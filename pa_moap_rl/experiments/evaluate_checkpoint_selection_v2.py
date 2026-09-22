"""Re-evaluate run03 numbered checkpoints with the candidate inference budget."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from pa_moap_rl.checkpointing import method_matrix_hash, validate_checkpoint_metadata
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.experiments.formal_training import (
    evaluate_validation_pairs,
    load_validation_pairs,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.ppo_solver import PPOConfig, build_actor_critic, resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(rows: list[dict[str, Any]], *, update: int) -> dict[str, Any]:
    scores = np.asarray([float(row["total_score"]) for row in rows])
    gaps = np.asarray([float(row["relative_gap_to_exact"]) for row in rows])
    runtimes = np.asarray([float(row["runtime"]) for row in rows])
    result = {
        "update": update,
        "count": len(rows),
        "mean_total_score": float(np.mean(scores)),
        "median_total_score": float(np.median(scores)),
        "p90_relative_gap": float(np.quantile(gaps, 0.90)),
        "mean_relative_gap": float(np.mean(gaps)),
        "worst_relative_gap": float(np.max(gaps)),
        "mean_runtime_sec": float(np.mean(runtimes)),
        "p95_runtime_sec": float(np.quantile(runtimes, 0.95)),
    }
    for threshold in (0.01, 0.02, 0.03, 0.05):
        result[f"gap_le_{int(threshold * 100)}pct_rate"] = float(
            np.mean(gaps <= threshold + 1.0e-12)
        )
    return result


def _better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    tolerance = 1.0e-12
    score = float(candidate["mean_total_score"])
    best_score = float(incumbent["mean_total_score"])
    if score > best_score + tolerance:
        return True
    if abs(score - best_score) > tolerance:
        return False
    p90 = float(candidate["p90_relative_gap"])
    best_p90 = float(incumbent["p90_relative_gap"])
    if p90 < best_p90 - tolerance:
        return True
    if abs(p90 - best_p90) > tolerance:
        return False
    return int(candidate["update"]) < int(incumbent["update"])


def _write_hash_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_csv_atomic(output_dir / "output_hashes.csv", rows)


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or int(config.get("schema_version", 0)) != 1
        or config.get("status") != "stage6_diagnostic_not_frozen"
    ):
        raise ValueError("Invalid stage6 checkpoint diagnostic config.")
    objective = load_objective_spec(config["objective"]["config"])
    if objective.config_hash != config["objective"]["config_hash"]:
        raise ValueError("Objective hash mismatch.")
    source = config["source_run"]
    pilot_config_path = Path(source["config"])
    pilot = yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
    run_dir = Path(source["run_dir"])
    run_metadata_path = run_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if run_metadata["objective_hash"] != objective.config_hash:
        raise ValueError("Source run objective mismatch.")
    for item in run_metadata["inputs"].values():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Source run input drift: {item['path']}")
    data_hash = hashlib.sha256(
        "".join(
            item["sha256"] for item in run_metadata["inputs"].values()
        ).encode("ascii")
    ).hexdigest()
    if data_hash != run_metadata["data_hash"]:
        raise ValueError("Source run data hash mismatch.")

    data = pilot["data"]
    pairs = load_validation_pairs(
        data["validation_pairs"],
        scenario_bank=data["validation_scenarios"],
        objective=objective,
        allow_controlled_held_out=bool(
            data.get("allow_controlled_held_out_validation", False)
        ),
    )
    if len(pairs) != 72:
        raise ValueError("Checkpoint selection diagnostic requires 72 pairs.")
    project_config = load_config()
    first_instance = load_instance_json(pairs[0].instance_path)
    device = resolve_device(config["run"]["device"])
    if config["checkpoint_selection_diagnostic"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    output_dir = Path(config["run"]["output_dir"]).resolve()
    checkpoint_output = output_dir / "checkpoints"
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(config_path)
    updates = [int(value) for value in source["checkpoint_updates"]]
    metadata = {
        "schema_version": 1,
        "diagnostic_id": config["diagnostic_id"],
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "objective_hash": objective.config_hash,
        "source_run": str(run_dir.resolve()),
        "source_run_metadata_sha256": _sha256(run_metadata_path),
        "source_data_hash": data_hash,
        "device": str(device),
        "checkpoint_updates": updates,
        "inference_max_steps": int(
            config["checkpoint_selection_diagnostic"]["inference_max_steps"]
        ),
        "inference_patience": int(
            config["checkpoint_selection_diagnostic"]["inference_patience"]
        ),
        "pair_count": len(pairs),
        "parameters_frozen": False,
    }
    _write_json_atomic(output_dir / "run_metadata.json", metadata)

    summaries = []
    started = time.perf_counter()
    for index, update in enumerate(updates, start=1):
        checkpoint_path = (
            run_dir / "checkpoints" / f"update_{update:06d}.pt"
        )
        output_csv = checkpoint_output / f"update_{update:06d}.csv"
        output_meta = checkpoint_output / f"update_{update:06d}.json"
        checkpoint_hash = _sha256(checkpoint_path)
        if output_csv.is_file() and output_meta.is_file():
            existing = json.loads(output_meta.read_text(encoding="utf-8"))
            if (
                existing.get("config_sha256") != config_hash
                or existing.get("checkpoint_sha256") != checkpoint_hash
                or existing.get("objective_hash") != objective.config_hash
                or int(existing.get("row_count", 0)) != 72
                or existing.get("output_csv_sha256") != _sha256(output_csv)
            ):
                raise ValueError(
                    f"Existing checkpoint evaluation mismatch: {output_csv}"
                )
            summary = existing["summary"]
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if int(checkpoint["update"]) != update:
                raise ValueError(f"Checkpoint update mismatch: {checkpoint_path}")
            validate_checkpoint_metadata(
                checkpoint,
                objective=objective,
                method_hash=method_matrix_hash(first_instance),
                data_hash=data_hash,
            )
            ppo_config = replace(
                PPOConfig(**checkpoint["ppo_config"]),
                device=str(device),
                env_max_steps=metadata["inference_max_steps"],
                env_patience=metadata["inference_patience"],
            )
            model = build_actor_critic(
                first_instance,
                config=project_config,
                device=device,
                hidden_dim=checkpoint.get("hidden_dim"),
                objective=objective,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            rows = evaluate_validation_pairs(
                model,
                pairs,
                project_config=project_config,
                ppo_config=ppo_config,
                seed=int(pilot["run"]["seed"]),
                objective=objective,
            )
            for row in rows:
                row["checkpoint_update"] = update
                row["checkpoint_sha256"] = checkpoint_hash
                row["inference_max_steps"] = metadata["inference_max_steps"]
                row["inference_patience"] = metadata["inference_patience"]
            _write_csv_atomic(output_csv, rows)
            summary = _stats(rows, update=update)
            _write_json_atomic(
                output_meta,
                {
                    "schema_version": 1,
                    "config_sha256": config_hash,
                    "objective_hash": objective.config_hash,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_update": update,
                    "row_count": len(rows),
                    "output_csv_sha256": _sha256(output_csv),
                    "summary": summary,
                },
            )
        summaries.append(summary)
        print(f"completed checkpoint {index}/{len(updates)}: update {update}", flush=True)

    best = None
    for item in summaries:
        if _better(item, best):
            best = item
    _write_csv_atomic(output_dir / "checkpoint_summary.csv", summaries)
    decision = {
        "schema_version": 1,
        "status": "diagnostic_completed_not_frozen",
        "objective_hash": objective.config_hash,
        "source_data_hash": data_hash,
        "inference_max_steps": metadata["inference_max_steps"],
        "inference_patience": metadata["inference_patience"],
        "selection_rule": config["checkpoint_selection_diagnostic"],
        "selected_update": int(best["update"]),
        "selected_summary": best,
        "previous_128_32_selected_update": 70,
        "selection_changed": int(best["update"]) != 70,
        "checkpoint_summaries": summaries,
        "parameters_frozen": False,
    }
    _write_json_atomic(output_dir / "checkpoint_selection_diagnostic.json", decision)
    metadata.update(
        {
            "status": "completed_not_frozen",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": time.perf_counter() - started,
            "selected_update": int(best["update"]),
            "selection_changed": decision["selection_changed"],
        }
    )
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    _write_hash_manifest(output_dir)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "pa_moap_rl/configs/"
            "ppo_freeze_decision_diagnostic_four_topology_v2.yaml"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
