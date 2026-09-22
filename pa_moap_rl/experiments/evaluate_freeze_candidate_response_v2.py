"""Evaluate the stage6 checkpoint candidate on the controlled 72x4 profiles."""

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
from pa_moap_rl.data.scenarios import apply_scenario
from pa_moap_rl.experiments.run_preference_response_v2 import (
    _assignment_hash,
    _load_and_validate as load_preference_context,
)
from pa_moap_rl.experiments.train_ppo_batch import _policy_rollout
from pa_moap_rl.models.preference_repair_v2 import build_versioned_actor_critic
from pa_moap_rl.solvers.ppo_solver import PPOConfig, build_actor_critic, resolve_device
from pa_moap_rl.utils.scoring import score_assignment


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


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
    _write_csv(output_dir / "output_hashes.csv", rows)


def run(
    config_path: str | Path, *, validate_only: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or int(config.get("schema_version", 0)) != 1
        or config.get("status") not in {
            "stage6_diagnostic_not_frozen",
            "ablation_controlled_cross_confirmed",
        }
    ):
        raise ValueError("Invalid stage6 response diagnostic config.")
    context = load_preference_context(
        Path(config["preference_response_config"]).resolve(),
        output_override=None,
    )
    objective = context["objective"]
    checkpoint_path = Path(config["checkpoint"]["path"])
    checkpoint_hash = _sha256(checkpoint_path)
    expected_checkpoint_hash = config["checkpoint"].get("checkpoint_sha256")
    if expected_checkpoint_hash is not None and checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("Candidate checkpoint hash mismatch.")
    device = resolve_device(config["inference"]["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_update = int(config["checkpoint"]["expected_update"])
    if int(checkpoint["update"]) != expected_update:
        raise ValueError("Candidate checkpoint update mismatch.")
    first_row = context["core"][context["pairs"][0]["original_id"]]
    first_instance = load_instance_json(Path(first_row["output_json"]))
    checkpoint_data_hash = context["ppo_data_hash"]
    source_metadata_path = config["checkpoint"].get("run_metadata")
    source_metadata = None
    if source_metadata_path is not None:
        source_metadata = json.loads(
            Path(source_metadata_path).read_text(encoding="utf-8")
        )
        for item in source_metadata["inputs"].values():
            if _sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Candidate training input drift: {item['path']}")
        checkpoint_data_hash = hashlib.sha256(
            "".join(
                item["sha256"] for item in source_metadata["inputs"].values()
            ).encode("ascii")
        ).hexdigest()
        if checkpoint_data_hash != source_metadata["data_hash"]:
            raise ValueError("Candidate run data hash mismatch.")
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_matrix_hash(first_instance),
        data_hash=checkpoint_data_hash,
    )
    experiment_spec = checkpoint.get("experiment_spec")
    if source_metadata is not None:
        if experiment_spec is None or experiment_spec != source_metadata.get("experiment_spec"):
            raise ValueError("Candidate experiment_spec mismatch.")
        expected_arm = config["checkpoint"].get("arm_id")
        if expected_arm is not None and experiment_spec.get("arm_id") != expected_arm:
            raise ValueError("Candidate arm id mismatch.")
    if config["inference"]["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    project_config = load_config()
    ppo_config = replace(
        PPOConfig(**checkpoint["ppo_config"]),
        device=str(device),
        env_max_steps=int(config["inference"]["max_steps"]),
        env_patience=int(config["inference"]["patience"]),
    )
    if experiment_spec is None:
        model = build_actor_critic(
            first_instance,
            config=project_config,
            device=device,
            hidden_dim=checkpoint.get("hidden_dim"),
            objective=objective,
        )
    else:
        model = build_versioned_actor_critic(
            first_instance,
            model_version=experiment_spec["model"]["model_version"],
            config=project_config,
            device=device,
            hidden_dim=checkpoint.get("hidden_dim"),
            objective=objective,
        )
        model.experiment_spec = experiment_spec
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if validate_only:
        return {
            "status": "validated_no_evaluation",
            "checkpoint_update": expected_update,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_data_hash": checkpoint_data_hash,
            "experiment_spec": experiment_spec,
            "inference_max_steps": ppo_config.env_max_steps,
            "inference_patience": ppo_config.env_patience,
        }

    output_dir = Path(config["run"]["output_dir"]).resolve()
    pair_dir = output_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(config_path)
    metadata = {
        "schema_version": 1,
        "diagnostic_id": config["diagnostic_id"],
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "objective_hash": objective.config_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_update": expected_update,
        "checkpoint_data_hash": checkpoint_data_hash,
        "experiment_spec": experiment_spec,
        "inference_max_steps": ppo_config.env_max_steps,
        "inference_patience": ppo_config.env_patience,
        "pair_count": len(context["pairs"]),
        "parameters_frozen": False,
        "pure_ppo_only": True,
    }
    _write_json_atomic(output_dir / "run_metadata.json", metadata)

    started = time.perf_counter()
    completed = 0
    for pair in context["pairs"]:
        output = pair_dir / f"{pair['pair_id']}.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("config_sha256") != config_hash
                or existing.get("checkpoint_sha256") != checkpoint_hash
                or existing.get("objective_hash") != objective.config_hash
                or existing.get("pair_id") != pair["pair_id"]
            ):
                raise ValueError(f"Existing pair mismatch: {output}")
            completed += 1
            continue
        core_row = context["core"][pair["original_id"]]
        base = load_instance_json(Path(core_row["output_json"]))
        scenario = context["scenarios"][pair["scenario_id"]]
        instance = apply_scenario(
            base, scenario, project_config, objective=objective
        )
        exact_path = context["exact_dir"] / f"{pair['pair_id']}.json"
        expected_exact_hash = context["exact_hashes"][exact_path.name]
        if _sha256(exact_path) != expected_exact_hash:
            raise ValueError(f"Exact result hash mismatch: {pair['pair_id']}")
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
        exact_score = float(exact["score"]["total_score"])
        result = _policy_rollout(
            instance,
            model=model,
            project_config=project_config,
            ppo_config=ppo_config,
            deterministic=True,
            objective=objective,
        )
        recomputed = score_assignment(
            result.assignment, instance=instance, objective=objective
        )
        recomputation_error = abs(
            float(recomputed.total_score) - float(result.score.total_score)
        )
        gap = exact_score - float(result.score.total_score)
        if gap < -1.0e-9:
            raise RuntimeError(f"Candidate exceeds exact: {pair['pair_id']}")
        if (
            result.metrics["hard_feasible_rate"] != 1.0
            or result.metrics["mask_violation_count"] != 0
        ):
            raise RuntimeError(f"Candidate is infeasible: {pair['pair_id']}")
        payload = {
            "schema_version": 1,
            "config_sha256": config_hash,
            "objective_hash": objective.config_hash,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_update": expected_update,
            **{
                key: pair[key]
                for key in (
                    "pair_id",
                    "original_id",
                    "source_family_id",
                    "topology",
                    "size_group",
                    "scenario_id",
                )
            },
            "n": instance.n,
            "assignment": result.assignment.tolist(),
            "assignment_hash": _assignment_hash(result.assignment),
            "runtime_sec": result.runtime,
            "score": result.score.as_dict(),
            "exact_score": exact_score,
            "relative_gap_to_exact": max(0.0, gap)
            / max(abs(exact_score), objective.epsilon),
            "hard_feasible_rate": result.metrics["hard_feasible_rate"],
            "mask_violation_count": result.metrics["mask_violation_count"],
            "score_recomputation_error": recomputation_error,
        }
        _write_json_atomic(output, payload)
        completed += 1
        if completed % 24 == 0:
            print(f"completed {completed}/{len(context['pairs'])}", flush=True)

    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(pair_dir.glob("*.json"))
    ]
    if len(payloads) != 288:
        raise ValueError("Expected 288 candidate response payloads.")
    lookup = {
        (item["original_id"], item["scenario_id"]): item for item in payloads
    }
    response_rows = []
    baseline_id = "controlled_balanced"
    for item in payloads:
        if item["scenario_id"] == baseline_id:
            continue
        baseline = lookup[(item["original_id"], baseline_id)]
        core_row = context["core"][item["original_id"]]
        base = load_instance_json(Path(core_row["output_json"]))
        scenario = context["scenarios"][item["scenario_id"]]
        shifted = apply_scenario(
            base, scenario, project_config, objective=objective
        )
        no_reopt = score_assignment(
            baseline["assignment"], instance=shifted, objective=objective
        )
        gain = float(item["score"]["total_score"]) - no_reopt.total_score
        response_rows.append(
            {
                key: item[key]
                for key in (
                    "pair_id",
                    "original_id",
                    "source_family_id",
                    "topology",
                    "size_group",
                    "scenario_id",
                    "n",
                    "relative_gap_to_exact",
                )
            }
            | {
                "reoptimization_gain": gain,
                "positive_gain": gain > 1.0e-9,
                "assignment_change_rate": float(
                    np.mean(
                        np.asarray(item["assignment"])
                        != np.asarray(baseline["assignment"])
                    )
                ),
            }
        )
    _write_csv(output_dir / "response_evidence.csv", response_rows)
    summary_rows = []
    for scenario_id in sorted({row["scenario_id"] for row in response_rows}):
        for size_group in (None, "group9"):
            values = [
                row
                for row in response_rows
                if row["scenario_id"] == scenario_id
                and (size_group is None or row["size_group"] == size_group)
            ]
            summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "scope": "overall" if size_group is None else "size_group",
                    "scope_value": "all" if size_group is None else size_group,
                    "count": len(values),
                    "positive_gain_rate": float(
                        np.mean([row["positive_gain"] for row in values])
                    ),
                    **{
                        f"reoptimization_gain_{key}": value
                        for key, value in _stats(
                            [float(row["reoptimization_gain"]) for row in values]
                        ).items()
                    },
                    **{
                        f"assignment_change_rate_{key}": value
                        for key, value in _stats(
                            [
                                float(row["assignment_change_rate"])
                                for row in values
                            ]
                        ).items()
                    },
                    **{
                        f"relative_gap_{key}": value
                        for key, value in _stats(
                            [
                                float(row["relative_gap_to_exact"])
                                for row in values
                            ]
                        ).items()
                    },
                }
            )
    _write_csv(output_dir / "response_summary.csv", summary_rows)
    overall_gaps = [
        float(item["relative_gap_to_exact"]) for item in payloads
    ]
    summary = {
        "schema_version": 1,
        "status": "completed_not_frozen",
        "objective_hash": objective.config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_update": expected_update,
        "experiment_spec": experiment_spec,
        "inference_max_steps": ppo_config.env_max_steps,
        "inference_patience": ppo_config.env_patience,
        "pair_count": len(payloads),
        "mean_relative_gap": float(np.mean(overall_gaps)),
        "p90_relative_gap": float(np.quantile(overall_gaps, 0.90)),
        "response_summary": summary_rows,
        "all_hard_feasible": all(
            item["hard_feasible_rate"] == 1.0
            and item["mask_violation_count"] == 0
            for item in payloads
        ),
        "max_score_recomputation_error": float(
            max(item.get("score_recomputation_error", 0.0) for item in payloads)
        ),
        "parameters_frozen": False,
    }
    _write_json_atomic(output_dir / "candidate_response_summary.json", summary)
    metadata.update(
        {
            "status": "completed_not_frozen",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": time.perf_counter() - started,
            "completed_pairs": completed,
        }
    )
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    _write_hash_manifest(output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "pa_moap_rl/configs/"
            "ppo_freeze_candidate_response_update60_v2.yaml"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, validate_only=args.validate_only),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
