"""Run resumable fixed-checkpoint search-budget and patience diagnostics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import yaml

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario
from pa_moap_rl.experiments.run_preference_response_v2 import (
    _load_and_validate as load_preference_context,
)
from pa_moap_rl.experiments.search_budget_trace_v2 import (
    build_search_conditions,
    derive_condition_from_master,
    deterministic_policy_trace,
)
from pa_moap_rl.solvers.ppo_solver import build_actor_critic, resolve_device
from pa_moap_rl.utils.scoring import score_assignment


ALLOWED_STATUSES = {"awaiting_user_confirmation", "user_confirmed"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_gzip_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    temporary.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_hash_manifest(output_dir: Path) -> Path:
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
    output = output_dir / "output_hashes.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "sha256", "size_bytes"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def _load_config(
    config_path: Path, *, output_override: Path | None, require_confirmed: bool
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported search-budget configuration schema.")
    status = config.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported search-budget configuration status: {status}")
    if require_confirmed and status != "user_confirmed":
        raise ValueError(
            "Search-budget protocol is not user-confirmed; full evaluation is blocked."
        )
    preference_config = Path(config["preference_response_config"]).resolve()
    context = load_preference_context(preference_config, output_override=None)
    output_dir = (
        output_override or Path(config["run"]["output_dir"])
    ).resolve()
    config_hash = _sha256(config_path)

    protocol = config["search_protocol"]
    all_conditions = []
    for row in context["core"].values():
        all_conditions.extend(
            build_search_conditions(protocol, n=int(row["N"]))
        )
    if not all_conditions:
        raise ValueError("Search protocol resolved to no conditions.")
    if config["analysis"]["accepted_move_definition"] != (
        "reward > environment.improvement_eps"
    ):
        raise ValueError("Accepted-move definition does not match trace semantics.")
    return config, context, output_dir, config_hash


def validate_protocol(config_path: str | Path) -> dict[str, Any]:
    """Validate immutable inputs and resolve the proposed condition ranges."""

    path = Path(config_path).resolve()
    config, context, output_dir, config_hash = _load_config(
        path, output_override=None, require_confirmed=False
    )
    budgets = []
    unique_execution_settings = set()
    conceptual_condition_count = 0
    for row in context["core"].values():
        conditions = build_search_conditions(
            config["search_protocol"], n=int(row["N"])
        )
        conceptual_condition_count += len(conditions) * 4
        for item in conditions:
            budgets.append(item.max_steps)
            unique_execution_settings.add(item.execution_key)
    return {
        "schema_version": 1,
        "status": config["status"],
        "config_path": str(path),
        "config_sha256": config_hash,
        "output_dir": str(output_dir),
        "objective_hash": context["objective"].config_hash,
        "checkpoint_sha256": context["checkpoint_hash"],
        "checkpoint_update": int(context["checkpoint"]["update"]),
        "original_count": len(context["core"]),
        "pair_count": len(context["pairs"]),
        "conceptual_rollout_count": conceptual_condition_count,
        "unique_execution_settings": [
            {"max_steps": item[0], "patience": item[1]}
            for item in sorted(unique_execution_settings)
        ],
        "min_resolved_budget": min(budgets),
        "max_resolved_budget": max(budgets),
        "full_run_blocked_until_user_confirmed": config["status"]
        != "user_confirmed",
    }


def run_experiment(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    limit_originals: int | None = None,
) -> dict[str, Any]:
    """Run all confirmed conditions with per-rollout atomic resume files."""

    config_path = Path(config_path).resolve()
    config, context, output_dir, config_hash = _load_config(
        config_path,
        output_override=(
            None if output_override is None else Path(output_override).resolve()
        ),
        require_confirmed=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_root = output_dir / "pair_traces"
    rollout_root.mkdir(parents=True, exist_ok=True)

    original_ids = sorted(context["core"])
    if limit_originals is not None:
        if limit_originals <= 0:
            raise ValueError("limit_originals must be positive.")
        original_ids = original_ids[:limit_originals]
    selected = set(original_ids)
    pairs = [row for row in context["pairs"] if row["original_id"] in selected]

    preference_config = context["config"]
    device = resolve_device(preference_config["ppo"]["device"])
    if preference_config["ppo"].get("deterministic", False):
        torch.use_deterministic_algorithms(True)
    project_config = load_config()
    checkpoint = context["checkpoint"]
    first_row = context["core"][pairs[0]["original_id"]]
    first_instance = load_instance_json(Path(first_row["output_json"]))
    model = build_actor_critic(
        first_instance,
        config=project_config,
        device=device,
        hidden_dim=checkpoint.get("hidden_dim"),
        objective=context["objective"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    expected_condition_results = sum(
        len(
            build_search_conditions(
                config["search_protocol"],
                n=int(context["core"][pair["original_id"]]["N"]),
            )
        )
        for pair in pairs
    )
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "objective_hash": context["objective"].config_hash,
        "checkpoint_path": str(context["checkpoint_path"]),
        "checkpoint_sha256": context["checkpoint_hash"],
        "checkpoint_update": int(checkpoint["update"]),
        "checkpoint_data_hash": context["ppo_data_hash"],
        "device": str(device),
        "original_count": len(original_ids),
        "pair_count": len(pairs),
        "expected_master_trace_count": len(pairs),
        "expected_condition_result_count": expected_condition_results,
        "execution_reduction_note": (
            "Each deterministic pair uses one longest master trace; all max_steps "
            "and patience conditions are exact stopping-rule truncations."
        ),
        "resume": bool(config["run"].get("resume", True)),
        "search_protocol": config["search_protocol"],
    }
    _write_json_atomic(output_dir / "run_metadata.json", metadata)

    completed_pairs = 0
    executed_master_traces = 0
    started = time.perf_counter()
    for pair_index, pair in enumerate(pairs, start=1):
        core_row = context["core"][pair["original_id"]]
        base = load_instance_json(Path(core_row["output_json"]))
        scenario = context["scenarios"][pair["scenario_id"]]
        instance = apply_scenario(
            base, scenario, project_config, objective=context["objective"]
        )
        exact_path = context["exact_dir"] / f"{pair['pair_id']}.json"
        expected_exact_hash = context["exact_hashes"].get(exact_path.name)
        if not exact_path.is_file() or _sha256(exact_path) != expected_exact_hash:
            raise ValueError(f"Exact result hash mismatch: {pair['pair_id']}")
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
        if (
            exact.get("objective_hash") != context["objective"].config_hash
            or not exact.get("proven_optimal")
            or float(exact.get("certified_gap", 1.0)) > 1.0e-12
        ):
            raise ValueError(f"Invalid exact reference: {pair['pair_id']}")
        exact_score = float(exact["score"]["total_score"])
        recomputed = score_assignment(
            exact["assignment"], instance=instance, objective=context["objective"]
        )
        if abs(recomputed.total_score - exact_score) > 1.0e-9:
            raise ValueError(f"Exact score recomputation mismatch: {pair['pair_id']}")

        conditions = build_search_conditions(
            config["search_protocol"], n=instance.n
        )
        output = rollout_root / f"{pair['pair_id']}.json.gz"
        expected_condition_specs = {
            item.condition_id: asdict(item) for item in conditions
        }
        if output.is_file():
            existing = _read_gzip_json(output)
            if (
                existing.get("config_sha256") != config_hash
                or existing.get("objective_hash")
                != context["objective"].config_hash
                or existing.get("checkpoint_sha256")
                != context["checkpoint_hash"]
                or existing.get("pair_id") != pair["pair_id"]
                or existing.get("condition_specs") != expected_condition_specs
            ):
                raise ValueError(
                    f"Existing master-trace provenance mismatch: {output}"
                )
            completed_pairs += 1
        else:
            master_steps = max(item.max_steps for item in conditions)
            master = deterministic_policy_trace(
                instance,
                model=model,
                project_config=project_config,
                objective=context["objective"],
                max_steps=master_steps,
                patience=master_steps,
                exact_score=exact_score,
            )
            executed_master_traces += 1
            condition_results = {
                condition.condition_id: derive_condition_from_master(
                    master,
                    instance=instance,
                    objective=context["objective"],
                    max_steps=condition.max_steps,
                    patience=condition.patience,
                )
                for condition in conditions
            }
            if any(
                item["hard_feasible_rate"] != 1.0
                or item["mask_violation_count"] != 0
                or item["best_score_recompute_error"] > 1.0e-9
                for item in condition_results.values()
            ):
                raise RuntimeError(f"Derived rollout audit failed: {pair['pair_id']}")
            payload = {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "config_sha256": config_hash,
                "objective_hash": context["objective"].config_hash,
                "checkpoint_sha256": context["checkpoint_hash"],
                "checkpoint_update": int(checkpoint["update"]),
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
                "student_template": scenario.student_template,
                "teacher_template": scenario.teacher_template,
                "n": int(instance.n),
                "instance_path": core_row["output_json"],
                "instance_sha256": core_row["output_json_sha256"],
                "exact_result_path": str(exact_path),
                "exact_result_sha256": expected_exact_hash,
                "condition_specs": expected_condition_specs,
                "master_trace": master,
                "condition_results": condition_results,
            }
            _write_gzip_json_atomic(output, payload)
            completed_pairs += 1
        if pair_index % 12 == 0 or pair_index == len(pairs):
            print(
                f"completed pairs {pair_index}/{len(pairs)}; "
                f"master traces {completed_pairs}/{len(pairs)}",
                flush=True,
            )

    metadata.update(
        {
            "status": "rollout_evaluation_completed",
            "completed_master_trace_count": completed_pairs,
            "completed_condition_result_count": expected_condition_results,
            "executed_master_trace_count_this_attempt": executed_master_traces,
            "last_attempt_elapsed_sec": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    metadata["output_hash_manifest"] = str(output_dir / "output_hashes.csv")
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    _write_hash_manifest(output_dir)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "pa_moap_rl/configs/"
            "search_budget_diagnostic_four_topology_v2.yaml"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--limit-originals", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_protocol(args.config)
    else:
        result = run_experiment(
            args.config,
            output_override=args.output_dir,
            limit_originals=args.limit_originals,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
