"""Run the four-topology 72x4 controlled dual-actor preference evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pa_moap_rl.checkpointing import method_matrix_hash, validate_checkpoint_metadata
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_bank
from pa_moap_rl.experiments.train_ppo_batch import _policy_rollout
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.greedy_solver import solve_scalarized_independent_greedy
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.ppo_solver import PPOConfig, build_actor_critic, resolve_device
from pa_moap_rl.utils.metrics import make_solver_result
from pa_moap_rl.utils.scoring import score_assignment


EXPECTED_SCENARIOS = {
    "controlled_balanced",
    "controlled_student_only",
    "controlled_teacher_only",
    "controlled_conflict",
}
EXPECTED_METHODS = {
    "gurobi_exact",
    "scalarized_independent_greedy",
    "local_search_best_improvement",
    "ppo_shared_policy",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _assignment_hash(assignment: np.ndarray | list[int]) -> str:
    payload = json.dumps(
        [int(value) for value in assignment], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _result_record(result: Any, *, exact_score: float) -> dict[str, Any]:
    assignment = np.asarray(result.assignment, dtype=np.int64)
    score = result.score
    absolute_gap = exact_score - float(score.total_score)
    if absolute_gap < -1.0e-9:
        raise RuntimeError(
            f"{result.solver_name} exceeds the proven optimum by {-absolute_gap}."
        )
    return {
        "solver_name": result.solver_name,
        "assignment": assignment.tolist(),
        "assignment_hash": _assignment_hash(assignment),
        "runtime": float(result.runtime),
        "total_score": float(score.total_score),
        "effect_score": float(score.effect_score),
        "student_pref_score": float(score.student_pref_score),
        "teacher_pref_score": float(score.teacher_pref_score),
        "preference_score": float(
            (score.student_pref_score + score.teacher_pref_score) / 2.0
        ),
        "global_score": float(score.global_score),
        "entropy": float(score.entropy),
        "max_method_share": float(np.max(score.method_distribution)),
        "method_distribution": score.method_distribution.tolist(),
        "diversity_violation": float(score.diversity_violation),
        "cap_violation": float(score.cap_violation),
        "soft_penalty": float(score.soft_penalty),
        "effect_contribution": float(score.effect_contribution),
        "student_contribution": float(score.student_contribution),
        "teacher_contribution": float(score.teacher_contribution),
        "global_contribution": float(score.global_contribution),
        "entropy_penalty_contribution": float(score.entropy_penalty_contribution),
        "cap_penalty_contribution": float(score.cap_penalty_contribution),
        "hard_feasible_rate": float(result.metrics["hard_feasible_rate"]),
        "mask_violation_count": int(result.metrics["mask_violation_count"]),
        "exact_total_score": exact_score,
        "absolute_gap_to_exact": max(0.0, absolute_gap),
        "relative_gap_to_exact": max(0.0, absolute_gap) / max(abs(exact_score), 1.0e-8),
        "history_length": len(result.history),
    }


def _load_and_validate(config_path: Path, *, output_override: Path | None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported preference-response configuration schema.")
    if config.get("status") != "user_confirmed":
        raise ValueError("Preference-response configuration is not user-confirmed.")
    configured_methods = set(config["methods"])
    if configured_methods != EXPECTED_METHODS:
        raise ValueError(f"Expected exactly four methods, got {configured_methods}.")

    objective_path = Path(config["objective"]["config"])
    objective = load_objective_spec(objective_path)
    if objective.config_hash != config["objective"]["config_hash"]:
        raise ValueError("Frozen objective hash mismatch.")

    data = config["data"]
    core_path = Path(data["core_manifest"])
    pair_path = Path(data["pair_manifest"])
    scenario_path = Path(data["scenario_bank"])
    exact_dir = Path(data["exact_result_dir"])
    exact_hash_path = Path(data["exact_hash_manifest"])
    required_paths = [core_path, pair_path, scenario_path, exact_dir, exact_hash_path]
    if any(not path.exists() for path in required_paths):
        missing = [str(path) for path in required_paths if not path.exists()]
        raise FileNotFoundError(f"Missing preference-response inputs: {missing}")

    core_rows = _read_csv(core_path)
    core = {row["original_id"]: row for row in core_rows}
    if len(core_rows) != 72 or len(core) != 72:
        raise ValueError("The controlled core must contain 72 unique instances.")
    for row in core_rows:
        instance_path = Path(row["output_json"])
        if not instance_path.is_file() or _sha256(instance_path) != row["output_json_sha256"]:
            raise ValueError(f"Instance hash mismatch: {row['original_id']}")

    scenarios = {item.scenario_id: item for item in load_scenario_bank(scenario_path)}
    if set(scenarios) != EXPECTED_SCENARIOS:
        raise ValueError(f"Controlled scenario set mismatch: {set(scenarios)}")
    for scenario in scenarios.values():
        if scenario.schema_version != 2 or scenario.access_status != data["access_status"]:
            raise ValueError(f"Invalid controlled scenario metadata: {scenario.scenario_id}")

    pair_rows = _read_csv(pair_path)
    if len(pair_rows) != 288 or len({row["pair_id"] for row in pair_rows}) != 288:
        raise ValueError("The controlled cross must contain 288 unique pairs.")
    coverage: dict[str, set[str]] = {}
    for row in pair_rows:
        if row["original_id"] not in core:
            raise ValueError(f"Unknown core instance: {row['original_id']}")
        if row["scenario_id"] not in scenarios:
            raise ValueError(f"Unknown controlled scenario: {row['scenario_id']}")
        for field in ("source_family_id", "topology", "size_group"):
            if row[field] != core[row["original_id"]][field]:
                raise ValueError(f"Pair/core {field} mismatch: {row['pair_id']}")
        coverage.setdefault(row["original_id"], set()).add(row["scenario_id"])
    if any(values != EXPECTED_SCENARIOS for values in coverage.values()):
        raise ValueError("Every core instance must be crossed with all four scenarios.")

    exact_prefix = (
        "controlled_cross/instances/"
        f"{objective.config_hash}/"
    )
    exact_hashes = {
        Path(row["relative_path"]).name: row["sha256"]
        for row in _read_csv(exact_hash_path)
        if row["relative_path"].startswith(exact_prefix)
    }
    if len(exact_hashes) != 288:
        raise ValueError("Exact hash manifest does not contain 288 b075 controlled results.")

    ppo = config["ppo"]
    checkpoint_path = Path(ppo["checkpoint"])
    run_metadata_path = Path(ppo["run_metadata"])
    ppo_hash_path = Path(ppo["output_hash_manifest"])
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    for item in run_metadata["inputs"].values():
        actual = _sha256(Path(item["path"]))
        if actual != item["sha256"]:
            raise ValueError(f"PPO training input drift: {item['path']}")
    data_hash = hashlib.sha256(
        "".join(item["sha256"] for item in run_metadata["inputs"].values()).encode("ascii")
    ).hexdigest()
    if data_hash != run_metadata["data_hash"]:
        raise ValueError("PPO run data hash mismatch.")
    checkpoint_hashes = {
        row["relative_path"]: row["sha256"] for row in _read_csv(ppo_hash_path)
    }
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hashes.get("checkpoints/best.pt") != checkpoint_hash:
        raise ValueError("PPO best checkpoint hash mismatch.")

    first_base = load_instance_json(Path(core_rows[0]["output_json"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["update"]) != int(ppo["expected_update"]):
        raise ValueError("Unexpected PPO best checkpoint update.")
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_matrix_hash(first_base),
        data_hash=data_hash,
    )
    output_dir = (output_override or Path(config["run"]["output_dir"])).resolve()
    return {
        "config": config,
        "objective": objective,
        "core": core,
        "pairs": pair_rows,
        "scenarios": scenarios,
        "exact_dir": exact_dir,
        "exact_hashes": exact_hashes,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "ppo_data_hash": data_hash,
        "output_dir": output_dir,
        "input_paths": [core_path, pair_path, scenario_path, exact_hash_path, config_path],
    }


def run_experiment(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    limit_originals: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    context = _load_and_validate(
        config_path,
        output_override=None if output_override is None else Path(output_override),
    )
    config = context["config"]
    objective = context["objective"]
    output_dir: Path = context["output_dir"]
    result_dir = output_dir / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    original_ids = sorted(context["core"])
    if limit_originals is not None:
        if limit_originals < 1:
            raise ValueError("limit_originals must be positive.")
        original_ids = original_ids[:limit_originals]
    selected = set(original_ids)
    pairs = [row for row in context["pairs"] if row["original_id"] in selected]

    device = resolve_device(config["ppo"]["device"])
    if config["ppo"].get("deterministic", False):
        torch.use_deterministic_algorithms(True)
    project_config = load_config()
    checkpoint = context["checkpoint"]
    ppo_config = replace(PPOConfig(**checkpoint["ppo_config"]), device=str(device))
    first_row = context["core"][pairs[0]["original_id"]]
    first_instance = load_instance_json(Path(first_row["output_json"]))
    model = build_actor_critic(
        first_instance,
        config=project_config,
        device=device,
        hidden_dim=checkpoint.get("hidden_dim"),
        objective=objective,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "objective_hash": objective.config_hash,
        "checkpoint_path": str(context["checkpoint_path"]),
        "checkpoint_sha256": context["checkpoint_hash"],
        "checkpoint_update": int(checkpoint["update"]),
        "checkpoint_data_hash": context["ppo_data_hash"],
        "device": str(device),
        "pair_count": len(pairs),
        "original_count": len(original_ids),
        "input_hashes": {str(path): _sha256(path) for path in context["input_paths"]},
        "status": "running",
    }
    _write_json_atomic(output_dir / "run_metadata.json", metadata)

    completed = 0
    started = time.perf_counter()
    for pair in pairs:
        output = result_dir / f"{pair['pair_id']}.json"
        if output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if (
                existing.get("objective_hash") != objective.config_hash
                or existing.get("checkpoint_sha256") != context["checkpoint_hash"]
                or set(existing.get("methods", {})) != EXPECTED_METHODS
            ):
                raise ValueError(f"Existing pair result provenance mismatch: {output}")
            completed += 1
            continue

        core_row = context["core"][pair["original_id"]]
        base = load_instance_json(Path(core_row["output_json"]))
        scenario = context["scenarios"][pair["scenario_id"]]
        instance = apply_scenario(base, scenario, project_config, objective=objective)
        exact_path = context["exact_dir"] / f"{pair['pair_id']}.json"
        expected_exact_hash = context["exact_hashes"].get(exact_path.name)
        if not exact_path.is_file() or _sha256(exact_path) != expected_exact_hash:
            raise ValueError(f"Exact result hash mismatch: {pair['pair_id']}")
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
        if (
            exact.get("pair_id") != pair["pair_id"]
            or exact.get("objective_hash") != objective.config_hash
            or not exact.get("proven_optimal")
            or float(exact.get("certified_gap", 1.0)) > 1.0e-12
        ):
            raise ValueError(f"Invalid exact reference: {pair['pair_id']}")
        exact_score = float(exact["score"]["total_score"])
        recomputed = score_assignment(exact["assignment"], instance=instance, objective=objective)
        if abs(recomputed.total_score - exact_score) > 1.0e-9:
            raise ValueError(f"Exact score recomputation mismatch: {pair['pair_id']}")

        results = [
            make_solver_result(
                solver_name="gurobi_exact",
                instance=instance,
                assignment=exact["assignment"],
                runtime=float(exact["total_wall_runtime"]),
                objective=objective,
            ),
            solve_scalarized_independent_greedy(instance, objective=objective),
            local_search_best_improvement(
                instance,
                max_iter=instance.n * instance.m,
                objective=objective,
            ),
            _policy_rollout(
                instance,
                model=model,
                project_config=project_config,
                ppo_config=ppo_config,
                deterministic=True,
                objective=objective,
            ),
        ]
        methods = {
            result.solver_name: _result_record(result, exact_score=exact_score)
            for result in results
        }
        if set(methods) != EXPECTED_METHODS:
            raise RuntimeError(f"Method set mismatch for {pair['pair_id']}: {set(methods)}")
        if any(
            item["hard_feasible_rate"] != 1.0 or item["mask_violation_count"] != 0
            for item in methods.values()
        ):
            raise RuntimeError(f"Infeasible method result: {pair['pair_id']}")
        payload = {
            "schema_version": 1,
            **{key: pair[key] for key in (
                "pair_id", "original_id", "source_family_id", "topology",
                "size_group", "scenario_id",
            )},
            "student_template": scenario.student_template,
            "teacher_template": scenario.teacher_template,
            "instance_path": core_row["output_json"],
            "instance_sha256": core_row["output_json_sha256"],
            "exact_result_path": str(exact_path),
            "exact_result_sha256": expected_exact_hash,
            "objective_hash": objective.config_hash,
            "checkpoint_sha256": context["checkpoint_hash"],
            "checkpoint_update": int(checkpoint["update"]),
            "n": int(instance.n),
            "methods": methods,
        }
        _write_json_atomic(output, payload)
        completed += 1
        if completed % 12 == 0 or completed == len(pairs):
            print(f"completed {completed}/{len(pairs)} pairs", flush=True)

    pair_files = sorted(result_dir.glob("*.json"))
    pair_mtimes = [path.stat().st_mtime for path in pair_files]
    metadata.update(
        {
            "status": "pair_evaluation_completed",
            "completed_pairs": completed,
            "last_attempt_elapsed_sec": time.perf_counter() - started,
            "pair_file_first_written_at_utc": datetime.fromtimestamp(
                min(pair_mtimes), tz=timezone.utc
            ).isoformat(),
            "pair_file_last_written_at_utc": datetime.fromtimestamp(
                max(pair_mtimes), tz=timezone.utc
            ).isoformat(),
            "observed_pair_write_span_sec": max(pair_mtimes) - min(pair_mtimes),
            "runtime_accounting_note": (
                "The process was externally interrupted and resumed; the pair-write "
                "span is an observed wall-clock bound, while last_attempt_elapsed_sec "
                "covers only the final idempotent completion attempt."
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/preference_response_four_topology_v2.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--limit-originals", type=int)
    args = parser.parse_args()
    result = run_experiment(
        args.config,
        output_override=args.output_dir,
        limit_originals=args.limit_originals,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
