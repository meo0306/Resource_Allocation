"""Compare PPO, one-point local search, Gurobi, and PPO+short search.

The default command runs the agreed 45-instance validation pilot: one fixed
instance from each of 5 topologies x 9 size groups.  It never reads the test
split unless the caller explicitly supplies a different manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_bank
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.solvers.gurobi_exact_solver import solve_gurobi
from pa_moap_rl.solvers.hybrid_solver import ppo_plus_short_local_search
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.ppo_solver import (
    PPOConfig,
    build_actor_critic,
    observation_to_model_batch,
    resolve_device,
)
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result


TOPOLOGY_ORDER = ("uniform", "random", "staircase", "hourglass", "bottleneck")
CANONICAL_LOCAL_SEARCH_NAME = "one_point_best_improvement_local_search"


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "output_json" not in rows[0]:
        raise ValueError("Manifest must be a non-empty CSV with an output_json column.")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = dict(row)
        record["manifest_index"] = index
        record["n"] = int(row["n"])
        records.append(record)
    return records


def select_validation_pilot(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one median-N instance for every topology x size-group cell."""

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for topology in TOPOLOGY_ORDER:
        topology_rows = [
            row for row in records if str(row.get("topology", "")).lower() == topology
        ]
        for group_index in range(1, 10):
            group = f"group{group_index}"
            candidates = [row for row in topology_rows if row.get("group") == group]
            if not candidates:
                missing.append(f"{topology}/{group}")
                continue
            candidates.sort(key=lambda row: (int(row["n"]), str(row["instance_name"])))
            selected.append(candidates[len(candidates) // 2])
    if missing:
        raise ValueError(f"Validation manifest lacks pilot cells: {', '.join(missing)}")
    if len(selected) != 45:
        raise AssertionError(f"Expected 45 pilot records, got {len(selected)}.")
    return selected


def solve_deterministic_policy(
    instance,
    model,
    *,
    ppo_config: PPOConfig,
    project_config,
) -> SolverResult:
    """Run the shared PPO policy greedily and return the best visited solution."""

    env_settings = project_config.default["environment"]
    env = MethodAssignmentEnv(
        max_steps=int(
            ppo_config.env_max_steps
            if ppo_config.env_max_steps is not None
            else env_settings["max_steps"]
        ),
        patience=int(
            ppo_config.env_patience
            if ppo_config.env_patience is not None
            else env_settings["patience"]
        ),
        improvement_eps=float(env_settings["improvement_eps"]),
        config=project_config,
    )
    observation = env.reset(instance)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    done = False
    steps = 0
    illegal_actions = 0
    while not done:
        with torch.inference_mode():
            output = model(observation_to_model_batch(observation, device=device))
            flat_action = int(torch.argmax(output.masked_logits.reshape(-1)).item())
        node, method = divmod(flat_action, instance.m)
        if not bool(observation["action_mask"][node, method]):
            illegal_actions += 1
        observation, _reward, done, _info = env.step((node, method))
        steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - start
    result = make_solver_result(
        solver_name="ppo",
        instance=instance,
        assignment=env.best_assignment,
        runtime=runtime,
        history=[float(env.best_score)],
    )
    result.metrics.update(
        {
            "policy_steps": steps,
            "illegal_action_count": illegal_actions,
            "deterministic_inference": True,
        }
    )
    return result


def _staged_gurobi(
    instance,
    *,
    initial_time_limit: float,
    retry_time_limit: float,
    acceptable_gap: float,
    seed: int,
    threads: int | None,
    log_file: str | None,
) -> SolverResult:
    first = solve_gurobi(
        instance,
        time_limit=initial_time_limit,
        mip_gap=0.0,
        seed=seed,
        threads=threads,
        log_file=log_file,
        output_flag=bool(log_file),
    )
    should_retry = (
        not bool(first.metrics["proven_optimal"])
        and float(first.metrics["mip_gap"]) > float(acceptable_gap)
    )
    if not should_retry:
        first.metrics.update(
            {
                "gurobi_stages": 1,
                "stage1_status": first.metrics["gurobi_status"],
                "stage1_mip_gap": first.metrics["mip_gap"],
                "stage1_runtime": first.runtime,
            }
        )
        return first

    second = solve_gurobi(
        instance,
        time_limit=retry_time_limit,
        mip_gap=0.0,
        seed=seed,
        threads=threads,
        log_file=log_file,
        output_flag=bool(log_file),
        warm_start=first.assignment,
    )
    combined = make_solver_result(
        solver_name="gurobi_exact_or_bounded",
        instance=instance,
        assignment=second.assignment,
        runtime=first.runtime + second.runtime,
        history=[first.score.total_score, second.score.total_score],
    )
    combined.metrics.update(second.metrics)
    combined.metrics["runtime"] = combined.runtime
    combined.metrics.update(
        {
            "gurobi_stages": 2,
            "stage1_status": first.metrics["gurobi_status"],
            "stage1_mip_gap": first.metrics["mip_gap"],
            "stage1_objective_bound": first.metrics["objective_bound"],
            "stage1_runtime": first.runtime,
            "stage2_runtime": second.runtime,
        }
    )
    return combined


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _result_row(
    record: dict[str, Any],
    scenario_id: str,
    result: SolverResult,
    *,
    solver_name: str | None = None,
) -> dict[str, Any]:
    row = {
        "manifest_index": int(record["manifest_index"]),
        "instance_uid": record.get("instance_uid"),
        "instance_name": record.get("instance_name"),
        "instance_path": record.get("output_json"),
        "topology": str(record.get("topology", "")).lower(),
        "group": record.get("group"),
        "n": int(record["n"]),
        "scenario_id": scenario_id,
        "solver_name": solver_name or result.solver_name,
        "assignment": result.assignment.tolist(),
    }
    row.update({key: _json_value(value) for key, value in result.metrics.items()})
    row["solver_name"] = solver_name or result.solver_name
    return row


def _add_exact_gaps(rows: list[dict[str, Any]], exact: SolverResult) -> None:
    incumbent = float(exact.score.total_score)
    bound = float(exact.metrics["objective_bound"])
    denominator = max(abs(incumbent), 1.0e-12)
    for row in rows:
        score = float(row["total_score"])
        row["gap_to_gurobi_incumbent"] = incumbent - score
        row["worst_case_gap_to_gurobi_bound"] = bound - score
        row["relative_gap_to_gurobi_incumbent"] = (incumbent - score) / denominator
        row["gurobi_incumbent"] = incumbent
        row["gurobi_bound"] = bound
        row["gurobi_proven_optimal"] = bool(exact.metrics["proven_optimal"])


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "paired_results.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row if key != "assignment"})
    with (output_dir / "paired_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(
    *,
    checkpoint_path: str | Path,
    validation_manifest: str | Path,
    scenario_bank: str | Path,
    output_dir: str | Path,
    move_budgets: tuple[int, ...] = (4, 8, 16, 32),
    initial_time_limit: float = 120.0,
    retry_time_limit: float = 600.0,
    acceptable_gap: float = 0.01,
    seed: int = 0,
    threads: int | None = None,
    device: str = "auto",
    pilot: bool = True,
) -> list[dict[str, Any]]:
    project_config = load_config()
    records = _read_manifest(validation_manifest)
    if pilot:
        records = select_validation_pilot(records)
    scenarios = load_scenario_bank(scenario_bank)
    resolved_device = resolve_device(device)
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=resolved_device, weights_only=False
    )
    ppo_config = PPOConfig(**checkpoint["ppo_config"])
    first_instance = load_instance_json(records[0]["output_json"])
    model = build_actor_critic(
        first_instance,
        config=project_config,
        device=resolved_device,
        hidden_dim=checkpoint.get("hidden_dim"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output = Path(output_dir)
    all_rows: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        base_instance = load_instance_json(record["output_json"])
        scenario = scenarios[int(record["manifest_index"]) % len(scenarios)]
        instance = apply_scenario(base_instance, scenario, project_config)

        ppo = solve_deterministic_policy(
            instance, model, ppo_config=ppo_config, project_config=project_config
        )
        local = local_search_best_improvement(instance)
        results: list[tuple[str, SolverResult]] = [
            ("ppo", ppo),
            (CANONICAL_LOCAL_SEARCH_NAME, local),
        ]
        for budget in move_budgets:
            hybrid = ppo_plus_short_local_search(
                instance,
                ppo.assignment,
                accepted_move_budget=budget,
                ppo_runtime=ppo.runtime,
            )
            results.append((hybrid.solver_name, hybrid))

        log_dir = output / "gurobi_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["instance_uid"]))
        exact = _staged_gurobi(
            instance,
            initial_time_limit=initial_time_limit,
            retry_time_limit=retry_time_limit,
            acceptable_gap=acceptable_gap,
            seed=seed,
            threads=threads,
            log_file=str(log_dir / f"{safe_uid}.log"),
        )
        results.append(("gurobi_exact_or_bounded", exact))
        instance_rows = [
            _result_row(record, scenario.scenario_id, result, solver_name=name)
            for name, result in results
        ]
        _add_exact_gaps(instance_rows, exact)
        all_rows.extend(instance_rows)
        _write_outputs(all_rows, output)
        print(
            json.dumps(
                {
                    "completed": position + 1,
                    "total": len(records),
                    "instance_uid": record["instance_uid"],
                    "gurobi_status": exact.metrics["gurobi_status"],
                    "gurobi_gap": exact.metrics["mip_gap"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "validation_manifest": str(Path(validation_manifest).resolve()),
        "scenario_bank": str(Path(scenario_bank).resolve()),
        "pilot": bool(pilot),
        "instance_count": len(records),
        "move_budgets": list(move_budgets),
        "initial_time_limit": float(initial_time_limit),
        "retry_time_limit": float(retry_time_limit),
        "acceptable_gap": float(acceptable_gap),
        "seed": int(seed),
        "threads": threads,
        "device": str(resolved_device),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="results/formal_joint_seed1_run01/checkpoints/best.pt",
    )
    parser.add_argument("--validation-manifest", default="data_processed/splits/validation.csv")
    parser.add_argument(
        "--scenario-bank",
        default="data_processed/splits/scenarios/validation_interpolation.json",
    )
    parser.add_argument("--output-dir", default="results/exact_hybrid_validation_pilot")
    parser.add_argument("--move-budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--initial-time-limit", type=float, default=120.0)
    parser.add_argument("--retry-time-limit", type=float, default=600.0)
    parser.add_argument("--acceptable-gap", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--full-validation", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = run_comparison(
        checkpoint_path=args.checkpoint,
        validation_manifest=args.validation_manifest,
        scenario_bank=args.scenario_bank,
        output_dir=args.output_dir,
        move_budgets=tuple(args.move_budgets),
        initial_time_limit=args.initial_time_limit,
        retry_time_limit=args.retry_time_limit,
        acceptable_gap=args.acceptable_gap,
        seed=args.seed,
        threads=args.threads,
        device=args.device,
        pilot=not args.full_validation,
    )
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
