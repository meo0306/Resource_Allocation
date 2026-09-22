"""One-instance end-to-end smoke test for the exact/hybrid comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_bank
from pa_moap_rl.experiments.compare_exact_hybrid import (
    _read_manifest,
    solve_deterministic_policy,
)
from pa_moap_rl.solvers.gurobi_exact_solver import solve_gurobi
from pa_moap_rl.solvers.hybrid_solver import ppo_plus_short_local_search
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.ppo_solver import PPOConfig, build_actor_critic, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="results/formal_joint_seed1_run01/checkpoints/best.pt",
    )
    parser.add_argument("--manifest", default="data_processed/splits/validation.csv")
    parser.add_argument(
        "--scenario-bank",
        default="data_processed/splits/scenarios/validation_interpolation.json",
    )
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/exact_hybrid_smoke/result.json")
    args = parser.parse_args()

    project_config = load_config()
    records = _read_manifest(args.manifest)
    record = min(records, key=lambda item: (int(item["n"]), int(item["manifest_index"])))
    scenarios = load_scenario_bank(args.scenario_bank)
    scenario = scenarios[int(record["manifest_index"]) % len(scenarios)]
    instance = apply_scenario(
        load_instance_json(record["output_json"]), scenario, project_config
    )

    device = resolve_device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    ppo_config = PPOConfig(**checkpoint["ppo_config"])
    model = build_actor_critic(
        instance,
        config=project_config,
        device=device,
        hidden_dim=checkpoint.get("hidden_dim"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ppo = solve_deterministic_policy(
        instance, model, ppo_config=ppo_config, project_config=project_config
    )
    local = local_search_best_improvement(instance)
    hybrid = ppo_plus_short_local_search(
        instance, ppo.assignment, accepted_move_budget=4, ppo_runtime=ppo.runtime
    )
    exact = solve_gurobi(instance, time_limit=args.time_limit, output_flag=False)
    if hybrid.score.total_score < ppo.score.total_score - 1.0e-10:
        raise AssertionError("Hybrid refinement reduced the PPO objective.")
    if exact.metrics["objective_bound"] < exact.score.total_score - 1.0e-8:
        raise AssertionError("Gurobi objective bound is below its incumbent.")
    if exact.metrics["objective_recompute_error"] > 1.0e-8:
        raise AssertionError("Gurobi and scoring.py objectives disagree.")

    result = {
        "manifest_index": int(record["manifest_index"]),
        "instance_uid": record["instance_uid"],
        "topology": record["topology"],
        "group": record["group"],
        "n": int(record["n"]),
        "scenario_id": scenario.scenario_id,
        "scores": {
            "ppo": ppo.score.total_score,
            "one_point_best_improvement_local_search": local.score.total_score,
            "ppo_plus_short_local_search_4": hybrid.score.total_score,
            "gurobi_incumbent": exact.score.total_score,
            "gurobi_bound": exact.metrics["objective_bound"],
        },
        "runtimes": {
            "ppo": ppo.runtime,
            "one_point_best_improvement_local_search": local.runtime,
            "ppo_plus_short_local_search_4": hybrid.runtime,
            "gurobi": exact.runtime,
        },
        "gurobi_status": exact.metrics["gurobi_status"],
        "gurobi_mip_gap": exact.metrics["mip_gap"],
        "objective_recompute_error": exact.metrics["objective_recompute_error"],
        "all_hard_feasible": all(
            result_item.metrics["hard_feasible_rate"] == 1.0
            for result_item in (ppo, local, hybrid, exact)
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
