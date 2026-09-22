"""Remeasure Gurobi with model construction included and merge fair runtimes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_bank
from pa_moap_rl.experiments.compare_exact_hybrid import _read_manifest
from pa_moap_rl.solvers.gurobi_exact_solver import solve_gurobi


def remeasure(
    *,
    paired_csv: str | Path,
    manifest: str | Path,
    scenario_bank: str | Path,
    output_dir: str | Path,
    time_limit: float = 120.0,
    seed: int = 0,
) -> pd.DataFrame:
    frame = pd.read_csv(paired_csv)
    exact_rows = frame[frame["solver_name"] == "gurobi_exact_or_bounded"].copy()
    records = {int(row["manifest_index"]): row for row in _read_manifest(manifest)}
    scenarios = load_scenario_bank(scenario_bank)
    config = load_config()
    measurements: list[dict] = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for position, (_, old_row) in enumerate(exact_rows.iterrows(), start=1):
        manifest_index = int(old_row["manifest_index"])
        record = records[manifest_index]
        scenario = scenarios[manifest_index % len(scenarios)]
        instance = apply_scenario(
            load_instance_json(record["output_json"]), scenario, config
        )
        start = time.perf_counter()
        result = solve_gurobi(
            instance,
            time_limit=time_limit,
            mip_gap=0.0,
            seed=seed,
            output_flag=False,
        )
        end_to_end_runtime = time.perf_counter() - start
        old_score = float(old_row["total_score"])
        if abs(result.score.total_score - old_score) > 1.0e-8:
            raise AssertionError(
                f"Remeasured optimum differs for {record['instance_uid']}: "
                f"{result.score.total_score} versus {old_score}."
            )
        measurement = {
            "manifest_index": manifest_index,
            "instance_uid": record["instance_uid"],
            "topology": record["topology"],
            "group": record["group"],
            "n": int(record["n"]),
            "scenario_id": scenario.scenario_id,
            "total_score": result.score.total_score,
            "end_to_end_runtime": end_to_end_runtime,
            "optimize_runtime": float(result.metrics["solver_runtime"]),
            "model_build_and_overhead_runtime": max(
                0.0, end_to_end_runtime - float(result.metrics["solver_runtime"])
            ),
            "proven_optimal": bool(result.metrics["proven_optimal"]),
            "mip_gap": float(result.metrics["mip_gap"]),
            "objective_recompute_error": float(
                result.metrics["objective_recompute_error"]
            ),
        }
        measurements.append(measurement)
        pd.DataFrame(measurements).to_csv(
            output / "gurobi_runtime_remeasurement.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            json.dumps(
                {
                    "completed": position,
                    "total": len(exact_rows),
                    "instance_uid": record["instance_uid"],
                    "end_to_end_runtime": end_to_end_runtime,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    measured = pd.DataFrame(measurements)
    lookup = measured.set_index(["manifest_index", "scenario_id"])
    corrected = frame.copy()
    corrected["runtime_definition"] = "complete_solver_call"
    corrected["gurobi_optimize_runtime"] = pd.NA
    corrected["gurobi_model_build_and_overhead_runtime"] = pd.NA
    exact_mask = corrected["solver_name"] == "gurobi_exact_or_bounded"
    for index in corrected.index[exact_mask]:
        key = (
            int(corrected.at[index, "manifest_index"]),
            corrected.at[index, "scenario_id"],
        )
        value = lookup.loc[key]
        corrected.at[index, "runtime"] = float(value["end_to_end_runtime"])
        corrected.at[index, "gurobi_optimize_runtime"] = float(
            value["optimize_runtime"]
        )
        corrected.at[index, "gurobi_model_build_and_overhead_runtime"] = float(
            value["model_build_and_overhead_runtime"]
        )
    corrected.to_csv(
        output / "paired_results_fair_runtime.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return measured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-csv",
        default="results/exact_hybrid_validation_pilot/paired_results.csv",
    )
    parser.add_argument("--manifest", default="data_processed/splits/validation.csv")
    parser.add_argument(
        "--scenario-bank",
        default="data_processed/splits/scenarios/validation_interpolation.json",
    )
    parser.add_argument(
        "--output-dir", default="results/exact_hybrid_validation_pilot/fair_runtime"
    )
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    measurements = remeasure(
        paired_csv=args.paired_csv,
        manifest=args.manifest,
        scenario_bank=args.scenario_bank,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        seed=args.seed,
    )
    print(json.dumps({"rows": len(measurements)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
