"""Summarize the exact/hybrid paired comparison and select a move budget."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_NAME = "one_point_best_improvement_local_search"


def _p90(series: pd.Series) -> float:
    return float(np.quantile(series.astype(float), 0.90))


def summarize(input_csv: str | Path, output_dir: str | Path) -> dict:
    frame = pd.read_csv(input_csv)
    required = {
        "instance_uid",
        "scenario_id",
        "solver_name",
        "total_score",
        "runtime",
        "hard_feasible_rate",
        "gap_to_gurobi_incumbent",
        "worst_case_gap_to_gurobi_bound",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Comparison CSV lacks columns: {', '.join(missing)}")

    summary = (
        frame.groupby("solver_name", sort=False)
        .agg(
            cases=("total_score", "size"),
            score_mean=("total_score", "mean"),
            score_std=("total_score", "std"),
            score_min=("total_score", "min"),
            runtime_mean=("runtime", "mean"),
            runtime_median=("runtime", "median"),
            runtime_p90=("runtime", _p90),
            runtime_max=("runtime", "max"),
            feasible_rate=("hard_feasible_rate", "mean"),
            gap_lower_mean=("gap_to_gurobi_incumbent", "mean"),
            gap_upper_mean=("worst_case_gap_to_gurobi_bound", "mean"),
        )
        .reset_index()
    )
    topology = (
        frame.groupby(["topology", "solver_name"], sort=False)
        .agg(
            cases=("total_score", "size"),
            score_mean=("total_score", "mean"),
            score_std=("total_score", "std"),
            runtime_mean=("runtime", "mean"),
            gap_lower_mean=("gap_to_gurobi_incumbent", "mean"),
            gap_upper_mean=("worst_case_gap_to_gurobi_bound", "mean"),
        )
        .reset_index()
    )

    exact_rows = frame[frame["solver_name"] == "gurobi_exact_or_bounded"]
    if exact_rows.empty:
        raise ValueError("No gurobi_exact_or_bounded rows found.")
    exact_stats = {
        "cases": int(len(exact_rows)),
        "proven_optimal_rate": float(exact_rows["gurobi_proven_optimal"].astype(bool).mean()),
        "mip_gap_mean": float(exact_rows["mip_gap"].mean()),
        "mip_gap_p90": _p90(exact_rows["mip_gap"]),
        "two_stage_rate": float((exact_rows["gurobi_stages"] == 2).mean()),
        "objective_recompute_error_max": float(exact_rows["objective_recompute_error"].max()),
    }

    means = summary.set_index("solver_name")
    if "ppo" not in means.index or LOCAL_NAME not in means.index:
        raise ValueError("Both PPO and canonical local-search rows are required.")
    ppo_score = float(means.loc["ppo", "score_mean"])
    local_score = float(means.loc[LOCAL_NAME, "score_mean"])
    local_runtime = float(means.loc[LOCAL_NAME, "runtime_mean"])
    full_gain = local_score - ppo_score
    candidates = []
    pattern = re.compile(r"ppo_plus_short_local_search_(\d+)$")
    for solver_name, row in means.iterrows():
        match = pattern.match(str(solver_name))
        if not match:
            continue
        recovered = (
            (float(row["score_mean"]) - ppo_score) / full_gain
            if full_gain > 1.0e-12
            else 1.0
        )
        runtime_ratio = (
            float(row["runtime_mean"]) / local_runtime
            if local_runtime > 1.0e-12
            else float("inf")
        )
        candidates.append(
            {
                "solver_name": solver_name,
                "accepted_move_budget": int(match.group(1)),
                "score_mean": float(row["score_mean"]),
                "runtime_mean": float(row["runtime_mean"]),
                "recovered_local_search_gain": float(recovered),
                "runtime_ratio_to_full_local_search": float(runtime_ratio),
                "feasible_rate": float(row["feasible_rate"]),
                "eligible": bool(
                    float(row["feasible_rate"]) == 1.0
                    and recovered >= 0.50
                    and runtime_ratio <= 0.20
                ),
            }
        )
    candidates.sort(key=lambda row: row["accepted_move_budget"])
    eligible = [row for row in candidates if row["eligible"]]
    selected = eligible[0] if eligible else None
    decision = {
        "full_local_search_gain_over_ppo": float(full_gain),
        "selection_rule": "minimum budget with >=50% gain recovery and <=20% runtime",
        "selected_budget": None if selected is None else selected["accepted_move_budget"],
        "selection_status": "selected" if selected is not None else "no_candidate_meets_gate",
        "candidates": candidates,
        "gurobi": exact_stats,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "algorithm_summary.csv", index=False, encoding="utf-8-sig")
    topology.to_csv(output / "topology_summary.csv", index=False, encoding="utf-8-sig")
    (output / "hybrid_budget_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_text = (
        f"选择 accepted_move_budget={selected['accepted_move_budget']}。"
        if selected is not None
        else "没有候选同时满足质量恢复与时间门禁；暂不冻结混合预算。"
    )
    report = f"""# 精确界与混合精炼 pilot 决策报告

## Gurobi 可解性

- 算例数：{exact_stats['cases']}
- 证得最优比例：{exact_stats['proven_optimal_rate']:.2%}
- MIP gap 均值 / P90：{exact_stats['mip_gap_mean']:.6f} / {exact_stats['mip_gap_p90']:.6f}
- 进入 600 秒复核比例：{exact_stats['two_stage_rate']:.2%}
- 目标重算误差最大值：{exact_stats['objective_recompute_error_max']:.3e}

## 混合预算

- 完整单点局部搜索相对 PPO 的平均增益：{full_gain:.6f}
- 决策：{selected_text}

该结论仅来自固定 validation pilot，不得据此陈述测试集性能。详细逐算法与逐拓扑结果见同目录 CSV。
"""
    (output / "decision_report.md").write_text(report, encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="results/exact_hybrid_validation_pilot/paired_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/exact_hybrid_validation_pilot/summary",
    )
    args = parser.parse_args()
    decision = summarize(args.input_csv, args.output_dir)
    print(json.dumps(decision, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
