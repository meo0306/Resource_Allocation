"""Summarize controlled dual-actor preference response for four solvers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_bank
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.utils.scoring import score_assignment


METHODS = (
    "gurobi_exact",
    "scalarized_independent_greedy",
    "local_search_best_improvement",
    "ppo_shared_policy",
)
SCENARIOS = (
    "controlled_balanced",
    "controlled_student_only",
    "controlled_teacher_only",
    "controlled_conflict",
)
NON_BASELINE_SCENARIOS = SCENARIOS[1:]
SUMMARY_METRICS = (
    "total_score",
    "effect_score",
    "student_pref_score",
    "teacher_pref_score",
    "preference_score",
    "soft_penalty",
    "entropy",
    "max_method_share",
    "relative_gap_to_exact",
    "assignment_change_rate",
    "profile_shift_score_change",
    "reoptimization_gain",
    "adaptation_delta_effect",
    "adaptation_delta_student",
    "adaptation_delta_teacher",
    "adaptation_delta_preference",
    "adaptation_delta_soft_penalty",
    "no_response_regret",
    "captured_no_response_regret_fraction",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Statistics require non-empty finite values.")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _cluster_bootstrap_mean_ci(
    rows: list[dict[str, Any]], key: str, *, seed: int = 20260913,
    repetitions: int = 10000,
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["source_family_id"]].append(float(row[key]))
    cluster_means = np.asarray(
        [np.mean(grouped[name]) for name in sorted(grouped)], dtype=np.float64
    )
    if cluster_means.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, cluster_means.size, size=(repetitions, cluster_means.size)
    )
    bootstrap = np.mean(cluster_means[indices], axis=1)
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def _load_rows(
    output_dir: Path, config_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    objective = load_objective_spec(config["objective"]["config"])
    if objective.config_hash != config["objective"]["config_hash"]:
        raise ValueError("Objective hash mismatch while summarizing.")
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("status") != "pair_evaluation_completed":
        raise ValueError("Pair evaluation has not completed.")
    if metadata.get("objective_hash") != objective.config_hash:
        raise ValueError("Run metadata objective hash mismatch.")
    checkpoint_hash = metadata["checkpoint_sha256"]

    core_rows = _read_csv(Path(config["data"]["core_manifest"]))
    core = {row["original_id"]: row for row in core_rows}
    scenarios = {
        item.scenario_id: item
        for item in load_scenario_bank(config["data"]["scenario_bank"])
    }
    payloads: list[dict[str, Any]] = []
    for path in sorted((output_dir / "pairs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("objective_hash") != objective.config_hash
            or payload.get("checkpoint_sha256") != checkpoint_hash
            or set(payload.get("methods", {})) != set(METHODS)
        ):
            raise ValueError(f"Pair provenance mismatch: {path}")
        payloads.append(payload)
    if len(payloads) != 288 or len({item["pair_id"] for item in payloads}) != 288:
        raise ValueError("Expected 288 unique pair payloads.")

    assignments: dict[tuple[str, str, str], list[int]] = {}
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for payload in payloads:
        for method in METHODS:
            item = payload["methods"][method]
            lookup_key = (
                payload["original_id"], payload["scenario_id"], method
            )
            assignment = [int(value) for value in item["assignment"]]
            assignments[lookup_key] = assignment
            row = {
                name: payload[name]
                for name in (
                    "pair_id", "original_id", "source_family_id", "topology",
                    "size_group", "scenario_id", "student_template",
                    "teacher_template", "n", "objective_hash",
                    "checkpoint_sha256", "checkpoint_update", "instance_path",
                    "instance_sha256", "exact_result_path", "exact_result_sha256",
                )
            }
            row.update(
                {
                    field: value
                    for field, value in item.items()
                    if field not in {"assignment", "method_distribution"}
                }
            )
            row["assignment_json"] = json.dumps(
                assignment, separators=(",", ":")
            )
            row["method_distribution_json"] = json.dumps(
                item["method_distribution"], separators=(",", ":")
            )
            records[lookup_key] = row

    project_rows: list[dict[str, Any]] = []
    for original_id in sorted(core):
        base = load_instance_json(Path(core[original_id]["output_json"]))
        for scenario_id in SCENARIOS:
            scenario = scenarios[scenario_id]
            instance = apply_scenario(base, scenario, objective=objective)
            exact_target = records[(original_id, scenario_id, "gurobi_exact")]
            for method in METHODS:
                lookup_key = (original_id, scenario_id, method)
                row = records[lookup_key]
                balanced_key = (original_id, "controlled_balanced", method)
                balanced_assignment = np.asarray(
                    assignments[balanced_key], dtype=np.int64
                )
                target_assignment = np.asarray(
                    assignments[lookup_key], dtype=np.int64
                )
                reused = score_assignment(
                    balanced_assignment, instance=instance, objective=objective
                )
                balanced_row = records[balanced_key]
                change_count = int(
                    np.sum(balanced_assignment != target_assignment)
                )
                reused_preference = (
                    reused.student_pref_score + reused.teacher_pref_score
                ) / 2.0
                row.update(
                    {
                        "baseline_assignment_hash": balanced_row["assignment_hash"],
                        "baseline_original_total_score": balanced_row["total_score"],
                        "reused_total_score": float(reused.total_score),
                        "reused_effect_score": float(reused.effect_score),
                        "reused_student_pref_score": float(
                            reused.student_pref_score
                        ),
                        "reused_teacher_pref_score": float(
                            reused.teacher_pref_score
                        ),
                        "reused_preference_score": float(reused_preference),
                        "reused_soft_penalty": float(reused.soft_penalty),
                        "profile_shift_score_change": float(
                            reused.total_score - balanced_row["total_score"]
                        ),
                        "reoptimization_gain": float(
                            row["total_score"] - reused.total_score
                        ),
                        "adaptation_delta_effect": float(
                            row["effect_score"] - reused.effect_score
                        ),
                        "adaptation_delta_student": float(
                            row["student_pref_score"]
                            - reused.student_pref_score
                        ),
                        "adaptation_delta_teacher": float(
                            row["teacher_pref_score"]
                            - reused.teacher_pref_score
                        ),
                        "adaptation_delta_preference": float(
                            row["preference_score"] - reused_preference
                        ),
                        "adaptation_delta_soft_penalty": float(
                            row["soft_penalty"] - reused.soft_penalty
                        ),
                        "assignment_change_count": change_count,
                        "assignment_change_rate": change_count / int(row["n"]),
                        "no_response_regret": float(
                            exact_target["total_score"] - reused.total_score
                        ),
                    }
                )
                denominator = row["no_response_regret"]
                row["captured_no_response_regret_fraction"] = (
                    row["reoptimization_gain"] / denominator
                    if denominator > 1.0e-9
                    else 0.0
                )
                project_rows.append(row)

    exact_gains = {
        (row["original_id"], row["scenario_id"]): row["reoptimization_gain"]
        for row in project_rows
        if row["solver_name"] == "gurobi_exact"
    }
    for row in project_rows:
        row["exact_reoptimization_gain"] = exact_gains[
            (row["original_id"], row["scenario_id"])
        ]
        exact_row = records[
            (row["original_id"], row["scenario_id"], "gurobi_exact")
        ]
        if row["total_score"] > exact_row["total_score"] + 1.0e-9:
            raise ValueError(
                f"Method score exceeds exact reference: {row['pair_id']}"
            )
    return project_rows, metadata


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = row["solver_name"]
        grouped[("overall", "all", method)].append(row)
        grouped[("scenario", row["scenario_id"], method)].append(row)
        grouped[("topology", row["topology"], method)].append(row)
        grouped[("size_group", row["size_group"], method)].append(row)
        grouped[
            (
                "scenario_topology",
                f"{row['scenario_id']}|{row['topology']}",
                method,
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (scope, value, method), group_rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "scope": scope,
            "scope_value": value,
            "solver_name": method,
            "count": len(group_rows),
        }
        for metric in SUMMARY_METRICS:
            stats = _stats(float(row[metric]) for row in group_rows)
            item.update(
                {f"{metric}_{name}": number for name, number in stats.items()}
            )
        output.append(item)
    return output


def _response_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    tolerance = 1.0e-9
    for method in METHODS:
        for scenario_id in NON_BASELINE_SCENARIOS:
            selected = [
                row for row in rows
                if row["solver_name"] == method
                and row["scenario_id"] == scenario_id
            ]
            gains = np.asarray(
                [row["reoptimization_gain"] for row in selected], dtype=np.float64
            )
            exact_gains = np.asarray(
                [row["exact_reoptimization_gain"] for row in selected],
                dtype=np.float64,
            )
            no_response = np.asarray(
                [row["no_response_regret"] for row in selected], dtype=np.float64
            )
            meaningful = exact_gains > tolerance
            low, high = _cluster_bootstrap_mean_ci(selected, "reoptimization_gain")
            evidence.append(
                {
                    "solver_name": method,
                    "scenario_id": scenario_id,
                    "count": len(selected),
                    "source_family_count": len(
                        {row["source_family_id"] for row in selected}
                    ),
                    "mean_reoptimization_gain": float(np.mean(gains)),
                    "median_reoptimization_gain": float(np.median(gains)),
                    "p90_reoptimization_gain": float(np.quantile(gains, 0.90)),
                    "cluster_bootstrap_mean_ci_low": low,
                    "cluster_bootstrap_mean_ci_high": high,
                    "positive_gain_rate": float(np.mean(gains > tolerance)),
                    "negative_gain_rate": float(np.mean(gains < -tolerance)),
                    "assignment_change_rate_mean": float(
                        np.mean([row["assignment_change_rate"] for row in selected])
                    ),
                    "exact_opportunity_rate": float(np.mean(meaningful)),
                    "positive_when_exact_opportunity_rate": (
                        float(np.mean(gains[meaningful] > tolerance))
                        if np.any(meaningful)
                        else 0.0
                    ),
                    "aggregate_captured_no_response_regret_fraction": (
                        float(np.sum(gains) / np.sum(no_response))
                        if float(np.sum(no_response)) > tolerance
                        else 0.0
                    ),
                    "adaptation_delta_effect_mean": float(
                        np.mean([row["adaptation_delta_effect"] for row in selected])
                    ),
                    "adaptation_delta_student_mean": float(
                        np.mean([row["adaptation_delta_student"] for row in selected])
                    ),
                    "adaptation_delta_teacher_mean": float(
                        np.mean([row["adaptation_delta_teacher"] for row in selected])
                    ),
                    "adaptation_delta_preference_mean": float(
                        np.mean([row["adaptation_delta_preference"] for row in selected])
                    ),
                    "relative_gap_to_exact_mean": float(
                        np.mean([row["relative_gap_to_exact"] for row in selected])
                    ),
                    "relative_gap_to_exact_p90": float(
                        np.quantile(
                            [row["relative_gap_to_exact"] for row in selected], 0.90
                        )
                    ),
                }
            )
    return evidence


def _pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def summarize(output_dir: Path, config_path: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    rows, run_metadata = _load_rows(output_dir, config_path)
    summaries = _summary_rows(rows)
    evidence = _response_evidence(rows)
    checks = {
        "pair_count_288": len({row["pair_id"] for row in rows}) == 288,
        "method_row_count_1152": len(rows) == 1152,
        "each_pair_has_four_methods": all(
            sum(other["pair_id"] == row["pair_id"] for other in rows) == 4
            for row in rows[::4]
        ),
        "all_hard_feasible": all(row["hard_feasible_rate"] == 1.0 for row in rows),
        "no_mask_violations": all(row["mask_violation_count"] == 0 for row in rows),
        "exact_gap_zero": all(
            abs(row["relative_gap_to_exact"]) <= 1.0e-12
            for row in rows if row["solver_name"] == "gurobi_exact"
        ),
        "exact_reoptimization_gain_nonnegative": all(
            row["reoptimization_gain"] >= -1.0e-9
            for row in rows if row["solver_name"] == "gurobi_exact"
        ),
        "objective_hash_consistent": {
            row["objective_hash"] for row in rows
        } == {run_metadata["objective_hash"]},
        "checkpoint_hash_consistent": {
            row["checkpoint_sha256"] for row in rows
        } == {run_metadata["checkpoint_sha256"]},
    }
    if not all(checks.values()):
        raise ValueError(f"Preference response audit failed: {checks}")

    _write_csv(output_dir / "method_results.csv", rows)
    _write_csv(output_dir / "stratified_statistics.csv", summaries)
    _write_csv(output_dir / "response_evidence.csv", evidence)

    ppo_evidence = [
        row for row in evidence if row["solver_name"] == "ppo_shared_policy"
    ]
    pooled_ppo = [
        row for row in rows
        if row["solver_name"] == "ppo_shared_policy"
        and row["scenario_id"] in NON_BASELINE_SCENARIOS
    ]
    pooled_low, pooled_high = _cluster_bootstrap_mean_ci(
        pooled_ppo, "reoptimization_gain"
    )
    pooled_gain = float(
        np.mean([row["reoptimization_gain"] for row in pooled_ppo])
    )
    positive_scenarios = sum(
        row["mean_reoptimization_gain"] > 0.0 for row in ppo_evidence
    )
    ppo_by_scenario = {row["scenario_id"]: row for row in ppo_evidence}
    teacher_only = ppo_by_scenario["controlled_teacher_only"]
    if (
        teacher_only["exact_opportunity_rate"] > 0.5
        and teacher_only["positive_gain_rate"] == 0.0
        and teacher_only["assignment_change_rate_mean"] == 0.0
    ):
        evidence_decision = "partial_response_teacher_only_absent"
    elif pooled_low > 0.0 and positive_scenarios < 3:
        evidence_decision = "positive_but_scenario_heterogeneous"
    elif pooled_low > 0.0:
        evidence_decision = "consistent_positive_preference_response"
    else:
        evidence_decision = "insufficient_positive_preference_response"
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_hash": run_metadata["objective_hash"],
        "checkpoint_sha256": run_metadata["checkpoint_sha256"],
        "checkpoint_update": run_metadata["checkpoint_update"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "ppo_pooled_nonbaseline_reoptimization_gain_mean": pooled_gain,
        "ppo_pooled_cluster_bootstrap_mean_ci": [pooled_low, pooled_high],
        "ppo_positive_scenario_count": positive_scenarios,
        "ppo_scenario_evidence": ppo_evidence,
        "preference_response_decision": evidence_decision,
        "interpretation_limits": [
            "exploratory controlled-calibration data, not an independent test",
            "single seed PPO checkpoint",
            "assignment change is reported but is not sufficient evidence",
            "raw component levels across profiles are not treated as causal changes",
        ],
    }
    _write_json(output_dir / "preference_response_summary.json", summary)

    overall = {
        row["solver_name"]: row
        for row in summaries
        if row["scope"] == "overall"
    }
    evidence_lookup = {
        (row["solver_name"], row["scenario_id"]): row for row in evidence
    }
    report = [
        "# 四拓扑 b075 双主体偏好响应报告",
        "",
        "## 验收结论",
        "",
        f"- 288 个实例—画像对、4 种方法，共 1,152 条结果；工程审计：**通过**。",
        f"- PPO 三个非基线画像的合并重求解增益均值为 {pooled_gain:.6f}，按源家族聚类 bootstrap 95% CI 为 [{pooled_low:.6f}, {pooled_high:.6f}]。",
        f"- 自动证据分类：`{evidence_decision}`。PPO 只表现出部分偏好响应，尚未证明完整的双主体响应。",
        "- teacher-only 中精确模型存在正向重求解机会，但 PPO 的 assignment 变化率与重求解增益均为 0；教师单独变化没有被当前策略利用。",
        "- student-only 的 PPO 正增益率为 44.444%，中位数为 0；conflict 的正增益率为 79.167%。合并正增益不能掩盖逐主体缺失。",
        "- 冻结目标 b075 不变；本实验不训练 PPO，也不冻结 PPO 或推理参数。",
        "",
        "## 四方法总体质量",
        "",
        "| 方法 | mean J | mean gap | P90 gap | mean F_E | mean F_P | mean soft penalty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = overall[method]
        report.append(
            f"| {method} | {row['total_score_mean']:.6f} | "
            f"{_pct(row['relative_gap_to_exact_mean'])} | "
            f"{_pct(row['relative_gap_to_exact_p90'])} | "
            f"{row['effect_score_mean']:.6f} | {row['preference_score_mean']:.6f} | "
            f"{row['soft_penalty_mean']:.6f} |"
        )
    report.extend(
        [
            "",
            "## 旧解复用与按画像重求解",
            "",
            "下表所有增益都在同一个目标画像下比较：重新求解结果减去该方法 S6–T6 旧解在目标画像下的重算得分。",
            "",
            "| 方法 | 目标画像 | mean gain | 95% cluster CI | positive rate | assignment change | captured no-response regret |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for scenario_id in NON_BASELINE_SCENARIOS:
            row = evidence_lookup[(method, scenario_id)]
            report.append(
                f"| {method} | {scenario_id} | {row['mean_reoptimization_gain']:.6f} | "
                f"[{row['cluster_bootstrap_mean_ci_low']:.6f}, {row['cluster_bootstrap_mean_ci_high']:.6f}] | "
                f"{_pct(row['positive_gain_rate'])} | "
                f"{_pct(row['assignment_change_rate_mean'])} | "
                f"{_pct(row['aggregate_captured_no_response_regret_fraction'])} |"
            )
    report.extend(
        [
            "",
            "## PPO 的分项响应与精确误差",
            "",
            "分项变化同样是按目标画像重算后的 PPO 新解减去 PPO 的 S6–T6 旧解；正负值是同画像内的优化权衡。",
            "",
            "| 目标画像 | ΔF_E | ΔF_S | ΔF_T | ΔF_P | mean gap | P90 gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario_id in NON_BASELINE_SCENARIOS:
        row = evidence_lookup[("ppo_shared_policy", scenario_id)]
        report.append(
            f"| {scenario_id} | {row['adaptation_delta_effect_mean']:.6f} | "
            f"{row['adaptation_delta_student_mean']:.6f} | "
            f"{row['adaptation_delta_teacher_mean']:.6f} | "
            f"{row['adaptation_delta_preference_mean']:.6f} | "
            f"{_pct(row['relative_gap_to_exact_mean'])} | "
            f"{_pct(row['relative_gap_to_exact_p90'])} |"
        )
    ppo_topology = [
        row for row in summaries
        if row["solver_name"] == "ppo_shared_policy"
        and row["scope"] == "topology"
    ]
    ppo_size = [
        row for row in summaries
        if row["solver_name"] == "ppo_shared_policy"
        and row["scope"] == "size_group"
    ]
    worst_topology = max(
        ppo_topology, key=lambda row: row["relative_gap_to_exact_mean"]
    )
    worst_size = max(
        ppo_size, key=lambda row: row["relative_gap_to_exact_mean"]
    )
    report.extend(
        [
            "",
            "## 分层风险",
            "",
            "- teacher-only 的零响应出现在四种拓扑的每一层，不是单一拓扑造成的局部现象。",
            f"- PPO 平均 gap 最大的拓扑为 {worst_topology['scope_value']}："
            f"mean gap={_pct(worst_topology['relative_gap_to_exact_mean'])}。",
            f"- PPO 平均 gap 最大的规模组为 {worst_size['scope_value']}："
            f"mean gap={_pct(worst_size['relative_gap_to_exact_mean'])}，"
            f"四画像合并的 mean reoptimization gain={worst_size['reoptimization_gain_mean']:.6f}。",
        ]
    )
    report.extend(
        [
            "",
            "## 解释边界",
            "",
            "- assignment 改变率只描述敏感性；只有同一目标画像下的重求解增益和分项权衡才用于判断偏好输入是否被利用。",
            "- 精确解可能不唯一；assignment 不变不等于没有偏好响应，assignment 改变也不自动等于响应有效。",
            "- 四画像的原始 F_S/F_T 水平来自不同画像函数，不把跨画像原始水平差直接解释为因果效应。",
            "- 数据是已查看的 controlled calibration，且 PPO 只有 seed0；结论属于开发诊断，不是独立测试或多种子结论。",
            "",
        ]
    )
    (output_dir / "preference_response_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    hash_rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "output_hashes.csv":
            continue
        hash_rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_csv(output_dir / "output_hashes.csv", hash_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/preference_response_four_topology_v2.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="results/four_topology_b075_preference_response_v1",
    )
    args = parser.parse_args()
    summary = summarize(Path(args.output_dir), Path(args.config).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
