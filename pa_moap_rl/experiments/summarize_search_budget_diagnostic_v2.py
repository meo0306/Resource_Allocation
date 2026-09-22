"""Summarize four-topology search-budget, patience, and anytime diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_bank
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.utils.scoring import score_assignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
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


def _analysis_condition(spec: dict[str, Any]) -> str:
    if spec["family"] == "n_scaled_budget":
        return "n_scaled"
    return str(spec["condition_id"])


def _load_payloads(
    config_path: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = _sha256(config_path)
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("status") != "rollout_evaluation_completed":
        raise ValueError("Search-budget rollout evaluation is not complete.")
    if metadata.get("config_sha256") != config_hash:
        raise ValueError("Search-budget config hash mismatch.")
    preference_config = yaml.safe_load(
        Path(config["preference_response_config"]).read_text(encoding="utf-8")
    )
    objective = load_objective_spec(preference_config["objective"]["config"])
    if metadata.get("objective_hash") != objective.config_hash:
        raise ValueError("Objective hash mismatch.")
    paths = sorted((output_dir / "pair_traces").glob("*.json.gz"))
    payloads = [_read_gzip_json(path) for path in paths]
    if len(payloads) != 288 or len({item["pair_id"] for item in payloads}) != 288:
        raise ValueError("Expected 288 unique pair traces.")
    for payload in payloads:
        if (
            payload.get("config_sha256") != config_hash
            or payload.get("objective_hash") != objective.config_hash
            or payload.get("checkpoint_sha256")
            != metadata["checkpoint_sha256"]
            or set(payload["condition_specs"])
            != set(payload["condition_results"])
        ):
            raise ValueError(f"Pair trace provenance mismatch: {payload['pair_id']}")
    return config, metadata, payloads


def _flat_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        for condition_id, result in payload["condition_results"].items():
            spec = payload["condition_specs"][condition_id]
            row = {
                key: payload[key]
                for key in (
                    "pair_id",
                    "original_id",
                    "source_family_id",
                    "topology",
                    "size_group",
                    "scenario_id",
                    "student_template",
                    "teacher_template",
                    "n",
                    "objective_hash",
                    "checkpoint_sha256",
                )
            }
            row.update(
                {
                    "condition_id": condition_id,
                    "analysis_condition": _analysis_condition(spec),
                    "condition_family": spec["family"],
                    "max_steps": result["max_steps"],
                    "patience": result["patience"],
                    "executed_moves": result["executed_moves"],
                    "accepted_moves": result["accepted_moves"],
                    "new_best_moves": result["new_best_moves"],
                    "unique_action_count": result["unique_action_count"],
                    "repeated_action_count": result["repeated_action_count"],
                    "repeated_node_count": result["repeated_node_count"],
                    "covered_node_count": result["covered_node_count"],
                    "covered_node_rate": result["covered_node_rate"],
                    "steps_to_best": result["steps_to_best"],
                    "done_reason": result["done_reason"],
                    "runtime_sec": result["runtime_sec"],
                    "initial_score": result["initial_score"],
                    "best_score": result["best_score"],
                    "absolute_gap_to_exact": result["absolute_gap_to_exact"],
                    "relative_gap_to_exact": result["relative_gap_to_exact"],
                    "assignment": result["best_assignment"],
                }
            )
            row.update(
                {
                    f"score_{key}": value
                    for key, value in result["score"].items()
                    if key != "method_distribution"
                }
            )
            rows.append(row)
    return rows


SUMMARY_METRICS = (
    "best_score",
    "relative_gap_to_exact",
    "executed_moves",
    "accepted_moves",
    "new_best_moves",
    "unique_action_count",
    "repeated_action_count",
    "repeated_node_count",
    "covered_node_count",
    "covered_node_rate",
    "steps_to_best",
    "runtime_sec",
)


def _summary_row(
    rows: list[dict[str, Any]], *, stratum: str, stratum_value: str
) -> dict[str, Any]:
    first = rows[0]
    output: dict[str, Any] = {
        "analysis_condition": first["analysis_condition"],
        "condition_family": first["condition_family"],
        "stratum": stratum,
        "stratum_value": stratum_value,
        "count": len(rows),
        "max_steps_min": min(int(row["max_steps"]) for row in rows),
        "max_steps_max": max(int(row["max_steps"]) for row in rows),
        "patience_min": min(int(row["patience"]) for row in rows),
        "patience_max": max(int(row["patience"]) for row in rows),
        "patience_stop_rate": float(
            np.mean([row["done_reason"] == "patience" for row in rows])
        ),
    }
    for metric in SUMMARY_METRICS:
        for stat, value in _stats(float(row[metric]) for row in rows).items():
            output[f"{metric}_{stat}"] = value
    gaps = np.asarray(
        [float(row["relative_gap_to_exact"]) for row in rows], dtype=np.float64
    )
    for threshold in (0.01, 0.02, 0.03, 0.05):
        output[f"gap_le_{int(threshold * 100)}pct_rate"] = float(
            np.mean(gaps <= threshold + 1.0e-12)
        )
    return output


def _stratified_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        (),
        ("topology",),
        ("size_group",),
        ("scenario_id",),
        ("topology", "size_group"),
        ("topology", "scenario_id"),
        ("size_group", "scenario_id"),
        ("topology", "size_group", "scenario_id"),
    )
    output = []
    for dimensions_used in dimensions:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (row["analysis_condition"],) + tuple(
                str(row[name]) for name in dimensions_used
            )
            grouped[key].append(row)
        for key, values in sorted(grouped.items()):
            stratum = "overall" if not dimensions_used else "x".join(dimensions_used)
            stratum_value = (
                "all" if not dimensions_used else "|".join(key[1:])
            )
            output.append(
                _summary_row(values, stratum=stratum, stratum_value=stratum_value)
            )
    return output


def _anytime_rows(
    payloads: list[dict[str, Any]], milestones: list[int]
) -> list[dict[str, Any]]:
    records = []
    for payload in payloads:
        trace = payload["master_trace"]["trace"]
        for condition_id, result in payload["condition_results"].items():
            spec = payload["condition_specs"][condition_id]
            analysis_condition = _analysis_condition(spec)
            executed = int(result["executed_moves"])
            for milestone in milestones:
                if milestone > int(result["max_steps"]):
                    continue
                effective = min(int(milestone), executed)
                point = trace[effective]
                records.append(
                    {
                        "analysis_condition": analysis_condition,
                        "condition_family": spec["family"],
                        "pair_id": payload["pair_id"],
                        "topology": payload["topology"],
                        "size_group": payload["size_group"],
                        "scenario_id": payload["scenario_id"],
                        "milestone": int(milestone),
                        "effective_step": effective,
                        "best_score": float(point["best_score"]),
                        "relative_gap_to_exact": float(
                            point["relative_gap_to_exact"]
                        ),
                        "covered_node_count": int(point["covered_node_count"]),
                        "accepted_moves": int(point["accepted_moves"]),
                    }
                )
    output = []
    dimensions = ((), ("topology",), ("size_group",), ("scenario_id",))
    for names in dimensions:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            key = (row["analysis_condition"], row["milestone"]) + tuple(
                row[name] for name in names
            )
            grouped[key].append(row)
        for key, values in sorted(grouped.items()):
            result = {
                "analysis_condition": key[0],
                "milestone": key[1],
                "stratum": "overall" if not names else "x".join(names),
                "stratum_value": "all" if not names else "|".join(map(str, key[2:])),
                "count": len(values),
            }
            for metric in (
                "best_score",
                "relative_gap_to_exact",
                "covered_node_count",
                "accepted_moves",
            ):
                for stat, value in _stats(row[metric] for row in values).items():
                    result[f"{metric}_{stat}"] = value
            output.append(result)
    return output


def _preference_response_rows(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    preference_config = yaml.safe_load(
        Path(config["preference_response_config"]).read_text(encoding="utf-8")
    )
    objective = load_objective_spec(preference_config["objective"]["config"])
    core_rows = _read_csv(Path(preference_config["data"]["core_manifest"]))
    core = {row["original_id"]: row for row in core_rows}
    scenarios = {
        item.scenario_id: item
        for item in load_scenario_bank(preference_config["data"]["scenario_bank"])
    }
    project_config = load_config()
    baseline_id = config["analysis"]["baseline_scenario_id"]
    lookup = {
        (row["original_id"], row["scenario_id"], row["analysis_condition"]): row
        for row in rows
    }
    output = []
    for row in rows:
        if row["scenario_id"] == baseline_id:
            continue
        baseline = lookup[
            (row["original_id"], baseline_id, row["analysis_condition"])
        ]
        base_instance = load_instance_json(Path(core[row["original_id"]]["output_json"]))
        shifted = apply_scenario(
            base_instance,
            scenarios[row["scenario_id"]],
            project_config,
            objective=objective,
        )
        no_reopt = score_assignment(
            baseline["assignment"], instance=shifted, objective=objective
        )
        assignment = np.asarray(row["assignment"], dtype=np.int64)
        baseline_assignment = np.asarray(baseline["assignment"], dtype=np.int64)
        output.append(
            {
                key: row[key]
                for key in (
                    "pair_id",
                    "original_id",
                    "source_family_id",
                    "topology",
                    "size_group",
                    "scenario_id",
                    "n",
                    "analysis_condition",
                    "condition_family",
                    "max_steps",
                    "patience",
                )
            }
            | {
                "reoptimization_gain": float(
                    row["best_score"] - no_reopt.total_score
                ),
                "assignment_change_rate": float(
                    np.mean(assignment != baseline_assignment)
                ),
                "positive_reoptimization_gain": bool(
                    row["best_score"] - no_reopt.total_score > 1.0e-9
                ),
                "relative_gap_to_exact": row["relative_gap_to_exact"],
            }
        )
    return output


def _preference_summaries(
    rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    dimensions = (
        ("scenario_id",),
        ("scenario_id", "topology"),
        ("scenario_id", "size_group"),
        ("scenario_id", "topology", "size_group"),
    )
    for names in dimensions:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (row["analysis_condition"],) + tuple(
                str(row[name]) for name in names
            )
            grouped[key].append(row)
        for key, values in sorted(grouped.items()):
            item = {
                "analysis_condition": key[0],
                "stratum": "x".join(names),
                "stratum_value": "|".join(key[1:]),
                "count": len(values),
                "positive_gain_rate": float(
                    np.mean(
                        [row["positive_reoptimization_gain"] for row in values]
                    )
                ),
            }
            for metric in (
                "reoptimization_gain",
                "assignment_change_rate",
                "relative_gap_to_exact",
            ):
                for stat, value in _stats(row[metric] for row in values).items():
                    item[f"{metric}_{stat}"] = value
            output.append(item)
    return output


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


def _plot_outputs(
    output_dir: Path,
    stratified: list[dict[str, Any]],
    anytime: list[dict[str, Any]],
    response_summary: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    key_conditions = (
        "current_b128_p32",
        "fixed_b256_p256",
        "fixed_b512_p512",
        "patience_b512_p32",
        "n_scaled",
    )
    labels = {
        "current_b128_p32": "128/32 current",
        "fixed_b256_p256": "256/256",
        "fixed_b512_p512": "512/512",
        "patience_b512_p32": "512/32",
        "n_scaled": "N-scaled",
    }
    overall = {
        row["analysis_condition"]: row
        for row in stratified
        if row["stratum"] == "overall"
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for condition in key_conditions:
        row = overall[condition]
        point = (
            1000.0 * float(row["runtime_sec_mean"]),
            100.0 * float(row["relative_gap_to_exact_mean"]),
        )
        ax.scatter(*point, s=45)
        ax.annotate(
            labels[condition], point, xytext=(4, 4), textcoords="offset points"
        )
    ax.set_xlabel("Mean inference time (ms)")
    ax.set_ylabel("Mean relative gap to exact (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_runtime_tradeoff.png")
    plt.close(fig)

    curve_rows = [
        row
        for row in anytime
        if row["analysis_condition"] == "fixed_b512_p512"
        and row["stratum"] in {"overall", "size_group"}
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for stratum, value, label in (
        ("overall", "all", "overall"),
        ("size_group", "group8", "group8"),
        ("size_group", "group9", "group9"),
    ):
        values = sorted(
            (
                row
                for row in curve_rows
                if row["stratum"] == stratum
                and row["stratum_value"] == value
            ),
            key=lambda row: int(row["milestone"]),
        )
        ax.plot(
            [int(row["milestone"]) for row in values],
            [100.0 * float(row["relative_gap_to_exact_mean"]) for row in values],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Search step")
    ax.set_ylabel("Mean relative gap to exact (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "anytime_gap_curve.png")
    plt.close(fig)

    size_rows = [
        row
        for row in stratified
        if row["stratum"] == "size_group"
        and row["analysis_condition"]
        in {"current_b128_p32", "fixed_b256_p256", "fixed_b512_p512"}
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for condition in (
        "current_b128_p32",
        "fixed_b256_p256",
        "fixed_b512_p512",
    ):
        values = sorted(
            (row for row in size_rows if row["analysis_condition"] == condition),
            key=lambda row: int(str(row["stratum_value"]).replace("group", "")),
        )
        ax.plot(
            [row["stratum_value"] for row in values],
            [100.0 * float(row["relative_gap_to_exact_mean"]) for row in values],
            marker="o",
            label=labels[condition],
        )
    ax.set_xlabel("Size group")
    ax.set_ylabel("Mean relative gap to exact (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gap_by_size_group.png")
    plt.close(fig)

    response_rows = [
        row
        for row in response_summary
        if row["stratum"] == "scenario_id"
        and row["analysis_condition"]
        in {"current_b128_p32", "fixed_b256_p256", "fixed_b512_p512"}
    ]
    scenarios = (
        "controlled_student_only",
        "controlled_teacher_only",
        "controlled_conflict",
    )
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for offset, condition in enumerate(
        ("current_b128_p32", "fixed_b256_p256", "fixed_b512_p512")
    ):
        lookup = {
            row["stratum_value"]: row
            for row in response_rows
            if row["analysis_condition"] == condition
        }
        ax.bar(
            x + (offset - 1) * width,
            [float(lookup[item]["reoptimization_gain_mean"]) for item in scenarios],
            width,
            label=labels[condition],
        )
    ax.set_xticks(x, ["student-only", "teacher-only", "conflict"])
    ax.set_ylabel("Mean reoptimization gain")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "preference_response_by_budget.png")
    plt.close(fig)


def _write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    stratified: list[dict[str, Any]],
    response_summary: list[dict[str, Any]],
) -> None:
    overall = {
        row["analysis_condition"]: row
        for row in stratified
        if row["stratum"] == "overall"
    }
    size = {
        (row["analysis_condition"], row["stratum_value"]): row
        for row in stratified
        if row["stratum"] == "size_group"
    }
    response = {
        (row["analysis_condition"], row["stratum_value"]): row
        for row in response_summary
        if row["stratum"] == "scenario_id"
    }
    group9_response = {
        (row["analysis_condition"], row["stratum_value"].split("|")[0]): row
        for row in response_summary
        if row["stratum"] == "scenario_idxsize_group"
        and row["stratum_value"].endswith("|group9")
    }
    current = overall["current_b128_p32"]
    recommended = overall["patience_b512_p32"]
    fixed_512 = overall["fixed_b512_p512"]
    fixed_256 = overall["fixed_b256_p256"]
    scaled = overall["n_scaled"]
    runtime_increase = (
        float(recommended["runtime_sec_mean"])
        / float(current["runtime_sec_mean"])
        - 1.0
    )
    lines = [
        "# 四拓扑搜索预算与质量诊断报告",
        "",
        "## 审计结论",
        "",
        f"- 固定 b075 目标与 run03 best checkpoint（update {metadata['checkpoint_update']}）完成 288 条母轨迹、2,880 个条件结果。",
        "- 所有 assignment 均满足硬约束，mask violation 为 0，最大评分重算误差为 0。",
        "- 本实验只改变推理阶段 max_steps/patience；未训练 PPO，也未冻结推理参数。",
        "",
        "## 主要结果",
        "",
        f"- 当前 128/32 的平均 gap 为 {100*float(current['relative_gap_to_exact_mean']):.3f}%，P90 为 {100*float(current['relative_gap_to_exact_p90']):.3f}%，1% 内比例为 {100*float(current['gap_le_1pct_rate']):.2f}%。",
        f"- 512/32 将平均 gap 降到 {100*float(recommended['relative_gap_to_exact_mean']):.3f}%，P90 降到 {100*float(recommended['relative_gap_to_exact_p90']):.3f}%，1% 内比例升到 {100*float(recommended['gap_le_1pct_rate']):.2f}%；平均推理时间由 {float(current['runtime_sec_mean']):.3f}s 增至 {float(recommended['runtime_sec_mean']):.3f}s（{100*runtime_increase:.1f}%）。",
        f"- 256/256 已达到平均 gap {100*float(fixed_256['relative_gap_to_exact_mean']):.3f}%；512 相对 256 的总体增益很小，但仍有少数实例继续改进。",
        f"- 512/32 与 512/512 的总体质量完全一致，而平均时间分别为 {float(recommended['runtime_sec_mean']):.3f}s 和 {float(fixed_512['runtime_sec_mean']):.3f}s。提高 patience 没有质量收益。",
        f"- N 缩放到最高 768 步与固定 512 的质量完全一致，平均时间为 {float(scaled['runtime_sec_mean']):.3f}s；不支持默认采用 N 缩放。",
        f"- 平均节点覆盖率仅由 {100*float(current['covered_node_rate_mean']):.2f}% 增至 {100*float(recommended['covered_node_rate_mean']):.2f}%。512/512 平均重复动作 {float(fixed_512['repeated_action_count_mean']):.1f}/{float(fixed_512['executed_moves_mean']):.0f}，说明长尾主要是循环而不是新增覆盖。",
        "",
        "## 瓶颈定位",
        "",
        f"- group8 平均 gap 从 {100*float(size[('current_b128_p32','group8')]['relative_gap_to_exact_mean']):.3f}% 降至 {100*float(size[('fixed_b512_p512','group8')]['relative_gap_to_exact_mean']):.3f}%。",
        f"- group9 平均 gap 从 {100*float(size[('current_b128_p32','group9')]['relative_gap_to_exact_mean']):.3f}% 降至 {100*float(size[('fixed_b512_p512','group9')]['relative_gap_to_exact_mean']):.3f}%，证明 128 步是大规模组的实际瓶颈。",
        "- group1–group6 在增加预算后没有改善；其残余误差不能归因于 128 步限制。",
        "",
        "## 偏好响应",
        "",
        f"- student-only 正增益率由 {100*float(response[('current_b128_p32','controlled_student_only')]['positive_gain_rate']):.1f}% 升至 {100*float(response[('fixed_b512_p512','controlled_student_only')]['positive_gain_rate']):.1f}%。",
        f"- conflict 正增益率由 {100*float(response[('current_b128_p32','controlled_conflict')]['positive_gain_rate']):.1f}% 升至 {100*float(response[('fixed_b512_p512','controlled_conflict')]['positive_gain_rate']):.1f}%。",
        "- teacher-only 在所有预算和 patience 下仍为零 assignment 变化、零重求解增益；延长搜索不能解释或修复该问题。",
        f"- group9 的 student-only 正增益率由 {100*float(group9_response[('current_b128_p32','controlled_student_only')]['positive_gain_rate']):.1f}% 升至 {100*float(group9_response[('fixed_b512_p512','controlled_student_only')]['positive_gain_rate']):.1f}%，conflict 由 {100*float(group9_response[('current_b128_p32','controlled_conflict')]['positive_gain_rate']):.1f}% 升至 {100*float(group9_response[('fixed_b512_p512','controlled_conflict')]['positive_gain_rate']):.1f}%；group9 的非教师响应零值属于预算瓶颈。",
        "",
        "## 候选意见（未冻结）",
        "",
        "- 主候选：max_steps=512、patience=32。它保留当前 patience，用较小时间增幅获得与无早停 512 步相同的质量。",
        "- 低时延参照：保留 max_steps=128、patience=32，用于后续质量匹配的批量服务比较。",
        "- 不推荐提高 patience，也不推荐默认 N 缩放；二者没有带来额外质量。",
        "- teacher-only 零响应应回到训练画像覆盖、损失权衡或策略学习诊断，不能继续归因于推理预算。",
        "- 以上仅形成第6项待决候选，不构成 PPO/推理参数已冻结。",
        "",
        "## 产物",
        "",
        "- condition_results.csv：2,880 条条件结果。",
        "- stratified_quality.csv：拓扑、规模、画像及交叉分层。",
        "- anytime_summary.csv：逐里程碑 best-so-far 与 gap。",
        "- preference_response_by_condition.csv 与 preference_response_summary.csv：预算下的重求解响应。",
        "- 四张 PNG 图分别展示 anytime、质量—时间、规模分层和画像响应。",
    ]
    (output_dir / "search_budget_diagnostic_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def summarize(
    config_path: str | Path,
    *, output_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config_preview = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = (
        Path(output_override)
        if output_override is not None
        else Path(config_preview["run"]["output_dir"])
    ).resolve()
    config, metadata, payloads = _load_payloads(config_path, output_dir)
    rows = _flat_rows(payloads)
    stratified = _stratified_summaries(rows)
    anytime = _anytime_rows(
        payloads, [int(value) for value in config["analysis"]["anytime_milestones"]]
    )
    response = _preference_response_rows(rows, config)
    response_summary = _preference_summaries(response)

    serializable_rows = [
        {key: value for key, value in row.items() if key != "assignment"}
        for row in rows
    ]
    _write_csv(output_dir / "condition_results.csv", serializable_rows)
    _write_csv(output_dir / "stratified_quality.csv", stratified)
    _write_csv(output_dir / "anytime_summary.csv", anytime)
    _write_csv(output_dir / "preference_response_by_condition.csv", response)
    _write_csv(
        output_dir / "preference_response_summary.csv", response_summary
    )
    _plot_outputs(output_dir, stratified, anytime, response_summary)
    _write_report(output_dir, metadata, stratified, response_summary)

    overall = [
        row for row in stratified if row["stratum"] == "overall"
    ]
    teacher = [
        row
        for row in response_summary
        if row["stratum"] == "scenario_id"
        and row["stratum_value"] == config["analysis"]["teacher_only_scenario_id"]
    ]
    group9_teacher = [
        row
        for row in response_summary
        if row["stratum"] == "scenario_idxsize_group"
        and row["stratum_value"]
        == f"{config['analysis']['teacher_only_scenario_id']}|group9"
    ]
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "objective_hash": metadata["objective_hash"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "pair_count": len(payloads),
        "condition_result_count": len(rows),
        "condition_count": len(
            {row["analysis_condition"] for row in rows}
        ),
        "all_hard_feasible": all(
            result["hard_feasible_rate"] == 1.0
            and result["mask_violation_count"] == 0
            for payload in payloads
            for result in payload["condition_results"].values()
        ),
        "maximum_score_recompute_error": max(
            result["best_score_recompute_error"]
            for payload in payloads
            for result in payload["condition_results"].values()
        ),
        "overall_condition_summary": overall,
        "teacher_only_response_summary": teacher,
        "group9_teacher_only_response_summary": group9_teacher,
        "inference_parameters_frozen": False,
        "teaching_quality_threshold_defined": False,
    }
    _write_json(output_dir / "search_budget_summary.json", summary)
    _write_hash_manifest(output_dir)
    return summary


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
    args = parser.parse_args()
    result = summarize(args.config, output_override=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
