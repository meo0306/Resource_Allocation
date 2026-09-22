"""Audit one completed four-topology seed-0 pilot without changing raw outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pa_moap_rl.checkpointing import validate_checkpoint_metadata
from pa_moap_rl.objective import load_objective_spec


OBJECTIVE_HASH = "b075006e79dc73a06ce76049ac0e3350c2d6b9b3ceb79db6fec57294e5b51606"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in row {row}")
    return value


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


def _validation_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    gaps = [_f(row, "relative_gap_to_exact") for row in rows]
    summary: dict[str, Any] = {
        "count": len(rows),
        "total_score": _stats(_f(row, "total_score") for row in rows),
        "relative_gap_to_exact": _stats(gaps),
        "effect_score": _stats(_f(row, "effect_score") for row in rows),
        "student_pref_score": _stats(_f(row, "student_pref_score") for row in rows),
        "teacher_pref_score": _stats(_f(row, "teacher_pref_score") for row in rows),
        "soft_penalty": _stats(_f(row, "soft_penalty") for row in rows),
        "method_distribution_entropy": _stats(_f(row, "entropy") for row in rows),
        "max_method_share": _stats(_f(row, "max_method_share") for row in rows),
        "hard_feasible_rate": _stats(_f(row, "hard_feasible_rate") for row in rows),
        "mask_violation_count": int(sum(int(row["mask_violation_count"]) for row in rows)),
    }
    summary["gap_threshold_rates"] = {
        f"within_{limit:d}pct": float(np.mean(np.asarray(gaps) <= limit / 100.0))
        for limit in (1, 2, 3, 5)
    }
    return summary


def _group_best(rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    output: list[dict[str, Any]] = []
    for value in sorted(grouped):
        summary = _validation_summary(grouped[value])
        gap = summary["relative_gap_to_exact"]
        rates = summary["gap_threshold_rates"]
        output.append(
            {
                "dimension": key,
                "value": value,
                "count": summary["count"],
                "mean_gap": gap["mean"],
                "median_gap": gap["median"],
                "p90_gap": gap["p90"],
                "max_gap": gap["max"],
                "within_1pct": rates["within_1pct"],
                "within_2pct": rates["within_2pct"],
                "within_3pct": rates["within_3pct"],
                "within_5pct": rates["within_5pct"],
            }
        )
    return output


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def audit_pilot(output_dir: Path, objective_path: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    raw_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
    train_updates = _read_csv(output_dir / "metrics" / "train_updates.csv")
    train_instances = _read_csv(output_dir / "metrics" / "train_instances.csv")
    validation_instances = _read_csv(output_dir / "metrics" / "validation_instances.csv")

    validation_by_update: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in validation_instances:
        validation_by_update[int(row["update"])].append(row)
    validation_updates = sorted(validation_by_update)
    update_summaries = {
        update: _validation_summary(validation_by_update[update])
        for update in validation_updates
    }
    best_update = max(
        validation_updates,
        key=lambda update: (update_summaries[update]["total_score"]["mean"], -update),
    )
    initial_update = validation_updates[0]
    latest_update = validation_updates[-1]
    best_rows = validation_by_update[best_update]

    critical_train_fields = (
        "approx_kl",
        "clip_fraction",
        "normalized_entropy",
        "grad_norm",
        "transitions_per_sec",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
    )
    training_health = {
        field: _stats(_f(row, field) for row in train_updates)
        for field in critical_train_fields
    }
    training_health["batch_group_counts"] = dict(
        sorted(Counter(row["batch_group"] for row in train_updates).items())
    )

    expected_eval_updates = list(range(0, 101, 10))
    validation_count_per_update = Counter(int(row["update"]) for row in validation_instances)
    objective_hashes = {row["objective_hash"] for row in validation_instances + train_instances}
    checks = {
        "run_completed_to_update_100": raw_summary.get("status") == "completed"
        and int(raw_summary.get("last_update", -1)) == 100,
        "train_update_count_100": len(train_updates) == 100,
        "train_instance_count_400": len(train_instances) == 400,
        "validation_instance_count_792": len(validation_instances) == 792,
        "validation_updates_0_to_100_every_10": validation_updates == expected_eval_updates,
        "validation_count_72_each_update": all(
            validation_count_per_update[update] == 72 for update in expected_eval_updates
        ),
        "all_training_batches_have_four_topologies": all(
            int(row["unique_topologies"]) == 4 for row in train_updates
        ),
        "all_training_batches_have_four_unique_sources": all(
            int(row["unique_sources"]) == 4 for row in train_updates
        ),
        "no_unknown_training_group": all(
            row["batch_group"].startswith("group") for row in train_updates
        ),
        "no_illegal_training_actions": all(
            int(row["illegal_action_count"]) == 0 for row in train_updates
        ),
        "all_validation_assignments_hard_feasible": all(
            _f(row, "hard_feasible_rate") == 1.0 for row in validation_instances
        ),
        "no_validation_mask_violations": all(
            int(row["mask_violation_count"]) == 0 for row in validation_instances
        ),
        "objective_hash_is_frozen_b075": objective_hashes == {OBJECTIVE_HASH},
        "deterministic_cuda_recorded": metadata.get("device") == "cuda"
        and metadata.get("deterministic_algorithms") is True,
    }

    spec = load_objective_spec(objective_path)
    if spec.config_hash != OBJECTIVE_HASH:
        raise ValueError("The supplied objective is not frozen b075.")
    input_hashes = [str(item["sha256"]) for item in metadata["inputs"].values()]
    recomputed_data_hash = hashlib.sha256("".join(input_hashes).encode("ascii")).hexdigest()
    method_hash = str(metadata["method_matrix_hash"])
    checkpoint_audit: dict[str, Any] = {
        "objective_hash": spec.config_hash,
        "recorded_data_hash": metadata["data_hash"],
        "recomputed_data_hash": recomputed_data_hash,
        "method_matrix_hash": method_hash,
        "checkpoints": {},
    }
    for name, expected_update in (("best.pt", best_update), ("latest.pt", latest_update)):
        path = output_dir / "checkpoints" / name
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_metadata(
            checkpoint,
            objective=spec,
            method_hash=method_hash,
            data_hash=recomputed_data_hash,
        )
        checkpoint_audit["checkpoints"][name] = {
            "update": int(checkpoint["update"]),
            "expected_update": expected_update,
            "update_matches": int(checkpoint["update"]) == expected_update,
            "sha256": _sha256(path),
            "provenance_valid": True,
        }
    checkpoint_audit["data_hash_matches"] = recomputed_data_hash == metadata["data_hash"]
    checkpoint_audit["all_passed"] = checkpoint_audit["data_hash_matches"] and all(
        item["update_matches"] and item["provenance_valid"]
        for item in checkpoint_audit["checkpoints"].values()
    )

    scenario_to_topology: dict[str, set[str]] = defaultdict(set)
    for row in best_rows:
        scenario_to_topology[row["scenario_id"]].add(row["topology"])
    scenario_topology_map = {
        key: sorted(value) for key, value in sorted(scenario_to_topology.items())
    }
    scenario_topology_confounded = all(len(value) == 1 for value in scenario_to_topology.values())

    initial = update_summaries[initial_update]
    best = update_summaries[best_update]
    latest = update_summaries[latest_update]
    best_to_latest_score_drop = best["total_score"]["mean"] - latest["total_score"]["mean"]
    best_to_latest_gap_increase = (
        latest["relative_gap_to_exact"]["mean"] - best["relative_gap_to_exact"]["mean"]
    )
    post_best_degradation = best_to_latest_score_drop > 0.01 or best_to_latest_gap_increase > 0.01
    engineering_pass = all(checks.values()) and checkpoint_audit["all_passed"]
    if not engineering_pass:
        recommendation = "inspect_failed_engineering_checks"
    elif post_best_degradation:
        recommendation = "adjust_ppo_before_preference_and_budget_diagnostics"
    else:
        recommendation = "proceed_to_preference_and_budget_diagnostics"
    stratified = (
        _group_best(best_rows, "topology")
        + _group_best(best_rows, "group")
        + _group_best(best_rows, "scenario_id")
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(output_dir),
        "raw_run_summary_sha256": _sha256(output_dir / "run_summary.json"),
        "objective_hash": OBJECTIVE_HASH,
        "checks": checks,
        "checkpoint_audit": checkpoint_audit,
        "engineering_pass": engineering_pass,
        "initial_update": initial_update,
        "best_update": best_update,
        "latest_update": latest_update,
        "initial": initial,
        "best": best,
        "latest": latest,
        "best_to_latest_score_drop": best_to_latest_score_drop,
        "best_to_latest_gap_increase": best_to_latest_gap_increase,
        "post_best_degradation": post_best_degradation,
        "training_health": training_health,
        "scenario_topology_map": scenario_topology_map,
        "scenario_topology_confounded": scenario_topology_confounded,
        "recommendation": recommendation,
    }
    _write_json(output_dir / "pilot_audit_summary.json", payload)
    _write_json(output_dir / "checkpoint_audit.json", checkpoint_audit)
    _write_csv(output_dir / "best_update_stratified.csv", stratified)

    topology_rows = [row for row in stratified if row["dimension"] == "topology"]
    size_rows = [row for row in stratified if row["dimension"] == "group"]
    report = [
        f"# 四拓扑 b075 seed0 pilot 决策报告：{output_dir.name}",
        "",
        "## 结论",
        "",
        f"- 工程与恢复链验收：**{'通过' if engineering_pass else '未通过'}**。",
        f"- 最佳 checkpoint：update {best_update}；验证集平均 J={best['total_score']['mean']:.9f}，平均 gap={_format_pct(best['relative_gap_to_exact']['mean'])}，P90 gap={_format_pct(best['relative_gap_to_exact']['p90'])}。",
        f"- latest checkpoint：update {latest_update}；平均 J={latest['total_score']['mean']:.9f}，平均 gap={_format_pct(latest['relative_gap_to_exact']['mean'])}，P90 gap={_format_pct(latest['relative_gap_to_exact']['p90'])}。",
        f"- update {best_update} 后到 update {latest_update} 的平均 J 下降 {best_to_latest_score_drop:.9f}，平均 gap 增加 {_format_pct(best_to_latest_gap_increase)}。",
        (
            "- 这构成明确的后程退化；建议先做稳定化调整 pilot，再进入偏好响应与搜索预算诊断。"
            if post_best_degradation
            else "- 未发现实质性后程退化；该运行满足进入偏好响应与搜索预算诊断的稳定性条件。"
        ),
        "- 冻结目标 b075 不变；PPO 与推理参数仍未冻结。",
        "",
        "## 运行和数据完整性",
        "",
        "- 100 个训练 update、400 个训练实例记录；validation 在 update 0/10/.../100 共 11 次，每次固定 72 对，共 792 条。",
        "- 每个训练 batch 均含四种拓扑与四个不同源家族；非法动作 0；validation 全部硬约束可行、mask violation 为 0。",
        f"- best/latest checkpoint 的 objective、method matrix、data provenance 均通过严格校验；data hash 为 `{recomputed_data_hash}`。",
        "",
        "## 最佳 checkpoint 的总体质量",
        "",
        f"- F_E={best['effect_score']['mean']:.6f}，F_S={best['student_pref_score']['mean']:.6f}，F_T={best['teacher_pref_score']['mean']:.6f}。",
        f"- 平均软惩罚={best['soft_penalty']['mean']:.6f}，方法分布熵={best['method_distribution_entropy']['mean']:.6f}，最大方法占比={best['max_method_share']['mean']:.6f}。",
        f"- gap 达标率：1%={_format_pct(best['gap_threshold_rates']['within_1pct'])}，2%={_format_pct(best['gap_threshold_rates']['within_2pct'])}，3%={_format_pct(best['gap_threshold_rates']['within_3pct'])}，5%={_format_pct(best['gap_threshold_rates']['within_5pct'])}。",
        "",
        "## 按拓扑分层（best update）",
        "",
        "| topology | mean gap | P90 gap | max gap | within 1% | within 2% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['value']} | {_format_pct(row['mean_gap'])} | {_format_pct(row['p90_gap'])} | {_format_pct(row['max_gap'])} | {_format_pct(row['within_1pct'])} | {_format_pct(row['within_2pct'])} |"
        for row in topology_rows
    )
    report.extend(
        [
            "",
            "## 按规模组分层（best update）",
            "",
            "| group | mean gap | P90 gap | max gap | within 1% | within 2% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    report.extend(
        f"| {row['value']} | {_format_pct(row['mean_gap'])} | {_format_pct(row['p90_gap'])} | {_format_pct(row['max_gap'])} | {_format_pct(row['within_1pct'])} | {_format_pct(row['within_2pct'])} |"
        for row in size_rows
    )
    report.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这 72 对 validation 的受控画像与拓扑一一对应，存在完全混杂；它适合做固定工程/质量监控，不支持独立解释画像响应或比较四类画像。画像响应必须在后续 72×4 全交叉设计中检验。",
            "- 本次仅为 seed0 短程开发 pilot，不能替代多种子结论，也不能把 1%/2%/3%/5% gap 档位解释为教学质量阈值。",
            "- 本报告不改变冻结目标，不冻结 PPO 或推理参数，不启动正式多种子训练。",
            "",
        ]
    )
    (output_dir / "pilot_decision_report.md").write_text("\n".join(report), encoding="utf-8")

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
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--objective",
        type=Path,
        default=Path("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml"),
    )
    args = parser.parse_args()
    summary = audit_pilot(args.output_dir, args.objective)
    print(json.dumps({
        "engineering_pass": summary["engineering_pass"],
        "best_update": summary["best_update"],
        "latest_update": summary["latest_update"],
        "recommendation": summary["recommendation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
