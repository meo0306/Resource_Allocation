"""Run the pre-registered Stage11 Phase-A dynamic-preference diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import PreferenceScenario, apply_scenario
from pa_moap_rl.experiments.dynamic_preference_adjustment_v1 import (
    DYNAMIC_ENGINE_VERSION,
    build_dynamic_scenarios,
    pareto_method_ids,
    run_dynamic_policy,
)
from pa_moap_rl.experiments.run_service_efficiency_v1 import (
    _load_model_bundle,
    _validate_frozen_config,
)
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.greedy_solver import scalarized_independent_greedy
from pa_moap_rl.solvers.gurobi_exact_solver import solve_gurobi
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result
from pa_moap_rl.utils.scoring import score_assignment


DEFAULT_CONFIG = Path("pa_moap_rl/configs/dynamic_preference_adjustment_phase_a_v1.yaml")
RUNNER_VERSION = "dynamic_preference_phase_a_runner_v1"
METHOD_IDS = (
    "old_solution",
    "scalarized_greedy",
    "ppo_restart",
    "ppo_old_start",
    "local_old_start",
    "gurobi_rebuild_default_start",
    "gurobi_rebuild_old_start",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _assignment_sha256(assignment: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(assignment, dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _case_key(original_id: str) -> str:
    stem = original_id.replace("/", "__").replace("\\", "__")
    return f"{stem}__{hashlib.sha256(original_id.encode('utf-8')).hexdigest()[:10]}"


def _select_strata(core_rows: Sequence[dict[str, str]], expected: int) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in core_rows:
        grouped.setdefault((row["size_group"], row["topology"]), []).append(row)
    selected = [sorted(rows, key=lambda row: row["original_id"])[0] for _, rows in sorted(grouped.items())]
    if len(selected) != int(expected):
        raise ValueError(f"Expected {expected} topology-size strata, got {len(selected)}.")
    return selected


def _condition(
    row: dict[str, str],
    scenario: PreferenceScenario,
    *,
    project_config: Any,
    objective: ObjectiveSpec,
) -> Any:
    base = load_instance_json(row["output_json"])
    return apply_scenario(base, scenario, project_config, objective=objective)


def _timed(call: Callable[[], Any], device: torch.device | None = None) -> tuple[Any, float]:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = call()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def _method_row(
    *,
    method_id: str,
    assignment: np.ndarray,
    instance: Any,
    objective: ObjectiveSpec,
    solve_latency_sec: float,
    conditioning_latency_sec: float,
    old_assignment: np.ndarray,
    old_score: float,
    reference_incumbent: float,
    reference_bound: float,
    reference_proven_optimal: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = make_solver_result(
        solver_name=method_id,
        instance=instance,
        assignment=assignment,
        runtime=solve_latency_sec,
        objective=objective,
    )
    score = float(result.score.total_score)
    denominator = max(abs(float(reference_bound)), float(objective.epsilon))
    incumbent_denominator = max(abs(float(reference_incumbent)), float(objective.epsilon))
    changes = int(np.sum(np.asarray(assignment, dtype=np.int64) != old_assignment))
    row: dict[str, Any] = {
        "method_id": method_id,
        "total_score": score,
        "reference_incumbent": float(reference_incumbent),
        "reference_bound": float(reference_bound),
        "reference_proven_optimal": bool(reference_proven_optimal),
        "bounded_gap": max(0.0, float(reference_bound) - score) / denominator,
        "incumbent_gap": max(0.0, float(reference_incumbent) - score) / incumbent_denominator,
        "improvement_over_old": score - float(old_score),
        "positive_improvement_over_old": bool(score > float(old_score) + 1.0e-12),
        "assignment_change_count": changes,
        "assignment_change_rate": changes / float(instance.n),
        "solve_latency_sec": float(solve_latency_sec),
        "conditioning_latency_sec": float(conditioning_latency_sec),
        "end_to_end_latency_sec": float(conditioning_latency_sec + solve_latency_sec),
        "hard_feasible": bool(result.metrics["hard_feasible_rate"] == 1.0),
        "mask_violation_count": int(result.metrics["mask_violation_count"]),
        "assignment": result.assignment.tolist(),
        "assignment_sha256": _assignment_sha256(result.assignment),
        "effect_score": float(result.score.effect_score),
        "student_pref_score": float(result.score.student_pref_score),
        "teacher_pref_score": float(result.score.teacher_pref_score),
        "global_score": float(result.score.global_score),
        "soft_penalty": float(result.score.soft_penalty),
    }
    if extra:
        row.update(extra)
    return row


def _run_case(
    *,
    row: dict[str, str],
    scenario: PreferenceScenario,
    baseline_instance: Any,
    old_assignment: np.ndarray,
    config: dict[str, Any],
    project_config: Any,
    objective: ObjectiveSpec,
    model: Any,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    instance, conditioning_latency = _timed(
        lambda: _condition(row, scenario, project_config=project_config, objective=objective)
    )
    old_score = float(score_assignment(old_assignment, instance=instance, objective=objective).total_score)
    reference_cfg = config["methods"]["quality_reference"]
    reference, reference_latency = _timed(
        lambda: solve_gurobi(
            instance,
            time_limit=float(reference_cfg["time_limit_sec"]),
            mip_gap=0.0,
            seed=int(reference_cfg["seed"]),
            threads=int(reference_cfg["threads"]),
            objective=objective,
        )
    )
    reference_incumbent = float(reference.score.total_score)
    reference_bound = float(reference.metrics["objective_bound"])
    reference_proven = bool(reference.metrics["proven_optimal"])
    methods: list[dict[str, Any]] = []
    methods.append(
        _method_row(
            method_id="old_solution",
            assignment=old_assignment,
            instance=instance,
            objective=objective,
            solve_latency_sec=0.0,
            conditioning_latency_sec=conditioning_latency,
            old_assignment=old_assignment,
            old_score=old_score,
            reference_incumbent=reference_incumbent,
            reference_bound=reference_bound,
            reference_proven_optimal=reference_proven,
        )
    )

    scalar, scalar_latency = _timed(
        lambda: scalarized_independent_greedy(instance, objective=objective)
    )
    methods.append(
        _method_row(
            method_id="scalarized_greedy",
            assignment=scalar.assignment,
            instance=instance,
            objective=objective,
            solve_latency_sec=scalar_latency,
            conditioning_latency_sec=conditioning_latency,
            old_assignment=old_assignment,
            old_score=old_score,
            reference_incumbent=reference_incumbent,
            reference_bound=reference_bound,
            reference_proven_optimal=reference_proven,
        )
    )

    ppo_cfg = config["methods"]["ppo_restart"]
    ppo_restart = run_dynamic_policy(
        instance=instance,
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=int(ppo_cfg["max_steps"]),
        patience=int(ppo_cfg["patience"]),
        initial_assignment=None,
    )
    methods.append(
        _method_row(
            method_id="ppo_restart",
            assignment=ppo_restart.assignment,
            instance=instance,
            objective=objective,
            solve_latency_sec=ppo_restart.runtime_sec,
            conditioning_latency_sec=conditioning_latency,
            old_assignment=old_assignment,
            old_score=old_score,
            reference_incumbent=reference_incumbent,
            reference_bound=reference_bound,
            reference_proven_optimal=reference_proven,
            extra={
                "step_count": ppo_restart.step_count,
                "steps_to_best": ppo_restart.steps_to_best,
                "done_reason": ppo_restart.done_reason,
            },
        )
    )

    ppo_old_cfg = config["methods"]["ppo_old_start"]
    ppo_old = run_dynamic_policy(
        instance=instance,
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=int(ppo_old_cfg["max_steps"]),
        patience=int(ppo_old_cfg["patience"]),
        initial_assignment=old_assignment,
    )
    methods.append(
        _method_row(
            method_id="ppo_old_start",
            assignment=ppo_old.assignment,
            instance=instance,
            objective=objective,
            solve_latency_sec=ppo_old.runtime_sec,
            conditioning_latency_sec=conditioning_latency,
            old_assignment=old_assignment,
            old_score=old_score,
            reference_incumbent=reference_incumbent,
            reference_bound=reference_bound,
            reference_proven_optimal=reference_proven,
            extra={
                "step_count": ppo_old.step_count,
                "steps_to_best": ppo_old.steps_to_best,
                "done_reason": ppo_old.done_reason,
            },
        )
    )

    local_cfg = config["methods"]["local_old_start"]
    local, local_latency = _timed(
        lambda: local_search_best_improvement(
            instance,
            initial_assignment=old_assignment,
            max_iter=int(local_cfg["max_iter"]),
            objective=objective,
        )
    )
    methods.append(
        _method_row(
            method_id="local_old_start",
            assignment=local.assignment,
            instance=instance,
            objective=objective,
            solve_latency_sec=local_latency,
            conditioning_latency_sec=conditioning_latency,
            old_assignment=old_assignment,
            old_score=old_score,
            reference_incumbent=reference_incumbent,
            reference_bound=reference_bound,
            reference_proven_optimal=reference_proven,
            extra={"accepted_moves": max(0, len(local.history) - 1)},
        )
    )

    for method_id, warm_start in (
        ("gurobi_rebuild_default_start", None),
        ("gurobi_rebuild_old_start", old_assignment),
    ):
        gurobi_cfg = config["methods"][method_id]
        gurobi, gurobi_latency = _timed(
            lambda warm_start=warm_start, gurobi_cfg=gurobi_cfg: solve_gurobi(
                instance,
                time_limit=float(gurobi_cfg["time_limit_sec"]),
                mip_gap=0.0,
                seed=int(gurobi_cfg["seed"]),
                threads=int(gurobi_cfg["threads"]),
                warm_start=warm_start,
                objective=objective,
            )
        )
        methods.append(
            _method_row(
                method_id=method_id,
                assignment=gurobi.assignment,
                instance=instance,
                objective=objective,
                solve_latency_sec=gurobi_latency,
                conditioning_latency_sec=conditioning_latency,
                old_assignment=old_assignment,
                old_score=old_score,
                reference_incumbent=reference_incumbent,
                reference_bound=reference_bound,
                reference_proven_optimal=reference_proven,
                extra={
                    "gurobi_status": gurobi.metrics["gurobi_status"],
                    "gurobi_proven_optimal": bool(gurobi.metrics["proven_optimal"]),
                    "gurobi_mip_gap": float(gurobi.metrics["mip_gap"]),
                    "gurobi_objective_bound": float(gurobi.metrics["objective_bound"]),
                },
            )
        )
    if [item["method_id"] for item in methods] != list(METHOD_IDS):
        raise RuntimeError("Dynamic method order or coverage mismatch.")
    if any(not item["hard_feasible"] or item["mask_violation_count"] for item in methods):
        raise RuntimeError("Dynamic-preference method produced an infeasible assignment.")
    direction, fraction_label = scenario.scenario_id.rsplit("__", 1)
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "engine_version": DYNAMIC_ENGINE_VERSION,
        "case": {
            "original_id": row["original_id"],
            "source_family_id": row["source_family_id"],
            "size_group": row["size_group"],
            "topology": row["topology"],
            "N": int(row["N"]),
            "V": int(row["V"]),
            "scenario_id": scenario.scenario_id,
            "direction": direction,
            "fraction": int(fraction_label[1:]) / 100.0,
        },
        "common_old_solution": {
            "source": "frozen_ppo_on_dynamic_baseline_s6_t6",
            "assignment_sha256": _assignment_sha256(old_assignment),
            "baseline_score": float(
                score_assignment(old_assignment, instance=baseline_instance, objective=objective).total_score
            ),
            "score_under_new_preferences": old_score,
        },
        "conditioning_latency_sec": conditioning_latency,
        "quality_reference": {
            "solve_latency_sec": reference_latency,
            "incumbent": reference_incumbent,
            "bound": reference_bound,
            "proven_optimal": reference_proven,
            "status": reference.metrics["gurobi_status"],
            "mip_gap": float(reference.metrics["mip_gap"]),
            "objective_recompute_error": float(reference.metrics["objective_recompute_error"]),
            "assignment_sha256": _assignment_sha256(reference.assignment),
        },
        "methods": methods,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _aggregate(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items()):
        latencies = [float(item["end_to_end_latency_sec"]) for item in values]
        gaps = [float(item["bounded_gap"]) for item in values]
        gains = [float(item["improvement_over_old"]) for item in values]
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(
            {
                "count": len(values),
                "mean_bounded_gap": float(statistics.fmean(gaps)),
                "median_bounded_gap": float(statistics.median(gaps)),
                "p90_bounded_gap": _quantile(gaps, 0.90),
                "worst_bounded_gap": max(gaps),
                "mean_improvement_over_old": float(statistics.fmean(gains)),
                "positive_improvement_rate": float(
                    statistics.fmean(float(item["positive_improvement_over_old"]) for item in values)
                ),
                "mean_assignment_change_rate": float(
                    statistics.fmean(float(item["assignment_change_rate"]) for item in values)
                ),
                "mean_latency_sec": float(statistics.fmean(latencies)),
                "median_latency_sec": float(statistics.median(latencies)),
                "p95_latency_sec": _quantile(latencies, 0.95),
                "all_hard_feasible": all(bool(item["hard_feasible"]) for item in values),
                "mask_violation_count": sum(int(item["mask_violation_count"]) for item in values),
            }
        )
        summaries.append(summary)
    return summaries


def _summarize(
    *,
    config: dict[str, Any],
    output_dir: Path,
    payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    flat_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for payload in payloads:
        case = payload["case"]
        reference_rows.append({**case, **payload["quality_reference"]})
        for method in payload["methods"]:
            flat_rows.append({**case, **{key: value for key, value in method.items() if key != "assignment"}})
    cell_summary = _aggregate(flat_rows, ("direction", "fraction", "method_id"))
    overall_summary = _aggregate(flat_rows, ("method_id",))
    teacher_summary = [row for row in cell_summary if row["direction"] == "teacher_only"]
    pareto_cells: list[dict[str, Any]] = []
    for direction in sorted({row["direction"] for row in cell_summary}):
        for fraction in sorted({float(row["fraction"]) for row in cell_summary}):
            cell = [
                row for row in cell_summary
                if row["direction"] == direction and float(row["fraction"]) == fraction
            ]
            ids = pareto_method_ids(cell)
            pareto_cells.append({"direction": direction, "fraction": fraction, "method_ids": ids})
    ppo_pareto_count = sum("ppo_old_start" in cell["method_ids"] for cell in pareto_cells)
    overall_by_id = {row["method_id"]: row for row in overall_summary}
    warm = overall_by_id["ppo_old_start"]
    restart = overall_by_id["ppo_restart"]
    gate = config["success_gate"]
    latency_reduction = 1.0 - float(warm["median_latency_sec"]) / float(restart["median_latency_sec"])
    gap_delta = float(warm["mean_bounded_gap"]) - float(restart["mean_bounded_gap"])
    faster_route = (
        latency_reduction >= float(gate["required_median_latency_reduction_vs_restart"])
        and gap_delta <= float(gate["max_mean_gap_regression_vs_restart"])
    )
    quality_improvement = float(restart["mean_bounded_gap"]) - float(warm["mean_bounded_gap"])
    quality_route = (
        quality_improvement >= float(gate["required_mean_gap_improvement_if_not_faster"])
        and float(warm["median_latency_sec"])
        <= float(restart["median_latency_sec"]) * (1.0 + float(gate["max_latency_increase_if_quality_route"]))
    )
    dominated_by: list[str] = []
    for baseline_id in gate["dominance_baselines"]:
        baseline = overall_by_id[baseline_id]
        no_worse = (
            float(baseline["mean_bounded_gap"]) <= float(warm["mean_bounded_gap"]) + 1.0e-12
            and float(baseline["median_latency_sec"]) <= float(warm["median_latency_sec"]) + 1.0e-12
        )
        strict = (
            float(baseline["mean_bounded_gap"]) < float(warm["mean_bounded_gap"]) - 1.0e-12
            or float(baseline["median_latency_sec"]) < float(warm["median_latency_sec"]) - 1.0e-12
        )
        if no_worse and strict:
            dominated_by.append(str(baseline_id))
    decision = {
        "ppo_old_start_pareto_cell_count": int(ppo_pareto_count),
        "required_pareto_cell_count": int(gate["required_ppo_old_start_pareto_cells"]),
        "median_latency_reduction_vs_restart": latency_reduction,
        "mean_gap_delta_vs_restart": gap_delta,
        "restart_benefit_route_passed": bool(faster_route or quality_route),
        "dominated_by_registered_baselines": dominated_by,
        "not_dominated_by_registered_baselines": not dominated_by,
    }
    eligible = (
        ppo_pareto_count >= int(gate["required_ppo_old_start_pareto_cells"])
        and bool(faster_route or quality_route)
        and not dominated_by
    )
    decision.update(
        {
            "eligible_for_phase_b": bool(eligible),
            "status": "phase_a_signal_continue" if eligible else "stop_after_phase_a_no_signal",
            "persistent_gurobi_model_reuse_required_next": bool(eligible),
        }
    )
    proof_rate = float(statistics.fmean(float(row["proven_optimal"]) for row in reference_rows))
    summary = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "status": "completed",
        "development_only": True,
        "no_training": True,
        "case_count": len(payloads),
        "method_result_count": len(flat_rows),
        "reference_optimal_proof_rate": proof_rate,
        "all_hard_feasible": all(row["all_hard_feasible"] for row in overall_summary),
        "mask_violation_count": sum(int(row["mask_violation_count"]) for row in overall_summary),
        "overall_summary": overall_summary,
        "cell_summary": cell_summary,
        "teacher_only_summary": teacher_summary,
        "pareto_cells": pareto_cells,
        "decision": decision,
        "interpretation_limits": {
            "switching_cost_excluded": True,
            "persistent_gurobi_model_reuse_not_implemented": True,
            "teacher_only_zero_response_must_remain_separate": True,
            "existing_development_content_only": True,
        },
    }
    _write_csv(output_dir / "dynamic_method_results.csv", flat_rows)
    _write_csv(output_dir / "dynamic_cell_summary.csv", cell_summary)
    _write_csv(output_dir / "dynamic_overall_summary.csv", overall_summary)
    _write_csv(output_dir / "quality_references.csv", reference_rows)
    _write_json_atomic(output_dir / "dynamic_preference_summary.json", summary)
    lines = [
        "# 方案4 Phase A：动态偏好调整诊断",
        "",
        f"- 状态：{decision['status']}。",
        f"- 样本：{len(payloads)} 个实例—动态条件；方法结果 {len(flat_rows)} 条。",
        f"- 5 秒质量参考最优证明率：{proof_rate:.2%}。",
        f"- PPO 旧解启动进入 Pareto 的条件数：{ppo_pareto_count}/6。",
        f"- 相对 PPO 重启的中位时延下降：{latency_reduction:.2%}。",
        f"- 被登记基线同时支配：{', '.join(dominated_by) if dominated_by else '无'}。",
        "- 未训练 PPO；未加入 assignment 切换成本；未实现持久化 Gurobi 模型复用。",
        "",
        "## 总体结果",
        "",
        "| 方法 | mean bounded gap | 正增益率 | median latency(s) | assignment change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall_summary:
        lines.append(
            f"| {row['method_id']} | {float(row['mean_bounded_gap']):.4%} | "
            f"{float(row['positive_improvement_rate']):.2%} | "
            f"{float(row['median_latency_sec']):.6f} | "
            f"{float(row['mean_assignment_change_rate']):.2%} |"
        )
    (output_dir / "dynamic_preference_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _hash_manifest(output_dir: Path, config_path: Path) -> None:
    inputs = [
        config_path,
        Path("pa_moap_rl/experiments/dynamic_preference_adjustment_v1.py"),
        Path("pa_moap_rl/experiments/run_dynamic_preference_adjustment_v1.py"),
    ]
    outputs = [
        path for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "evidence_hashes.json"
    ]
    _write_json_atomic(
        output_dir / "evidence_hashes.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {str(path): _sha256(path) for path in inputs},
            "outputs": {str(path): _sha256(path) for path in outputs},
        },
    )


def run(
    config_path: str | Path,
    *,
    validate_only: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    if config.get("status") != "user_confirmed":
        raise ValueError("Dynamic-preference Phase A is not user-confirmed.")
    if config.get("engine_version") != DYNAMIC_ENGINE_VERSION:
        raise ValueError("Dynamic-preference engine version mismatch.")
    frozen, objective = _validate_frozen_config(config)
    project_config = load_config()
    core_rows = _read_csv(config["data"]["core_manifest"])
    selected = _select_strata(core_rows, int(config["data"]["expected_strata"]))
    baseline_scenario, transition_scenarios = build_dynamic_scenarios(config)
    first_instance = _condition(
        selected[0], baseline_scenario, project_config=project_config, objective=objective
    )
    requested_device = torch.device(str(config["measurement"]["device"]))
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Dynamic-preference Phase A requires the registered CUDA device.")
    if bool(config["measurement"]["deterministic"]):
        torch.use_deterministic_algorithms(True)
    model, ppo_config, source_metadata, model_load_sec = _load_model_bundle(
        config=config,
        frozen=frozen,
        objective=objective,
        first_instance=first_instance,
        device=requested_device,
    )
    validation = {
        "status": "validated_no_experiment" if validate_only else "validated",
        "runner_version": RUNNER_VERSION,
        "engine_version": DYNAMIC_ENGINE_VERSION,
        "config_sha256": _sha256(config_path),
        "objective_hash": objective.config_hash,
        "checkpoint_sha256": config["frozen_ppo"]["checkpoint_sha256"],
        "source_data_hash": source_metadata["data_hash"],
        "selected_strata": len(selected),
        "transition_count_per_instance": len(transition_scenarios),
        "model_load_sec": model_load_sec,
        "frozen_inference": {
            "max_steps": int(config["methods"]["ppo_restart"]["max_steps"]),
            "patience": int(config["methods"]["ppo_restart"]["patience"]),
            "checkpoint_ppo_config_max_steps": int(ppo_config.env_max_steps),
            "checkpoint_ppo_config_patience": int(ppo_config.env_patience),
        },
    }
    if validate_only:
        return validation
    run_rows = selected[:1] if smoke else selected
    run_scenarios = transition_scenarios[:2] if smoke else transition_scenarios
    suffix = "_smoke" if smoke else ""
    output_dir = Path(str(config["run"]["output_dir"]) + suffix).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        **validation,
        "status": "running",
        "mode": "smoke" if smoke else "formal_phase_a",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(requested_device) if requested_device.type == "cuda" else None,
        "no_training": True,
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "objective_hash", "checkpoint_sha256", "mode"):
            if previous.get(key) != metadata.get(key):
                raise ValueError(f"Existing dynamic output mismatch: {key}")
    _write_json_atomic(metadata_path, metadata)
    _write_json_atomic(
        output_dir / "scenario_snapshot.json",
        {
            "baseline": baseline_scenario.to_dict(),
            "transitions": [item.to_dict() for item in transition_scenarios],
        },
    )
    _write_csv(output_dir / "selected_strata.csv", run_rows)
    payloads: list[dict[str, Any]] = []
    for row in run_rows:
        baseline_instance = _condition(
            row, baseline_scenario, project_config=project_config, objective=objective
        )
        baseline_cache_path = output_dir / "baseline_old_solutions" / f"{_case_key(row['original_id'])}.json"
        if baseline_cache_path.exists():
            baseline_cache = json.loads(baseline_cache_path.read_text(encoding="utf-8"))
            old_assignment = np.asarray(baseline_cache["assignment"], dtype=np.int64)
        else:
            old_result = run_dynamic_policy(
                instance=baseline_instance,
                model=model,
                project_config=project_config,
                objective=objective,
                max_steps=int(config["methods"]["ppo_restart"]["max_steps"]),
                patience=int(config["methods"]["ppo_restart"]["patience"]),
                initial_assignment=None,
            )
            old_assignment = old_result.assignment
            _write_json_atomic(
                baseline_cache_path,
                {
                    "original_id": row["original_id"],
                    "scenario_id": baseline_scenario.scenario_id,
                    "assignment": old_assignment.tolist(),
                    "assignment_sha256": _assignment_sha256(old_assignment),
                    "total_score": old_result.total_score,
                    "runtime_sec": old_result.runtime_sec,
                    "step_count": old_result.step_count,
                    "done_reason": old_result.done_reason,
                },
            )
        for scenario in run_scenarios:
            result_path = (
                output_dir / "instances" / _case_key(row["original_id"])
                / f"{scenario.scenario_id}.json"
            )
            if result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                payload = _run_case(
                    row=row,
                    scenario=scenario,
                    baseline_instance=baseline_instance,
                    old_assignment=old_assignment,
                    config=config,
                    project_config=project_config,
                    objective=objective,
                    model=model,
                )
                payload["config_sha256"] = validation["config_sha256"]
                payload["objective_hash"] = objective.config_hash
                payload["checkpoint_sha256"] = config["frozen_ppo"]["checkpoint_sha256"]
                _write_json_atomic(result_path, payload)
            payloads.append(payload)
    summary = _summarize(config=config, output_dir=output_dir, payloads=payloads)
    metadata.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dynamic_case_count": len(payloads),
            "method_result_count": len(payloads) * len(METHOD_IDS),
            "decision_status": summary["decision"]["status"],
            "eligible_for_phase_b": summary["decision"]["eligible_for_phase_b"],
        }
    )
    _write_json_atomic(metadata_path, metadata)
    _hash_manifest(output_dir, config_path)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    result = run(args.config, validate_only=args.validate_only, smoke=args.smoke_test)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
