"""实验输出与解质量指标。

baseline、PPO 和批量实验都通过 `SolverResult` 返回结果。这里把可行性、
评分分项、运行时间和改进幅度整理成统一字典，保证不同求解器写出的 CSV
字段一致，便于后续 `analyze_results.py` 复现表格和图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.utils.masks import build_action_mask
from pa_moap_rl.utils.scoring import ScoreBreakdown, score_assignment


@dataclass(frozen=True)
class SolverResult:
    """baseline 与学习型 solver 共用的返回对象。"""

    solver_name: str
    assignment: np.ndarray
    score: ScoreBreakdown
    runtime: float
    metrics: dict[str, Any]
    history: list[float] = field(default_factory=list)


def as_assignment_array(assignment: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    """把 assignment 转为一维整数数组。"""

    values = np.asarray(assignment, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"assignment must be 1D, got shape {values.shape}.")
    return values


def mask_violation_count(instance: AssignmentInstance, assignment: np.ndarray | list[int]) -> int:
    """统计违反静态 `feasible_mask` 的节点数。"""

    values = as_assignment_array(assignment)
    if values.shape != (instance.n,):
        raise ValueError(f"assignment shape must be {(instance.n,)}, got {values.shape}.")

    invalid = (values < 0) | (values >= instance.m)
    count = int(np.sum(invalid))
    valid_rows = ~invalid
    if np.any(valid_rows):
        rows = np.arange(instance.n)[valid_rows]
        count += int(np.sum(~instance.feasible_mask[rows, values[valid_rows]]))
    return count


def is_hard_feasible(instance: AssignmentInstance, assignment: np.ndarray | list[int]) -> bool:
    """判断 assignment 是否满足所有静态硬可行性约束。"""

    return mask_violation_count(instance, assignment) == 0


def valid_action_ratio(instance: AssignmentInstance, assignment: np.ndarray | list[int]) -> float:
    """当前 assignment 下动态可替换动作的比例。"""

    action_mask = build_action_mask(as_assignment_array(assignment), instance.feasible_mask)
    return float(np.mean(action_mask))


def evaluate_assignment(
    instance: AssignmentInstance,
    assignment: np.ndarray | list[int],
    *,
    solver_name: str | None = None,
    runtime: float = 0.0,
    initial_score: float | None = None,
    greedy_score: float | None = None,
) -> dict[str, Any]:
    """计算可行性、评分、运行时间和可选改进幅度指标。"""

    values = as_assignment_array(assignment)
    score = score_assignment(values, instance=instance)
    # 这里的字段名直接进入实验 CSV，改名会影响分析脚本。
    metrics: dict[str, Any] = {
        "solver_name": solver_name,
        "hard_feasible_rate": 1.0 if is_hard_feasible(instance, values) else 0.0,
        "mask_violation_count": mask_violation_count(instance, values),
        "valid_action_ratio": valid_action_ratio(instance, values),
        "total_score": score.total_score,
        "effect_score": score.effect_score,
        "student_pref_score": score.student_pref_score,
        "teacher_pref_score": score.teacher_pref_score,
        "global_score": score.global_score,
        "soft_penalty": score.soft_penalty,
        "entropy": score.entropy,
        "diversity_violation": score.diversity_violation,
        "cap_violation": score.cap_violation,
        "runtime": float(runtime),
    }
    if initial_score is not None:
        metrics["improvement_over_initial"] = float(score.total_score - initial_score)
    if greedy_score is not None:
        metrics["improvement_over_greedy"] = float(score.total_score - greedy_score)
    return metrics


def make_solver_result(
    *,
    solver_name: str,
    instance: AssignmentInstance,
    assignment: np.ndarray | list[int],
    runtime: float,
    history: list[float] | None = None,
    initial_score: float | None = None,
    greedy_score: float | None = None,
) -> SolverResult:
    """根据最终 assignment 构造标准 `SolverResult`。"""

    values = as_assignment_array(assignment).copy()
    score = score_assignment(values, instance=instance)
    metrics = evaluate_assignment(
        instance,
        values,
        solver_name=solver_name,
        runtime=runtime,
        initial_score=initial_score,
        greedy_score=greedy_score,
    )
    return SolverResult(
        solver_name=solver_name,
        assignment=values,
        score=score,
        runtime=float(runtime),
        metrics=metrics,
        history=list(history or [score.total_score]),
    )


__all__ = [
    "SolverResult",
    "as_assignment_array",
    "evaluate_assignment",
    "is_hard_feasible",
    "make_solver_result",
    "mask_violation_count",
    "valid_action_ratio",
]
