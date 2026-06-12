"""PA-MOAP 的贪心 baseline 求解器。

这里包含三类参照：只看理论效果的 effect-only greedy；只看局部加权效果/
偏好的 scalarized independent greedy；以及在完整目标 J 上反复执行最佳单点
改进的 one-point greedy improvement。
"""

from __future__ import annotations

import time

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result
from pa_moap_rl.utils.scoring import score_assignment


def _argmax_feasible(score_matrix: np.ndarray, feasible_mask: np.ndarray) -> np.ndarray:
    """在每一行中选择可行且分数最高的方法。"""

    scores = np.asarray(score_matrix, dtype=np.float64)
    feasible = np.asarray(feasible_mask, dtype=bool)
    if scores.shape != feasible.shape:
        raise ValueError(f"score_matrix shape {scores.shape} must equal feasible_mask shape {feasible.shape}.")
    if np.any(feasible.sum(axis=1) < 1):
        raise ValueError("each node must have at least one feasible method.")
    masked_scores = np.where(feasible, scores, -np.inf)
    return np.argmax(masked_scores, axis=1).astype(np.int64)


def effect_only_assignment(instance: AssignmentInstance) -> np.ndarray:
    """为每个节点选择 `effect_matrix` 最大的可行方法。"""

    return _argmax_feasible(instance.effect_matrix, instance.feasible_mask)


def solve_effect_only_greedy(instance: AssignmentInstance) -> SolverResult:
    """只最大化局部理论效果的 baseline。"""

    start = time.perf_counter()
    assignment = effect_only_assignment(instance)
    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="effect_only_greedy",
        instance=instance,
        assignment=assignment,
        runtime=runtime,
    )


def scalarized_independent_assignment(instance: AssignmentInstance) -> np.ndarray:
    """在局部加权分数下为每个节点独立选择最优可行方法。"""

    weights = instance.weights
    # 此处不考虑 F_G 和软约束，因为它们依赖全局方法分布。
    local_scores = (
        float(weights["effect"]) * instance.effect_matrix
        + float(weights["student"]) * instance.student_pref
        + float(weights["teacher"]) * instance.teacher_pref
    )
    return _argmax_feasible(local_scores, instance.feasible_mask)


def solve_scalarized_independent_greedy(instance: AssignmentInstance) -> SolverResult:
    """最大化局部加权效果/偏好的独立贪心 baseline。"""

    start = time.perf_counter()
    assignment = scalarized_independent_assignment(instance)
    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="scalarized_independent_greedy",
        instance=instance,
        assignment=assignment,
        runtime=runtime,
        greedy_score=solve_effect_only_greedy(instance).score.total_score,
    )


def one_point_greedy_improvement(
    instance: AssignmentInstance,
    initial_assignment: np.ndarray | list[int] | None = None,
    *,
    max_iter: int | None = None,
    tolerance: float = 1.0e-12,
) -> SolverResult:
    """在完整目标 `J` 上反复执行收益最大的正向单点替换。"""

    start = time.perf_counter()
    assignment = (
        scalarized_independent_assignment(instance)
        if initial_assignment is None
        else np.asarray(initial_assignment, dtype=np.int64).copy()
    )
    if assignment.shape != (instance.n,):
        raise ValueError(f"initial_assignment shape must be {(instance.n,)}, got {assignment.shape}.")
    if np.any(~instance.feasible_mask[np.arange(instance.n), assignment]):
        raise ValueError("initial_assignment violates feasible_mask.")

    current_score = score_assignment(assignment, instance=instance).total_score
    initial_score = current_score
    history = [current_score]
    iterations = 0
    max_iter = int(max_iter if max_iter is not None else instance.n * instance.m)

    # 每轮穷举所有合法单点替换，若不存在正收益动作则到达一阶局部最优。
    while iterations < max_iter:
        best_delta = 0.0
        best_action: tuple[int, int] | None = None
        for i in range(instance.n):
            current_method = int(assignment[i])
            for method in np.flatnonzero(instance.feasible_mask[i]):
                method = int(method)
                if method == current_method:
                    continue
                candidate = assignment.copy()
                candidate[i] = method
                candidate_score = score_assignment(candidate, instance=instance).total_score
                delta = candidate_score - current_score
                if delta > best_delta + tolerance:
                    best_delta = delta
                    best_action = (i, method)

        if best_action is None:
            break
        assignment[best_action[0]] = best_action[1]
        current_score += best_delta
        history.append(current_score)
        iterations += 1

    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="one_point_greedy_improvement",
        instance=instance,
        assignment=assignment,
        runtime=runtime,
        history=history,
        initial_score=initial_score,
        greedy_score=solve_effect_only_greedy(instance).score.total_score,
    )


def effect_only_greedy(instance: AssignmentInstance) -> SolverResult:
    """`solve_effect_only_greedy` 的别名。"""

    return solve_effect_only_greedy(instance)


def scalarized_independent_greedy(instance: AssignmentInstance) -> SolverResult:
    """`solve_scalarized_independent_greedy` 的别名。"""

    return solve_scalarized_independent_greedy(instance)


__all__ = [
    "effect_only_assignment",
    "effect_only_greedy",
    "one_point_greedy_improvement",
    "scalarized_independent_assignment",
    "scalarized_independent_greedy",
    "solve_effect_only_greedy",
    "solve_scalarized_independent_greedy",
]
