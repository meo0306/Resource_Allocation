"""PA-MOAP 的单点替换局部搜索求解器。

局部搜索从一个可行 assignment 出发，每次只替换一个知识点的方法，并用完整
目标 `J` 判断是否接受。`best` 策略每轮选择收益最大的替换；`first` 策略找到
第一个正收益替换就接受，可配合 shuffle 产生随机搜索路径。
"""

from __future__ import annotations

import time

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.greedy_solver import effect_only_greedy, scalarized_independent_assignment
from pa_moap_rl.solvers.random_solver import random_legal_assignment
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result
from pa_moap_rl.utils.scoring import IncrementalAssignmentScorer, score_assignment


def _validate_initial(instance: AssignmentInstance, assignment: np.ndarray | list[int]) -> np.ndarray:
    """校验初始 assignment 是否 shape 正确且满足静态可行性。"""

    values = np.asarray(assignment, dtype=np.int64).copy()
    if values.shape != (instance.n,):
        raise ValueError(f"initial_assignment shape must be {(instance.n,)}, got {values.shape}.")
    if np.any(values < 0) or np.any(values >= instance.m):
        raise ValueError("initial_assignment contains out-of-range method ids.")
    if np.any(~instance.feasible_mask[np.arange(instance.n), values]):
        raise ValueError("initial_assignment violates feasible_mask.")
    return values


def _best_improving_action(
    instance: AssignmentInstance,
    assignment: np.ndarray,
    current_score: float,
    tolerance: float,
    objective: ObjectiveSpec | None = None,
) -> tuple[tuple[int, int] | None, float]:
    """穷举所有合法单点替换，返回收益最大的正向动作。"""

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
            candidate_score = score_assignment(
                candidate, instance=instance, objective=objective
            ).total_score
            delta = candidate_score - current_score
            if delta > best_delta + tolerance:
                best_delta = delta
                best_action = (i, method)
    return best_action, best_delta


def _first_improving_action(
    instance: AssignmentInstance,
    assignment: np.ndarray,
    current_score: float,
    tolerance: float,
    rng: np.random.Generator | None = None,
    objective: ObjectiveSpec | None = None,
) -> tuple[tuple[int, int] | None, float]:
    """按节点/方法顺序或随机顺序返回第一个正向改进动作。"""

    nodes = np.arange(instance.n)
    if rng is not None:
        nodes = rng.permutation(nodes)
    for i_raw in nodes:
        i = int(i_raw)
        methods = np.flatnonzero(instance.feasible_mask[i])
        if rng is not None:
            methods = rng.permutation(methods)
        current_method = int(assignment[i])
        for method_raw in methods:
            method = int(method_raw)
            if method == current_method:
                continue
            candidate = assignment.copy()
            candidate[i] = method
            candidate_score = score_assignment(
                candidate, instance=instance, objective=objective
            ).total_score
            delta = candidate_score - current_score
            if delta > tolerance:
                return (i, method), float(delta)
    return None, 0.0


def _run_local_search_full_reference(
    instance: AssignmentInstance,
    assignment: np.ndarray,
    *,
    strategy: str,
    max_iter: int | None,
    tolerance: float,
    rng: np.random.Generator | None = None,
    objective: ObjectiveSpec | None = None,
) -> tuple[np.ndarray, list[float]]:
    """执行局部搜索主循环并记录每次接受动作后的目标值。"""

    current_score = score_assignment(
        assignment, instance=instance, objective=objective
    ).total_score
    history = [current_score]
    max_iter = int(max_iter if max_iter is not None else instance.n * instance.m)

    for _ in range(max_iter):
        if strategy == "best":
            action, delta = _best_improving_action(
                instance, assignment, current_score, tolerance, objective
            )
        elif strategy == "first":
            action, delta = _first_improving_action(
                instance,
                assignment,
                current_score,
                tolerance,
                rng=rng,
                objective=objective,
            )
        else:
            raise ValueError(f"unknown local search strategy: {strategy!r}")

        # 找不到正收益动作时停止，此时是当前邻域定义下的局部最优。
        if action is None:
            break
        assignment[action[0]] = action[1]
        current_score += delta
        history.append(current_score)
    return assignment, history


def _run_local_search(
    instance: AssignmentInstance,
    assignment: np.ndarray,
    *,
    strategy: str,
    max_iter: int | None,
    tolerance: float,
    rng: np.random.Generator | None = None,
    objective: ObjectiveSpec | None = None,
) -> tuple[np.ndarray, list[float]]:
    '''Run local search using cached O(M) replacement scores.'''

    scorer = IncrementalAssignmentScorer(assignment, instance, objective=objective)
    history = [scorer.score]
    limit = int(max_iter if max_iter is not None else instance.n * instance.m)
    for _ in range(limit):
        nodes = rng.permutation(instance.n) if rng is not None else np.arange(instance.n)
        best_action: tuple[int, int] | None = None
        best_delta = 0.0
        for raw_i in nodes:
            i = int(raw_i)
            methods = np.flatnonzero(instance.feasible_mask[i])
            if rng is not None:
                methods = rng.permutation(methods)
            for raw_method in methods:
                method = int(raw_method)
                if method == int(scorer.assignment[i]):
                    continue
                delta = scorer.candidate_delta(i, method)
                if delta > tolerance and strategy == 'first':
                    best_action = (i, method)
                    best_delta = delta
                    break
                if delta > best_delta + tolerance:
                    best_action = (i, method)
                    best_delta = delta
            if best_action is not None and strategy == 'first':
                break
        if strategy not in {'best', 'first'}:
            raise ValueError(f'unknown local search strategy: {strategy!r}')
        if best_action is None:
            break
        scorer.apply(*best_action)
        history.append(scorer.score)
    return scorer.assignment.copy(), history


def local_search_best_improvement(
    instance: AssignmentInstance,
    initial_assignment: np.ndarray | list[int] | None = None,
    *,
    max_iter: int | None = None,
    tolerance: float = 1.0e-12,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """运行确定性的 best-improvement 单点局部搜索。"""

    start = time.perf_counter()
    assignment = (
        scalarized_independent_assignment(instance, objective=objective)
        if initial_assignment is None
        else _validate_initial(instance, initial_assignment)
    )
    initial_score = score_assignment(assignment, instance=instance, objective=objective).total_score
    assignment, history = _run_local_search(
        instance,
        assignment,
        strategy="best",
        max_iter=max_iter,
        tolerance=tolerance,
        objective=objective,
    )
    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="local_search_best_improvement",
        instance=instance,
        assignment=assignment,
        runtime=runtime,
        history=history,
        initial_score=initial_score,
        greedy_score=effect_only_greedy(instance, objective=objective).score.total_score,
        objective=objective,
    )


def local_search_first_improvement(
    instance: AssignmentInstance,
    initial_assignment: np.ndarray | list[int] | None = None,
    *,
    max_iter: int | None = None,
    tolerance: float = 1.0e-12,
    seed: int | None = None,
    objective: ObjectiveSpec | None = None,
    shuffle: bool = False,
) -> SolverResult:
    """运行 first-improvement 单点局部搜索。"""

    start = time.perf_counter()
    rng = np.random.default_rng(seed) if shuffle else None
    assignment = (
        scalarized_independent_assignment(instance, objective=objective)
        if initial_assignment is None
        else _validate_initial(instance, initial_assignment)
    )
    initial_score = score_assignment(assignment, instance=instance, objective=objective).total_score
    assignment, history = _run_local_search(
        instance,
        assignment,
        strategy="first",
        max_iter=max_iter,
        tolerance=tolerance,
        rng=rng,
        objective=objective,
    )
    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="local_search_first_improvement",
        instance=instance,
        assignment=assignment,
        runtime=runtime,
        history=history,
        initial_score=initial_score,
        greedy_score=effect_only_greedy(instance, objective=objective).score.total_score,
        objective=objective,
    )


def random_restart_local_search(
    instance: AssignmentInstance,
    *,
    restarts: int = 8,
    max_iter: int | None = None,
    tolerance: float = 1.0e-12,
    seed: int | None = None,
    objective: ObjectiveSpec | None = None,
    strategy: str = "best",
) -> SolverResult:
    """从多个随机可行初解出发运行局部搜索，并返回最好结果。"""

    if restarts <= 0:
        raise ValueError("restarts must be positive.")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_history: list[float] = []
    best_score = -np.inf

    # 多启动用于降低单个局部最优对结果的影响。
    for _ in range(int(restarts)):
        assignment = random_legal_assignment(instance, rng)
        assignment, history = _run_local_search(
            instance,
            assignment,
            strategy=strategy,
            max_iter=max_iter,
            tolerance=tolerance,
            objective=objective,
            rng=rng if strategy == "first" else None,
        )
        score = score_assignment(assignment, instance=instance, objective=objective).total_score
        if score > best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_history = history

    assert best_assignment is not None
    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name=f"random_restart_local_search_{strategy}",
        instance=instance,
        assignment=best_assignment,
        runtime=runtime,
        history=best_history,
        greedy_score=effect_only_greedy(instance, objective=objective).score.total_score,
        objective=objective,
    )


def has_positive_one_point_improvement(
    instance: AssignmentInstance,
    assignment: np.ndarray | list[int],
    *,
    tolerance: float = 1.0e-12,
    objective: ObjectiveSpec | None = None,
) -> bool:
    """判断当前 assignment 是否还存在任意正收益单点替换。"""

    values = _validate_initial(instance, assignment)
    scorer = IncrementalAssignmentScorer(values, instance, objective=objective)
    for i in range(instance.n):
        for method in np.flatnonzero(instance.feasible_mask[i]):
            if int(method) != int(values[i]) and scorer.candidate_delta(i, int(method)) > tolerance:
                return True
    return False


__all__ = [
    "has_positive_one_point_improvement",
    "local_search_best_improvement",
    "local_search_first_improvement",
    "random_restart_local_search",
]
