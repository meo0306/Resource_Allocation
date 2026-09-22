"""随机可行 baseline。

该 baseline 只保证每个知识点选择静态 `feasible_mask` 允许的方法，不使用任何
评分信息。它主要用于提供最低复杂度的随机参照和 seed 可复现实验。
"""

from __future__ import annotations

import time

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result


def random_legal_assignment(instance: AssignmentInstance, rng: np.random.Generator) -> np.ndarray:
    """对每个节点从静态可行方法集合中均匀采样一个方法。"""

    assignment = np.empty(instance.n, dtype=np.int64)
    for i in range(instance.n):
        legal = np.flatnonzero(instance.feasible_mask[i])
        if legal.size == 0:
            raise ValueError(f"node {i} has no feasible method.")
        assignment[i] = int(rng.choice(legal))
    return assignment


def solve_random_legal(
    instance: AssignmentInstance,
    *,
    seed: int | None = None,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """运行一次随机可行 baseline，并包装为 `SolverResult`。"""

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    assignment = random_legal_assignment(instance, rng)
    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="random_legal",
        instance=instance,
        assignment=assignment,
        runtime=runtime,
        objective=objective,
    )


def random_legal(
    instance: AssignmentInstance,
    *,
    seed: int | None = None,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """`solve_random_legal` 的简短别名。"""

    return solve_random_legal(instance, seed=seed, objective=objective)


__all__ = ["random_legal", "random_legal_assignment", "solve_random_legal"]
