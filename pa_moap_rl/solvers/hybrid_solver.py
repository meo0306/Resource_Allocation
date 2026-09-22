"""Hybrid refinement that starts from a PPO assignment.

This module intentionally reuses the existing one-point best-improvement local
search.  It does not define another neighborhood or another local-search rule;
it only supplies the PPO solution as the initial assignment and limits the
number of accepted moves.
"""

from __future__ import annotations

import time

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.local_search_solver import (
    has_positive_one_point_improvement,
    local_search_best_improvement,
)
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result
from pa_moap_rl.utils.scoring import score_assignment


def ppo_plus_short_local_search(
    instance: AssignmentInstance,
    ppo_assignment: np.ndarray | list[int],
    *,
    accepted_move_budget: int,
    ppo_runtime: float = 0.0,
    tolerance: float = 1.0e-12,
    certify_if_budget_exhausted: bool = False,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """Refine a PPO result with a bounded number of best-improvement moves."""

    budget = int(accepted_move_budget)
    if budget <= 0:
        raise ValueError("accepted_move_budget must be positive.")
    initial = np.asarray(ppo_assignment, dtype=np.int64)
    initial_score = score_assignment(initial, instance=instance, objective=objective).total_score

    start = time.perf_counter()
    refined = local_search_best_improvement(
        instance,
        initial_assignment=initial,
        max_iter=budget,
        tolerance=tolerance,
        objective=objective,
    )
    refinement_runtime = time.perf_counter() - start
    accepted_moves = max(0, len(refined.history) - 1)
    stopped_early = accepted_moves < budget
    certified = stopped_early
    if not certified and certify_if_budget_exhausted:
        certified = not has_positive_one_point_improvement(
            instance, refined.assignment, tolerance=tolerance, objective=objective
        )

    result = make_solver_result(
        solver_name=f"ppo_plus_short_local_search_{budget}",
        instance=instance,
        assignment=refined.assignment,
        runtime=float(ppo_runtime) + refinement_runtime,
        history=refined.history,
        initial_score=initial_score,
        objective=objective,
    )
    result.metrics.update(
        {
            "ppo_runtime": float(ppo_runtime),
            "refinement_runtime": refinement_runtime,
            "accepted_move_budget": budget,
            "accepted_moves": accepted_moves,
            "termination_reason": (
                "one_point_local_optimum" if certified else "accepted_move_limit"
            ),
            "iteration_limit_reached": not certified,
            "one_point_local_optimum_certified": certified,
            "improvement_over_ppo": float(result.score.total_score - initial_score),
        }
    )
    return result


__all__ = ["ppo_plus_short_local_search"]
