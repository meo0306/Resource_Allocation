"""Exact or bounded PA-MOAP solving with Gurobi.

Entropy is represented at every attainable integer method count. Therefore
the piecewise-linear representation is exact at all feasible assignments.
Time-limited runs retain a certified incumbent and objective bound.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Iterable

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.greedy_solver import scalarized_independent_assignment
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result
from pa_moap_rl.utils.scoring import (
    DEFAULT_ENTROPY_MIN,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_CAP,
    DEFAULT_LAMBDA_DIV,
    DEFAULT_METHOD_CAP,
    score_assignment,
)


def _require_gurobi() -> tuple[Any, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError(
            "Gurobi solving requires the optional 'gurobipy' package and a "
            "valid local license. Install it with `pip install gurobipy`."
        ) from exc
    return gp, GRB


def gurobi_available() -> bool:
    """Return whether gurobipy is importable (not whether a license is valid)."""

    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return False
    return True


def _cap_vector(pi_cap: float | np.ndarray, method_count: int) -> np.ndarray:
    cap = np.asarray(pi_cap, dtype=np.float64)
    if cap.ndim == 0:
        cap = np.full(method_count, float(cap), dtype=np.float64)
    if cap.shape != (method_count,):
        raise ValueError(f"pi_cap must be scalar or shape {(method_count,)}, got {cap.shape}.")
    if np.any(cap <= 0.0) or np.any(cap > 1.0):
        raise ValueError("pi_cap values must be in (0, 1].")
    return cap


def _validate_warm_start(
    instance: AssignmentInstance,
    assignment: np.ndarray | list[int],
) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    if values.shape != (instance.n,):
        raise ValueError(f"warm_start shape must be {(instance.n,)}, got {values.shape}.")
    if np.any(values < 0) or np.any(values >= instance.m):
        raise ValueError("warm_start contains an out-of-range method id.")
    if np.any(~instance.feasible_mask[np.arange(instance.n), values]):
        raise ValueError("warm_start violates feasible_mask.")
    return values.copy()


def solve_gurobi(
    instance: AssignmentInstance,
    *,
    time_limit: float | None = 120.0,
    mip_gap: float = 0.0,
    seed: int = 0,
    threads: int | None = None,
    output_flag: bool = False,
    log_file: str | None = None,
    warm_start: np.ndarray | list[int] | None = None,
    H_min: float = DEFAULT_ENTROPY_MIN,
    pi_cap: float | np.ndarray = DEFAULT_METHOD_CAP,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
    lambda_cap: float = DEFAULT_LAMBDA_CAP,
    epsilon: float = DEFAULT_EPSILON,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """Solve one instance exactly or return a certified time-limit bound."""

    if objective is not None and (
        float(H_min) != DEFAULT_ENTROPY_MIN
        or np.asarray(pi_cap).ndim != 0
        or float(np.asarray(pi_cap)) != DEFAULT_METHOD_CAP
        or float(lambda_div) != DEFAULT_LAMBDA_DIV
        or float(lambda_cap) != DEFAULT_LAMBDA_CAP
        or float(epsilon) != DEFAULT_EPSILON
    ):
        raise ValueError(
            "Do not combine ObjectiveSpec with external threshold or penalty overrides."
        )
    if time_limit is not None and float(time_limit) <= 0.0:
        raise ValueError("time_limit must be positive or None.")
    if float(mip_gap) < 0.0:
        raise ValueError("mip_gap must be non-negative.")
    if not 0.0 <= float(H_min) <= 1.0:
        raise ValueError("H_min must be in [0, 1].")

    if objective is not None:
        H_min = objective.entropy_min
        pi_cap = objective.method_cap
        epsilon = objective.epsilon
        target_values = objective.target_distribution
        alpha_effect = objective.alpha_effect
        alpha_student = objective.alpha_student
        alpha_teacher = objective.alpha_teacher
        alpha_global = objective.alpha_global
        beta_entropy = objective.beta_entropy
        beta_cap = objective.beta_cap
    else:
        target_values = tuple(float(value) for value in instance.target_distribution)
        weights = instance.weights
        alpha_effect = float(weights['effect'])
        alpha_student = float(weights['student'])
        alpha_teacher = float(weights['teacher'])
        alpha_global = float(weights['global'])
        beta_entropy = float(weights['soft']) * float(lambda_div)
        beta_cap = float(weights['soft']) * float(lambda_cap)

    gp, GRB = _require_gurobi()
    cap = _cap_vector(pi_cap, instance.m)
    initial = _validate_warm_start(
        instance,
        scalarized_independent_assignment(instance, objective=objective)
        if warm_start is None
        else warm_start,
    )

    model = gp.Model(f"pa_moap_{instance.instance_name}")
    model.Params.OutputFlag = int(bool(output_flag))
    model.Params.Seed = int(seed)
    model.Params.MIPGap = float(mip_gap)
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.OptimalityTol = 1.0e-9
    model.Params.IntFeasTol = 1.0e-9
    model.Params.NumericFocus = 3
    model.Params.IntegralityFocus = 1
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if threads is not None:
        if int(threads) <= 0:
            raise ValueError("threads must be positive or None.")
        model.Params.Threads = int(threads)
    if log_file:
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 0
        model.Params.LogFile = str(log_file)

    n = float(instance.n)
    x: dict[tuple[int, int], Any] = {}
    for i in range(instance.n):
        for raw_method in np.flatnonzero(instance.feasible_mask[i]):
            m = int(raw_method)
            x[i, m] = model.addVar(vtype=GRB.BINARY, name=f"x[{i},{m}]")
    model.update()
    for i in range(instance.n):
        model.addConstr(
            gp.quicksum(x[i, int(m)] for m in np.flatnonzero(instance.feasible_mask[i])) == 1,
            name=f"assign[{i}]",
        )

    counts = model.addVars(instance.m, lb=0.0, ub=n, vtype=GRB.INTEGER, name="count")
    for m in range(instance.m):
        model.addConstr(
            counts[m] == gp.quicksum(x[i, m] for i in range(instance.n) if (i, m) in x),
            name=f"count_def[{m}]",
        )

    abs_deviation = model.addVars(instance.m, lb=0.0, name="target_abs")
    cap_excess = model.addVars(instance.m, lb=0.0, name="cap_excess")
    entropy_terms = model.addVars(instance.m, lb=0.0, name="entropy_term")
    count_points = list(range(instance.n + 1))
    log_m = math.log(instance.m) if instance.m > 1 else 1.0
    for m in range(instance.m):
        distribution = counts[m] / n
        target = float(target_values[m]) if target_values is not None else 0.0
        model.addConstr(abs_deviation[m] >= distribution - target, name=f"target_pos[{m}]")
        model.addConstr(abs_deviation[m] >= target - distribution, name=f"target_neg[{m}]")
        model.addConstr(cap_excess[m] >= distribution - float(cap[m]), name=f"cap[{m}]")
        entropy_values = []
        for count in count_points:
            probability = count / n
            contribution = (
                0.0
                if probability <= 0.0 or instance.m <= 1
                else -probability * math.log(probability + float(epsilon)) / log_m
            )
            entropy_values.append(max(0.0, contribution))
        model.addGenConstrPWL(
            counts[m], entropy_terms[m], count_points, entropy_values, name=f"entropy_pwl[{m}]"
        )

    entropy = model.addVar(lb=0.0, ub=1.0, name="entropy")
    diversity_shortfall = model.addVar(lb=0.0, name="diversity_shortfall")
    model.addConstr(
        entropy == gp.quicksum(entropy_terms[m] for m in range(instance.m)),
        name="entropy_sum",
    )
    model.addConstr(
        diversity_shortfall >= float(H_min) - entropy,
        name="diversity_shortfall_def",
    )

    weights = instance.weights if objective is None else {
        'effect': alpha_effect,
        'student': alpha_student,
        'teacher': alpha_teacher,
        'global': alpha_global,
        'soft': 1.0,
    }
    if objective is not None:
        lambda_div = beta_entropy
        lambda_cap = beta_cap
    local_objective = gp.quicksum(
        x[i, m]
        * (
            float(weights["effect"]) * float(instance.effect_matrix[i, m])
            + float(weights["student"]) * float(instance.student_pref[i, m])
            + float(weights["teacher"]) * float(instance.teacher_pref[i, m])
        )
        / n
        for (i, m) in x
    )
    global_objective = float(weights["global"]) * (
        1.0 - 0.5 * gp.quicksum(abs_deviation[m] for m in range(instance.m))
    )
    soft_penalty = float(weights["soft"]) * (
        float(lambda_div) * diversity_shortfall * diversity_shortfall
        + float(lambda_cap)
        * gp.quicksum(cap_excess[m] * cap_excess[m] for m in range(instance.m))
    )
    model.setObjective(local_objective + global_objective - soft_penalty, GRB.MAXIMIZE)
    for (i, m), variable in x.items():
        variable.Start = 1.0 if int(initial[i]) == m else 0.0

    start = time.perf_counter()
    model.optimize()
    wall_runtime = time.perf_counter() - start
    status_names = {
        GRB.LOADED: "loaded",
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.CUTOFF: "cutoff",
        GRB.ITERATION_LIMIT: "iteration_limit",
        GRB.NODE_LIMIT: "node_limit",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SOLUTION_LIMIT: "solution_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.NUMERIC: "numeric",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    status_name = status_names.get(int(model.Status), f"status_{int(model.Status)}")
    if int(model.SolCount) <= 0:
        raise RuntimeError(f"Gurobi returned {status_name} without a feasible incumbent.")

    assignment = np.empty(instance.n, dtype=np.int64)
    for i in range(instance.n):
        legal = [int(m) for m in np.flatnonzero(instance.feasible_mask[i])]
        assignment[i] = max(legal, key=lambda m: float(x[i, m].X))
    result = make_solver_result(
        solver_name="gurobi_exact_or_bounded",
        instance=instance,
        assignment=assignment,
        runtime=wall_runtime,
        history=[float(model.ObjVal)],
        initial_score=score_assignment(initial, instance=instance, objective=objective).total_score,
        objective=objective,
    )
    objective_bound = float(model.ObjBound)
    recomputed = float(result.score.total_score)
    recompute_error = abs(float(model.ObjVal) - recomputed)
    if recompute_error > 1.0e-9:
        model.dispose()
        raise RuntimeError(
            f"Gurobi objective recomputation error {recompute_error:.3e} exceeds 1e-9."
        )
    result.metrics.update(
        {
            "gurobi_status": status_name,
            "status_code": int(model.Status),
            "proven_optimal": bool(int(model.Status) == GRB.OPTIMAL),
            "solution_count": int(model.SolCount),
            "objective_incumbent": float(model.ObjVal),
            "objective_bound": objective_bound,
            "absolute_optimality_gap": max(0.0, objective_bound - recomputed),
            "mip_gap": float(model.MIPGap),
            "node_count": float(model.NodeCount),
            "simplex_iterations": float(model.IterCount),
            "solver_runtime": float(model.Runtime),
            "objective_recompute_error": recompute_error,
            "time_limit": None if time_limit is None else float(time_limit),
            "requested_mip_gap": float(mip_gap),
            "gurobi_seed": int(seed),
            "gurobi_threads": None if threads is None else int(threads),
        }
    )
    model.dispose()
    return result


def brute_force_optimum(
    instance: AssignmentInstance,
    *,
    max_assignments: int = 1_000_000,
    H_min: float = DEFAULT_ENTROPY_MIN,
    pi_cap: float | np.ndarray = DEFAULT_METHOD_CAP,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
    lambda_cap: float = DEFAULT_LAMBDA_CAP,
    epsilon: float = DEFAULT_EPSILON,
    objective: ObjectiveSpec | None = None,
) -> tuple[np.ndarray, float]:
    """Enumerate a tiny instance, intended only for formulation tests."""

    legal_methods: list[Iterable[int]] = [
        [int(m) for m in np.flatnonzero(instance.feasible_mask[i])]
        for i in range(instance.n)
    ]
    search_size = math.prod(len(methods) for methods in legal_methods)
    if search_size > int(max_assignments):
        raise ValueError(
            f"Brute-force search has {search_size} assignments, exceeding "
            f"max_assignments={max_assignments}."
        )
    best_assignment: np.ndarray | None = None
    best_score = -math.inf
    for candidate in itertools.product(*legal_methods):
        assignment = np.asarray(candidate, dtype=np.int64)
        score = score_assignment(
            assignment,
            instance=instance,
            H_min=H_min,
            pi_cap=pi_cap,
            lambda_div=lambda_div,
            lambda_cap=lambda_cap,
            epsilon=epsilon,
            objective=objective,
        ).total_score
        if score > best_score:
            best_score = float(score)
            best_assignment = assignment.copy()
    assert best_assignment is not None
    return best_assignment, best_score


__all__ = ["brute_force_optimum", "gurobi_available", "solve_gurobi"]
