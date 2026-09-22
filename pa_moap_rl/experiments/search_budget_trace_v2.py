"""Deterministic PPO inference traces for search-budget diagnostics.

Inference-budget controls remain separate from PPO training hyperparameters.
Every executed single-node replacement and best-so-far incumbent is retained
so that anytime curves can be recomputed without rerunning the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import time
from typing import Any, Mapping

import torch

from pa_moap_rl.configs import PAConfig
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.actor_critic import MaskedActorCritic, unflatten_action
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.ppo_solver import observation_to_model_batch
from pa_moap_rl.utils.metrics import mask_violation_count
from pa_moap_rl.utils.scoring import score_assignment


@dataclass(frozen=True)
class SearchCondition:
    """One conceptual search setting for a particular instance size."""

    condition_id: str
    family: str
    max_steps: int
    patience: int
    requested_budget: str

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.patience <= 0:
            raise ValueError("max_steps and patience must be positive.")

    @property
    def execution_key(self) -> tuple[int, int]:
        """Settings that determine the actual deterministic rollout."""

        return self.max_steps, self.patience


def scaled_budget_for_n(n: int, *, unit: int = 128) -> int:
    """Round N upward to a positive multiple of the unit."""

    if n <= 0 or unit <= 0:
        raise ValueError("n and unit must be positive.")
    return int(unit * ceil(n / unit))


def build_search_conditions(
    protocol: Mapping[str, Any], *, n: int
) -> list[SearchCondition]:
    """Resolve fixed, N-scaled, and patience-control settings for one N."""

    conditions: list[SearchCondition] = []
    for anchor in protocol.get("anchor_conditions", []):
        max_steps = int(anchor["max_steps"])
        patience = int(anchor["patience"])
        conditions.append(
            SearchCondition(
                condition_id=str(anchor["condition_id"]),
                family="current_anchor",
                max_steps=max_steps,
                patience=patience,
                requested_budget=str(max_steps),
            )
        )

    fixed_budgets = [int(value) for value in protocol["fixed_step_budgets"]]
    if not fixed_budgets or any(value <= 0 for value in fixed_budgets):
        raise ValueError("fixed_step_budgets must contain positive integers.")
    if len(set(fixed_budgets)) != len(fixed_budgets):
        raise ValueError("fixed_step_budgets must be unique.")

    budget_patience = protocol.get("budget_scan_patience", "equal_budget")
    for budget in fixed_budgets:
        patience = budget if budget_patience == "equal_budget" else int(budget_patience)
        conditions.append(
            SearchCondition(
                condition_id=f"fixed_b{budget}_p{patience}",
                family="fixed_budget",
                max_steps=budget,
                patience=patience,
                requested_budget=str(budget),
            )
        )

    scaled = protocol["n_scaled_budget"]
    if scaled.get("formula") != "unit_times_ceil_n_over_unit":
        raise ValueError("Unsupported n_scaled_budget formula.")
    unit = int(scaled["unit"])
    scaled_budget = scaled_budget_for_n(n, unit=unit)
    scaled_patience = (
        scaled_budget
        if scaled.get("patience", "equal_budget") == "equal_budget"
        else int(scaled["patience"])
    )
    conditions.append(
        SearchCondition(
            condition_id=f"n_scaled_b{scaled_budget}_p{scaled_patience}",
            family="n_scaled_budget",
            max_steps=scaled_budget,
            patience=scaled_patience,
            requested_budget="n_scaled",
        )
    )

    patience_scan = protocol["patience_scan"]
    patience_budget = int(patience_scan["max_steps"])
    patience_values = [int(value) for value in patience_scan["values"]]
    if (
        patience_budget <= 0
        or not patience_values
        or any(value <= 0 for value in patience_values)
    ):
        raise ValueError("patience_scan values and max_steps must be positive.")
    if len(set(patience_values)) != len(patience_values):
        raise ValueError("patience_scan values must be unique.")
    for patience in patience_values:
        conditions.append(
            SearchCondition(
                condition_id=f"patience_b{patience_budget}_p{patience}",
                family="patience_control",
                max_steps=patience_budget,
                patience=patience,
                requested_budget=str(patience_budget),
            )
        )

    ids = [item.condition_id for item in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Search condition ids must be unique.")
    return conditions


def deterministic_policy_trace(
    instance: AssignmentInstance,
    *,
    model: MaskedActorCritic,
    project_config: PAConfig,
    objective: ObjectiveSpec,
    max_steps: int,
    patience: int,
    exact_score: float,
) -> dict[str, Any]:
    """Run masked argmax inference and retain a complete anytime trace."""

    if max_steps <= 0 or patience <= 0:
        raise ValueError("max_steps and patience must be positive.")
    env_cfg = project_config.default["environment"]
    env = MethodAssignmentEnv(
        max_steps=max_steps,
        patience=patience,
        improvement_eps=float(env_cfg["improvement_eps"]),
        config=project_config,
        objective=objective,
    )
    started = time.perf_counter()
    obs = env.reset(instance)
    device = next(model.parameters()).device
    initial_assignment = env.assignment.copy()
    initial_score = float(env.current_score.total_score)
    trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "elapsed_sec": 0.0,
            "current_score": initial_score,
            "best_score": initial_score,
            "absolute_gap_to_exact": max(0.0, exact_score - initial_score),
            "relative_gap_to_exact": max(0.0, exact_score - initial_score)
            / max(abs(exact_score), 1.0e-8),
            "accepted_moves": 0,
            "unique_action_count": 0,
            "covered_node_count": 0,
            "repeated_action_count": 0,
            "repeated_node_count": 0,
            "no_improve_steps": 0,
            "steps_to_best": 0,
        }
    ]
    seen_actions: set[tuple[int, int]] = set()
    seen_nodes: set[int] = set()
    accepted_moves = 0
    positive_reward_moves = 0
    new_best_moves = 0
    repeated_action_count = 0
    repeated_node_count = 0

    while not env.done:
        with torch.no_grad():
            batch = observation_to_model_batch(obs, device=device)
            output = model(batch)
            flat_action = torch.argmax(
                output.masked_logits.reshape(1, -1), dim=1
            )
        node_tensor, method_tensor = unflatten_action(flat_action, instance.m)
        node_index = int(node_tensor.item())
        method_index = int(method_tensor.item())
        action = (node_index, method_index)
        repeated_action = action in seen_actions
        repeated_node = node_index in seen_nodes
        if repeated_action:
            repeated_action_count += 1
        if repeated_node:
            repeated_node_count += 1
        seen_actions.add(action)
        seen_nodes.add(node_index)

        previous_best = float(env.best_score)
        obs, reward, _, info = env.step(action)
        positive_reward = reward > env.improvement_eps
        new_best = float(info["best_score"]) > previous_best + env.improvement_eps
        if positive_reward:
            positive_reward_moves += 1
            accepted_moves += 1
        if new_best:
            new_best_moves += 1
        best_score = float(info["best_score"])
        trace.append(
            {
                "step": int(info["step_count"]),
                "elapsed_sec": float(time.perf_counter() - started),
                "node_index": node_index,
                "method_index": method_index,
                "reward": float(reward),
                "current_score": float(info["current_score"]["total_score"]),
                "best_score": best_score,
                "absolute_gap_to_exact": max(0.0, exact_score - best_score),
                "relative_gap_to_exact": max(0.0, exact_score - best_score)
                / max(abs(exact_score), 1.0e-8),
                "positive_reward": bool(positive_reward),
                "new_best": bool(new_best),
                "repeated_action": bool(repeated_action),
                "repeated_node": bool(repeated_node),
                "accepted_moves": accepted_moves,
                "unique_action_count": len(seen_actions),
                "covered_node_count": len(seen_nodes),
                "repeated_action_count": repeated_action_count,
                "repeated_node_count": repeated_node_count,
                "no_improve_steps": int(info["no_improve_steps"]),
                "steps_to_best": int(info["steps_to_best"]),
            }
        )

    runtime = time.perf_counter() - started
    best_assignment = env.best_assignment.copy()
    best_breakdown = score_assignment(
        best_assignment, instance=instance, objective=objective
    )
    recompute_error = abs(float(env.best_score) - float(best_breakdown.total_score))
    if recompute_error > 1.0e-9:
        raise RuntimeError(f"Best-score recomputation mismatch: {recompute_error}")
    gap = exact_score - float(best_breakdown.total_score)
    if gap < -1.0e-9:
        raise RuntimeError(f"PPO incumbent exceeds proven optimum by {-gap}.")
    violations = mask_violation_count(instance, best_assignment)

    return {
        "schema_version": 1,
        "max_steps": int(max_steps),
        "patience": int(patience),
        "executed_moves": int(env.step_count),
        "accepted_moves": accepted_moves,
        "accepted_move_definition": "reward > environment.improvement_eps",
        "positive_reward_moves": positive_reward_moves,
        "new_best_moves": new_best_moves,
        "unique_action_count": len(seen_actions),
        "repeated_action_count": repeated_action_count,
        "repeated_node_count": repeated_node_count,
        "covered_node_count": len(seen_nodes),
        "covered_node_rate": len(seen_nodes) / instance.n,
        "steps_to_best": int(env.steps_to_best),
        "done_reason": env.done_reason,
        "runtime_sec": float(runtime),
        "initial_assignment": initial_assignment.tolist(),
        "best_assignment": best_assignment.tolist(),
        "initial_score": initial_score,
        "best_score": float(best_breakdown.total_score),
        "best_score_recompute_error": recompute_error,
        "exact_score": float(exact_score),
        "absolute_gap_to_exact": max(0.0, gap),
        "relative_gap_to_exact": max(0.0, gap)
        / max(abs(exact_score), 1.0e-8),
        "hard_feasible_rate": 1.0 if violations == 0 else 0.0,
        "mask_violation_count": violations,
        "score": best_breakdown.as_dict(),
        "trace": trace,
    }


def derive_condition_from_master(
    master: Mapping[str, Any],
    *,
    instance: AssignmentInstance,
    objective: ObjectiveSpec,
    max_steps: int,
    patience: int,
) -> dict[str, Any]:
    """Derive an exact stopping-condition result from one longer trajectory."""

    if max_steps <= 0 or patience <= 0:
        raise ValueError("max_steps and patience must be positive.")
    trace = list(master["trace"])
    if not trace or int(trace[0]["step"]) != 0:
        raise ValueError("Master trace must begin at step zero.")
    if int(master["max_steps"]) < max_steps:
        raise ValueError("Master trace is shorter than the requested max_steps.")
    stop_row = None
    done_reason = None
    for row in trace[1:]:
        step = int(row["step"])
        if step >= max_steps:
            stop_row = row
            done_reason = "max_steps"
            break
        if int(row["no_improve_steps"]) >= patience:
            stop_row = row
            done_reason = "patience"
            break
    if stop_row is None:
        raise ValueError("Master trace cannot resolve the requested condition.")

    stop_step = int(stop_row["step"])
    steps_to_best = int(stop_row["steps_to_best"])
    assignment = list(master["initial_assignment"])
    for row in trace[1 : steps_to_best + 1]:
        assignment[int(row["node_index"])] = int(row["method_index"])
    breakdown = score_assignment(assignment, instance=instance, objective=objective)
    recompute_error = abs(
        float(breakdown.total_score) - float(stop_row["best_score"])
    )
    if recompute_error > 1.0e-9:
        raise RuntimeError(
            f"Derived best-score recomputation mismatch: {recompute_error}"
        )
    exact_score = float(master["exact_score"])
    gap = exact_score - float(breakdown.total_score)
    if gap < -1.0e-9:
        raise RuntimeError(
            f"Derived incumbent exceeds proven optimum by {-gap}."
        )
    violations = mask_violation_count(instance, assignment)
    return {
        "schema_version": 1,
        "derived_from_master": True,
        "max_steps": int(max_steps),
        "patience": int(patience),
        "executed_moves": stop_step,
        "accepted_moves": int(stop_row["accepted_moves"]),
        "accepted_move_definition": master["accepted_move_definition"],
        "positive_reward_moves": int(stop_row["accepted_moves"]),
        "new_best_moves": sum(
            bool(row.get("new_best", False)) for row in trace[1 : stop_step + 1]
        ),
        "unique_action_count": int(stop_row["unique_action_count"]),
        "repeated_action_count": int(stop_row["repeated_action_count"]),
        "repeated_node_count": int(stop_row["repeated_node_count"]),
        "covered_node_count": int(stop_row["covered_node_count"]),
        "covered_node_rate": int(stop_row["covered_node_count"]) / instance.n,
        "steps_to_best": steps_to_best,
        "done_reason": done_reason,
        "runtime_sec": float(stop_row["elapsed_sec"]),
        "initial_score": float(trace[0]["best_score"]),
        "best_assignment": assignment,
        "best_score": float(breakdown.total_score),
        "best_score_recompute_error": recompute_error,
        "exact_score": exact_score,
        "absolute_gap_to_exact": max(0.0, gap),
        "relative_gap_to_exact": max(0.0, gap)
        / max(abs(exact_score), 1.0e-8),
        "hard_feasible_rate": 1.0 if violations == 0 else 0.0,
        "mask_violation_count": violations,
        "score": breakdown.as_dict(),
        "master_trace_stop_step": stop_step,
    }


__all__ = [
    "SearchCondition",
    "build_search_conditions",
    "derive_condition_from_master",
    "deterministic_policy_trace",
    "scaled_budget_for_n",
]
