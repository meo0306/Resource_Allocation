"""Versioned service-efficiency primitives for frozen four-topology PPO.

This module is deliberately independent from the training rollout collector.
It runs one finite episode per request, compacts completed requests out of the
active model batch, and never resets an environment after completion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from pa_moap_rl.configs import PAConfig
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.actor_critic import MaskedActorCritic
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.greedy_solver import (
    effect_only_greedy,
    scalarized_independent_greedy,
)
from pa_moap_rl.solvers.gurobi_exact_solver import solve_gurobi
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.ppo_solver import pad_and_stack_observations
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result


SERVICE_ENGINE_VERSION = "batched_policy_service_v1"
BASELINE_SERVICE_VERSION = "threaded_request_service_v1"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    """Return stable latency statistics in seconds."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": _quantile(array.tolist(), 0.95),
        "max": float(array.max()),
    }


@dataclass(frozen=True)
class PolicyRequestResult:
    request_id: str
    assignment: np.ndarray
    score: float
    completion_latency_sec: float
    step_count: int
    steps_to_best: int
    done_reason: str
    hard_feasible: bool
    mask_violation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "assignment": self.assignment.tolist(),
            "total_score": float(self.score),
            "completion_latency_sec": float(self.completion_latency_sec),
            "step_count": int(self.step_count),
            "steps_to_best": int(self.steps_to_best),
            "done_reason": self.done_reason,
            "hard_feasible": bool(self.hard_feasible),
            "mask_violation_count": int(self.mask_violation_count),
        }


@dataclass(frozen=True)
class PolicyBatchResult:
    requested_batch_size: int
    actual_batch_size: int
    batch_latency_sec: float
    reset_latency_sec: float
    model_forward_sec: float
    environment_step_sec: float
    postprocess_sec: float
    synchronized_iterations: int
    mean_active_batch: float
    mean_padding_ratio: float
    peak_padding_ratio: float
    requests: tuple[PolicyRequestResult, ...]

    @property
    def throughput_requests_per_sec(self) -> float:
        return float(self.actual_batch_size / self.batch_latency_sec)

    def to_dict(self) -> dict[str, Any]:
        request_latencies = [item.completion_latency_sec for item in self.requests]
        return {
            "service_engine_version": SERVICE_ENGINE_VERSION,
            "requested_batch_size": int(self.requested_batch_size),
            "actual_batch_size": int(self.actual_batch_size),
            "batch_latency_sec": float(self.batch_latency_sec),
            "throughput_requests_per_sec": self.throughput_requests_per_sec,
            "reset_latency_sec": float(self.reset_latency_sec),
            "model_forward_sec": float(self.model_forward_sec),
            "environment_step_sec": float(self.environment_step_sec),
            "postprocess_sec": float(self.postprocess_sec),
            "synchronized_iterations": int(self.synchronized_iterations),
            "mean_active_batch": float(self.mean_active_batch),
            "mean_padding_ratio": float(self.mean_padding_ratio),
            "peak_padding_ratio": float(self.peak_padding_ratio),
            "request_latency": latency_summary(request_latencies),
            "requests": [item.to_dict() for item in self.requests],
        }


def run_batched_policy_service(
    *,
    request_ids: Sequence[str],
    instances: Sequence[AssignmentInstance],
    model: MaskedActorCritic,
    project_config: PAConfig,
    objective: ObjectiveSpec,
    max_steps: int,
    patience: int,
    requested_batch_size: int | None = None,
) -> PolicyBatchResult:
    """Run deterministic finite policy episodes in one compacting model batch."""

    if not instances or len(request_ids) != len(instances):
        raise ValueError("request_ids and instances must be non-empty and have equal length.")
    if len({instance.m for instance in instances}) != 1:
        raise ValueError("All requests must share the method count.")
    if len({instance.k for instance in instances}) != 1:
        raise ValueError("All requests must share the category count.")
    requested = int(requested_batch_size or len(instances))
    if requested < len(instances):
        raise ValueError("requested_batch_size cannot be smaller than the actual batch.")

    device = next(model.parameters()).device
    _synchronize(device)
    total_start = time.perf_counter()
    reset_start = total_start
    envs = [
        MethodAssignmentEnv(
            max_steps=int(max_steps),
            patience=int(patience),
            config=project_config,
            objective=objective,
        )
        for _ in instances
    ]
    observations = [env.reset(instance) for env, instance in zip(envs, instances)]
    reset_latency = time.perf_counter() - reset_start

    active = list(range(len(instances)))
    completion = [math.nan for _ in instances]
    active_sizes: list[int] = []
    padding_ratios: list[float] = []
    model_forward_sec = 0.0
    environment_step_sec = 0.0
    iterations = 0

    while active:
        iterations += 1
        active_sizes.append(len(active))
        active_observations = [observations[index] for index in active]
        node_counts = [instances[index].n for index in active]
        max_nodes = max(node_counts)
        padding_ratios.append(
            1.0 - float(sum(node_counts)) / float(len(node_counts) * max_nodes)
        )

        _synchronize(device)
        model_start = time.perf_counter()
        with torch.inference_mode():
            batch = pad_and_stack_observations(active_observations, device=device)
            output = model(batch)
            flat_actions = torch.argmax(
                output.masked_logits.reshape(len(active), -1), dim=1
            )
        _synchronize(device)
        model_forward_sec += time.perf_counter() - model_start
        actions = flat_actions.detach().cpu().numpy().astype(np.int64, copy=False)

        step_start = time.perf_counter()
        next_active: list[int] = []
        for position, request_index in enumerate(active):
            instance = instances[request_index]
            flat_action = int(actions[position])
            node_index, method_index = divmod(flat_action, instance.m)
            observation, _reward, done, _info = envs[request_index].step(
                (node_index, method_index)
            )
            observations[request_index] = observation
            if done:
                completion[request_index] = time.perf_counter() - total_start
            else:
                next_active.append(request_index)
        environment_step_sec += time.perf_counter() - step_start
        active = next_active

    _synchronize(device)
    rollout_end = time.perf_counter()
    postprocess_start = rollout_end
    request_results: list[PolicyRequestResult] = []
    for index, (request_id, instance, env) in enumerate(zip(request_ids, instances, envs)):
        assert env.best_assignment is not None
        result = make_solver_result(
            solver_name="ppo_frozen_batched",
            instance=instance,
            assignment=env.best_assignment,
            runtime=float(completion[index]),
            objective=objective,
        )
        request_results.append(
            PolicyRequestResult(
                request_id=str(request_id),
                assignment=result.assignment,
                score=float(result.score.total_score),
                completion_latency_sec=float(completion[index]),
                step_count=int(env.step_count),
                steps_to_best=int(env.steps_to_best),
                done_reason=str(env.done_reason),
                hard_feasible=bool(result.metrics["hard_feasible_rate"] == 1.0),
                mask_violation_count=int(result.metrics["mask_violation_count"]),
            )
        )
    postprocess_sec = time.perf_counter() - postprocess_start
    batch_latency = time.perf_counter() - total_start
    return PolicyBatchResult(
        requested_batch_size=requested,
        actual_batch_size=len(instances),
        batch_latency_sec=batch_latency,
        reset_latency_sec=reset_latency,
        model_forward_sec=model_forward_sec,
        environment_step_sec=environment_step_sec,
        postprocess_sec=postprocess_sec,
        synchronized_iterations=iterations,
        mean_active_batch=float(np.mean(active_sizes)),
        mean_padding_ratio=float(np.mean(padding_ratios)),
        peak_padding_ratio=float(max(padding_ratios)),
        requests=tuple(request_results),
    )


def assert_batch_equivalent(
    sequential: Sequence[PolicyRequestResult],
    batched: Sequence[PolicyRequestResult],
    *,
    score_tolerance: float = 1.0e-9,
) -> None:
    """Fail if batch execution changes deterministic request results."""

    if len(sequential) != len(batched):
        raise AssertionError("Sequential and batched result counts differ.")
    sequential_by_id = {item.request_id: item for item in sequential}
    batched_by_id = {item.request_id: item for item in batched}
    if set(sequential_by_id) != set(batched_by_id):
        raise AssertionError("Sequential and batched request ids differ.")
    for request_id, expected in sequential_by_id.items():
        actual = batched_by_id[request_id]
        if not np.array_equal(expected.assignment, actual.assignment):
            raise AssertionError(f"Batch assignment mismatch for {request_id}.")
        if abs(expected.score - actual.score) > float(score_tolerance):
            raise AssertionError(f"Batch score mismatch for {request_id}.")
        if expected.step_count != actual.step_count or expected.done_reason != actual.done_reason:
            raise AssertionError(f"Batch stopping mismatch for {request_id}.")


def build_baseline_solver(
    solver_id: str,
    *,
    objective: ObjectiveSpec,
    max_iter: int | None = None,
    time_limit: float | None = None,
    gurobi_threads: int = 1,
    seed: int = 0,
) -> Callable[[AssignmentInstance], SolverResult]:
    """Build one explicit-objective baseline callable."""

    if solver_id == "effect_greedy":
        return lambda instance: effect_only_greedy(instance, objective=objective)
    if solver_id == "scalarized_greedy":
        return lambda instance: scalarized_independent_greedy(instance, objective=objective)
    if solver_id == "local_search_best":
        return lambda instance: local_search_best_improvement(
            instance,
            max_iter=max_iter,
            objective=objective,
        )
    if solver_id == "gurobi":
        if time_limit is None:
            raise ValueError("Gurobi baseline requires time_limit.")
        return lambda instance: solve_gurobi(
            instance,
            time_limit=float(time_limit),
            mip_gap=0.0,
            seed=int(seed),
            threads=int(gurobi_threads),
            objective=objective,
        )
    raise ValueError(f"Unknown solver_id: {solver_id}")


def run_threaded_baseline_service(
    *,
    request_ids: Sequence[str],
    instances: Sequence[AssignmentInstance],
    solver: Callable[[AssignmentInstance], SolverResult],
    concurrency: int,
) -> dict[str, Any]:
    """Serve independent baseline requests with explicit host concurrency."""

    if not instances or len(request_ids) != len(instances):
        raise ValueError("request_ids and instances must be non-empty and have equal length.")
    if int(concurrency) <= 0:
        raise ValueError("concurrency must be positive.")

    def run_one(index: int) -> tuple[int, SolverResult, float]:
        started = time.perf_counter()
        result = solver(instances[index])
        return index, result, time.perf_counter() - started

    wall_start = time.perf_counter()
    completed: list[tuple[int, SolverResult, float]] = []
    with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
        futures = [executor.submit(run_one, index) for index in range(len(instances))]
        for future in as_completed(futures):
            completed.append(future.result())
    wall_latency = time.perf_counter() - wall_start
    completed.sort(key=lambda item: item[0])
    rows = []
    for index, result, latency in completed:
        rows.append(
            {
                "request_id": str(request_ids[index]),
                "total_score": float(result.score.total_score),
                "latency_sec": float(latency),
                "hard_feasible": bool(result.metrics["hard_feasible_rate"] == 1.0),
                "mask_violation_count": int(result.metrics["mask_violation_count"]),
                "assignment": result.assignment.tolist(),
                "solver_metrics": dict(result.metrics),
            }
        )
    return {
        "baseline_service_version": BASELINE_SERVICE_VERSION,
        "request_count": len(rows),
        "concurrency": int(concurrency),
        "wall_latency_sec": float(wall_latency),
        "throughput_requests_per_sec": float(len(rows) / wall_latency),
        "request_latency": latency_summary([row["latency_sec"] for row in rows]),
        "requests": rows,
    }


__all__ = [
    "BASELINE_SERVICE_VERSION",
    "PolicyBatchResult",
    "PolicyRequestResult",
    "SERVICE_ENGINE_VERSION",
    "assert_batch_equivalent",
    "build_baseline_solver",
    "latency_summary",
    "run_batched_policy_service",
    "run_threaded_baseline_service",
]
