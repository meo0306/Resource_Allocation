"""Versioned primitives for Stage11 dynamic-preference adjustment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Sequence

import numpy as np
import torch

from pa_moap_rl.configs import PAConfig
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.data.scenarios import PreferenceScenario
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.actor_critic import MaskedActorCritic
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.ppo_solver import pad_and_stack_observations
from pa_moap_rl.utils.metrics import make_solver_result


DYNAMIC_ENGINE_VERSION = "dynamic_preference_adjustment_v1"


def interpolate_profile(
    baseline: Sequence[float],
    target: Sequence[float],
    fraction: float,
) -> np.ndarray:
    """Linearly interpolate a six-dimensional profile without hidden clipping."""

    start = np.asarray(baseline, dtype=np.float64)
    end = np.asarray(target, dtype=np.float64)
    if start.shape != (6,) or end.shape != (6,):
        raise ValueError("Dynamic preference profiles must have length 6.")
    value = float(fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Dynamic preference fraction must be in [0, 1].")
    return start + value * (end - start)


def build_dynamic_scenarios(config: dict[str, Any]) -> tuple[PreferenceScenario, list[PreferenceScenario]]:
    """Build deterministic objective-independent baseline and transition scenarios."""

    settings = config["dynamic_preferences"]
    baseline_student = np.asarray(settings["baseline"]["student_profile"], dtype=np.float64)
    baseline_teacher = np.asarray(settings["baseline"]["teacher_profile"], dtype=np.float64)
    transition_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    baseline = PreferenceScenario(
        scenario_id=str(settings["baseline"]["scenario_id"]),
        student_template=str(settings["baseline"]["student_template"]),
        teacher_template=str(settings["baseline"]["teacher_template"]),
        student_profile=baseline_student,
        teacher_profile=baseline_teacher,
        target_distribution=None,
        weights=None,
        held_out=False,
        seed=int(settings["seed"]),
        schema_version=2,
        access_status="development_dynamic_preference_phase_a",
        perturbation_config_hash=transition_hash,
    )
    scenarios: list[PreferenceScenario] = []
    for transition in settings["transitions"]:
        target_student = np.asarray(transition["target_student_profile"], dtype=np.float64)
        target_teacher = np.asarray(transition["target_teacher_profile"], dtype=np.float64)
        for fraction in settings["fractions"]:
            fraction_value = float(fraction)
            fraction_label = f"d{int(round(100 * fraction_value)):03d}"
            scenarios.append(
                PreferenceScenario(
                    scenario_id=f"{transition['id']}__{fraction_label}",
                    student_template=(
                        f"{settings['baseline']['student_template']}_to_"
                        f"{transition['target_student_template']}@{fraction_value:.2f}"
                    ),
                    teacher_template=(
                        f"{settings['baseline']['teacher_template']}_to_"
                        f"{transition['target_teacher_template']}@{fraction_value:.2f}"
                    ),
                    student_profile=interpolate_profile(
                        baseline_student, target_student, fraction_value
                    ),
                    teacher_profile=interpolate_profile(
                        baseline_teacher, target_teacher, fraction_value
                    ),
                    target_distribution=None,
                    weights=None,
                    held_out=False,
                    seed=int(settings["seed"]),
                    schema_version=2,
                    access_status="development_dynamic_preference_phase_a",
                    perturbation_config_hash=transition_hash,
                )
            )
    expected = len(settings["transitions"]) * len(settings["fractions"])
    if len(scenarios) != expected or len({item.scenario_id for item in scenarios}) != expected:
        raise ValueError("Dynamic scenario construction is not one-to-one.")
    return baseline, scenarios


@dataclass(frozen=True)
class DynamicPolicyResult:
    assignment: np.ndarray
    total_score: float
    runtime_sec: float
    step_count: int
    steps_to_best: int
    done_reason: str
    hard_feasible: bool
    mask_violation_count: int


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_dynamic_policy(
    *,
    instance: AssignmentInstance,
    model: MaskedActorCritic,
    project_config: PAConfig,
    objective: ObjectiveSpec,
    max_steps: int,
    patience: int,
    initial_assignment: np.ndarray | Sequence[int] | None,
) -> DynamicPolicyResult:
    """Run one deterministic finite PPO episode from an explicit initial state."""

    device = next(model.parameters()).device
    _synchronize(device)
    started = time.perf_counter()
    env = MethodAssignmentEnv(
        max_steps=int(max_steps),
        patience=int(patience),
        config=project_config,
        objective=objective,
    )
    observation = env.reset(instance, initial_assignment=initial_assignment)
    while not env.done:
        _synchronize(device)
        with torch.inference_mode():
            batch = pad_and_stack_observations([observation], device=device)
            output = model(batch)
            flat_action = int(torch.argmax(output.masked_logits.reshape(1, -1), dim=1).item())
        node_index, method_index = divmod(flat_action, instance.m)
        observation, _reward, _done, _info = env.step((node_index, method_index))
    _synchronize(device)
    assert env.best_assignment is not None
    result = make_solver_result(
        solver_name="ppo_frozen_dynamic_v1",
        instance=instance,
        assignment=env.best_assignment,
        runtime=time.perf_counter() - started,
        objective=objective,
    )
    return DynamicPolicyResult(
        assignment=result.assignment,
        total_score=float(result.score.total_score),
        runtime_sec=float(result.runtime),
        step_count=int(env.step_count),
        steps_to_best=int(env.steps_to_best),
        done_reason=str(env.done_reason),
        hard_feasible=bool(result.metrics["hard_feasible_rate"] == 1.0),
        mask_violation_count=int(result.metrics["mask_violation_count"]),
    )


def pareto_method_ids(
    rows: Sequence[dict[str, Any]],
    *,
    quality_key: str = "mean_bounded_gap",
    latency_key: str = "median_latency_sec",
    tolerance: float = 1.0e-12,
) -> list[str]:
    """Return stable ids non-dominated when minimizing both gap and latency."""

    ordered = sorted(rows, key=lambda row: str(row["method_id"]))
    result: list[str] = []
    for candidate in ordered:
        dominated = False
        candidate_quality = float(candidate[quality_key])
        candidate_latency = float(candidate[latency_key])
        for other in ordered:
            if other is candidate:
                continue
            other_quality = float(other[quality_key])
            other_latency = float(other[latency_key])
            no_worse = (
                other_quality <= candidate_quality + tolerance
                and other_latency <= candidate_latency + tolerance
            )
            strictly_better = (
                other_quality < candidate_quality - tolerance
                or other_latency < candidate_latency - tolerance
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["method_id"]))
    return result


__all__ = [
    "DYNAMIC_ENGINE_VERSION",
    "DynamicPolicyResult",
    "build_dynamic_scenarios",
    "interpolate_profile",
    "pareto_method_ids",
    "run_dynamic_policy",
]
