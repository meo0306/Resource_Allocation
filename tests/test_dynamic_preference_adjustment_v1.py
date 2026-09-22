"""Tests for the versioned Stage11 dynamic-preference diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.experiments.dynamic_preference_adjustment_v1 import (
    build_dynamic_scenarios,
    interpolate_profile,
    pareto_method_ids,
    run_dynamic_policy,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.greedy_solver import scalarized_independent_greedy
from pa_moap_rl.solvers.ppo_solver import build_actor_critic


CONFIG_PATH = "pa_moap_rl/configs/dynamic_preference_adjustment_phase_a_v1.yaml"


def _config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _instance():
    return build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )


def test_interpolation_and_registered_scenarios_are_exact() -> None:
    midpoint = interpolate_profile([0.0] * 6, [1.0] * 6, 0.25)
    assert np.allclose(midpoint, 0.25)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpolate_profile([0.0] * 6, [1.0] * 6, 1.1)

    baseline, transitions = build_dynamic_scenarios(_config())
    assert baseline.schema_version == 2
    assert len(transitions) == 6
    assert len({item.scenario_id for item in transitions}) == 6
    student_small = next(item for item in transitions if item.scenario_id == "student_only__d025")
    expected = np.asarray([0.60, 0.60, 0.60, 0.60, 0.60, 0.70]) + 0.25 * (
        np.asarray([0.30, 0.55, 0.65, 0.90, 0.45, 0.80])
        - np.asarray([0.60, 0.60, 0.60, 0.60, 0.60, 0.70])
    )
    assert np.allclose(student_small.student_profile, expected)
    assert student_small.weights is None and student_small.target_distribution is None


def test_dynamic_policy_preserves_or_improves_explicit_old_solution() -> None:
    torch.manual_seed(20260922)
    instance = _instance()
    objective = load_objective_spec("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    project_config = load_config()
    model = build_actor_critic(
        instance,
        config=project_config,
        device="cpu",
        hidden_dim=16,
        objective=objective,
    )
    model.eval()
    initial = scalarized_independent_greedy(instance, objective=objective)
    result = run_dynamic_policy(
        instance=instance,
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=4,
        patience=4,
        initial_assignment=initial.assignment,
    )
    assert result.total_score >= initial.score.total_score - 1.0e-12
    assert result.hard_feasible
    assert result.mask_violation_count == 0
    assert result.step_count == 4


def test_pareto_method_ids_and_protocol_freeze() -> None:
    rows = [
        {"method_id": "fast", "mean_bounded_gap": 0.02, "median_latency_sec": 0.1},
        {"method_id": "balanced", "mean_bounded_gap": 0.01, "median_latency_sec": 0.2},
        {"method_id": "dominated", "mean_bounded_gap": 0.03, "median_latency_sec": 0.3},
    ]
    assert pareto_method_ids(rows) == ["balanced", "fast"]
    config = _config()
    assert config["status"] == "user_confirmed"
    assert config["run"]["no_training"] is True
    assert config["dynamic_preferences"]["fractions"] == [0.25, 1.0]
    assert config["run"]["persistent_gurobi_model_reuse_deferred_until_gate_signal"] is True
