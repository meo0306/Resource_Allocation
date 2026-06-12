"""Tests for baseline solvers."""

import numpy as np
import pytest

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.solvers.greedy_solver import (
    effect_only_assignment,
    effect_only_greedy,
    one_point_greedy_improvement,
    scalarized_independent_greedy,
)
from pa_moap_rl.solvers.local_search_solver import (
    has_positive_one_point_improvement,
    local_search_best_improvement,
    local_search_first_improvement,
    random_restart_local_search,
)
from pa_moap_rl.solvers.random_solver import random_legal
from pa_moap_rl.utils.metrics import evaluate_assignment, is_hard_feasible, mask_violation_count


def _example_instance():
    return build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )


def test_random_legal_is_seeded_and_feasible() -> None:
    instance = _example_instance()

    result_a = random_legal(instance, seed=7)
    result_b = random_legal(instance, seed=7)

    np.testing.assert_array_equal(result_a.assignment, result_b.assignment)
    assert is_hard_feasible(instance, result_a.assignment)
    assert result_a.metrics["mask_violation_count"] == 0
    assert result_a.metrics["hard_feasible_rate"] == 1.0


def test_effect_only_greedy_selects_max_effect_feasible_method() -> None:
    instance = _example_instance()

    assignment = effect_only_assignment(instance)
    result = effect_only_greedy(instance)

    expected = np.argmax(np.where(instance.feasible_mask, instance.effect_matrix, -np.inf), axis=1)
    np.testing.assert_array_equal(assignment, expected)
    np.testing.assert_array_equal(result.assignment, expected)
    assert is_hard_feasible(instance, result.assignment)


def test_scalarized_and_one_point_greedy_are_feasible_and_monotonic() -> None:
    instance = _example_instance()
    scalarized = scalarized_independent_greedy(instance)
    improved = one_point_greedy_improvement(instance, scalarized.assignment)

    assert is_hard_feasible(instance, scalarized.assignment)
    assert is_hard_feasible(instance, improved.assignment)
    assert improved.score.total_score >= scalarized.score.total_score - 1.0e-12
    assert all(
        later >= earlier - 1.0e-12
        for earlier, later in zip(improved.history, improved.history[1:])
    )


def test_local_search_best_improvement_reaches_one_point_local_optimum() -> None:
    instance = _example_instance()
    result = local_search_best_improvement(instance)

    assert is_hard_feasible(instance, result.assignment)
    assert not has_positive_one_point_improvement(instance, result.assignment)
    assert all(later >= earlier - 1.0e-12 for earlier, later in zip(result.history, result.history[1:]))


def test_first_improvement_and_random_restart_return_valid_results() -> None:
    instance = _example_instance()

    first = local_search_first_improvement(instance, shuffle=True, seed=11, max_iter=20)
    restart = random_restart_local_search(instance, restarts=3, max_iter=10, seed=13)

    assert is_hard_feasible(instance, first.assignment)
    assert is_hard_feasible(instance, restart.assignment)
    assert first.metrics["mask_violation_count"] == 0
    assert restart.metrics["mask_violation_count"] == 0
    assert first.history
    assert restart.history


def test_metrics_are_complete_and_count_mask_violations() -> None:
    instance = _example_instance()
    result = effect_only_greedy(instance)
    metrics = evaluate_assignment(instance, result.assignment, solver_name="check", runtime=0.1)

    required = {
        "hard_feasible_rate",
        "mask_violation_count",
        "valid_action_ratio",
        "total_score",
        "effect_score",
        "student_pref_score",
        "teacher_pref_score",
        "global_score",
        "soft_penalty",
        "runtime",
    }
    assert required <= set(metrics)
    assert metrics["runtime"] == pytest.approx(0.1)
    assert 0.0 <= metrics["valid_action_ratio"] <= 1.0

    bad = result.assignment.copy()
    bad[0] = instance.m
    assert mask_violation_count(instance, bad) == 1
