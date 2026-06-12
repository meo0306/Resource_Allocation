"""Tests for PA-MOAP scoring functions."""

import numpy as np
import pytest

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.utils.scoring import F_E, F_G, F_S, F_T, H, J, V_cap, V_div, delta_score, method_distribution, score_assignment


def test_method_distribution_and_objective_components() -> None:
    assignment = np.asarray([0, 1, 1, 2])
    effect = np.asarray(
        [
            [0.9, 0.1, 0.2],
            [0.3, 0.8, 0.4],
            [0.2, 0.7, 0.5],
            [0.1, 0.2, 0.6],
        ]
    )
    student = effect + 0.01
    teacher = effect + 0.02
    target = np.asarray([0.25, 0.50, 0.25])

    np.testing.assert_allclose(method_distribution(assignment, 3), [0.25, 0.50, 0.25])
    assert F_E(assignment, effect) == pytest.approx((0.9 + 0.8 + 0.7 + 0.6) / 4.0)
    assert F_S(assignment, student) == pytest.approx(F_E(assignment, effect) + 0.01)
    assert F_T(assignment, teacher) == pytest.approx(F_E(assignment, effect) + 0.02)
    assert F_G(assignment, target) == pytest.approx(1.0)
    assert H(assignment, M=3) > 0.0
    assert V_div(assignment, M=3, H_min=0.0) == pytest.approx(0.0)
    assert V_cap(assignment, M=3, pi_cap=0.35) == pytest.approx((0.50 - 0.35) ** 2)


def test_score_assignment_uses_means_and_weights() -> None:
    assignment = np.asarray([0, 1])
    effect = np.asarray([[1.0, 0.0], [0.0, 0.8]])
    student = np.asarray([[0.5, 0.2], [0.3, 0.7]])
    teacher = np.asarray([[0.4, 0.1], [0.2, 0.6]])
    target = np.asarray([0.5, 0.5])
    weights = {"effect": 0.2, "student": 0.2, "teacher": 0.2, "global": 0.2, "soft": 0.2}

    score = score_assignment(
        assignment,
        effect_matrix=effect,
        student_pref=student,
        teacher_pref=teacher,
        target_distribution=target,
        weights=weights,
        H_min=0.0,
        pi_cap=1.0,
    )

    assert score.effect_score == pytest.approx(0.9)
    assert score.student_pref_score == pytest.approx(0.6)
    assert score.teacher_pref_score == pytest.approx(0.5)
    assert score.global_score == pytest.approx(1.0)
    assert score.soft_penalty == pytest.approx(0.0)
    assert score.total_score == pytest.approx(0.2 * (0.9 + 0.6 + 0.5 + 1.0))
    assert J(
        assignment,
        effect_matrix=effect,
        student_pref=student,
        teacher_pref=teacher,
        target_distribution=target,
        weights=weights,
        H_min=0.0,
        pi_cap=1.0,
    ) == pytest.approx(score.total_score)


def test_delta_score_matches_full_recompute_on_assignment_instance() -> None:
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )
    assignment = np.argmax(instance.effect_matrix, axis=1)
    node_index = 0
    new_method = (int(assignment[node_index]) + 1) % instance.m

    delta = delta_score(assignment, node_index, new_method, instance=instance)
    updated = assignment.copy()
    updated[node_index] = new_method

    assert delta == pytest.approx(J(updated, instance=instance) - J(assignment, instance=instance))
    assert score_assignment(assignment, instance=instance).total_score == pytest.approx(J(assignment, instance=instance))
