"""Tests for exact bounds and PPO-seeded short local refinement."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.solvers.gurobi_exact_solver import (
    brute_force_optimum,
    gurobi_available,
    solve_gurobi,
)
from pa_moap_rl.solvers.greedy_solver import effect_only_assignment
from pa_moap_rl.solvers.hybrid_solver import ppo_plus_short_local_search
from pa_moap_rl.utils.metrics import is_hard_feasible
from pa_moap_rl.utils.scoring import score_assignment


def _example_instance():
    return build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )


def _tiny_instance():
    base = _example_instance()
    size = 3
    selected_ids = list(base.selected_ids[:size])
    return replace(
        base,
        instance_name=f"{base.instance_name}_tiny",
        selected_ids=selected_ids,
        original_to_local_id={node_id: i for i, node_id in enumerate(selected_ids)},
        n=size,
        category_id=base.category_id[:size].copy(),
        cognitive_load=base.cognitive_load[:size].copy(),
        concept_need=base.concept_need[:size].copy(),
        effect_matrix=base.effect_matrix[:size].copy(),
        student_pref=base.student_pref[:size].copy(),
        teacher_pref=base.teacher_pref[:size].copy(),
        feasible_mask=base.feasible_mask[:size].copy(),
    )


def test_brute_force_optimum_matches_its_recomputed_score() -> None:
    instance = _tiny_instance()
    assignment, optimum = brute_force_optimum(instance)

    assert is_hard_feasible(instance, assignment)
    assert score_assignment(assignment, instance=instance).total_score == pytest.approx(optimum)


def test_short_hybrid_is_feasible_monotonic_and_budgeted() -> None:
    instance = _example_instance()
    start = effect_only_assignment(instance)
    initial_score = score_assignment(start, instance=instance).total_score

    result = ppo_plus_short_local_search(
        instance,
        start,
        accepted_move_budget=4,
        ppo_runtime=0.25,
    )

    assert is_hard_feasible(instance, result.assignment)
    assert result.score.total_score >= initial_score - 1.0e-12
    assert 0 <= result.metrics["accepted_moves"] <= 4
    assert result.metrics["ppo_runtime"] == pytest.approx(0.25)
    assert result.runtime >= 0.25


@pytest.mark.skipif(not gurobi_available(), reason="gurobipy is not installed")
def test_gurobi_matches_brute_force_on_tiny_instance() -> None:
    instance = _tiny_instance()
    _assignment, brute_score = brute_force_optimum(instance)

    result = solve_gurobi(instance, time_limit=30.0, output_flag=False)

    assert result.metrics["proven_optimal"]
    assert result.score.total_score == pytest.approx(brute_score, abs=1.0e-8)
    assert result.metrics["objective_bound"] == pytest.approx(brute_score, abs=1.0e-8)
    assert result.metrics["objective_recompute_error"] <= 1.0e-8
    assert np.asarray(result.assignment).shape == (instance.n,)
