"""Acceptance tests for the four-topology objective-calibration interfaces."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pa_moap_rl.checkpointing import (
    checkpoint_metadata,
    instance_data_hash,
    method_matrix_hash,
    validate_checkpoint_metadata,
)
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.scenarios import apply_scenario, generate_scenario_bank
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.objective import ObjectiveSpec, load_objective_spec
from pa_moap_rl.solvers.gurobi_exact_solver import (
    brute_force_optimum,
    gurobi_available,
    solve_gurobi,
)
from pa_moap_rl.solvers.greedy_solver import solve_scalarized_independent_greedy
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.utils.scoring import IncrementalAssignmentScorer, delta_score, score_assignment


RHO = (0.18, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.05, 0.05, 0.06)


def _instance():
    return build_assignment_instance_from_selection(
        'examples',
        load_selection_csv('examples/summary_ga_own_group1_260417_210154.csv')[0],
    )


def _tiny_instance():
    base = _instance()
    size = 3
    selected = list(base.selected_ids[:size])
    return replace(
        base,
        instance_name=f'{base.instance_name}_objective_v2_tiny',
        selected_ids=selected,
        original_to_local_id={node_id: index for index, node_id in enumerate(selected)},
        n=size,
        category_id=base.category_id[:size].copy(),
        cognitive_load=base.cognitive_load[:size].copy(),
        concept_need=base.concept_need[:size].copy(),
        effect_matrix=base.effect_matrix[:size].copy(),
        student_pref=base.student_pref[:size].copy(),
        teacher_pref=base.teacher_pref[:size].copy(),
        feasible_mask=base.feasible_mask[:size].copy(),
    )


def _objective(*, global_weight: float = 0.05) -> ObjectiveSpec:
    return ObjectiveSpec(
        objective_id=f'test_global_{global_weight}',
        schema_version=2,
        alpha_effect=0.35,
        alpha_preference=0.65 - global_weight,
        alpha_global=global_weight,
        entropy_min=0.65,
        method_cap=0.25,
        beta_entropy=0.30,
        beta_cap=0.10,
        target_distribution=RHO if global_weight else None,
        epsilon=1.0e-8,
    )


def test_objective_validation_and_equal_preference_derivation() -> None:
    spec = _objective()
    assert spec.alpha_student == pytest.approx(spec.alpha_preference / 2.0)
    assert spec.alpha_teacher == pytest.approx(spec.alpha_preference / 2.0)
    assert len(spec.config_hash) == 64
    with pytest.raises(ValueError):
        ObjectiveSpec.from_dict(
            {
                **spec.to_dict(include_hash=False),
                'alpha_student': spec.alpha_student + 0.01,
            }
        )
    with pytest.raises(ValueError):
        ObjectiveSpec.from_dict(
            {**spec.to_dict(include_hash=False), 'lambda_soft': 0.2}
        )
    with pytest.raises(ValueError):
        replace(spec, target_distribution=(0.5, 0.5))


def test_nondefault_full_delta_and_incremental_scores_match_every_action() -> None:
    instance = _instance()
    objective = _objective(global_weight=0.0)
    assignment = np.argmax(
        np.where(instance.feasible_mask, instance.effect_matrix, -np.inf), axis=1
    )
    scorer = IncrementalAssignmentScorer(assignment, instance, objective=objective)
    for node in range(instance.n):
        for method in np.flatnonzero(instance.feasible_mask[node]):
            updated = assignment.copy()
            updated[node] = int(method)
            full = score_assignment(updated, instance=instance, objective=objective)
            incremental = scorer.candidate_breakdown(node, int(method))
            assert incremental.total_score == pytest.approx(full.total_score, abs=1.0e-12)
            assert delta_score(
                assignment,
                node,
                int(method),
                instance=instance,
                objective=objective,
            ) == pytest.approx(full.total_score - scorer.score, abs=1.0e-12)


@pytest.mark.skipif(not gurobi_available(), reason='gurobipy is not installed')
@pytest.mark.parametrize('global_weight', [0.0, 0.05])
def test_gurobi_matches_bruteforce_with_v2_penalties(global_weight: float) -> None:
    instance = _tiny_instance()
    objective = _objective(global_weight=global_weight)
    _, brute_score = brute_force_optimum(instance, objective=objective)
    result = solve_gurobi(
        instance,
        time_limit=30.0,
        threads=1,
        seed=20260912,
        objective=objective,
    )
    assert result.score.total_score == pytest.approx(brute_score, abs=1.0e-9)
    assert result.metrics['objective_recompute_error'] <= 1.0e-9
    assert result.metrics['objective_hash'] == objective.config_hash


def test_env_greedy_and_local_search_share_objective_hash() -> None:
    instance = _instance()
    objective = _objective(global_weight=0.0)
    env = MethodAssignmentEnv(objective=objective, max_steps=2, patience=2)
    observation = env.reset(instance)
    np.testing.assert_array_equal(observation['target_distribution'], np.zeros(instance.m))
    greedy = solve_scalarized_independent_greedy(instance, objective=objective)
    local = local_search_best_improvement(
        instance, initial_assignment=greedy.assignment, max_iter=2, objective=objective
    )
    assert env.current_score.objective_hash == objective.config_hash
    assert greedy.metrics['objective_hash'] == objective.config_hash
    assert local.metrics['objective_hash'] == objective.config_hash


def test_scenario_v2_is_objective_independent_and_reproducible() -> None:
    first = generate_scenario_bank(
        3, 123, objective_independent=True, access_status='test'
    )
    second = generate_scenario_bank(
        3, 123, objective_independent=True, access_status='test'
    )
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert 'weights' not in first[0].to_dict()
    assert 'target_distribution' not in first[0].to_dict()
    with pytest.raises(ValueError):
        apply_scenario(_instance(), first[0])
    conditioned = apply_scenario(_instance(), first[0], objective=_objective())
    assert conditioned.metadata['scenario']['scenario_id'] == first[0].scenario_id


def test_checkpoint_provenance_rejects_all_three_mismatch_classes() -> None:
    instance = _instance()
    objective = _objective()
    method_hash = method_matrix_hash(instance)
    data_hash = instance_data_hash(instance)
    checkpoint = {
        'provenance': checkpoint_metadata(
            objective, method_hash=method_hash, data_hash=data_hash
        )
    }
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_hash,
        data_hash=data_hash,
    )
    for overrides in (
        {'objective': replace(objective, objective_id='different')},
        {'method_hash': '0' * 64},
        {'data_hash': 'f' * 64},
    ):
        with pytest.raises(ValueError):
            validate_checkpoint_metadata(
                checkpoint,
                objective=overrides.get('objective', objective),
                method_hash=overrides.get('method_hash', method_hash),
                data_hash=overrides.get('data_hash', data_hash),
            )
    with pytest.raises(ValueError):
        validate_checkpoint_metadata(
            {},
            objective=objective,
            method_hash=method_hash,
            data_hash=data_hash,
        )


def test_generated_four_topology_protocol_audit() -> None:
    root = Path('data_processed/four_topology_exploratory_v2')
    audit = json.loads((root / 'data_audit.json').read_text(encoding='utf-8'))
    assert audit['roles']['dev_train']['instances'] == 2520
    assert audit['roles']['dev_calibration']['instances'] == 540
    assert audit['roles']['dev_seen_diagnostic']['instances'] == 540
    assert audit['calibration_core'] == {'instances': 72, 'source_families': 18}
    assert set(audit['family_intersections'].values()) == {0}
    assert audit['excluded_topologies'] == ['staircase']
    assert 'previously viewed' in audit['dev_seen_diagnostic_notice']


def test_frozen_four_topology_objective_matches_confirmed_hash() -> None:
    config = Path('pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml')
    spec = load_objective_spec(config)
    assert spec.objective_id == 'main_no_reference__h0.65__c0.35__be1.00__bc1.00'
    assert spec.config_hash == (
        'b075006e79dc73a06ce76049ac0e3350c2d6b9b3ceb79db6fec57294e5b51606'
    )
    assert spec.alpha_effect == pytest.approx(0.35)
    assert spec.alpha_student == pytest.approx(0.325)
    assert spec.alpha_teacher == pytest.approx(0.325)
    assert spec.alpha_global == 0.0
    assert spec.target_distribution is None
    assert spec.entropy_min == pytest.approx(0.65)
    assert spec.method_cap == pytest.approx(0.35)
    assert spec.beta_entropy == pytest.approx(1.0)
    assert spec.beta_cap == pytest.approx(1.0)
