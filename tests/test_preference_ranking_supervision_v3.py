"""Tests for separately versioned D/E preference-ranking supervision."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from pa_moap_rl.configs import load_config
from pa_moap_rl.checkpointing import checkpoint_metadata, method_matrix_hash
from pa_moap_rl.data.build_assignment_instance import (
    build_assignment_instance_from_selection,
)
from pa_moap_rl.data.scenarios import apply_scenario, load_scenario_config
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.paired_preference_sampling_v2 import (
    counterfactual_triplet,
)
from pa_moap_rl.experiments.formal_training import (
    load_training_checkpoint,
    save_checkpoint,
)
from pa_moap_rl.experiments.preference_ranking_supervision_v3 import (
    COUNTERFACTUAL_CONTRAST_VERSION,
    EXACT_DELTA_RANK_LOSS_VERSION,
    PreferenceAuxiliaryLoss,
    PreferenceSupervisionSpec,
    exact_delta_objective,
    masked_listwise_rank_loss,
)
from pa_moap_rl.experiments.run_preference_ranking_ablation_v3 import (
    load_and_validate_protocol,
    run_arm_from_config,
)
from pa_moap_rl.experiments.summarize_preference_ranking_ablation_v3 import (
    validate as validate_ranking_evaluation,
)
from pa_moap_rl.models.preference_repair_v2 import (
    SYMMETRIC_MODEL_VERSION,
    build_versioned_actor_critic,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.ppo_solver import (
    PPOConfig,
    collect_rollout,
    collect_vectorized_rollout,
    observation_to_model_batch,
    update_ppo,
    ppo_config_from_project,
)
from pa_moap_rl.utils.scoring import score_assignment


OBJECTIVE_PATH = "pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml"
PROTOCOL_PATH = "pa_moap_rl/configs/preference_ranking_ablation_protocol_v3.yaml"


def _example_instance():
    record = load_selection_csv(
        "examples/summary_ga_own_group1_260417_210154.csv"
    )[0]
    return build_assignment_instance_from_selection("examples", record)


def _paired_rollout():
    project = load_config()
    objective = load_objective_spec(OBJECTIVE_PATH)
    base = _example_instance()
    scenarios = counterfactual_triplet(
        np.random.default_rng(310),
        scenario_config=load_scenario_config(),
        project_config=project,
        objective_independent=True,
        prefix="ranking-test",
    )
    instances = [
        apply_scenario(base, scenario, project, objective=objective)
        for scenario in scenarios
    ]
    instances.append(instances[0])
    record = {"output_json": "same-content.json"}
    path = Path("same-content.json")
    prepared = [
        (record, path, scenario, instance)
        for scenario, instance in zip(
            [*scenarios, scenarios[0]],
            instances,
            strict=True,
        )
    ]
    model = build_versioned_actor_critic(
        instances[0],
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    config = PPOConfig(
        rollout_steps=4,
        minibatch_size=16,
        update_epochs=1,
        device="cpu",
        seed=13,
    )
    torch.manual_seed(13)
    rollout = collect_vectorized_rollout(
        envs=[
            MethodAssignmentEnv(
                objective=objective,
                max_steps=4,
                patience=4,
            )
            for _ in instances
        ],
        instances=instances,
        model=model,
        config=config,
    )
    return model, rollout, prepared, objective


def test_exact_delta_labels_match_full_score_difference() -> None:
    objective = load_objective_spec(OBJECTIVE_PATH)
    base = _example_instance()
    scenario = counterfactual_triplet(
        np.random.default_rng(41),
        scenario_config=load_scenario_config(),
        project_config=load_config(),
        objective_independent=True,
        prefix="delta-test",
    )[2]
    instance = apply_scenario(base, scenario, objective=objective)
    observation = MethodAssignmentEnv(
        objective=objective,
        max_steps=2,
        patience=2,
    ).reset(instance)
    batch = observation_to_model_batch(observation)
    labels = exact_delta_objective(batch, objective=objective)[0].numpy()
    current = score_assignment(
        observation["assignment"],
        instance=instance,
        objective=objective,
    ).total_score
    expected = np.zeros_like(labels, dtype=np.float64)
    for node, method in zip(*np.nonzero(observation["action_mask"]), strict=True):
        candidate = observation["assignment"].copy()
        candidate[node] = method
        expected[node, method] = (
            score_assignment(
                candidate,
                instance=instance,
                objective=objective,
            ).total_score
            - current
        )
    np.testing.assert_allclose(
        labels[observation["action_mask"]],
        expected[observation["action_mask"]],
        atol=5.0e-7,
        rtol=0.0,
    )


def test_listwise_rank_loss_prefers_exactly_aligned_logits() -> None:
    delta = torch.tensor([[[0.03, 0.01, -0.02], [0.00, 0.02, -0.01]]])
    mask = torch.tensor([[[True, True, False], [True, True, True]]])
    temperature = 0.01
    aligned = delta / temperature
    reversed_logits = -aligned
    aligned_loss = masked_listwise_rank_loss(
        aligned,
        delta,
        mask,
        target_temperature=temperature,
    )
    reversed_loss = masked_listwise_rank_loss(
        reversed_logits,
        delta,
        mask,
        target_temperature=temperature,
    )
    assert float(aligned_loss) == pytest.approx(0.0, abs=1.0e-7)
    assert float(reversed_loss) > float(aligned_loss)


def test_E_counterfactual_loss_is_finite_and_supervises_both_directions() -> None:
    model, rollout, prepared, objective = _paired_rollout()
    spec = PreferenceSupervisionSpec(
        ranking_version=EXACT_DELTA_RANK_LOSS_VERSION,
        ranking_weight=0.03,
        target_temperature=0.0025,
        contrast_version=COUNTERFACTUAL_CONTRAST_VERSION,
        contrast_weight=0.03,
    )
    callback = PreferenceAuxiliaryLoss(
        prepared=prepared,
        rollout=rollout,
        objective=objective,
        spec=spec,
    )
    indices = torch.arange(rollout.actions.shape[0])
    evaluated = model.evaluate_actions(
        rollout.observations,
        flat_action=rollout.actions,
    )
    result = callback(
        model=model,
        rollout=rollout,
        indices=indices,
        evaluated=evaluated,
    )
    assert torch.isfinite(result["loss"])
    assert result["metrics"]["auxiliary_contrast_direction_samples"] == 8.0
    assert result["metrics"]["auxiliary_contrast_loss"] >= 0.0
    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ranking_protocol_is_locked_and_verifies_old_artifacts() -> None:
    _, current, _, _, _ = load_and_validate_protocol(
        PROTOCOL_PATH,
        arm_id='D',
    )
    if current['status'] == 'user_confirmed':
        confirmed_specs = {}
        for confirmed_arm in ('D', 'E'):
            result = run_arm_from_config(
                PROTOCOL_PATH,
                arm_id=confirmed_arm,
                validate_only=True,
            )
            assert result['status'] == 'validated_no_training'
            confirmed_specs[confirmed_arm] = result['experiment_spec']
        assert confirmed_specs['D']['model'] == confirmed_specs['E']['model']
        assert confirmed_specs['D']['sampling'] == confirmed_specs['E']['sampling']
        assert (
            confirmed_specs['D']['auxiliary_supervision']
            != confirmed_specs['E']['auxiliary_supervision']
        )
        return
    specs = {}
    for arm in ("D", "E"):
        _, config, _, _, experiment_spec = load_and_validate_protocol(
            PROTOCOL_PATH,
            arm_id=arm,
        )
        assert config["status"] == "awaiting_user_confirmation_no_training"
        assert experiment_spec["experiment_spec_schema_version"] == 2
        assert experiment_spec["arm_id"] == arm
        specs[arm] = experiment_spec
    assert specs["D"]["model"] == specs["E"]["model"]
    assert specs["D"]["sampling"] == specs["E"]["sampling"]
    assert specs["D"]["auxiliary_supervision"] != specs["E"]["auxiliary_supervision"]
    with pytest.raises(PermissionError, match="locked"):
        run_arm_from_config(PROTOCOL_PATH, arm_id="D")


def test_default_off_auxiliary_interface_preserves_legacy_update() -> None:
    instance = _example_instance()
    config = PPOConfig(
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        device="cpu",
        seed=19,
    )
    source = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=load_objective_spec(OBJECTIVE_PATH),
        device="cpu",
        hidden_dim=16,
    )
    left = deepcopy(source)
    right = deepcopy(source)
    torch.manual_seed(19)
    rollout, _ = collect_rollout(
        env=MethodAssignmentEnv(max_steps=4, patience=4),
        instance=instance,
        model=source,
        config=config,
    )
    left_optimizer = torch.optim.Adam(left.parameters(), lr=1.0e-3)
    right_optimizer = torch.optim.Adam(right.parameters(), lr=1.0e-3)
    torch.manual_seed(991)
    left_metrics = update_ppo(
        model=left,
        optimizer=left_optimizer,
        rollout=rollout,
        config=config,
    )
    torch.manual_seed(991)
    right_metrics = update_ppo(
        model=right,
        optimizer=right_optimizer,
        rollout=rollout,
        config=config,
        auxiliary_loss_fn=None,
    )
    assert left_metrics == right_metrics
    for left_parameter, right_parameter in zip(
        left.parameters(),
        right.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left_parameter, right_parameter)


def test_D_checkpoint_cannot_restore_as_E(tmp_path: Path) -> None:
    _, _, objective, _, d_spec = load_and_validate_protocol(
        PROTOCOL_PATH,
        arm_id="D",
    )
    _, _, _, _, e_spec = load_and_validate_protocol(
        PROTOCOL_PATH,
        arm_id="E",
    )
    instance = _example_instance()
    d_model = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    d_model.experiment_spec = d_spec
    optimizer = torch.optim.Adam(d_model.parameters(), lr=1.0e-4)
    ppo_config = ppo_config_from_project(
        load_config(),
        learning_rate=1.0e-4,
    )
    provenance = checkpoint_metadata(
        objective,
        method_hash=method_matrix_hash(instance),
        data_hash="ranking-test-data",
    )
    checkpoint = tmp_path / "D.pt"
    save_checkpoint(
        checkpoint,
        model=d_model,
        optimizer=optimizer,
        update=1,
        hidden_dim=16,
        ppo_config=ppo_config,
        train_rows=[],
        eval_rows=[],
        python_rng=random.Random(1),
        numpy_rng=np.random.default_rng(1),
        best_validation_score=0.0,
        provenance=provenance,
    )
    e_model = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    e_model.experiment_spec = e_spec
    with pytest.raises(ValueError, match="experiment_spec mismatch"):
        load_training_checkpoint(
            checkpoint,
            model=e_model,
            optimizer=torch.optim.Adam(e_model.parameters(), lr=1.0e-4),
            python_rng=random.Random(1),
            numpy_rng=np.random.default_rng(1),
            device=torch.device("cpu"),
            objective=objective,
            method_hash=method_matrix_hash(instance),
            data_hash="ranking-test-data",
        )


def test_D_E_controlled_cross_protocol_validates_without_inference() -> None:
    result = validate_ranking_evaluation(
        'pa_moap_rl/configs/preference_ranking_ablation_evaluation_v3.yaml'
    )
    assert result['status'] == 'validated_no_training_no_inference'
    assert result['arm_count'] == 2
    assert result['pair_count_per_arm'] == 288
    assert result['training_executed'] is False
