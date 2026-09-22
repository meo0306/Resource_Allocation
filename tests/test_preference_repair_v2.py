"""Tests for versioned pure-PPO preference repair components."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pa_moap_rl.checkpointing import checkpoint_metadata, method_matrix_hash
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.scenarios import apply_scenario, generate_scenario_bank, load_scenario_config
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.formal_training import load_training_checkpoint, save_checkpoint
from pa_moap_rl.experiments.paired_preference_sampling_v2 import counterfactual_triplet
from pa_moap_rl.experiments.run_preference_repair_ablation_v2 import (
    load_and_validate_protocol,
    run_arm_from_config,
)
from pa_moap_rl.models.preference_repair_v2 import (
    SYMMETRIC_MODEL_VERSION,
    build_versioned_actor_critic,
    model_spec,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.ppo_solver import observation_to_model_batch, ppo_config_from_project
from pa_moap_rl.utils.scoring import score_assignment


OBJECTIVE_PATH = "pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml"
PROTOCOL_PATH = "pa_moap_rl/configs/preference_repair_ablation_protocol_v2.yaml"


def example_instance():
    record = load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0]
    return build_assignment_instance_from_selection("examples", record)


def conditioned_instance():
    objective = load_objective_spec(OBJECTIVE_PATH)
    scenario = generate_scenario_bank(
        1,
        seed=910,
        objective_independent=True,
    )[0]
    return apply_scenario(example_instance(), scenario, objective=objective), objective


def test_counterfactual_triplet_is_reproducible_and_shares_unchanged_actor() -> None:
    config = load_config()
    scenario_config = load_scenario_config()
    left = counterfactual_triplet(
        np.random.default_rng(41),
        scenario_config=scenario_config,
        project_config=config,
        objective_independent=True,
        prefix="test",
    )
    right = counterfactual_triplet(
        np.random.default_rng(41),
        scenario_config=scenario_config,
        project_config=config,
        objective_independent=True,
        prefix="test",
    )
    assert [(item.student_template, item.teacher_template) for item in left] == [
        ("S6", "T6"),
        ("S4", "T6"),
        ("S6", "T4"),
    ]
    for first, second in zip(left, right, strict=True):
        assert first.schema_version == 2
        assert first.weights is None
        assert first.target_distribution is None
        np.testing.assert_array_equal(first.student_profile, second.student_profile)
        np.testing.assert_array_equal(first.teacher_profile, second.teacher_profile)
    np.testing.assert_array_equal(left[0].teacher_profile, left[1].teacher_profile)
    np.testing.assert_array_equal(left[0].student_profile, left[2].student_profile)


def test_symmetric_model_is_exchange_invariant() -> None:
    instance, objective = conditioned_instance()
    env = MethodAssignmentEnv(objective=objective, max_steps=2, patience=2)
    observation = env.reset(instance)
    model = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    model.eval()
    batch = observation_to_model_batch(observation)
    swapped = {key: value.clone() for key, value in batch.items()}
    swapped["student_pref"] = batch["teacher_pref"].clone()
    swapped["teacher_pref"] = batch["student_pref"].clone()
    swapped["current_scores"][:, 1] = batch["current_scores"][:, 2]
    swapped["current_scores"][:, 2] = batch["current_scores"][:, 1]
    with torch.no_grad():
        original_output = model(batch)
        swapped_output = model(swapped)
    torch.testing.assert_close(
        original_output.action_logits,
        swapped_output.action_logits,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    torch.testing.assert_close(
        original_output.state_value,
        swapped_output.state_value,
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_delta_objective_feature_matches_full_score_difference() -> None:
    instance, objective = conditioned_instance()
    env = MethodAssignmentEnv(objective=objective, max_steps=2, patience=2)
    observation = env.reset(instance)
    model = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    with torch.no_grad():
        output = model(observation_to_model_batch(observation))
    encoded_delta = output.encoder_output.pair_feature[0, :, :, -3].cpu().numpy()
    current_assignment = observation["assignment"]
    current_score = score_assignment(
        current_assignment,
        instance=instance,
        objective=objective,
    ).total_score
    expected = np.zeros((instance.n, instance.m), dtype=np.float64)
    for node in range(instance.n):
        for method in range(instance.m):
            candidate = current_assignment.copy()
            candidate[node] = method
            expected[node, method] = (
                score_assignment(candidate, instance=instance, objective=objective).total_score
                - current_score
            )
    np.testing.assert_allclose(encoded_delta, expected, atol=5.0e-7, rtol=0.0)


def test_ablation_protocol_versions_and_training_lock(tmp_path: Path) -> None:
    specs = {}
    for arm in ("A", "B", "C"):
        _, config, _, spec = load_and_validate_protocol(PROTOCOL_PATH, arm_id=arm)
        assert config["status"] == "user_confirmed"
        assert spec["arm_id"] == arm
        assert spec["model"]["feature_version"]
        assert spec["model"]["network_architecture_version"]
        specs[arm] = spec
    assert specs["A"]["sampling"] != specs["B"]["sampling"]
    assert specs["B"]["model"] == specs["C"]["model"]
    locked_protocol = dict(config)
    locked_protocol["status"] = "awaiting_user_confirmation_no_training"
    locked_path = tmp_path / "locked_protocol.yaml"
    locked_path.write_text(
        yaml.safe_dump(locked_protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="locked"):
        run_arm_from_config(locked_path, arm_id="A")


def test_versioned_checkpoint_refuses_experiment_spec_mismatch(tmp_path: Path) -> None:
    instance, objective = conditioned_instance()
    model = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    model.experiment_spec = {"arm_id": "B", "model": model_spec(SYMMETRIC_MODEL_VERSION)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    ppo_config = ppo_config_from_project(load_config(), learning_rate=1.0e-4)
    path = tmp_path / "versioned.pt"
    provenance = checkpoint_metadata(
        objective,
        method_hash=method_matrix_hash(instance),
        data_hash="test-data-hash",
    )
    save_checkpoint(
        path,
        model=model,
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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 3
    assert payload["experiment_spec"] == model.experiment_spec

    restored_model = build_versioned_actor_critic(
        instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        device="cpu",
        hidden_dim=16,
    )
    restored_model.experiment_spec = {
        "arm_id": "C",
        "model": model_spec(SYMMETRIC_MODEL_VERSION),
    }
    with pytest.raises(ValueError, match="experiment_spec mismatch"):
        load_training_checkpoint(
            path,
            model=restored_model,
            optimizer=torch.optim.Adam(restored_model.parameters(), lr=1.0e-4),
            python_rng=random.Random(1),
            numpy_rng=np.random.default_rng(1),
            device=torch.device("cpu"),
            objective=objective,
            method_hash=method_matrix_hash(instance),
            data_hash="test-data-hash",
        )
