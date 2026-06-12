"""Tests for neural encoders."""

import numpy as np
import torch

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.encoders import AssignmentStateEncoder, GlobalEncoder, MethodEncoder, NodeEncoder, PairFeatureBuilder


def _batch_from_example(batch_size: int = 2):
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )
    obs = MethodAssignmentEnv().reset(instance)
    batch = {
        "category_id": torch.tensor(np.stack([obs["category_id"]] * batch_size), dtype=torch.long),
        "concept_need": torch.tensor(np.stack([obs["concept_need"]] * batch_size), dtype=torch.float32),
        "method_attr": torch.tensor(obs["method_attr"], dtype=torch.float32),
        "assignment": torch.tensor(np.stack([obs["assignment"]] * batch_size), dtype=torch.long),
        "effect_matrix": torch.tensor(np.stack([obs["effect_matrix"]] * batch_size), dtype=torch.float32),
        "student_pref": torch.tensor(np.stack([obs["student_pref"]] * batch_size), dtype=torch.float32),
        "teacher_pref": torch.tensor(np.stack([obs["teacher_pref"]] * batch_size), dtype=torch.float32),
        "node_mask": torch.tensor(np.stack([obs["node_mask"]] * batch_size), dtype=torch.bool),
        "method_distribution": torch.tensor(np.stack([obs["method_distribution"]] * batch_size), dtype=torch.float32),
        "target_distribution": torch.tensor(np.stack([obs["target_distribution"]] * batch_size), dtype=torch.float32),
        "current_scores": torch.tensor(np.stack([obs["current_scores"]] * batch_size), dtype=torch.float32),
        "action_mask": torch.tensor(np.stack([obs["action_mask"]] * batch_size), dtype=torch.bool),
    }
    return instance, batch


def test_individual_encoder_shapes() -> None:
    instance, batch = _batch_from_example(batch_size=2)
    hidden_dim = 16

    node_encoder = NodeEncoder(num_categories=instance.k, num_methods=instance.m, hidden_dim=hidden_dim)
    method_encoder = MethodEncoder(num_methods=instance.m, hidden_dim=hidden_dim)
    global_encoder = GlobalEncoder(num_methods=instance.m, hidden_dim=hidden_dim)

    node_repr = node_encoder(
        category_id=batch["category_id"],
        concept_need=batch["concept_need"],
        assignment=batch["assignment"],
        effect_matrix=batch["effect_matrix"],
        student_pref=batch["student_pref"],
        teacher_pref=batch["teacher_pref"],
    )
    method_repr = method_encoder(batch["method_attr"], batch_size=2)
    global_repr = global_encoder(
        node_repr=node_repr,
        node_mask=batch["node_mask"],
        method_distribution=batch["method_distribution"],
        target_distribution=batch["target_distribution"],
        current_scores=batch["current_scores"],
    )

    assert node_repr.shape == (2, instance.n, hidden_dim)
    assert method_repr.shape == (2, instance.m, hidden_dim)
    assert global_repr.shape == (2, hidden_dim)


def test_pair_feature_builder_shape_and_current_indicator() -> None:
    instance, batch = _batch_from_example(batch_size=2)
    hidden_dim = 16
    node_repr = torch.randn(2, instance.n, hidden_dim)
    method_repr = torch.randn(2, instance.m, hidden_dim)
    global_repr = torch.randn(2, hidden_dim)
    builder = PairFeatureBuilder(num_methods=instance.m)

    pair_feature = builder(
        node_repr=node_repr,
        method_repr=method_repr,
        global_repr=global_repr,
        assignment=batch["assignment"],
        effect_matrix=batch["effect_matrix"],
        student_pref=batch["student_pref"],
        teacher_pref=batch["teacher_pref"],
        method_distribution=batch["method_distribution"],
        target_distribution=batch["target_distribution"],
        node_mask=batch["node_mask"],
    )

    assert pair_feature.shape == (2, instance.n, instance.m, 3 * hidden_dim + 9)
    current_indicator = pair_feature[..., -1]
    for b in range(2):
        for i in range(instance.n):
            assert current_indicator[b, i, batch["assignment"][b, i]] == 1.0
            assert current_indicator[b, i].sum() == 1.0


def test_composite_assignment_state_encoder_shapes() -> None:
    instance, batch = _batch_from_example(batch_size=2)
    hidden_dim = 16
    encoder = AssignmentStateEncoder(num_categories=instance.k, num_methods=instance.m, hidden_dim=hidden_dim)

    output = encoder(
        category_id=batch["category_id"],
        concept_need=batch["concept_need"],
        method_attr=batch["method_attr"],
        assignment=batch["assignment"],
        effect_matrix=batch["effect_matrix"],
        student_pref=batch["student_pref"],
        teacher_pref=batch["teacher_pref"],
        node_mask=batch["node_mask"],
        method_distribution=batch["method_distribution"],
        target_distribution=batch["target_distribution"],
        current_scores=batch["current_scores"],
    )

    assert output.node_repr.shape == (2, instance.n, hidden_dim)
    assert output.method_repr.shape == (2, instance.m, hidden_dim)
    assert output.global_repr.shape == (2, hidden_dim)
    assert output.pair_feature.shape == (2, instance.n, instance.m, 3 * hidden_dim + 9)
