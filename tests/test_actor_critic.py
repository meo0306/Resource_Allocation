"""Tests for masked Actor-Critic."""

import numpy as np
import pytest
import torch

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.actor_critic import MASKED_LOGIT_VALUE, MaskedActorCritic, flatten_action, unflatten_action


def _batch_from_example(batch_size: int = 2):
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )
    obs = MethodAssignmentEnv().reset(instance)
    return instance, {
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


def test_actor_critic_forward_shapes_and_mask_application() -> None:
    instance, batch = _batch_from_example(batch_size=2)
    model = MaskedActorCritic(num_categories=instance.k, num_methods=instance.m, hidden_dim=16)

    output = model(batch)

    assert output.action_logits.shape == (2, instance.n, instance.m)
    assert output.masked_logits.shape == (2, instance.n, instance.m)
    assert output.state_value.shape == (2, 1)
    assert torch.all(output.masked_logits[~batch["action_mask"]] <= MASKED_LOGIT_VALUE / 2.0)
    assert torch.allclose(output.masked_logits[batch["action_mask"]], output.action_logits[batch["action_mask"]])


def test_sample_action_never_selects_masked_action() -> None:
    instance, batch = _batch_from_example(batch_size=2)
    model = MaskedActorCritic(num_categories=instance.k, num_methods=instance.m, hidden_dim=16)

    sample = model.sample_action(batch)

    assert sample["flat_action"].shape == (2,)
    assert sample["log_prob"].shape == (2,)
    assert sample["entropy"].shape == (2,)
    for b in range(2):
        assert batch["action_mask"][b, sample["node_index"][b], sample["method_index"][b]]


def test_evaluate_actions_and_backward() -> None:
    instance, batch = _batch_from_example(batch_size=2)
    model = MaskedActorCritic(num_categories=instance.k, num_methods=instance.m, hidden_dim=16)
    sample = model.sample_action(batch)

    evaluated = model.evaluate_actions(batch, flat_action=sample["flat_action"])
    loss = -(evaluated["log_prob"].mean() + 0.01 * evaluated["entropy"].mean()) + evaluated["state_value"].pow(2).mean()
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_flatten_unflatten_mapping() -> None:
    node = torch.tensor([0, 2, 4])
    method = torch.tensor([1, 3, 5])
    flat = flatten_action(node, method, num_methods=11)
    recovered_node, recovered_method = unflatten_action(flat, num_methods=11)

    torch.testing.assert_close(recovered_node, node)
    torch.testing.assert_close(recovered_method, method)


def test_all_masked_state_raises_clear_error() -> None:
    instance, batch = _batch_from_example(batch_size=1)
    model = MaskedActorCritic(num_categories=instance.k, num_methods=instance.m, hidden_dim=16)
    batch["action_mask"] = torch.zeros_like(batch["action_mask"], dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one legal action"):
        model(batch)
