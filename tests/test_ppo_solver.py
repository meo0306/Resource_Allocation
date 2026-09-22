"""Tests for minimal masked PPO solver."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.solvers.ppo_solver import (
    PPOConfig,
    build_actor_critic,
    collect_rollout,
    collect_vectorized_rollout,
    compute_gae,
    pad_and_stack_observations,
    solve_ppo,
    update_ppo,
)
from pa_moap_rl.utils.metrics import is_hard_feasible


def _example_instance():
    return build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )


def test_compute_gae_matches_discounted_returns_when_values_are_zero() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    values = torch.zeros(3)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=torch.tensor(0.0),
        gamma=0.9,
        gae_lambda=1.0,
    )

    expected = torch.tensor([1.0 + 0.9 + 0.81, 1.0 + 0.9, 1.0])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_collect_rollout_has_no_illegal_actions_and_expected_shapes() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=4, patience=4)
    model = build_actor_critic(instance, device="cpu", hidden_dim=16)
    config = PPOConfig(rollout_steps=4, minibatch_size=2, update_epochs=1, device="cpu", seed=3)

    rollout, _ = collect_rollout(env=env, instance=instance, model=model, config=config)

    assert rollout.actions.shape == (4,)
    assert rollout.returns.shape == (4,)
    assert rollout.advantages.shape == (4,)
    assert rollout.illegal_action_count == 0
    assert rollout.observations["assignment"].shape == (4, instance.n)


def _truncated_instance(instance, n: int):
    selected_ids = instance.selected_ids[:n]
    return replace(
        instance,
        instance_name=f'{instance.instance_name}-n{n}',
        selected_ids=selected_ids,
        original_to_local_id={
            original: index for index, original in enumerate(selected_ids)
        },
        n=n,
        category_id=instance.category_id[:n].copy(),
        cognitive_load=instance.cognitive_load[:n].copy(),
        concept_need=instance.concept_need[:n].copy(),
        effect_matrix=instance.effect_matrix[:n].copy(),
        student_pref=instance.student_pref[:n].copy(),
        teacher_pref=instance.teacher_pref[:n].copy(),
        feasible_mask=instance.feasible_mask[:n].copy(),
    )


def test_vectorized_rollout_pads_nodes_and_pools_equal_budgets() -> None:
    large = _example_instance()
    small = _truncated_instance(large, large.n - 3)
    envs = [
        MethodAssignmentEnv(max_steps=4, patience=4),
        MethodAssignmentEnv(max_steps=4, patience=4),
    ]
    model = build_actor_critic(large, device='cpu', hidden_dim=16)
    config = PPOConfig(
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        device='cpu',
        seed=13,
    )
    initial = [
        env.reset(instance) for env, instance in zip(envs, [small, large])
    ]
    padded = pad_and_stack_observations(initial)
    assert padded['assignment'].shape == (2, large.n)
    assert not padded['node_mask'][0, small.n:].any()
    assert not padded['action_mask'][0, small.n:].any()

    torch.manual_seed(13)
    rollout = collect_vectorized_rollout(
        envs=envs,
        instances=[small, large],
        model=model,
        config=config,
    )
    assert rollout.actions.shape == (8,)
    assert rollout.observations['assignment'].shape == (8, large.n)
    assert rollout.environment_count == 2
    assert rollout.steps_per_environment == 4
    assert rollout.illegal_action_count == 0
    assert len(rollout.instance_summaries) == 2
    for advantages in rollout.advantages.reshape(2, 4):
        assert float(advantages.mean()) == pytest.approx(0.0, abs=1.0e-5)
        assert float(advantages.std(unbiased=False)) == pytest.approx(1.0, abs=1.0e-4)


def test_vectorized_batch_one_matches_single_environment_collection() -> None:
    instance = _example_instance()
    config = PPOConfig(
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        device='cpu',
        seed=17,
    )
    model = build_actor_critic(instance, device='cpu', hidden_dim=16)
    torch.manual_seed(17)
    single, _ = collect_rollout(
        env=MethodAssignmentEnv(max_steps=4, patience=4),
        instance=instance,
        model=model,
        config=config,
    )
    torch.manual_seed(17)
    joint = collect_vectorized_rollout(
        envs=[MethodAssignmentEnv(max_steps=4, patience=4)],
        instances=[instance],
        model=model,
        config=config,
    )
    torch.testing.assert_close(joint.actions, single.actions)
    torch.testing.assert_close(joint.old_log_probs, single.old_log_probs)
    torch.testing.assert_close(joint.rewards, single.rewards)
    torch.testing.assert_close(joint.returns, single.returns)
    torch.testing.assert_close(joint.advantages, single.advantages)
    np.testing.assert_array_equal(
        joint.instance_summaries[0].best_assignment,
        single.best_assignment,
    )


def test_update_ppo_runs_backward_and_returns_losses() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=4, patience=4)
    model = build_actor_critic(instance, device="cpu", hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    config = PPOConfig(rollout_steps=4, minibatch_size=2, update_epochs=1, device="cpu", seed=5)
    rollout, _ = collect_rollout(env=env, instance=instance, model=model, config=config)

    metrics = update_ppo(model=model, optimizer=optimizer, rollout=rollout, config=config)

    assert {"policy_loss", "value_loss", "entropy", "approx_kl", "loss"} <= set(metrics)
    assert np.isfinite(list(metrics.values())).all()
    assert metrics['ppo_epochs_executed'] == pytest.approx(1.0)
    assert metrics['optimizer_minibatches'] == pytest.approx(2.0)
    assert metrics['target_kl_triggered'] == pytest.approx(0.0)


def test_target_kl_stops_after_first_completed_epoch() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=4, patience=4)
    model = build_actor_critic(instance, device='cpu', hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    config = PPOConfig(
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=4,
        target_kl=1.0e-12,
        device='cpu',
        seed=11,
    )
    rollout, _ = collect_rollout(
        env=env,
        instance=instance,
        model=model,
        config=config,
    )

    metrics = update_ppo(
        model=model,
        optimizer=optimizer,
        rollout=rollout,
        config=config,
    )

    assert metrics['target_kl_triggered'] == pytest.approx(1.0)
    assert metrics['target_kl_observed'] > config.target_kl
    assert metrics['ppo_epochs_executed'] == pytest.approx(1.0)
    assert metrics['optimizer_minibatches'] == pytest.approx(2.0)


def test_solve_ppo_smoke_returns_feasible_solution_and_log(tmp_path: Path) -> None:
    instance = _example_instance()
    config = PPOConfig(
        rollout_steps=4,
        update_epochs=1,
        minibatch_size=2,
        total_updates=1,
        seed=7,
        device="cpu",
        env_max_steps=4,
        env_patience=4,
    )
    log_path = tmp_path / "ppo_log.csv"
    checkpoint_path = tmp_path / "ppo.pt"

    result = solve_ppo(
        instance,
        config=config,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
    )

    assert is_hard_feasible(instance, result.assignment)
    assert result.metrics["illegal_action_rate"] == pytest.approx(0.0)
    assert result.metrics["updates"] == 1
    assert log_path.exists()
    assert checkpoint_path.exists()
