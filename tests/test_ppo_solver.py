"""Tests for minimal masked PPO solver."""

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
    compute_gae,
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
