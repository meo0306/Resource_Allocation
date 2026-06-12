"""Tests for MethodAssignmentEnv."""

import numpy as np
import pytest

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.utils.scoring import J


def _example_instance():
    return build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )


def _first_legal_action(action_mask: np.ndarray) -> tuple[int, int]:
    indices = np.argwhere(action_mask)
    assert indices.size > 0
    return int(indices[0, 0]), int(indices[0, 1])


def test_reset_returns_complete_observation_and_effect_greedy_initial_solution() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=4, patience=4)
    obs = env.reset(instance)

    assert obs["assignment"].shape == (instance.n,)
    assert obs["method_distribution"].shape == (instance.m,)
    assert obs["current_scores"].shape == (6,)
    assert obs["action_mask"].shape == (instance.n, instance.m)
    assert obs["node_mask"].all()
    assert np.all(instance.feasible_mask[np.arange(instance.n), obs["assignment"]])

    expected = np.argmax(np.where(instance.feasible_mask, instance.effect_matrix, -np.inf), axis=1)
    np.testing.assert_array_equal(obs["assignment"], expected)
    np.testing.assert_allclose(obs["method_distribution"], env.method_dist)
    np.testing.assert_allclose(obs["current_scores"], env.current_scores)


def test_step_updates_assignment_scores_distribution_and_best_state() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=8, patience=8)
    obs = env.reset(instance)
    before_assignment = obs["assignment"].copy()
    before_score = float(obs["current_scores"][-1])
    action = _first_legal_action(obs["action_mask"])

    next_obs, reward, done, info = env.step(action)

    expected_assignment = before_assignment.copy()
    expected_assignment[action[0]] = action[1]
    np.testing.assert_array_equal(next_obs["assignment"], expected_assignment)
    assert reward == pytest.approx(float(next_obs["current_scores"][-1]) - before_score)
    assert reward == pytest.approx(env.compute_reward(before_assignment, expected_assignment))
    assert float(next_obs["current_scores"][-1]) == pytest.approx(J(expected_assignment, instance=instance))
    np.testing.assert_allclose(next_obs["method_distribution"], env.method_dist)
    assert not next_obs["action_mask"][action[0], action[1]]
    assert done is False
    assert info["step_count"] == 1

    if reward > env.improvement_eps:
        np.testing.assert_array_equal(env.best_assignment, expected_assignment)
        assert env.best_score == pytest.approx(float(next_obs["current_scores"][-1]))
    else:
        np.testing.assert_array_equal(env.best_assignment, before_assignment)
        assert env.best_score == pytest.approx(before_score)


def test_illegal_action_cannot_execute() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv()
    obs = env.reset(instance)
    current_method = int(obs["assignment"][0])

    with pytest.raises(ValueError):
        env.step((0, current_method))

    with pytest.raises(ValueError):
        env.reset(instance, initial_assignment=np.full(instance.n, instance.m, dtype=np.int64))


def test_done_by_max_steps() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=1, patience=99)
    obs = env.reset(instance)

    _, _, done, info = env.step(_first_legal_action(obs["action_mask"]))

    assert done
    assert info["done_reason"] == "max_steps"


def test_done_by_patience_and_best_so_far_survives_negative_or_small_reward() -> None:
    instance = _example_instance()
    env = MethodAssignmentEnv(max_steps=99, patience=1, improvement_eps=10.0)
    obs = env.reset(instance)
    initial_assignment = obs["assignment"].copy()
    initial_best = float(obs["current_scores"][-1])

    _, _, done, info = env.step(_first_legal_action(obs["action_mask"]))

    assert done
    assert info["done_reason"] == "patience"
    assert env.best_score == pytest.approx(initial_best)
    np.testing.assert_array_equal(env.best_assignment, initial_assignment)


def test_step_before_reset_and_step_after_done_fail_clearly() -> None:
    env = MethodAssignmentEnv(max_steps=1, patience=99)
    with pytest.raises(RuntimeError):
        env.get_action_mask()

    obs = env.reset(_example_instance())
    env.step(_first_legal_action(obs["action_mask"]))
    with pytest.raises(RuntimeError):
        env.step(_first_legal_action(env.get_action_mask()))
