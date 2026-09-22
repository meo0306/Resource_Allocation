"""Consistency tests for the teacher-response action diagnostic."""

from pathlib import Path

import numpy as np
import pytest

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import (
    build_assignment_instance_from_selection,
)
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.diagnose_teacher_response_v2 import (
    _candidate_rewards,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.utils.scoring import score_assignment


def test_vectorized_candidate_rewards_match_full_recomputation() -> None:
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv(
            "examples/summary_ga_own_group1_260417_210154.csv"
        )[0],
    )
    objective = load_objective_spec(
        Path("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    )
    env = MethodAssignmentEnv(
        max_steps=4,
        patience=4,
        config=load_config(),
        objective=objective,
    )
    observation = env.reset(instance)
    assignment = observation["assignment"]
    current = score_assignment(
        assignment, instance=instance, objective=objective
    ).total_score
    rewards = _candidate_rewards(instance, assignment, objective)

    for node in range(instance.n):
        for method in np.flatnonzero(instance.feasible_mask[node]):
            method = int(method)
            if method == int(assignment[node]):
                assert rewards[node, method] == -np.inf
                continue
            candidate = assignment.copy()
            candidate[node] = method
            expected = (
                score_assignment(
                    candidate, instance=instance, objective=objective
                ).total_score
                - current
            )
            assert rewards[node, method] == pytest.approx(expected, abs=1.0e-12)
