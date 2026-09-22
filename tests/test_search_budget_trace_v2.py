"""Regression tests for search-budget condition resolution and traces."""

from pathlib import Path

import pytest

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import (
    build_assignment_instance_from_selection,
)
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.experiments.search_budget_trace_v2 import (
    build_search_conditions,
    derive_condition_from_master,
    deterministic_policy_trace,
    scaled_budget_for_n,
)
from pa_moap_rl.experiments.evaluate_checkpoint_selection_v2 import _better
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.ppo_solver import build_actor_critic


def _protocol():
    return {
        "anchor_conditions": [
            {
                "condition_id": "current_b128_p32",
                "max_steps": 128,
                "patience": 32,
            }
        ],
        "fixed_step_budgets": [128, 256, 512],
        "budget_scan_patience": "equal_budget",
        "n_scaled_budget": {
            "formula": "unit_times_ceil_n_over_unit",
            "unit": 128,
            "patience": "equal_budget",
        },
        "patience_scan": {"max_steps": 512, "values": [32, 64, 128]},
    }


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, 128), (128, 128), (129, 256), (512, 512), (743, 768)],
)
def test_scaled_budget_rounds_up_to_cover_n(n: int, expected: int) -> None:
    assert scaled_budget_for_n(n) == expected


def test_condition_grid_separates_budget_and_patience_controls() -> None:
    conditions = build_search_conditions(_protocol(), n=743)
    by_id = {item.condition_id: item for item in conditions}
    assert by_id["current_b128_p32"].execution_key == (128, 32)
    assert by_id["fixed_b128_p128"].execution_key == (128, 128)
    assert by_id["fixed_b512_p512"].execution_key == (512, 512)
    assert by_id["n_scaled_b768_p768"].execution_key == (768, 768)
    assert by_id["patience_b512_p32"].execution_key == (512, 32)
    assert by_id["patience_b512_p128"].execution_key == (512, 128)


def test_trace_is_complete_feasible_and_best_so_far_monotonic() -> None:
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv(
            "examples/summary_ga_own_group1_260417_210154.csv"
        )[0],
    )
    objective = load_objective_spec(
        Path("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    )
    project_config = load_config()
    model = build_actor_critic(
        instance,
        config=project_config,
        device="cpu",
        hidden_dim=16,
        objective=objective,
    )
    result = deterministic_policy_trace(
        instance,
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=4,
        patience=4,
        exact_score=2.0,
    )

    assert result["executed_moves"] == 4
    assert result["done_reason"] == "max_steps"
    assert len(result["trace"]) == 5
    assert result["hard_feasible_rate"] == 1.0
    assert result["mask_violation_count"] == 0
    assert result["best_score_recompute_error"] <= 1.0e-9
    best_scores = [row["best_score"] for row in result["trace"]]
    assert best_scores == sorted(best_scores)
    assert result["unique_action_count"] + result["repeated_action_count"] == 4
    assert result["accepted_moves"] == result["positive_reward_moves"]


def test_invalid_condition_protocol_is_rejected() -> None:
    protocol = _protocol()
    protocol["fixed_step_budgets"] = [128, 128]
    with pytest.raises(ValueError, match="unique"):
        build_search_conditions(protocol, n=100)


@pytest.mark.parametrize(("max_steps", "patience"), [(4, 4), (8, 2)])
def test_master_trace_truncation_matches_direct_rollout(
    max_steps: int, patience: int
) -> None:
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv(
            "examples/summary_ga_own_group1_260417_210154.csv"
        )[0],
    )
    objective = load_objective_spec(
        Path("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    )
    project_config = load_config()
    model = build_actor_critic(
        instance,
        config=project_config,
        device="cpu",
        hidden_dim=16,
        objective=objective,
    )
    model.eval()
    master = deterministic_policy_trace(
        instance,
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=8,
        patience=8,
        exact_score=2.0,
    )
    derived = derive_condition_from_master(
        master,
        instance=instance,
        objective=objective,
        max_steps=max_steps,
        patience=patience,
    )
    direct = deterministic_policy_trace(
        instance,
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=max_steps,
        patience=patience,
        exact_score=2.0,
    )

    assert derived["best_assignment"] == direct["best_assignment"]
    assert derived["best_score"] == pytest.approx(direct["best_score"], abs=1.0e-12)
    assert derived["executed_moves"] == direct["executed_moves"]
    assert derived["accepted_moves"] == direct["accepted_moves"]
    assert derived["covered_node_count"] == direct["covered_node_count"]
    assert derived["done_reason"] == direct["done_reason"]


def test_checkpoint_selection_uses_score_then_p90_then_earliest() -> None:
    incumbent = {
        "update": 70,
        "mean_total_score": 0.75,
        "p90_relative_gap": 0.02,
    }
    assert _better(
        {
            "update": 80,
            "mean_total_score": 0.751,
            "p90_relative_gap": 0.03,
        },
        incumbent,
    )
    assert _better(
        {
            "update": 80,
            "mean_total_score": 0.75,
            "p90_relative_gap": 0.019,
        },
        incumbent,
    )
    assert _better(
        {
            "update": 60,
            "mean_total_score": 0.75,
            "p90_relative_gap": 0.02,
        },
        incumbent,
    )
    assert not _better(
        {
            "update": 80,
            "mean_total_score": 0.75,
            "p90_relative_gap": 0.02,
        },
        incumbent,
    )
