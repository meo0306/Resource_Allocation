"""Tests for the versioned Stage7 batch-service engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.build_service_scenario_bank_v1 import build as build_service_scenario_bank
from pa_moap_rl.data.scenarios import load_scenario_bank
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.experiments import run_service_efficiency_v1 as service_runner
from pa_moap_rl.experiments.service_efficiency_v1 import (
    assert_batch_equivalent,
    build_baseline_solver,
    latency_summary,
    run_batched_policy_service,
    run_threaded_baseline_service,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.ppo_solver import build_actor_critic


def _instance():
    return build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )


def _truncate(instance, n: int):
    selected_ids = instance.selected_ids[:n]
    return replace(
        instance,
        instance_name=f"{instance.instance_name}-n{n}",
        selected_ids=selected_ids,
        original_to_local_id={original: index for index, original in enumerate(selected_ids)},
        n=n,
        category_id=instance.category_id[:n].copy(),
        cognitive_load=instance.cognitive_load[:n].copy(),
        concept_need=instance.concept_need[:n].copy(),
        effect_matrix=instance.effect_matrix[:n].copy(),
        student_pref=instance.student_pref[:n].copy(),
        teacher_pref=instance.teacher_pref[:n].copy(),
        feasible_mask=instance.feasible_mask[:n].copy(),
    )


def test_true_batch_matches_independent_deterministic_rollouts() -> None:
    torch.manual_seed(20260922)
    large = _instance()
    small = _truncate(large, large.n - 3)
    objective = load_objective_spec("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    project_config = load_config()
    model = build_actor_critic(
        large,
        config=project_config,
        device="cpu",
        hidden_dim=16,
        objective=objective,
    )
    model.eval()

    sequential = []
    for request_id, instance in (("large", large), ("small", small)):
        result = run_batched_policy_service(
            request_ids=[request_id],
            instances=[instance],
            model=model,
            project_config=project_config,
            objective=objective,
            max_steps=4,
            patience=4,
            requested_batch_size=1,
        )
        sequential.extend(result.requests)
    batched = run_batched_policy_service(
        request_ids=["large", "small"],
        instances=[large, small],
        model=model,
        project_config=project_config,
        objective=objective,
        max_steps=4,
        patience=4,
        requested_batch_size=2,
    )

    assert_batch_equivalent(sequential, batched.requests)
    assert batched.actual_batch_size == 2
    assert batched.synchronized_iterations == 4
    assert batched.mean_padding_ratio > 0.0
    assert all(item.hard_feasible for item in batched.requests)


def test_threaded_baseline_service_preserves_request_order_and_feasibility() -> None:
    instance = _instance()
    objective = load_objective_spec("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    solver = build_baseline_solver("scalarized_greedy", objective=objective)
    result = run_threaded_baseline_service(
        request_ids=["a", "b"],
        instances=[instance, instance],
        solver=solver,
        concurrency=2,
    )

    assert [row["request_id"] for row in result["requests"]] == ["a", "b"]
    assert result["throughput_requests_per_sec"] > 0.0
    assert all(row["hard_feasible"] for row in result["requests"])
    assert all(row["mask_violation_count"] == 0 for row in result["requests"])


def test_latency_summary_and_invalid_solver_validation() -> None:
    summary = latency_summary([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["p95"] == pytest.approx(float(np.quantile([1.0, 2.0, 3.0], 0.95)))

    objective = load_objective_spec("pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml")
    with pytest.raises(ValueError, match="Unknown solver_id"):
        build_baseline_solver("missing", objective=objective)


def test_service_scenario_bank_is_deterministic_and_profile_unique(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_metadata = build_service_scenario_bank(output=first, count=128, seed=20260922)
    second_metadata = build_service_scenario_bank(output=second, count=128, seed=20260922)

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["scenario_count"] == 128
    assert first_metadata["profile_unique_count"] == 128
    scenarios = load_scenario_bank(first)
    assert all(item.schema_version == 2 for item in scenarios)
    assert all(item.weights is None and item.target_distribution is None for item in scenarios)


def test_same_content_workload_has_distinct_profiles_and_exact_baseline_count(
    tmp_path,
    monkeypatch,
) -> None:
    bank = tmp_path / "service.json"
    build_service_scenario_bank(output=bank, count=128, seed=20260922)
    scenarios = load_scenario_bank(bank)
    context = {
        "core_rows": [
            {"size_group": "group1", "topology": "fully_connected", "original_id": "anchor"}
        ],
        "service_scenarios": {item.scenario_id: item for item in scenarios},
    }
    batches = service_runner._workload_batches(
        context=context,
        workload_id="same_content_many_preferences",
        batch_size=128,
        batch_count=2,
    )
    for batch in batches:
        assert len(batch) == 128
        assert len({item[2].scenario_id for item in batch}) == 128
        assert len(
            {
                (
                    tuple(item[2].student_profile.tolist()),
                    tuple(item[2].teacher_profile.tolist()),
                )
                for item in batch
            }
        ) == 128

    monkeypatch.setattr(
        service_runner,
        "_prepare_batch",
        lambda _context, specification: (
            [item[0] for item in specification],
            list(specification),
            0.0,
        ),
    )
    request_ids, specifications, _ = service_runner._baseline_workload_requests(
        context,
        "same_content_many_preferences",
        36,
    )
    assert len(request_ids) == len(specifications) == 36
    assert len({item[2].scenario_id for item in specifications}) == 36
