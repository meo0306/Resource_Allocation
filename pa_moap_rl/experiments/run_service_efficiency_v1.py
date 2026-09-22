"""Run the frozen four-topology batch-service efficiency experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml

from pa_moap_rl.checkpointing import method_matrix_hash, validate_checkpoint_metadata
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import PreferenceScenario, apply_scenario, load_scenario_bank
from pa_moap_rl.experiments.service_efficiency_v1 import (
    SERVICE_ENGINE_VERSION,
    assert_batch_equivalent,
    build_baseline_solver,
    latency_summary,
    run_batched_policy_service,
    run_threaded_baseline_service,
)
from pa_moap_rl.objective import ObjectiveSpec, load_objective_spec
from pa_moap_rl.solvers.ppo_solver import PPOConfig, build_actor_critic


DEFAULT_CONFIG = Path("pa_moap_rl/configs/service_efficiency_benchmark_four_topology_v1.yaml")
RUNNER_VERSION = "service_efficiency_runner_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignment_hash(values: Sequence[int]) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in values for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def _compact_request(row: dict[str, Any]) -> dict[str, Any]:
    compact = dict(row)
    assignment = compact.pop("assignment", None)
    if assignment is not None:
        compact["assignment_sha256"] = _assignment_hash(assignment)
        compact["assignment_length"] = len(assignment)
    metrics = compact.get("solver_metrics")
    if isinstance(metrics, dict):
        compact["solver_metrics"] = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return compact


def _compact_batch(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["requests"] = [_compact_request(row) for row in value["requests"]]
    return result


def _source_checkpoint_data_hash(source_metadata: dict[str, Any]) -> str:
    for item in source_metadata["inputs"].values():
        if _sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen PPO training input drift: {item['path']}")
    return hashlib.sha256(
        "".join(item["sha256"] for item in source_metadata["inputs"].values()).encode("ascii")
    ).hexdigest()


def _validate_frozen_config(config: dict[str, Any]) -> tuple[dict[str, Any], ObjectiveSpec]:
    frozen_entry = config["frozen_ppo"]
    frozen_path = Path(frozen_entry["config"])
    if _sha256(frozen_path) != frozen_entry["config_sha256"]:
        raise ValueError("Frozen PPO config hash mismatch.")
    frozen = _read_yaml(frozen_path)
    if frozen["status"] != "frozen_by_user_mainline_return_after_failed_stage6a_branch":
        raise ValueError("Stage7 requires the user-confirmed mainline freeze.")
    if frozen["problem_and_model"]["model_version"] != "legacy_separate_v1":
        raise ValueError("Stage7 v1 is registered for legacy_separate_v1 only.")
    if frozen["stage6a_branch"]["status"] != "closed_failed_exploratory_branch_not_selected":
        raise ValueError("Stage6A branch state mismatch.")
    objective = load_objective_spec(config["objective"]["config"])
    if objective.config_hash != config["objective"]["config_hash"]:
        raise ValueError("Frozen objective hash mismatch.")
    if objective.config_hash != frozen["objective"]["config_hash"]:
        raise ValueError("PPO freeze and benchmark objective mismatch.")
    checkpoint_path = Path(frozen_entry["checkpoint"])
    if _sha256(checkpoint_path) != frozen_entry["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch.")
    if frozen_entry["checkpoint_sha256"] != frozen["seed0_reference_checkpoint"]["sha256"]:
        raise ValueError("Benchmark checkpoint differs from frozen seed0 reference.")
    return frozen, objective


def _load_model_bundle(
    *,
    config: dict[str, Any],
    frozen: dict[str, Any],
    objective: ObjectiveSpec,
    first_instance: Any,
    device: torch.device,
) -> tuple[Any, PPOConfig, dict[str, Any], float]:
    started = time.perf_counter()
    checkpoint_path = Path(config["frozen_ppo"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["update"]) != int(config["frozen_ppo"]["expected_update"]):
        raise ValueError("Frozen checkpoint update mismatch.")
    if checkpoint.get("experiment_spec") is not None:
        raise ValueError("Stage7 mainline checkpoint must use the original legacy model path.")
    source_metadata_path = Path(frozen["evidence"]["source_run_metadata"]["path"])
    if _sha256(source_metadata_path) != frozen["evidence"]["source_run_metadata"]["sha256"]:
        raise ValueError("Frozen source metadata hash mismatch.")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    checkpoint_data_hash = _source_checkpoint_data_hash(source_metadata)
    if checkpoint_data_hash != source_metadata["data_hash"]:
        raise ValueError("Frozen checkpoint data hash mismatch.")
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_matrix_hash(first_instance),
        data_hash=checkpoint_data_hash,
    )
    project_config = load_config()
    model = build_actor_critic(
        first_instance,
        config=project_config,
        device=device,
        hidden_dim=checkpoint.get("hidden_dim"),
        objective=objective,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ppo_config = replace(
        PPOConfig(**checkpoint["ppo_config"]),
        device=str(device),
        env_max_steps=int(frozen["evaluation_and_inference"]["env_max_steps"]),
        env_patience=int(frozen["evaluation_and_inference"]["env_patience"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, ppo_config, source_metadata, time.perf_counter() - started


def _load_context(config: dict[str, Any], objective: ObjectiveSpec) -> dict[str, Any]:
    data = config["data"]
    manifest_rows = _read_csv(data["instance_manifest"])
    by_original = {row["original_id"]: row for row in manifest_rows}
    core_rows = _read_csv(data["core_manifest"])
    controlled_pairs = _read_csv(data["controlled_pairs"])
    interpolation_pairs = _read_csv(data["interpolation_pairs"])
    controlled_scenarios = {
        item.scenario_id: item for item in load_scenario_bank(data["controlled_scenarios"])
    }
    interpolation_scenarios = {
        item.scenario_id: item for item in load_scenario_bank(data["interpolation_scenarios"])
    }
    service_scenarios = {
        item.scenario_id: item for item in load_scenario_bank(data["service_scenarios"])
    }
    if any(item.schema_version != 2 for item in controlled_scenarios.values()):
        raise ValueError("Controlled scenario bank must use schema v2.")
    if any(item.schema_version != 2 for item in interpolation_scenarios.values()):
        raise ValueError("Interpolation scenario bank must use schema v2.")
    if any(item.schema_version != 2 for item in service_scenarios.values()):
        raise ValueError("Service scenario bank must use schema v2.")
    expected_service_hash = data.get("service_scenarios_sha256")
    if expected_service_hash and _sha256(data["service_scenarios"]) != expected_service_hash:
        raise ValueError("Service scenario bank hash mismatch.")
    required_service_count = max(int(value) for value in config["service_workloads"]["batch_sizes"])
    if len(service_scenarios) < required_service_count:
        raise ValueError(
            f"Service scenario bank has {len(service_scenarios)} rows; "
            f"at least {required_service_count} are required."
        )
    service_profile_pairs = {
        (
            tuple(float(value) for value in item.student_profile),
            tuple(float(value) for value in item.teacher_profile),
        )
        for item in service_scenarios.values()
    }
    if len(service_profile_pairs) != len(service_scenarios):
        raise ValueError("Service scenario bank contains duplicate preference profiles.")
    for row in core_rows:
        if row["original_id"] not in by_original:
            raise ValueError(f"Core instance missing from dev calibration: {row['original_id']}")
    return {
        "manifest_rows": manifest_rows,
        "by_original": by_original,
        "core_rows": core_rows,
        "controlled_pairs": controlled_pairs,
        "interpolation_pairs": interpolation_pairs,
        "controlled_scenarios": controlled_scenarios,
        "interpolation_scenarios": interpolation_scenarios,
        "service_scenarios": service_scenarios,
        "exact_dir": Path(data["controlled_exact_dir"]),
        "objective": objective,
        "project_config": load_config(),
    }


def _condition(context: dict[str, Any], original_id: str, scenario: PreferenceScenario) -> Any:
    row = context["by_original"][original_id]
    base = load_instance_json(row["output_json"])
    return apply_scenario(
        base,
        scenario,
        context["project_config"],
        objective=context["objective"],
    )


def _quality_requests(context: dict[str, Any], count: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for pair in context["controlled_pairs"]:
        grouped.setdefault((pair["size_group"], pair["topology"]), []).append(pair)
    selected: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(grouped)):
        candidates = sorted(grouped[key], key=lambda row: row["pair_id"])
        pair = candidates[index % len(candidates)]
        exact_path = context["exact_dir"] / f"{pair['pair_id']}.json"
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
        if exact["objective_hash"] != context["objective"].config_hash:
            raise ValueError(f"Exact objective mismatch: {pair['pair_id']}")
        selected.append(
            {
                **pair,
                "exact_score": float(exact["score"]["total_score"]),
                "exact_sha256": _sha256(exact_path),
            }
        )
    if len(selected) != 36:
        raise ValueError(f"Expected 36 topology-size strata, got {len(selected)}.")
    return selected[: int(count)]


def _prepare_quality(context: dict[str, Any], requests: Sequence[dict[str, Any]]) -> tuple[list[str], list[Any]]:
    ids: list[str] = []
    instances: list[Any] = []
    for row in requests:
        ids.append(row["pair_id"])
        scenario = context["controlled_scenarios"][row["scenario_id"]]
        instances.append(_condition(context, row["original_id"], scenario))
    return ids, instances


def _relative_gap(exact: float, score: float) -> float:
    return float(max(0.0, exact - score) / max(abs(exact), 1.0e-12))


def _quality_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gaps = [float(row["relative_gap"]) for row in rows]
    latencies = [float(row["latency_sec"]) for row in rows]
    return {
        "count": len(rows),
        "mean_relative_gap": float(statistics.fmean(gaps)),
        "median_relative_gap": float(statistics.median(gaps)),
        "p90_relative_gap": float(np.quantile(gaps, 0.90)),
        "worst_relative_gap": float(max(gaps)),
        "latency": latency_summary(latencies),
        "all_hard_feasible": all(bool(row["hard_feasible"]) for row in rows),
        "mask_violation_count": sum(int(row["mask_violation_count"]) for row in rows),
    }


def _run_preflight(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    model: Any,
    quality_requests: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    settings = config["preflight"]
    subset = quality_requests[: int(settings["equivalence_request_count"])]
    request_ids, instances = _prepare_quality(context, subset)
    sequential = []
    for request_id, instance in zip(request_ids, instances):
        result = run_batched_policy_service(
            request_ids=[request_id],
            instances=[instance],
            model=model,
            project_config=context["project_config"],
            objective=context["objective"],
            max_steps=int(settings["max_steps"]),
            patience=int(settings["patience"]),
            requested_batch_size=1,
        )
        sequential.extend(result.requests)
    batched = run_batched_policy_service(
        request_ids=request_ids,
        instances=instances,
        model=model,
        project_config=context["project_config"],
        objective=context["objective"],
        max_steps=int(settings["max_steps"]),
        patience=int(settings["patience"]),
        requested_batch_size=len(instances),
    )
    assert_batch_equivalent(
        sequential,
        batched.requests,
        score_tolerance=float(settings["score_tolerance"]),
    )
    result = {
        "status": "passed",
        "request_count": len(instances),
        "score_tolerance": float(settings["score_tolerance"]),
        "batch_result": _compact_batch(batched.to_dict()),
    }
    _write_json_atomic(output_dir / "preflight_equivalence.json", result)
    return result


def _run_quality_frontier(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    model: Any,
    quality_requests: list[dict[str, Any]],
    output_dir: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    settings = config["quality_frontier"]
    request_ids, instances = _prepare_quality(context, quality_requests)
    exact_by_id = {row["pair_id"]: float(row["exact_score"]) for row in quality_requests}
    summaries: list[dict[str, Any]] = []

    ppo_budgets = settings["ppo_budgets"][:1] if smoke else settings["ppo_budgets"]
    ppo_batch_size = min(int(settings["ppo_batch_size"]), len(instances))
    for budget in ppo_budgets:
        cell = output_dir / "quality" / f"{budget['id']}.json"
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for offset in range(0, len(instances), ppo_batch_size):
            batch_ids = request_ids[offset : offset + ppo_batch_size]
            batch_instances = instances[offset : offset + ppo_batch_size]
            result = run_batched_policy_service(
                request_ids=batch_ids,
                instances=batch_instances,
                model=model,
                project_config=context["project_config"],
                objective=context["objective"],
                max_steps=int(budget["max_steps"]),
                patience=int(budget["patience"]),
                requested_batch_size=ppo_batch_size,
            )
            for request in result.requests:
                rows.append(
                    {
                        "request_id": request.request_id,
                        "total_score": request.score,
                        "relative_gap": _relative_gap(exact_by_id[request.request_id], request.score),
                        "latency_sec": request.completion_latency_sec,
                        "hard_feasible": request.hard_feasible,
                        "mask_violation_count": request.mask_violation_count,
                        "assignment_sha256": _assignment_hash(request.assignment),
                    }
                )
        summary = _quality_summary(rows)
        summary.update(
            {
                "solver_id": budget["id"],
                "solver_family": "ppo",
                "wall_latency_sec": time.perf_counter() - started,
                "batch_size": ppo_batch_size,
                "max_steps": int(budget["max_steps"]),
                "patience": int(budget["patience"]),
            }
        )
        _write_json_atomic(cell, {"summary": summary, "requests": rows})
        summaries.append(summary)

    baseline_specs = settings["baselines"]
    if smoke:
        baseline_specs = [item for item in baseline_specs if item["solver"] != "gurobi"][:3]
    for spec in baseline_specs:
        solver = build_baseline_solver(
            spec["solver"],
            objective=context["objective"],
            max_iter=spec.get("max_iter"),
            time_limit=spec.get("time_limit"),
            gurobi_threads=int(spec.get("gurobi_threads", 1)),
            seed=20260922,
        )
        result = run_threaded_baseline_service(
            request_ids=request_ids,
            instances=instances,
            solver=solver,
            concurrency=int(spec["concurrency"]),
        )
        rows = []
        for request in result["requests"]:
            rows.append(
                {
                    "request_id": request["request_id"],
                    "total_score": request["total_score"],
                    "relative_gap": _relative_gap(
                        exact_by_id[request["request_id"]], request["total_score"]
                    ),
                    "latency_sec": request["latency_sec"],
                    "hard_feasible": request["hard_feasible"],
                    "mask_violation_count": request["mask_violation_count"],
                    "assignment_sha256": _assignment_hash(request["assignment"]),
                }
            )
        summary = _quality_summary(rows)
        summary.update(
            {
                "solver_id": spec["id"],
                "solver_family": spec["solver"],
                "wall_latency_sec": result["wall_latency_sec"],
                "throughput_requests_per_sec": result["throughput_requests_per_sec"],
                "concurrency": int(spec["concurrency"]),
                "max_iter": spec.get("max_iter"),
                "time_limit": spec.get("time_limit"),
            }
        )
        _write_json_atomic(output_dir / "quality" / f"{spec['id']}.json", {"summary": summary, "requests": rows})
        summaries.append(summary)
    _write_csv(output_dir / "quality_frontier.csv", summaries)
    return summaries


def _workload_batches(
    *,
    context: dict[str, Any],
    workload_id: str,
    batch_size: int,
    batch_count: int,
) -> list[list[tuple[str, str, PreferenceScenario]]]:
    result: list[list[tuple[str, str, PreferenceScenario]]] = []
    if workload_id == "same_content_many_preferences":
        anchors = sorted(context["core_rows"], key=lambda row: (row["size_group"], row["topology"], row["original_id"]))
        scenarios = [context["service_scenarios"][key] for key in sorted(context["service_scenarios"])]
        for batch_index in range(batch_count):
            anchor = anchors[(batch_index * 7) % len(anchors)]
            batch = []
            for slot in range(batch_size):
                scenario = scenarios[(batch_index * batch_size + slot) % len(scenarios)]
                request_id = f"same-b{batch_size:03d}-{batch_index:03d}-{slot:03d}"
                batch.append((request_id, anchor["original_id"], scenario))
            scenario_ids = [item[2].scenario_id for item in batch]
            profile_pairs = {
                (
                    tuple(float(value) for value in item[2].student_profile),
                    tuple(float(value) for value in item[2].teacher_profile),
                )
                for item in batch
            }
            if len(set(scenario_ids)) != batch_size or len(profile_pairs) != batch_size:
                raise ValueError("Same-content workload must use distinct preferences within each batch.")
            result.append(batch)
        return result

    pairs = list(context["interpolation_pairs"])
    if workload_id == "new_content_new_preferences_bucketed":
        pairs.sort(key=lambda row: (int(context["by_original"][row["original_id"]]["N"]), row["pair_id"]))
        for batch_index in range(batch_count):
            if batch_count == 1:
                start = 0
            else:
                start = round((len(pairs) - batch_size) * batch_index / (batch_count - 1))
            chosen = [pairs[(start + slot) % len(pairs)] for slot in range(batch_size)]
            result.append(
                [
                    (
                        f"bucket-b{batch_size:03d}-{batch_index:03d}-{slot:03d}",
                        row["original_id"],
                        context["interpolation_scenarios"][row["scenario_id"]],
                    )
                    for slot, row in enumerate(chosen)
                ]
            )
        return result

    if workload_id == "new_content_new_preferences_padding":
        pairs.sort(key=lambda row: (int(context["by_original"][row["original_id"]]["N"]), row["pair_id"]))
        for batch_index in range(batch_count):
            stride = max(1, len(pairs) // batch_size)
            chosen = [pairs[(batch_index + slot * stride) % len(pairs)] for slot in range(batch_size)]
            result.append(
                [
                    (
                        f"padding-b{batch_size:03d}-{batch_index:03d}-{slot:03d}",
                        row["original_id"],
                        context["interpolation_scenarios"][row["scenario_id"]],
                    )
                    for slot, row in enumerate(chosen)
                ]
            )
        return result
    raise ValueError(f"Unknown workload_id: {workload_id}")


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"process_time_sec": time.process_time()}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        result["rss_bytes"] = int(process.memory_info().rss)
        result["cpu_percent"] = float(process.cpu_percent(interval=None))
    except ImportError:
        result["rss_bytes"] = None
        result["cpu_percent"] = None
    if device.type == "cuda":
        result["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        result["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return result


def _prepare_batch(context: dict[str, Any], spec: Sequence[tuple[str, str, PreferenceScenario]]) -> tuple[list[str], list[Any], float]:
    started = time.perf_counter()
    request_ids = [item[0] for item in spec]
    instances = [_condition(context, item[1], item[2]) for item in spec]
    return request_ids, instances, time.perf_counter() - started


def _run_ppo_service_cells(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    model: Any,
    output_dir: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    service = config["service_workloads"]
    batch_sizes = [1, 8] if smoke else [int(value) for value in service["batch_sizes"]]
    warmup = 0 if smoke else int(service["warmup_batches_per_cell"])
    measured = 1 if smoke else int(service["measured_batches_per_cell"])
    workload_ids = [
        "same_content_many_preferences",
        "new_content_new_preferences_bucketed",
        "new_content_new_preferences_padding",
    ]
    budgets = config["quality_frontier"]["ppo_budgets"][:1] if smoke else config["quality_frontier"]["ppo_budgets"]
    device = next(model.parameters()).device
    summaries: list[dict[str, Any]] = []
    for budget in budgets:
        for workload_id in workload_ids:
            for batch_size in batch_sizes:
                cell_path = output_dir / "service" / "ppo" / budget["id"] / workload_id / f"batch_{batch_size:03d}.json"
                if cell_path.is_file():
                    existing = json.loads(cell_path.read_text(encoding="utf-8"))
                    summaries.append(existing["summary"])
                    continue
                specifications = _workload_batches(
                    context=context,
                    workload_id=workload_id,
                    batch_size=batch_size,
                    batch_count=warmup + measured,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                before = _resource_snapshot(device)
                measured_batches: list[dict[str, Any]] = []
                for batch_index, specification in enumerate(specifications):
                    request_ids, instances, prepare_sec = _prepare_batch(context, specification)
                    result = run_batched_policy_service(
                        request_ids=request_ids,
                        instances=instances,
                        model=model,
                        project_config=context["project_config"],
                        objective=context["objective"],
                        max_steps=int(budget["max_steps"]),
                        patience=int(budget["patience"]),
                        requested_batch_size=batch_size,
                    )
                    if batch_index >= warmup:
                        row = _compact_batch(result.to_dict())
                        row["prepare_latency_sec"] = prepare_sec
                        row["end_to_end_batch_latency_sec"] = prepare_sec + result.batch_latency_sec
                        for request in row["requests"]:
                            request["end_to_end_completion_latency_sec"] = prepare_sec + request["completion_latency_sec"]
                        measured_batches.append(row)
                after = _resource_snapshot(device)
                request_rows = [request for batch in measured_batches for request in batch["requests"]]
                total_end_to_end = sum(float(batch["end_to_end_batch_latency_sec"]) for batch in measured_batches)
                summary = {
                    "solver_id": budget["id"],
                    "solver_family": "ppo",
                    "workload_id": workload_id,
                    "requested_batch_size": batch_size,
                    "measured_batch_count": len(measured_batches),
                    "request_count": len(request_rows),
                    "throughput_requests_per_sec": len(request_rows) / total_end_to_end,
                    "batch_latency": latency_summary([float(batch["end_to_end_batch_latency_sec"]) for batch in measured_batches]),
                    "request_latency": latency_summary([float(row["end_to_end_completion_latency_sec"]) for row in request_rows]),
                    "mean_padding_ratio": float(statistics.fmean(float(batch["mean_padding_ratio"]) for batch in measured_batches)),
                    "peak_padding_ratio": max(float(batch["peak_padding_ratio"]) for batch in measured_batches),
                    "max_steps": int(budget["max_steps"]),
                    "patience": int(budget["patience"]),
                }
                payload = {"summary": summary, "resource_before": before, "resource_after": after, "batches": measured_batches}
                _write_json_atomic(cell_path, payload)
                summaries.append(summary)
    _write_csv(output_dir / "ppo_service_summary.csv", summaries)
    return summaries


def _baseline_workload_requests(context: dict[str, Any], workload_id: str, count: int) -> tuple[list[str], list[Any], float]:
    if workload_id == "same_content_many_preferences":
        anchor = sorted(context["core_rows"], key=lambda row: (row["size_group"], row["topology"], row["original_id"]))[len(context["core_rows"]) // 2]
        scenarios = [context["service_scenarios"][key] for key in sorted(context["service_scenarios"])[:count]]
        if len(scenarios) != count:
            raise ValueError(f"Expected {count} distinct baseline service scenarios, got {len(scenarios)}.")
        spec = [(f"baseline-same-{index:03d}", anchor["original_id"], scenario) for index, scenario in enumerate(scenarios)]
    else:
        pairs = sorted(context["interpolation_pairs"], key=lambda row: row["pair_id"])
        indices = np.linspace(0, len(pairs) - 1, num=count, dtype=int)
        spec = []
        for index, pair_index in enumerate(indices):
            pair = pairs[int(pair_index)]
            spec.append((f"baseline-mixed-{index:03d}", pair["original_id"], context["interpolation_scenarios"][pair["scenario_id"]]))
    return _prepare_batch(context, spec)


def _run_baseline_service_cells(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    output_dir: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    settings = config["baseline_service"]
    request_count = 4 if smoke else int(settings["request_count_per_workload"])
    concurrency_levels = [1] if smoke else [int(value) for value in settings["concurrency_levels"]]
    specs = settings["configurations"][:1] if smoke else settings["configurations"]
    summaries: list[dict[str, Any]] = []
    for workload_id in ("same_content_many_preferences", "new_content_new_preferences"):
        request_ids, instances, prepare_sec = _baseline_workload_requests(context, workload_id, request_count)
        for spec in specs:
            for concurrency in concurrency_levels:
                cell_path = output_dir / "service" / "baselines" / workload_id / f"{spec['id']}__c{concurrency}.json"
                if cell_path.is_file():
                    existing = json.loads(cell_path.read_text(encoding="utf-8"))
                    summaries.append(existing["summary"])
                    continue
                solver = build_baseline_solver(
                    spec["solver"],
                    objective=context["objective"],
                    max_iter=spec.get("max_iter"),
                    time_limit=spec.get("time_limit"),
                    gurobi_threads=int(spec.get("gurobi_threads", 1)),
                    seed=20260922,
                )
                result = run_threaded_baseline_service(
                    request_ids=request_ids,
                    instances=instances,
                    solver=solver,
                    concurrency=concurrency,
                )
                rows = [_compact_request(row) for row in result["requests"]]
                end_to_end = prepare_sec + float(result["wall_latency_sec"])
                summary = {
                    "solver_id": spec["id"],
                    "solver_family": spec["solver"],
                    "workload_id": workload_id,
                    "concurrency": concurrency,
                    "request_count": len(rows),
                    "prepare_latency_sec": prepare_sec,
                    "wall_latency_sec": result["wall_latency_sec"],
                    "end_to_end_latency_sec": end_to_end,
                    "throughput_requests_per_sec": len(rows) / end_to_end,
                    "request_latency": result["request_latency"],
                }
                _write_json_atomic(cell_path, {"summary": summary, "requests": rows})
                summaries.append(summary)
    _write_csv(output_dir / "baseline_service_summary.csv", summaries)
    return summaries


def _write_report(
    *,
    output_dir: Path,
    quality: Sequence[dict[str, Any]],
    ppo_service: Sequence[dict[str, Any]],
    baseline_service: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    frozen_quality = next(row for row in quality if row["solver_id"] == "ppo_512_32_frozen")
    tolerance = float(config["quality_frontier"]["match_tolerance"])
    matches = [
        row["solver_id"]
        for row in quality
        if row["solver_family"] != "ppo"
        and row["mean_relative_gap"] <= frozen_quality["mean_relative_gap"] + tolerance
        and row["p90_relative_gap"] <= frozen_quality["p90_relative_gap"] + tolerance
    ]
    summary = {
        "status": "completed",
        "quality_frontier": list(quality),
        "frozen_ppo_quality": frozen_quality,
        "quality_matched_baselines": matches,
        "ppo_service": list(ppo_service),
        "baseline_service": list(baseline_service),
        "interpretation": {
            "development_only": True,
            "teacher_only_limitation_remains": True,
            "no_assumed_ppo_advantage": True,
        },
    }
    _write_json_atomic(output_dir / "service_efficiency_summary.json", summary)
    lines = [
        "# 四拓扑批量服务效率实验",
        "",
        "- 状态：完成。",
        "- PPO：Stage6 冻结主线 update60，legacy_separate_v1。",
        "- 质量匹配基线：" + (", ".join(matches) if matches else "无；仅报告质量—时间前沿"),
        "- 数据解释：开发集效率实验，不是独立测试。",
        "- 已知局限：teacher-only 零响应仍然存在，本实验不声称该问题已解决。",
        "",
        "## 质量前沿",
        "",
        "| 方法 | mean gap | P90 gap | mean latency(s) |",
        "|---|---:|---:|---:|",
    ]
    for row in quality:
        lines.append(
            f"| {row['solver_id']} | {row['mean_relative_gap']:.4%} | "
            f"{row['p90_relative_gap']:.4%} | {row['latency']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## PPO 服务吞吐",
            "",
            "| 预算 | 负载 | batch | req/s | P95 request(s) | padding mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in ppo_service:
        lines.append(
            f"| {row['solver_id']} | {row['workload_id']} | {row['requested_batch_size']} | "
            f"{row['throughput_requests_per_sec']:.3f} | {row['request_latency']['p95']:.6f} | "
            f"{row['mean_padding_ratio']:.3f} |"
        )
    (output_dir / "service_efficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    *,
    smoke: bool = False,
    validate_only: bool = False,
    cold_probe: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    if config.get("status") not in {"user_authorized", "user_authorized_protocol_correction"}:
        raise ValueError("Service-efficiency protocol is not user-authorized.")
    if config.get("engine_version") != SERVICE_ENGINE_VERSION:
        raise ValueError("Unknown service engine version.")
    frozen, objective = _validate_frozen_config(config)
    context = _load_context(config, objective)
    first_row = context["core_rows"][0]
    first_scenario = next(iter(context["controlled_scenarios"].values()))
    first_instance = _condition(context, first_row["original_id"], first_scenario)
    requested_device = str(config["measurement"]["device"])
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage7 requires the registered CUDA device.")
    if bool(config["measurement"]["deterministic"]):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    model, ppo_config, source_metadata, model_load_sec = _load_model_bundle(
        config=config,
        frozen=frozen,
        objective=objective,
        first_instance=first_instance,
        device=device,
    )
    validation = {
        "status": "validated_no_experiment" if validate_only else "validated",
        "runner_version": RUNNER_VERSION,
        "engine_version": SERVICE_ENGINE_VERSION,
        "config_sha256": _sha256(config_path),
        "frozen_config_sha256": config["frozen_ppo"]["config_sha256"],
        "objective_hash": objective.config_hash,
        "checkpoint_sha256": config["frozen_ppo"]["checkpoint_sha256"],
        "checkpoint_update": int(config["frozen_ppo"]["expected_update"]),
        "device": str(device),
        "model_load_sec": model_load_sec,
        "source_data_hash": source_metadata["data_hash"],
        "frozen_inference": {
            "max_steps": ppo_config.env_max_steps,
            "patience": ppo_config.env_patience,
        },
    }
    if validate_only:
        return validation

    if cold_probe:
        request = _quality_requests(context, 1)[0]
        scenario = context["controlled_scenarios"][request["scenario_id"]]
        request_ids, instances, prepare_sec = _prepare_batch(
            context,
            [("cold-start-000", request["original_id"], scenario)],
        )
        frozen_budget = next(
            item for item in config["quality_frontier"]["ppo_budgets"]
            if item["id"] == "ppo_512_32_frozen"
        )
        result = run_batched_policy_service(
            request_ids=request_ids,
            instances=instances,
            model=model,
            project_config=context["project_config"],
            objective=context["objective"],
            max_steps=int(frozen_budget["max_steps"]),
            patience=int(frozen_budget["patience"]),
            requested_batch_size=1,
        )
        return {
            **validation,
            "status": "cold_probe_completed",
            "prepare_latency_sec": prepare_sec,
            "first_inference_latency_sec": result.batch_latency_sec,
            "cold_start_total_sec": time.perf_counter() - run_started,
            "request": _compact_request(result.requests[0].to_dict()),
        }

    root = Path(config["run"]["output_dir"] + ("_smoke" if smoke else "")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "run_metadata.json"
    metadata = {
        **validation,
        "status": "running",
        "mode": "smoke" if smoke else "formal",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "output_dir": str(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "no_training": True,
    }
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "objective_hash", "checkpoint_sha256", "mode"):
            if previous.get(key) != metadata.get(key):
                raise ValueError(f"Existing Stage7 output mismatch: {key}")
    _write_json_atomic(metadata_path, metadata)

    quality_count = 4 if smoke else int(config["quality_frontier"]["stratified_request_count"])
    quality_requests = _quality_requests(context, quality_count)
    preflight_config = config
    if smoke:
        preflight_config = dict(config)
        preflight_config["preflight"] = dict(config["preflight"])
        preflight_config["preflight"]["max_steps"] = 32
        preflight_config["preflight"]["patience"] = 16
    preflight = _run_preflight(
        config=preflight_config,
        context=context,
        model=model,
        quality_requests=quality_requests,
        output_dir=root,
    )
    quality = _run_quality_frontier(
        config=config,
        context=context,
        model=model,
        quality_requests=quality_requests,
        output_dir=root,
        smoke=smoke,
    )
    ppo_service = _run_ppo_service_cells(
        config=config,
        context=context,
        model=model,
        output_dir=root,
        smoke=smoke,
    )
    baseline_service = _run_baseline_service_cells(
        config=config,
        context=context,
        output_dir=root,
        smoke=smoke,
    )
    if not smoke:
        _write_report(
            output_dir=root,
            quality=quality,
            ppo_service=ppo_service,
            baseline_service=baseline_service,
            config=config,
        )
    metadata.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preflight_status": preflight["status"],
            "quality_configuration_count": len(quality),
            "ppo_service_cell_count": len(ppo_service),
            "baseline_service_cell_count": len(baseline_service),
        }
    )
    _write_json_atomic(metadata_path, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--cold-probe", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config,
        smoke=args.smoke_test,
        validate_only=args.validate_only,
        cold_probe=args.cold_probe,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
