"""Audit and finalize the corrected Stage7 service-efficiency run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import yaml

from pa_moap_rl.data.scenarios import load_scenario_bank


FINALIZER_VERSION = "service_efficiency_finalizer_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["run"]["output_dir"]).resolve()
    metadata = _read_json(output_dir / "run_metadata.json")
    if metadata.get("status") != "completed" or metadata.get("mode") != "formal":
        raise ValueError("Formal Stage7 run is not complete.")
    summary = _read_json(output_dir / "service_efficiency_summary.json")
    cold = _read_json(output_dir / "cold_start_measurements.json")
    quality = summary["quality_frontier"]
    ppo_service = summary["ppo_service"]
    baseline_service = summary["baseline_service"]
    if (len(quality), len(ppo_service), len(baseline_service)) != (10, 24, 12):
        raise ValueError("Unexpected Stage7 configuration counts.")
    if int(cold["repetitions"]) != int(config["measurement"]["cold_start_repetitions"]):
        raise ValueError("Cold-start repetition count mismatch.")

    scenarios = load_scenario_bank(config["data"]["service_scenarios"])
    profile_pairs = {
        (
            tuple(float(value) for value in item.student_profile),
            tuple(float(value) for value in item.teacher_profile),
        )
        for item in scenarios
    }
    if len(scenarios) != 128 or len(profile_pairs) != 128:
        raise ValueError("Corrected service bank must contain 128 unique profiles.")

    hard_feasible = True
    mask_violations = 0
    score_rows = 0
    for path in sorted((output_dir / "quality").glob("*.json")):
        payload = _read_json(path)
        for row in payload["requests"]:
            score_rows += 1
            hard_feasible = hard_feasible and bool(row["hard_feasible"])
            mask_violations += int(row["mask_violation_count"])
    service_rows = 0
    cuda_peaks: list[int] = []
    for path in sorted((output_dir / "service" / "ppo").rglob("batch_*.json")):
        payload = _read_json(path)
        cell = payload["summary"]
        expected = int(cell["requested_batch_size"]) * int(cell["measured_batch_count"])
        if int(cell["request_count"]) != expected:
            raise ValueError(f"PPO service request count mismatch: {path}")
        for batch in payload["batches"]:
            if int(batch["actual_batch_size"]) != int(batch["requested_batch_size"]):
                raise ValueError(f"PPO actual batch mismatch: {path}")
            for row in batch["requests"]:
                service_rows += 1
                hard_feasible = hard_feasible and bool(row["hard_feasible"])
                mask_violations += int(row["mask_violation_count"])
        peak = payload["resource_after"].get("cuda_peak_allocated_bytes")
        if peak is not None:
            cuda_peaks.append(int(peak))
    for path in sorted((output_dir / "service" / "baselines").rglob("*.json")):
        payload = _read_json(path)
        if int(payload["summary"]["request_count"]) != int(config["baseline_service"]["request_count_per_workload"]):
            raise ValueError(f"Baseline request count mismatch: {path}")
        for row in payload["requests"]:
            service_rows += 1
            hard_feasible = hard_feasible and bool(row["hard_feasible"])
            mask_violations += int(row["mask_violation_count"])
    if not hard_feasible or mask_violations != 0:
        raise ValueError("Feasibility audit failed.")

    cold_rows = cold["measurements"]
    cold_summary = {
        "repetitions": len(cold_rows),
        "subprocess_wall_mean_sec": _mean([float(row["subprocess_wall_sec"]) for row in cold_rows]),
        "in_process_total_mean_sec": _mean([float(row["probe"]["cold_start_total_sec"]) for row in cold_rows]),
        "model_load_mean_sec": _mean([float(row["probe"]["model_load_sec"]) for row in cold_rows]),
        "first_inference_mean_sec": _mean([float(row["probe"]["first_inference_latency_sec"]) for row in cold_rows]),
    }
    training_summary_path = Path(
        "results/four_topology_b075_seed0_pilot_stabilization_run03/run_summary.json"
    )
    training_summary = _read_json(training_summary_path)
    training_sec = float(training_summary["training_elapsed_sec"])
    amortization = {
        "training_elapsed_sec": training_sec,
        "reported_separately_from_online_latency": True,
        "per_request_training_sec": {
            str(volume): training_sec / volume for volume in (1_000, 10_000, 100_000, 1_000_000)
        },
        "source": str(training_summary_path),
        "source_sha256": _sha256(training_summary_path),
    }

    frozen_quality = next(row for row in quality if row["solver_id"] == "ppo_512_32_frozen")
    scalar_quality = next(row for row in quality if row["solver_id"] == "scalarized_greedy")
    best_ppo = sorted(ppo_service, key=lambda row: float(row["throughput_requests_per_sec"]), reverse=True)[0]
    scalar_cells = [row for row in baseline_service if row["solver_id"] == "scalarized_greedy"]
    audit = {
        "schema_version": 1,
        "finalizer_version": FINALIZER_VERSION,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run01_invalidated_notice_present": (
            Path("results/four_topology_b075_service_efficiency_v1/INVALIDATED_PROTOCOL_DEVIATION.md").is_file()
        ),
        "service_scenario_count": len(scenarios),
        "service_profile_unique_count": len(profile_pairs),
        "quality_configuration_count": len(quality),
        "ppo_service_cell_count": len(ppo_service),
        "baseline_service_cell_count": len(baseline_service),
        "quality_request_rows": score_rows,
        "service_request_rows": service_rows,
        "all_hard_feasible": hard_feasible,
        "mask_violation_count": mask_violations,
        "max_cuda_peak_allocated_bytes": max(cuda_peaks) if cuda_peaks else None,
        "cold_start": cold_summary,
        "training_amortization": amortization,
        "frozen_ppo_quality": frozen_quality,
        "scalarized_greedy_quality": scalar_quality,
        "best_ppo_service_cell": best_ppo,
        "scalarized_greedy_service_cells": scalar_cells,
        "known_limitation": "teacher-only zero response remains unresolved",
        "development_only": True,
    }
    _write_json(output_dir / "final_audit_summary.json", audit)
    _write_json(output_dir / "training_amortization.json", amortization)

    lines = [
        "# 四拓扑批量服务效率实验（run02）",
        "",
        "- 状态：正式运行完成，协议修正审计通过。",
        "- 范围：现有开发数据上的效率实验，不是独立测试。",
        "- 模型：冻结主线 `legacy_separate_v1`，run03 update60，确定性 512/32。",
        "- 数据：同内容多偏好使用 128 个批内唯一、目标无关的服务画像。",
        "- 已知局限：teacher-only 零响应仍未解决，本实验不改变该结论。",
        "",
        "## 质量前沿",
        "",
        "| 方法 | mean gap | P90 gap | 平均延迟（秒） |",
        "|---|---:|---:|---:|",
    ]
    for row in quality:
        lines.append(
            f"| {row['solver_id']} | {float(row['mean_relative_gap']):.4%} | "
            f"{float(row['p90_relative_gap']):.4%} | {float(row['latency']['mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## PPO 批量服务吞吐",
            "",
            "| 预算 | 负载 | batch | 请求/秒 | P95 请求延迟（秒） | 平均 padding |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in ppo_service:
        lines.append(
            f"| {row['solver_id']} | {row['workload_id']} | {row['requested_batch_size']} | "
            f"{float(row['throughput_requests_per_sec']):.3f} | "
            f"{float(row['request_latency']['p95']):.6f} | {float(row['mean_padding_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 冷启动、资源与训练摊销",
            "",
            f"- 独立进程冷启动均值：{cold_summary['subprocess_wall_mean_sec']:.3f} 秒。",
            f"- 进程内加载至首请求完成均值：{cold_summary['in_process_total_mean_sec']:.3f} 秒。",
            f"- 首次推理均值：{cold_summary['first_inference_mean_sec']:.3f} 秒。",
            f"- CUDA 峰值已分配显存：{(max(cuda_peaks) / 1024**2 if cuda_peaks else 0):.1f} MiB。",
            f"- seed0 训练耗时单独报告为 {training_sec:.3f} 秒；不计入在线延迟。",
            "",
            "## 解释边界",
            "",
            "质量匹配和服务吞吐必须联合解读；不预设 PPO 优于 greedy、局部搜索或 Gurobi。",
            "run01 仅保留作机械执行记录，因画像复用偏差不得作为正式证据。",
        ]
    )
    report_path = output_dir / "service_efficiency_report_utf8.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    code_and_inputs = [
        config_path,
        Path(config["frozen_ppo"]["config"]),
        Path(config["frozen_ppo"]["checkpoint"]),
        Path(config["objective"]["config"]),
        Path(config["data"]["service_scenarios"]),
        Path("pa_moap_rl/experiments/service_efficiency_v1.py"),
        Path("pa_moap_rl/experiments/run_service_efficiency_v1.py"),
        Path("pa_moap_rl/experiments/run_service_cold_start_repetitions_v1.py"),
        Path("pa_moap_rl/experiments/finalize_service_efficiency_v1.py"),
    ]
    output_files = [
        path for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "evidence_hashes.json"
    ]
    hashes = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_and_inputs": {str(path): _sha256(path) for path in code_and_inputs},
        "outputs": {str(path): _sha256(path) for path in output_files},
    }
    _write_json(output_dir / "evidence_hashes.json", hashes)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/service_efficiency_benchmark_four_topology_v1_run02.yaml",
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
