"""Apply confirmed gates to A/B/C pure-PPO controlled-cross results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ARM_ORDER = ("A", "B", "C")
SCENARIOS = (
    "controlled_student_only",
    "controlled_teacher_only",
    "controlled_conflict",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _cluster_ci(
    rows: list[dict[str, str]], *, seed: int, repetitions: int
) -> list[float]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        clusters.setdefault(row["source_family_id"], []).append(
            float(row["reoptimization_gain"])
        )
    cluster_means = np.asarray(
        [np.mean(clusters[key]) for key in sorted(clusters)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, cluster_means.size, size=(repetitions, cluster_means.size)
    )
    values = np.mean(cluster_means[indices], axis=1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _fixed_72(run_dir: Path, update: int) -> dict[str, Any]:
    rows = _read_csv(run_dir / "metrics" / "validation_updates.csv")
    selected = [
        row
        for row in rows
        if int(row["update"]) == update
        and row["scope"] == "overall"
        and row["scope_value"] == "all"
    ]
    if len(selected) != 1:
        raise ValueError(f"Ambiguous fixed-72 summary for {run_dir} update {update}.")
    row = selected[0]
    return {
        "count": int(row["count"]),
        "mean_total_score": float(row["total_score_mean"]),
        "mean_relative_gap": float(row["relative_gap_to_exact_mean"]),
        "p90_relative_gap": float(row["relative_gap_to_exact_p90"]),
        "hard_feasible_rate": float(row["hard_feasible_rate_mean"]),
        "mask_violation_count": int(float(row["mask_violation_count_sum"])),
    }


def summarize(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "protocol_confirmed_post_training_evaluation":
        raise ValueError("Evaluation configuration is not confirmed.")
    protocol_path = Path(config["protocol"]["path"])
    if _sha256(protocol_path) != config["protocol"]["sha256"]:
        raise ValueError("Protocol hash mismatch while summarizing.")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    gates = protocol["acceptance_gates"]
    output_dir = Path(config["run"]["output_dir"]).resolve()
    arm_summaries: dict[str, Any] = {}
    gate_rows: dict[str, Any] = {}
    eligible: list[str] = []
    for arm_id in ARM_ORDER:
        arm_cfg = config["arms"][arm_id]
        arm_dir = output_dir / f"arm_{arm_id}"
        candidate = json.loads(
            (arm_dir / "candidate_response_summary.json").read_text(encoding="utf-8")
        )
        if (
            candidate.get("status") != "completed_not_frozen"
            or candidate.get("checkpoint_sha256") != arm_cfg["checkpoint_sha256"]
            or int(candidate.get("checkpoint_update", -1)) != int(arm_cfg["expected_update"])
            or int(candidate.get("pair_count", 0)) != 288
        ):
            raise ValueError(f"Arm {arm_id} candidate summary mismatch.")
        rows = _read_csv(arm_dir / "response_evidence.csv")
        if len(rows) != 216:
            raise ValueError(f"Arm {arm_id} must have 216 non-baseline response rows.")
        response: dict[str, Any] = {}
        for scenario_id in SCENARIOS:
            selected = [row for row in rows if row["scenario_id"] == scenario_id]
            gains = [float(row["reoptimization_gain"]) for row in selected]
            stable = int(config["bootstrap"]["seed"]) + int(
                hashlib.sha256(f"{arm_id}:{scenario_id}".encode("ascii")).hexdigest()[:8],
                16,
            )
            response[scenario_id] = {
                "count": len(selected),
                "source_family_count": len({row["source_family_id"] for row in selected}),
                "positive_gain_rate": float(np.mean(np.asarray(gains) > 1.0e-9)),
                "reoptimization_gain": _stats(gains),
                "cluster_bootstrap_mean_ci": _cluster_ci(
                    selected,
                    seed=stable,
                    repetitions=int(config["bootstrap"]["repetitions"]),
                ),
                "assignment_change_rate": _stats(
                    [float(row["assignment_change_rate"]) for row in selected]
                ),
                "relative_gap_to_exact": _stats(
                    [float(row["relative_gap_to_exact"]) for row in selected]
                ),
            }
        fixed = _fixed_72(
            Path(arm_cfg["run_dir"]), int(arm_cfg["expected_update"])
        )
        arm_summary = {
            "checkpoint_update": int(arm_cfg["expected_update"]),
            "checkpoint_sha256": arm_cfg["checkpoint_sha256"],
            "fixed_72": fixed,
            "controlled_cross": {
                "mean_relative_gap": float(candidate["mean_relative_gap"]),
                "p90_relative_gap": float(candidate["p90_relative_gap"]),
            },
            "all_hard_feasible": bool(candidate["all_hard_feasible"]),
            "max_score_recomputation_error": float(
                candidate["max_score_recomputation_error"]
            ),
            "response": response,
        }
        teacher = response["controlled_teacher_only"]
        student = response["controlled_student_only"]
        conflict = response["controlled_conflict"]
        checks = {
            "engineering_feasible": arm_summary["all_hard_feasible"],
            "engineering_recomputation": arm_summary["max_score_recomputation_error"]
            <= float(gates["engineering"]["max_score_recomputation_error"]),
            "fixed_72_mean_gap": fixed["mean_relative_gap"]
            <= float(gates["fixed_72_quality"]["max_mean_exact_relative_gap"]),
            "fixed_72_p90_gap": fixed["p90_relative_gap"]
            <= float(gates["fixed_72_quality"]["max_p90_exact_relative_gap"]),
            "teacher_positive_rate": teacher["positive_gain_rate"]
            >= float(gates["pure_ppo_response"]["min_teacher_only_positive_gain_rate"]),
            "teacher_mean_gain": teacher["reoptimization_gain"]["mean"]
            >= float(gates["pure_ppo_response"]["min_teacher_only_mean_reoptimization_gain"]),
            "teacher_bootstrap_lower_positive": teacher["cluster_bootstrap_mean_ci"][0] > 0.0,
            "student_positive_rate": student["positive_gain_rate"]
            >= float(gates["pure_ppo_response"]["min_student_only_positive_gain_rate"]),
            "conflict_positive_rate": conflict["positive_gain_rate"]
            >= float(gates["pure_ppo_response"]["min_conflict_positive_gain_rate"]),
        }
        passed = all(checks.values())
        if passed:
            eligible.append(arm_id)
        arm_summaries[arm_id] = arm_summary
        gate_rows[arm_id] = {"passed": passed, "checks": checks}
    complexity = {"A": 0, "B": 1, "C": 2}
    ranking = sorted(
        eligible,
        key=lambda arm_id: (
            -arm_summaries[arm_id]["response"]["controlled_teacher_only"]["reoptimization_gain"]["mean"],
            -arm_summaries[arm_id]["fixed_72"]["mean_total_score"],
            arm_summaries[arm_id]["fixed_72"]["p90_relative_gap"],
            complexity[arm_id],
        ),
    )
    summary = {
        "schema_version": 1,
        "status": "completed_not_frozen",
        "config_sha256": _sha256(path),
        "protocol_sha256": config["protocol"]["sha256"],
        "arms": arm_summaries,
        "gate_decision": {
            "arms": gate_rows,
            "eligible_arms": eligible,
            "ranking_among_eligible": ranking,
            "recommended_arm": ranking[0] if ranking else None,
        },
        "interpretation_limits": [
            "controlled calibration data already used during development",
            "single seed0 checkpoint per arm",
            "pure PPO only; no local-search or hybrid post-processing",
            "recommendation does not freeze PPO or inference parameters",
        ],
    }
    _write_json(output_dir / "ablation_evaluation_summary.json", summary)
    lines = [
        "# A/B/C pure-PPO controlled-cross decision",
        "",
        "| Arm | Fixed72 mean gap | P90 gap | Teacher +rate | Teacher mean | CI low | Student +rate | Conflict +rate | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm_id in ARM_ORDER:
        item = arm_summaries[arm_id]
        teacher = item["response"]["controlled_teacher_only"]
        lines.append(
            f"| {arm_id} | {100*item['fixed_72']['mean_relative_gap']:.3f}% | "
            f"{100*item['fixed_72']['p90_relative_gap']:.3f}% | "
            f"{100*teacher['positive_gain_rate']:.1f}% | "
            f"{teacher['reoptimization_gain']['mean']:.8f} | "
            f"{teacher['cluster_bootstrap_mean_ci'][0]:.8f} | "
            f"{100*item['response']['controlled_student_only']['positive_gain_rate']:.1f}% | "
            f"{100*item['response']['controlled_conflict']['positive_gain_rate']:.1f}% | "
            f"{'PASS' if gate_rows[arm_id]['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Eligible arms: {eligible}",
            f"Recommended arm under confirmed rule: {ranking[0] if ranking else None}",
            "",
            "This is exploratory controlled-calibration evidence and does not freeze parameters.",
        ]
    )
    (output_dir / "ablation_evaluation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/preference_repair_ablation_evaluation_v2.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
