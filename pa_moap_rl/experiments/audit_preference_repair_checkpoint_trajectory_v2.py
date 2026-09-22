"""Audit teacher-only response across every A/B/C checkpoint without training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from pa_moap_rl.checkpointing import method_matrix_hash, validate_checkpoint_metadata
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.diagnose_teacher_response_v2 import _candidate_rewards
from pa_moap_rl.experiments.run_preference_response_v2 import (
    _assignment_hash,
    _load_and_validate as load_preference_context,
)
from pa_moap_rl.experiments.train_ppo_batch import _policy_rollout
from pa_moap_rl.models.preference_repair_v2 import build_versioned_actor_critic
from pa_moap_rl.solvers.ppo_solver import (
    PPOConfig,
    observation_to_model_batch,
    resolve_device,
)
from pa_moap_rl.utils.scoring import score_assignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
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


def _source_data_hash(metadata: dict[str, Any]) -> str:
    input_hashes: list[str] = []
    for item in metadata["inputs"].values():
        path = Path(item["path"])
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Training input drift: {path}")
        input_hashes.append(actual)
    result = hashlib.sha256("".join(input_hashes).encode("ascii")).hexdigest()
    if result != metadata["data_hash"]:
        raise ValueError("Training run data hash mismatch.")
    return result


def _full_policy_vector(
    model: Any, observation: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(observation_to_model_batch(observation, device=device))
    mask = observation["action_mask"].reshape(-1).astype(bool)
    logits = output.masked_logits.reshape(-1).detach().cpu().numpy()
    legal = np.flatnonzero(mask)
    legal_logits = logits[legal]
    probabilities = np.exp(legal_logits - np.max(legal_logits))
    probabilities /= np.sum(probabilities)
    return logits, legal, probabilities


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 1.0e-15 or np.std(right) <= 1.0e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _screen_metrics(
    *,
    model: Any,
    balanced_observation: dict[str, np.ndarray],
    teacher_observation: dict[str, np.ndarray],
    teacher_instance: Any,
    objective: Any,
    tolerance: float,
    top_k: list[int],
) -> dict[str, Any]:
    assignment = np.asarray(teacher_observation["assignment"], dtype=np.int64)
    if not np.array_equal(assignment, balanced_observation["assignment"]):
        raise RuntimeError("Controlled profiles have different initial assignments.")
    balanced_logits, balanced_legal, balanced_prob = _full_policy_vector(
        model, balanced_observation
    )
    teacher_logits, teacher_legal, teacher_prob = _full_policy_vector(
        model, teacher_observation
    )
    if not np.array_equal(balanced_legal, teacher_legal):
        raise RuntimeError("Controlled profiles have different legal action masks.")

    rewards = _candidate_rewards(
        teacher_instance, assignment, objective
    ).reshape(-1)
    legal_rewards = rewards[teacher_legal]
    order = np.argsort(-teacher_logits[teacher_legal], kind="stable")
    ranked_flat = teacher_legal[order]
    ranked_rewards = rewards[ranked_flat]
    chosen_flat = int(ranked_flat[0])
    best_flat = int(teacher_legal[int(np.argmax(legal_rewards))])
    best_rank = int(np.flatnonzero(ranked_flat == best_flat)[0]) + 1
    positive_positions = np.flatnonzero(ranked_rewards > tolerance)
    first_positive_rank = (
        int(positive_positions[0]) + 1 if positive_positions.size else len(ranked_flat) + 1
    )
    result: dict[str, Any] = {
        "legal_action_count": int(teacher_legal.size),
        "positive_action_count": int(np.sum(legal_rewards > tolerance)),
        "positive_action_rate": float(np.mean(legal_rewards > tolerance)),
        "policy_top_action_reward": float(rewards[chosen_flat]),
        "policy_top_action_positive": bool(rewards[chosen_flat] > tolerance),
        "true_best_action_reward": float(rewards[best_flat]),
        "true_best_action_policy_rank": best_rank,
        "first_positive_action_policy_rank": first_positive_rank,
        "reward_logit_pearson": _safe_pearson(
            teacher_logits[teacher_legal], legal_rewards
        ),
        "teacher_logit_change_mean": float(
            np.mean(
                np.abs(
                    teacher_logits[teacher_legal] - balanced_logits[balanced_legal]
                )
            )
        ),
        "teacher_probability_tv": float(0.5 * np.sum(np.abs(teacher_prob - balanced_prob))),
        "teacher_changes_top_action": bool(
            np.argmax(teacher_logits) != np.argmax(balanced_logits)
        ),
    }
    for value in top_k:
        bounded = min(int(value), ranked_rewards.size)
        result[f"top{value}_contains_positive"] = bool(
            np.any(ranked_rewards[:bounded] > tolerance)
        )
        result[f"true_best_in_top{value}"] = bool(best_rank <= value)
    return result


def _summarize_screen(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm_id, update in sorted({(row["arm_id"], row["update"]) for row in rows}):
        selected = [
            row for row in rows if row["arm_id"] == arm_id and row["update"] == update
        ]
        summary: dict[str, Any] = {
            "arm_id": arm_id,
            "update": int(update),
            "checkpoint_sha256": selected[0]["checkpoint_sha256"],
            "model_version": selected[0]["model_version"],
            "instance_count": len(selected),
            "top_action_positive_rate": float(
                np.mean([row["policy_top_action_positive"] for row in selected])
            ),
            "teacher_changes_top_action_rate": float(
                np.mean([row["teacher_changes_top_action"] for row in selected])
            ),
        }
        for field in (
            "positive_action_rate",
            "policy_top_action_reward",
            "true_best_action_reward",
            "true_best_action_policy_rank",
            "first_positive_action_policy_rank",
            "reward_logit_pearson",
            "teacher_logit_change_mean",
            "teacher_probability_tv",
        ):
            for key, value in _stats([float(row[field]) for row in selected]).items():
                summary[f"{field}_{key}"] = value
        for key in sorted(selected[0]):
            if key.startswith("top") and key.endswith("contains_positive"):
                summary[f"{key}_rate"] = float(
                    np.mean([bool(row[key]) for row in selected])
                )
            if key.startswith("true_best_in_top"):
                summary[f"{key}_rate"] = float(
                    np.mean([bool(row[key]) for row in selected])
                )
        summaries.append(summary)
    return summaries


def select_followup_checkpoints(
    summaries: list[dict[str, Any]],
    selected_updates: dict[str, int],
) -> list[dict[str, Any]]:
    """Apply the registered deterministic at-most-three-per-arm rule."""

    chosen: list[dict[str, Any]] = []
    for arm_id in sorted(selected_updates):
        rows = [row for row in summaries if row["arm_id"] == arm_id]
        if not rows:
            raise ValueError(f"No screen rows for arm {arm_id}.")
        by_update = {int(row["update"]): row for row in rows}
        selected_update = int(selected_updates[arm_id])
        if selected_update not in by_update:
            raise ValueError(f"Selected update {selected_update} missing for arm {arm_id}.")
        reasons: dict[int, set[str]] = {selected_update: {"protocol_selected"}}
        max_positive = min(
            rows,
            key=lambda row: (
                -float(row["top_action_positive_rate"]),
                float(row["true_best_action_policy_rank_median"]),
                int(row["update"]),
            ),
        )
        reasons.setdefault(int(max_positive["update"]), set()).add(
            "max_top_action_positive_rate"
        )
        min_rank = min(
            rows,
            key=lambda row: (
                float(row["true_best_action_policy_rank_median"]),
                -float(row["top_action_positive_rate"]),
                int(row["update"]),
            ),
        )
        reasons.setdefault(int(min_rank["update"]), set()).add(
            "min_median_true_best_rank"
        )
        for update in sorted(reasons):
            chosen.append(
                {
                    **by_update[update],
                    "selection_reasons": ";".join(sorted(reasons[update])),
                }
            )
    return chosen


def _load_model(
    *,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    first_instance: Any,
    objective: Any,
    project_config: Any,
    device: torch.device,
) -> tuple[Any, dict[str, Any], str]:
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data_hash = _source_data_hash(metadata)
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_matrix_hash(first_instance),
        data_hash=data_hash,
    )
    experiment_spec = checkpoint.get("experiment_spec")
    if experiment_spec is None or experiment_spec != metadata.get("experiment_spec"):
        raise ValueError(f"Checkpoint experiment_spec mismatch: {checkpoint_path}")
    model = build_versioned_actor_critic(
        first_instance,
        model_version=experiment_spec["model"]["model_version"],
        config=project_config,
        device=device,
        hidden_dim=checkpoint.get("hidden_dim"),
        objective=objective,
    )
    model.experiment_spec = experiment_spec
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, checkpoint_hash


def _screen(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    output_dir: Path,
    project_config: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objective = context["objective"]
    baseline_id = config["controlled_profiles"]["baseline_scenario_id"]
    teacher_id = config["controlled_profiles"]["teacher_scenario_id"]
    device = resolve_device(config["screen"]["device"])
    first_row = next(iter(context["core"].values()))
    first_instance = load_instance_json(Path(first_row["output_json"]))
    cached: list[tuple[str, dict[str, str], Any, Any, dict[str, Any], dict[str, Any]]] = []
    for original_id, core_row in context["core"].items():
        base = load_instance_json(Path(core_row["output_json"]))
        balanced = apply_scenario(
            base, context["scenarios"][baseline_id], project_config, objective=objective
        )
        teacher = apply_scenario(
            base, context["scenarios"][teacher_id], project_config, objective=objective
        )
        balanced_obs = MethodAssignmentEnv(
            max_steps=1, patience=1, config=project_config, objective=objective
        ).reset(balanced)
        teacher_obs = MethodAssignmentEnv(
            max_steps=1, patience=1, config=project_config, objective=objective
        ).reset(teacher)
        cached.append(
            (original_id, core_row, balanced, teacher, balanced_obs, teacher_obs)
        )

    rows: list[dict[str, Any]] = []
    total = len(config["arms"]) * len(config["screen"]["updates"])
    completed = 0
    for arm_id, arm in config["arms"].items():
        run_dir = Path(arm["run_dir"])
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata["experiment_spec"]["arm_id"] != arm_id:
            raise ValueError(f"Arm metadata mismatch for {arm_id}.")
        for update in config["screen"]["updates"]:
            checkpoint_path = run_dir / "checkpoints" / f"update_{int(update):06d}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            model, checkpoint, checkpoint_hash = _load_model(
                checkpoint_path=checkpoint_path,
                metadata=metadata,
                first_instance=first_instance,
                objective=objective,
                project_config=project_config,
                device=device,
            )
            if int(checkpoint["update"]) != int(update):
                raise ValueError(f"Checkpoint update mismatch: {checkpoint_path}")
            model_version = checkpoint["experiment_spec"]["model"]["model_version"]
            for original_id, core_row, _, teacher, balanced_obs, teacher_obs in cached:
                rows.append(
                    {
                        "arm_id": arm_id,
                        "update": int(update),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_hash,
                        "model_version": model_version,
                        "original_id": original_id,
                        "source_family_id": core_row["source_family_id"],
                        "topology": core_row["topology"],
                        "size_group": core_row["size_group"],
                        "n": teacher.n,
                        **_screen_metrics(
                            model=model,
                            balanced_observation=balanced_obs,
                            teacher_observation=teacher_obs,
                            teacher_instance=teacher,
                            objective=objective,
                            tolerance=float(config["screen"]["positive_gain_tolerance"]),
                            top_k=[int(value) for value in config["screen"]["top_k"]],
                        ),
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            completed += 1
            print(f"screened {completed}/{total}: {arm_id} update {update}", flush=True)
    summaries = _summarize_screen(rows)
    _write_csv(output_dir / "screen_rows.csv", rows)
    _write_csv(output_dir / "checkpoint_screen_summary.csv", summaries)
    return rows, summaries


def _exact_score(context: dict[str, Any], original_id: str, scenario_id: str) -> float:
    pair = next(
        row
        for row in context["pairs"]
        if row["original_id"] == original_id and row["scenario_id"] == scenario_id
    )
    path = context["exact_dir"] / f"{pair['pair_id']}.json"
    if _sha256(path) != context["exact_hashes"][path.name]:
        raise ValueError(f"Exact result hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["score"]["total_score"])


def _full_followup(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    selected: list[dict[str, Any]],
    output_dir: Path,
    project_config: Any,
) -> list[dict[str, Any]]:
    objective = context["objective"]
    baseline_id = config["controlled_profiles"]["baseline_scenario_id"]
    teacher_id = config["controlled_profiles"]["teacher_scenario_id"]
    device = resolve_device(config["followup"]["device"])
    first_row = next(iter(context["core"].values()))
    first_instance = load_instance_json(Path(first_row["output_json"]))
    config_hash = _sha256(Path(config["_config_path"]))
    summaries: list[dict[str, Any]] = []
    for selected_index, selection in enumerate(selected, start=1):
        arm_id = selection["arm_id"]
        update = int(selection["update"])
        run_dir = Path(config["arms"][arm_id]["run_dir"])
        checkpoint_path = run_dir / "checkpoints" / f"update_{update:06d}.pt"
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        model, checkpoint, checkpoint_hash = _load_model(
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            first_instance=first_instance,
            objective=objective,
            project_config=project_config,
            device=device,
        )
        ppo_config = replace(
            PPOConfig(**checkpoint["ppo_config"]),
            device=str(device),
            env_max_steps=int(config["followup"]["max_steps"]),
            env_patience=int(config["followup"]["patience"]),
        )
        candidate_dir = output_dir / "full_response" / f"{arm_id}_update_{update:06d}"
        pair_dir = candidate_dir / "instances"
        pair_dir.mkdir(parents=True, exist_ok=True)
        instance_rows: list[dict[str, Any]] = []
        for instance_index, (original_id, core_row) in enumerate(
            context["core"].items(), start=1
        ):
            path = pair_dir / f"{original_id}.json"
            if path.is_file() and config["run"].get("resume", False):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("config_sha256") != config_hash
                    or payload.get("checkpoint_sha256") != checkpoint_hash
                    or payload.get("original_id") != original_id
                ):
                    raise ValueError(f"Resume artifact mismatch: {path}")
            else:
                base = load_instance_json(Path(core_row["output_json"]))
                balanced = apply_scenario(
                    base,
                    context["scenarios"][baseline_id],
                    project_config,
                    objective=objective,
                )
                teacher = apply_scenario(
                    base,
                    context["scenarios"][teacher_id],
                    project_config,
                    objective=objective,
                )
                balanced_result = _policy_rollout(
                    balanced,
                    model=model,
                    project_config=project_config,
                    ppo_config=ppo_config,
                    deterministic=True,
                    objective=objective,
                )
                teacher_result = _policy_rollout(
                    teacher,
                    model=model,
                    project_config=project_config,
                    ppo_config=ppo_config,
                    deterministic=True,
                    objective=objective,
                )
                balanced_recomputed = score_assignment(
                    balanced_result.assignment, instance=balanced, objective=objective
                )
                teacher_recomputed = score_assignment(
                    teacher_result.assignment, instance=teacher, objective=objective
                )
                no_reopt = score_assignment(
                    balanced_result.assignment, instance=teacher, objective=objective
                )
                balanced_exact = _exact_score(context, original_id, baseline_id)
                teacher_exact = _exact_score(context, original_id, teacher_id)
                payload = {
                    "schema_version": 1,
                    "audit_version": config["audit_version"],
                    "config_sha256": config_hash,
                    "objective_hash": objective.config_hash,
                    "arm_id": arm_id,
                    "update": update,
                    "checkpoint_sha256": checkpoint_hash,
                    "model_version": checkpoint["experiment_spec"]["model"]["model_version"],
                    "original_id": original_id,
                    "source_family_id": core_row["source_family_id"],
                    "topology": core_row["topology"],
                    "size_group": core_row["size_group"],
                    "n": teacher.n,
                    "balanced_assignment": balanced_result.assignment.tolist(),
                    "balanced_assignment_hash": _assignment_hash(balanced_result.assignment),
                    "teacher_assignment": teacher_result.assignment.tolist(),
                    "teacher_assignment_hash": _assignment_hash(teacher_result.assignment),
                    "balanced_score": float(balanced_recomputed.total_score),
                    "teacher_score": float(teacher_recomputed.total_score),
                    "teacher_no_reoptimization_score": float(no_reopt.total_score),
                    "reoptimization_gain": float(
                        teacher_recomputed.total_score - no_reopt.total_score
                    ),
                    "positive_gain": bool(
                        teacher_recomputed.total_score - no_reopt.total_score
                        > float(config["followup"]["positive_gain_tolerance"])
                    ),
                    "assignment_change_rate": float(
                        np.mean(balanced_result.assignment != teacher_result.assignment)
                    ),
                    "balanced_relative_gap": float(
                        max(0.0, balanced_exact - balanced_recomputed.total_score)
                        / max(abs(balanced_exact), objective.epsilon)
                    ),
                    "teacher_relative_gap": float(
                        max(0.0, teacher_exact - teacher_recomputed.total_score)
                        / max(abs(teacher_exact), objective.epsilon)
                    ),
                    "balanced_score_recomputation_error": float(
                        abs(
                            balanced_recomputed.total_score
                            - balanced_result.score.total_score
                        )
                    ),
                    "teacher_score_recomputation_error": float(
                        abs(
                            teacher_recomputed.total_score
                            - teacher_result.score.total_score
                        )
                    ),
                }
                _write_json_atomic(path, payload)
            instance_rows.append(
                {
                    key: payload[key]
                    for key in (
                        "arm_id",
                        "update",
                        "checkpoint_sha256",
                        "model_version",
                        "original_id",
                        "source_family_id",
                        "topology",
                        "size_group",
                        "n",
                        "reoptimization_gain",
                        "positive_gain",
                        "assignment_change_rate",
                        "balanced_relative_gap",
                        "teacher_relative_gap",
                        "balanced_score_recomputation_error",
                        "teacher_score_recomputation_error",
                    )
                }
            )
            if instance_index % 24 == 0:
                print(
                    f"followup {selected_index}/{len(selected)} {arm_id} u{update}: "
                    f"{instance_index}/72",
                    flush=True,
                )
        _write_csv(candidate_dir / "response_rows.csv", instance_rows)
        summary: dict[str, Any] = {
            "arm_id": arm_id,
            "update": update,
            "checkpoint_sha256": checkpoint_hash,
            "model_version": selection["model_version"],
            "selection_reasons": selection["selection_reasons"],
            "instance_count": len(instance_rows),
            "positive_response_rate": float(
                np.mean([row["positive_gain"] for row in instance_rows])
            ),
            "assignment_change_rate": float(
                np.mean([float(row["assignment_change_rate"]) > 0 for row in instance_rows])
            ),
            "max_recomputation_error": float(
                max(
                    max(
                        float(row["balanced_score_recomputation_error"]),
                        float(row["teacher_score_recomputation_error"]),
                    )
                    for row in instance_rows
                )
            ),
        }
        for field in (
            "reoptimization_gain",
            "assignment_change_rate",
            "balanced_relative_gap",
            "teacher_relative_gap",
        ):
            for key, value in _stats([float(row[field]) for row in instance_rows]).items():
                summary[f"{field}_{key}"] = value
        _write_json_atomic(candidate_dir / "response_summary.json", summary)
        summaries.append(summary)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_csv(output_dir / "full_response_summary.csv", summaries)
    return summaries


def _write_report(
    path: Path,
    *,
    screen: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    full: list[dict[str, Any]],
) -> None:
    lines = [
        "# A/B/C checkpoint teacher-response trajectory audit",
        "",
        "Status: completed without training; this audit does not freeze PPO or inference settings.",
        "",
        "## Screen conclusion",
        "",
    ]
    for arm_id in sorted({row["arm_id"] for row in screen}):
        rows = [row for row in screen if row["arm_id"] == arm_id]
        best_positive = max(rows, key=lambda row: float(row["top_action_positive_rate"]))
        best_rank = min(rows, key=lambda row: float(row["true_best_action_policy_rank_median"]))
        lines.append(
            f"- Arm {arm_id}: maximum initial teacher-positive argmax rate "
            f"{float(best_positive['top_action_positive_rate']):.2%} at update "
            f"{best_positive['update']}; best median true-action rank "
            f"{float(best_rank['true_best_action_policy_rank_median']):.1f} at update "
            f"{best_rank['update']}."
        )
    lines.extend(["", "## Registered full-response follow-up", ""])
    full_lookup = {(row["arm_id"], row["update"]): row for row in full}
    for row in selected:
        result = full_lookup[(row["arm_id"], row["update"])]
        lines.append(
            f"- {row['arm_id']} update {row['update']} "
            f"({row['selection_reasons']}): positive teacher response "
            f"{float(result['positive_response_rate']):.2%}, mean gain "
            f"{float(result['reoptimization_gain_mean']):.12f}, teacher P90 gap "
            f"{float(result['teacher_relative_gap_p90']):.4%}."
        )
    any_signal = any(float(row["positive_response_rate"]) > 0 for row in full)
    lines.extend(
        [
            "",
            "## Decision implication",
            "",
            (
                "At least one intermediate checkpoint has genuine teacher-only response; "
                "the corresponding update is evidence for transient signal and requires a "
                "separate recovery decision."
                if any_signal
                else "No registered follow-up checkpoint has genuine teacher-only response. "
                "The failure is therefore not explained solely by selecting the wrong terminal checkpoint."
            ),
            "",
            "No training was executed and no model, objective, or inference setting was frozen.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hash_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "output_hashes.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_csv(output_dir / "output_hashes.csv", rows)


def run(
    config_path: str | Path,
    *,
    validate_only: bool = False,
    screen_only: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or int(config.get("schema_version", 0)) != 1
        or config.get("status") != "user_confirmed_no_training"
        or not config.get("run", {}).get("no_training", False)
    ):
        raise ValueError("Checkpoint trajectory audit is not authorized as no-training.")
    if _sha256(Path(config["protocol"]["path"])) != config["protocol"]["sha256"]:
        raise ValueError("Ablation protocol hash mismatch.")
    context = load_preference_context(
        Path(config["preference_response_config"]).resolve(), output_override=None
    )
    objective = context["objective"]
    if objective.config_hash != config["objective"]["config_hash"]:
        raise ValueError("Frozen objective hash mismatch.")
    expected_updates = list(range(0, 101, 10))
    if [int(value) for value in config["screen"]["updates"]] != expected_updates:
        raise ValueError("The audit must cover update 0,10,...,100.")
    for arm_id, arm in config["arms"].items():
        run_dir = Path(arm["run_dir"])
        for update in expected_updates:
            if not (run_dir / "checkpoints" / f"update_{update:06d}.pt").is_file():
                raise FileNotFoundError(f"Missing {arm_id} update {update} checkpoint.")
    if validate_only:
        return {
            "status": "validated_no_training_no_inference",
            "audit_version": config["audit_version"],
            "arm_count": len(config["arms"]),
            "checkpoint_count": len(config["arms"]) * len(expected_updates),
            "instance_count": len(context["core"]),
            "training_executed": False,
        }

    if config["screen"].get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    output_dir = Path(config["run"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["_config_path"] = str(config_path)
    project_config = load_config()
    started = time.perf_counter()
    metadata = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "audit_version": config["audit_version"],
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "objective_hash": objective.config_hash,
        "checkpoint_count": len(config["arms"]) * len(expected_updates),
        "training_executed": False,
        "parameters_frozen": False,
    }
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    _, screen_summary = _screen(
        config=config,
        context=context,
        output_dir=output_dir,
        project_config=project_config,
    )
    selected = select_followup_checkpoints(
        screen_summary,
        {arm_id: int(arm["selected_update"]) for arm_id, arm in config["arms"].items()},
    )
    _write_csv(output_dir / "followup_selection.csv", selected)
    if screen_only:
        metadata.update(
            {
                "status": "screen_completed_no_training",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": time.perf_counter() - started,
                "followup_checkpoint_count": len(selected),
            }
        )
        _write_json_atomic(output_dir / "run_metadata.json", metadata)
        _write_hash_manifest(output_dir)
        return metadata
    full = _full_followup(
        config=config,
        context=context,
        selected=selected,
        output_dir=output_dir,
        project_config=project_config,
    )
    any_signal = any(float(row["positive_response_rate"]) > 0 for row in full)
    summary = {
        "schema_version": 1,
        "status": "completed_no_training_not_frozen",
        "audit_version": config["audit_version"],
        "objective_hash": objective.config_hash,
        "checkpoint_count_screened": len(screen_summary),
        "instance_count_per_checkpoint": len(context["core"]),
        "followup_checkpoint_count": len(selected),
        "transient_teacher_signal_found": any_signal,
        "screen_summary": screen_summary,
        "followup_selection": selected,
        "full_response_summary": full,
        "training_executed": False,
        "parameters_frozen": False,
    }
    _write_json_atomic(output_dir / "audit_summary.json", summary)
    _write_report(
        output_dir / "checkpoint_teacher_trajectory_report.md",
        screen=screen_summary,
        selected=selected,
        full=full,
    )
    metadata.update(
        {
            "status": "completed_no_training_not_frozen",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": time.perf_counter() - started,
            "followup_checkpoint_count": len(selected),
            "transient_teacher_signal_found": any_signal,
        }
    )
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    _write_hash_manifest(output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "pa_moap_rl/configs/"
            "preference_repair_checkpoint_trajectory_audit_v2.yaml"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, validate_only=args.validate_only, screen_only=args.screen_only),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
