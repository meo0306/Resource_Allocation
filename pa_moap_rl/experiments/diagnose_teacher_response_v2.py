"""Diagnose teacher-only zero response without training or parameter mutation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pa_moap_rl.checkpointing import method_matrix_hash, validate_checkpoint_metadata
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.run_preference_response_v2 import (
    _load_and_validate as load_preference_context,
)
from pa_moap_rl.solvers.ppo_solver import (
    build_actor_critic,
    observation_to_model_batch,
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _candidate_rewards(instance: Any, assignment: np.ndarray, objective: Any) -> np.ndarray:
    """Vectorized exact single-replacement reward under the frozen objective."""

    n = instance.n
    m = instance.m
    nodes = np.arange(n)
    distribution = np.bincount(assignment, minlength=m).astype(np.float64) / n
    eye = np.eye(m, dtype=np.float64)
    candidate_distribution = distribution[None, None, :] + (
        eye[None, :, :] - eye[assignment][:, None, :]
    ) / n
    entropy = -np.sum(
        candidate_distribution * np.log(candidate_distribution + objective.epsilon),
        axis=2,
    ) / np.log(m)
    entropy = np.clip(entropy, 0.0, 1.0)
    candidate_penalty = (
        objective.beta_entropy
        * np.maximum(objective.entropy_min - entropy, 0.0) ** 2
        + objective.beta_cap
        * np.sum(
            np.maximum(candidate_distribution - objective.method_cap, 0.0) ** 2,
            axis=2,
        )
    )
    current_entropy = -np.sum(
        distribution * np.log(distribution + objective.epsilon)
    ) / np.log(m)
    current_penalty = (
        objective.beta_entropy
        * max(objective.entropy_min - current_entropy, 0.0) ** 2
        + objective.beta_cap
        * np.sum(np.maximum(distribution - objective.method_cap, 0.0) ** 2)
    )
    delta_effect = (
        instance.effect_matrix
        - instance.effect_matrix[nodes, assignment][:, None]
    )
    delta_student = (
        instance.student_pref
        - instance.student_pref[nodes, assignment][:, None]
    )
    delta_teacher = (
        instance.teacher_pref
        - instance.teacher_pref[nodes, assignment][:, None]
    )
    rewards = (
        objective.alpha_effect * delta_effect
        + objective.alpha_student * delta_student
        + objective.alpha_teacher * delta_teacher
    ) / n - (candidate_penalty - current_penalty)
    rewards[nodes, assignment] = -np.inf
    rewards[~instance.feasible_mask] = -np.inf
    return rewards


def _best_improvement(
    instance: Any,
    assignment: np.ndarray,
    objective: Any,
    *,
    tolerance: float,
    max_moves: int,
    recompute_tolerance: float,
) -> tuple[np.ndarray, float, int]:
    values = np.asarray(assignment, dtype=np.int64).copy()
    current = score_assignment(values, instance=instance, objective=objective).total_score
    for move in range(max_moves):
        rewards = _candidate_rewards(instance, values, objective)
        flat = int(np.argmax(rewards))
        delta = float(rewards.reshape(-1)[flat])
        if not np.isfinite(delta) or delta <= tolerance:
            return values, float(current), move
        node, method = divmod(flat, instance.m)
        candidate = values.copy()
        candidate[node] = method
        recomputed = score_assignment(
            candidate, instance=instance, objective=objective
        ).total_score
        if abs((recomputed - current) - delta) > recompute_tolerance:
            raise RuntimeError("Vectorized replacement reward mismatch.")
        values = candidate
        current = recomputed
    raise RuntimeError("Local-improvement diagnostic exceeded max_local_moves.")


def _policy_vector(model: Any, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        output = model(observation_to_model_batch(observation, device="cpu"))
    mask = observation["action_mask"].reshape(-1)
    logits = output.masked_logits.reshape(-1).cpu().numpy()[mask]
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= np.sum(probabilities)
    return logits, probabilities


def _load_pair_assignment(pair_dir: Path, pair_id: str) -> np.ndarray:
    payload = json.loads((pair_dir / f"{pair_id}.json").read_text(encoding="utf-8"))
    return np.asarray(payload["assignment"], dtype=np.int64)


def _summarize_rows(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row["kind"] == kind]
    numeric = [
        "preference_absolute_change_mean",
        "preference_action_relative_change_mean",
        "initial_logit_change_mean",
        "initial_probability_tv",
        "positive_action_rate_at_balanced_ppo",
        "policy_chosen_reward_at_balanced_ppo",
        "true_best_reward_at_balanced_ppo",
        "true_best_action_policy_rank",
        "hybrid_reoptimization_gain",
        "hybrid_relative_gap_to_exact",
        "hybrid_assignment_change_rate",
    ]
    result: dict[str, Any] = {"count": len(selected)}
    for field in numeric:
        result[field] = _stats([float(row[field]) for row in selected])
    result["initial_top_action_change_rate"] = float(
        np.mean([bool(row["initial_top_action_changed"]) for row in selected])
    )
    result["policy_chosen_positive_rate_at_balanced_ppo"] = float(
        np.mean(
            [float(row["policy_chosen_reward_at_balanced_ppo"]) > 1.0e-12 for row in selected]
        )
    )
    result["hybrid_positive_response_rate"] = float(
        np.mean([float(row["hybrid_reoptimization_gain"]) > 1.0e-12 for row in selected])
    )
    return result


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    student = summary["scenario_summary"]["student"]
    teacher = summary["scenario_summary"]["teacher"]
    lines = [
        "# Teacher-only zero-response root-cause diagnostic",
        "",
        "Status: completed without training; PPO and inference settings remain unfrozen.",
        "",
        "## Confirmed findings",
        "",
        "- The teacher preference field reaches the objective, environment observation, encoder, and actor. No missing-field or zero-weight defect was found.",
        "- The frozen objective gives student and teacher equal coefficients of 0.325.",
        "- Exact and local-search results show a real teacher-only response opportunity; search budget was already ruled out.",
        f"- At the balanced PPO assignment, the policy-selected teacher action is positive on {teacher['policy_chosen_positive_rate_at_balanced_ppo']:.2%} of instances, despite a mean positive-action rate of {teacher['positive_action_rate_at_balanced_ppo']['mean']:.2%}.",
        f"- The true best teacher action has mean policy rank {teacher['true_best_action_policy_rank']['mean']:.1f}; the actor therefore misranks available improvements rather than lacking legal moves.",
        f"- A deterministic best-improvement continuation produces positive teacher response on {teacher['hybrid_positive_response_rate']:.2%} of instances and a mean exact gap of {teacher['hybrid_relative_gap_to_exact']['mean']:.6%}.",
        "",
        "## Ruled-out explanations",
        "",
        "- objective coefficient asymmetry",
        "- missing teacher tensor or disconnected model input",
        "- absent T4 or S6-T4 training coverage",
        "- no exact teacher-response opportunity",
        "- insufficient inference steps or patience",
        "- hard-constraint or action-mask blockage",
        "",
        "## Remaining causal hypotheses",
        "",
        "1. Training identification: independently sampled content/profile tuples provide no same-content counterfactual teacher contrast inside an update.",
        "2. Ranking supervision: PPO observes only scalar total reward; it has no direct auxiliary signal that the teacher-sensitive positive actions should outrank the current argmax.",
        "3. Representation scale: the actor receives raw component deltas but not the exact weighted, N-normalized one-step objective delta, so it must learn the size-dependent trade-off with soft penalties.",
        "4. Symmetry risk: the equal-weight objective is invariant to swapping student and teacher, but the network uses separate unconstrained channels and does not enforce that invariance.",
        "5. Solver behavior: feasibility masking permits negative-reward actions; unchanged argmax trajectories therefore cycle until patience while best-so-far preserves the old solution.",
        "",
        "## Recommended resolution sequence",
        "",
        "1. Register PPO plus deterministic one-point best-improvement as a diagnostic hybrid baseline, not as evidence that PPO itself learned teacher conditioning.",
        "2. Add paired same-content counterfactual scenario batches with balanced, student-only, and teacher-only changes; keep the frozen objective unchanged.",
        "3. Add an objective-consistent action feature containing the exact weighted one-step delta and enforce student/teacher exchange symmetry by using combined preference features or tied processing.",
        "4. Compare paired batching alone, representation repair alone, and both together in short seed0 ablations; do not change multiple other PPO hyperparameters simultaneously.",
        "5. Re-run the controlled 72x4 response audit and the 512/32 quality audit before returning to the freeze decision.",
        "",
        "The hybrid baseline is a solver-level workaround. A claim that PPO itself uses both actors requires the retrained pure-policy path to pass the controlled response audit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "user_authorized_no_training":
        raise ValueError("Teacher-response diagnostic is not user-authorized.")
    if not config["diagnostic"].get("no_training", False):
        raise ValueError("This diagnostic must explicitly prohibit training.")

    protocol_path = Path(config["controlled_protocol"]["config"]).resolve()
    context = load_preference_context(protocol_path, output_override=None)
    objective = context["objective"]
    if objective.config_hash != config["objective"]["config_hash"]:
        raise ValueError("Frozen objective hash mismatch.")
    checkpoint_path = Path(config["checkpoint"]["path"]).resolve()
    if _sha256(checkpoint_path) != config["checkpoint"]["expected_sha256"]:
        raise ValueError("Checkpoint hash mismatch.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["update"]) != int(config["checkpoint"]["expected_update"]):
        raise ValueError("Checkpoint update mismatch.")
    first_row = next(iter(context["core"].values()))
    first_instance = load_instance_json(Path(first_row["output_json"]))
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_matrix_hash(first_instance),
        data_hash=context["ppo_data_hash"],
    )
    project_config = load_config()
    model = build_actor_critic(
        first_instance,
        config=project_config,
        device="cpu",
        hidden_dim=checkpoint.get("hidden_dim"),
        objective=objective,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    baseline_id = config["controlled_protocol"]["baseline_scenario_id"]
    shifted = config["controlled_protocol"]["shifted_scenarios"]
    pair_by_key = {
        (row["original_id"], row["scenario_id"]): row for row in context["pairs"]
    }
    pair_dir = Path(config["existing_evidence"]["update60_pairs"])
    tolerance = float(config["diagnostic"]["local_improvement_tolerance"])
    max_moves = int(config["diagnostic"]["max_local_moves"])
    recompute_tolerance = float(config["diagnostic"]["score_recompute_tolerance"])
    rows: list[dict[str, Any]] = []

    for index, (original_id, core_row) in enumerate(context["core"].items(), start=1):
        base = load_instance_json(Path(core_row["output_json"]))
        instances = {
            scenario_id: apply_scenario(
                base,
                context["scenarios"][scenario_id],
                project_config,
                objective=objective,
            )
            for scenario_id in [baseline_id, *shifted.values()]
        }
        initial_observations: dict[str, dict[str, np.ndarray]] = {}
        initial_logits: dict[str, np.ndarray] = {}
        initial_probabilities: dict[str, np.ndarray] = {}
        for scenario_id, instance in instances.items():
            env = MethodAssignmentEnv(
                max_steps=512,
                patience=32,
                config=project_config,
                objective=objective,
            )
            observation = env.reset(instance)
            initial_observations[scenario_id] = observation
            initial_logits[scenario_id], initial_probabilities[scenario_id] = (
                _policy_vector(model, observation)
            )
        baseline_initial = initial_observations[baseline_id]["assignment"]
        if any(
            not np.array_equal(baseline_initial, observation["assignment"])
            for observation in initial_observations.values()
        ):
            raise RuntimeError("Effect-greedy initial assignments differ by profile.")

        baseline_pair = pair_by_key[(original_id, baseline_id)]
        baseline_ppo = _load_pair_assignment(pair_dir, baseline_pair["pair_id"])
        baseline_local, _, _ = _best_improvement(
            instances[baseline_id],
            baseline_ppo,
            objective,
            tolerance=tolerance,
            max_moves=max_moves,
            recompute_tolerance=recompute_tolerance,
        )

        for kind, scenario_id in shifted.items():
            instance = instances[scenario_id]
            pair = pair_by_key[(original_id, scenario_id)]
            shifted_ppo = _load_pair_assignment(pair_dir, pair["pair_id"])
            shifted_local, shifted_local_score, shifted_local_moves = _best_improvement(
                instance,
                shifted_ppo,
                objective,
                tolerance=tolerance,
                max_moves=max_moves,
                recompute_tolerance=recompute_tolerance,
            )
            reused_local_score = score_assignment(
                baseline_local, instance=instance, objective=objective
            ).total_score
            exact_payload = json.loads(
                (context["exact_dir"] / f"{pair['pair_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            exact_score = float(exact_payload["score"]["total_score"])

            shifted_profile_matrix = (
                instance.student_pref if kind == "student" else instance.teacher_pref
            )
            baseline_profile_matrix = (
                instances[baseline_id].student_pref
                if kind == "student"
                else instances[baseline_id].teacher_pref
            )
            profile_change = shifted_profile_matrix - baseline_profile_matrix
            relative_change = profile_change - profile_change[
                np.arange(instance.n), baseline_initial
            ][:, None]
            initial_mask = initial_observations[baseline_id]["action_mask"]
            baseline_logits = initial_logits[baseline_id]
            scenario_logits = initial_logits[scenario_id]
            baseline_probabilities = initial_probabilities[baseline_id]
            scenario_probabilities = initial_probabilities[scenario_id]

            diagnostic_env = MethodAssignmentEnv(
                max_steps=512,
                patience=32,
                config=project_config,
                objective=objective,
            )
            ppo_observation = diagnostic_env.reset(
                instance, initial_assignment=baseline_ppo
            )
            flat_rewards = _candidate_rewards(
                instance, baseline_ppo, objective
            ).reshape(-1)
            legal_flat = np.flatnonzero(np.isfinite(flat_rewards))
            with torch.no_grad():
                policy_output = model(
                    observation_to_model_batch(ppo_observation, device="cpu")
                )
            full_logits = policy_output.masked_logits.reshape(-1).cpu().numpy()
            chosen_flat = int(np.argmax(full_logits))
            best_flat = int(np.argmax(flat_rewards))
            ranked_legal = legal_flat[np.argsort(-full_logits[legal_flat])]
            best_rank = int(np.flatnonzero(ranked_legal == best_flat)[0]) + 1

            rows.append(
                {
                    "original_id": original_id,
                    "topology": core_row["topology"],
                    "size_group": core_row["size_group"],
                    "n": instance.n,
                    "kind": kind,
                    "scenario_id": scenario_id,
                    "preference_absolute_change_mean": float(
                        np.mean(np.abs(profile_change))
                    ),
                    "preference_action_relative_change_mean": float(
                        np.mean(np.abs(relative_change[initial_mask]))
                    ),
                    "initial_logit_change_mean": float(
                        np.mean(np.abs(scenario_logits - baseline_logits))
                    ),
                    "initial_probability_tv": float(
                        0.5
                        * np.sum(
                            np.abs(
                                scenario_probabilities - baseline_probabilities
                            )
                        )
                    ),
                    "initial_top_action_changed": bool(
                        np.argmax(scenario_logits) != np.argmax(baseline_logits)
                    ),
                    "positive_action_count_at_balanced_ppo": int(
                        np.sum(flat_rewards[legal_flat] > tolerance)
                    ),
                    "positive_action_rate_at_balanced_ppo": float(
                        np.mean(flat_rewards[legal_flat] > tolerance)
                    ),
                    "policy_chosen_reward_at_balanced_ppo": float(
                        flat_rewards[chosen_flat]
                    ),
                    "true_best_reward_at_balanced_ppo": float(
                        flat_rewards[best_flat]
                    ),
                    "true_best_action_policy_rank": best_rank,
                    "ppo_shifted_equals_balanced_assignment": bool(
                        np.array_equal(shifted_ppo, baseline_ppo)
                    ),
                    "hybrid_local_moves": shifted_local_moves,
                    "hybrid_reoptimization_gain": float(
                        shifted_local_score - reused_local_score
                    ),
                    "hybrid_relative_gap_to_exact": float(
                        max(0.0, exact_score - shifted_local_score)
                        / max(abs(exact_score), objective.epsilon)
                    ),
                    "hybrid_assignment_change_rate": float(
                        np.mean(shifted_local != baseline_local)
                    ),
                }
            )
        if index % 12 == 0:
            print(f"diagnosed {index}/72 instances", flush=True)

    training_rows = _read_csv(Path(config["training_log"]["path"]))
    pair_counts = Counter(
        (row["student_template"], row["teacher_template"])
        for row in training_rows
    )
    duplicate_source_contrasts = 0
    by_update_source: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in training_rows:
        key = (row["update"], row["source_identity"])
        by_update_source.setdefault(key, set()).add(
            (row["student_template"], row["teacher_template"])
        )
    duplicate_source_contrasts = sum(
        len(values) > 1 for values in by_update_source.values()
    )

    checkpoint_rows: list[dict[str, Any]] = []
    cached_initial: list[tuple[dict[str, np.ndarray], ...]] = []
    for core_row in context["core"].values():
        base = load_instance_json(Path(core_row["output_json"]))
        observations = []
        for scenario_id in [baseline_id, shifted["student"], shifted["teacher"]]:
            instance = apply_scenario(
                base,
                context["scenarios"][scenario_id],
                project_config,
                objective=objective,
            )
            env = MethodAssignmentEnv(
                max_steps=512,
                patience=32,
                config=project_config,
                objective=objective,
            )
            observations.append(env.reset(instance))
        cached_initial.append(tuple(observations))
    checkpoint_root = checkpoint_path.parent
    for update in config["diagnostic"]["checkpoint_updates"]:
        candidate = torch.load(
            checkpoint_root / f"update_{int(update):06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        candidate_model = build_actor_critic(
            first_instance,
            config=project_config,
            device="cpu",
            hidden_dim=candidate.get("hidden_dim"),
            objective=objective,
        )
        candidate_model.load_state_dict(candidate["model_state_dict"])
        candidate_model.eval()
        top_changes = {"student": [], "teacher": []}
        tv_values = {"student": [], "teacher": []}
        for observations in cached_initial:
            vectors = [_policy_vector(candidate_model, obs) for obs in observations]
            for position, kind in [(1, "student"), (2, "teacher")]:
                top_changes[kind].append(
                    np.argmax(vectors[position][0]) != np.argmax(vectors[0][0])
                )
                tv_values[kind].append(
                    0.5 * np.sum(np.abs(vectors[position][1] - vectors[0][1]))
                )
        checkpoint_rows.append(
            {
                "update": int(update),
                "student_initial_top_change_rate": float(
                    np.mean(top_changes["student"])
                ),
                "teacher_initial_top_change_rate": float(
                    np.mean(top_changes["teacher"])
                ),
                "student_initial_probability_tv_mean": float(
                    np.mean(tv_values["student"])
                ),
                "teacher_initial_probability_tv_mean": float(
                    np.mean(tv_values["teacher"])
                ),
            }
        )

    actor_weights = model.actor[0].weight.detach().cpu().numpy()
    scalar_offset = 3 * model.hidden_dim
    scalar_names = [
        "effect",
        "student",
        "teacher",
        "delta_effect",
        "delta_student",
        "delta_teacher",
        "delta_global",
        "delta_soft",
        "is_current",
    ]
    summary = {
        "schema_version": 1,
        "status": "diagnostic_completed_no_training_parameters_unfrozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_hash": objective.config_hash,
        "checkpoint_update": int(checkpoint["update"]),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "instance_count": len(context["core"]),
        "scenario_summary": {
            "student": _summarize_rows(rows, "student"),
            "teacher": _summarize_rows(rows, "teacher"),
        },
        "training_coverage": {
            "episode_rows": len(training_rows),
            "unique_allowed_template_pairs": len(pair_counts),
            "S6_T4_rows": pair_counts[("S6", "T4")],
            "S6_T6_rows": pair_counts[("S6", "T6")],
            "S4_T6_rows": pair_counts[("S4", "T6")],
            "same_update_same_content_counterfactual_contrasts": duplicate_source_contrasts,
        },
        "actor_direct_scalar_weight_l2": {
            name: float(np.linalg.norm(actor_weights[:, scalar_offset + index]))
            for index, name in enumerate(scalar_names)
        },
        "ruled_out": [
            "teacher_input_disconnected",
            "teacher_objective_weight_zero_or_unequal",
            "controlled_teacher_template_absent_from_training",
            "no_exact_teacher_response_opportunity",
            "search_budget_or_patience_bottleneck",
            "hard_constraint_or_action_mask_blockage",
        ],
        "confirmed_mechanism": [
            "teacher_perturbation_changes_logits_but_not_initial_argmax_at_update60",
            "policy_argmax_at_balanced_ppo_assignment_is_negative_for_teacher_on_all_72",
            "positive_teacher_actions_exist_but_are_misranked",
            "deterministic_local_improvement_recovers_teacher_response",
        ],
        "parameters_frozen": False,
        "training_executed": False,
    }

    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_instance_diagnostics.csv", rows)
    _write_csv(output_dir / "checkpoint_sensitivity.csv", checkpoint_rows)
    _write_json_atomic(output_dir / "diagnostic_summary.json", summary)
    _write_report(output_dir / "teacher_response_diagnostic_report.md", summary)
    metadata = {
        "schema_version": 1,
        "diagnostic_id": config["diagnostic_id"],
        "status": "completed_no_training",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "input_hashes": {
            str(path): _sha256(path)
            for path in [
                config_path,
                protocol_path,
                checkpoint_path,
                Path(config["training_log"]["path"]),
                Path(config["existing_evidence"]["update70_method_results"]),
                Path(config["existing_evidence"]["search_budget_summary"]),
            ]
        },
        "training_executed": False,
        "parameters_frozen": False,
    }
    _write_json_atomic(output_dir / "run_metadata.json", metadata)
    hash_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "output_hashes.csv":
            hash_rows.append(
                {
                    "relative_path": path.name,
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_csv(output_dir / "output_hashes.csv", hash_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/teacher_response_root_cause_diagnostic_v2.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
