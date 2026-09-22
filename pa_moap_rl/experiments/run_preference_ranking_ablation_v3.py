"""Validate or run the separately versioned D/E preference-ranking ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.formal_training import (
    build_training_buckets,
    load_training_records,
    run_formal_training,
)
from pa_moap_rl.experiments.paired_preference_sampling_v2 import (
    PAIRED_SAMPLER_VERSION,
    build_paired_counterfactual_batch,
    sampling_spec,
)
from pa_moap_rl.experiments.preference_ranking_supervision_v3 import (
    AUXILIARY_LOSS_INTEGRATION_VERSION,
    PreferenceSupervisionSpec,
    build_preference_auxiliary_loss,
)
from pa_moap_rl.models.preference_repair_v2 import (
    PROBLEM_FORMULATION_VERSION,
    SYMMETRIC_MODEL_VERSION,
    build_versioned_actor_critic,
    model_spec,
)
from pa_moap_rl.objective import load_objective_spec
from pa_moap_rl.solvers.ppo_solver import (
    PPOConfig,
    collect_vectorized_rollout,
)


TRAINING_INTEGRATION_VERSION = "formal_training_counterfactual_ranking_v2"
EXPERIMENT_SPEC_SCHEMA_VERSION = 2
CHECKPOINT_PAYLOAD_VERSION = 3
LOCKED_STATUS = "awaiting_user_confirmation_no_training"
CONFIRMED_STATUS = "user_confirmed"


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Ranking protocol is missing {key!r}.")
    return mapping[key]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_immutable_baseline(config: dict[str, Any]) -> None:
    entries = _required(config, "immutable_baseline_artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("immutable_baseline_artifacts must be a non-empty list.")
    for entry in entries:
        path = Path(_required(entry, "path"))
        if not path.is_file():
            raise FileNotFoundError(f"Immutable baseline artifact is missing: {path}")
        actual = _sha256(path)
        expected = str(_required(entry, "sha256"))
        if actual != expected:
            raise ValueError(f"Immutable baseline artifact drift: {path}")


def load_and_validate_protocol(
    config_path: str | Path,
    *,
    arm_id: str,
) -> tuple[Path, dict[str, Any], Any, PreferenceSupervisionSpec, dict[str, Any]]:
    """Validate protocol, frozen objective, old artifacts, and D/E versions."""

    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported preference-ranking protocol schema.")
    if config.get("status") not in {LOCKED_STATUS, CONFIRMED_STATUS}:
        raise ValueError("Unknown ranking protocol status; training remains locked.")
    versions = _required(config, "versions")
    if versions.get("problem_formulation_version") != PROBLEM_FORMULATION_VERSION:
        raise ValueError("Problem formulation version mismatch.")
    if versions.get("training_integration_version") != TRAINING_INTEGRATION_VERSION:
        raise ValueError("Training integration version mismatch.")
    if (
        versions.get("auxiliary_loss_integration_version")
        != AUXILIARY_LOSS_INTEGRATION_VERSION
    ):
        raise ValueError("Auxiliary loss integration version mismatch.")
    if int(versions.get("checkpoint_payload_version", -1)) != CHECKPOINT_PAYLOAD_VERSION:
        raise ValueError("Checkpoint payload version mismatch.")
    if (
        int(versions.get("experiment_spec_schema_version", -1))
        != EXPERIMENT_SPEC_SCHEMA_VERSION
    ):
        raise ValueError("Experiment specification schema mismatch.")

    _validate_immutable_baseline(config)
    objective_cfg = _required(config, "objective")
    objective = load_objective_spec(_required(objective_cfg, "config"))
    if objective.objective_id != str(_required(objective_cfg, "objective_id")):
        raise ValueError("Objective id mismatch.")
    if objective.config_hash != str(_required(objective_cfg, "config_hash")):
        raise ValueError("Frozen objective hash mismatch.")

    arms = _required(config, "arms")
    if set(arms) != {"D", "E"}:
        raise ValueError("The ranking protocol must define exactly arms D and E.")
    if arm_id not in arms:
        raise ValueError(f"Unknown arm {arm_id!r}; expected D or E.")
    outputs = [str(_required(arm, "output_dir")) for arm in arms.values()]
    if len(outputs) != len(set(outputs)):
        raise ValueError("D and E must have distinct output directories.")
    baseline_root = Path(_required(config["read_only_baseline"], "results_root")).resolve()
    for output in outputs:
        resolved = Path(output).resolve()
        if resolved == baseline_root or baseline_root in resolved.parents:
            raise ValueError("D/E output must not be nested in the old A/B/C results root.")

    for candidate_id, candidate in arms.items():
        if candidate.get("sampler_version") != PAIRED_SAMPLER_VERSION:
            raise ValueError(f"Arm {candidate_id} must retain paired sampling.")
        if candidate.get("model_version") != SYMMETRIC_MODEL_VERSION:
            raise ValueError(f"Arm {candidate_id} must retain the symmetric v2 model.")
        actual_model = model_spec(SYMMETRIC_MODEL_VERSION)
        for key in ("feature_version", "network_architecture_version"):
            if candidate.get(key) != actual_model[key]:
                raise ValueError(f"Arm {candidate_id} {key} mismatch.")
        PreferenceSupervisionSpec.from_mapping(
            _required(candidate, "auxiliary_supervision")
        )

    d_spec = PreferenceSupervisionSpec.from_mapping(
        arms["D"]["auxiliary_supervision"]
    )
    e_spec = PreferenceSupervisionSpec.from_mapping(
        arms["E"]["auxiliary_supervision"]
    )
    if d_spec.contrast_weight != 0.0 or e_spec.contrast_weight <= 0.0:
        raise ValueError("D must be ranking-only and E must add counterfactual contrast.")
    if (
        d_spec.ranking_version != e_spec.ranking_version
        or d_spec.ranking_weight != e_spec.ranking_weight
        or d_spec.target_temperature != e_spec.target_temperature
    ):
        raise ValueError("D/E may differ only by the registered contrast term.")

    run = _required(config, "common_run")
    if str(_required(run, "rollout_mode")) != "joint":
        raise ValueError("Counterfactual ranking supervision requires joint rollout.")
    if int(_required(run, "instances_per_update")) % 4 != 0:
        raise ValueError("Paired D/E sampling requires four-environment blocks.")

    arm = arms[arm_id]
    supervision = PreferenceSupervisionSpec.from_mapping(
        arm["auxiliary_supervision"]
    )
    experiment_spec = {
        "experiment_spec_schema_version": EXPERIMENT_SPEC_SCHEMA_VERSION,
        "protocol_id": str(_required(config, "protocol_id")),
        "protocol_config_sha256": _sha256(path),
        "arm_id": arm_id,
        "problem_formulation_version": PROBLEM_FORMULATION_VERSION,
        "training_integration_version": TRAINING_INTEGRATION_VERSION,
        "checkpoint_payload_version": CHECKPOINT_PAYLOAD_VERSION,
        "objective": {
            "objective_id": objective.objective_id,
            "schema_version": objective.schema_version,
            "config_hash": objective.config_hash,
        },
        "sampling": sampling_spec(PAIRED_SAMPLER_VERSION),
        "model": model_spec(SYMMETRIC_MODEL_VERSION),
        "auxiliary_supervision": supervision.to_dict(),
    }
    return path, config, objective, supervision, experiment_spec


def run_arm_from_config(
    config_path: str | Path,
    *,
    arm_id: str,
    validate_only: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    path, config, objective, supervision, experiment_spec = (
        load_and_validate_protocol(config_path, arm_id=arm_id)
    )
    arm = config["arms"][arm_id]
    if validate_only:
        return {
            "status": "validated_no_training",
            "training_started": False,
            "protocol_status": config["status"],
            "immutable_baseline_verified": True,
            "output_dir": arm["output_dir"],
            "experiment_spec": experiment_spec,
        }
    if config["status"] != CONFIRMED_STATUS:
        raise PermissionError(
            "D/E training is locked until the protocol status is explicitly "
            "changed to user_confirmed."
        )

    output = Path(_required(arm, "output_dir"))
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Ranking arm output already contains files; explicit matching "
            f"resume is required: {output}"
        )

    def versioned_model_builder(instance: Any, **kwargs: Any) -> Any:
        model = build_versioned_actor_critic(
            instance,
            model_version=SYMMETRIC_MODEL_VERSION,
            **kwargs,
        )
        model.experiment_spec = experiment_spec
        return model

    def auxiliary_loss_builder(**kwargs: Any) -> Any:
        return build_preference_auxiliary_loss(
            **kwargs,
            spec=supervision,
        )

    data = config["data"]
    run = config["common_run"]
    return run_formal_training(
        train_manifest=_required(data, "train_manifest"),
        validation_manifest=_required(data, "validation_manifest"),
        validation_scenarios=_required(data, "validation_scenarios"),
        validation_pairs_manifest=_required(data, "validation_pairs"),
        allow_controlled_held_out_validation=bool(
            data.get("allow_controlled_held_out_validation", False)
        ),
        data_audit=_required(data, "data_audit"),
        pilot_config=path,
        output_dir=output,
        seed=int(_required(run, "seed")),
        device=str(_required(run, "device")),
        total_updates=int(_required(run, "total_updates")),
        instances_per_update=int(_required(run, "instances_per_update")),
        rollout_steps=int(_required(run, "rollout_steps")),
        minibatch_size=int(_required(run, "minibatch_size")),
        update_epochs=int(_required(run, "update_epochs")),
        learning_rate=float(_required(run, "learning_rate")),
        entropy_coef=float(_required(run, "entropy_coef")),
        target_kl=float(_required(run, "target_kl")),
        eval_interval=int(_required(run, "eval_interval")),
        checkpoint_interval=int(_required(run, "checkpoint_interval")),
        hidden_dim=int(_required(run, "hidden_dim")),
        env_max_steps=int(_required(run, "env_max_steps")),
        env_patience=int(_required(run, "env_patience")),
        eval_env_max_steps=int(_required(run, "eval_env_max_steps")),
        eval_env_patience=int(_required(run, "eval_env_patience")),
        resume_from=resume_from,
        rollout_mode=str(_required(run, "rollout_mode")),
        objective=objective,
        evaluate_at_start=bool(run.get("evaluate_at_start", False)),
        deterministic=bool(run.get("deterministic", False)),
        model_builder=versioned_model_builder,
        training_batch_builder=build_paired_counterfactual_batch,
        auxiliary_loss_builder=auxiliary_loss_builder,
    )


def run_loss_preflight(
    config_path: str | Path,
    *,
    arm_id: str,
) -> dict[str, Any]:
    """Run forward/backward diagnostics without an optimizer step or file writes."""

    path, config, objective, supervision, experiment_spec = (
        load_and_validate_protocol(config_path, arm_id=arm_id)
    )
    data = config["data"]
    run = config["common_run"]
    seed = int(_required(run, "seed"))
    project_config = load_config()
    records = load_training_records(_required(data, "train_manifest"))
    buckets = build_training_buckets(records)
    train_paths = [Path(record["output_json"]) for record in records]
    prepared = build_paired_counterfactual_batch(
        train_buckets=buckets,
        train_paths=train_paths,
        instances_per_update=int(_required(run, "instances_per_update")),
        python_rng=random.Random(seed),
        numpy_rng=np.random.default_rng(seed),
        project_config=project_config,
        objective=objective,
    )
    first_instance = load_instance_json(Path(prepared[0][1]))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model = build_versioned_actor_critic(
        first_instance,
        model_version=SYMMETRIC_MODEL_VERSION,
        objective=objective,
        config=project_config,
        device="cpu",
        hidden_dim=int(_required(run, "hidden_dim")),
    )
    model.experiment_spec = experiment_spec
    preflight_steps = int(config["preflight"].get("rollout_steps", 4))
    ppo_config = PPOConfig(
        rollout_steps=preflight_steps,
        minibatch_size=len(prepared) * preflight_steps,
        update_epochs=1,
        seed=seed,
        device="cpu",
        env_max_steps=preflight_steps,
        env_patience=preflight_steps,
    )
    torch.manual_seed(seed)
    rollout = collect_vectorized_rollout(
        envs=[
            MethodAssignmentEnv(
                config=project_config,
                objective=objective,
                max_steps=preflight_steps,
                patience=preflight_steps,
            )
            for _ in prepared
        ],
        instances=[item[3] for item in prepared],
        model=model,
        config=ppo_config,
    )
    evaluated = model.evaluate_actions(
        rollout.observations,
        flat_action=rollout.actions,
    )
    callback = build_preference_auxiliary_loss(
        prepared=prepared,
        rollout=rollout,
        objective=objective,
        project_config=project_config,
        spec=supervision,
    )
    result = callback(
        model=model,
        rollout=rollout,
        indices=torch.arange(rollout.actions.shape[0]),
        evaluated=evaluated,
    )
    advantages = rollout.advantages
    if advantages.shape[0] > 1:
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1.0e-8)
    ratio = torch.exp(evaluated["log_prob"] - rollout.old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - float(ppo_config.clip_epsilon),
        1.0 + float(ppo_config.clip_epsilon),
    ) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(
        evaluated["state_value"],
        rollout.returns,
    )
    entropy = model._distribution(evaluated["masked_logits"]).entropy().mean()
    base_loss = (
        policy_loss
        + float(ppo_config.value_coef) * value_loss
        - float(ppo_config.entropy_coef) * entropy
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    base_gradients = torch.autograd.grad(
        base_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    ranking_gradients = torch.autograd.grad(
        supervision.ranking_weight * result["components"]["ranking_loss"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    contrast_gradients = (
        torch.autograd.grad(
            supervision.contrast_weight * result["components"]["contrast_loss"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        if supervision.contrast_weight > 0.0
        else tuple(None for _ in parameters)
    )
    auxiliary_gradients = torch.autograd.grad(
        result["loss"],
        parameters,
        allow_unused=True,
    )
    base_gradients = [
        value.detach() for value in base_gradients if value is not None
    ]
    ranking_gradients = [
        value.detach() for value in ranking_gradients if value is not None
    ]
    contrast_gradients = [
        value.detach() for value in contrast_gradients if value is not None
    ]
    auxiliary_gradients = [
        value.detach() for value in auxiliary_gradients if value is not None
    ]
    gradients = [
        *base_gradients,
        *ranking_gradients,
        *contrast_gradients,
        *auxiliary_gradients,
    ]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError("D/E preflight produced missing or non-finite gradients.")
    base_gradient_norm = float(
        torch.sqrt(
            sum(torch.sum(value.double().pow(2)) for value in base_gradients)
        ).item()
    )
    auxiliary_gradient_norm = float(
        torch.sqrt(
            sum(torch.sum(value.double().pow(2)) for value in auxiliary_gradients)
        ).item()
    )
    ranking_gradient_norm = float(
        torch.sqrt(
            sum(torch.sum(value.double().pow(2)) for value in ranking_gradients)
        ).item()
    )
    contrast_gradient_norm = (
        float(
            torch.sqrt(
                sum(
                    torch.sum(value.double().pow(2))
                    for value in contrast_gradients
                )
            ).item()
        )
        if contrast_gradients
        else 0.0
    )
    ranking_ratio = ranking_gradient_norm / max(base_gradient_norm, 1.0e-12)
    contrast_ratio = contrast_gradient_norm / max(base_gradient_norm, 1.0e-12)
    auxiliary_ratio = auxiliary_gradient_norm / max(base_gradient_norm, 1.0e-12)
    gates = config['preflight_gates']
    ranking_range = gates['ranking_to_base_gradient_ratio_range']
    contrast_range = gates['contrast_to_base_gradient_ratio_range_for_E']
    ranking_gate = float(ranking_range[0]) <= ranking_ratio <= float(ranking_range[1])
    contrast_gate = float(contrast_range[0]) <= contrast_ratio <= float(contrast_range[1])
    if arm_id == 'D':
        contrast_gate = True
    auxiliary_gate = auxiliary_ratio <= float(
        gates['auxiliary_to_base_gradient_ratio_max']
    )
    gate_checks = {
        'ranking_gradient_ratio': ranking_gate,
        'contrast_gradient_ratio': contrast_gate,
        'total_auxiliary_gradient_ratio': auxiliary_gate,
    }
    if not all(gate_checks.values()):
        raise RuntimeError('D/E gradient preflight gates failed')
    result['metrics']['gradient_gate_checks'] = gate_checks
    result['metrics']['gradient_gates_passed'] = True
    return {
        "status": "loss_preflight_completed_no_training",
        "training_started": False,
        "optimizer_step_executed": False,
        "files_written": False,
        "protocol_status": config["status"],
        "protocol_config_sha256": _sha256(path),
        "arm_id": arm_id,
        "experiment_spec": experiment_spec,
        "environment_count": len(prepared),
        "rollout_steps": preflight_steps,
        "transition_count": int(rollout.actions.shape[0]),
        "batch_max_n": max(item[3].n for item in prepared),
        "ppo_base_loss": float(base_loss.detach().item()),
        "ppo_base_gradient_l2_norm": base_gradient_norm,
        "auxiliary_gradient_l2_norm": auxiliary_gradient_norm,
        "ranking_gradient_l2_norm": ranking_gradient_norm,
        "contrast_gradient_l2_norm": contrast_gradient_norm,
        "ranking_to_base_gradient_ratio": (
            ranking_gradient_norm / max(base_gradient_norm, 1.0e-12)
        ),
        "contrast_to_base_gradient_ratio": (
            contrast_gradient_norm / max(base_gradient_norm, 1.0e-12)
        ),
        "auxiliary_to_base_gradient_ratio": (
            auxiliary_gradient_norm / max(base_gradient_norm, 1.0e-12)
        ),
        **result["metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "pa_moap_rl/configs/"
            "preference_ranking_ablation_protocol_v3.yaml"
        ),
    )
    parser.add_argument("--arm", choices=("D", "E"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--loss-preflight", action="store_true")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    result = (
        run_loss_preflight(args.config, arm_id=args.arm)
        if args.loss_preflight
        else run_arm_from_config(
            args.config,
            arm_id=args.arm,
            validate_only=args.validate_only,
            resume_from=args.resume_from,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
