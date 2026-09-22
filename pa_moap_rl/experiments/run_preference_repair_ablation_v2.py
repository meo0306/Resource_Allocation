"""Validate or run one explicitly confirmed pure-PPO preference-repair arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from pa_moap_rl.experiments.formal_training import run_formal_training
from pa_moap_rl.experiments.paired_preference_sampling_v2 import (
    INDEPENDENT_SAMPLER_VERSION,
    PAIRED_SAMPLER_VERSION,
    SUPPORTED_SAMPLER_VERSIONS,
    build_paired_counterfactual_batch,
    sampling_spec,
)
from pa_moap_rl.models.preference_repair_v2 import (
    SUPPORTED_MODEL_VERSIONS,
    build_versioned_actor_critic,
    model_spec,
)
from pa_moap_rl.objective import load_objective_spec


TRAINING_INTEGRATION_VERSION = "formal_training_versioned_injection_v1"
CHECKPOINT_PAYLOAD_VERSION = 3
CONFIRMED_STATUS = "user_confirmed"
LOCKED_STATUS = "awaiting_user_confirmation_no_training"


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Ablation protocol is missing {key!r}.")
    return mapping[key]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_protocol(
    config_path: str | Path,
    *,
    arm_id: str,
) -> tuple[Path, dict[str, Any], Any, dict[str, Any]]:
    """Validate all immutable protocol labels without creating output files."""

    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported preference-repair protocol schema.")
    if config.get("status") not in {LOCKED_STATUS, CONFIRMED_STATUS}:
        raise ValueError("Unknown protocol status; training remains locked.")
    versions = _required(config, "versions")
    if versions.get("training_integration_version") != TRAINING_INTEGRATION_VERSION:
        raise ValueError("Training integration version mismatch.")
    if int(versions.get("checkpoint_payload_version", -1)) != CHECKPOINT_PAYLOAD_VERSION:
        raise ValueError("Checkpoint payload version mismatch.")

    objective_cfg = _required(config, "objective")
    objective = load_objective_spec(_required(objective_cfg, "config"))
    if objective.objective_id != str(_required(objective_cfg, "objective_id")):
        raise ValueError("Objective id mismatch.")
    if objective.config_hash != str(_required(objective_cfg, "config_hash")):
        raise ValueError("Objective hash differs from the frozen configuration.")

    arms = _required(config, "arms")
    if set(arms) != {"A", "B", "C"}:
        raise ValueError("The protocol must define exactly arms A, B, and C.")
    if arm_id not in arms:
        raise ValueError(f"Unknown arm {arm_id!r}; expected A, B, or C.")
    outputs = [str(_required(arm, "output_dir")) for arm in arms.values()]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Each ablation arm must have a unique output directory.")

    for candidate_id, candidate in arms.items():
        sampler_version = str(_required(candidate, "sampler_version"))
        model_version = str(_required(candidate, "model_version"))
        if sampler_version not in SUPPORTED_SAMPLER_VERSIONS:
            raise ValueError(f"Arm {candidate_id} has an unsupported sampler version.")
        if model_version not in SUPPORTED_MODEL_VERSIONS:
            raise ValueError(f"Arm {candidate_id} has an unsupported model version.")
        actual_model_spec = model_spec(model_version)
        for key in ("feature_version", "network_architecture_version"):
            if candidate.get(key) != actual_model_spec[key]:
                raise ValueError(f"Arm {candidate_id} {key} does not match its implementation.")
        if (
            sampler_version == PAIRED_SAMPLER_VERSION
            and int(_required(config["common_run"], "instances_per_update")) % 4 != 0
        ):
            raise ValueError("Paired sampling requires instances_per_update divisible by four.")

    arm = arms[arm_id]
    experiment_spec = {
        "experiment_spec_schema_version": 1,
        "protocol_id": str(_required(config, "protocol_id")),
        "protocol_config_sha256": _sha256(path),
        "arm_id": arm_id,
        "training_integration_version": TRAINING_INTEGRATION_VERSION,
        "checkpoint_payload_version": CHECKPOINT_PAYLOAD_VERSION,
        "objective": {
            "objective_id": objective.objective_id,
            "schema_version": objective.schema_version,
            "config_hash": objective.config_hash,
        },
        "sampling": sampling_spec(str(arm["sampler_version"])),
        "model": model_spec(str(arm["model_version"])),
    }
    return path, config, objective, experiment_spec


def run_arm_from_config(
    config_path: str | Path,
    *,
    arm_id: str,
    validate_only: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    path, config, objective, experiment_spec = load_and_validate_protocol(
        config_path, arm_id=arm_id
    )
    arm = config["arms"][arm_id]
    if validate_only:
        return {
            "status": "validated_no_training",
            "training_started": False,
            "protocol_status": config["status"],
            "output_dir": arm["output_dir"],
            "experiment_spec": experiment_spec,
        }
    if config["status"] != CONFIRMED_STATUS:
        raise PermissionError(
            "Training is locked until the protocol status is explicitly changed to user_confirmed."
        )

    output = Path(_required(arm, "output_dir"))
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Arm output already contains files; use an explicit matching checkpoint: {output}"
        )

    model_version = str(_required(arm, "model_version"))

    def versioned_model_builder(instance: Any, **kwargs: Any) -> Any:
        model = build_versioned_actor_critic(
            instance,
            model_version=model_version,
            **kwargs,
        )
        model.experiment_spec = experiment_spec
        return model

    sampler_version = str(_required(arm, "sampler_version"))
    batch_builder = (
        build_paired_counterfactual_batch
        if sampler_version == PAIRED_SAMPLER_VERSION
        else None
    )
    if sampler_version not in {INDEPENDENT_SAMPLER_VERSION, PAIRED_SAMPLER_VERSION}:
        raise ValueError("Unsupported sampler version after protocol validation.")

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
        training_batch_builder=batch_builder,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pa_moap_rl/configs/preference_repair_ablation_protocol_v2.yaml",
    )
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume-from")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_arm_from_config(
        args.config,
        arm_id=args.arm,
        validate_only=args.validate_only,
        resume_from=args.resume_from,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
