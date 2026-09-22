"""Regression checks for the user-confirmed Stage6 mainline freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = ROOT / "pa_moap_rl/configs/ppo_inference_frozen_four_topology_v2.yaml"
DECISION = ROOT / "results/four_topology_b075_ppo_inference_freeze_v1/freeze_decision.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_stage6_mainline_freeze_is_complete_and_self_consistent() -> None:
    config = yaml.safe_load(FROZEN_CONFIG.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    assert config["status"] == "frozen_by_user_mainline_return_after_failed_stage6a_branch"
    assert config["decision_scope"]["training_hyperparameters_frozen"] is True
    assert config["decision_scope"]["inference_parameters_frozen"] is True
    assert config["problem_and_model"]["model_version"] == "legacy_separate_v1"
    assert config["training"]["learning_rate"] == 1.0e-4
    assert config["training"]["target_kl"] == 0.03
    assert config["evaluation_and_inference"]["env_max_steps"] == 512
    assert config["evaluation_and_inference"]["env_patience"] == 32
    assert config["seed0_reference_checkpoint"]["update"] == 60
    assert config["known_limitations"]["teacher_only_positive_reoptimization_gain_count"] == 0
    assert config["stage6a_branch"]["status"] == "closed_failed_exploratory_branch_not_selected"
    assert config["workflow"]["stage7_batch_service_efficiency_unblocked"] is True

    checkpoint = ROOT / config["seed0_reference_checkpoint"]["path"]
    assert checkpoint.is_file()
    assert _sha256(checkpoint) == config["seed0_reference_checkpoint"]["sha256"]
    assert _sha256(FROZEN_CONFIG) == decision["frozen_config"]["sha256"]


def test_historical_candidate_remains_unmodified_and_non_authoritative() -> None:
    candidate = ROOT / "pa_moap_rl/configs/ppo_inference_freeze_candidate_four_topology_v2.yaml"
    candidate_config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    frozen_config = yaml.safe_load(FROZEN_CONFIG.read_text(encoding="utf-8"))

    assert candidate_config["status"] == "awaiting_user_decision_not_frozen"
    assert _sha256(candidate) == frozen_config["evidence"]["historical_candidate"]["sha256"]
    assert frozen_config["freeze_record_id"] != candidate_config["freeze_record_id"]
