"""Versioned auxiliary supervision for same-state preference action ranking.

This module does not change the MDP, frozen objective, sampler, or actor
architecture.  It adds two independently versioned training-only signals:

* an exact-delta listwise ranking loss over legal replacement actions; and
* a same-state counterfactual logit-alignment loss for paired profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pa_moap_rl.objective import ObjectiveSpec


EXACT_DELTA_RANK_LOSS_VERSION = "exact_delta_listwise_rank_v1"
COUNTERFACTUAL_CONTRAST_VERSION = "same_state_counterfactual_logit_alignment_v1"
AUXILIARY_LOSS_INTEGRATION_VERSION = "ppo_auxiliary_loss_callback_v1"
NO_CONTRAST_VERSION = "none"


@dataclass(frozen=True)
class PreferenceSupervisionSpec:
    """Immutable loss settings stored in every D/E experiment specification."""

    ranking_version: str
    ranking_weight: float
    target_temperature: float
    contrast_version: str = NO_CONTRAST_VERSION
    contrast_weight: float = 0.0
    include_student_direction: bool = True
    include_teacher_direction: bool = True

    def __post_init__(self) -> None:
        if self.ranking_version != EXACT_DELTA_RANK_LOSS_VERSION:
            raise ValueError("Unsupported preference ranking loss version.")
        if self.ranking_weight <= 0.0:
            raise ValueError("ranking_weight must be positive.")
        if self.target_temperature <= 0.0:
            raise ValueError("target_temperature must be positive.")
        if self.contrast_version not in {
            NO_CONTRAST_VERSION,
            COUNTERFACTUAL_CONTRAST_VERSION,
        }:
            raise ValueError("Unsupported counterfactual contrast version.")
        if self.contrast_weight < 0.0:
            raise ValueError("contrast_weight must be non-negative.")
        enabled = self.contrast_version != NO_CONTRAST_VERSION
        if enabled != (self.contrast_weight > 0.0):
            raise ValueError(
                "contrast_version and contrast_weight must enable or disable together."
            )
        if enabled and not (
            self.include_student_direction and self.include_teacher_direction
        ):
            raise ValueError(
                "Counterfactual alignment must supervise both actors symmetrically."
            )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PreferenceSupervisionSpec":
        return cls(
            ranking_version=str(values["ranking_version"]),
            ranking_weight=float(values["ranking_weight"]),
            target_temperature=float(values["target_temperature"]),
            contrast_version=str(values.get("contrast_version", NO_CONTRAST_VERSION)),
            contrast_weight=float(values.get("contrast_weight", 0.0)),
            include_student_direction=bool(
                values.get("include_student_direction", True)
            ),
            include_teacher_direction=bool(
                values.get("include_teacher_direction", True)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "integration_version": AUXILIARY_LOSS_INTEGRATION_VERSION,
            "ranking_version": self.ranking_version,
            "ranking_weight": self.ranking_weight,
            "target_temperature": self.target_temperature,
            "contrast_version": self.contrast_version,
            "contrast_weight": self.contrast_weight,
            "include_student_direction": self.include_student_direction,
            "include_teacher_direction": self.include_teacher_direction,
        }


def _soft_penalty(
    distribution: torch.Tensor,
    *,
    objective: ObjectiveSpec,
) -> torch.Tensor:
    method_count = distribution.shape[-1]
    if method_count <= 1:
        entropy = torch.zeros(
            distribution.shape[:-1],
            dtype=distribution.dtype,
            device=distribution.device,
        )
    else:
        denominator = torch.log(
            torch.tensor(
                float(method_count),
                dtype=distribution.dtype,
                device=distribution.device,
            )
        )
        entropy = -torch.sum(
            distribution * torch.log(distribution + float(objective.epsilon)),
            dim=-1,
        ) / denominator
        entropy = entropy.clamp(0.0, 1.0)
    diversity = torch.clamp(
        float(objective.entropy_min) - entropy,
        min=0.0,
    ).pow(2)
    cap = torch.clamp(
        distribution - float(objective.method_cap),
        min=0.0,
    ).pow(2).sum(dim=-1)
    return (
        float(objective.beta_entropy) * diversity
        + float(objective.beta_cap) * cap
    )


def exact_delta_objective(
    observations: dict[str, torch.Tensor],
    *,
    objective: ObjectiveSpec,
) -> torch.Tensor:
    """Return exact one-replacement delta-J labels for a padded model batch."""

    effect = observations["effect_matrix"]
    student = observations["student_pref"]
    teacher = observations["teacher_pref"]
    assignment = observations["assignment"].long()
    node_mask = observations["node_mask"].bool()
    distribution = observations["method_distribution"]
    target = observations["target_distribution"]
    batch_size, node_count, method_count = effect.shape
    if student.shape != effect.shape or teacher.shape != effect.shape:
        raise ValueError("Preference matrices must match effect_matrix shape.")
    if assignment.shape != (batch_size, node_count):
        raise ValueError("Assignment shape is inconsistent with effect_matrix.")
    active_count = node_mask.to(effect.dtype).sum(dim=1).clamp_min(1.0)
    gather_index = assignment.unsqueeze(-1)
    delta_effect = effect - torch.gather(effect, dim=2, index=gather_index)
    delta_student = student - torch.gather(student, dim=2, index=gather_index)
    delta_teacher = teacher - torch.gather(teacher, dim=2, index=gather_index)

    one_hot_current = F.one_hot(
        assignment,
        num_classes=method_count,
    ).to(effect.dtype)
    method_eye = torch.eye(
        method_count,
        dtype=effect.dtype,
        device=effect.device,
    )
    candidate_distribution = distribution[:, None, None, :] + (
        method_eye.view(1, 1, method_count, method_count)
        - one_hot_current[:, :, None, :]
    ) / active_count.view(batch_size, 1, 1, 1)
    candidate_distribution = candidate_distribution.clamp_min(0.0)
    current_soft = _soft_penalty(distribution, objective=objective)
    candidate_soft = _soft_penalty(
        candidate_distribution.reshape(-1, method_count),
        objective=objective,
    ).reshape(batch_size, node_count, method_count)
    delta_soft = candidate_soft - current_soft[:, None, None]

    if float(objective.alpha_global) > 0.0:
        current_global = 1.0 - 0.5 * torch.sum(
            torch.abs(distribution - target),
            dim=-1,
        )
        candidate_global = 1.0 - 0.5 * torch.sum(
            torch.abs(candidate_distribution - target[:, None, None, :]),
            dim=-1,
        )
        delta_global = candidate_global - current_global[:, None, None]
    else:
        delta_global = torch.zeros_like(delta_effect)

    local_scale = active_count[:, None, None]
    delta = (
        float(objective.alpha_effect) * delta_effect / local_scale
        + float(objective.alpha_student) * delta_student / local_scale
        + float(objective.alpha_teacher) * delta_teacher / local_scale
        + float(objective.alpha_global) * delta_global
        - delta_soft
    )
    return delta.masked_fill(~node_mask[:, :, None], 0.0)


def recompute_current_scores(
    observations: dict[str, torch.Tensor],
    *,
    objective: ObjectiveSpec,
) -> torch.Tensor:
    """Recompute [F_E,F_S,F_T,F_G,V_soft,J] after profile substitution."""

    assignment = observations["assignment"].long()
    node_mask = observations["node_mask"].bool()
    gather_index = assignment.unsqueeze(-1)
    active_count = node_mask.to(observations["effect_matrix"].dtype).sum(
        dim=1
    ).clamp_min(1.0)

    def selected_mean(name: str) -> torch.Tensor:
        selected = torch.gather(
            observations[name],
            dim=2,
            index=gather_index,
        ).squeeze(-1)
        return torch.sum(selected * node_mask.to(selected.dtype), dim=1) / active_count

    effect_score = selected_mean("effect_matrix")
    student_score = selected_mean("student_pref")
    teacher_score = selected_mean("teacher_pref")
    distribution = observations["method_distribution"]
    if float(objective.alpha_global) > 0.0:
        global_score = 1.0 - 0.5 * torch.sum(
            torch.abs(distribution - observations["target_distribution"]),
            dim=-1,
        )
    else:
        global_score = torch.zeros_like(effect_score)
    soft_penalty = _soft_penalty(distribution, objective=objective)
    total = (
        float(objective.alpha_effect) * effect_score
        + float(objective.alpha_student) * student_score
        + float(objective.alpha_teacher) * teacher_score
        + float(objective.alpha_global) * global_score
        - soft_penalty
    )
    return torch.stack(
        [
            effect_score,
            student_score,
            teacher_score,
            global_score,
            soft_penalty,
            total,
        ],
        dim=1,
    )


def masked_listwise_rank_loss(
    logits: torch.Tensor,
    target_delta: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    target_temperature: float,
) -> torch.Tensor:
    """KL(target delta ranking || policy ranking) over legal actions."""

    if logits.shape != target_delta.shape or logits.shape != action_mask.shape:
        raise ValueError("Logits, target_delta, and action_mask must share shape.")
    batch_size = logits.shape[0]
    legal = action_mask.reshape(batch_size, -1).bool()
    if not bool(torch.all(legal.any(dim=1)).item()):
        raise ValueError("Every ranking sample must have at least one legal action.")
    policy_flat = logits.reshape(batch_size, -1).masked_fill(~legal, -1.0e9)
    target_flat = (
        target_delta.detach().reshape(batch_size, -1)
        / float(target_temperature)
    ).masked_fill(~legal, -1.0e9)
    target_log_probability = F.log_softmax(target_flat, dim=1)
    target_probability = torch.exp(target_log_probability)
    policy_log_probability = F.log_softmax(policy_flat, dim=1)
    return torch.sum(
        target_probability
        * (target_log_probability - policy_log_probability),
        dim=1,
    ).mean()


def _padded_profile(
    matrix: np.ndarray,
    *,
    max_nodes: int,
    method_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != method_count:
        raise ValueError("Counterfactual preference profile has invalid shape.")
    if values.shape[0] > max_nodes:
        raise ValueError("Counterfactual profile exceeds rollout padding.")
    result = torch.zeros(
        (max_nodes, method_count),
        dtype=dtype,
        device=device,
    )
    result[: values.shape[0]] = torch.as_tensor(
        values,
        dtype=dtype,
        device=device,
    )
    return result


def _counterfactual_profiles(
    *,
    prepared: list[tuple[Any, Any, Any, Any]],
    rollout: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(prepared) % 4 != 0:
        raise ValueError("Counterfactual supervision requires four-environment blocks.")
    max_nodes = int(rollout.observations["assignment"].shape[1])
    method_count = int(rollout.observations["effect_matrix"].shape[2])
    dtype = rollout.observations["effect_matrix"].dtype
    device = rollout.observations["effect_matrix"].device
    student_profiles = []
    teacher_profiles = []
    for start in range(0, len(prepared), 4):
        balanced_record, balanced_path, balanced_scenario, balanced = prepared[start]
        student_record, student_path, student_scenario, student = prepared[start + 1]
        teacher_record, teacher_path, teacher_scenario, teacher = prepared[start + 2]
        if not (
            balanced_path == student_path == teacher_path
            and balanced_record.get("output_json") == student_record.get("output_json")
            and balanced_record.get("output_json") == teacher_record.get("output_json")
        ):
            raise ValueError("Counterfactual triplet does not share one content instance.")
        templates = [
            (balanced_scenario.student_template, balanced_scenario.teacher_template),
            (student_scenario.student_template, student_scenario.teacher_template),
            (teacher_scenario.student_template, teacher_scenario.teacher_template),
        ]
        if templates != [("S6", "T6"), ("S4", "T6"), ("S6", "T4")]:
            raise ValueError("Unexpected paired counterfactual template order.")
        np.testing.assert_array_equal(balanced.teacher_pref, student.teacher_pref)
        np.testing.assert_array_equal(balanced.student_pref, teacher.student_pref)
        student_profiles.append(
            _padded_profile(
                student.student_pref,
                max_nodes=max_nodes,
                method_count=method_count,
                dtype=dtype,
                device=device,
            )
        )
        teacher_profiles.append(
            _padded_profile(
                teacher.teacher_pref,
                max_nodes=max_nodes,
                method_count=method_count,
                dtype=dtype,
                device=device,
            )
        )
    return torch.stack(student_profiles), torch.stack(teacher_profiles)


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def _direction_alignment_loss(
    predicted_difference: torch.Tensor,
    target_difference: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    target_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = predicted_difference.shape[0]
    legal = action_mask.reshape(batch_size, -1).to(predicted_difference.dtype)
    count = legal.sum(dim=1).clamp_min(1.0)
    predicted = predicted_difference.reshape(batch_size, -1)
    target = (
        target_difference.detach().reshape(batch_size, -1)
        / float(target_temperature)
    )
    predicted = (predicted - (predicted * legal).sum(dim=1, keepdim=True) / count[:, None]) * legal
    target = (target - (target * legal).sum(dim=1, keepdim=True) / count[:, None]) * legal
    importance = torch.abs(target) * legal
    importance_sum = importance.sum(dim=1)
    valid = importance_sum > 1.0e-12
    if not bool(valid.any().item()):
        zero = predicted.sum() * 0.0
        return zero, valid
    elementwise = F.smooth_l1_loss(
        predicted[valid],
        target[valid],
        reduction="none",
    )
    valid_importance = importance[valid]
    per_sample = torch.sum(elementwise * valid_importance, dim=1) / (
        valid_importance.sum(dim=1).clamp_min(1.0e-12)
    )
    return per_sample.mean(), valid


class PreferenceAuxiliaryLoss:
    """Callable PPO auxiliary loss bound to one versioned joint rollout."""

    def __init__(
        self,
        *,
        prepared: list[tuple[Any, Any, Any, Any]],
        rollout: Any,
        objective: ObjectiveSpec,
        spec: PreferenceSupervisionSpec,
    ) -> None:
        if int(rollout.environment_count) != len(prepared):
            raise ValueError("Prepared environments do not match joint rollout.")
        self.rollout = rollout
        self.objective = objective
        self.spec = spec
        self.steps = int(rollout.steps_per_environment)
        self.student_profiles: torch.Tensor | None = None
        self.teacher_profiles: torch.Tensor | None = None
        if spec.contrast_version != NO_CONTRAST_VERSION:
            self.student_profiles, self.teacher_profiles = _counterfactual_profiles(
                prepared=prepared,
                rollout=rollout,
            )

    def __call__(
        self,
        *,
        model: Any,
        rollout: Any,
        indices: torch.Tensor,
        evaluated: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        if rollout is not self.rollout:
            raise ValueError("Auxiliary loss received a different rollout.")
        observations = {
            key: value[indices] for key, value in rollout.observations.items()
        }
        target_delta = exact_delta_objective(
            observations,
            objective=self.objective,
        )
        ranking_loss = masked_listwise_rank_loss(
            evaluated["masked_logits"],
            target_delta,
            observations["action_mask"],
            target_temperature=self.spec.target_temperature,
        )
        total = self.spec.ranking_weight * ranking_loss
        contrast_loss = total * 0.0
        contrast_samples = 0

        if self.spec.contrast_version != NO_CONTRAST_VERSION:
            environment_indices = torch.div(
                indices,
                self.steps,
                rounding_mode="floor",
            )
            balanced_local = torch.nonzero(
                environment_indices.remainder(4) == 0,
                as_tuple=False,
            ).squeeze(-1)
            if balanced_local.numel() > 0:
                block_indices = torch.div(
                    environment_indices[balanced_local],
                    4,
                    rounding_mode="floor",
                )
                balanced = {
                    key: value[balanced_local]
                    for key, value in observations.items()
                }
                student = _clone_batch(balanced)
                teacher = _clone_batch(balanced)
                assert self.student_profiles is not None
                assert self.teacher_profiles is not None
                student["student_pref"] = self.student_profiles[block_indices]
                teacher["teacher_pref"] = self.teacher_profiles[block_indices]
                student["current_scores"] = recompute_current_scores(
                    student,
                    objective=self.objective,
                )
                teacher["current_scores"] = recompute_current_scores(
                    teacher,
                    objective=self.objective,
                )
                counterfactual = {
                    key: torch.cat([student[key], teacher[key]], dim=0)
                    for key in balanced
                }
                counterfactual_logits = model(counterfactual).masked_logits
                count = balanced_local.numel()
                student_logits = counterfactual_logits[:count]
                teacher_logits = counterfactual_logits[count:]
                balanced_logits = evaluated["masked_logits"][balanced_local]
                balanced_delta = exact_delta_objective(
                    balanced,
                    objective=self.objective,
                )
                student_delta = exact_delta_objective(
                    student,
                    objective=self.objective,
                )
                teacher_delta = exact_delta_objective(
                    teacher,
                    objective=self.objective,
                )
                student_loss, student_valid = _direction_alignment_loss(
                    student_logits - balanced_logits,
                    student_delta - balanced_delta,
                    balanced["action_mask"],
                    target_temperature=self.spec.target_temperature,
                )
                teacher_loss, teacher_valid = _direction_alignment_loss(
                    teacher_logits - balanced_logits,
                    teacher_delta - balanced_delta,
                    balanced["action_mask"],
                    target_temperature=self.spec.target_temperature,
                )
                contrast_loss = 0.5 * (student_loss + teacher_loss)
                contrast_samples = int(
                    student_valid.sum().item() + teacher_valid.sum().item()
                )
                total = total + self.spec.contrast_weight * contrast_loss

        legal = observations["action_mask"].reshape(indices.shape[0], -1)
        flat_delta = target_delta.reshape(indices.shape[0], -1)
        chosen = torch.argmax(evaluated["masked_logits"].reshape(indices.shape[0], -1), dim=1)
        chosen_delta = torch.gather(flat_delta, dim=1, index=chosen[:, None]).squeeze(1)
        return {
            "loss": total,
            "components": {
                "ranking_loss": ranking_loss,
                "contrast_loss": contrast_loss,
            },
            "metrics": {
                "auxiliary_total_loss": float(total.detach().item()),
                "auxiliary_ranking_loss": float(ranking_loss.detach().item()),
                "auxiliary_contrast_loss": float(contrast_loss.detach().item()),
                "auxiliary_policy_top_positive_rate": float(
                    (chosen_delta > 1.0e-9).float().mean().item()
                ),
                "auxiliary_legal_action_rate": float(legal.float().mean().item()),
                "auxiliary_contrast_direction_samples": float(contrast_samples),
            },
        }


def build_preference_auxiliary_loss(
    *,
    prepared: list[tuple[Any, Any, Any, Any]],
    rollout: Any,
    objective: ObjectiveSpec,
    project_config: Any,
    spec: PreferenceSupervisionSpec,
) -> PreferenceAuxiliaryLoss:
    """Build one update-scoped callback; project_config is accepted for parity."""

    del project_config
    return PreferenceAuxiliaryLoss(
        prepared=prepared,
        rollout=rollout,
        objective=objective,
        spec=spec,
    )


def supervision_spec(values: dict[str, Any]) -> dict[str, Any]:
    return PreferenceSupervisionSpec.from_mapping(values).to_dict()


__all__ = [
    "AUXILIARY_LOSS_INTEGRATION_VERSION",
    "COUNTERFACTUAL_CONTRAST_VERSION",
    "EXACT_DELTA_RANK_LOSS_VERSION",
    "NO_CONTRAST_VERSION",
    "PreferenceAuxiliaryLoss",
    "PreferenceSupervisionSpec",
    "build_preference_auxiliary_loss",
    "exact_delta_objective",
    "masked_listwise_rank_loss",
    "recompute_current_scores",
    "supervision_spec",
]
