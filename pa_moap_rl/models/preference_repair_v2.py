"""Versioned PPO model components for dual-actor preference repair.

The legacy model remains available as ``legacy_separate_v1``.  The v2 model
adds an exact one-step objective delta and enforces student/teacher exchange
symmetry in every preference-derived feature consumed by the policy.
"""

from __future__ import annotations

from typing import Any

import torch

from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.models.actor_critic import MaskedActorCritic
from pa_moap_rl.models.encoders import (
    AssignmentStateEncoder,
    GlobalEncoder,
    NodeEncoder,
    PairFeatureBuilder,
)
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.ppo_solver import build_actor_critic


LEGACY_MODEL_VERSION = "legacy_separate_v1"
SYMMETRIC_MODEL_VERSION = "objective_delta_symmetric_v2"
SUPPORTED_MODEL_VERSIONS = {LEGACY_MODEL_VERSION, SYMMETRIC_MODEL_VERSION}

# These labels deliberately version independently. A bundle-level model
# label alone is not sufficient to audit or safely restore an ablation.
PROBLEM_FORMULATION_VERSION = "single_replacement_masked_mdp_v1"
LEGACY_FEATURE_VERSION = "separate_student_teacher_features_v1"
SYMMETRIC_FEATURE_VERSION = "objective_consistent_symmetric_features_v1"
LEGACY_NETWORK_VERSION = "masked_actor_critic_separate_channels_v1"
SYMMETRIC_NETWORK_VERSION = "masked_actor_critic_exchange_invariant_v1"


class SymmetricNodeEncoder(NodeEncoder):
    """Encode current preferences through swap-invariant mean and gap terms."""

    def forward(
        self,
        category_id: torch.Tensor,
        concept_need: torch.Tensor,
        assignment: torch.Tensor,
        effect_matrix: torch.Tensor,
        student_pref: torch.Tensor,
        teacher_pref: torch.Tensor,
    ) -> torch.Tensor:
        category_emb = self.category_embedding(category_id.long())
        assigned_method_emb = self.method_embedding(assignment.long())
        gather_index = assignment.long().unsqueeze(-1)
        current_effect = torch.gather(effect_matrix, dim=2, index=gather_index)
        current_student = torch.gather(student_pref, dim=2, index=gather_index)
        current_teacher = torch.gather(teacher_pref, dim=2, index=gather_index)
        current_preference_mean = 0.5 * (current_student + current_teacher)
        current_preference_gap = torch.abs(current_student - current_teacher)
        features = torch.cat(
            [
                category_emb,
                concept_need,
                assigned_method_emb,
                current_effect,
                current_preference_mean,
                current_preference_gap,
            ],
            dim=-1,
        )
        return self.net(features)


class SymmetricGlobalEncoder(GlobalEncoder):
    """Replace separate F_S/F_T inputs by swap-invariant mean and gap."""

    def forward(
        self,
        node_repr: torch.Tensor,
        node_mask: torch.Tensor,
        method_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> torch.Tensor:
        symmetric_scores = torch.stack(
            [
                current_scores[:, 0],
                0.5 * (current_scores[:, 1] + current_scores[:, 2]),
                torch.abs(current_scores[:, 1] - current_scores[:, 2]),
                current_scores[:, 3],
                current_scores[:, 4],
                current_scores[:, 5],
            ],
            dim=1,
        )
        return super().forward(
            node_repr=node_repr,
            node_mask=node_mask,
            method_distribution=method_distribution,
            target_distribution=target_distribution,
            current_scores=symmetric_scores,
        )


class ObjectiveDeltaSymmetricPairFeatureBuilder(PairFeatureBuilder):
    """Build nine symmetric scalars including exact frozen-objective delta J."""

    scalar_feature_names = (
        "effect",
        "preference_mean",
        "preference_gap",
        "delta_effect",
        "delta_preference_mean",
        "delta_preference_gap",
        "delta_objective",
        "delta_soft_penalty",
        "is_current",
    )

    def __init__(self, *, objective: ObjectiveSpec, num_methods: int) -> None:
        if abs(objective.alpha_student - objective.alpha_teacher) > objective.epsilon:
            raise ValueError("The symmetric v2 encoder requires equal student/teacher weights.")
        super().__init__(
            num_methods=num_methods,
            entropy_min=objective.entropy_min,
            method_cap=objective.method_cap,
            lambda_div=objective.beta_entropy,
            lambda_cap=objective.beta_cap,
            epsilon=objective.epsilon,
        )
        self.alpha_effect = float(objective.alpha_effect)
        self.alpha_student = float(objective.alpha_student)
        self.alpha_teacher = float(objective.alpha_teacher)
        self.alpha_global = float(objective.alpha_global)

    def forward(
        self,
        node_repr: torch.Tensor,
        method_repr: torch.Tensor,
        global_repr: torch.Tensor,
        assignment: torch.Tensor,
        effect_matrix: torch.Tensor,
        student_pref: torch.Tensor,
        teacher_pref: torch.Tensor,
        method_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = node_repr.shape
        method_count = method_repr.shape[1]
        if method_count != self.num_methods:
            raise ValueError(
                f"method_repr method count must be {self.num_methods}, got {method_count}."
            )
        node_features = node_repr.unsqueeze(2).expand(-1, -1, method_count, -1)
        method_features = method_repr.unsqueeze(1).expand(-1, node_count, -1, -1)
        global_features = global_repr[:, None, None, :].expand(
            -1, node_count, method_count, -1
        )
        gather_index = assignment.long().unsqueeze(-1)
        current_effect = torch.gather(effect_matrix, dim=2, index=gather_index)
        current_student = torch.gather(student_pref, dim=2, index=gather_index)
        current_teacher = torch.gather(teacher_pref, dim=2, index=gather_index)
        delta_effect = effect_matrix - current_effect
        delta_student = student_pref - current_student
        delta_teacher = teacher_pref - current_teacher
        preference_mean = 0.5 * (student_pref + teacher_pref)
        preference_gap = torch.abs(student_pref - teacher_pref)
        delta_preference_mean = 0.5 * (delta_student + delta_teacher)
        delta_preference_gap = torch.abs(delta_student - delta_teacher)
        delta_global, delta_soft = self._global_deltas(
            assignment=assignment,
            method_distribution=method_distribution,
            target_distribution=target_distribution,
            node_mask=node_mask,
        )
        active_count = node_mask.to(effect_matrix.dtype).sum(dim=1).clamp_min(1.0)
        local_scale = active_count.view(batch_size, 1, 1)
        delta_objective = (
            self.alpha_effect * delta_effect / local_scale
            + self.alpha_student * delta_student / local_scale
            + self.alpha_teacher * delta_teacher / local_scale
            + self.alpha_global * delta_global
            - delta_soft
        )
        method_ids = torch.arange(method_count, device=assignment.device)
        is_current = (
            method_ids.view(1, 1, method_count)
            == assignment.long().unsqueeze(-1)
        ).to(effect_matrix.dtype)
        scalar_features = torch.stack(
            [
                effect_matrix,
                preference_mean,
                preference_gap,
                delta_effect,
                delta_preference_mean,
                delta_preference_gap,
                delta_objective,
                delta_soft,
                is_current,
            ],
            dim=-1,
        )
        return torch.cat(
            [node_features, method_features, global_features, scalar_features], dim=-1
        )


class ObjectiveDeltaSymmetricStateEncoder(AssignmentStateEncoder):
    """Drop-in state encoder with objective-consistent symmetric preferences."""

    def __init__(
        self,
        *,
        num_categories: int,
        num_methods: int,
        hidden_dim: int,
        category_embedding_dim: int,
        method_embedding_dim: int,
        objective: ObjectiveSpec,
    ) -> None:
        super().__init__(
            num_categories=num_categories,
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            category_embedding_dim=category_embedding_dim,
            method_embedding_dim=method_embedding_dim,
            entropy_min=objective.entropy_min,
            method_cap=objective.method_cap,
            lambda_div=objective.beta_entropy,
            lambda_cap=objective.beta_cap,
            epsilon=objective.epsilon,
        )
        self.node_encoder = SymmetricNodeEncoder(
            num_categories=num_categories,
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            category_embedding_dim=category_embedding_dim,
            method_embedding_dim=method_embedding_dim,
        )
        self.global_encoder = SymmetricGlobalEncoder(
            num_methods=num_methods, hidden_dim=hidden_dim
        )
        self.pair_builder = ObjectiveDeltaSymmetricPairFeatureBuilder(
            objective=objective, num_methods=num_methods
        )


class ObjectiveDeltaSymmetricActorCritic(MaskedActorCritic):
    """Actor-Critic whose preference representation is exactly swap invariant."""

    model_version = SYMMETRIC_MODEL_VERSION

    def __init__(
        self,
        *,
        num_categories: int,
        num_methods: int,
        hidden_dim: int,
        category_embedding_dim: int,
        method_embedding_dim: int,
        objective: ObjectiveSpec,
    ) -> None:
        super().__init__(
            num_categories=num_categories,
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            category_embedding_dim=category_embedding_dim,
            method_embedding_dim=method_embedding_dim,
            entropy_min=objective.entropy_min,
            method_cap=objective.method_cap,
            lambda_div=objective.beta_entropy,
            lambda_cap=objective.beta_cap,
            epsilon=objective.epsilon,
        )
        self.encoder = ObjectiveDeltaSymmetricStateEncoder(
            num_categories=num_categories,
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            category_embedding_dim=category_embedding_dim,
            method_embedding_dim=method_embedding_dim,
            objective=objective,
        )


def model_spec(model_version: str) -> dict[str, Any]:
    if model_version == LEGACY_MODEL_VERSION:
        return {
            "problem_formulation_version": PROBLEM_FORMULATION_VERSION,
            "model_version": LEGACY_MODEL_VERSION,
            "feature_version": LEGACY_FEATURE_VERSION,
            "network_architecture_version": LEGACY_NETWORK_VERSION,
            "preference_representation": "separate_student_teacher_v1",
            "objective_delta_feature": False,
            "student_teacher_exchange_invariant": False,
        }
    if model_version == SYMMETRIC_MODEL_VERSION:
        return {
            "problem_formulation_version": PROBLEM_FORMULATION_VERSION,
            "model_version": SYMMETRIC_MODEL_VERSION,
            "feature_version": SYMMETRIC_FEATURE_VERSION,
            "network_architecture_version": SYMMETRIC_NETWORK_VERSION,
            "preference_representation": "mean_absolute_gap_v2",
            "objective_delta_feature": True,
            "student_teacher_exchange_invariant": True,
        }
    raise ValueError(f"Unsupported model_version: {model_version}")


def build_versioned_actor_critic(
    instance: AssignmentInstance,
    *,
    model_version: str,
    objective: ObjectiveSpec,
    config: PAConfig | None = None,
    device: torch.device | str = "auto",
    hidden_dim: int | None = None,
) -> MaskedActorCritic:
    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(f"Unsupported model_version: {model_version}")
    project_config = config if config is not None else load_config()
    if model_version == LEGACY_MODEL_VERSION:
        model = build_actor_critic(
            instance,
            config=project_config,
            device=device,
            hidden_dim=hidden_dim,
            objective=objective,
        )
    else:
        model_config = project_config.default["model"]
        model = ObjectiveDeltaSymmetricActorCritic(
            num_categories=instance.k,
            num_methods=instance.m,
            hidden_dim=int(hidden_dim or model_config["hidden_dim"]),
            category_embedding_dim=int(model_config["category_embedding_dim"]),
            method_embedding_dim=int(model_config["method_embedding_dim"]),
            objective=objective,
        )
        resolved_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        model.to(resolved_device)
    setattr(model, "model_spec", model_spec(model_version))
    return model


__all__ = [
    "LEGACY_MODEL_VERSION",
    "SYMMETRIC_MODEL_VERSION",
    "PROBLEM_FORMULATION_VERSION",
    "LEGACY_FEATURE_VERSION",
    "SYMMETRIC_FEATURE_VERSION",
    "LEGACY_NETWORK_VERSION",
    "SYMMETRIC_NETWORK_VERSION",
    "SUPPORTED_MODEL_VERSIONS",
    "ObjectiveDeltaSymmetricActorCritic",
    "ObjectiveDeltaSymmetricPairFeatureBuilder",
    "build_versioned_actor_critic",
    "model_spec",
]
