"""节点、方法、全局状态和候选动作 pair feature 的编码器。

模型输入来自环境 observation。编码流程是：
节点状态 `[B,N]` -> `node_repr [B,N,H]`；
方法属性 `[B,M,6]` -> `method_repr [B,M,H]`；
整体状态 -> `global_repr [B,H]`；
最后为每个候选替换动作 `(i,m)` 拼接 `pair_feature [B,N,M,3H+9]`。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1.0e-8) -> torch.Tensor:
    """按 bool mask 对指定维度做 mean pooling，忽略 padding 节点。"""

    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = torch.sum(values * weights, dim=dim)
    denominator = torch.sum(weights, dim=dim).clamp_min(eps)
    return numerator / denominator


@dataclass(frozen=True)
class EncoderOutput:
    """完整 encoder 前向传播的中间表示。"""

    node_repr: torch.Tensor
    method_repr: torch.Tensor
    global_repr: torch.Tensor
    pair_feature: torch.Tensor


class NodeEncoder(nn.Module):
    """编码当前 assignment 下的知识点状态。"""

    def __init__(
        self,
        *,
        num_categories: int,
        num_methods: int,
        hidden_dim: int = 64,
        category_embedding_dim: int = 8,
        method_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)
        self.method_embedding = nn.Embedding(num_methods, method_embedding_dim)
        input_dim = category_embedding_dim + 6 + method_embedding_dim + 3
        self.net = _mlp(input_dim, hidden_dim, hidden_dim)

    def forward(
        self,
        category_id: torch.Tensor,
        concept_need: torch.Tensor,
        assignment: torch.Tensor,
        effect_matrix: torch.Tensor,
        student_pref: torch.Tensor,
        teacher_pref: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `node_repr`，shape 为 `[B, N, H]`。"""

        category_emb = self.category_embedding(category_id.long())
        assigned_method_emb = self.method_embedding(assignment.long())
        # 取出每个节点当前已分配方法对应的效果和偏好分数。
        gather_index = assignment.long().unsqueeze(-1)
        current_effect = torch.gather(effect_matrix, dim=2, index=gather_index).to(concept_need.dtype)
        current_student = torch.gather(student_pref, dim=2, index=gather_index).to(concept_need.dtype)
        current_teacher = torch.gather(teacher_pref, dim=2, index=gather_index).to(concept_need.dtype)
        features = torch.cat(
            [
                category_emb,
                concept_need,
                assigned_method_emb,
                current_effect,
                current_student,
                current_teacher,
            ],
            dim=-1,
        )
        return self.net(features)


class MethodEncoder(nn.Module):
    """编码静态教学方法属性。"""

    def __init__(
        self,
        *,
        num_methods: int,
        hidden_dim: int = 64,
        method_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_methods = int(num_methods)
        self.method_embedding = nn.Embedding(num_methods, method_embedding_dim)
        self.net = _mlp(method_embedding_dim + 6, hidden_dim, hidden_dim)

    def forward(self, method_attr: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """返回 `method_repr`，支持 `[M,6]` 单份方法属性或 `[B,M,6]` batch。"""

        if method_attr.ndim == 2:
            method_id = torch.arange(method_attr.shape[0], device=method_attr.device)
            method_emb = self.method_embedding(method_id)
            features = torch.cat([method_emb, method_attr], dim=-1)
            encoded = self.net(features)
            if batch_size is None:
                return encoded
            return encoded.unsqueeze(0).expand(int(batch_size), -1, -1)

        if method_attr.ndim != 3:
            raise ValueError(f"method_attr must be 2D or 3D, got shape {tuple(method_attr.shape)}.")
        batch_size, method_count, _ = method_attr.shape
        method_id = torch.arange(method_count, device=method_attr.device)
        method_emb = self.method_embedding(method_id).unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([method_emb, method_attr], dim=-1)
        return self.net(features)


class GlobalEncoder(nn.Module):
    """由 pooled 节点、方法分布和当前分数编码整体 assignment 状态。"""

    def __init__(self, *, num_methods: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = _mlp(hidden_dim + 3 * int(num_methods) + 6, hidden_dim, hidden_dim)

    def forward(
        self,
        node_repr: torch.Tensor,
        node_mask: torch.Tensor,
        method_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> torch.Tensor:
        """返回 `global_repr`，shape 为 `[B, H]`。"""

        pooled_nodes = masked_mean(node_repr, node_mask, dim=1)
        features = torch.cat(
            [
                pooled_nodes,
                method_distribution,
                target_distribution,
                method_distribution - target_distribution,
                current_scores,
            ],
            dim=-1,
        )
        return self.net(features)


class PairFeatureBuilder(nn.Module):
    """构造候选替换动作特征 `xi_im`，shape 为 `[B,N,M,3H+9]`。"""

    def __init__(
        self,
        *,
        num_methods: int,
        entropy_min: float = 0.55,
        method_cap: float = 0.35,
        lambda_div: float = 1.0,
        lambda_cap: float = 1.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.num_methods = int(num_methods)
        self.entropy_min = float(entropy_min)
        self.method_cap = float(method_cap)
        self.lambda_div = float(lambda_div)
        self.lambda_cap = float(lambda_cap)
        self.epsilon = float(epsilon)

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
        """为每个 `(节点 i, 候选方法 m)` 拼接局部、方法、全局和标量增量特征。"""

        batch_size, node_count, hidden_dim = node_repr.shape
        method_count = method_repr.shape[1]
        if method_count != self.num_methods:
            raise ValueError(f"method_repr method count must be {self.num_methods}, got {method_count}.")

        # 将节点、方法和全局表示广播到每个候选动作 `(i, m)`。
        h = node_repr.unsqueeze(2).expand(-1, -1, method_count, -1)
        mu = method_repr.unsqueeze(1).expand(-1, node_count, -1, -1)
        g = global_repr[:, None, None, :].expand(-1, node_count, method_count, -1)

        gather_index = assignment.long().unsqueeze(-1)
        current_effect = torch.gather(effect_matrix, dim=2, index=gather_index)
        current_student = torch.gather(student_pref, dim=2, index=gather_index)
        current_teacher = torch.gather(teacher_pref, dim=2, index=gather_index)

        # 标量增量衡量“从当前方法换到候选方法”对局部分项的直接影响。
        delta_effect = effect_matrix - current_effect
        delta_student = student_pref - current_student
        delta_teacher = teacher_pref - current_teacher

        delta_global, delta_soft = self._global_deltas(
            assignment=assignment,
            method_distribution=method_distribution,
            target_distribution=target_distribution,
            node_mask=node_mask,
        )

        method_ids = torch.arange(method_count, device=assignment.device)
        is_current = (method_ids.view(1, 1, method_count) == assignment.long().unsqueeze(-1)).to(effect_matrix.dtype)

        scalar_features = torch.stack(
            [
                effect_matrix,
                student_pref,
                teacher_pref,
                delta_effect,
                delta_student,
                delta_teacher,
                delta_global,
                delta_soft,
                is_current,
            ],
            dim=-1,
        )
        return torch.cat([h, mu, g, scalar_features], dim=-1)

    def _global_deltas(
        self,
        *,
        assignment: torch.Tensor,
        method_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """估计每个候选替换动作对 `F_G` 和软惩罚项的分布层面增量。"""

        batch_size, node_count = assignment.shape
        method_count = method_distribution.shape[1]
        current_global = self._global_score(method_distribution, target_distribution)
        current_soft = self._soft_penalty(method_distribution)

        active_count = node_mask.to(method_distribution.dtype).sum(dim=1).clamp_min(1.0)
        one_hot_current = torch.nn.functional.one_hot(assignment.long(), num_classes=method_count).to(method_distribution.dtype)
        method_eye = torch.eye(method_count, dtype=method_distribution.dtype, device=assignment.device)
        # 单点替换会从当前方法减去 1/N_active，并给候选方法加上 1/N_active。
        candidate = method_distribution[:, None, None, :] + (
            method_eye.view(1, 1, method_count, method_count) - one_hot_current[:, :, None, :]
        ) / active_count.view(batch_size, 1, 1, 1)
        candidate = candidate.clamp_min(0.0)

        target = target_distribution[:, None, None, :]
        candidate_global = 1.0 - 0.5 * torch.sum(torch.abs(candidate - target), dim=-1)
        flat_candidate = candidate.reshape(batch_size * node_count * method_count, method_count)
        candidate_soft = self._soft_penalty(flat_candidate).reshape(batch_size, node_count, method_count)

        delta_global = candidate_global - current_global.view(batch_size, 1, 1)
        delta_soft = candidate_soft - current_soft.view(batch_size, 1, 1)
        return delta_global, delta_soft

    def _global_score(self, distribution: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
        return 1.0 - 0.5 * torch.sum(torch.abs(distribution - target_distribution), dim=-1)

    def _soft_penalty(self, distribution: torch.Tensor) -> torch.Tensor:
        method_count = distribution.shape[-1]
        if method_count <= 1:
            entropy = torch.zeros(distribution.shape[:-1], dtype=distribution.dtype, device=distribution.device)
        else:
            entropy = -torch.sum(distribution * torch.log(distribution + self.epsilon), dim=-1) / torch.log(
                torch.tensor(float(method_count), dtype=distribution.dtype, device=distribution.device)
            )
            entropy = entropy.clamp(0.0, 1.0)
        div_violation = torch.clamp(self.entropy_min - entropy, min=0.0).pow(2)
        cap_violation = torch.clamp(distribution - self.method_cap, min=0.0).pow(2).sum(dim=-1)
        return self.lambda_div * div_violation + self.lambda_cap * cap_violation


class AssignmentStateEncoder(nn.Module):
    """组合式状态编码器，统一返回节点、方法、全局和 pair feature。"""

    def __init__(
        self,
        *,
        num_categories: int,
        num_methods: int,
        hidden_dim: int = 64,
        category_embedding_dim: int = 8,
        method_embedding_dim: int = 8,
        entropy_min: float = 0.55,
        method_cap: float = 0.35,
        lambda_div: float = 1.0,
        lambda_cap: float = 1.0,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.node_encoder = NodeEncoder(
            num_categories=num_categories,
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            category_embedding_dim=category_embedding_dim,
            method_embedding_dim=method_embedding_dim,
        )
        self.method_encoder = MethodEncoder(
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            method_embedding_dim=method_embedding_dim,
        )
        self.global_encoder = GlobalEncoder(num_methods=num_methods, hidden_dim=hidden_dim)
        self.pair_builder = PairFeatureBuilder(
            num_methods=num_methods,
            entropy_min=entropy_min,
            method_cap=method_cap,
            lambda_div=lambda_div,
            lambda_cap=lambda_cap,
            epsilon=epsilon,
        )

    def forward(
        self,
        *,
        category_id: torch.Tensor,
        concept_need: torch.Tensor,
        method_attr: torch.Tensor,
        assignment: torch.Tensor,
        effect_matrix: torch.Tensor,
        student_pref: torch.Tensor,
        teacher_pref: torch.Tensor,
        node_mask: torch.Tensor,
        method_distribution: torch.Tensor,
        target_distribution: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> EncoderOutput:
        """执行完整编码链路，供 Actor-Critic 直接消费。"""

        node_repr = self.node_encoder(
            category_id=category_id,
            concept_need=concept_need,
            assignment=assignment,
            effect_matrix=effect_matrix,
            student_pref=student_pref,
            teacher_pref=teacher_pref,
        )
        method_repr = self.method_encoder(method_attr=method_attr, batch_size=category_id.shape[0])
        global_repr = self.global_encoder(
            node_repr=node_repr,
            node_mask=node_mask,
            method_distribution=method_distribution,
            target_distribution=target_distribution,
            current_scores=current_scores,
        )
        pair_feature = self.pair_builder(
            node_repr=node_repr,
            method_repr=method_repr,
            global_repr=global_repr,
            assignment=assignment,
            effect_matrix=effect_matrix,
            student_pref=student_pref,
            teacher_pref=teacher_pref,
            method_distribution=method_distribution,
            target_distribution=target_distribution,
            node_mask=node_mask,
        )
        return EncoderOutput(
            node_repr=node_repr,
            method_repr=method_repr,
            global_repr=global_repr,
            pair_feature=pair_feature,
        )


__all__ = [
    "AssignmentStateEncoder",
    "EncoderOutput",
    "GlobalEncoder",
    "MethodEncoder",
    "NodeEncoder",
    "PairFeatureBuilder",
    "masked_mean",
]
