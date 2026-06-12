"""用于教学方法分配的 masked Actor-Critic 网络。

Actor 对每个候选替换动作 `(i,m)` 输出 logits，并使用 `action_mask` 将非法动作
置为极小值；Critic 只基于全局状态表示估计 state value。动作在采样和 PPO
更新时使用扁平编号 `flat = i * M + m`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from pa_moap_rl.models.encoders import AssignmentStateEncoder, EncoderOutput


MASKED_LOGIT_VALUE = -1.0e9


@dataclass(frozen=True)
class ActorCriticOutput:
    """Masked Actor-Critic 前向传播输出。"""

    action_logits: torch.Tensor
    masked_logits: torch.Tensor
    state_value: torch.Tensor
    encoder_output: EncoderOutput


def flatten_action(node_index: torch.Tensor, method_index: torch.Tensor, num_methods: int) -> torch.Tensor:
    """将动作 `(i, m)` 映射为扁平动作编号 `i * M + m`。"""

    return node_index.long() * int(num_methods) + method_index.long()


def unflatten_action(flat_action: torch.Tensor, num_methods: int) -> tuple[torch.Tensor, torch.Tensor]:
    """将扁平动作编号还原为 `(i, m)`。"""

    flat = flat_action.long()
    return flat // int(num_methods), flat % int(num_methods)


class MaskedActorCritic(nn.Module):
    """面向单点方法替换动作的 masked Actor-Critic 网络。"""

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
        self.num_methods = int(num_methods)
        self.hidden_dim = int(hidden_dim)
        self.encoder = AssignmentStateEncoder(
            num_categories=num_categories,
            num_methods=num_methods,
            hidden_dim=hidden_dim,
            category_embedding_dim=category_embedding_dim,
            method_embedding_dim=method_embedding_dim,
            entropy_min=entropy_min,
            method_cap=method_cap,
            lambda_div=lambda_div,
            lambda_cap=lambda_cap,
            epsilon=epsilon,
        )
        pair_dim = 3 * hidden_dim + 9
        self.actor = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, Any] | None = None, **kwargs: Any) -> ActorCriticOutput:
        """执行前向传播，并用 `action_mask` 屏蔽非法 actor logits。"""

        inputs = dict(batch or {})
        inputs.update(kwargs)
        tensor_inputs = self._tensorize_inputs(inputs)
        action_mask = tensor_inputs.pop("action_mask")

        encoder_output = self.encoder(**tensor_inputs)
        action_logits = self.actor(encoder_output.pair_feature).squeeze(-1)
        self._validate_action_mask(action_mask, action_logits)
        if not torch.all(action_mask.reshape(action_mask.shape[0], -1).any(dim=1)):
            raise ValueError("Each batch item must have at least one legal action.")
        # Categorical 只从 masked_logits 采样，非法动作概率近似为 0。
        masked_logits = action_logits.masked_fill(~action_mask.bool(), MASKED_LOGIT_VALUE)
        state_value = self.critic(encoder_output.global_repr)
        return ActorCriticOutput(
            action_logits=action_logits,
            masked_logits=masked_logits,
            state_value=state_value,
            encoder_output=encoder_output,
        )

    def sample_action(self, batch: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        """从 masked policy 中采样合法动作，并同时返回扁平编号和 `(i,m)`。"""

        output = self.forward(batch, **kwargs)
        distribution = self._distribution(output.masked_logits)
        flat_action = distribution.sample()
        node_index, method_index = unflatten_action(flat_action, self.num_methods)
        return {
            "flat_action": flat_action,
            "node_index": node_index,
            "method_index": method_index,
            "log_prob": distribution.log_prob(flat_action),
            "entropy": distribution.entropy(),
            "state_value": output.state_value.squeeze(-1),
            "masked_logits": output.masked_logits,
        }

    def evaluate_actions(
        self,
        batch: dict[str, Any] | None = None,
        *,
        flat_action: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """评估给定扁平动作的 log-probability、entropy 和 state value。"""

        output = self.forward(batch, **kwargs)
        distribution = self._distribution(output.masked_logits)
        flat_action = flat_action.long()
        return {
            "log_prob": distribution.log_prob(flat_action),
            "entropy": distribution.entropy(),
            "state_value": output.state_value.squeeze(-1),
            "masked_logits": output.masked_logits,
        }

    def _distribution(self, masked_logits: torch.Tensor) -> Categorical:
        batch_size = masked_logits.shape[0]
        return Categorical(logits=masked_logits.reshape(batch_size, -1))

    def _tensorize_inputs(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """把 observation 字典统一转为当前模型设备上的 Tensor。"""

        required = {
            "category_id",
            "concept_need",
            "method_attr",
            "assignment",
            "effect_matrix",
            "student_pref",
            "teacher_pref",
            "node_mask",
            "method_distribution",
            "target_distribution",
            "current_scores",
            "action_mask",
        }
        missing = sorted(required - set(inputs))
        if missing:
            raise ValueError(f"Missing Actor-Critic inputs: {', '.join(missing)}.")

        device = next(self.parameters()).device
        tensors: dict[str, torch.Tensor] = {}
        for name in required:
            value = inputs[name]
            if isinstance(value, torch.Tensor):
                tensor = value.to(device)
            else:
                tensor = torch.as_tensor(value, device=device)
            tensors[name] = tensor

        tensors["category_id"] = tensors["category_id"].long()
        tensors["assignment"] = tensors["assignment"].long()
        tensors["node_mask"] = tensors["node_mask"].bool()
        tensors["action_mask"] = tensors["action_mask"].bool()
        for name in (
            "concept_need",
            "method_attr",
            "effect_matrix",
            "student_pref",
            "teacher_pref",
            "method_distribution",
            "target_distribution",
            "current_scores",
        ):
            tensors[name] = tensors[name].float()
        return tensors

    @staticmethod
    def _validate_action_mask(action_mask: torch.Tensor, action_logits: torch.Tensor) -> None:
        if action_mask.shape != action_logits.shape:
            raise ValueError(
                f"action_mask shape {tuple(action_mask.shape)} must equal action_logits shape {tuple(action_logits.shape)}."
            )


ActorCritic = MaskedActorCritic


__all__ = [
    "ActorCritic",
    "ActorCriticOutput",
    "MASKED_LOGIT_VALUE",
    "MaskedActorCritic",
    "flatten_action",
    "unflatten_action",
]
