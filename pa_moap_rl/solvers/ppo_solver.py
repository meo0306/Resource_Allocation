"""PA-MOAP 的最小 masked PPO 求解器。

该模块实现单实例 PPO 闭环：环境采样 rollout、用 GAE 估计优势、执行 clipped
PPO 更新，并返回训练过程中环境记录到的 best-so-far assignment。所有动作都
经过 `action_mask`，理论上 illegal action count 应保持为 0。
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pa_moap_rl.checkpointing import checkpoint_metadata, instance_data_hash, method_matrix_hash
from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.actor_critic import MaskedActorCritic
from pa_moap_rl.objective import ObjectiveSpec, legacy_objective_spec
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result


MODEL_INPUT_KEYS = [
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
]


@dataclass(frozen=True)
class PPOConfig:
    """最小 PPO 训练超参数。"""

    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 0.50
    target_kl: float | None = None
    total_updates: int = 4
    seed: int | None = None
    device: str = "auto"
    env_max_steps: int | None = None
    env_patience: int | None = None


@dataclass
class RolloutBatch:
    """一次 rollout 采集到的训练张量和轻量诊断信息。"""

    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_returns: list[float]
    illegal_action_count: int
    best_assignment: np.ndarray
    best_score: float


@dataclass
class InstanceRolloutSummary:
    '''Per-environment diagnostics retained after joint collection.'''

    rewards: torch.Tensor
    episode_returns: list[float]
    illegal_action_count: int
    best_assignment: np.ndarray
    best_score: float


@dataclass
class VectorizedRolloutBatch:
    '''Pooled on-policy transitions from multiple synchronized environments.'''

    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    instance_summaries: list[InstanceRolloutSummary]
    illegal_action_count: int
    environment_count: int
    steps_per_environment: int


def ppo_config_from_project(config: PAConfig | None = None, **overrides: Any) -> PPOConfig:
    """从 YAML 默认值和显式覆盖项构造 `PPOConfig`。"""

    cfg = config if config is not None else load_config()
    ppo = cfg.default["ppo"]
    values = {
        "rollout_steps": int(ppo["rollout_steps"]),
        "update_epochs": int(ppo["update_epochs"]),
        "minibatch_size": int(ppo["minibatch_size"]),
        "gamma": float(ppo["gamma"]),
        "gae_lambda": float(ppo["gae_lambda"]),
        "clip_epsilon": float(ppo["clip_epsilon"]),
        "value_coef": float(ppo["value_coef"]),
        "entropy_coef": float(ppo["entropy_coef"]),
        "learning_rate": float(ppo["learning_rate"]),
        "max_grad_norm": float(ppo["max_grad_norm"]),
    }
    configured_target_kl = ppo.get('target_kl')
    values['target_kl'] = (
        float(configured_target_kl) if configured_target_kl is not None else None
    )
    values.update(overrides)
    return PPOConfig(**values)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """解析训练设备，`auto` 优先使用可用 CUDA。"""

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return resolved


def observation_to_model_batch(obs: dict[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """把单个环境 observation 转为 batch size = 1 的模型输入。"""

    return stack_observations([obs], device=device)


def stack_observations(observations: list[dict[str, np.ndarray]], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """堆叠多个环境 observation，形成模型可直接消费的 Tensor batch。"""

    if not observations:
        raise ValueError("observations must be non-empty.")
    stacked: dict[str, torch.Tensor] = {}
    for key in MODEL_INPUT_KEYS:
        if key not in observations[0]:
            raise ValueError(f"observation missing key {key!r}.")
        array = np.stack([obs[key] for obs in observations], axis=0)
        tensor = torch.as_tensor(array, device=device)
        if key in {"category_id", "assignment"}:
            tensor = tensor.long()
        elif key in {"node_mask", "action_mask"}:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        stacked[key] = tensor
    return stacked


NODE_AXIS_KEYS = {
    'category_id',
    'concept_need',
    'assignment',
    'effect_matrix',
    'student_pref',
    'teacher_pref',
    'node_mask',
    'action_mask',
}


def pad_and_stack_observations(
    observations: list[dict[str, np.ndarray]],
    device: torch.device | str = 'cpu',
) -> dict[str, torch.Tensor]:
    '''Pad variable-node observations and stack them into one model batch.'''

    if not observations:
        raise ValueError('observations must be non-empty.')
    node_counts = [int(np.asarray(obs['category_id']).shape[0]) for obs in observations]
    max_nodes = max(node_counts)
    stacked: dict[str, torch.Tensor] = {}
    for key in MODEL_INPUT_KEYS:
        arrays: list[np.ndarray] = []
        for obs, node_count in zip(observations, node_counts):
            if key not in obs:
                raise ValueError(f'observation missing key {key!r}.')
            array = np.asarray(obs[key])
            if key in NODE_AXIS_KEYS and node_count < max_nodes:
                pad_width = [(0, max_nodes - node_count)]
                pad_width.extend((0, 0) for _ in range(array.ndim - 1))
                array = np.pad(array, pad_width, mode='constant', constant_values=0)
            arrays.append(array)
        tensor = torch.as_tensor(np.stack(arrays, axis=0), device=device)
        if key in {'category_id', 'assignment'}:
            tensor = tensor.long()
        elif key in {'node_mask', 'action_mask'}:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        stacked[key] = tensor
    return stacked


def build_actor_critic(
    instance: AssignmentInstance,
    *,
    config: PAConfig | None = None,
    device: torch.device | str = "auto",
    hidden_dim: int | None = None,
    objective: ObjectiveSpec | None = None,
) -> MaskedActorCritic:
    """根据算例的类别数 K 和方法数 M 创建 Actor-Critic 模型。"""

    cfg = config if config is not None else load_config()
    model_cfg = cfg.default["model"]
    soft = cfg.default["soft_constraints"]
    entropy_min = float(soft["entropy_min"])
    method_cap = float(soft["method_cap"])
    lambda_div = float(soft["lambda_div"])
    lambda_cap = float(soft["lambda_cap"])
    epsilon = float(soft["epsilon"])
    if objective is not None:
        entropy_min = objective.entropy_min
        method_cap = objective.method_cap
        lambda_div = objective.beta_entropy
        lambda_cap = objective.beta_cap
        epsilon = objective.epsilon
    resolved_device = resolve_device(device)
    model = MaskedActorCritic(
        num_categories=instance.k,
        num_methods=instance.m,
        hidden_dim=int(hidden_dim if hidden_dim is not None else model_cfg["hidden_dim"]),
        category_embedding_dim=int(model_cfg["category_embedding_dim"]),
        method_embedding_dim=int(model_cfg["method_embedding_dim"]),
        entropy_min=entropy_min,
        method_cap=method_cap,
        lambda_div=lambda_div,
        lambda_cap=lambda_cap,
        epsilon=epsilon,
    )
    return model.to(resolved_device)


def collect_rollout(
    *,
    env: MethodAssignmentEnv,
    instance: AssignmentInstance,
    model: MaskedActorCritic,
    config: PPOConfig,
    obs: dict[str, np.ndarray] | None = None,
) -> tuple[RolloutBatch, dict[str, np.ndarray]]:
    """从单个环境采集固定长度 rollout。"""

    device = next(model.parameters()).device
    current_obs = obs if obs is not None else env.reset(instance)
    observations: list[dict[str, np.ndarray]] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    episode_returns: list[float] = []
    current_episode_return = 0.0
    illegal_action_count = 0
    rollout_best_assignment = env.best_assignment.copy()
    rollout_best_score = float(env.best_score)

    for _ in range(config.rollout_steps):
        observations.append(current_obs)
        with torch.no_grad():
            sample = model.sample_action(observation_to_model_batch(current_obs, device=device))
        node_index = int(sample["node_index"].item())
        method_index = int(sample["method_index"].item())
        if not bool(current_obs["action_mask"][node_index, method_index]):
            illegal_action_count += 1
        next_obs, reward, done, _ = env.step((node_index, method_index))
        if float(env.best_score) > rollout_best_score:
            rollout_best_score = float(env.best_score)
            rollout_best_assignment = env.best_assignment.copy()

        actions.append(int(sample["flat_action"].item()))
        log_probs.append(float(sample["log_prob"].item()))
        values.append(float(sample["state_value"].item()))
        rewards.append(float(reward))
        dones.append(float(done))
        current_episode_return += float(reward)

        # episode 结束后立即 reset，使一个 rollout 可以跨越多个 episode。
        if done:
            episode_returns.append(current_episode_return)
            current_episode_return = 0.0
            current_obs = env.reset(instance)
        else:
            current_obs = next_obs

    # rollout 末尾的 next_value 用于 bootstrap GAE。
    with torch.no_grad():
        next_value = float(model(observation_to_model_batch(current_obs, device=device)).state_value.squeeze(-1).item())

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    advantages, returns = compute_gae(
        rewards=rewards_t,
        dones=dones_t,
        values=values_t,
        next_value=torch.tensor(next_value, dtype=torch.float32, device=device),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return (
        RolloutBatch(
            observations=stack_observations(observations, device=device),
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            old_log_probs=torch.tensor(log_probs, dtype=torch.float32, device=device),
            values=values_t,
            rewards=rewards_t,
            dones=dones_t,
            returns=returns,
            advantages=advantages,
            episode_returns=episode_returns,
            illegal_action_count=illegal_action_count,
            best_assignment=rollout_best_assignment,
            best_score=rollout_best_score,
        ),
        current_obs,
    )


def collect_vectorized_rollout(
    *,
    envs: list[MethodAssignmentEnv],
    instances: list[AssignmentInstance],
    model: MaskedActorCritic,
    config: PPOConfig,
) -> VectorizedRolloutBatch:
    '''Collect synchronized on-policy trajectories and pool them for joint PPO.'''

    if not envs or len(envs) != len(instances):
        raise ValueError('envs and instances must be non-empty and have equal length.')
    method_counts = {instance.m for instance in instances}
    category_counts = {instance.k for instance in instances}
    if len(method_counts) != 1 or len(category_counts) != 1:
        raise ValueError('All vectorized environments must share method and category counts.')

    device = next(model.parameters()).device
    environment_count = len(envs)
    steps = int(config.rollout_steps)
    current_observations = [
        env.reset(instance) for env, instance in zip(envs, instances)
    ]
    observations_by_env: list[list[dict[str, np.ndarray]]] = [
        [] for _ in envs
    ]
    actions_by_env: list[list[int]] = [[] for _ in envs]
    log_probs_by_env: list[list[float]] = [[] for _ in envs]
    values_by_env: list[list[float]] = [[] for _ in envs]
    rewards_by_env: list[list[float]] = [[] for _ in envs]
    dones_by_env: list[list[float]] = [[] for _ in envs]
    episode_returns: list[list[float]] = [[] for _ in envs]
    current_episode_returns = [0.0 for _ in envs]
    illegal_counts = [0 for _ in envs]
    best_assignments = [env.best_assignment.copy() for env in envs]
    best_scores = [float(env.best_score) for env in envs]

    for _ in range(steps):
        for index, observation in enumerate(current_observations):
            observations_by_env[index].append(observation)
        with torch.inference_mode():
            model_batch = pad_and_stack_observations(
                current_observations,
                device=device,
            )
            sample = model.sample_action(model_batch)
            sampled = torch.stack(
                [
                    sample['flat_action'].float(),
                    sample['log_prob'],
                    sample['state_value'],
                ],
                dim=1,
            ).detach().cpu().numpy()

        for index, (env, instance) in enumerate(zip(envs, instances)):
            flat_action = int(sampled[index, 0])
            node_index = flat_action // instance.m
            method_index = flat_action % instance.m
            action_mask = current_observations[index]['action_mask']
            legal = (
                node_index < instance.n
                and bool(action_mask[node_index, method_index])
            )
            if not legal:
                illegal_counts[index] += 1
            next_observation, reward, done, _ = env.step(
                (node_index, method_index)
            )
            if float(env.best_score) > best_scores[index]:
                best_scores[index] = float(env.best_score)
                best_assignments[index] = env.best_assignment.copy()

            actions_by_env[index].append(flat_action)
            log_probs_by_env[index].append(float(sampled[index, 1]))
            values_by_env[index].append(float(sampled[index, 2]))
            rewards_by_env[index].append(float(reward))
            dones_by_env[index].append(float(done))
            current_episode_returns[index] += float(reward)
            if done:
                episode_returns[index].append(current_episode_returns[index])
                current_episode_returns[index] = 0.0
                current_observations[index] = env.reset(instance)
            else:
                current_observations[index] = next_observation

    with torch.inference_mode():
        next_batch = pad_and_stack_observations(
            current_observations,
            device=device,
        )
        next_values = model(next_batch).state_value.squeeze(-1)

    action_tensors: list[torch.Tensor] = []
    log_prob_tensors: list[torch.Tensor] = []
    value_tensors: list[torch.Tensor] = []
    reward_tensors: list[torch.Tensor] = []
    done_tensors: list[torch.Tensor] = []
    return_tensors: list[torch.Tensor] = []
    advantage_tensors: list[torch.Tensor] = []
    summaries: list[InstanceRolloutSummary] = []
    for index in range(environment_count):
        rewards = torch.tensor(
            rewards_by_env[index],
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            dones_by_env[index],
            dtype=torch.float32,
            device=device,
        )
        values = torch.tensor(
            values_by_env[index],
            dtype=torch.float32,
            device=device,
        )
        advantages, returns = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_value=next_values[index],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        if environment_count > 1 and steps > 1:
            advantages = (
                (advantages - advantages.mean())
                / (advantages.std(unbiased=False) + 1.0e-8)
            )
        action_tensors.append(
            torch.tensor(actions_by_env[index], dtype=torch.long, device=device)
        )
        log_prob_tensors.append(
            torch.tensor(
                log_probs_by_env[index],
                dtype=torch.float32,
                device=device,
            )
        )
        value_tensors.append(values)
        reward_tensors.append(rewards)
        done_tensors.append(dones)
        return_tensors.append(returns)
        advantage_tensors.append(advantages)
        summaries.append(
            InstanceRolloutSummary(
                rewards=rewards,
                episode_returns=episode_returns[index],
                illegal_action_count=illegal_counts[index],
                best_assignment=best_assignments[index],
                best_score=best_scores[index],
            )
        )

    flattened_observations = [
        observation
        for environment_observations in observations_by_env
        for observation in environment_observations
    ]
    return VectorizedRolloutBatch(
        observations=pad_and_stack_observations(
            flattened_observations,
            device=device,
        ),
        actions=torch.cat(action_tensors),
        old_log_probs=torch.cat(log_prob_tensors),
        values=torch.cat(value_tensors),
        rewards=torch.cat(reward_tensors),
        dones=torch.cat(done_tensors),
        returns=torch.cat(return_tensors),
        advantages=torch.cat(advantage_tensors),
        instance_summaries=summaries,
        illegal_action_count=sum(illegal_counts),
        environment_count=environment_count,
        steps_per_environment=steps,
    )


def compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Generalized Advantage Estimation (GAE) 和 returns。"""

    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    # 从后往前递推，遇到 done 时 next_nonterminal 会切断跨 episode 的 bootstrap。
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - dones[step]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[step]
            next_values = values[step + 1]
        delta = rewards[step] + float(gamma) * next_values * next_nonterminal - values[step]
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def update_ppo(
    *,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PPOConfig,
    auxiliary_loss_fn: Any | None = None,
) -> dict[str, float]:
    """基于一个 rollout 执行 clipped PPO 多轮 minibatch 更新。"""

    sample_count = rollout.actions.shape[0]
    minibatch_size = min(int(config.minibatch_size), sample_count)
    advantages = rollout.advantages
    if sample_count > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

    last_metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "loss": 0.0,
    }
    executed_epochs = 0
    optimizer_minibatches = 0
    target_kl_triggered = False
    target_kl_observed = 0.0
    for _epoch_index in range(config.update_epochs):
        if target_kl_triggered:
            break
        permutation = torch.randperm(sample_count, device=rollout.actions.device)
        for start in range(0, sample_count, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            mb_obs = {key: value[indices] for key, value in rollout.observations.items()}
            evaluated = model.evaluate_actions(mb_obs, flat_action=rollout.actions[indices])
            new_log_prob = evaluated["log_prob"]
            entropy = evaluated["entropy"].mean()
            value = evaluated["state_value"]

            # PPO ratio 比较更新后策略与采样时旧策略对同一动作的概率。
            log_ratio = new_log_prob - rollout.old_log_probs[indices]
            ratio = torch.exp(log_ratio)
            mb_advantages = advantages[indices]
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - float(config.clip_epsilon),
                1.0 + float(config.clip_epsilon),
            ) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(value, rollout.returns[indices])
            loss = policy_loss + float(config.value_coef) * value_loss - float(config.entropy_coef) * entropy
            auxiliary_metrics: dict[str, float] = {}
            if auxiliary_loss_fn is not None:
                auxiliary_result = auxiliary_loss_fn(
                    model=model,
                    rollout=rollout,
                    indices=indices,
                    evaluated=evaluated,
                )
                if not isinstance(auxiliary_result, dict) or "loss" not in auxiliary_result:
                    raise ValueError("auxiliary_loss_fn must return a mapping containing loss.")
                auxiliary_loss = auxiliary_result["loss"]
                if not isinstance(auxiliary_loss, torch.Tensor) or auxiliary_loss.ndim != 0:
                    raise ValueError("Auxiliary loss must be a scalar tensor.")
                if not bool(torch.isfinite(auxiliary_loss).item()):
                    raise ValueError("Auxiliary loss must be finite.")
                loss = loss + auxiliary_loss
                for name, value in auxiliary_result.get("metrics", {}).items():
                    metric_name = str(name)
                    if not metric_name.startswith("auxiliary_"):
                        raise ValueError("Auxiliary metric names must start with auxiliary_.")
                    auxiliary_metrics[metric_name] = float(value)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            optimizer.step()
            optimizer_minibatches += 1

            if start + minibatch_size >= sample_count:
                executed_epochs += 1
                if config.target_kl is not None:
                    kl_sum = 0.0
                    with torch.no_grad():
                        for diagnostic_start in range(
                            0, sample_count, minibatch_size
                        ):
                            diagnostic_indices = slice(
                                diagnostic_start,
                                diagnostic_start + minibatch_size,
                            )
                            diagnostic_observations = {
                                key: value[diagnostic_indices]
                                for key, value in rollout.observations.items()
                            }
                            epoch_evaluated = model.evaluate_actions(
                                diagnostic_observations,
                                flat_action=rollout.actions[diagnostic_indices],
                            )
                            kl_sum += float(
                                torch.sum(
                                    rollout.old_log_probs[diagnostic_indices]
                                    - epoch_evaluated['log_prob']
                                ).item()
                            )
                    target_kl_observed = abs(kl_sum / sample_count)
                    target_kl_triggered = (
                        target_kl_observed > float(config.target_kl)
                    )

            with torch.no_grad():
                approx_kl = torch.mean(rollout.old_log_probs[indices] - new_log_prob).abs()
            last_metrics = {
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
                "approx_kl": float(approx_kl.detach().item()),
                "loss": float(loss.detach().item()),
                **auxiliary_metrics,
            }
    with torch.no_grad():
        clip_fraction = torch.mean(
            (torch.abs(ratio - 1.0) > float(config.clip_epsilon)).float()
        )
        target_variance = torch.var(rollout.returns, unbiased=False)
        if float(target_variance.item()) > 1.0e-12:
            residual_variance = torch.var(rollout.returns - rollout.values, unbiased=False)
            explained_variance = 1.0 - residual_variance / target_variance
        else:
            explained_variance = torch.zeros((), device=rollout.returns.device)
    last_metrics['clip_fraction'] = float(clip_fraction.item())
    last_metrics['grad_norm'] = float(grad_norm.detach().item())
    last_metrics['explained_variance'] = float(explained_variance.item())
    last_metrics['advantage_mean'] = float(rollout.advantages.mean().item())
    last_metrics['advantage_std'] = float(rollout.advantages.std(unbiased=False).item())
    last_metrics['return_mean'] = float(rollout.returns.mean().item())
    last_metrics['return_std'] = float(rollout.returns.std(unbiased=False).item())
    last_metrics['ppo_epochs_executed'] = float(executed_epochs)
    last_metrics['optimizer_minibatches'] = float(optimizer_minibatches)
    last_metrics['target_kl_observed'] = float(target_kl_observed)
    last_metrics['target_kl_triggered'] = float(target_kl_triggered)
    return last_metrics


def solve_ppo(
    instance: AssignmentInstance,
    *,
    config: PPOConfig | None = None,
    project_config: PAConfig | None = None,
    model: MaskedActorCritic | None = None,
    log_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """在单个实例上训练 masked PPO，并返回环境记录的 best-so-far 解。"""

    project_cfg = project_config if project_config is not None else load_config()
    ppo_cfg = config if config is not None else ppo_config_from_project(project_cfg)
    soft = project_cfg.default["soft_constraints"]
    effective_objective = objective or legacy_objective_spec(
        instance,
        H_min=float(soft["entropy_min"]),
        pi_cap=float(soft["method_cap"]),
        lambda_div=float(soft["lambda_div"]),
        lambda_cap=float(soft["lambda_cap"]),
        epsilon=float(soft["epsilon"]),
    )
    if ppo_cfg.seed is not None:
        np.random.seed(ppo_cfg.seed)
        torch.manual_seed(ppo_cfg.seed)

    device = resolve_device(ppo_cfg.device)
    env_cfg = project_cfg.default["environment"]
    env = MethodAssignmentEnv(
        max_steps=int(ppo_cfg.env_max_steps if ppo_cfg.env_max_steps is not None else env_cfg["max_steps"]),
        patience=int(ppo_cfg.env_patience if ppo_cfg.env_patience is not None else env_cfg["patience"]),
        improvement_eps=float(env_cfg["improvement_eps"]),
        config=project_cfg,
        objective=effective_objective,
    )
    obs = env.reset(instance)
    global_best_assignment = env.best_assignment.copy()
    global_best_score = float(env.best_score)
    model = model if model is not None else build_actor_critic(
        instance,
        config=project_cfg,
        device=device,
        objective=effective_objective,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo_cfg.learning_rate))

    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    total_illegal = 0
    episode_returns: list[float] = []
    for update in range(int(ppo_cfg.total_updates)):
        rollout, obs = collect_rollout(env=env, instance=instance, model=model, config=ppo_cfg, obs=obs)
        if rollout.best_score > global_best_score:
            global_best_score = rollout.best_score
            global_best_assignment = rollout.best_assignment.copy()
        update_metrics = update_ppo(model=model, optimizer=optimizer, rollout=rollout, config=ppo_cfg)
        total_illegal += rollout.illegal_action_count
        episode_returns.extend(rollout.episode_returns)
        row = {
            "update": update,
            "best_score": float(env.best_score),
            "current_score": float(env.current_scores[-1]),
            "mean_reward": float(rollout.rewards.mean().item()),
            "episode_return": float(np.mean(rollout.episode_returns)) if rollout.episode_returns else 0.0,
            "illegal_action_count": int(rollout.illegal_action_count),
            **update_metrics,
        }
        row['best_score'] = global_best_score
        rows.append(row)

    runtime = time.perf_counter() - start_time
    if log_path is not None:
        write_training_log(rows, log_path)
    if checkpoint_path is not None:
        output = Path(checkpoint_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "model_state_dict": model.state_dict(),
                "config": ppo_cfg.__dict__,
                "ppo_config": ppo_cfg.__dict__,
                "log": rows,
                "provenance": checkpoint_metadata(
                    effective_objective,
                    method_hash=method_matrix_hash(instance),
                    data_hash=instance_data_hash(instance),
                ),
            },
            output,
        )

    result = make_solver_result(
        solver_name="ppo",
        instance=instance,
        assignment=global_best_assignment,
        runtime=runtime,
        history=[float(row["best_score"]) for row in rows],
        objective=effective_objective,
    )
    result.metrics["illegal_action_count"] = int(total_illegal)
    result.metrics["illegal_action_rate"] = float(total_illegal / max(1, int(ppo_cfg.rollout_steps) * int(ppo_cfg.total_updates)))
    result.metrics["episode_return"] = float(np.mean(episode_returns)) if episode_returns else 0.0
    result.metrics["updates"] = int(ppo_cfg.total_updates)
    return result


def write_training_log(rows: list[dict[str, Any]], path: str | Path) -> None:
    """将 PPO 训练日志写为 CSV。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "PPOConfig",
    "RolloutBatch",
    "build_actor_critic",
    "collect_rollout",
    "compute_gae",
    "observation_to_model_batch",
    "ppo_config_from_project",
    "resolve_device",
    "solve_ppo",
    "stack_observations",
    "update_ppo",
    "write_training_log",
]
