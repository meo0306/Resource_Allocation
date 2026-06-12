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

from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.models.actor_critic import MaskedActorCritic
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


def build_actor_critic(
    instance: AssignmentInstance,
    *,
    config: PAConfig | None = None,
    device: torch.device | str = "auto",
    hidden_dim: int | None = None,
) -> MaskedActorCritic:
    """根据算例的类别数 K 和方法数 M 创建 Actor-Critic 模型。"""

    cfg = config if config is not None else load_config()
    model_cfg = cfg.default["model"]
    soft = cfg.default["soft_constraints"]
    resolved_device = resolve_device(device)
    model = MaskedActorCritic(
        num_categories=instance.k,
        num_methods=instance.m,
        hidden_dim=int(hidden_dim if hidden_dim is not None else model_cfg["hidden_dim"]),
        category_embedding_dim=int(model_cfg["category_embedding_dim"]),
        method_embedding_dim=int(model_cfg["method_embedding_dim"]),
        entropy_min=float(soft["entropy_min"]),
        method_cap=float(soft["method_cap"]),
        lambda_div=float(soft["lambda_div"]),
        lambda_cap=float(soft["lambda_cap"]),
        epsilon=float(soft["epsilon"]),
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

    for _ in range(config.rollout_steps):
        observations.append(current_obs)
        with torch.no_grad():
            sample = model.sample_action(observation_to_model_batch(current_obs, device=device))
        node_index = int(sample["node_index"].item())
        method_index = int(sample["method_index"].item())
        if not bool(current_obs["action_mask"][node_index, method_index]):
            illegal_action_count += 1
        next_obs, reward, done, _ = env.step((node_index, method_index))

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
        ),
        current_obs,
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
    for _ in range(config.update_epochs):
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean(rollout.old_log_probs[indices] - new_log_prob).abs()
            last_metrics = {
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
                "approx_kl": float(approx_kl.detach().item()),
                "loss": float(loss.detach().item()),
            }
    return last_metrics


def solve_ppo(
    instance: AssignmentInstance,
    *,
    config: PPOConfig | None = None,
    project_config: PAConfig | None = None,
    model: MaskedActorCritic | None = None,
    log_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> SolverResult:
    """在单个实例上训练 masked PPO，并返回环境记录的 best-so-far 解。"""

    project_cfg = project_config if project_config is not None else load_config()
    ppo_cfg = config if config is not None else ppo_config_from_project(project_cfg)
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
    )
    obs = env.reset(instance)
    model = model if model is not None else build_actor_critic(instance, config=project_cfg, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo_cfg.learning_rate))

    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    total_illegal = 0
    episode_returns: list[float] = []
    for update in range(int(ppo_cfg.total_updates)):
        rollout, obs = collect_rollout(env=env, instance=instance, model=model, config=ppo_cfg, obs=obs)
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
        rows.append(row)

    runtime = time.perf_counter() - start_time
    if log_path is not None:
        write_training_log(rows, log_path)
    if checkpoint_path is not None:
        output = Path(checkpoint_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": ppo_cfg.__dict__, "log": rows}, output)

    result = make_solver_result(
        solver_name="ppo",
        instance=instance,
        assignment=env.best_assignment,
        runtime=runtime,
        history=[float(row["best_score"]) for row in rows],
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
