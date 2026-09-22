"""跨多个 assignment instance 训练一个共享 masked PPO 模型。

与 `train_ppo.py` 的单实例训练不同，这里会在多个实例之间采样 rollout，
用同一个 Actor-Critic 更新参数，并定期在 eval 实例上运行确定性 policy
评估。输出包括训练日志、评估日志、checkpoint、best assignment 和可选
TensorBoard scalars。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pa_moap_rl.checkpointing import checkpoint_metadata, collection_data_hash, method_matrix_hash
from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.run_batch import discover_instance_paths
from pa_moap_rl.models.actor_critic import MaskedActorCritic, unflatten_action
from pa_moap_rl.objective import ObjectiveSpec, legacy_objective_spec
from pa_moap_rl.solvers.greedy_solver import effect_only_greedy, scalarized_independent_greedy
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.ppo_solver import (
    PPOConfig,
    build_actor_critic,
    collect_rollout,
    observation_to_model_batch,
    ppo_config_from_project,
    resolve_device,
    update_ppo,
)
from pa_moap_rl.solvers.random_solver import random_legal
from pa_moap_rl.utils.metrics import SolverResult, make_solver_result


@dataclass(frozen=True)
class SharedBatchTrainingResult:
    """一次共享 PPO 批训练的输出路径和最终日志行。"""

    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    checkpoint_path: Path
    train_log_path: Path
    eval_log_path: Path
    assignment_dir: Path
    tensorboard_dir: Path | None = None


def _write_rows_legacy(rows: list[dict[str, Any]], path: str | Path) -> None:
    """写出 train/eval CSV，字段集合由所有行共同决定。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "update",
        "phase",
        "group",
        "instance_name",
        "instance_path",
        "seed",
        "solver_name",
    ]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _replace_with_retry(
    temporary: Path,
    output: Path,
    *,
    attempts: int = 50,
    delay_seconds: float = 0.1,
) -> None:
    '''Replace output while tolerating short-lived Windows reader locks.'''

    for attempt in range(attempts):
        try:
            temporary.replace(output)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay_seconds)


def _write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    '''Atomically replace a CSV so concurrent readers never see partial data.'''

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    if not rows:
        temporary.write_text('', encoding='utf-8')
        _replace_with_retry(temporary, output)
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        'update',
        'phase',
        'group',
        'instance_name',
        'instance_path',
        'seed',
        'solver_name',
    ]
    preferred = [field for field in preferred if field in fieldnames]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _replace_with_retry(temporary, output)


def _create_summary_writer(tensorboard_dir: str | Path | None) -> Any | None:
    """按需创建 TensorBoard SummaryWriter。"""

    if tensorboard_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the tensorboard package. "
            "Install it with: python -m pip install tensorboard"
        ) from exc

    output = Path(tensorboard_dir)
    output.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(output))


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))


def _write_train_tensorboard_scalars(writer: Any | None, row: dict[str, Any]) -> None:
    """把训练聚合行写入 TensorBoard。"""

    if writer is None:
        return
    update = int(row["update"])
    for key in (
        "best_score",
        "current_score",
        "mean_reward",
        "episode_return",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "loss",
        "illegal_action_rate",
        "elapsed_sec",
    ):
        value = row.get(key)
        if _is_scalar(value):
            writer.add_scalar(f"train/{key}", float(value), update)
    writer.flush()


def _write_eval_tensorboard_scalars(writer: Any | None, rows: list[dict[str, Any]], update: int) -> None:
    """把一次评估结果按 solver 聚合后写入 TensorBoard。"""

    if writer is None or not rows:
        return
    by_solver: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        solver_name = str(row.get("solver_name") or "unknown")
        by_solver.setdefault(solver_name, []).append(row)

    for solver_name, solver_rows in by_solver.items():
        total_scores = [float(row["total_score"]) for row in solver_rows if _is_scalar(row.get("total_score"))]
        hard_feasible = [
            float(row["hard_feasible_rate"]) for row in solver_rows if _is_scalar(row.get("hard_feasible_rate"))
        ]
        if total_scores:
            writer.add_scalar(f"eval/{solver_name}/total_score", float(np.mean(total_scores)), int(update))
        if hard_feasible:
            writer.add_scalar(f"eval/{solver_name}/hard_feasible_rate", float(np.mean(hard_feasible)), int(update))
    writer.flush()


def _build_env(
    project_config: PAConfig,
    ppo_config: PPOConfig,
    objective: ObjectiveSpec | None = None,
) -> MethodAssignmentEnv:
    """根据项目配置和 PPO 覆盖项创建环境。"""

    env_cfg = project_config.default["environment"]
    return MethodAssignmentEnv(
        max_steps=int(ppo_config.env_max_steps if ppo_config.env_max_steps is not None else env_cfg["max_steps"]),
        patience=int(ppo_config.env_patience if ppo_config.env_patience is not None else env_cfg["patience"]),
        improvement_eps=float(env_cfg["improvement_eps"]),
        config=project_config,
        objective=objective,
    )


def _assert_shared_shape(instances: list[AssignmentInstance]) -> None:
    """确认共享模型可用于所有实例：方法数 M 和类别数 K 必须一致。"""

    if not instances:
        raise ValueError("instances must be non-empty.")
    m = instances[0].m
    k = instances[0].k
    mismatched = [instance.instance_name for instance in instances if instance.m != m or instance.k != k]
    if mismatched:
        raise ValueError(
            "Shared PPO v1 requires all instances to have the same M and K; "
            f"mismatched examples: {mismatched[:5]}"
        )


def _load_instances(paths: list[Path]) -> list[AssignmentInstance]:
    """加载实例并校验共享模型所需的 shape 一致性。"""

    instances = [load_instance_json(path) for path in paths]
    _assert_shared_shape(instances)
    return instances


def _policy_rollout(
    instance: AssignmentInstance,
    *,
    model: MaskedActorCritic,
    project_config: PAConfig,
    ppo_config: PPOConfig,
    deterministic: bool = True,
    objective: ObjectiveSpec | None = None,
) -> SolverResult:
    """无梯度运行一次 policy episode，并返回环境记录的 best assignment。"""

    start = time.perf_counter()
    env = _build_env(project_config, ppo_config, objective)
    obs = env.reset(instance)
    device = next(model.parameters()).device

    while not env.done:
        with torch.no_grad():
            batch = observation_to_model_batch(obs, device=device)
            if deterministic:
                output = model(batch)
                flat_action = torch.argmax(output.masked_logits.reshape(1, -1), dim=1)
            else:
                flat_action = model.sample_action(batch)["flat_action"]
        node_index, method_index = unflatten_action(flat_action, instance.m)
        obs, _, _, _ = env.step((int(node_index.item()), int(method_index.item())))

    runtime = time.perf_counter() - start
    return make_solver_result(
        solver_name="ppo_shared_policy",
        instance=instance,
        assignment=env.best_assignment,
        runtime=runtime,
        history=[float(env.best_score)],
        objective=objective,
    )


def _result_row(
    *,
    update: int,
    phase: str,
    instance_path: Path,
    instance: AssignmentInstance,
    result: SolverResult,
    seed: int,
) -> dict[str, Any]:
    """将单个评估结果转为 eval CSV 行。"""

    row = {
        "update": int(update),
        "phase": phase,
        "group": instance_path.parent.name,
        "instance_name": instance.instance_name,
        "instance_path": str(instance_path),
        "n": int(instance.n),
        "m": int(instance.m),
        "k": int(instance.k),
        "seed": int(seed),
        "solver_name": result.solver_name,
    }
    for key, value in result.metrics.items():
        if key != "solver_name":
            row[key] = value
    return row


def _baseline_results(
    instance: AssignmentInstance,
    *,
    seed: int,
    include_local_search: bool,
    objective: ObjectiveSpec | None = None,
) -> list[SolverResult]:
    """返回 eval 阶段要对比的 baseline 结果。"""

    results = [
        random_legal(instance, seed=seed, objective=objective),
        effect_only_greedy(instance, objective=objective),
        scalarized_independent_greedy(instance, objective=objective),
    ]
    if include_local_search:
        results.append(
            local_search_best_improvement(
                instance, max_iter=instance.n * instance.m, objective=objective
            )
        )
    return results


def _evaluate(
    *,
    update: int,
    eval_paths: list[Path],
    eval_instances: list[AssignmentInstance],
    model: MaskedActorCritic,
    project_config: PAConfig,
    ppo_config: PPOConfig,
    seed: int,
    assignment_dir: Path,
    include_baselines: bool,
    include_local_search: bool,
    objective: ObjectiveSpec | None = None,
) -> list[dict[str, Any]]:
    """在 eval 实例上运行当前 policy 和可选 baseline，并写出 best assignment。"""

    rows: list[dict[str, Any]] = []
    update_dir = assignment_dir / f"update_{int(update):04d}"
    update_dir.mkdir(parents=True, exist_ok=True)

    for path, instance in zip(eval_paths, eval_instances):
        policy_result = _policy_rollout(
            instance,
            model=model,
            project_config=project_config,
            ppo_config=ppo_config,
            deterministic=True,
            objective=objective,
        )
        assignment_path = update_dir / f"{instance.instance_name}.assignment.json"
        assignment_path.write_text(
            json.dumps(
                {
                    "solver_name": policy_result.solver_name,
                    "instance_name": instance.instance_name,
                    "assignment": policy_result.assignment.astype(int).tolist(),
                    "metrics": policy_result.metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        policy_row = _result_row(
            update=update,
            phase="eval",
            instance_path=path,
            instance=instance,
            result=policy_result,
            seed=seed,
        )
        policy_row["best_assignment_path"] = str(assignment_path)
        rows.append(policy_row)

        if include_baselines:
            for result in _baseline_results(
                instance,
                seed=seed,
                include_local_search=include_local_search,
                objective=objective,
            ):
                rows.append(
                    _result_row(
                        update=update,
                        phase="baseline",
                        instance_path=path,
                        instance=instance,
                        result=result,
                        seed=seed,
                    )
                )
    return rows


def run_shared_batch_training(
    *,
    instance_root: str | Path = "instance",
    output_dir: str | Path = "results/ppo_batch",
    group: str | None = None,
    limit: int | None = None,
    eval_limit: int = 3,
    seed: int = 0,
    device: str = "auto",
    total_updates: int | None = None,
    instances_per_update: int = 2,
    rollout_steps: int | None = None,
    minibatch_size: int | None = None,
    update_epochs: int | None = None,
    eval_interval: int = 1,
    include_baselines: bool = True,
    include_local_search: bool = True,
    hidden_dim: int | None = None,
    env_max_steps: int | None = None,
    env_patience: int | None = None,
    tensorboard_dir: str | Path | None = None,
    objective: ObjectiveSpec | None = None,
) -> SharedBatchTrainingResult:
    """通过循环采样多个实例 rollout 来训练一个共享模型。"""

    if instances_per_update <= 0:
        raise ValueError("instances_per_update must be positive.")
    if eval_limit < 0:
        raise ValueError("eval_limit must be non-negative.")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")

    project_config = load_config()
    paths = discover_instance_paths(instance_root, limit=limit, group=group)
    instances = _load_instances(paths)
    # 前 eval_count 个实例固定为评估集，其余用于训练；若实例太少则复用全集训练。
    eval_count = min(int(eval_limit), len(instances))
    eval_paths = paths[:eval_count]
    eval_instances = instances[:eval_count]
    train_paths = paths[eval_count:] or paths
    train_instances = instances[eval_count:] or instances
    soft = project_config.default["soft_constraints"]
    effective_objective = objective or legacy_objective_spec(
        instances[0],
        H_min=float(soft["entropy_min"]),
        pi_cap=float(soft["method_cap"]),
        lambda_div=float(soft["lambda_div"]),
        lambda_cap=float(soft["lambda_cap"]),
        epsilon=float(soft["epsilon"]),
    )

    overrides: dict[str, Any] = {
        "seed": int(seed),
        "device": str(resolve_device(device)),
    }
    if total_updates is not None:
        overrides["total_updates"] = int(total_updates)
    if rollout_steps is not None:
        overrides["rollout_steps"] = int(rollout_steps)
    if minibatch_size is not None:
        overrides["minibatch_size"] = int(minibatch_size)
    if update_epochs is not None:
        overrides["update_epochs"] = int(update_epochs)
    if env_max_steps is not None:
        overrides["env_max_steps"] = int(env_max_steps)
    if env_patience is not None:
        overrides["env_patience"] = int(env_patience)
    ppo_config = ppo_config_from_project(project_config, **overrides)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = resolve_device(ppo_config.device)
    model = build_actor_critic(
        train_instances[0],
        config=project_config,
        device=resolved_device,
        hidden_dim=hidden_dim,
        objective=effective_objective,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo_config.learning_rate))

    output = Path(output_dir)
    train_log_path = output / "train_log.csv"
    eval_log_path = output / "eval_log.csv"
    checkpoint_path = output / "checkpoint.pt"
    assignment_dir = output / "best_assignments"
    tensorboard_path = Path(tensorboard_dir) if tensorboard_dir is not None else None
    writer = _create_summary_writer(tensorboard_path)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    rng = random.Random(seed)

    if eval_instances:
        # update=0 先评估随机初始化模型和 baseline，作为训练前基线。
        eval_rows.extend(
            _evaluate(
                update=0,
                eval_paths=eval_paths,
                eval_instances=eval_instances,
                model=model,
                project_config=project_config,
                ppo_config=ppo_config,
                seed=seed,
                assignment_dir=assignment_dir,
                include_baselines=include_baselines,
                include_local_search=include_local_search,
                objective=effective_objective,
            )
        )
        _write_eval_tensorboard_scalars(writer, eval_rows, update=0)
    _write_rows(train_rows, train_log_path)
    _write_rows(eval_rows, eval_log_path)

    for update in range(1, int(ppo_config.total_updates) + 1):
        # 每个 update 随机抽若干训练实例，各自采 rollout 后立即用共享模型更新。
        sample_count = min(int(instances_per_update), len(train_instances))
        sampled_indices = rng.sample(range(len(train_instances)), k=sample_count)
        per_instance_rows: list[dict[str, Any]] = []
        total_illegal = 0

        for index in sampled_indices:
            instance = train_instances[index]
            instance_path = train_paths[index]
            env = _build_env(project_config, ppo_config, effective_objective)
            rollout, _ = collect_rollout(env=env, instance=instance, model=model, config=ppo_config)
            update_metrics = update_ppo(model=model, optimizer=optimizer, rollout=rollout, config=ppo_config)
            total_illegal += int(rollout.illegal_action_count)
            per_instance_rows.append(
                {
                    "update": int(update),
                    "phase": "train",
                    "group": instance_path.parent.name,
                    "instance_name": instance.instance_name,
                    "instance_path": str(instance_path),
                    "n": int(instance.n),
                    "m": int(instance.m),
                    "k": int(instance.k),
                    "seed": int(seed),
                    "solver_name": "ppo_shared",
                    "rollout_steps": int(ppo_config.rollout_steps),
                    "best_score": float(env.best_score),
                    "current_score": float(env.current_scores[-1]),
                    "mean_reward": float(rollout.rewards.mean().item()),
                    "episode_return": float(np.mean(rollout.episode_returns)) if rollout.episode_returns else 0.0,
                    "illegal_action_count": int(rollout.illegal_action_count),
                    **update_metrics,
                }
            )

        elapsed_sec = time.perf_counter() - start_time
        # train_summary 行用于快速看整体训练趋势和 TensorBoard 曲线。
        aggregate = {
            "update": int(update),
            "phase": "train_summary",
            "group": group or "all",
            "instance_name": "shared_batch",
            "instance_path": "",
            "n": "",
            "m": int(train_instances[0].m),
            "k": int(train_instances[0].k),
            "seed": int(seed),
            "solver_name": "ppo_shared",
            "instances_per_update": sample_count,
            "rollout_steps": int(ppo_config.rollout_steps),
            "elapsed_sec": float(elapsed_sec),
            "illegal_action_count": int(total_illegal),
            "illegal_action_rate": float(total_illegal / max(1, sample_count * int(ppo_config.rollout_steps))),
        }
        for key in (
            "best_score",
            "current_score",
            "mean_reward",
            "episode_return",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "loss",
        ):
            aggregate[key] = float(np.mean([row[key] for row in per_instance_rows]))
        train_rows.extend(per_instance_rows)
        train_rows.append(aggregate)
        _write_train_tensorboard_scalars(writer, aggregate)
        _write_rows(train_rows, train_log_path)

        if eval_instances and (update % int(eval_interval) == 0 or update == int(ppo_config.total_updates)):
            new_eval_rows = _evaluate(
                update=update,
                eval_paths=eval_paths,
                eval_instances=eval_instances,
                model=model,
                project_config=project_config,
                ppo_config=ppo_config,
                seed=seed,
                assignment_dir=assignment_dir,
                include_baselines=include_baselines,
                include_local_search=include_local_search,
                objective=effective_objective,
            )
            eval_rows.extend(new_eval_rows)
            _write_eval_tensorboard_scalars(writer, new_eval_rows, update=update)
            _write_rows(eval_rows, eval_log_path)

    _write_rows(train_rows, train_log_path)
    _write_rows(eval_rows, eval_log_path)
    if writer is not None:
        writer.close()
    # checkpoint 保存模型参数、配置和日志，便于复现实验或继续分析。
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_state_dict": model.state_dict(),
            "config": ppo_config.__dict__,
            "ppo_config": ppo_config.__dict__,
            "train_log": train_rows,
            "eval_log": eval_rows,
            "train_instance_paths": [str(path) for path in train_paths],
            "eval_instance_paths": [str(path) for path in eval_paths],
            "provenance": checkpoint_metadata(
                effective_objective,
                method_hash=method_matrix_hash(instances[0]),
                data_hash=collection_data_hash(instances),
            ),
        },
        checkpoint_path,
    )
    return SharedBatchTrainingResult(
        train_rows=train_rows,
        eval_rows=eval_rows,
        checkpoint_path=checkpoint_path,
        train_log_path=train_log_path,
        eval_log_path=eval_log_path,
        assignment_dir=assignment_dir,
        tensorboard_dir=tensorboard_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", default="instance")
    parser.add_argument("--output-dir", default="results/ppo_batch")
    parser.add_argument("--group", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--instances-per-update", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--env-max-steps", type=int, default=None)
    parser.add_argument("--env-patience", type=int, default=None)
    parser.add_argument("--tensorboard-dir", default=None, help="Optional TensorBoard log directory.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-local-search", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limit = 2 if args.smoke_test and args.limit is None else args.limit
    group = "group1" if args.smoke_test and args.group is None else args.group
    output_dir = "results/ppo_batch_smoke" if args.smoke_test and args.output_dir == "results/ppo_batch" else args.output_dir
    result = run_shared_batch_training(
        instance_root=args.instance_root,
        output_dir=output_dir,
        group=group,
        limit=limit,
        eval_limit=1 if args.smoke_test and args.eval_limit == 3 else args.eval_limit,
        seed=args.seed,
        device=args.device,
        total_updates=1 if args.smoke_test and args.updates is None else args.updates,
        instances_per_update=1 if args.smoke_test and args.instances_per_update == 2 else args.instances_per_update,
        rollout_steps=4 if args.smoke_test and args.rollout_steps is None else args.rollout_steps,
        minibatch_size=2 if args.smoke_test and args.minibatch_size is None else args.minibatch_size,
        update_epochs=1 if args.smoke_test and args.update_epochs is None else args.update_epochs,
        eval_interval=args.eval_interval,
        include_baselines=not args.skip_baselines,
        include_local_search=not args.skip_local_search and not args.smoke_test,
        hidden_dim=args.hidden_dim,
        env_max_steps=4 if args.smoke_test and args.env_max_steps is None else args.env_max_steps,
        env_patience=4 if args.smoke_test and args.env_patience is None else args.env_patience,
        tensorboard_dir=args.tensorboard_dir,
    )
    print(
        json.dumps(
            {
                "train_rows": len(result.train_rows),
                "eval_rows": len(result.eval_rows),
                "train_log_path": str(result.train_log_path),
                "eval_log_path": str(result.eval_log_path),
                "checkpoint_path": str(result.checkpoint_path),
                "assignment_dir": str(result.assignment_dir),
                "tensorboard_dir": str(result.tensorboard_dir) if result.tensorboard_dir is not None else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
