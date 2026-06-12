"""训练或 smoke-test 单实例 masked PPO。

该命令可以直接读取处理后的 assignment JSON，也可以在 smoke-test 中从
`examples/` 的 `.sm + 内容一 CSV` 临时构造算例。输出包括训练日志 CSV、
模型 checkpoint 和 best assignment JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.solvers.ppo_solver import PPOConfig, ppo_config_from_project, resolve_device, solve_ppo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or smoke-test masked PPO for PA-MOAP.")
    parser.add_argument("--instance-json", type=Path, default=None, help="Processed assignment instance JSON.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short deterministic smoke test.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None, help="Reserved for future CLI model override.")
    parser.add_argument("--log-path", type=Path, default=Path("results/ppo_smoke_log.csv"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("results/ppo_smoke.pt"))
    parser.add_argument("--assignment-path", type=Path, default=Path("results/ppo_smoke_assignment.json"))
    return parser


def load_smoke_instance():
    """从 examples 构造一个无需预处理产物的 smoke-test 算例。"""

    record = load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0]
    return build_assignment_instance_from_selection("examples", record)


def load_instance(path: Path | None):
    """优先读取用户指定 JSON；未指定时使用 smoke-test 示例算例。"""

    if path is None:
        return load_smoke_instance()
    return load_instance_json(path)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_config = load_config()
    overrides = {
        "seed": args.seed,
        "device": str(resolve_device(args.device)),
    }
    if args.smoke_test:
        # smoke-test 使用极小训练步数，目标是验证闭环和 mask，不追求收敛质量。
        overrides.update(
            {
                "total_updates": args.updates if args.updates is not None else 2,
                "rollout_steps": args.rollout_steps if args.rollout_steps is not None else 16,
                "minibatch_size": args.minibatch_size if args.minibatch_size is not None else 8,
                "update_epochs": args.update_epochs if args.update_epochs is not None else 2,
                "env_max_steps": 8,
                "env_patience": 8,
            }
        )
    else:
        if args.updates is not None:
            overrides["total_updates"] = args.updates
        if args.rollout_steps is not None:
            overrides["rollout_steps"] = args.rollout_steps
        if args.minibatch_size is not None:
            overrides["minibatch_size"] = args.minibatch_size
        if args.update_epochs is not None:
            overrides["update_epochs"] = args.update_epochs

    ppo_config = ppo_config_from_project(project_config, **overrides)
    instance = load_instance(args.instance_json)
    result = solve_ppo(
        instance,
        config=ppo_config,
        project_config=project_config,
        log_path=args.log_path,
        checkpoint_path=args.checkpoint_path,
    )

    # 将 best-so-far assignment 单独写出，便于后续人工检查或结果分析。
    args.assignment_path.parent.mkdir(parents=True, exist_ok=True)
    args.assignment_path.write_text(
        json.dumps(
            {
                "solver_name": result.solver_name,
                "instance_name": instance.instance_name,
                "assignment": result.assignment.astype(int).tolist(),
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "device": str(resolve_device(args.device)),
                "total_score": result.score.total_score,
                "illegal_action_rate": result.metrics["illegal_action_rate"],
                "log_path": str(args.log_path),
                "checkpoint_path": str(args.checkpoint_path),
                "assignment_path": str(args.assignment_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
