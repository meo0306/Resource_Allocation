"""批量运行 PA-MOAP 实验。

该入口读取已经转换好的 `*.assignment.json`，对每个实例运行一组 baseline
或短 PPO smoke solver，并把所有 `SolverResult.metrics` 展平写入 CSV。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.solvers.greedy_solver import effect_only_greedy, scalarized_independent_greedy
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.ppo_solver import PPOConfig, solve_ppo
from pa_moap_rl.solvers.random_solver import random_legal
from pa_moap_rl.utils.metrics import SolverResult


SolverFactory = Callable[[AssignmentInstance, int], SolverResult]


def discover_instance_paths(instance_root: str | Path, *, limit: int | None = None, group: str | None = None) -> list[Path]:
    """在实例根目录或指定 group 下查找已处理的 assignment JSON。"""

    root = Path(instance_root)
    search_root = root / group if group is not None else root
    paths = sorted(search_root.rglob("*.assignment.json"))
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"No *.assignment.json files found under {search_root}.")
    return paths


def default_solver_factories(*, include_local_search: bool = True, include_ppo: bool = False) -> dict[str, SolverFactory]:
    """返回批量实验默认 solver 注册表。"""

    # factory 统一签名为 `(instance, seed) -> SolverResult`，便于批量循环调用。
    solvers: dict[str, SolverFactory] = {
        "random_legal": lambda instance, seed: random_legal(instance, seed=seed),
        "effect_only_greedy": lambda instance, seed: effect_only_greedy(instance),
        "scalarized_independent_greedy": lambda instance, seed: scalarized_independent_greedy(instance),
    }
    if include_local_search:
        solvers["local_search_best_improvement"] = lambda instance, seed: local_search_best_improvement(
            instance,
            max_iter=instance.n * instance.m,
        )
    if include_ppo:
        solvers["ppo_smoke"] = lambda instance, seed: solve_ppo(
            instance,
            config=PPOConfig(
                rollout_steps=16,
                update_epochs=1,
                minibatch_size=8,
                total_updates=1,
                seed=seed,
                device="auto",
                env_max_steps=8,
                env_patience=8,
            ),
        )
    return solvers


def result_to_row(
    *,
    instance_path: Path,
    instance: AssignmentInstance,
    result: SolverResult,
    seed: int,
    variant: str = "default",
) -> dict:
    """将单个 solver 结果展平成一行 CSV 记录。"""

    row = {
        "variant": variant,
        "group": instance_path.parent.name,
        "instance_name": instance.instance_name,
        "instance_path": str(instance_path),
        "n": instance.n,
        "m": instance.m,
        "k": instance.k,
        "seed": seed,
        "solver_name": result.solver_name,
    }
    for key, value in result.metrics.items():
        if key == "solver_name":
            continue
        row[key] = value
    return row


def run_batch_experiments(
    *,
    instance_root: str | Path = "instance",
    output_csv: str | Path = "results/batch_results.csv",
    limit: int | None = None,
    group: str | None = None,
    seeds: list[int] | None = None,
    include_local_search: bool = True,
    include_ppo: bool = False,
) -> list[dict]:
    """在一批处理后算例上运行配置的 solver，并保存统一 CSV。"""

    seeds = seeds or [0]
    paths = discover_instance_paths(instance_root, limit=limit, group=group)
    solvers = default_solver_factories(include_local_search=include_local_search, include_ppo=include_ppo)
    rows: list[dict] = []

    for path in tqdm(paths, desc="Running batch experiments", unit="instance"):
        instance = load_instance_json(path)
        for seed in seeds:
            for solver in solvers.values():
                result = solver(instance, int(seed))
                rows.append(result_to_row(instance_path=path, instance=instance, result=result, seed=int(seed)))

    write_rows(rows, output_csv)
    return rows


def write_rows(rows: list[dict], output_csv: str | Path) -> None:
    """用稳定字段顺序写出多行实验结果 CSV。"""

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "variant",
        "group",
        "instance_name",
        "instance_path",
        "n",
        "m",
        "k",
        "seed",
        "solver_name",
    ]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch PA-MOAP experiments.")
    parser.add_argument("--instance-root", default="instance")
    parser.add_argument("--output-csv", default="results/batch_results.csv")
    parser.add_argument("--group", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--include-ppo", action="store_true", help="Also run a short PPO smoke solver.")
    parser.add_argument("--skip-local-search", action="store_true", help="Skip local search for fast framework tests.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one small instance with fast baseline solvers.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limit = 1 if args.smoke_test and args.limit is None else args.limit
    group = "group1" if args.smoke_test and args.group is None else args.group
    output_csv = "results/smoke_batch.csv" if args.smoke_test and args.output_csv == "results/batch_results.csv" else args.output_csv
    rows = run_batch_experiments(
        instance_root=args.instance_root,
        output_csv=output_csv,
        limit=limit,
        group=group,
        seeds=_parse_seeds(args.seeds),
        include_local_search=not args.skip_local_search and not args.smoke_test,
        include_ppo=args.include_ppo,
    )
    print(f"Wrote {len(rows)} rows to {output_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
