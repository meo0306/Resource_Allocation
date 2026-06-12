"""运行 PA-MOAP 消融实验。

消融实验不改变输入算例矩阵，只通过替换权重或目标分布构造轻量 variant，
然后比较启发式初解、局部搜索和随机初解的结果。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.experiments.run_batch import discover_instance_paths, result_to_row, write_rows
from pa_moap_rl.solvers.greedy_solver import scalarized_independent_assignment
from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement
from pa_moap_rl.solvers.random_solver import random_legal_assignment
from pa_moap_rl.utils.metrics import make_solver_result


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """把消融后的权重重新归一化，保证目标 `J` 权重总和仍为 1。"""

    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Ablation weights must have positive total mass.")
    return {key: float(value) / total for key, value in weights.items()}


def ablation_variants(instance: AssignmentInstance) -> dict[str, AssignmentInstance]:
    """基于一个算例创建 v1 消融 variant。"""

    base_weights = dict(instance.weights)
    # no_preference 移除学生/教师偏好项；no_global 移除分布与软约束项。
    no_pref_weights = normalize_weights({**base_weights, "student": 0.0, "teacher": 0.0})
    no_global_weights = normalize_weights({**base_weights, "global": 0.0, "soft": 0.0})
    uniform_target = np.full(instance.m, 1.0 / instance.m, dtype=np.float64)

    return {
        "full": instance,
        "no_preference": replace(instance, weights=no_pref_weights),
        "no_global": replace(instance, weights=no_global_weights),
        "uniform_target": replace(instance, target_distribution=uniform_target),
    }


def run_ablation_experiments(
    *,
    instance_root: str | Path = "instance",
    output_csv: str | Path = "results/ablation_results.csv",
    limit: int | None = None,
    group: str | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    """运行轻量消融实验并保存统一 CSV。"""

    seeds = seeds or [0]
    rows: list[dict] = []
    paths = discover_instance_paths(instance_root, limit=limit, group=group)
    for path in tqdm(paths, desc="Running ablations", unit="instance"):
        base_instance = load_instance_json(path)
        for variant_name, instance in ablation_variants(base_instance).items():
            heuristic_initial = scalarized_independent_assignment(instance)
            heuristic_result = local_search_best_improvement(instance, heuristic_initial)
            rows.append(
                result_to_row(
                    instance_path=path,
                    instance=instance,
                    result=heuristic_result,
                    seed=0,
                    variant=f"{variant_name}:heuristic_init",
                )
            )

            for seed in seeds:
                rng = np.random.default_rng(seed)
                random_initial = random_legal_assignment(instance, rng)
                random_score = make_solver_result(
                    solver_name="random_initial",
                    instance=instance,
                    assignment=random_initial,
                    runtime=0.0,
                )
                rows.append(
                    result_to_row(
                        instance_path=path,
                        instance=instance,
                        result=random_score,
                        seed=int(seed),
                        variant=f"{variant_name}:random_init",
                    )
                )

    write_rows(rows, output_csv)
    return rows


def _parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PA-MOAP ablation experiments.")
    parser.add_argument("--instance-root", default="instance")
    parser.add_argument("--output-csv", default="results/ablation_results.csv")
    parser.add_argument("--group", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limit = 1 if args.smoke_test and args.limit is None else args.limit
    group = "group1" if args.smoke_test and args.group is None else args.group
    output_csv = "results/smoke_ablation.csv" if args.smoke_test and args.output_csv == "results/ablation_results.csv" else args.output_csv
    rows = run_ablation_experiments(
        instance_root=args.instance_root,
        output_csv=output_csv,
        limit=limit,
        group=group,
        seeds=_parse_seeds(args.seeds),
    )
    print(f"Wrote {len(rows)} rows to {output_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
