"""分析 PA-MOAP 实验结果 CSV 并生成表格/图。

该脚本可以读取 batch、ablation 或 PPO train log。若输入包含 solver 评分列，
会生成实验汇总表、case 表和对比图；若输入包含 `update/best_score`，会生成
PPO 训练曲线。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SCORE_COLUMNS = [
    "total_score",
    "effect_score",
    "student_pref_score",
    "teacher_pref_score",
    "global_score",
    "soft_penalty",
    "runtime",
]


def load_results(input_paths: list[str | Path]) -> pd.DataFrame:
    """读取并合并一个或多个实验 CSV。"""

    frames = []
    for path in input_paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(csv_path)
        frame["source_file"] = str(csv_path)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one input CSV is required.")
    return pd.concat(frames, ignore_index=True)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """按 variant/solver 聚合关键数值指标。"""

    group_cols = [col for col in ("variant", "solver_name") if col in df.columns]
    if not group_cols:
        raise ValueError("Input results must include solver_name or variant columns.")
    numeric_cols = [col for col in SCORE_COLUMNS if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["count", "mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary.reset_index()


def build_case_table(df: pd.DataFrame) -> pd.DataFrame:
    """按实例/variant 选择 total_score 最高的一行，形成 case 表。"""

    required = {"instance_name", "total_score"}
    if not required.issubset(df.columns):
        return df.head(20).copy()
    group_cols = [col for col in ("variant", "instance_name") if col in df.columns]
    idx = df.groupby(group_cols)["total_score"].idxmax() if group_cols else df["total_score"].nlargest(20).index
    columns = [
        col
        for col in (
            "variant",
            "group",
            "instance_name",
            "n",
            "seed",
            "solver_name",
            "total_score",
            "effect_score",
            "student_pref_score",
            "teacher_pref_score",
            "global_score",
            "soft_penalty",
            "runtime",
        )
        if col in df.columns
    ]
    return df.loc[idx, columns].sort_values(columns[:2] if len(columns) >= 2 else columns).reset_index(drop=True)


def plot_total_score(summary: pd.DataFrame, figure_dir: str | Path) -> Path | None:
    """绘制不同 solver/variant 的平均 total score 柱状图。"""

    if "total_score_mean" not in summary.columns:
        return None
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = summary.apply(
        lambda row: f"{row.get('variant', 'default')}\n{row.get('solver_name', '')}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(summary)), 4.5))
    ax.bar(range(len(summary)), summary["total_score_mean"])
    ax.set_ylabel("Mean total score")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "total_score_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_tradeoff(df: pd.DataFrame, figure_dir: str | Path) -> Path | None:
    """绘制效果分数与平均偏好分数的 trade-off 散点图。"""

    needed = {"effect_score", "student_pref_score", "teacher_pref_score"}
    if not needed.issubset(df.columns):
        return None
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    preference = (df["student_pref_score"] + df["teacher_pref_score"]) / 2.0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for solver, group in df.assign(preference_score=preference).groupby("solver_name" if "solver_name" in df.columns else "source_file"):
        ax.scatter(group["effect_score"], group["preference_score"], label=str(solver), alpha=0.75)
    ax.set_xlabel("Effect score")
    ax.set_ylabel("Preference score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "effect_preference_tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training_curve(df: pd.DataFrame, figure_dir: str | Path) -> Path | None:
    """当输入包含训练日志字段时绘制 PPO best_score 曲线。"""

    if not {"update", "best_score"}.issubset(df.columns):
        return None
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for source, group in df.groupby("source_file"):
        ax.plot(group["update"], group["best_score"], marker="o", label=Path(source).name)
    ax.set_xlabel("Update")
    ax.set_ylabel("Best score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "ppo_training_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def analyze_results(
    *,
    input_paths: list[str | Path],
    table_dir: str | Path = "tables",
    figure_dir: str | Path = "figures",
) -> dict[str, str]:
    """分析结果 CSV，并写出可复现实验表格和图。"""

    df = load_results(input_paths)
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if {"solver_name", "total_score"}.issubset(df.columns):
        summary = summarize_results(df)
        summary_path = table_dir / "experiment_summary.csv"
        summary.to_csv(summary_path, index=False)
        outputs["summary"] = str(summary_path)

        case_table = build_case_table(df)
        case_path = table_dir / "case_table.csv"
        case_table.to_csv(case_path, index=False)
        outputs["case_table"] = str(case_path)

        total_plot = plot_total_score(summary, figure_dir)
        if total_plot is not None:
            outputs["total_score_plot"] = str(total_plot)
        tradeoff_plot = plot_tradeoff(df, figure_dir)
        if tradeoff_plot is not None:
            outputs["tradeoff_plot"] = str(tradeoff_plot)

    training_plot = plot_training_curve(df, figure_dir)
    if training_plot is not None:
        outputs["training_curve"] = str(training_plot)

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PA-MOAP experiment result files.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more result CSV files.")
    parser.add_argument("--table-dir", default="tables")
    parser.add_argument("--figure-dir", default="figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = analyze_results(input_paths=args.input, table_dir=args.table_dir, figure_dir=args.figure_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
