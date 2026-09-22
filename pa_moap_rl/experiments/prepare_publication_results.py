'''Prepare traceable statistics and publication-ready figures for the final PPO runs.'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False


SEEDS = (0, 1, 2)
DATASETS = ('interpolation', 'heldout')
PPO_NAME = 'ppo_shared_policy'
METHOD_ORDER = (
    'random_legal',
    'effect_only_greedy',
    'scalarized_independent_greedy',
    PPO_NAME,
    'local_search_best_improvement',
)
METHOD_LABELS = {
    'random_legal': 'Random legal',
    'effect_only_greedy': 'Effect-only',
    'scalarized_independent_greedy': 'Scalar greedy',
    PPO_NAME: 'Masked PPO',
    'local_search_best_improvement': 'Local search',
}
METHOD_COLORS = {
    'random_legal': '#B4C0E4',
    'effect_only_greedy': '#7884B4',
    'scalarized_independent_greedy': '#484878',
    PPO_NAME: '#0F4D92',
    'local_search_best_improvement': '#E9A6A1',
}
SEED_COLORS = {0: '#0F4D92', 1: '#42949E', 2: '#9A4D8E'}
TOPOLOGY_ORDER = ('uniform', 'random', 'staircase', 'hourglass', 'bottleneck')
TOPOLOGY_COLORS = {
    'uniform': '#0F4D92',
    'random': '#42949E',
    'staircase': '#B64342',
    'hourglass': '#9A4D8E',
    'bottleneck': '#767676',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    samples: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError('Cannot bootstrap an empty sample.')
    if samples < 1:
        raise ValueError('Bootstrap samples must be positive.')
    estimates: list[np.ndarray] = []
    remaining = samples
    chunk_size = max(1, min(1000, samples))
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, array.size, size=(current, array.size))
        estimates.append(array[indices].mean(axis=1))
        remaining -= current
    boot = np.concatenate(estimates)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _two_sided_sign_test(differences: Iterable[float], tolerance: float = 1e-12) -> float:
    values = np.asarray(list(differences), dtype=float)
    wins = int(np.sum(values > tolerance))
    losses = int(np.sum(values < -tolerance))
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return [base.with_suffix(ext) for ext in ('.svg', '.pdf', '.png')]


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        ha='left',
        va='bottom',
    )


def _read_training(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        run_dir = project_root / 'results' / f'formal_joint_seed{seed}_run01' / 'metrics'
        train = pd.read_csv(run_dir / 'train_updates.csv')
        validation = pd.read_csv(run_dir / 'validation_updates.csv')
        train['seed'] = seed
        validation['seed'] = seed
        train_frames.append(train)
        validation_frames.append(validation)

        overall = validation.loc[validation['scope'] == 'overall'].sort_values('update')
        best_index = overall['total_score_mean'].idxmax()
        best = overall.loc[best_index]
        summary_rows.append(
            {
                'seed': seed,
                'updates': int(train['update'].max()),
                'training_instances': int(train['instances_per_update'].sum()),
                'transitions': int(train['transitions'].sum()),
                'validation_points': int(len(overall)),
                'best_validation_score': float(best['total_score_mean']),
                'best_update': int(best['update']),
                'last_validation_score': float(overall.iloc[-1]['total_score_mean']),
                'last_four_mean': float(overall.tail(4)['total_score_mean'].mean()),
                'normalized_entropy_min': float(train['normalized_entropy'].min()),
                'normalized_entropy_end': float(train.iloc[-1]['normalized_entropy']),
                'kl_mean': float(train['approx_kl'].mean()),
                'kl_p90': float(train['approx_kl'].quantile(0.9)),
                'kl_max': float(train['approx_kl'].max()),
                'clip_fraction_mean': float(train['clip_fraction'].mean()),
                'clip_fraction_p90': float(train['clip_fraction'].quantile(0.9)),
                'illegal_action_count': int(train['illegal_action_count'].sum()),
                'update_time_seconds': float(train['update_elapsed_sec'].sum()),
                'validation_runtime_seconds': float(
                    (overall['runtime_mean'] * overall['count']).sum()
                ),
                'rollout_time_seconds': float(train['rollout_elapsed_sec'].sum()),
                'ppo_time_seconds': float(train['ppo_elapsed_sec'].sum()),
                'transitions_per_second_mean': float(train['transitions_per_sec'].mean()),
                'gpu_peak_allocated_mb': float(train['gpu_peak_allocated_mb'].max()),
                'gpu_peak_reserved_mb': float(train['gpu_peak_reserved_mb'].max()),
            }
        )
    return (
        pd.concat(train_frames, ignore_index=True),
        pd.concat(validation_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def _read_test_results(project_root: Path) -> dict[str, pd.DataFrame]:
    base = project_root / 'results' / 'final_evaluation'
    paths = {
        'interpolation': base / 'seed1_best_test_interpolation_with_baselines.csv',
        'heldout': base / 'seed1_best_test_heldout_with_baselines.csv',
    }
    return {dataset: pd.read_csv(path) for dataset, path in paths.items()}


def _validate_test_pairing(frame: pd.DataFrame, dataset: str) -> int:
    required = {
        'instance_path',
        'scenario_id',
        'solver_name',
        'total_score',
        'runtime',
        'topology',
        'hard_feasible_rate',
        'mask_violation_count',
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'{dataset} is missing columns: {sorted(missing)}')
    unexpected = set(frame['solver_name']).difference(METHOD_ORDER)
    if unexpected:
        raise ValueError(f'{dataset} contains unexpected solvers: {sorted(unexpected)}')
    counts: set[int] = set()
    reference_keys: set[tuple[str, str]] | None = None
    for method in METHOD_ORDER:
        rows = frame.loc[frame['solver_name'] == method]
        keys = list(zip(rows['instance_path'], rows['scenario_id']))
        if len(keys) != len(set(keys)):
            raise ValueError(f'{dataset}/{method} contains duplicate instance-scenario pairs.')
        key_set = set(keys)
        if reference_keys is None:
            reference_keys = key_set
        elif key_set != reference_keys:
            raise ValueError(f'{dataset}/{method} does not match the paired test bank.')
        counts.add(len(keys))
    if len(counts) != 1:
        raise ValueError(f'{dataset} solver counts differ: {sorted(counts)}')
    return counts.pop()


def _algorithm_summary(
    frames: dict[str, pd.DataFrame],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for dataset in DATASETS:
        frame = frames[dataset]
        count = _validate_test_pairing(frame, dataset)
        for method in METHOD_ORDER:
            subset = frame.loc[frame['solver_name'] == method]
            low, high = _bootstrap_mean_ci(
                subset['total_score'], samples=bootstrap_samples, rng=rng
            )
            rows.append(
                {
                    'dataset': dataset,
                    'solver_name': method,
                    'solver_label': METHOD_LABELS[method],
                    'n_instances': count,
                    'total_score_mean': float(subset['total_score'].mean()),
                    'total_score_std_across_instances': float(subset['total_score'].std(ddof=0)),
                    'total_score_ci95_low': low,
                    'total_score_ci95_high': high,
                    'runtime_mean_seconds': float(subset['runtime'].mean()),
                    'runtime_max_seconds': float(subset['runtime'].max()),
                    'hard_feasible': bool((subset['hard_feasible_rate'] == 1.0).all()),
                    'mask_violation_count': int(subset['mask_violation_count'].sum()),
                }
            )
    return pd.DataFrame(rows)


def _paired_comparisons(
    frames: dict[str, pd.DataFrame],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    topology_rows: list[dict[str, float | int | str]] = []
    instance_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(bootstrap_seed + 1)
    keys = ['instance_path', 'scenario_id']
    for dataset in DATASETS:
        frame = frames[dataset]
        ppo = frame.loc[frame['solver_name'] == PPO_NAME].set_index(keys).sort_index()
        for baseline in METHOD_ORDER:
            if baseline == PPO_NAME:
                continue
            other = frame.loc[frame['solver_name'] == baseline].set_index(keys).sort_index()
            if not ppo.index.equals(other.index):
                raise ValueError(f'{dataset}/{baseline} pairing changed after indexing.')
            difference = ppo['total_score'].to_numpy() - other['total_score'].to_numpy()
            low, high = _bootstrap_mean_ci(
                difference, samples=bootstrap_samples, rng=rng
            )
            standard_deviation = float(np.std(difference, ddof=1))
            rows.append(
                {
                    'dataset': dataset,
                    'baseline': baseline,
                    'baseline_label': METHOD_LABELS[baseline],
                    'n_pairs': int(difference.size),
                    'ppo_mean': float(ppo['total_score'].mean()),
                    'baseline_mean': float(other['total_score'].mean()),
                    'mean_difference_ppo_minus_baseline': float(difference.mean()),
                    'difference_ci95_low': low,
                    'difference_ci95_high': high,
                    'win_rate': float(np.mean(difference > 1e-12)),
                    'tie_rate': float(np.mean(np.abs(difference) <= 1e-12)),
                    'loss_rate': float(np.mean(difference < -1e-12)),
                    'paired_cohen_dz': (
                        float(difference.mean() / standard_deviation)
                        if standard_deviation > 0
                        else float('nan')
                    ),
                    'sign_test_p_value': _two_sided_sign_test(difference),
                    'runtime_ratio_baseline_over_ppo': float(
                        other['runtime'].mean() / ppo['runtime'].mean()
                    ),
                }
            )
            for index, difference_value in zip(ppo.index, difference):
                instance_rows.append(
                    {
                        'dataset': dataset,
                        'instance_path': index[0],
                        'scenario_id': index[1],
                        'topology': str(ppo.loc[index, 'topology']),
                        'baseline': baseline,
                        'ppo_score': float(ppo.loc[index, 'total_score']),
                        'baseline_score': float(other.loc[index, 'total_score']),
                        'difference_ppo_minus_baseline': float(difference_value),
                        'ppo_runtime_seconds': float(ppo.loc[index, 'runtime']),
                        'baseline_runtime_seconds': float(other.loc[index, 'runtime']),
                    }
                )
            for topology in TOPOLOGY_ORDER:
                mask = ppo['topology'].to_numpy() == topology
                topology_difference = difference[mask]
                top_low, top_high = _bootstrap_mean_ci(
                    topology_difference, samples=bootstrap_samples, rng=rng
                )
                topology_rows.append(
                    {
                        'dataset': dataset,
                        'topology': topology,
                        'baseline': baseline,
                        'n_pairs': int(topology_difference.size),
                        'mean_difference_ppo_minus_baseline': float(topology_difference.mean()),
                        'difference_ci95_low': top_low,
                        'difference_ci95_high': top_high,
                        'win_rate': float(np.mean(topology_difference > 1e-12)),
                    }
                )
    adjusted = _holm_adjust([float(row['sign_test_p_value']) for row in rows])
    for row, value in zip(rows, adjusted):
        row['sign_test_p_value_holm'] = value
    return pd.DataFrame(rows), pd.DataFrame(topology_rows), pd.DataFrame(instance_rows)


def _load_cross_seed_summary(project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base = project_root / 'results' / 'final_evaluation'
    for dataset in DATASETS:
        frame = pd.read_csv(base / f'{dataset}_multiseed' / 'ppo_cross_seed_summary.csv')
        frame['dataset'] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _load_seed_level_test_summary(project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base = project_root / 'results' / 'final_evaluation'
    for dataset in DATASETS:
        frame = pd.read_csv(base / f'{dataset}_multiseed' / 'ppo_seed_summary.csv')
        frame['dataset'] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _plot_training(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    overall = validation.loc[validation['scope'] == 'overall']
    for seed in SEEDS:
        subset = overall.loc[overall['seed'] == seed].sort_values('update')
        ax_a.plot(
            subset['update'],
            subset['total_score_mean'],
            color=SEED_COLORS[seed],
            linewidth=1.4,
            marker='o',
            markersize=2.2,
            label=f'Seed {seed}',
        )
        best = subset.loc[subset['total_score_mean'].idxmax()]
        ax_a.scatter(
            [best['update']], [best['total_score_mean']],
            color=SEED_COLORS[seed], edgecolor='white', linewidth=0.5, s=26, zorder=4,
        )
    ax_a.set(xlabel='PPO update', ylabel='Validation total score')
    ax_a.legend(ncol=3, loc='lower right')
    _add_panel_label(ax_a, 'a')

    selected = validation.loc[
        (validation['seed'] == 1) & (validation['scope'] == 'topology')
    ]
    for topology in TOPOLOGY_ORDER:
        subset = selected.loc[selected['scope_value'] == topology].sort_values('update')
        ax_b.plot(
            subset['update'], subset['total_score_mean'],
            color=TOPOLOGY_COLORS[topology], linewidth=1.2, label=topology.capitalize(),
        )
    ax_b.set(xlabel='PPO update', ylabel='Validation score by topology')
    ax_b.legend(ncol=2, fontsize=6.4, loc='lower right')
    _add_panel_label(ax_b, 'b')

    for seed in SEEDS:
        subset = train.loc[train['seed'] == seed].sort_values('update')
        smooth = subset['normalized_entropy'].rolling(25, min_periods=1).mean()
        ax_c.plot(subset['update'], smooth, color=SEED_COLORS[seed], linewidth=1.3)
    ax_c.axhline(0.45, color='#B64342', linestyle='--', linewidth=0.9, label='Freeze gate')
    ax_c.set(xlabel='PPO update', ylabel='Normalized entropy\n(25-update mean)')
    ax_c.legend(loc='lower right')
    _add_panel_label(ax_c, 'c')

    averaged = (
        train.groupby('update', as_index=False)[['approx_kl', 'clip_fraction']]
        .mean()
        .sort_values('update')
    )
    kl = averaged['approx_kl'].rolling(25, min_periods=1).mean()
    clip = averaged['clip_fraction'].rolling(25, min_periods=1).mean()
    line_kl = ax_d.plot(
        averaged['update'], kl, color='#0F4D92', linewidth=1.4, label='Approx. KL'
    )[0]
    ax_d.axhline(0.05, color='#0F4D92', linestyle=':', linewidth=0.8)
    twin = ax_d.twinx()
    line_clip = twin.plot(
        averaged['update'], clip, color='#B64342', linewidth=1.2, label='Clip fraction'
    )[0]
    twin.axhline(0.25, color='#B64342', linestyle=':', linewidth=0.8)
    ax_d.set(xlabel='PPO update', ylabel='Approx. KL\n(25-update mean)')
    twin.set_ylabel('Clip fraction\n(25-update mean)', color='#B64342')
    twin.tick_params(axis='y', colors='#B64342')
    ax_d.legend([line_kl, line_clip], ['Approx. KL', 'Clip fraction'], loc='upper right')
    _add_panel_label(ax_d, 'd')

    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.09, top=0.96, wspace=0.38, hspace=0.34)
    return _save_figure(fig, figure_dir / 'figure_1_training_dynamics')


def _plot_algorithm_comparison(
    algorithm_summary: pd.DataFrame,
    paired: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={'width_ratios': [1.25, 1.25, 1.0]})
    ax_a, ax_b, ax_c = axes
    x = np.arange(len(METHOD_ORDER))
    offsets = {'interpolation': -0.10, 'heldout': 0.10}
    markers = {'interpolation': 'o', 'heldout': 's'}
    for dataset in DATASETS:
        subset = algorithm_summary.loc[algorithm_summary['dataset'] == dataset].set_index('solver_name').loc[list(METHOD_ORDER)]
        values = subset['total_score_mean'].to_numpy()
        low = values - subset['total_score_ci95_low'].to_numpy()
        high = subset['total_score_ci95_high'].to_numpy() - values
        ax_a.errorbar(
            x + offsets[dataset], values, yerr=np.vstack([low, high]),
            fmt=markers[dataset], markersize=4, capsize=2.2, linewidth=1.0,
            color='#0F4D92' if dataset == 'interpolation' else '#B64342',
            label='Interpolation' if dataset == 'interpolation' else 'Held-out',
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([METHOD_LABELS[name] for name in METHOD_ORDER], rotation=35, ha='right')
    ax_a.set_ylabel('Mean total score (95% bootstrap CI)')
    ax_a.set_ylim(0.59, 0.70)
    ax_a.legend(loc='lower right', fontsize=6.5)
    _add_panel_label(ax_a, 'a')

    forest_order = (
        'random_legal',
        'effect_only_greedy',
        'scalarized_independent_greedy',
        'local_search_best_improvement',
    )
    y = np.arange(len(forest_order))[::-1]
    for dataset in DATASETS:
        subset = paired.loc[paired['dataset'] == dataset].set_index('baseline').loc[list(forest_order)]
        offset = -0.09 if dataset == 'interpolation' else 0.09
        ax_b.errorbar(
            subset['mean_difference_ppo_minus_baseline'], y + offset,
            xerr=np.vstack([
                subset['mean_difference_ppo_minus_baseline'] - subset['difference_ci95_low'],
                subset['difference_ci95_high'] - subset['mean_difference_ppo_minus_baseline'],
            ]),
            fmt=markers[dataset], markersize=3.8, capsize=2.0, linewidth=1.0,
            color='#0F4D92' if dataset == 'interpolation' else '#B64342',
        )
    ax_b.axvline(0.0, color='#767676', linestyle='--', linewidth=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f'vs {METHOD_LABELS[name]}' for name in forest_order])
    ax_b.set_xlabel('Paired score difference (PPO − baseline)')
    _add_panel_label(ax_b, 'b')

    for method in METHOD_ORDER:
        subset = algorithm_summary.loc[algorithm_summary['solver_name'] == method].set_index('dataset')
        ax_c.plot(
            subset.loc[list(DATASETS), 'runtime_mean_seconds'],
            subset.loc[list(DATASETS), 'total_score_mean'],
            color=METHOD_COLORS[method], linewidth=0.8, alpha=0.75,
        )
        for dataset in DATASETS:
            row = subset.loc[dataset]
            ax_c.scatter(
                row['runtime_mean_seconds'], row['total_score_mean'],
                color=METHOD_COLORS[method], marker=markers[dataset], s=23,
                edgecolor='white', linewidth=0.4, zorder=3,
            )
        label_x = float(subset['runtime_mean_seconds'].max())
        label_y = float(subset['total_score_mean'].mean())
        horizontal_alignment = 'left'
        x_offset = 3
        if method == 'local_search_best_improvement':
            label_x = float(subset['runtime_mean_seconds'].min())
            horizontal_alignment = 'right'
            x_offset = -3
        ax_c.annotate(
            METHOD_LABELS[method],
            (label_x, label_y),
            xytext=(x_offset, 0),
            textcoords='offset points',
            ha=horizontal_alignment,
            va='center',
            fontsize=5.8,
            color=METHOD_COLORS[method],
        )
    ax_c.set_xscale('log')
    ax_c.set(xlabel='Mean runtime per instance\n(s, log scale)', ylabel='Mean total score')
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color='#4D4D4D', label='Interpolation'),
        plt.Line2D([], [], marker='s', linestyle='', color='#4D4D4D', label='Held-out'),
    ]
    ax_c.legend(handles=handles, fontsize=6.2, loc='upper left')
    _add_panel_label(ax_c, 'c')

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.94, wspace=0.48)
    return _save_figure(fig, figure_dir / 'figure_2_quality_speed_comparison')


def _plot_generalization(
    cross_seed: pd.DataFrame,
    seed_level: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={'width_ratios': [1.45, 1.0]})
    ax_a, ax_b = axes
    topology = cross_seed.loc[cross_seed['scope'] == 'topology'].copy()
    x = np.arange(len(TOPOLOGY_ORDER))
    for dataset, offset, color, marker in (
        ('interpolation', -0.10, '#0F4D92', 'o'),
        ('heldout', 0.10, '#B64342', 's'),
    ):
        subset = topology.loc[topology['dataset'] == dataset].set_index('scope_value').loc[list(TOPOLOGY_ORDER)]
        ax_a.errorbar(
            x + offset, subset['seed_mean'], yerr=subset['seed_population_std'],
            fmt=marker, markersize=4.3, capsize=2.3, linewidth=1.0, color=color,
            label=dataset.capitalize() if dataset == 'interpolation' else 'Held-out',
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([name.capitalize() for name in TOPOLOGY_ORDER], rotation=25, ha='right')
    ax_a.set_ylabel('PPO score across three seeds (mean ± SD)')
    ax_a.set_ylim(0.60, 0.70)
    ax_a.legend(loc='lower right')
    _add_panel_label(ax_a, 'a')

    rows: list[dict[str, float | str]] = []
    seed_delta_rows: list[dict[str, float | str]] = []
    scopes = [('overall', 'all'), *[('topology', topology) for topology in TOPOLOGY_ORDER]]
    for scope, value in scopes:
        subset = seed_level.loc[
            (seed_level['scope'] == scope) & (seed_level['scope_value'] == value)
        ]
        interpolation = (
            subset.loc[subset['dataset'] == 'interpolation']
            .set_index('seed_id')['total_score_mean']
            .sort_index()
        )
        heldout = (
            subset.loc[subset['dataset'] == 'heldout']
            .set_index('seed_id')['total_score_mean']
            .sort_index()
        )
        if not interpolation.index.equals(heldout.index):
            raise ValueError(f'Cross-dataset seed mismatch for {scope}/{value}.')
        differences = heldout - interpolation
        rows.append(
            {
                'scope_value': 'Overall' if value == 'all' else value.capitalize(),
                'delta_heldout_minus_interpolation_mean': float(differences.mean()),
                'delta_heldout_minus_interpolation_sd': float(differences.std(ddof=0)),
            }
        )
        for seed_id, difference in differences.items():
            seed_delta_rows.append(
                {
                    'scope_value': 'Overall' if value == 'all' else value.capitalize(),
                    'seed_id': seed_id,
                    'delta_heldout_minus_interpolation': float(difference),
                }
            )
    delta = pd.DataFrame(rows)
    colors = [
        '#42949E' if value >= 0 else '#B64342'
        for value in delta['delta_heldout_minus_interpolation_mean']
    ]
    ax_b.barh(
        np.arange(len(delta))[::-1], delta['delta_heldout_minus_interpolation_mean'],
        xerr=delta['delta_heldout_minus_interpolation_sd'],
        color=colors, edgecolor='white', linewidth=0.4,
        error_kw={'elinewidth': 0.8, 'capsize': 2, 'capthick': 0.8, 'ecolor': '#4D4D4D'},
    )
    ax_b.axvline(0.0, color='#767676', linestyle='--', linewidth=0.8)
    ax_b.set_yticks(np.arange(len(delta))[::-1])
    ax_b.set_yticklabels(delta['scope_value'])
    ax_b.set_xlabel('Held-out − interpolation score')
    _add_panel_label(ax_b, 'b')

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.94, wspace=0.38)
    paths = _save_figure(fig, figure_dir / 'figure_3_generalization_by_topology')
    delta.to_csv(figure_dir.parent / 'source_data' / 'generalization_delta.csv', index=False)
    pd.DataFrame(seed_delta_rows).to_csv(
        figure_dir.parent / 'source_data' / 'generalization_delta_by_seed.csv', index=False
    )
    return paths


def _write_report(
    output_dir: Path,
    training_summary: pd.DataFrame,
    algorithm_summary: pd.DataFrame,
    paired: pd.DataFrame,
    cross_seed: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> Path:
    selected = training_summary.loc[training_summary['seed'] == 1].iloc[0]
    interpolation = paired.loc[paired['dataset'] == 'interpolation'].set_index('baseline')
    heldout = paired.loc[paired['dataset'] == 'heldout'].set_index('baseline')
    cross_overall = cross_seed.loc[cross_seed['scope'] == 'overall'].set_index('dataset')
    scalar = 'scalarized_independent_greedy'
    local = 'local_search_best_improvement'
    update_compute_hours = training_summary['update_time_seconds'].sum() / 3600.0
    validation_compute_hours = training_summary['validation_runtime_seconds'].sum() / 3600.0
    report = f'''# PPO 现有结果整理包

本目录由冻结后的三种子训练日志和正式配对测试 CSV 自动生成。

## 核心结论

共享 masked PPO 在 interpolation 和 held-out 测试中分别比标量贪婪高
{interpolation.loc[scalar, 'mean_difference_ppo_minus_baseline']:.6f} 和
{heldout.loc[scalar, 'mean_difference_ppo_minus_baseline']:.6f}。PPO 分别比完整局部搜索低
{abs(interpolation.loc[local, 'mean_difference_ppo_minus_baseline']):.6f} 和
{abs(heldout.loc[local, 'mean_difference_ppo_minus_baseline']):.6f}，但局部搜索耗时分别是 PPO 的
{interpolation.loc[local, 'runtime_ratio_baseline_over_ppo']:.1f} 倍和
{heldout.loc[local, 'runtime_ratio_baseline_over_ppo']:.1f} 倍。

## 冻结训练结果

- seed 0、1、2 均训练 1500 updates；每次 update 使用 4 个实例和 512 条 transitions。
- 部署模型按固定 validation 选择 seed 1，其最佳分数为
  {selected['best_validation_score']:.6f}，出现在 update {int(selected['best_update'])}。
- 三种子日志累计的 PPO update 计时为 {update_compute_hours:.2f} 小时，validation 实例
  推理计时为 {validation_compute_hours:.2f} 小时；两者合计 {update_compute_hours + validation_compute_hours:.2f} 小时，
  不含进程启动、checkpoint 写入和中断恢复开销。
- 所有正式训练与测试结果均满足硬约束，非法动作和 mask violation 均为 0。

## 泛化结果

- Interpolation：PPO 三种子均值 {cross_overall.loc['interpolation', 'seed_mean']:.6f}
  +/- {cross_overall.loc['interpolation', 'seed_population_std']:.6f} SD。
- Held-out：PPO 三种子均值 {cross_overall.loc['heldout', 'seed_mean']:.6f}
  +/- {cross_overall.loc['heldout', 'seed_population_std']:.6f} SD。
- Held-out 指五类已知拓扑上的未见画像组合，不表示对完全未见拓扑类型的泛化。

## 统计口径

- 每套测试使用 675 个严格配对的 instance-scenario 样本。
- 总分区间为 {bootstrap_samples:,} 次百分位 bootstrap 的 95% CI。
- 配对差定义为 PPO 减去对比算法；正值表示 PPO 更优。
- 表格同时报告胜/平/负率、配对 Cohen's dz、精确双侧符号检验，以及对 8 项预设比较的 Holm 校正。
- 实例级 bootstrap 区间和三种子波动回答不同问题，已分开报告。
- 项目当前未依赖 SciPy，因此本整理包没有加入 Wilcoxon signed-rank 检验。

## 图件

1. `figures/figure_1_training_dynamics.*`：validation 变化、seed1 分拓扑表现、标准化熵和 KL/clip。
2. `figures/figure_2_quality_speed_comparison.*`：测试总分、PPO 配对差及质量-速度折中。
3. `figures/figure_3_generalization_by_topology.*`：三种子拓扑结果及 held-out 相对 interpolation 的变化。

每张图均提供可编辑文本 SVG、PDF 和 300 dpi PNG。

## 表格与源数据

- `tables/training_summary.csv`：逐种子训练审计。
- `tables/algorithm_summary.csv`：逐测试集和算法的总分区间、耗时与可行性。
- `tables/paired_comparisons.csv`：预设 PPO-基线配对比较。
- `tables/paired_topology_comparisons.csv`：按拓扑拆分的配对差。
- `tables/cross_seed_summary.csv`：三种子总体与拓扑泛化结果。
- `source_data/paired_instance_differences.csv`：每个测试样本的 PPO-基线差。
- `source_data/` 其余文件：各图面板直接使用的绘图数据。

## 结论边界

当前结果支持“质量-时延折中”：PPO 优于简单构造式基线，并能泛化到未见画像组合，但总分尚未
超过完整 best-improvement 局部搜索。PPO 消融和固定权重敏感性实验不在本整理包中，不能描述为已完成。
'''
    path = output_dir / 'README.md'
    path.write_text(report, encoding='utf-8')
    return path


def prepare_publication_results(
    project_root: str | Path,
    *,
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260908,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    figure_dir = output / 'figures'
    table_dir = output / 'tables'
    source_dir = output / 'source_data'
    for directory in (figure_dir, table_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train, validation, training_summary = _read_training(root)
    test_frames = _read_test_results(root)
    algorithm_summary = _algorithm_summary(
        test_frames,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired, paired_topology, paired_instances = _paired_comparisons(
        test_frames,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    cross_seed = _load_cross_seed_summary(root)
    seed_level = _load_seed_level_test_summary(root)

    table_outputs = {
        'training_summary': table_dir / 'training_summary.csv',
        'algorithm_summary': table_dir / 'algorithm_summary.csv',
        'paired_comparisons': table_dir / 'paired_comparisons.csv',
        'paired_topology_comparisons': table_dir / 'paired_topology_comparisons.csv',
        'cross_seed_summary': table_dir / 'cross_seed_summary.csv',
    }
    training_summary.to_csv(table_outputs['training_summary'], index=False)
    algorithm_summary.to_csv(table_outputs['algorithm_summary'], index=False)
    paired.to_csv(table_outputs['paired_comparisons'], index=False)
    paired_topology.to_csv(table_outputs['paired_topology_comparisons'], index=False)
    cross_seed.to_csv(table_outputs['cross_seed_summary'], index=False)

    diagnostic_columns = [
        'seed', 'update', 'normalized_entropy', 'approx_kl', 'clip_fraction',
        'explained_variance', 'grad_norm', 'transitions_per_sec', 'update_elapsed_sec',
        'rollout_elapsed_sec', 'ppo_elapsed_sec', 'gpu_peak_allocated_mb',
        'gpu_peak_reserved_mb', 'illegal_action_count',
    ]
    overall_validation = validation.loc[validation['scope'] == 'overall'].copy()
    topology_validation = validation.loc[validation['scope'] == 'topology'].copy()
    train.loc[:, diagnostic_columns].to_csv(
        source_dir / 'training_diagnostics.csv', index=False
    )
    overall_validation.to_csv(source_dir / 'validation_overall.csv', index=False)
    topology_validation.to_csv(source_dir / 'validation_by_topology.csv', index=False)
    algorithm_summary.to_csv(source_dir / 'algorithm_comparison.csv', index=False)
    paired.to_csv(source_dir / 'paired_score_differences.csv', index=False)
    paired_instances.to_csv(source_dir / 'paired_instance_differences.csv', index=False)
    cross_seed.to_csv(source_dir / 'generalization_cross_seed.csv', index=False)
    seed_level.to_csv(source_dir / 'generalization_seed_level.csv', index=False)

    figure_outputs = []
    figure_outputs.extend(_plot_training(train, validation, figure_dir))
    figure_outputs.extend(_plot_algorithm_comparison(algorithm_summary, paired, figure_dir))
    figure_outputs.extend(_plot_generalization(cross_seed, seed_level, figure_dir))
    report_path = _write_report(
        output,
        training_summary,
        algorithm_summary,
        paired,
        cross_seed,
        bootstrap_samples=bootstrap_samples,
    )

    source_paths: list[Path] = [
        root / 'results' / f'formal_joint_seed{seed}_run01' / 'metrics' / name
        for seed in SEEDS
        for name in ('train_updates.csv', 'validation_updates.csv')
    ]
    source_paths.extend(
        [
            root / 'results' / 'final_evaluation' / 'seed1_best_test_interpolation_with_baselines.csv',
            root / 'results' / 'final_evaluation' / 'seed1_best_test_heldout_with_baselines.csv',
        ]
    )
    source_paths.extend(
        root / 'results' / 'final_evaluation' / f'{dataset}_multiseed' / 'ppo_cross_seed_summary.csv'
        for dataset in DATASETS
    )
    source_paths.extend(
        root / 'results' / 'final_evaluation' / f'{dataset}_multiseed' / 'ppo_seed_summary.csv'
        for dataset in DATASETS
    )
    manifest = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': 'python/matplotlib',
        'bootstrap_samples': bootstrap_samples,
        'bootstrap_seed': bootstrap_seed,
        'test_pairs_per_dataset': {
            dataset: _validate_test_pairing(frame, dataset)
            for dataset, frame in test_frames.items()
        },
        'source_files': [
            {'path': str(path.relative_to(root)), 'sha256': _sha256(path)}
            for path in source_paths
        ],
        'outputs': [
            str(path.relative_to(output))
            for path in [
                *table_outputs.values(),
                *figure_outputs,
                *sorted(source_dir.glob('*.csv')),
                report_path,
            ]
        ],
    }
    manifest_path = output / 'artifact_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return {
        'output_dir': str(output),
        'report': str(report_path),
        'manifest': str(manifest_path),
        'figures': [str(path) for path in figure_outputs],
        'tables': {name: str(path) for name, path in table_outputs.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output-dir', default='results/publication_ready_20260908')
    parser.add_argument('--bootstrap-samples', type=int, default=10_000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260908)
    args = parser.parse_args(argv)
    result = prepare_publication_results(
        args.project_root,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
