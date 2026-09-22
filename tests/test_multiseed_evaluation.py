'''Tests for paired multi-seed PPO test summaries.'''

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pa_moap_rl.experiments.summarize_multiseed_evaluation import (
    summarize_multiseed_evaluation,
)


def _evaluation(path: Path, score: float, *, suffix: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            'instance_path': f'instance-{index}{suffix}',
            'scenario_id': f'scenario-{index}',
            'topology': 'staircase' if index == 0 else 'uniform',
            'solver_name': 'ppo_shared_policy',
            'total_score': score + index * 0.01,
            'runtime': 0.1,
            'hard_feasible_rate': 1.0,
            'mask_violation_count': 0,
        }
        for index in range(2)
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_multiseed_summary_emits_paired_topology_statistics(tmp_path: Path) -> None:
    runs: dict[str, Path] = {}
    for index, score in enumerate((0.60, 0.62, 0.64)):
        seed_id = f'seed{index}'
        runs[seed_id] = tmp_path / f'{seed_id}.csv'
        _evaluation(runs[seed_id], score)
    output = tmp_path / 'summary'
    result = summarize_multiseed_evaluation(runs, output_dir=output)

    assert result['all_pairs_match'] is True
    assert result['pairs_per_seed'] == 2
    overall = next(
        row for row in result['cross_seed_summary']
        if row['scope'] == 'overall'
    )
    assert overall['seed_mean'] == pytest.approx(0.625)
    assert overall['all_hard_feasible'] is True
    assert (output / 'ppo_paired_results.csv').is_file()


def test_multiseed_summary_rejects_pair_mismatch(tmp_path: Path) -> None:
    first = tmp_path / 'first.csv'
    second = tmp_path / 'second.csv'
    _evaluation(first, 0.60)
    _evaluation(second, 0.60, suffix='-other')

    with pytest.raises(ValueError, match='Test pair mismatch'):
        summarize_multiseed_evaluation(
            {'seed0': first, 'seed1': second},
            output_dir=tmp_path / 'summary',
        )
