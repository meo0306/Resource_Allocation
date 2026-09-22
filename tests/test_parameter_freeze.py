'''Tests for the three-seed PPO parameter freeze decision.'''

from __future__ import annotations

import csv
from pathlib import Path

from pa_moap_rl.experiments.freeze_ppo_parameters import (
    evaluate_parameter_freeze,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed(
    root: Path,
    best_score: float,
    *,
    entropy: float = 0.60,
    max_update: int = 1000,
) -> None:
    updates = (1, max_update - 25, max_update)
    scores = (best_score - 0.004, best_score, best_score - 0.003)
    metrics = root / 'metrics'
    _write(
        metrics / 'train_updates.csv',
        [
            {
                'update': update,
                'approx_kl': 0.02,
                'normalized_entropy': entropy,
                'illegal_action_count': 0,
            }
            for update in updates
        ],
    )
    validation_updates: list[dict[str, object]] = []
    validation_instances: list[dict[str, object]] = []
    for update, score in zip(updates, scores):
        validation_updates.append(
            {
                'update': update,
                'scope': 'overall',
                'scope_value': 'all',
                'total_score_mean': score,
            }
        )
        for topology in ('random', 'staircase'):
            validation_updates.append(
                {
                    'update': update,
                    'scope': 'topology',
                    'scope_value': topology,
                    'total_score_mean': score,
                }
            )
            validation_instances.append(
                {
                    'update': update,
                    'topology': topology,
                    'hard_feasible_rate': 1.0,
                }
            )
    _write(metrics / 'validation_updates.csv', validation_updates)
    _write(metrics / 'validation_instances.csv', validation_instances)


def test_parameter_freeze_passes_and_requests_extension(tmp_path: Path) -> None:
    runs: dict[str, Path] = {}
    for index, score in enumerate((0.664, 0.663, 0.662)):
        seed_id = f'seed{index}'
        runs[seed_id] = tmp_path / seed_id
        _seed(runs[seed_id], score)
    output = tmp_path / 'freeze'
    decision = evaluate_parameter_freeze(
        runs,
        reference_seed_id='seed0',
        max_update=1000,
        output_dir=output,
    )
    assert decision['parameters_frozen'] is True
    assert decision['extend_all_to_1500'] is True
    assert decision['aggregate']['best_score_median'] == 0.663
    assert next(
        row for row in decision['topology_summary']
        if row['topology'] == 'staircase'
    )['passes'] is True
    assert (output / 'freeze_decision_report.md').is_file()


def test_parameter_freeze_rejects_entropy_collapse(tmp_path: Path) -> None:
    runs: dict[str, Path] = {}
    for index, score in enumerate((0.664, 0.663, 0.662)):
        seed_id = f'seed{index}'
        runs[seed_id] = tmp_path / seed_id
        _seed(runs[seed_id], score, entropy=0.40 if index == 2 else 0.60)
    decision = evaluate_parameter_freeze(
        runs,
        reference_seed_id='seed0',
        max_update=1000,
        output_dir=tmp_path / 'freeze',
    )
    assert decision['parameters_frozen'] is False
    assert decision['extend_all_to_1500'] is False
    assert 'entropy_min' in decision['failed_gates']


def test_completed_1500_run_is_not_recommended_for_repeat_extension(
    tmp_path: Path,
) -> None:
    runs: dict[str, Path] = {}
    for index, score in enumerate((0.664, 0.663, 0.662)):
        seed_id = f'seed{index}'
        runs[seed_id] = tmp_path / seed_id
        _seed(runs[seed_id], score, max_update=1500)
    decision = evaluate_parameter_freeze(
        runs,
        reference_seed_id='seed0',
        max_update=1500,
        output_dir=tmp_path / 'freeze',
    )
    assert decision['parameters_frozen'] is True
    assert decision['extend_all_to_1500'] is False
