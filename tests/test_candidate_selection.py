'''Tests for deterministic PPO candidate selection artifacts.'''

from __future__ import annotations

import csv
from pathlib import Path

from pa_moap_rl.experiments.select_ppo_candidates import select_candidates


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate(
    root: Path,
    *,
    scores: tuple[float, float],
    kl: float,
    schedule_suffix: str = '',
) -> None:
    metrics = root / 'metrics'
    _write(
        metrics / 'train_updates.csv',
        [
            {
                'update': update,
                'approx_kl': kl,
                'clip_fraction': 0.10,
                'normalized_entropy': 0.70,
                'illegal_action_count': 0,
            }
            for update in (1, 2)
        ],
    )
    _write(
        metrics / 'train_instances.csv',
        [
            {
                'update': update,
                'environment_index': environment,
                'instance_path': f'instance-{update}-{environment}{schedule_suffix}',
                'scenario_id': f'scenario-{update}-{environment}',
            }
            for update in (1, 2)
            for environment in (0, 1)
        ],
    )
    validation_updates: list[dict[str, object]] = []
    validation_instances: list[dict[str, object]] = []
    for update, score in zip((1, 2), scores):
        validation_updates.append(
            {
                'update': update,
                'scope': 'overall',
                'scope_value': 'all',
                'total_score_mean': score,
            }
        )
        for topology in ('random', 'uniform'):
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
                    'instance_path': f'validation-{topology}',
                    'scenario_id': f'scenario-{topology}',
                    'topology': topology,
                    'total_score': score,
                    'hard_feasible_rate': 1.0,
                }
            )
    _write(metrics / 'validation_updates.csv', validation_updates)
    _write(metrics / 'validation_instances.csv', validation_instances)


def test_candidate_selection_emits_ranked_artifacts(tmp_path: Path) -> None:
    baseline = tmp_path / 'baseline'
    stable = tmp_path / 'stable'
    _candidate(baseline, scores=(0.65, 0.66), kl=0.04)
    _candidate(stable, scores=(0.65, 0.66), kl=0.02)
    output = tmp_path / 'selection'
    decision = select_candidates(
        {'B': baseline, 'C1': stable},
        baseline_id='B',
        max_update=2,
        output_dir=output,
        advance_count=1,
    )
    assert decision['advancing_candidates'] == ['C1']
    assert decision['all_candidates_schedule_match'] is True
    for name in (
        'candidate_summary.csv',
        'training_schedule_pairs.csv',
        'validation_pairs.csv',
        'topology_comparison.csv',
        'decision.json',
        'decision_report.md',
    ):
        assert (output / name).is_file()


def test_schedule_mismatch_disqualifies_candidate(tmp_path: Path) -> None:
    baseline = tmp_path / 'baseline'
    mismatch = tmp_path / 'mismatch'
    _candidate(baseline, scores=(0.65, 0.66), kl=0.04)
    _candidate(mismatch, scores=(0.66, 0.67), kl=0.02, schedule_suffix='-other')
    decision = select_candidates(
        {'B': baseline, 'C1': mismatch},
        baseline_id='B',
        max_update=2,
        output_dir=tmp_path / 'selection',
        advance_count=2,
    )
    candidate = next(
        row for row in decision['candidate_summary'] if row['candidate_id'] == 'C1'
    )
    assert candidate['eligible'] is False
    assert 'training_schedule_mismatch' in candidate['disqualification_reasons']
