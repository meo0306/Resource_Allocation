'''Evaluate multi-seed PPO runs and decide whether parameters can be frozen.'''

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable


@dataclass(frozen=True)
class FreezeThresholds:
    best_score_median_minimum: float = 0.662
    best_score_std_maximum: float = 0.006
    best_to_last_four_gap_maximum: float = 0.015
    kl_p90_median_maximum: float = 0.05
    kl_p90_seed_maximum: float = 0.06
    entropy_minimum: float = 0.45
    topology_regression_tolerance: float = 0.005
    refresh_after_update: int = 800
    refresh_seed_minimum: int = 2


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = sorted({key for row in values for key in row})
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Cannot compute a quantile of an empty sequence.')
    return ordered[round((len(ordered) - 1) * probability)]


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_seed_metrics(run_dir: str | Path, max_update: int) -> dict[str, Any]:
    '''Summarize one completed seed without consulting any test artifact.'''

    root = Path(run_dir)
    metrics = root / 'metrics'
    train_updates = [
        row for row in _read_csv(metrics / 'train_updates.csv')
        if int(row['update']) <= max_update
    ]
    validation_updates = [
        row for row in _read_csv(metrics / 'validation_updates.csv')
        if int(row['update']) <= max_update
    ]
    validation_instances = [
        row for row in _read_csv(metrics / 'validation_instances.csv')
        if int(row['update']) <= max_update
    ]
    if not train_updates or int(train_updates[-1]['update']) != max_update:
        reached = int(train_updates[-1]['update']) if train_updates else 0
        raise ValueError(f'{root} reached update {reached}, expected {max_update}.')
    overall = [row for row in validation_updates if row['scope'] == 'overall']
    if not overall or int(overall[-1]['update']) != max_update:
        reached = int(overall[-1]['update']) if overall else 0
        raise ValueError(f'{root} validation reached update {reached}, expected {max_update}.')
    best_row = max(overall, key=lambda row: _float(row, 'total_score_mean'))
    best_update = int(best_row['update'])
    scores = [_float(row, 'total_score_mean') for row in overall]
    last_four_mean = mean(scores[-4:])
    best_score = _float(best_row, 'total_score_mean')
    topology_scores = {
        row['scope_value']: _float(row, 'total_score_mean')
        for row in validation_updates
        if row['scope'] == 'topology' and int(row['update']) == best_update
    }
    kl = [_float(row, 'approx_kl') for row in train_updates]
    entropy = [_float(row, 'normalized_entropy') for row in train_updates]
    summary = {
        'run_dir': str(root.resolve()),
        'max_update': max_update,
        'validation_points': len(overall),
        'best_validation_score': best_score,
        'best_update': best_update,
        'last_four_mean': last_four_mean,
        'best_to_last_four_gap': best_score - last_four_mean,
        'kl_p90': _quantile(kl, 0.90),
        'entropy_min': min(entropy),
        'illegal_action_count': sum(
            int(float(row['illegal_action_count'])) for row in train_updates
        ),
        'hard_feasible_rate_min': min(
            _float(row, 'hard_feasible_rate') for row in validation_instances
        ),
    }
    return {'summary': summary, 'topology_scores': topology_scores}


def _write_report(path: Path, decision: dict[str, Any]) -> None:
    lines = [
        '# PPO parameter freeze decision through update {}'.format(
            decision['max_update']
        ),
        '',
        '- Parameters frozen: {}'.format(decision['parameters_frozen']),
        '- Extend all seeds to 1500: {}'.format(decision['extend_all_to_1500']),
        '- Failed gates: {}'.format(';'.join(decision['failed_gates']) or 'none'),
        '- Median best validation score: {:.6f}'.format(
            decision['aggregate']['best_score_median']
        ),
        '- Best-score population standard deviation: {:.6f}'.format(
            decision['aggregate']['best_score_std']
        ),
        '- Median KL P90: {:.6f}'.format(
            decision['aggregate']['kl_p90_median']
        ),
        '- Seeds refreshing best after update 800: {}'.format(
            decision['aggregate']['refresh_seed_count']
        ),
        '',
        '| Seed | Best | Best update | Last-4 mean | Gap | KL P90 | Entropy min | Feasible | Illegal |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in decision['seed_summary']:
        lines.append(
            '| {seed_id} | {best_validation_score:.6f} | {best_update} | '
            '{last_four_mean:.6f} | {best_to_last_four_gap:.6f} | {kl_p90:.6f} | '
            '{entropy_min:.6f} | {hard_feasible_rate_min:.3f} | '
            '{illegal_action_count} |'.format(**row)
        )
    lines.extend(
        [
            '',
            '| Topology | Reference | Median | Mean | Min | Delta vs reference | Pass |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for row in decision['topology_summary']:
        lines.append(
            '| {topology} | {reference_score:.6f} | {score_median:.6f} | '
            '{score_mean:.6f} | {score_min:.6f} | {median_delta:.6f} | '
            '{passes} |'.format(**row)
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def evaluate_parameter_freeze(
    runs: dict[str, str | Path],
    *,
    reference_seed_id: str,
    max_update: int,
    output_dir: str | Path,
    thresholds: FreezeThresholds | None = None,
) -> dict[str, Any]:
    '''Apply the approved three-seed freeze and extension rules.'''

    if len(runs) != 3:
        raise ValueError('Exactly three seed runs are required.')
    if reference_seed_id not in runs:
        raise ValueError(f'Unknown reference seed: {reference_seed_id}')
    limits = thresholds or FreezeThresholds()
    loaded = {
        seed_id: load_seed_metrics(path, max_update)
        for seed_id, path in runs.items()
    }
    seed_rows: list[dict[str, Any]] = []
    for seed_id in sorted(loaded):
        row = dict(loaded[seed_id]['summary'])
        row['seed_id'] = seed_id
        row['refreshed_after_threshold'] = (
            row['best_update'] > limits.refresh_after_update
        )
        seed_rows.append(row)
    best_scores = [row['best_validation_score'] for row in seed_rows]
    kl_p90 = [row['kl_p90'] for row in seed_rows]
    reference = loaded[reference_seed_id]['topology_scores']
    topology_rows: list[dict[str, Any]] = []
    for topology in sorted(reference):
        values = [
            loaded[seed_id]['topology_scores'][topology]
            for seed_id in sorted(loaded)
        ]
        score_median = median(values)
        reference_score = reference[topology]
        median_delta = score_median - reference_score
        topology_rows.append(
            {
                'topology': topology,
                'reference_seed_id': reference_seed_id,
                'reference_score': reference_score,
                'score_median': score_median,
                'score_mean': mean(values),
                'score_min': min(values),
                'median_delta': median_delta,
                'passes': median_delta >= -limits.topology_regression_tolerance,
            }
        )
    aggregate = {
        'best_score_median': median(best_scores),
        'best_score_std': pstdev(best_scores),
        'kl_p90_median': median(kl_p90),
        'kl_p90_max': max(kl_p90),
        'refresh_seed_count': sum(
            row['refreshed_after_threshold'] for row in seed_rows
        ),
    }
    checks = (
        (
            all(row['hard_feasible_rate_min'] == 1.0 for row in seed_rows),
            'hard_feasibility',
        ),
        (all(row['illegal_action_count'] == 0 for row in seed_rows), 'illegal_actions'),
        (
            aggregate['best_score_median'] >= limits.best_score_median_minimum,
            'best_score_median',
        ),
        (
            aggregate['best_score_std'] <= limits.best_score_std_maximum,
            'best_score_std',
        ),
        (
            all(
                row['best_to_last_four_gap'] <= limits.best_to_last_four_gap_maximum
                for row in seed_rows
            ),
            'best_to_last_four_gap',
        ),
        (
            aggregate['kl_p90_median'] <= limits.kl_p90_median_maximum,
            'kl_p90_median',
        ),
        (
            aggregate['kl_p90_max'] <= limits.kl_p90_seed_maximum,
            'kl_p90_seed_max',
        ),
        (
            all(row['entropy_min'] >= limits.entropy_minimum for row in seed_rows),
            'entropy_min',
        ),
        (all(row['passes'] for row in topology_rows), 'topology_regression'),
    )
    failed = [name for passed, name in checks if not passed]
    frozen = not failed
    extend = (
        max_update < 1500
        and frozen
        and aggregate['refresh_seed_count'] >= limits.refresh_seed_minimum
    )
    decision = {
        'schema_version': 1,
        'max_update': max_update,
        'reference_seed_id': reference_seed_id,
        'parameters_frozen': frozen,
        'extend_all_to_1500': extend,
        'failed_gates': failed,
        'thresholds': asdict(limits),
        'aggregate': aggregate,
        'seed_summary': seed_rows,
        'topology_summary': topology_rows,
        'topology_rule': (
            'For each topology, the median score across seeds at each seed best '
            'checkpoint must be no more than 0.005 below the seed0 reference score.'
        ),
        'standard_deviation_definition': 'population standard deviation',
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'seed_summary.csv', seed_rows)
    _write_csv(output / 'topology_summary.csv', topology_rows)
    (output / 'freeze_decision.json').write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    _write_report(output / 'freeze_decision_report.md', decision)
    return decision


def _parse_run(value: str) -> tuple[str, Path]:
    seed_id, separator, path = value.partition('=')
    if not separator or not seed_id or not path:
        raise argparse.ArgumentTypeError('Run must use SEED_ID=RUN_DIR.')
    return seed_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, type=_parse_run)
    parser.add_argument('--reference-seed-id', default='seed0')
    parser.add_argument('--max-update', required=True, type=int)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError('Seed IDs must be unique.')
    decision = evaluate_parameter_freeze(
        runs,
        reference_seed_id=args.reference_seed_id,
        max_update=args.max_update,
        output_dir=args.output_dir,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
