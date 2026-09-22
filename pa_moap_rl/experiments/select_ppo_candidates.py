'''Select PPO parameter candidates from fixed-validation training artifacts.'''

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class SelectionThresholds:
    robust_score_tolerance: float = 0.002
    best_score_tolerance: float = 0.002
    topology_tolerance: float = 0.005
    entropy_minimum: float = 0.50
    entropy_end_minimum: float = 0.55
    kl_p90_maximum: float = 0.05
    clip_mean_maximum: float = 0.25
    quality_tie_tolerance: float = 0.001


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


def _schedule_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row['update']), int(row.get('environment_index') or 0)


def _schedule_value(row: dict[str, str]) -> tuple[str, str]:
    return row['instance_path'], row['scenario_id']


def _validation_key(row: dict[str, str]) -> tuple[int, str, str]:
    return int(row['update']), row['instance_path'], row['scenario_id']


def load_candidate_metrics(run_dir: str | Path, max_update: int) -> dict[str, Any]:
    '''Load and summarize one formal-training run through max_update.'''

    root = Path(run_dir)
    metrics = root / 'metrics'
    train_updates = [
        row for row in _read_csv(metrics / 'train_updates.csv')
        if int(row['update']) <= max_update
    ]
    train_instances = [
        row for row in _read_csv(metrics / 'train_instances.csv')
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
    topology_scores = {
        row['scope_value']: _float(row, 'total_score_mean')
        for row in validation_updates
        if row['scope'] == 'topology' and int(row['update']) == best_update
    }
    validation_scores = [_float(row, 'total_score_mean') for row in overall]
    validation_mean = mean(validation_scores)
    last_four_mean = mean(validation_scores[-4:])
    best_score = max(validation_scores)
    robust_score = 0.50 * validation_mean + 0.30 * last_four_mean + 0.20 * best_score
    kl = [_float(row, 'approx_kl') for row in train_updates]
    clip = [_float(row, 'clip_fraction') for row in train_updates]
    entropy = [_float(row, 'normalized_entropy') for row in train_updates]
    illegal_count = sum(int(float(row['illegal_action_count'])) for row in train_updates)
    hard_feasible = min(_float(row, 'hard_feasible_rate') for row in validation_instances)
    schedule = {_schedule_key(row): _schedule_value(row) for row in train_instances}
    validation = {_validation_key(row): row for row in validation_instances}
    summary = {
        'run_dir': str(root.resolve()),
        'max_update': max_update,
        'validation_points': len(overall),
        'validation_mean': validation_mean,
        'last_four_mean': last_four_mean,
        'best_validation_score': best_score,
        'best_update': best_update,
        'robust_quality_score': robust_score,
        'worst_topology_score': min(topology_scores.values()),
        'kl_mean': mean(kl),
        'kl_p90': _quantile(kl, 0.90),
        'kl_max': max(kl),
        'kl_gt_003_count': sum(value > 0.03 for value in kl),
        'kl_gt_003_rate': sum(value > 0.03 for value in kl) / len(kl),
        'clip_fraction_mean': mean(clip),
        'clip_fraction_p90': _quantile(clip, 0.90),
        'entropy_min': min(entropy),
        'entropy_end': entropy[-1],
        'illegal_action_count': illegal_count,
        'hard_feasible_rate_min': hard_feasible,
    }
    return {
        'summary': summary,
        'topology_scores': topology_scores,
        'schedule': schedule,
        'validation': validation,
    }


def _rank_compare(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> int:
    quality_gap = float(left['robust_quality_score']) - float(right['robust_quality_score'])
    if abs(quality_gap) > tolerance:
        return -1 if quality_gap > 0 else 1
    for key in ('kl_p90', 'clip_fraction_mean'):
        gap = float(left[key]) - float(right[key])
        if abs(gap) > 1.0e-12:
            return -1 if gap < 0 else 1
    topology_gap = float(left['worst_topology_score']) - float(right['worst_topology_score'])
    if abs(topology_gap) > 1.0e-12:
        return -1 if topology_gap > 0 else 1
    if str(left['candidate_id']) == str(right['candidate_id']):
        return 0
    return -1 if str(left['candidate_id']) < str(right['candidate_id']) else 1


def _schedule_pairs(
    candidate_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    matches_all = True
    for key in sorted(set(baseline['schedule']) | set(candidate['schedule'])):
        base_value = baseline['schedule'].get(key)
        candidate_value = candidate['schedule'].get(key)
        matches = base_value == candidate_value
        matches_all = matches_all and matches
        rows.append(
            {
                'candidate_id': candidate_id,
                'update': key[0],
                'environment_index': key[1],
                'baseline_instance_path': base_value[0] if base_value else '',
                'baseline_scenario_id': base_value[1] if base_value else '',
                'candidate_instance_path': candidate_value[0] if candidate_value else '',
                'candidate_scenario_id': candidate_value[1] if candidate_value else '',
                'matches': matches,
            }
        )
    return matches_all, rows


def _topology_pairs(
    candidate_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    deltas: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    topologies = sorted(set(baseline['topology_scores']) | set(candidate['topology_scores']))
    for topology in topologies:
        base_score = baseline['topology_scores'].get(topology)
        candidate_score = candidate['topology_scores'].get(topology)
        delta = (
            candidate_score - base_score
            if base_score is not None and candidate_score is not None
            else float('-inf')
        )
        deltas[topology] = delta
        rows.append(
            {
                'candidate_id': candidate_id,
                'candidate_best_update': candidate['summary']['best_update'],
                'topology': topology,
                'baseline_score': base_score,
                'candidate_score': candidate_score,
                'delta': delta,
            }
        )
    return deltas, rows


def _validation_pairs(
    candidate_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(set(baseline['validation']) & set(candidate['validation']))
    for key in keys:
        baseline_row = baseline['validation'][key]
        candidate_row = candidate['validation'][key]
        baseline_score = _float(baseline_row, 'total_score')
        candidate_score = _float(candidate_row, 'total_score')
        rows.append(
            {
                'candidate_id': candidate_id,
                'update': key[0],
                'instance_path': key[1],
                'scenario_id': key[2],
                'topology': candidate_row['topology'],
                'baseline_score': baseline_score,
                'candidate_score': candidate_score,
                'delta': candidate_score - baseline_score,
            }
        )
    return rows


def _gate_summary(
    candidate_id: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    schedule_matches: bool,
    topology_deltas: dict[str, float],
    limits: SelectionThresholds,
) -> dict[str, Any]:
    summary = dict(candidate['summary'])
    baseline_summary = baseline['summary']
    reasons: list[str] = []
    checks = (
        (not schedule_matches, 'training_schedule_mismatch'),
        (summary['hard_feasible_rate_min'] < 1.0, 'hard_feasibility_below_one'),
        (summary['illegal_action_count'] != 0, 'illegal_actions'),
        (
            summary['robust_quality_score']
            < baseline_summary['robust_quality_score'] - limits.robust_score_tolerance,
            'robust_quality_below_tolerance',
        ),
        (
            summary['best_validation_score']
            < baseline_summary['best_validation_score'] - limits.best_score_tolerance,
            'best_score_below_tolerance',
        ),
        (
            any(delta < -limits.topology_tolerance for delta in topology_deltas.values()),
            'topology_regression',
        ),
        (summary['entropy_min'] < limits.entropy_minimum, 'entropy_min_below_threshold'),
        (summary['entropy_end'] < limits.entropy_end_minimum, 'entropy_end_below_threshold'),
        (summary['kl_p90'] > limits.kl_p90_maximum, 'kl_p90_above_threshold'),
        (summary['clip_fraction_mean'] > limits.clip_mean_maximum, 'clip_mean_above_threshold'),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    summary.update(
        {
            'candidate_id': candidate_id,
            'is_baseline': candidate is baseline,
            'schedule_matches': schedule_matches,
            'minimum_topology_delta': min(topology_deltas.values()),
            'eligible': not reasons,
            'disqualification_reasons': ';'.join(reasons),
        }
    )
    return summary


def _write_report(path: Path, decision: dict[str, Any]) -> None:
    advancing = decision['advancing_candidates']
    advancing_text = ', '.join(advancing) if advancing else 'none'
    max_update = decision['max_update']
    baseline_id = decision['baseline_id']
    schedules_match = decision['all_candidates_schedule_match']
    lines = [
        f'# PPO candidate selection through update {max_update}',
        '',
        f'- Baseline: `{baseline_id}`',
        f'- Advancing: `{advancing_text}`',
        f'- All training schedules match: `{schedules_match}`',
        '',
        '| Candidate | Eligible | Rank | S | Best | KL P90 | Clip mean | Min topology delta | Reasons |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in sorted(decision['candidate_summary'], key=lambda item: item['candidate_id']):
        lines.append(
            '| {candidate_id} | {eligible} | {eligible_rank} | {robust_quality_score:.6f} | '
            '{best_validation_score:.6f} | {kl_p90:.6f} | {clip_fraction_mean:.6f} | '
            '{minimum_topology_delta:.6f} | {disqualification_reasons} |'.format(**row)
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def select_candidates(
    candidates: dict[str, str | Path],
    *,
    baseline_id: str,
    max_update: int,
    output_dir: str | Path,
    advance_count: int = 2,
    thresholds: SelectionThresholds | None = None,
) -> dict[str, Any]:
    '''Evaluate candidates, enforce gates, and emit reproducible decision artifacts.'''

    if baseline_id not in candidates:
        raise ValueError(f'Unknown baseline candidate: {baseline_id}')
    limits = thresholds or SelectionThresholds()
    loaded = {
        candidate_id: load_candidate_metrics(path, max_update)
        for candidate_id, path in candidates.items()
    }
    baseline = loaded[baseline_id]
    schedule_rows: list[dict[str, Any]] = []
    validation_pair_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate_id, candidate in loaded.items():
        schedule_matches, candidate_schedule_rows = _schedule_pairs(
            candidate_id, baseline, candidate
        )
        topology_deltas, candidate_topology_rows = _topology_pairs(
            candidate_id, baseline, candidate
        )
        schedule_rows.extend(candidate_schedule_rows)
        topology_rows.extend(candidate_topology_rows)
        if candidate_id != baseline_id:
            validation_pair_rows.extend(
                _validation_pairs(candidate_id, baseline, candidate)
            )
        summaries.append(
            _gate_summary(
                candidate_id,
                candidate,
                baseline,
                schedule_matches=schedule_matches,
                topology_deltas=topology_deltas,
                limits=limits,
            )
        )
    eligible = [row for row in summaries if row['eligible']]
    ranked = sorted(
        eligible,
        key=cmp_to_key(
            lambda left, right: _rank_compare(
                left, right, limits.quality_tie_tolerance
            )
        ),
    )
    advancing = [row['candidate_id'] for row in ranked[:advance_count]]
    rank_lookup = {row['candidate_id']: index + 1 for index, row in enumerate(ranked)}
    for row in summaries:
        row['eligible_rank'] = rank_lookup.get(row['candidate_id'], '')
        row['advances'] = row['candidate_id'] in advancing
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'candidate_summary.csv', summaries)
    _write_csv(output / 'training_schedule_pairs.csv', schedule_rows)
    _write_csv(output / 'validation_pairs.csv', validation_pair_rows)
    _write_csv(output / 'topology_comparison.csv', topology_rows)
    decision = {
        'schema_version': 1,
        'baseline_id': baseline_id,
        'max_update': max_update,
        'advance_count': advance_count,
        'advancing_candidates': advancing,
        'all_candidates_schedule_match': all(
            row['schedule_matches'] for row in summaries
        ),
        'thresholds': limits.__dict__,
        'candidate_summary': summaries,
    }
    (output / 'decision.json').write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    _write_report(output / 'decision_report.md', decision)
    return decision


def _parse_candidate(value: str) -> tuple[str, Path]:
    candidate_id, separator, path = value.partition('=')
    if not separator or not candidate_id or not path:
        raise argparse.ArgumentTypeError('Candidate must use ID=RUN_DIR.')
    return candidate_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, type=_parse_candidate)
    parser.add_argument('--baseline-id', default='B')
    parser.add_argument('--max-update', required=True, type=int)
    parser.add_argument('--advance-count', default=2, type=int)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError('Candidate IDs must be unique.')
    result = select_candidates(
        candidates,
        baseline_id=args.baseline_id,
        max_update=args.max_update,
        output_dir=args.output_dir,
        advance_count=args.advance_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
