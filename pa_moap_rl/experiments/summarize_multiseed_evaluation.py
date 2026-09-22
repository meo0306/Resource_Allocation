'''Build paired and aggregate reports for multi-seed PPO test evaluations.'''

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in values for key in row})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row['instance_path'], row['scenario_id']


def _scope_rows(rows: list[dict[str, str]]) -> list[tuple[str, str, list[dict[str, str]]]]:
    scoped: list[tuple[str, str, list[dict[str, str]]]] = [('overall', 'all', rows)]
    for topology in sorted({row['topology'] for row in rows}):
        scoped.append(
            (
                'topology',
                topology,
                [row for row in rows if row['topology'] == topology],
            )
        )
    return scoped


def summarize_multiseed_evaluation(
    runs: dict[str, str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    '''Validate exact test pairing and summarize PPO scores across seeds.'''

    if len(runs) < 2:
        raise ValueError('At least two seed evaluations are required.')
    loaded: dict[str, list[dict[str, str]]] = {}
    for seed_id, path in runs.items():
        policy_rows = [
            row for row in _read_csv(Path(path))
            if row['solver_name'] == 'ppo_shared_policy'
        ]
        if not policy_rows:
            raise ValueError(f'No PPO rows found for {seed_id}.')
        keys = [_pair_key(row) for row in policy_rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate PPO test pairs for {seed_id}.')
        loaded[seed_id] = policy_rows
    reference_id = sorted(loaded)[0]
    reference_keys = {_pair_key(row) for row in loaded[reference_id]}
    mismatched = [
        seed_id for seed_id, rows in loaded.items()
        if {_pair_key(row) for row in rows} != reference_keys
    ]
    if mismatched:
        raise ValueError(f'Test pair mismatch for seeds: {mismatched}')

    paired_rows: list[dict[str, Any]] = []
    seed_summary: list[dict[str, Any]] = []
    for seed_id in sorted(loaded):
        rows = loaded[seed_id]
        for row in rows:
            paired_rows.append({**row, 'evaluation_seed_id': seed_id})
        for scope, scope_value, scope_values in _scope_rows(rows):
            scores = [float(row['total_score']) for row in scope_values]
            runtimes = [float(row['runtime']) for row in scope_values]
            seed_summary.append(
                {
                    'seed_id': seed_id,
                    'scope': scope,
                    'scope_value': scope_value,
                    'count': len(scope_values),
                    'total_score_mean': mean(scores),
                    'total_score_std': pstdev(scores),
                    'total_score_min': min(scores),
                    'total_score_max': max(scores),
                    'runtime_mean': mean(runtimes),
                    'hard_feasible_rate_min': min(
                        float(row['hard_feasible_rate']) for row in scope_values
                    ),
                    'mask_violation_count_sum': sum(
                        int(float(row['mask_violation_count'])) for row in scope_values
                    ),
                }
            )
    aggregate_rows: list[dict[str, Any]] = []
    scopes = sorted({(row['scope'], row['scope_value']) for row in seed_summary})
    for scope, scope_value in scopes:
        rows = [
            row for row in seed_summary
            if row['scope'] == scope and row['scope_value'] == scope_value
        ]
        values = [row['total_score_mean'] for row in rows]
        aggregate_rows.append(
            {
                'scope': scope,
                'scope_value': scope_value,
                'seed_count': len(values),
                'seed_mean': mean(values),
                'seed_population_std': pstdev(values),
                'seed_min': min(values),
                'seed_max': max(values),
                'all_hard_feasible': all(
                    row['hard_feasible_rate_min'] == 1.0 for row in rows
                ),
                'mask_violation_count_sum': sum(
                    row['mask_violation_count_sum'] for row in rows
                ),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'ppo_paired_results.csv', paired_rows)
    _write_csv(output / 'ppo_seed_summary.csv', seed_summary)
    _write_csv(output / 'ppo_cross_seed_summary.csv', aggregate_rows)
    result = {
        'schema_version': 1,
        'seed_ids': sorted(loaded),
        'pairs_per_seed': len(reference_keys),
        'all_pairs_match': True,
        'seed_summary': seed_summary,
        'cross_seed_summary': aggregate_rows,
    }
    (output / 'ppo_multiseed_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return result


def _parse_run(value: str) -> tuple[str, Path]:
    seed_id, separator, path = value.partition('=')
    if not separator or not seed_id or not path:
        raise argparse.ArgumentTypeError('Run must use SEED_ID=CSV_PATH.')
    return seed_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, type=_parse_run)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError('Seed IDs must be unique.')
    result = summarize_multiseed_evaluation(runs, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
