'''Build leakage-safe train/validation/test manifests and scenario banks.'''

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pa_moap_rl.data.scenarios import (
    generate_scenario_bank,
    load_scenario_config,
    save_scenario_bank,
)


SPLIT_RATIOS = {'train': 0.70, 'validation': 0.15, 'test': 0.15}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No rows found in manifest {path}.')
    required = {'topology', 'group', 'instance_name', 'output_json'}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f'Manifest lacks columns: {sorted(missing)}')
    return rows


def split_source_id(row: dict[str, str]) -> str:
    '''Keep all topology variants of the same group/seed in one split.'''

    return row['group'] + '/' + Path(row['instance_name']).stem


def assign_source_splits(
    rows: list[dict[str, str]],
    seed: int = 20260906,
) -> dict[str, str]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[row['group']].add(split_source_id(row))
    assignments: dict[str, str] = {}
    for group in sorted(by_group):
        identities = sorted(by_group[group])
        rng = np.random.default_rng(seed + sum(ord(char) for char in group))
        rng.shuffle(identities)
        train_end = int(round(len(identities) * SPLIT_RATIOS['train']))
        validation_end = train_end + int(round(len(identities) * SPLIT_RATIOS['validation']))
        for index, identity in enumerate(identities):
            split = 'train' if index < train_end else 'validation' if index < validation_end else 'test'
            assignments[identity] = split
    return assignments

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 20260906,
) -> dict[str, Any]:
    '''Write deterministic split manifests, audit summary, and scenario banks.'''

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir)
    source_rows = read_manifest(manifest_path)
    assignments = assign_source_splits(source_rows, seed=seed)
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_RATIOS}
    seen: dict[str, set[str]] = {name: set() for name in SPLIT_RATIOS}
    for raw in source_rows:
        row = dict(raw)
        identity = split_source_id(row)
        split = assignments[identity]
        output_json = Path(row['output_json'])
        if not output_json.is_absolute():
            cwd_candidate = output_json.resolve()
            output_json = (
                cwd_candidate
                if cwd_candidate.is_file()
                else (manifest_path.parent / output_json).resolve()
            )
        row['output_json'] = str(output_json)
        row['source_identity'] = identity
        row['split'] = split
        split_rows[split].append(row)
        seen[split].add(identity)
    if any(seen[left] & seen[right] for left in seen for right in seen if left < right):
        raise RuntimeError('Source identity leakage detected between splits.')

    fields = list(source_rows[0]) + ['source_identity', 'split']
    for split, rows in split_rows.items():
        _write_csv(output_dir / f'{split}.csv', rows, fields)

    topology_counts = {
        split: dict(Counter(row['topology'] for row in rows))
        for split, rows in split_rows.items()
    }
    summary = {
        'seed': seed,
        'ratios': SPLIT_RATIOS,
        'manifest': str(manifest_path),
        'row_counts': {split: len(rows) for split, rows in split_rows.items()},
        'source_identity_counts': {split: len(seen[split]) for split in seen},
        'topology_counts': topology_counts,
        'leakage_check': 'passed',
    }
    (output_dir / 'dataset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    scenario_config = load_scenario_config()
    scenario_dir = output_dir / 'scenarios'
    save_scenario_bank(
        generate_scenario_bank(30, seed + 1, scenario_config=scenario_config),
        scenario_dir / 'validation_interpolation.json',
    )
    save_scenario_bank(
        generate_scenario_bank(60, seed + 2, scenario_config=scenario_config),
        scenario_dir / 'test_interpolation.json',
    )
    save_scenario_bank(
        generate_scenario_bank(30, seed + 3, held_out=True, scenario_config=scenario_config),
        scenario_dir / 'test_heldout.json',
    )
    return summary

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data_processed/base_instances/manifest.csv')
    parser.add_argument('--output-dir', default='data_processed/splits')
    parser.add_argument('--seed', type=int, default=20260906)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_dataset(args.manifest, args.output_dir, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
