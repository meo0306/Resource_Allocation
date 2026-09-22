'''Prepare and audit the fixed exact-reference join for the four-topology seed0 pilot.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pa_moap_rl.experiments.formal_training import load_validation_pairs
from pa_moap_rl.objective import load_objective_spec


FIELDNAMES = (
    'pair_id',
    'original_id',
    'source_family_id',
    'topology',
    'size_group',
    'scenario_id',
    'output_json',
    'output_json_sha256',
    'exact_result_path',
    'exact_result_sha256',
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Manifest must not be empty: {path}')
    return rows


def _write_csv_atomic(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def _write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    temporary.replace(output)


def build_seed0_validation_join(
    *,
    core_manifest: str | Path,
    pair_manifest: str | Path,
    scenario_bank: str | Path,
    calibration_results: str | Path,
    objective_config: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    '''Join the fixed 72 pairs to immutable instances and b075 exact results.'''

    objective = load_objective_spec(objective_config)
    core_path = Path(core_manifest).resolve()
    pair_path = Path(pair_manifest).resolve()
    scenario_path = Path(scenario_bank).resolve()
    calibration_path = Path(calibration_results).resolve()
    output_path = Path(output_manifest).resolve()

    core_rows = _read_rows(core_path)
    pair_rows = _read_rows(pair_path)
    if len(pair_rows) != 72:
        raise ValueError(f'Seed0 pilot requires exactly 72 validation pairs, got {len(pair_rows)}.')
    core_by_id = {row['original_id']: row for row in core_rows}
    if len(core_by_id) != len(core_rows):
        raise ValueError('Core manifest contains duplicate original_id values.')

    exact_dir = calibration_path / 'screen' / 'instances' / objective.config_hash
    joined: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    strata: set[tuple[str, str]] = set()
    for pair in pair_rows:
        pair_id = pair['pair_id']
        if pair_id in seen_pairs:
            raise ValueError(f'Duplicate pair_id: {pair_id}')
        seen_pairs.add(pair_id)
        original_id = pair['original_id']
        if original_id not in core_by_id:
            raise ValueError(f'{pair_id} is absent from the 72-instance core manifest.')
        source = core_by_id[original_id]
        for key in ('source_family_id', 'topology', 'size_group'):
            if pair[key] != source[key]:
                raise ValueError(f'{pair_id} {key} does not match the core manifest.')
        instance_path = Path(source['output_json']).resolve()
        if not instance_path.is_file():
            raise FileNotFoundError(f'{pair_id} instance does not exist: {instance_path}')
        instance_hash = _sha256(instance_path)
        if instance_hash != source['output_json_sha256']:
            raise ValueError(f'{pair_id} instance hash differs from the audited core manifest.')

        exact_path = (exact_dir / f'{pair_id}.json').resolve()
        if not exact_path.is_file():
            raise FileNotFoundError(f'{pair_id} exact result does not exist: {exact_path}')
        exact = json.loads(exact_path.read_text(encoding='utf-8'))
        expected = {
            'pair_id': pair_id,
            'original_id': original_id,
            'scenario_id': pair['scenario_id'],
            'objective_hash': objective.config_hash,
        }
        for key, value in expected.items():
            if str(exact.get(key)) != str(value):
                raise ValueError(f'{pair_id} exact result {key} mismatch.')
        if not bool(exact.get('proven_optimal')):
            raise ValueError(f'{pair_id} exact result is not proven optimal.')
        if float(exact.get('certified_gap', float('inf'))) > 1.0e-12:
            raise ValueError(f'{pair_id} exact result has a nonzero certified gap.')
        if float(exact.get('objective_recompute_error', float('inf'))) > 1.0e-9:
            raise ValueError(f'{pair_id} exact result recomputation exceeds 1e-9.')
        joined.append(
            {
                'pair_id': pair_id,
                'original_id': original_id,
                'source_family_id': pair['source_family_id'],
                'topology': pair['topology'],
                'size_group': pair['size_group'],
                'scenario_id': pair['scenario_id'],
                'output_json': str(instance_path),
                'output_json_sha256': instance_hash,
                'exact_result_path': str(exact_path),
                'exact_result_sha256': _sha256(exact_path),
            }
        )
        strata.add((pair['size_group'], pair['topology']))

    if len(strata) != 9 * 4:
        raise ValueError(f'Expected all 36 size-topology strata, got {len(strata)}.')
    _write_csv_atomic(joined, output_path)
    verified = load_validation_pairs(
        output_path,
        scenario_bank=scenario_path,
        objective=objective,
        allow_controlled_held_out=True,
    )
    metadata = {
        'schema_version': 1,
        'pilot_id': 'four_topology_b075_seed0_pilot_v1',
        'status': 'ready',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'pair_count': len(verified),
        'size_topology_strata': len(strata),
        'objective_id': objective.objective_id,
        'objective_hash': objective.config_hash,
        'inputs': {
            'core_manifest': {'path': str(core_path), 'sha256': _sha256(core_path)},
            'pair_manifest': {'path': str(pair_path), 'sha256': _sha256(pair_path)},
            'scenario_bank': {'path': str(scenario_path), 'sha256': _sha256(scenario_path)},
            'objective_config': {
                'path': str(Path(objective_config).resolve()),
                'sha256': _sha256(objective_config),
            },
            'calibration_results': str(calibration_path),
        },
        'output_manifest': {'path': str(output_path), 'sha256': _sha256(output_path)},
    }
    metadata_path = output_path.with_suffix('.metadata.json')
    _write_json_atomic(metadata, metadata_path)
    metadata['metadata_path'] = str(metadata_path)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--core-manifest',
        default='data_processed/four_topology_exploratory_v2/calibration_core_72.csv',
    )
    parser.add_argument(
        '--pair-manifest',
        default='data_processed/four_topology_exploratory_v2/screen_pairs_72.csv',
    )
    parser.add_argument(
        '--scenario-bank',
        default='data_processed/four_topology_exploratory_v2/controlled_scenarios_v2.json',
    )
    parser.add_argument(
        '--calibration-results',
        default='results/objective_calibration_four_topology_v2',
    )
    parser.add_argument(
        '--objective-config',
        default='pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml',
    )
    parser.add_argument(
        '--output-manifest',
        default='data_processed/four_topology_exploratory_v2/seed0_pilot_validation_pairs_72.csv',
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = build_seed0_validation_join(
        core_manifest=args.core_manifest,
        pair_manifest=args.pair_manifest,
        scenario_bank=args.scenario_bank,
        calibration_results=args.calibration_results,
        objective_config=args.objective_config,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
