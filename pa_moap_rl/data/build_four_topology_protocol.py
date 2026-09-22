"""Build and audit the exploratory four-topology v2 data protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pa_moap_rl.data.scenarios import (
    PreferenceScenario,
    generate_scenario_bank,
    load_scenario_config,
    save_scenario_bank,
    scenario_pairs,
)


ALLOWED_TOPOLOGIES = ('bottleneck', 'hourglass', 'random', 'uniform')
ROLE_SOURCES = {
    'dev_train': 'train.csv',
    'dev_calibration': 'validation.csv',
    'dev_seen_diagnostic': 'test.csv',
}
EXPECTED = {
    'dev_train': {'families': 630, 'instances': 2520},
    'dev_calibration': {'families': 135, 'instances': 540},
    'dev_seen_diagnostic': {'families': 135, 'instances': 540},
}
CONTROLLED_PAIRS = (
    ('controlled_balanced', 'S6', 'T6'),
    ('controlled_student_only', 'S4', 'T6'),
    ('controlled_teacher_only', 'S6', 'T4'),
    ('controlled_conflict', 'S1', 'T4'),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f'Refusing to write empty table: {path}.')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f'Inconsistent CSV fields for {path}.')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _family_id(row: dict[str, str]) -> str:
    value = row.get('source_identity')
    if value:
        return value.replace('\\', '/')
    return f"{row['group']}/{Path(row['instance_name']).stem}"


def _enrich_rows(
    rows: list[dict[str, str]],
    *,
    role: str,
    project_root: Path,
    hash_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row['topology'] in ALLOWED_TOPOLOGIES]
    result: list[dict[str, Any]] = []
    for row in filtered:
        paths = {
            'source_sm_sha256': _resolve(project_root, row['source_sm']),
            'source_csv_sha256': _resolve(project_root, row['source_csv']),
            'output_json_sha256': _resolve(project_root, row['output_json']),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Missing protocol input paths: {missing[:5]}.')
        hashes: dict[str, str] = {}
        for key, path in paths.items():
            resolved = path.resolve()
            if resolved not in hash_cache:
                hash_cache[resolved] = sha256_file(resolved)
            hashes[key] = hash_cache[resolved]
        match = re.search(r'inst_V(\d+)', row['instance_name'])
        if match is None:
            raise ValueError(f"Cannot parse V from {row['instance_name']!r}.")
        access_status = (
            'previously_viewed_legacy_results'
            if role == 'dev_seen_diagnostic'
            else 'development_access'
        )
        result.append(
            {
                'original_id': row['instance_uid'],
                'source_family_id': _family_id(row),
                'topology': row['topology'],
                'size_group': row['group'],
                'V': int(match.group(1)),
                'N': int(row['n']),
                'M': int(row['m']),
                'K': int(row['k']),
                'source_sm': row['source_sm'],
                'source_csv': row['source_csv'],
                'output_json': row['output_json'],
                'old_split': row.get('split', ''),
                'research_role': role,
                'access_status': access_status,
                **hashes,
            }
        )
    return sorted(
        result,
        key=lambda row: (row['size_group'], row['source_family_id'], row['topology']),
    )


def _audit_roles(role_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    families_by_role: dict[str, set[str]] = {}
    role_summary: dict[str, Any] = {}
    for role, rows in role_rows.items():
        families = {str(row['source_family_id']) for row in rows}
        families_by_role[role] = families
        topology_counts = Counter(str(row['topology']) for row in rows)
        group_counts = Counter(str(row['size_group']) for row in rows)
        variants: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            variants[str(row['source_family_id'])].add(str(row['topology']))
        bad_families = [
            family
            for family, topologies in variants.items()
            if topologies != set(ALLOWED_TOPOLOGIES)
        ]
        expected = EXPECTED[role]
        if len(rows) != expected['instances'] or len(families) != expected['families']:
            raise ValueError(
                f'{role} count mismatch: {len(rows)} rows/{len(families)} families; '
                f'expected {expected}.'
            )
        if bad_families:
            raise ValueError(f'{role} families do not have exactly four topologies: {bad_families[:5]}.')
        role_summary[role] = {
            'instances': len(rows),
            'source_families': len(families),
            'topology_counts': dict(sorted(topology_counts.items())),
            'group_counts': dict(sorted(group_counts.items())),
            'access_status': sorted({str(row['access_status']) for row in rows}),
        }
    intersections: dict[str, int] = {}
    roles = list(role_rows)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = families_by_role[left] & families_by_role[right]
            intersections[f'{left}__{right}'] = len(overlap)
            if overlap:
                raise ValueError(f'Family leakage between {left} and {right}: {sorted(overlap)[:5]}.')
    return {'roles': role_summary, 'family_intersections': intersections}


def _select_calibration_core(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group_family[str(row['size_group'])][str(row['source_family_id'])].append(row)
    selected: list[dict[str, Any]] = []
    if len(by_group_family) != 9:
        raise ValueError(f'Expected 9 size groups, found {len(by_group_family)}.')
    for group in sorted(by_group_family):
        families = by_group_family[group]
        means = {
            family: float(np.mean([float(row['N']) for row in family_rows]))
            for family, family_rows in families.items()
        }
        values = np.asarray(list(means.values()), dtype=np.float64)
        targets = (float(np.quantile(values, 0.50)), float(np.quantile(values, 0.75)))
        chosen: list[str] = []
        for target in targets:
            candidates = sorted(means, key=lambda family: (abs(means[family] - target), family))
            family = next(item for item in candidates if item not in chosen)
            chosen.append(family)
        for family in chosen:
            selected.extend(families[family])
    selected = sorted(
        selected,
        key=lambda row: (row['size_group'], row['source_family_id'], row['topology']),
    )
    if len(selected) != 72 or len({row['source_family_id'] for row in selected}) != 18:
        raise ValueError('Calibration core must contain 72 rows from 18 source families.')
    return selected


def _controlled_scenarios(config: dict[str, Any]) -> list[PreferenceScenario]:
    perturbation_hash = _canonical_hash(
        {'mode': 'none', 'source_perturbation_config': config['perturbation']}
    )
    held_out = set(scenario_pairs(config, held_out=True))
    result: list[PreferenceScenario] = []
    for scenario_id, student_id, teacher_id in CONTROLLED_PAIRS:
        result.append(
            PreferenceScenario(
                scenario_id=scenario_id,
                student_template=student_id,
                teacher_template=teacher_id,
                student_profile=np.asarray(
                    config['student_templates'][student_id]['values'], dtype=np.float64
                ),
                teacher_profile=np.asarray(
                    config['teacher_templates'][teacher_id]['values'], dtype=np.float64
                ),
                target_distribution=None,
                weights=None,
                held_out=(student_id, teacher_id) in held_out,
                seed=0,
                schema_version=2,
                access_status='controlled_calibration',
                perturbation_config_hash=perturbation_hash,
            )
        )
    return result


def build_protocol(
    *,
    project_root: str | Path,
    split_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    splits = Path(split_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing protocol directory: {output}.')

    hash_cache: dict[Path, str] = {}
    role_rows: dict[str, list[dict[str, Any]]] = {}
    source_manifest_hashes: dict[str, str] = {}
    for role, filename in ROLE_SOURCES.items():
        path = splits / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        source_manifest_hashes[filename] = sha256_file(path)
        role_rows[role] = _enrich_rows(
            _read_csv(path), role=role, project_root=root, hash_cache=hash_cache
        )
    audit = _audit_roles(role_rows)
    core = _select_calibration_core(role_rows['dev_calibration'])
    scenario_config = load_scenario_config()
    controlled = _controlled_scenarios(scenario_config)
    interpolation = generate_scenario_bank(
        count=30,
        seed=2026091201,
        held_out=False,
        scenario_config=scenario_config,
        objective_independent=True,
        access_status='calibration_interpolation',
    )

    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=str(output.parent)))
    try:
        for role, rows in role_rows.items():
            _write_csv(temporary / f'{role}.csv', rows)
        _write_csv(temporary / 'calibration_core_72.csv', core)
        access_rows = [
            {
                'original_id': row['original_id'],
                'source_family_id': row['source_family_id'],
                'research_role': row['research_role'],
                'access_status': row['access_status'],
                'previously_viewed': str(
                    row['research_role'] == 'dev_seen_diagnostic'
                ).lower(),
            }
            for role in ROLE_SOURCES
            for row in role_rows[role]
        ]
        _write_csv(temporary / 'access_registry.csv', access_rows)
        save_scenario_bank(controlled, temporary / 'controlled_scenarios_v2.json')
        save_scenario_bank(interpolation, temporary / 'interpolation_scenarios_v2.json')
        screen_pairs = [
            {
                'pair_id': f'screen-{index:04d}',
                'original_id': row['original_id'],
                'source_family_id': row['source_family_id'],
                'topology': row['topology'],
                'size_group': row['size_group'],
                'scenario_id': controlled[index % len(controlled)].scenario_id,
            }
            for index, row in enumerate(core)
        ]
        _write_csv(temporary / 'screen_pairs_72.csv', screen_pairs)
        cross_pairs = [
            {
                'pair_id': f'controlled-{instance_index:04d}-{scenario_index:02d}',
                'original_id': row['original_id'],
                'source_family_id': row['source_family_id'],
                'topology': row['topology'],
                'size_group': row['size_group'],
                'scenario_id': scenario.scenario_id,
            }
            for instance_index, row in enumerate(core)
            for scenario_index, scenario in enumerate(controlled)
        ]
        _write_csv(temporary / 'controlled_pairs_288.csv', cross_pairs)
        full_pairs = [
            {
                'pair_id': f'full-{index:04d}',
                'original_id': row['original_id'],
                'source_family_id': row['source_family_id'],
                'topology': row['topology'],
                'size_group': row['size_group'],
                'scenario_id': interpolation[index % len(interpolation)].scenario_id,
            }
            for index, row in enumerate(role_rows['dev_calibration'])
        ]
        _write_csv(temporary / 'full_calibration_pairs_540.csv', full_pairs)

        generated_hashes = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != 'data_audit.json'
        }
        audit.update(
            {
                'schema_version': 2,
                'protocol_id': 'four_topology_exploratory_v2',
                'allowed_topologies': list(ALLOWED_TOPOLOGIES),
                'excluded_topologies': ['staircase'],
                'lineage_policy': 'preserve_existing_family_split_without_reshuffle',
                'dev_seen_diagnostic_notice': (
                    'Legacy results for this role were previously viewed; it is not an independent test set.'
                ),
                'calibration_core': {'instances': 72, 'source_families': 18},
                'controlled_cross_pairs': len(cross_pairs),
                'full_calibration_pairs': len(full_pairs),
                'source_manifest_hashes': source_manifest_hashes,
                'generated_file_hashes': generated_hashes,
            }
        )
        (temporary / 'data_audit.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
            encoding='utf-8',
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--split-dir', type=Path, default=Path('data_processed/splits'))
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data_processed/four_topology_exploratory_v2'),
    )
    args = parser.parse_args()
    audit = build_protocol(
        project_root=args.project_root,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
