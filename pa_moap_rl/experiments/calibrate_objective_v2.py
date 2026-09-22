"""Resumable exact calibration for the exploratory four-topology protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import PreferenceScenario, apply_scenario, load_scenario_bank
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.solvers.gurobi_exact_solver import solve_gurobi
from pa_moap_rl.utils.scoring import score_assignment


CONFIG_PATH = Path('pa_moap_rl/configs/objective_calibration_v2.yaml')
PROTOCOL_DIR = Path('data_processed/four_topology_exploratory_v2')
RESULT_DIR = Path('results/objective_calibration_four_topology_v2')


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_output_hash_manifest(result_dir: Path) -> Path:
    output = result_dir / 'output_hashes.csv'
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.rglob('*')):
        if (
            not path.is_file()
            or path == output
            or path.name == 'run_metadata.json'
            or path.name.startswith('runner_stdout')
            or path.name.startswith('runner_stderr')
        ):
            continue
        if path.suffix == '.tmp':
            continue
        if path == output or path.name == 'run_metadata.json' or path.name.startswith('runner_'):
            continue
        rows.append(
            {
                'relative_path': path.relative_to(result_dir).as_posix(),
                'size_bytes': path.stat().st_size,
                'sha256': _sha256(path),
            }
        )
    _write_csv(output, rows)
    return output


def _semantic_key(spec: ObjectiveSpec) -> tuple[Any, ...]:
    entropy_min = spec.entropy_min if spec.beta_entropy else 0.0
    method_cap = spec.method_cap if spec.beta_cap else 1.0
    return (
        spec.alpha_effect,
        spec.alpha_preference,
        spec.alpha_global,
        entropy_min,
        method_cap,
        spec.beta_entropy,
        spec.beta_cap,
        spec.target_distribution,
        spec.epsilon,
    )


def enumerate_candidates(path: str | Path = CONFIG_PATH) -> list[ObjectiveSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    defaults = dict(data['defaults'])
    candidates: list[ObjectiveSpec] = []
    for raw in data['candidates']:
        candidates.append(ObjectiveSpec.from_dict({**defaults, **raw}))
    scan = data['scan']
    for structure in scan['structures']:
        target = None if float(structure['alpha_global']) == 0.0 else defaults['target_distribution']
        if scan.get('include_unpenalized_per_structure'):
            candidates.append(
                ObjectiveSpec.from_dict(
                    {
                        **defaults,
                        **structure,
                        'objective_id': f"{structure['name']}__unpenalized",
                        'target_distribution': target,
                    }
                )
            )
        for entropy_min in scan['entropy_min']:
            for method_cap in scan['method_cap']:
                for beta_entropy, beta_cap in scan['penalty_pairs']:
                    objective_id = (
                        f"{structure['name']}__h{float(entropy_min):.2f}"
                        f"__c{float(method_cap):.2f}__be{float(beta_entropy):.2f}"
                        f"__bc{float(beta_cap):.2f}"
                    )
                    candidates.append(
                        ObjectiveSpec.from_dict(
                            {
                                **defaults,
                                **structure,
                                'objective_id': objective_id,
                                'entropy_min': entropy_min,
                                'method_cap': method_cap,
                                'beta_entropy': beta_entropy,
                                'beta_cap': beta_cap,
                                'target_distribution': target,
                            }
                        )
                    )
    unique: dict[tuple[Any, ...], ObjectiveSpec] = {}
    for spec in candidates:
        unique.setdefault(_semantic_key(spec), spec)
    return sorted(unique.values(), key=lambda item: item.config_hash)


def _relative_gap(incumbent: float, bound: float) -> float:
    return max(0.0, bound - incumbent) / max(1.0e-12, abs(incumbent))


def _solve_task(task: dict[str, Any]) -> str:
    output = Path(task['output'])
    if output.is_file():
        return str(output)
    spec = ObjectiveSpec.from_dict(task['objective'])
    base = load_instance_json(task['instance_path'])
    scenario = PreferenceScenario.from_dict(task['scenario'])
    instance = apply_scenario(base, scenario, objective=spec)
    log_dir = Path(task['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    stage1 = solve_gurobi(
        instance,
        time_limit=float(task['stage1_seconds']),
        seed=int(task['seed']),
        threads=1,
        log_file=str(log_dir / 'stage1.log'),
        objective=spec,
    )
    selected = stage1
    stages = [{'stage': 1, 'new_search_tree': True, 'runtime': stage1.runtime, **stage1.metrics}]
    stage1_gap = _relative_gap(
        float(stage1.score.total_score), float(stage1.metrics['objective_bound'])
    )
    if not stage1.metrics['proven_optimal'] and stage1_gap > 0.01:
        stage2 = solve_gurobi(
            instance,
            time_limit=float(task['stage2_seconds']),
            seed=int(task['seed']),
            threads=1,
            log_file=str(log_dir / 'stage2.log'),
            warm_start=stage1.assignment,
            objective=spec,
        )
        selected = stage2
        stages.append(
            {
                'stage': 2,
                'new_search_tree': True,
                'warm_started_from_stage1_incumbent': True,
                'runtime': stage2.runtime,
                **stage2.metrics,
            }
        )
    score = selected.score.as_dict()
    result = {
        'schema_version': 2,
        'phase': task['phase'],
        'pair_id': task['pair_id'],
        'original_id': task['original_id'],
        'source_family_id': task['source_family_id'],
        'topology': task['topology'],
        'size_group': task['size_group'],
        'scenario_id': scenario.scenario_id,
        'student_template': scenario.student_template,
        'teacher_template': scenario.teacher_template,
        'objective': spec.to_dict(),
        'objective_hash': spec.config_hash,
        'assignment': selected.assignment.astype(int).tolist(),
        'score': score,
        'max_method_share': float(max(score['method_distribution'])),
        'proven_optimal': bool(selected.metrics['proven_optimal']),
        'objective_incumbent': float(selected.score.total_score),
        'objective_bound': float(selected.metrics['objective_bound']),
        'certified_gap': _relative_gap(
            float(selected.score.total_score), float(selected.metrics['objective_bound'])
        ),
        'objective_recompute_error': float(selected.metrics['objective_recompute_error']),
        'stages': stages,
        'total_wall_runtime': float(sum(float(stage['runtime']) for stage in stages)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(output)
    return str(output)


def _tasks_for_phase(
    *,
    phase: str,
    pair_file: Path,
    scenario_file: Path,
    manifest_file: Path,
    candidates: Iterable[ObjectiveSpec],
    result_dir: Path,
    stage1_seconds: float,
    stage2_seconds: float,
    seed: int,
) -> list[dict[str, Any]]:
    manifests = {row['original_id']: row for row in _read_csv(manifest_file)}
    scenarios = {item.scenario_id: item for item in load_scenario_bank(scenario_file)}
    tasks: list[dict[str, Any]] = []
    pairs = _read_csv(pair_file)
    for spec in candidates:
        for pair in pairs:
            manifest = manifests[pair['original_id']]
            scenario = scenarios[pair['scenario_id']]
            output = result_dir / phase / 'instances' / spec.config_hash / f"{pair['pair_id']}.json"
            tasks.append(
                {
                    'phase': phase,
                    'pair_id': pair['pair_id'],
                    'original_id': pair['original_id'],
                    'source_family_id': pair['source_family_id'],
                    'topology': pair['topology'],
                    'size_group': pair['size_group'],
                    'instance_path': manifest['output_json'],
                    'scenario': scenario.to_dict(),
                    'objective': spec.to_dict(include_hash=False),
                    'output': str(output),
                    'log_dir': str(result_dir / 'logs' / phase / spec.config_hash / pair['pair_id']),
                    'stage1_seconds': stage1_seconds,
                    'stage2_seconds': stage2_seconds,
                    'seed': seed,
                }
            )
    return tasks


def _old_profile_tasks(
    *,
    manifest_file: Path,
    candidates: Iterable[ObjectiveSpec],
    result_dir: Path,
    stage1_seconds: float,
    stage2_seconds: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows = _read_csv(manifest_file)
    tasks: list[dict[str, Any]] = []
    for spec in candidates:
        for index, row in enumerate(rows):
            base = load_instance_json(row['output_json'])
            scenario = PreferenceScenario(
                scenario_id='legacy_embedded_profile',
                student_template='embedded',
                teacher_template='embedded',
                student_profile=base.student_profile.copy(),
                teacher_profile=base.teacher_profile.copy(),
                target_distribution=None,
                weights=None,
                held_out=False,
                seed=0,
                schema_version=2,
                access_status='previously_used_embedded_profile',
                perturbation_config_hash='none',
            )
            pair_id = f'old-profile-{index:04d}'
            output = (
                result_dir / 'old_profile' / 'instances' / spec.config_hash / f'{pair_id}.json'
            )
            tasks.append(
                {
                    'phase': 'old_profile',
                    'pair_id': pair_id,
                    'original_id': row['original_id'],
                    'source_family_id': row['source_family_id'],
                    'topology': row['topology'],
                    'size_group': row['size_group'],
                    'instance_path': row['output_json'],
                    'scenario': scenario.to_dict(),
                    'objective': spec.to_dict(include_hash=False),
                    'output': str(output),
                    'log_dir': str(
                        result_dir / 'logs' / 'old_profile' / spec.config_hash / pair_id
                    ),
                    'stage1_seconds': stage1_seconds,
                    'stage2_seconds': stage2_seconds,
                    'seed': seed,
                }
            )
    return tasks


def _profile_reuse_rows(
    *,
    old_results: list[dict[str, Any]],
    controlled_results: list[dict[str, Any]],
    manifest_file: Path,
    scenario_file: Path,
) -> list[dict[str, Any]]:
    manifests = {row['original_id']: row for row in _read_csv(manifest_file)}
    scenarios = {item.scenario_id: item for item in load_scenario_bank(scenario_file)}
    old_lookup = {
        (item['objective_hash'], item['original_id']): item for item in old_results
    }
    output: list[dict[str, Any]] = []
    for item in controlled_results:
        old = old_lookup[(item['objective_hash'], item['original_id'])]
        spec = ObjectiveSpec.from_dict(item['objective'])
        base = load_instance_json(manifests[item['original_id']]['output_json'])
        conditioned = apply_scenario(
            base, scenarios[item['scenario_id']], objective=spec
        )
        reused = score_assignment(
            old['assignment'], instance=conditioned, objective=spec
        ).total_score
        old_score = float(old['score']['total_score'])
        reoptimized = float(item['score']['total_score'])
        output.append(
            {
                'objective_id': spec.objective_id,
                'objective_hash': spec.config_hash,
                'original_id': item['original_id'],
                'topology': item['topology'],
                'size_group': item['size_group'],
                'scenario_id': item['scenario_id'],
                'old_profile_optimum': old_score,
                'old_assignment_under_new_profile': reused,
                'profile_shift_loss': old_score - reused,
                'resolve_gain': reoptimized - reused,
                'new_profile_optimum': reoptimized,
            }
        )
    return output


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[Path]:
    pending = [task for task in tasks if not Path(task['output']).is_file()]
    if workers <= 1:
        for index, task in enumerate(pending, 1):
            _solve_task(task)
            if index % 25 == 0:
                print(f'completed {index}/{len(pending)} pending solves', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_solve_task, task) for task in pending]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 25 == 0:
                    print(f'completed {index}/{len(pending)} pending solves', flush=True)
    return [Path(task['output']) for task in tasks]


def _load_results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding='utf-8')) for path in paths]


def aggregate_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result['objective_hash'], []).append(result)
    rows: list[dict[str, Any]] = []
    for objective_hash, values in grouped.items():
        spec = values[0]['objective']
        strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for value in values:
            strata.setdefault((value['topology'], value['size_group']), []).append(value)
        proof_rate = float(np.mean([item['proven_optimal'] for item in values]))
        remaining = [float(item['certified_gap']) for item in values if not item['proven_optimal']]
        stratum_proof_min = min(
            float(np.mean([item['proven_optimal'] for item in items]))
            for items in strata.values()
        )
        scores = [item['score'] for item in values]
        row = {
            'objective_id': spec['objective_id'],
            'objective_hash': objective_hash,
            'alpha_effect': spec['alpha_effect'],
            'alpha_preference': spec['alpha_preference'],
            'alpha_global': spec['alpha_global'],
            'entropy_min': spec['entropy_min'],
            'method_cap': spec['method_cap'],
            'beta_entropy': spec['beta_entropy'],
            'beta_cap': spec['beta_cap'],
            'samples': len(values),
            'mean_total_score': float(np.mean([score['total_score'] for score in scores])),
            'mean_effect': float(np.mean([score['effect_score'] for score in scores])),
            'mean_student': float(np.mean([score['student_pref_score'] for score in scores])),
            'mean_teacher': float(np.mean([score['teacher_pref_score'] for score in scores])),
            'mean_preference': float(
                np.mean(
                    [
                        (score['student_pref_score'] + score['teacher_pref_score']) / 2.0
                        for score in scores
                    ]
                )
            ),
            'mean_global': float(np.mean([score['global_score'] for score in scores])),
            'mean_entropy': float(np.mean([score['entropy'] for score in scores])),
            'mean_max_method_share': float(np.mean([item['max_method_share'] for item in values])),
            'mean_diversity_violation': float(
                np.mean([score['diversity_violation'] for score in scores])
            ),
            'mean_cap_violation': float(np.mean([score['cap_violation'] for score in scores])),
            'mean_entropy_penalty': float(
                np.mean([score['entropy_penalty_contribution'] for score in scores])
            ),
            'mean_cap_penalty': float(
                np.mean([score['cap_penalty_contribution'] for score in scores])
            ),
            'mean_effect_contribution': float(
                np.mean([score['effect_contribution'] for score in scores])
            ),
            'mean_student_contribution': float(
                np.mean([score['student_contribution'] for score in scores])
            ),
            'mean_teacher_contribution': float(
                np.mean([score['teacher_contribution'] for score in scores])
            ),
            'mean_preference_contribution': float(
                np.mean(
                    [
                        score['student_contribution'] + score['teacher_contribution']
                        for score in scores
                    ]
                )
            ),
            'mean_global_contribution': float(
                np.mean([score['global_contribution'] for score in scores])
            ),
            'proof_rate': proof_rate,
            'max_unproven_gap': max(remaining, default=0.0),
            'min_topology_size_proof_rate': stratum_proof_min,
            'max_recompute_error': max(float(item['objective_recompute_error']) for item in values),
            'total_solver_wall_seconds': float(sum(item['total_wall_runtime'] for item in values)),
        }
        row['eligible'] = (
            row['proof_rate'] >= 0.95
            and row['max_unproven_gap'] <= 0.01
            and row['min_topology_size_proof_rate'] >= 0.90
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row['objective_hash'])


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = (
        left['mean_effect'],
        left['mean_preference'],
        left['mean_entropy'],
        -left['mean_max_method_share'],
    )
    right_values = (
        right['mean_effect'],
        right['mean_preference'],
        right['mean_entropy'],
        -right['mean_max_method_share'],
    )
    return all(a >= b - 1.0e-12 for a, b in zip(left_values, right_values)) and any(
        a > b + 1.0e-12 for a, b in zip(left_values, right_values)
    )


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row['eligible']]
    return [
        row
        for row in eligible
        if not any(
            _dominates(other, row)
            for other in eligible
            if other['objective_hash'] != row['objective_hash']
        )
    ]


def select_by_crowding(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return sorted(rows, key=lambda row: row['objective_hash'])
    metrics = (
        ('mean_effect', True),
        ('mean_preference', True),
        ('mean_entropy', True),
        ('mean_max_method_share', False),
    )
    selected: set[str] = set()
    distance = {row['objective_hash']: 0.0 for row in rows}
    for key, maximize in metrics:
        ordered = sorted(rows, key=lambda row: (row[key], row['objective_hash']))
        selected.add((ordered[-1] if maximize else ordered[0])['objective_hash'])
        selected.add((ordered[0] if maximize else ordered[-1])['objective_hash'])
        span = float(ordered[-1][key] - ordered[0][key])
        if span <= 1.0e-15:
            continue
        distance[ordered[0]['objective_hash']] = math.inf
        distance[ordered[-1]['objective_hash']] = math.inf
        for index in range(1, len(ordered) - 1):
            distance[ordered[index]['objective_hash']] += (
                float(ordered[index + 1][key]) - float(ordered[index - 1][key])
            ) / span
    structures = (
        [row for row in rows if float(row['alpha_global']) == 0.0],
        [row for row in rows if abs(float(row['alpha_global']) - 0.05) <= 1.0e-12],
    )
    for options in structures:
        if options:
            selected.add(
                max(options, key=lambda row: (distance[row['objective_hash']], row['objective_hash']))[
                    'objective_hash'
                ]
            )
    ranked = sorted(
        rows,
        key=lambda row: (-distance[row['objective_hash']], row['objective_hash']),
    )
    for row in ranked:
        if len(selected) >= limit:
            break
        selected.add(row['objective_hash'])
    lookup = {row['objective_hash']: row for row in rows}
    return [lookup[key] for key in sorted(selected)[:limit]]


def _add_unpenalized_comparisons(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    by_hash_pair = {
        (item['objective_hash'], item['pair_id']): item for item in results
    }
    baseline_hash: dict[str, str] = {}
    for row in rows:
        if float(row['beta_entropy']) == 0.0 and float(row['beta_cap']) == 0.0:
            structure = (
                float(row['alpha_effect']),
                float(row['alpha_preference']),
                float(row['alpha_global']),
            )
            baseline_hash[str(structure)] = row['objective_hash']
    pairs_by_hash: dict[str, list[str]] = {}
    for item in results:
        pairs_by_hash.setdefault(item['objective_hash'], []).append(item['pair_id'])
    for row in rows:
        structure = str(
            (
                float(row['alpha_effect']),
                float(row['alpha_preference']),
                float(row['alpha_global']),
            )
        )
        reference = baseline_hash.get(structure)
        if reference is None:
            row['effect_loss_vs_unpenalized'] = ''
            row['preference_gain_vs_unpenalized'] = ''
            continue
        effect_deltas: list[float] = []
        preference_deltas: list[float] = []
        for pair_id in pairs_by_hash.get(row['objective_hash'], []):
            current = by_hash_pair[(row['objective_hash'], pair_id)]['score']
            baseline = by_hash_pair.get((reference, pair_id))
            if baseline is None:
                continue
            base_score = baseline['score']
            effect_deltas.append(base_score['effect_score'] - current['effect_score'])
            preference_deltas.append(
                (
                    current['student_pref_score']
                    + current['teacher_pref_score']
                    - base_score['student_pref_score']
                    - base_score['teacher_pref_score']
                )
                / 2.0
            )
        row['effect_loss_vs_unpenalized'] = float(np.mean(effect_deltas)) if effect_deltas else ''
        row['preference_gain_vs_unpenalized'] = (
            float(np.mean(preference_deltas)) if preference_deltas else ''
        )


def _stratified_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = ('topology', 'size_group', 'scenario_id')
    output: list[dict[str, Any]] = []
    for dimension in dimensions:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in results:
            grouped.setdefault((item['objective_hash'], item[dimension]), []).append(item)
        for (objective_hash, stratum), values in sorted(grouped.items()):
            totals = np.asarray([item['score']['total_score'] for item in values], dtype=float)
            output.append(
                {
                    'objective_hash': objective_hash,
                    'objective_id': values[0]['objective']['objective_id'],
                    'dimension': dimension,
                    'stratum': stratum,
                    'samples': len(values),
                    'mean_total_score': float(np.mean(totals)),
                    'median_total_score': float(np.median(totals)),
                    'p90_loss_from_best': float(np.quantile(np.max(totals) - totals, 0.90)),
                    'worst_total_score': float(np.min(totals)),
                    'mean_effect': float(
                        np.mean([item['score']['effect_score'] for item in values])
                    ),
                    'mean_preference': float(
                        np.mean(
                            [
                                (
                                    item['score']['student_pref_score']
                                    + item['score']['teacher_pref_score']
                                )
                                / 2.0
                                for item in values
                            ]
                        )
                    ),
                    'mean_entropy': float(np.mean([item['score']['entropy'] for item in values])),
                    'worst_max_method_share': float(
                        np.max([item['max_method_share'] for item in values])
                    ),
                }
            )
    return output


def _plot_pareto(rows: list[dict[str, Any]], selected: set[str], path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    scatter = axis.scatter(
        [row['mean_effect'] for row in rows],
        [row['mean_preference'] for row in rows],
        c=[row['mean_entropy'] for row in rows],
        s=[30 + 500 * (1.0 - row['mean_max_method_share']) for row in rows],
        cmap='viridis',
        alpha=0.72,
    )
    chosen = [row for row in rows if row['objective_hash'] in selected]
    axis.scatter(
        [row['mean_effect'] for row in chosen],
        [row['mean_preference'] for row in chosen],
        facecolors='none',
        edgecolors='crimson',
        s=130,
        linewidths=1.5,
        label='shortlist',
    )
    axis.set_xlabel('Mean F_E')
    axis.set_ylabel('Mean F_P')
    axis.set_title('Four-topology objective calibration Pareto candidates')
    axis.legend()
    figure.colorbar(scatter, ax=axis, label='Mean normalized entropy')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    *,
    result_dir: Path,
    screen_rows: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    controlled_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    profile_reuse_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        '# 四拓扑目标校准 Pareto 决策报告',
        '',
        f'- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}',
        f'- 筛选候选数：{len(screen_rows)}',
        f'- 首轮 Pareto 复核数：{len(shortlist_rows)}',
        f'- 完整 calibration 复核数：{len(final_rows)}',
        '- 状态：仅提出候选建议；目标尚未冻结。',
        '- PPO：未用于目标筛选，本阶段未启动 PPO 训练。',
        '',
        '## 求解门槛',
        '',
    ]
    eligible = [row for row in screen_rows if row['eligible']]
    lines.append(
        f'{len(eligible)}/{len(screen_rows)} 个候选达到 95% 最优证明率、其余 gap≤1%、'
        '拓扑—规模层证明率≥90% 的门槛。'
    )
    lines.extend(['', '## 最终候选', ''])
    if final_rows:
        lines.append(
            '| objective_id | αE/αP/αG | Hmin/cap | βH/βC | F_E | F_P | entropy | max share |'
        )
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for row in final_rows:
            lines.append(
                f"| {row['objective_id']} | {row['alpha_effect']:.2f}/"
                f"{row['alpha_preference']:.2f}/{row['alpha_global']:.2f} | "
                f"{row['entropy_min']:.2f}/{row['method_cap']:.2f} | "
                f"{row['beta_entropy']:.2f}/{row['beta_cap']:.2f} | "
                f"{row['mean_effect']:.6f} | {row['mean_preference']:.6f} | "
                f"{row['mean_entropy']:.6f} | {row['mean_max_method_share']:.6f} |"
            )
    else:
        lines.append('没有候选通过求解门槛，因此未执行强行排名。')
    references: list[dict[str, Any]] = []
    historical = [
        row for row in screen_rows if row['objective_id'] == 'historical_equivalent_anchor'
    ]
    if historical:
        references.append(historical[0])
    for global_weight in (0.0, 0.05):
        options = [
            row
            for row in screen_rows
            if row['eligible'] and abs(float(row['alpha_global']) - global_weight) <= 1.0e-12
        ]
        if options:
            references.append(
                max(
                    options,
                    key=lambda row: (
                        row['mean_effect']
                        + row['mean_preference']
                        + row['mean_entropy']
                        - row['mean_max_method_share'],
                        row['objective_hash'],
                    ),
                )
            )
    lines.extend(['', '## 必备参照', ''])
    for row in references:
        lines.append(
            f"- {row['objective_id']} ({row['objective_hash'][:12]}): "
            f"F_E={row['mean_effect']:.6f}, F_P={row['mean_preference']:.6f}, "
            f"H={row['mean_entropy']:.6f}, max share={row['mean_max_method_share']:.6f}."
        )
    if final_rows:
        metrics = (
            ('mean_effect', True),
            ('mean_preference', True),
            ('mean_entropy', True),
            ('mean_max_method_share', False),
        )
        composite = {row['objective_hash']: 0.0 for row in final_rows}
        for key, maximize in metrics:
            values = np.asarray([float(row[key]) for row in final_rows])
            span = float(values.max() - values.min())
            for row in final_rows:
                normalized = 0.5 if span <= 1.0e-15 else (
                    (float(row[key]) - float(values.min())) / span
                )
                composite[row['objective_hash']] += normalized if maximize else 1.0 - normalized
        recommendation = max(
            final_rows,
            key=lambda row: (composite[row['objective_hash']], row['objective_hash']),
        )
        lines.extend(
            [
                '',
                '## 推荐意见（待冻结）',
                '',
                f"建议优先审阅 {recommendation['objective_id']} "
                f"({recommendation['objective_hash']})。该建议只依据四个等权标准化 Pareto "
                '维度的平衡位置，不代表目标已经冻结，也不构成教学因果证明。',
            ]
        )
    reuse = profile_reuse_rows or []
    if reuse:
        lines.extend(
            [
                '',
                '## 旧画像解复用与重求解',
                '',
                f"- 旧画像 assignment 在新画像下的平均分数变化："
                f"{float(np.mean([row['profile_shift_loss'] for row in reuse])):.6f}",
                f"- 在新画像下重新精确求解的平均增益："
                f"{float(np.mean([row['resolve_gain'] for row in reuse])):.6f}",
                f"- 重新求解增益最差值："
                f"{float(np.min([row['resolve_gain'] for row in reuse])):.6f}",
            ]
        )
    lines.extend(
        [
            '',
            '## 解释边界',
            '',
            '现有数据可以比较目标项、方法多样性和软约束之间的经验权衡，但不能单独证明'
            '某个系数在教学上具有因果最优性，也不能把“分布更均衡”直接解释为学习效果更好。',
            '最终目标需由用户结合教学意义冻结；本报告不会生成或暗示“已冻结目标”。',
        ]
    )
    (result_dir / 'pareto_decision_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _stratified_rows_v2(
    results: list[dict[str, Any]],
    *,
    phase: str,
    dimensions: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    maximize = {
        'total_score',
        'effect',
        'student',
        'teacher',
        'preference',
        'global',
        'entropy',
    }
    for source_field, dimension in dimensions:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in results:
            grouped.setdefault(
                (item['objective_hash'], str(item[source_field])), []
            ).append(item)
        for (objective_hash, stratum), values in sorted(grouped.items()):
            score_rows = [item['score'] for item in values]
            metric_values = {
                'total_score': np.asarray(
                    [score['total_score'] for score in score_rows], dtype=float
                ),
                'effect': np.asarray(
                    [score['effect_score'] for score in score_rows], dtype=float
                ),
                'student': np.asarray(
                    [score['student_pref_score'] for score in score_rows], dtype=float
                ),
                'teacher': np.asarray(
                    [score['teacher_pref_score'] for score in score_rows], dtype=float
                ),
                'preference': np.asarray(
                    [
                        (score['student_pref_score'] + score['teacher_pref_score']) / 2.0
                        for score in score_rows
                    ],
                    dtype=float,
                ),
                'global': np.asarray(
                    [score['global_score'] for score in score_rows], dtype=float
                ),
                'entropy': np.asarray(
                    [score['entropy'] for score in score_rows], dtype=float
                ),
                'max_method_share': np.asarray(
                    [item['max_method_share'] for item in values], dtype=float
                ),
                'diversity_violation': np.asarray(
                    [score['diversity_violation'] for score in score_rows], dtype=float
                ),
                'cap_violation': np.asarray(
                    [score['cap_violation'] for score in score_rows], dtype=float
                ),
            }
            row: dict[str, Any] = {
                'phase': phase,
                'objective_hash': objective_hash,
                'objective_id': values[0]['objective']['objective_id'],
                'dimension': dimension,
                'stratum': stratum,
                'samples': len(values),
            }
            for metric, array in metric_values.items():
                row[f'mean_{metric}'] = float(np.mean(array))
                row[f'median_{metric}'] = float(np.median(array))
                row[f'p90_{metric}'] = float(np.quantile(array, 0.90))
                row[f'worst_{metric}'] = (
                    float(np.min(array)) if metric in maximize else float(np.max(array))
                )
            totals = metric_values['total_score']
            row['p90_loss_from_best'] = float(
                np.quantile(float(np.max(totals)) - totals, 0.90)
            )
            output.append(row)
    return output


def _balanced_recommendation(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not rows:
        return None
    metrics = (
        ('mean_effect', True),
        ('mean_preference', True),
        ('mean_entropy', True),
        ('mean_max_method_share', False),
    )
    scores = {row['objective_hash']: 0.0 for row in rows}
    for key, maximize_metric in metrics:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        low = float(np.min(values))
        span = float(np.max(values) - low)
        for row in rows:
            normalized = 0.5 if span <= 1.0e-15 else (float(row[key]) - low) / span
            scores[row['objective_hash']] += (
                normalized if maximize_metric else 1.0 - normalized
            )
    return max(rows, key=lambda row: (scores[row['objective_hash']], row['objective_hash']))


def _write_report_v2(
    *,
    result_dir: Path,
    screen_rows: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    controlled_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    profile_reuse_rows: list[dict[str, Any]] | None = None,
) -> None:
    eligible_screen = [row for row in screen_rows if row['eligible']]
    eligible_controlled = [row for row in controlled_rows if row['eligible']]
    final_front = pareto_front(final_rows)
    decision_pool = final_front or [row for row in final_rows if row['eligible']]
    overall = _balanced_recommendation(decision_pool)
    main = _balanced_recommendation(
        [row for row in decision_pool if abs(float(row['alpha_global'])) <= 1.0e-12]
    )
    sensitivity = _balanced_recommendation(
        [
            row
            for row in decision_pool
            if abs(float(row['alpha_global']) - 0.05) <= 1.0e-12
        ]
    )
    max_final_error = max(
        (float(row['max_recompute_error']) for row in final_rows), default=0.0
    )
    min_final_proof = min(
        (float(row['proof_rate']) for row in final_rows), default=0.0
    )
    min_final_stratum = min(
        (float(row['min_topology_size_proof_rate']) for row in final_rows),
        default=0.0,
    )
    max_final_gap = max(
        (float(row['max_unproven_gap']) for row in final_rows), default=0.0
    )
    lines = [
        '# \u56db\u62d3\u6251\u76ee\u6807\u6821\u51c6 Pareto \u51b3\u7b56\u62a5\u544a',
        '',
        f'- \u751f\u6210\u65f6\u95f4\uff08UTC\uff09\uff1a{datetime.now(timezone.utc).isoformat()}',
        f'- \u626b\u63cf\u5019\u9009\uff1a{len(screen_rows)}\uff1b\u7b5b\u9009 Pareto \u590d\u6838\uff1a{len(shortlist_rows)}\uff1b\u5b8c\u6574\u590d\u6838\uff1a{len(final_rows)}',
        '- \u72b6\u6001\uff1a\u4ec5\u63d0\u51fa\u63a8\u8350\uff0c\u76ee\u6807\u5c1a\u672a\u51bb\u7ed3\u3002',
        '- PPO\uff1a\u672c\u9636\u6bb5\u672a\u542f\u52a8 PPO \u8bad\u7ec3\uff0c\u4e5f\u672a\u4f7f\u7528 PPO \u5206\u6570\u7b5b\u9009\u76ee\u6807\u3002',
        '',
        '## \u6570\u636e\u4e0e\u6c42\u89e3\u95e8\u69db',
        '',
        '- \u4f7f\u7528\u65e2\u5b9a\u56db\u62d3\u6251\u63a2\u7d22\u6027\u534f\u8bae\uff1b\u672a\u91cd\u65b0\u968f\u673a\u5212\u5206\uff0c\u672a\u751f\u6210\u65b0\u7684\u5e95\u5c42\u5185\u5bb9\u6570\u636e\u3002',
        f"- \u7b5b\u9009\u9636\u6bb5\u901a\u8fc7\u95e8\u69db\uff1a{len(eligible_screen)}/{len(screen_rows)}\u3002",
        f"- \u53d7\u63a7\u4ea4\u53c9\u9636\u6bb5\u901a\u8fc7\u95e8\u69db\uff1a{len(eligible_controlled)}/{len(controlled_rows)}\u3002",
        f"- \u6700\u7ec8 540 \u6761\u590d\u6838\uff1a\u6700\u4f4e\u8bc1\u4f18\u7387 {min_final_proof:.6f}\uff0c\u6700\u5dee\u62d3\u6251\u2014\u89c4\u6a21\u5c42\u8bc1\u4f18\u7387 {min_final_stratum:.6f}\uff0c\u6700\u5927\u672a\u8bc1\u4f18 gap {max_final_gap:.6g}\u3002",
        f"- \u6700\u5927\u8bc4\u5206\u91cd\u7b97\u8bef\u5dee\uff1a{max_final_error:.3e}\uff08\u9a8c\u6536\u9608\u503c 1e-9\uff09\u3002",
        '',
        '## \u6700\u7ec8\u5019\u9009\uff1a\u539f\u59cb\u5206\u9879\u4e0e\u52a0\u6743\u8d21\u732e',
        '',
        '| objective_id | \u03b1E/\u03b1P/\u03b1G | Hmin/cap | \u03b2H/\u03b2C | F_E | F_S | F_T | F_P | F_G | wE | wP | wG |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in final_rows:
        preference_contribution = float(
            row.get(
                'mean_preference_contribution',
                float(row['mean_student_contribution'])
                + float(row['mean_teacher_contribution']),
            )
        )
        lines.append(
            f"| {row['objective_id']} | {float(row['alpha_effect']):.2f}/"
            f"{float(row['alpha_preference']):.2f}/{float(row['alpha_global']):.2f} | "
            f"{float(row['entropy_min']):.2f}/{float(row['method_cap']):.2f} | "
            f"{float(row['beta_entropy']):.2f}/{float(row['beta_cap']):.2f} | "
            f"{float(row['mean_effect']):.6f} | {float(row['mean_student']):.6f} | "
            f"{float(row['mean_teacher']):.6f} | {float(row['mean_preference']):.6f} | "
            f"{float(row['mean_global']):.6f} | "
            f"{float(row['mean_effect_contribution']):.6f} | "
            f"{preference_contribution:.6f} | "
            f"{float(row['mean_global_contribution']):.6f} |"
        )
    lines.extend(
        [
            '',
            '\u5176\u4e2d wP \u4e3a\u5b66\u751f\u4e0e\u6559\u5e08\u52a0\u6743\u8d21\u732e\u4e4b\u548c\uff1b\u5b66\u751f\u3001\u6559\u5e08\u59cb\u7ec8\u5404\u5360 \u03b1P/2\u3002F_G=0 \u65f6\uff0cF_G \u4e0e wG \u8bb0\u4e3a 0\u3002',
            '',
            '## \u65b9\u6cd5\u5206\u5e03\u4e0e\u8f6f\u60e9\u7f5a',
            '',
            '| objective_id | entropy | max share | entropy violation | cap violation | entropy penalty | cap penalty | total score |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {row['objective_id']} | {float(row['mean_entropy']):.6f} | "
            f"{float(row['mean_max_method_share']):.6f} | "
            f"{float(row['mean_diversity_violation']):.6g} | "
            f"{float(row['mean_cap_violation']):.6g} | "
            f"{float(row['mean_entropy_penalty']):.6g} | "
            f"{float(row['mean_cap_penalty']):.6g} | "
            f"{float(row['mean_total_score']):.6f} |"
        )
    lines.extend(
        [
            '',
            '## \u76f8\u5bf9\u65e0\u60e9\u7f5a\u89e3\u7684\u53d8\u5316\uff0872 \u6761\u7b5b\u9009\u5bf9\uff09',
            '',
            '\u8be5\u6bd4\u8f83\u4f7f\u7528\u540c\u4e00\u7ed3\u6784\u7684 (\u03b2H,\u03b2C)=(0,0) \u7cbe\u786e\u89e3\u4f5c\u4e3a\u57fa\u7ebf\uff1b\u6b63\u7684\u6548\u679c\u635f\u5931\u8868\u793a\u60e9\u7f5a\u89e3\u964d\u4f4e\u4e86 F_E\uff0c\u6b63\u7684\u504f\u597d\u6536\u76ca\u8868\u793a\u63d0\u9ad8\u4e86 F_P\u3002',
            '',
            '| objective_id | effect loss | preference gain |',
            '|---|---:|---:|',
        ]
    )
    for row in final_rows:
        effect_loss = row.get('screen_effect_loss_vs_unpenalized', '')
        preference_gain = row.get('screen_preference_gain_vs_unpenalized', '')
        effect_text = 'n/a' if effect_loss == '' else f'{float(effect_loss):.6g}'
        preference_text = (
            'n/a' if preference_gain == '' else f'{float(preference_gain):.6g}'
        )
        lines.append(
            f"| {row['objective_id']} | {effect_text} | {preference_text} |"
        )
    historical = next(
        (
            row
            for row in controlled_rows
            if row['objective_id'] == 'historical_equivalent_anchor'
        ),
        next(
            (
                row
                for row in screen_rows
                if row['objective_id'] == 'historical_equivalent_anchor'
            ),
            None,
        ),
    )
    lines.extend(['', '## \u5fc5\u5907\u53c2\u7167\u4e0e\u63a8\u8350\uff08\u5f85\u51bb\u7ed3\uff09', ''])
    if historical is not None:
        lines.append(
            f"- \u5386\u53f2\u7b49\u4ef7\u951a\u70b9 {historical['objective_hash'][:12]}\uff1a"
            f"F_E={float(historical['mean_effect']):.6f}\uff0c"
            f"F_P={float(historical['mean_preference']):.6f}\uff0c"
            f"entropy={float(historical['mean_entropy']):.6f}\uff0c"
            f"max share={float(historical['mean_max_method_share']):.6f}\uff0c"
            f"eligible={historical['eligible']}\u3002"
        )
    if main is not None:
        lines.append(
            f"- F_G=0 \u4e3b\u5019\u9009\uff1a{main['objective_id']} "
            f"({main['objective_hash']})\u3002"
        )
    if sensitivity is not None:
        lines.append(
            f"- F_G=0.05 \u654f\u611f\u6027\u5019\u9009\uff1a{sensitivity['objective_id']} "
            f"({sensitivity['objective_hash']})\u3002"
        )
    if overall is not None:
        lines.append(
            f"- \u56db\u4e2a\u7b49\u6743\u6807\u51c6\u5316 Pareto \u7ef4\u5ea6\u4e0b\u7684\u4f18\u5148\u5ba1\u9605\u9879\uff1a{overall['objective_id']} "
            f"({overall['objective_hash']})\u3002\u8fd9\u53ea\u662f\u7ecf\u9a8c\u6298\u4e2d\u63a8\u8350\uff0c\u4e0d\u4ee3\u8868\u76ee\u6807\u5df2\u51bb\u7ed3\u3002"
        )
    reuse = profile_reuse_rows or []
    if reuse:
        shift = np.asarray([float(row['profile_shift_loss']) for row in reuse])
        gain = np.asarray([float(row['resolve_gain']) for row in reuse])
        lines.extend(
            [
                '',
                '## \u65e7\u753b\u50cf\u89e3\u590d\u7528\u4e0e\u91cd\u65b0\u6c42\u89e3',
                '',
                '\u201c\u753b\u50cf\u53d8\u5316\u5dee\u201d\u5b9a\u4e49\u4e3a\u65e7\u753b\u50cf\u6700\u4f18\u5206\u6570\u51cf\u53bb\u65e7 assignment \u5728\u65b0\u753b\u50cf\u4e0b\u7684\u5206\u6570\uff1b\u5b83\u53ef\u4e3a\u8d1f\uff0c\u56e0\u6b64\u4e0d\u5e94\u4e00\u6982\u89e3\u91ca\u4e3a\u635f\u5931\u3002',
                '',
                '| metric | mean | median | P90 | worst |',
                '|---|---:|---:|---:|---:|',
                f"| profile shift difference | {float(np.mean(shift)):.6f} | {float(np.median(shift)):.6f} | {float(np.quantile(shift, 0.90)):.6f} | {float(np.max(shift)):.6f} |",
                f"| re-solve gain | {float(np.mean(gain)):.6f} | {float(np.median(gain)):.6f} | {float(np.quantile(gain, 0.90)):.6f} | {float(np.min(gain)):.6f} |",
            ]
        )
    stratified_path = result_dir / 'stratified_statistics.csv'
    stratified_rows = _read_csv(stratified_path) if stratified_path.is_file() else []
    dimension_counts: dict[str, int] = {}
    for row in stratified_rows:
        dimension_counts[row['dimension']] = dimension_counts.get(row['dimension'], 0) + 1
    lines.extend(
        [
            '',
            '## \u5206\u5c42\u7edf\u8ba1\u4e0e\u56fe',
            '',
            '- [\u5b8c\u6574\u5019\u9009\u6c47\u603b](final_candidate_summary.csv)',
            '- [\u5206\u5c42\u7edf\u8ba1](stratified_statistics.csv)\uff1a\u6bcf\u4e2a\u5019\u9009\u6309\u62d3\u6251\u3001\u89c4\u6a21\u7ec4\u3001\u63d2\u503c\u753b\u50cf\u4e0e4\u7c7b\u53d7\u63a7\u753b\u50cf\u7ed9\u51fa mean\u3001median\u3001P90 \u548c worst\uff1b'
            + '\uff1b'.join(f'{key}={value} \u884c' for key, value in sorted(dimension_counts.items()))
            + '\u3002',
            '- [\u7b5b\u9009 Pareto \u56fe](screen_pareto.png)',
            '- [\u6700\u7ec8 Pareto \u56fe](final_pareto.png)',
            '- [\u65e7\u753b\u50cf\u590d\u7528\u4e0e\u91cd\u6c42\u89e3\u660e\u7ec6](old_profile_reuse_and_resolve_gain.csv)',
            '',
            '## \u89e3\u91ca\u8fb9\u754c',
            '',
            '\u73b0\u6709\u6570\u636e\u53ef\u4ee5\u6bd4\u8f83\u76ee\u6807\u5206\u9879\u3001\u65b9\u6cd5\u591a\u6837\u6027\u548c\u8f6f\u7ea6\u675f\u4e4b\u95f4\u7684\u7ecf\u9a8c\u6743\u8861\uff0c\u4f46\u4e0d\u80fd\u5355\u72ec\u8bc1\u660e\u67d0\u4e2a\u7cfb\u6570\u5728\u6559\u5b66\u4e0a\u5177\u6709\u56e0\u679c\u6700\u4f18\u6027\u3002',
            '\u65b9\u6cd5\u5206\u5e03\u66f4\u5747\u8861\u4e5f\u4e0d\u80fd\u76f4\u63a5\u89e3\u91ca\u4e3a\u5b66\u4e60\u6548\u679c\u66f4\u597d\u3002\u6700\u7ec8\u76ee\u6807\u4ecd\u9700\u7528\u6237\u7ed3\u5408\u6559\u5b66\u610f\u4e49\u51bb\u7ed3\uff1b\u672c\u62a5\u544a\u4e0d\u751f\u6210\u6216\u6697\u793a\u201c\u5df2\u51bb\u7ed3\u76ee\u6807\u201d\u3002',
        ]
    )
    (result_dir / 'pareto_decision_report.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )


def run_calibration(
    *,
    config_path: Path,
    protocol_dir: Path,
    result_dir: Path,
    workers: int,
    stage1_seconds: float,
    stage2_seconds: float,
    seed: int,
) -> dict[str, Any]:
    candidates = enumerate_candidates(config_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = result_dir / 'run_metadata.json'
    current_inputs = {
        'schema_version': 2,
        'status': 'running',
        'created_or_resumed_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_path': str(config_path.resolve()),
        'config_sha256': _sha256(config_path),
        'data_audit_sha256': _sha256(protocol_dir / 'data_audit.json'),
        'candidate_count': len(candidates),
        'gurobi_seed': seed,
        'gurobi_threads': 1,
        'stage1_seconds': stage1_seconds,
        'stage2_seconds': stage2_seconds,
        'python': sys.version,
        'platform': platform.platform(),
        'implementation_file_hashes': {
            str(path): _sha256(path)
            for path in (
                Path('pa_moap_rl/objective.py'),
                Path('pa_moap_rl/checkpointing.py'),
                Path('pa_moap_rl/utils/scoring.py'),
                Path('pa_moap_rl/solvers/gurobi_exact_solver.py'),
                Path('pa_moap_rl/experiments/calibrate_objective_v2.py'),
            )
        },
    }
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding='utf-8'))
        for key in ('config_sha256', 'data_audit_sha256', 'candidate_count'):
            if existing.get(key) != current_inputs[key]:
                raise ValueError(f'Resume metadata mismatch for {key}.')
    else:
        metadata_path.write_text(
            json.dumps(current_inputs, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    (result_dir / 'candidate_specs.json').write_text(
        json.dumps([spec.to_dict() for spec in candidates], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    screen_tasks = _tasks_for_phase(
        phase='screen',
        pair_file=protocol_dir / 'screen_pairs_72.csv',
        scenario_file=protocol_dir / 'controlled_scenarios_v2.json',
        manifest_file=protocol_dir / 'dev_calibration.csv',
        candidates=candidates,
        result_dir=result_dir,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        seed=seed,
    )
    screen_results = _load_results(run_tasks(screen_tasks, workers))
    screen_rows = aggregate_candidates(screen_results)
    _add_unpenalized_comparisons(screen_rows, screen_results)
    _write_csv(result_dir / 'screen_candidate_summary.csv', screen_rows)
    front = pareto_front(screen_rows)
    shortlist_rows = select_by_crowding(front, limit=12)
    _write_csv(result_dir / 'screen_pareto_shortlist.csv', shortlist_rows)
    _plot_pareto(
        screen_rows,
        {row['objective_hash'] for row in shortlist_rows},
        result_dir / 'screen_pareto.png',
    )
    if not shortlist_rows:
        _write_report_v2(
            result_dir=result_dir,
            screen_rows=screen_rows,
            shortlist_rows=[],
            controlled_rows=[],
            final_rows=[],
        )
        current_inputs['status'] = 'bounded_diagnostic_only'
        metadata_path.write_text(json.dumps(current_inputs, indent=2), encoding='utf-8')
        _write_output_hash_manifest(result_dir)
        return current_inputs

    specs = {spec.config_hash: spec for spec in candidates}
    shortlist_specs = [specs[row['objective_hash']] for row in shortlist_rows]
    old_profile_tasks = _old_profile_tasks(
        manifest_file=protocol_dir / 'calibration_core_72.csv',
        candidates=shortlist_specs,
        result_dir=result_dir,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        seed=seed,
    )
    old_profile_results = _load_results(run_tasks(old_profile_tasks, workers))
    controlled_tasks = _tasks_for_phase(
        phase='controlled_cross',
        pair_file=protocol_dir / 'controlled_pairs_288.csv',
        scenario_file=protocol_dir / 'controlled_scenarios_v2.json',
        manifest_file=protocol_dir / 'dev_calibration.csv',
        candidates=shortlist_specs,
        result_dir=result_dir,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        seed=seed,
    )
    controlled_results = _load_results(run_tasks(controlled_tasks, workers))
    profile_reuse_rows = _profile_reuse_rows(
        old_results=old_profile_results,
        controlled_results=controlled_results,
        manifest_file=protocol_dir / 'dev_calibration.csv',
        scenario_file=protocol_dir / 'controlled_scenarios_v2.json',
    )
    _write_csv(result_dir / 'old_profile_reuse_and_resolve_gain.csv', profile_reuse_rows)
    controlled_rows = aggregate_candidates(controlled_results)
    _write_csv(result_dir / 'controlled_candidate_summary.csv', controlled_rows)
    controlled_front = pareto_front(controlled_rows)
    final_pre_rows = select_by_crowding(controlled_front, limit=6)
    final_specs = [specs[row['objective_hash']] for row in final_pre_rows]

    full_tasks = _tasks_for_phase(
        phase='full_calibration',
        pair_file=protocol_dir / 'full_calibration_pairs_540.csv',
        scenario_file=protocol_dir / 'interpolation_scenarios_v2.json',
        manifest_file=protocol_dir / 'dev_calibration.csv',
        candidates=final_specs,
        result_dir=result_dir,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        seed=seed,
    )
    full_results = _load_results(run_tasks(full_tasks, workers))
    final_rows = aggregate_candidates(full_results)
    screen_lookup = {row['objective_hash']: row for row in screen_rows}
    for row in final_rows:
        screen_row = screen_lookup[row['objective_hash']]
        row['screen_effect_loss_vs_unpenalized'] = screen_row.get(
            'effect_loss_vs_unpenalized', ''
        )
        row['screen_preference_gain_vs_unpenalized'] = screen_row.get(
            'preference_gain_vs_unpenalized', ''
        )
    final_front_hashes = {
        row['objective_hash'] for row in pareto_front(final_rows)
    }
    for row in final_rows:
        row['final_nondominated'] = row['objective_hash'] in final_front_hashes
    _write_csv(result_dir / 'final_candidate_summary.csv', final_rows)
    final_hashes = {row['objective_hash'] for row in final_rows}
    controlled_final_results = [
        item for item in controlled_results if item['objective_hash'] in final_hashes
    ]
    stratified_rows = _stratified_rows_v2(
        full_results,
        phase='full_calibration',
        dimensions=(
            ('topology', 'topology'),
            ('size_group', 'size_group'),
            ('scenario_id', 'interpolation_profile'),
        ),
    )
    stratified_rows.extend(
        _stratified_rows_v2(
            controlled_final_results,
            phase='controlled_cross',
            dimensions=(('scenario_id', 'controlled_profile_type'),),
        )
    )
    _write_csv(result_dir / 'stratified_statistics.csv', stratified_rows)
    _plot_pareto(final_rows, final_front_hashes, result_dir / 'final_pareto.png')
    _write_report_v2(
        result_dir=result_dir,
        screen_rows=screen_rows,
        shortlist_rows=shortlist_rows,
        controlled_rows=controlled_rows,
        final_rows=final_rows,
        profile_reuse_rows=profile_reuse_rows,
    )
    current_inputs['status'] = 'complete_waiting_for_objective_freeze'
    current_inputs['screen_shortlist_hashes'] = [
        row['objective_hash'] for row in shortlist_rows
    ]
    current_inputs['final_candidate_hashes'] = [row['objective_hash'] for row in final_rows]
    current_inputs['final_pareto_hashes'] = sorted(final_front_hashes)
    current_inputs['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    output_hash_manifest = _write_output_hash_manifest(result_dir)
    current_inputs['output_hash_manifest'] = str(output_hash_manifest)
    current_inputs['output_hash_manifest_sha256'] = _sha256(output_hash_manifest)
    metadata_path.write_text(
        json.dumps(current_inputs, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    hash_manifest = _write_output_hash_manifest(result_dir)
    current_inputs['output_hash_manifest'] = str(hash_manifest)
    current_inputs['output_hash_manifest_sha256'] = _sha256(hash_manifest)
    metadata_path.write_text(
        json.dumps(current_inputs, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return current_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--protocol-dir', type=Path, default=PROTOCOL_DIR)
    parser.add_argument('--result-dir', type=Path, default=RESULT_DIR)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--stage1-seconds', type=float, default=120.0)
    parser.add_argument('--stage2-seconds', type=float, default=600.0)
    parser.add_argument('--seed', type=int, default=20260912)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive.')
    summary = run_calibration(
        config_path=args.config,
        protocol_dir=args.protocol_dir,
        result_dir=args.result_dir,
        workers=args.workers,
        stage1_seconds=args.stage1_seconds,
        stage2_seconds=args.stage2_seconds,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
