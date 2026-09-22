'''Summarize D/E controlled-cross response and endpoint mechanism gates.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import apply_scenario
from pa_moap_rl.envs.method_assignment_env import MethodAssignmentEnv
from pa_moap_rl.experiments.audit_preference_repair_checkpoint_trajectory_v2 import (
    _load_model,
    _screen_metrics,
)
from pa_moap_rl.experiments.run_preference_response_v2 import (
    _load_and_validate as load_preference_context,
)
from pa_moap_rl.experiments.summarize_preference_repair_ablation_v2 import (
    _cluster_ci,
    _fixed_72,
    _stats,
)
from pa_moap_rl.solvers.ppo_solver import resolve_device


SUMMARY_VERSION = 'preference_ranking_controlled_cross_summary_v1'
ARM_ORDER = ('D', 'E')
SCENARIOS = (
    'controlled_student_only',
    'controlled_teacher_only',
    'controlled_conflict',
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('Refusing to write an empty D/E summary CSV.')
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if (
        not isinstance(config, dict)
        or config.get('status') != 'protocol_confirmed_post_training_evaluation'
        or not config.get('run', {}).get('no_training', False)
    ):
        raise ValueError('D/E evaluation configuration is not confirmed as no-training.')
    protocol_path = Path(config['protocol']['path'])
    if _sha256(protocol_path) != config['protocol']['sha256']:
        raise ValueError('D/E protocol hash mismatch while evaluating.')
    protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'user_confirmed':
        raise ValueError('D/E training protocol is no longer user-confirmed.')
    config['_protocol'] = protocol
    config['_config_path'] = str(path)
    return config


def validate(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _load_config(path)
    context = load_preference_context(
        Path(config['preference_response_config']).resolve(),
        output_override=None,
    )
    if context['objective'].config_hash != config['objective']['config_hash']:
        raise ValueError('Frozen objective hash mismatch in D/E evaluation.')
    for arm_id in ARM_ORDER:
        arm = config['arms'][arm_id]
        checkpoint = Path(arm['checkpoint'])
        if _sha256(checkpoint) != arm['checkpoint_sha256']:
            raise ValueError('D/E checkpoint hash mismatch for arm ' + arm_id)
        evaluation_config = yaml.safe_load(
            Path(arm['evaluation_config']).read_text(encoding='utf-8')
        )
        if (
            evaluation_config['checkpoint']['arm_id'] != arm_id
            or evaluation_config['checkpoint']['checkpoint_sha256']
            != arm['checkpoint_sha256']
            or int(evaluation_config['checkpoint']['expected_update'])
            != int(arm['expected_update'])
        ):
            raise ValueError('D/E arm evaluation config mismatch for ' + arm_id)
    return {
        'status': 'validated_no_training_no_inference',
        'summary_version': SUMMARY_VERSION,
        'protocol_sha256': config['protocol']['sha256'],
        'arm_count': len(ARM_ORDER),
        'pair_count_per_arm': len(context['pairs']),
        'training_executed': False,
    }


def _response_summary(
    rows: list[dict[str, str]],
    *,
    arm_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scenario_id in SCENARIOS:
        selected = [row for row in rows if row['scenario_id'] == scenario_id]
        if len(selected) != 72:
            raise ValueError('Expected 72 response rows for ' + arm_id + ':' + scenario_id)
        gains = [float(row['reoptimization_gain']) for row in selected]
        stable = int(config['bootstrap']['seed']) + int(
            hashlib.sha256((arm_id + ':' + scenario_id).encode('ascii')).hexdigest()[:8],
            16,
        )
        result[scenario_id] = {
            'count': len(selected),
            'source_family_count': len({row['source_family_id'] for row in selected}),
            'positive_gain_rate': float(np.mean(np.asarray(gains) > 1.0e-9)),
            'reoptimization_gain': _stats(gains),
            'cluster_bootstrap_mean_ci': _cluster_ci(
                selected,
                seed=stable,
                repetitions=int(config['bootstrap']['repetitions']),
            ),
            'assignment_change_rate': _stats(
                [float(row['assignment_change_rate']) for row in selected]
            ),
            'relative_gap_to_exact': _stats(
                [float(row['relative_gap_to_exact']) for row in selected]
            ),
        }
    return result


def _mechanism_rows(
    *,
    arm_id: str,
    arm: dict[str, Any],
    config: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    objective = context['objective']
    project = load_config()
    device = resolve_device(config['inference']['device'])
    first_row = next(iter(context['core'].values()))
    first_instance = load_instance_json(Path(first_row['output_json']))
    metadata = json.loads(
        (Path(arm['run_dir']) / 'run_metadata.json').read_text(encoding='utf-8')
    )
    model, checkpoint, checkpoint_hash = _load_model(
        checkpoint_path=Path(arm['checkpoint']),
        metadata=metadata,
        first_instance=first_instance,
        objective=objective,
        project_config=project,
        device=device,
    )
    if (
        checkpoint_hash != arm['checkpoint_sha256']
        or int(checkpoint['update']) != int(arm['expected_update'])
        or checkpoint['experiment_spec']['arm_id'] != arm_id
    ):
        raise ValueError('D/E mechanism checkpoint mismatch for arm ' + arm_id)
    baseline_id = config['mechanism']['baseline_scenario_id']
    teacher_id = config['mechanism']['teacher_scenario_id']
    tolerance = float(config['mechanism']['positive_gain_tolerance'])
    pairs = {
        (row['original_id'], row['scenario_id']): row
        for row in context['pairs']
    }
    pair_dir = Path(arm['evaluation_dir']) / 'pairs'
    response = {
        row['original_id']: row
        for row in _read_csv(Path(arm['evaluation_dir']) / 'response_evidence.csv')
        if row['scenario_id'] == teacher_id
    }
    rows: list[dict[str, Any]] = []
    for original_id, core_row in context['core'].items():
        base = load_instance_json(Path(core_row['output_json']))
        balanced = apply_scenario(
            base,
            context['scenarios'][baseline_id],
            project,
            objective=objective,
        )
        teacher = apply_scenario(
            base,
            context['scenarios'][teacher_id],
            project,
            objective=objective,
        )
        baseline_pair = pairs[(original_id, baseline_id)]
        teacher_pair = pairs[(original_id, teacher_id)]
        baseline_payload = json.loads(
            (pair_dir / (baseline_pair['pair_id'] + '.json')).read_text(
                encoding='utf-8'
            )
        )
        teacher_payload = json.loads(
            (pair_dir / (teacher_pair['pair_id'] + '.json')).read_text(
                encoding='utf-8'
            )
        )
        assignment = np.asarray(baseline_payload['assignment'], dtype=np.int64)
        balanced_observation = MethodAssignmentEnv(
            max_steps=1,
            patience=1,
            config=project,
            objective=objective,
        ).reset(balanced, initial_assignment=assignment)
        teacher_observation = MethodAssignmentEnv(
            max_steps=1,
            patience=1,
            config=project,
            objective=objective,
        ).reset(teacher, initial_assignment=assignment)
        screen = _screen_metrics(
            model=model,
            balanced_observation=balanced_observation,
            teacher_observation=teacher_observation,
            teacher_instance=teacher,
            objective=objective,
            tolerance=tolerance,
            top_k=[int(value) for value in config['mechanism']['top_k']],
        )
        gain = float(response[original_id]['reoptimization_gain'])
        diverged = not np.array_equal(
            assignment,
            np.asarray(teacher_payload['assignment'], dtype=np.int64),
        )
        rows.append(
            {
                'arm_id': arm_id,
                'original_id': original_id,
                'source_family_id': core_row['source_family_id'],
                'topology': core_row['topology'],
                'size_group': core_row['size_group'],
                'n': teacher.n,
                'reoptimization_gain': gain,
                'trajectory_diverged': diverged,
                'profitable_trajectory_divergence': diverged and gain > tolerance,
                **screen,
            }
        )
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return rows


def _hash_manifest(output_dir: Path) -> None:
    excluded = {
        'evaluation_output_hashes.csv',
        'evaluation_sequence_state.json',
    }
    rows = []
    for path in sorted(output_dir.rglob('*')):
        if path.is_file() and path.name not in excluded and '_sequence_logs' not in path.parts:
            rows.append(
                {
                    'relative_path': path.relative_to(output_dir).as_posix(),
                    'sha256': _sha256(path),
                    'size_bytes': path.stat().st_size,
                }
            )
    _write_csv(output_dir / 'evaluation_output_hashes.csv', rows)


def summarize(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _load_config(path)
    context = load_preference_context(
        Path(config['preference_response_config']).resolve(),
        output_override=None,
    )
    objective = context['objective']
    if objective.config_hash != config['objective']['config_hash']:
        raise ValueError('Frozen objective hash mismatch in D/E summary.')
    gates = config['_protocol']['acceptance_gates']
    output_dir = Path(config['run']['output_dir']).resolve()
    arm_summaries: dict[str, Any] = {}
    gate_rows: dict[str, Any] = {}
    mechanism_rows: list[dict[str, Any]] = []
    eligible: list[str] = []
    for arm_id in ARM_ORDER:
        arm = config['arms'][arm_id]
        arm_dir = Path(arm['evaluation_dir'])
        candidate = json.loads(
            (arm_dir / 'candidate_response_summary.json').read_text(encoding='utf-8')
        )
        experiment = candidate.get('experiment_spec', {})
        if (
            candidate.get('status') != 'completed_not_frozen'
            or candidate.get('checkpoint_sha256') != arm['checkpoint_sha256']
            or int(candidate.get('checkpoint_update', -1)) != int(arm['expected_update'])
            or int(candidate.get('pair_count', 0)) != 288
            or experiment.get('arm_id') != arm_id
            or experiment.get('protocol_config_sha256') != config['protocol']['sha256']
        ):
            raise ValueError('D/E candidate summary mismatch for arm ' + arm_id)
        response_rows = _read_csv(arm_dir / 'response_evidence.csv')
        if len(response_rows) != 216:
            raise ValueError('D/E response evidence must contain 216 rows.')
        response = _response_summary(response_rows, arm_id=arm_id, config=config)
        fixed = _fixed_72(Path(arm['run_dir']), int(arm['expected_update']))
        arm_mechanism_rows = _mechanism_rows(
            arm_id=arm_id,
            arm=arm,
            config=config,
            context=context,
        )
        mechanism_rows.extend(arm_mechanism_rows)
        mechanism = {
            'instance_count': len(arm_mechanism_rows),
            'teacher_endpoint_policy_positive_action_rate': float(
                np.mean(
                    [row['policy_top_action_positive'] for row in arm_mechanism_rows]
                )
            ),
            'teacher_best_action_policy_rank': _stats(
                [float(row['true_best_action_policy_rank']) for row in arm_mechanism_rows]
            ),
            'profitable_trajectory_divergence_count': int(
                sum(row['profitable_trajectory_divergence'] for row in arm_mechanism_rows)
            ),
            'profitable_trajectory_divergence_rate': float(
                np.mean(
                    [
                        row['profitable_trajectory_divergence']
                        for row in arm_mechanism_rows
                    ]
                )
            ),
        }
        teacher = response['controlled_teacher_only']
        student = response['controlled_student_only']
        conflict = response['controlled_conflict']
        arm_summary = {
            'checkpoint_update': int(arm['expected_update']),
            'checkpoint_sha256': arm['checkpoint_sha256'],
            'fixed_72': fixed,
            'controlled_cross': {
                'mean_relative_gap': float(candidate['mean_relative_gap']),
                'p90_relative_gap': float(candidate['p90_relative_gap']),
            },
            'all_hard_feasible': bool(candidate['all_hard_feasible']),
            'max_score_recomputation_error': float(
                candidate['max_score_recomputation_error']
            ),
            'response': response,
            'mechanism': mechanism,
        }
        checks = {
            'engineering_feasible': arm_summary['all_hard_feasible'],
            'engineering_recomputation': arm_summary['max_score_recomputation_error']
            <= float(gates['engineering']['max_score_recomputation_error']),
            'fixed_72_mean_gap': fixed['mean_relative_gap']
            <= float(gates['fixed_72_quality']['max_mean_exact_relative_gap']),
            'fixed_72_p90_gap': fixed['p90_relative_gap']
            <= float(gates['fixed_72_quality']['max_p90_exact_relative_gap']),
            'teacher_endpoint_positive_rate': mechanism[
                'teacher_endpoint_policy_positive_action_rate'
            ]
            >= float(
                gates['mechanism']['min_teacher_endpoint_policy_positive_action_rate']
            ),
            'teacher_best_action_median_rank': mechanism[
                'teacher_best_action_policy_rank'
            ]['median']
            <= float(gates['mechanism']['max_teacher_best_action_median_policy_rank']),
            'profitable_trajectory_divergence': mechanism[
                'profitable_trajectory_divergence_count'
            ]
            > 0,
            'teacher_positive_rate': teacher['positive_gain_rate']
            >= float(gates['pure_ppo_response']['min_teacher_only_positive_gain_rate']),
            'teacher_mean_gain': teacher['reoptimization_gain']['mean']
            >= float(
                gates['pure_ppo_response']['min_teacher_only_mean_reoptimization_gain']
            ),
            'teacher_bootstrap_lower_positive': teacher[
                'cluster_bootstrap_mean_ci'
            ][0]
            > 0.0,
            'student_positive_rate': student['positive_gain_rate']
            >= float(gates['pure_ppo_response']['min_student_only_positive_gain_rate']),
            'conflict_positive_rate': conflict['positive_gain_rate']
            >= float(gates['pure_ppo_response']['min_conflict_positive_gain_rate']),
        }
        passed = all(checks.values())
        if passed:
            eligible.append(arm_id)
        arm_summaries[arm_id] = arm_summary
        gate_rows[arm_id] = {'passed': passed, 'checks': checks}
    complexity = {'D': 0, 'E': 1}
    ranking = sorted(
        eligible,
        key=lambda arm_id: (
            -arm_summaries[arm_id]['response']['controlled_teacher_only'][
                'reoptimization_gain'
            ]['mean'],
            -arm_summaries[arm_id]['fixed_72']['mean_total_score'],
            arm_summaries[arm_id]['fixed_72']['p90_relative_gap'],
            complexity[arm_id],
        ),
    )
    summary = {
        'schema_version': 1,
        'summary_version': SUMMARY_VERSION,
        'status': 'completed_not_frozen',
        'config_sha256': _sha256(path),
        'protocol_sha256': config['protocol']['sha256'],
        'arms': arm_summaries,
        'gate_decision': {
            'arms': gate_rows,
            'eligible_arms': eligible,
            'ranking_among_eligible': ranking,
            'recommended_arm': ranking[0] if ranking else None,
        },
        'interpretation_limits': [
            'controlled calibration data already used during development',
            'single seed0 checkpoint per arm',
            'pure PPO only; no local-search or hybrid post-processing',
            'mechanism metrics use the balanced PPO endpoint under teacher-only scoring',
            'recommendation does not freeze PPO or inference parameters',
        ],
    }
    _write_csv(output_dir / 'mechanism_endpoint_rows.csv', mechanism_rows)
    _write_json(output_dir / 'ranking_evaluation_summary.json', summary)
    lines = [
        '# D/E pure-PPO controlled-cross decision',
        '',
        '| Arm | Fixed72 mean gap | P90 gap | Teacher +rate | Teacher mean | CI low | Endpoint +rate | Best rank median | Student +rate | Conflict +rate | Gate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for arm_id in ARM_ORDER:
        item = arm_summaries[arm_id]
        teacher = item['response']['controlled_teacher_only']
        mechanism = item['mechanism']
        lines.append(
            '| {} | {:.3f}% | {:.3f}% | {:.1f}% | {:.8f} | {:.8f} | '
            '{:.1f}% | {:.1f} | {:.1f}% | {:.1f}% | {} |'.format(
                arm_id,
                100 * item['fixed_72']['mean_relative_gap'],
                100 * item['fixed_72']['p90_relative_gap'],
                100 * teacher['positive_gain_rate'],
                teacher['reoptimization_gain']['mean'],
                teacher['cluster_bootstrap_mean_ci'][0],
                100 * mechanism['teacher_endpoint_policy_positive_action_rate'],
                mechanism['teacher_best_action_policy_rank']['median'],
                100
                * item['response']['controlled_student_only']['positive_gain_rate'],
                100 * item['response']['controlled_conflict']['positive_gain_rate'],
                'PASS' if gate_rows[arm_id]['passed'] else 'FAIL',
            )
        )
    lines.extend(
        [
            '',
            'Eligible arms: ' + str(eligible),
            'Recommended arm under confirmed rule: '
            + str(ranking[0] if ranking else None),
            '',
            'This is exploratory controlled-calibration evidence and does not freeze parameters.',
        ]
    )
    (output_dir / 'ranking_evaluation_report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    _hash_manifest(output_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default=(
            'pa_moap_rl/configs/'
            'preference_ranking_ablation_evaluation_v3.yaml'
        ),
    )
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    result = validate(args.config) if args.validate_only else summarize(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
