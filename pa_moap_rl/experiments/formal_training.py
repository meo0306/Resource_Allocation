'''Formal shared-policy PPO training with dynamic preference scenarios.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pa_moap_rl.checkpointing import (
    checkpoint_metadata,
    method_matrix_hash,
    validate_checkpoint_metadata,
)
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import (
    PreferenceScenario,
    apply_scenario,
    load_scenario_bank,
    load_scenario_config,
    sample_scenario,
)
from pa_moap_rl.experiments.train_ppo_batch import (
    _baseline_results,
    _build_env,
    _create_summary_writer,
    _policy_rollout,
    _write_rows,
)
from pa_moap_rl.objective import ObjectiveSpec, legacy_objective_spec, load_objective_spec
from pa_moap_rl.solvers.ppo_solver import (
    build_actor_critic,
    collect_rollout,
    collect_vectorized_rollout,
    ppo_config_from_project,
    resolve_device,
    update_ppo,
)
from pa_moap_rl.utils.scoring import score_assignment


@dataclass(frozen=True)
class ValidationPair:
    '''One explicitly joined instance/scenario/exact-reference validation item.'''

    pair_id: str
    instance_path: Path
    scenario: PreferenceScenario
    exact_score: float
    exact_result_path: Path
    exact_result_sha256: str
    original_id: str
    source_family_id: str
    topology: str
    size_group: str


def load_split_paths(path: str | Path, limit: int | None = None) -> list[Path]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (
                row.get('size_group') or row.get('group', ''),
                row.get('topology', ''),
            )
            buckets.setdefault(key, []).append(row)
        selected: list[dict[str, str]] = []
        offset = 0
        while len(selected) < limit:
            added = False
            for key in sorted(buckets):
                if offset < len(buckets[key]):
                    selected.append(buckets[key][offset])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            offset += 1
        rows = selected
    paths = [Path(row['output_json']) for row in rows]
    missing = [item for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} split entries do not exist; first: {missing[0]}')
    return paths


def load_training_records(path: str | Path) -> list[dict[str, str]]:
    '''Load manifest metadata needed for size-aware stratified batches.'''

    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError('Training manifest must contain at least one instance.')
    missing = [
        Path(record['output_json'])
        for record in records
        if not Path(record['output_json']).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} training entries do not exist; first: {missing[0]}'
        )
    return records


def build_training_buckets(
    records: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    '''Use dataset groups as size-homogeneous buckets.'''

    buckets: dict[str, list[dict[str, str]]] = {}
    for record in records:
        bucket = record.get('size_group') or record.get('group') or 'ungrouped'
        buckets.setdefault(bucket, []).append(record)
    return buckets


def _source_family_id(record: dict[str, str]) -> str:
    return record.get('source_family_id') or record.get('source_identity') or record['output_json']


def sample_stratified_training_batch(
    buckets: dict[str, list[dict[str, str]]],
    *,
    sample_count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    '''Sample one size bucket, distinct contents, and diverse topologies.'''

    if sample_count < 1:
        raise ValueError('sample_count must be positive.')
    eligible = [
        (name, records)
        for name, records in sorted(buckets.items())
        if len(records) >= sample_count
    ]
    if not eligible:
        raise ValueError(
            f'No size bucket contains at least {sample_count} training records.'
        )
    bucket_names = [name for name, _ in eligible]
    selected_bucket = rng.choice(bucket_names)
    candidates = list(buckets[selected_bucket])
    by_topology: dict[str, list[dict[str, str]]] = {}
    for record in candidates:
        by_topology.setdefault(record.get('topology', 'unknown'), []).append(record)

    topology_order = list(by_topology)
    rng.shuffle(topology_order)
    selected: list[dict[str, str]] = []
    selected_paths: set[str] = set()
    selected_sources: set[str] = set()
    for topology in topology_order:
        options = [
            record
            for record in by_topology[topology]
            if _source_family_id(record) not in selected_sources
        ]
        if not options:
            continue
        record = rng.choice(options)
        selected.append(record)
        selected_paths.add(record['output_json'])
        selected_sources.add(_source_family_id(record))
        if len(selected) == sample_count:
            return selected

    rng.shuffle(candidates)
    for record in candidates:
        source = _source_family_id(record)
        if record['output_json'] in selected_paths or source in selected_sources:
            continue
        selected.append(record)
        selected_paths.add(record['output_json'])
        selected_sources.add(source)
        if len(selected) == sample_count:
            return selected
    for record in candidates:
        if record['output_json'] in selected_paths:
            continue
        selected.append(record)
        if len(selected) == sample_count:
            return selected
    raise RuntimeError('Could not construct the requested training batch.')


def evaluate_model(
    model: torch.nn.Module,
    paths: list[Path],
    scenarios: list[PreferenceScenario],
    *,
    project_config: Any,
    ppo_config: Any,
    seed: int,
    include_baselines: bool = False,
    include_local_search: bool = False,
    objective: ObjectiveSpec | None = None,
) -> list[dict[str, Any]]:
    '''Evaluate a fixed scenario bank deterministically for fair comparisons.'''

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        base = load_instance_json(path)
        scenario = scenarios[index % len(scenarios)]
        instance = apply_scenario(base, scenario, project_config, objective=objective)
        results = [
            _policy_rollout(
                instance,
                model=model,
                project_config=project_config,
                ppo_config=ppo_config,
                deterministic=True,
                objective=objective,
            )
        ]
        if include_baselines:
            results.extend(
                _baseline_results(
                    instance,
                    seed=seed + index,
                    include_local_search=include_local_search,
                    objective=objective,
                )
            )
        metadata = instance.metadata or {}
        for result in results:
            row = {
                'instance_path': str(path),
                'instance_name': instance.instance_name,
                'topology': metadata.get('topology', path.parents[1].name),
                'group': metadata.get('group', path.parent.name),
                'n': instance.n,
                'scenario_id': scenario.scenario_id,
                'student_template': scenario.student_template,
                'teacher_template': scenario.teacher_template,
                'held_out': scenario.held_out,
                'scenario_seed': scenario.seed,
                'student_profile_json': json.dumps(scenario.student_profile.tolist()),
                'teacher_profile_json': json.dumps(scenario.teacher_profile.tolist()),
                'target_distribution_json': json.dumps(
                    (
                        list(objective.target_distribution)
                        if objective is not None and objective.target_distribution is not None
                        else (
                            scenario.target_distribution.tolist()
                            if scenario.target_distribution is not None
                            else None
                        )
                    )
                ),
                'weights_json': json.dumps(
                    objective.to_dict() if objective is not None else scenario.weights,
                    sort_keys=True,
                ),
                'seed': seed,
                'solver_name': result.solver_name,
            }
            row.update({key: value for key, value in result.metrics.items() if key != 'solver_name'})
            rows.append(row)
    return rows

def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    hidden_dim: int | None,
    ppo_config: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    python_rng: random.Random,
    numpy_rng: np.random.Generator,
    best_validation_score: float,
    best_validation_p90_gap: float = float('inf'),
    provenance: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 3 if getattr(model, 'experiment_spec', None) is not None else 2,
        'update': update,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hidden_dim': hidden_dim,
        'ppo_config': dict(ppo_config.__dict__),
        'train_rows': train_rows,
        'eval_rows': eval_rows,
        'python_rng_state': python_rng.getstate(),
        'numpy_rng_state': numpy_rng.bit_generator.state,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'best_validation_score': best_validation_score,
        'best_validation_p90_gap': best_validation_p90_gap,
        'provenance': provenance,
        'experiment_spec': getattr(model, 'experiment_spec', None),
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    python_rng: random.Random,
    numpy_rng: np.random.Generator,
    device: torch.device,
    objective: ObjectiveSpec | None = None,
    method_hash: str | None = None,
    data_hash: str | None = None,
    legacy_mode: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if objective is None or method_hash is None or data_hash is None:
        raise ValueError(
            'Checkpoint restore requires an explicit ObjectiveSpec, method hash, and data hash.'
        )
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_hash,
        data_hash=data_hash,
        legacy_mode=legacy_mode,
    )
    expected_experiment_spec = getattr(model, 'experiment_spec', None)
    actual_experiment_spec = checkpoint.get('experiment_spec')
    if expected_experiment_spec is not None and int(checkpoint.get('format_version', 0)) < 3:
        raise ValueError(
            'Versioned ablation restore requires checkpoint format_version >= 3.'
        )
    if actual_experiment_spec != expected_experiment_spec:
        raise ValueError(
            'Checkpoint experiment_spec mismatch: the model/sampler version '
            'must match exactly and cannot silently fall back.'
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    python_rng.setstate(checkpoint['python_rng_state'])
    numpy_rng.bit_generator.state = checkpoint['numpy_rng_state']
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    if device.type == 'cuda' and checkpoint.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng_state']])
    return checkpoint


def _mean_policy_score(rows: list[dict[str, Any]]) -> float:
    values = [
        float(row['total_score'])
        for row in rows
        if row.get('solver_name') == 'ppo_shared_policy'
    ]
    return float(np.mean(values)) if values else -float('inf')


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_validation_pairs(
    path: str | Path,
    *,
    scenario_bank: str | Path,
    objective: ObjectiveSpec,
    allow_controlled_held_out: bool = False,
) -> list[ValidationPair]:
    '''Load and strictly validate an explicit instance/scenario/exact-result join.'''

    scenarios = {
        scenario.scenario_id: scenario
        for scenario in load_scenario_bank(scenario_bank)
    }
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Validation pair manifest must not be empty.')
    required = {
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
    }
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(
            'Validation pair manifest is missing columns: '
            + ', '.join(sorted(missing_columns))
        )
    pair_ids: set[str] = set()
    pairs: list[ValidationPair] = []
    for row in rows:
        pair_id = row['pair_id']
        if pair_id in pair_ids:
            raise ValueError(f'Duplicate validation pair_id: {pair_id}.')
        pair_ids.add(pair_id)
        scenario_id = row['scenario_id']
        if scenario_id not in scenarios:
            raise ValueError(f'Unknown scenario_id {scenario_id!r} for {pair_id}.')
        scenario = scenarios[scenario_id]
        if scenario.schema_version != 2:
            raise ValueError(f'{pair_id} must use Scenario v2.')
        controlled_held_out = (
            scenario.held_out
            and scenario.access_status == 'controlled_calibration'
            and allow_controlled_held_out
        )
        if scenario.held_out and not controlled_held_out:
            raise ValueError(
                f'{pair_id} uses a held-out scenario without the explicit '
                'controlled-calibration validation opt-in.'
            )
        instance_path = Path(row['output_json'])
        exact_path = Path(row['exact_result_path'])
        for label, item in (('instance', instance_path), ('exact result', exact_path)):
            if not item.is_file():
                raise FileNotFoundError(f'{pair_id} {label} does not exist: {item}')
        instance_hash = _sha256(instance_path)
        if instance_hash != row['output_json_sha256']:
            raise ValueError(f'{pair_id} instance hash mismatch.')
        exact_hash = _sha256(exact_path)
        if exact_hash != row['exact_result_sha256']:
            raise ValueError(f'{pair_id} exact-result hash mismatch.')
        exact = json.loads(exact_path.read_text(encoding='utf-8'))
        if exact.get('objective_hash') != objective.config_hash:
            raise ValueError(f'{pair_id} exact objective hash mismatch.')
        if not bool(exact.get('proven_optimal')):
            raise ValueError(f'{pair_id} exact result is not proven optimal.')
        if str(exact.get('pair_id')) != pair_id:
            raise ValueError(f'{pair_id} exact result pair_id mismatch.')
        exact_score = float(exact['score']['total_score'])
        base = load_instance_json(instance_path)
        conditioned = apply_scenario(base, scenario, objective=objective)
        recomputed = score_assignment(
            exact['assignment'],
            instance=conditioned,
            objective=objective,
        ).total_score
        if abs(recomputed - exact_score) > 1.0e-9:
            raise ValueError(f'{pair_id} exact score recomputation exceeds 1e-9.')
        pairs.append(
            ValidationPair(
                pair_id=pair_id,
                instance_path=instance_path,
                scenario=scenario,
                exact_score=exact_score,
                exact_result_path=exact_path,
                exact_result_sha256=exact_hash,
                original_id=row['original_id'],
                source_family_id=row['source_family_id'],
                topology=row['topology'],
                size_group=row['size_group'],
            )
        )
    return pairs


def evaluate_validation_pairs(
    model: torch.nn.Module,
    pairs: list[ValidationPair],
    *,
    project_config: Any,
    ppo_config: Any,
    seed: int,
    objective: ObjectiveSpec,
) -> list[dict[str, Any]]:
    '''Deterministically evaluate explicitly joined validation pairs.'''

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        base = load_instance_json(pair.instance_path)
        instance = apply_scenario(
            base,
            pair.scenario,
            project_config,
            objective=objective,
        )
        result = _policy_rollout(
            instance,
            model=model,
            project_config=project_config,
            ppo_config=ppo_config,
            deterministic=True,
            objective=objective,
        )
        score = float(result.score.total_score)
        absolute_gap = pair.exact_score - score
        if absolute_gap < -1.0e-9:
            raise RuntimeError(
                f'{pair.pair_id} policy score exceeds proven optimum by {-absolute_gap}.'
            )
        relative_gap = max(0.0, absolute_gap) / max(
            abs(pair.exact_score),
            float(objective.epsilon),
        )
        row = {
            'pair_id': pair.pair_id,
            'original_id': pair.original_id,
            'source_family_id': pair.source_family_id,
            'instance_path': str(pair.instance_path),
            'instance_name': instance.instance_name,
            'topology': pair.topology,
            'group': pair.size_group,
            'n': instance.n,
            'scenario_id': pair.scenario.scenario_id,
            'student_template': pair.scenario.student_template,
            'teacher_template': pair.scenario.teacher_template,
            'scenario_seed': pair.scenario.seed,
            'student_profile_json': json.dumps(pair.scenario.student_profile.tolist()),
            'teacher_profile_json': json.dumps(pair.scenario.teacher_profile.tolist()),
            'target_distribution_json': json.dumps(
                list(objective.target_distribution)
                if objective.target_distribution is not None
                else None
            ),
            'weights_json': json.dumps(objective.to_dict(), sort_keys=True),
            'objective_id': objective.objective_id,
            'objective_hash': objective.config_hash,
            'exact_result_path': str(pair.exact_result_path),
            'exact_result_sha256': pair.exact_result_sha256,
            'exact_total_score': pair.exact_score,
            'absolute_gap_to_exact': max(0.0, absolute_gap),
            'relative_gap_to_exact': relative_gap,
            'method_distribution_json': json.dumps(
                result.score.method_distribution.tolist()
            ),
            'max_method_share': float(np.max(result.score.method_distribution)),
            'seed': seed,
            'solver_name': result.solver_name,
        }
        row.update(
            {
                key: value
                for key, value in result.metrics.items()
                if key != 'solver_name'
            }
        )
        rows.append(row)
    return rows




def validation_policy_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    policy_rows = [
        row for row in rows
        if row.get('solver_name') == 'ppo_shared_policy'
    ]
    if not policy_rows:
        return {
            'mean_total_score': -float('inf'),
            'p90_relative_gap': float('inf'),
        }
    result = {
        'mean_total_score': float(
            np.mean([float(row['total_score']) for row in policy_rows])
        ),
        'p90_relative_gap': float('inf'),
    }
    gaps = [
        float(row['relative_gap_to_exact'])
        for row in policy_rows
        if row.get('relative_gap_to_exact') not in (None, '')
    ]
    if gaps:
        result['p90_relative_gap'] = float(np.quantile(gaps, 0.90))
    return result


def _is_better_validation(
    candidate: dict[str, float],
    incumbent_score: float,
    incumbent_p90_gap: float,
) -> bool:
    score = candidate['mean_total_score']
    p90_gap = candidate['p90_relative_gap']
    if score > incumbent_score + 1.0e-12:
        return True
    return (
        abs(score - incumbent_score) <= 1.0e-12
        and p90_gap < incumbent_p90_gap - 1.0e-12
    )

def _write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    temporary.replace(output)


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if np.isfinite(number):
                values.append(number)
    return values


def _mean_field(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = _numeric_values(rows, key)
    return float(np.mean(values)) if values else default


def aggregate_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    '''Build one stable, plotting-ready table for overall and topology curves.'''

    metric_names = (
        'total_score',
        'effect_score',
        'student_pref_score',
        'teacher_pref_score',
        'global_score',
        'soft_penalty',
        'entropy',
        'hard_feasible_rate',
        'valid_action_ratio',
        'runtime',
        'absolute_gap_to_exact',
        'relative_gap_to_exact',
    )
    policy_rows = [row for row in rows if row.get('solver_name') == 'ppo_shared_policy']
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in policy_rows:
        update = int(row['update'])
        grouped.setdefault((update, 'overall', 'all'), []).append(row)
        topology = str(row.get('topology', 'unknown'))
        grouped.setdefault((update, 'topology', topology), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (update, scope, scope_value), group_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            'update': update,
            'scope': scope,
            'scope_value': scope_value,
            'count': len(group_rows),
        }
        for metric in metric_names:
            values = _numeric_values(group_rows, metric)
            if not values:
                continue
            aggregate[f'{metric}_mean'] = float(np.mean(values))
            aggregate[f'{metric}_std'] = float(np.std(values))
            aggregate[f'{metric}_min'] = float(np.min(values))
            aggregate[f'{metric}_max'] = float(np.max(values))
            aggregate[f'{metric}_median'] = float(np.median(values))
            aggregate[f'{metric}_p90'] = float(np.quantile(values, 0.90))
        aggregate['mask_violation_count_sum'] = int(
            sum(float(row.get('mask_violation_count', 0)) for row in group_rows)
        )
        aggregates.append(aggregate)
    return aggregates


def _write_research_tables(
    *,
    output: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    metrics_dir = output / 'metrics'
    paths = {
        'train_instances': metrics_dir / 'train_instances.csv',
        'train_updates': metrics_dir / 'train_updates.csv',
        'validation_instances': metrics_dir / 'validation_instances.csv',
        'validation_updates': metrics_dir / 'validation_updates.csv',
    }
    _write_rows([row for row in train_rows if row.get('phase') == 'train'], paths['train_instances'])
    _write_rows(
        [row for row in train_rows if row.get('phase') == 'train_summary'],
        paths['train_updates'],
    )
    _write_rows(eval_rows, paths['validation_instances'])
    _write_rows(aggregate_validation_rows(eval_rows), paths['validation_updates'])
    return paths


def _write_train_tensorboard(writer: Any | None, row: dict[str, Any]) -> None:
    if writer is None:
        return
    update = int(row['update'])
    groups = {
        'train': (
            'best_score',
            'mean_reward',
            'episode_return',
            'entropy',
            'normalized_entropy',
            'policy_loss',
            'value_loss',
            'loss',
            'approx_kl',
            'clip_fraction',
            'grad_norm',
            'explained_variance',
            'illegal_action_rate',
        ),
        'system': (
            'transitions_per_sec',
            'update_elapsed_sec',
            'rollout_elapsed_sec',
            'ppo_elapsed_sec',
            'optimizer_minibatches',
            'ppo_epochs_executed',
            'target_kl',
            'target_kl_observed',
            'target_kl_triggered',
            'unique_topologies',
            'unique_sources',
            'batch_max_n',
            'gpu_peak_allocated_mb',
            'gpu_peak_reserved_mb',
            'learning_rate',
            'entropy_coef',
        ),
        'auxiliary': tuple(
            sorted(name for name in row if name.startswith('auxiliary_'))
        ),
    }
    for group, names in groups.items():
        for name in names:
            values = _numeric_values([row], name)
            if values:
                writer.add_scalar(f'{group}/{name}', values[0], update)
    writer.flush()


def _write_validation_tensorboard(
    writer: Any | None,
    rows: list[dict[str, Any]],
    *,
    update: int,
    best_validation_score: float,
) -> None:
    if writer is None:
        return
    aggregates = aggregate_validation_rows(rows)
    overall = next(row for row in aggregates if row['scope'] == 'overall')
    for name in (
        'total_score_mean',
        'total_score_std',
        'effect_score_mean',
        'student_pref_score_mean',
        'teacher_pref_score_mean',
        'global_score_mean',
        'soft_penalty_mean',
        'hard_feasible_rate_mean',
    ):
        values = _numeric_values([overall], name)
        if values:
            writer.add_scalar(f'validation/{name}', values[0], update)
    writer.add_scalar('validation/best_total_score_mean', best_validation_score, update)
    writer.add_scalar(
        'validation/gap_to_best',
        float(overall['total_score_mean']) - best_validation_score,
        update,
    )
    for row in aggregates:
        if row['scope'] == 'topology':
            scope_value = row['scope_value']
            writer.add_scalar(
                f'validation_by_topology/{scope_value}',
                float(row['total_score_mean']),
                update,
            )
    writer.flush()


def _run_metadata(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    validation_scenarios: str | Path,
    output: Path,
    seed: int,
    device: torch.device,
    hidden_dim: int | None,
    ppo_config: Any,
    evaluation_ppo_config: Any,
    instances_per_update: int,
    eval_interval: int,
    eval_limit: int | None,
    checkpoint_interval: int,
    resume_from: str | Path | None,
    tensorboard_dir: Path | None,
    rollout_mode: str,
) -> dict[str, Any]:
    inputs = {}
    for name, path in (
        ('train_manifest', train_manifest),
        ('validation_manifest', validation_manifest),
        ('validation_scenarios', validation_scenarios),
    ):
        resolved = Path(path).resolve()
        inputs[name] = {'path': str(resolved), 'sha256': _sha256(resolved)}
    return {
        'schema_version': 1,
        'created_at_utc': _utc_now(),
        'output_dir': str(output.resolve()),
        'resume_from': str(Path(resume_from).resolve()) if resume_from is not None else None,
        'seed': seed,
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': (
            torch.cuda.get_device_name(device) if device.type == 'cuda' else None
        ),
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'hidden_dim': hidden_dim,
        'ppo_config': dict(ppo_config.__dict__),
        'evaluation_ppo_config': dict(evaluation_ppo_config.__dict__),
        'training_and_evaluation_environment_separated': (
            ppo_config.env_max_steps != evaluation_ppo_config.env_max_steps
            or ppo_config.env_patience != evaluation_ppo_config.env_patience
        ),
        'instances_per_update': instances_per_update,
        'rollout_mode': rollout_mode,
        'advantage_normalization': 'per_instance_then_pooled',
        'training_batch_sampler': 'size_group_content_unique_topology_diverse',
        'eval_interval': eval_interval,
        'eval_limit': eval_limit,
        'checkpoint_interval': checkpoint_interval,
        'tensorboard_dir': str(tensorboard_dir.resolve()) if tensorboard_dir is not None else None,
        'inputs': inputs,
    }


def run_formal_training(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    validation_scenarios: str | Path,
    validation_pairs_manifest: str | Path | None = None,
    allow_controlled_held_out_validation: bool = False,
    data_audit: str | Path | None = None,
    pilot_config: str | Path | None = None,
    output_dir: str | Path,
    seed: int = 0,
    device: str = 'auto',
    total_updates: int | None = None,
    instances_per_update: int = 2,
    rollout_steps: int | None = None,
    minibatch_size: int | None = None,
    update_epochs: int | None = None,
    learning_rate: float | None = None,
    entropy_coef: float | None = None,
    target_kl: float | None = None,
    eval_interval: int = 10,
    eval_limit: int | None = 30,
    checkpoint_interval: int = 10,
    hidden_dim: int | None = None,
    env_max_steps: int | None = None,
    env_patience: int | None = None,
    eval_env_max_steps: int | None = None,
    eval_env_patience: int | None = None,
    resume_from: str | Path | None = None,
    tensorboard_dir: str | Path | None = None,
    enable_tensorboard: bool = True,
    rollout_mode: str = 'joint',
    objective: ObjectiveSpec | None = None,
    evaluate_at_start: bool = False,
    deterministic: bool = False,
    model_builder: Any | None = None,
    training_batch_builder: Any | None = None,
    auxiliary_loss_builder: Any | None = None,
) -> dict[str, Any]:
    '''Train one shared policy over content, topology, and preference variation.'''

    if min(instances_per_update, eval_interval, checkpoint_interval) < 1:
        raise ValueError('Sampling and interval values must be positive.')
    if rollout_mode not in {'joint', 'sequential'}:
        raise ValueError('rollout_mode must be joint or sequential.')
    if auxiliary_loss_builder is not None and rollout_mode != 'joint':
        raise ValueError('Versioned auxiliary supervision requires joint rollout mode.')
    project_config = load_config()
    scenario_config = load_scenario_config()
    train_records = load_training_records(train_manifest)
    train_buckets = build_training_buckets(train_records)
    train_paths = [Path(record['output_json']) for record in train_records]
    validation_paths = (
        [] if validation_pairs_manifest is not None
        else load_split_paths(validation_manifest, limit=eval_limit)
    )
    scenarios = load_scenario_bank(validation_scenarios)
    first_instance = load_instance_json(train_paths[0])
    soft = project_config.default['soft_constraints']
    effective_objective = objective or legacy_objective_spec(
        first_instance,
        H_min=float(soft['entropy_min']),
        pi_cap=float(soft['method_cap']),
        lambda_div=float(soft['lambda_div']),
        lambda_cap=float(soft['lambda_cap']),
        epsilon=float(soft['epsilon']),
    )
    validation_pairs = (
        load_validation_pairs(
            validation_pairs_manifest,
            scenario_bank=validation_scenarios,
            objective=effective_objective,
            allow_controlled_held_out=allow_controlled_held_out_validation,
        )
        if validation_pairs_manifest is not None
        else None
    )
    hash_paths = [Path(train_manifest), Path(validation_manifest), Path(validation_scenarios)]
    if validation_pairs_manifest is not None:
        hash_paths.append(Path(validation_pairs_manifest))
    if data_audit is not None:
        hash_paths.append(Path(data_audit))
    if pilot_config is not None:
        hash_paths.append(Path(pilot_config))
    run_data_hash = hashlib.sha256(
        ''.join(_sha256(path) for path in hash_paths).encode('ascii')
    ).hexdigest()
    run_method_hash = method_matrix_hash(first_instance)
    provenance = checkpoint_metadata(
        effective_objective,
        method_hash=run_method_hash,
        data_hash=run_data_hash,
    )
    overrides: dict[str, Any] = {'seed': seed, 'device': str(resolve_device(device))}
    if total_updates is not None:
        overrides['total_updates'] = int(total_updates)
    if rollout_steps is not None:
        overrides['rollout_steps'] = int(rollout_steps)
    if minibatch_size is not None:
        overrides['minibatch_size'] = int(minibatch_size)
    if update_epochs is not None:
        overrides['update_epochs'] = int(update_epochs)
    if learning_rate is not None:
        overrides['learning_rate'] = float(learning_rate)
    if entropy_coef is not None:
        overrides['entropy_coef'] = float(entropy_coef)
    if target_kl is not None:
        if target_kl <= 0.0:
            raise ValueError('target_kl must be positive when enabled.')
        overrides['target_kl'] = float(target_kl)
    if env_max_steps is not None:
        overrides['env_max_steps'] = int(env_max_steps)
    if env_patience is not None:
        overrides['env_patience'] = int(env_patience)
    ppo_config = ppo_config_from_project(project_config, **overrides)
    evaluation_ppo_config = replace(
        ppo_config,
        env_max_steps=(
            int(eval_env_max_steps)
            if eval_env_max_steps is not None
            else ppo_config.env_max_steps
        ),
        env_patience=(
            int(eval_env_patience)
            if eval_env_patience is not None
            else ppo_config.env_patience
        ),
    )
    resolved_device = resolve_device(ppo_config.device)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if resolved_device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    model = (
        build_actor_critic(
            first_instance,
            config=project_config,
            device=resolved_device,
            hidden_dim=hidden_dim,
            objective=effective_objective,
        )
        if model_builder is None
        else model_builder(
            first_instance,
            config=project_config,
            device=resolved_device,
            hidden_dim=hidden_dim,
            objective=effective_objective,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo_config.learning_rate))
    output = Path(output_dir)
    checkpoint_dir = output / 'checkpoints'
    train_log = output / 'train_log.csv'
    eval_log = output / 'validation_log.csv'
    research_dir = output / 'metrics'
    checkpoint_index = output / 'checkpoint_index.csv'
    resolved_tensorboard_dir = (
        Path(tensorboard_dir)
        if tensorboard_dir is not None
        else output / 'tensorboard'
    )
    if not enable_tensorboard:
        resolved_tensorboard_dir = None
    writer = _create_summary_writer(resolved_tensorboard_dir)
    checkpoint_rows: list[dict[str, Any]] = []
    if checkpoint_index.is_file():
        with checkpoint_index.open('r', encoding='utf-8', newline='') as handle:
            checkpoint_rows = list(csv.DictReader(handle))
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    start_update = 1
    best_validation_score = -float('inf')
    best_validation_p90_gap = float('inf')
    if resume_from is not None:
        restored = load_training_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            python_rng=python_rng,
            numpy_rng=numpy_rng,
            device=resolved_device,
            objective=effective_objective,
            method_hash=run_method_hash,
            data_hash=run_data_hash,
            legacy_mode=False,
        )
        start_update = int(restored['update']) + 1
        train_rows = list(restored.get('train_rows', []))
        eval_rows = list(restored.get('eval_rows', []))
        best_validation_p90_gap = float(
            restored.get('best_validation_p90_gap', float('inf'))
        )
        best_validation_score = float(restored.get('best_validation_score', -float('inf')))
        if learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = float(ppo_config.learning_rate)

    final_update = int(ppo_config.total_updates)
    if start_update > final_update:
        raise ValueError(f'Checkpoint already reached update {start_update - 1}, target is {final_update}.')

    metadata = _run_metadata(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        validation_scenarios=validation_scenarios,
        output=output,
        seed=seed,
        device=resolved_device,
        hidden_dim=hidden_dim,
        ppo_config=ppo_config,
        evaluation_ppo_config=evaluation_ppo_config,
        instances_per_update=instances_per_update,
        eval_interval=eval_interval,
        eval_limit=eval_limit,
        checkpoint_interval=checkpoint_interval,
        resume_from=resume_from,
        tensorboard_dir=resolved_tensorboard_dir,
        rollout_mode=rollout_mode,
    )
    metadata['start_update'] = start_update
    metadata['target_update'] = final_update
    metadata['objective'] = effective_objective.to_dict()
    metadata['objective_hash'] = effective_objective.config_hash
    metadata['method_matrix_hash'] = run_method_hash
    metadata['data_hash'] = run_data_hash
    metadata['schema_version'] = 2
    metadata['deterministic_algorithms'] = deterministic
    metadata['experiment_spec'] = getattr(model, 'experiment_spec', None)
    metadata['versioned_model_builder'] = model_builder is not None
    metadata['versioned_training_batch_builder'] = training_batch_builder is not None
    metadata['versioned_auxiliary_loss_builder'] = auxiliary_loss_builder is not None
    metadata['validation_pair_count'] = (
        len(validation_pairs) if validation_pairs is not None else len(validation_paths)
    )
    metadata['allow_controlled_held_out_validation'] = (
        allow_controlled_held_out_validation
    )
    if validation_pairs_manifest is not None:
        resolved = Path(validation_pairs_manifest).resolve()
        metadata['inputs']['validation_pairs_manifest'] = {
            'path': str(resolved),
            'sha256': _sha256(resolved),
        }
    if data_audit is not None:
        resolved = Path(data_audit).resolve()
        metadata['inputs']['data_audit'] = {
            'path': str(resolved),
            'sha256': _sha256(resolved),
        }
    if pilot_config is not None:
        resolved = Path(pilot_config).resolve()
        metadata['inputs']['pilot_config'] = {
            'path': str(resolved),
            'sha256': _sha256(resolved),
        }
    _write_json_atomic(metadata, output / 'run_metadata.json')
    _write_rows(train_rows, train_log)
    _write_rows(eval_rows, eval_log)
    _write_research_tables(output=output, train_rows=train_rows, eval_rows=eval_rows)
    validation_score = best_validation_score
    validation_p90_gap = best_validation_p90_gap
    if start_update == 1 and evaluate_at_start:
        initial_rows = (
            evaluate_validation_pairs(
                model,
                validation_pairs,
                project_config=project_config,
                ppo_config=evaluation_ppo_config,
                seed=seed,
                objective=effective_objective,
            )
            if validation_pairs is not None
            else evaluate_model(
                model,
                validation_paths,
                scenarios,
                project_config=project_config,
                ppo_config=evaluation_ppo_config,
                seed=seed,
                objective=effective_objective,
            )
        )
        for row in initial_rows:
            row['update'] = 0
        eval_rows.extend(initial_rows)
        initial_summary = validation_policy_summary(initial_rows)
        validation_score = initial_summary['mean_total_score']
        validation_p90_gap = initial_summary['p90_relative_gap']
        best_validation_score = validation_score
        best_validation_p90_gap = validation_p90_gap
        checkpoint_args = {
            'model': model,
            'optimizer': optimizer,
            'update': 0,
            'hidden_dim': hidden_dim,
            'ppo_config': ppo_config,
            'train_rows': train_rows,
            'eval_rows': eval_rows,
            'python_rng': python_rng,
            'numpy_rng': numpy_rng,
            'best_validation_score': best_validation_score,
            'best_validation_p90_gap': best_validation_p90_gap,
            'provenance': provenance,
        }
        save_checkpoint(checkpoint_dir / 'update_000000.pt', **checkpoint_args)
        save_checkpoint(checkpoint_dir / 'best.pt', **checkpoint_args)
        checkpoint_rows.extend(
            [
                {
                    'update': 0,
                    'kind': 'numbered',
                    'path': str(checkpoint_dir / 'update_000000.pt'),
                    'validation_score': validation_score,
                    'validation_p90_gap': validation_p90_gap,
                    'saved_at_utc': _utc_now(),
                },
                {
                    'update': 0,
                    'kind': 'best',
                    'path': str(checkpoint_dir / 'best.pt'),
                    'validation_score': validation_score,
                    'validation_p90_gap': validation_p90_gap,
                    'saved_at_utc': _utc_now(),
                },
            ]
        )
        _write_rows(checkpoint_rows, checkpoint_index)
        _write_rows(eval_rows, eval_log)
        _write_research_tables(output=output, train_rows=train_rows, eval_rows=eval_rows)


    if resume_from is not None and not (checkpoint_dir / 'best.pt').exists():
        save_checkpoint(
            checkpoint_dir / 'best.pt',
            model=model,
            optimizer=optimizer,
            update=start_update - 1,
            hidden_dim=hidden_dim,
            ppo_config=ppo_config,
            train_rows=train_rows,
            eval_rows=eval_rows,
            python_rng=python_rng,
            numpy_rng=numpy_rng,
            best_validation_score=best_validation_score,
            best_validation_p90_gap=best_validation_p90_gap,
            provenance=provenance,
        )
        checkpoint_rows.append(
            {
                'update': start_update - 1,
                'kind': 'inherited_best',
                'path': str(checkpoint_dir / 'best.pt'),
                'validation_score': best_validation_score,
                'saved_at_utc': _utc_now(),
            }
        )
        _write_rows(checkpoint_rows, checkpoint_index)

    training_start = time.perf_counter()
    for update in range(start_update, final_update + 1):
        update_start = time.perf_counter()
        if resolved_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(resolved_device)
        update_row_start = len(train_rows)
        if training_batch_builder is None:
            requested_count = min(instances_per_update, len(train_paths))
            selected_records = sample_stratified_training_batch(
                train_buckets,
                sample_count=requested_count,
                rng=python_rng,
            )
            prepared: list[
                tuple[dict[str, str], Path, PreferenceScenario, Any]
            ] = []
            for record in selected_records:
                path = Path(record['output_json'])
                base = load_instance_json(path)
                scenario = sample_scenario(
                    numpy_rng,
                    scenario_config=scenario_config,
                    pa_config=project_config,
                    held_out=False,
                    objective_independent=effective_objective.schema_version == 2,
                )
                instance = apply_scenario(
                    base, scenario, project_config, objective=effective_objective
                )
                prepared.append((record, path, scenario, instance))
        else:
            prepared = training_batch_builder(
                train_buckets=train_buckets,
                train_paths=train_paths,
                instances_per_update=instances_per_update,
                python_rng=python_rng,
                numpy_rng=numpy_rng,
                project_config=project_config,
                objective=effective_objective,
            )
        sample_count = len(prepared)
        batch_group = (
            prepared[0][0].get('size_group')
            or prepared[0][0].get('group')
            or 'unknown'
        )
        batch_max_n = max(instance.n for _, _, _, instance in prepared)

        rollout_elapsed = 0.0
        ppo_elapsed = 0.0
        rollout_summaries: list[Any] = []
        metric_rows: list[dict[str, float]] = []
        if rollout_mode == 'joint':
            envs = [
                _build_env(project_config, ppo_config, effective_objective)
                for _ in prepared
            ]
            rollout_start = time.perf_counter()
            joint_rollout = collect_vectorized_rollout(
                envs=envs,
                instances=[
                    instance for _, _, _, instance in prepared
                ],
                model=model,
                config=ppo_config,
            )
            rollout_elapsed = time.perf_counter() - rollout_start
            auxiliary_loss_fn = (
                None
                if auxiliary_loss_builder is None
                else auxiliary_loss_builder(
                    prepared=prepared,
                    rollout=joint_rollout,
                    objective=effective_objective,
                    project_config=project_config,
                )
            )
            ppo_start = time.perf_counter()
            joint_metrics = update_ppo(
                model=model,
                optimizer=optimizer,
                rollout=joint_rollout,
                config=ppo_config,
                auxiliary_loss_fn=auxiliary_loss_fn,
            )
            ppo_elapsed = time.perf_counter() - ppo_start
            rollout_summaries = list(joint_rollout.instance_summaries)
            metric_rows = [joint_metrics for _ in prepared]
            entropy_normalizer = float(
                np.mean(
                    [
                        np.log(max(2, instance.n * (instance.m - 1)))
                        for _, _, _, instance in prepared
                    ]
                )
            )
        else:
            entropy_normalizer = 0.0
            for _, _, _, instance in prepared:
                env = _build_env(project_config, ppo_config, effective_objective)
                rollout_start = time.perf_counter()
                rollout, _ = collect_rollout(
                    env=env,
                    instance=instance,
                    model=model,
                    config=ppo_config,
                )
                rollout_elapsed += time.perf_counter() - rollout_start
                ppo_start = time.perf_counter()
                metrics = update_ppo(
                    model=model,
                    optimizer=optimizer,
                    rollout=rollout,
                    config=ppo_config,
                )
                ppo_elapsed += time.perf_counter() - ppo_start
                rollout_summaries.append(rollout)
                metric_rows.append(metrics)

        if rollout_mode == 'joint':
            optimizer_minibatches = int(metric_rows[0]['optimizer_minibatches'])
            ppo_epochs_executed = float(metric_rows[0]['ppo_epochs_executed'])
            target_kl_observed = float(metric_rows[0]['target_kl_observed'])
            target_kl_triggered = bool(metric_rows[0]['target_kl_triggered'])
        else:
            optimizer_minibatches = int(
                sum(row['optimizer_minibatches'] for row in metric_rows)
            )
            ppo_epochs_executed = float(
                np.mean([row['ppo_epochs_executed'] for row in metric_rows])
            )
            target_kl_observed = float(
                max(row['target_kl_observed'] for row in metric_rows)
            )
            target_kl_triggered = any(
                bool(row['target_kl_triggered']) for row in metric_rows
            )

        for environment_index, (
            prepared_item,
            rollout_summary,
            metrics,
        ) in enumerate(zip(prepared, rollout_summaries, metric_rows)):
            record, path, scenario, instance = prepared_item
            best_breakdown = score_assignment(
                rollout_summary.best_assignment,
                instance=instance,
                objective=effective_objective,
            )
            if abs(best_breakdown.total_score - float(rollout_summary.best_score)) > 1.0e-9:
                raise RuntimeError('Training score recomputation mismatch exceeds 1e-9.')
            metadata = instance.metadata or {}
            normalizer = (
                entropy_normalizer
                if rollout_mode == 'joint'
                else float(np.log(max(2, instance.n * (instance.m - 1))))
            )
            train_rows.append(
                {
                    'update': update,
                    'phase': 'train',
                    'rollout_mode': rollout_mode,
                    'environment_index': environment_index,
                    'batch_group': batch_group,
                    'batch_max_n': batch_max_n,
                    'source_family_id': _source_family_id(record),
                    'source_identity': _source_family_id(record),
                    'instance_path': str(path),
                    'instance_name': instance.instance_name,
                    'topology': metadata.get('topology', path.parents[1].name),
                    'group': (
                        metadata.get('group')
                        or record.get('size_group')
                        or record.get('group')
                        or path.parent.name
                    ),
                    'n': instance.n,
                    'm': instance.m,
                    'k': instance.k,
                    'scenario_id': scenario.scenario_id,
                    'student_template': scenario.student_template,
                    'teacher_template': scenario.teacher_template,
                    'scenario_seed': scenario.seed,
                    'student_profile_json': json.dumps(scenario.student_profile.tolist()),
                    'teacher_profile_json': json.dumps(scenario.teacher_profile.tolist()),
                    'target_distribution_json': json.dumps(
                        list(effective_objective.target_distribution)
                        if effective_objective.target_distribution is not None
                        else None
                    ),
                    'weights_json': json.dumps(effective_objective.to_dict(), sort_keys=True),
                    'objective_id': effective_objective.objective_id,
                    'objective_hash': effective_objective.config_hash,
                    'best_score': rollout_summary.best_score,
                    'mean_reward': float(rollout_summary.rewards.mean().item()),
                    'reward_std': float(
                        rollout_summary.rewards.std(unbiased=False).item()
                    ),
                    'episode_return': (
                        float(np.mean(rollout_summary.episode_returns))
                        if rollout_summary.episode_returns
                        else 0.0
                    ),
                    'effect_score': best_breakdown.effect_score,
                    'student_pref_score': best_breakdown.student_pref_score,
                    'teacher_pref_score': best_breakdown.teacher_pref_score,
                    'global_score': best_breakdown.global_score,
                    'soft_penalty': best_breakdown.soft_penalty,
                    'illegal_action_count': rollout_summary.illegal_action_count,
                    'illegal_action_rate': float(
                        rollout_summary.illegal_action_count
                        / max(1, int(ppo_config.rollout_steps))
                    ),
                    'normalized_entropy': float(
                        metrics['entropy'] / normalizer
                    ),
                    **metrics,
                }
            )
        recent_rows = train_rows[update_row_start:]
        update_elapsed = time.perf_counter() - update_start
        train_rows.append(
            {
                'update': update,
                'phase': 'train_summary',
                'instance_name': 'shared_policy',
                'rollout_mode': rollout_mode,
                'batch_group': batch_group,
                'batch_max_n': batch_max_n,
                'unique_topologies': len(
                    {row.get('topology') for row in recent_rows}
                ),
                'unique_sources': len(
                    {row.get('source_identity') for row in recent_rows}
                ),
                'instances_per_update': sample_count,
                'rollout_steps': int(ppo_config.rollout_steps),
                'transitions': sample_count * int(ppo_config.rollout_steps),
                'optimizer_minibatches': optimizer_minibatches,
                'ppo_epochs_executed': ppo_epochs_executed,
                'target_kl': (
                    float(ppo_config.target_kl)
                    if ppo_config.target_kl is not None
                    else ''
                ),
                'target_kl_observed': target_kl_observed,
                'target_kl_triggered': float(target_kl_triggered),
                'rollout_elapsed_sec': rollout_elapsed,
                'ppo_elapsed_sec': ppo_elapsed,
                'update_elapsed_sec': update_elapsed,
                'transitions_per_sec': sample_count * int(ppo_config.rollout_steps) / max(update_elapsed, 1e-9),
                'best_score': float(np.mean([row['best_score'] for row in recent_rows])),
                'mean_reward': float(np.mean([row['mean_reward'] for row in recent_rows])),
                'reward_std': _mean_field(recent_rows, 'reward_std'),
                'episode_return': _mean_field(recent_rows, 'episode_return'),
                'effect_score': _mean_field(recent_rows, 'effect_score'),
                'student_pref_score': _mean_field(recent_rows, 'student_pref_score'),
                'teacher_pref_score': _mean_field(recent_rows, 'teacher_pref_score'),
                'global_score': _mean_field(recent_rows, 'global_score'),
                'soft_penalty': _mean_field(recent_rows, 'soft_penalty'),
                'entropy': float(np.mean([row['entropy'] for row in recent_rows])),
                'normalized_entropy': float(
                    np.mean([row['normalized_entropy'] for row in recent_rows])
                ),
                'policy_loss': float(np.mean([row['policy_loss'] for row in recent_rows])),
                'value_loss': float(np.mean([row['value_loss'] for row in recent_rows])),
                'loss': _mean_field(recent_rows, 'loss'),
                **{
                    key: _mean_field(recent_rows, key)
                    for key in recent_rows[0]
                    if key.startswith('auxiliary_')
                },
                'approx_kl': float(np.mean([row['approx_kl'] for row in recent_rows])),
                'clip_fraction': _mean_field(recent_rows, 'clip_fraction'),
                'grad_norm': _mean_field(recent_rows, 'grad_norm'),
                'explained_variance': _mean_field(recent_rows, 'explained_variance'),
                'advantage_mean': _mean_field(recent_rows, 'advantage_mean'),
                'advantage_std': _mean_field(recent_rows, 'advantage_std'),
                'return_mean': _mean_field(recent_rows, 'return_mean'),
                'return_std': _mean_field(recent_rows, 'return_std'),
                'illegal_action_count': int(sum(row['illegal_action_count'] for row in recent_rows)),
                'illegal_action_rate': float(
                    sum(row['illegal_action_count'] for row in recent_rows)
                    / max(1, sample_count * int(ppo_config.rollout_steps))
                ),
                'learning_rate': float(optimizer.param_groups[0]['lr']),
                'entropy_coef': float(ppo_config.entropy_coef),
                'gpu_peak_allocated_mb': (
                    torch.cuda.max_memory_allocated(resolved_device) / (1024**2)
                    if resolved_device.type == 'cuda'
                    else 0.0
                ),
                'gpu_peak_reserved_mb': (
                    torch.cuda.max_memory_reserved(resolved_device) / (1024**2)
                    if resolved_device.type == 'cuda'
                    else 0.0
                ),
            }
        )
        _write_rows(train_rows, train_log)
        _write_rows(
            [row for row in train_rows if row.get('phase') == 'train'],
            research_dir / 'train_instances.csv',
        )
        _write_rows(
            [row for row in train_rows if row.get('phase') == 'train_summary'],
            research_dir / 'train_updates.csv',
        )
        _write_train_tensorboard(writer, train_rows[-1])

        should_evaluate = update % eval_interval == 0 or update == final_update
        if should_evaluate:
            new_rows = (
                evaluate_validation_pairs(
                    model,
                    validation_pairs,
                    project_config=project_config,
                    ppo_config=evaluation_ppo_config,
                    seed=seed,
                    objective=effective_objective,
                )
                if validation_pairs is not None
                else evaluate_model(
                    model,
                    validation_paths,
                    scenarios,
                    project_config=project_config,
                    ppo_config=evaluation_ppo_config,
                    seed=seed,
                    objective=effective_objective,
                )
            )
            for row in new_rows:
                row['update'] = update
            eval_rows.extend(new_rows)
            _write_rows(eval_rows, eval_log)
            validation_summary = validation_policy_summary(new_rows)
            validation_score = validation_summary['mean_total_score']
            validation_p90_gap = validation_summary['p90_relative_gap']
            if _is_better_validation(
                validation_summary,
                best_validation_score,
                best_validation_p90_gap,
            ):
                best_validation_score = validation_score
                best_validation_p90_gap = validation_p90_gap
                save_checkpoint(
                    checkpoint_dir / 'best.pt',
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    hidden_dim=hidden_dim,
                    ppo_config=ppo_config,
                    train_rows=train_rows,
                    eval_rows=eval_rows,
                    python_rng=python_rng,
                    numpy_rng=numpy_rng,
                    best_validation_score=best_validation_score,
                    best_validation_p90_gap=best_validation_p90_gap,
                    provenance=provenance,
                )
                checkpoint_rows.append(
                    {
                        'update': update,
                        'kind': 'best',
                        'path': str(checkpoint_dir / 'best.pt'),
                        'validation_score': validation_score,
                        'validation_p90_gap': validation_p90_gap,
                        'saved_at_utc': _utc_now(),
                    }
                )
                _write_rows(checkpoint_rows, checkpoint_index)
            _write_rows(eval_rows, research_dir / 'validation_instances.csv')
            _write_rows(
                aggregate_validation_rows(eval_rows),
                research_dir / 'validation_updates.csv',
            )
            _write_validation_tensorboard(
                writer,
                new_rows,
                update=update,
                best_validation_score=best_validation_score,
            )
            _write_json_atomic(
                {
                    'schema_version': 2,
                    'status': 'running',
                    'last_update': update,
                    'target_update': final_update,
                    'latest_validation_score': validation_score,
                    'best_validation_score': best_validation_score,
                    'latest_validation_p90_gap': (
                        validation_p90_gap
                        if np.isfinite(validation_p90_gap)
                        else None
                    ),
                    'best_validation_p90_gap': (
                        best_validation_p90_gap
                        if np.isfinite(best_validation_p90_gap)
                        else None
                    ),
                    'train_instance_rows': len(
                        [row for row in train_rows if row.get('phase') == 'train']
                    ),
                    'validation_instance_rows': len(eval_rows),
                    'updated_at_utc': _utc_now(),
                },
                output / 'run_summary.json',
            )

        if update % checkpoint_interval == 0 or update == final_update:
            numbered_path = checkpoint_dir / f'update_{update:06d}.pt'
            checkpoint_args = {
                'model': model,
                'optimizer': optimizer,
                'update': update,
                'hidden_dim': hidden_dim,
                'ppo_config': ppo_config,
                'train_rows': train_rows,
                'eval_rows': eval_rows,
                'python_rng': python_rng,
                'numpy_rng': numpy_rng,
                'best_validation_score': best_validation_score,
                'provenance': provenance,
                'best_validation_p90_gap': best_validation_p90_gap,
            }
            save_checkpoint(numbered_path, **checkpoint_args)
            save_checkpoint(checkpoint_dir / 'latest.pt', **checkpoint_args)
            checkpoint_rows.append(
                {
                    'update': update,
                    'kind': 'numbered',
                    'path': str(numbered_path),
                    'validation_score': (
                        validation_score if should_evaluate else None
                    ),
                    'saved_at_utc': _utc_now(),
                    'validation_p90_gap': validation_p90_gap if should_evaluate else None,
                }
            )
            _write_rows(checkpoint_rows, checkpoint_index)

    training_elapsed = time.perf_counter() - training_start
    if writer is not None:
        writer.flush()
        writer.close()
    _write_json_atomic(
        {
            'schema_version': 2,
            'status': 'completed',
            'last_update': final_update,
            'target_update': final_update,
            'latest_validation_score': validation_score,
            'best_validation_score': best_validation_score,
            'latest_validation_p90_gap': (
                validation_p90_gap
                if np.isfinite(validation_p90_gap)
                else None
            ),
            'best_validation_p90_gap': (
                best_validation_p90_gap
                if np.isfinite(best_validation_p90_gap)
                else None
            ),
            'train_instance_rows': len(
                [row for row in train_rows if row.get('phase') == 'train']
            ),
            'validation_instance_rows': len(eval_rows),
            'training_elapsed_sec': training_elapsed,
            'completed_at_utc': _utc_now(),
        },
        output / 'run_summary.json',
    )
    return {
        'updates': final_update,
        'train_rows': len(train_rows),
        'eval_rows': len(eval_rows),
        'best_validation_score': best_validation_score,
        'latest_checkpoint': str(checkpoint_dir / 'latest.pt'),
        'best_checkpoint': str(checkpoint_dir / 'best.pt'),
        'train_log': str(train_log),
        'validation_log': str(eval_log),
        'metrics_dir': str(research_dir),
        'tensorboard_dir': (
            str(resolved_tensorboard_dir)
            if resolved_tensorboard_dir is not None
            else None
        ),
        'run_metadata': str(output / 'run_metadata.json'),
        'run_summary': str(output / 'run_summary.json'),
        'checkpoint_index': str(checkpoint_index),
        'device': str(resolved_device),
        'training_elapsed_sec': training_elapsed,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', default='data_processed/splits/train.csv')
    parser.add_argument('--validation-manifest', default='data_processed/splits/validation.csv')
    parser.add_argument(
        '--validation-scenarios',
        default='data_processed/splits/scenarios/validation_interpolation.json',
    )
    parser.add_argument('--validation-pairs')
    parser.add_argument(
        '--allow-controlled-held-out-validation',
        action='store_true',
        help=(
            'Allow held-out Scenario v2 rows only when their access_status is '
            'controlled_calibration; this does not make them an independent test.'
        ),
    )
    parser.add_argument('--data-audit')
    parser.add_argument('--objective-config')
    parser.add_argument('--objective-id')
    parser.add_argument('--legacy-v1', action='store_true')
    parser.add_argument('--output-dir', default='results/formal_ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--total-updates', type=int)
    parser.add_argument('--instances-per-update', type=int, default=2)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--minibatch-size', type=int)
    parser.add_argument('--update-epochs', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--entropy-coef', type=float)
    parser.add_argument('--target-kl', type=float)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--eval-limit', type=int, default=30)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int)
    parser.add_argument('--env-max-steps', type=int)
    parser.add_argument('--env-patience', type=int)
    parser.add_argument('--eval-env-max-steps', type=int)
    parser.add_argument('--eval-env-patience', type=int)
    parser.add_argument('--resume-from')
    parser.add_argument('--tensorboard-dir')
    parser.add_argument('--disable-tensorboard', action='store_true')
    parser.add_argument(
        '--rollout-mode',
        choices=('joint', 'sequential'),
        default='joint',
    )
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--evaluate-at-start', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    smoke = args.smoke_test
    if args.objective_config is None and not args.legacy_v1:
        raise SystemExit(
            'New training CLI requires --objective-config; use --legacy-v1 only '
            'for an explicitly requested historical run.'
        )
    objective = (
        load_objective_spec(args.objective_config, args.objective_id)
        if args.objective_config is not None
        else None
    )
    result = run_formal_training(
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        validation_scenarios=args.validation_scenarios,
        output_dir=args.output_dir,
        seed=args.seed,
        validation_pairs_manifest=args.validation_pairs,
        allow_controlled_held_out_validation=(
            args.allow_controlled_held_out_validation
        ),
        data_audit=args.data_audit,
        device=args.device,
        total_updates=1 if smoke else args.total_updates,
        instances_per_update=1 if smoke else args.instances_per_update,
        rollout_steps=2 if smoke else args.rollout_steps,
        minibatch_size=2 if smoke else args.minibatch_size,
        update_epochs=1 if smoke else args.update_epochs,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        target_kl=args.target_kl,
        eval_interval=1 if smoke else args.eval_interval,
        eval_limit=1 if smoke else args.eval_limit,
        checkpoint_interval=1 if smoke else args.checkpoint_interval,
        hidden_dim=args.hidden_dim,
        env_max_steps=4 if smoke else args.env_max_steps,
        env_patience=4 if smoke else args.env_patience,
        eval_env_max_steps=4 if smoke else args.eval_env_max_steps,
        eval_env_patience=4 if smoke else args.eval_env_patience,
        resume_from=args.resume_from,
        tensorboard_dir=args.tensorboard_dir,
        enable_tensorboard=not args.disable_tensorboard,
        rollout_mode=args.rollout_mode,
        objective=objective,
        evaluate_at_start=args.evaluate_at_start,
        deterministic=args.deterministic,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
