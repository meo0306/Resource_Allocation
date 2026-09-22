'''Regression tests for formal-training foundation changes.'''

import csv
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.build_dataset import assign_source_splits, split_source_id
from pa_moap_rl.data.loader import save_instance_json
from pa_moap_rl.data.scenarios import (
    apply_scenario,
    generate_scenario_bank,
    load_scenario_config,
    save_scenario_bank,
    scenario_pairs,
)
from pa_moap_rl.data.selection_loader import load_selection_csv
from pa_moap_rl.experiments.formal_training import (
    build_training_buckets,
    load_split_paths,
    load_validation_pairs,
    run_formal_training,
    sample_stratified_training_batch,
)
from pa_moap_rl.objective import ObjectiveSpec
from pa_moap_rl.utils.scoring import IncrementalAssignmentScorer, score_assignment


def example_instance():
    record = load_selection_csv('examples/summary_ga_own_group1_260417_210154.csv')[0]
    return build_assignment_instance_from_selection('examples', record)


def test_scenario_catalogue_and_application_are_reproducible() -> None:
    config = load_scenario_config()
    assert len(scenario_pairs(config, held_out=False)) == 30
    assert len(scenario_pairs(config, held_out=True)) == 6
    left = generate_scenario_bank(3, seed=17)
    right = generate_scenario_bank(3, seed=17)
    np.testing.assert_allclose(left[0].student_profile, right[0].student_profile)
    assert left[0].scenario_id == right[0].scenario_id
    held_out_bank = generate_scenario_bank(6, seed=19, held_out=True)
    assert {
        (scenario.student_template, scenario.teacher_template)
        for scenario in held_out_bank
    } == set(scenario_pairs(config, held_out=True))

    base = example_instance()
    conditioned = apply_scenario(base, left[0])
    np.testing.assert_allclose(conditioned.effect_matrix, base.effect_matrix)
    assert not np.allclose(conditioned.student_pref, base.student_pref)
    assert conditioned.metadata['scenario']['scenario_id'] == left[0].scenario_id


def test_topology_variants_share_one_split() -> None:
    rows = [
        {
            'topology': topology,
            'group': 'group1',
            'instance_name': f'inst_V60_w6-10_s{seed}.sm',
            'output_json': 'unused',
        }
        for topology in ('uniform', 'random', 'staircase', 'hourglass', 'bottleneck')
        for seed in range(20)
    ]
    assignments = assign_source_splits(rows, seed=23)
    for seed in range(20):
        variants = [row for row in rows if row['instance_name'].endswith(f's{seed}.sm')]
        assert len({assignments[split_source_id(row)] for row in variants}) == 1
    counts = {split: list(assignments.values()).count(split) for split in ('train', 'validation', 'test')}
    assert counts == {'train': 14, 'validation': 3, 'test': 3}


def test_incremental_score_matches_full_recomputation() -> None:
    instance = example_instance()
    assignment = np.argmax(instance.effect_matrix, axis=1)
    scorer = IncrementalAssignmentScorer(assignment, instance)
    rng = np.random.default_rng(9)
    for _ in range(20):
        node = int(rng.integers(instance.n))
        method = int(rng.integers(instance.m))
        candidate = scorer.assignment.copy()
        candidate[node] = method
        expected = score_assignment(candidate, instance=instance).total_score
        assert scorer.candidate_breakdown(node, method).total_score == pytest.approx(expected, abs=1e-12)
        scorer.apply(node, method)
        assert scorer.score == pytest.approx(expected, abs=1e-12)


def test_formal_training_checkpoint_and_resume(tmp_path: Path) -> None:
    instance_path = tmp_path / 'instance.assignment.json'
    save_instance_json(example_instance(), instance_path)
    fields = ['topology', 'group', 'instance_name', 'output_json']
    for name in ('train', 'validation'):
        with (tmp_path / f'{name}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    'topology': 'uniform',
                    'group': 'group1',
                    'instance_name': 'example.sm',
                    'output_json': str(instance_path),
                }
            )
    scenarios_path = tmp_path / 'scenarios.json'
    save_scenario_bank(generate_scenario_bank(2, seed=31), scenarios_path)
    output = tmp_path / 'formal'
    first = run_formal_training(
        train_manifest=tmp_path / 'train.csv',
        validation_manifest=tmp_path / 'validation.csv',
        validation_scenarios=scenarios_path,
        output_dir=output,
        seed=5,
        device='cpu',
        total_updates=1,
        instances_per_update=1,
        rollout_steps=2,
        minibatch_size=2,
        update_epochs=1,
        eval_interval=1,
        eval_limit=1,
        checkpoint_interval=1,
        hidden_dim=16,
        env_max_steps=2,
        env_patience=2,
        eval_env_max_steps=4,
        eval_env_patience=3,
    )
    latest = Path(first['latest_checkpoint'])
    assert latest.exists()
    assert any(Path(first['tensorboard_dir']).glob('events.out.tfevents.*'))
    for filename in (
        'train_instances.csv',
        'train_updates.csv',
        'validation_instances.csv',
        'validation_updates.csv',
    ):
        assert (Path(first['metrics_dir']) / filename).exists()
    metadata = json.loads(Path(first['run_metadata']).read_text(encoding='utf-8'))
    summary_data = json.loads(Path(first['run_summary']).read_text(encoding='utf-8'))
    assert metadata['inputs']['train_manifest']['sha256']
    assert metadata['ppo_config']['env_max_steps'] == 2
    assert metadata['ppo_config']['env_patience'] == 2
    assert metadata['evaluation_ppo_config']['env_max_steps'] == 4
    assert metadata['evaluation_ppo_config']['env_patience'] == 3
    assert metadata['training_and_evaluation_environment_separated']
    assert summary_data['status'] == 'completed'
    with Path(first['train_log']).open('r', encoding='utf-8', newline='') as handle:
        logged = list(csv.DictReader(handle))
    summary = next(row for row in logged if row['phase'] == 'train_summary')
    assert float(summary['transitions_per_sec']) > 0.0
    assert 0.0 <= float(summary['normalized_entropy']) <= 1.01
    resumed = run_formal_training(
        train_manifest=tmp_path / 'train.csv',
        validation_manifest=tmp_path / 'validation.csv',
        validation_scenarios=scenarios_path,
        output_dir=output,
        seed=5,
        device='cpu',
        total_updates=2,
        instances_per_update=1,
        rollout_steps=2,
        minibatch_size=2,
        update_epochs=1,
        eval_interval=1,
        eval_limit=1,
        checkpoint_interval=1,
        hidden_dim=16,
        env_max_steps=2,
        env_patience=2,
        resume_from=latest,
        learning_rate=1.0e-4,
        target_kl=0.03,
    )
    assert resumed['updates'] == 2
    assert resumed['train_rows'] == 4
    restored = torch.load(resumed['latest_checkpoint'], map_location='cpu', weights_only=False)
    assert restored['optimizer_state_dict']['param_groups'][0]['lr'] == pytest.approx(1.0e-4)
    assert restored['ppo_config']['learning_rate'] == pytest.approx(1.0e-4)
    assert restored['ppo_config']['target_kl'] == pytest.approx(0.03)
    with (Path(resumed['metrics_dir']) / 'train_updates.csv').open(
        'r', encoding='utf-8', newline=''
    ) as handle:
        update_rows = list(csv.DictReader(handle))
    assert update_rows[-1]['ppo_epochs_executed']
    assert update_rows[-1]['optimizer_minibatches']
    assert update_rows[-1]['target_kl_triggered']
    with Path(resumed['checkpoint_index']).open('r', encoding='utf-8', newline='') as handle:
        checkpoint_index = list(csv.DictReader(handle))
    assert {int(row['update']) for row in checkpoint_index} >= {1, 2}
    assert not list(output.rglob('*.tmp'))


def test_training_batch_is_size_homogeneous_content_unique_and_topology_diverse() -> None:
    records = [
        {
            'group': group,
            'topology': topology,
            'source_identity': f'{group}/source-{source}',
            'output_json': f'{group}-{topology}-{source}.json',
        }
        for group in ('group1', 'group2')
        for topology in (
            'uniform',
            'random',
            'staircase',
            'hourglass',
            'bottleneck',
        )
        for source in range(5)
    ]
    selected = sample_stratified_training_batch(
        build_training_buckets(records),
        sample_count=4,
        rng=random.Random(23),
    )
    assert len({record['group'] for record in selected}) == 1
    assert len({record['topology'] for record in selected}) == 4
    assert len({record['source_identity'] for record in selected}) == 4


def test_training_batch_accepts_four_topology_size_group_column() -> None:
    records = [
        {
            'size_group': size_group,
            'topology': topology,
            'source_family_id': f'{size_group}/source-{source}',
            'output_json': f'{size_group}-{topology}-{source}.json',
        }
        for size_group in ('group1', 'group2')
        for topology in ('uniform', 'random', 'hourglass', 'bottleneck')
        for source in range(4)
    ]
    selected = sample_stratified_training_batch(
        build_training_buckets(records),
        sample_count=4,
        rng=random.Random(29),
    )
    assert len({record['size_group'] for record in selected}) == 1
    assert len({record['topology'] for record in selected}) == 4
    assert len({record['source_family_id'] for record in selected}) == 4


def test_limited_split_loading_round_robins_groups_and_topologies(tmp_path: Path) -> None:
    fields = ['topology', 'group', 'instance_name', 'output_json']
    manifest = tmp_path / 'validation.csv'
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for topology in ('hourglass', 'random', 'uniform'):
            for group in ('group1', 'group2'):
                path = tmp_path / f'{group}-{topology}.json'
                path.write_text('{}', encoding='utf-8')
                writer.writerow(
                    {
                        'topology': topology,
                        'group': group,
                        'instance_name': path.name,
                        'output_json': str(path),
                    }
                )
    selected = load_split_paths(manifest, limit=6)
    assert {path.stem.split('-', 1)[0] for path in selected} == {'group1', 'group2'}
    assert {path.stem.split('-', 1)[1] for path in selected} == {'hourglass', 'random', 'uniform'}


def test_controlled_held_out_validation_requires_explicit_opt_in(tmp_path: Path) -> None:
    objective = ObjectiveSpec(
        objective_id='test_v2',
        schema_version=2,
        alpha_effect=0.35,
        alpha_preference=0.65,
        alpha_global=0.0,
        entropy_min=0.65,
        method_cap=0.35,
        beta_entropy=1.0,
        beta_cap=1.0,
        target_distribution=None,
        epsilon=1.0e-8,
    )
    instance_path = tmp_path / 'instance.assignment.json'
    base = example_instance()
    save_instance_json(base, instance_path)
    scenario = replace(
        generate_scenario_bank(
            1,
            seed=71,
            objective_independent=True,
        )[0],
        held_out=True,
        access_status='controlled_calibration',
    )
    scenario_path = tmp_path / 'scenarios.json'
    save_scenario_bank([scenario], scenario_path)
    conditioned = apply_scenario(base, scenario, objective=objective)
    assignment = np.argmax(conditioned.effect_matrix, axis=1)
    score = score_assignment(assignment, instance=conditioned, objective=objective)
    exact_path = tmp_path / 'pair-0.json'
    exact_path.write_text(
        json.dumps(
            {
                'pair_id': 'pair-0',
                'objective_hash': objective.config_hash,
                'proven_optimal': True,
                'assignment': assignment.tolist(),
                'score': {'total_score': score.total_score},
            }
        ),
        encoding='utf-8',
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tmp_path / 'pairs.csv'
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                'pair_id': 'pair-0',
                'original_id': 'original-0',
                'source_family_id': 'group1/source-0',
                'topology': 'uniform',
                'size_group': 'group1',
                'scenario_id': scenario.scenario_id,
                'output_json': str(instance_path),
                'output_json_sha256': digest(instance_path),
                'exact_result_path': str(exact_path),
                'exact_result_sha256': digest(exact_path),
            }
        )

    with pytest.raises(ValueError, match='explicit controlled-calibration'):
        load_validation_pairs(
            manifest,
            scenario_bank=scenario_path,
            objective=objective,
        )
    pairs = load_validation_pairs(
        manifest,
        scenario_bank=scenario_path,
        objective=objective,
        allow_controlled_held_out=True,
    )
    assert len(pairs) == 1
    assert pairs[0].exact_score == pytest.approx(score.total_score)
