'''Evaluate a saved shared PPO checkpoint on fixed test scenarios.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pa_moap_rl.checkpointing import method_matrix_hash, validate_checkpoint_metadata
from pa_moap_rl.configs import load_config
from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import load_scenario_bank
from pa_moap_rl.experiments.formal_training import evaluate_model, load_split_paths
from pa_moap_rl.experiments.train_ppo_batch import _write_rows
from pa_moap_rl.objective import ObjectiveSpec, load_objective_spec
from pa_moap_rl.solvers.ppo_solver import PPOConfig, build_actor_critic, resolve_device


def _slice_evaluation_inputs(
    paths: list,
    scenarios: list,
    *,
    offset: int,
    limit: int | None,
) -> tuple[list, list]:
    '''Slice paths while preserving the full-run index-to-scenario mapping.'''

    if offset < 0:
        raise ValueError('offset must be non-negative.')
    if not scenarios:
        raise ValueError('scenarios must be non-empty.')
    stop = None if limit is None else offset + limit
    selected_paths = paths[offset:stop]
    if not selected_paths:
        raise ValueError('The requested evaluation slice is empty.')
    scenario_offset = offset % len(scenarios)
    rotated_scenarios = scenarios[scenario_offset:] + scenarios[:scenario_offset]
    return selected_paths, rotated_scenarios


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    test_manifest: str | Path,
    scenario_bank: str | Path,
    output_csv: str | Path,
    device: str = 'auto',
    offset: int = 0,
    limit: int | None = None,
    include_baselines: bool = True,
    include_local_search: bool = False,
    seed: int = 0,
    objective: ObjectiveSpec | None = None,
    expected_data_hash: str | None = None,
    legacy_mode: bool = False,
) -> list[dict]:
    if objective is None or expected_data_hash is None:
        raise ValueError(
            'Evaluation requires an explicit ObjectiveSpec and expected_data_hash.'
        )
    project_config = load_config()
    resolved_device = resolve_device(device)
    all_paths = load_split_paths(test_manifest, limit=None)
    scenarios = load_scenario_bank(scenario_bank)
    paths, scenarios = _slice_evaluation_inputs(
        all_paths,
        scenarios,
        offset=offset,
        limit=limit,
    )
    checkpoint = torch.load(Path(checkpoint_path), map_location=resolved_device, weights_only=False)
    first_instance = load_instance_json(paths[0])
    validate_checkpoint_metadata(
        checkpoint,
        objective=objective,
        method_hash=method_matrix_hash(first_instance),
        data_hash=expected_data_hash,
        legacy_mode=legacy_mode,
    )
    ppo_config = PPOConfig(**checkpoint['ppo_config'])
    ppo_config = PPOConfig(**{**ppo_config.__dict__, 'device': str(resolved_device)})
    model = build_actor_critic(
        first_instance,
        config=project_config,
        device=resolved_device,
        hidden_dim=checkpoint.get('hidden_dim'),
        objective=objective,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    rows = evaluate_model(
        model,
        paths,
        scenarios,
        project_config=project_config,
        ppo_config=ppo_config,
        seed=seed,
        include_baselines=include_baselines,
        include_local_search=include_local_search,
        objective=objective,
    )
    _write_rows(rows, output_csv)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-manifest', default='data_processed/splits/test.csv')
    parser.add_argument(
        '--scenario-bank',
        default='data_processed/splits/scenarios/test_interpolation.json',
    )
    parser.add_argument('--output-csv', default='results/formal_ppo/test_results.csv')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--include-local-search', action='store_true')
    parser.add_argument('--objective-config', required=True)
    parser.add_argument('--objective-id')
    parser.add_argument('--expected-data-hash', required=True)
    parser.add_argument('--legacy-checkpoint', action='store_true')
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    objective = load_objective_spec(args.objective_config, args.objective_id)
    rows = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_manifest=args.test_manifest,
        scenario_bank=args.scenario_bank,
        output_csv=args.output_csv,
        device=args.device,
        offset=args.offset,
        limit=args.limit,
        include_baselines=not args.skip_baselines,
        include_local_search=args.include_local_search,
        seed=args.seed,
        objective=objective,
        expected_data_hash=args.expected_data_hash,
        legacy_mode=args.legacy_checkpoint,
    )
    print(json.dumps({'rows': len(rows), 'output_csv': args.output_csv}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
