'''Run or resume the confirmed four-topology b075 seed0 pilot from one YAML file.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pa_moap_rl.experiments.formal_training import run_formal_training
from pa_moap_rl.objective import load_objective_spec


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f'Pilot configuration is missing {key!r}.')
    return mapping[key]


def run_pilot_from_config(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or int(config.get('schema_version', 0)) != 1:
        raise ValueError('Unsupported seed0 pilot configuration schema.')
    if config.get('status') != 'user_confirmed':
        raise ValueError('Seed0 pilot configuration is not user-confirmed.')
    objective_cfg = _required(config, 'objective')
    data = _required(config, 'data')
    run = _required(config, 'run')
    objective_path = Path(_required(objective_cfg, 'config'))
    objective = load_objective_spec(objective_path)
    if objective.config_hash != str(_required(objective_cfg, 'config_hash')):
        raise ValueError('Pilot objective hash differs from the frozen configuration.')

    output = Path(_required(run, 'output_dir'))
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'Pilot output already contains files; pass --resume-from explicitly: {output}'
        )
    target_kl = run.get('target_kl')
    return run_formal_training(
        train_manifest=_required(data, 'train_manifest'),
        validation_manifest=_required(data, 'validation_manifest'),
        validation_scenarios=_required(data, 'validation_scenarios'),
        validation_pairs_manifest=_required(data, 'validation_pairs'),
        allow_controlled_held_out_validation=bool(
            data.get('allow_controlled_held_out_validation', False)
        ),
        data_audit=_required(data, 'data_audit'),
        pilot_config=path,
        output_dir=output,
        seed=int(_required(run, 'seed')),
        device=str(_required(run, 'device')),
        total_updates=int(_required(run, 'total_updates')),
        instances_per_update=int(_required(run, 'instances_per_update')),
        rollout_steps=int(_required(run, 'rollout_steps')),
        minibatch_size=int(_required(run, 'minibatch_size')),
        update_epochs=int(_required(run, 'update_epochs')),
        learning_rate=float(_required(run, 'learning_rate')),
        entropy_coef=float(_required(run, 'entropy_coef')),
        target_kl=None if target_kl is None else float(target_kl),
        eval_interval=int(_required(run, 'eval_interval')),
        checkpoint_interval=int(_required(run, 'checkpoint_interval')),
        hidden_dim=int(_required(run, 'hidden_dim')),
        env_max_steps=int(_required(run, 'env_max_steps')),
        env_patience=int(_required(run, 'env_patience')),
        eval_env_max_steps=(
            None
            if run.get('eval_env_max_steps') is None
            else int(run['eval_env_max_steps'])
        ),
        eval_env_patience=(
            None
            if run.get('eval_env_patience') is None
            else int(run['eval_env_patience'])
        ),
        resume_from=resume_from,
        rollout_mode=str(_required(run, 'rollout_mode')),
        objective=objective,
        evaluate_at_start=bool(run.get('evaluate_at_start', False)),
        deterministic=bool(run.get('deterministic', False)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='pa_moap_rl/configs/seed0_pilot_four_topology_v2.yaml',
    )
    parser.add_argument('--resume-from')
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_pilot_from_config(args.config, resume_from=args.resume_from)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
