'''Rebuild a clean TensorBoard event stream from formal-training CSV metrics.'''

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from torch.utils.tensorboard import SummaryWriter


TRAIN_GROUPS = {
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
}

VALIDATION_NAMES = (
    'total_score_mean',
    'total_score_std',
    'effect_score_mean',
    'student_pref_score_mean',
    'teacher_pref_score_mean',
    'global_score_mean',
    'soft_penalty_mean',
    'hard_feasible_rate_mean',
)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, '')
    if value in ('', None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _updates(rows: Iterable[dict[str, str]]) -> list[int]:
    return sorted({int(row['update']) for row in rows})


def rebuild_tensorboard(
    run_dir: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    '''Write one clean, complete event stream from immutable research CSVs.'''

    root = Path(run_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'TensorBoard output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    train_rows = _rows(root / 'metrics' / 'train_updates.csv')
    validation_rows = _rows(root / 'metrics' / 'validation_updates.csv')
    writer = SummaryWriter(log_dir=str(output))
    try:
        for row in train_rows:
            update = int(row['update'])
            for group, names in TRAIN_GROUPS.items():
                for name in names:
                    value = _number(row, name)
                    if value is not None:
                        writer.add_scalar(f'{group}/{name}', value, update)
        best_score = -float('inf')
        for update in _updates(validation_rows):
            current = [
                row for row in validation_rows
                if int(row['update']) == update
            ]
            overall = next(row for row in current if row['scope'] == 'overall')
            for name in VALIDATION_NAMES:
                value = _number(overall, name)
                if value is not None:
                    writer.add_scalar(f'validation/{name}', value, update)
            current_score = float(overall['total_score_mean'])
            best_score = max(best_score, current_score)
            writer.add_scalar('validation/best_total_score_mean', best_score, update)
            writer.add_scalar(
                'validation/gap_to_best', current_score - best_score, update
            )
            for row in current:
                if row['scope'] == 'topology':
                    writer.add_scalar(
                        'validation_by_topology/{}'.format(row['scope_value']),
                        float(row['total_score_mean']),
                        update,
                    )
        writer.flush()
    finally:
        writer.close()
    result = {
        'schema_version': 1,
        'source_run_dir': str(root.resolve()),
        'output_dir': str(output.resolve()),
        'train_updates': len(train_rows),
        'validation_points': len(_updates(validation_rows)),
    }
    (output / 'rebuild_metadata.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = rebuild_tensorboard(args.run_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
