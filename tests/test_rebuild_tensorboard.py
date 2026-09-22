'''Tests for rebuilding TensorBoard events from research CSV files.'''

from __future__ import annotations

import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pa_moap_rl.experiments.rebuild_tensorboard import rebuild_tensorboard


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_rebuild_tensorboard_writes_train_validation_and_topology(
    tmp_path: Path,
) -> None:
    run = tmp_path / 'run'
    _write(
        run / 'metrics' / 'train_updates.csv',
        [
            {
                'update': update,
                'approx_kl': 0.01 * update,
                'normalized_entropy': 0.7,
            }
            for update in (1, 2)
        ],
    )
    _write(
        run / 'metrics' / 'validation_updates.csv',
        [
            {
                'update': update,
                'scope': scope,
                'scope_value': scope_value,
                'total_score_mean': score,
            }
            for update, score in ((1, 0.60), (2, 0.65))
            for scope, scope_value in (('overall', 'all'), ('topology', 'staircase'))
        ],
    )
    output = tmp_path / 'tensorboard'
    result = rebuild_tensorboard(run, output_dir=output)
    events = EventAccumulator(str(output)).Reload()

    assert result['train_updates'] == 2
    assert result['validation_points'] == 2
    assert len(events.Scalars('train/approx_kl')) == 2
    assert len(events.Scalars('validation/total_score_mean')) == 2
    assert len(events.Scalars('validation_by_topology/staircase')) == 2
    assert (output / 'rebuild_metadata.json').is_file()
