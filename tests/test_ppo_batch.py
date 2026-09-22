"""Tests for shared PPO batch training."""

from pathlib import Path

import pandas as pd

from pa_moap_rl.experiments.train_ppo_batch import (
    _replace_with_retry,
    run_shared_batch_training,
)


def test_csv_replace_retries_short_windows_reader_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temporary = tmp_path / 'metrics.csv.tmp'
    output = tmp_path / 'metrics.csv'
    temporary.write_text('new', encoding='utf-8')
    output.write_text('old', encoding='utf-8')
    real_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError('reader still owns the target')
        return real_replace(path, target)

    monkeypatch.setattr(Path, 'replace', flaky_replace)
    _replace_with_retry(temporary, output, attempts=3, delay_seconds=0.0)

    assert attempts == 3
    assert output.read_text(encoding='utf-8') == 'new'


def test_shared_batch_training_smoke_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo_batch"
    tensorboard_dir = tmp_path / "tensorboard"

    result = run_shared_batch_training(
        instance_root="instance",
        output_dir=output_dir,
        group="group1",
        limit=2,
        eval_limit=1,
        seed=11,
        device="cpu",
        total_updates=1,
        instances_per_update=1,
        rollout_steps=4,
        minibatch_size=2,
        update_epochs=1,
        env_max_steps=4,
        env_patience=4,
        include_baselines=True,
        include_local_search=False,
        hidden_dim=16,
        tensorboard_dir=tensorboard_dir,
    )

    assert result.train_log_path.exists()
    assert result.eval_log_path.exists()
    assert result.checkpoint_path.exists()
    assert result.assignment_dir.exists()
    assert result.tensorboard_dir == tensorboard_dir
    assert any(result.assignment_dir.rglob("*.assignment.json"))
    assert any(tensorboard_dir.glob("events.out.tfevents.*"))

    train_df = pd.read_csv(result.train_log_path)
    eval_df = pd.read_csv(result.eval_log_path)
    assert {"train", "train_summary"} <= set(train_df["phase"])
    assert "ppo_shared_policy" in set(eval_df["solver_name"])
    assert float(train_df.loc[train_df["phase"] == "train_summary", "illegal_action_rate"].iloc[-1]) == 0.0
