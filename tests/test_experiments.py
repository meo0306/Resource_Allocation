"""Tests for Phase 8 experiment framework."""

from pathlib import Path

import pandas as pd

from pa_moap_rl.experiments.analyze_results import analyze_results
from pa_moap_rl.experiments.run_ablation import run_ablation_experiments
from pa_moap_rl.experiments.run_batch import run_batch_experiments


def test_run_batch_smoke_writes_complete_results(tmp_path: Path) -> None:
    output_csv = tmp_path / "batch.csv"

    rows = run_batch_experiments(
        instance_root="instance",
        output_csv=output_csv,
        group="group1",
        limit=1,
        seeds=[0],
        include_local_search=False,
        include_ppo=False,
    )

    assert output_csv.exists()
    assert len(rows) == 3
    df = pd.read_csv(output_csv)
    assert {"solver_name", "total_score", "effect_score", "runtime", "mask_violation_count"} <= set(df.columns)
    assert set(df["solver_name"]) == {"random_legal", "effect_only_greedy", "scalarized_independent_greedy"}
    assert int(df["mask_violation_count"].sum()) == 0


def test_run_ablation_smoke_writes_variant_results(tmp_path: Path) -> None:
    output_csv = tmp_path / "ablation.csv"

    rows = run_ablation_experiments(
        instance_root="instance",
        output_csv=output_csv,
        group="group1",
        limit=1,
        seeds=[0],
    )

    assert output_csv.exists()
    assert len(rows) == 8
    df = pd.read_csv(output_csv)
    assert "variant" in df.columns
    assert any(value.startswith("no_preference") for value in df["variant"])
    assert any(value.startswith("no_global") for value in df["variant"])
    assert int(df["mask_violation_count"].sum()) == 0


def test_analyze_results_writes_tables_and_figures(tmp_path: Path) -> None:
    result_csv = tmp_path / "batch.csv"
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    run_batch_experiments(
        instance_root="instance",
        output_csv=result_csv,
        group="group1",
        limit=1,
        seeds=[0],
        include_local_search=False,
        include_ppo=False,
    )

    outputs = analyze_results(input_paths=[result_csv], table_dir=table_dir, figure_dir=figure_dir)

    assert Path(outputs["summary"]).exists()
    assert Path(outputs["case_table"]).exists()
    assert Path(outputs["total_score_plot"]).exists()
    assert Path(outputs["tradeoff_plot"]).exists()
