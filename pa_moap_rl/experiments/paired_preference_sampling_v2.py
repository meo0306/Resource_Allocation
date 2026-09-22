"""Versioned same-content counterfactual preference sampling for PPO repair."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
from typing import Any

import numpy as np

from pa_moap_rl.data.loader import load_instance_json
from pa_moap_rl.data.scenarios import (
    PreferenceScenario,
    apply_scenario,
    load_scenario_config,
    sample_scenario,
)


INDEPENDENT_SAMPLER_VERSION = "independent_random_v1"
PAIRED_SAMPLER_VERSION = "paired_counterfactual_v1"
SUPPORTED_SAMPLER_VERSIONS = {
    INDEPENDENT_SAMPLER_VERSION,
    PAIRED_SAMPLER_VERSION,
}


def sampling_spec(sampler_version: str) -> dict[str, Any]:
    if sampler_version == INDEPENDENT_SAMPLER_VERSION:
        return {
            "sampler_version": sampler_version,
            "same_content_counterfactual": False,
            "batch_layout": "four_independent_content_profile_pairs",
        }
    if sampler_version == PAIRED_SAMPLER_VERSION:
        return {
            "sampler_version": sampler_version,
            "same_content_counterfactual": True,
            "batch_layout": "one_triplet_content_plus_one_independent_random_content_per_four_envs",
            "triplet_templates": ["S6-T6", "S4-T6", "S6-T4"],
            "shared_unchanged_actor_profile": True,
        }
    raise ValueError(f"Unsupported sampler_version: {sampler_version}")


def counterfactual_triplet(
    rng: np.random.Generator,
    *,
    scenario_config: dict[str, Any],
    project_config: Any,
    objective_independent: bool,
    prefix: str,
) -> list[PreferenceScenario]:
    """Create balanced/student-only/teacher-only scenarios with exact sharing."""

    balanced = sample_scenario(
        rng,
        scenario_config=scenario_config,
        pa_config=project_config,
        held_out=False,
        scenario_id=f"{prefix}-balanced",
        template_pair=("S6", "T6"),
        objective_independent=objective_independent,
        access_status="development_counterfactual_train",
    )
    student_draw = sample_scenario(
        rng,
        scenario_config=scenario_config,
        pa_config=project_config,
        held_out=False,
        scenario_id=f"{prefix}-student-draw",
        template_pair=("S4", "T6"),
        objective_independent=objective_independent,
        access_status="development_counterfactual_train",
    )
    teacher_draw = sample_scenario(
        rng,
        scenario_config=scenario_config,
        pa_config=project_config,
        held_out=False,
        scenario_id=f"{prefix}-teacher-draw",
        template_pair=("S6", "T4"),
        objective_independent=objective_independent,
        access_status="development_counterfactual_train",
    )
    student_only = replace(
        balanced,
        scenario_id=f"{prefix}-student-only",
        student_template="S4",
        student_profile=student_draw.student_profile.copy(),
        seed=student_draw.seed,
    )
    teacher_only = replace(
        balanced,
        scenario_id=f"{prefix}-teacher-only",
        teacher_template="T4",
        teacher_profile=teacher_draw.teacher_profile.copy(),
        seed=teacher_draw.seed,
    )
    return [balanced, student_only, teacher_only]


def build_paired_counterfactual_batch(
    *,
    train_buckets: dict[Any, Any],
    train_paths: list[Path],
    instances_per_update: int,
    python_rng: random.Random,
    numpy_rng: np.random.Generator,
    project_config: Any,
    objective: Any,
) -> list[tuple[dict[str, str], Path, PreferenceScenario, Any]]:
    """Build versioned blocks of three paired and one independent environment."""

    if instances_per_update < 4 or instances_per_update % 4 != 0:
        raise ValueError(
            "paired_counterfactual_v1 requires instances_per_update to be a multiple of 4."
        )
    from pa_moap_rl.experiments.formal_training import (
        sample_stratified_training_batch,
    )

    block_count = instances_per_update // 4
    selected = sample_stratified_training_batch(
        train_buckets,
        sample_count=2 * block_count,
        rng=python_rng,
    )
    scenario_config = load_scenario_config()
    prepared: list[tuple[dict[str, str], Path, PreferenceScenario, Any]] = []
    objective_independent = objective.schema_version == 2
    for block in range(block_count):
        anchor_record = selected[2 * block]
        random_record = selected[2 * block + 1]
        anchor_path = Path(anchor_record["output_json"])
        anchor_base = load_instance_json(anchor_path)
        prefix = f"paired-v1-b{block}-{int(numpy_rng.integers(0, 2**31 - 1)):010d}"
        for scenario in counterfactual_triplet(
            numpy_rng,
            scenario_config=scenario_config,
            project_config=project_config,
            objective_independent=objective_independent,
            prefix=prefix,
        ):
            prepared.append(
                (
                    anchor_record,
                    anchor_path,
                    scenario,
                    apply_scenario(
                        anchor_base,
                        scenario,
                        project_config,
                        objective=objective,
                    ),
                )
            )
        random_path = Path(random_record["output_json"])
        random_base = load_instance_json(random_path)
        random_scenario = sample_scenario(
            numpy_rng,
            scenario_config=scenario_config,
            pa_config=project_config,
            held_out=False,
            scenario_id=f"{prefix}-independent-random",
            objective_independent=objective_independent,
            access_status="development_random_train",
        )
        prepared.append(
            (
                random_record,
                random_path,
                random_scenario,
                apply_scenario(
                    random_base,
                    random_scenario,
                    project_config,
                    objective=objective,
                ),
            )
        )
    if len(prepared) != instances_per_update:
        raise RuntimeError("Paired sampler returned an unexpected batch size.")
    return prepared


__all__ = [
    "INDEPENDENT_SAMPLER_VERSION",
    "PAIRED_SAMPLER_VERSION",
    "SUPPORTED_SAMPLER_VERSIONS",
    "build_paired_counterfactual_batch",
    "counterfactual_triplet",
    "sampling_spec",
]
