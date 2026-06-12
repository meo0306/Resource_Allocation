"""Tests for assignment instance construction."""

import numpy as np
import pytest

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_selection
from pa_moap_rl.data.loader import (
    load_instance_json,
    load_instance_npz,
    save_instance_json,
    save_instance_npz,
)
from pa_moap_rl.data.selection_loader import find_instance_file, load_pre_data_pairs, load_selection_csv


def test_load_content_one_csv_and_find_instance_file() -> None:
    records = load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")
    first = records[0]

    assert first.instance_name == "inst_V60_w6-10_s670487.sm"
    assert first.x.shape == (60,)
    assert len(first.selected_ids) == 43
    assert first.selected_ids[:3] == [1, 2, 5]
    assert find_instance_file("examples", first.instance_name).name == first.instance_name


def test_build_assignment_instance_shapes_and_values() -> None:
    config = load_config()
    first = load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0]
    instance = build_assignment_instance_from_selection("examples", first, config=config)

    assert instance.n == len(first.selected_ids)
    assert instance.m == 11
    assert instance.k == 6
    assert instance.category_id.shape == (instance.n,)
    assert instance.cognitive_load.shape == (instance.n,)
    assert instance.concept_need.shape == (instance.n, 6)
    assert instance.method_attr.shape == (11, 6)
    assert instance.effect_matrix.shape == (instance.n, 11)
    assert instance.student_pref.shape == (instance.n, 11)
    assert instance.teacher_pref.shape == (instance.n, 11)
    assert instance.feasible_mask.shape == (instance.n, 11)
    assert instance.feasible_mask.dtype == np.bool_
    assert instance.feasible_mask.all()

    category_prototype = np.asarray(config.method["category_prototype"], dtype=float)
    effect_by_category = np.asarray(config.method["effect_matrix_by_category"], dtype=float)
    np.testing.assert_allclose(instance.concept_need[:, :5], category_prototype[instance.category_id])
    np.testing.assert_allclose(instance.concept_need[:, 5], instance.cognitive_load)
    np.testing.assert_allclose(instance.effect_matrix[0, :], effect_by_category[:, instance.category_id[0]])
    assert np.all(instance.student_pref >= 0.0)
    assert np.all(instance.student_pref <= 1.0)
    assert np.all(instance.teacher_pref >= 0.0)
    assert np.all(instance.teacher_pref <= 1.0)


def test_assignment_instance_json_npz_roundtrip(tmp_path) -> None:
    instance = build_assignment_instance_from_selection(
        "examples",
        load_selection_csv("examples/summary_ga_own_group1_260417_210154.csv")[0],
    )
    json_path = tmp_path / "instance.json"
    npz_path = tmp_path / "instance.npz"

    save_instance_json(instance, json_path)
    loaded_json = load_instance_json(json_path)
    save_instance_npz(instance, npz_path)
    loaded_npz = load_instance_npz(npz_path)

    for loaded in (loaded_json, loaded_npz):
        assert loaded.instance_name == instance.instance_name
        assert loaded.selected_ids == instance.selected_ids
        np.testing.assert_array_equal(loaded.category_id, instance.category_id)
        np.testing.assert_allclose(loaded.concept_need, instance.concept_need)
        np.testing.assert_allclose(loaded.effect_matrix, instance.effect_matrix)
        np.testing.assert_array_equal(loaded.feasible_mask, instance.feasible_mask)


def test_pre_data_first_pair_builds() -> None:
    pairs = load_pre_data_pairs("pre_data/instance", "pre_data/x")
    assert pairs
    sm_path, record = pairs[0]
    assert sm_path.exists()
    instance = build_assignment_instance_from_selection("pre_data/instance", record)
    assert instance.source_sm.endswith(record.instance_name)
    assert instance.n == int(record.x.sum())
    assert instance.concept_need.shape == (instance.n, 6)
    assert instance.effect_matrix.shape == (instance.n, 11)
    assert instance.feasible_mask.all()
