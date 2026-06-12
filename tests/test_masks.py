"""Tests for feasible and action masks."""

import numpy as np
import pytest

from pa_moap_rl.utils.masks import (
    action_mask,
    assert_action_legal,
    build_action_mask,
    is_action_legal,
    valid_action_indices,
    validate_feasible_mask,
)


def test_single_instance_action_mask_excludes_current_and_infeasible_methods() -> None:
    assignment = np.asarray([0, 1, 2])
    feasible = np.asarray(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ]
    )

    mask = build_action_mask(assignment, feasible)

    expected = np.asarray(
        [
            [False, True, False],
            [False, False, True],
            [True, False, False],
        ]
    )
    np.testing.assert_array_equal(mask, expected)
    assert is_action_legal(mask, 0, 1)
    assert not is_action_legal(mask, 0, 0)
    assert valid_action_indices(mask).shape == (3, 2)
    assert_action_legal(mask, 2, 0)
    with pytest.raises(ValueError):
        assert_action_legal(mask, 2, 2)


def test_batch_action_mask_respects_padding_node_mask() -> None:
    assignment = np.asarray([[0, 1, 0], [2, 1, 0]])
    feasible = np.ones((2, 3, 3), dtype=bool)
    feasible[1, 0, 0] = False
    node_mask = np.asarray([[True, True, False], [True, False, False]])

    mask = action_mask(assignment, feasible, node_mask=node_mask)

    assert mask.shape == (2, 3, 3)
    assert not mask[0, 0, 0]
    assert mask[0, 0, 1]
    assert not mask[0, 2].any()
    assert not mask[1, 0, 2]
    assert not mask[1, 0, 0]
    assert mask[1, 0, 1]
    assert not mask[1, 1].any()
    assert not mask[1, 2].any()


def test_validate_feasible_mask_rejects_empty_rows_and_bad_shapes() -> None:
    with pytest.raises(ValueError):
        validate_feasible_mask(np.asarray([[True, False], [False, False]]))
    with pytest.raises(ValueError):
        validate_feasible_mask(np.ones((2, 2, 1), dtype=bool))
    with pytest.raises(ValueError):
        build_action_mask(np.asarray([0, 1]), np.ones((3, 2), dtype=bool))
