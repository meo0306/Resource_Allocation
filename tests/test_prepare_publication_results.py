from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_moap_rl.experiments.prepare_publication_results import (
    METHOD_ORDER,
    _bootstrap_mean_ci,
    _holm_adjust,
    _two_sided_sign_test,
    _validate_test_pairing,
)


def _paired_frame() -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for index in range(3):
            rows.append(
                {
                    'instance_path': f'instance-{index}',
                    'scenario_id': f'scenario-{index}',
                    'solver_name': method,
                    'total_score': 0.5 + index * 0.01,
                    'runtime': 0.1,
                    'topology': 'uniform',
                    'hard_feasible_rate': 1.0,
                    'mask_violation_count': 0,
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    first = _bootstrap_mean_ci(
        values, samples=500, rng=np.random.default_rng(42)
    )
    second = _bootstrap_mean_ci(
        values, samples=500, rng=np.random.default_rng(42)
    )
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_exact_sign_test_and_holm_adjustment() -> None:
    assert _two_sided_sign_test([1.0, 2.0, 3.0]) == pytest.approx(0.25)
    assert _two_sided_sign_test([0.0, 0.0]) == 1.0
    assert _holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_validate_test_pairing_rejects_missing_pair() -> None:
    frame = _paired_frame()
    assert _validate_test_pairing(frame, 'synthetic') == 3

    missing = frame.drop(
        frame.loc[
            (frame['solver_name'] == METHOD_ORDER[-1])
            & (frame['scenario_id'] == 'scenario-2')
        ].index
    )
    with pytest.raises(ValueError, match='paired test bank'):
        _validate_test_pairing(missing, 'synthetic')
