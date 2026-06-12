"""Tests for PA-MOAP configuration loading and validation."""

from pathlib import Path

import pytest

from pa_moap_rl.configs import ConfigError, load_config


def test_default_config_shapes_and_values() -> None:
    config = load_config()

    method = config.method
    profile = config.profile
    default = config.default

    assert len(method["categories"]) == 6
    assert len(method["methods"]) == 11
    assert len(method["category_prototype"]) == 6
    assert all(len(row) == 5 for row in method["category_prototype"])
    assert len(method["method_attr"]) == 11
    assert all(len(row) == 6 for row in method["method_attr"])
    assert len(method["effect_matrix_by_category"]) == 11
    assert all(len(row) == 6 for row in method["effect_matrix_by_category"])

    assert len(profile["student_profile"]) == 6
    assert len(profile["teacher_profile"]) == 6
    assert all(0.0 <= value <= 1.0 for value in profile["student_profile"])
    assert all(0.0 <= value <= 1.0 for value in profile["teacher_profile"])
    assert len(profile["target_distribution"]) == 11
    assert sum(profile["target_distribution"]) == pytest.approx(1.0)
    assert sum(profile["weights"].values()) == pytest.approx(1.0)

    assert default["preference"]["lambda_q"] + default["preference"]["lambda_r"] == pytest.approx(1.0)
    assert sum(default["preference"]["activity_weights"]) == pytest.approx(1.0)
    assert default["soft_constraints"]["entropy_min"] == 0.55
    assert default["soft_constraints"]["method_cap"] == 0.35


def test_config_loader_rejects_bad_shape(tmp_path: Path) -> None:
    source = Path("pa_moap_rl/configs")
    for name in ("default.yaml", "profile_config.yaml"):
        (tmp_path / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    bad_method = (source / "method_config.yaml").read_text(encoding="utf-8")
    bad_method = bad_method.replace("  - [0.90, 0.20, 0.15, 0.20, 0.20]", "  - [0.90, 0.20]")
    (tmp_path / "method_config.yaml").write_text(bad_method, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(tmp_path)
