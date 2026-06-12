"""加载并校验 PA-MOAP 的 YAML 配置文件。

项目的数学参数分散在三个 YAML 中：默认训练/约束参数、教学方法矩阵、学生/
教师画像与目标分布。这里统一读取并做 shape、取值范围和归一化校验，避免
后续构造实例或训练时才暴露配置错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """当配置文件缺字段、shape 不匹配或数值越界时抛出。"""


@dataclass(frozen=True)
class PAConfig:
    """已经通过校验的 PA-MOAP 配置集合。"""

    default: dict[str, Any]
    method: dict[str, Any]
    profile: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")
    return data


def _shape(matrix: Any) -> tuple[int, ...]:
    if not isinstance(matrix, list):
        raise ConfigError("Expected a list matrix.")
    if not matrix:
        return (0,)
    if all(not isinstance(row, list) for row in matrix):
        return (len(matrix),)
    if not all(isinstance(row, list) for row in matrix):
        raise ConfigError("Matrix rows must be consistently nested lists.")
    row_lengths = {len(row) for row in matrix}
    if len(row_lengths) != 1:
        raise ConfigError("Matrix rows must have equal length.")
    return (len(matrix), row_lengths.pop())


def _ensure_range(name: str, values: Any) -> None:
    if isinstance(values, dict):
        iterable = values.values()
    elif isinstance(values, list):
        iterable = values
    else:
        iterable = [values]
    for value in iterable:
        if isinstance(value, list):
            _ensure_range(name, value)
            continue
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{name} contains a non-numeric value: {value!r}")
        if value < 0.0 or value > 1.0:
            raise ConfigError(f"{name} value {value!r} is outside [0, 1].")


def validate_config(config: PAConfig) -> None:
    """校验矩阵 shape、向量长度、数值范围和概率分布归一化。"""

    method = config.method
    profile = config.profile
    default = config.default

    categories = method.get("categories")
    methods = method.get("methods")
    if not isinstance(categories, list) or not categories:
        raise ConfigError("method_config.yaml must define non-empty categories.")
    if not isinstance(methods, list) or not methods:
        raise ConfigError("method_config.yaml must define non-empty methods.")

    # 当前 v1 数学定义固定为 6 类知识点和 11 种教学方法。
    k = len(categories)
    m = len(methods)
    if k != 6:
        raise ConfigError(f"Expected 6 categories, got {k}.")
    if m != 11:
        raise ConfigError(f"Expected 11 methods, got {m}.")

    # method_config 中的矩阵是后续构造 concept_need/effect_matrix 的依据。
    expected_shapes = {
        "category_prototype": (k, 5),
        "method_attr": (m, 6),
        "effect_matrix_by_category": (m, k),
    }
    for key, expected in expected_shapes.items():
        actual = _shape(method.get(key))
        if actual != expected:
            raise ConfigError(f"{key} shape must be {expected}, got {actual}.")
        _ensure_range(key, method[key])

    for key in ("student_profile", "teacher_profile"):
        actual = _shape(profile.get(key))
        if actual != (6,):
            raise ConfigError(f"{key} shape must be (6,), got {actual}.")
        _ensure_range(key, profile[key])

    # 目标分布用于全局分布项 F_G，因此必须和方法数 M 对齐并归一化。
    target_distribution = profile.get("target_distribution")
    actual = _shape(target_distribution)
    if actual != (m,):
        raise ConfigError(f"target_distribution shape must be ({m},), got {actual}.")
    _ensure_range("target_distribution", target_distribution)
    if abs(sum(target_distribution) - 1.0) > 1.0e-6:
        raise ConfigError("target_distribution must sum to 1.")

    # 五个目标项权重共同构成标量目标 J，要求键完整且总和为 1。
    weights = profile.get("weights")
    if not isinstance(weights, dict):
        raise ConfigError("weights must be a mapping.")
    expected_weight_keys = {"effect", "student", "teacher", "global", "soft"}
    if set(weights) != expected_weight_keys:
        raise ConfigError(f"weights keys must be {sorted(expected_weight_keys)}.")
    _ensure_range("weights", weights)
    if abs(sum(weights.values()) - 1.0) > 1.0e-9:
        raise ConfigError("weights must sum to 1.")

    preference = default.get("preference")
    if not isinstance(preference, dict):
        raise ConfigError("default.yaml must define preference.")
    if abs(preference.get("lambda_q", 0.0) + preference.get("lambda_r", 0.0) - 1.0) > 1.0e-9:
        raise ConfigError("lambda_q + lambda_r must equal 1.")
    activity_weights = preference.get("activity_weights")
    if _shape(activity_weights) != (5,):
        raise ConfigError("activity_weights shape must be (5,).")
    _ensure_range("activity_weights", activity_weights)
    if abs(sum(activity_weights) - 1.0) > 1.0e-9:
        raise ConfigError("activity_weights must sum to 1.")

    soft_constraints = default.get("soft_constraints")
    if not isinstance(soft_constraints, dict):
        raise ConfigError("default.yaml must define soft_constraints.")
    _ensure_range(
        "soft_constraints",
        [
            soft_constraints.get("entropy_min"),
            soft_constraints.get("method_cap"),
            soft_constraints.get("lambda_div"),
            soft_constraints.get("lambda_cap"),
        ],
    )


def load_config(config_dir: str | Path | None = None) -> PAConfig:
    """读取并校验三个 YAML 配置文件。"""

    base = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parent
    default = _load_yaml(base / "default.yaml")
    method = _load_yaml(base / "method_config.yaml")
    profile = _load_yaml(base / "profile_config.yaml")
    config = PAConfig(default=default, method=method, profile=profile)
    validate_config(config)
    return config
