"""PA-MOAP 目标函数与单点替换增量评分。

本模块实现内容二的标量目标：
`J = w_E F_E + w_S F_S + w_T F_T + w_G F_G - w_soft V_soft`。
所有局部项都使用均值，避免实例规模 N 变大时目标值自然放大；全局项通过
教学方法使用分布 `pi` 与目标分布 `rho` 的距离、熵和使用上限计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_ENTROPY_MIN = 0.55
DEFAULT_METHOD_CAP = 0.35
DEFAULT_LAMBDA_DIV = 1.0
DEFAULT_LAMBDA_CAP = 1.0
DEFAULT_EPSILON = 1.0e-8


@dataclass(frozen=True)
class ScoreBreakdown:
    """一个完整 assignment 的目标函数分解结果。"""

    total_score: float
    effect_score: float
    student_pref_score: float
    teacher_pref_score: float
    global_score: float
    entropy: float
    diversity_violation: float
    cap_violation: float
    soft_penalty: float
    method_distribution: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        """返回适合写入 JSON/CSV 的评分分解。"""

        return {
            "total_score": self.total_score,
            "effect_score": self.effect_score,
            "student_pref_score": self.student_pref_score,
            "teacher_pref_score": self.teacher_pref_score,
            "global_score": self.global_score,
            "entropy": self.entropy,
            "diversity_violation": self.diversity_violation,
            "cap_violation": self.cap_violation,
            "soft_penalty": self.soft_penalty,
            "method_distribution": self.method_distribution.tolist(),
        }


def _as_assignment(assignment: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"assignment must be a 1D array, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("assignment must be non-empty.")
    if np.any(values < 0):
        raise ValueError("assignment contains negative method ids.")
    return values


def _as_matrix(name: str, values: np.ndarray, n: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {matrix.shape}.")
    if matrix.shape[0] != n:
        raise ValueError(f"{name} first dimension must equal assignment length {n}, got {matrix.shape[0]}.")
    return matrix


def _resolve_m(assignment: np.ndarray, M: int | None, matrix: np.ndarray | None = None) -> int:
    if M is None:
        if matrix is not None:
            M = int(matrix.shape[1])
        else:
            M = int(assignment.max()) + 1
    M = int(M)
    if M <= 0:
        raise ValueError("M must be positive.")
    if assignment.size and int(assignment.max()) >= M:
        raise ValueError(f"assignment contains method id {int(assignment.max())}, but M={M}.")
    return M


def _selected_values(name: str, assignment: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    matrix = _as_matrix(name, matrix, assignment.size)
    if int(assignment.max()) >= matrix.shape[1]:
        raise ValueError(f"assignment contains a method id outside {name} columns.")
    return matrix[np.arange(assignment.size), assignment]


def _as_distribution(
    assignment_or_distribution: np.ndarray | list[float] | tuple[float, ...],
    M: int | None = None,
) -> np.ndarray:
    values = np.asarray(assignment_or_distribution)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D assignment or distribution, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("assignment or distribution must be non-empty.")

    if np.issubdtype(values.dtype, np.integer):
        return method_distribution(values.astype(np.int64), M)

    distribution = values.astype(np.float64)
    if np.any(distribution < -1.0e-12):
        raise ValueError("distribution contains negative values.")
    total = float(distribution.sum())
    if total <= 0.0:
        raise ValueError("distribution must have positive mass.")
    if abs(total - 1.0) > 1.0e-8:
        distribution = distribution / total
    if M is not None and distribution.shape[0] != int(M):
        raise ValueError(f"distribution length must equal M={M}, got {distribution.shape[0]}.")
    return distribution


def method_distribution(assignment: np.ndarray | list[int] | tuple[int, ...], M: int | None = None) -> np.ndarray:
    """计算一维 assignment 的教学方法使用比例 `pi_m`。"""

    values = _as_assignment(assignment)
    method_count = _resolve_m(values, M)
    counts = np.bincount(values, minlength=method_count).astype(np.float64)
    return counts / float(values.size)


def F_E(assignment: np.ndarray | list[int] | tuple[int, ...], effect_matrix: np.ndarray) -> float:
    """理论教学效果匹配均值 `F_E`。"""

    values = _as_assignment(assignment)
    return float(np.mean(_selected_values("effect_matrix", values, effect_matrix)))


def F_S(assignment: np.ndarray | list[int] | tuple[int, ...], student_pref: np.ndarray) -> float:
    """学生偏好满足度均值 `F_S`。"""

    values = _as_assignment(assignment)
    return float(np.mean(_selected_values("student_pref", values, student_pref)))


def F_T(assignment: np.ndarray | list[int] | tuple[int, ...], teacher_pref: np.ndarray) -> float:
    """教师偏好满足度均值 `F_T`。"""

    values = _as_assignment(assignment)
    return float(np.mean(_selected_values("teacher_pref", values, teacher_pref)))


def F_G(
    assignment: np.ndarray | list[int] | tuple[int, ...],
    target_distribution: np.ndarray | list[float] | tuple[float, ...],
    M: int | None = None,
) -> float:
    """全局方法分布得分 `F_G = 1 - 0.5 * L1(pi, rho)`。"""

    values = _as_assignment(assignment)
    target = np.asarray(target_distribution, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError(f"target_distribution must be 1D, got shape {target.shape}.")
    method_count = _resolve_m(values, M if M is not None else target.shape[0])
    if target.shape[0] != method_count:
        raise ValueError(f"target_distribution length must be {method_count}, got {target.shape[0]}.")
    if np.any(target < -1.0e-12):
        raise ValueError("target_distribution contains negative values.")
    target_sum = float(target.sum())
    if abs(target_sum - 1.0) > 1.0e-8:
        raise ValueError("target_distribution must sum to 1.")

    pi = method_distribution(values, method_count)
    return float(1.0 - 0.5 * np.sum(np.abs(pi - target)))


def H(
    assignment_or_distribution: np.ndarray | list[float] | tuple[float, ...],
    M: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """当前方法分布的归一化熵，取值约束到 `[0, 1]`。"""

    pi = _as_distribution(assignment_or_distribution, M)
    method_count = int(pi.shape[0])
    if method_count <= 1:
        return 0.0
    entropy = -float(np.sum(pi * np.log(pi + float(epsilon)))) / float(np.log(method_count))
    return float(np.clip(entropy, 0.0, 1.0))


def V_div(
    assignment_or_distribution: np.ndarray | list[float] | tuple[float, ...],
    M: int | None = None,
    H_min: float = DEFAULT_ENTROPY_MIN,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """多样性软约束违反量 `max(0, H_min - H)^2`。"""

    gap = max(0.0, float(H_min) - H(assignment_or_distribution, M=M, epsilon=epsilon))
    return float(gap * gap)


def V_cap(
    assignment_or_distribution: np.ndarray | list[float] | tuple[float, ...],
    M: int | None = None,
    pi_cap: float | np.ndarray = DEFAULT_METHOD_CAP,
) -> float:
    """每种方法使用比例上限的平方违反量。"""

    pi = _as_distribution(assignment_or_distribution, M)
    cap = np.asarray(pi_cap, dtype=np.float64)
    if cap.ndim == 0:
        cap = np.full(pi.shape, float(cap), dtype=np.float64)
    if cap.shape != pi.shape:
        raise ValueError(f"pi_cap shape must be scalar or {pi.shape}, got {cap.shape}.")
    if np.any(cap <= 0.0) or np.any(cap > 1.0):
        raise ValueError("pi_cap values must be in (0, 1].")
    violation = np.maximum(0.0, pi - cap)
    return float(np.sum(violation * violation))


def V_soft(
    assignment_or_distribution: np.ndarray | list[float] | tuple[float, ...],
    M: int | None = None,
    H_min: float = DEFAULT_ENTROPY_MIN,
    pi_cap: float | np.ndarray = DEFAULT_METHOD_CAP,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
    lambda_cap: float = DEFAULT_LAMBDA_CAP,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """软约束总惩罚 `lambda_div * V_div + lambda_cap * V_cap`。"""

    return float(
        float(lambda_div) * V_div(assignment_or_distribution, M=M, H_min=H_min, epsilon=epsilon)
        + float(lambda_cap) * V_cap(assignment_or_distribution, M=M, pi_cap=pi_cap)
    )


def score_assignment(
    assignment: np.ndarray | list[int] | tuple[int, ...],
    *,
    effect_matrix: np.ndarray | None = None,
    student_pref: np.ndarray | None = None,
    teacher_pref: np.ndarray | None = None,
    target_distribution: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
    instance: Any | None = None,
    H_min: float = DEFAULT_ENTROPY_MIN,
    pi_cap: float | np.ndarray = DEFAULT_METHOD_CAP,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
    lambda_cap: float = DEFAULT_LAMBDA_CAP,
    epsilon: float = DEFAULT_EPSILON,
) -> ScoreBreakdown:
    """计算一个 assignment 的所有 PA-MOAP 目标分项。"""

    if instance is not None:
        # 常规调用直接传 AssignmentInstance，避免在 solver 中反复拆矩阵参数。
        effect_matrix = instance.effect_matrix
        student_pref = instance.student_pref
        teacher_pref = instance.teacher_pref
        target_distribution = instance.target_distribution
        weights = instance.weights

    missing = [
        name
        for name, value in (
            ("effect_matrix", effect_matrix),
            ("student_pref", student_pref),
            ("teacher_pref", teacher_pref),
            ("target_distribution", target_distribution),
            ("weights", weights),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing scoring inputs: {', '.join(missing)}.")

    values = _as_assignment(assignment)
    method_count = _resolve_m(values, None, np.asarray(effect_matrix))
    pi = method_distribution(values, method_count)
    effect_score = F_E(values, np.asarray(effect_matrix))
    student_score = F_S(values, np.asarray(student_pref))
    teacher_score = F_T(values, np.asarray(teacher_pref))
    global_score = F_G(values, np.asarray(target_distribution), method_count)
    entropy = H(pi, epsilon=epsilon)
    diversity_violation = V_div(pi, H_min=H_min, epsilon=epsilon)
    cap_violation = V_cap(pi, pi_cap=pi_cap)
    soft_penalty = float(lambda_div) * diversity_violation + float(lambda_cap) * cap_violation

    # 权重键必须完整，防止某一项被静默遗漏或写错名字。
    required_weights = {"effect", "student", "teacher", "global", "soft"}
    if set(weights) != required_weights:
        raise ValueError(f"weights keys must be {sorted(required_weights)}.")
    total = (
        float(weights["effect"]) * effect_score
        + float(weights["student"]) * student_score
        + float(weights["teacher"]) * teacher_score
        + float(weights["global"]) * global_score
        - float(weights["soft"]) * soft_penalty
    )

    return ScoreBreakdown(
        total_score=float(total),
        effect_score=effect_score,
        student_pref_score=student_score,
        teacher_pref_score=teacher_score,
        global_score=global_score,
        entropy=entropy,
        diversity_violation=diversity_violation,
        cap_violation=cap_violation,
        soft_penalty=soft_penalty,
        method_distribution=pi,
    )


def J(assignment: np.ndarray | list[int] | tuple[int, ...], **kwargs: Any) -> float:
    """返回标量化 PA-MOAP 目标值 `J`。"""

    return score_assignment(assignment, **kwargs).total_score


def delta_score(
    assignment: np.ndarray | list[int] | tuple[int, ...],
    node_index: int,
    new_method: int,
    **score_kwargs: Any,
) -> float:
    """返回单点替换的收益 `J(y with y_i=new_method) - J(y)`。"""

    values = _as_assignment(assignment)
    node_index = int(node_index)
    new_method = int(new_method)
    if node_index < 0 or node_index >= values.size:
        raise IndexError(f"node_index {node_index} outside assignment length {values.size}.")
    if new_method < 0:
        raise ValueError("new_method must be non-negative.")

    current = score_assignment(values, **score_kwargs).total_score
    updated = values.copy()
    updated[node_index] = new_method
    new_value = score_assignment(updated, **score_kwargs).total_score
    return float(new_value - current)


def replacement_delta(*args: Any, **kwargs: Any) -> float:
    """`delta_score` 的别名，供局部搜索代码使用。"""

    return delta_score(*args, **kwargs)
