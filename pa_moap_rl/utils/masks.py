"""静态可行性 mask 与动态动作 mask 的构造工具。

`feasible_mask[i, m]` 表示知识点 i 是否允许使用方法 m，是算例自带的硬约束。
`action_mask[i, m]` 表示当前状态下是否允许执行“把知识点 i 替换为方法 m”的
动作，因此还必须排除当前已经使用的方法；batch 版本还会排除 padding 节点。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_feasible_mask(
    feasible_mask: np.ndarray | list[list[bool]],
    *,
    n: int | None = None,
    m: int | None = None,
    require_any: bool = True,
) -> np.ndarray:
    """校验并返回 bool 类型的静态可行性矩阵。"""

    mask = np.asarray(feasible_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"feasible_mask must be 2D, got shape {mask.shape}.")
    if n is not None and mask.shape[0] != int(n):
        raise ValueError(f"feasible_mask first dimension must be n={int(n)}, got {mask.shape[0]}.")
    if m is not None and mask.shape[1] != int(m):
        raise ValueError(f"feasible_mask second dimension must be m={int(m)}, got {mask.shape[1]}.")
    if require_any and np.any(mask.sum(axis=1) < 1):
        bad = np.flatnonzero(mask.sum(axis=1) < 1)
        raise ValueError(f"every node must have at least one feasible method; invalid rows: {bad[:10].tolist()}.")
    return mask


def feasible_mask_from_instance(instance: Any) -> np.ndarray:
    """从 `AssignmentInstance` 中取出并校验静态 `feasible_mask`。"""

    return validate_feasible_mask(instance.feasible_mask, n=instance.n, m=instance.m)


def _validate_assignment_1d(assignment: np.ndarray, method_count: int) -> np.ndarray:
    values = np.asarray(assignment, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"assignment must be 1D, got shape {values.shape}.")
    if np.any(values < 0) or np.any(values >= method_count):
        raise ValueError("assignment contains method ids outside feasible_mask columns.")
    return values


def _action_mask_single(
    assignment: np.ndarray | list[int],
    feasible_mask: np.ndarray | list[list[bool]],
    node_mask: np.ndarray | list[bool] | None = None,
) -> np.ndarray:
    """构造单实例动态动作 mask。"""

    feasible = validate_feasible_mask(feasible_mask)
    values = _validate_assignment_1d(np.asarray(assignment), feasible.shape[1])
    if values.shape[0] != feasible.shape[0]:
        raise ValueError(f"assignment length {values.shape[0]} must equal feasible_mask rows {feasible.shape[0]}.")

    mask = feasible.copy()
    mask[np.arange(values.size), values] = False
    if node_mask is not None:
        active_nodes = np.asarray(node_mask, dtype=bool)
        if active_nodes.shape != (values.size,):
            raise ValueError(f"node_mask shape must be {(values.size,)}, got {active_nodes.shape}.")
        mask &= active_nodes[:, None]
    return mask


def build_action_mask(
    assignment: np.ndarray | list[int] | list[list[int]],
    feasible_mask: np.ndarray | list[list[bool]] | list[list[list[bool]]],
    node_mask: np.ndarray | list[bool] | list[list[bool]] | None = None,
) -> np.ndarray:
    """构造单实例或 batch 的动态替换动作 mask。

    Single instance:
        ``action_mask[i, m] = feasible_mask[i, m] and m != assignment[i]``

    Batch:
        ``action_mask[b, i, m] = node_mask[b, i] and feasible_mask[b, i, m]
        and m != assignment[b, i]``.
    """

    assignment_array = np.asarray(assignment, dtype=np.int64)
    feasible_array = np.asarray(feasible_mask, dtype=bool)

    if assignment_array.ndim == 1:
        return _action_mask_single(assignment_array, feasible_array, node_mask=node_mask)

    if assignment_array.ndim != 2:
        raise ValueError(f"assignment must be 1D or 2D, got shape {assignment_array.shape}.")
    if feasible_array.ndim != 3:
        raise ValueError(f"batched feasible_mask must be 3D, got shape {feasible_array.shape}.")
    if feasible_array.shape[:2] != assignment_array.shape:
        raise ValueError(
            f"batched feasible_mask leading shape {feasible_array.shape[:2]} must equal "
            f"assignment shape {assignment_array.shape}."
        )

    batch_size, node_count, method_count = feasible_array.shape
    if np.any(assignment_array < 0) or np.any(assignment_array >= method_count):
        raise ValueError("assignment contains method ids outside feasible_mask columns.")

    # 先继承静态可行性，再把“替换成当前方法”的无效动作置 False。
    mask = feasible_array.copy()
    batch_idx = np.arange(batch_size)[:, None]
    node_idx = np.arange(node_count)[None, :]
    mask[batch_idx, node_idx, assignment_array] = False

    if node_mask is not None:
        active_nodes = np.asarray(node_mask, dtype=bool)
        if active_nodes.shape != assignment_array.shape:
            raise ValueError(f"node_mask shape must be {assignment_array.shape}, got {active_nodes.shape}.")
        mask &= active_nodes[:, :, None]
    return mask


def action_mask(*args: Any, **kwargs: Any) -> np.ndarray:
    """`build_action_mask` 的别名。"""

    return build_action_mask(*args, **kwargs)


def valid_action_indices(mask: np.ndarray | list[list[bool]]) -> np.ndarray:
    """返回 action mask 中所有合法动作坐标。"""

    values = np.asarray(mask, dtype=bool)
    if values.ndim not in (2, 3):
        raise ValueError(f"mask must be 2D or 3D, got shape {values.shape}.")
    return np.argwhere(values)


def is_action_legal(mask: np.ndarray | list[list[bool]], node_index: int, method_index: int, batch_index: int | None = None) -> bool:
    """检查一个动作坐标在给定 action mask 下是否合法。"""

    values = np.asarray(mask, dtype=bool)
    node_index = int(node_index)
    method_index = int(method_index)
    if values.ndim == 2:
        if batch_index is not None:
            raise ValueError("batch_index is only valid for a batched mask.")
        return bool(values[node_index, method_index])
    if values.ndim == 3:
        if batch_index is None:
            raise ValueError("batch_index is required for a batched mask.")
        return bool(values[int(batch_index), node_index, method_index])
    raise ValueError(f"mask must be 2D or 3D, got shape {values.shape}.")


def assert_action_legal(
    mask: np.ndarray | list[list[bool]],
    node_index: int,
    method_index: int,
    batch_index: int | None = None,
) -> None:
    """当动作被 mask 掉时抛出清晰错误。"""

    if not is_action_legal(mask, node_index=node_index, method_index=method_index, batch_index=batch_index):
        prefix = f"batch {batch_index}, " if batch_index is not None else ""
        raise ValueError(f"Illegal action: {prefix}node {int(node_index)}, method {int(method_index)}.")
