"""内容二 assignment instance 的数据结构与校验工具。

本模块只定义“数据长什么样”和“哪些 shape/取值必须成立”，不负责解析、
构造或求解。后续环境、solver、模型都依赖这里的 `AssignmentInstance`，
因此这里的校验是整条流水线的第一道防线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class InstanceValidationError(ValueError):
    """当内容二算例不满足预期 schema 时抛出。"""


@dataclass(frozen=True)
class SMNode:
    """从标准 `.sm` 文件 `NODE ATTRIBUTES` 区块解析得到的一个原始节点。"""

    original_id: int
    category_name: str
    category_id: int
    importance: float
    timeliness: float
    measurability: float
    teaching_time: float
    cognitive_load: float
    external_resource_demand: float


@dataclass(frozen=True)
class SMInstance:
    """解析后的 `.sm` 原始算例。

    当前内容二只直接使用 `category_id` 和 `cognitive_load`。其他字段保留下来，
    一是方便追溯原始输入，二是为以后重新接入容量/先修关系等约束留接口。
    """

    instance_name: str
    source_sm: str
    description: str | None
    num_nodes_total: int
    random_seed: int | None
    nodes: list[SMNode]
    precedence: dict[int, list[int]]
    capacities: np.ndarray
    id_mapping: dict[int, int]

    @property
    def category_id(self) -> np.ndarray:
        return np.asarray([node.category_id for node in self.nodes], dtype=np.int64)

    @property
    def cognitive_load(self) -> np.ndarray:
        return np.asarray([node.cognitive_load for node in self.nodes], dtype=np.float64)


@dataclass(frozen=True)
class SelectionRecord:
    """内容一求解器输出中的一条选择记录。

    `x[j-1] == 1` 表示原始 `.sm` 中 `nodnr == j` 的知识点被选中。
    """

    instance_name: str
    selected_ids: list[int]
    x: np.ndarray
    row_type: str = "instance"
    source_csv: str | None = None
    instance_dir: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssignmentInstance:
    """内容二教学方法分配算例。

    核心任务是在 `n` 个已选知识点上，为每个点选择 `m` 种教学方法中的一种。
    这里保存的矩阵都已经按本地节点编号 `0..n-1` 对齐，solver 不再需要理解
    `.sm` 原始节点编号。
    """

    instance_name: str
    source_sm: str
    selected_ids: list[int]
    original_to_local_id: dict[int, int]
    n: int
    m: int
    k: int
    category_names: list[str]
    method_names: list[str]
    category_id: np.ndarray
    cognitive_load: np.ndarray
    concept_need: np.ndarray
    method_attr: np.ndarray
    effect_matrix: np.ndarray
    student_profile: np.ndarray
    teacher_profile: np.ndarray
    student_pref: np.ndarray
    teacher_pref: np.ndarray
    target_distribution: np.ndarray
    feasible_mask: np.ndarray
    weights: dict[str, float]
    metadata: dict[str, Any] | None = None

    @property
    def N(self) -> int:
        return self.n

    @property
    def M(self) -> int:
        return self.m

    @property
    def K(self) -> int:
        return self.k


def validate_assignment_instance(instance: AssignmentInstance) -> None:
    """校验内容二算例的 shape、索引范围和基础数值范围。

    这里刻意做 fail fast：如果实例构造阶段已经出现矩阵维度错位，继续训练
    PPO 或运行 baseline 会得到很难定位的错误。
    """

    n, m, k = instance.n, instance.m, instance.k
    expected = {
        "category_id": (n,),
        "cognitive_load": (n,),
        "concept_need": (n, 6),
        "method_attr": (m, 6),
        "effect_matrix": (n, m),
        "student_profile": (6,),
        "teacher_profile": (6,),
        "student_pref": (n, m),
        "teacher_pref": (n, m),
        "target_distribution": (m,),
        "feasible_mask": (n, m),
    }
    # 所有核心矩阵必须围绕同一组局部节点和方法编号展开。
    for name, shape in expected.items():
        actual = getattr(instance, name).shape
        if actual != shape:
            raise InstanceValidationError(f"{name} shape must be {shape}, got {actual}.")

    if len(instance.category_names) != k:
        raise InstanceValidationError("category_names length must equal k.")
    if len(instance.method_names) != m:
        raise InstanceValidationError("method_names length must equal m.")
    if len(instance.selected_ids) != n:
        raise InstanceValidationError("selected_ids length must equal n.")
    if set(instance.original_to_local_id) != set(instance.selected_ids):
        raise InstanceValidationError("original_to_local_id keys must match selected_ids.")
    if not np.issubdtype(instance.category_id.dtype, np.integer):
        raise InstanceValidationError("category_id must have integer dtype.")
    if np.any(instance.category_id < 0) or np.any(instance.category_id >= k):
        raise InstanceValidationError("category_id contains out-of-range category ids.")
    if not np.issubdtype(instance.feasible_mask.dtype, np.bool_):
        raise InstanceValidationError("feasible_mask must have bool dtype.")
    if np.any(instance.feasible_mask.sum(axis=1) < 1):
        raise InstanceValidationError("each selected node must have at least one feasible method.")
    if abs(float(instance.target_distribution.sum()) - 1.0) > 1.0e-6:
        raise InstanceValidationError("target_distribution must sum to 1.")

    # 当前模型假定所有偏好、画像、效果和需求值都已归一化到 [0, 1]。
    for name in (
        "cognitive_load",
        "concept_need",
        "method_attr",
        "effect_matrix",
        "student_profile",
        "teacher_profile",
        "student_pref",
        "teacher_pref",
        "target_distribution",
    ):
        values = getattr(instance, name)
        if np.any(values < -1.0e-12) or np.any(values > 1.0 + 1.0e-12):
            raise InstanceValidationError(f"{name} contains values outside [0, 1].")
