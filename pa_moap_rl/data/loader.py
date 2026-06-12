"""内容二算例的 JSON/NPZ 读写工具。

`AssignmentInstance` 内部大量使用 NumPy 数组；保存为 JSON 时需要转成普通
list，加载时再恢复 dtype 和 shape，并重新运行 schema 校验。这样可以保证
手工编辑或跨脚本传递后的算例仍然满足统一接口。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance, validate_assignment_instance


# 这些字段在 dataclass 中是 NumPy 数组，序列化时需要集中处理。
_ARRAY_FIELDS = {
    "category_id",
    "cognitive_load",
    "concept_need",
    "method_attr",
    "effect_matrix",
    "student_profile",
    "teacher_profile",
    "student_pref",
    "teacher_pref",
    "target_distribution",
    "feasible_mask",
}


def assignment_instance_to_dict(instance: AssignmentInstance) -> dict[str, Any]:
    """将 `AssignmentInstance` 转为 JSON 兼容字典。"""

    data: dict[str, Any] = {
        "instance_name": instance.instance_name,
        "source_sm": instance.source_sm,
        "selected_ids": list(instance.selected_ids),
        "original_to_local_id": {str(key): int(value) for key, value in instance.original_to_local_id.items()},
        "N": instance.n,
        "M": instance.m,
        "K": instance.k,
        "category_names": list(instance.category_names),
        "method_names": list(instance.method_names),
        "weights": dict(instance.weights),
        "metadata": instance.metadata or {},
    }
    # feasible_mask 在 JSON 中保存为 0/1，更便于肉眼检查。
    for field in _ARRAY_FIELDS:
        value = getattr(instance, field)
        data[field] = value.astype(int).tolist() if field == "feasible_mask" else value.tolist()
    return data


def assignment_instance_from_dict(data: dict[str, Any]) -> AssignmentInstance:
    """从普通字典恢复 `AssignmentInstance`，并立即校验 schema。"""

    instance = AssignmentInstance(
        instance_name=data["instance_name"],
        source_sm=data["source_sm"],
        selected_ids=[int(value) for value in data["selected_ids"]],
        original_to_local_id={int(key): int(value) for key, value in data["original_to_local_id"].items()},
        n=int(data.get("N", data.get("n"))),
        m=int(data.get("M", data.get("m"))),
        k=int(data.get("K", data.get("k"))),
        category_names=list(data["category_names"]),
        method_names=list(data["method_names"]),
        category_id=np.asarray(data["category_id"], dtype=np.int64),
        cognitive_load=np.asarray(data["cognitive_load"], dtype=np.float64),
        concept_need=np.asarray(data["concept_need"], dtype=np.float64),
        method_attr=np.asarray(data["method_attr"], dtype=np.float64),
        effect_matrix=np.asarray(data["effect_matrix"], dtype=np.float64),
        student_profile=np.asarray(data["student_profile"], dtype=np.float64),
        teacher_profile=np.asarray(data["teacher_profile"], dtype=np.float64),
        student_pref=np.asarray(data["student_pref"], dtype=np.float64),
        teacher_pref=np.asarray(data["teacher_pref"], dtype=np.float64),
        target_distribution=np.asarray(data["target_distribution"], dtype=np.float64),
        feasible_mask=np.asarray(data["feasible_mask"], dtype=bool),
        weights={key: float(value) for key, value in data["weights"].items()},
        metadata=data.get("metadata") or None,
    )
    validate_assignment_instance(instance)
    return instance


def save_instance_json(instance: AssignmentInstance, path: str | Path) -> None:
    """以 UTF-8 JSON 保存内容二算例。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(assignment_instance_to_dict(instance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_instance_json(path: str | Path) -> AssignmentInstance:
    """从 UTF-8 JSON 加载内容二算例。"""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Assignment instance JSON must contain an object.")
    return assignment_instance_from_dict(data)


def save_instance_npz(instance: AssignmentInstance, path: str | Path) -> None:
    """以压缩 NPZ 保存数组字段，并把标量元数据打包进 JSON 字段。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scalar_data = assignment_instance_to_dict(instance)
    arrays = {field: getattr(instance, field) for field in _ARRAY_FIELDS}
    arrays["json_metadata"] = np.asarray(json.dumps({key: value for key, value in scalar_data.items() if key not in _ARRAY_FIELDS}, ensure_ascii=False))
    np.savez_compressed(output, **arrays)


def load_instance_npz(path: str | Path) -> AssignmentInstance:
    """加载 `save_instance_npz` 写出的压缩算例。"""

    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["json_metadata"]))
        for field in _ARRAY_FIELDS:
            metadata[field] = data[field].tolist()
    return assignment_instance_from_dict(metadata)
