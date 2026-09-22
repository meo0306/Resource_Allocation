"""读取内容一阶段输出的知识点选择结果。

内容一负责从原始知识点集合中选择一个子集，内容二只接收这个子集并为其分配
教学方法。这里最重要的约定是编号转换：CSV/JSON 中的选择向量 `x` 是 0-based
数组，但它的位置 `j-1` 对应 `.sm` 原始节点编号 `nodnr == j`。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from pa_moap_rl.data.instance_schema import SelectionRecord


class SelectionLoadError(ValueError):
    """当内容一选择结果无法解析或不满足二进制向量约定时抛出。"""


_INT_RE = re.compile(r"-?\d+")


def parse_x_vector(value: str | list[int] | list[float] | np.ndarray) -> np.ndarray:
    """解析内容一的二进制选择向量。

    CSV 中常见格式是 NumPy 风格字符串，例如 ``"[1 0 1\n 0]"``；JSON 可直接
    给普通 list。解析后统一返回一维 `int64` 数组。
    """

    if isinstance(value, np.ndarray):
        vector = value.astype(np.int64)
    elif isinstance(value, list):
        vector = np.asarray(value, dtype=np.int64)
    elif isinstance(value, str):
        numbers = [int(token) for token in _INT_RE.findall(value)]
        if not numbers:
            raise SelectionLoadError("Selection vector string contains no integers.")
        vector = np.asarray(numbers, dtype=np.int64)
    else:
        raise SelectionLoadError(f"Unsupported selection vector type: {type(value)!r}")

    if vector.ndim != 1:
        raise SelectionLoadError("Selection vector must be one-dimensional.")
    unique = set(vector.tolist())
    if not unique.issubset({0, 1}):
        raise SelectionLoadError(f"Selection vector must be binary, got values {sorted(unique)}.")
    return vector


def selected_ids_from_x(x: np.ndarray) -> list[int]:
    """把 0-based 选择向量位置转换为 `.sm` 原始 `nodnr` 编号。"""

    return [idx + 1 for idx, selected in enumerate(x.tolist()) if int(selected) == 1]


def load_selection_json(path: str | Path, instance_name: str | None = None) -> SelectionRecord:
    """读取 JSON 选择文件，支持 `selected_ids` 或 `x_star` 两种输入。"""

    json_path = Path(path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SelectionLoadError("Selection JSON must contain an object.")
    name = instance_name or data.get("instance_name") or json_path.stem

    if "selected_ids" in data:
        # 直接给原始 nodnr 时，反向构造一个等价的 x 向量，方便后续统一处理。
        selected_ids = [int(value) for value in data["selected_ids"]]
        if not selected_ids:
            raise SelectionLoadError("selected_ids must be non-empty.")
        max_id = max(selected_ids)
        x = np.zeros(max_id, dtype=np.int64)
        for original_id in selected_ids:
            if original_id < 1:
                raise SelectionLoadError("selected_ids must use original .sm ids starting at 1.")
            x[original_id - 1] = 1
    elif "x_star" in data:
        x = parse_x_vector(data["x_star"])
        selected_ids = selected_ids_from_x(x)
    else:
        raise SelectionLoadError("Selection JSON must contain selected_ids or x_star.")

    return SelectionRecord(
        instance_name=name,
        selected_ids=selected_ids,
        x=x,
        source_csv=None,
        metadata={key: value for key, value in data.items() if key not in {"selected_ids", "x_star"}},
    )


def load_selection_csv(path: str | Path, row_type: str = "instance") -> list[SelectionRecord]:
    """读取内容一 summary CSV，并筛选指定 `row_type` 的实例行。"""

    csv_path = Path(path)
    records: list[SelectionRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"row_type", "instance_name", "x"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SelectionLoadError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            if row.get("row_type") != row_type:
                continue
            x = parse_x_vector(row["x"])
            records.append(
                SelectionRecord(
                    instance_name=row["instance_name"],
                    selected_ids=selected_ids_from_x(x),
                    x=x,
                    row_type=row.get("row_type", row_type),
                    source_csv=str(csv_path),
                    instance_dir=row.get("instance_dir") or None,
                    metadata={key: value for key, value in row.items() if key not in {"x"}},
                )
            )
    if not records:
        raise SelectionLoadError(f"No {row_type!r} rows found in {csv_path}.")
    return records


def find_instance_file(instance_root: str | Path, instance_name: str) -> Path:
    """在算例根目录下按实例名查找对应 `.sm` 文件。"""

    root = Path(instance_root)
    candidates = list(root.rglob(instance_name))
    if not candidates and not instance_name.endswith(".sm"):
        candidates = list(root.rglob(f"{instance_name}.sm"))
    if not candidates:
        raise SelectionLoadError(f"Could not find {instance_name!r} under {root}.")
    if len(candidates) > 1:
        normalized = [path for path in candidates if path.name == instance_name]
        if len(normalized) == 1:
            return normalized[0]
        raise SelectionLoadError(f"Multiple instance files named {instance_name!r} under {root}.")
    return candidates[0]


def _load_pre_data_pairs_flat(instance_root: str | Path, selection_root: str | Path) -> list[tuple[Path, SelectionRecord]]:
    """批量读取 `pre_data` 中的 CSV 记录，并与同名 `.sm` 文件配对。"""

    pairs: list[tuple[Path, SelectionRecord]] = []
    for csv_path in sorted(Path(selection_root).rglob("*.csv")):
        for record in load_selection_csv(csv_path):
            pairs.append((find_instance_file(instance_root, record.instance_name), record))
    return pairs

# This topology-aware definition intentionally supersedes the legacy flat-layout
# loader above while retaining backward compatibility for old datasets.
def load_pre_data_pairs(instance_root: str | Path, selection_root: str | Path) -> list[tuple[Path, SelectionRecord]]:
    '''Pair selections within their topology directory to disambiguate names.'''

    pairs: list[tuple[Path, SelectionRecord]] = []
    instance_base = Path(instance_root)
    selection_base = Path(selection_root)
    for csv_path in sorted(selection_base.rglob('*.csv')):
        relative = csv_path.relative_to(selection_base)
        topology = relative.parts[0] if len(relative.parts) > 1 else None
        scoped_root = instance_base / topology if topology and (instance_base / topology).is_dir() else instance_base
        for record in load_selection_csv(csv_path):
            if scoped_root != instance_base:
                metadata = dict(record.metadata or {})
                metadata['input_topology'] = topology
                metadata['topology'] = 'bottleneck' if topology == 'manual_bottleneck' else topology
                record = SelectionRecord(
                    instance_name=record.instance_name,
                    selected_ids=record.selected_ids,
                    x=record.x,
                    row_type=record.row_type,
                    source_csv=record.source_csv,
                    instance_dir=record.instance_dir,
                    metadata=metadata,
                )
            pairs.append((find_instance_file(scoped_root, record.instance_name), record))
    return pairs
