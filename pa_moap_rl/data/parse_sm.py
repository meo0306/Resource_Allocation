"""解析标准化 `.sm` 资源分配算例。

`.sm` 是上游资源组合阶段使用的原始算例格式。本项目的内容二阶段只取其中
两类字段：`type` 映射为知识点类别 `category_id`，`q_2` 映射为认知负荷
`cognitive_load`。先修关系、容量和其他 p/q 字段会被解析并保留，但当前求解
教学方法分配时不参与约束。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.instance_schema import SMInstance, SMNode


class SMParseError(ValueError):
    """当 `.sm` 文件缺少必要区块或字段格式不匹配时抛出。"""


def _split_key_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _section_lines(lines: list[str], start_marker: str) -> list[str]:
    """截取某个 `.sm` 标记区块下的正文行。"""

    for start, line in enumerate(lines):
        if line.strip() == start_marker:
            break
    else:
        raise SMParseError(f"Missing section {start_marker!r}.")

    content: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("************************************************************************"):
            if content:
                break
            continue
        content.append(line.rstrip("\n"))
    return content


def parse_sm(path: str | Path, category_to_id: dict[str, int] | None = None) -> SMInstance:
    """解析一个标准化 `.sm` 文件。

    `NODE ATTRIBUTES` 区块必须包含：
    ``nodnr. <type> <p_1> <p_2> <p_3> <q_1> <q_2> <q_3>``.

    内容二使用 `type -> category_id`、`q_2 -> cognitive_load`。如果调用者
    不传 `category_to_id`，就使用默认 YAML 配置中的类别顺序。
    """

    sm_path = Path(path)
    if category_to_id is None:
        config = load_config()
        category_to_id = {name: idx for idx, name in enumerate(config.method["categories"])}

    lines = sm_path.read_text(encoding="utf-8").splitlines()
    # 头部元数据使用 `key: value` 形式，解析后用于校验节点数和记录随机种子。
    metadata: dict[str, str] = {}
    for line in lines:
        pair = _split_key_value(line)
        if pair is not None:
            metadata[pair[0]] = pair[1]

    instance_name = metadata.get("Instance name", sm_path.stem)
    description = metadata.get("Description")
    try:
        num_nodes_total = int(metadata["Nodes (incl. supersource/sink)"])
    except KeyError as exc:
        raise SMParseError("Missing 'Nodes (incl. supersource/sink)' metadata.") from exc
    random_seed = int(metadata["Random seed"]) if metadata.get("Random seed") else None

    # 先修关系当前不参与内容二求解，但保留在 SMInstance 中便于追溯。
    precedence_lines = _section_lines(lines, "PRECEDENCE RELATIONS:")
    precedence: dict[int, list[int]] = {}
    for line in precedence_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("nodnr."):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise SMParseError(f"Malformed precedence row: {line!r}")
        node_id = int(parts[0])
        successor_count = int(parts[1])
        successors = [int(value) for value in parts[2:]]
        if len(successors) != successor_count:
            raise SMParseError(f"Successor count mismatch for node {node_id}.")
        precedence[node_id] = successors

    # 节点属性是内容二构造 category_id/cognitive_load 的主要来源。
    attribute_lines = _section_lines(lines, "NODE ATTRIBUTES:")
    nodes: list[SMNode] = []
    for line in attribute_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("nodnr."):
            continue
        parts = stripped.split()
        if len(parts) != 8:
            raise SMParseError(f"Malformed node attribute row: {line!r}")
        original_id = int(parts[0])
        category_name = parts[1]
        if category_name not in category_to_id:
            raise SMParseError(f"Unknown category {category_name!r} for node {original_id}.")
        values = [float(value) for value in parts[2:]]
        nodes.append(
            SMNode(
                original_id=original_id,
                category_name=category_name,
                category_id=category_to_id[category_name],
                importance=values[0],
                timeliness=values[1],
                measurability=values[2],
                teaching_time=values[3],
                cognitive_load=values[4],
                external_resource_demand=values[5],
            )
        )

    if len(nodes) != num_nodes_total:
        raise SMParseError(f"Expected {num_nodes_total} nodes, parsed {len(nodes)}.")

    capacity_lines = _section_lines(lines, "COST CAPACITIES:")
    capacity_rows = [line.strip() for line in capacity_lines if line.strip()]
    if len(capacity_rows) < 2:
        raise SMParseError("COST CAPACITIES must contain header and values rows.")
    capacities = np.asarray([float(value) for value in capacity_rows[1].split()], dtype=np.float64)
    if capacities.shape != (3,):
        raise SMParseError(f"Expected 3 cost capacity values, got {capacities.shape}.")

    # `id_mapping` 将 `.sm` 原始 nodnr 映射到 Python 0-based 本地数组下标。
    id_mapping = {node.original_id: idx for idx, node in enumerate(nodes)}
    return SMInstance(
        instance_name=instance_name,
        source_sm=str(sm_path),
        description=description,
        num_nodes_total=num_nodes_total,
        random_seed=random_seed,
        nodes=nodes,
        precedence=precedence,
        capacities=capacities,
        id_mapping=id_mapping,
    )
