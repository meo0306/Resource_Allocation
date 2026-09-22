"""把 `pre_data` 中的 `.sm` 与内容一 CSV 批量转换为内容二 JSON 算例。

这是数据预处理命令行入口。它只生成可重复构造的中间产物，默认输出到
`instance/` 或调用者指定目录；这些批量 JSON 与 manifest 属于实验产物，
不应直接作为核心源码提交。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from pa_moap_rl.configs import load_config
from pa_moap_rl.data.build_assignment_instance import build_assignment_instance_from_files
from pa_moap_rl.data.loader import save_instance_json
from pa_moap_rl.data.selection_loader import load_pre_data_pairs


@dataclass(frozen=True)
class ConvertedInstance:
    """manifest 中的一行转换结果摘要。"""

    group: str
    instance_name: str
    source_sm: str
    source_csv: str
    output_json: str
    n: int
    m: int
    k: int
    selected_count: int


def _group_name_from_sm_path(sm_path: Path) -> str:
    return sm_path.parent.name


def convert_pre_data_flat(
    instance_root: str | Path,
    selection_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> list[ConvertedInstance]:
    """批量转换所有匹配到的 `.sm + 选择记录`，并按 group 写出 JSON。"""

    instance_root = Path(instance_root)
    selection_root = Path(selection_root)
    output_root = Path(output_root)
    config = load_config()
    # `pairs` 已经完成同名 `.sm` 和内容一 CSV 行的匹配。
    pairs = load_pre_data_pairs(instance_root, selection_root)
    converted: list[ConvertedInstance] = []

    for sm_path, record in tqdm(pairs, desc="Converting assignment instances", unit="instance"):
        group = _group_name_from_sm_path(sm_path)
        group_dir = output_root / group
        output_path = group_dir / f"{Path(record.instance_name).stem}.assignment.json"
        # 默认跳过已存在文件，避免无意覆盖已有实验输入。
        if output_path.exists() and not overwrite:
            continue

        metadata = {
            "source_csv": record.source_csv,
            "source_row_type": record.row_type,
            "selection_format": "content_one_x_vector",
        }
        instance = build_assignment_instance_from_files(
            sm_path=sm_path,
            selected_ids=record.selected_ids,
            config=config,
            metadata=metadata,
        )
        save_instance_json(instance, output_path)
        converted.append(
            ConvertedInstance(
                group=group,
                instance_name=instance.instance_name,
                source_sm=str(sm_path),
                source_csv=record.source_csv or "",
                output_json=str(output_path.relative_to(output_root)),
                n=instance.n,
                m=instance.m,
                k=instance.k,
                selected_count=len(record.selected_ids),
            )
        )

    _write_manifests_flat(output_root, converted)
    return converted


def _write_manifests_flat(output_root: Path, rows: list[ConvertedInstance]) -> None:
    """分别写出 group 级 manifest 和根目录总 manifest。"""

    fieldnames = [
        "group",
        "instance_name",
        "source_sm",
        "source_csv",
        "output_json",
        "n",
        "m",
        "k",
        "selected_count",
    ]
    by_group: dict[str, list[ConvertedInstance]] = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)

    for group, group_rows in by_group.items():
        manifest_path = output_root / group / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in group_rows:
                writer.writerow(row.__dict__)

    all_manifest_path = output_root / "manifest.csv"
    all_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with all_manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


@dataclass(frozen=True)
class ConvertedInstanceV2:
    topology: str
    input_topology: str
    group: str
    instance_uid: str
    instance_name: str
    source_sm: str
    source_csv: str
    output_json: str
    n: int
    m: int
    k: int
    selected_count: int


def _write_manifests_v2(output_root: Path, rows: list[ConvertedInstanceV2]) -> None:
    fieldnames = list(ConvertedInstanceV2.__dataclass_fields__)
    grouped: dict[tuple[str, str], list[ConvertedInstanceV2]] = {}
    for row in rows:
        grouped.setdefault((row.topology, row.group), []).append(row)
    for (topology, group), group_rows in grouped.items():
        path = output_root / topology / group / 'manifest.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.__dict__ for row in group_rows)
    path = output_root / 'manifest.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)


def convert_pre_data(
    instance_root: str | Path,
    selection_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> list[ConvertedInstanceV2]:
    '''Convert topology-scoped source pairs without filename collisions.'''

    from pa_moap_rl.data.loader import load_instance_json

    instance_root = Path(instance_root)
    output_root = Path(output_root)
    config = load_config()
    pairs = load_pre_data_pairs(instance_root, selection_root)
    converted: list[ConvertedInstanceV2] = []
    for sm_path, record in tqdm(pairs, desc='Converting assignment instances', unit='instance'):
        input_topology = str((record.metadata or {}).get('input_topology', 'legacy'))
        topology = str((record.metadata or {}).get('topology', input_topology))
        group = _group_name_from_sm_path(sm_path)
        stem = Path(record.instance_name).stem
        uid = f'{topology}/{group}/{stem}'
        output_path = output_root / topology / group / f'{stem}.assignment.json'
        metadata = {
            'source_csv': record.source_csv,
            'source_row_type': record.row_type,
            'selection_format': 'content_one_x_vector',
            'topology': topology,
            'input_topology': input_topology,
            'group': group,
            'instance_uid': uid,
        }
        if output_path.exists() and not overwrite:
            instance = load_instance_json(output_path)
        else:
            instance = build_assignment_instance_from_files(
                sm_path=sm_path,
                selected_ids=record.selected_ids,
                config=config,
                metadata=metadata,
            )
            save_instance_json(instance, output_path)
        converted.append(
            ConvertedInstanceV2(
                topology=topology,
                input_topology=input_topology,
                group=group,
                instance_uid=uid,
                instance_name=instance.instance_name,
                source_sm=str(sm_path),
                source_csv=record.source_csv or '',
                output_json=str(output_path),
                n=instance.n,
                m=instance.m,
                k=instance.k,
                selected_count=len(record.selected_ids),
            )
        )
    _write_manifests_v2(output_root, converted)
    return converted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", default="pre_data/instance", help="Root directory containing grouped .sm files.")
    parser.add_argument("--selection-root", default="pre_data/x", help="Root directory containing content-one summary CSV files.")
    parser.add_argument("--output-root", default="instance", help="Output directory for grouped assignment JSON files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted JSON files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = convert_pre_data(
        instance_root=args.instance_root,
        selection_root=args.selection_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(rows)} assignment instances into {args.output_root}.")


if __name__ == "__main__":
    main()
