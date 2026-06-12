"""由 `.sm` 原始数据和内容一选择结果构造内容二 assignment instance。

这里是数据流水线的核心转换层：输入是原始知识点属性和被选中的原始节点编号，
输出是 solver/env/model 都能直接消费的矩阵化算例。构造完成后，所有节点都
使用局部编号 `0..N-1`，不再暴露 `.sm` 的 `nodnr` 编号细节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.instance_schema import (
    AssignmentInstance,
    InstanceValidationError,
    SMInstance,
    SelectionRecord,
    validate_assignment_instance,
)
from pa_moap_rl.data.parse_sm import parse_sm
from pa_moap_rl.data.selection_loader import find_instance_file


def build_concept_need(category_id: np.ndarray, cognitive_load: np.ndarray, category_prototype: np.ndarray) -> np.ndarray:
    """构造知识点需求矩阵 `concept_need`，shape 为 `[N, 6]`。

    前 5 维来自类别原型矩阵，最后 1 维使用该知识点的认知负荷。
    """

    need = np.empty((category_id.shape[0], 6), dtype=np.float64)
    need[:, :5] = category_prototype[category_id]
    need[:, 5] = cognitive_load
    return need


def build_pair_feature_tensor(
    concept_need: np.ndarray,
    cognitive_load: np.ndarray,
    method_attr: np.ndarray,
    lambda_q: float,
    lambda_r: float,
) -> np.ndarray:
    """构造知识点-方法交互特征 `z_im`，shape 为 `[N, M, 6]`。

    前 5 维刻画知识点需求与方法属性的逐维匹配，第 6 维综合认知负荷和方法
    活动强度，用于学生/教师偏好计算。
    """

    n = concept_need.shape[0]
    m = method_attr.shape[0]
    z = np.empty((n, m, 6), dtype=np.float64)
    z[:, :, :5] = concept_need[:, None, :5] * method_attr[None, :, :5]
    z[:, :, 5] = lambda_q * cognitive_load[:, None] + lambda_r * method_attr[None, :, 5]
    return z


def build_preference_matrix(
    pair_features: np.ndarray,
    profile: np.ndarray,
    alpha: float,
    activity_weights: np.ndarray,
) -> np.ndarray:
    """计算某个画像主体的偏好矩阵 `P_im`，shape 为 `[N, M]`。

    `profile` 可以是学生画像或教师画像；`alpha` 控制活动匹配项和负荷适配项
    的折中。
    """

    phi = np.sum(activity_weights[None, None, :] * (1.0 - np.abs(pair_features[:, :, :5] - profile[None, None, :5])), axis=2)
    psi = 1.0 - np.maximum(0.0, pair_features[:, :, 5] - profile[5])
    return np.clip(alpha * phi + (1.0 - alpha) * psi, 0.0, 1.0)


def _ensure_config(config: PAConfig | None) -> PAConfig:
    return config if config is not None else load_config()


def build_assignment_instance(
    sm_instance: SMInstance,
    selected_ids: Iterable[int],
    config: PAConfig | None = None,
    feasible_mask: np.ndarray | None = None,
    metadata: dict | None = None,
) -> AssignmentInstance:
    """从已解析 `.sm` 与选中节点编号构造内容二算例。"""

    cfg = _ensure_config(config)
    selected_ids = [int(value) for value in selected_ids]
    if not selected_ids:
        raise InstanceValidationError("selected_ids must be non-empty.")

    # 这里仍使用 `.sm` 原始 nodnr 校验输入，防止内容一选择结果与算例错配。
    missing = [original_id for original_id in selected_ids if original_id not in sm_instance.id_mapping]
    if missing:
        raise InstanceValidationError(f"selected_ids not found in source .sm: {missing[:10]}")

    # 转换为本地数组下标后，后续矩阵都按 `0..N-1` 排列。
    node_indices = [sm_instance.id_mapping[original_id] for original_id in selected_ids]
    category_id = sm_instance.category_id[node_indices].astype(np.int64)
    cognitive_load = sm_instance.cognitive_load[node_indices].astype(np.float64)

    method = cfg.method
    profile = cfg.profile
    default = cfg.default

    category_prototype = np.asarray(method["category_prototype"], dtype=np.float64)
    method_attr = np.asarray(method["method_attr"], dtype=np.float64)
    effect_by_category = np.asarray(method["effect_matrix_by_category"], dtype=np.float64)
    student_profile = np.asarray(profile["student_profile"], dtype=np.float64)
    teacher_profile = np.asarray(profile["teacher_profile"], dtype=np.float64)
    target_distribution = np.asarray(profile["target_distribution"], dtype=np.float64)
    weights = {key: float(value) for key, value in profile["weights"].items()}

    # effect_matrix 的行是知识点，列是教学方法；配置中按方法 x 类别存储，因此需要转置。
    concept_need = build_concept_need(category_id, cognitive_load, category_prototype)
    effect_matrix = effect_by_category[:, category_id].T.copy()

    preference = default["preference"]
    pair_features = build_pair_feature_tensor(
        concept_need=concept_need,
        cognitive_load=cognitive_load,
        method_attr=method_attr,
        lambda_q=float(preference["lambda_q"]),
        lambda_r=float(preference["lambda_r"]),
    )
    activity_weights = np.asarray(preference["activity_weights"], dtype=np.float64)
    student_pref = build_preference_matrix(
        pair_features=pair_features,
        profile=student_profile,
        alpha=float(preference["alpha_student"]),
        activity_weights=activity_weights,
    )
    teacher_pref = build_preference_matrix(
        pair_features=pair_features,
        profile=teacher_profile,
        alpha=float(preference["alpha_teacher"]),
        activity_weights=activity_weights,
    )

    n = len(selected_ids)
    m = len(method["methods"])
    # v1 中没有额外硬约束时，所有知识点-方法组合默认可行。
    if feasible_mask is None:
        feasible_mask = np.ones((n, m), dtype=bool)
    else:
        feasible_mask = np.asarray(feasible_mask, dtype=bool)

    instance = AssignmentInstance(
        instance_name=sm_instance.instance_name,
        source_sm=sm_instance.source_sm,
        selected_ids=selected_ids,
        original_to_local_id={original_id: local_id for local_id, original_id in enumerate(selected_ids)},
        n=n,
        m=m,
        k=len(method["categories"]),
        category_names=list(method["categories"]),
        method_names=list(method["methods"]),
        category_id=category_id,
        cognitive_load=cognitive_load,
        concept_need=concept_need,
        method_attr=method_attr,
        effect_matrix=effect_matrix,
        student_profile=student_profile,
        teacher_profile=teacher_profile,
        student_pref=student_pref,
        teacher_pref=teacher_pref,
        target_distribution=target_distribution,
        feasible_mask=feasible_mask,
        weights=weights,
        metadata=metadata,
    )
    validate_assignment_instance(instance)
    return instance


def build_assignment_instance_from_files(
    sm_path: str | Path,
    selected_ids: Iterable[int],
    config: PAConfig | None = None,
    feasible_mask: np.ndarray | None = None,
    metadata: dict | None = None,
) -> AssignmentInstance:
    """解析 `.sm` 文件并直接构造内容二算例。"""

    cfg = _ensure_config(config)
    sm_instance = parse_sm(sm_path, category_to_id={name: idx for idx, name in enumerate(cfg.method["categories"])})
    return build_assignment_instance(sm_instance, selected_ids, config=cfg, feasible_mask=feasible_mask, metadata=metadata)


def build_assignment_instance_from_selection(
    instance_root: str | Path,
    selection: SelectionRecord,
    config: PAConfig | None = None,
) -> AssignmentInstance:
    """根据 `SelectionRecord` 在实例根目录中找到 `.sm` 并构造内容二算例。"""

    sm_path = find_instance_file(instance_root, selection.instance_name)
    metadata = {
        "source_csv": selection.source_csv,
        "content_one_metadata": selection.metadata or {},
    }
    return build_assignment_instance_from_files(sm_path, selection.selected_ids, config=config, metadata=metadata)
