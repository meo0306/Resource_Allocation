"""Strict provenance helpers for versioned PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np

from pa_moap_rl.data.instance_schema import AssignmentInstance
from pa_moap_rl.objective import ObjectiveSpec


CHECKPOINT_SCHEMA_VERSION = 2


def _sha256_payload(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder='big', signed=False))
        digest.update(part)
    return digest.hexdigest()


def _array_bytes(values: np.ndarray) -> bytes:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {'dtype': str(array.dtype), 'shape': list(array.shape)},
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return header + b'\0' + array.tobytes(order='C')


def method_matrix_hash(instance: AssignmentInstance) -> str:
    """Hash method identity and attributes used by the policy network."""

    names = json.dumps(instance.method_names, ensure_ascii=False, separators=(',', ':'))
    return _sha256_payload([names.encode('utf-8'), _array_bytes(instance.method_attr)])


def instance_data_hash(instance: AssignmentInstance) -> str:
    """Hash all assignment inputs that can change scores or action legality."""

    identity = json.dumps(
        {
            'instance_name': instance.instance_name,
            'source_sm': instance.source_sm,
            'selected_ids': list(instance.selected_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    arrays = (
        instance.category_id,
        instance.cognitive_load,
        instance.concept_need,
        instance.method_attr,
        instance.effect_matrix,
        instance.student_profile,
        instance.teacher_profile,
        instance.student_pref,
        instance.teacher_pref,
        instance.target_distribution,
        instance.feasible_mask,
    )
    return _sha256_payload([identity, *(_array_bytes(array) for array in arrays)])


def collection_data_hash(instances: Iterable[AssignmentInstance]) -> str:
    hashes = sorted(instance_data_hash(instance) for instance in instances)
    return _sha256_payload([value.encode('ascii') for value in hashes])


def checkpoint_metadata(
    objective: ObjectiveSpec,
    *,
    method_hash: str,
    data_hash: str,
) -> dict[str, Any]:
    return {
        'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
        'objective': objective.to_dict(),
        'objective_hash': objective.config_hash,
        'method_matrix_hash': str(method_hash),
        'data_hash': str(data_hash),
    }


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    objective: ObjectiveSpec,
    method_hash: str,
    data_hash: str,
    legacy_mode: bool = False,
) -> None:
    """Fail immediately on objective, method-matrix, or data drift."""

    metadata = checkpoint.get('provenance')
    if metadata is None:
        if legacy_mode and objective.schema_version == 1:
            return
        raise ValueError(
            'Legacy checkpoint has no provenance. Reading it requires explicit '
            'legacy_mode=True and an explicit schema-v1 ObjectiveSpec.'
        )
    if int(metadata.get('checkpoint_schema_version', -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('Unsupported checkpoint schema version.')
    expected = {
        'objective_hash': objective.config_hash,
        'method_matrix_hash': str(method_hash),
        'data_hash': str(data_hash),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f'Checkpoint provenance mismatch: {mismatches}.')
    restored = ObjectiveSpec.from_dict(metadata['objective'])
    if restored.config_hash != objective.config_hash:
        raise ValueError('Checkpoint objective snapshot does not match its hash.')


__all__ = [
    'CHECKPOINT_SCHEMA_VERSION',
    'checkpoint_metadata',
    'collection_data_hash',
    'instance_data_hash',
    'method_matrix_hash',
    'validate_checkpoint_metadata',
]
