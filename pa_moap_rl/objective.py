"""Versioned objective specifications for PA-MOAP scoring and solvers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ObjectiveSpec:
    """An immutable, hashable scalar-objective definition.

    Schema v2 normalizes the three positive research weights and keeps the
    entropy and method-cap penalties as two independently identifiable terms.
    Schema v1 exists only to reproduce historical scores exactly.
    """

    objective_id: str
    schema_version: int
    alpha_effect: float
    alpha_preference: float
    alpha_global: float
    entropy_min: float
    method_cap: float
    beta_entropy: float
    beta_cap: float
    target_distribution: tuple[float, ...] | None
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if not self.objective_id.strip():
            raise ValueError("objective_id must be non-empty.")
        if self.schema_version not in {1, 2}:
            raise ValueError("schema_version must be 1 or 2.")
        positive = np.asarray(
            [self.alpha_effect, self.alpha_preference, self.alpha_global],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(positive)) or np.any(positive < 0.0):
            raise ValueError("Positive objective weights must be finite and non-negative.")
        if self.schema_version == 2 and abs(float(positive.sum()) - 1.0) > 1.0e-9:
            raise ValueError("Schema-v2 positive objective weights must sum to 1.")
        for name, value in (
            ("entropy_min", self.entropy_min),
            ("method_cap", self.method_cap),
            ("beta_entropy", self.beta_entropy),
            ("beta_cap", self.beta_cap),
            ("epsilon", self.epsilon),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= float(self.entropy_min) <= 1.0:
            raise ValueError("entropy_min must be in [0, 1].")
        if not 0.0 < float(self.method_cap) <= 1.0:
            raise ValueError("method_cap must be in (0, 1].")
        if min(float(self.beta_entropy), float(self.beta_cap)) < 0.0:
            raise ValueError("Penalty coefficients must be non-negative.")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.alpha_global > 0.0 and self.target_distribution is None:
            raise ValueError("target_distribution is required when alpha_global is positive.")
        if self.target_distribution is not None:
            target = np.asarray(self.target_distribution, dtype=np.float64)
            if self.schema_version == 2 and target.size != 11:
                raise ValueError('Schema-v2 target_distribution must have length 11.')
            if target.ndim != 1 or target.size == 0:
                raise ValueError("target_distribution must be a non-empty vector.")
            if np.any(~np.isfinite(target)) or np.any(target < 0.0):
                raise ValueError("target_distribution must be finite and non-negative.")
            if abs(float(target.sum()) - 1.0) > 1.0e-8:
                raise ValueError("target_distribution must sum to 1.")

    @property
    def alpha_student(self) -> float:
        return float(self.alpha_preference) / 2.0

    @property
    def alpha_teacher(self) -> float:
        return float(self.alpha_preference) / 2.0

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "objective_id": self.objective_id,
            "schema_version": self.schema_version,
            "alpha_effect": float(self.alpha_effect),
            "alpha_preference": float(self.alpha_preference),
            "alpha_global": float(self.alpha_global),
            "alpha_student": self.alpha_student,
            "alpha_teacher": self.alpha_teacher,
            "entropy_min": float(self.entropy_min),
            "method_cap": float(self.method_cap),
            "beta_entropy": float(self.beta_entropy),
            "beta_cap": float(self.beta_cap),
            "target_distribution": (
                list(self.target_distribution) if self.target_distribution is not None else None
            ),
            "epsilon": float(self.epsilon),
        }
        if include_hash:
            result["config_hash"] = self.config_hash
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveSpec":
        positive = data.get("positive_weights", {})
        penalties = data.get("soft_constraints", {})
        forbidden = {'lambda_soft', 'lambda_div', 'lambda_cap'} & set(data)
        if forbidden:
            raise ValueError(
                'ObjectiveSpec owns both penalty coefficients; external penalty keys are forbidden: '
                + ', '.join(sorted(forbidden))
            )
        spec = cls(
            objective_id=str(data["objective_id"]),
            schema_version=int(data.get("schema_version", 2)),
            alpha_effect=float(data.get("alpha_effect", positive.get("effect"))),
            alpha_preference=float(data.get("alpha_preference", positive.get("preference"))),
            alpha_global=float(data.get("alpha_global", positive.get("global"))),
            entropy_min=float(data.get("entropy_min", penalties.get("entropy_min"))),
            method_cap=float(data.get("method_cap", penalties.get("method_cap"))),
            beta_entropy=float(data.get("beta_entropy", penalties.get("beta_entropy"))),
            beta_cap=float(data.get("beta_cap", penalties.get("beta_cap"))),
            target_distribution=(
                tuple(float(value) for value in data["target_distribution"])
                if data.get("target_distribution") is not None
                else None
            ),
            epsilon=float(data.get("epsilon", penalties.get("epsilon", 1.0e-8))),
        )
        for key, expected in (
            ("alpha_student", spec.alpha_student),
            ("alpha_teacher", spec.alpha_teacher),
        ):
            if key in data and abs(float(data[key]) - expected) > 1.0e-12:
                raise ValueError(f"{key} must equal alpha_preference / 2.")
        return spec


def load_objective_spec(path: str | Path, objective_id: str | None = None) -> ObjectiveSpec:
    """Load one spec from a direct mapping or a calibration catalogue."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Objective configuration must be a mapping.")
    if "candidates" in data:
        if objective_id is None:
            raise ValueError("objective_id is required for a calibration catalogue.")
        candidates = data["candidates"]
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list.")
        match = next((item for item in candidates if item.get("objective_id") == objective_id), None)
        if match is None:
            raise ValueError(f"Unknown objective_id {objective_id!r}.")
        merged = dict(data.get("defaults", {}))
        merged.update(match)
        data = merged
    spec = ObjectiveSpec.from_dict(data)
    expected = data.get("config_hash")
    if expected is not None and str(expected) != spec.config_hash:
        raise ValueError("Objective configuration hash mismatch.")
    return spec


def legacy_objective_spec(instance: Any, *, H_min: float, pi_cap: float,
                          lambda_div: float, lambda_cap: float, epsilon: float) -> ObjectiveSpec:
    """Translate one historical instance into an exactly equivalent schema-v1 spec."""

    weights = instance.weights
    return ObjectiveSpec(
        objective_id="legacy_v1",
        schema_version=1,
        alpha_effect=float(weights["effect"]),
        alpha_preference=float(weights["student"]) + float(weights["teacher"]),
        alpha_global=float(weights["global"]),
        entropy_min=float(H_min),
        method_cap=float(pi_cap),
        beta_entropy=float(weights["soft"]) * float(lambda_div),
        beta_cap=float(weights["soft"]) * float(lambda_cap),
        target_distribution=tuple(float(value) for value in instance.target_distribution),
        epsilon=float(epsilon),
    )


__all__ = ["ObjectiveSpec", "legacy_objective_spec", "load_objective_spec"]
