"""Preference-scenario generation for shared-policy training and evaluation."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.build_assignment_instance import build_pair_feature_tensor, build_preference_matrix
from pa_moap_rl.data.instance_schema import AssignmentInstance, validate_assignment_instance
from pa_moap_rl.objective import ObjectiveSpec


@dataclass(frozen=True)
class PreferenceScenario:
    '''One reproducible student-teacher preference realization.'''

    scenario_id: str
    student_template: str
    teacher_template: str
    student_profile: np.ndarray
    teacher_profile: np.ndarray
    target_distribution: np.ndarray | None
    weights: dict[str, float] | None
    held_out: bool
    seed: int
    schema_version: int = 1
    access_status: str = 'legacy'
    perturbation_config_hash: str = ''

    def to_dict(self) -> dict[str, Any]:
        result = {
            'scenario_id': self.scenario_id,
            'student_template': self.student_template,
            'teacher_template': self.teacher_template,
            'student_profile': self.student_profile.tolist(),
            'teacher_profile': self.teacher_profile.tolist(),
            'held_out': self.held_out,
            'seed': self.seed,
            'schema_version': self.schema_version,
            'access_status': self.access_status,
            'perturbation_config_hash': self.perturbation_config_hash,
        }
        if self.schema_version == 1:
            result['target_distribution'] = self.target_distribution.tolist()
            result['weights'] = dict(self.weights)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PreferenceScenario':
        return cls(
            scenario_id=str(data['scenario_id']),
            student_template=str(data['student_template']),
            teacher_template=str(data['teacher_template']),
            student_profile=np.asarray(data['student_profile'], dtype=np.float64),
            teacher_profile=np.asarray(data['teacher_profile'], dtype=np.float64),
            target_distribution=(
                np.asarray(data['target_distribution'], dtype=np.float64)
                if data.get('target_distribution') is not None
                else None
            ),
            weights=(
                {key: float(value) for key, value in data['weights'].items()}
                if data.get('weights') is not None
                else None
            ),
            held_out=bool(data['held_out']),
            seed=int(data['seed']),
            schema_version=int(data.get('schema_version', 1)),
            access_status=str(data.get('access_status', 'legacy')),
            perturbation_config_hash=str(data.get('perturbation_config_hash', '')),
        )

def load_scenario_config(path: str | Path | None = None) -> dict[str, Any]:
    '''Load and validate the six-by-six teaching profile catalogue.'''

    config_path = Path(path) if path else Path(__file__).parents[1] / 'configs' / 'scenario_config.yaml'
    data = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('scenario_config.yaml must contain a mapping.')
    dimensions = data.get('dimensions', [])
    students = data.get('student_templates', {})
    teachers = data.get('teacher_templates', {})
    if len(dimensions) != 6 or len(students) != 6 or len(teachers) != 6:
        raise ValueError('Scenario config must define six dimensions and 6x6 templates.')
    for kind, templates in (('student', students), ('teacher', teachers)):
        for template_id, template in templates.items():
            values = np.asarray(template.get('values', []), dtype=np.float64)
            if values.shape != (6,) or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f'Invalid {kind} template {template_id!r}.')
    held_out = {tuple(pair) for pair in data.get('held_out_pairs', [])}
    all_pairs = set(product(students, teachers))
    if not held_out or not held_out <= all_pairs:
        raise ValueError('held_out_pairs must be a non-empty subset of the 6x6 pairs.')
    return data


def scenario_pairs(config: dict[str, Any], held_out: bool) -> list[tuple[str, str]]:
    '''Return template pairs assigned to interpolation or held-out evaluation.'''

    excluded = {tuple(pair) for pair in config['held_out_pairs']}
    pairs = list(product(config['student_templates'], config['teacher_templates']))
    return [pair for pair in pairs if (pair in excluded) == held_out]


def _perturb(values: list[float], config: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    settings = config['perturbation']
    noise = rng.normal(0.0, float(settings['std']), size=6)
    noise = np.clip(noise, -float(settings['max_abs']), float(settings['max_abs']))
    return np.clip(
        np.asarray(values, dtype=np.float64) + noise,
        float(settings['clip_min']),
        float(settings['clip_max']),
    )


def sample_scenario(
    rng: np.random.Generator,
    scenario_config: dict[str, Any] | None = None,
    pa_config: PAConfig | None = None,
    held_out: bool = False,
    scenario_id: str | None = None,
    template_pair: tuple[str, str] | None = None,
    objective_independent: bool = False,
    access_status: str = 'development',
) -> PreferenceScenario:
    '''Sample one continuously perturbed, reproducible preference scenario.'''

    config = scenario_config or load_scenario_config()
    base = pa_config or load_config()
    pairs = scenario_pairs(config, held_out=held_out)
    if template_pair is not None and template_pair not in pairs:
        raise ValueError(f'Template pair {template_pair!r} is not valid for held_out={held_out}.')
    student_id, teacher_id = template_pair or pairs[int(rng.integers(len(pairs)))]
    seed = int(rng.integers(0, 2**31 - 1))
    local_rng = np.random.default_rng(seed)
    student = _perturb(config['student_templates'][student_id]['values'], config, local_rng)
    teacher = _perturb(config['teacher_templates'][teacher_id]['values'], config, local_rng)
    perturbation_hash = hashlib.sha256(
        json.dumps(config['perturbation'], sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return PreferenceScenario(
        scenario_id=scenario_id or f'{student_id}-{teacher_id}-{seed:010d}',
        student_template=student_id,
        teacher_template=teacher_id,
        student_profile=student,
        teacher_profile=teacher,
        target_distribution=(
            None
            if objective_independent
            else np.asarray(base.profile['target_distribution'], dtype=np.float64)
        ),
        weights=(
            None
            if objective_independent
            else {key: float(value) for key, value in base.profile['weights'].items()}
        ),
        held_out=held_out,
        seed=seed,
        schema_version=2 if objective_independent else 1,
        access_status=access_status,
        perturbation_config_hash=perturbation_hash,
    )


def apply_scenario(
    instance: AssignmentInstance,
    scenario: PreferenceScenario,
    pa_config: PAConfig | None = None,
    objective: ObjectiveSpec | None = None,
) -> AssignmentInstance:
    '''Recompute preference matrices while preserving fixed teaching effects.'''

    config = pa_config or load_config()
    preference = config.default['preference']
    pair_features = build_pair_feature_tensor(
        concept_need=instance.concept_need,
        cognitive_load=instance.cognitive_load,
        method_attr=instance.method_attr,
        lambda_q=float(preference['lambda_q']),
        lambda_r=float(preference['lambda_r']),
    )
    activity_weights = np.asarray(preference['activity_weights'], dtype=np.float64)
    student_pref = build_preference_matrix(
        pair_features,
        scenario.student_profile,
        float(preference['alpha_student']),
        activity_weights,
    )
    teacher_pref = build_preference_matrix(
        pair_features,
        scenario.teacher_profile,
        float(preference['alpha_teacher']),
        activity_weights,
    )
    metadata = dict(instance.metadata or {})
    metadata['scenario'] = {
        'scenario_id': scenario.scenario_id,
        'student_template': scenario.student_template,
        'teacher_template': scenario.teacher_template,
        'held_out': scenario.held_out,
        'seed': scenario.seed,
        'schema_version': scenario.schema_version,
        'access_status': scenario.access_status,
        'perturbation_config_hash': scenario.perturbation_config_hash,
    }
    if objective is not None:
        metadata['scenario']['objective_id'] = objective.objective_id
        metadata['scenario']['objective_hash'] = objective.config_hash
    if scenario.schema_version == 2 and objective is None:
        raise ValueError('Scenario v2 requires an explicit ObjectiveSpec.')
    replacements: dict[str, Any] = {}
    if scenario.schema_version == 1 and objective is None:
        if scenario.target_distribution is None or scenario.weights is None:
            raise ValueError('Legacy scenario is missing embedded objective fields.')
        replacements['target_distribution'] = scenario.target_distribution.copy()
        replacements['weights'] = dict(scenario.weights)
    conditioned = replace(
        instance,
        student_profile=scenario.student_profile.copy(),
        teacher_profile=scenario.teacher_profile.copy(),
        student_pref=student_pref,
        teacher_pref=teacher_pref,
        metadata=metadata,
        **replacements,
    )
    validate_assignment_instance(conditioned)
    return conditioned


def generate_scenario_bank(
    count: int,
    seed: int,
    held_out: bool = False,
    scenario_config: dict[str, Any] | None = None,
    pa_config: PAConfig | None = None,
    objective_independent: bool = False,
    access_status: str = 'development',
) -> list[PreferenceScenario]:
    '''Generate a fixed scenario bank for comparable validation or test runs.'''

    if count < 1:
        raise ValueError('count must be positive.')
    rng = np.random.default_rng(seed)
    prefix = 'heldout' if held_out else 'interp'
    config = scenario_config or load_scenario_config()
    pairs = scenario_pairs(config, held_out=held_out)
    return [
        sample_scenario(
            rng,
            scenario_config=config,
            pa_config=pa_config,
            held_out=held_out,
            scenario_id=f'{prefix}-{index:04d}',
            template_pair=pairs[index % len(pairs)],
            objective_independent=objective_independent,
            access_status=access_status,
        )
        for index in range(count)
    ]


def save_scenario_bank(scenarios: list[PreferenceScenario], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([scenario.to_dict() for scenario in scenarios], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def load_scenario_bank(path: str | Path) -> list[PreferenceScenario]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('Scenario bank must contain a non-empty list.')
    return [PreferenceScenario.from_dict(item) for item in data]
