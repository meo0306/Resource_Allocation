"""教学方法分配的 Gymnasium-like 环境。

环境状态是一份完整 assignment；动作是单点替换 `(node_index, method_index)`。
每一步 reward 定义为新旧标量目标差值 `J(next) - J(current)`，同时维护
best-so-far assignment，供 PPO 或其他序列决策算法返回最优历史解。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pa_moap_rl.configs import PAConfig, load_config
from pa_moap_rl.data.instance_schema import AssignmentInstance, validate_assignment_instance
from pa_moap_rl.utils.masks import assert_action_legal, build_action_mask, feasible_mask_from_instance
from pa_moap_rl.utils.scoring import ScoreBreakdown, method_distribution, score_assignment


class MethodAssignmentEnv:
    """用于 PA-MOAP 局部替换搜索的小型环境。"""

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        patience: int | None = None,
        improvement_eps: float | None = None,
        config: PAConfig | None = None,
        entropy_min: float | None = None,
        method_cap: float | np.ndarray | None = None,
        lambda_div: float | None = None,
        lambda_cap: float | None = None,
        epsilon: float | None = None,
    ) -> None:
        """初始化环境超参数与软约束参数。

        如果调用者没有显式传入参数，就从默认 YAML 配置读取。
        """

        cfg = config if config is not None else load_config()
        environment = cfg.default["environment"]
        soft_constraints = cfg.default["soft_constraints"]

        self.max_steps = int(max_steps if max_steps is not None else environment["max_steps"])
        self.patience = int(patience if patience is not None else environment["patience"])
        self.improvement_eps = float(
            improvement_eps if improvement_eps is not None else environment["improvement_eps"]
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")

        self.entropy_min = float(entropy_min if entropy_min is not None else soft_constraints["entropy_min"])
        self.method_cap = method_cap if method_cap is not None else float(soft_constraints["method_cap"])
        self.lambda_div = float(lambda_div if lambda_div is not None else soft_constraints["lambda_div"])
        self.lambda_cap = float(lambda_cap if lambda_cap is not None else soft_constraints["lambda_cap"])
        self.epsilon = float(epsilon if epsilon is not None else soft_constraints["epsilon"])

        self.instance: AssignmentInstance | None = None
        self.assignment: np.ndarray | None = None
        self.current_score: ScoreBreakdown | None = None
        self.best_assignment: np.ndarray | None = None
        self.best_score: float | None = None
        self.steps_to_best = 0
        self.step_count = 0
        self.no_improve_steps = 0
        self.done = False
        self.done_reason: str | None = None

    def reset(self, instance: AssignmentInstance, initial_assignment: np.ndarray | list[int] | None = None) -> dict[str, np.ndarray]:
        """开始一个新 episode，并返回初始 observation。"""

        validate_assignment_instance(instance)
        self.instance = instance
        if initial_assignment is None:
            # 默认初解使用 effect-only greedy，保证每个节点先有一个可行方法。
            assignment = self._effect_greedy_initial_assignment(instance)
        else:
            assignment = np.asarray(initial_assignment, dtype=np.int64)
        self._validate_assignment(instance, assignment)

        self.assignment = assignment.copy()
        self.current_score = self._score(self.assignment)
        self.best_assignment = self.assignment.copy()
        self.best_score = self.current_score.total_score
        self.steps_to_best = 0
        self.step_count = 0
        self.no_improve_steps = 0
        self.done = False
        self.done_reason = None
        return self._observation()

    def step(self, action: tuple[int, int]) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """执行一个合法替换动作 `(node_index, method_index)`。"""

        self._require_reset()
        if self.done:
            raise RuntimeError("Cannot call step() after the episode is done; call reset() first.")

        node_index, method_index = self._parse_action(action)
        action_mask = self.get_action_mask()
        assert_action_legal(action_mask, node_index, method_index)

        assert self.assignment is not None
        assert self.current_score is not None

        # reward 只反映这一步的目标增量，best-so-far 单独维护。
        previous_score = self.current_score
        self.assignment[node_index] = method_index
        self.current_score = self._score(self.assignment)
        reward = float(self.current_score.total_score - previous_score.total_score)

        self.step_count += 1
        if self.best_score is None or self.current_score.total_score > self.best_score + self.improvement_eps:
            self.best_score = self.current_score.total_score
            self.best_assignment = self.assignment.copy()
            self.steps_to_best = self.step_count
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        self.done, self.done_reason = self._compute_done()
        obs = self._observation()
        info = {
            "previous_score": previous_score.as_dict(),
            "current_score": self.current_score.as_dict(),
            "best_score": float(self.best_score),
            "best_assignment": self.best_assignment.copy(),
            "steps_to_best": int(self.steps_to_best),
            "step_count": int(self.step_count),
            "no_improve_steps": int(self.no_improve_steps),
            "done_reason": self.done_reason,
        }
        return obs, reward, self.done, info

    def get_action_mask(self) -> np.ndarray:
        """返回当前状态下的动态替换动作 mask。"""

        self._require_reset()
        assert self.instance is not None
        assert self.assignment is not None
        return build_action_mask(self.assignment, self.instance.feasible_mask)

    def compute_reward(
        self,
        previous_assignment: np.ndarray | list[int],
        next_assignment: np.ndarray | list[int],
    ) -> float:
        """在当前实例上计算 `J(next_assignment) - J(previous_assignment)`。"""

        self._require_reset()
        previous = np.asarray(previous_assignment, dtype=np.int64)
        next_values = np.asarray(next_assignment, dtype=np.int64)
        assert self.instance is not None
        self._validate_assignment(self.instance, previous)
        self._validate_assignment(self.instance, next_values)
        return float(self._score(next_values).total_score - self._score(previous).total_score)

    @property
    def current_scores(self) -> np.ndarray:
        """当前评分向量 `[F_E, F_S, F_T, F_G, V_soft, J]`。"""

        self._require_reset()
        assert self.current_score is not None
        return self._score_vector(self.current_score)

    @property
    def method_dist(self) -> np.ndarray:
        """当前教学方法使用分布。"""

        self._require_reset()
        assert self.current_score is not None
        return self.current_score.method_distribution.copy()

    def _observation(self) -> dict[str, np.ndarray]:
        """组装模型和实验脚本共用的 observation 字典。"""

        assert self.instance is not None
        assert self.assignment is not None
        assert self.current_score is not None

        return {
            "category_id": self.instance.category_id.copy(),
            "cognitive_load": self.instance.cognitive_load.copy(),
            "concept_need": self.instance.concept_need.copy(),
            "node_mask": np.ones(self.instance.n, dtype=bool),
            "method_id": np.arange(self.instance.m, dtype=np.int64),
            "method_attr": self.instance.method_attr.copy(),
            "assignment": self.assignment.copy(),
            "effect_matrix": self.instance.effect_matrix.copy(),
            "student_pref": self.instance.student_pref.copy(),
            "teacher_pref": self.instance.teacher_pref.copy(),
            "feasible_mask": self.instance.feasible_mask.copy(),
            "action_mask": self.get_action_mask(),
            "method_distribution": self.current_score.method_distribution.copy(),
            "method_dist": self.current_score.method_distribution.copy(),
            "target_distribution": self.instance.target_distribution.copy(),
            "target_dist": self.instance.target_distribution.copy(),
            "current_scores": self._score_vector(self.current_score),
        }

    def _score(self, assignment: np.ndarray) -> ScoreBreakdown:
        """使用当前环境软约束参数计算 assignment 评分。"""

        assert self.instance is not None
        return score_assignment(
            assignment,
            instance=self.instance,
            H_min=self.entropy_min,
            pi_cap=self.method_cap,
            lambda_div=self.lambda_div,
            lambda_cap=self.lambda_cap,
            epsilon=self.epsilon,
        )

    @staticmethod
    def _score_vector(score: ScoreBreakdown) -> np.ndarray:
        return np.asarray(
            [
                score.effect_score,
                score.student_pref_score,
                score.teacher_pref_score,
                score.global_score,
                score.soft_penalty,
                score.total_score,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _effect_greedy_initial_assignment(instance: AssignmentInstance) -> np.ndarray:
        """用效果矩阵构造可行初解。"""

        feasible = feasible_mask_from_instance(instance)
        masked_effect = np.where(feasible, instance.effect_matrix, -np.inf)
        assignment = np.argmax(masked_effect, axis=1).astype(np.int64)
        if np.any(~np.isfinite(masked_effect[np.arange(instance.n), assignment])):
            raise ValueError("Cannot build an initial assignment because at least one node has no feasible method.")
        return assignment

    @staticmethod
    def _validate_assignment(instance: AssignmentInstance, assignment: np.ndarray) -> None:
        values = np.asarray(assignment, dtype=np.int64)
        if values.shape != (instance.n,):
            raise ValueError(f"assignment shape must be {(instance.n,)}, got {values.shape}.")
        if np.any(values < 0) or np.any(values >= instance.m):
            raise ValueError("assignment contains out-of-range method ids.")
        if not np.all(instance.feasible_mask[np.arange(instance.n), values]):
            bad = np.flatnonzero(~instance.feasible_mask[np.arange(instance.n), values])
            raise ValueError(f"assignment violates feasible_mask at rows: {bad[:10].tolist()}.")

    @staticmethod
    def _parse_action(action: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("action must be a tuple: (node_index, method_index).")
        return int(action[0]), int(action[1])

    def _compute_done(self) -> tuple[bool, str | None]:
        """根据步数上限和无改进耐心值判断 episode 是否结束。"""

        if self.step_count >= self.max_steps:
            return True, "max_steps"
        if self.no_improve_steps >= self.patience:
            return True, "patience"
        return False, None

    def _require_reset(self) -> None:
        if self.instance is None or self.assignment is None or self.current_score is None:
            raise RuntimeError("Environment has not been reset.")


__all__ = ["MethodAssignmentEnv"]
