"""Compiled distillation planning for AEG artifacts."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class DistillationMode(enum.Enum):
    """Supported PRD distillation modes."""

    LOGIT = "logit"
    FEATURE = "feature"
    REASONING = "reasoning"
    SELF = "self"


@dataclass(frozen=True)
class DistillationPlan:
    """A deterministic plan for training or compiling a student AEG."""

    teacher_model: str
    student_model: str
    modes: tuple[DistillationMode, ...]
    datasets: tuple[str, ...]
    temperature: float = 2.0
    alpha: float = 0.7
    feature_layers: tuple[int, ...] = field(default_factory=tuple)
    target_quality_retention: float = 0.95

    def loss_weights(self) -> dict[str, float]:
        active = {mode.value for mode in self.modes}
        weights = {
            "hard_label_ce": 1.0 - self.alpha,
            "kl_logits": self.alpha if "logit" in active else 0.0,
            "hidden_state_mse": 0.35 if "feature" in active else 0.0,
            "reasoning_trace_ce": 0.45 if "reasoning" in active else 0.0,
            "self_consistency": 0.25 if "self" in active else 0.0,
        }
        total = sum(weights.values()) or 1.0
        return {key: round(value / total, 6) for key, value in weights.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "modes": [mode.value for mode in self.modes],
            "datasets": list(self.datasets),
            "temperature": self.temperature,
            "alpha": self.alpha,
            "feature_layers": list(self.feature_layers),
            "target_quality_retention": self.target_quality_retention,
            "loss_weights": self.loss_weights(),
            "eval_gate": {"max_relative_regression": 1.0 - self.target_quality_retention},
        }


class DistillationPipeline:
    """Plan compiled distillation jobs and their AEG artifact contracts."""

    def plan(
        self,
        teacher_model: str,
        student_model: str,
        task_type: str = "general",
        compression_target: float = 0.25,
    ) -> DistillationPlan:
        modes = self._modes_for_task(task_type)
        datasets = self._datasets_for_task(task_type)
        feature_layers = self._feature_layers(compression_target)
        retention = 0.97 if compression_target >= 0.5 else 0.95
        return DistillationPlan(
            teacher_model=teacher_model,
            student_model=student_model,
            modes=modes,
            datasets=datasets,
            feature_layers=feature_layers,
            target_quality_retention=retention,
        )

    def compile_manifest(self, plan: DistillationPlan) -> dict[str, Any]:
        payload = plan.to_dict()
        payload.update(
            {
                "version": "distillation/1.0",
                "compiler_outputs": ["student.aeg", "eval_report.json", "provenance/manifest.json"],
                "quality_gates": ["perplexity", "task_eval", "safety_regression"],
            }
        )
        return payload

    def _modes_for_task(self, task_type: str) -> tuple[DistillationMode, ...]:
        normalized = task_type.lower()
        if normalized in {"math", "code", "reasoning"}:
            return (DistillationMode.LOGIT, DistillationMode.FEATURE, DistillationMode.REASONING)
        if normalized in {"chat", "general"}:
            return (DistillationMode.LOGIT, DistillationMode.SELF)
        return (DistillationMode.LOGIT, DistillationMode.FEATURE)

    def _datasets_for_task(self, task_type: str) -> tuple[str, ...]:
        normalized = task_type.lower()
        if normalized == "code":
            return ("humaneval", "mbpp", "code_contests")
        if normalized in {"math", "reasoning"}:
            return ("gsm8k", "math-500", "aime_style")
        return ("mmlu", "hellaswag", "arena_conversations")

    def _feature_layers(self, compression_target: float) -> tuple[int, ...]:
        if compression_target < 0.25:
            return (0, 2, 4, 8, 16, 24)
        if compression_target < 0.5:
            return (0, 4, 12, 20)
        return (0, 8, 16)
