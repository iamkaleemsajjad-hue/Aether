"""
Precision assignment engine.

Given a sensitivity map and a quality budget, assigns the optimal precision
format to each layer. The engine supports multiple assignment strategies
(per-layer, per-block) and produces a precision_map.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import SENSITIVITY_CRITICAL_THRESHOLD, SENSITIVITY_HIGH_THRESHOLD, SENSITIVITY_MEDIUM_THRESHOLD
from aether.core.exceptions import PrecisionAssignmentError
from aether.quantization.sensitivity import LayerSensitivity
from aether.utils.logging import get_logger

logger = get_logger(__name__)

PRECISION_TIER_MAP: list[tuple[float, str]] = [
    (SENSITIVITY_CRITICAL_THRESHOLD, "BF16"),
    (SENSITIVITY_HIGH_THRESHOLD, "FP8"),
    (SENSITIVITY_MEDIUM_THRESHOLD, "Q4_K_M"),
]


@dataclass
class AssignmentResult:
    """Result of a precision assignment run."""

    precision_map: dict[str, str] = field(default_factory=dict)
    quality_estimate: float = 0.0
    memory_saved_bytes: int = 0
    memory_original_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.memory_original_bytes <= 0:
            return 1.0
        return self.memory_original_bytes / max(self.memory_saved_bytes, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_map": self.precision_map,
            "quality_estimate": self.quality_estimate,
            "memory_saved_bytes": self.memory_saved_bytes,
            "memory_original_bytes": self.memory_original_bytes,
            "compression_ratio": self.compression_ratio,
        }


class PrecisionAssigner:
    """Assigns mixed precision to layers based on sensitivity scores."""

    def __init__(self, quality_budget: float = 0.02) -> None:
        self.quality_budget = quality_budget

    def assign(
        self,
        sensitivity_scores: dict[str, LayerSensitivity],
        total_params_size_bytes: int = 0,
    ) -> AssignmentResult:
        """Assign precision to each layer based on sensitivity.

        Args:
            sensitivity_scores: Sensitivity scores keyed by layer name.
            total_params_size_bytes: Total parameter size in bytes at BF16.

        Returns:
            An AssignmentResult with the precision map and quality estimate.
        """
        precision_map: dict[str, str] = {}
        for layer_name, score in sorted(
            sensitivity_scores.items(),
            key=lambda x: (x[1].score if x[1].layer_index >= 0 else float("inf"), x[0]),
            reverse=True,
        ):
            if score.score >= SENSITIVITY_CRITICAL_THRESHOLD:
                precision = "BF16"
            elif score.score >= SENSITIVITY_HIGH_THRESHOLD:
                precision = "FP8"
            elif score.score >= SENSITIVITY_MEDIUM_THRESHOLD:
                precision = "Q4_K_M"
            else:
                precision = "Q3_K"
            precision_map[layer_name] = precision

        # Estimate quality impact
        quality_estimate = self._estimate_quality(precision_map, sensitivity_scores)
        if quality_estimate > self.quality_budget:
            logger.warning(
                "Quality estimate %.4f exceeds budget %.4f — tightening may overshoot",
                quality_estimate,
                self.quality_budget,
            )

        # Memory estimates
        memory_original = total_params_size_bytes or len(precision_map) * 4096 * 4096 * 2
        memory_used = self._estimate_memory(precision_map, total_params_size_bytes or memory_original)
        memory_saved = memory_original - memory_used

        return AssignmentResult(
            precision_map=precision_map,
            quality_estimate=quality_estimate,
            memory_saved_bytes=memory_saved,
            memory_original_bytes=memory_original,
        )

    def assign_uniform(self, layers: int, precision: str) -> AssignmentResult:
        """Assign uniform precision to all layers."""
        precision_map: dict[str, str] = {}
        for i in range(layers):
            precision_map[f"layer_{i}"] = precision
        precision_map["embedding"] = "BF16"
        precision_map["lm_head"] = "BF16"
        return AssignmentResult(
            precision_map=precision_map,
            quality_estimate=0.0,
        )

    def _estimate_quality(
        self,
        precision_map: dict[str, str],
        sensitivity_scores: dict[str, LayerSensitivity],
    ) -> float:
        """Estimate total perplexity increase from the precision map.

        This is a lightweight heuristic based on bit width differences. A full
        computation would run the actual model with calibration data.
        """
        total_delta = 0.0
        for layer_name, precision in precision_map.items():
            base_bits = 16.0
            q_bits = float({"BF16": 16, "FP8": 8, "Q4_K_M": 4, "Q3_K": 3, "Q2_K": 2}.get(precision.upper(), 16))
            bit_reduction = max(0.0, (base_bits - q_bits) / base_bits)
            layer_score = sensitivity_scores.get(layer_name, LayerSensitivity(layer_name=layer_name, layer_index=0, score=0.5))
            total_delta += bit_reduction * layer_score.score * 0.01
        return total_delta

    def _estimate_memory(self, precision_map: dict[str, str], original_bytes: int) -> int:
        """Estimate memory used after quantization."""
        from aether.core.constants import PRECISION_SIZES_BYTES

        avg_bits = sum(
            PRECISION_SIZES_BYTES.get(p.upper(), 2.0) * 8
            for p in precision_map.values()
        ) / max(len(precision_map), 1)
        ratio = avg_bits / 16.0
        return int(original_bytes * ratio)

    def __repr__(self) -> str:
        return f"PrecisionAssigner(budget={self.quality_budget})"
