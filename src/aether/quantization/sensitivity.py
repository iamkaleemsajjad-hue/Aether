"""
Per-layer sensitivity scoring for mixed-precision quantization.

Computes a sensitivity score for each layer by estimating the perplexity
impact of quantizing that layer. Used by the precision assignment engine
to allocate higher precision to more sensitive layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.core.constants import SENSITIVITY_CRITICAL_THRESHOLD, SENSITIVITY_HIGH_THRESHOLD, SENSITIVITY_MEDIUM_THRESHOLD
from aether.core.exceptions import CalibrationError
from aether.core.types import ModelArchitecture
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LayerSensitivity:
    """Sensitivity score for a single layer."""

    layer_name: str
    layer_index: int
    score: float
    params_fraction: float = 0.0
    is_attention: bool = False
    is_moe: bool = False

    @property
    def tier(self) -> str:
        """Return the recommended precision tier."""
        if self.score >= SENSITIVITY_CRITICAL_THRESHOLD:
            return "critical"
        if self.score >= SENSITIVITY_HIGH_THRESHOLD:
            return "high"
        if self.score >= SENSITIVITY_MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "score": self.score,
            "tier": self.tier,
            "params_fraction": self.params_fraction,
            "is_attention": self.is_attention,
            "is_moe": self.is_moe,
        }


class SensitivityScorer:
    """Computes per-layer sensitivity scores.

    Uses a combination of analytical heuristics (depth-based, parameter-based)
    and observed calibration data (perplexity deltas) to produce a sensitivity
    map. A higher score means the layer degrades more when quantized.
    """

    def __init__(self) -> None:
        self._calibration_results: dict[str, float] = {}

    def compute(
        self,
        architecture: ModelArchitecture,
        perplexity_by_layer: dict[str, float] | None = None,
    ) -> dict[str, LayerSensitivity]:
        """Compute sensitivity scores for all layers.

        Args:
            architecture: Model architecture metadata.
            perplexity_by_layer: Optional observed perplexity when each layer is quantized.

        Returns:
            Dictionary mapping layer names to LayerSensitivity instances.
        """
        scores: dict[str, LayerSensitivity] = {}
        for i in range(architecture.layers):
            layer_name = f"layer_{i}"
            if perplexity_by_layer and layer_name in perplexity_by_layer:
                raw_score = perplexity_by_layer[layer_name]
                # Normalize to [0, 1]
                score = min(1.0, raw_score / 10.0)
            else:
                depth_factor = 1.0 - (i / max(architecture.layers, 1)) * 0.5
                score = 0.5 + depth_factor * 0.3

            # Boost attention layers (they're more sensitive)
            is_attention = (i % 2 == 0)
            if is_attention:
                score = min(1.0, score * 1.1)

            scores[layer_name] = LayerSensitivity(
                layer_name=layer_name,
                layer_index=i,
                score=round(score, 4),
                params_fraction=1.0 / max(architecture.layers, 1),
                is_attention=is_attention,
                is_moe=architecture.is_moe and i >= architecture.layers // 2,
            )

        # Add embedding and lm_head
        for extra in ["embedding", "lm_head"]:
            scores[extra] = LayerSensitivity(
                layer_name=extra,
                layer_index=-1,
                score=0.95,
                is_attention=False,
            )

        self._calibration_results = {k: v.score for k, v in scores.items()}
        return scores

    def compute_block_sensitivity(
        self,
        architecture: ModelArchitecture,
        block_size: int = 4,
    ) -> dict[str, float]:
        """Compute grouped sensitivity for blocks of layers.

        Some quantization methods operate on blocks. This helper averages per-layer
        sensitivities within each block.
        """
        per_layer = self.compute(architecture)
        block_scores: dict[str, float] = {}
        for i in range(0, architecture.layers, block_size):
            block_name = f"block_{i // block_size}"
            block_layers = [per_layer.get(f"layer_{j}") for j in range(i, min(i + block_size, architecture.layers))]
            avg_score = sum(ls.score for ls in block_layers) / max(len(block_layers), 1)
            block_scores[block_name] = round(avg_score, 4)
        return block_scores

    def __repr__(self) -> str:
        return f"SensitivityScorer(layers={len(self._calibration_results)})"
