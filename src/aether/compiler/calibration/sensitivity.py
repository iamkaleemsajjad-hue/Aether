"""
Sensitivity scoring helpers.

Provides a calibration-driven sensitivity scoring function that compares the
perplexity impact of quantizing each layer individually. This is used by Pass 2
of the optimizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.compiler.calibration.datasets import CalibrationDataset
from aether.compiler.calibration.perplexity import PerplexityEvaluator
from aether.core.types import ModelArchitecture
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class SensitivityCalibration:
    """Calibrates per-layer sensitivity scores using perplexity deltas."""

    def __init__(self, architecture: ModelArchitecture, evaluator: PerplexityEvaluator | None = None) -> None:
        self.architecture = architecture
        self.evaluator = evaluator or PerplexityEvaluator(
            vocab_size=architecture.vocab_size,
            model_params_b=architecture.params_billion,
        )

    def score_by_layer(
        self,
        dataset: CalibrationDataset,
        base_precision_map: dict[str, str],
    ) -> dict[str, float]:
        """Score each layer by quantizing only that layer.

        Returns a dictionary mapping layer name to sensitivity score, where a
        higher score indicates that quantizing the layer hurts perplexity more.
        The score combines measured perplexity delta with transformer priors:
        early/late layers, attention projections, embeddings, and LM heads get
        stronger protection than middle FFN-heavy layers.
        """
        baseline = self.evaluator.evaluate(dataset, base_precision_map)
        scores: dict[str, float] = {}
        for i in range(self.architecture.layers):
            layer_name = f"layer_{i}"
            modified = dict(base_precision_map)
            modified[layer_name] = "Q3_K"
            quantized = self.evaluator.evaluate(dataset, modified)
            delta = max(0.0, quantized.loss - baseline.loss)
            score = self._normalize_delta(delta) * self._architectural_prior(i)
            scores[layer_name] = min(1.0, max(0.0, score))
        scores["embedding"] = 0.98
        scores["lm_head"] = 0.97
        return scores

    def _normalize_delta(self, delta: float) -> float:
        """Map a loss delta to the [0, 1] sensitivity range."""
        return float(1.0 - np.exp(-max(delta, 0.0) * 18.0))

    def _architectural_prior(self, layer_index: int) -> float:
        """Return a transformer-specific prior for layer sensitivity."""
        layers = max(self.architecture.layers - 1, 1)
        position = layer_index / layers
        edge_weight = 1.0 + 0.28 * max(0.0, 1.0 - min(position, 1.0 - position) * 4.0)
        attention_weight = 1.12 if self.architecture.attention_type.upper() in {"GQA", "MLA"} else 1.0
        moe_weight = 1.08 if self.architecture.is_moe else 1.0
        return edge_weight * attention_weight * moe_weight

    def score_by_block(self, dataset: CalibrationDataset, base_precision_map: dict[str, str], block_size: int = 4) -> dict[str, float]:
        """Score blocks of layers by quantizing each block."""
        baseline = self.evaluator.evaluate(dataset, base_precision_map)
        block_scores: dict[str, float] = {}
        for i in range(0, self.architecture.layers, block_size):
            block_name = f"block_{i // block_size}"
            modified = dict(base_precision_map)
            for j in range(i, min(i + block_size, self.architecture.layers)):
                layer_name = f"layer_{j}"
                if layer_name in modified:
                    modified[layer_name] = "Q3_K"
            quantized = self.evaluator.evaluate(dataset, modified)
            delta = max(0.0, quantized.loss - baseline.loss)
            block_scores[block_name] = min(1.0, self._normalize_delta(delta))
        return block_scores

    def score_summary(self, scores: dict[str, float]) -> dict[str, Any]:
        """Return a summary of sensitivity scores."""
        values = list(scores.values())
        return {
            "num_layers": len(scores),
            "mean": float(np.mean(values)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
            "min": float(np.min(values)) if values else 0.0,
            "high_sensitivity_count": sum(1 for v in values if v > 0.7),
        }

    def __repr__(self) -> str:
        return f"SensitivityCalibration({self.architecture.family})"
