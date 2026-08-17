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
    """Calibrates per-layer sensitivity scores.

    Two scoring methods are supported, selected by data availability:

    ``weight_reconstruction_error``
        Preferred.  Quantizes each layer's *real* bound weight tensors to the
        probe precision and measures the relative reconstruction error.  This
        is computed from the actual checkpoint weights the graph carries and
        is the standard one-shot approximation when activation Hessians are
        unavailable (cf. GPTQ/AWQ weight-error proxies).

    ``text_entropy_proxy``
        Fallback for weightless graphs.  A deterministic text-entropy
        perplexity delta combined with documented architectural priors.  The
        method is always recorded so consumers can distinguish measured from
        estimated scores.
    """

    #: Probe precision used to measure per-layer quantization damage.
    PROBE_PRECISION = "Q4_K_M"

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
        layer_weights: dict[str, list[np.ndarray]] | None = None,
    ) -> dict[str, float]:
        """Score each layer by quantizing only that layer.

        Returns a dictionary mapping layer name to sensitivity score, where a
        higher score indicates that quantizing the layer hurts perplexity more.
        Weight-derived scores combine the measured relative reconstruction
        error of the layer's real tensors with transformer priors: early/late
        layers, attention projections, embeddings, and LM heads get stronger
        protection than middle FFN-heavy layers.
        """
        scores: dict[str, float] = {}
        if layer_weights:
            for i in range(self.architecture.layers):
                layer_name = f"layer_{i}"
                tensors = layer_weights.get(layer_name) or []
                error = self._mean_relative_error(tensors)
                score = self._normalize_weight_error(error) * self._architectural_prior(i)
                scores[layer_name] = min(1.0, max(0.0, score))
            for special, fallback in (("embedding", 0.98), ("lm_head", 0.97)):
                tensors = layer_weights.get(special) or []
                if tensors:
                    error = self._mean_relative_error(tensors)
                    scores[special] = min(1.0, max(0.0, self._normalize_weight_error(error)))
                else:
                    # Documented conservative prior: the PRD classifies the
                    # embedding and output projection as the most protected
                    # tensors when no weights are available to measure.
                    scores[special] = fallback
            return scores

        baseline = self.evaluator.evaluate(dataset, base_precision_map)
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

    @classmethod
    def scoring_method(cls, layer_weights: dict[str, list[np.ndarray]] | None) -> str:
        """Return the human-readable method name that ``score_by_layer`` uses."""
        return "weight_reconstruction_error" if layer_weights else "text_entropy_proxy"

    #: Relative reconstruction error at which a tensor is fully sensitive.
    #: Q4_K_M block quantization yields ~3-6% error on well-behaved tensors
    #: and >12% on outlier-heavy (sensitive) tensors, so 12% maps to 1.0.
    WEIGHT_ERROR_FULL_SENSITIVITY = 0.12

    @classmethod
    def _normalize_weight_error(cls, error: float) -> float:
        """Map a relative reconstruction error linearly to [0, 1]."""
        return min(1.0, max(0.0, error / cls.WEIGHT_ERROR_FULL_SENSITIVITY))

    def _mean_relative_error(self, tensors: list[np.ndarray]) -> float:
        """Mean relative Frobenius reconstruction error under the probe precision.

        Quantizes each tensor with the production codec and measures how much
        of the original signal survives; tensors that cannot be probed are
        skipped rather than assigned a fabricated error.
        """
        from aether.quantization.formats import dequantize_tensor, quantize_tensor

        errors: list[float] = []
        for tensor in tensors:
            array = np.asarray(tensor, dtype=np.float32)
            if array.size < 2 or not np.isfinite(array).all():
                continue
            try:
                reconstructed = dequantize_tensor(quantize_tensor(array, self.PROBE_PRECISION))
            except Exception:  # noqa: BLE001 - unsupported shape/precision: skip
                continue
            denominator = float(np.linalg.norm(array))
            if denominator <= 0.0:
                continue
            errors.append(float(np.linalg.norm(array - reconstructed)) / denominator)
        return float(np.mean(errors)) if errors else 0.0

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
