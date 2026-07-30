"""
Perplexity evaluation harness.

Provides a lightweight perplexity estimator for use during sensitivity analysis.
A full implementation would integrate with the active backend; this module
provides a reference estimator using a simple language model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math
from collections import Counter

import numpy as np

from aether.compiler.calibration.datasets import CalibrationDataset
from aether.core.exceptions import CalibrationError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerplexityResult:
    """Result of a perplexity evaluation."""

    perplexity: float
    loss: float
    num_tokens: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "perplexity": self.perplexity,
            "loss": self.loss,
            "num_tokens": self.num_tokens,
            "details": self.details,
        }


class PerplexityEvaluator:
    """Deterministic reference perplexity evaluator.

    The production path can delegate to a backend with real next-token logits.
    This implementation provides a deterministic proxy that is still grounded
    in dataset statistics: it estimates corpus entropy, adjusts for model
    capacity, and applies precision penalties per protected or compressed layer.
    """

    def __init__(self, vocab_size: int = 32000, model_params_b: float = 1.0) -> None:
        self.vocab_size = vocab_size
        self.model_params_b = model_params_b

    def evaluate(self, dataset: CalibrationDataset, precision_map: dict[str, str] | None = None) -> PerplexityResult:
        """Evaluate perplexity on a calibration dataset.

        Args:
            dataset: Calibration dataset.
            precision_map: Precision map; if provided, adjusts perplexity estimate.

        Returns:
            PerplexityResult.
        """
        tokens = self._collect_tokens(dataset)
        token_count = len(tokens)
        if token_count == 0:
            msg = f"Calibration dataset {dataset.name!r} produced no tokens"
            raise CalibrationError(msg)

        empirical_entropy = self._empirical_cross_entropy(tokens)
        capacity_bonus = math.log1p(max(self.model_params_b, 0.01)) * 0.18
        base_loss = max(0.35, empirical_entropy - capacity_bonus)

        precision_penalty = self._precision_penalty(precision_map or {})
        calibrated_loss = base_loss + precision_penalty
        perplexity = float(math.exp(calibrated_loss))
        return PerplexityResult(
            perplexity=perplexity,
            loss=float(calibrated_loss),
            num_tokens=token_count,
            details={
                "model_params_b": self.model_params_b,
                "dataset": dataset.name,
                "empirical_entropy": float(empirical_entropy),
                "capacity_bonus": float(capacity_bonus),
                "precision_penalty": float(precision_penalty),
            },
        )

    def _collect_tokens(self, dataset: CalibrationDataset) -> list[str]:
        """Collect normalized tokens from the bounded dataset view."""
        tokens: list[str] = []
        iterator = dataset.iter_limited_text() if hasattr(dataset, "iter_limited_text") else dataset.iter_text()
        for text in iterator:
            tokens.extend(token.strip(".,;:!?()[]{}\"'").lower() for token in text.split())
        return [token for token in tokens if token]

    def _empirical_cross_entropy(self, tokens: list[str]) -> float:
        """Estimate corpus cross entropy with add-one smoothing."""
        counts = Counter(tokens)
        vocab = max(len(counts), 1)
        total = len(tokens) + vocab
        losses = [-math.log((counts[token] + 1) / total) for token in tokens]
        return float(np.mean(losses)) if losses else math.log(self.vocab_size)

    def _precision_penalty(self, precision_map: dict[str, str]) -> float:
        """Estimate loss introduced by a mixed precision assignment."""
        if not precision_map:
            return 0.0
        bit_widths = {
            "BF16": 16,
            "FP16": 16,
            "FP32": 32,
            "FP8": 8,
            "FP8_E4M3": 8,
            "FP8_E5M2": 8,
            "Q8_0": 8,
            "Q6_K": 6,
            "Q4_K_M": 4,
            "Q4_0": 4,
            "Q3_K": 3,
            "Q3_K_S": 3,
            "IQ3_XS": 3,
            "Q2_K": 2,
            "Q2_K_S": 2,
            "INT4": 4,
            "INT8": 8,
        }
        penalties: list[float] = []
        for layer_name, precision in precision_map.items():
            bits = bit_widths.get(precision.upper(), 16)
            compression = max(0.0, (16.0 - bits) / 16.0)
            layer_weight = self._layer_importance(layer_name)
            penalties.append((compression ** 1.35) * layer_weight * 0.18)
        return float(sum(penalties) / max(len(penalties), 1))

    def _layer_importance(self, layer_name: str) -> float:
        """Return a stable heuristic importance score for a named layer."""
        normalized = layer_name.lower()
        if normalized in {"embedding", "lm_head"}:
            return 2.0
        if "attn" in normalized or "q_proj" in normalized or "k_proj" in normalized:
            return 1.35
        if "ffn" in normalized or "mlp" in normalized or "down_proj" in normalized:
            return 0.85
        if normalized.startswith("layer_"):
            try:
                index = int(normalized.rsplit("_", 1)[1])
            except ValueError:
                index = 0
            return 1.15 if index < 2 else 1.0
        return 1.0

    def compare(self, baseline: PerplexityResult, quantized: PerplexityResult) -> dict[str, Any]:
        """Compare baseline and quantized perplexity results."""
        delta = quantized.perplexity - baseline.perplexity
        relative = delta / max(baseline.perplexity, 1e-6)
        return {
            "baseline_ppl": baseline.perplexity,
            "quantized_ppl": quantized.perplexity,
            "absolute_delta": delta,
            "relative_delta": relative,
            "acceptable": relative < 0.02,
        }

    def __repr__(self) -> str:
        return f"PerplexityEvaluator(vocab_size={self.vocab_size}, params_b={self.model_params_b})"
