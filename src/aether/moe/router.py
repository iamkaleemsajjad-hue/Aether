"""
MoE router — threshold-based expert routing.

Implements the dynamic threshold-based routing described in the DynaMoE
research. Routes each token to a subset of experts based on activation
thresholds rather than a fixed top-K.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.core.constants import MOE_HOT_THRESHOLD, MOE_WARM_THRESHOLD
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ThresholdRouter:
    """Dynamically routes tokens to experts using activation thresholds.

    Unlike static top-K routing (always route to exactly K experts), threshold
    routing sends each token to any expert whose gate score exceeds a learned
    or configured threshold. This naturally adapts: easy tokens use fewer
    experts, hard tokens use more.
    """

    def __init__(self, num_experts: int, hot_threshold: float = MOE_HOT_THRESHOLD, warm_threshold: float = MOE_WARM_THRESHOLD) -> None:
        self.num_experts = num_experts
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self._activation_counts: np.ndarray = np.zeros(num_experts, dtype=np.int64)
        self._total_tokens: int = 0

    def compute_gates(self, logits: np.ndarray) -> np.ndarray:
        """Compute gate scores using softmax over expert logits.

        Args:
            logits: Expert logits of shape (batch, num_experts).

        Returns:
            Gate probability distribution of the same shape.
        """
        max_logits = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def route(self, gates: np.ndarray) -> list[list[int]]:
        """Route each token to experts whose gate score exceeds the threshold.

        Args:
            gates: Gate probabilities of shape (batch, num_experts).

        Returns:
            List of expert indices per token in the batch.
        """
        batch = gates.shape[0]
        routing: list[list[int]] = []
        for i in range(batch):
            scores = gates[i]
            mask = scores >= self.hot_threshold
            if not mask.any():
                mask[scores.argmax()] = True
            experts = mask.nonzero()[0].tolist()
            routing.append(experts)
        self._total_tokens += batch
        for token_experts in routing:
            for e_idx in token_experts:
                self._activation_counts[e_idx] += 1
        return routing

    def expert_activation_rates(self) -> list[float]:
        """Return per-expert activation rates in [0, 1]."""
        if self._total_tokens == 0:
            return [0.0] * self.num_experts
        return [float(count) / max(self._total_tokens, 1) for count in self._activation_counts]

    def classify_experts(self) -> dict[str, list[int]]:
        """Classify experts into hot/warm/cold tiers based on activation rates."""
        rates = self.expert_activation_rates()
        hot: list[int] = []
        warm: list[int] = []
        cold: list[int] = []
        for i, rate in enumerate(rates):
            if rate >= self.hot_threshold:
                hot.append(i)
            elif rate >= self.warm_threshold:
                warm.append(i)
            else:
                cold.append(i)
        return {"hot": hot, "warm": warm, "cold": cold}

    def reset(self) -> None:
        """Reset activation counters."""
        self._activation_counts.fill(0)
        self._total_tokens = 0

    def __repr__(self) -> str:
        tiers = self.classify_experts()
        return f"ThresholdRouter({self.num_experts} experts, hot={len(tiers['hot'])}, warm={len(tiers['warm'])}, cold={len(tiers['cold'])})"
