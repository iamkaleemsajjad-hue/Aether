"""
Intra-expert sparsity analysis.

Identifies dead or low-impact channels within expert feed-forward networks.
These channels can be pruned or downcast without meaningful quality loss,
improving the effective throughput of MoE inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ExpertSparsityAnalyzer:
    """Analyzes per-expert weight matrices to find sparsity patterns.

    Identifies:
    - Dead channels (weights near zero for all inputs)
    - Low-magnitude channels (below a relative threshold)
    - Activation sparsity patterns across calibration tokens
    """

    def __init__(self, sparsity_threshold: float = 0.01) -> None:
        self.sparsity_threshold = sparsity_threshold
        self._channel_activity: dict[str, np.ndarray] = {}

    def analyze_weights(self, expert_idx: int, gate_weight: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray) -> dict[str, Any]:
        """Analyze sparsity in an expert's FFN weights.

        Args:
            expert_idx: Expert index.
            gate_weight: Gate projection weight of shape (hidden_size, intermediate_size).
            up_weight: Up projection weight of shape (hidden_size, intermediate_size).
            down_weight: Down projection weight of shape (intermediate_size, hidden_size).

        Returns:
            Dictionary with sparsity metrics.
        """
        gate_norms = np.linalg.norm(gate_weight, axis=0)
        up_norms = np.linalg.norm(up_weight, axis=0)
        down_norms = np.linalg.norm(down_weight, axis=1)
        merged_norms = (gate_norms + up_norms) / 2.0
        total = len(merged_norms)
        dead_channels = int((merged_norms < self.sparsity_threshold).sum())
        low_channels = int(((merged_norms >= self.sparsity_threshold) & (merged_norms < self.sparsity_threshold * 10)).sum())
        active_channels = total - dead_channels - low_channels
        return {
            "expert_idx": expert_idx,
            "total_channels": total,
            "dead_channels": dead_channels,
            "low_channels": low_channels,
            "active_channels": active_channels,
            "dead_ratio": dead_channels / max(total, 1),
            "low_ratio": low_channels / max(total, 1),
            "active_ratio": active_channels / max(total, 1),
        }

    def analyze_activations(self, expert_idx: int, activations: np.ndarray) -> dict[str, Any]:
        """Analyze activation sparsity for an expert.

        Args:
            expert_idx: Expert index.
            activations: Activation matrix of shape (num_tokens, intermediate_size).

        Returns:
            Dictionary with activation sparsity metrics.
        """
        if activations.ndim != 2:
            return {"expert_idx": expert_idx, "error": "Expected 2D activation matrix"}
        total_neurons = activations.shape[1]
        zero_neurons = int((activations.max(axis=0) < self.sparsity_threshold).sum())
        rarely_active = int(((activations > self.sparsity_threshold).mean(axis=0) < 0.05).sum())
        self._channel_activity[f"expert_{expert_idx}"] = (activations > self.sparsity_threshold).mean(axis=0)
        return {
            "expert_idx": expert_idx,
            "total_neurons": total_neurons,
            "always_dead": zero_neurons,
            "rarely_active": rarely_active,
            "dead_ratio": zero_neurons / max(total_neurons, 1),
            "rarely_active_ratio": rarely_active / max(total_neurons, 1),
        }

    def sparsity_report(self) -> dict[str, Any]:
        """Return a global sparsity report across all analyzed experts."""
        total_channels = 0
        total_dead = 0
        for name, activity in self._channel_activity.items():
            total_channels += len(activity)
            total_dead += int((activity < self.sparsity_threshold).sum())
        return {
            "experts_analyzed": len(self._channel_activity),
            "total_channels": total_channels,
            "total_dead_channels": total_dead,
            "overall_dead_ratio": total_dead / max(total_channels, 1),
            "threshold": self.sparsity_threshold,
        }

    def __repr__(self) -> str:
        return f"ExpertSparsityAnalyzer(threshold={self.sparsity_threshold})"
