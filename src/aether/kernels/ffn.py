"""
Fused FFN (feed-forward network) kernel.

Provides a single fused kernel for the transformer FFN block: gate, up, SiLU,
and down projections merged into a single operation to reduce kernel launch
overhead and intermediate memory bandwidth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.kernels.base import Kernel
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class FFNKernel(Kernel):
    """Fused feed-forward network kernel.

    Supports SwiGLU, GeGLU, and vanilla GELU activations. The fused
    implementation computes gate(x) * silu(up(x)) in a single pass,
    then multiplies by down(x).
    """

    name = "ffn"
    supported_formats = ["swiglu", "geglu", "gelu"]

    def __init__(self, activation: str = "swiglu") -> None:
        super().__init__()
        self.activation = activation.lower()

    def forward(self, x: np.ndarray, gate_weight: np.ndarray | None = None, up_weight: np.ndarray | None = None, down_weight: np.ndarray | None = None) -> np.ndarray:
        """Execute the fused FFN.

        Args:
            x: Input tensor (batch, hidden_size).
            gate_weight: Gate projection weight (hidden_size, intermediate_size).
            up_weight: Up projection weight (hidden_size, intermediate_size).
            down_weight: Down projection weight (intermediate_size, hidden_size).

        Returns:
            Output tensor (batch, hidden_size).
        """
        if gate_weight is not None and up_weight is not None:
            gate = np.matmul(x, gate_weight.T)
            up = np.matmul(x, up_weight.T)
        else:
            gate = x
            up = x
        activated = self._activate(up)
        hidden = gate * activated
        if down_weight is not None:
            return np.matmul(hidden, down_weight.T)
        return hidden

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured activation function."""
        if self.activation == "swiglu":
            return x / (1 + np.exp(-x))  # SiLU / Swish
        if self.activation == "geglu":
            return x * 0.5 * (1 + np.math.erf(x / np.sqrt(2.0)))
        return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
