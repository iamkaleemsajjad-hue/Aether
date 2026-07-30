"""
Normalization kernels (RMSNorm, LayerNorm).

Provides fused normalization implementations with optional weight and bias.
Used in transformer feed-forward and attention blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.kernels.base import Kernel
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class RMSNormKernel(Kernel):
    """Root Mean Square Layer Normalization.

    Computes: out = x * rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps).
    """

    name = "rms_norm"
    supported_formats = ["rms_norm", "rms_norm_quant"]

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """Apply RMSNorm.

        Args:
            x: Input tensor. Last dimension is normalized.
            weight: Optional learnable weight of shape (hidden_size,).

        Returns:
            Normalized tensor, same shape as x.
        """
        variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        rms = np.sqrt(variance + self.eps)
        x_normed = x.astype(np.float32) / rms
        if weight is not None:
            x_normed = x_normed * weight.astype(np.float32)
        return x_normed.astype(x.dtype)


class LayerNormKernel(Kernel):
    """Standard Layer Normalization.

    Computes: out = (x - mean) / sqrt(var + eps) * weight + bias.
    """

    name = "layer_norm"
    supported_formats = ["layer_norm"]

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: np.ndarray, weight: np.ndarray | None = None, bias: np.ndarray | None = None) -> np.ndarray:
        """Apply LayerNorm.

        Args:
            x: Input tensor. Last dimension is normalized.
            weight: Optional learnable weight of shape (hidden_size,).
            bias: Optional learnable bias of shape (hidden_size,).

        Returns:
            Normalized tensor.
        """
        mean = np.mean(x.astype(np.float32), axis=-1, keepdims=True)
        var = np.var(x.astype(np.float32), axis=-1, keepdims=True)
        x_normed = (x.astype(np.float32) - mean) / np.sqrt(var + self.eps)
        if weight is not None:
            x_normed = x_normed * weight.astype(np.float32)
        if bias is not None:
            x_normed = x_normed + bias.astype(np.float32)
        return x_normed.astype(x.dtype)
