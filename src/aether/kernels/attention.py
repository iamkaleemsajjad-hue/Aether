"""
Attention kernel dispatch.

Provides the base attention interface and dispatches to backend-specific
attention implementations (FlashAttention, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.kernels.base import Kernel
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class AttentionKernel(Kernel):
    """Base class for attention implementations."""

    name = "attention"
    supported_formats: list[str] = ["flash_attention_2", "vanilla", "paged"]

    def __init__(self, head_dim: int = 128, softmax_scale: float | None = None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reference attention forward pass.

        Args:
            query: (batch, heads, seq_len, head_dim)
            key: (batch, heads, kv_len, head_dim)
            value: (batch, heads, kv_len, head_dim)
            mask: optional (seq_len, kv_len) attention mask

        Returns:
            Output tensor (batch, heads, seq_len, head_dim).
        """
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * self.softmax_scale
        if mask is not None:
            scores = scores + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        return np.matmul(attn, value)
