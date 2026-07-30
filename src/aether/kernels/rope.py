"""
Rotary Position Embedding (RoPE) kernels.

Provides position encoding for attention queries and keys using rotary
embeddings. Supports base RoPE, YaRN, NTK-aware, and linear scaling variants.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.kernels.base import Kernel
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class RoPEKernel(Kernel):
    """Rotary Position Embedding kernel.

    Applies rotary embeddings to query and key tensors. Supports:
    - Standard RoPE (interleaved)
    - YaRN (context extension scaling)
    - NTK-aware (frequency scaling)
    - Linear (simple scaling)
    """

    name = "rope"
    supported_formats = ["standard", "yarn", "ntk", "linear"]

    def __init__(self, theta: float = 10000.0, scaling_type: str = "standard", scaling_factor: float = 1.0, max_position: int = 131072) -> None:
        super().__init__()
        self.theta = theta
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.max_position = max_position

    def compute_freqs(self, head_dim: int, seq_len: int) -> np.ndarray:
        """Compute the frequency tensor for RoPE.

        Args:
            head_dim: Dimension of each attention head.
            seq_len: Sequence length.

        Returns:
            Frequency tensor of shape (seq_len, head_dim // 2).
        """
        positions = np.arange(seq_len, dtype=np.float32)
        dim_indices = np.arange(0, head_dim, 2, dtype=np.float32)
        theta_vals = self.theta ** (-dim_indices / head_dim)
        if self.scaling_type == "yarn":
            ratio = seq_len / self.max_position
            if ratio >= 1.0:
                theta_vals *= ratio ** (dim_indices / (head_dim - 2))
        elif self.scaling_type == "ntk":
            theta_vals *= (self.scaling_factor ** (head_dim / (head_dim - 2)))
        elif self.scaling_type == "linear":
            positions = positions / self.scaling_factor
        freqs = np.outer(positions, theta_vals)
        return freqs

    def apply(self, x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Apply rotary embeddings to a tensor.

        Args:
            x: Input tensor of shape (..., seq_len, head_dim).
            freqs: Frequency tensor of shape (seq_len, head_dim // 2).

        Returns:
            Rotated tensor of same shape as x.
        """
        seq_len = x.shape[-2]
        head_dim = x.shape[-1]
        x_reshaped = x.reshape(*x.shape[:-1], head_dim // 2, 2)
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]
        f = freqs[:seq_len, :] if len(freqs) >= seq_len else np.pad(freqs, ((0, seq_len - len(freqs)), (0, 0)))
        cos = np.cos(f).reshape(1, 1, seq_len, -1)
        sin = np.sin(f).reshape(1, 1, seq_len, -1)
        rotated0 = x0 * cos - x1 * sin
        rotated1 = x0 * sin + x1 * cos
        result = np.stack([rotated0, rotated1], axis=-1)
        return result.reshape(*x.shape)

    def batch_apply(self, q: np.ndarray, k: np.ndarray, seq_len: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Apply RoPE to both query and key tensors.

        Args:
            q: Query tensor (batch, heads, seq_len, head_dim).
            k: Key tensor (batch, heads, kv_len, head_dim).
            seq_len: Current sequence length.
            head_dim: Head dimension.

        Returns:
            Tuple of (rotated_query, rotated_key).
        """
        freqs = self.compute_freqs(head_dim, seq_len)
        q_rot = self.apply(q, freqs)
        k_rot = self.apply(k, freqs) if k.shape[-2] == seq_len else self.apply(k, freqs[:k.shape[-2]])
        return q_rot, k_rot
