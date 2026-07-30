"""
Quantized GEMM kernel dispatch.

Provides a dispatch layer for matrix multiplication across precision formats.
For each precision combination (BF16*Q4_K_M, Q4_K_M*K4_K_M, etc.), dispatches
to the appropriate backend-specific or reference implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.kernels.base import Kernel
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class GEMMKernel(Kernel):
    """General matrix multiply for various precision formats."""

    name = "gemm"
    supported_formats = ["bf16", "q4_k_m", "q8_0", "int8", "int4"]

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: np.ndarray, b: np.ndarray, dtype: str = "bf16") -> np.ndarray:
        """Dispatch GEMM by dtype.

        Args:
            a: Left operand.
            b: Right operand.
            dtype: Precision format string.

        Returns:
            GEMM result.
        """
        if dtype.upper() == "BF16":
            return self._bf16_gemm(a, b)
        if dtype.upper() in ("Q4_K_M", "Q8_0", "INT8", "INT4"):
            return self._quantized_gemm(a, b)
        msg = f"Unsupported GEMM dtype: {dtype}"
        raise ValueError(msg)

    def _bf16_gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """BF16 reference matmul."""
        return np.matmul(a, b.T) if a.ndim == 2 and b.ndim == 2 else np.matmul(a, b)

    def _quantized_gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Quantized reference matmul (dequantize then multiply)."""
        return np.matmul(a.astype(np.float32), b.T.astype(np.float32)) if a.ndim == 2 else np.matmul(a, b)
