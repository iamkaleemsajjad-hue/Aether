"""
Quantization formats and data structures.

Defines the quantization format identifier, quantized tensor representation,
and helper functions for quantize/dequantize operations. The actual quantization
kernels live in the backend plugins; this module provides the Python-level
format descriptors and lightweight reference implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.core.constants import PRECISION_SIZES_BYTES
from aether.core.types import DType, Precision


class QuantizationFormat:
    """Describes a quantization format with its properties.

    Each format has a name, bit width, block size (for block-quantized formats),
    and a flag indicating whether it is symmetric or asymmetric.
    """

    def __init__(self, precision: str) -> None:
        self.precision = precision
        self.bit_width = PRECISION_SIZES_BYTES.get(precision.upper(), 2.0) * 8
        self.is_quantized = precision.upper().startswith("Q") or precision.upper().startswith("I")

    @property
    def name(self) -> str:
        return self.precision

    @property
    def block_size(self) -> int:
        """Return the typical block size for this format.

        Block-quantized formats like Q4_K_M group weights into blocks.
        Unquantized formats return 1 (no grouping).
        """
        if not self.is_quantized:
            return 1
        return 32 if "Q4" in self.precision or "Q3" in self.precision else 64

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio relative to BF16 (16 bits)."""
        return 16.0 / self.bit_width if self.bit_width > 0 else 1.0

    def __repr__(self) -> str:
        return f"QuantizationFormat({self.precision}, {self.bit_width:.1f} bits)"

    @staticmethod
    def from_precision(p: Precision) -> QuantizationFormat:
        return QuantizationFormat(p.value)

    @staticmethod
    def from_string(value: str) -> QuantizationFormat:
        return QuantizationFormat(value)


@dataclass
class QuantizedTensor:
    """A quantized weight tensor.

    Stores the packed quantized data along with scale and zero-point metadata
    needed for dequantization.
    """

    precision: str
    shape: tuple[int, ...]
    data: np.ndarray
    scales: np.ndarray | None = None
    zero_points: np.ndarray | None = None
    block_size: int = 32

    @property
    def original_size_bytes(self) -> int:
        """Size of the original FP16/BF16 tensor in bytes."""
        return int(np.prod(self.shape)) * 2

    @property
    def compressed_size_bytes(self) -> int:
        """Compressed size including metadata overhead."""
        return int(self.data.nbytes) + (self.scales.nbytes if self.scales is not None else 0)

    @property
    def compression_ratio(self) -> float:
        """Actual compression ratio achieved."""
        if self.compressed_size_bytes <= 0:
            return 1.0
        return self.original_size_bytes / self.compressed_size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "shape": list(self.shape),
            "data_bytes": int(self.data.nbytes),
            "scales_bytes": int(self.scales.nbytes) if self.scales is not None else 0,
            "block_size": self.block_size,
        }

    def __repr__(self) -> str:
        return f"QuantizedTensor({self.precision}, shape={self.shape}, ratio={self.compression_ratio:.2f}x)"


def quantize_tensor(weights: np.ndarray, precision: str, block_size: int = 32) -> QuantizedTensor:
    """Quantize a floating-point weight tensor to a quantized format.

    This is a reference implementation using block-wise quantization. Real
    deployments use backend-specific quantization kernels.
    """
    original_dtype = weights.dtype
    flat = weights.astype(np.float32).ravel()
    n = len(flat)
    num_blocks = (n + block_size - 1) // block_size
    scales = np.zeros(num_blocks, dtype=np.float16)
    qdata = np.zeros(n, dtype=np.int8)

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = flat[start:end]
        scale = block.max() / 127.0 if block.max() > 1e-8 else 1.0
        scales[i] = scale
        qdata[start:end] = np.clip(np.round(block / scale), -128, 127).astype(np.int8)

    qdata = qdata.reshape(weights.shape)
    return QuantizedTensor(
        precision=precision,
        shape=weights.shape,
        data=qdata,
        scales=scales,
        block_size=block_size,
    )


def dequantize_tensor(qt: QuantizedTensor) -> np.ndarray:
    """Dequantize a quantized tensor back to approximate float.

    This is a reference implementation for testing and validation.
    """
    flat = qt.data.astype(np.float32).ravel()
    n = len(flat)
    result = np.zeros(n, dtype=np.float32)
    for i in range((n + qt.block_size - 1) // qt.block_size):
        start = i * qt.block_size
        end = min(start + qt.block_size, n)
        if qt.scales is not None and i < len(qt.scales):
            result[start:end] = flat[start:end] * qt.scales[i]
        else:
            result[start:end] = flat[start:end]
    return result.reshape(qt.shape)
