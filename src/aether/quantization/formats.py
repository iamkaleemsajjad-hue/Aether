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

from aether.core.constants import PRECISION_BITS, PRECISION_SIZES_BYTES
from aether.core.types import DType, Precision
from aether.quantization.codecs import Codec, PassthroughCodec, get_codec
from aether.quantization.packing import BitPacker


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

    Stores the bit-packed quantized payload along with the per-block scale and
    zero-point metadata required for dequantization. ``data`` holds packed bytes
    for sub-8-bit formats and plain arrays otherwise; ``num_elements`` records the
    logical element count so unpacking can trim padding.
    """

    precision: str
    shape: tuple[int, ...]
    data: np.ndarray
    scales: np.ndarray | None = None
    zero_points: np.ndarray | None = None
    block_size: int = 32
    #: Bits per element for the stored payload.
    bits: int = 8
    #: True when ``data`` is bit-packed and must be unpacked before decoding.
    packed: bool = False
    #: Logical element count (``prod(shape)``), needed to trim unpack padding.
    num_elements: int = 0
    #: Extra codec state (e.g. FP8 variant) preserved for reconstruction.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.num_elements:
            self.num_elements = int(np.prod(self.shape)) if self.shape else 0

    @property
    def original_size_bytes(self) -> int:
        """Size of the original BF16/FP16 tensor in bytes."""
        return int(np.prod(self.shape)) * 2

    @property
    def payload_bytes(self) -> int:
        """Size of the quantized payload alone, excluding block metadata."""
        return int(self.data.nbytes)

    @property
    def metadata_bytes(self) -> int:
        """Size of the per-block scale and zero-point metadata."""
        total = int(self.scales.nbytes) if self.scales is not None else 0
        if self.zero_points is not None:
            total += int(self.zero_points.nbytes)
        return total

    @property
    def compressed_size_bytes(self) -> int:
        """Compressed size including per-block metadata overhead."""
        return self.payload_bytes + self.metadata_bytes

    @property
    def compression_ratio(self) -> float:
        """Actual compression ratio achieved versus BF16, metadata included."""
        if self.compressed_size_bytes <= 0:
            return 1.0
        return self.original_size_bytes / self.compressed_size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "shape": list(self.shape),
            "bits": self.bits,
            "packed": self.packed,
            "block_size": self.block_size,
            "num_elements": self.num_elements,
            "data_bytes": self.payload_bytes,
            "scales_bytes": int(self.scales.nbytes) if self.scales is not None else 0,
            "zero_points_bytes": int(self.zero_points.nbytes) if self.zero_points is not None else 0,
            "compression_ratio": round(self.compression_ratio, 4),
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            f"QuantizedTensor({self.precision}, shape={self.shape}, "
            f"{self.bits}-bit, ratio={self.compression_ratio:.2f}x)"
        )


#: Largest magnitude representable by float16, used to guard scale downcasting.
_FP16_MAX = 65504.0
#: Smallest positive normal float16; scales below this would underflow to zero.
_FP16_MIN_NORMAL = 6.104e-05


def _compact_block_metadata(values: np.ndarray) -> np.ndarray:
    """Downcast per-block scales/zero-points to float16 when it is lossless enough.

    Production block formats (Q4_K_M, Q8_0, NF4) store fp16 scales, which halves
    metadata overhead. Downcasting is skipped when any value would overflow or
    underflow float16, so pathological weight ranges stay exact.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float16)
    magnitude = np.abs(arr)
    nonzero = magnitude[magnitude > 0]
    if nonzero.size and (nonzero.max() > _FP16_MAX or nonzero.min() < _FP16_MIN_NORMAL):
        return arr
    return arr.astype(np.float16)


def _pad_to_blocks(flat: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    """Reshape a flat array into ``(num_blocks, block_size)``, zero-padding the tail.

    Padding with zeros is safe for every codec because all of them preserve zero
    exactly, so padded lanes never perturb a block's scale beyond its true absmax.
    """
    n = flat.size
    num_blocks = (n + block_size - 1) // block_size
    padded_len = num_blocks * block_size
    if padded_len != n:
        flat = np.concatenate([flat, np.zeros(padded_len - n, dtype=np.float32)])
    return flat.reshape(num_blocks, block_size), n


def quantize_tensor(weights: np.ndarray, precision: str, block_size: int = 32) -> QuantizedTensor:
    """Quantize a floating-point weight tensor to a quantized format.

    Dispatches to the codec registered for ``precision`` (see
    :mod:`aether.quantization.codecs`), then bit-packs the resulting codes for
    sub-8-bit formats so the stored payload reflects the format's true bit width.

    Args:
        weights: Float weight tensor of any shape.
        precision: Target precision identifier, e.g. ``"Q4_K_M"``, ``"NF4"``, ``"FP8"``.
        block_size: Elements per quantization block for block-scaled formats.

    Returns:
        A :class:`QuantizedTensor` carrying the packed payload and block metadata.

    Raises:
        ValueError: If ``precision`` is unknown or ``block_size`` is not positive.
    """
    if block_size <= 0:
        msg = f"block_size must be positive, got {block_size}"
        raise ValueError(msg)

    codec = get_codec(precision)
    arr = np.ascontiguousarray(weights, dtype=np.float32)

    # Float passthrough formats are stored densely, not block-quantized. BF16 is
    # persisted as the high 16 bits of the float32 pattern so a 16-bit format
    # actually occupies 16 bits on disk.
    if isinstance(codec, PassthroughCodec):
        rounded = codec.round_to_format(arr)
        if codec.name == "FP16":
            stored: np.ndarray = rounded.astype(np.float16)
        elif codec.name == "BF16":
            stored = (rounded.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        else:
            stored = rounded
        return QuantizedTensor(
            precision=precision,
            shape=tuple(weights.shape),
            data=stored,
            scales=None,
            zero_points=None,
            block_size=1,
            bits=codec.bits,
            packed=False,
            num_elements=int(arr.size),
            metadata={"codec": codec.name, "storage": str(stored.dtype)},
        )

    blocks, n = _pad_to_blocks(arr.ravel(), block_size)
    codes, scales, zero_points = codec.encode(blocks)

    # Bit-pack sub-8-bit codes so compression is real rather than nominal.
    packed = False
    payload: np.ndarray = codes.reshape(-1)
    if codec.packable and codec.bits < 8:
        payload = BitPacker(codec.bits).pack(payload)
        packed = True

    # Asymmetric zero points are integers by construction; storing them as int16
    # keeps `code == zero_point` decoding to exactly 0.0 regardless of how much
    # precision the scale loses, which is what preserves pruned zeros.
    stored_scales = _compact_block_metadata(scales)
    stored_zero_points = zero_points.astype(np.int16) if codec.asymmetric else None

    return QuantizedTensor(
        precision=precision,
        shape=tuple(weights.shape),
        data=payload,
        scales=stored_scales,
        zero_points=stored_zero_points,
        block_size=block_size,
        bits=codec.bits,
        packed=packed,
        num_elements=int(n),
        metadata={"codec": codec.name, "padded_elements": int(blocks.size - n)},
    )


def dequantize_tensor(qt: QuantizedTensor) -> np.ndarray:
    """Dequantize a quantized tensor back to approximate float32.

    Inverts :func:`quantize_tensor`: unpacks the bit-packed payload when needed,
    decodes through the format's codec, then trims block padding and restores the
    original shape.

    Args:
        qt: The tensor produced by :func:`quantize_tensor`.

    Returns:
        A float32 array with the same shape as the original weights.
    """
    codec = get_codec(qt.precision)

    if isinstance(codec, PassthroughCodec):
        data = np.asarray(qt.data)
        if codec.name == "BF16" and data.dtype == np.uint16:
            # Re-expand the stored high 16 bits into a full float32 pattern.
            widened = (data.astype(np.uint32) << np.uint32(16)).view(np.float32)
            return widened.reshape(qt.shape)
        return data.astype(np.float32).reshape(qt.shape)

    total_elements = qt.num_elements or int(np.prod(qt.shape))
    num_blocks = (total_elements + qt.block_size - 1) // qt.block_size
    padded_len = num_blocks * qt.block_size

    codes = (
        BitPacker(qt.bits).unpack(qt.data, padded_len)
        if qt.packed
        else np.asarray(qt.data).reshape(-1)[:padded_len]
    )
    codes = codes.reshape(num_blocks, qt.block_size)

    scales = qt.scales if qt.scales is not None else np.ones(num_blocks, dtype=np.float32)
    if qt.zero_points is not None:
        zero_points = np.asarray(qt.zero_points, dtype=np.float32)
    else:
        # Codecs that do not persist zero points reconstruct them from the codec
        # itself (symmetric codes are stored biased by 2^(bits-1)).
        zero_points = codec.default_zero_points(num_blocks)

    decoded = codec.decode(codes, np.asarray(scales, dtype=np.float32), zero_points)
    return decoded.reshape(-1)[:total_elements].reshape(qt.shape)
