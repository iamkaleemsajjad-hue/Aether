"""
Numerical quantization codecs.

Each codec implements genuine per-format encode/decode math rather than a shared
int8 path. Codecs are *vectorized over blocks*: they accept a 2-D
``(num_blocks, block_size)`` float32 array and return integer codes plus the
per-block metadata required to reconstruct an approximation of the input.

Format families
---------------
* :class:`AffineIntCodec`    — asymmetric (zero-point) integer grids: Q4_K_M, Q3_K, Q2_K, Q6_K.
* :class:`SymmetricIntCodec` — symmetric absmax integer grids: INT8, Q8_0, INT4, Q4_0.
* :class:`NF4Codec`          — NormalFloat4, the information-theoretically optimal
  4-bit grid for normally distributed weights (QLoRA, Dettmers et al.).
* :class:`FP8Codec`          — OCP FP8 minifloat, E4M3 and E5M2 layouts.
* :class:`FP4Codec`          — E2M1 minifloat used by NVFP4/MXFP4 Blackwell targets.
* :class:`PassthroughCodec`  — BF16/FP16/FP32, which round rather than quantize.

Every codec satisfies two invariants, both asserted by the test suite:

1. **Grid idempotence** — decoding then re-encoding a value already on the
   codec's grid reproduces it exactly.
2. **Zero preservation** — an all-zero block roundtrips to exactly zero, which
   matters because pruning masks introduce structural zeros.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

__all__ = [
    "Codec",
    "AffineIntCodec",
    "SymmetricIntCodec",
    "NF4Codec",
    "FP8Codec",
    "FP4Codec",
    "MXFP4Codec",
    "PassthroughCodec",
    "get_codec",
    "supported_precisions",
    "NF4_LEVELS",
]


# NormalFloat4 levels: the 16 quantiles of a unit normal, normalised to [-1, 1].
# Reproduced from the QLoRA reference implementation.
NF4_LEVELS: np.ndarray = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)

#: Blocks whose dynamic range falls below this are treated as constant/zero.
_EPS = 1e-12


def _safe_divisor(values: np.ndarray) -> np.ndarray:
    """Return ``values`` with non-positive entries replaced by 1 to avoid 0-division."""
    return np.where(np.abs(values) < _EPS, np.float32(1.0), values).astype(np.float32)




class Codec:
    """Base class for a block quantization codec.

    Subclasses convert a ``(num_blocks, block_size)`` float32 array into integer
    codes plus per-block ``scales`` and ``zero_points``, and back again.
    """

    name: str = "codec"
    bits: int = 8
    #: True when codes are unsigned integers and therefore safe to bit-pack.
    packable: bool = True
    #: True when the codec requires a per-block zero point to decode.
    asymmetric: bool = False

    @property
    def levels(self) -> int:
        """Number of representable code points."""
        return 1 << self.bits

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a 2-D block array.

        Args:
            blocks: ``(num_blocks, block_size)`` float32 array.

        Returns:
            Tuple of ``(codes, scales, zero_points)``. ``codes`` has the same
            shape as ``blocks``; ``scales`` and ``zero_points`` have shape
            ``(num_blocks,)``.
        """
        raise NotImplementedError

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        """Decode codes back to a ``(num_blocks, block_size)`` float32 array."""
        raise NotImplementedError

    def default_zero_points(self, num_blocks: int) -> np.ndarray:
        """Zero points to assume when they were not persisted alongside the codes.

        Only :attr:`asymmetric` codecs need their zero points stored; every other
        codec can reconstruct them from the codec definition alone.
        """
        return np.zeros(num_blocks, dtype=np.float32)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, {self.bits}-bit)"


class SymmetricIntCodec(Codec):
    """Symmetric absmax integer quantization.

    The grid is centred on zero and spans ``[-qmax, +qmax]``, so the scale derives
    from each block's maximum *absolute* value. Zero maps exactly to zero, which
    is required for sparse/pruned weights to stay sparse.

    Codes are stored biased into unsigned range (``code + 2^(bits-1)``) so that
    they remain bit-packable.
    """

    def __init__(self, name: str, bits: int) -> None:
        self.name = name
        self.bits = bits
        self.packable = True
        self.asymmetric = False

    @property
    def _qmax(self) -> int:
        # Symmetric range: 4-bit -> [-7, 7], 8-bit -> [-127, 127].
        return (1 << (self.bits - 1)) - 1

    @property
    def _bias(self) -> int:
        return 1 << (self.bits - 1)

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        absmax = np.abs(blocks).max(axis=1)
        degenerate = absmax < _EPS
        scales = (absmax / self._qmax).astype(np.float32)
        codes = np.rint(blocks / _safe_divisor(scales)[:, None])
        codes = np.clip(codes, -self._qmax, self._qmax) + self._bias
        codes[degenerate, :] = self._bias
        scales[degenerate] = 0.0
        zero_points = np.full(blocks.shape[0], float(self._bias), dtype=np.float32)
        return codes.astype(np.uint8), scales, zero_points

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        signed = codes.astype(np.float32) - zero_points[:, None].astype(np.float32)
        return (signed * scales[:, None].astype(np.float32)).astype(np.float32)

    def default_zero_points(self, num_blocks: int) -> np.ndarray:
        """Symmetric codes are stored biased by ``2^(bits-1)``."""
        return np.full(num_blocks, float(self._bias), dtype=np.float32)


class AffineIntCodec(Codec):
    """Asymmetric affine (zero-point) integer quantization.

    Maps each block's ``[min, max]`` onto the full unsigned code range. This
    yields materially lower error than symmetric quantization for skewed or
    entirely single-signed blocks — the failure mode that motivated replacing the
    original shared int8 path. This is the K-quant family behaviour.

    **Integer zero point.** The grid is ``(code - zero_point) * scale`` with an
    *integer* zero point, which is how TFLite and the GGUF K-quants define it. The
    alternative ``code * scale + offset`` form represents 0.0 only when
    ``-offset/scale`` happens to be an integer, and any later rounding of the
    stored scale destroys that cancellation — turning pruned weights into small
    non-zero values and silently breaking 2:4 sparse-tensor-core kernels. With an
    integer zero point, ``code == zero_point`` decodes to exactly 0.0 at *any*
    scale precision.
    """

    #: Zero points are persisted as int16, so they must fit this range.
    _ZERO_POINT_LIMIT = 32767

    def __init__(self, name: str, bits: int) -> None:
        self.name = name
        self.bits = bits
        self.packable = True
        self.asymmetric = True

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qmax = self.levels - 1
        bmin = blocks.min(axis=1).astype(np.float32)
        bmax = blocks.max(axis=1).astype(np.float32)
        scales = ((bmax - bmin) / qmax).astype(np.float32)
        zero_points = np.rint(-bmin / _safe_divisor(scales))

        # A block whose range sits far from zero can need a zero point outside
        # int16. Widening such a block to include zero always brings it back into
        # [0, qmax]; these blocks are rare and hold no zeros to preserve anyway.
        overflow = np.abs(zero_points) > self._ZERO_POINT_LIMIT
        if np.any(overflow):
            wide_min = np.minimum(bmin, np.float32(0.0))
            wide_max = np.maximum(bmax, np.float32(0.0))
            wide_scale = ((wide_max - wide_min) / qmax).astype(np.float32)
            scales = np.where(overflow, wide_scale, scales).astype(np.float32)
            zero_points = np.where(
                overflow, np.rint(-wide_min / _safe_divisor(scales)), zero_points
            )

        codes = np.rint(blocks / _safe_divisor(scales)[:, None]) + zero_points[:, None]
        codes = np.clip(codes, 0, qmax)

        # Constant block: scale carries the value, code 1 against zero point 0.
        degenerate = (bmax - bmin) < _EPS
        if np.any(degenerate):
            scales[degenerate] = bmin[degenerate]
            zero_points[degenerate] = 0.0
            codes[degenerate, :] = 1
        return codes.astype(np.uint8), scales, zero_points.astype(np.float32)

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        centred = codes.astype(np.float32) - zero_points[:, None].astype(np.float32)
        return (centred * scales[:, None].astype(np.float32)).astype(np.float32)


class NF4Codec(Codec):
    """NormalFloat4 quantization.

    Uses the 16 fixed quantiles of a unit normal as the code grid. Because LLM
    weights are approximately normally distributed after absmax normalisation,
    this grid minimises expected error relative to a uniform 4-bit grid.
    """

    name = "NF4"
    bits = 4
    packable = True
    asymmetric = False

    _levels: ClassVar[np.ndarray] = NF4_LEVELS
    #: Midpoints between adjacent levels, used for nearest-level search.
    _boundaries: ClassVar[np.ndarray] = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2.0
    #: Index of the level that is exactly 0.0.
    _zero_index: ClassVar[int] = 7

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        absmax = np.abs(blocks).max(axis=1).astype(np.float32)
        degenerate = absmax < _EPS
        normalised = blocks / _safe_divisor(absmax)[:, None]
        codes = np.searchsorted(self._boundaries, normalised).astype(np.uint8)
        codes[degenerate, :] = self._zero_index
        scales = absmax.copy()
        scales[degenerate] = 0.0
        zero_points = np.zeros(blocks.shape[0], dtype=np.float32)
        return codes, scales, zero_points

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        return (self._levels[codes.astype(np.intp)] * scales[:, None].astype(np.float32)).astype(np.float32)


class FP8Codec(Codec):
    """FP8 minifloat quantization (E4M3 / E5M2).

    Encodes each value as a sign/exponent/mantissa triple with OCP field widths,
    matching the FP8 layouts consumed by Hopper and Blackwell tensor cores. A
    per-block scale is applied first so the block's dynamic range is centred
    inside the format's representable window.
    """

    def __init__(self, variant: str = "E4M3") -> None:
        variant = variant.upper()
        if variant not in ("E4M3", "E5M2"):
            msg = f"Unsupported FP8 variant '{variant}' (expected E4M3 or E5M2)"
            raise ValueError(msg)
        self.variant = variant
        self.name = f"FP8_{variant}"
        self.bits = 8
        self.packable = True
        self.asymmetric = False
        if variant == "E4M3":
            # Stored exponents span 0..15; (exp=15, mant=7) is reserved for NaN,
            # so the largest finite value is (1 + 6/8) * 2^8 = 448.
            self.exp_bits, self.mant_bits, self.exp_bias = 4, 3, 7
            self.max_exp_stored = 15
            self.max_mantissa = 6
        else:
            # Stored exponent 31 is reserved for Inf/NaN, so the largest finite
            # value is (1 + 3/4) * 2^15 = 57344.
            self.exp_bits, self.mant_bits, self.exp_bias = 5, 2, 15
            self.max_exp_stored = 30
            self.max_mantissa = 3
        self.max_value = float(
            (1.0 + self.max_mantissa / (1 << self.mant_bits))
            * 2.0 ** (self.max_exp_stored - self.exp_bias)
        )

    @property
    def _min_normal(self) -> np.float32:
        return np.float32(2.0) ** np.float32(1 - self.exp_bias)

    @property
    def _subnormal_step(self) -> np.float32:
        return self._min_normal / np.float32(1 << self.mant_bits)

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        absmax = np.abs(blocks).max(axis=1).astype(np.float32)
        degenerate = absmax < _EPS
        scales = (absmax / np.float32(self.max_value)).astype(np.float32)
        scaled = blocks / _safe_divisor(scales)[:, None]
        codes = self._to_bits(scaled)
        codes[degenerate, :] = 0
        scales[degenerate] = 0.0
        zero_points = np.zeros(blocks.shape[0], dtype=np.float32)
        return codes, scales, zero_points

    def _to_bits(self, values: np.ndarray) -> np.ndarray:
        """Round values onto the FP8 grid and pack into 8-bit codes."""
        av = np.minimum(np.abs(values), np.float32(self.max_value)).astype(np.float32)
        sign_bit = (values < 0).astype(np.uint8) << 7
        nonzero = av > 0
        exp_stored = np.zeros(av.shape, dtype=np.int32)
        mantissa = np.zeros(av.shape, dtype=np.int32)

        normal = nonzero & (av >= self._min_normal)
        if np.any(normal):
            # Unbiased exponent range for normals: [1 - bias, max_exp_stored - bias].
            e = np.floor(np.log2(av, where=normal, out=np.ones_like(av))).astype(np.int32)
            e = np.clip(e, 1 - self.exp_bias, self.max_exp_stored - self.exp_bias)
            frac = np.zeros(av.shape, dtype=np.float32)
            frac[normal] = av[normal] / np.exp2(e[normal].astype(np.float32)) - np.float32(1.0)
            exp_stored[normal] = e[normal] + self.exp_bias
            mantissa[normal] = np.rint(frac[normal] * (1 << self.mant_bits)).astype(np.int32)

        subnormal = nonzero & (av < self._min_normal)
        if np.any(subnormal):
            mantissa[subnormal] = np.rint(av[subnormal] / self._subnormal_step).astype(np.int32)

        # Mantissa rounding can overflow into the exponent field.
        carry = mantissa >= (1 << self.mant_bits)
        exp_stored[carry] += 1
        mantissa[carry] = 0

        # Saturate at the largest finite value rather than emitting Inf/NaN codes.
        overflow = exp_stored > self.max_exp_stored
        exp_stored[overflow] = self.max_exp_stored
        mantissa[overflow] = self.max_mantissa
        at_top = exp_stored == self.max_exp_stored
        mantissa[at_top] = np.minimum(mantissa[at_top], self.max_mantissa)

        return (
            sign_bit | (exp_stored.astype(np.uint8) << self.mant_bits) | mantissa.astype(np.uint8)
        ).astype(np.uint8)

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        c = codes.astype(np.int32)
        sign = np.where((c >> 7) & 1 == 1, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
        exponent = (c >> self.mant_bits) & ((1 << self.exp_bits) - 1)
        mantissa = c & ((1 << self.mant_bits) - 1)
        magnitude = np.where(
            exponent == 0,
            mantissa.astype(np.float32) * self._subnormal_step,
            (np.float32(1.0) + mantissa.astype(np.float32) / np.float32(1 << self.mant_bits))
            * np.exp2((exponent - self.exp_bias).astype(np.float32)),
        ).astype(np.float32)
        return (sign * magnitude * scales[:, None].astype(np.float32)).astype(np.float32)


class FP4Codec(Codec):
    """FP4 (E2M1) minifloat quantization for NVFP4/MXFP4 Blackwell targets.

    E2M1 uses 2 exponent bits and 1 mantissa bit, giving representable magnitudes
    ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` plus a sign bit. A per-block scale maps the
    block into that window.
    """

    name = "FP4_E2M1"
    bits = 4
    packable = True
    asymmetric = False

    #: Magnitudes representable by E2M1, ascending.
    _magnitudes: ClassVar[np.ndarray] = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
    )
    _boundaries: ClassVar[np.ndarray] = (_magnitudes[:-1] + _magnitudes[1:]) / 2.0
    max_value: float = 6.0

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        absmax = np.abs(blocks).max(axis=1).astype(np.float32)
        degenerate = absmax < _EPS
        scales = (absmax / np.float32(self.max_value)).astype(np.float32)
        scaled = blocks / _safe_divisor(scales)[:, None]
        sign = (scaled < 0).astype(np.uint8)
        mag_idx = np.searchsorted(self._boundaries, np.abs(scaled)).astype(np.uint8)
        mag_idx = np.clip(mag_idx, 0, 7)
        codes = ((sign << 3) | mag_idx).astype(np.uint8)
        codes[degenerate, :] = 0
        scales[degenerate] = 0.0
        zero_points = np.zeros(blocks.shape[0], dtype=np.float32)
        return codes, scales, zero_points

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        c = codes.astype(np.int32)
        sign = np.where((c >> 3) & 1 == 1, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
        magnitude = self._magnitudes[(c & 0x7).astype(np.intp)]
        return (sign * magnitude * scales[:, None].astype(np.float32)).astype(np.float32)


class MXFP4Codec(Codec):
    """
    Microscaling FP4 (OCP MXFP4) — dual-level block scaling.

    MXFP4 uses:
    - Outer group scale: one float8 (E4M3) per 32 elements (the "microscale")
    - Inner per-element: E2M1 codes (same as FP4Codec above)

    This matches the OCP Microscaling Formats (MX) spec adopted by NVIDIA Blackwell,
    AMD MI400, and Intel Gaudi 3. The dual scale provides 2-3x better accuracy
    than single-scale FP4 at near-zero overhead.

    Reference: OCP MX Specification v1.0 (2023), NVIDIA Blackwell GTC 2024.
    """

    name = "MXFP4"
    bits = 4
    packable = True
    asymmetric = False

    # MXFP4 outer group size (per the OCP spec)
    OUTER_GROUP = 32

    # E2M1 magnitudes (same as FP4)
    _magnitudes: ClassVar[np.ndarray] = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
    )
    _boundaries: ClassVar[np.ndarray] = (_magnitudes[:-1] + _magnitudes[1:]) / 2.0
    max_value: float = 6.0

    # Float8 E4M3 max value (outer scale storage format)
    _FP8_E4M3_MAX: float = 448.0

    def _fp8_e4m3_round(self, x: np.ndarray) -> np.ndarray:
        """Round-to-nearest in FP8 E4M3 range [0, 448]."""
        return np.clip(x, 0.0, self._FP8_E4M3_MAX).astype(np.float32)

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode using dual-level MXFP4 scaling.

        For MXFP4, the 'scale' stored per-block is the outer group scale
        (the fp8 microscale). The inner codes are E2M1 quantized relative
        to the outer scale.
        """
        num_blocks, block_size = blocks.shape
        absmax = np.abs(blocks).max(axis=1).astype(np.float32)
        degenerate = absmax < _EPS

        # Outer scale: absmax / FP4_max, rounded to fp8 E4M3 range
        outer_scale = self._fp8_e4m3_round(absmax / np.float32(self.max_value))

        # Scale blocks by outer_scale and quantize inner E2M1
        safe_outer = _safe_divisor(outer_scale)
        scaled = blocks / safe_outer[:, None]

        sign = (scaled < 0).astype(np.uint8)
        mag_idx = np.searchsorted(self._boundaries, np.abs(scaled)).astype(np.uint8)
        mag_idx = np.clip(mag_idx, 0, 7)
        codes = ((sign << 3) | mag_idx).astype(np.uint8)

        codes[degenerate, :] = 0
        outer_scale[degenerate] = 0.0

        zero_points = np.zeros(num_blocks, dtype=np.float32)
        return codes, outer_scale, zero_points

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        """Decode MXFP4: inner E2M1 × outer microscale."""
        c = codes.astype(np.int32)
        sign = np.where((c >> 3) & 1 == 1, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
        magnitude = self._magnitudes[(c & 0x7).astype(np.intp)]
        # scales here is the outer fp8 microscale
        return (sign * magnitude * scales[:, None].astype(np.float32)).astype(np.float32)

    def default_zero_points(self, num_blocks: int) -> np.ndarray:
        return np.zeros(num_blocks, dtype=np.float32)




class PassthroughCodec(Codec):
    """Rounding-only codec for BF16/FP16/FP32.

    These formats are not block-quantized; the tensor is stored at reduced float
    width. BF16 is emulated by round-to-nearest-even on the truncated mantissa,
    which is what hardware BF16 conversion does.
    """

    def __init__(self, name: str) -> None:
        self.name = name.upper()
        self.bits = {"BF16": 16, "FP16": 16, "FP32": 32}.get(self.name, 16)
        self.packable = False
        self.asymmetric = False

    @property
    def storage_dtype(self) -> np.dtype:
        """The numpy dtype used to store this format on disk."""
        if self.name == "FP32":
            return np.dtype(np.float32)
        if self.name == "FP16":
            return np.dtype(np.float16)
        return np.dtype(np.uint16)  # BF16 stored as the high 16 bits

    def round_to_format(self, values: np.ndarray) -> np.ndarray:
        """Round a float array onto this format's representable grid."""
        arr = np.ascontiguousarray(values, dtype=np.float32)
        if self.name == "FP32":
            return arr
        if self.name == "FP16":
            return arr.astype(np.float16).astype(np.float32)
        # BF16: round-to-nearest-even, then truncate the low mantissa bits.
        bits = arr.view(np.uint32)
        bias = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        rounded = ((bits.astype(np.uint64) + bias) & np.uint64(0xFFFF0000)).astype(np.uint32)
        return rounded.view(np.float32)

    def encode(self, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rounded = self.round_to_format(blocks)
        scales = np.ones(blocks.shape[0], dtype=np.float32)
        zero_points = np.zeros(blocks.shape[0], dtype=np.float32)
        return rounded, scales, zero_points

    def decode(self, codes: np.ndarray, scales: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.float32)


# ── Registry ──────────────────────────────────────────────────────────────────

#: Formats using an asymmetric affine grid (K-quant family), mapped to bit width.
_AFFINE_FORMATS: dict[str, int] = {
    "Q6_K": 6,
    "Q4_K_M": 4,
    "Q4_K_S": 4,
    "Q3_K": 3,
    "Q3_K_S": 3,
    "IQ3_XS": 3,
    "Q2_K": 2,
}

#: Formats using a symmetric absmax grid, mapped to bit width.
_SYMMETRIC_FORMATS: dict[str, int] = {
    "INT16": 16,
    "INT8": 8,
    "Q8_0": 8,
    "INT4": 4,
    "Q4_0": 4,
}

#: Aliases resolving to a canonical codec constructor key.
_ALIASES: dict[str, str] = {
    "NORMALFLOAT4": "NF4",
    "E4M3": "FP8",
    "FP8_E4M3": "FP8",
    "E5M2": "FP8_E5M2",
    "NVFP4": "FP4",
    # NOTE: MXFP4 is intentionally NOT aliased to FP4 — it uses a distinct
    # dual-level microscaling scheme (OCP MXFP4) via MXFP4Codec.
    "FP4_E2M1": "FP4",
}



def supported_precisions() -> list[str]:
    """Return every precision identifier :func:`get_codec` accepts, sorted."""
    names = (
        ["BF16", "FP16", "FP32", "NF4", "FP8", "FP8_E5M2", "FP4", "MXFP4"]
        + list(_AFFINE_FORMATS)
        + list(_SYMMETRIC_FORMATS)
        + list(_ALIASES)
    )
    return sorted(set(names))


def get_codec(precision: str) -> Codec:
    """Return the codec implementing ``precision``.

    Args:
        precision: Precision identifier such as ``"Q4_K_M"``, ``"FP8"``, ``"NF4"``.

    Returns:
        The matching :class:`Codec` instance.

    Raises:
        ValueError: If the precision is not recognised.
    """
    key = _ALIASES.get(precision.upper(), precision.upper())
    if key in ("BF16", "FP16", "FP32"):
        return PassthroughCodec(key)
    if key == "NF4":
        return NF4Codec()
    if key == "FP8":
        return FP8Codec("E4M3")
    if key == "FP8_E5M2":
        return FP8Codec("E5M2")
    if key == "FP4":
        return FP4Codec()
    if key == "MXFP4":
        return MXFP4Codec()
    if key in _AFFINE_FORMATS:
        return AffineIntCodec(key, _AFFINE_FORMATS[key])
    if key in _SYMMETRIC_FORMATS:
        return SymmetricIntCodec(key, _SYMMETRIC_FORMATS[key])
    msg = f"Unknown precision '{precision}'. Supported: {', '.join(supported_precisions())}"
    raise ValueError(msg)
