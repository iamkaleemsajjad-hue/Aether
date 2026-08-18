"""Numerical correctness tests for GGUF K-quant dequantization.

Every test compares Aether's pure-Python K-quant decoders against the
``gguf`` package's numpy reference implementations (an independent
transcription of llama.cpp's ggml-quanta.c). A mismatch means Aether would
silently compute wrong logits from real GGUF models.

These tests also verify the hard "no zeros placeholder" invariant: valid
quantized data must never dequantize to an all-zeros array.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

gguf = pytest.importorskip("gguf", reason="gguf reference package not installed")
from gguf import GGMLQuantizationType  # noqa: E402

from aether.compiler.stage1_ingestion import gguf_loader as gl  # noqa: E402

_GGML_QUANT_TO_AETHER = {
    GGMLQuantizationType.Q2_K: gl._GGML_TYPE_Q2_K,
    GGMLQuantizationType.Q3_K: gl._GGML_TYPE_Q3_K,
    GGMLQuantizationType.Q4_K: gl._GGML_TYPE_Q4_K,
    GGMLQuantizationType.Q5_K: gl._GGML_TYPE_Q5_K,
    GGMLQuantizationType.Q6_K: gl._GGML_TYPE_Q6_K,
    GGMLQuantizationType.Q8_0: gl._GGML_TYPE_Q8_0,
    GGMLQuantizationType.Q4_0: gl._GGML_TYPE_Q4_0,
    GGMLQuantizationType.Q4_1: gl._GGML_TYPE_Q4_1,
    GGMLQuantizationType.Q5_0: gl._GGML_TYPE_Q5_0,
    GGMLQuantizationType.Q5_1: gl._GGML_TYPE_Q5_1,
    # Q8_K is excluded: the gguf reference package raises NotImplementedError
    # for it, so it is covered by the hand-computed block test below instead.
}

_BLOCK_SHAPES = {
    gl._GGML_TYPE_Q2_K: 84,
    gl._GGML_TYPE_Q3_K: 110,
    gl._GGML_TYPE_Q4_K: 144,
    gl._GGML_TYPE_Q5_K: 176,
    gl._GGML_TYPE_Q6_K: 210,
    gl._GGML_TYPE_Q8_0: 34,
    gl._GGML_TYPE_Q4_0: 18,
    gl._GGML_TYPE_Q4_1: 20,
    gl._GGML_TYPE_Q5_0: 22,
    gl._GGML_TYPE_Q5_1: 24,
    gl._GGML_TYPE_Q8_K: 292,
}

#: Types whose blocks hold 32 elements (K-quants hold 256 per super-block).
_BLOCK32_TYPES = frozenset({
    gl._GGML_TYPE_Q8_0, gl._GGML_TYPE_Q4_0,
    gl._GGML_TYPE_Q4_1, gl._GGML_TYPE_Q5_0, gl._GGML_TYPE_Q5_1,
})


def _reference_dequant(raw: bytes, block_bytes: int, ggml_type: GGMLQuantizationType, n_blocks: int) -> np.ndarray:
    """Dequantize with the gguf package's independent reference implementation."""
    data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_bytes)
    return gguf.quants.dequantize(data, ggml_type).astype(np.float32).reshape(-1)


def _aether_dequant(raw: bytes, aether_type: int, num_elems: int) -> np.ndarray:
    return gl._DEQUANT_FN[aether_type](raw, num_elems)


@pytest.mark.parametrize("ggml_type", list(_GGML_QUANT_TO_AETHER))
def test_kquant_matches_gguf_reference(ggml_type: GGMLQuantizationType) -> None:
    """Aether's dequantizer must be bit-compatible with the gguf reference."""
    aether_type = _GGML_QUANT_TO_AETHER[ggml_type]
    block_bytes = _BLOCK_SHAPES[aether_type]
    n_blocks = 4  # exercise multi-block ordering including both 128-elem halves
    num_elems = n_blocks * (32 if aether_type in _BLOCK32_TYPES else 256)
    block_bytes = _BLOCK_SHAPES[aether_type]

    rng = np.random.default_rng(1234)
    raw = rng.integers(0, 256, size=block_bytes * n_blocks, dtype=np.uint8).tobytes()

    expected = _reference_dequant(raw, block_bytes, ggml_type, n_blocks)
    actual = _aether_dequant(raw, aether_type, num_elems)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_q4_k_known_block() -> None:
    """Hand-computed Q4_K block verifies scale unpacking independent of loops."""
    # Block: d=1.0, dmin=0.5; scales bytes 0..7 = [10, 20, 30, 40, 50, 60, 70, 80]
    # with get_scale_min_k4: j<4 -> scale=q[j]&63, min=q[j+4]&63.
    d = np.float16(1.0)
    dmin = np.float16(0.5)
    scales = bytes([10, 20, 30, 40, 50, 60, 70, 80] + [0] * 4)
    # sub-block 0: scale = 10, min = 50&63 = 50
    # first element: qs[0] low nibble = 5 -> value = 1.0*10*5 - 0.5*50 = 25.0
    qs = bytearray(128)
    qs[0] = 0xA5  # lo nibble 5 (elem 0), hi nibble 10 (elem 32)
    block = struct.pack("<e", d) + struct.pack("<e", dmin) + scales + bytes(qs)
    assert len(block) == 144
    out = gl._dequant_q4_k(block, 256)
    assert out[0] == pytest.approx(1.0 * 10 * 5 - 0.5 * 50)
    # sub-block 1 uses hi nibbles of qs bytes 0..31 with scale/min of j=1:
    # scale = 20, min = 60&63 = 60 -> elem 32 = 1.0*20*10 - 0.5*60
    assert out[32] == pytest.approx(1.0 * 20 * 10 - 0.5 * 60)


def test_q6_k_known_block() -> None:
    """Hand-computed Q6_K block: element 0 uses ql[0] lo nibble + qh[0] bits 0-1."""
    ql = bytearray(128)
    qh = bytearray(64)
    sc = bytes([1] * 16)  # int8 scales all 1
    d = np.float16(2.0)
    ql[0] = 0x3  # lo nibble 3, hi nibble 0
    qh[0] = 0x1  # hi2 = 1 for element 0
    block = bytes(ql) + bytes(qh) + sc + struct.pack("<e", d)
    assert len(block) == 210
    out = gl._dequant_q6_k(block, 256)
    # elem 0: q1 = (3 | (1 << 4)) - 32 = -13; y = d * sc[0] * q1 = -26.0
    assert out[0] == pytest.approx(-26.0)
    # elem 64: q3 = (ql[0] >> 4 | ((qh[0]>>4)&3)<<4) - 32 = -32
    assert out[64] == pytest.approx(-64.0)


@pytest.mark.parametrize(
    "aether_type,block_bytes,elements_per_block",
    [
        (gl._GGML_TYPE_Q2_K, 84, 256),
        (gl._GGML_TYPE_Q3_K, 110, 256),
        (gl._GGML_TYPE_Q4_K, 144, 256),
        (gl._GGML_TYPE_Q5_K, 176, 256),
        (gl._GGML_TYPE_Q6_K, 210, 256),
        (gl._GGML_TYPE_Q8_K, 292, 256),
        (gl._GGML_TYPE_Q4_1, 20, 32),
        (gl._GGML_TYPE_Q5_0, 22, 32),
        (gl._GGML_TYPE_Q5_1, 24, 32),
    ],
)
def test_kquant_never_returns_all_zeros(
    aether_type: int, block_bytes: int, elements_per_block: int
) -> None:
    """Valid random quantized data must never dequantize to all zeros."""
    rng = np.random.default_rng(99)
    raw = rng.integers(0, 256, size=block_bytes * 2, dtype=np.uint8).tobytes()
    out = gl._DEQUANT_FN[aether_type](raw, elements_per_block * 2)
    assert out.shape == (elements_per_block * 2,)
    assert not np.all(out == 0.0), "dequantization produced placeholder zeros"


@pytest.mark.parametrize(
    "aether_type,block_bytes",
    [
        (gl._GGML_TYPE_Q2_K, 84),
        (gl._GGML_TYPE_Q3_K, 110),
        (gl._GGML_TYPE_Q4_K, 144),
        (gl._GGML_TYPE_Q5_K, 176),
        (gl._GGML_TYPE_Q6_K, 210),
        (gl._GGML_TYPE_Q8_K, 292),
        (gl._GGML_TYPE_Q4_1, 20),
        (gl._GGML_TYPE_Q5_0, 22),
        (gl._GGML_TYPE_Q5_1, 24),
    ],
)
def test_kquant_rejects_bad_element_count(aether_type: int, block_bytes: int) -> None:
    """A size that does not divide into whole super-blocks must fail closed."""
    from aether.core.exceptions import UnsupportedFormatError

    with pytest.raises(UnsupportedFormatError):
        gl._DEQUANT_FN[aether_type](b"\x00" * block_bytes, 100)


def test_q4_1_known_block() -> None:
    """Hand-computed Q4_1 block: value = d * nibble + m."""
    d = np.float16(0.5)
    m = np.float16(1.0)
    qs = bytearray(16)
    qs[0] = 0x21  # lo nibble 1 (elem 0), hi nibble 2 (elem 16)
    block = struct.pack("<e", d) + struct.pack("<e", m) + bytes(qs)
    assert len(block) == 20
    out = gl._dequant_q4_1(block, 32)
    assert out[0] == pytest.approx(0.5 * 1 + 1.0)
    assert out[16] == pytest.approx(0.5 * 2 + 1.0)


def test_q5_0_known_block() -> None:
    """Hand-computed Q5_0 block: 5-bit code minus 16, scaled by d."""
    d = np.float16(2.0)
    qh = struct.pack("<I", 0)  # all high bits clear
    qs = bytearray(16)
    qs[0] = 0x30  # lo nibble 0 (elem 0), hi nibble 3 (elem 16)
    block = struct.pack("<e", d) + qh + bytes(qs)
    assert len(block) == 22
    out = gl._dequant_q5_0(block, 32)
    assert out[0] == pytest.approx(2.0 * (0 - 16.0))
    assert out[16] == pytest.approx(2.0 * (3 - 16.0))
    # Setting qh bit 0 adds 16 to element 0's code.
    block_hi = struct.pack("<e", d) + struct.pack("<I", 1) + bytes(qs)
    out_hi = gl._dequant_q5_0(block_hi, 32)
    assert out_hi[0] == pytest.approx(2.0 * ((0 | 16) - 16.0))


def test_q5_1_known_block() -> None:
    """Hand-computed Q5_1 block: value = d * code + m."""
    d = np.float16(0.25)
    m = np.float16(0.5)
    qh = struct.pack("<I", 1 << 16)  # high bit for element 16
    qs = bytearray(16)
    qs[0] = 0x10  # lo nibble 0 (elem 0), hi nibble 1 (elem 16)
    block = struct.pack("<e", d) + struct.pack("<e", m) + qh + bytes(qs)
    assert len(block) == 24
    out = gl._dequant_q5_1(block, 32)
    assert out[0] == pytest.approx(0.25 * 0 + 0.5)
    # element 16: lo2 = 1, hi2 = (qh >> 16) & 1 = 1 -> code = 1 | 16 = 17
    assert out[16] == pytest.approx(0.25 * 17 + 0.5)


def test_q8_k_known_block() -> None:
    """Hand-computed Q8_K block: fp32 scale d, int8 quants, trailing bsums.

    Block layout (292 bytes): ``[4-byte fp32 d][256 int8 quants][16 int16
    bsums]`` per ggml-common.h; bsums are dot-product partial sums and are
    ignored by dequantization.
    """
    d = 0.5
    qs = np.arange(256).astype(np.int8).tobytes()  # wraps to negatives past 127
    bsums = bytes(32)
    block = struct.pack("<f", d) + qs + bsums
    assert len(block) == 292
    out = gl._dequant_q8_k(block, 256)
    expected = np.arange(256).astype(np.int8).astype(np.float32) * 0.5
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_complete_loader_kquant_delegates_no_zeros() -> None:
    """GGUFLoaderComplete must never return zeros for K-quant data."""
    import io

    from aether.compiler.stage1_ingestion.gguf_loader_complete import GGUFReader

    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, size=144 * 2, dtype=np.uint8).tobytes()
    reader = GGUFReader.__new__(GGUFReader)
    out = reader._dequantize_k_quant(io.BytesIO(raw), (512,), gl._GGML_TYPE_Q4_K)
    assert out.shape == (512,)
    assert not np.all(out == 0.0)

    # And an unsupported type must raise, not fabricate data.
    from aether.core.exceptions import UnsupportedFormatError

    with pytest.raises(UnsupportedFormatError):
        reader._dequantize_k_quant(io.BytesIO(b""), (4,), 42)
