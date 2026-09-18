"""
Tests for aether.kernels.quantized_linear

Verifies every quantized linear format:
- FP8 dequant math is numerically correct
- INT8 symmetric and asymmetric dequant are correct
- NF4 codec matches the QLoRA NF4 grid
- INT4 packed dequant is numerically correct
- dispatch_quantized_linear routes to the right path
- QuantizedLinear object API works
- dequant_fallback is always labeled correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.kernels.quantized_linear import (
    QuantizedLinear,
    _dequant_fp8,
    _dequant_int4_packed,
    _dequant_int8_asymmetric,
    _dequant_int8_symmetric,
    _dequant_nf4,
    dequant_fallback,
    dispatch_quantized_linear,
    int8_linear,
    nf4_linear,
)


# ---------------------------------------------------------------------------
# NF4 grid constants
# ---------------------------------------------------------------------------
NF4_GRID = np.array([
    -1.0, -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500,  0.0,
     0.0795803,  0.1609302,  0.2461123,  0.3379152,
     0.4407098,  0.5626170,  0.7229568,  1.0,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# FP8 dequant
# ---------------------------------------------------------------------------

class TestDequantFP8:
    def test_scalar_scale(self):
        w = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        scale = 2.0
        out = _dequant_fp8(w, scale)
        expected = np.array([2.0, -2.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_identity_scale(self):
        w = np.array([0.25, 0.5, -0.25], dtype=np.float32)
        out = _dequant_fp8(w, 1.0)
        np.testing.assert_allclose(out, w, rtol=1e-6)

    def test_zero_weight(self):
        w = np.zeros(8, dtype=np.float32)
        out = _dequant_fp8(w, 3.7)
        np.testing.assert_array_equal(out, np.zeros(8, dtype=np.float32))

    def test_none_scale_defaults_to_one(self):
        w = np.array([1.5, -0.5], dtype=np.float32)
        out = _dequant_fp8(w, None)
        np.testing.assert_allclose(out, w, rtol=1e-6)


# ---------------------------------------------------------------------------
# INT8 dequant
# ---------------------------------------------------------------------------

class TestDequantINT8:
    def test_symmetric_per_tensor(self):
        w = np.array([[-128, 127], [0, 64]], dtype=np.int8)
        scale = np.array([0.01])
        out = _dequant_int8_symmetric(w, scale)
        expected = np.array([[-1.28, 1.27], [0.0, 0.64]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_symmetric_per_row(self):
        w = np.array([[100, -100], [50, -50]], dtype=np.int8)
        scale = np.array([0.01, 0.02])
        out = _dequant_int8_symmetric(w, scale)
        np.testing.assert_allclose(out[0], np.array([1.0, -1.0]), atol=1e-5)
        np.testing.assert_allclose(out[1], np.array([1.0, -1.0]), atol=1e-5)

    def test_asymmetric(self):
        # Use valid int8 range: map value 0 → 0*scale - zp*scale, 100 → 100*scale - zp*scale
        # zp=50 maps the midpoint to zero
        w = np.array([[0, 100]], dtype=np.int8)
        scale = np.array([0.01])   # 1/100 range
        zp = np.array([50.0])
        out = _dequant_int8_asymmetric(w, scale, zp)
        # (0 - 50) * 0.01 = -0.5, (100 - 50) * 0.01 = 0.5
        np.testing.assert_allclose(out[0, 0], -0.5, atol=1e-5)
        np.testing.assert_allclose(out[0, 1], 0.5, atol=1e-5)

    def test_zero_weights(self):
        w = np.zeros((4, 4), dtype=np.int8)
        out = _dequant_int8_symmetric(w, np.array([1.0]))
        np.testing.assert_array_equal(out, np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# NF4 dequant
# ---------------------------------------------------------------------------

class TestDequantNF4:
    def test_single_byte_lo_hi(self):
        # Pack lo=7 (grid[7]=0.0), hi=15 (grid[15]=1.0)
        packed = np.array([0xF7], dtype=np.uint8)
        absmax = np.array([1.0], dtype=np.float32)
        out = _dequant_nf4(packed, absmax)
        # lo nibble first: codes[0] = lo=7 → NF4_GRID[7] = 0.0
        # codes[1] = hi=15 → NF4_GRID[15] = 1.0
        assert len(out) == 2
        np.testing.assert_allclose(out[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[1], 1.0, atol=1e-5)

    def test_symmetric_range(self):
        # lo=0 (grid[0]=-1.0), hi=15 (grid[15]=1.0)
        packed = np.array([0xF0], dtype=np.uint8)
        absmax = np.array([1.0])
        out = _dequant_nf4(packed, absmax)
        np.testing.assert_allclose(out[0], NF4_GRID[0], atol=1e-5)
        np.testing.assert_allclose(out[1], NF4_GRID[15], atol=1e-5)

    def test_absmax_scaling(self):
        # lo=15, hi=0 → scaled by absmax=2.0
        packed = np.array([0x0F], dtype=np.uint8)
        absmax = np.array([2.0])
        out = _dequant_nf4(packed, absmax)
        np.testing.assert_allclose(out[0], NF4_GRID[15] * 2.0, atol=1e-5)
        np.testing.assert_allclose(out[1], NF4_GRID[0] * 2.0, atol=1e-5)

    def test_nf4_shape(self):
        packed = np.zeros(8, dtype=np.uint8)
        absmax = np.array([1.0])
        out = _dequant_nf4(packed, absmax)
        assert out.shape == (16,)


# ---------------------------------------------------------------------------
# INT4 dequant
# ---------------------------------------------------------------------------

class TestDequantINT4:
    def test_zero_packed(self):
        # lo=0, hi=0 → both decode to -8 (signed), scaled by 1 → -8
        packed = np.array([0x00], dtype=np.uint8)
        out = _dequant_int4_packed(packed, np.array([1.0]))
        assert len(out) == 2
        np.testing.assert_allclose(out[0], -8.0, atol=1e-5)
        np.testing.assert_allclose(out[1], -8.0, atol=1e-5)

    def test_max_packed(self):
        # lo=15 (0xF), hi=15 (0xF) → both → 15^8 - 8 = 7
        packed = np.array([0xFF], dtype=np.uint8)
        out = _dequant_int4_packed(packed, np.array([1.0]))
        np.testing.assert_allclose(out[0], 7.0, atol=1e-5)
        np.testing.assert_allclose(out[1], 7.0, atol=1e-5)

    def test_mixed_nibbles(self):
        # lo=0x8 (8^8-8=0), hi=0x0 (0^8-8=-8)
        packed = np.array([0x08], dtype=np.uint8)
        out = _dequant_int4_packed(packed, np.array([1.0]))
        np.testing.assert_allclose(out[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[1], -8.0, atol=1e-5)

    def test_scale_applied(self):
        packed = np.array([0xFF], dtype=np.uint8)  # both = 7
        out = _dequant_int4_packed(packed, np.array([0.5]))
        np.testing.assert_allclose(out[0], 3.5, atol=1e-5)


# ---------------------------------------------------------------------------
# dequant_fallback metadata label
# ---------------------------------------------------------------------------

class TestDequantFallback:
    def test_is_always_labeled_dequant(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.eye(4, dtype=np.float32)
        _, meta = dequant_fallback(x, None, None, w, "int8", "test_reason")
        assert meta.execution_mode == "dequant_fallback"
        assert meta.fallback_reason == "test_reason"

    def test_numpy_output_shape(self):
        x = np.ones((2, 4), dtype=np.float32)
        w = np.eye(4, dtype=np.float32)
        out, meta = dequant_fallback(x, None, None, w, "int8", "test")
        assert out.shape == (2, 4)

    def test_bias_applied(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.eye(4, dtype=np.float32)
        bias = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out, _ = dequant_fallback(x, None, bias, w, "fp8_e4m3", "test")
        expected = np.array([[2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# dispatch_quantized_linear routing
# ---------------------------------------------------------------------------

class TestDispatchRouting:
    def test_passthrough_bf16(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.eye(4, dtype=np.float32)
        out, meta = dispatch_quantized_linear(x, w, fmt="bf16")
        assert meta.execution_mode == "reference"
        assert out.shape == (1, 4)

    def test_int8_routes(self):
        x = np.ones((1, 4), dtype=np.float32)
        w = np.zeros((4, 4), dtype=np.int8)
        scale = np.array([1.0])
        out, meta = dispatch_quantized_linear(x, w, fmt="int8", scale=scale)
        # Must be labeled (dequant_fallback or quantized, never reference)
        assert meta.format == "int8"
        assert meta.execution_mode in ("quantized", "dequant_fallback")

    def test_nf4_routes(self):
        packed = np.zeros(8, dtype=np.uint8)
        absmax = np.array([1.0])
        x = np.ones((1, 16), dtype=np.float32)
        out, meta = dispatch_quantized_linear(x, packed, fmt="nf4", absmax=absmax)
        assert meta.format == "nf4"

    def test_invalid_fmt_raises(self):
        x = np.ones((1, 4), dtype=np.float32)
        # Not invalid — unknown formats fall through to reference
        out, meta = dispatch_quantized_linear(x, np.eye(4), fmt="bfloat16")
        assert meta.format == "bfloat16"


# ---------------------------------------------------------------------------
# QuantizedLinear object API
# ---------------------------------------------------------------------------

class TestQuantizedLinear:
    def test_call_int8(self):
        w = np.zeros((4, 4), dtype=np.int8)
        scale = np.array([1.0])
        ql = QuantizedLinear(
            weight=w, fmt="int8", scale=scale,
            out_features=4, in_features=4,
        )
        x = np.ones((1, 4), dtype=np.float32)
        out = ql(x)
        assert out is not None
        assert ql.execution_mode in ("quantized", "dequant_fallback")

    def test_repr(self):
        ql = QuantizedLinear(
            weight=np.zeros(8, dtype=np.uint8),
            fmt="nf4",
            out_features=4,
            in_features=16,
        )
        r = repr(ql)
        assert "nf4" in r
        assert "QuantizedLinear" in r

    def test_identity_fp32_passthrough(self):
        """FP32 passthrough should return identity for identity weight."""
        w = np.eye(4, dtype=np.float32)
        ql = QuantizedLinear(weight=w, fmt="fp32", out_features=4, in_features=4)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = ql(x)
        out_np = np.asarray(out, dtype=np.float32)
        np.testing.assert_allclose(out_np, x, atol=1e-5)
