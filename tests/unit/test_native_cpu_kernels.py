"""Tests for the natively compiled CPU kernels.

These exercise both execution paths. The numpy-reference path is always tested;
the compiled path is tested only when a host C++ compiler exists, and the two are
compared against each other so a codegen bug cannot hide behind the fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.kernels.native_cpu import (
    CPU_KERNEL_SOURCE,
    CompilerToolchain,
    NativeCPUKernels,
    detect_toolchain,
    get_native_kernels,
)

#: Tolerance for float32 accumulation differences between C++ and numpy.
ATOL = 1e-4
RTOL = 1e-5

HAS_COMPILER = detect_toolchain() is not None
requires_compiler = pytest.mark.skipif(
    not HAS_COMPILER, reason="no host C++ compiler available"
)


@pytest.fixture(scope="module")
def native() -> NativeCPUKernels:
    """Kernels using the compiled path when possible."""
    kernels = get_native_kernels()
    kernels.ensure_compiled()
    return kernels


@pytest.fixture(scope="module")
def reference() -> NativeCPUKernels:
    """Kernels pinned to the numpy reference path."""
    kernels = NativeCPUKernels()
    kernels.toolchain = None
    return kernels


@pytest.fixture
def rs() -> np.random.RandomState:
    return np.random.RandomState(0)


class TestToolchainDetection:
    def test_detection_returns_toolchain_or_none(self) -> None:
        toolchain = detect_toolchain()
        assert toolchain is None or isinstance(toolchain, CompilerToolchain)

    @requires_compiler
    def test_detected_toolchain_has_an_executable(self) -> None:
        toolchain = detect_toolchain()
        assert toolchain is not None
        assert toolchain.executable
        assert toolchain.name

    def test_library_suffix_is_platform_appropriate(self) -> None:
        toolchain = CompilerToolchain(name="g++", executable="g++")
        assert toolchain.library_suffix in (".dll", ".so", ".dylib")

    def test_build_command_includes_source_and_output(self) -> None:
        from pathlib import Path

        toolchain = CompilerToolchain(name="g++", executable="g++", base_flags=("-O3",))
        command = toolchain.build_command(Path("k.cpp"), Path("k.so"))
        assert "k.cpp" in command[-1]
        assert "k.so" in " ".join(command)
        assert "-shared" in command

    def test_msvc_command_uses_msvc_syntax(self) -> None:
        from pathlib import Path

        toolchain = CompilerToolchain(name="msvc", executable="cl", is_msvc=True)
        command = toolchain.build_command(Path("k.cpp"), Path("k.dll"))
        assert "/LD" in command
        assert any(arg.startswith("/Fe:") for arg in command)

    def test_source_does_not_use_fast_math_flag(self) -> None:
        """-ffast-math would break numerical parity with the numpy references."""
        toolchain = detect_toolchain()
        if toolchain is not None:
            assert "-ffast-math" not in toolchain.base_flags


class TestCompilation:
    @requires_compiler
    def test_compiles_successfully(self, native: NativeCPUKernels) -> None:
        assert native.is_native, native.build_error
        assert native.library_path is not None
        assert native.library_path.exists()

    @requires_compiler
    def test_all_declared_symbols_are_present(self, native: NativeCPUKernels) -> None:
        """_bind_signatures raises if a symbol is missing, so this must pass."""
        assert len(native.available_kernels()) == 10

    @requires_compiler
    def test_repeated_calls_reuse_the_cached_library(self, native: NativeCPUKernels) -> None:
        first = native.library_path
        assert native.ensure_compiled()
        assert native.library_path == first

    def test_source_declares_every_bound_symbol(self) -> None:
        """Each ctypes signature must correspond to a definition in the source."""
        for symbol in get_native_kernels().available_kernels():
            assert f"{symbol}(" in CPU_KERNEL_SOURCE, symbol

    def test_source_defines_the_export_macro_per_platform(self) -> None:
        assert "#define AETHER_EXPORT" in CPU_KERNEL_SOURCE
        assert "__declspec(dllexport)" in CPU_KERNEL_SOURCE  # Windows
        assert 'visibility("default")' in CPU_KERNEL_SOURCE  # ELF/Mach-O

    def test_missing_toolchain_degrades_gracefully(self, reference: NativeCPUKernels) -> None:
        assert reference.ensure_compiled() is False
        assert reference.is_native is False
        assert reference.build_error is not None

    def test_repr_reports_active_backend(self, reference: NativeCPUKernels) -> None:
        assert "numpy-reference" in repr(reference)


class TestRMSNorm:
    def test_matches_closed_form(self, reference: NativeCPUKernels, rs) -> None:
        x = rs.randn(4, 32).astype(np.float32)
        w = np.abs(rs.randn(32)).astype(np.float32) + 0.5
        expected = x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-5) * w
        np.testing.assert_allclose(reference.rmsnorm(x, w), expected, atol=ATOL, rtol=RTOL)

    @requires_compiler
    def test_native_matches_reference(
        self, native: NativeCPUKernels, reference: NativeCPUKernels, rs
    ) -> None:
        x = rs.randn(16, 128).astype(np.float32)
        w = np.abs(rs.randn(128)).astype(np.float32) + 0.5
        np.testing.assert_allclose(
            native.rmsnorm(x, w), reference.rmsnorm(x, w), atol=ATOL, rtol=RTOL
        )

    def test_unit_weight_normalises_to_unit_rms(self, native: NativeCPUKernels, rs) -> None:
        x = rs.randn(8, 64).astype(np.float32)
        out = native.rmsnorm(x, np.ones(64, dtype=np.float32), eps=0.0)
        rms = np.sqrt(np.mean(out.astype(np.float64) ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-3)

    def test_rejects_mismatched_weight(self, native: NativeCPUKernels, rs) -> None:
        with pytest.raises(ValueError, match="last axis"):
            native.rmsnorm(rs.randn(2, 8).astype(np.float32), np.ones(4, dtype=np.float32))

    def test_long_rows_stay_accurate(self, native: NativeCPUKernels, rs) -> None:
        """Accumulating in float32 over long rows loses precision; kernel uses double."""
        x = rs.randn(2, 8192).astype(np.float32)
        w = np.ones(8192, dtype=np.float32)
        expected = x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(native.rmsnorm(x, w), expected, atol=1e-3)


class TestActivations:
    def test_silu_matches_formula(self, reference: NativeCPUKernels, rs) -> None:
        x = rs.randn(4, 16).astype(np.float32)
        expected = x / (1.0 + np.exp(-x.astype(np.float64)))
        np.testing.assert_allclose(reference.silu(x), expected, atol=ATOL, rtol=RTOL)

    def test_silu_of_zero_is_zero(self, native: NativeCPUKernels) -> None:
        assert native.silu(np.zeros(8, dtype=np.float32)).tolist() == [0.0] * 8

    def test_silu_is_monotonic_above_threshold(self, native: NativeCPUKernels) -> None:
        x = np.linspace(0.0, 10.0, 64, dtype=np.float32)
        out = native.silu(x)
        assert np.all(np.diff(out) > 0)

    @requires_compiler
    def test_native_silu_matches_reference(
        self, native: NativeCPUKernels, reference: NativeCPUKernels, rs
    ) -> None:
        x = (rs.randn(64, 64) * 8).astype(np.float32)
        np.testing.assert_allclose(native.silu(x), reference.silu(x), atol=ATOL, rtol=RTOL)

    def test_swiglu_equals_silu_times_up(self, native: NativeCPUKernels, rs) -> None:
        gate = rs.randn(4, 16).astype(np.float32)
        up = rs.randn(4, 16).astype(np.float32)
        np.testing.assert_allclose(
            native.swiglu(gate, up), native.silu(gate) * up, atol=ATOL, rtol=RTOL
        )

    def test_swiglu_rejects_shape_mismatch(self, native: NativeCPUKernels, rs) -> None:
        with pytest.raises(ValueError, match="does not match"):
            native.swiglu(rs.randn(2, 4).astype(np.float32), rs.randn(2, 8).astype(np.float32))

    @requires_compiler
    def test_native_swiglu_matches_reference(
        self, native: NativeCPUKernels, reference: NativeCPUKernels, rs
    ) -> None:
        gate = rs.randn(32, 64).astype(np.float32)
        up = rs.randn(32, 64).astype(np.float32)
        np.testing.assert_allclose(
            native.swiglu(gate, up), reference.swiglu(gate, up), atol=ATOL, rtol=RTOL
        )


class TestSoftmax:
    def test_rows_sum_to_one(self, native: NativeCPUKernels, rs) -> None:
        out = native.softmax(rs.randn(8, 50).astype(np.float32))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-5)

    def test_output_is_a_probability_distribution(self, native: NativeCPUKernels, rs) -> None:
        out = native.softmax(rs.randn(4, 32).astype(np.float32))
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_is_shift_invariant(self, native: NativeCPUKernels, rs) -> None:
        x = rs.randn(4, 16).astype(np.float32)
        np.testing.assert_allclose(native.softmax(x), native.softmax(x + 10.0), atol=1e-5)

    def test_large_logits_do_not_overflow(self, native: NativeCPUKernels) -> None:
        """Max-shifting is what keeps exp() finite here."""
        x = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
        out = native.softmax(x)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5)

    @requires_compiler
    def test_native_matches_reference(
        self, native: NativeCPUKernels, reference: NativeCPUKernels, rs
    ) -> None:
        x = (rs.randn(16, 128) * 5).astype(np.float32)
        np.testing.assert_allclose(native.softmax(x), reference.softmax(x), atol=ATOL, rtol=RTOL)


class TestSGEMM:
    def test_matches_numpy_matmul(self, native: NativeCPUKernels, rs) -> None:
        a = rs.randn(16, 24).astype(np.float32)
        b = rs.randn(24, 12).astype(np.float32)
        np.testing.assert_allclose(native.sgemm(a, b), a @ b, atol=1e-3, rtol=1e-4)

    @requires_compiler
    def test_native_path_matches_blas_path(self, native: NativeCPUKernels, rs) -> None:
        """The compiled kernel must agree with the BLAS default it defers to."""
        a = rs.randn(64, 96).astype(np.float32)
        b = rs.randn(96, 48).astype(np.float32)
        np.testing.assert_allclose(
            native.sgemm(a, b, force_native=True), native.sgemm(a, b), atol=1e-3, rtol=1e-4
        )

    @requires_compiler
    def test_native_path_handles_block_boundaries(self, native: NativeCPUKernels, rs) -> None:
        """Sizes that are not multiples of the 64/256/64 blocking must still be exact."""
        for m, k, n in [(1, 1, 1), (65, 67, 257), (63, 3, 129), (100, 100, 7)]:
            a = rs.randn(m, k).astype(np.float32)
            b = rs.randn(k, n).astype(np.float32)
            np.testing.assert_allclose(
                native.sgemm(a, b, force_native=True), a @ b, atol=1e-3, rtol=1e-4
            )

    @requires_compiler
    def test_native_path_handles_sparse_weights(self, native: NativeCPUKernels, rs) -> None:
        a = rs.randn(32, 64).astype(np.float32)
        a[a < 0.5] = 0.0
        b = rs.randn(64, 16).astype(np.float32)
        np.testing.assert_allclose(
            native.sgemm(a, b, force_native=True), a @ b, atol=1e-3, rtol=1e-4
        )

    def test_rejects_misaligned_shapes(self, native: NativeCPUKernels, rs) -> None:
        with pytest.raises(ValueError, match="not aligned"):
            native.sgemm(rs.randn(4, 8).astype(np.float32), rs.randn(9, 4).astype(np.float32))

    def test_rejects_non_2d_inputs(self, native: NativeCPUKernels, rs) -> None:
        with pytest.raises(ValueError, match="2-D inputs"):
            native.sgemm(rs.randn(2, 2, 2).astype(np.float32), rs.randn(2, 2).astype(np.float32))

    def test_identity_matrix_is_a_no_op(self, native: NativeCPUKernels, rs) -> None:
        a = rs.randn(8, 8).astype(np.float32)
        np.testing.assert_allclose(native.sgemm(a, np.eye(8, dtype=np.float32)), a, atol=1e-5)


class TestRoPE:
    @pytest.fixture
    def tables(self) -> tuple[np.ndarray, np.ndarray]:
        head_dim, max_pos = 64, 128
        positions = np.arange(max_pos)[:, None]
        inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
        angles = positions * inv_freq
        return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def test_preserves_shape(self, native: NativeCPUKernels, tables, rs) -> None:
        cos, sin = tables
        x = rs.randn(8, 4, 64).astype(np.float32)
        assert native.rope(x, cos, sin).shape == x.shape

    def test_position_zero_is_identity(self, native: NativeCPUKernels, tables, rs) -> None:
        """cos(0)=1, sin(0)=0, so the first position must pass through unchanged."""
        cos, sin = tables
        x = rs.randn(1, 2, 64).astype(np.float32)
        np.testing.assert_allclose(native.rope(x, cos, sin), x, atol=1e-5)

    def test_preserves_pairwise_norm(self, native: NativeCPUKernels, tables, rs) -> None:
        """Rotation is norm-preserving on each (d, d+half) pair."""
        cos, sin = tables
        x = rs.randn(6, 2, 64).astype(np.float32)
        out = native.rope(x, cos, sin)
        half = 32
        before = x[..., :half] ** 2 + x[..., half:] ** 2
        after = out[..., :half] ** 2 + out[..., half:] ** 2
        np.testing.assert_allclose(after, before, rtol=1e-4, atol=1e-4)

    def test_does_not_mutate_input(self, native: NativeCPUKernels, tables, rs) -> None:
        cos, sin = tables
        x = rs.randn(4, 2, 64).astype(np.float32)
        original = x.copy()
        native.rope(x, cos, sin)
        np.testing.assert_array_equal(x, original)

    def test_position_offset_shifts_the_rotation(
        self, native: NativeCPUKernels, tables, rs
    ) -> None:
        cos, sin = tables
        x = rs.randn(4, 2, 64).astype(np.float32)
        assert not np.allclose(
            native.rope(x, cos, sin, position_offset=0),
            native.rope(x, cos, sin, position_offset=8),
        )

    @requires_compiler
    def test_native_matches_reference(
        self, native: NativeCPUKernels, reference: NativeCPUKernels, tables, rs
    ) -> None:
        cos, sin = tables
        x = rs.randn(16, 8, 64).astype(np.float32)
        np.testing.assert_allclose(
            native.rope(x, cos, sin), reference.rope(x, cos, sin), atol=ATOL, rtol=RTOL
        )

    def test_rejects_odd_head_dim(self, native: NativeCPUKernels, tables, rs) -> None:
        cos, sin = tables
        with pytest.raises(ValueError, match="head_dim must be even"):
            native.rope(rs.randn(2, 2, 15).astype(np.float32), cos, sin)

    def test_rejects_wrong_rank(self, native: NativeCPUKernels, tables, rs) -> None:
        cos, sin = tables
        with pytest.raises(ValueError, match="seq, heads, head_dim"):
            native.rope(rs.randn(4, 64).astype(np.float32), cos, sin)


class TestArgmax:
    def test_matches_numpy(self, native: NativeCPUKernels, rs) -> None:
        for _ in range(5):
            logits = rs.randn(1000).astype(np.float32)
            assert native.argmax(logits) == int(np.argmax(logits))

    def test_finds_a_planted_maximum(self, native: NativeCPUKernels) -> None:
        logits = np.zeros(64, dtype=np.float32)
        logits[42] = 1.0
        assert native.argmax(logits) == 42

    def test_returns_first_index_on_ties(self, native: NativeCPUKernels) -> None:
        assert native.argmax(np.ones(8, dtype=np.float32)) == 0

    def test_handles_all_negative_logits(self, native: NativeCPUKernels) -> None:
        logits = np.array([-5.0, -1.0, -3.0], dtype=np.float32)
        assert native.argmax(logits) == 1

    def test_rejects_empty_input(self, native: NativeCPUKernels) -> None:
        with pytest.raises(ValueError, match="empty"):
            native.argmax(np.array([], dtype=np.float32))


class TestNativeDequantize:
    @pytest.mark.parametrize("precision", ["Q8_0", "INT8", "Q4_K_M", "NF4", "FP8"])
    def test_matches_the_codec_path(self, native: NativeCPUKernels, precision: str, rs) -> None:
        from aether.quantization.formats import dequantize_tensor, quantize_tensor

        tensor = quantize_tensor(rs.randn(64, 64).astype(np.float32), precision, 32)
        np.testing.assert_allclose(
            native.dequantize(tensor), dequantize_tensor(tensor), atol=1e-5
        )

    def test_preserves_shape(self, native: NativeCPUKernels, rs) -> None:
        from aether.quantization.formats import quantize_tensor

        tensor = quantize_tensor(rs.randn(17, 33).astype(np.float32), "Q8_0", 32)
        assert native.dequantize(tensor).shape == (17, 33)

    def test_sparse_weights_stay_sparse(self, native: NativeCPUKernels, rs) -> None:
        from aether.quantization.formats import quantize_tensor

        weights = rs.randn(32, 64).astype(np.float32)
        weights[weights < 0.5] = 0.0
        tensor = quantize_tensor(weights, "Q8_0", 32)
        assert np.all(native.dequantize(tensor)[weights == 0.0] == 0.0)


class TestSharedInstance:
    def test_get_native_kernels_is_a_singleton(self) -> None:
        assert get_native_kernels() is get_native_kernels()

    def test_shared_instance_lists_all_kernels(self) -> None:
        assert "aether_rmsnorm" in get_native_kernels().available_kernels()
