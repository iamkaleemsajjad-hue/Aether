"""Tests for the numerical quantization codecs.

These assert the properties that make the codecs trustworthy rather than merely
loose error bounds: per-format bit widths, exact zero preservation, grid
idempotence, monotonic error versus bit width, and correct handling of the
single-signed blocks that the original shared-int8 path corrupted.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.quantization.codecs import (
    NF4_LEVELS,
    AffineIntCodec,
    FP4Codec,
    FP8Codec,
    MXFP4Codec,
    NF4Codec,
    PassthroughCodec,
    SymmetricIntCodec,
    TernaryCodec,
    get_codec,
    supported_precisions,
)
from aether.quantization.formats import dequantize_tensor, quantize_tensor

#: Every precision the registry resolves, used for property-based sweeps.
ALL_PRECISIONS = supported_precisions()

#: Block-quantized formats, ordered from most to fewest bits.
BLOCK_FORMATS = ["Q8_0", "INT8", "FP8", "FP8_E5M2", "Q6_K", "NF4", "Q4_K_M", "Q4_0", "FP4", "Q3_K", "Q2_K"]


@pytest.fixture
def normal_blocks() -> np.ndarray:
    """A (64, 32) block array drawn from a unit normal, like real LLM weights."""
    return np.random.RandomState(0).randn(64, 32).astype(np.float32)


class TestCodecRegistry:
    def test_resolves_every_supported_precision(self) -> None:
        for precision in ALL_PRECISIONS:
            assert get_codec(precision) is not None

    def test_unknown_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown precision"):
            get_codec("Q9_TOTALLY_FAKE")

    def test_error_message_lists_supported_formats(self) -> None:
        with pytest.raises(ValueError, match="Q4_K_M"):
            get_codec("nope")

    @pytest.mark.parametrize(
        ("precision", "expected"),
        [
            ("Q4_K_M", AffineIntCodec),
            ("Q3_K", AffineIntCodec),
            ("INT8", SymmetricIntCodec),
            ("Q8_0", SymmetricIntCodec),
            ("NF4", NF4Codec),
            ("FP8", FP8Codec),
            ("FP4", FP4Codec),
            ("BF16", PassthroughCodec),
            ("TERNARY", TernaryCodec),
        ],
    )
    def test_dispatches_to_correct_family(self, precision: str, expected: type) -> None:
        assert isinstance(get_codec(precision), expected)

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("NVFP4", "FP4"), ("E4M3", "FP8"), ("normalfloat4", "NF4")],
    )
    def test_aliases_resolve_to_same_codec(self, alias: str, canonical: str) -> None:
        assert get_codec(alias).name == get_codec(canonical).name

    def test_mxfp4_is_a_distinct_codec_not_an_fp4_alias(self) -> None:
        """
        MXFP4 must not collapse into FP4.

        Both use E2M1 codes, but OCP MXFP4 adds an outer FP8-E4M3 microscale
        per 32-element group on top of the per-block scale. That second scale
        level is what makes it more accurate than single-scale FP4, so
        resolving MXFP4 to FP4Codec would silently drop precision on Blackwell
        / MI400 / Gaudi3 targets. PRD §18 lists them as separate formats.
        """
        mxfp4 = get_codec("MXFP4")
        fp4 = get_codec("FP4")

        assert isinstance(mxfp4, MXFP4Codec)
        assert mxfp4.name == "MXFP4"
        assert mxfp4.name != fp4.name
        # Same nominal width, different scaling structure.
        assert mxfp4.bits == fp4.bits == 4
        assert mxfp4.OUTER_GROUP == 32

    def test_mxfp4_outer_microscale_beats_plain_fp4_on_varied_magnitudes(self) -> None:
        """
        The dual-level scale must earn its keep.

        Across groups whose magnitudes differ by orders of magnitude, the outer
        microscale should reconstruct at least as accurately as single-scale
        FP4 — that is the whole reason the format exists.
        """
        rng = np.random.RandomState(7)
        # Blocks spanning wildly different dynamic ranges.
        scales = np.array([1e-3, 1.0, 1e2, 1e3], dtype=np.float32).repeat(8)
        blocks = (rng.randn(32, 32) * scales[:, None]).astype(np.float32)

        def reconstruction_error(codec) -> float:
            codes, scale, zero = codec.encode(blocks)
            out = codec.decode(codes, scale, zero)
            denom = float(np.abs(blocks).mean())
            return float(np.abs(out - blocks).mean() / denom)

        mxfp4_err = reconstruction_error(get_codec("MXFP4"))
        fp4_err = reconstruction_error(get_codec("FP4"))

        assert mxfp4_err <= fp4_err * 1.05

    def test_precision_is_case_insensitive(self) -> None:
        assert get_codec("q4_k_m").name == get_codec("Q4_K_M").name

    def test_bitnet_alias_and_packed_tensor_roundtrip(self) -> None:
        assert get_codec("bitnet").name == "TERNARY"
        values = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
        tensor = quantize_tensor(values, "TERNARY", block_size=32)
        assert tensor.packed is True
        assert tensor.bits == 2
        assert tensor.data.nbytes == 16
        restored = dequantize_tensor(tensor)
        assert restored.shape == values.shape
        assert np.isfinite(restored).all()
        assert np.all(restored[values == 0.0] == 0.0)

    @pytest.mark.parametrize(
        ("precision", "bits"),
        [("Q2_K", 2), ("Q3_K", 3), ("Q4_K_M", 4), ("NF4", 4), ("FP4", 4), ("Q6_K", 6), ("FP8", 8), ("INT8", 8)],
    )
    def test_bit_widths_are_distinct_per_format(self, precision: str, bits: int) -> None:
        """Regression: every format once shared one int8 path and reported 8 bits."""
        assert get_codec(precision).bits == bits


class TestZeroPreservation:
    """Structural zeros must survive quantization or pruning silently breaks."""

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_all_zero_block_roundtrips_to_exact_zero(self, precision: str) -> None:
        codec = get_codec(precision)
        blocks = np.zeros((4, 32), dtype=np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        assert np.all(decoded == 0.0)

    @pytest.mark.parametrize("precision", BLOCK_FORMATS)
    def test_sparse_block_keeps_its_zeros(self, precision: str) -> None:
        codec = get_codec(precision)
        blocks = np.random.RandomState(1).randn(8, 32).astype(np.float32)
        blocks[blocks < 0.5] = 0.0
        decoded = codec.decode(*codec.encode(blocks))
        assert np.all(decoded[blocks == 0.0] == 0.0)


class TestGridIdempotence:
    """Re-encoding an on-grid value must not drift; otherwise error compounds."""

    @pytest.mark.parametrize("precision", BLOCK_FORMATS + ["BF16", "FP16", "FP32"])
    def test_second_roundtrip_is_stable(self, precision: str, normal_blocks: np.ndarray) -> None:
        codec = get_codec(precision)
        first = codec.decode(*codec.encode(normal_blocks))
        second = codec.decode(*codec.encode(first))
        np.testing.assert_allclose(first, second, atol=1e-5)


class TestSingleSignedBlocks:
    """The original codec destroyed all-negative blocks; these lock in the fix.

    Error bounds differ by grid type. NF4 and FP4 use *fixed* grids centred on
    zero, so a block that never crosses zero can only reach roughly a third of
    their levels — a real property of those formats, not a defect. Adaptive grids
    (affine and symmetric) rescale to the block and stay far tighter.
    """

    #: Fixed-grid formats waste their unused half on single-signed data.
    FIXED_GRID_TOLERANCE = 0.20
    #: Adaptive grids should track a single-signed block closely.
    ADAPTIVE_GRID_TOLERANCE = 0.10
    FIXED_GRID_FORMATS = frozenset({"NF4", "FP4"})

    def _tolerance(self, precision: str) -> float:
        return (
            self.FIXED_GRID_TOLERANCE
            if precision in self.FIXED_GRID_FORMATS
            else self.ADAPTIVE_GRID_TOLERANCE
        )

    @pytest.mark.parametrize("precision", BLOCK_FORMATS)
    def test_all_negative_block_is_not_collapsed(self, precision: str) -> None:
        blocks = np.linspace(-1.0, -0.5, 32, dtype=np.float32).reshape(1, 32)
        codec = get_codec(precision)
        decoded = codec.decode(*codec.encode(blocks))
        # The old bug produced a constant -1.0 block: one distinct value, err 0.5.
        assert len(np.unique(decoded)) > 1, "block collapsed to a single value"
        assert np.abs(blocks - decoded).max() < self._tolerance(precision)

    @pytest.mark.parametrize("precision", BLOCK_FORMATS)
    def test_all_positive_block_roundtrips(self, precision: str) -> None:
        blocks = np.linspace(0.5, 1.0, 32, dtype=np.float32).reshape(1, 32)
        codec = get_codec(precision)
        decoded = codec.decode(*codec.encode(blocks))
        assert len(np.unique(decoded)) > 1, "block collapsed to a single value"
        assert np.abs(blocks - decoded).max() < self._tolerance(precision)

    @pytest.mark.parametrize("precision", ["Q4_K_M", "Q8_0", "INT8", "FP8"])
    def test_adaptive_grids_track_negative_blocks_tightly(self, precision: str) -> None:
        """Regression guard on the exact case that used to yield 0.5 max error."""
        blocks = np.linspace(-1.0, -0.5, 32, dtype=np.float32).reshape(1, 32)
        codec = get_codec(precision)
        decoded = codec.decode(*codec.encode(blocks))
        assert np.abs(blocks - decoded).max() < 0.05

    def test_affine_codec_beats_symmetric_on_offset_data(self) -> None:
        """Asymmetric grids should win on data far from zero — the whole point."""
        blocks = (np.random.RandomState(2).randn(16, 32) * 0.05 + 10.0).astype(np.float32)
        affine = get_codec("Q4_K_M")
        symmetric = get_codec("Q4_0")
        affine_err = np.abs(blocks - affine.decode(*affine.encode(blocks))).mean()
        symmetric_err = np.abs(blocks - symmetric.decode(*symmetric.encode(blocks))).mean()
        assert affine_err < symmetric_err


class TestErrorMonotonicity:
    def test_error_decreases_as_bits_increase(self, normal_blocks: np.ndarray) -> None:
        ladder = ["Q2_K", "Q3_K", "Q4_K_M", "Q6_K", "Q8_0"]
        errors = []
        for precision in ladder:
            codec = get_codec(precision)
            decoded = codec.decode(*codec.encode(normal_blocks))
            errors.append(float(np.sqrt(np.mean((normal_blocks - decoded) ** 2))))
        assert errors == sorted(errors, reverse=True), dict(zip(ladder, errors))

    def test_nf4_beats_uniform_int4_on_normal_data(self, normal_blocks: np.ndarray) -> None:
        """NF4's normal-quantile grid is the reason it exists."""
        nf4 = get_codec("NF4")
        int4 = get_codec("Q4_0")
        nf4_err = np.sqrt(np.mean((normal_blocks - nf4.decode(*nf4.encode(normal_blocks))) ** 2))
        int4_err = np.sqrt(np.mean((normal_blocks - int4.decode(*int4.encode(normal_blocks))) ** 2))
        assert nf4_err < int4_err


class TestNF4Codec:
    def test_grid_has_16_levels_spanning_unit_range(self) -> None:
        assert len(NF4_LEVELS) == 16
        assert NF4_LEVELS[0] == pytest.approx(-1.0)
        assert NF4_LEVELS[-1] == pytest.approx(1.0)

    def test_grid_is_sorted_and_contains_exact_zero(self) -> None:
        assert np.all(np.diff(NF4_LEVELS) > 0)
        assert 0.0 in NF4_LEVELS

    def test_on_grid_values_encode_exactly(self) -> None:
        """A block whose values are already NF4 levels must roundtrip exactly."""
        codec = NF4Codec()
        blocks = np.tile(NF4_LEVELS, (2, 2)).astype(np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        np.testing.assert_allclose(decoded, blocks, atol=1e-6)

    def test_codes_stay_within_4_bit_range(self, normal_blocks: np.ndarray) -> None:
        codes, _, _ = NF4Codec().encode(normal_blocks)
        assert codes.min() >= 0
        assert codes.max() <= 15


class TestFP8Codec:
    @pytest.mark.parametrize(("variant", "max_value"), [("E4M3", 448.0), ("E5M2", 57344.0)])
    def test_max_representable_matches_ocp_spec(self, variant: str, max_value: float) -> None:
        assert FP8Codec(variant).max_value == pytest.approx(max_value)

    def test_rejects_unknown_variant(self) -> None:
        with pytest.raises(ValueError, match="Unsupported FP8 variant"):
            FP8Codec("E7M0")

    @pytest.mark.parametrize("variant", ["E4M3", "E5M2"])
    def test_sign_is_preserved(self, variant: str) -> None:
        codec = FP8Codec(variant)
        blocks = np.array([[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] * 4], dtype=np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        assert np.all(np.sign(decoded) == np.sign(blocks))

    def test_e4m3_more_accurate_than_e5m2_on_normal_data(self, normal_blocks: np.ndarray) -> None:
        """E4M3 trades exponent range for mantissa bits, so it wins on tight ranges."""
        e4m3, e5m2 = FP8Codec("E4M3"), FP8Codec("E5M2")
        err_4 = np.sqrt(np.mean((normal_blocks - e4m3.decode(*e4m3.encode(normal_blocks))) ** 2))
        err_5 = np.sqrt(np.mean((normal_blocks - e5m2.decode(*e5m2.encode(normal_blocks))) ** 2))
        assert err_4 < err_5

    def test_does_not_emit_nan_or_inf(self) -> None:
        """Saturation must clamp to max finite rather than producing Inf/NaN codes."""
        codec = FP8Codec("E4M3")
        blocks = np.array([[1e30, -1e30, 1e-30, 0.0] * 8], dtype=np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        assert np.all(np.isfinite(decoded))

    def test_relative_error_within_mantissa_resolution(self) -> None:
        """E4M3 has 3 mantissa bits, so relative error must stay near 2^-4."""
        codec = FP8Codec("E4M3")
        blocks = np.full((1, 32), 1.0, dtype=np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        assert np.abs(decoded - 1.0).max() < 0.07


class TestFP4Codec:
    def test_magnitudes_match_e2m1_grid(self) -> None:
        assert FP4Codec.max_value == pytest.approx(6.0)
        np.testing.assert_allclose(
            FP4Codec._magnitudes, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        )

    def test_codes_stay_within_4_bit_range(self, normal_blocks: np.ndarray) -> None:
        codes, _, _ = FP4Codec().encode(normal_blocks)
        assert codes.max() <= 15

    def test_on_grid_values_encode_exactly(self) -> None:
        codec = FP4Codec()
        # Scaled so absmax maps exactly onto the top of the E2M1 grid.
        blocks = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4], dtype=np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        np.testing.assert_allclose(decoded, blocks, atol=1e-6)


class TestPassthroughCodec:
    def test_fp32_is_lossless(self) -> None:
        codec = PassthroughCodec("FP32")
        blocks = np.random.RandomState(3).randn(4, 32).astype(np.float32)
        np.testing.assert_array_equal(codec.round_to_format(blocks), blocks)

    def test_bf16_truncates_mantissa_but_keeps_magnitude(self) -> None:
        codec = PassthroughCodec("BF16")
        blocks = np.random.RandomState(4).randn(4, 32).astype(np.float32)
        rounded = codec.round_to_format(blocks)
        # BF16 keeps 8 mantissa bits -> ~2^-9 relative error.
        assert np.abs(rounded - blocks).max() < 0.01
        assert np.all(np.sign(rounded) == np.sign(blocks))

    def test_bf16_low_mantissa_bits_are_zero(self) -> None:
        codec = PassthroughCodec("BF16")
        blocks = np.random.RandomState(5).randn(2, 32).astype(np.float32)
        rounded = codec.round_to_format(blocks)
        assert np.all(rounded.view(np.uint32) & np.uint32(0x0000FFFF) == 0)

    def test_bf16_is_idempotent(self) -> None:
        codec = PassthroughCodec("BF16")
        blocks = np.random.RandomState(6).randn(2, 32).astype(np.float32)
        once = codec.round_to_format(blocks)
        np.testing.assert_array_equal(codec.round_to_format(once), once)

    def test_fp16_matches_numpy_cast(self) -> None:
        codec = PassthroughCodec("FP16")
        blocks = np.random.RandomState(7).randn(2, 32).astype(np.float32)
        expected = blocks.astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(codec.round_to_format(blocks), expected)


class TestCodecShapes:
    @pytest.mark.parametrize("precision", BLOCK_FORMATS)
    def test_encode_shapes_are_consistent(self, precision: str) -> None:
        codec = get_codec(precision)
        blocks = np.random.RandomState(8).randn(7, 16).astype(np.float32)
        codes, scales, zero_points = codec.encode(blocks)
        assert codes.shape == blocks.shape
        assert scales.shape == (7,)
        assert zero_points.shape == (7,)

    @pytest.mark.parametrize("precision", BLOCK_FORMATS)
    def test_single_block_and_single_element(self, precision: str) -> None:
        codec = get_codec(precision)
        for shape in [(1, 1), (1, 32), (3, 1)]:
            blocks = np.random.RandomState(9).randn(*shape).astype(np.float32)
            decoded = codec.decode(*codec.encode(blocks))
            assert decoded.shape == shape

    @pytest.mark.parametrize("precision", BLOCK_FORMATS)
    def test_extreme_magnitudes_stay_finite(self, precision: str) -> None:
        codec = get_codec(precision)
        blocks = np.array([[1e-8] * 16 + [1e8] * 16], dtype=np.float32)
        decoded = codec.decode(*codec.encode(blocks))
        assert np.all(np.isfinite(decoded))
