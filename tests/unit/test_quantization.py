"""Tests for the quantization package."""

from __future__ import annotations

import numpy as np
import pytest

from aether.core.types import ModelArchitecture
from aether.quantization import BitPacker, PrecisionAssigner, Quantizer, SensitivityScorer
from aether.quantization.formats import QuantizationFormat, QuantizedTensor, dequantize_tensor, quantize_tensor


class TestQuantizationFormat:
    def test_from_string(self) -> None:
        fmt = QuantizationFormat.from_string("Q4_K_M")
        assert fmt.precision == "Q4_K_M"
        assert fmt.bit_width == 4
        assert fmt.is_quantized

    def test_unquantized(self) -> None:
        fmt = QuantizationFormat.from_string("BF16")
        assert not fmt.is_quantized
        assert fmt.block_size == 1

    def test_compression_ratio(self) -> None:
        fmt = QuantizationFormat.from_string("Q4_K_M")
        assert fmt.compression_ratio == 4.0


class TestQuantizeDequantize:
    def test_roundtrip_q4(self) -> None:
        weights = np.random.RandomState(0).randn(64, 64).astype(np.float32)
        qt = quantize_tensor(weights, "Q4_K_M", block_size=32)
        assert qt.precision == "Q4_K_M"
        assert qt.shape == (64, 64)
        dequantized = dequantize_tensor(qt)
        assert dequantized.shape == weights.shape
        # A 4-bit affine grid over unit-normal data lands near 0.08 RMSE. The old
        # bound of mse < 1.0 was loose enough to pass a codec that collapsed
        # entire blocks to a constant, which is how that bug went unnoticed.
        rmse = float(np.sqrt(np.mean((dequantized - weights) ** 2)))
        assert rmse < 0.15, f"Q4_K_M RMSE {rmse:.4f} is worse than the format allows"

    @pytest.mark.parametrize(
        ("precision", "max_rmse"),
        [
            ("BF16", 0.005),
            ("FP16", 0.001),
            ("Q8_0", 0.01),
            ("FP8", 0.05),
            ("Q6_K", 0.03),
            ("NF4", 0.10),
            ("Q4_K_M", 0.10),
            ("Q3_K", 0.20),
            ("Q2_K", 0.45),
        ],
    )
    def test_roundtrip_error_within_format_budget(self, precision: str, max_rmse: float) -> None:
        weights = np.random.RandomState(1).randn(64, 64).astype(np.float32)
        dequantized = dequantize_tensor(quantize_tensor(weights, precision, block_size=32))
        rmse = float(np.sqrt(np.mean((dequantized - weights) ** 2)))
        assert rmse < max_rmse, f"{precision} RMSE {rmse:.4f} exceeds budget {max_rmse}"

    @pytest.mark.parametrize("precision", ["Q4_K_M", "NF4", "Q3_K", "Q2_K", "FP4", "Q6_K"])
    def test_sub_byte_formats_are_bit_packed(self, precision: str) -> None:
        """Payload bytes must reflect the real bit width, not a padded int8 array."""
        weights = np.random.RandomState(2).randn(64, 64).astype(np.float32)
        qt = quantize_tensor(weights, precision, block_size=32)
        assert qt.packed
        expected = (weights.size * qt.bits + 7) // 8
        assert qt.payload_bytes == expected

    @pytest.mark.parametrize(
        ("precision", "min_ratio"),
        [("Q8_0", 1.7), ("Q6_K", 2.0), ("Q4_K_M", 3.0), ("Q3_K", 3.5), ("Q2_K", 5.0)],
    )
    def test_compression_ratio_is_real(self, precision: str, min_ratio: float) -> None:
        """Every format once reported the same 1.88x because all used int8."""
        weights = np.random.RandomState(3).randn(128, 128).astype(np.float32)
        qt = quantize_tensor(weights, precision, block_size=32)
        assert qt.compression_ratio >= min_ratio

    @pytest.mark.parametrize("size", [1, 7, 31, 33, 100, 1000])
    def test_non_multiple_of_block_size_preserves_shape(self, size: int) -> None:
        weights = np.random.RandomState(size).randn(size).astype(np.float32)
        for precision in ("Q4_K_M", "NF4", "Q3_K", "FP8"):
            dequantized = dequantize_tensor(quantize_tensor(weights, precision, block_size=32))
            assert dequantized.shape == weights.shape

    @pytest.mark.parametrize("shape", [(3, 5), (7, 11, 2), (1, 1), (64, 1)])
    def test_arbitrary_shapes_survive_roundtrip(self, shape: tuple[int, ...]) -> None:
        weights = np.random.RandomState(4).randn(*shape).astype(np.float32)
        assert dequantize_tensor(quantize_tensor(weights, "Q4_K_M", 32)).shape == shape

    def test_all_negative_tensor_is_not_collapsed(self) -> None:
        """Regression: the old scale used block.max(), destroying negative blocks."""
        weights = np.linspace(-1.0, -0.5, 256, dtype=np.float32)
        dequantized = dequantize_tensor(quantize_tensor(weights, "Q4_K_M", 32))
        assert np.abs(weights - dequantized).max() < 0.02
        assert len(np.unique(dequantized)) > 1

    @pytest.mark.parametrize("precision", ["Q4_K_M", "NF4", "FP8", "INT8", "Q8_0"])
    def test_structural_zeros_survive(self, precision: str) -> None:
        """Pruned weights must stay exactly zero after quantization."""
        weights = np.random.RandomState(5).randn(64, 64).astype(np.float32)
        weights[weights < 0.5] = 0.0
        dequantized = dequantize_tensor(quantize_tensor(weights, precision, 32))
        assert np.all(dequantized[weights == 0.0] == 0.0)

    def test_bf16_no_quant(self) -> None:
        weights = np.random.randn(16, 16).astype(np.float32)
        qt = quantize_tensor(weights, "BF16", block_size=32)
        assert qt.precision == "BF16"

    def test_bf16_stores_sixteen_bits_per_element(self) -> None:
        """BF16 is a 16-bit format, so it must not occupy float32 storage."""
        weights = np.random.RandomState(6).randn(64, 64).astype(np.float32)
        qt = quantize_tensor(weights, "BF16", block_size=32)
        assert qt.payload_bytes == weights.size * 2
        assert qt.compression_ratio == pytest.approx(1.0)

    def test_rejects_invalid_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size must be positive"):
            quantize_tensor(np.zeros(8, dtype=np.float32), "Q4_K_M", block_size=0)

    def test_rejects_unknown_precision(self) -> None:
        with pytest.raises(ValueError, match="Unknown precision"):
            quantize_tensor(np.zeros(8, dtype=np.float32), "Q5_FAKE", block_size=32)


class TestQuantizer:
    def test_quantize_simple(self) -> None:
        quantizer = Quantizer()
        weights = {"layer_0.weight": np.random.randn(32, 32).astype(np.float32)}
        precision_map = {"layer_0": "Q4_K_M"}
        result = quantizer.quantize(weights, precision_map)
        assert "layer_0.weight" in result
        assert isinstance(result["layer_0.weight"], QuantizedTensor)

    def test_estimate_compression(self) -> None:
        quantizer = Quantizer()
        weights = {"w1": np.random.randn(128, 128).astype(np.float32)}
        precisions = {"default": "Q3_K"}
        estimation = quantizer.estimate_compression(weights, precisions)
        assert estimation["original_bytes"] > estimation["compressed_bytes"]
        assert estimation["compression_ratio"] > 1.0


class TestSensitivityScorer:
    def test_compute(self) -> None:
        arch = ModelArchitecture(
            family="llama_family",
            params_billion=1.0,
            layers=8,
            hidden_size=256,
            num_attention_heads=4,
        )
        scorer = SensitivityScorer()
        scores = scorer.compute(arch)
        assert "layer_0" in scores
        assert "embedding" in scores
        assert "lm_head" in scores
        assert scores["embedding"].score > 0.8
        assert scores["lm_head"].score > 0.8

    def test_tier_classification(self) -> None:
        arch = ModelArchitecture(family="test", layers=12, hidden_size=512, num_attention_heads=8, params_billion=0.1)
        scorer = SensitivityScorer()
        scores = scorer.compute(arch)
        for ls in scores.values():
            assert ls.tier in ("critical", "high", "medium", "low")


class TestPrecisionAssigner:
    def test_assign(self, sample_precision_map: dict[str, str]) -> None:
        architecture = ModelArchitecture(
            family="test",
            params_billion=1.0,
            layers=len(sample_precision_map) - 2,
            hidden_size=512,
            num_attention_heads=8,
        )
        scorer = SensitivityScorer()
        scores = scorer.compute(architecture)
        assigner = PrecisionAssigner(quality_budget=0.02)
        result = assigner.assign(scores)
        assert "embedding" in result.precision_map
        assert "lm_head" in result.precision_map
        assert result.precision_map["embedding"] == "BF16"

    def test_assign_uniform(self) -> None:
        assigner = PrecisionAssigner()
        result = assigner.assign_uniform(4, "Q4_K_M")
        assert len(result.precision_map) == 6
        assert all(p == "Q4_K_M" for k, p in result.precision_map.items() if k.startswith("layer"))


class TestBitPacker:
    def test_pack_unpack_4bit(self) -> None:
        packer = BitPacker(bit_width=4)
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
        packed = packer.pack(values)
        unpacked = packer.unpack(packed, len(values))
        assert np.array_equal(values, unpacked)

    def test_pack_unpack_3bit(self) -> None:
        packer = BitPacker(bit_width=3)
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.uint8)
        packed = packer.pack(values)
        unpacked = packer.unpack(packed, len(values))
        assert np.array_equal(values, unpacked)
