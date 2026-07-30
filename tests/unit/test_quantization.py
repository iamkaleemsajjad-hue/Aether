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
        weights = np.random.randn(64, 64).astype(np.float32)
        qt = quantize_tensor(weights, "Q4_K_M", block_size=32)
        assert qt.precision == "Q4_K_M"
        assert qt.shape == (64, 64)
        dequantized = dequantize_tensor(qt)
        assert dequantized.shape == weights.shape
        mse = np.mean((dequantized - weights) ** 2)
        assert mse < 1.0

    def test_bf16_no_quant(self) -> None:
        weights = np.random.randn(16, 16).astype(np.float32)
        qt = quantize_tensor(weights, "BF16", block_size=32)
        assert qt.precision == "BF16"


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
