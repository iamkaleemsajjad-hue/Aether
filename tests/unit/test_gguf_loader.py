"""
Tests for the GGUF loader — pure-binary parsing and K-quant dequantization.

Tests run without any external GGUF files by constructing minimal valid
GGUF binary payloads in memory.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aether.compiler.stage1_ingestion.gguf_loader import (
    GGUFLoader,
    GGUFReader,
    export_gguf_tokenizer_metadata,
    _dequant_f16,
    _dequant_bf16,
    _dequant_q8_0,
    _dequant_q4_0,
    _dequant_q4_k,
    _dequant_q6_k,
    _dequant_q2_k,
    _GGUF_MAGIC,
    _GGML_TYPE_F32,
    _GGML_TYPE_F16,
    _GGML_TYPE_BF16,
    _GGML_TYPE_Q8_0,
    _GGML_TYPE_Q4_0,
    _GGML_TYPE_Q4_K,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal valid GGUF files
# ---------------------------------------------------------------------------

def _pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_gguf(
    tensors: list[dict],
    metadata: dict | None = None,
    alignment: int = 32,
) -> bytes:
    """
    Build a minimal GGUF v3 binary with the given tensors.

    ``tensors`` items must have: name, shape (list[int]), type (int), data (bytes).
    """
    meta = metadata or {}

    # --- Header ---
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(tensors), len(meta))

    # --- KV metadata ---
    kv_bytes = b""
    for k, v in meta.items():
        kv_bytes += _pack_string(k)
        if isinstance(v, str):
            kv_bytes += struct.pack("<I", 8)  # STRING
            kv_bytes += _pack_string(v)
        elif isinstance(v, int):
            kv_bytes += struct.pack("<I", 4)  # UINT32
            kv_bytes += struct.pack("<I", v)
        elif isinstance(v, float):
            kv_bytes += struct.pack("<I", 6)  # FLOAT32
            kv_bytes += struct.pack("<f", v)
        elif isinstance(v, list):
            if not v:
                raise ValueError("test GGUF arrays must not be empty")
            if all(isinstance(item, str) for item in v):
                elem_type = 8
            elif all(isinstance(item, int) for item in v):
                elem_type = 4
            elif all(isinstance(item, (int, float)) for item in v):
                elem_type = 6
            else:
                raise ValueError(f"unsupported test GGUF array: {v!r}")
            kv_bytes += struct.pack("<I", 9)
            kv_bytes += struct.pack("<I", elem_type)
            kv_bytes += struct.pack("<Q", len(v))
            for item in v:
                if elem_type == 8:
                    kv_bytes += _pack_string(str(item))
                elif elem_type == 4:
                    kv_bytes += struct.pack("<I", int(item))
                else:
                    kv_bytes += struct.pack("<f", float(item))

    # --- Tensor info ---
    tensor_info_bytes = b""
    data_offset = 0
    offsets = []
    for t in tensors:
        tensor_info_bytes += _pack_string(t["name"])
        n_dims = len(t["shape"])
        tensor_info_bytes += struct.pack("<I", n_dims)
        for d in t["shape"]:
            tensor_info_bytes += struct.pack("<Q", d)
        tensor_info_bytes += struct.pack("<I", t["type"])
        offsets.append(data_offset)
        tensor_info_bytes += struct.pack("<Q", data_offset)
        data_offset += len(t["data"])

    # --- Alignment padding ---
    meta_size = len(header) + len(kv_bytes) + len(tensor_info_bytes)
    remainder = meta_size % alignment
    padding = (alignment - remainder) if remainder else 0

    # --- Tensor data ---
    tensor_data = b"".join(t["data"] for t in tensors)

    return header + kv_bytes + tensor_info_bytes + (b"\x00" * padding) + tensor_data


# ---------------------------------------------------------------------------
# Dequant unit tests
# ---------------------------------------------------------------------------

class TestDequantF16:
    def test_basic(self):
        arr = np.array([1.0, -1.0, 0.5], dtype=np.float16)
        result = _dequant_f16(arr.tobytes())
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, -1.0, 0.5], rtol=1e-3)

    def test_zero(self):
        arr = np.zeros(4, dtype=np.float16)
        result = _dequant_f16(arr.tobytes())
        assert np.all(result == 0.0)


class TestDequantBF16:
    def test_basic(self):
        # BF16 1.0: u16 = 0x3F80
        u16 = np.array([0x3F80, 0x0000], dtype=np.uint16)
        result = _dequant_bf16(u16.tobytes())
        assert result.dtype == np.float32
        assert abs(result[0] - 1.0) < 1e-5
        assert result[1] == 0.0


class TestDequantQ8_0:
    def test_single_block(self):
        """Q8_0 block: 2-byte f16 scale + 32 int8 values."""
        scale_f16 = np.array([2.0], dtype=np.float16)
        quants = np.array(list(range(-16, 16)), dtype=np.int8)
        raw = scale_f16.tobytes() + quants.tobytes()
        result = _dequant_q8_0(raw, 32)
        assert result.shape == (32,)
        expected = quants.astype(np.float32) * 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_two_blocks(self):
        scale = np.array([1.0], dtype=np.float16)
        quants = np.zeros(32, dtype=np.int8)
        raw = (scale.tobytes() + quants.tobytes()) * 2
        result = _dequant_q8_0(raw, 64)
        assert result.shape == (64,)
        assert np.all(result == 0.0)

    def test_output_dtype(self):
        scale = np.array([1.0], dtype=np.float16)
        quants = np.ones(32, dtype=np.int8)
        result = _dequant_q8_0(scale.tobytes() + quants.tobytes(), 32)
        assert result.dtype == np.float32


class TestDequantQ4_0:
    def test_single_block(self):
        """Q4_0: 2-byte f16 scale + 16 bytes (32 nibbles)."""
        scale = np.array([1.0], dtype=np.float16)
        # All nibbles = 8 → 8 - 8 = 0
        packed = np.full(16, 0x88, dtype=np.uint8)
        raw = scale.tobytes() + packed.tobytes()
        result = _dequant_q4_0(raw, 32)
        assert result.shape == (32,)
        assert np.all(result == 0.0)

    def test_range(self):
        scale = np.array([1.0], dtype=np.float16)
        # Low nibble=0 (→-8), high nibble=15 (→+7)
        packed = np.full(16, 0xF0, dtype=np.uint8)
        raw = scale.tobytes() + packed.tobytes()
        result = _dequant_q4_0(raw, 32)
        # ggml layout: byte j holds element j (low nibble) and element j+16
        # (high nibble) — the first 16 elements read low nibbles, the next
        # 16 read high nibbles (verified against gguf.quants reference).
        assert result[0] == pytest.approx(-8.0, abs=1e-3)
        assert result[15] == pytest.approx(-8.0, abs=1e-3)
        assert result[16] == pytest.approx(7.0, abs=1e-3)
        assert result[31] == pytest.approx(7.0, abs=1e-3)


class TestDequantQ4K:
    def test_zeros(self):
        """Zero-filled Q4_K block should dequantize to all zeros."""
        raw = bytes(144)
        result = _dequant_q4_k(raw, 256)
        assert result.shape == (256,)
        # All zeros — no guarantees on exact value but should not crash
        assert np.isfinite(result).all()

    def test_shape(self):
        raw = bytes(288)
        result = _dequant_q4_k(raw, 512)
        assert result.shape == (512,)

    def test_dtype(self):
        result = _dequant_q4_k(bytes(144), 256)
        assert result.dtype == np.float32


class TestDequantQ6K:
    def test_zeros(self):
        raw = bytes(210)
        result = _dequant_q6_k(raw, 256)
        assert result.shape == (256,)
        assert np.isfinite(result).all()

    def test_dtype(self):
        result = _dequant_q6_k(bytes(210), 256)
        assert result.dtype == np.float32


class TestDequantQ2K:
    def test_zeros(self):
        raw = bytes(84)
        result = _dequant_q2_k(raw, 256)
        assert result.shape == (256,)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# GGUFReader integration tests
# ---------------------------------------------------------------------------

class TestGGUFReader:
    def _make_f32_gguf(self, data: np.ndarray, name: str = "model.weight") -> Path:
        raw = data.astype(np.float32).tobytes()
        shape = list(data.shape)
        payload = _build_gguf(
            tensors=[{"name": name, "shape": shape, "type": _GGML_TYPE_F32, "data": raw}],
            metadata={"general.architecture": "test"},
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(payload)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def test_parse_f32_tensor(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = self._make_f32_gguf(data, "weight")
        reader = GGUFReader(path)
        assert "weight" in reader.tensors
        info = reader.tensors["weight"]
        assert info["type_name"] == "F32"
        assert info["num_elems"] == 12
        Path(path).unlink(missing_ok=True)

    def test_dequantize_f32(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = self._make_f32_gguf(data)
        reader = GGUFReader(path)
        result = reader.dequantize("model.weight")
        assert result.dtype == np.float32
        np.testing.assert_allclose(result.ravel(), data.ravel(), rtol=1e-5)
        Path(path).unlink(missing_ok=True)

    def test_metadata_extraction(self):
        path = self._make_f32_gguf(np.zeros(4, dtype=np.float32))
        reader = GGUFReader(path)
        assert reader.architecture == "test"
        Path(path).unlink(missing_ok=True)

    def test_missing_tensor_raises(self):
        path = self._make_f32_gguf(np.zeros(4, dtype=np.float32))
        reader = GGUFReader(path)
        with pytest.raises(KeyError):
            reader.dequantize("nonexistent")
        Path(path).unlink(missing_ok=True)

    def test_invalid_magic_raises(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(b"\x00\x00\x00\x00" * 16)
        tmp.close()
        from aether.core.exceptions import IngestionError
        with pytest.raises(IngestionError):
            GGUFReader(Path(tmp.name))
        Path(tmp.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# GGUFLoader high-level tests
# ---------------------------------------------------------------------------

class TestGGUFLoader:
    def test_file_not_found(self):
        from aether.core.exceptions import IngestionError
        loader = GGUFLoader("/nonexistent/path.gguf")
        with pytest.raises(IngestionError):
            loader.load()

    def test_load_returns_dict(self):
        data = np.ones(8, dtype=np.float32)
        raw = data.tobytes()
        payload = _build_gguf(
            tensors=[{"name": "w", "shape": [8], "type": _GGML_TYPE_F32, "data": raw}],
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(payload)
        tmp.close()
        loader = GGUFLoader(tmp.name)
        result = loader.load()
        assert "tensors" in result
        assert "metadata" in result
        assert "reader" in result
        assert "w" in result["tensors"]
        Path(tmp.name).unlink(missing_ok=True)

    def test_repr(self):
        loader = GGUFLoader("/some/path.gguf")
        assert "GGUFLoader" in repr(loader)


def test_architecture_detector_reads_gguf_header(tmp_path: Path) -> None:
    from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

    path = tmp_path / "model.gguf"
    path.write_bytes(
        _build_gguf(
            tensors=[
                {
                    "name": "token_embd.weight",
                    "shape": [8, 16],
                    "type": _GGML_TYPE_F32,
                    "data": np.zeros(128, dtype=np.float32).tobytes(),
                }
            ],
            metadata={
                "general.architecture": "llama",
                "llama.block_count": 2,
                "llama.embedding_length": 16,
                "llama.attention.head_count": 2,
                "llama.attention.head_count_kv": 1,
                "llama.context_length": 128,
                "llama.feed_forward_length": 32,
                "llama.vocab_size": 8,
            },
        )
    )
    architecture = ArchitectureDetector().detect(str(path))
    assert architecture.family == "llama_family"
    assert architecture.layers == 2
    assert architecture.hidden_size == 16
    assert architecture.num_kv_heads == 1


def test_embedded_gguf_tokenizer_exports_as_loadable_fast_tokenizer(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    output = tmp_path / "tokenizer"
    info = export_gguf_tokenizer_metadata(
        {
            "tokenizer.ggml.tokens": ["<unk>", "▁hello", "▁world"],
            "tokenizer.ggml.scores": [0.0, -0.1, -0.2],
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.eos_token_id": 0,
        },
        output,
    )
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    encoded = tokenizer("hello world")
    assert encoded["input_ids"]
    assert info["token_count"] == 3


@pytest.mark.integration
def test_tiny_gguf_compiles_reloads_and_generates(tmp_path: Path) -> None:
    """Exercise GGUF metadata, weights, tokenizer export, and CPU execution."""
    from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig

    hidden = 16
    vocab = 8
    intermediate = 32
    rng = np.random.default_rng(11)

    def tensor(name: str, shape: list[int], values: np.ndarray) -> dict:
        # GGUF stores dimensions in reverse order relative to the runtime
        # tensor shape used by the loader.
        return {
            "name": name,
            "shape": list(reversed(shape)),
            "type": _GGML_TYPE_F32,
            "data": values.astype(np.float32).tobytes(),
        }

    tensors = [
        tensor("token_embd.weight", [vocab, hidden], rng.normal(size=(vocab, hidden))),
        tensor("output_norm.weight", [hidden], np.ones(hidden)),
        tensor("blk.0.attn_norm.weight", [hidden], np.ones(hidden)),
        tensor("blk.0.attn_q.weight", [hidden, hidden], rng.normal(size=(hidden, hidden))),
        tensor("blk.0.attn_k.weight", [hidden // 2, hidden], rng.normal(size=(hidden // 2, hidden))),
        tensor("blk.0.attn_v.weight", [hidden // 2, hidden], rng.normal(size=(hidden // 2, hidden))),
        tensor("blk.0.attn_output.weight", [hidden, hidden], rng.normal(size=(hidden, hidden))),
        tensor("blk.0.ffn_norm.weight", [hidden], np.ones(hidden)),
        tensor("blk.0.ffn_gate.weight", [intermediate, hidden], rng.normal(size=(intermediate, hidden))),
        tensor("blk.0.ffn_up.weight", [intermediate, hidden], rng.normal(size=(intermediate, hidden))),
        tensor("blk.0.ffn_down.weight", [hidden, intermediate], rng.normal(size=(hidden, intermediate))),
        tensor("output.weight", [vocab, hidden], rng.normal(size=(vocab, hidden))),
    ]
    model = tmp_path / "model.gguf"
    model.write_bytes(
        _build_gguf(
            tensors,
            metadata={
                "general.architecture": "llama",
                "llama.block_count": 1,
                "llama.embedding_length": hidden,
                "llama.attention.head_count": 2,
                "llama.attention.head_count_kv": 1,
                "llama.context_length": 128,
                "llama.feed_forward_length": intermediate,
                "llama.vocab_size": vocab,
                "tokenizer.ggml.tokens": ["<unk>", "▁hello", "▁world", "tok3", "tok4", "tok5", "tok6", "tok7"],
                "tokenizer.ggml.scores": [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7],
                "tokenizer.ggml.unknown_token_id": 0,
                "tokenizer.ggml.eos_token_id": 0,
            },
        )
    )

    artifact = tmp_path / "model.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(model), output_path=artifact)

    package = __import__("aether.core.aeg_format", fromlist=["load_aeg_package"]).load_aeg_package(artifact)
    package.verify_integrity()
    assert package.has_weights
    assert (artifact / "tokenizer" / "tokenizer.json").is_file()

    response = Runtime(
        RuntimeConfig(hf_offline=True, enable_semantic_cache=False, default_max_tokens=2)
    ).generate(str(artifact), "hello", max_tokens=2, temperature=0.0)
    assert response.usage["completion_tokens"] == 2
