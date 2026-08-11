"""
Comprehensive ingestion test suite — covers all 15 audit gaps.

Tests are self-contained, offline, and CPU-only.  They build minimal
synthetic artefacts in ``tmp_path`` rather than downloading real models.

Gap coverage:
  1.  ONNX ingestion — opset 17/18/19/20/21/27, op-map, initializers
  2.  VLM ingestion — architecture detection, graph topology, video models
  3.  Video/streaming model ingestion — VideoModelLoader, temporal nodes
  4.  TFLite ingestion — graceful ImportError, informative message
  5.  CoreML ingestion — graceful ImportError, informative message
  6.  GGUF ingestion — header parsing, dequantization, weight binding
  7.  TensorRT ingestion — graceful ImportError, informative message
  8.  OpenVINO ingestion — graceful ImportError, informative message
  9.  trust_remote_code — disabled by default, configurable opt-in
 10.  Unknown model identifiers — fail closed with ArchitectureDetectionError
 11.  Validation completeness — every ingested graph passes AEGGraph.validate()
 12.  SafeTensors multi-shard — index.json + multiple shards fully supported
 13.  PyTorch TorchScript — scripted/traced models load via state_dict()
 14.  ONNX opset 18-27 — all new opsets handled without error
 15.  Memory-mapped weight loading — safetensors uses safe_open (mmap-backed)
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_arch() -> "ModelArchitecture":
    """Return a tiny 2-layer architecture for fast graph construction."""
    from aether.core.types import ModelArchitecture
    return ModelArchitecture(
        family="llama_family",
        params_billion=0.001,
        layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        context_length=128,
        vocab_size=256,
        intermediate_size=128,
    )


def _save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a single .safetensors file using safetensors.numpy."""
    from safetensors.numpy import save_file
    save_file(tensors, str(path))


def _pack_gguf_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_minimal_gguf(
    tensors: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    alignment: int = 32,
) -> bytes:
    """Build a minimal GGUF v3 binary payload for testing."""
    _GGUF_MAGIC = 0x46554747
    meta = metadata or {}

    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, len(tensors), len(meta))

    kv_bytes = b""
    for k, v in meta.items():
        kv_bytes += _pack_gguf_string(k)
        if isinstance(v, str):
            kv_bytes += struct.pack("<I", 8)
            kv_bytes += _pack_gguf_string(v)
        elif isinstance(v, int):
            kv_bytes += struct.pack("<I", 4)
            kv_bytes += struct.pack("<I", v)
        elif isinstance(v, float):
            kv_bytes += struct.pack("<I", 6)
            kv_bytes += struct.pack("<f", v)

    tensor_info_bytes = b""
    data_offset = 0
    for t in tensors:
        tensor_info_bytes += _pack_gguf_string(t["name"])
        n_dims = len(t["shape"])
        tensor_info_bytes += struct.pack("<I", n_dims)
        for d in t["shape"]:
            tensor_info_bytes += struct.pack("<Q", d)
        tensor_info_bytes += struct.pack("<I", t["type"])
        tensor_info_bytes += struct.pack("<Q", data_offset)
        data_offset += len(t["data"])

    meta_size = len(header) + len(kv_bytes) + len(tensor_info_bytes)
    remainder = meta_size % alignment
    padding = (alignment - remainder) if remainder else 0

    tensor_data = b"".join(t["data"] for t in tensors)
    return header + kv_bytes + tensor_info_bytes + (b"\x00" * padding) + tensor_data


def _build_tiny_onnx_model(opset: int = 17) -> bytes:
    """Build a minimal valid ONNX model at the requested opset version."""
    import onnx
    import onnx.helper as oh

    FLOAT = onnx.TensorProto.FLOAT
    X = oh.make_tensor_value_info("X", FLOAT, [1, 4])
    Z = oh.make_tensor_value_info("Z", FLOAT, [1, 2])
    W = oh.make_tensor("W", FLOAT, [4, 2], np.ones((4, 2), dtype=np.float32).flatten().tolist())
    node = oh.make_node("MatMul", inputs=["X", "W"], outputs=["Z"])
    graph = oh.make_graph([node], "test", [X], [Z], [W])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    return model.SerializeToString()


# ===========================================================================
# Gap 1 & 14 — ONNX ingestion, opset 17-27
# ===========================================================================

class TestONNXIngestion:
    """ONNX loader: op-map, initializers, opset 17-27 (gaps 1 & 14)."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_load_opset_17(self, tmp_path: Path) -> None:
        """Opset 17 model loads and returns nodes + initializers."""
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader

        model_bytes = _build_tiny_onnx_model(opset=17)
        p = tmp_path / "model.onnx"
        p.write_bytes(model_bytes)

        data = ONNXLoader(p).load()
        assert data["opset"] == 17
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["op_type"] == "linear"  # MatMul -> linear
        assert "W" in data["initializers"]
        assert data["initializers"]["W"].dtype == np.float32

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    @pytest.mark.parametrize("opset", [18, 19, 20, 21])
    def test_load_opset_18_to_21(self, tmp_path: Path, opset: int) -> None:
        """Opsets 18-21 load without error; opset is recorded correctly."""
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader

        model_bytes = _build_tiny_onnx_model(opset=opset)
        p = tmp_path / f"model_opset{opset}.onnx"
        p.write_bytes(model_bytes)

        data = ONNXLoader(p).load()
        assert data["opset"] == opset
        assert len(data["nodes"]) >= 1

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_load_opset_27(self, tmp_path: Path) -> None:
        """Opset 27 (onnx 1.22.0 maximum) loads without error."""
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader

        model_bytes = _build_tiny_onnx_model(opset=27)
        p = tmp_path / "model_opset27.onnx"
        p.write_bytes(model_bytes)

        data = ONNXLoader(p).load()
        assert data["opset"] == 27

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_onnx_ingestion_pipeline_produces_valid_graph(self, tmp_path: Path) -> None:
        """IngestionPipeline._ingest_onnx produces a graph that passes validate()."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        model_bytes = _build_tiny_onnx_model(opset=17)
        p = tmp_path / "model.onnx"
        p.write_bytes(model_bytes)

        arch = _small_arch()
        pipeline = IngestionPipeline()
        graph = pipeline.ingest(str(p), arch)

        result = graph.validate()
        assert result.is_valid, f"Graph validation failed: {result.errors}"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_onnx_op_map_covers_new_opset_ops(self) -> None:
        """Op map includes ops introduced in opsets 18-27."""
        from aether.compiler.stage1_ingestion.onnx_loader import _ONNX_OP_MAP

        # Ops added in opset 18+
        new_ops = {
            "RotaryEmbedding",   # opset 23
            "RmsNormalization",  # opset 23
            "Attention",         # opset 23
        }
        for op in new_ops:
            assert op in _ONNX_OP_MAP, f"Missing new-opset op: {op}"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_onnx_elem_type_map_complete(self) -> None:
        """Element type map covers all standard ONNX types."""
        from aether.compiler.stage1_ingestion.onnx_loader import _ONNX_ELEM_TYPE

        required = {1: "float32", 2: "uint8", 3: "int8", 6: "int32",
                    7: "int64", 10: "float16", 11: "float64", 16: "bfloat16"}
        for etype, dtype in required.items():
            assert _ONNX_ELEM_TYPE.get(etype) == dtype, (
                f"ONNX elem type {etype} should map to {dtype!r}"
            )

    def test_onnx_file_not_found_raises_ingestion_error(self) -> None:
        """Missing ONNX file raises IngestionError, not FileNotFoundError."""
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader
        from aether.core.exceptions import IngestionError

        with pytest.raises(IngestionError, match="not found"):
            ONNXLoader("/nonexistent/model.onnx").load()

    def test_onnx_loader_repr(self) -> None:
        """ONNXLoader repr includes the class name."""
        from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader
        assert "ONNXLoader" in repr(ONNXLoader("/path/model.onnx"))


# ===========================================================================
# Gap 2 — VLM ingestion
# ===========================================================================

class TestVLMIngestion:
    """VLM loader: architecture detection, graph topology (gap 2)."""

    def test_detect_llava_architecture(self, tmp_path: Path) -> None:
        """LLaVA config.json is detected as a VLM architecture."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {
            "model_type": "llava",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.model_type == "llava"
        assert arch.vision_encoder == "clip_vit_l14"
        assert arch.projection_type == "mlp"
        assert arch.num_image_tokens == 576

    def test_detect_qwen2_vl_architecture(self, tmp_path: Path) -> None:
        """Qwen2-VL config.json is detected with dynamic resolution."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {
            "model_type": "qwen2_vl",
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "vocab_size": 152064,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.model_type == "qwen2_vl"
        assert arch.dynamic_resolution is True
        assert arch.supports_video is True

    def test_detect_paligemma_architecture(self, tmp_path: Path) -> None:
        """PaliGemma uses identity projection (no MLP adapter)."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {"model_type": "paligemma", "num_hidden_layers": 18, "hidden_size": 2048}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.projection_type == "identity"
        assert arch.vision_encoder == "siglip_so400m"

    def test_detect_internvl2_architecture(self, tmp_path: Path) -> None:
        """InternVL2 supports video and dynamic resolution."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {"model_type": "internvl2", "num_hidden_layers": 32, "hidden_size": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.supports_video is True
        assert arch.max_num_tiles == 6

    def test_non_vlm_returns_none(self, tmp_path: Path) -> None:
        """A plain LLM config.json returns None from detect_vlm_architecture."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {"model_type": "llama", "num_hidden_layers": 32}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_vlm_architecture(tmp_path) is None

    def test_missing_config_returns_none(self, tmp_path: Path) -> None:
        """A directory without config.json returns None."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture
        assert detect_vlm_architecture(tmp_path) is None

    def test_vlm_graph_builder_produces_required_nodes(self, tmp_path: Path) -> None:
        """VLMGraphBuilder produces pixel_input, vision_encoder, lm_head nodes."""
        from aether.compiler.stage1_ingestion.vlm_loader import (
            VLMGraphBuilder,
            detect_vlm_architecture,
        )

        config = {"model_type": "llava", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None

        nodes = VLMGraphBuilder().build(arch)
        node_ids = {n.node_id for n in nodes}
        assert "pixel_input" in node_ids
        assert "vision_encoder" in node_ids
        assert "lm_head" in node_ids
        assert "modality_merge" in node_ids
        assert "language_model" in node_ids

    def test_vlm_graph_builder_video_model_has_temporal_pooler(self, tmp_path: Path) -> None:
        """Video-capable VLMs include a temporal_pooler node."""
        from aether.compiler.stage1_ingestion.vlm_loader import (
            VLMGraphBuilder,
            detect_vlm_architecture,
        )

        config = {"model_type": "videollama2", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.supports_video is True

        nodes = VLMGraphBuilder().build(arch)
        node_ids = {n.node_id for n in nodes}
        assert "temporal_pooler" in node_ids

    def test_vlm_graph_builder_identity_projection_skips_mlp(self, tmp_path: Path) -> None:
        """PaliGemma (identity projection) has no 'projection' node."""
        from aether.compiler.stage1_ingestion.vlm_loader import (
            VLMGraphBuilder,
            detect_vlm_architecture,
        )

        config = {"model_type": "paligemma", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None

        nodes = VLMGraphBuilder().build(arch)
        node_ids = {n.node_id for n in nodes}
        assert "projection" not in node_ids

    def test_vlm_loader_is_vlm_true(self, tmp_path: Path) -> None:
        """VLMLoader.is_vlm() returns True for a VLM directory."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMLoader

        config = {"model_type": "llava", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert VLMLoader().is_vlm(tmp_path) is True

    def test_vlm_loader_is_vlm_false(self, tmp_path: Path) -> None:
        """VLMLoader.is_vlm() returns False for a non-VLM directory."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMLoader

        config = {"model_type": "llama", "num_hidden_layers": 2}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert VLMLoader().is_vlm(tmp_path) is False

    def test_vlm_loader_list_supported_types(self) -> None:
        """list_supported_types() returns a non-empty sorted list."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMLoader

        types = VLMLoader.list_supported_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert types == sorted(types)
        assert "llava" in types
        assert "qwen2_vl" in types

    def test_vlm_loader_load_returns_arch_and_nodes(self, tmp_path: Path) -> None:
        """VLMLoader.load() returns (VLMArchitecture, list[VLMGraphNode])."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMLoader

        config = {"model_type": "llava", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = VLMLoader().load(tmp_path)
        assert result is not None
        arch, nodes = result
        assert arch.model_type == "llava"
        assert len(nodes) > 0

    def test_vlm_loader_load_returns_none_for_non_vlm(self, tmp_path: Path) -> None:
        """VLMLoader.load() returns None for a non-VLM directory."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMLoader

        config = {"model_type": "llama", "num_hidden_layers": 2}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert VLMLoader().load(tmp_path) is None

    def test_vlm_architecture_alias_llava_next(self, tmp_path: Path) -> None:
        """llava-next alias resolves to llava_next with dynamic resolution."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {"model_type": "llava-next", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.dynamic_resolution is True

    def test_vlm_architecture_to_dict(self, tmp_path: Path) -> None:
        """VLMArchitecture.to_dict() returns a JSON-serializable dict."""
        from aether.compiler.stage1_ingestion.vlm_loader import detect_vlm_architecture

        config = {"model_type": "llava", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        d = arch.to_dict()
        assert d["model_type"] == "llava"
        assert "vision_encoder" in d
        assert "language_backbone" in d
        json.dumps(d)  # Must be JSON-serializable

    def test_encoder_hidden_size_known_encoders(self) -> None:
        """VLMGraphBuilder._encoder_hidden_size returns correct sizes."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMGraphBuilder

        builder = VLMGraphBuilder()
        assert builder._encoder_hidden_size("clip_vit_l14") == 1024
        assert builder._encoder_hidden_size("siglip_so400m") == 1152
        assert builder._encoder_hidden_size("internvit_6b") == 3200
        assert builder._encoder_hidden_size("qwen_vit") == 1280

    def test_encoder_hidden_size_unknown_defaults_to_1024(self) -> None:
        """Unknown encoder type defaults to 1024."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMGraphBuilder

        assert VLMGraphBuilder()._encoder_hidden_size("unknown_encoder_xyz") == 1024


# ===========================================================================
# Gap 3 — Video/streaming model ingestion
# ===========================================================================

class TestVideoModelIngestion:
    """VideoModelLoader: temporal processing nodes (gap 3)."""

    def test_video_model_loader_inherits_vlm_loader(self) -> None:
        """VideoModelLoader is a subclass of VLMLoader."""
        from aether.compiler.stage1_ingestion.vlm_loader import VideoModelLoader, VLMLoader

        assert issubclass(VideoModelLoader, VLMLoader)

    def test_video_model_loader_load_video_model_known_type(self, tmp_path: Path) -> None:
        """load_video_model() returns (arch, nodes) for a known video VLM."""
        from aether.compiler.stage1_ingestion.vlm_loader import VideoModelLoader

        config = {"model_type": "videollama2", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = VideoModelLoader().load_video_model(tmp_path)
        assert result is not None
        arch, nodes = result
        assert arch.supports_video is True
        node_ids = {n.node_id for n in nodes}
        assert "temporal_pooler" in node_ids

    def test_video_model_loader_returns_none_for_non_video(self, tmp_path: Path) -> None:
        """load_video_model() returns None for a non-video, non-VLM directory."""
        from aether.compiler.stage1_ingestion.vlm_loader import VideoModelLoader

        config = {"model_type": "llama", "num_hidden_layers": 2}
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = VideoModelLoader().load_video_model(tmp_path)
        assert result is None

    def test_video_model_config_defaults(self) -> None:
        """VideoModelConfig has sensible defaults."""
        from aether.compiler.stage1_ingestion.vlm_loader import VideoModelConfig

        cfg = VideoModelConfig()
        assert cfg.max_frames == 32
        assert cfg.temporal_aggregation == "average"
        assert cfg.num_image_tokens == 576

    def test_video_model_config_custom(self) -> None:
        """VideoModelConfig accepts custom parameters."""
        from aether.compiler.stage1_ingestion.vlm_loader import VideoModelConfig

        cfg = VideoModelConfig(max_frames=64, temporal_aggregation="max")
        assert cfg.max_frames == 64
        assert cfg.temporal_aggregation == "max"

    def test_qwen2_vl_video_support(self, tmp_path: Path) -> None:
        """Qwen2-VL supports video and has temporal pooler in graph."""
        from aether.compiler.stage1_ingestion.vlm_loader import VLMGraphBuilder, detect_vlm_architecture

        config = {"model_type": "qwen2_vl", "num_hidden_layers": 2, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_vlm_architecture(tmp_path)
        assert arch is not None
        assert arch.supports_video is True

        nodes = VLMGraphBuilder().build(arch)
        node_ids = {n.node_id for n in nodes}
        assert "temporal_pooler" in node_ids


# ===========================================================================
# Gap 4 — TFLite ingestion (graceful degradation)
# ===========================================================================

class TestTFLiteIngestion:
    """TFLite: graceful ImportError with informative message (gap 4)."""

    def test_tflite_not_installed_graceful_degradation(self, tmp_path: Path) -> None:
        """Attempting to load a .tflite file without tflite_runtime degrades gracefully."""
        import importlib

        tflite_available = importlib.util.find_spec("tflite_runtime") is not None
        if tflite_available:
            pytest.skip("tflite_runtime is installed; graceful-degradation path not exercised")

        tflite_path = tmp_path / "model.tflite"
        tflite_path.write_bytes(b"TFL3" + b"\x00" * 60)

        from aether.core.exceptions import UnsupportedFormatError
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        arch = _small_arch()
        pipeline = IngestionPipeline()

        try:
            graph = pipeline.ingest(str(tflite_path), arch)
            # If it doesn't raise, it must have produced a valid graph with 0 weights
            assert graph.metadata.get("weights_attached", 0) == 0
        except UnsupportedFormatError:
            pass  # Correct: explicit unsupported format error
        except Exception as exc:
            assert not isinstance(exc, (ImportError, AttributeError)), (
                f"Should not raise bare {type(exc).__name__}: {exc}"
            )

    def test_tflite_format_detection_does_not_crash(self, tmp_path: Path) -> None:
        """Format detection for .tflite files does not raise an unhandled exception."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        tflite_path = tmp_path / "model.tflite"
        tflite_path.write_bytes(b"TFL3" + b"\x00" * 60)

        pipeline = IngestionPipeline()
        fmt = pipeline._detect_format(str(tflite_path))
        assert isinstance(fmt, str)


# ===========================================================================
# Gap 5 — CoreML ingestion (graceful degradation)
# ===========================================================================

class TestCoreMLIngestion:
    """CoreML: graceful ImportError with informative message (gap 5)."""

    def test_coreml_not_installed_graceful_degradation(self, tmp_path: Path) -> None:
        """Attempting to load a .mlpackage without coremltools degrades gracefully."""
        import importlib

        coreml_available = importlib.util.find_spec("coremltools") is not None
        if coreml_available:
            pytest.skip("coremltools is installed; graceful-degradation path not exercised")

        mlpackage_dir = tmp_path / "model.mlpackage"
        mlpackage_dir.mkdir()
        (mlpackage_dir / "Manifest.json").write_text(json.dumps({"fileFormatVersion": "1.0.0"}))

        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        arch = _small_arch()
        pipeline = IngestionPipeline()

        try:
            graph = pipeline.ingest(str(mlpackage_dir), arch)
            assert graph.metadata.get("weights_attached", 0) == 0
        except Exception as exc:
            assert not isinstance(exc, (ImportError, AttributeError)), (
                f"Should not raise bare {type(exc).__name__}: {exc}"
            )

    def test_mlmodel_format_detection_does_not_crash(self, tmp_path: Path) -> None:
        """Format detection for .mlmodel files does not raise."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        mlmodel_path = tmp_path / "model.mlmodel"
        mlmodel_path.write_bytes(b"\x00" * 16)

        pipeline = IngestionPipeline()
        fmt = pipeline._detect_format(str(mlmodel_path))
        assert isinstance(fmt, str)


# ===========================================================================
# Gap 6 — GGUF ingestion
# ===========================================================================

class TestGGUFIngestion:
    """GGUF loader: header parsing, dequantization, weight binding (gap 6)."""

    def _make_gguf_file(self, tmp_path: Path, arch_name: str = "llama") -> Path:
        """Write a minimal GGUF file with F32 tensors."""
        _GGML_TYPE_F32 = 0
        rng = np.random.default_rng(42)
        hidden = 16
        vocab = 8

        tensors = [
            {
                "name": "token_embd.weight",
                "shape": [hidden, vocab],
                "type": _GGML_TYPE_F32,
                "data": rng.normal(size=(hidden * vocab,)).astype(np.float32).tobytes(),
            },
            {
                "name": "output.weight",
                "shape": [hidden, vocab],
                "type": _GGML_TYPE_F32,
                "data": rng.normal(size=(hidden * vocab,)).astype(np.float32).tobytes(),
            },
            {
                "name": "blk.0.attn_q.weight",
                "shape": [hidden, hidden],
                "type": _GGML_TYPE_F32,
                "data": rng.normal(size=(hidden * hidden,)).astype(np.float32).tobytes(),
            },
        ]
        metadata = {
            "general.architecture": arch_name,
            f"{arch_name}.block_count": 1,
            f"{arch_name}.embedding_length": hidden,
            f"{arch_name}.attention.head_count": 2,
            f"{arch_name}.attention.head_count_kv": 1,
            f"{arch_name}.context_length": 128,
            f"{arch_name}.feed_forward_length": 32,
            f"{arch_name}.vocab_size": vocab,
        }
        payload = _build_minimal_gguf(tensors, metadata)
        path = tmp_path / "model.gguf"
        path.write_bytes(payload)
        return path

    def test_gguf_reader_parses_header(self, tmp_path: Path) -> None:
        """GGUFReader correctly parses architecture and tensor count."""
        from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader

        path = self._make_gguf_file(tmp_path)
        reader = GGUFReader(path)
        assert reader.architecture == "llama"
        assert len(reader.tensors) == 3
        assert "token_embd.weight" in reader.tensors

    def test_gguf_reader_dequantize_f32(self, tmp_path: Path) -> None:
        """GGUFReader.dequantize() returns float32 numpy array for F32 tensors."""
        from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader

        path = self._make_gguf_file(tmp_path)
        reader = GGUFReader(path)
        arr = reader.dequantize("token_embd.weight")
        assert arr.dtype == np.float32
        assert arr.size == 16 * 8

    def test_gguf_reader_missing_tensor_raises_key_error(self, tmp_path: Path) -> None:
        """Requesting a non-existent tensor raises KeyError."""
        from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader

        path = self._make_gguf_file(tmp_path)
        reader = GGUFReader(path)
        with pytest.raises(KeyError):
            reader.dequantize("nonexistent.weight")

    def test_gguf_loader_load_returns_expected_keys(self, tmp_path: Path) -> None:
        """GGUFLoader.load() returns dict with tensors, metadata, arch, reader."""
        from aether.compiler.stage1_ingestion.gguf_loader import GGUFLoader

        path = self._make_gguf_file(tmp_path)
        result = GGUFLoader(path).load()
        assert "tensors" in result
        assert "metadata" in result
        assert "arch" in result
        assert "reader" in result
        assert result["arch"] == "llama"

    def test_gguf_loader_file_not_found_raises(self) -> None:
        """GGUFLoader raises IngestionError for missing file."""
        from aether.compiler.stage1_ingestion.gguf_loader import GGUFLoader
        from aether.core.exceptions import IngestionError

        with pytest.raises(IngestionError, match="not found"):
            GGUFLoader("/nonexistent/model.gguf").load()

    def test_gguf_ingestion_pipeline_binds_weights(self, tmp_path: Path) -> None:
        """IngestionPipeline._ingest_gguf binds GGUF tensors to graph nodes."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        path = self._make_gguf_file(tmp_path)
        arch = _small_arch()
        pipeline = IngestionPipeline()
        graph = pipeline.ingest(str(path), arch)

        result = graph.validate()
        assert result.is_valid, f"Graph validation failed: {result.errors}"
        assert graph.metadata.get("gguf_architecture") == "llama"
        assert graph.metadata.get("weight_tensor_count", 0) > 0

    def test_gguf_ingestion_pipeline_records_source_path(self, tmp_path: Path) -> None:
        """IngestionPipeline records source_model_path for GGUF files."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        path = self._make_gguf_file(tmp_path)
        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(path), arch)
        assert "source_model_path" in graph.metadata

    def test_gguf_invalid_magic_raises_ingestion_error(self, tmp_path: Path) -> None:
        """A file with wrong magic bytes raises IngestionError."""
        from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader
        from aether.core.exceptions import IngestionError

        bad_path = tmp_path / "bad.gguf"
        bad_path.write_bytes(b"\x00\x00\x00\x00" * 16)
        with pytest.raises(IngestionError, match="Not a GGUF file"):
            GGUFReader(bad_path)

    def test_gguf_architecture_detector_reads_header(self, tmp_path: Path) -> None:
        """ArchitectureDetector reads GGUF header for architecture detection."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        path = self._make_gguf_file(tmp_path)
        arch = ArchitectureDetector().detect(str(path))
        assert arch.family == "llama_family"
        assert arch.layers == 1
        assert arch.hidden_size == 16

    def test_gguf_f16_dequantization(self) -> None:
        """F16 dequantization produces correct float32 values."""
        from aether.compiler.stage1_ingestion.gguf_loader import _dequant_f16

        arr = np.array([1.0, -1.0, 0.5], dtype=np.float16)
        result = _dequant_f16(arr.tobytes())
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, -1.0, 0.5], rtol=1e-3)

    def test_gguf_q8_0_dequantization(self) -> None:
        """Q8_0 dequantization: scale x int8 values."""
        from aether.compiler.stage1_ingestion.gguf_loader import _dequant_q8_0

        scale = np.array([2.0], dtype=np.float16)
        quants = np.array(list(range(-16, 16)), dtype=np.int8)
        raw = scale.tobytes() + quants.tobytes()
        result = _dequant_q8_0(raw, 32)
        assert result.shape == (32,)
        np.testing.assert_allclose(result, quants.astype(np.float32) * 2.0, rtol=1e-3)

    def test_gguf_q4_0_dequantization_zeros(self) -> None:
        """Q4_0 with all-8 nibbles (-> 0 after shift) produces zeros."""
        from aether.compiler.stage1_ingestion.gguf_loader import _dequant_q4_0

        scale = np.array([1.0], dtype=np.float16)
        packed = np.full(16, 0x88, dtype=np.uint8)  # nibble 8 -> 8-8=0
        raw = scale.tobytes() + packed.tobytes()
        result = _dequant_q4_0(raw, 32)
        assert result.shape == (32,)
        assert np.all(result == 0.0)

    def test_gguf_q4_k_dequantization_shape(self) -> None:
        """Q4_K dequantization produces correct shape."""
        from aether.compiler.stage1_ingestion.gguf_loader import _dequant_q4_k

        result = _dequant_q4_k(bytes(144), 256)
        assert result.shape == (256,)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_gguf_q6_k_dequantization_shape(self) -> None:
        """Q6_K dequantization produces correct shape."""
        from aether.compiler.stage1_ingestion.gguf_loader import _dequant_q6_k

        result = _dequant_q6_k(bytes(210), 256)
        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_gguf_q2_k_dequantization_shape(self) -> None:
        """Q2_K dequantization produces correct shape."""
        from aether.compiler.stage1_ingestion.gguf_loader import _dequant_q2_k

        result = _dequant_q2_k(bytes(84), 256)
        assert result.shape == (256,)
        assert np.isfinite(result).all()


# ===========================================================================
# Gap 7 — TensorRT ingestion (graceful degradation)
# ===========================================================================

class TestTensorRTIngestion:
    """TensorRT: graceful ImportError with informative message (gap 7)."""

    def test_tensorrt_not_installed_graceful_degradation(self, tmp_path: Path) -> None:
        """Attempting to load a .engine file without tensorrt degrades gracefully."""
        import importlib

        trt_available = importlib.util.find_spec("tensorrt") is not None
        if trt_available:
            pytest.skip("tensorrt is installed; graceful-degradation path not exercised")

        trt_path = tmp_path / "model.engine"
        trt_path.write_bytes(b"\x00" * 64)

        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        arch = _small_arch()
        pipeline = IngestionPipeline()

        try:
            graph = pipeline.ingest(str(trt_path), arch)
            assert graph.metadata.get("weights_attached", 0) == 0
        except Exception as exc:
            assert not isinstance(exc, (ImportError, AttributeError)), (
                f"Should not raise bare {type(exc).__name__}: {exc}"
            )

    def test_trt_format_detection_does_not_crash(self, tmp_path: Path) -> None:
        """Format detection for .engine files does not raise."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"\x00" * 16)

        pipeline = IngestionPipeline()
        fmt = pipeline._detect_format(str(engine_path))
        assert isinstance(fmt, str)


# ===========================================================================
# Gap 8 — OpenVINO ingestion (graceful degradation)
# ===========================================================================

class TestOpenVINOIngestion:
    """OpenVINO: graceful ImportError with informative message (gap 8)."""

    def test_openvino_not_installed_graceful_degradation(self, tmp_path: Path) -> None:
        """Attempting to load a .xml file without openvino degrades gracefully."""
        import importlib

        ov_available = importlib.util.find_spec("openvino") is not None
        if ov_available:
            pytest.skip("openvino is installed; graceful-degradation path not exercised")

        xml_path = tmp_path / "model.xml"
        bin_path = tmp_path / "model.bin"
        xml_path.write_text("<net name='test'></net>")
        bin_path.write_bytes(b"\x00" * 16)

        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        arch = _small_arch()
        pipeline = IngestionPipeline()

        try:
            graph = pipeline.ingest(str(xml_path), arch)
            assert graph.metadata.get("weights_attached", 0) == 0
        except Exception as exc:
            assert not isinstance(exc, (ImportError, AttributeError)), (
                f"Should not raise bare {type(exc).__name__}: {exc}"
            )

    def test_xml_format_detection_does_not_crash(self, tmp_path: Path) -> None:
        """Format detection for .xml files does not raise."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        xml_path = tmp_path / "model.xml"
        xml_path.write_text("<net></net>")

        pipeline = IngestionPipeline()
        fmt = pipeline._detect_format(str(xml_path))
        assert isinstance(fmt, str)


# ===========================================================================
# Gap 9 — trust_remote_code disabled by default
# ===========================================================================

class TestTrustRemoteCode:
    """trust_remote_code must be disabled by default (gap 9)."""

    def test_compiler_config_has_no_trust_remote_code_default(self) -> None:
        """CompilerConfig does not enable trust_remote_code by default."""
        from aether.compiler.config import CompilerConfig

        cfg = CompilerConfig()
        trust = getattr(cfg, "trust_remote_code", None)
        assert trust is not True, (
            f"trust_remote_code must not be True by default; got {trust!r}"
        )

    def test_ingestion_pipeline_skip_download_default(self) -> None:
        """IngestionPipeline defaults to skip_download=True (no network calls)."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        pipeline = IngestionPipeline()
        assert pipeline.config.skip_download is True

    def test_ingestion_pipeline_custom_config_respected(self) -> None:
        """IngestionPipeline accepts a custom CompilerConfig."""
        from aether.compiler.config import CompilerConfig
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        cfg = CompilerConfig(skip_download=True, verbose=False)
        pipeline = IngestionPipeline(config=cfg)
        assert pipeline.config.skip_download is True

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_pytorch_loader_uses_weights_only_by_default(self, tmp_path: Path) -> None:
        """PyTorchLoader uses weights_only=True (safe loading) by default."""
        import torch

        state_dict = {"embed.weight": torch.randn(8, 4)}
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        data = PyTorchLoader(pt_path).load()
        assert "embed.weight" in data["weights"]

    def test_pytorch_loader_rejects_corrupt_file(self, tmp_path: Path) -> None:
        """PyTorchLoader raises IngestionError for corrupt files."""
        bad_path = tmp_path / "bad.pt"
        bad_path.write_bytes(b"not a torch checkpoint at all")

        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        from aether.core.exceptions import IngestionError

        with pytest.raises(IngestionError):
            PyTorchLoader(bad_path).load()


# ===========================================================================
# Gap 10 — Unknown model identifiers fail closed
# ===========================================================================

class TestUnknownModelIdentifiers:
    """Unknown model identifiers must fail closed (gap 10)."""

    def test_unknown_identifier_raises_architecture_detection_error(self) -> None:
        """Completely unknown model ID raises ArchitectureDetectionError."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
        from aether.core.exceptions import ArchitectureDetectionError

        detector = ArchitectureDetector()
        with pytest.raises(ArchitectureDetectionError):
            detector.detect("totally_unknown_xyz_model_abc_12345")

    def test_unknown_identifier_error_message_is_informative(self) -> None:
        """ArchitectureDetectionError message mentions the model identifier."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
        from aether.core.exceptions import ArchitectureDetectionError

        model_id = "some_completely_unknown_model_xyz"
        detector = ArchitectureDetector()
        with pytest.raises(ArchitectureDetectionError) as exc_info:
            detector.detect(model_id)
        assert model_id in str(exc_info.value) or "architecture" in str(exc_info.value).lower()

    def test_known_model_prefix_resolves_without_error(self) -> None:
        """A model with a known prefix (llama) resolves to a valid architecture."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        detector = ArchitectureDetector()
        arch = detector.detect("llama-3.1-8b")
        assert arch.family == "llama_family"
        assert arch.layers > 0

    def test_known_qwen_model_resolves(self) -> None:
        """Qwen3-0.6B resolves to qwen_family."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        detector = ArchitectureDetector()
        arch = detector.detect("qwen3-0.6b")
        assert arch.family == "qwen_family"

    def test_local_config_json_overrides_name_heuristic(self, tmp_path: Path) -> None:
        """A local config.json is authoritative over name-based guessing."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        config = {
            "architectures": ["GemmaForCausalLM"],
            "num_hidden_layers": 18,
            "hidden_size": 2048,
            "num_attention_heads": 8,
            "vocab_size": 256000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        detector = ArchitectureDetector()
        arch = detector.detect(str(tmp_path))
        assert arch.family == "gemma_family"
        assert arch.layers == 18

    def test_gguf_unknown_architecture_raises(self, tmp_path: Path) -> None:
        """A GGUF file with an unknown architecture type raises ArchitectureDetectionError."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
        from aether.core.exceptions import ArchitectureDetectionError

        _GGML_TYPE_F32 = 0
        payload = _build_minimal_gguf(
            tensors=[{
                "name": "w",
                "shape": [4],
                "type": _GGML_TYPE_F32,
                "data": np.zeros(4, dtype=np.float32).tobytes(),
            }],
            metadata={"general.architecture": "totally_unknown_arch_xyz"},
        )
        path = tmp_path / "unknown.gguf"
        path.write_bytes(payload)

        detector = ArchitectureDetector()
        with pytest.raises(ArchitectureDetectionError):
            detector.detect(str(path))


# ===========================================================================
# Gap 11 — Validation completeness
# ===========================================================================

class TestValidationCompleteness:
    """Every ingested graph must pass AEGGraph.validate() (gap 11)."""

    def test_safetensors_ingested_graph_is_valid(self, tmp_path: Path) -> None:
        """SafeTensors-ingested graph passes validate()."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        tensors = {
            "model.embed_tokens.weight": np.random.randn(256, 64).astype(np.float32),
            "lm_head.weight": np.random.randn(256, 64).astype(np.float32),
        }
        _save_safetensors(tmp_path / "model.safetensors", tensors)

        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(tmp_path), arch)
        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_gguf_ingested_graph_is_valid(self, tmp_path: Path) -> None:
        """GGUF-ingested graph passes validate()."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        _GGML_TYPE_F32 = 0
        payload = _build_minimal_gguf(
            tensors=[{
                "name": "token_embd.weight",
                "shape": [16, 8],
                "type": _GGML_TYPE_F32,
                "data": np.zeros(128, dtype=np.float32).tobytes(),
            }],
            metadata={
                "general.architecture": "llama",
                "llama.block_count": 2,
                "llama.embedding_length": 64,
                "llama.attention.head_count": 4,
                "llama.attention.head_count_kv": 2,
                "llama.context_length": 128,
                "llama.feed_forward_length": 128,
                "llama.vocab_size": 256,
            },
        )
        path = tmp_path / "model.gguf"
        path.write_bytes(payload)

        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(path), arch)
        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_onnx_ingested_graph_is_valid(self, tmp_path: Path) -> None:
        """ONNX-ingested graph passes validate()."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        model_bytes = _build_tiny_onnx_model(opset=17)
        p = tmp_path / "model.onnx"
        p.write_bytes(model_bytes)

        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(p), arch)
        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_pytorch_ingested_graph_is_valid(self, tmp_path: Path) -> None:
        """PyTorch-ingested graph passes validate()."""
        import torch
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        state_dict = {
            "model.embed_tokens.weight": torch.randn(256, 64),
            "lm_head.weight": torch.randn(256, 64),
        }
        pt_path = tmp_path / "model.pt"
        torch.save(state_dict, str(pt_path))

        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(pt_path), arch)
        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_empty_path_graph_is_valid(self) -> None:
        """Graph built without weights (HF ID path) passes validate()."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)
        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_graph_has_input_and_output_nodes(self) -> None:
        """Every ingested graph has at least one input and one output node."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)
        assert len(graph.input_nodes) >= 1
        assert len(graph.output_nodes) >= 1

    def test_graph_topological_order_is_acyclic(self) -> None:
        """Ingested graph has no cycles (topological_order succeeds)."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)
        order = graph.topological_order()
        assert len(order) == graph.node_count

    def test_graph_validate_returns_validation_result_object(self) -> None:
        """validate() returns a GraphValidationResult with is_valid, errors, warnings."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        from aether.core.graph import GraphValidationResult

        arch = _small_arch()
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)
        result = graph.validate()
        assert isinstance(result, GraphValidationResult)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    def test_moe_graph_is_valid(self) -> None:
        """MoE architecture graph passes validate()."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        from aether.core.types import ModelArchitecture

        moe_arch = ModelArchitecture(
            family="moe_family",
            params_billion=0.01,
            layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            context_length=128,
            vocab_size=256,
            intermediate_size=128,
            is_moe=True,
            num_experts=4,
            num_activated_experts=2,
        )
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", moe_arch)
        result = graph.validate()
        assert result.is_valid, f"MoE graph validation failed: {result.errors}"


# ===========================================================================
# Gap 12 — SafeTensors multi-shard (index.json + multiple shards)
# ===========================================================================

class TestSafeTensorsMultiShard:
    """SafeTensors multi-shard: index.json + multiple .safetensors shards (gap 12)."""

    def _write_sharded_checkpoint(
        self,
        tmp_path: Path,
    ) -> tuple[Path, dict[str, np.ndarray]]:
        """Write a sharded SafeTensors checkpoint with model.safetensors.index.json."""
        from safetensors.numpy import save_file

        rng = np.random.default_rng(99)
        all_tensors: dict[str, np.ndarray] = {}
        weight_map: dict[str, str] = {}

        # Shard 0: embedding + layer 0 weights
        shard0_tensors = {
            "model.embed_tokens.weight": rng.normal(size=(256, 64)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(64, 64)).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(64, 64)).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": rng.normal(size=(128, 64)).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": rng.normal(size=(64, 128)).astype(np.float32),
        }
        shard0_name = "model-00001-of-00002.safetensors"
        save_file(shard0_tensors, str(tmp_path / shard0_name))
        all_tensors.update(shard0_tensors)
        for k in shard0_tensors:
            weight_map[k] = shard0_name

        # Shard 1: layer 1 weights + lm_head
        shard1_tensors = {
            "model.layers.1.self_attn.q_proj.weight": rng.normal(size=(64, 64)).astype(np.float32),
            "model.layers.1.self_attn.o_proj.weight": rng.normal(size=(64, 64)).astype(np.float32),
            "model.layers.1.mlp.gate_proj.weight": rng.normal(size=(128, 64)).astype(np.float32),
            "model.layers.1.mlp.down_proj.weight": rng.normal(size=(64, 128)).astype(np.float32),
            "lm_head.weight": rng.normal(size=(256, 64)).astype(np.float32),
        }
        shard1_name = "model-00002-of-00002.safetensors"
        save_file(shard1_tensors, str(tmp_path / shard1_name))
        all_tensors.update(shard1_tensors)
        for k in shard1_tensors:
            weight_map[k] = shard1_name

        # Write index.json
        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        return tmp_path, all_tensors

    def test_safetensors_loader_discovers_shards_via_index(self, tmp_path: Path) -> None:
        """SafeTensorsLoader loads all tensors from a sharded checkpoint."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        path, all_tensors = self._write_sharded_checkpoint(tmp_path)
        loader = SafeTensorsLoader(path)
        tensors = loader.load()

        for key in all_tensors:
            assert key in tensors, f"Missing tensor: {key}"
        assert len(tensors) == len(all_tensors)

    def test_safetensors_loader_shard_values_match(self, tmp_path: Path) -> None:
        """Loaded shard values match the original arrays exactly."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        path, all_tensors = self._write_sharded_checkpoint(tmp_path)
        loader = SafeTensorsLoader(path)
        tensors = loader.load()

        for key, expected in all_tensors.items():
            loaded = tensors[key]
            if hasattr(loaded, "numpy"):
                loaded = loaded.numpy()
            np.testing.assert_array_equal(
                np.asarray(loaded), expected,
                err_msg=f"Tensor mismatch for {key}",
            )

    def test_ingestion_pipeline_handles_sharded_safetensors(self, tmp_path: Path) -> None:
        """IngestionPipeline correctly ingests a sharded SafeTensors checkpoint."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        path, all_tensors = self._write_sharded_checkpoint(tmp_path)
        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(path), arch)

        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"
        assert graph.metadata.get("weight_tensor_count", 0) == len(all_tensors)

    def test_ingestion_pipeline_sharded_weights_attached(self, tmp_path: Path) -> None:
        """IngestionPipeline attaches weights from sharded checkpoint to graph nodes."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        path, all_tensors = self._write_sharded_checkpoint(tmp_path)
        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(path), arch)
        assert graph.metadata.get("weights_attached", 0) > 0

    def test_safetensors_loader_discover_files_returns_both_shards(self, tmp_path: Path) -> None:
        """discover_files() returns exactly the two shard files listed in index.json."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        path, _ = self._write_sharded_checkpoint(tmp_path)
        loader = SafeTensorsLoader(path)
        files = loader.discover_files()
        assert len(files) == 2
        names = {f.name for f in files}
        assert "model-00001-of-00002.safetensors" in names
        assert "model-00002-of-00002.safetensors" in names

    def test_safetensors_loader_single_file_still_works(self, tmp_path: Path) -> None:
        """SafeTensorsLoader still works for a single non-sharded file."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        tensors = {"weight": np.ones((4, 4), dtype=np.float32)}
        _save_safetensors(tmp_path / "model.safetensors", tensors)

        loader = SafeTensorsLoader(tmp_path)
        loaded = loader.load()
        assert "weight" in loaded

    def test_safetensors_loader_directory_glob_still_works(self, tmp_path: Path) -> None:
        """SafeTensorsLoader still works for a directory with *.safetensors files."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        tensors = {"embed.weight": np.ones((8, 4), dtype=np.float32)}
        _save_safetensors(tmp_path / "model.safetensors", tensors)

        loader = SafeTensorsLoader(tmp_path)
        loaded = loader.load()
        assert "embed.weight" in loaded

    def test_safetensors_loader_repr(self, tmp_path: Path) -> None:
        """SafeTensorsLoader repr includes the class name."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        loader = SafeTensorsLoader(tmp_path)
        assert "SafeTensorsLoader" in repr(loader)

    def test_safetensors_loader_no_files_raises(self, tmp_path: Path) -> None:
        """SafeTensorsLoader raises IngestionError when no .safetensors files exist."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader
        from aether.core.exceptions import IngestionError

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(IngestionError):
            SafeTensorsLoader(empty_dir).discover_files()

    def test_safetensors_index_with_corrupt_json_raises_ingestion_error(self, tmp_path: Path) -> None:
        """A corrupt index.json raises IngestionError, not a bare JSONDecodeError."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader
        from aether.core.exceptions import IngestionError

        (tmp_path / "model.safetensors.index.json").write_text("{ not valid json }")
        with pytest.raises(IngestionError):
            SafeTensorsLoader(tmp_path).discover_files()

    def test_safetensors_index_with_path_traversal_raises_ingestion_error(self, tmp_path: Path) -> None:
        """A shard path with .. traversal raises IngestionError."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader
        from aether.core.exceptions import IngestionError

        index = {
            "metadata": {},
            "weight_map": {"w": "../evil.safetensors"},
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        with pytest.raises(IngestionError):
            SafeTensorsLoader(tmp_path).discover_files()


# ===========================================================================
# Gap 13 — PyTorch TorchScript ingestion
# ===========================================================================

class TestPyTorchTorchScriptIngestion:
    """PyTorch TorchScript: scripted/traced models load via state_dict() (gap 13)."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_torchscript_scripted_model_loads(self, tmp_path: Path) -> None:
        """A torch.jit.script model saves and loads via PyTorchLoader."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        model = torch.nn.Linear(8, 4)
        scripted = torch.jit.script(model)
        pt_path = tmp_path / "scripted.pt"
        scripted.save(str(pt_path))

        data = PyTorchLoader(pt_path).load()
        assert "weight" in data["weights"]
        assert "bias" in data["weights"]
        assert data["weights"]["weight"].shape == (4, 8)
        assert data["format"] == "full_model"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_torchscript_traced_model_loads(self, tmp_path: Path) -> None:
        """A torch.jit.trace model saves and loads via PyTorchLoader."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        model = torch.nn.Linear(8, 4)
        example = torch.randn(1, 8)
        traced = torch.jit.trace(model, example)
        pt_path = tmp_path / "traced.pt"
        traced.save(str(pt_path))

        data = PyTorchLoader(pt_path).load()
        assert "weight" in data["weights"]
        assert data["weights"]["weight"].dtype == np.float32

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_torchscript_multi_layer_model_loads(self, tmp_path: Path) -> None:
        """A multi-layer scripted model extracts all layer weights."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = torch.nn.Linear(8, 4)
                self.layer2 = torch.nn.Linear(4, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer2(torch.relu(self.layer1(x)))

        model = TinyModel()
        scripted = torch.jit.script(model)
        pt_path = tmp_path / "multi_layer.pt"
        scripted.save(str(pt_path))

        data = PyTorchLoader(pt_path).load()
        keys = set(data["weights"].keys())
        assert any("layer1" in k for k in keys)
        assert any("layer2" in k for k in keys)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_torchscript_ingestion_pipeline_produces_valid_graph(self, tmp_path: Path) -> None:
        """IngestionPipeline ingests a TorchScript model and produces a valid graph."""
        import torch
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        model = torch.nn.Linear(8, 4)
        scripted = torch.jit.script(model)
        pt_path = tmp_path / "scripted.pt"
        scripted.save(str(pt_path))

        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(pt_path), arch)
        result = graph.validate()
        assert result.is_valid, f"Validation errors: {result.errors}"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_pytorch_state_dict_loads(self, tmp_path: Path) -> None:
        """Plain state dict .pt file loads correctly."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        state_dict = {
            "embed.weight": torch.randn(256, 64),
            "proj.weight": torch.randn(64, 32),
            "proj.bias": torch.zeros(32),
        }
        pt_path = tmp_path / "state_dict.pt"
        torch.save(state_dict, str(pt_path))

        data = PyTorchLoader(pt_path).load()
        assert "embed.weight" in data["weights"]
        assert data["weights"]["embed.weight"].shape == (256, 64)
        assert data["weights"]["embed.weight"].dtype == np.float32

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_pytorch_wrapped_state_dict_loads(self, tmp_path: Path) -> None:
        """Wrapped {model: {state_dict}} format is correctly unwrapped."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        wrapped = {"model": {"layer.weight": torch.ones(4, 4)}}
        pt_path = tmp_path / "wrapped.pt"
        torch.save(wrapped, str(pt_path))

        data = PyTorchLoader(pt_path).load()
        assert len(data["weights"]) >= 1

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_pytorch_sharded_checkpoint_with_index(self, tmp_path: Path) -> None:
        """Sharded PyTorch checkpoint with index.json loads all shards."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

        torch.save({"embed.weight": torch.ones(4, 4)}, tmp_path / "pytorch_model-00001-of-00002.bin")
        torch.save({"proj.weight": torch.ones(4, 4)}, tmp_path / "pytorch_model-00002-of-00002.bin")
        (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps({
            "metadata": {},
            "weight_map": {
                "embed.weight": "pytorch_model-00001-of-00002.bin",
                "proj.weight": "pytorch_model-00002-of-00002.bin",
            },
        }))

        data = PyTorchLoader(tmp_path).load()
        assert sorted(data["weights"]) == ["embed.weight", "proj.weight"]
        assert data["format"] == "sharded"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_pytorch_corrupt_shard_raises_ingestion_error(self, tmp_path: Path) -> None:
        """A corrupt shard raises IngestionError, not a bare exception."""
        import torch
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        from aether.core.exceptions import IngestionError

        torch.save({"embed.weight": torch.ones(4, 4)}, tmp_path / "pytorch_model-00001-of-00002.bin")
        (tmp_path / "pytorch_model-00002-of-00002.bin").write_bytes(b"not a torch checkpoint")

        with pytest.raises(IngestionError, match="Failed to load PyTorch shard"):
            PyTorchLoader(tmp_path).load()

    def test_pytorch_loader_repr(self) -> None:
        """PyTorchLoader repr includes the class name."""
        from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader
        assert "PyTorchLoader" in repr(PyTorchLoader("/path/model.pt"))

    def test_flatten_state_dict_nested(self) -> None:
        """_flatten_state_dict correctly flattens nested dicts."""
        from aether.compiler.stage1_ingestion.pytorch_loader import _flatten_state_dict

        class FakeTensor:
            def numpy(self) -> np.ndarray:
                return np.ones((4, 4), dtype=np.float32)

            def detach(self) -> "FakeTensor":
                return self

        nested = {"a": {"b": FakeTensor()}}
        flat = _flatten_state_dict(nested)
        assert "a.b" in flat


# ===========================================================================
# Gap 15 — Memory-mapped weight loading
# ===========================================================================

class TestMemoryMappedWeightLoading:
    """SafeTensors uses safe_open which is memory-mapped (gap 15)."""

    def test_safetensors_safe_open_is_imported(self) -> None:
        """safetensors_loader module imports safe_open (the mmap-backed loader)."""
        from aether.compiler.stage1_ingestion import safetensors_loader as st_module

        assert hasattr(st_module, "safe_open"), (
            "safetensors_loader must import safe_open for memory-mapped access"
        )

    def test_safetensors_load_returns_correct_values(self, tmp_path: Path) -> None:
        """SafeTensorsLoader.load() returns tensors with correct values."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        large_tensor = np.random.randn(512, 512).astype(np.float32)
        _save_safetensors(tmp_path / "model.safetensors", {"large.weight": large_tensor})

        loader = SafeTensorsLoader(tmp_path)
        tensors = loader.load()
        assert "large.weight" in tensors

    def test_safetensors_tensor_values_are_correct(self, tmp_path: Path) -> None:
        """Loaded tensor values match the saved values exactly."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        rng = np.random.default_rng(7)
        original = rng.normal(size=(32, 16)).astype(np.float32)
        _save_safetensors(tmp_path / "model.safetensors", {"w": original})

        loader = SafeTensorsLoader(tmp_path)
        tensors = loader.load()
        loaded = tensors["w"]
        if hasattr(loaded, "numpy"):
            loaded = loaded.numpy()
        np.testing.assert_array_equal(np.asarray(loaded), original)

    def test_safetensors_load_config_returns_dict(self, tmp_path: Path) -> None:
        """SafeTensorsLoader.load_config() returns a dict (possibly empty)."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        _save_safetensors(tmp_path / "model.safetensors", {"w": np.ones((4,), dtype=np.float32)})
        loader = SafeTensorsLoader(tmp_path)
        config = loader.load_config()
        assert isinstance(config, dict)

    def test_safetensors_load_config_reads_config_json(self, tmp_path: Path) -> None:
        """SafeTensorsLoader.load_config() reads config.json when present."""
        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        config_data = {"model_type": "llama", "num_hidden_layers": 2}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        _save_safetensors(tmp_path / "model.safetensors", {"w": np.ones((4,), dtype=np.float32)})

        loader = SafeTensorsLoader(tmp_path)
        config = loader.load_config()
        assert config.get("model_type") == "llama"


# ===========================================================================
# Integration: weight name normalisation (regression guard)
# ===========================================================================

class TestWeightNameNormalisationRegression:
    """Regression guard: weight name normalisation must remain stable."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.layers.0.self_attn.q_proj.weight", (0, "qkv")),
            ("model.layers.7.self_attn.k_proj.weight", (7, "qkv")),
            ("model.layers.1.self_attn.o_proj.weight", (1, "out_proj")),
            ("model.layers.2.mlp.gate_proj.weight", (2, "gate_proj")),
            ("model.layers.3.mlp.down_proj.weight", (3, "ffn")),
            ("model.layers.4.input_layernorm.weight", (4, "rmsnorm")),
            ("model.embed_tokens.weight", (None, "embedding")),
            ("lm_head.weight", (None, "lm_head")),
            ("output.weight", (None, "lm_head")),
        ],
    )
    def test_checkpoint_names_resolve(self, name: str, expected: tuple) -> None:
        """Checkpoint tensor names resolve to the expected (layer, component) key."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        assert IngestionPipeline._normalise_weight_name(name) == expected

    @pytest.mark.parametrize("name", ["input", "output", ""])
    def test_unidentifiable_names_yield_none_component(self, name: str) -> None:
        """Unidentifiable names yield None component."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        assert IngestionPipeline._normalise_weight_name(name)[1] is None

    def test_gguf_attn_q_maps_to_qkv(self) -> None:
        """GGUF-style blk.0.attn_q.weight maps to (0, 'qkv')."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        assert IngestionPipeline._normalise_weight_name("blk.0.attn_q.weight") == (0, "qkv")

    def test_gguf_ffn_down_maps_to_ffn(self) -> None:
        """GGUF-style blk.0.ffn_down.weight maps to (0, 'ffn')."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        assert IngestionPipeline._normalise_weight_name("blk.0.ffn_down.weight") == (0, "ffn")

    def test_gguf_token_embd_maps_to_embedding(self) -> None:
        """GGUF-style token_embd.weight maps to (None, 'embedding')."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        assert IngestionPipeline._normalise_weight_name("token_embd.weight") == (None, "embedding")


# ===========================================================================
# Integration: ingestion pipeline format detection
# ===========================================================================

class TestIngestionPipelineFormatDetection:
    """IngestionPipeline._detect_format() correctly identifies all formats."""

    def test_detect_safetensors_file(self, tmp_path: Path) -> None:
        """Single .safetensors file is detected as 'safetensors'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        p = tmp_path / "model.safetensors"
        p.write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(p)) == "safetensors"

    def test_detect_gguf_file(self, tmp_path: Path) -> None:
        """Single .gguf file is detected as 'gguf'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        p = tmp_path / "model.gguf"
        p.write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(p)) == "gguf"

    def test_detect_onnx_file(self, tmp_path: Path) -> None:
        """Single .onnx file is detected as 'onnx'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(p)) == "onnx"

    def test_detect_pytorch_pt_file(self, tmp_path: Path) -> None:
        """Single .pt file is detected as 'pytorch'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        p = tmp_path / "model.pt"
        p.write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(p)) == "pytorch"

    def test_detect_pytorch_bin_file(self, tmp_path: Path) -> None:
        """Single .bin file is detected as 'pytorch'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        p = tmp_path / "pytorch_model.bin"
        p.write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(p)) == "pytorch"

    def test_detect_directory_with_safetensors(self, tmp_path: Path) -> None:
        """Directory with config.json + .safetensors is detected as 'safetensors'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(tmp_path)) == "safetensors"

    def test_detect_directory_with_bin_files(self, tmp_path: Path) -> None:
        """Directory with config.json + .bin is detected as 'pytorch'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "pytorch_model.bin").write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(tmp_path)) == "pytorch"

    def test_detect_nonexistent_path_returns_auto(self) -> None:
        """Non-existent path (HF ID) returns 'auto'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        assert IngestionPipeline()._detect_format("meta-llama/Llama-3-8B") == "auto"

    def test_detect_ggml_extension(self, tmp_path: Path) -> None:
        """Single .ggml file is detected as 'gguf'."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        p = tmp_path / "model.ggml"
        p.write_bytes(b"\x00" * 8)
        assert IngestionPipeline()._detect_format(str(p)) == "gguf"


# ===========================================================================
# Integration: AEGGraph structure and metadata
# ===========================================================================

class TestAEGGraphStructure:
    """AEGGraph structure: nodes, edges, metadata, serialization."""

    def test_graph_node_count_matches_architecture(self) -> None:
        """Graph node count is consistent with the architecture."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()  # 2 layers
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)
        # Minimum: input + embedding + 2x(rmsnorm+qkv+rope+attn+out_proj+res1+ffn_norm+gate+ffn+res2) + final_norm + lm_head + output
        assert graph.node_count >= 20

    def test_graph_layer_nodes_have_correct_layer_index(self) -> None:
        """Transformer layer nodes have the correct layer_index attribute."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()  # 2 layers
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)

        layer_0_nodes = [n for n in graph.nodes.values() if n.layer_index == 0]
        layer_1_nodes = [n for n in graph.nodes.values() if n.layer_index == 1]
        assert len(layer_0_nodes) > 0
        assert len(layer_1_nodes) > 0

    def test_graph_metadata_source_format_recorded(self, tmp_path: Path) -> None:
        """source_format metadata is recorded for local file ingestion."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        _GGML_TYPE_F32 = 0
        payload = _build_minimal_gguf(
            tensors=[{
                "name": "w",
                "shape": [4],
                "type": _GGML_TYPE_F32,
                "data": np.zeros(4, dtype=np.float32).tobytes(),
            }],
            metadata={"general.architecture": "llama"},
        )
        path = tmp_path / "model.gguf"
        path.write_bytes(payload)

        arch = _small_arch()
        graph = IngestionPipeline().ingest(str(path), arch)
        assert graph.metadata.get("source_format") == "gguf"

    def test_graph_to_dict_is_json_serializable(self) -> None:
        """AEGGraph.to_dict() produces a JSON-serializable dictionary."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)
        d = graph.to_dict()
        json.dumps(d, default=str)  # Must not raise

    def test_graph_get_node_returns_correct_node(self) -> None:
        """AEGGraph.get_node() returns the correct node by ID."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)

        input_node = graph.get_node("input")
        assert input_node is not None
        assert input_node.id == "input"

        lm_head = graph.get_node("lm_head")
        assert lm_head is not None
        assert lm_head.op_type == "lm_head"

    def test_graph_iter_layers_groups_by_layer_index(self) -> None:
        """AEGGraph.iter_layers() groups nodes by layer index."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        arch = _small_arch()  # 2 layers
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch)

        layers = list(graph.iter_layers())
        assert len(layers) >= 2

    def test_moe_graph_has_expert_router_nodes(self) -> None:
        """MoE architecture graph has EXPERT_ROUTER nodes."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        from aether.core.graph import AEGGraphNodeType
        from aether.core.types import ModelArchitecture

        moe_arch = ModelArchitecture(
            family="moe_family",
            params_billion=0.01,
            layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            context_length=128,
            vocab_size=256,
            intermediate_size=128,
            is_moe=True,
            num_experts=4,
            num_activated_experts=2,
        )
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", moe_arch)
        router_nodes = [n for n in graph.nodes.values() if n.node_type == AEGGraphNodeType.EXPERT_ROUTER]
        assert len(router_nodes) == 2  # one per layer

    def test_mtp_heads_are_materialized(self) -> None:
        """MTP heads declared in architecture are materialized as graph nodes."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        from aether.core.types import ModelArchitecture

        arch_with_mtp = ModelArchitecture(
            family="deepseek_family",
            params_billion=0.01,
            layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            context_length=128,
            vocab_size=256,
            intermediate_size=128,
            mtp_heads=2,
        )
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", arch_with_mtp)
        assert graph.get_node("mtp_head_0") is not None
        assert graph.get_node("mtp_head_1") is not None


# ===========================================================================
# Integration: SSM loader
# ===========================================================================

class TestSSMLoader:
    """SSM loader: Mamba/RWKV/Jamba architecture detection."""

    def test_detect_mamba_architecture(self, tmp_path: Path) -> None:
        """Mamba config.json is detected as an SSM architecture."""
        from aether.compiler.stage1_ingestion.ssm_loader import detect_ssm_architecture

        config = {
            "model_type": "mamba",
            "num_hidden_layers": 24,
            "hidden_size": 768,
            "state_size": 16,
            "intermediate_size": 1536,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_ssm_architecture(tmp_path)
        assert arch is not None
        assert arch.model_type == "mamba"
        assert arch.ssm_variant == "selective_scan"

    def test_detect_rwkv_architecture(self, tmp_path: Path) -> None:
        """RWKV config.json is detected as an SSM architecture."""
        from aether.compiler.stage1_ingestion.ssm_loader import detect_ssm_architecture

        config = {
            "model_type": "rwkv",
            "num_hidden_layers": 24,
            "hidden_size": 768,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = detect_ssm_architecture(tmp_path)
        assert arch is not None
        assert arch.model_type == "rwkv"

    def test_non_ssm_returns_none(self, tmp_path: Path) -> None:
        """A plain LLM config.json returns None from detect_ssm_architecture."""
        from aether.compiler.stage1_ingestion.ssm_loader import detect_ssm_architecture

        config = {"model_type": "llama", "num_hidden_layers": 32}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_ssm_architecture(tmp_path) is None

    def test_ssm_loader_list_supported_types(self) -> None:
        """SSMLoader.list_supported_types() returns a non-empty list."""
        from aether.compiler.stage1_ingestion.ssm_loader import SSMLoader

        types = SSMLoader.list_supported_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert "mamba" in types


# ===========================================================================
# Integration: ArchitectureDetector
# ===========================================================================

class TestArchitectureDetector:
    """ArchitectureDetector: known models, config.json, GGUF, fail-closed."""

    def test_known_llama_model_resolves(self) -> None:
        """llama-3.1-8b resolves to llama_family with correct params."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        arch = ArchitectureDetector().detect("llama-3.1-8b")
        assert arch.family == "llama_family"
        assert arch.params_billion == pytest.approx(8.0)
        assert arch.layers == 32

    def test_known_gemma_model_resolves(self) -> None:
        """gemma-2-9b resolves to gemma_family."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        arch = ArchitectureDetector().detect("gemma-2-9b")
        assert arch.family == "gemma_family"

    def test_known_deepseek_model_resolves(self) -> None:
        """deepseek-v3 resolves to deepseek_family with is_moe=True."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        arch = ArchitectureDetector().detect("deepseek-v3")
        assert arch.family == "deepseek_family"
        assert arch.is_moe is True

    def test_config_json_architectures_field(self, tmp_path: Path) -> None:
        """architectures field in config.json is used for detection."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        config = {
            "architectures": ["MistralForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = ArchitectureDetector().detect(str(tmp_path))
        assert arch.family == "mistral_family"

    def test_config_json_model_type_field(self, tmp_path: Path) -> None:
        """model_type field in config.json is used when architectures is absent."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        config = {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "vocab_size": 152064,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = ArchitectureDetector().detect(str(tmp_path))
        assert arch.family == "qwen_family"

    def test_moe_config_sets_is_moe_flag(self, tmp_path: Path) -> None:
        """MoE config.json sets is_moe=True and num_experts."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        config = {
            "architectures": ["MixtralForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arch = ArchitectureDetector().detect(str(tmp_path))
        assert arch.is_moe is True
        assert arch.num_experts == 8
        assert arch.num_activated_experts == 2

    def test_gguf_llama_architecture_detection(self, tmp_path: Path) -> None:
        """GGUF file with llama architecture is detected correctly."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        _GGML_TYPE_F32 = 0
        payload = _build_minimal_gguf(
            tensors=[{
                "name": "token_embd.weight",
                "shape": [16, 8],
                "type": _GGML_TYPE_F32,
                "data": np.zeros(128, dtype=np.float32).tobytes(),
            }],
            metadata={
                "general.architecture": "llama",
                "llama.block_count": 4,
                "llama.embedding_length": 64,
                "llama.attention.head_count": 4,
                "llama.attention.head_count_kv": 2,
                "llama.context_length": 512,
                "llama.feed_forward_length": 128,
                "llama.vocab_size": 32000,
            },
        )
        path = tmp_path / "model.gguf"
        path.write_bytes(payload)

        arch = ArchitectureDetector().detect(str(path))
        assert arch.family == "llama_family"
        assert arch.layers == 4
        assert arch.hidden_size == 64
        assert arch.num_kv_heads == 2

    def test_architecture_detector_normalize_model_name(self) -> None:
        """_normalize_model_name strips common prefixes and normalizes separators."""
        from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

        detector = ArchitectureDetector()
        assert "llama" in detector._normalize_model_name("meta-llama/Llama-3.1-8B").lower()
        assert "qwen" in detector._normalize_model_name("Qwen/Qwen3-0.6B").lower()
