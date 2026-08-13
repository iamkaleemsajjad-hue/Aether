"""
Tests for the new specialised ingestion loaders:
  - VideoModelLoader (video_loader.py)
  - MLALoader (mla_loader.py)
  - MoELoader (moe_loader.py)

All tests run without real model weights, exercising the full graph-building
logic through config-dict injection.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest


# ============================================================================
# Helpers
# ============================================================================

def _make_model_dir(tmp_path: Path, config: dict[str, Any]) -> Path:
    """Write config.json into tmp_path and return the dir."""
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


# ============================================================================
# VideoModelLoader tests
# ============================================================================

class TestVideoModelLoader:
    """Tests for aether.compiler.stage1_ingestion.video_loader.VideoModelLoader."""

    def _import(self):
        from aether.compiler.stage1_ingestion.video_loader import (
            VideoModelLoader,
            VideoArchitecture,
            load_video_model,
            detect_video_architecture,
        )
        return VideoModelLoader, VideoArchitecture, load_video_model, detect_video_architecture

    def test_import(self):
        loader_cls, arch_cls, load_fn, detect_fn = self._import()
        assert loader_cls is not None
        assert arch_cls is not None

    def test_video_architecture_dataclass(self):
        _, VideoArchitecture, _, _ = self._import()
        arch = VideoArchitecture(
            model_type="video_llama2",
            video_encoder="clip_vit_l14",
            temporal_aggregator="temporal_attn",
            language_backbone="llama2_13b",
            projection_type="mlp",
            frame_resolution=224,
            patch_size=14,
            max_frames=16,
            tokens_per_frame=256,
            temporal_compression=4,
        )
        assert arch.total_visual_tokens == 16 * 256 // 4
        assert arch.max_frames == 16

    def test_detect_video_architecture_video_llama(self):
        _, _, _, detect_fn = self._import()
        config = {
            "model_type": "video_llama",
            "max_frames": 8,
        }
        arch = detect_fn(config)
        assert arch is not None
        assert arch.model_type == "video_llama"
        assert arch.max_frames == 8

    def test_detect_video_architecture_llava_video(self):
        _, _, _, detect_fn = self._import()
        config = {"model_type": "llava_video"}
        arch = detect_fn(config)
        assert arch is not None
        assert "llava" in arch.model_type.lower() or "video" in arch.model_type.lower()

    def test_detect_video_architecture_unknown_returns_none(self):
        _, _, _, detect_fn = self._import()
        config = {"model_type": "llama"}  # Not a video model
        arch = detect_fn(config)
        assert arch is None

    def test_detect_video_architecture_from_video_keys(self):
        _, _, _, detect_fn = self._import()
        config = {
            "model_type": "custom_model",
            "max_frames": 16,          # video-specific key → triggers detection
            "num_query_tokens": 32,
        }
        arch = detect_fn(config)
        assert arch is not None

    def test_load_missing_path_raises(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        missing = tmp_path / "nonexistent_model"
        loader = VideoModelLoader(missing)
        with pytest.raises(Exception):  # IngestionError or FileNotFoundError
            loader.load()

    def test_load_video_llama_config(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {
            "model_type": "video_llama",
            "max_frames": 8,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "vision_config": {"image_size": 224, "patch_size": 14},
        }
        model_dir = _make_model_dir(tmp_path, config)
        loader = VideoModelLoader(model_dir)
        result = loader.load()
        assert result["format"] == "video_model"
        assert "graph" in result
        assert "architecture" in result
        assert "metadata" in result

    def test_load_video_llama2_config(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {
            "model_type": "video_llama2",
            "max_frames": 16,
            "num_hidden_layers": 8,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        arch = result["architecture"]
        assert arch.max_frames == 16

    def test_load_videochat2_config(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "videochat2", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        assert result["format"] == "video_model"

    def test_load_llava_video_config(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "llava_video", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        arch = result["architecture"]
        assert arch.model_type == "llava_video"

    def test_graph_has_video_encoder_node(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "video_llama", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        graph = result["graph"]
        nodes = (
            list(graph.nodes.values())
            if hasattr(graph, "nodes")
            else list(getattr(graph, "_nodes", {}).values())
        )
        op_types = [getattr(n, "op_type", "") for n in nodes]
        assert any("video" in op or "encoder" in op for op in op_types)

    def test_graph_has_lm_head_node(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "videochat2", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        ids = [getattr(n, "id", "") for n in nodes]
        assert "lm_head" in ids

    def test_metadata_contains_video_info(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "video_llama2", "max_frames": 16, "num_hidden_layers": 4}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        meta = result["metadata"]
        assert meta.architecture.max_frames == 16

    def test_has_audio_video_llama(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "video_llama", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        assert result["architecture"].has_audio is True

    def test_temporal_compression_applied(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "video_llama2", "max_frames": 16, "num_hidden_layers": 4}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        arch = result["architecture"]
        assert arch.total_visual_tokens == arch.max_frames * arch.tokens_per_frame // arch.temporal_compression

    def test_config_override_max_frames(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "video_llama", "max_frames": 32, "num_hidden_layers": 4}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        assert result["architecture"].max_frames == 32

    def test_unknown_video_model_type(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {
            "model_type": "custom_video_model",
            "max_frames": 8,
            "num_hidden_layers": 4,
            "vision_config": {"image_size": 224, "patch_size": 14},
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        assert result["format"] == "video_model"  # Should succeed via heuristic detection

    def test_convenience_function_load_video_model(self, tmp_path):
        _, _, load_video_model, _ = self._import()
        config = {"model_type": "videochat2", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = load_video_model(model_dir)
        assert "graph" in result and "architecture" in result

    def test_type_alias_llava_next_video(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "llava_next_video", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        assert result["format"] == "video_model"

    def test_internvideo2(self, tmp_path):
        VideoModelLoader, *_ = self._import()
        config = {"model_type": "internvideo2", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = VideoModelLoader(model_dir).load()
        arch = result["architecture"]
        assert arch.model_type == "internvideo2"


# ============================================================================
# MLALoader tests
# ============================================================================

class TestMLALoader:
    """Tests for aether.compiler.stage1_ingestion.mla_loader.MLALoader."""

    def _import(self):
        from aether.compiler.stage1_ingestion.mla_loader import (
            MLALoader,
            MLAArchitecture,
            load_mla_model,
            is_mla_model,
            _parse_mla_config,
        )
        return MLALoader, MLAArchitecture, load_mla_model, is_mla_model, _parse_mla_config

    def test_import(self):
        loader_cls, *_ = self._import()
        assert loader_cls is not None

    def test_mla_architecture_kv_compression_ratio(self):
        _, MLAArchitecture, *_ = self._import()
        arch = MLAArchitecture(
            model_type="deepseek_v2",
            family="deepseek",
            hidden_size=5120,
            num_layers=60,
            num_heads=128,
            vocab_size=102400,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
        )
        # Compression ratio should be >> 1 (MLA compresses KV)
        assert arch.kv_compression_ratio > 1.0
        # For DeepSeek-V2 it's typically ~5x
        assert arch.kv_compression_ratio > 3.0

    def test_mla_architecture_q_head_dim(self):
        _, MLAArchitecture, *_ = self._import()
        arch = MLAArchitecture(
            model_type="deepseek_v2", family="deepseek",
            hidden_size=5120, num_layers=60, num_heads=128, vocab_size=102400,
            kv_lora_rank=512, q_lora_rank=1536,
            qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        )
        assert arch.q_head_dim == 64 + 128  # rope + nope

    def test_is_mla_model_by_kv_lora_rank(self):
        *_, is_mla_model, _ = self._import()
        assert is_mla_model({"kv_lora_rank": 512}) is True
        assert is_mla_model({"model_type": "llama"}) is False
        assert is_mla_model({"model_type": "deepseek_v2"}) is True

    def test_parse_deepseek_v2_config(self):
        *_, _, parse_fn = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_routed_experts": 160,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
        }
        arch = parse_fn(config)
        assert arch.kv_lora_rank == 512
        assert arch.num_experts == 160
        assert arch.is_moe is True

    def test_parse_deepseek_v3_defaults(self):
        *_, _, parse_fn = self._import()
        config = {"model_type": "deepseek_v3"}
        arch = parse_fn(config)
        assert arch.hidden_size == 7168
        assert arch.num_experts == 256

    def test_load_missing_path_raises(self, tmp_path):
        MLALoader, *_ = self._import()
        with pytest.raises(Exception):
            MLALoader(tmp_path / "nonexistent").load()

    def test_load_deepseek_v2_config(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 128,
            "vocab_size": 102400,
            "n_routed_experts": 16,
            "num_experts_per_tok": 6,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        assert result["format"] == "mla_model"
        assert "graph" in result
        assert "architecture" in result
        assert "kv_compression_ratio" in result

    def test_kv_compression_ratio_in_result(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 4,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        assert result["kv_compression_ratio"] > 1.0

    def test_graph_has_mla_attention_nodes(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 6,
            "first_k_dense_replace": 2,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        op_types = {getattr(n, "op_type", "") for n in nodes}
        assert "aeg.mla_attention" in op_types
        assert "aeg.attention" in op_types  # Dense layers use standard MHA

    def test_hybrid_dense_mla_layers(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 8,
            "first_k_dense_replace": 3,  # First 3 layers = dense MHA
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        arch = result["architecture"]
        assert arch.first_k_dense_replace == 3

    def test_graph_has_lm_head(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {"model_type": "deepseek_v2", "kv_lora_rank": 512,
                  "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "num_hidden_layers": 4}
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        ids = {getattr(n, "id", "") for n in nodes}
        assert "lm_head" in ids

    def test_moe_layer_nodes_present(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 4,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        op_types = [getattr(n, "op_type", "") for n in nodes]
        assert "aeg.moe_layer" in op_types

    def test_deepseek_r1_alias(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_r1",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 4,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        assert result["format"] == "mla_model"

    def test_metadata_mla_config_set(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 4,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        graph = result["graph"]
        if hasattr(graph, "get_metadata") or hasattr(graph, "_metadata"):
            meta = getattr(graph, "_metadata", {})
            assert "mla_config" in meta

    def test_convenience_function(self, tmp_path):
        _, _, load_mla_model, *_ = self._import()
        config = {"model_type": "deepseek_v2", "kv_lora_rank": 512,
                  "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "num_hidden_layers": 4}
        model_dir = _make_model_dir(tmp_path, config)
        result = load_mla_model(model_dir)
        assert result["format"] == "mla_model"

    def test_rope_theta_override(self, tmp_path):
        MLALoader, *_ = self._import()
        config = {
            "model_type": "deepseek_r1",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 4,
            "rope_theta": 500000.0,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MLALoader(model_dir).load()
        arch = result["architecture"]
        assert arch.rope_theta == 500000.0


# ============================================================================
# MoELoader tests
# ============================================================================

class TestMoELoader:
    """Tests for aether.compiler.stage1_ingestion.moe_loader.MoELoader."""

    def _import(self):
        from aether.compiler.stage1_ingestion.moe_loader import (
            MoELoader,
            MoEArchitecture,
            load_moe_model,
            is_moe_model,
            _parse_moe_config,
            _classify_experts,
        )
        return MoELoader, MoEArchitecture, load_moe_model, is_moe_model, _parse_moe_config, _classify_experts

    def test_import(self):
        loader_cls, *_ = self._import()
        assert loader_cls is not None

    def test_moe_architecture_sparsity(self):
        _, MoEArchitecture, *_ = self._import()
        arch = MoEArchitecture(
            model_type="mixtral", family="mistral",
            hidden_size=4096, num_layers=32, num_heads=32,
            vocab_size=32000, intermediate_size=14336,
            num_experts=8, num_experts_per_token=2,
            expert_intermediate_size=14336, num_shared_experts=0,
            num_key_value_heads=8, router_aux_loss_coef=0.02,
        )
        assert arch.sparsity == pytest.approx(2 / 8)
        assert arch.sparsity == pytest.approx(0.25)

    def test_classify_experts_counts(self):
        *_, classify_fn = self._import()
        tiers = classify_fn(8, hot_threshold=0.20, warm_threshold=0.50)
        assert set(tiers.keys()) == {"hot", "warm", "cold"}
        total = len(tiers["hot"]) + len(tiers["warm"]) + len(tiers["cold"])
        assert total == 8

    def test_classify_experts_no_overlap(self):
        *_, classify_fn = self._import()
        tiers = classify_fn(64)
        hot_set = set(tiers["hot"])
        warm_set = set(tiers["warm"])
        cold_set = set(tiers["cold"])
        assert not (hot_set & warm_set)
        assert not (hot_set & cold_set)
        assert not (warm_set & cold_set)

    def test_classify_experts_all_covered(self):
        *_, classify_fn = self._import()
        for n in [8, 16, 64, 128, 256]:
            tiers = classify_fn(n)
            total = len(tiers["hot"]) + len(tiers["warm"]) + len(tiers["cold"])
            assert total == n, f"Expert count mismatch for n={n}: got {total}"

    def test_classify_experts_hot_fraction(self):
        *_, classify_fn = self._import()
        tiers = classify_fn(100, hot_threshold=0.20)
        assert len(tiers["hot"]) == 20

    def test_is_moe_model_by_num_local_experts(self):
        *_, is_moe, _, _ = self._import()
        assert is_moe({"num_local_experts": 8}) is True
        assert is_moe({"n_routed_experts": 64}) is True
        assert is_moe({"model_type": "llama"}) is False

    def test_parse_mixtral_config(self):
        *_, _, parse_fn, _ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        }
        arch = parse_fn(config)
        assert arch.num_experts == 8
        assert arch.num_experts_per_token == 2

    def test_parse_qwen_moe_config(self):
        *_, _, parse_fn, _ = self._import()
        config = {
            "model_type": "qwen_moe",
            "num_local_experts": 60,
            "num_experts_per_tok": 4,
            "n_shared_experts": 4,
        }
        arch = parse_fn(config)
        assert arch.num_shared_experts == 4

    def test_parse_jamba_alternating_layers(self):
        *_, _, parse_fn, _ = self._import()
        config = {
            "model_type": "jamba",
            "moe_layer_frequency": 2,  # every 2nd layer is MoE
            "num_hidden_layers": 32,
            "num_local_experts": 16,
        }
        arch = parse_fn(config)
        assert len(arch.moe_layers) == 16   # 32/2
        assert len(arch.dense_layers) == 16

    def test_load_missing_path_raises(self, tmp_path):
        MoELoader, *_ = self._import()
        with pytest.raises(Exception):
            MoELoader(tmp_path / "nonexistent").load()

    def test_load_mixtral_config(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "hidden_size": 4096,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        assert result["format"] == "moe_model"
        assert "graph" in result
        assert "architecture" in result
        assert "expert_tiers" in result

    def test_expert_tiers_in_result(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        tiers = result["expert_tiers"]
        assert {"hot", "warm", "cold"} <= set(tiers.keys())
        assert sum(len(v) for v in tiers.values()) == 8

    def test_graph_has_moe_router_node(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        op_types = {getattr(n, "op_type", "") for n in nodes}
        assert "aeg.moe_router" in op_types

    def test_graph_has_expert_ffn_nodes(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        op_types = [getattr(n, "op_type", "") for n in nodes]
        # Each of the 4 MoE layers should have 8 expert nodes
        expert_count = op_types.count("aeg.expert_ffn")
        assert expert_count == 4 * 8

    def test_shared_experts_present_qwen_moe(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "qwen_moe",
            "num_local_experts": 60,
            "num_experts_per_tok": 4,
            "n_shared_experts": 4,
            "num_hidden_layers": 4,
            "vocab_size": 151936,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        op_types = [getattr(n, "op_type", "") for n in nodes]
        shared_count = op_types.count("aeg.shared_expert_ffn")
        assert shared_count == 4 * 4  # 4 layers × 4 shared experts

    def test_dense_layers_have_standard_ffn(self, tmp_path):
        MoELoader, *_ = self._import()
        # Jamba: alternating MoE / dense layers
        config = {
            "model_type": "jamba",
            "num_local_experts": 16,
            "num_experts_per_tok": 2,
            "moe_layer_frequency": 2,  # every 2nd is MoE
            "num_hidden_layers": 4,    # layers 0,2 = dense; 1,3 = MoE
            "vocab_size": 65536,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        op_types = [getattr(n, "op_type", "") for n in nodes]
        assert "aeg.swiglu_ffn" in op_types   # Dense layers have standard FFN
        assert "aeg.moe_router" in op_types    # MoE layers have router

    def test_graph_has_embedding_and_lm_head(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        ids = {getattr(n, "id", "") for n in nodes}
        assert "embedding" in ids
        assert "lm_head" in ids

    def test_router_tier_annotations_in_attributes(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        router_nodes = [n for n in nodes if getattr(n, "op_type", "") == "aeg.moe_router"]
        assert len(router_nodes) >= 1
        router = router_nodes[0]
        attrs = getattr(router, "attributes", {}) or {}
        assert "hot_experts" in attrs
        assert "warm_experts" in attrs
        assert "cold_experts" in attrs

    def test_expert_tier_annotation_in_ffn_nodes(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        nodes = list(getattr(graph, "nodes", {}).values()) or list(getattr(graph, "_nodes", {}).values())
        expert_nodes = [n for n in nodes if getattr(n, "op_type", "") == "aeg.expert_ffn"]
        assert len(expert_nodes) > 0
        # Every expert should have a tier annotation
        tiers_present = {(getattr(n, "attributes", {}) or {}).get("tier") for n in expert_nodes}
        assert tiers_present <= {"hot", "warm", "cold"}
        assert tiers_present  # At least one tier present

    def test_graph_moe_metadata(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        graph = result["graph"]
        meta = getattr(graph, "_metadata", {})
        assert "moe_config" in meta

    def test_convenience_function(self, tmp_path):
        _, _, load_moe_model, *_ = self._import()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = load_moe_model(model_dir)
        assert result["format"] == "moe_model"

    def test_dbrx_config(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "dbrx",
            "num_hidden_layers": 4,
            "vocab_size": 100352,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        assert result["format"] == "moe_model"
        assert result["architecture"].model_type == "dbrx"

    def test_olmoe_config(self, tmp_path):
        MoELoader, *_ = self._import()
        config = {
            "model_type": "olmoe",
            "num_local_experts": 64,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 4,
            "vocab_size": 50304,
        }
        model_dir = _make_model_dir(tmp_path, config)
        result = MoELoader(model_dir).load()
        assert result["architecture"].num_experts == 64


# ============================================================================
# Ingestion pipeline specialised dispatch tests
# ============================================================================

class TestIngestionSpecialisedDispatch:
    """Tests that IngestionPipeline._try_specialised_loader routes correctly."""

    def _make_arch(self):
        from aether.core.types import ModelArchitecture
        return ModelArchitecture(
            family="test",
            params_billion=7.0,
            layers=4,
            hidden_size=4096,
            num_attention_heads=32,
            context_length=4096,
        )

    def test_import_pipeline(self):
        from aether.compiler.stage1_ingestion import IngestionPipeline
        assert IngestionPipeline is not None

    def test_try_specialised_loader_mla_deepseek(self, tmp_path):
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        pipeline = IngestionPipeline()
        arch = self._make_arch()
        config = {
            "model_type": "deepseek_v2",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "num_hidden_layers": 4,
        }
        model_dir = _make_model_dir(tmp_path, config)
        graph = pipeline._try_specialised_loader(str(model_dir), arch, "safetensors")
        assert graph is not None
        # Should have come via MLALoader
        meta = getattr(graph, "_metadata", {}) or {}
        assert meta.get("specialised_loader_format") == "mla_model"

    def test_try_specialised_loader_moe_mixtral(self, tmp_path):
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        pipeline = IngestionPipeline()
        arch = self._make_arch()
        config = {
            "model_type": "mixtral",
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
        }
        model_dir = _make_model_dir(tmp_path, config)
        graph = pipeline._try_specialised_loader(str(model_dir), arch, "safetensors")
        assert graph is not None
        meta = getattr(graph, "_metadata", {}) or {}
        assert meta.get("specialised_loader_format") == "moe_model"

    def test_try_specialised_loader_video(self, tmp_path):
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        pipeline = IngestionPipeline()
        arch = self._make_arch()
        config = {"model_type": "video_llama", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        graph = pipeline._try_specialised_loader(str(model_dir), arch, "safetensors")
        assert graph is not None
        meta = getattr(graph, "_metadata", {}) or {}
        assert meta.get("specialised_loader_format") == "video_model"

    def test_try_specialised_loader_returns_none_for_llama(self, tmp_path):
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        pipeline = IngestionPipeline()
        arch = self._make_arch()
        config = {"model_type": "llama", "num_hidden_layers": 4, "vocab_size": 32000}
        model_dir = _make_model_dir(tmp_path, config)
        result = pipeline._try_specialised_loader(str(model_dir), arch, "safetensors")
        assert result is None  # No specialised loader for plain LLaMA

    def test_ingestion_package_exports(self):
        from aether.compiler.stage1_ingestion import (
            VideoModelLoader, MLALoader, MoELoader,
            VideoArchitecture, MLAArchitecture, MoEArchitecture,
            load_video_model, load_mla_model, load_moe_model,
            is_mla_model, is_moe_model,
        )
        assert VideoModelLoader is not None
        assert MLALoader is not None
        assert MoELoader is not None
