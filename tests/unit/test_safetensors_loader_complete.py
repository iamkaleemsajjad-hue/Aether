"""
Comprehensive tests for SafeTensors loader.

Tests all aspects of SafeTensors loading including:
- Single file loading
- Multi-shard loading with index.json
- Weight validation
- Integrity checking
- Error handling
- Security (path traversal, etc.)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader
from aether.core.exceptions import IngestionError


class TestSafeTensorsLoaderBasic:
    """Basic SafeTensors loading functionality."""

    def test_single_file_loading(self, tmp_path):
        """Test loading a single .safetensors file."""
        # Create a simple model
        tensors = {
            "embedding.weight": torch.randn(1000, 512),
            "layer.0.attn.q_proj.weight": torch.randn(512, 512),
            "layer.0.attn.k_proj.weight": torch.randn(512, 512),
            "lm_head.weight": torch.randn(1000, 512),
        }

        model_file = tmp_path / "model.safetensors"
        save_file(tensors, model_file)

        loader = SafeTensorsLoader(model_file)
        loaded = loader.load()

        assert len(loaded) == 4
        assert "embedding.weight" in loaded
        assert loaded["embedding.weight"].shape == torch.Size([1000, 512])

    def test_directory_single_shard(self, tmp_path):
        """Test loading from directory with single safetensors file."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        tensors = {"weight": torch.randn(10, 10)}
        save_file(tensors, model_dir / "model.safetensors")

        loader = SafeTensorsLoader(model_dir)
        loaded = loader.load()

        assert len(loaded) == 1
        assert "weight" in loaded

    def test_multi_shard_with_index(self, tmp_path):
        """Test loading multi-shard model with index.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create shards
        shard1 = {"layer.0.weight": torch.randn(100, 100)}
        shard2 = {"layer.1.weight": torch.randn(100, 100)}

        save_file(shard1, model_dir / "model-00001-of-00002.safetensors")
        save_file(shard2, model_dir / "model-00002-of-00002.safetensors")

        # Create index
        index = {
            "metadata": {"total_size": 80000},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = SafeTensorsLoader(model_dir)
        loaded = loader.load()

        assert len(loaded) == 2
        assert "layer.0.weight" in loaded
        assert "layer.1.weight" in loaded

    def test_no_safetensors_files_error(self, tmp_path):
        """Test error when no safetensors files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = SafeTensorsLoader(empty_dir)
        with pytest.raises(IngestionError, match="No SafeTensors files found"):
            loader.discover_files()

    def test_config_loading(self, tmp_path):
        """Test loading config.json alongside model."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        tensors = {"weight": torch.randn(10, 10)}
        save_file(tensors, model_dir / "model.safetensors")

        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loaded_config = loader.load_config()

        assert loaded_config["model_type"] == "llama"
        assert loaded_config["hidden_size"] == 4096


class TestSafeTensorsValidation:
    """Test weight validation functionality."""

    def test_validate_complete_model(self, tmp_path):
        """Test validation of complete model with all expected tensors."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create complete Llama-style model
        tensors = {
            "model.embed_tokens.weight": torch.randn(32000, 512),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(512, 512),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(1024, 512),
            "model.layers.0.mlp.up_proj.weight": torch.randn(1024, 512),
            "model.layers.0.mlp.down_proj.weight": torch.randn(512, 1024),
            "model.layers.0.input_layernorm.weight": torch.randn(512),
            "lm_head.weight": torch.randn(32000, 512),
        }
        save_file(tensors, model_dir / "model.safetensors")

        config = {
            "model_type": "llama",
            "hidden_size": 512,
            "num_hidden_layers": 1,
            "vocab_size": 32000,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loader.load()
        report = loader.validate_weights()

        assert report["valid"] is True
        assert len(report["errors"]) == 0

    def test_validate_missing_critical_tensors(self, tmp_path):
        """Test validation catches missing critical tensors."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Missing lm_head
        tensors = {
            "model.embed_tokens.weight": torch.randn(1000, 512),
            "model.layers.0.weight": torch.randn(512, 512),
        }
        save_file(tensors, model_dir / "model.safetensors")

        config = {"model_type": "llama", "num_hidden_layers": 1}
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loader.load()
        report = loader.validate_weights()

        assert report["valid"] is False
        assert any("lm_head" in err for err in report["errors"])

    def test_validate_nan_detection(self, tmp_path):
        """Test validation detects NaN values."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create tensor with NaN
        weight = torch.randn(10, 10)
        weight[0, 0] = float('nan')
        tensors = {"weight": weight}
        save_file(tensors, model_dir / "model.safetensors")

        loader = SafeTensorsLoader(model_dir)
        loader.load()
        report = loader.validate_weights()

        assert report["valid"] is False
        assert any("NaN" in err for err in report["errors"])

    def test_validate_layer_count_mismatch(self, tmp_path):
        """Test validation warns about layer count mismatches."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Only include tensors for 1 layer but config says 32
        tensors = {
            "model.layers.0.weight": torch.randn(512, 512),
        }
        save_file(tensors, model_dir / "model.safetensors")

        config = {"model_type": "llama", "num_hidden_layers": 32}
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loader.load()
        report = loader.validate_weights()

        assert len(report["warnings"]) > 0


class TestSafeTensorsIntegrity:
    """Test integrity checking functionality."""

    def test_compute_sha256(self, tmp_path):
        """Test SHA-256 hash computation for integrity."""
        model_file = tmp_path / "model.safetensors"
        tensors = {
            "weight1": torch.randn(10, 10),
            "weight2": torch.randn(5, 5),
        }
        save_file(tensors, model_file)

        loader = SafeTensorsLoader(model_file)
        loader.load()
        hash1 = loader.compute_sha256()

        # Hash should be deterministic
        loader2 = SafeTensorsLoader(model_file)
        loader2.load()
        hash2 = loader2.compute_sha256()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex string

    def test_sha256_changes_with_weights(self, tmp_path):
        """Test that hash changes when weights change."""
        model_file1 = tmp_path / "model1.safetensors"
        model_file2 = tmp_path / "model2.safetensors"

        tensors1 = {"weight": torch.randn(10, 10)}
        tensors2 = {"weight": torch.randn(10, 10)}  # Different random values

        save_file(tensors1, model_file1)
        save_file(tensors2, model_file2)

        loader1 = SafeTensorsLoader(model_file1)
        loader1.load()
        hash1 = loader1.compute_sha256()

        loader2 = SafeTensorsLoader(model_file2)
        loader2.load()
        hash2 = loader2.compute_sha256()

        assert hash1 != hash2


class TestSafeTensorsSecurity:
    """Test security features (path traversal, etc.)."""

    def test_path_traversal_prevention_in_index(self, tmp_path):
        """Test that path traversal in index.json is rejected."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Malicious index with path traversal
        index = {
            "weight_map": {
                "weight": "../../../etc/passwd.safetensors"
            }
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = SafeTensorsLoader(model_dir)
        with pytest.raises(IngestionError, match="unsafe shard path"):
            loader.discover_files()

    def test_absolute_path_prevention_in_index(self, tmp_path):
        """Test that absolute paths in index.json are rejected."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Malicious index with absolute path
        index = {
            "weight_map": {
                "weight": "/etc/passwd.safetensors"
            }
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = SafeTensorsLoader(model_dir)
        with pytest.raises(IngestionError, match="unsafe shard path"):
            loader.discover_files()

    def test_missing_shard_file_error(self, tmp_path):
        """Test error when index references non-existent shard."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        index = {
            "weight_map": {
                "weight": "nonexistent.safetensors"
            }
        }
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        loader = SafeTensorsLoader(model_dir)
        with pytest.raises(IngestionError, match="shard file not found"):
            loader.discover_files()


class TestSafeTensorsModelFamilies:
    """Test compatibility with different model families."""

    def test_llama_architecture(self, tmp_path):
        """Test Llama-style model loading."""
        model_dir = tmp_path / "llama_model"
        model_dir.mkdir()

        tensors = {
            "model.embed_tokens.weight": torch.randn(32000, 4096),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.norm.weight": torch.randn(4096),
            "lm_head.weight": torch.randn(32000, 4096),
        }
        save_file(tensors, model_dir / "model.safetensors")

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loaded = loader.load()
        assert len(loaded) == 4

    def test_qwen_architecture(self, tmp_path):
        """Test Qwen-style model loading."""
        model_dir = tmp_path / "qwen_model"
        model_dir.mkdir()

        tensors = {
            "transformer.wte.weight": torch.randn(152064, 1536),
            "transformer.h.0.attn.c_attn.weight": torch.randn(4608, 1536),
            "lm_head.weight": torch.randn(152064, 1536),
        }
        save_file(tensors, model_dir / "model.safetensors")

        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loaded = loader.load()
        assert len(loaded) == 3

    def test_moe_architecture(self, tmp_path):
        """Test MoE model loading with expert tensors."""
        model_dir = tmp_path / "moe_model"
        model_dir.mkdir()

        tensors = {
            "model.embed_tokens.weight": torch.randn(32000, 4096),
            "model.layers.0.block_sparse_moe.gate.weight": torch.randn(8, 4096),
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(14336, 4096),
            "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(14336, 4096),
            "lm_head.weight": torch.randn(32000, 4096),
        }
        save_file(tensors, model_dir / "model.safetensors")

        config = {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "num_local_experts": 8,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        loader = SafeTensorsLoader(model_dir)
        loaded = loader.load()
        report = loader.validate_weights()

        # Should recognize MoE patterns
        assert len(loaded) == 5
        assert report["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
