"""
Shared test fixtures and configuration.

Provides test models, minimal AEG packages, and reusable test helpers
for the Aether test suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aether.core.aeg_format import AEGPackage
from aether.core.aeg_ir import AEGIRModule, AEGInstruction, AEGOpCode, AEGOperand, Block, Function
from aether.core.types import ModelArchitecture


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep the default suite deterministic when external model access is absent.

    Network-marked tests remain available with ``AETHER_RUN_NETWORK_TESTS=1``;
    they must not turn a disconnected CI worker into a wall of connection
    failures or conceal the local test results.
    """
    if os.environ.get("AETHER_RUN_NETWORK_TESTS", "").lower() in {"1", "true", "yes"}:
        return
    skip = pytest.mark.skip(reason="network tests disabled; set AETHER_RUN_NETWORK_TESTS=1 to enable")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary Aether cache directory."""
    return tmp_path / ".aether"


@pytest.fixture
def minimal_config_json() -> dict[str, Any]:
    """Return a minimal HuggingFace config.json for testing."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 128,
        "intermediate_size": 256,
    }


@pytest.fixture
def small_architecture() -> ModelArchitecture:
    """Return a small synthetic model architecture for testing."""
    return ModelArchitecture(
        family="llama_family",
        params_billion=0.001,
        layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        context_length=128,
        vocab_size=1000,
        intermediate_size=256,
    )


@pytest.fixture
def minimal_aeg_ir() -> AEGIRModule:
    """Return a minimal AEG-IR module for testing the compiler pipeline."""
    module = AEGIRModule(version="AEG-IR/1.0")
    func = Function(name="model")
    block = Block(name="entry")
    input_op = AEGOperand(name="input", type_str="tensor<*xi64>")
    output_op = AEGOperand(name="logits", type_str="tensor<*xbf16>")
    func.parameters = [input_op]
    func.results = [output_op]
    block.add_instruction(
        AEGInstruction(
            results=[AEGOperand(name="emb", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.EMBEDDING,
            inputs=[input_op.name],
            attributes={"vocab_size": 1000, "hidden_size": 64},
        )
    )
    block.add_instruction(
        AEGInstruction(
            results=[AEGOperand(name="layer0_out", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.TRANSPOSE,
            inputs=["emb"],
        )
    )
    block.add_instruction(
        AEGInstruction(
            results=[AEGOperand(name="logits", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.ADD,
            inputs=["layer0_out", "emb"],
        )
    )
    func.add_block(block)
    module.add_function(func)
    return module


@pytest.fixture
def minimal_aeg_package(tmp_path: Path, small_architecture: ModelArchitecture) -> AEGPackage:
    """Create and return a minimal AEG package on disk for testing."""
    package_path = tmp_path / "test_model.aeg"
    package = AEGPackage.create(package_path, model_id="test-model", aether_version="0.1.0")
    package.manifest.architecture = small_architecture
    package.manifest.memory_requirements = None  # Will be recalculated
    package.set_precision_map({
        "embedding": "BF16",
        "layer_0": "Q4_K_M",
        "layer_1": "Q4_K_M",
        "lm_head": "BF16",
    })
    ir = AEGIRModule(version="AEG-IR/1.0")
    func = Function(name="model")
    block = Block(name="entry")
    block.add_instruction(
        AEGInstruction(
            results=[AEGOperand(name="emb", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.EMBEDDING,
            inputs=["%input"],
            attributes={"vocab_size": 1000, "hidden_size": 64},
        )
    )
    func.add_block(block)
    ir.add_function(func)
    package.ir = ir
    package.save()
    return package


@pytest.fixture
def tiny_local_safetensors_model(tmp_path: Path) -> Path:
    """Write a real, offline, tokenizer-backed tiny Llama checkpoint."""
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    path = tmp_path / "tiny-llama"
    path.mkdir()
    vocab_size, hidden, intermediate = 32, 16, 32
    (path / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "num_hidden_layers": 1, "hidden_size": hidden,
        "intermediate_size": intermediate, "num_attention_heads": 2,
        "num_key_value_heads": 1, "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }), encoding="utf-8")
    rng = np.random.default_rng(7)
    tensors = {
        "model.embed_tokens.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.norm.weight": np.ones(hidden, dtype="float32"),
        "lm_head.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(16, hidden)).astype("float32"),
        "model.layers.0.self_attn.k_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
        "model.layers.0.self_attn.v_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
        "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "model.layers.0.mlp.gate_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
        "model.layers.0.mlp.up_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
        "model.layers.0.mlp.down_proj.weight": rng.normal(size=(hidden, intermediate)).astype("float32"),
    }
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    vocab.update({f"tok{i}": i + 3 for i in range(vocab_size - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>").save_pretrained(str(path))
    return path


@pytest.fixture
def sample_precision_map() -> dict[str, str]:
    """Return a sample precision map."""
    return {
        "embedding": "BF16",
        "layer_0": "BF16",
        "layer_1": "FP8",
        "layer_2": "Q4_K_M",
        "layer_3": "Q4_K_M",
        "layer_4": "Q3_K",
        "layer_5": "Q3_K",
        "layer_6": "Q4_K_M",
        "layer_7": "FP8",
        "layer_8": "BF16",
        "layer_9": "Q4_K_M",
        "layer_10": "Q4_K_M",
        "layer_11": "Q3_K",
        "lm_head": "BF16",
    }
