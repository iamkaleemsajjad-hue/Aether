"""
Shared test fixtures and configuration.

Provides test models, minimal AEG packages, and reusable test helpers
for the Aether test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aether.core.aeg_format import AEGPackage
from aether.core.aeg_ir import AEGIRModule, AEGInstruction, AEGOpCode, AEGOperand, Block, Function
from aether.core.types import ModelArchitecture


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
