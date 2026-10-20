"""An explicit FP32 request must retain source parameter bits, including globals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from aether.compiler.config import CompilerConfig
from aether.compiler.stage2_optimizer.optimizer import PrecisionAssignmentPass
from aether.compiler.weight_quantizer import quantize_graph_weights
from aether.core.aeg_format import AEGPackage
from aether.core.graph import AEGGraph, AEGGraphNode, AEGGraphNodeType
from aether.core.types import ModelArchitecture
from aether.quantization.formats import dequantize_tensor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("mode", ["uniform", "manual"])
def test_explicit_fp32_globals_survive_precision_assignment(mode: str) -> None:
    """Neither a uniform FP32 request nor named overrides may become BF16."""
    config = CompilerConfig(
        precision="FP32" if mode == "uniform" else None,
        precision_assignment_mode=mode,
        manual_precision_map={"embedding": "FP32", "lm_head": "FP32"},
    )
    architecture = ModelArchitecture(
        family="llama_family", params_billion=0.001, layers=1, hidden_size=8, num_attention_heads=2
    )
    _, report = PrecisionAssignmentPass().run(AEGGraph(name="source"), architecture, config)
    assert report.status == "applied"
    assert report.details["precision_map"]["embedding"] == "FP32"
    assert report.details["precision_map"]["lm_head"] == "FP32"


def test_fp32_quantizer_preserves_globals_and_all_bias_values(tmp_path: Path) -> None:
    """Compare actual codec outputs to source bits, without a quality success flag."""
    rng = np.random.default_rng(192)
    graph = AEGGraph(name="source")
    originals = {}
    for name, shape in (("embedding", (16, 8)), ("lm_head", (16, 8)), ("final_norm", (8,))):
        values = rng.normal(size=shape).astype(np.float32)
        graph.add_node(
            AEGGraphNode(
                id=name,
                name=name,
                node_type=AEGGraphNodeType.OPERATION,
                op_type=name,
                attributes={"weight": values},
            )
        )
        originals[name] = values
    values = rng.normal(size=(24, 8)).astype(np.float32)
    attributes: dict[str, Any] = {
        "weight": values,
        "num_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 4,
    }
    for index, projection in enumerate(("q", "k", "v")):
        bias = rng.normal(size=8).astype(np.float32)
        attributes[projection + "_bias"] = bias
        originals[f"layer_0_{projection}_proj"] = values[index * 8 : (index + 1) * 8]
        originals[f"layer_0_{projection}_proj_bias"] = bias
    graph.add_node(
        AEGGraphNode(
            id="layer_0_qkv",
            name="QKV",
            node_type=AEGGraphNodeType.OPERATION,
            op_type="qkv_proj",
            layer_index=0,
            attributes=attributes,
        )
    )
    weight, bias = (
        rng.normal(size=(12, 8)).astype(np.float32),
        rng.normal(size=12).astype(np.float32),
    )
    graph.add_node(
        AEGGraphNode(
            id="layer_0_gate_proj",
            name="gate",
            node_type=AEGGraphNodeType.OPERATION,
            op_type="linear",
            layer_index=0,
            attributes={"weight": weight, "bias": bias},
        )
    )
    originals.update(layer_0_gate_proj=weight, layer_0_gate_proj_bias=bias)
    package = AEGPackage.create(tmp_path / "parameters.aeg", model_id="source", aether_version="1")
    quantize_graph_weights(graph, package, default_precision="FP32")
    assert package.weights.keys() == originals.keys()
    for name, original in originals.items():
        encoded = package.weights[name]
        assert encoded.precision == "FP32", name
        np.testing.assert_array_equal(dequantize_tensor(encoded), original)
