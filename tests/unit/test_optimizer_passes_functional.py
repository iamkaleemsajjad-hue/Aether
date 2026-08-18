"""Named functional regression coverage for all 22 optimizer registrations.

The test uses a real AEGGraph and ModelArchitecture.  Opt-in passes remain
reported as ``skipped`` when their required artifact inputs are absent; this
is deliberate fail-closed behavior and is asserted separately from failures.
"""

from __future__ import annotations

import numpy as np

from aether.compiler.config import CompilerConfig
from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
from aether.core.graph import AEGGraph, AEGGraphNode, AEGGraphNodeType
from aether.core.types import ModelArchitecture


EXPECTED_PASS_NAMES = [
    "operator_fusion",
    "sensitivity_analysis",
    "precision_assignment",
    "kv_cache_structuring",
    "moe_routing",
    "parallelism_discovery",
    "reasoning_graph",
    "sparse_attention",
    "pruning_sparsity",
    "mtp_head_compilation",
    "grammar_constraint_compilation",
    "model_merging",
    "ttt_fast_weight_injection",
    "semantic_kv_compression",
    "cross_layer_kv_sharing",
    "green_energy_compilation",
    "tee_kernel_wrapping",
    "mdlm_drafter_compilation",
    "sub2bit_quantization",
    "video_token_compression",
    "advanced_peft_compilation",
    "rlvr_verifier_head_injection",
]


def _real_graph() -> AEGGraph:
    graph = AEGGraph(name="optimizer-functional-regression")
    for layer in range(2):
        graph.add_node(
            AEGGraphNode(
                id=f"layer_{layer}.q_proj",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"layer {layer} q projection",
                op_type="linear",
                layer_index=layer,
                attributes={
                    "weight": np.arange(64, dtype=np.float32).reshape(8, 8),
                },
            )
        )
    return graph


def _architecture() -> ModelArchitecture:
    return ModelArchitecture(
        family="llama_family",
        params_billion=0.001,
        layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_kv_heads=2,
        vocab_size=32,
    )


def test_pipeline_registers_all_22_passes_in_prd_order() -> None:
    pipeline = OptimizerPipeline(CompilerConfig())
    assert pipeline.pass_count == 22
    assert [item.name for item in pipeline._passes] == EXPECTED_PASS_NAMES


def test_real_graph_executes_without_failed_passes() -> None:
    result, reports = OptimizerPipeline(CompilerConfig()).run(
        _real_graph(), _architecture()
    )
    assert len(reports) == 22
    assert not [report for report in reports if report.status == "failed"]
    assert "sensitivity_map" in result.metadata
    assert "reasoning_graph" in result.metadata
    assert "sparsity_plan" in result.metadata
    assert any(report.status == "applied" for report in reports)


def test_opt_in_passes_never_claim_success_without_required_inputs() -> None:
    config = CompilerConfig(
        enable_tee=True,
        enable_mdlm_drafter=True,
        enable_video_compression=True,
        enable_advanced_peft=True,
    )
    _, reports = OptimizerPipeline(config).run(_real_graph(), _architecture())
    by_name = {report.pass_name: report for report in reports}
    assert by_name["tee_kernel_wrapping"].status == "skipped"
    assert by_name["mdlm_drafter_compilation"].status == "skipped"
    assert by_name["video_token_compression"].status == "skipped"
    assert by_name["advanced_peft_compilation"].status == "skipped"

