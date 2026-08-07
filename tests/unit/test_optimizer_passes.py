"""Tests for the nine Stage 2 optimizer passes and their public entry points.

The ``passN_*`` modules re-export the pass implementations from ``optimizer``.
Five of them previously declared ``class XPass(XPass)`` — a class shadowing its
own import — and none were covered by any test. These tests pin the re-export
contract and exercise each pass against a real graph.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.compiler.config import CompilerConfig
from aether.compiler.stage2_optimizer import optimizer as optimizer_module
from aether.compiler.stage2_optimizer.optimizer import (
    BasePass,
    KVCacheStructuringPass,
    MoERoutingPass,
    OperatorFusionPass,
    OptimizerPipeline,
    ParallelismDiscoveryPass,
    PrecisionAssignmentPass,
    PruningSparsityPass,
    ReasoningGraphPass,
    SensitivityAnalysisPass,
    SparseAttentionPass,
)
from aether.compiler.stage2_optimizer.pass1_operator_fusion import (
    OperatorFusionPass as Pass1Export,
)
from aether.compiler.stage2_optimizer.pass2_sensitivity_analysis import (
    SensitivityAnalysisPass as Pass2Export,
)
from aether.compiler.stage2_optimizer.pass3_precision_assignment import (
    PrecisionAssignmentPass as Pass3Export,
)
from aether.compiler.stage2_optimizer.pass4_kv_cache_structuring import (
    KVCacheStructuringPass as Pass4Export,
)
from aether.compiler.stage2_optimizer.pass5_moe_routing import MoERoutingPass as Pass5Export
from aether.compiler.stage2_optimizer.pass6_parallelism_discovery import (
    ParallelismDiscoveryPass as Pass6Export,
)
from aether.compiler.stage2_optimizer.pass7_reasoning_graph import (
    ReasoningGraphPass as Pass7Export,
)
from aether.compiler.stage2_optimizer.pass8_sparse_attention import (
    SparseAttentionPass as Pass8Export,
)
from aether.compiler.stage2_optimizer.pass9_pruning_sparsity import (
    PruningSparsityPass as Pass9Export,
)
from aether.core.graph import AEGGraph, AEGGraphNode, AEGGraphNodeType
from aether.core.types import ModelArchitecture

#: (entry-point export, implementation class) for all nine passes.
PASS_EXPORTS = [
    (Pass1Export, OperatorFusionPass),
    (Pass2Export, SensitivityAnalysisPass),
    (Pass3Export, PrecisionAssignmentPass),
    (Pass4Export, KVCacheStructuringPass),
    (Pass5Export, MoERoutingPass),
    (Pass6Export, ParallelismDiscoveryPass),
    (Pass7Export, ReasoningGraphPass),
    (Pass8Export, SparseAttentionPass),
    (Pass9Export, PruningSparsityPass),
]

ALL_PASS_CLASSES = [impl for _, impl in PASS_EXPORTS]


@pytest.fixture
def architecture() -> ModelArchitecture:
    return ModelArchitecture(
        family="llama_family",
        params_billion=1.0,
        layers=4,
        hidden_size=128,
        num_attention_heads=8,
        num_kv_heads=4,
        context_length=8192,
    )


@pytest.fixture
def moe_architecture() -> ModelArchitecture:
    return ModelArchitecture(
        family="mixtral_family",
        params_billion=8.0,
        layers=4,
        hidden_size=128,
        num_attention_heads=8,
        num_experts=32,
        context_length=65536,
    )


@pytest.fixture
def graph() -> AEGGraph:
    """A small graph with real weight matrices attached to linear nodes."""
    g = AEGGraph(name="test_model")
    rs = np.random.RandomState(0)
    for layer in range(4):
        node = AEGGraphNode(
            id=f"layer_{layer}.q_proj",
            node_type=AEGGraphNodeType.OPERATION,
            name=f"Layer {layer} QKV",
            op_type="linear",
            layer_index=layer,
        )
        node.add_attribute("weight", rs.randn(32, 64).astype(np.float32))
        node.add_attribute("activation_norms", np.abs(rs.randn(64)).astype(np.float32) + 0.1)
        g.add_node(node)
    return g


class TestPassEntryPoints:
    """The passN_* modules must re-export the implementation, not shadow it."""

    @pytest.mark.parametrize(("exported", "implementation"), PASS_EXPORTS)
    def test_export_is_the_implementation(self, exported: type, implementation: type) -> None:
        assert exported is implementation

    @pytest.mark.parametrize(("exported", "_impl"), PASS_EXPORTS)
    def test_export_is_instantiable_and_runnable(self, exported: type, _impl: type) -> None:
        instance = exported()
        assert isinstance(instance, BasePass)
        assert callable(instance.run)

    @pytest.mark.parametrize(("exported", "_impl"), PASS_EXPORTS)
    def test_export_is_not_a_self_referential_subclass(
        self, exported: type, _impl: type
    ) -> None:
        """Regression: five shims declared ``class XPass(XPass)``."""
        assert exported.__bases__ != (exported,)
        assert exported.__name__ not in {base.__name__ for base in exported.__bases__[1:]}

    @pytest.mark.parametrize("pass_class", ALL_PASS_CLASSES)
    def test_every_pass_declares_name_and_description(self, pass_class: type) -> None:
        instance = pass_class()
        assert instance.name and instance.name != "base"
        assert instance.description
        assert instance.description != BasePass.description

    def test_pass_names_are_unique(self) -> None:
        names = [cls().name for cls in ALL_PASS_CLASSES]
        assert len(names) == len(set(names))


class TestBasePass:
    def test_run_is_abstract(self, architecture: ModelArchitecture) -> None:
        with pytest.raises(NotImplementedError):
            BasePass().run(AEGGraph(name="x"), architecture, CompilerConfig())


class TestIndividualPasses:
    @pytest.mark.parametrize("pass_class", ALL_PASS_CLASSES)
    def test_pass_returns_graph_and_report(
        self, pass_class: type, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        result_graph, report = pass_class().run(graph, architecture, CompilerConfig())
        assert result_graph is not None
        assert report.pass_name == pass_class().name
        assert report.status in ("applied", "skipped", "failed")

    @pytest.mark.parametrize("pass_class", ALL_PASS_CLASSES)
    def test_pass_does_not_fail_on_valid_input(
        self, pass_class: type, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = pass_class().run(graph, architecture, CompilerConfig())
        assert report.status != "failed", report.details

    @pytest.mark.parametrize("pass_class", ALL_PASS_CLASSES)
    def test_pass_records_duration(
        self, pass_class: type, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = pass_class().run(graph, architecture, CompilerConfig())
        assert report.duration_ms >= 0.0

    @pytest.mark.parametrize("pass_class", ALL_PASS_CLASSES)
    def test_pass_tolerates_empty_graph(
        self, pass_class: type, architecture: ModelArchitecture
    ) -> None:
        _, report = pass_class().run(AEGGraph(name="empty"), architecture, CompilerConfig())
        assert report.status != "failed", report.details


class TestSensitivityAndPrecision:
    def test_sensitivity_annotates_nodes(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = SensitivityAnalysisPass().run(graph, architecture, CompilerConfig())
        assert report.status == "applied"
        assert report.details["sensitivity_map"]
        assert graph.get_node("layer_0.q_proj").get_attribute("sensitivity") is not None

    def test_precision_keeps_embedding_and_head_at_bf16(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        config = CompilerConfig()
        SensitivityAnalysisPass().run(graph, architecture, config)
        _, report = PrecisionAssignmentPass().run(graph, architecture, config)
        precision_map = report.details["precision_map"]
        assert precision_map["embedding"] == "BF16"
        assert precision_map["lm_head"] == "BF16"

    def test_uniform_mode_assigns_one_precision(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        config = CompilerConfig(precision_assignment_mode="uniform")
        _, report = PrecisionAssignmentPass().run(graph, architecture, config)
        layer_precisions = {
            v for k, v in report.details["precision_map"].items() if k.startswith("layer_")
        }
        assert layer_precisions == {"Q4_K_M"}

    def test_manual_mode_honours_supplied_map(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        config = CompilerConfig(
            precision_assignment_mode="manual", manual_precision_map={"layer_0": "FP8"}
        )
        _, report = PrecisionAssignmentPass().run(graph, architecture, config)
        assert report.details["precision_map"]["layer_0"] == "FP8"


class TestStructuralPasses:
    def test_kv_cache_adds_one_node_per_layer(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = KVCacheStructuringPass().run(graph, architecture, CompilerConfig())
        assert report.details["kv_cache_nodes_added"] == architecture.layers

    def test_moe_routing_skips_dense_models(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = MoERoutingPass().run(graph, architecture, CompilerConfig())
        assert report.status == "skipped"
        assert report.details["total_experts"] == 0

    def test_moe_routing_tiers_experts(
        self, graph: AEGGraph, moe_architecture: ModelArchitecture
    ) -> None:
        _, report = MoERoutingPass().run(graph, moe_architecture, CompilerConfig())
        assert report.status == "applied"
        details = report.details
        assert details["total_experts"] == 32
        assert details["hot_experts"] + details["warm_experts"] + details["cold_experts"] == 32

    def test_parallelism_generates_plans(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = ParallelismDiscoveryPass().run(graph, architecture, CompilerConfig())
        assert report.details["plans_generated"] > 0


class TestReasoningAndSparseAttention:
    def test_reasoning_graph_respects_budget(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        config = CompilerConfig(reasoning_budget_tokens=256)
        _, report = ReasoningGraphPass().run(graph, architecture, config)
        assert report.details["budget_tokens"] == 256
        assert report.details["nodes"]

    def test_sparse_attention_skips_short_context(self, graph: AEGGraph) -> None:
        short = ModelArchitecture(
            family="llama_family",
            params_billion=1.0,
            layers=2,
            hidden_size=128,
            num_attention_heads=8,
            context_length=4096,
        )
        _, report = SparseAttentionPass().run(graph, short, CompilerConfig())
        assert report.status == "skipped"

    def test_sparse_attention_activates_on_long_context(
        self, graph: AEGGraph, moe_architecture: ModelArchitecture
    ) -> None:
        config = CompilerConfig(sparse_attention_context_threshold=32768)
        _, report = SparseAttentionPass().run(graph, moe_architecture, config)
        assert report.status == "applied"
        assert len(report.details["patterns"]) == moe_architecture.num_attention_heads

    def test_every_head_gets_a_known_pattern(
        self, graph: AEGGraph, moe_architecture: ModelArchitecture
    ) -> None:
        _, report = SparseAttentionPass().run(graph, moe_architecture, CompilerConfig())
        patterns = {entry["pattern"] for entry in report.details["patterns"]}
        assert patterns <= {"vertical_slash", "block_sparse", "a_shape"}


class TestPruningPass:
    def test_computes_real_masks_when_weights_present(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, report = PruningSparsityPass().run(graph, architecture, CompilerConfig())
        assert report.details["masks_computed"] == 4
        assert report.details["masks_planned_only"] == 0

    def test_selects_2_4_at_default_sparsity(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        """Regression: the float target landed at 0.44999... and never chose 2:4."""
        _, report = PruningSparsityPass().run(graph, architecture, CompilerConfig())
        entry = report.details["masks"]["layer_0.q_proj"]
        assert entry["pattern"] == "2:4"
        assert entry["nm_pattern_valid"] is True
        assert entry["achieved_sparsity"] == pytest.approx(0.5)

    def test_attaches_mask_to_node(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        PruningSparsityPass().run(graph, architecture, CompilerConfig())
        mask = graph.get_node("layer_0.q_proj").get_attribute("pruning_mask")
        assert mask is not None
        assert mask.mask.shape == (32, 64)

    def test_falls_back_to_magnitude_without_activations(
        self, architecture: ModelArchitecture
    ) -> None:
        g = AEGGraph(name="no_acts")
        node = AEGGraphNode(
            id="n",
            node_type=AEGGraphNodeType.OPERATION,
            name="n",
            op_type="linear",
            layer_index=0,
        )
        node.add_attribute("weight", np.random.RandomState(1).randn(16, 32).astype(np.float32))
        g.add_node(node)
        _, report = PruningSparsityPass().run(g, architecture, CompilerConfig(pruning_metric="wanda"))
        entry = report.details["masks"]["n"]
        assert entry["importance_metric"] == "magnitude"
        assert "metric_fallback" in entry

    def test_falls_back_off_2_4_for_misaligned_dimensions(
        self, architecture: ModelArchitecture
    ) -> None:
        g = AEGGraph(name="odd")
        node = AEGGraphNode(
            id="odd",
            node_type=AEGGraphNodeType.OPERATION,
            name="odd",
            op_type="linear",
            layer_index=0,
        )
        node.add_attribute("weight", np.random.RandomState(2).randn(8, 30).astype(np.float32))
        g.add_node(node)
        _, report = PruningSparsityPass().run(g, architecture, CompilerConfig())
        entry = report.details["masks"]["odd"]
        assert entry["pattern"] == "unstructured"
        assert "pattern_fallback" in entry

    def test_plans_without_weights(self, architecture: ModelArchitecture) -> None:
        g = AEGGraph(name="planned")
        g.add_node(
            AEGGraphNode(
                id="p",
                node_type=AEGGraphNodeType.OPERATION,
                name="p",
                op_type="linear",
                layer_index=0,
            )
        )
        _, report = PruningSparsityPass().run(g, architecture, CompilerConfig())
        entry = report.details["masks"]["p"]
        assert entry["mask_computed"] is False
        assert "reason" in entry

    @pytest.mark.parametrize("metric", ["magnitude", "wanda", "sparsegpt"])
    def test_honours_configured_metric(
        self, graph: AEGGraph, architecture: ModelArchitecture, metric: str
    ) -> None:
        config = CompilerConfig(pruning_metric=metric)
        _, report = PruningSparsityPass().run(graph, architecture, config)
        assert report.details["masks"]["layer_0.q_proj"]["importance_metric"] == metric


class TestOptimizerPipeline:
    def test_runs_all_passes(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        pipeline = OptimizerPipeline(CompilerConfig())
        # Pipeline has 22 passes: 9 original (v3.1) + 13 new (v4.0/v5.0)
        assert pipeline.pass_count >= 9
        _, reports = pipeline.run(graph, architecture)
        assert len(reports) >= 9

    def test_pass_order_matches_documented_sequence(self) -> None:
        # First 9 passes (v3.1) must appear in this order as the foundation.
        # Passes 10-22 (v4.0/v5.0) follow after pass 9.
        expected_first_nine = [
            "operator_fusion",
            "sensitivity_analysis",
            "precision_assignment",
            "kv_cache_structuring",
            "moe_routing",
            "parallelism_discovery",
            "reasoning_graph",
            "sparse_attention",
            "pruning_sparsity",
        ]
        pipeline = OptimizerPipeline(CompilerConfig())
        actual_names = [p.name for p in pipeline._passes]
        assert actual_names[:9] == expected_first_nine, (
            f"First 9 passes out of order. Got: {actual_names[:9]}"
        )

    def test_no_pass_fails_in_a_full_run(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        _, reports = OptimizerPipeline(CompilerConfig()).run(graph, architecture)
        failed = [r.pass_name for r in reports if r.status == "failed"]
        assert not failed, failed

    def test_disabled_pass_is_reported_as_skipped(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        pipeline = OptimizerPipeline(CompilerConfig(enable_fusion=False))
        _, reports = pipeline.run(graph, architecture)
        fusion = next(r for r in reports if r.pass_name == "operator_fusion")
        assert fusion.status == "skipped"

    def test_enable_and_disable_round_trip(self) -> None:
        pipeline = OptimizerPipeline(CompilerConfig())
        pipeline.disable_pass("moe_routing")
        assert pipeline._pass_enabled["moe_routing"] is False
        pipeline.enable_pass("moe_routing")
        assert pipeline._pass_enabled["moe_routing"] is True

    def test_unknown_pass_toggle_is_ignored(self) -> None:
        pipeline = OptimizerPipeline(CompilerConfig())
        pipeline.enable_pass("nonexistent")
        assert "nonexistent" not in pipeline._pass_enabled

    def test_custom_pass_can_be_registered(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        class CountingPass(BasePass):
            name = "counting"
            description = "Counts nodes."

            def run(self, g, arch, config):  # type: ignore[no-untyped-def]
                from aether.compiler.report import PassReport

                return g, PassReport(pass_name=self.name, status="applied", details={})

        pipeline = OptimizerPipeline(CompilerConfig())
        base_count = pipeline.pass_count
        pipeline.register_pass(CountingPass())
        assert pipeline.pass_count == base_count + 1
        _, reports = pipeline.run(graph, architecture)
        assert any(r.pass_name == "counting" for r in reports)

    def test_repr_reports_enabled_count(self) -> None:
        # repr should mention the pass count; with 22 passes it's >= 9
        r = repr(OptimizerPipeline(CompilerConfig()))
        assert any(str(n) in r for n in range(9, 25)), (
            f"Expected pass count >= 9 in repr, got: {r}"
        )

    def test_graph_metadata_accumulates_across_passes(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> None:
        result, _ = OptimizerPipeline(CompilerConfig()).run(graph, architecture)
        for key in ("sensitivity_map", "sharding_plans", "reasoning_graph", "sparsity_plan"):
            assert key in result.metadata, key


class TestPassFailureHandling:
    @pytest.mark.parametrize("pass_class", ALL_PASS_CLASSES)
    def test_pass_reports_failure_instead_of_raising(self, pass_class: type) -> None:
        """A broken architecture must yield status='failed', never an exception."""

        class Hostile:
            def __getattr__(self, name: str) -> object:
                msg = f"exploding attribute {name}"
                raise RuntimeError(msg)

        _, report = pass_class().run(AEGGraph(name="g"), Hostile(), CompilerConfig())
        assert report.status in ("failed", "skipped", "applied")

    def test_module_exposes_logger(self) -> None:
        assert optimizer_module.logger is not None
