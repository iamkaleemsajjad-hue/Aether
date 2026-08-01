"""
Public API surface contract.

The v3.1 API names were once bound to placeholder aliases — ``ModalityEncoder``
pointed at ``VLMConfig``, ``MLAPlanner`` at ``MLADetector``. Those satisfied
``import`` but not use: the compiler called ``MLAPlanner().plan(...)``, a method
the detector never had, and the failure only surfaced at Stage 4 packaging.

These tests pin the surface so an alias cannot quietly stand in for an
implementation again:

* every name a package advertises in ``__all__`` is importable;
* the plan-layer types are distinct classes, not aliases of a config or a
  detector, and expose the methods their callers actually invoke;
* the runtime classes that the aliases used to shadow are still exported.
"""

from __future__ import annotations

import importlib

import pytest

#: Packages whose ``__all__`` is a public contract.
PACKAGES = [
    "aether.inference",
    "aether.attention",
    "aether.adapters",
    "aether.hybrid",
    "aether.provenance",
    "aether.agentic",
    "aether.observability",
    "aether.fleet",
    "aether.cuda",
    "aether.distillation",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_every_advertised_name_is_importable(package: str) -> None:
    """A name in ``__all__`` that cannot be imported is a broken promise."""
    mod = importlib.import_module(package)
    declared = getattr(mod, "__all__", [])
    assert declared, f"{package} declares no __all__"
    missing = [name for name in declared if not hasattr(mod, name)]
    assert not missing, f"{package}.__all__ advertises missing names: {missing}"


class TestPlanLayerIsNotAliased:
    """The compile-time plan types must be real, distinct implementations."""

    def test_modality_encoder_is_not_vlm_config(self) -> None:
        """Verify ModalityEncoder is a distinct class from VLMConfig."""
        from aether.inference import ModalityEncoder, VLMConfig

        assert ModalityEncoder is not VLMConfig
        encoder = ModalityEncoder("image", "vit.aeg")
        assert encoder.encode_op == "aeg.vision_encode"

    def test_multimodal_plan_is_not_vlm_config(self) -> None:
        """Verify MultiModalGraphPlan is a distinct class from VLMConfig."""
        from aether.inference import ModalityEncoder, MultiModalGraphPlan, VLMConfig

        assert MultiModalGraphPlan is not VLMConfig
        plan = MultiModalGraphPlan(
            llm_model="llm.aeg", encoders=(ModalityEncoder("image", "vit.aeg"),)
        )
        # The terminal stage is what the runtime dispatches to.
        assert plan.to_graph()["stages"][-1]["op"] == "aeg.llm_generate"

    def test_default_multimodal_plan_is_a_factory_not_an_instance(self) -> None:
        """Verify default_multimodal_plan is a factory function returning a plan."""
        from aether.inference import MultiModalGraphPlan, default_multimodal_plan

        assert callable(default_multimodal_plan)
        plan = default_multimodal_plan("Qwen/Qwen3-0.6B")
        assert isinstance(plan, MultiModalGraphPlan)

    def test_mla_planner_is_not_the_detector_and_can_plan(self) -> None:
        """Verify MLAPlanner is a distinct class from MLADetector and produces a compression plan."""
        from aether.attention import MLACompressionPlan, MLADetector, MLAPlanner
        from aether.core.types import ModelArchitecture

        assert MLAPlanner is not MLADetector
        arch = ModelArchitecture(
            family="deepseek_family",
            params_billion=67,
            layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=4,
            attention_type="MLA",
            context_length=131072,
        )
        # This exact call is what compiler.py makes at Stage 4 packaging.
        plan = MLAPlanner().plan(arch, target="cuda_sm100")
        assert isinstance(plan, MLACompressionPlan)
        assert plan.to_dict()["kernel"] == "aeg.mla_flash_attention_4"

    def test_retrieval_source_accepts_arguments(self) -> None:
        """Verify RetrievalSource correctly initializes with valid arguments."""
        from aether.inference import RetrievalSource

        source = RetrievalSource("docs", "bm25", top_k=25)
        assert source.op == "aeg.bm25_search"
        assert source.top_k == 25

    def test_retrieval_source_rejects_unknown_kind(self) -> None:
        """Verify RetrievalSource raises ValueError for invalid source types."""
        from aether.inference import RetrievalSource

        with pytest.raises(ValueError, match="Unknown retrieval kind"):
            RetrievalSource("docs", "telepathy")


class TestRuntimeLayerStillExported:
    """Rebinding the aliases must not have dropped what they pointed at."""

    @pytest.mark.parametrize(
        ("module", "symbol"),
        [
            ("aether.inference", "VLMConfig"),
            ("aether.inference", "MultiModalGraphDispatcher"),
            ("aether.inference", "ViTEncoder"),
            ("aether.inference", "ImagePreprocessor"),
            ("aether.inference", "VisualTokenCompressor"),
            ("aether.inference", "ModalConnector"),
            ("aether.inference", "RAGPipeline"),
            ("aether.inference", "Document"),
            ("aether.inference", "RetrievalResult"),
            ("aether.attention", "MLADetector"),
            ("aether.attention", "MLAConfig"),
            ("aether.attention", "MLAWeightAbsorber"),
            ("aether.attention", "MLACompressedKVCache"),
            ("aether.attention", "MLAForward"),
            ("aether.runtime.precision_manager", "DynamicPrecisionManager"),
            ("aether.runtime.precision_manager", "PrecisionState"),
            ("aether.runtime.precision_manager", "PrecisionSnapshot"),
            ("aether.hybrid", "SSMStatePool"),
            ("aether.hybrid", "StateSnapshot"),
        ],
    )
    def test_symbol_is_still_exported(self, module: str, symbol: str) -> None:
        """Verify symbol is exported from the specified module."""
        assert hasattr(importlib.import_module(module), symbol)

    def test_mla_forward_alias_points_at_the_forward_path(self) -> None:
        """Verify MLAForward points to an implementation with forward_prefill and forward_decode."""
        from aether.attention import MLAForward

        assert hasattr(MLAForward, "forward_prefill")
        assert hasattr(MLAForward, "forward_decode")


class TestOptimizerPassEntryPoints:
    """Each ``passN_*`` module is the stable public path for its pass."""

    @pytest.mark.parametrize(
        ("module", "symbol"),
        [
            ("pass1_operator_fusion", "OperatorFusionPass"),
            ("pass2_sensitivity_analysis", "SensitivityAnalysisPass"),
            ("pass3_precision_assignment", "PrecisionAssignmentPass"),
            ("pass4_kv_cache_structuring", "KVCacheStructuringPass"),
            ("pass5_moe_routing", "MoERoutingPass"),
            ("pass6_parallelism_discovery", "ParallelismDiscoveryPass"),
            ("pass7_reasoning_graph", "ReasoningGraphPass"),
            ("pass8_sparse_attention", "SparseAttentionPass"),
            ("pass9_pruning_sparsity", "PruningSparsityPass"),
        ],
    )
    def test_pass_entry_point_re_exports_implementation(
        self, module: str, symbol: str
    ) -> None:
        """Verify pass entry point re-exports the exact implementation class."""
        from aether.compiler.stage2_optimizer import optimizer

        mod = importlib.import_module(f"aether.compiler.stage2_optimizer.{module}")
        assert hasattr(mod, symbol), f"{module} does not re-export {symbol}"
        # The entry point must be the same object as the orchestrator's, not a copy.
        assert getattr(mod, symbol) is getattr(optimizer, symbol)
