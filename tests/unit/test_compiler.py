"""
Tests for the compiler pipeline and configuration.
"""

from __future__ import annotations

import pytest

from aether import Compiler, CompilerConfig
from aether.compiler.config import CompilerConfig
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
from aether.core.exceptions import CompilerConfigError
from aether.core.types import ModelArchitecture


class TestCompilerConfig:
    """Tests for compiler configuration validation."""

    def test_default_config(self) -> None:
        config = CompilerConfig()
        assert config.quality_budget == 0.02
        assert config.optimization_level == 2
        assert config.targets == ["auto"]

    def test_quality_budget_out_of_range(self) -> None:
        with pytest.raises(CompilerConfigError):
            CompilerConfig(quality_budget=1.5)

    def test_invalid_optimization_level(self) -> None:
        with pytest.raises(CompilerConfigError):
            CompilerConfig(optimization_level=5)

    def test_resolve_targets_auto(self) -> None:
        config = CompilerConfig(targets=["auto"])
        targets = config.get_targets()
        assert len(targets) >= 1
        assert "auto" not in targets

    def test_resolve_targets_explicit(self) -> None:
        config = CompilerConfig(targets=["cpu_avx512", "cuda_sm90"])
        targets = config.get_targets()
        assert "cpu_avx512" in targets
        assert "cuda_sm90" in targets

    def test_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("AETHER_QUALITY_BUDGET", "0.01")
        monkeypatch.setenv("AETHER_OPTIMIZATION_LEVEL", "3")
        config = CompilerConfig.from_env()
        assert config.quality_budget == 0.01
        assert config.optimization_level == 3


class TestArchitectureDetector:
    """Tests for architecture detection."""

    def test_detect_qwen3(self) -> None:
        detector = ArchitectureDetector()
        arch = detector.detect("Qwen/Qwen3-8B")
        assert arch.family == "qwen_family"
        assert arch.params_billion == 8.0
        assert arch.layers == 32

    def test_detect_llama(self) -> None:
        detector = ArchitectureDetector()
        arch = detector.detect("meta-llama/Llama-3.3-70B")
        assert arch.family == "llama_family"
        assert arch.params_billion == 70.0

    def test_detect_unknown_defaults(self) -> None:
        detector = ArchitectureDetector()
        arch = detector.detect("unknown/custom-model")
        assert arch.family == "llama_family"
        assert arch.layers == 32

    def test_from_config(self) -> None:
        detector = ArchitectureDetector()
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "intermediate_size": 2816,
        }
        arch = detector._from_config(config)
        assert arch.family == "qwen_family"
        assert arch.layers == 24

    def test_check_compatibility_moe(self) -> None:
        detector = ArchitectureDetector()
        arch = ModelArchitecture(
            family="deepseek_family",
            params_billion=671.0,
            layers=80,
            hidden_size=7168,
            num_attention_heads=64,
            is_moe=True,
            num_experts=256,
        )
        warnings = detector.check_compatibility(arch)
        assert any("MoE" in w for w in warnings)


class TestOptimizerPipeline:
    """Tests for the optimizer pipeline."""

    def test_pipeline_run(self, small_architecture: ModelArchitecture) -> None:
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        from aether.compiler.config import CompilerConfig

        config = CompilerConfig(optimization_level=2)
        ingestion = IngestionPipeline(config)
        graph = ingestion._build_architecture_graph(AEGGraph(name="test"), small_architecture)  # noqa: SLF001
        pipeline = OptimizerPipeline(config)
        optimized, reports = pipeline.run(graph, small_architecture)
        assert len(reports) == 6
        assert all(r.status in ("applied", "skipped", "failed") for r in reports)

    def test_disable_pass(self) -> None:
        config = CompilerConfig()
        pipeline = OptimizerPipeline(config)
        pipeline.disable_pass("operator_fusion")
        assert not pipeline._pass_enabled["operator_fusion"]

    def test_enable_pass(self) -> None:
        config = CompilerConfig()
        pipeline = OptimizerPipeline(config)
        pipeline.disable_pass("operator_fusion")
        pipeline.enable_pass("operator_fusion")
        assert pipeline._pass_enabled["operator_fusion"]


class TestCompilerPlan:
    """Tests for the compiler dry-run plan."""

    def test_plan_returns_opportunities(self) -> None:
        compiler = Compiler()
        plan = compiler.plan("Qwen/Qwen3-8B")
        assert plan.model_id == "Qwen/Qwen3-8B"
        assert plan.total_opportunities > 0
        assert plan.is_feasible

    def test_plan_unknown_model(self) -> None:
        compiler = Compiler()
        plan = compiler.plan("unknown/unknown-model-xyz")
        # Unknown model gets default architecture so it should still be feasible
        assert plan.is_feasible or len(plan.errors) == 0

    def test_compile_small_model(self, tmp_cache_dir) -> None:
        compiler = Compiler(CompilerConfig(targets=["cpu_avx512"], overwrite=True, dry_run=False))
        aeg = compiler.compile("Qwen/Qwen3-0.6B", output_path=tmp_cache_dir / "qwen3-0.6b.aeg")
        assert aeg.root.exists()
        assert (aeg.root / "manifest.json").exists()

    def test_quality_report(self, minimal_aeg_package) -> None:
        compiler = Compiler()
        report = compiler.quality_report(minimal_aeg_package)
        assert report.model_id == "test-model"
        assert report.is_success


from aether.core.graph import AEGGraph
