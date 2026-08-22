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

    def test_detect_unknown_fails_closed(self) -> None:
        detector = ArchitectureDetector()
        import pytest
        with pytest.raises(Exception, match="Could not identify architecture"):
            detector.detect("unknown/custom-model")

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

    def test_from_config_preserves_non_derived_qwen_geometry(self) -> None:
        """Qwen3's head dimension and numerical constants come from config."""
        arch = ArchitectureDetector()._from_config(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "num_hidden_layers": 28,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "intermediate_size": 3072,
            }
        )
        assert arch.head_dim == 128
        assert arch.norm_eps == 1e-6
        assert arch.rope_theta == 1_000_000.0
        assert arch.qk_norm is True

    def test_qwen_checkpoint_norm_names_are_bound_to_runtime_components(self) -> None:
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        normalise = IngestionPipeline._normalise_weight_name
        assert normalise("model.layers.0.self_attn.q_norm.weight") == (0, "q_norm")
        assert normalise("model.layers.0.self_attn.k_norm.weight") == (0, "k_norm")
        assert normalise("model.norm.weight") == (None, "final_norm")

    def test_approximate_quality_sensitive_passes_are_opt_in(self) -> None:
        config = CompilerConfig()
        assert config.enable_sparse_attention is True
        assert config.enable_pruning is False

    @pytest.mark.parametrize(
        ("model_type", "expected_family"),
        [
            ("llama", "llama_family"),
            ("qwen3", "qwen_family"),
            ("gemma3", "gemma_family"),
            ("mixtral", "moe_family"),
            ("deepseek_v3", "deepseek_family"),
            ("mamba", "hybrid_ssm_family"),
        ],
    )
    def test_from_config_model_type_aliases(
        self, model_type: str, expected_family: str
    ) -> None:
        """Configs without an ``architectures`` list must still identify safely."""
        detector = ArchitectureDetector()
        arch = detector._from_config(
            {
                "model_type": model_type,
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "vocab_size": 32,
            }
        )
        assert arch.family == expected_family

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
        # Pipeline now runs all 22 passes (PRD v3.1 passes 1–9, v4.0 passes 10–17,
        # v5.0 passes 18–22).  Opt-in passes emit a "skipped" report, so every
        # pass always contributes exactly one PassReport.
        assert len(reports) == 22
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
        assert not plan.is_feasible
        assert plan.errors

    def test_compile_small_model(self, tmp_cache_dir, tiny_local_safetensors_model) -> None:
        compiler = Compiler(CompilerConfig(targets=["cpu_avx512"], overwrite=True, dry_run=False))
        aeg = compiler.compile(str(tiny_local_safetensors_model), output_path=tmp_cache_dir / "qwen3-0.6b.aeg")
        assert aeg.root.exists()
        assert (aeg.root / "manifest.json").exists()

    def test_hub_snapshot_config_replaces_pre_download_name_geometry(
        self, tmp_path, tiny_local_safetensors_model, monkeypatch
    ) -> None:
        """A downloaded checkpoint, not a name table, owns executable geometry."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
        from aether.runtime.aeg_loader import load_engine_from_path

        # Simulate the Hub path without network access.  The Qwen name causes
        # the initial detector to have a provisional (and intentionally stale)
        # vocabulary, while the materialized checkpoint is a valid tiny model.
        monkeypatch.setenv("AETHER_HF_OFFLINE", "1")
        monkeypatch.setattr(
            IngestionPipeline,
            "_download_hf_snapshot",
            lambda _self, _model: str(tiny_local_safetensors_model),
        )
        artifact = tmp_path / "hub-snapshot.aeg"
        package = Compiler(
            CompilerConfig(
                targets=["cpu_avx512"],
                overwrite=True,
                skip_download=False,
                calibration_tokens=8,
                cache_dir=str(tmp_path / "cache"),
            )
        ).compile("Qwen/Qwen3-0.6B", output_path=artifact)

        assert package.manifest is not None
        assert package.manifest.architecture.vocab_size == 32
        assert package.manifest.architecture.layers == 1
        engine = load_engine_from_path(artifact)
        logits, _ = engine.forward([1, 2])
        assert logits.shape == (2, 32)

    def test_quality_report(self, minimal_aeg_package) -> None:
        compiler = Compiler()
        report = compiler.quality_report(minimal_aeg_package)
        assert report.model_id == "test-model"
        assert report.is_success


from aether.core.graph import AEGGraph
