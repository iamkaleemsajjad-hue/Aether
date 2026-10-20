"""Default policy and public pass selections must govern real execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aether import Compiler, CompilerConfig
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline

if TYPE_CHECKING:
    from pathlib import Path

    from aether.core.graph import AEGGraph
    from aether.core.types import ModelArchitecture


def _fusion_only(enabled: bool) -> CompilerConfig:
    return CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        skip_download=True,
        enable_fusion=enabled,
        enable_sensitivity=False,
        enable_precision_assignment=False,
        enable_kv_cache_structuring=False,
        enable_moe_routing=False,
        enable_parallelism_discovery=False,
        enable_reasoning_graph=False,
        enable_sparse_attention=False,
    )


def _ingest(source: Path) -> tuple[AEGGraph, ModelArchitecture]:
    architecture = ArchitectureDetector().detect(str(source))
    return IngestionPipeline(_fusion_only(False)).ingest(str(source), architecture), architecture


def test_default_constructor_and_execution_enable_only_consumed_precision(
    tiny_local_safetensors_model: Path,
) -> None:
    """Unused graph plans and unvalidated sparse behavior require explicit opt-in."""
    pipeline = OptimizerPipeline(CompilerConfig())
    assert {name for name, enabled in pipeline._pass_enabled.items() if enabled} == {
        "precision_assignment"
    }
    graph, architecture = _ingest(tiny_local_safetensors_model)
    _, reports = pipeline.run(graph, architecture)
    assert len(reports) == 22
    assert {report.pass_name for report in reports if report.status == "applied"} == {
        "precision_assignment"
    }
    assert all(report.status in {"applied", "skipped"} for report in reports)
    assert "attention_head_patterns" not in graph.metadata


@pytest.mark.parametrize("requested", [False, True])
def test_explicit_pipeline_selection_survives_config_sync_and_repeated_runs(
    tiny_local_safetensors_model: Path,
    requested: bool,
) -> None:
    """The public toggle must change real graph rewriting on every run."""
    pipeline = OptimizerPipeline(_fusion_only(not requested))
    if requested:
        pipeline.enable_pass("operator_fusion")
    else:
        pipeline.disable_pass("operator_fusion")
    for _ in range(2):
        graph, architecture = _ingest(tiny_local_safetensors_model)
        _, reports = pipeline.run(graph, architecture)
        fusion = next(report for report in reports if report.pass_name == "operator_fusion")
        assert fusion.status == ("applied" if requested else "skipped")
        if requested:
            assert fusion.details["fused_count"] > 0
            assert any(node.attributes.get("is_fused_away") for node in graph.nodes.values())
        else:
            assert "fusion_accounting" not in graph.metadata


def test_config_changes_still_apply_without_an_explicit_pipeline_override(
    tiny_local_safetensors_model: Path,
) -> None:
    """Selection overrides must not freeze ordinary mutable CompilerConfig fields."""
    config = _fusion_only(False)
    pipeline = OptimizerPipeline(config)
    config.enable_fusion = True
    graph, architecture = _ingest(tiny_local_safetensors_model)
    _, reports = pipeline.run(graph, architecture)
    fusion = next(report for report in reports if report.pass_name == "operator_fusion")
    assert fusion.details["fused_count"] > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_packaged_fusion_report_describes_ir_without_claiming_runtime_savings(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    enabled: bool,
) -> None:
    """Skipped fusion is empty; real IR rewriting does not prove saved launches."""
    config = _fusion_only(enabled)
    config.cache_dir = str(tmp_path / "cache")
    compiler = Compiler(config)
    package = compiler.compile(
        str(tiny_local_safetensors_model), output_path=tmp_path / "model.aeg"
    )
    package.verify_integrity()
    assert package.manifest is not None
    optimization = package.manifest.optimization
    if enabled:
        accounting = package.metadata["fusion_accounting"]
        assert accounting["fused_groups"] > 0
        assert optimization.fused_ops_count == accounting["fused_groups"]
        assert set(optimization.fusion_passes_applied) == set(accounting["patterns"])
    else:
        assert optimization.fused_ops_count == 0
        assert optimization.fusion_passes_applied == []
    quality = compiler.quality_report(package)
    assert quality.fusion is not None
    assert quality.fusion.memory_round_trips_saved == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_sharding_inspection_payload_requires_an_applied_producer(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    enabled: bool,
) -> None:
    """Disabled discovery cannot silently emit generic multi-GPU templates."""
    config = _fusion_only(False)
    config.enable_parallelism_discovery = enabled
    config.cache_dir = str(tmp_path / "cache")
    package = Compiler(config).compile(
        str(tiny_local_safetensors_model), output_path=tmp_path / "model.aeg"
    )
    package.verify_integrity()
    assert bool(package.sharding_plans) is enabled
    assert bool(list((package.root / "parallelism").glob("*gpu.json"))) is enabled
    if enabled:
        assert {count: plan.to_dict() for count, plan in package.sharding_plans.items()} == {
            int(count): payload for count, payload in package.metadata["sharding_plans"].items()
        }


def test_experimental_fusion_preview_does_not_invent_engine_benefits(
    tiny_local_safetensors_model: Path,
) -> None:
    """The inspection plan cannot claim saved memory, latency or confidence."""
    plan = Compiler(_fusion_only(True)).plan(str(tiny_local_safetensors_model))
    assert plan.fusion_opportunities
    for opportunity in plan.fusion_opportunities:
        assert opportunity.estimated_memory_saved_mb == 0
        assert opportunity.estimated_latency_reduction_ms == 0
        assert opportunity.confidence == 0


@pytest.mark.parametrize("requested", [False, True])
def test_pipeline_selection_reaches_internally_gated_grammar_compilation(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    requested: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected pass must receive the same effective flag as the scheduler."""
    config = _fusion_only(False)
    config.grammar_backend = "builtin"
    config.grammar_schema = 'root ::= "hello"'
    config.enable_grammar_constraint = not requested
    pipeline = OptimizerPipeline(config)
    if requested:
        pipeline.enable_pass("grammar_constraint_compilation")
    else:
        pipeline.disable_pass("grammar_constraint_compilation")
    graph, architecture = _ingest(tiny_local_safetensors_model)
    monkeypatch.setattr(graph, "output_dir", str(tmp_path), raising=False)
    _, reports = pipeline.run(graph, architecture)
    report = next(item for item in reports if item.pass_name == "grammar_constraint_compilation")
    assert report.status == ("applied" if requested else "skipped")
    assert (tmp_path / "grammar/fsm.bin").is_file() is requested
    assert config.enable_grammar_constraint is (not requested)
    if requested:
        assert report.details["tokenizer_aware"] is True
        assert report.details["fsa_transitions"] > 0
