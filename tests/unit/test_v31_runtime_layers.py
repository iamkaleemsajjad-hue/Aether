"""Tests for PRD v3.1 runtime and platform layers."""

from __future__ import annotations

from aether.agentic import AgentWorkflowOptimizer, ToolCall
from aether.attention import MLAPlanner
from aether.compiler.config import CompilerConfig
from aether.core.aeg_format import AEGPackage
from aether.core.types import ModelArchitecture
from aether.cuda import CUDAGraphCapturePlan, CUDAGraphSelector
from aether.distillation import DistillationMode, DistillationPipeline
from aether.fleet import AetherFleetManager, FleetConfig, FleetNode, HotReloadRouter
from aether.inference.multimodal import ModalityEncoder, MultiModalGraphPlan
from aether.observability import ABRolloutController, DriftMonitor, EvalGate, EvalResult, TelemetrySnapshot
from aether.runtime.eagle import EAGLE3Planner


def test_agentic_workflow_optimizer_compiles_meta_tools_and_routes() -> None:
    trace = [
        ToolCall("retrieve", {"query": "string"}, average_latency_ms=10.0),
        ToolCall("rerank", {"documents": "array"}, average_latency_ms=20.0),
        ToolCall("generate", {"context": "string"}, average_latency_ms=80.0, writes_context=True),
    ]
    optimizer = AgentWorkflowOptimizer(min_sequence_frequency=2)
    plan = optimizer.compile([trace, trace])
    assert plan["meta_tools"]
    assert plan["context_cache"]["enabled"] is True
    route = optimizer.route_for_prompt("debug and analyze this complex tool plan", tool_count=3)
    assert route.model_tier in {"medium", "large"}


def test_observability_eval_gate_drift_and_rollout() -> None:
    results = [
        EvalResult("hellaswag", 0.8, 0.79),
        EvalResult("mmlu", 0.75, 0.74),
        EvalResult("gsm8k", 0.7, 0.7),
        EvalResult("math-500", 0.65, 0.65),
        EvalResult("humaneval", 0.55, 0.55),
    ]
    decision = EvalGate(max_relative_regression=0.02).evaluate(results)
    assert decision.passed is True
    monitor = DriftMonitor(baseline_win_rate=0.6, alert_drop=0.05, min_samples=2)
    monitor.record(TelemetrySnapshot(100, 50, 0.8, 0.7, 10, 0.5, 0.6, win_rate=0.5))
    status = monitor.record(TelemetrySnapshot(100, 55, 0.8, 0.7, 10, 0.5, 0.6, win_rate=0.48))
    assert status["alert"] is True
    rollout = ABRolloutController("exp", candidate_percent=0.5)
    assert rollout.assign("request-1") in {"control", "candidate"}
    assert rollout.ramp(gate_passed=False, drift_alert=False) == 0.0


def test_fleet_manager_and_hot_reload_router() -> None:
    manager = AetherFleetManager([
        FleetNode("gpu-a", "cuda_sm100", gpu_count=8, memory_gb=640, region="us-east"),
        FleetNode("cpu-a", "cpu_avx512", memory_gb=128, region="us-east"),
    ])
    handle = manager.deploy("model.aeg", FleetConfig(replicas=2, min_gpu_memory_gb=80, preferred_targets=("cuda_sm100",)))
    assert len(handle.assignments) == 2
    assert handle.assignments[0]["kernel_profile"]["cuda_graphs"] is True
    router = HotReloadRouter("v1.aeg", "v2.aeg", candidate_percent=1.0)
    assert router.route("req") == "v2.aeg"
    assert router.rollback() == "v1.aeg"


def test_distillation_pipeline_modes_and_loss_weights() -> None:
    pipeline = DistillationPipeline()
    plan = pipeline.plan("teacher", "student", task_type="reasoning", compression_target=0.2)
    assert DistillationMode.REASONING in plan.modes
    manifest = pipeline.compile_manifest(plan)
    assert manifest["loss_weights"]["reasoning_trace_ce"] > 0
    assert "student.aeg" in manifest["compiler_outputs"]


def test_cuda_graph_selector_rounds_dynamic_shapes() -> None:
    selector = CUDAGraphSelector(CUDAGraphCapturePlan("cuda_sm100"))
    assert selector.select_graph("decode", 9)["bucket"] == 16
    assert selector.select_graph("prefill", 513)["bucket"] == 1024
    assert selector.select_graph("decode", 128)["mode"] == "eager_piecewise"


def test_mla_and_eagle3_planners_use_architecture() -> None:
    architecture = ModelArchitecture(
        family="deepseek_family",
        params_billion=67,
        layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=4,
        attention_type="MLA",
        context_length=131072,
    )
    mla = MLAPlanner().plan(architecture, target="cuda_sm100")
    assert mla.enabled is True
    assert mla.compression_ratio > 1
    assert mla.kernel == "aeg.mla_flash_attention_4"
    eagle = EAGLE3Planner().plan(architecture, draft_model_id="draft")
    assert eagle.attention_drift_correction is True
    assert len(eagle.fusion_layers) > 1


def test_multimodal_unified_graph_plan() -> None:
    plan = MultiModalGraphPlan(
        llm_model="llm.aeg",
        encoders=(ModalityEncoder("image", "vit.aeg"), ModalityEncoder("video", "video.aeg", token_budget=100000)),
    )
    graph = plan.to_graph()
    assert graph["stages"][-1]["op"] == "aeg.llm_generate"
    assert graph["optimizations"]["mm_sparse_attention"] is True


def test_compiled_package_emits_remaining_v31_artifact_contracts(tmp_path, tiny_local_safetensors_model) -> None:
    from aether import Compiler

    compiler = Compiler(CompilerConfig(targets=["cpu_avx512"], overwrite=True, calibration_tokens=128))
    package = compiler.compile(str(tiny_local_safetensors_model), output_path=tmp_path / "qwen.aeg")
    loaded = AEGPackage(package.root).load()
    expected = [
        "agentic/workflow_cache.json",
        "observability/eval_gates.json",
        "observability/drift_monitor.json",
        "observability/metrics_schema.json",
        "rollout/ab_config.json",
        "fleet/deployment_plan.json",
        "fleet/hot_reload.json",
        "distillation/plan.json",
        "cuda_graphs/capture_manifest.json",
        "mla/plan.json",
        "speculation/eagle3.json",
        "multimodal/graph.json",
    ]
    assert loaded.manifest is not None
    for artifact in expected:
        assert (loaded.root / artifact).exists(), artifact
        assert artifact in loaded.manifest.artifacts
    assert loaded.metadata["agentic_workflow"]["meta_tools"]
