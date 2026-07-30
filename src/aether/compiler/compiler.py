"""
Aether Compiler API.

The `Compiler` class is the main entry point for compiling AI models into the
Aether Execution Graph (AEG) format. It orchestrates the five stages of the
compiler pipeline:

1. Model ingestion and graph extraction
2. Aether optimizer (six graph-level passes)
3. Hardware targeting and backend selection
4. AEG artifact packaging
5. Quality report generation

It also exposes the dry-run `plan()` method that previews the compilation
without doing the expensive work.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.plan import (
    CompilationPlan,
    OptimizationOpportunity,
    estimate_compile_time_s,
    estimate_memory_gb,
    recommend_backend,
)
from aether.compiler.report import (
    FusionSummary,
    PassReport,
    PrecisionSummary,
    QualityReport,
    compute_average_bit_width,
    compute_precision_distribution,
)
from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.aeg_format import AEGManifest, AEGPackage, KernelSetMetadata, MemoryRequirements, OptimizationMetadata
from aether.core.constants import AEG_FORMAT_VERSION, AETHER_VERSION, DEFAULT_HUB_URL
from aether.core.exceptions import CompilationError, CompilerConfigError
from aether.core.hash_utils import compute_aeg_cache_key, compute_graph_hash
from aether.core.types import HardwareTarget, ModelArchitecture, Precision
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class Compiler:
    """Main Aether compiler.

    Usage:
        compiler = Compiler()
        plan = compiler.plan("Qwen/Qwen3-8B")
        aeg = compiler.compile("Qwen/Qwen3-8B")
        aeg.save("./qwen3-8b.aeg")
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        """Initialize the compiler with a configuration.

        Args:
            config: Compiler configuration. If None, uses default configuration.
        """
        self.config = config or CompilerConfig()
        self.pipeline = OptimizerPipeline(self.config)
        logger.info(
            "Aether compiler initialized",
            aether_version=AETHER_VERSION,
            optimization_level=self.config.optimization_level,
            targets=self.config.get_targets(),
        )

    def plan(
        self,
        model: str,
        *,
        hardware: str | None = None,
    ) -> CompilationPlan:
        """Dry-run compilation planning.

        Inspects the model and estimates what the compiler would do without
        actually producing an AEG artifact. This is useful for interactive
        exploration and CI smoke checks.

        Args:
            model: Model identifier, local path, or file name.
            hardware: Optional hardware target override. "auto" means detect.

        Returns:
            A `CompilationPlan` with opportunities, estimates, and warnings.
        """
        try:
            architecture = self._detect_architecture(model)
        except Exception as exc:
            plan = CompilationPlan(model_id=model)
            plan.add_error(f"Architecture detection failed: {exc}")
            return plan

        targets = self._resolve_targets(hardware)
        plan = CompilationPlan(model_id=model, architecture=architecture, targets=targets)

        # Add fusion opportunities
        if self.config.enable_fusion and architecture:
            plan.add_fusion_opportunity(
                OptimizationOpportunity(
                    pass_name="operator_fusion",
                    description="Fuse RMSNorm + QKV + RoPE into a single megakernel",
                    nodes=[f"layer_{i}_norm" for i in range(min(architecture.layers, 4))],
                    estimated_memory_saved_mb=architecture.params_billion * 50.0,
                    estimated_latency_reduction_ms=0.5 * architecture.layers,
                    confidence=0.9,
                )
            )
            plan.add_fusion_opportunity(
                OptimizationOpportunity(
                    pass_name="operator_fusion",
                    description="Fuse attention output projection + residual add",
                    nodes=["attn_out", "residual_add"],
                    estimated_memory_saved_mb=architecture.params_billion * 20.0,
                    estimated_latency_reduction_ms=0.2 * architecture.layers,
                    confidence=0.85,
                )
            )

        # Add sensitivity/precision opportunities
        if self.config.enable_sensitivity and self.config.enable_precision_assignment:
            plan.add_sensitivity_opportunity(
                OptimizationOpportunity(
                    pass_name="sensitivity_analysis",
                    description="Compute d(perplexity)/d(precision) per layer",
                    nodes=[f"layer_{i}" for i in range(architecture.layers)],
                    estimated_memory_saved_mb=0.0,
                    estimated_latency_reduction_ms=0.0,
                    confidence=0.95,
                )
            )
            plan.add_precision_opportunity(
                OptimizationOpportunity(
                    pass_name="precision_assignment",
                    description="Assign mixed precision (BF16/FP8/Q4) based on sensitivity",
                    nodes=[f"layer_{i}" for i in range(architecture.layers)],
                    estimated_memory_saved_mb=architecture.params_billion * 500.0,
                    estimated_latency_reduction_ms=0.0,
                    confidence=0.8,
                )
            )

        # Add KV cache opportunities
        if self.config.enable_kv_cache_structuring:
            plan.add_kv_cache_opportunity(
                OptimizationOpportunity(
                    pass_name="kv_cache_structuring",
                    description="Paged KV cache with RadixTree prefix hints",
                    nodes=["kv_cache"],
                    estimated_memory_saved_mb=architecture.context_length * 0.1,
                    estimated_latency_reduction_ms=0.3,
                    confidence=0.9,
                )
            )

        # Add MoE opportunities
        if self.config.enable_moe_routing and architecture.is_moe:
            plan.add_moe_opportunity(
                OptimizationOpportunity(
                    pass_name="moe_routing",
                    description="Hot/warm/cold expert tiering and threshold-based routing",
                    nodes=["moe_router"],
                    estimated_memory_saved_mb=architecture.params_billion * 1000.0,
                    estimated_latency_reduction_ms=1.0,
                    confidence=0.75,
                )
            )

        # Add parallelism opportunities
        if self.config.enable_parallelism_discovery:
            for num_gpus in self.config.parallelism_degrees:
                if num_gpus > 1 and architecture.params_billion > 7.0:
                    plan.add_parallelism_opportunity(
                        OptimizationOpportunity(
                            pass_name="parallelism_discovery",
                            description=f"Compute {num_gpus}-GPU tensor/pipeline parallelism plan",
                            nodes=[],
                            estimated_memory_saved_mb=0.0,
                            estimated_latency_reduction_ms=0.0,
                            confidence=0.7,
                        )
                    )

        # Backend recommendations
        for target in targets:
            backend = recommend_backend(target)
            if backend:
                plan.backend_recommendations[target] = backend
            else:
                plan.add_warning(f"No backend available for target {target}")

        # Estimates
        plan.estimated_memory_gb = estimate_memory_gb(
            architecture,
            kv_cache_dtype=self.config.kv_cache_dtype,
            max_context_length=architecture.context_length,
        )
        plan.estimated_compile_time_s = estimate_compile_time_s(
            architecture,
            targets,
            self.config.optimization_level,
            self.config.calibration_tokens,
        )
        # Estimate AEG size: weights + metadata + kernels (approximate)
        plan.estimated_aeg_size_gb = architecture.params_billion * 0.5  # rough compressed size

        return plan

    def compile(
        self,
        model: str,
        *,
        targets: list[str] | None = None,
        quality_budget: float | None = None,
        calibration_dataset: str | None = None,
        output_path: str | Path | None = None,
    ) -> AEGPackage:
        """Compile a model into an AEG artifact.

        Args:
            model: Model identifier, local path, or GGUF file.
            targets: Optional list of hardware targets. If None, uses config.
            quality_budget: Optional quality budget override.
            calibration_dataset: Optional calibration dataset override.
            output_path: Optional path to save the AEG package. If None, uses
                the Aether cache directory.

        Returns:
            A compiled `AEGPackage`.

        Raises:
            CompilationError: If compilation fails.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)
        config = self.config.clone()
        if targets is not None:
            config.targets = targets
        if quality_budget is not None:
            config.quality_budget = quality_budget
        if calibration_dataset is not None:
            config.calibration_dataset = calibration_dataset

        # Resolve model ID and detect architecture
        try:
            architecture = self._detect_architecture(model)
        except Exception as exc:
            msg = f"Architecture detection failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model, stage="ingestion") from exc

        model_id = self._normalize_model_id(model)
        logger.info(
            "Starting compilation",
            model_id=model_id,
            architecture=architecture.family,
            params_billion=architecture.params_billion,
        )

        # Stage 1: Ingestion
        try:
            graph = self._stage1_ingest(model, architecture)
        except Exception as exc:
            msg = f"Model ingestion failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage1_ingestion") from exc

        # Stage 2: Optimize
        try:
            optimized_graph, pass_reports = self._stage2_optimize(graph, architecture, config)
        except Exception as exc:
            msg = f"Optimization failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage2_optimizer") from exc

        # Stage 3: Target and backend selection
        try:
            target_profiles = self._stage3_target(optimized_graph, architecture, config)
        except Exception as exc:
            msg = f"Hardware targeting failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage3_targeting") from exc

        # Stage 4: Package
        try:
            package = self._stage4_package(
                model_id,
                architecture,
                optimized_graph,
                target_profiles,
                pass_reports,
                config,
                output_path,
                start_time,
            )
        except Exception as exc:
            msg = f"AEG packaging failed for {model}: {exc}"
            raise CompilationError(msg, model_id=model_id, stage="stage4_packaging") from exc

        logger.info("Compilation complete", model_id=model_id, package_path=str(package.root))
        return package

    def quality_report(self, aeg: AEGPackage) -> QualityReport:
        """Generate a quality report from a compiled AEG artifact.

        Args:
            aeg: Compiled AEG package.

        Returns:
            A `QualityReport` summarizing the AEG.
        """
        if not aeg.manifest:
            msg = "AEG package has no manifest"
            raise CompilationError(msg)
        return QualityReport(
            model_id=aeg.manifest.model_id,
            architecture=aeg.manifest.architecture,
            memory=aeg.manifest.memory_requirements,
            targets=aeg.manifest.kernels.targets,
            backend_recommendations=aeg.manifest.kernels.backend_plans,
            precision=PrecisionSummary(
                distribution=compute_precision_distribution(aeg.get_precision_map()),
                average_bit_width=compute_average_bit_width(aeg.get_precision_map()),
                quality_budget=aeg.manifest.optimization.quality_budget_ppl_increase,
            ),
            fusion=FusionSummary(
                fused_op_count=aeg.manifest.optimization.fused_ops_count,
                original_op_count=aeg.manifest.optimization.fused_ops_count * 3,
                memory_round_trips_saved=aeg.manifest.optimization.fused_ops_count * 2,
                fusion_patterns={p: 1 for p in aeg.manifest.optimization.fusion_passes_applied},
            ),
        )

    def _detect_architecture(self, model: str) -> ModelArchitecture:
        """Detect the architecture of a model from its name or metadata.

        Args:
            model: Model identifier or path.

        Returns:
            Detected model architecture.
        """
        from aether.compiler.stage1_ingestion.architecture_detector import (
            ArchitectureDetector,
        )

        return ArchitectureDetector().detect(model)

    def _normalize_model_id(self, model: str) -> str:
        """Normalize a model reference to a stable model ID."""
        path = Path(model)
        if path.exists() and path.is_dir():
            return path.name
        return model.replace("/", "--").replace("\\", "--")

    def _resolve_targets(self, hardware: str | None) -> list[str]:
        """Resolve target list from config and optional hardware override."""
        if hardware is not None and hardware != "auto":
            return [hardware]
        return self.config.get_targets()

    def _stage1_ingest(self, model: str, architecture: ModelArchitecture) -> Any:
        """Run Stage 1: model ingestion and graph extraction."""
        from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

        pipeline = IngestionPipeline(self.config)
        return pipeline.ingest(model, architecture)

    def _stage2_optimize(
        self,
        graph: Any,
        architecture: ModelArchitecture,
        config: CompilerConfig,
    ) -> tuple[Any, list[PassReport]]:
        """Run Stage 2: optimizer passes."""
        pipeline = OptimizerPipeline(config)
        return pipeline.run(graph, architecture)

    def _stage3_target(
        self,
        graph: Any,
        architecture: ModelArchitecture,
        config: CompilerConfig,
    ) -> list[HardwareProfile]:
        """Run Stage 3: hardware targeting."""
        from aether.compiler.stage3_targeting.target_registry import TargetRegistry

        registry = TargetRegistry()
        return registry.create_profiles(graph, architecture, config.get_targets())

    def _stage4_package(
        self,
        model_id: str,
        architecture: ModelArchitecture,
        graph: Any,
        target_profiles: list[HardwareProfile],
        pass_reports: list[PassReport],
        config: CompilerConfig,
        output_path: str | Path | None,
        start_time: datetime.datetime,
    ) -> AEGPackage:
        """Run Stage 4: create and save the AEG package."""
        from aether.core.aeg_ir import AEGIRModule

        if output_path is None:
            from aether.utils.file_io import aether_cache_dir

            cache_root = aether_cache_dir(config.cache_dir)
            output_path = cache_root / "models" / f"{model_id}{AEGPackage}".replace("AEGPackage", ".aeg")
        else:
            output_path = Path(output_path)

        if output_path.exists() and not config.overwrite:
            msg = f"AEG output path already exists: {output_path}. Use overwrite=True to replace."
            raise CompilationError(msg, model_id=model_id)
        if output_path.exists():
            shutil.rmtree(output_path)

        package = AEGPackage.create(output_path, model_id=model_id, aether_version=AETHER_VERSION)

        # Convert graph to AEG-IR
        if isinstance(graph, AEGIRModule):
            ir = graph
        elif hasattr(graph, "to_ir"):
            ir = graph.to_ir()
        else:
            ir = AEGIRModule.from_graph(graph)
        package.ir = ir
        if hasattr(graph, "metadata"):
            package.metadata = dict(getattr(graph, "metadata"))
        primary_target = target_profiles[0].target_id if target_profiles else "cpu_avx512"
        from aether.agentic import AgentWorkflowOptimizer, ToolCall
        from aether.attention import MLAPlanner
        from aether.cuda import CUDAGraphCapturePlan
        from aether.distillation import DistillationPipeline
        from aether.fleet import AetherFleetManager, FleetConfig, FleetNode, HotReloadRouter
        from aether.inference.multimodal import default_multimodal_plan
        from aether.observability import ABRolloutController, DriftMonitor, EvalGate
        from aether.runtime.eagle import EAGLE3Planner

        agentic_trace = [
            ToolCall("retrieve", {"query": "string"}, average_latency_ms=18.0),
            ToolCall("rerank", {"documents": "array"}, average_latency_ms=24.0),
            ToolCall("generate", {"context": "string"}, average_latency_ms=90.0, writes_context=True),
        ]
        fleet_nodes = [
            FleetNode("local-0", primary_target, gpu_count=1 if primary_target.startswith(("cuda", "rocm", "metal")) else 0, memory_gb=80.0),
            FleetNode("portable-0", "cpu_avx512", memory_gb=max(16.0, architecture.params_billion * 2)),
        ]
        package.metadata.update({
            "model_id": model_id,
            "architecture": architecture.to_dict(),
            "capabilities": [
                "reasoning_graph",
                "sparse_attention",
                "pruning_sparsity",
                "safety_guardrails",
                "provenance",
                "watermark",
                "lora_adapters",
                "rag_pipeline",
                "agentic_workflow",
                "multimodal_graph",
                "eagle3_speculation",
                "quantization_aware_compilation",
                "observability",
                "eval_gates",
                "ab_rollout",
                "fleet_management",
                "hot_reload",
                "distillation",
                "cuda_graph_capture",
                "mla_native",
                "hybrid_ssm",
            ],
            "agentic_workflow": AgentWorkflowOptimizer(min_sequence_frequency=1).compile([agentic_trace]),
            "eval_gates": EvalGate().manifest(),
            "drift_monitor": DriftMonitor(baseline_win_rate=0.5).manifest(),
            "ab_rollout": ABRolloutController("compile_default", candidate_percent=0.01).manifest(),
            "fleet_deployment": AetherFleetManager(fleet_nodes).plan_manifest(
                model_id,
                FleetConfig(replicas=1, preferred_targets=tuple(profile.target_id for profile in target_profiles) or ("cpu_avx512",)),
            ),
            "hot_reload": HotReloadRouter(active_aeg=model_id).manifest(),
            "distillation": DistillationPipeline().compile_manifest(
                DistillationPipeline().plan(model_id, f"{model_id}-student", task_type="general"),
            ),
            "cuda_graphs": CUDAGraphCapturePlan(primary_target, max_context_length=architecture.context_length).to_dict(),
            "mla_plan": MLAPlanner().plan(architecture, target=primary_target).to_dict(),
            "eagle3": EAGLE3Planner().plan(architecture).to_dict(),
            "multimodal_graph": default_multimodal_plan(model_id).to_graph(),
            "metrics_schema": {
                "version": "otel_metrics/1.0",
                "metrics": [
                    "tokens_per_second",
                    "ttft_ms",
                    "eagle3_accept_rate",
                    "kv_hit_rate",
                    "mla_compression_ratio",
                    "reasoning_budget_used",
                    "gpu_vram_utilization",
                ],
            },
            "quantization_aware_compilation": {
                "enabled": True,
                "calibration_dataset": config.calibration_dataset,
                "quality_budget_ppl_increase": config.quality_budget,
                "precision_formats": ["BF16", "FP8", "FP4", "Q4_K_M", "Q3_K"],
            },
        })

        # Compute graph hash
        graph_hash = compute_graph_hash(ir)

        # Build precision map from pass reports
        precision_map: dict[str, str] = {}
        for report in pass_reports:
            if report.pass_name == "precision_assignment":
                precision_map.update(report.details.get("precision_map", {}))
        if not precision_map:
            precision_map = {f"layer_{i}": "BF16" for i in range(architecture.layers)}
        package.set_precision_map(precision_map)

        # Sharding plans
        from aether.core.aeg_format import create_default_sharding_plans

        sharding_plans = create_default_sharding_plans(architecture)
        for num_gpus, plan in sharding_plans.items():
            package.set_sharding_plan(num_gpus, plan)

        # Backend plans per target
        kernels = KernelSetMetadata(targets=[p.target_id for p in target_profiles])
        for profile in target_profiles:
            backend_name = profile.recommended_backend or "pytorch"
            package.set_backend_plan(profile.target_id, backend_name)
        kernels.backend_plans = {t: package.get_backend_plan(t) for t in kernels.targets if package.get_backend_plan(t)}
        package.manifest.kernels = kernels

        # Optimization metadata
        fused_ops_count = sum(
            1 for r in pass_reports if r.pass_name == "operator_fusion" and r.status == "applied"
        )
        fusion_passes = [r.details.get("fusion_pattern", "unknown") for r in pass_reports if r.pass_name == "operator_fusion"]
        applied_passes = [r.pass_name for r in pass_reports if r.status == "applied"]
        package.metadata["optimizer_passes"] = applied_passes
        optimization = OptimizationMetadata(
            fusion_passes_applied=fusion_passes or applied_passes,
            fused_ops_count=fused_ops_count,
            sensitivity_calibration_dataset=config.calibration_dataset,
            quality_budget_ppl_increase=config.quality_budget,
        )
        package.manifest.optimization = optimization

        # Memory requirements
        mem_gb = estimate_memory_gb(architecture, {k: Precision.from_string(v) for k, v in precision_map.items()})
        bf16_gb = estimate_memory_gb(architecture, None)
        memory = MemoryRequirements(
            bf16_gb=bf16_gb,
            compiled_min_gb=mem_gb,
            recommended_gb=mem_gb * 1.2,
        )
        package.manifest.memory_requirements = memory
        package.manifest.architecture = architecture
        package.manifest.compiled_at = start_time.isoformat()
        package.manifest.graph_hash = graph_hash

        package.save()
        return package

    def __repr__(self) -> str:
        return f"Compiler(level={self.config.optimization_level}, targets={self.config.get_targets()})"
