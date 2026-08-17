"""
Compilation planning — dry-run estimation of fusion opportunities, memory, and time.

The `CompilationPlan` class provides a pre-compilation report that lets users
inspect what the compiler would do without performing the expensive work. It is
returned by `compiler.plan()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import SUPPORTED_TARGET_IDS
from aether.core.types import HardwareTarget, ModelArchitecture, Precision, PrecisionTier


@dataclass
class OptimizationOpportunity:
    """A single optimization opportunity identified in a dry-run plan."""

    pass_name: str
    """Optimizer pass that would apply this opportunity."""

    description: str
    """Human-readable description of the opportunity."""

    nodes: list[str] = field(default_factory=list)
    """Graph node IDs involved in the opportunity."""

    estimated_memory_saved_mb: float = 0.0
    """Estimated memory saved in megabytes."""

    estimated_latency_reduction_ms: float = 0.0
    """Estimated latency reduction per token in milliseconds."""

    confidence: float = 1.0
    """Confidence score (0.0-1.0)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_name": self.pass_name,
            "description": self.description,
            "nodes": self.nodes,
            "estimated_memory_saved_mb": self.estimated_memory_saved_mb,
            "estimated_latency_reduction_ms": self.estimated_latency_reduction_ms,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OptimizationOpportunity:
        return OptimizationOpportunity(
            pass_name=data["pass_name"],
            description=data["description"],
            nodes=list(data.get("nodes", [])),
            estimated_memory_saved_mb=data.get("estimated_memory_saved_mb", 0.0),
            estimated_latency_reduction_ms=data.get("estimated_latency_reduction_ms", 0.0),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class CompilationPlan:
    """Result of a dry-run compilation plan.

    Provides a detailed breakdown of the work the compiler would perform
    without actually modifying any state or producing an AEG artifact.
    """

    model_id: str
    """Model identifier."""

    architecture: ModelArchitecture | None = None
    """Detected architecture."""

    targets: list[str] = field(default_factory=list)
    """Hardware targets that would be compiled."""

    fusion_opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    """Identified operator fusion opportunities."""

    sensitivity_opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    """Identified sensitivity analysis opportunities."""

    precision_opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    """Identified mixed-precision opportunities."""

    kv_cache_opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    """Identified KV cache structuring opportunities."""

    moe_opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    """Identified MoE routing opportunities."""

    parallelism_opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    """Identified parallelism opportunities."""

    estimated_memory_gb: float = 0.0
    """Estimated peak memory during compilation in GB."""

    estimated_compile_time_s: float = 0.0
    """Estimated compilation time in seconds."""

    estimated_aeg_size_gb: float = 0.0
    """Estimated final AEG artifact size in GB."""

    estimated_quality_loss_ppl: float | None = None
    """Estimated perplexity increase if known."""

    backend_recommendations: dict[str, str] = field(default_factory=dict)
    """Map of target_id -> recommended backend name."""

    warnings: list[str] = field(default_factory=list)
    """Planning warnings."""

    errors: list[str] = field(default_factory=list)
    """Planning errors that would prevent compilation."""

    def add_fusion_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        """Add a fusion opportunity to the plan."""
        self.fusion_opportunities.append(opportunity)

    def add_sensitivity_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        """Add a sensitivity opportunity to the plan."""
        self.sensitivity_opportunities.append(opportunity)

    def add_precision_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        """Add a precision opportunity to the plan."""
        self.precision_opportunities.append(opportunity)

    def add_kv_cache_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        """Add a KV cache opportunity to the plan."""
        self.kv_cache_opportunities.append(opportunity)

    def add_moe_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        """Add a MoE opportunity to the plan."""
        self.moe_opportunities.append(opportunity)

    def add_parallelism_opportunity(self, opportunity: OptimizationOpportunity) -> None:
        """Add a parallelism opportunity to the plan."""
        self.parallelism_opportunities.append(opportunity)

    def add_warning(self, message: str) -> None:
        """Add a planning warning."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add a planning error."""
        self.errors.append(message)

    @property
    def is_feasible(self) -> bool:
        """Return True if the plan has no errors."""
        return len(self.errors) == 0

    @property
    def total_opportunities(self) -> int:
        """Return total number of optimization opportunities."""
        return (
            len(self.fusion_opportunities)
            + len(self.sensitivity_opportunities)
            + len(self.precision_opportunities)
            + len(self.kv_cache_opportunities)
            + len(self.moe_opportunities)
            + len(self.parallelism_opportunities)
        )

    @property
    def estimated_memory_saved_gb(self) -> float:
        """Sum estimated memory savings across all opportunities in GB."""
        total = 0.0
        for opp in self._all_opportunities():
            total += opp.estimated_memory_saved_mb
        return total / 1024.0

    @property
    def estimated_latency_reduction_ms(self) -> float:
        """Sum estimated latency reductions across all opportunities in ms."""
        total = 0.0
        for opp in self._all_opportunities():
            total += opp.estimated_latency_reduction_ms
        return total

    def _all_opportunities(self) -> list[OptimizationOpportunity]:
        return (
            self.fusion_opportunities
            + self.sensitivity_opportunities
            + self.precision_opportunities
            + self.kv_cache_opportunities
            + self.moe_opportunities
            + self.parallelism_opportunities
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture.to_dict() if self.architecture else None,
            "targets": self.targets,
            "fusion_opportunities": [o.to_dict() for o in self.fusion_opportunities],
            "sensitivity_opportunities": [o.to_dict() for o in self.sensitivity_opportunities],
            "precision_opportunities": [o.to_dict() for o in self.precision_opportunities],
            "kv_cache_opportunities": [o.to_dict() for o in self.kv_cache_opportunities],
            "moe_opportunities": [o.to_dict() for o in self.moe_opportunities],
            "parallelism_opportunities": [o.to_dict() for o in self.parallelism_opportunities],
            "estimated_memory_gb": self.estimated_memory_gb,
            "estimated_compile_time_s": self.estimated_compile_time_s,
            "estimated_aeg_size_gb": self.estimated_aeg_size_gb,
            "estimated_quality_loss_ppl": self.estimated_quality_loss_ppl,
            "backend_recommendations": self.backend_recommendations,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CompilationPlan:
        arch = data.get("architecture")
        return CompilationPlan(
            model_id=data["model_id"],
            architecture=ModelArchitecture.from_dict(arch) if arch else None,
            targets=list(data.get("targets", [])),
            fusion_opportunities=[OptimizationOpportunity.from_dict(o) for o in data.get("fusion_opportunities", [])],
            sensitivity_opportunities=[OptimizationOpportunity.from_dict(o) for o in data.get("sensitivity_opportunities", [])],
            precision_opportunities=[OptimizationOpportunity.from_dict(o) for o in data.get("precision_opportunities", [])],
            kv_cache_opportunities=[OptimizationOpportunity.from_dict(o) for o in data.get("kv_cache_opportunities", [])],
            moe_opportunities=[OptimizationOpportunity.from_dict(o) for o in data.get("moe_opportunities", [])],
            parallelism_opportunities=[OptimizationOpportunity.from_dict(o) for o in data.get("parallelism_opportunities", [])],
            estimated_memory_gb=data.get("estimated_memory_gb", 0.0),
            estimated_compile_time_s=data.get("estimated_compile_time_s", 0.0),
            estimated_aeg_size_gb=data.get("estimated_aeg_size_gb", 0.0),
            estimated_quality_loss_ppl=data.get("estimated_quality_loss_ppl"),
            backend_recommendations=dict(data.get("backend_recommendations", {})),
            warnings=list(data.get("warnings", [])),
            errors=list(data.get("errors", [])),
        )

    def to_json(self, indent: int | None = None) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def __repr__(self) -> str:
        return (
            f"CompilationPlan({self.model_id}, {self.total_opportunities} opportunities, "
            f"feasible={self.is_feasible})"
        )


def estimate_memory_gb(
    architecture: ModelArchitecture,
    precision_map: dict[str, Precision] | None = None,
    kv_cache_dtype: str = "fp8",
    max_context_length: int = 4096,
) -> float:
    """Estimate the memory required to run a model at a given precision.

    Args:
        architecture: Model architecture.
        precision_map: Per-layer precision map. If None, uses BF16 for all.
        kv_cache_dtype: KV cache data type.
        max_context_length: Maximum context length.

    Returns:
        Estimated memory in GB.
    """
    # Weight memory
    if precision_map:
        def _bit_width(p: str | Precision) -> int:
            if isinstance(p, Precision):
                return p.bit_width
            return Precision.from_string(p).bit_width
        avg_bits = sum(_bit_width(p) for p in precision_map.values()) / len(precision_map)
        weight_gb = architecture.params_billion * 1e9 * (avg_bits / 8) / (1024**3)
    else:
        weight_gb = architecture.params_billion * 2.0
    # KV cache memory: 2 (K+V) * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
    head_dim = architecture.head_dim or (architecture.hidden_size // architecture.num_attention_heads)
    kv_bytes = {"fp8": 1, "fp16": 2, "bf16": 2}.get(kv_cache_dtype, 1)
    kv_gb = (
        2
        * architecture.layers
        * (architecture.num_kv_heads or architecture.num_attention_heads)
        * head_dim
        * max_context_length
        * kv_bytes
        / (1024**3)
    )
    # Activations and overhead
    activation_gb = weight_gb * 0.1
    return weight_gb + kv_gb + activation_gb


def estimate_compile_time_s(
    architecture: ModelArchitecture,
    targets: list[str],
    optimization_level: int,
    calibration_tokens: int,
) -> float:
    """Estimate compilation time based on model size, targets, and optimization level.

    Args:
        architecture: Model architecture.
        targets: Number of hardware targets.
        optimization_level: Optimization level (0-3).
        calibration_tokens: Number of calibration tokens.

    Returns:
        Estimated compilation time in seconds.
    """
    base = 30.0 + architecture.params_billion * 2.0
    target_multiplier = 1.0 + 0.3 * (len(targets) - 1)
    level_multiplier = {0: 0.2, 1: 0.7, 2: 1.0, 3: 1.5}[optimization_level]
    calibration_time = max(0.0, (calibration_tokens - 4096) / 1000.0)
    return base * target_multiplier * level_multiplier + calibration_time


def recommend_backend(target_id: str) -> str | None:
    """Recommend the best backend for a target.

    Returns the first available backend from the target's candidate list.
    For CPU targets, always returns ``"aether_cpu"`` as the final fallback
    because the native Aether CPU engine requires no external packages.

    Never returns ``"pytorch"`` unless torch is explicitly installed AND the
    target is a CUDA/GPU target that would require it. CPU targets use the
    native engine, not PyTorch.

    Args:
        target_id: Hardware target identifier.

    Returns:
        Backend name or ``None`` if the target is invalid or no backend is
        available (callers should emit a warning in that case).
    """
    try:
        target = HardwareTarget.from_string(target_id)
    except ValueError:
        return None
    candidates = target.backend_candidates
    # Check availability (optional imports)
    for backend in candidates:
        try:
            if backend == "aether_cpu":
                # Always available — no external dependency required.
                return backend
            if backend == "pytorch":
                # PyTorch is OPTIONAL. Only recommend it when installed.
                import torch  # noqa: F401,PLC0415
                return backend
            if backend == "vllm":
                import vllm  # noqa: F401,PLC0415
                return backend
            if backend == "llama.cpp":
                import llama_cpp  # noqa: F401,PLC0415
                return backend
            if backend == "mlx":
                import mlx.core  # noqa: F401,PLC0415
                return backend
            if backend == "onnxruntime":
                import onnxruntime  # noqa: F401,PLC0415
                return backend
            if backend == "tensorrt-llm":
                import tensorrt_llm  # noqa: F401,PLC0415
                return backend
        except ImportError:
            continue
    # For any CPU-class target, the native Aether engine is always available.
    if target_id.startswith("cpu_") or target_id == "cpu":
        return "aether_cpu"
    # For GPU targets, no backend is available on this machine — return None
    # so callers can emit an explicit warning rather than silently falling back.
    return None
