"""
Quality and compilation reports for Aether.

The `QualityReport` class captures the measured impact of Aether's optimizer
passes, including perplexity change, precision distribution, fusion summary,
kernel targeting, and memory requirements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aether.core.aeg_format import MemoryRequirements
from aether.core.types import ModelArchitecture, Precision


@dataclass
class PassReport:
    """Report for a single optimizer pass."""

    pass_name: str
    """Name of the pass."""

    status: str
    """Status: 'applied', 'skipped', 'failed'."""

    duration_ms: float = 0.0
    """Time spent in the pass in milliseconds."""

    nodes_affected: int = 0
    """Number of graph nodes affected."""

    details: dict[str, Any] = field(default_factory=dict)
    """Pass-specific details."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_name": self.pass_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "nodes_affected": self.nodes_affected,
            "details": self.details,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PassReport:
        return PassReport(
            pass_name=data["pass_name"],
            status=data["status"],
            duration_ms=data.get("duration_ms", 0.0),
            nodes_affected=data.get("nodes_affected", 0),
            details=dict(data.get("details", {})),
        )


@dataclass
class FusionSummary:
    """Summary of operator fusion results."""

    fused_op_count: int
    """Number of fused operations produced."""

    original_op_count: int
    """Original number of operations before fusion."""

    memory_round_trips_saved: int
    """Estimated number of intermediate DRAM round-trips saved."""

    fusion_patterns: dict[str, int] = field(default_factory=dict)
    """Histogram of fusion patterns applied."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fused_op_count": self.fused_op_count,
            "original_op_count": self.original_op_count,
            "memory_round_trips_saved": self.memory_round_trips_saved,
            "fusion_patterns": self.fusion_patterns,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> FusionSummary:
        return FusionSummary(
            fused_op_count=data["fused_op_count"],
            original_op_count=data["original_op_count"],
            memory_round_trips_saved=data["memory_round_trips_saved"],
            fusion_patterns=dict(data.get("fusion_patterns", {})),
        )


@dataclass
class PrecisionSummary:
    """Summary of precision assignment results."""

    distribution: dict[str, int]
    """Count of layers per precision format."""

    average_bit_width: float
    """Average bit width across all layers."""

    baseline_ppl: float | None = None
    """Perplexity at full BF16."""

    quantized_ppl: float | None = None
    """Perplexity after quantization."""

    ppl_increase: float | None = None
    """Measured perplexity increase."""

    quality_budget: float = 0.02
    """Configured quality budget."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "average_bit_width": self.average_bit_width,
            "baseline_ppl": self.baseline_ppl,
            "quantized_ppl": self.quantized_ppl,
            "ppl_increase": self.ppl_increase,
            "quality_budget": self.quality_budget,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PrecisionSummary:
        return PrecisionSummary(
            distribution=dict(data["distribution"]),
            average_bit_width=data["average_bit_width"],
            baseline_ppl=data.get("baseline_ppl"),
            quantized_ppl=data.get("quantized_ppl"),
            ppl_increase=data.get("ppl_increase"),
            quality_budget=data.get("quality_budget", 0.02),
        )


@dataclass
class QualityReport:
    """Compilation quality report.

    Returned by `compiler.compile()` and by `aeg.quality_report()`. It captures
    all measurable outcomes of the compiler pipeline.
    """

    model_id: str
    """Model identifier."""

    architecture: ModelArchitecture | None = None
    """Detected architecture."""

    passes: list[PassReport] = field(default_factory=list)
    """Per-pass reports."""

    fusion: FusionSummary | None = None
    """Fusion summary."""

    precision: PrecisionSummary | None = None
    """Precision summary."""

    memory: MemoryRequirements | None = None
    """Memory requirements."""

    targets: list[str] = field(default_factory=list)
    """Hardware targets compiled."""

    backend_recommendations: dict[str, str] = field(default_factory=dict)
    """Recommended backend per target."""

    warnings: list[str] = field(default_factory=list)
    """Quality warnings."""

    errors: list[str] = field(default_factory=list)
    """Quality errors."""

    def add_pass(self, pass_report: PassReport) -> None:
        """Add a pass report."""
        self.passes.append(pass_report)

    def add_warning(self, message: str) -> None:
        """Add a warning."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error."""
        self.errors.append(message)

    @property
    def is_success(self) -> bool:
        """Return True if there are no errors."""
        return len(self.errors) == 0

    @property
    def total_duration_ms(self) -> float:
        """Return total duration across all passes."""
        return sum(p.duration_ms for p in self.passes)

    @property
    def precision_distribution_pct(self) -> dict[str, float]:
        """Return precision distribution as percentages."""
        if not self.precision:
            return {}
        total = sum(self.precision.distribution.values())
        if total == 0:
            return {}
        return {k: f"{v / total * 100:.1f}%" for k, v in self.precision.distribution.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture.to_dict() if self.architecture else None,
            "passes": [p.to_dict() for p in self.passes],
            "fusion": self.fusion.to_dict() if self.fusion else None,
            "precision": self.precision.to_dict() if self.precision else None,
            "memory": self.memory.to_dict() if self.memory else None,
            "targets": self.targets,
            "backend_recommendations": self.backend_recommendations,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_text(self) -> str:
        """Return a human-readable text report."""
        lines = [f"Quality Report for {self.model_id}", "=" * 60]
        if self.architecture:
            lines.append(f"Architecture: {self.architecture.family} ({self.architecture.params_billion}B params)")
        lines.append(f"Total duration: {self.total_duration_ms:.1f} ms")
        if self.fusion:
            lines.append(
                f"Fusion: {self.fusion.original_op_count} -> {self.fusion.fused_op_count} ops, "
                f"{self.fusion.memory_round_trips_saved} round-trips saved"
            )
        if self.precision:
            lines.append(f"Precision: avg {self.precision.average_bit_width:.1f} bits, distribution {self.precision.distribution}")
            if self.precision.ppl_increase is not None:
                lines.append(f"Perplexity increase: {self.precision.ppl_increase:.4f} (budget: {self.precision.quality_budget})")
        if self.memory:
            lines.append(
                f"Memory: BF16={self.memory.bf16_gb:.1f}GB, compiled={self.memory.compiled_min_gb:.1f}GB, "
                f"recommended={self.memory.recommended_gb:.1f}GB"
            )
        lines.append(f"Targets: {', '.join(self.targets) or 'none'}")
        for target, backend in self.backend_recommendations.items():
            lines.append(f"  {target} -> {backend}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> QualityReport:
        arch = data.get("architecture")
        fusion = data.get("fusion")
        precision = data.get("precision")
        memory = data.get("memory")
        return QualityReport(
            model_id=data["model_id"],
            architecture=ModelArchitecture.from_dict(arch) if arch else None,
            passes=[PassReport.from_dict(p) for p in data.get("passes", [])],
            fusion=FusionSummary.from_dict(fusion) if fusion else None,
            precision=PrecisionSummary.from_dict(precision) if precision else None,
            memory=MemoryRequirements.from_dict(memory) if memory else None,
            targets=list(data.get("targets", [])),
            backend_recommendations=dict(data.get("backend_recommendations", {})),
            warnings=list(data.get("warnings", [])),
            errors=list(data.get("errors", [])),
        )

    @staticmethod
    def from_json(json_str: str) -> QualityReport:
        return QualityReport.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"QualityReport({self.model_id}, success={self.is_success}, passes={len(self.passes)})"


def create_empty_quality_report(model_id: str) -> QualityReport:
    """Create an empty quality report for a model."""
    return QualityReport(model_id=model_id)


def compute_precision_distribution(precision_map: dict[str, str]) -> dict[str, int]:
    """Compute a histogram of precision formats from a precision map."""
    distribution: dict[str, int] = {}
    for precision in precision_map.values():
        distribution[precision] = distribution.get(precision, 0) + 1
    return distribution


def compute_average_bit_width(precision_map: dict[str, str]) -> float:
    """Compute the average bit width across a precision map."""
    if not precision_map:
        return 16.0
    total = 0.0
    for precision in precision_map.values():
        total += Precision.from_string(precision).bit_width
    return total / len(precision_map)
