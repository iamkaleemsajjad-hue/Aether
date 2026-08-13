"""
Pass 3: Precision Assignment

Uses sensitivity analysis results to assign optimal precision to each layer,
maximizing performance while staying within quality budget.

Supports: FP32, FP16, BF16, FP8 (E4M3/E5M2), FP4, INT8, INT4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether.core.graph import AEGGraph, AEGNode
from aether.compiler.config import CompilerConfig
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PrecisionConfig:
    """Precision configuration for a layer."""
    layer_name: str
    assigned_precision: str
    weight_dtype: str
    activation_dtype: str
    kv_cache_dtype: str | None
    estimated_speedup: float
    estimated_memory_savings_mb: float


# Precision capabilities and tradeoffs
PRECISION_SPECS = {
    "fp32": {
        "bits": 32,
        "speedup": 1.0,
        "memory_factor": 1.0,
        "quality_impact": 0.0,
        "supported_hardware": ["cpu", "cuda", "rocm", "metal"],
    },
    "fp16": {
        "bits": 16,
        "speedup": 2.0,
        "memory_factor": 0.5,
        "quality_impact": 0.001,
        "supported_hardware": ["cpu", "cuda", "rocm", "metal"],
    },
    "bf16": {
        "bits": 16,
        "speedup": 2.0,
        "memory_factor": 0.5,
        "quality_impact": 0.0005,
        "supported_hardware": ["cuda_sm80+", "cpu"],
    },
    "fp8_e4m3": {
        "bits": 8,
        "speedup": 4.0,
        "memory_factor": 0.25,
        "quality_impact": 0.01,
        "supported_hardware": ["cuda_sm89+", "rocm_cdna3+"],
    },
    "fp8_e5m2": {
        "bits": 8,
        "speedup": 4.0,
        "memory_factor": 0.25,
        "quality_impact": 0.008,
        "supported_hardware": ["cuda_sm89+", "rocm_cdna3+"],
    },
    "fp4": {
        "bits": 4,
        "speedup": 8.0,
        "memory_factor": 0.125,
        "quality_impact": 0.05,
        "supported_hardware": ["cuda_sm100+"],  # Blackwell
    },
    "int8": {
        "bits": 8,
        "speedup": 4.0,
        "memory_factor": 0.25,
        "quality_impact": 0.02,
        "supported_hardware": ["cpu", "cuda", "rocm", "metal"],
    },
    "int4": {
        "bits": 4,
        "speedup": 8.0,
        "memory_factor": 0.125,
        "quality_impact": 0.04,
        "supported_hardware": ["cpu", "cuda", "rocm"],
    },
}


class PrecisionAssignmentPass:
    """Pass 3: Precision Assignment - assigns optimal precision per layer."""

    def __init__(self, config: CompilerConfig):
        self.config = config
        self.assignments: dict[str, PrecisionConfig] = {}
        self.target_hardware = self._detect_target_hardware()
        self.quality_budget = config.quality_budget or 0.02

    def run(self, graph: AEGGraph) -> AEGGraph:
        """Apply precision assignment based on sensitivity analysis."""
        if not self.config.enable_precision_assignment:
            logger.info("Precision assignment disabled, skipping")
            return graph

        logger.info("Running Pass 3: Precision Assignment")

        # Get sensitivity analysis results
        sensitivity_data = graph.get_metadata("sensitivity_analysis")
        if not sensitivity_data:
            logger.warning("No sensitivity analysis data found, using default precision")
            return self._apply_default_precision(graph)

        # Assign precision to each layer
        self._assign_layer_precisions(graph, sensitivity_data)

        # Validate quality budget
        if not self._validate_quality_budget(sensitivity_data):
            logger.warning("Quality budget exceeded, adjusting precisions")
            self._adjust_for_quality_budget(graph, sensitivity_data)

        # Apply assignments to graph
        self._apply_assignments_to_graph(graph)

        # Log summary
        self._log_assignment_summary()

        return graph

    def _detect_target_hardware(self) -> list[str]:
        """Detect target hardware from config."""
        targets = self.config.targets or ["auto"]

        if "auto" in targets:
            # Auto-detect available hardware
            detected = []
            try:
                import torch
                if torch.cuda.is_available():
                    # Get CUDA compute capability
                    cap = torch.cuda.get_device_capability()
                    if cap >= (9, 0):
                        detected.append("cuda_sm90")  # H100
                    elif cap >= (8, 9):
                        detected.append("cuda_sm89")  # RTX 4090
                    elif cap >= (8, 0):
                        detected.append("cuda_sm80")  # A100
                    else:
                        detected.append("cuda_sm70")  # V100
            except Exception:
                pass

            if not detected:
                detected.append("cpu")

            logger.info(f"Auto-detected hardware: {detected}")
            return detected

        return targets

    def _assign_layer_precisions(
        self, graph: AEGGraph, sensitivity_data: dict[str, Any]
    ):
        """Assign precision to each layer based on sensitivity."""
        per_layer = sensitivity_data.get("per_layer", {})

        for layer_name, layer_sens in per_layer.items():
            node = graph.get_node(layer_name)
            if not node:
                continue

            # Get recommended precision from sensitivity analysis
            recommended = layer_sens.get("recommended_precision", "fp16")
            sensitivity_score = layer_sens.get("sensitivity_score", 0.0)

            # Adjust based on hardware capabilities
            assigned = self._select_compatible_precision(
                recommended, sensitivity_score
            )

            # Determine activation and KV cache precisions
            weight_dtype, activation_dtype, kv_cache_dtype = self._determine_dtypes(
                assigned, node
            )

            # Calculate benefits
            speedup, memory_savings = self._calculate_benefits(assigned, node)

            self.assignments[layer_name] = PrecisionConfig(
                layer_name=layer_name,
                assigned_precision=assigned,
                weight_dtype=weight_dtype,
                activation_dtype=activation_dtype,
                kv_cache_dtype=kv_cache_dtype,
                estimated_speedup=speedup,
                estimated_memory_savings_mb=memory_savings,
            )

    def _select_compatible_precision(
        self, recommended: str, sensitivity_score: float
    ) -> str:
        """Select precision compatible with target hardware."""
        # Check if recommended precision is supported
        recommended_spec = PRECISION_SPECS.get(recommended)
        if recommended_spec:
            supported = recommended_spec["supported_hardware"]
            for target in self.target_hardware:
                target_base = target.split("_")[0]
                if target_base in supported or any(
                    s.startswith(target_base) for s in supported
                ):
                    return recommended

        # Fall back to next best supported precision
        precision_hierarchy = ["fp4", "int4", "fp8_e4m3", "int8", "fp16", "bf16", "fp32"]
        precision_hierarchy.reverse()  # Start from highest quality

        for prec in precision_hierarchy:
            spec = PRECISION_SPECS[prec]
            # Check hardware support
            supported = spec["supported_hardware"]
            for target in self.target_hardware:
                target_base = target.split("_")[0]
                if target_base in supported or any(
                    s.startswith(target_base) for s in supported
                ):
                    # Check if quality impact is acceptable
                    if spec["quality_impact"] <= self.quality_budget:
                        return prec

        # Ultimate fallback
        return "fp16"

    def _determine_dtypes(
        self, precision: str, node: AEGNode
    ) -> tuple[str, str, str | None]:
        """Determine weight, activation, and KV cache dtypes."""
        # Weight dtype
        if precision.startswith("fp"):
            weight_dtype = f"torch.{precision.replace('_', '.')}"
        elif precision.startswith("int"):
            weight_dtype = f"torch.{precision}"
        else:
            weight_dtype = "torch.float16"

        # Activation dtype (usually higher precision than weights)
        if precision in {"fp4", "int4"}:
            activation_dtype = "torch.float16"
        elif precision in {"fp8_e4m3", "fp8_e5m2", "int8"}:
            activation_dtype = "torch.float16"
        else:
            activation_dtype = weight_dtype

        # KV cache dtype (for attention layers)
        kv_cache_dtype = None
        op_type = getattr(node, "op_type", "").lower()
        if "attention" in op_type or "attn" in op_type:
            # KV cache can be more aggressive
            if precision in {"fp32", "bf16", "fp16"}:
                kv_cache_dtype = "torch.float8_e4m3fn"  # FP8 for KV
            elif precision in {"fp8_e4m3", "fp8_e5m2"}:
                kv_cache_dtype = "torch.float8_e4m3fn"
            elif precision in {"int8", "int4"}:
                kv_cache_dtype = "torch.int8"
            else:
                kv_cache_dtype = activation_dtype

        return weight_dtype, activation_dtype, kv_cache_dtype

    def _calculate_benefits(
        self, precision: str, node: AEGNode
    ) -> tuple[float, float]:
        """Calculate speedup and memory savings."""
        spec = PRECISION_SPECS.get(precision, PRECISION_SPECS["fp16"])

        speedup = spec["speedup"]
        memory_factor = spec["memory_factor"]

        # Estimate layer size
        # This is a rough estimate - real implementation would use actual tensor sizes
        estimated_layer_size_mb = 100.0  # Placeholder

        memory_savings = estimated_layer_size_mb * (1.0 - memory_factor)

        return speedup, memory_savings

    def _validate_quality_budget(self, sensitivity_data: dict[str, Any]) -> bool:
        """Validate that assigned precisions stay within quality budget."""
        baseline_ppl = sensitivity_data.get("baseline_perplexity", 10.0)
        max_allowed_ppl = baseline_ppl * (1 + self.quality_budget)

        # Calculate estimated perplexity with current assignments
        estimated_ppl = self._estimate_overall_perplexity(sensitivity_data)

        logger.info(
            f"Quality validation: baseline={baseline_ppl:.4f}, "
            f"estimated={estimated_ppl:.4f}, "
            f"max_allowed={max_allowed_ppl:.4f}"
        )

        return estimated_ppl <= max_allowed_ppl

    def _estimate_overall_perplexity(
        self, sensitivity_data: dict[str, Any]
    ) -> float:
        """Estimate overall perplexity with current precision assignments."""
        baseline = sensitivity_data.get("baseline_perplexity", 10.0)

        # Accumulate quality impacts from all assigned precisions
        total_impact = 0.0
        for assignment in self.assignments.values():
            spec = PRECISION_SPECS.get(assignment.assigned_precision, {})
            impact = spec.get("quality_impact", 0.0)
            total_impact += impact

        # Average impact across layers
        avg_impact = total_impact / max(len(self.assignments), 1)

        return baseline * (1 + avg_impact)

    def _adjust_for_quality_budget(
        self, graph: AEGGraph, sensitivity_data: dict[str, Any]
    ):
        """Adjust precision assignments to meet quality budget."""
        # Sort layers by sensitivity (most sensitive first)
        per_layer = sensitivity_data.get("per_layer", {})
        sorted_layers = sorted(
            per_layer.items(),
            key=lambda x: x[1].get("sensitivity_score", 0.0),
            reverse=True,
        )

        # Upgrade most sensitive layers to higher precision
        for layer_name, _ in sorted_layers:
            if layer_name not in self.assignments:
                continue

            assignment = self.assignments[layer_name]
            current_prec = assignment.assigned_precision

            # Try upgrading to next higher precision
            upgrade = self._get_higher_precision(current_prec)
            if upgrade and upgrade != current_prec:
                # Temporarily upgrade and check
                old_prec = assignment.assigned_precision
                assignment.assigned_precision = upgrade

                # Check if we now meet quality budget
                if self._validate_quality_budget(sensitivity_data):
                    logger.info(f"Upgraded {layer_name}: {old_prec} -> {upgrade}")
                    break
                else:
                    # Revert if not enough
                    assignment.assigned_precision = old_prec

    def _get_higher_precision(self, current: str) -> str | None:
        """Get next higher precision level."""
        hierarchy = {
            "int4": "int8",
            "fp4": "fp8_e4m3",
            "int8": "fp8_e4m3",
            "fp8_e5m2": "fp16",
            "fp8_e4m3": "fp16",
            "fp16": "bf16",
            "bf16": "fp32",
            "fp32": None,  # Already highest
        }
        return hierarchy.get(current)

    def _apply_default_precision(self, graph: AEGGraph) -> AEGGraph:
        """Apply default precision when no sensitivity data available."""
        logger.info("Applying default FP16 precision to all layers")

        for node in graph.get_nodes():
            op_type = getattr(node, "op_type", "").lower()
            if any(qt in op_type for qt in ["linear", "matmul", "conv", "attention"]):
                node.set_metadata("weight_dtype", "torch.float16")
                node.set_metadata("activation_dtype", "torch.float16")

        return graph

    def _apply_assignments_to_graph(self, graph: AEGGraph):
        """Apply precision assignments to graph nodes."""
        for layer_name, assignment in self.assignments.items():
            node = graph.get_node(layer_name)
            if not node:
                continue

            # Set precision metadata
            node.set_metadata("assigned_precision", assignment.assigned_precision)
            node.set_metadata("weight_dtype", assignment.weight_dtype)
            node.set_metadata("activation_dtype", assignment.activation_dtype)

            if assignment.kv_cache_dtype:
                node.set_metadata("kv_cache_dtype", assignment.kv_cache_dtype)

            node.set_metadata("precision_speedup", assignment.estimated_speedup)
            node.set_metadata(
                "precision_memory_savings_mb", assignment.estimated_memory_savings_mb
            )

        # Store overall assignment summary in graph
        graph.set_metadata("precision_assignment", {
            "quality_budget": self.quality_budget,
            "target_hardware": self.target_hardware,
            "per_layer": {
                name: {
                    "precision": a.assigned_precision,
                    "weight_dtype": a.weight_dtype,
                    "activation_dtype": a.activation_dtype,
                    "speedup": a.estimated_speedup,
                    "memory_savings_mb": a.estimated_memory_savings_mb,
                }
                for name, a in self.assignments.items()
            },
        })

    def _log_assignment_summary(self):
        """Log summary of precision assignments."""
        if not self.assignments:
            return

        # Count assignments by precision
        precision_counts = {}
        total_speedup = 1.0
        total_memory_savings = 0.0

        for assignment in self.assignments.values():
            prec = assignment.assigned_precision
            precision_counts[prec] = precision_counts.get(prec, 0) + 1
            total_speedup *= assignment.estimated_speedup
            total_memory_savings += assignment.estimated_memory_savings_mb

        # Calculate geometric mean speedup
        geom_mean_speedup = total_speedup ** (1.0 / len(self.assignments))

        logger.info(
            "Precision assignment summary",
            total_layers=len(self.assignments),
            precision_distribution=precision_counts,
            estimated_speedup=f"{geom_mean_speedup:.2f}x",
            memory_savings_mb=f"{total_memory_savings:.1f}MB",
            quality_budget=f"{self.quality_budget*100:.1f}%",
        )


def apply_precision_assignment(graph: AEGGraph, config: CompilerConfig) -> AEGGraph:
    """Convenience function to apply precision assignment pass."""
    pass_instance = PrecisionAssignmentPass(config)
    return pass_instance.run(graph)
