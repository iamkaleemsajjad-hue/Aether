"""
Compiler configuration for Aether.

Defines the `CompilerConfig` class and all configuration options that control the
five compiler stages: ingestion, optimization (six passes), hardware targeting, and
quality reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CALIBRATION_TOKENS,
    DEFAULT_FUSION_PASS,
    DEFAULT_KV_CACHE_PASS,
    DEFAULT_MOE_ROUTING_PASS,
    DEFAULT_OPTIMIZATION_LEVEL,
    DEFAULT_PARALLELISM_PASS,
    DEFAULT_PRECISION_PASS,
    DEFAULT_PRUNING_PASS,
    DEFAULT_QUALITY_BUDGET,
    DEFAULT_REASONING_GRAPH_PASS,
    DEFAULT_SENSITIVITY_PASS,
    DEFAULT_SPARSE_ATTENTION_PASS,
)
from aether.core.exceptions import CompilerConfigError
from aether.core.types import HardwareTarget


@dataclass
class CompilerConfig:
    """Configuration for the Aether compiler.

    All fields have sensible defaults and can be overridden programmatically or
    via the CLI / environment variables.
    """

    quality_budget: float = DEFAULT_QUALITY_BUDGET
    """Maximum allowed perplexity increase relative to BF16 baseline (0.02 = 2%)."""

    calibration_dataset: str = DEFAULT_CALIBRATION_DATASET
    """Dataset name used for sensitivity analysis and calibration."""

    calibration_tokens: int = DEFAULT_CALIBRATION_TOKENS
    """Number of tokens to draw from the calibration dataset."""

    targets: list[str] = field(default_factory=lambda: ["auto"])
    """Target hardware identifiers. 'auto' means detect current hardware."""

    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL
    """Optimization level: 0=none, 1=basic, 2=full, 3=aggressive."""

    enable_fusion: bool = DEFAULT_FUSION_PASS
    """Enable operator fusion pass (Pass 1)."""

    enable_sensitivity: bool = DEFAULT_SENSITIVITY_PASS
    """Enable sensitivity analysis pass (Pass 2)."""

    enable_precision_assignment: bool = DEFAULT_PRECISION_PASS
    """Enable precision assignment pass (Pass 3)."""

    enable_kv_cache_structuring: bool = DEFAULT_KV_CACHE_PASS
    """Enable KV cache structuring pass (Pass 4)."""

    enable_moe_routing: bool = DEFAULT_MOE_ROUTING_PASS
    """Enable MoE routing pass (Pass 5)."""

    enable_parallelism_discovery: bool = DEFAULT_PARALLELISM_PASS
    """Enable automatic parallelism discovery pass (Pass 6)."""

    enable_reasoning_graph: bool = DEFAULT_REASONING_GRAPH_PASS
    """Enable reasoning graph compilation pass (Pass 7)."""

    enable_sparse_attention: bool = DEFAULT_SPARSE_ATTENTION_PASS
    """Enable sparse attention pattern compilation pass (Pass 8)."""

    enable_pruning: bool = DEFAULT_PRUNING_PASS
    """Enable pruning and sparsity planning pass (Pass 9)."""

    reasoning_budget_tokens: int = 512
    """Default token budget for compiled reasoning graph nodes."""

    sparse_attention_context_threshold: int = 32768
    """Context length where MInference-style sparse attention plans activate."""

    pruning_target_sparsity: float = 0.5
    """Target sparsity for Wanda/SparseGPT-style mask planning."""

    upload_kernels: bool = False
    """Opt-in to upload compiled kernels to Aether Hub after compilation."""

    cache_dir: str = DEFAULT_CACHE_DIR
    """Directory for Aether's local cache."""

    hub_url: str | None = None
    """Aether Hub URL. None means use the default."""

    max_calibration_samples: int = 2048
    """Maximum number of sequences to use for calibration."""

    min_layer_samples: int = 128
    """Minimum number of tokens per layer for sensitivity estimation."""

    precision_assignment_mode: str = "sensitivity"
    """How to assign precision: 'sensitivity', 'uniform', 'manual'."""

    manual_precision_map: dict[str, str] = field(default_factory=dict)
    """If mode='manual', per-layer precision overrides."""

    sensitivity_bits_candidates: list[int] = field(default_factory=lambda: [4, 6, 8, 16])
    """Bit widths to evaluate during sensitivity analysis."""

    kv_cache_dtype: str = "fp8"
    """KV cache numeric format: 'fp8', 'fp16', 'bf16'."""

    kv_cache_cpu_gb: int = 32
    """CPU DRAM budget for KV cache (GB)."""

    kv_cache_nvme_gb: int = 200
    """NVMe SSD budget for KV cache (GB)."""

    max_prefill_chunk_size: int = 2048
    """Maximum prefill chunk size for KV cache pass."""

    moe_hot_threshold: float = 0.05
    """Activation threshold for hot experts."""

    moe_warm_threshold: float = 0.001
    """Activation threshold for warm experts."""

    parallelism_degrees: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    """GPU counts to evaluate for automatic parallelism discovery."""

    skip_download: bool = False
    """If True, do not download model weights (use local path only)."""

    output_format: str = "aeg"
    """Output format: 'aeg' or 'aeg-ir'."""

    overwrite: bool = False
    """If True, overwrite an existing AEG package."""

    dry_run: bool = False
    """If True, only plan compilation without producing an AEG."""

    verbose: bool = False
    """Enable verbose compiler output."""

    def __post_init__(self) -> None:
        """Validate the configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values and raise CompilerConfigError on failure."""
        if self.quality_budget < 0.0 or self.quality_budget > 1.0:
            msg = f"quality_budget must be in [0, 1], got {self.quality_budget}"
            raise CompilerConfigError(msg)
        if self.optimization_level < 0 or self.optimization_level > 3:
            msg = f"optimization_level must be 0-3, got {self.optimization_level}"
            raise CompilerConfigError(msg)
        if self.calibration_tokens < 0:
            msg = f"calibration_tokens must be >= 0, got {self.calibration_tokens}"
            raise CompilerConfigError(msg)
        if self.max_calibration_samples < 1:
            msg = f"max_calibration_samples must be >= 1, got {self.max_calibration_samples}"
            raise CompilerConfigError(msg)
        if self.precision_assignment_mode not in ("sensitivity", "uniform", "manual"):
            msg = f"Unknown precision_assignment_mode: {self.precision_assignment_mode}"
            raise CompilerConfigError(msg)
        if self.kv_cache_dtype not in ("fp8", "fp16", "bf16"):
            msg = f"Unknown kv_cache_dtype: {self.kv_cache_dtype}"
            raise CompilerConfigError(msg)
        if self.reasoning_budget_tokens < 1:
            msg = f"reasoning_budget_tokens must be >= 1, got {self.reasoning_budget_tokens}"
            raise CompilerConfigError(msg)
        if not 0.0 <= self.pruning_target_sparsity < 1.0:
            msg = f"pruning_target_sparsity must be in [0, 1), got {self.pruning_target_sparsity}"
            raise CompilerConfigError(msg)
        for target in self.targets:
            if target == "auto":
                continue
            try:
                HardwareTarget.from_string(target)
            except ValueError:
                from aether.core.constants import SUPPORTED_TARGET_IDS

                msg = f"Unknown target: {target}. Supported: {sorted(SUPPORTED_TARGET_IDS)}"
                raise CompilerConfigError(msg) from None

    def get_targets(self) -> list[str]:
        """Resolve 'auto' targets to the current hardware target."""
        from aether.core.types import HardwareTarget

        resolved: list[str] = []
        for target in self.targets:
            if target == "auto":
                resolved.append(HardwareTarget.auto().value)
            else:
                resolved.append(target)
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in resolved:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "quality_budget": self.quality_budget,
            "calibration_dataset": self.calibration_dataset,
            "calibration_tokens": self.calibration_tokens,
            "targets": self.targets,
            "optimization_level": self.optimization_level,
            "enable_fusion": self.enable_fusion,
            "enable_sensitivity": self.enable_sensitivity,
            "enable_precision_assignment": self.enable_precision_assignment,
            "enable_kv_cache_structuring": self.enable_kv_cache_structuring,
            "enable_moe_routing": self.enable_moe_routing,
            "enable_parallelism_discovery": self.enable_parallelism_discovery,
            "enable_reasoning_graph": self.enable_reasoning_graph,
            "enable_sparse_attention": self.enable_sparse_attention,
            "enable_pruning": self.enable_pruning,
            "reasoning_budget_tokens": self.reasoning_budget_tokens,
            "sparse_attention_context_threshold": self.sparse_attention_context_threshold,
            "pruning_target_sparsity": self.pruning_target_sparsity,
            "upload_kernels": self.upload_kernels,
            "cache_dir": self.cache_dir,
            "hub_url": self.hub_url,
            "max_calibration_samples": self.max_calibration_samples,
            "min_layer_samples": self.min_layer_samples,
            "precision_assignment_mode": self.precision_assignment_mode,
            "manual_precision_map": self.manual_precision_map,
            "sensitivity_bits_candidates": self.sensitivity_bits_candidates,
            "kv_cache_dtype": self.kv_cache_dtype,
            "kv_cache_cpu_gb": self.kv_cache_cpu_gb,
            "kv_cache_nvme_gb": self.kv_cache_nvme_gb,
            "max_prefill_chunk_size": self.max_prefill_chunk_size,
            "moe_hot_threshold": self.moe_hot_threshold,
            "moe_warm_threshold": self.moe_warm_threshold,
            "parallelism_degrees": self.parallelism_degrees,
            "skip_download": self.skip_download,
            "output_format": self.output_format,
            "overwrite": self.overwrite,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CompilerConfig:
        """Deserialize configuration from a dictionary."""
        return CompilerConfig(
            quality_budget=data.get("quality_budget", DEFAULT_QUALITY_BUDGET),
            calibration_dataset=data.get("calibration_dataset", DEFAULT_CALIBRATION_DATASET),
            calibration_tokens=data.get("calibration_tokens", DEFAULT_CALIBRATION_TOKENS),
            targets=list(data.get("targets", ["auto"])),
            optimization_level=data.get("optimization_level", DEFAULT_OPTIMIZATION_LEVEL),
            enable_fusion=data.get("enable_fusion", DEFAULT_FUSION_PASS),
            enable_sensitivity=data.get("enable_sensitivity", DEFAULT_SENSITIVITY_PASS),
            enable_precision_assignment=data.get("enable_precision_assignment", DEFAULT_PRECISION_PASS),
            enable_kv_cache_structuring=data.get("enable_kv_cache_structuring", DEFAULT_KV_CACHE_PASS),
            enable_moe_routing=data.get("enable_moe_routing", DEFAULT_MOE_ROUTING_PASS),
            enable_parallelism_discovery=data.get("enable_parallelism_discovery", DEFAULT_PARALLELISM_PASS),
            enable_reasoning_graph=data.get("enable_reasoning_graph", DEFAULT_REASONING_GRAPH_PASS),
            enable_sparse_attention=data.get("enable_sparse_attention", DEFAULT_SPARSE_ATTENTION_PASS),
            enable_pruning=data.get("enable_pruning", DEFAULT_PRUNING_PASS),
            reasoning_budget_tokens=data.get("reasoning_budget_tokens", 512),
            sparse_attention_context_threshold=data.get("sparse_attention_context_threshold", 32768),
            pruning_target_sparsity=data.get("pruning_target_sparsity", 0.5),
            upload_kernels=data.get("upload_kernels", False),
            cache_dir=data.get("cache_dir", DEFAULT_CACHE_DIR),
            hub_url=data.get("hub_url"),
            max_calibration_samples=data.get("max_calibration_samples", 2048),
            min_layer_samples=data.get("min_layer_samples", 128),
            precision_assignment_mode=data.get("precision_assignment_mode", "sensitivity"),
            manual_precision_map=dict(data.get("manual_precision_map", {})),
            sensitivity_bits_candidates=list(data.get("sensitivity_bits_candidates", [4, 6, 8, 16])),
            kv_cache_dtype=data.get("kv_cache_dtype", "fp8"),
            kv_cache_cpu_gb=data.get("kv_cache_cpu_gb", 32),
            kv_cache_nvme_gb=data.get("kv_cache_nvme_gb", 200),
            max_prefill_chunk_size=data.get("max_prefill_chunk_size", 2048),
            moe_hot_threshold=data.get("moe_hot_threshold", 0.05),
            moe_warm_threshold=data.get("moe_warm_threshold", 0.001),
            parallelism_degrees=list(data.get("parallelism_degrees", [1, 2, 4, 8])),
            skip_download=data.get("skip_download", False),
            output_format=data.get("output_format", "aeg"),
            overwrite=data.get("overwrite", False),
            dry_run=data.get("dry_run", False),
            verbose=data.get("verbose", False),
        )

    @staticmethod
    def from_env() -> CompilerConfig:
        """Load compiler configuration from environment variables.

        Supported environment variables:
            AETHER_QUALITY_BUDGET
            AETHER_CALIBRATION_DATASET
            AETHER_CALIBRATION_TOKENS
            AETHER_TARGETS
            AETHER_OPTIMIZATION_LEVEL
            AETHER_UPLOAD_KERNELS
            AETHER_CACHE_DIR
            AETHER_DRY_RUN
            AETHER_VERBOSE
        """
        import os

        config = CompilerConfig()
        if "AETHER_QUALITY_BUDGET" in os.environ:
            config.quality_budget = float(os.environ["AETHER_QUALITY_BUDGET"])
        if "AETHER_CALIBRATION_DATASET" in os.environ:
            config.calibration_dataset = os.environ["AETHER_CALIBRATION_DATASET"]
        if "AETHER_CALIBRATION_TOKENS" in os.environ:
            config.calibration_tokens = int(os.environ["AETHER_CALIBRATION_TOKENS"])
        if "AETHER_TARGETS" in os.environ:
            config.targets = [t.strip() for t in os.environ["AETHER_TARGETS"].split(",")]
        if "AETHER_OPTIMIZATION_LEVEL" in os.environ:
            config.optimization_level = int(os.environ["AETHER_OPTIMIZATION_LEVEL"])
        if "AETHER_UPLOAD_KERNELS" in os.environ:
            config.upload_kernels = os.environ["AETHER_UPLOAD_KERNELS"].lower() in ("1", "true", "yes")
        if "AETHER_CACHE_DIR" in os.environ:
            config.cache_dir = os.environ["AETHER_CACHE_DIR"]
        if "AETHER_DRY_RUN" in os.environ:
            config.dry_run = os.environ["AETHER_DRY_RUN"].lower() in ("1", "true", "yes")
        if "AETHER_VERBOSE" in os.environ:
            config.verbose = os.environ["AETHER_VERBOSE"].lower() in ("1", "true", "yes")
        config.validate()
        return config

    def clone(self) -> CompilerConfig:
        """Return a deep copy of this configuration."""
        return CompilerConfig.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"CompilerConfig(level={self.optimization_level}, "
            f"budget={self.quality_budget}, targets={self.targets})"
        )
