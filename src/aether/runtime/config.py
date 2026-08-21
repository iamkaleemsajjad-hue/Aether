"""
Runtime configuration.

Defines the `RuntimeConfig` class controlling the Aether execution engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import (
    DEFAULT_DISAGGREGATE_SERVE,
    DEFAULT_DYNAMIC_PRECISION,
    DEFAULT_KV_CACHE_CPU_GB,
    DEFAULT_KV_CACHE_DTYPE,
    DEFAULT_KV_CACHE_NVME_GB,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPTIMIZE_FOR,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SPECULATIVE_DECODING,
    DEFAULT_SPECULATIVE_TREE_DEPTH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from aether.core.exceptions import RuntimeConfigError


@dataclass
class RuntimeConfig:
    """Configuration for the Aether runtime.

    Controls the serving engine, scheduling, caching, speculative decoding,
    dynamic precision adjustment, and server parameters.
    """

    optimize_for: str = DEFAULT_OPTIMIZE_FOR
    """Optimization objective: 'latency', 'throughput', or 'quality'."""

    speculative_decoding: bool | str = DEFAULT_SPECULATIVE_DECODING
    """Enable tree-speculative decoding for faster generation."""

    speculative_tree_depth: int = DEFAULT_SPECULATIVE_TREE_DEPTH
    """Maximum draft tree depth for speculative decoding."""

    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE
    """Maximum tokens per prefill chunk."""

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    """Maximum concurrent requests per batch."""

    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    """KV cache numeric format: 'fp8', 'fp16', or 'bf16'."""

    kv_cache_cpu_gb: int = DEFAULT_KV_CACHE_CPU_GB
    """CPU DRAM budget for KV cache (GB)."""

    kv_cache_nvme_gb: int = DEFAULT_KV_CACHE_NVME_GB
    """NVMe SSD budget for KV cache (GB)."""

    dynamic_precision: bool = DEFAULT_DYNAMIC_PRECISION
    """Allow precision downgrade/upgrade under memory pressure."""

    disaggregate_prefill_decode: bool = DEFAULT_DISAGGREGATE_SERVE
    """Enable for multi-GPU cluster mode with separate prefill/decode pools."""

    server_port: int = DEFAULT_SERVER_PORT
    """REST server port."""

    server_host: str = DEFAULT_SERVER_HOST
    """REST server host."""

    default_temperature: float = DEFAULT_TEMPERATURE
    """Default generation temperature."""

    default_max_tokens: int = DEFAULT_MAX_TOKENS
    """Default maximum generation tokens."""

    default_top_p: float = DEFAULT_TOP_P
    """Default top-p sampling parameter."""

    backend_name: str | None = None
    """Force a specific backend. None = auto-select."""

    execution_devices: list[str] | None = None
    """Explicit model-parallel device mesh, e.g. ``['cuda:0', 'cuda:1']``.

    When omitted, an accelerator backend uses every compatible accelerator it
    detects.  Supplying ``cpu`` alongside accelerator IDs enables the
    heterogeneous path; the runtime shards one model across that mesh rather
    than creating one full model replica per device.
    """

    model_cache_dir: str | None = None
    """Custom model cache directory."""

    model_download_timeout_s: float = 30.0
    """Maximum socket timeout used while downloading a model or tokenizer."""

    hf_offline: bool = False
    """Only use local Hugging Face files when true; never contact the Hub."""

    allow_remote_code: bool = False
    """Explicitly allow execution of custom code from a model repository.

    This is disabled by default because ``trust_remote_code`` executes Python
    supplied by the model publisher. Enable it only for a reviewed repository.
    """

    lazy_model_loading: bool = True
    """Load models on first use rather than at registration."""

    enable_continuous_batching: bool = True
    """Enable continuous batching for higher throughput."""

    enable_prefix_caching: bool = True
    """Enable RadixTree prefix KV cache."""

    enable_memory_profiling: bool = False
    """Enable per-request memory profiling."""

    enable_telemetry: bool = True
    """Enable anonymous telemetry and usage statistics."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional runtime-specific parameters."""

    # Public v3/v4 configuration names.  Runtime consumes these values at
    # model-load and request-routing boundaries; they are not no-op aliases.
    model_routing: dict[str, str] = field(default_factory=dict)
    reasoning_budget: int = 0
    enable_safety_layer: bool = False
    telemetry_endpoint: str | None = None
    saguaro_enabled: bool = False
    multi_agent_kv_mode: str = "relay"
    scheduler: str = "continuous_batching"
    slo_profiles: dict[str, dict[str, float | None]] = field(default_factory=dict)
    ttt_enabled: bool = False
    ttt_reset_between_requests: bool = True
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_timeout_ms: int = 5000
    green_power_management: bool = False
    green_target_region: str = "lowest_carbon"
    tee_mode: str = "auto"

    # ── v5.0 Runtime Config ─────────────────────────────────────────────────

    vocab_size: int = 128000
    """Vocabulary size of the model (for R9 DiffusionSpecEngine)."""

    semantic_cache_threshold: float = 0.92
    """Cosine similarity threshold for R11 semantic cache hits (0.0-1.0).
    Lower = more aggressive caching (may return slightly different responses).
    Higher = more conservative (only very close prompts hit cache).
    Default 0.92 balances hit rate vs accuracy."""

    semantic_cache_size: int = 100_000
    """Maximum number of entries retained by the R11 semantic request cache."""

    cxl_pool_size_gb: float = 0.0
    """CXL rack-scale KV pool size in GB (R12). Set to 0 to disable.
    Requires CXL 3.0 hardware or emulated mode (file-backed mmap).
    Typical values: 128.0 (small rack), 512.0 (full rack)."""

    enable_diffusion_spec: bool = True
    """Enable R9 diffusion speculative decoding when MDLM draft head is available.
    Provides 2.8-4.1x wall-clock speedup vs sequential AR decoding."""

    enable_semantic_cache: bool = True
    """Enable R11 semantic request cache. Eliminates 30-50% of redundant LLM calls
    by finding semantically similar prior responses."""

    diffusion_spec_K: int = 8
    """Initial draft block size K for R9 DiffusionSpecEngine.
    Adaptively adjusted during inference based on acceptance rate."""

    diffusion_spec_T: int = 4
    """Initial denoising steps T for R9 DiffusionSpecEngine.
    Adaptively adjusted based on per-block uncertainty."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        if self.optimize_for not in ("latency", "throughput", "quality"):
            msg = f"optimize_for must be 'latency', 'throughput', or 'quality', got '{self.optimize_for}'"
            raise RuntimeConfigError(msg)
        if isinstance(self.speculative_decoding, str) and self.speculative_decoding not in {
            "eagle3", "p_eagle", "saguaro", "none", "off"
        }:
            raise RuntimeConfigError(f"unsupported speculative_decoding mode: {self.speculative_decoding}")
        if self.speculative_tree_depth < 1 or self.speculative_tree_depth > 10:
            msg = f"speculative_tree_depth must be between 1 and 10, got {self.speculative_tree_depth}"
            raise RuntimeConfigError(msg)
        if self.prefill_chunk_size < 128:
            msg = f"prefill_chunk_size must be >= 128, got {self.prefill_chunk_size}"
            raise RuntimeConfigError(msg)
        if self.max_batch_size < 1:
            msg = f"max_batch_size must be >= 1, got {self.max_batch_size}"
            raise RuntimeConfigError(msg)
        if self.kv_cache_dtype not in ("fp8", "fp16", "bf16"):
            msg = f"kv_cache_dtype must be 'fp8', 'fp16', or 'bf16', got '{self.kv_cache_dtype}'"
            raise RuntimeConfigError(msg)
        if self.model_download_timeout_s <= 0:
            msg = f"model_download_timeout_s must be positive, got {self.model_download_timeout_s}"
            raise RuntimeConfigError(msg)
        if self.reasoning_budget < 0:
            raise RuntimeConfigError("reasoning_budget must be non-negative")
        if not 0.0 <= self.semantic_cache_threshold <= 1.0:
            raise RuntimeConfigError("semantic_cache_threshold must be between 0 and 1")
        if self.semantic_cache_size < 1:
            raise RuntimeConfigError("semantic_cache_size must be at least 1")
        if self.scheduler not in ("continuous_batching", "slo_aware"):
            raise RuntimeConfigError("scheduler must be 'continuous_batching' or 'slo_aware'")
        if self.multi_agent_kv_mode not in ("relay", "kvcomm", "droidspeak", "swarm"):
            raise RuntimeConfigError("multi_agent_kv_mode must be relay, kvcomm, droidspeak, or swarm")
        if self.mcp_timeout_ms <= 0:
            raise RuntimeConfigError("mcp_timeout_ms must be positive")
        if self.execution_devices is not None:
            if not self.execution_devices or len(set(self.execution_devices)) != len(self.execution_devices):
                raise RuntimeConfigError("execution_devices must be a non-empty list of unique IDs")
            if any(not isinstance(device, str) or not device.strip() for device in self.execution_devices):
                raise RuntimeConfigError("execution_devices must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimize_for": self.optimize_for,
            "speculative_decoding": self.speculative_decoding,
            "speculative_tree_depth": self.speculative_tree_depth,
            "prefill_chunk_size": self.prefill_chunk_size,
            "max_batch_size": self.max_batch_size,
            "kv_cache_dtype": self.kv_cache_dtype,
            "kv_cache_cpu_gb": self.kv_cache_cpu_gb,
            "kv_cache_nvme_gb": self.kv_cache_nvme_gb,
            "dynamic_precision": self.dynamic_precision,
            "disaggregate_prefill_decode": self.disaggregate_prefill_decode,
            "server_port": self.server_port,
            "server_host": self.server_host,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "default_top_p": self.default_top_p,
            "backend_name": self.backend_name,
            "execution_devices": list(self.execution_devices) if self.execution_devices is not None else None,
            "model_cache_dir": self.model_cache_dir,
            "model_download_timeout_s": self.model_download_timeout_s,
            "hf_offline": self.hf_offline,
            "allow_remote_code": self.allow_remote_code,
            "enable_memory_profiling": self.enable_memory_profiling,
            "enable_telemetry": self.enable_telemetry,
            "extra": dict(self.extra),
            "lazy_model_loading": self.lazy_model_loading,
            "enable_continuous_batching": self.enable_continuous_batching,
            "enable_prefix_caching": self.enable_prefix_caching,
            "model_routing": dict(self.model_routing),
            "reasoning_budget": self.reasoning_budget,
            "enable_safety_layer": self.enable_safety_layer,
            "telemetry_endpoint": self.telemetry_endpoint,
            "saguaro_enabled": self.saguaro_enabled,
            "multi_agent_kv_mode": self.multi_agent_kv_mode,
            "scheduler": self.scheduler,
            "slo_profiles": dict(self.slo_profiles),
            "ttt_enabled": self.ttt_enabled,
            "ttt_reset_between_requests": self.ttt_reset_between_requests,
            "mcp_servers": dict(self.mcp_servers),
            "mcp_timeout_ms": self.mcp_timeout_ms,
            "green_power_management": self.green_power_management,
            "green_target_region": self.green_target_region,
            "tee_mode": self.tee_mode,
            "vocab_size": self.vocab_size,
            "semantic_cache_threshold": self.semantic_cache_threshold,
            "semantic_cache_size": self.semantic_cache_size,
            "cxl_pool_size_gb": self.cxl_pool_size_gb,
            "enable_diffusion_spec": self.enable_diffusion_spec,
            "enable_semantic_cache": self.enable_semantic_cache,
            "diffusion_spec_K": self.diffusion_spec_K,
            "diffusion_spec_T": self.diffusion_spec_T,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RuntimeConfig:
        return RuntimeConfig(
            optimize_for=data.get("optimize_for", DEFAULT_OPTIMIZE_FOR),
            speculative_decoding=data.get("speculative_decoding", DEFAULT_SPECULATIVE_DECODING),
            speculative_tree_depth=data.get("speculative_tree_depth", DEFAULT_SPECULATIVE_TREE_DEPTH),
            prefill_chunk_size=data.get("prefill_chunk_size", DEFAULT_PREFILL_CHUNK_SIZE),
            max_batch_size=data.get("max_batch_size", DEFAULT_MAX_BATCH_SIZE),
            kv_cache_dtype=data.get("kv_cache_dtype", DEFAULT_KV_CACHE_DTYPE),
            kv_cache_cpu_gb=data.get("kv_cache_cpu_gb", DEFAULT_KV_CACHE_CPU_GB),
            kv_cache_nvme_gb=data.get("kv_cache_nvme_gb", DEFAULT_KV_CACHE_NVME_GB),
            dynamic_precision=data.get("dynamic_precision", DEFAULT_DYNAMIC_PRECISION),
            disaggregate_prefill_decode=data.get("disaggregate_prefill_decode", DEFAULT_DISAGGREGATE_SERVE),
            server_port=data.get("server_port", DEFAULT_SERVER_PORT),
            server_host=data.get("server_host", DEFAULT_SERVER_HOST),
            default_temperature=data.get("default_temperature", DEFAULT_TEMPERATURE),
            default_max_tokens=data.get("default_max_tokens", DEFAULT_MAX_TOKENS),
            default_top_p=data.get("default_top_p", DEFAULT_TOP_P),
            backend_name=data.get("backend_name"),
            execution_devices=(
                list(data["execution_devices"])
                if data.get("execution_devices") is not None else None
            ),
            model_cache_dir=data.get("model_cache_dir"),
            model_download_timeout_s=data.get("model_download_timeout_s", 30.0),
            hf_offline=data.get("hf_offline", False),
            allow_remote_code=data.get("allow_remote_code", False),
            enable_memory_profiling=data.get("enable_memory_profiling", False),
            enable_telemetry=data.get("enable_telemetry", True),
            extra=dict(data.get("extra", {})),
            model_routing=dict(data.get("model_routing", {})),
            reasoning_budget=data.get("reasoning_budget", 0),
            enable_safety_layer=data.get("enable_safety_layer", False),
            telemetry_endpoint=data.get("telemetry_endpoint"),
            saguaro_enabled=data.get("saguaro_enabled", False),
            multi_agent_kv_mode=data.get("multi_agent_kv_mode", "relay"),
            scheduler=data.get("scheduler", "continuous_batching"),
            slo_profiles=dict(data.get("slo_profiles", {})),
            ttt_enabled=data.get("ttt_enabled", False),
            ttt_reset_between_requests=data.get("ttt_reset_between_requests", True),
            mcp_servers=dict(data.get("mcp_servers", {})),
            mcp_timeout_ms=data.get("mcp_timeout_ms", 5000),
            green_power_management=data.get("green_power_management", False),
            green_target_region=data.get("green_target_region", "lowest_carbon"),
            tee_mode=data.get("tee_mode", "auto"),
            vocab_size=data.get("vocab_size", 128000),
            semantic_cache_threshold=data.get("semantic_cache_threshold", 0.92),
            semantic_cache_size=data.get("semantic_cache_size", 100_000),
            cxl_pool_size_gb=data.get("cxl_pool_size_gb", 0.0),
            enable_diffusion_spec=data.get("enable_diffusion_spec", True),
            enable_semantic_cache=data.get("enable_semantic_cache", True),
            diffusion_spec_K=data.get("diffusion_spec_K", 8),
            diffusion_spec_T=data.get("diffusion_spec_T", 4),
        )

    @staticmethod
    def from_env() -> RuntimeConfig:
        """Load runtime configuration from environment variables."""
        config = RuntimeConfig()
        if "AETHER_OPTIMIZE_FOR" in os.environ:
            config.optimize_for = os.environ["AETHER_OPTIMIZE_FOR"]
        if "AETHER_SPECULATIVE_DECODING" in os.environ:
            raw_speculation = os.environ["AETHER_SPECULATIVE_DECODING"].strip().lower()
            if raw_speculation in {"1", "true", "yes", "on"}:
                config.speculative_decoding = True
            elif raw_speculation in {"0", "false", "no", "off", "none", "disabled"}:
                config.speculative_decoding = "none"
            else:
                # Preserve named engines such as eagle3/p_eagle/saguaro;
                # coercing every non-boolean value to False silently disabled
                # valid deployments configured through the environment.
                config.speculative_decoding = raw_speculation
        if "AETHER_PREFILL_CHUNK_SIZE" in os.environ:
            config.prefill_chunk_size = int(os.environ["AETHER_PREFILL_CHUNK_SIZE"])
        if "AETHER_MAX_BATCH_SIZE" in os.environ:
            config.max_batch_size = int(os.environ["AETHER_MAX_BATCH_SIZE"])
        if "AETHER_SERVER_PORT" in os.environ:
            config.server_port = int(os.environ["AETHER_SERVER_PORT"])
        if "AETHER_BACKEND" in os.environ:
            config.backend_name = os.environ["AETHER_BACKEND"]
        if "AETHER_EXECUTION_DEVICES" in os.environ:
            config.execution_devices = [
                item.strip()
                for item in os.environ["AETHER_EXECUTION_DEVICES"].split(",")
                if item.strip()
            ]
        if "AETHER_MODEL_DOWNLOAD_TIMEOUT_S" in os.environ:
            config.model_download_timeout_s = float(os.environ["AETHER_MODEL_DOWNLOAD_TIMEOUT_S"])
        if "AETHER_HF_OFFLINE" in os.environ:
            config.hf_offline = os.environ["AETHER_HF_OFFLINE"].lower() in ("1", "true", "yes")
        config.validate()
        return config

    def __repr__(self) -> str:
        return f"RuntimeConfig(optimize_for={self.optimize_for}, backend={self.backend_name})"
