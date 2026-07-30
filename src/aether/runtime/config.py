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

    speculative_decoding: bool = DEFAULT_SPECULATIVE_DECODING
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

    model_cache_dir: str | None = None
    """Custom model cache directory."""

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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        if self.optimize_for not in ("latency", "throughput", "quality"):
            msg = f"optimize_for must be 'latency', 'throughput', or 'quality', got '{self.optimize_for}'"
            raise RuntimeConfigError(msg)
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
            "lazy_model_loading": self.lazy_model_loading,
            "enable_continuous_batching": self.enable_continuous_batching,
            "enable_prefix_caching": self.enable_prefix_caching,
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
        )

    @staticmethod
    def from_env() -> RuntimeConfig:
        """Load runtime configuration from environment variables."""
        config = RuntimeConfig()
        if "AETHER_OPTIMIZE_FOR" in os.environ:
            config.optimize_for = os.environ["AETHER_OPTIMIZE_FOR"]
        if "AETHER_SPECULATIVE_DECODING" in os.environ:
            config.speculative_decoding = os.environ["AETHER_SPECULATIVE_DECODING"].lower() in ("1", "true", "yes")
        if "AETHER_PREFILL_CHUNK_SIZE" in os.environ:
            config.prefill_chunk_size = int(os.environ["AETHER_PREFILL_CHUNK_SIZE"])
        if "AETHER_MAX_BATCH_SIZE" in os.environ:
            config.max_batch_size = int(os.environ["AETHER_MAX_BATCH_SIZE"])
        if "AETHER_SERVER_PORT" in os.environ:
            config.server_port = int(os.environ["AETHER_SERVER_PORT"])
        if "AETHER_BACKEND" in os.environ:
            config.backend_name = os.environ["AETHER_BACKEND"]
        config.validate()
        return config

    def __repr__(self) -> str:
        return f"RuntimeConfig(optimize_for={self.optimize_for}, backend={self.backend_name})"
