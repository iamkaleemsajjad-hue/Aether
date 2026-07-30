"""
Memory estimation and tracking utilities.

Provides helpers to estimate model memory usage, KV cache requirements, and
available system memory. These are used by the compiler plan and runtime
scheduler to make placement decisions.
"""

from __future__ import annotations

from typing import Any

from aether.core.constants import PRECISION_SIZES_BYTES
from aether.core.types import ModelArchitecture
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def estimate_model_memory_gb(
    architecture: ModelArchitecture,
    precision_map: dict[str, str] | None = None,
    kv_cache_gb: float = 0.0,
    overhead_factor: float = 1.15,
) -> float:
    """Estimate total model memory in GB.

    Args:
        architecture: Model architecture metadata.
        precision_map: Precision assignment per layer. Defaults to BF16.
        kv_cache_gb: Additional KV cache memory budget in GB.
        overhead_factor: Multiplier for activation overhead.

    Returns:
        Estimated memory in GB.
    """
    if precision_map is None:
        precision_map = {"default": "BF16"}
    avg_bits = sum(PRECISION_SIZES_BYTES.get(p.upper(), 2.0) for p in precision_map.values()) / max(len(precision_map), 1)
    params_gb = architecture.params_billion * avg_bits * 8 / 1e9
    activation_gb = params_gb * 0.2 * overhead_factor
    total = params_gb + activation_gb + kv_cache_gb
    return total


def estimate_kv_cache_memory_gb(
    architecture: ModelArchitecture,
    sequence_length: int,
    batch_size: int = 1,
    dtype_bytes: float = 1.0,
) -> float:
    """Estimate KV cache memory for a given sequence length and batch size.

    Args:
        architecture: Model architecture metadata.
        sequence_length: Number of tokens per sequence.
        batch_size: Number of concurrent sequences.
        dtype_bytes: Bytes per KV element (e.g., 1.0 for FP8).

    Returns:
        Estimated KV cache memory in GB.
    """
    num_layers = architecture.layers
    num_kv_heads = architecture.num_kv_heads or architecture.num_attention_heads
    head_dim = architecture.head_dim or (architecture.hidden_size // architecture.num_attention_heads)
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * sequence_length * batch_size
    return total_bytes / (1024**3)


def available_system_memory_gb() -> float:
    """Return available system memory in GB.

    Requires psutil. Falls back to a conservative estimate if unavailable.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except Exception:
        return 4.0


def memory_pressure(used_gb: float, total_gb: float) -> float:
    """Return memory pressure as a value in [0.0, 1.0]."""
    if total_gb <= 0:
        return 0.0
    return max(0.0, min(1.0, used_gb / total_gb))


def fit_in_memory(
    architecture: ModelArchitecture,
    available_gb: float,
    precision_map: dict[str, str] | None = None,
    kv_cache_gb: float = 0.0,
) -> dict[str, Any]:
    """Determine whether a model fits in available memory and report slack.

    Returns:
        A dictionary with fit boolean, estimated memory, and slack.
    """
    estimated = estimate_model_memory_gb(architecture, precision_map, kv_cache_gb)
    fits = estimated <= available_gb
    return {
        "fits": fits,
        "estimated_gb": estimated,
        "available_gb": available_gb,
        "slack_gb": available_gb - estimated,
        "kv_cache_gb": kv_cache_gb,
    }


def recommend_batch_size(
    available_gb: float,
    model_memory_gb: float,
    kv_per_sequence_gb: float,
    max_batch_size: int = 256,
) -> int:
    """Recommend a conservative batch size based on memory constraints."""
    usable = max(0.0, available_gb - model_memory_gb)
    if kv_per_sequence_gb <= 0:
        return max_batch_size
    batch = int(usable // kv_per_sequence_gb)
    return max(1, min(batch, max_batch_size))
