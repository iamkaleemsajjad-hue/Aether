"""
Pass 4: KV Cache Structuring.

Annotates the AEG-IR with explicit KV cache graph nodes: paged block sizes,
radix-tree prefix hints, memory tier offload thresholds, and cache-sharing
policies.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import KVCacheStructuringPass
from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["KVCacheStructuringPass"]


class KVCacheStructuringPass(KVCacheStructuringPass):
    """Exported alias of the KV cache structuring pass."""
