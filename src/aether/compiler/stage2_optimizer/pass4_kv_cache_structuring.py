"""
Pass 4: KV Cache Structuring.

Annotates the AEG-IR with explicit KV cache graph nodes: paged block sizes,
radix-tree prefix hints, memory tier offload thresholds, and cache-sharing
policies.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import KVCacheStructuringPass

__all__ = ["KVCacheStructuringPass"]
