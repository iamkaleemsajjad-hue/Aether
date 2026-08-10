"""
Global KV cache manager with tiered storage and RadixTree prefix hints.

Implements L1 (GPU HBM), L2 (CPU DRAM), L3 (NVMe SSD), and L4 (Aether Hub)
KV cache tiers. The cache manager tracks active request blocks, prefix cache
hits, and eviction across tiers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aether.core.types import MemoryTier
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KVCacheBlock:
    """A single KV cache block."""

    block_id: int
    """Unique block identifier."""

    layer_index: int
    """Transformer layer index."""

    token_start: int
    """Start token position in the sequence."""

    token_count: int
    """Number of tokens in this block."""

    prefix_hash: str | None = None
    """Hash of the prefix for deduplication."""

    tier: MemoryTier = MemoryTier.L1_GPU_HBM
    """Current storage tier."""

    last_access: float = 0.0
    """Last access timestamp (monotonic)."""

    access_count: int = 0
    """Number of accesses."""

    def __post_init__(self) -> None:
        if self.token_count < 0:
            msg = "token_count must be non-negative"
            raise ValueError(msg)


class KVCacheManager:
    """Manager for the global tiered KV cache."""

    def __init__(
        self,
        dtype: str = "fp8",
        cpu_budget_gb: int = 32,
        nvme_budget_gb: int = 200,
    ) -> None:
        self.dtype = dtype
        self.cpu_budget_gb = cpu_budget_gb
        self.nvme_budget_gb = nvme_budget_gb
        self._blocks: dict[str, KVCacheBlock] = {}
        self._prefix_index: dict[str, str] = {}
        self._lock = threading.RLock()
        self._next_block_id = 0
        self._stats: dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "transfers": 0,
            "transferred_tokens": 0,
            "transfers_by_route": {},
            "last_transfer_monotonic": None,
        }
        logger.info(
            "KV cache manager initialized",
            dtype=dtype,
            cpu_budget_gb=cpu_budget_gb,
            nvme_budget_gb=nvme_budget_gb,
        )

    @property
    def block_count(self) -> int:
        """Return the number of tracked blocks."""
        with self._lock:
            return len(self._blocks)

    def allocate_block(
        self,
        layer_index: int,
        token_start: int,
        token_count: int,
        prefix_hash: str | None = None,
    ) -> KVCacheBlock:
        """Allocate a new KV cache block."""
        with self._lock:
            block_id = self._next_block_id
            self._next_block_id += 1
            block = KVCacheBlock(
                block_id=block_id,
                layer_index=layer_index,
                token_start=token_start,
                token_count=token_count,
                prefix_hash=prefix_hash,
            )
            self._blocks[str(block_id)] = block
            if prefix_hash:
                self._prefix_index[prefix_hash] = str(block_id)
            return block

    def find_prefix(self, prefix_hash: str) -> KVCacheBlock | None:
        """Find a cached block matching a prefix hash."""
        with self._lock:
            block_id = self._prefix_index.get(prefix_hash)
            if block_id is None:
                self._stats["misses"] += 1
                return None
            block = self._blocks.get(block_id)
            if block:
                block.access_count += 1
                self._stats["hits"] += 1
            return block

    def evict_to_tier(self, block_id: str, target_tier: MemoryTier) -> bool:
        """Move a block to a lower tier (e.g., L1 -> L2 -> L3)."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                return False
            if block.tier == target_tier:
                return True
            old_tier = block.tier
            block.tier = target_tier
            self._record_transfer(old_tier, target_tier, block.token_count)
            logger.debug(
                f"KV block {block_id} moved from {old_tier} to {target_tier}"
            )
            return True

    def _record_transfer(
        self,
        source_tier: MemoryTier,
        destination_tier: MemoryTier,
        token_count: int,
    ) -> None:
        """Record a real tier movement without inventing byte counts.

        KVCacheBlock stores token cardinality but not tensor shape or element
        width, so this manager can truthfully report blocks/tokens moved and
        route identity, but must not manufacture bandwidth or byte totals.
        """
        route = f"{source_tier.name}->{destination_tier.name}"
        self._stats["transfers"] += 1
        self._stats["transferred_tokens"] += max(0, int(token_count))
        routes = self._stats["transfers_by_route"]
        routes[route] = int(routes.get(route, 0)) + 1
        self._stats["last_transfer_monotonic"] = time.monotonic()

    def evict_lru(self, tier: MemoryTier | None = None) -> int:
        """Evict least-recently-used blocks from a tier."""
        with self._lock:
            candidates = [
                b for b in self._blocks.values()
                if tier is None or b.tier == tier
            ]
            if not candidates:
                return 0
            candidates.sort(key=lambda b: b.last_access)
            to_evict = candidates[: max(1, len(candidates) // 10)]
            for block in to_evict:
                if block.tier == MemoryTier.L1_GPU_HBM:
                    old_tier = block.tier
                    block.tier = MemoryTier.L2_CPU_DRAM
                    self._record_transfer(old_tier, block.tier, block.token_count)
                elif block.tier == MemoryTier.L2_CPU_DRAM:
                    old_tier = block.tier
                    block.tier = MemoryTier.L3_NVME
                    self._record_transfer(old_tier, block.tier, block.token_count)
                elif block.tier == MemoryTier.L3_NVME:
                    self._blocks.pop(str(block.block_id), None)
                    if block.prefix_hash:
                        self._prefix_index.pop(block.prefix_hash, None)
                    self._stats["evictions"] += 1
            return len(to_evict)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return dict(self._stats)

    def get_transfer_stats(self) -> dict[str, Any]:
        """Return measured local-tier movement statistics.

        This is intentionally separate from network/RDMA statistics.  The
        local manager can prove tier transitions, block counts, and tokens
        moved; it cannot claim NIXL, UCCL, RDMA, or GPU-initiated transfers.
        """
        with self._lock:
            return {
                "local_tier_transfers": int(self._stats["transfers"]),
                "transferred_tokens": int(self._stats["transferred_tokens"]),
                "transfers_by_route": dict(self._stats["transfers_by_route"]),
                "last_transfer_monotonic": self._stats["last_transfer_monotonic"],
            }

    def hit_rate(self) -> float:
        """Return the prefix cache hit rate."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            if total == 0:
                return 0.0
            return self._stats["hits"] / total

    def __repr__(self) -> str:
        return f"KVCacheManager(blocks={self.block_count}, hit_rate={self.hit_rate():.2f})"
