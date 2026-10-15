"""
Global KV cache manager with tiered storage and RadixTree prefix hints.

Implements L1 (GPU HBM), L2 (CPU DRAM), L3 (NVMe SSD), and L4 (Aether Hub)
KV cache tiers. The cache manager tracks active request blocks, prefix cache
hits, and eviction across tiers.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Any

from aether.core.types import MemoryTier
from aether.utils.logging import get_logger
from aether.runtime.kv_payload import CachePayload
from aether.runtime.kv_storage import KVPayloadStore

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
        *,
        payload_store: KVPayloadStore | None = None,
        hub_client: Any = None,
        hub_namespace: str | None = None,
    ) -> None:
        self.dtype = dtype
        self.cpu_budget_gb = cpu_budget_gb
        self.nvme_budget_gb = nvme_budget_gb
        self._blocks: dict[str, KVCacheBlock] = {}
        self._prefix_index: dict[str, str] = {}
        self._lock = threading.RLock()
        self._next_block_id = 0
        self.payload_store = payload_store
        self.hub_client = hub_client
        self.hub_namespace = hub_namespace
        self._payloads: dict[str, CachePayload] = {}
        self._payload_keys: dict[str, str] = {}
        self._payload_identities: dict[str, tuple[str, str]] = {}
        self._hub_keys: set[str] = set()
        self._verified_backing: dict[str, tuple[MemoryTier, str]] = {}
        self._payload_stats: dict[str, Any] = {"transfers": 0, "tensor_bytes": 0, "serialized_bytes": 0,
                                             "transfers_by_route": {}, "last_transfer_monotonic": None}
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
        # Recover descriptors only from committed local snapshots. Actual
        # tensors are read/checksummed lazily on reload.
        if payload_store is not None:
            for entry in payload_store.entries():
                descriptor = entry["metadata"].get("block")
                if not isinstance(descriptor, dict):
                    continue
                block = KVCacheBlock(**descriptor, tier=MemoryTier.L3_NVME)
                # Block integers are local allocation coordinates, not global
                # snapshot identities. Independent managers can persist the
                # same integer; every committed UUID key must survive recovery.
                if str(block.block_id) in self._blocks:
                    block.block_id = self._next_block_id
                block_id = str(block.block_id)
                self._blocks[block_id] = block
                self._payload_keys[block_id] = entry["cache_key"]
                self._payload_identities[block_id] = (entry["model_fingerprint"], entry["cache_kind"])
                self._next_block_id = max(self._next_block_id, block.block_id + 1)
                if block.prefix_hash:
                    self._prefix_index[block.prefix_hash] = block_id

    def attach_payload(self, block_id: str | int, payload: CachePayload) -> None:
        """Attach a real immutable host snapshot; do not infer GPU residency."""
        if not isinstance(payload, CachePayload):
            raise TypeError("Cache attachment requires a CachePayload")
        # Preserve a snapshot even if the caller subsequently mutates its
        # public metadata/mapping. TensorPayload buffers themselves are immutable.
        payload = CachePayload(payload.model_fingerprint, payload.cache_kind, payload.logical_length,
                               dict(payload.tensors), dict(payload.metadata))
        block_id = str(block_id)
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                raise KeyError(f"Unknown cache block: {block_id}")
            resident = sum(value.byte_size for key, value in self._payloads.items() if key != block_id)
            if resident + payload.byte_size > self.cpu_budget_gb * 1024**3:
                raise ValueError("CPU cache payload budget would be exceeded")
            existing = self._payload_identities.get(block_id)
            identity = (payload.model_fingerprint, payload.cache_kind)
            if existing is not None and existing != identity:
                raise ValueError("Cannot replace a block with a different model/cache identity")
            self._payloads[block_id] = payload
            self._verified_backing.pop(block_id, None)
            self._payload_keys.setdefault(block_id, uuid.uuid4().hex)
            self._payload_identities[block_id] = identity
            block.tier = MemoryTier.L2_CPU_DRAM
            block.last_access = time.monotonic()

    def import_payload(self, payload: CachePayload, *, layer_index: int = 0, token_start: int = 0,
                       prefix_hash: str | None = None) -> KVCacheBlock:
        with self._lock:
            previous = self._prefix_index.get(prefix_hash) if prefix_hash else None
            block = self.allocate_block(layer_index, token_start, payload.logical_length, prefix_hash)
            try:
                self.attach_payload(block.block_id, payload)
            except Exception:
                self._blocks.pop(str(block.block_id), None)
                if prefix_hash:
                    if previous is None:
                        self._prefix_index.pop(prefix_hash, None)
                    else:
                        self._prefix_index[prefix_hash] = previous
                raise
            return block

    def payload_key(self, block_id: str | int) -> str:
        """Return the persistent local/remote snapshot key for explicit handoff."""
        with self._lock:
            return self._payload_keys[str(block_id)]

    def block_for_payload_key(self, key: str) -> KVCacheBlock | None:
        """Find an existing/recovered descriptor without allocating duplicates."""
        with self._lock:
            return next((self._blocks[block_id] for block_id, stored_key in self._payload_keys.items()
                         if stored_key == key), None)

    def release_payload_residency(self, block_id: str | int) -> bool:
        """Drop a verified snapshot copy after an execution adapter imports it.

        The engine owns its reconstructed tensors. This manager returns to its
        committed backing tier without claiming the engine's device residency
        and without re-uploading or rewriting/fsyncing the same snapshot.
        """
        block_id = str(block_id)
        with self._lock:
            payload = self._payloads.get(block_id)
            if payload is None:
                return False
            backing = self._verified_backing.get(block_id)
            if backing is None or payload.content_hash() != backing[1]:
                raise ValueError("Cannot release an uncommitted or modified cache payload")
            self._payloads.pop(block_id)
            self._blocks[block_id].tier = backing[0]
            return True

    def discard_resident_payloads(self) -> int:
        """Release owned host snapshots when their Runtime finishes shutting down.

        This terminal cleanup does not flush or delete local/Hub snapshots and
        preserves their keys, identities and descriptors. Verified descriptors
        return to their backing tier. Unpersisted bytes are discarded without
        creating backing storage; descriptor tiers alone do not prove residency.
        Use ``release_payload_residency`` for recoverable live-cache eviction.

        Return the released tensor-byte count; repeated calls return zero.
        """
        with self._lock:
            released = sum(payload.byte_size for payload in self._payloads.values())
            for block_id in self._payloads:
                backing = self._verified_backing.get(block_id)
                if backing is not None:
                    self._blocks[block_id].tier = backing[0]
            self._payloads.clear()
            return released

    def export_payload(self, block_id: str | int, *, expected_model_fingerprint: str | None = None,
                       expected_cache_kind: str | None = None) -> CachePayload:
        block_id = str(block_id)
        with self._lock:
            identity = self._payload_identities.get(block_id)
            if identity is None:
                raise ValueError("Cache block is descriptor-only and contains no tensor snapshot")
            if expected_model_fingerprint is not None and identity[0] != expected_model_fingerprint:
                raise ValueError("Cache model fingerprint does not match the destination model")
            if expected_cache_kind is not None and identity[1] != expected_cache_kind:
                raise ValueError("Cache kind does not match the destination adapter")
            payload = self._payloads.get(block_id)
            if payload is None:
                payload = self.reload(block_id, expected_model_fingerprint=identity[0], expected_cache_kind=identity[1])
            return payload

    def _record_payload_transfer(self, source: MemoryTier, destination: MemoryTier,
                                 payload: CachePayload, serialized_bytes: int) -> None:
        route = f"{source.name}->{destination.name}"
        self._payload_stats["transfers"] += 1
        self._payload_stats["tensor_bytes"] += payload.byte_size
        self._payload_stats["serialized_bytes"] += serialized_bytes
        routes = self._payload_stats["transfers_by_route"]
        routes[route] = routes.get(route, 0) + 1
        self._payload_stats["last_transfer_monotonic"] = time.monotonic()

    def swap(self, block_id: str | int, target_tier: MemoryTier) -> bool:
        """Persist real bytes before releasing residency or changing the tier.

        Host snapshots can move between L2, local disk (L3), and explicitly
        configured authenticated HTTP Hub storage (L4). L1 requires an engine
        device import and is deliberately not inferred by this host manager.
        """
        block_id = str(block_id)
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                return False
            if block_id not in self._payload_identities:
                raise ValueError("Cannot transfer a descriptor-only block as tensor bytes")
            if target_tier == MemoryTier.L1_GPU_HBM:
                raise ValueError("GPU cache residency requires an execution adapter import")
            if target_tier == MemoryTier.L2_CPU_DRAM:
                self.reload(block_id)
                return True
            if target_tier not in (MemoryTier.L3_NVME, MemoryTier.L4_HUB):
                raise ValueError("Unsupported cache payload tier")
            if block.tier == target_tier and block_id not in self._payloads:
                return True
            source = block.tier
            was_resident = block_id in self._payloads
            payload = self.export_payload(block_id)
            key = self._payload_keys[block_id]
            try:
                if target_tier == MemoryTier.L3_NVME:
                    if self.payload_store is None:
                        raise ValueError("Local cache payload storage is not configured")
                    descriptor = asdict(block)
                    descriptor.pop("tier")
                    entry = self.payload_store.put(key, payload, metadata={"block": descriptor})
                else:
                    if self.hub_client is None or not self.hub_namespace:
                        raise ValueError("Explicit Hub client and namespace are required for cache uploads")
                    entry = self.hub_client.put_kv(self.hub_namespace, key, payload)
                    self._hub_keys.add(key)
            except Exception:
                if not was_resident:
                    self._payloads.pop(block_id, None)
                    block.tier = source
                raise
            # The destination ACK/commit succeeded; only now release host bytes.
            self._payloads.pop(block_id, None)
            block.tier = target_tier
            self._verified_backing[block_id] = (target_tier, entry["content_hash"])
            self._record_payload_transfer(source, target_tier, payload, entry["size_bytes"])
            return True

    def reload(self, block_id: str | int, *, expected_model_fingerprint: str | None = None,
               expected_cache_kind: str | None = None) -> CachePayload:
        """Restore verified bytes to host residency, leaving failed reads untouched."""
        block_id = str(block_id)
        with self._lock:
            block = self._blocks[block_id]
            identity = self._payload_identities[block_id]
            fingerprint = expected_model_fingerprint or identity[0]
            kind = expected_cache_kind or identity[1]
            if (fingerprint, kind) != identity:
                raise ValueError("Cache destination model/kind identity mismatch")
            resident = self._payloads.get(block_id)
            if resident is not None:
                block.last_access = time.monotonic()
                return resident
            key = self._payload_keys[block_id]
            source = block.tier
            if source == MemoryTier.L3_NVME and self.payload_store is not None:
                payload = self.payload_store.get(key, expected_model_fingerprint=fingerprint, expected_cache_kind=kind)
                entry = next((row for row in self.payload_store.entries() if row["cache_key"] == key), None)
                size = entry["size_bytes"] if entry else 0
            elif source == MemoryTier.L4_HUB and self.hub_client is not None and self.hub_namespace:
                payload = self.hub_client.get_kv(self.hub_namespace, key, expected_model_fingerprint=fingerprint,
                                                expected_cache_kind=kind)
                size = self.hub_client.last_operation["downloaded_bytes"]
            else:
                raise ValueError("The block's payload storage is not configured")
            if payload is None:
                raise ValueError("Stored cache snapshot is missing")
            self.attach_payload(block_id, payload)
            self._verified_backing[block_id] = (source, payload.content_hash())
            self._record_payload_transfer(source, MemoryTier.L2_CPU_DRAM, payload, size)
            return payload

    def import_from_hub(self, key: str, *, expected_model_fingerprint: str,
                        expected_cache_kind: str | None = None) -> KVCacheBlock:
        """Explicit remote/restart handoff; downloads are never automatic network probes."""
        if self.hub_client is None or not self.hub_namespace:
            raise ValueError("Explicit Hub client and namespace are required")
        payload = self.hub_client.get_kv(self.hub_namespace, key, expected_model_fingerprint=expected_model_fingerprint,
                                        expected_cache_kind=expected_cache_kind)
        block = self.import_payload(payload)
        self._payload_keys[str(block.block_id)] = key
        self._hub_keys.add(key)
        self._verified_backing[str(block.block_id)] = (MemoryTier.L4_HUB, payload.content_hash())
        self._record_payload_transfer(MemoryTier.L4_HUB, MemoryTier.L2_CPU_DRAM, payload,
                                      self.hub_client.last_operation["downloaded_bytes"])
        return block

    def remove(self, block_id: str | int) -> bool:
        """Remove a snapshot and its committed local/known remote backing bytes."""
        block_id = str(block_id)
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                return False
            key = self._payload_keys.get(block_id)
            if key in self._hub_keys:
                self.hub_client.delete_kv(self.hub_namespace, key)
                self._hub_keys.discard(key)
            if key is not None and self.payload_store is not None:
                self.payload_store.delete(key)
            self._blocks.pop(block_id)
            self._payloads.pop(block_id, None)
            self._payload_keys.pop(block_id, None)
            self._payload_identities.pop(block_id, None)
            self._verified_backing.pop(block_id, None)
            if block.prefix_hash and self._prefix_index.get(block.prefix_hash) == block_id:
                self._prefix_index.pop(block.prefix_hash, None)
            return True

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
                block.last_access = time.monotonic()
                self._stats["hits"] += 1
            return block

    def evict_to_tier(self, block_id: str, target_tier: MemoryTier) -> bool:
        """Move a block to a lower tier (e.g., L1 -> L2 -> L3)."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None:
                return False
            if block_id in self._payload_identities:
                return self.swap(block_id, target_tier)
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
        """Record a descriptor transition without inventing byte movement.

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
                if str(block.block_id) in self._payload_identities:
                    if block.tier == MemoryTier.L2_CPU_DRAM:
                        self.swap(block.block_id, MemoryTier.L3_NVME)
                    elif block.tier == MemoryTier.L3_NVME:
                        self.remove(block.block_id)
                    continue
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
                "legacy_transfer_scope": "descriptor_transitions",
                "payload_transfers": self._payload_stats["transfers"],
                "transferred_tensor_bytes": self._payload_stats["tensor_bytes"],
                "transferred_serialized_bytes": self._payload_stats["serialized_bytes"],
                "payload_transfers_by_route": dict(self._payload_stats["transfers_by_route"]),
                "resident_tensor_bytes": sum(payload.byte_size for payload in self._payloads.values()),
                "payload_storage_scope": "host_snapshot/local_disk/explicit_http",
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
