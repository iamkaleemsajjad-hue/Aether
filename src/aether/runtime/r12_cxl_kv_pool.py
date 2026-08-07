"""
R12 — CXL Rack-Scale KV Pool.

Implements a cache-coherent, rack-scale KV cache pool using Compute Express
Link (CXL) 3.0 shared memory. CXL enables low-latency (300-500 ns) access to
terabyte-scale memory pools shared across multiple GPU nodes in a rack.

This eliminates KV recomputation when migrating a model session across nodes.
Instead of recomputing the full KV cache (seconds), the new node reads directly
from the CXL pool (microseconds).

Architecture:
  1. **CXL Memory Pool**: Shared 128 GB-4 TB memory pool exposed via CXL 3.0
     Type-3 device. All nodes in the rack see the same physical address space.
  2. **BlueField-4 DPU Management**: NVIDIA CMX platform uses BlueField-4 DPU
     for pool management, bandwidth QoS, and namespace isolation.
  3. **3-Tier KV Platform**: Hot (GPU HBM) → Warm (CXL pool) → Cold (NVMe/GPUDirect)
  4. **TraCT Policy**: Transfer vs recompute decision engine that minimizes TTFT.
  5. **Background Defragmentation**: Daemon compacts fragmented CXL pool regions.

Storage hierarchy:
  L1 - GPU HBM:      1-6 TB/s BW, 0.1-0.3 ms access, ~80 GB capacity
  L2 - CXL pool:     300-500 ns latency, 64-512 GB/s BW, 128 GB-4 TB capacity
  L3 - NVMe-oF:      100 μs latency, 10-50 GB/s BW, petabyte-scale

Research basis:
  - TraCT CXL (arXiv 2026): CXL rack-scale KV; 300-500 ns latency
  - NVIDIA CMX (NVIDIA 2026): 3-tier KV platform; BlueField-4 DPU management
  - DUAL-BLADE NVMe-direct (arXiv 2026): NVMe-direct bypass for cold tier
  - GPUDirect Storage (NVIDIA 2024-2026): direct DMA GPU to NVMe
  - NVMe-oF for AI (Industry 2026): cross-node KV fabric
  - CXL 3.0 Specification (2024-2026): cache-coherent shared memory; 300-500 ns
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CXL pool capacity & layout constants
# ─────────────────────────────────────────────────────────────────────────────

# CXL block size (aligned to CXL 3.0 granule: 256 bytes)
CXL_BLOCK_BYTES = 4096        # 4 KB KV blocks (standard page size)
CXL_NAMESPACE_HEADER_BYTES = 64  # Per-namespace control header
MAX_NAMESPACE_COUNT = 1024    # Max concurrent tenants


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CXLKVBlock:
    """A KV cache block stored in the CXL pool.

    Attributes:
        block_id: Content-addressed ID (SHA-256 of KV data prefix hash).
        session_id: Session that owns this block.
        layer_indices: List of transformer layers whose KV data is in this block.
        seq_len: Number of tokens whose KV this block covers.
        dtype: Data type of KV data ('FP8', 'BF16', etc.).
        created_at: Creation timestamp.
        last_access: Last access timestamp (for LRU eviction).
        ref_count: Reference count (block is live while > 0).
        cxl_offset: Byte offset within CXL pool where this block resides.
        is_pinned: If True, block cannot be evicted.
        checksum: CRC32 of block data for integrity validation.
    """

    block_id: str
    session_id: str
    layer_indices: list[int]
    seq_len: int
    dtype: str = "FP8"
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    ref_count: int = 0
    cxl_offset: int = -1      # -1 = not yet placed in pool
    is_pinned: bool = False
    checksum: int = 0

    @property
    def size_bytes(self) -> int:
        """Estimated block size in bytes."""
        bytes_per_dtype = {"BF16": 2, "FP16": 2, "FP8": 1, "INT4": 1}
        bpd = bytes_per_dtype.get(self.dtype, 2)
        # Approximate: num_layers × seq_len × kv_heads × head_dim × K+V × bytes
        return len(self.layer_indices) * self.seq_len * 8 * 128 * 2 * bpd

    def touch(self) -> None:
        self.last_access = time.time()
        self.ref_count += 1

    def release(self) -> None:
        self.ref_count = max(0, self.ref_count - 1)


@dataclass
class CXLPoolStats:
    """Statistics for the CXL rack-scale KV pool."""

    total_capacity_bytes: int = 0
    used_bytes: int = 0
    block_count: int = 0
    session_count: int = 0
    cache_hits: int = 0          # pool read hits (session migration)
    cache_misses: int = 0        # had to recompute KV
    evictions: int = 0
    defrag_runs: int = 0
    bytes_transferred_in: int = 0   # from GPU to CXL pool
    bytes_transferred_out: int = 0  # from CXL pool to GPU
    ttft_reductions_ms: list[float] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        if self.total_capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_capacity_bytes

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def mean_ttft_reduction_ms(self) -> float:
        if not self.ttft_reductions_ms:
            return 0.0
        return sum(self.ttft_reductions_ms) / len(self.ttft_reductions_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_capacity_gb": round(self.total_capacity_bytes / 1e9, 2),
            "used_gb": round(self.used_bytes / 1e9, 3),
            "utilization_pct": round(self.utilization * 100, 1),
            "block_count": self.block_count,
            "session_count": self.session_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "defrag_runs": self.defrag_runs,
            "bytes_transferred_in_gb": round(self.bytes_transferred_in / 1e9, 3),
            "bytes_transferred_out_gb": round(self.bytes_transferred_out / 1e9, 3),
            "mean_ttft_reduction_ms": round(self.mean_ttft_reduction_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TraCT: Transfer vs Recompute Decision Engine
# Reference: TraCT CXL (arXiv 2026), NIKA (SCITEPRESS 2026)
# ─────────────────────────────────────────────────────────────────────────────

class TraCTPolicy:
    """Transfer vs Recompute Cost Tradeoff (TraCT) policy engine.

    For each session migration, computes whether it is cheaper to:
    (a) Transfer KV blocks from CXL pool to the new GPU (cost: transfer time)
    (b) Recompute the KV cache on the new GPU (cost: prefill latency)

    Decision function (TraCT arXiv 2026, Eq. 4):
      transfer_time(B) = B / bw_CXL         [seconds]
      recompute_time(L) = L / prefill_tps   [seconds]
      Transfer preferred iff: transfer_time < recompute_time

    Where:
      B = KV block size in bytes
      bw_CXL = CXL pool bandwidth (GB/s)
      L = number of tokens to recompute
      prefill_tps = target model prefill throughput (tokens/second)
    """

    def __init__(
        self,
        cxl_bandwidth_gbs: float = 128.0,    # CXL 3.0 pool BW
        prefill_tps: float = 50_000.0,        # target model prefill throughput
        nvlink_bandwidth_gbs: float = 900.0,  # for GPU-to-GPU transfer
    ) -> None:
        self.cxl_bandwidth_gbs = cxl_bandwidth_gbs
        self.prefill_tps = prefill_tps
        self.nvlink_bandwidth_gbs = nvlink_bandwidth_gbs

    def should_transfer(
        self,
        kv_size_bytes: int,
        seq_len: int,
        is_same_rack: bool = True,
    ) -> tuple[bool, float, float]:
        """Decide transfer vs recompute.

        Args:
            kv_size_bytes: Total bytes in the KV blocks to transfer.
            seq_len: Number of tokens to recompute if transfer not chosen.
            is_same_rack: True for CXL pool access; False for remote RDMA.

        Returns:
            Tuple of (should_transfer, transfer_time_ms, recompute_time_ms).
        """
        # Transfer time
        bw = self.cxl_bandwidth_gbs if is_same_rack else (self.nvlink_bandwidth_gbs * 0.5)
        transfer_time_ms = (kv_size_bytes / (bw * 1e9)) * 1000

        # Recompute time
        recompute_time_ms = (seq_len / self.prefill_tps) * 1000

        # Transfer if it's faster (TraCT criterion)
        should_transfer = transfer_time_ms < recompute_time_ms

        return should_transfer, transfer_time_ms, recompute_time_ms

    def nika_policy(
        self,
        kv_size_gb: float,
        bandwidth_gbps: float,
        decode_util: float,
    ) -> dict[str, Any]:
        """NIKA analytical policy (SCITEPRESS 2026).

        Determines KV transfer strategy based on network conditions and
        decode GPU utilization. Replicates Algorithm 1 from NIKA paper.

        Args:
            kv_size_gb: KV data size in GB.
            bandwidth_gbps: Available network bandwidth in Gbps.
            decode_util: Decode GPU utilization (0.0-1.0).

        Returns:
            Policy recommendation with expected TTFT reduction.
        """
        # NIKA: when decode GPU is >70% utilized, transfer KV to free up prefill GPU
        bandwidth_gbs = bandwidth_gbps / 8  # Gbps → GB/s
        transfer_time_s = kv_size_gb / bandwidth_gbs
        # Expected TTFT reduction = KV recompute time saved
        # Heuristic: recomputing 1 GB of KV ≈ 5 seconds on a typical model
        kv_recompute_time_s = kv_size_gb * 5.0
        ttft_reduction_s = max(0.0, kv_recompute_time_s - transfer_time_s)

        if decode_util > 0.70 and transfer_time_s < kv_recompute_time_s:
            strategy = "transfer"
        else:
            strategy = "recompute"

        return {
            "strategy": strategy,
            "kv_size_gb": kv_size_gb,
            "bandwidth_gbps": bandwidth_gbps,
            "decode_utilization": decode_util,
            "estimated_transfer_time_s": round(transfer_time_s, 3),
            "estimated_recompute_time_s": round(kv_recompute_time_s, 3),
            "estimated_ttft_reduction_s": round(ttft_reduction_s, 3),
            "estimated_ttft_reduction_pct": round(
                ttft_reduction_s / max(0.001, kv_recompute_time_s) * 100, 1
            ),
            "research_basis": "NIKA SCITEPRESS 2026 Algorithm 1",
        }


# ─────────────────────────────────────────────────────────────────────────────
# CXL Pool Backend
# ─────────────────────────────────────────────────────────────────────────────

class CXLPoolBackend:
    """Physical CXL pool access backend.

    Supports two modes:
    1. Real CXL: Uses /dev/cxl/* device files or NUMA-local CXL memory region
    2. Emulated: Uses a file-backed mmap (for development without CXL hardware)

    In production, this would interface with the BlueField-4 DPU management
    plane via NVIDIA CMX platform APIs.
    """

    def __init__(
        self,
        pool_size_gb: float = 128.0,
        device_path: str | None = None,
        emulated_path: str | None = None,
    ) -> None:
        self.pool_size_bytes = int(pool_size_gb * 1024 ** 3)
        self.device_path = device_path
        self.emulated_path = emulated_path
        self._mmap: mmap.mmap | None = None
        self._file: Any = None
        self._mode: str = "none"
        self._lock = threading.RLock()

        self._try_init()

    def _try_init(self) -> None:
        """Try to initialize CXL pool access."""
        # Try real CXL device
        if self.device_path and Path(self.device_path).exists():
            try:
                self._file = open(self.device_path, "r+b")
                self._mmap = mmap.mmap(self._file.fileno(), self.pool_size_bytes)
                self._mode = "cxl_device"
                logger.info(f"R12: CXL pool using real device {self.device_path} ({self.pool_size_bytes // 1e9:.0f} GB)")
                return
            except Exception as e:
                logger.warning(f"R12: CXL device init failed: {e}")

        # Emulated CXL via file-backed mmap
        if self.emulated_path:
            try:
                pool_file = Path(self.emulated_path)
                pool_file.parent.mkdir(parents=True, exist_ok=True)
                # Create/resize backing file
                if not pool_file.exists() or pool_file.stat().st_size != self.pool_size_bytes:
                    # Use a smaller emulated pool to avoid disk space issues
                    emulated_size = min(self.pool_size_bytes, 256 * 1024 * 1024)  # 256 MB max
                    with open(pool_file, "wb") as f:
                        f.seek(emulated_size - 1)
                        f.write(b"\x00")
                    actual_size = pool_file.stat().st_size
                else:
                    actual_size = pool_file.stat().st_size

                self._file = open(pool_file, "r+b")
                self._mmap = mmap.mmap(self._file.fileno(), actual_size)
                self._mode = "emulated"
                logger.info(f"R12: CXL pool emulated via mmap ({actual_size // 1024 // 1024} MB at {self.emulated_path})")
                return
            except Exception as e:
                logger.warning(f"R12: Emulated CXL init failed: {e}")

        # In-memory fallback (for test environments)
        self._mode = "memory"
        self._memory_store: dict[int, bytes] = {}
        logger.info("R12: CXL pool using in-memory fallback (no persistence)")

    def read(self, offset: int, length: int) -> bytes:
        """Read bytes from the CXL pool at given offset."""
        with self._lock:
            if self._mmap is not None:
                self._mmap.seek(offset)
                return self._mmap.read(length)
            elif self._mode == "memory":
                return self._memory_store.get(offset, b"\x00" * length)
            return b"\x00" * length

    def write(self, offset: int, data: bytes) -> bool:
        """Write bytes to the CXL pool at given offset."""
        with self._lock:
            try:
                if self._mmap is not None:
                    self._mmap.seek(offset)
                    self._mmap.write(data)
                    self._mmap.flush()
                    return True
                elif self._mode == "memory":
                    self._memory_store[offset] = data
                    return True
                return False
            except Exception as e:
                logger.error(f"R12: CXL write failed at offset {offset}: {e}")
                return False

    def close(self) -> None:
        """Release CXL pool resources."""
        with self._lock:
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if self._file is not None:
                self._file.close()
                self._file = None

    @property
    def mode(self) -> str:
        return self._mode


# ─────────────────────────────────────────────────────────────────────────────
# Main CXL KV Pool Manager
# ─────────────────────────────────────────────────────────────────────────────

class CXLRackScaleKVPool:
    """Runtime R12: CXL Rack-Scale KV Pool.

    Manages a shared KV cache pool across a rack of GPU nodes using CXL 3.0
    cache-coherent shared memory. Enables zero-recompute session migration.

    API:
        pool = CXLRackScaleKVPool(pool_size_gb=512)
        pool.store_kv(session_id, layer_indices, kv_data)
        kv = pool.load_kv(session_id, layer_indices)
        pool.migrate_session(src_node, dst_node, session_id)
        pool.defragment()

    Memory layout:
        [header: 4KB][namespace_table: 64KB][free_bitmap: 1MB][data_region: rest]
    """

    HEADER_OFFSET = 0
    HEADER_SIZE = 4096
    NAMESPACE_TABLE_SIZE = 65536
    FREE_BITMAP_SIZE = 1024 * 1024  # 1 MB → manages 4 KB blocks up to 512 GB pool

    def __init__(
        self,
        pool_size_gb: float = 128.0,
        device_path: str | None = None,
        emulated_path: str | None = None,
        cxl_bandwidth_gbs: float = 128.0,
        prefill_tps: float = 50_000.0,
        enable_defrag_daemon: bool = True,
        defrag_interval_s: float = 300.0,
    ) -> None:
        self.pool_size_gb = pool_size_gb
        self.cxl_bandwidth_gbs = cxl_bandwidth_gbs

        # Physical backend
        self._backend = CXLPoolBackend(
            pool_size_gb=pool_size_gb,
            device_path=device_path,
            emulated_path=emulated_path,
        )

        # TraCT policy engine
        self.tract = TraCTPolicy(
            cxl_bandwidth_gbs=cxl_bandwidth_gbs,
            prefill_tps=prefill_tps,
        )

        # Block registry (in memory, mirrored in CXL namespace table)
        self._blocks: dict[str, CXLKVBlock] = {}       # block_id → block
        self._session_blocks: dict[str, list[str]] = {} # session_id → [block_ids]
        self._free_offsets: list[int] = []              # available CXL block offsets
        self._next_offset = (
            self.HEADER_SIZE + self.NAMESPACE_TABLE_SIZE + self.FREE_BITMAP_SIZE
        )  # data region starts after control structures

        self._stats = CXLPoolStats(
            total_capacity_bytes=int(pool_size_gb * 1024 ** 3)
        )
        self._lock = threading.RLock()

        # Background defragmentation daemon
        self._defrag_thread: threading.Thread | None = None
        if enable_defrag_daemon:
            self._start_defrag_daemon(defrag_interval_s)

        logger.info(
            f"R12: CXL Rack-Scale KV Pool initialized "
            f"({pool_size_gb} GB, backend={self._backend.mode})"
        )

    # ─────── Public API ───────────────────────────────────────────────────────

    def store_kv(
        self,
        session_id: str,
        layer_indices: list[int],
        kv_data: bytes,
        seq_len: int = 0,
        dtype: str = "FP8",
        pin: bool = False,
    ) -> CXLKVBlock | None:
        """Store KV data for a session in the CXL pool.

        Args:
            session_id: Unique session identifier.
            layer_indices: Transformer layer indices this KV data covers.
            kv_data: Raw KV bytes to store.
            seq_len: Number of tokens this KV covers (for metadata).
            dtype: Data type of KV tensor ('FP8', 'BF16', etc.).
            pin: If True, block cannot be evicted.

        Returns:
            CXLKVBlock descriptor, or None if pool is full.
        """
        with self._lock:
            block_id = self._make_block_id(session_id, layer_indices, kv_data[:64])
            offset = self._allocate(len(kv_data))
            if offset is None:
                # Try LRU eviction
                if not self._evict_lru(len(kv_data)):
                    logger.warning("R12: CXL pool full, cannot store KV block")
                    return None
                offset = self._allocate(len(kv_data))
                if offset is None:
                    return None

            # Write to CXL pool
            success = self._backend.write(offset, kv_data)
            if not success:
                return None

            # Compute checksum
            import zlib
            checksum = zlib.crc32(kv_data) & 0xFFFFFFFF

            block = CXLKVBlock(
                block_id=block_id,
                session_id=session_id,
                layer_indices=layer_indices,
                seq_len=seq_len or len(kv_data) // max(1, len(layer_indices) * 2048),
                dtype=dtype,
                cxl_offset=offset,
                is_pinned=pin,
                ref_count=1,
                checksum=checksum,
            )

            self._blocks[block_id] = block
            self._session_blocks.setdefault(session_id, []).append(block_id)

            # Update stats
            self._stats.block_count += 1
            self._stats.used_bytes += len(kv_data)
            self._stats.bytes_transferred_in += len(kv_data)

            return block

    def load_kv(
        self,
        session_id: str,
        layer_indices: list[int],
        validate_checksum: bool = True,
    ) -> bytes | None:
        """Load KV data for a session from the CXL pool.

        Args:
            session_id: Session identifier.
            layer_indices: Layers to load.
            validate_checksum: If True, validate CRC32 integrity.

        Returns:
            Raw KV bytes, or None if not found or checksum failed.
        """
        with self._lock:
            block_ids = self._session_blocks.get(session_id, [])
            for block_id in block_ids:
                block = self._blocks.get(block_id)
                if block is None:
                    continue
                if set(block.layer_indices) != set(layer_indices):
                    continue

                # Read from CXL pool
                data_size = block.size_bytes
                data = self._backend.read(block.cxl_offset, data_size)

                if validate_checksum:
                    import zlib
                    actual_checksum = zlib.crc32(data) & 0xFFFFFFFF
                    if actual_checksum != block.checksum:
                        logger.error(
                            f"R12: Checksum mismatch for block {block_id[:8]} "
                            f"(stored={block.checksum:08x}, actual={actual_checksum:08x})"
                        )
                        self._stats.cache_misses += 1
                        return None

                block.touch()
                self._stats.cache_hits += 1
                self._stats.bytes_transferred_out += len(data)
                return data

            self._stats.cache_misses += 1
            return None

    def migrate_session(
        self,
        src_node_id: str,
        dst_node_id: str,
        session_id: str,
        seq_len: int = 0,
    ) -> dict[str, Any]:
        """Migrate a session's KV cache from source to destination node.

        Uses TraCT policy to decide transfer vs recompute. If transfer is chosen,
        the destination node reads directly from the shared CXL pool (zero-copy).

        Returns:
            Migration report with decision, timing, and TTFT reduction estimate.
        """
        with self._lock:
            block_ids = self._session_blocks.get(session_id, [])
            if not block_ids:
                return {
                    "decision": "recompute",
                    "reason": "No KV blocks found in pool for this session",
                    "session_id": session_id,
                }

            # Compute total KV size
            total_bytes = sum(
                self._blocks[bid].size_bytes
                for bid in block_ids
                if bid in self._blocks
            )

            # TraCT decision
            should_transfer, transfer_ms, recompute_ms = self.tract.should_transfer(
                kv_size_bytes=total_bytes,
                seq_len=seq_len,
                is_same_rack=(src_node_id.split("-")[0] == dst_node_id.split("-")[0]),
            )

            if should_transfer:
                # Transfer: dst node accesses CXL pool directly (shared address space)
                # In real CXL, this is just a pointer dereference — no data copy needed
                ttft_reduction_ms = max(0.0, recompute_ms - transfer_ms)
                self._stats.cache_hits += 1
                self._stats.ttft_reductions_ms.append(ttft_reduction_ms)
                decision = "transfer"
                logger.info(
                    f"R12: Session {session_id[:8]} migrated "
                    f"{src_node_id}→{dst_node_id} via CXL transfer "
                    f"({total_bytes / 1e6:.1f} MB, {transfer_ms:.1f} ms)"
                )
            else:
                self._stats.cache_misses += 1
                ttft_reduction_ms = 0.0
                decision = "recompute"
                logger.info(
                    f"R12: Session {session_id[:8]} will recompute "
                    f"(recompute={recompute_ms:.1f} ms < transfer={transfer_ms:.1f} ms)"
                )

            return {
                "decision": decision,
                "session_id": session_id,
                "src_node": src_node_id,
                "dst_node": dst_node_id,
                "kv_size_bytes": total_bytes,
                "kv_size_mb": round(total_bytes / 1e6, 2),
                "transfer_time_ms": round(transfer_ms, 2),
                "recompute_time_ms": round(recompute_ms, 2),
                "ttft_reduction_ms": round(ttft_reduction_ms, 2),
                "block_count": len(block_ids),
                "research_basis": "TraCT CXL arXiv 2026 + NIKA SCITEPRESS 2026",
            }

    def release_session(self, session_id: str) -> int:
        """Release all KV blocks for a session. Returns number of blocks freed."""
        with self._lock:
            block_ids = self._session_blocks.pop(session_id, [])
            freed = 0
            for block_id in block_ids:
                block = self._blocks.pop(block_id, None)
                if block and not block.is_pinned:
                    block.release()
                    if block.ref_count <= 0:
                        self._free_offsets.append(block.cxl_offset)
                        self._stats.used_bytes = max(0, self._stats.used_bytes - block.size_bytes)
                        self._stats.block_count -= 1
                        freed += 1
            self._stats.session_count = len(self._session_blocks)
            return freed

    def defragment(self) -> dict[str, Any]:
        """Compact the CXL pool to eliminate fragmentation.

        Moves blocks to lower offsets, coalescing free regions.
        In real CXL hardware, this is done by the BlueField-4 DPU.
        """
        with self._lock:
            start = time.perf_counter()
            moved = 0
            # Sort blocks by offset and compact
            live_blocks = sorted(
                [(b.cxl_offset, b) for b in self._blocks.values() if b.cxl_offset >= 0],
                key=lambda x: x[0],
            )
            new_offset = self._next_offset  # reset to data region start

            for old_offset, block in live_blocks:
                if old_offset != new_offset:
                    # Move block
                    data = self._backend.read(old_offset, block.size_bytes)
                    if data:
                        self._backend.write(new_offset, data)
                        block.cxl_offset = new_offset
                        moved += 1
                new_offset += max(CXL_BLOCK_BYTES, block.size_bytes)

            # Reclaim free space
            self._free_offsets.clear()
            self._stats.defrag_runs += 1
            duration_ms = (time.perf_counter() - start) * 1000

            result = {
                "blocks_moved": moved,
                "duration_ms": round(duration_ms, 1),
                "free_bytes_recovered": max(0, new_offset - self._next_offset),
                "defrag_run": self._stats.defrag_runs,
            }
            logger.info(f"R12: Defrag complete — {moved} blocks moved in {duration_ms:.1f} ms")
            return result

    def get_stats(self) -> dict[str, Any]:
        """Return pool statistics."""
        with self._lock:
            d = self._stats.to_dict()
            d["backend_mode"] = self._backend.mode
            d["pool_size_gb"] = self.pool_size_gb
            d["tract_policy"] = {
                "cxl_bandwidth_gbs": self.tract.cxl_bandwidth_gbs,
                "prefill_tps": self.tract.prefill_tps,
            }
            d["research_basis"] = [
                "TraCT CXL arXiv 2026",
                "NVIDIA CMX 2026",
                "CXL 3.0 Specification",
                "DUAL-BLADE NVMe-direct arXiv 2026",
            ]
            return d

    # ─────── Internal helpers ─────────────────────────────────────────────────

    def _make_block_id(self, session_id: str, layer_indices: list[int], kv_prefix: bytes) -> str:
        payload = f"{session_id}|{sorted(layer_indices)}|".encode() + kv_prefix
        return hashlib.sha256(payload).hexdigest()

    def _allocate(self, size: int) -> int | None:
        """Allocate space in the CXL pool. Returns byte offset, or None if full."""
        pool_max = int(self.pool_size_gb * 1024 ** 3)

        # Try reusing a free slot (simple first-fit)
        for i, offset in enumerate(self._free_offsets):
            if self._next_offset - offset >= size:
                self._free_offsets.pop(i)
                return offset

        # Allocate from end
        if self._next_offset + size > pool_max:
            return None
        offset = self._next_offset
        self._next_offset += max(CXL_BLOCK_BYTES, size)
        return offset

    def _evict_lru(self, needed_bytes: int) -> bool:
        """Evict LRU unpinned blocks until needed_bytes are freed."""
        evictable = [
            b for b in self._blocks.values()
            if not b.is_pinned and b.ref_count <= 0
        ]
        evictable.sort(key=lambda b: b.last_access)  # LRU first

        freed = 0
        for block in evictable:
            self._free_offsets.append(block.cxl_offset)
            freed += block.size_bytes
            session_id = block.session_id
            if session_id in self._session_blocks:
                try:
                    self._session_blocks[session_id].remove(block.block_id)
                except ValueError:
                    pass
            del self._blocks[block.block_id]
            self._stats.evictions += 1
            self._stats.block_count -= 1
            self._stats.used_bytes = max(0, self._stats.used_bytes - block.size_bytes)
            if freed >= needed_bytes:
                return True

        return freed >= needed_bytes

    def _start_defrag_daemon(self, interval_s: float) -> None:
        """Start background defragmentation thread."""
        def _daemon():
            while True:
                time.sleep(interval_s)
                try:
                    if self._stats.utilization > 0.80:
                        self.defragment()
                except Exception as e:
                    logger.debug(f"R12: Defrag daemon error: {e}")

        self._defrag_thread = threading.Thread(
            target=_daemon, daemon=True, name="cxl_defrag"
        )
        self._defrag_thread.start()
        logger.info(f"R12: Defrag daemon started (interval={interval_s}s)")

    def close(self) -> None:
        """Shut down the CXL pool and release resources."""
        self._backend.close()
        logger.info("R12: CXL pool closed")
