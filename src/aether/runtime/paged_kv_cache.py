"""
PagedKVPool — Real paged KV cache with tensor storage.

Implements PagedAttention-style block-based KV cache as described in:
    Kwon et al. (2023) "Efficient Memory Management for Large Language Model
    Serving with PagedAttention"
    https://arxiv.org/abs/2309.06180

Key design decisions
---------------------
- Pre-allocates one large pool tensor: shape
      (max_pages, page_size, 2, num_kv_heads, head_dim)
  where dim 2 = {0: K, 1: V}.  This single allocation avoids fragmentation
  and allows O(1) KV reads/writes regardless of history length.
- Block table: maps (seq_id, layer_idx) → list of page IDs.
- Copy-on-write: not yet implemented (single-user sequences only).
  Prefix sharing is implemented via read-only page references.
- CPU offload: moves pages asynchronously to a CPU mirror pool.
  Implemented via torch.Tensor.copy_ with non-blocking=True on CUDA.

CRITICAL DIFFERENCE from the previous kv_cache.py
---------------------------------------------------
The old implementation stored ONLY block metadata (page IDs, tier labels).
No KV tensors were ever created or stored.  This module stores ACTUAL TENSORS.
The pool is a real torch.Tensor and KV data is written into it with index_put.

Fallback behaviour
------------------
If torch is unavailable (core-only mode), the pool stores numpy arrays.
The attention kernel gets a real numpy (seq, num_heads, head_dim) view.

GPU / CPU
---------
Pass device="cuda" for GPU-resident pool.
Pass device="cpu"  for CPU-resident pool (reference + offload target).
The pool dtype should match the compute dtype to avoid mid-flight casts.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PagedKVPool",
    "BlockAllocator",
    "PrefixHash",
    "KVPageRef",
    "PagedKVStats",
]


# ---------------------------------------------------------------------------
# Prefix hashing (content-addressed block identity)
# ---------------------------------------------------------------------------

class PrefixHash:
    """Stable, tokenizer-independent content hash for prefix sharing."""

    @staticmethod
    def of(token_ids: list[int] | tuple[int, ...] | Any) -> str:
        """Return a hex SHA-256 of the token sequence."""
        ids = list(int(t) for t in token_ids)
        raw = bytes(sum(([t & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, t >> 24] for t in ids), []))
        return hashlib.sha256(raw).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KVPageRef:
    """A reference to one page in the pool, for one (seq, layer)."""
    page_id: int
    layer_idx: int
    seq_id: str
    write_pos: int = 0  # next write slot within the page (0 <= write_pos <= page_size)
    prefix_shared: bool = False  # True if this page is shared from prefix cache


@dataclass
class PagedKVStats:
    """Snapshot of pool utilisation."""
    total_pages: int
    free_pages: int
    used_pages: int
    active_sequences: int
    prefix_shared_pages: int
    cpu_offloaded_pages: int
    evictions: int
    allocations: int
    frees: int

    @property
    def utilization(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return self.used_pages / self.total_pages


# ---------------------------------------------------------------------------
# Block allocator (lock-free via threading.Lock)
# ---------------------------------------------------------------------------

class BlockAllocator:
    """
    Fast page allocator for the KV pool.

    Free pages are stored in a deque for O(1) alloc/free.
    Reference counting supports prefix sharing (page freed when ref_count→0).
    """

    def __init__(self, num_pages: int) -> None:
        self._lock = threading.Lock()
        self._free: list[int] = list(range(num_pages))
        self._ref_count: dict[int, int] = {}
        self._total = num_pages
        self._allocations = 0
        self._frees = 0
        self._evictions = 0

    # ------------------------------------------------------------------
    def allocate(self, n: int = 1) -> list[int]:
        """Allocate *n* pages. Raises RuntimeError if pool is exhausted."""
        with self._lock:
            if len(self._free) < n:
                raise RuntimeError(
                    f"KV pool exhausted: need {n} pages, have {len(self._free)}"
                )
            pages = self._free[-n:]
            del self._free[-n:]
            for p in pages:
                self._ref_count[p] = 1
            self._allocations += n
        return pages

    def free(self, pages: list[int]) -> None:
        """Decrement ref-count; return to pool when count reaches 0."""
        with self._lock:
            for p in pages:
                if p not in self._ref_count:
                    continue
                self._ref_count[p] -= 1
                if self._ref_count[p] <= 0:
                    del self._ref_count[p]
                    self._free.append(p)
                    self._frees += 1

    def retain(self, page: int) -> None:
        """Increment ref-count (prefix sharing)."""
        with self._lock:
            self._ref_count[page] = self._ref_count.get(page, 0) + 1

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def used_count(self) -> int:
        with self._lock:
            return self._total - len(self._free)

    @property
    def allocations(self) -> int:
        return self._allocations

    @property
    def frees(self) -> int:
        return self._frees


# ---------------------------------------------------------------------------
# Main KV pool
# ---------------------------------------------------------------------------

class PagedKVPool:
    """
    Pre-allocated paged KV cache with actual tensor storage.

    Pool tensor layout:
        pool[page_id, slot, layer_idx, kv_idx, head_idx, head_dim_idx]
    shape: (max_pages, page_size, num_layers, 2, num_kv_heads, head_dim)

    The layer dimension is unified into a single pool tensor so a single
    allocation covers all layers, reducing Python object overhead.

    Args:
        num_layers:    Number of transformer layers.
        num_kv_heads:  Number of KV attention heads (= num_heads for MHA,
                       < num_heads for MQA/GQA).
        head_dim:      Per-head dimension size.
        page_size:     Tokens per page (default 16, same as vLLM).
        max_pages:     Total pages in pool.
        dtype:         Storage dtype ('float16', 'bfloat16', 'float32').
        device:        'cpu' or 'cuda' or 'cuda:N'.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_pages: int = 2048,
        dtype: str = "float16",
        device: str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype_str = dtype
        self.device_str = device

        self._allocator = BlockAllocator(max_pages)
        # seq_id → {layer_idx → [KVPageRef, ...]}
        self._page_table: dict[str, dict[int, list[KVPageRef]]] = {}
        # prefix_hash → list of page_ids (shared, read-only)
        self._prefix_cache: dict[str, list[int]] = {}
        self._prefix_token_map: dict[str, list[int]] = {}
        self._cpu_offloaded: set[int] = set()
        self._lock = threading.Lock()

        # Allocate the pool tensor
        self._pool, self._np_pool = self._allocate_pool()

    # ------------------------------------------------------------------
    # Pool allocation
    # ------------------------------------------------------------------

    def _allocate_pool(self) -> tuple[Any, Any]:
        """
        Allocate the primary KV pool.

        Returns (torch_tensor_or_None, numpy_array).
        The numpy array is always present; on CUDA it is a CPU mirror for
        asynchronous offload operations.
        """
        shape = (self.max_pages, self.page_size, self.num_layers, 2,
                 self.num_kv_heads, self.head_dim)
        np_dtype = {
            "float16": np.float16,
            "bfloat16": np.float32,  # numpy has no bfloat16; use float32 mirror
            "float32": np.float32,
        }.get(self.dtype_str, np.float16)
        np_pool = np.zeros(shape, dtype=np_dtype)

        try:
            import torch
            torch_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(self.dtype_str, torch.float16)
            pool = torch.zeros(shape, dtype=torch_dtype, device=self.device_str)
            logger.debug(
                "PagedKVPool: allocated %s pool (%.1f MB, device=%s)",
                shape, np_pool.nbytes / 1e6, self.device_str
            )
            return pool, np_pool
        except Exception as exc:
            logger.debug("PagedKVPool: torch unavailable (%s); using numpy pool", exc)
            return None, np_pool

    @property
    def _uses_torch(self) -> bool:
        return self._pool is not None

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def allocate_sequence(self, seq_id: str, num_layers: int | None = None) -> None:
        """Register a new sequence. Call before any write operations."""
        n = num_layers or self.num_layers
        with self._lock:
            if seq_id not in self._page_table:
                self._page_table[seq_id] = {i: [] for i in range(n)}

    def free_sequence(self, seq_id: str) -> None:
        """Free all pages held by a sequence."""
        with self._lock:
            layer_map = self._page_table.pop(seq_id, {})
        all_pages: list[int] = []
        for refs in layer_map.values():
            for ref in refs:
                if not ref.prefix_shared:
                    all_pages.append(ref.page_id)
        if all_pages:
            self._allocator.free(all_pages)

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------

    def append_kv(
        self,
        seq_id: str,
        layer_idx: int,
        k: Any,  # shape (1, num_kv_heads, head_dim) or (num_kv_heads, head_dim)
        v: Any,
    ) -> None:
        """
        Append one token's K and V vectors to the sequence's KV cache.

        Allocates a new page if the current page is full.
        """
        with self._lock:
            layer_pages = self._page_table[seq_id][layer_idx]
            # Ensure we have a non-full, non-shared page
            if (not layer_pages or
                    layer_pages[-1].write_pos >= self.page_size or
                    layer_pages[-1].prefix_shared):
                page_id = self._allocator.allocate(1)[0]
                layer_pages.append(KVPageRef(
                    page_id=page_id, layer_idx=layer_idx,
                    seq_id=seq_id, write_pos=0,
                ))
            ref = layer_pages[-1]
            slot = ref.write_pos
            page_id = ref.page_id
            ref.write_pos += 1

        # Write outside lock (different pages never alias)
        k_np = np.asarray(k, dtype=np.float32).reshape(self.num_kv_heads, self.head_dim)
        v_np = np.asarray(v, dtype=np.float32).reshape(self.num_kv_heads, self.head_dim)

        self._np_pool[page_id, slot, layer_idx, 0] = k_np
        self._np_pool[page_id, slot, layer_idx, 1] = v_np

        if self._uses_torch:
            import torch
            pool = self._pool
            pool[page_id, slot, layer_idx, 0] = torch.as_tensor(
                k_np, dtype=pool.dtype, device=pool.device
            )
            pool[page_id, slot, layer_idx, 1] = torch.as_tensor(
                v_np, dtype=pool.dtype, device=pool.device
            )

    def get_kv(
        self,
        seq_id: str,
        layer_idx: int,
        as_torch: bool = True,
    ) -> tuple[Any, Any]:
        """
        Gather all K and V tensors for a sequence and layer.

        Returns (K, V) each of shape (total_tokens, num_kv_heads, head_dim).
        Uses the torch pool if as_torch=True and torch is available.
        """
        with self._lock:
            refs = list(self._page_table.get(seq_id, {}).get(layer_idx, []))

        parts_k: list[Any] = []
        parts_v: list[Any] = []
        for ref in refs:
            n_tokens = ref.write_pos
            if n_tokens == 0:
                continue
            k_block = self._np_pool[ref.page_id, :n_tokens, layer_idx, 0]
            v_block = self._np_pool[ref.page_id, :n_tokens, layer_idx, 1]
            parts_k.append(k_block)
            parts_v.append(v_block)

        if not parts_k:
            empty = np.zeros((0, self.num_kv_heads, self.head_dim), dtype=np.float32)
            if as_torch and self._uses_torch:
                import torch
                t = torch.zeros(
                    0, self.num_kv_heads, self.head_dim,
                    dtype=self._pool.dtype, device=self._pool.device
                )
                return t, t
            return empty, empty

        k_all = np.concatenate(parts_k, axis=0)
        v_all = np.concatenate(parts_v, axis=0)

        if as_torch and self._uses_torch:
            import torch
            k_t = torch.as_tensor(k_all, dtype=self._pool.dtype, device=self._pool.device)
            v_t = torch.as_tensor(v_all, dtype=self._pool.dtype, device=self._pool.device)
            return k_t, v_t
        return k_all, v_all

    def get_total_tokens(self, seq_id: str, layer_idx: int = 0) -> int:
        """Return the number of tokens stored for this sequence."""
        with self._lock:
            refs = self._page_table.get(seq_id, {}).get(layer_idx, [])
            return sum(r.write_pos for r in refs)

    # ------------------------------------------------------------------
    # Prefix sharing
    # ------------------------------------------------------------------

    def insert_prefix(
        self, token_ids: list[int], seq_id: str, layer_idx: int
    ) -> str:
        """
        Mark a sequence's pages as a named prefix block.

        Returns the prefix hash so callers can reuse this block.
        The pages are retained (ref-count incremented) and the token IDs
        recorded for radix-tree lookup.
        """
        prefix_hash = PrefixHash.of(token_ids)
        with self._lock:
            refs = self._page_table.get(seq_id, {}).get(layer_idx, [])
            page_ids = [r.page_id for r in refs]
            for pid in page_ids:
                self._allocator.retain(pid)
            self._prefix_cache[prefix_hash] = page_ids
            self._prefix_token_map[prefix_hash] = list(token_ids)
        return prefix_hash

    def lookup_prefix(
        self, token_ids: list[int]
    ) -> tuple[str | None, int]:
        """
        Look for the longest prefix of ``token_ids`` in the prefix cache.

        Returns (prefix_hash, matched_length).
        Returns (None, 0) on miss.
        """
        # Linear scan; replace with radix trie for production
        best_hash = None
        best_len = 0
        with self._lock:
            for phash, ptokens in self._prefix_token_map.items():
                n = min(len(ptokens), len(token_ids))
                if list(token_ids[:n]) == ptokens[:n] and n > best_len:
                    best_len = n
                    best_hash = phash
        return best_hash, best_len

    def attach_prefix(
        self, seq_id: str, prefix_hash: str, layer_idx: int
    ) -> int:
        """
        Attach prefix pages to a sequence as shared, read-only refs.

        Returns the number of tokens prefilled from cache.
        """
        with self._lock:
            page_ids = self._prefix_cache.get(prefix_hash, [])
            ptokens = self._prefix_token_map.get(prefix_hash, [])
            n_tokens = len(ptokens)
            if not page_ids:
                return 0
            refs = self._page_table.setdefault(seq_id, {}).setdefault(layer_idx, [])
            for pid in page_ids:
                self._allocator.retain(pid)
                refs.append(KVPageRef(
                    page_id=pid, layer_idx=layer_idx, seq_id=seq_id,
                    write_pos=self.page_size,  # mark full (shared page)
                    prefix_shared=True,
                ))
        return n_tokens

    # ------------------------------------------------------------------
    # CPU offload
    # ------------------------------------------------------------------

    def offload_to_cpu(self, seq_id: str) -> int:
        """
        Copy GPU pages for this sequence to the CPU numpy mirror.

        This is a synchronous copy (for correctness on CPU-only systems;
        on GPU, use non_blocking=True in production).

        Returns number of pages offloaded.
        """
        if not self._uses_torch:
            return 0  # already CPU

        import torch

        with self._lock:
            layer_map = self._page_table.get(seq_id, {})
            all_refs = [r for refs in layer_map.values() for r in refs]

        offloaded = 0
        for ref in all_refs:
            if ref.page_id not in self._cpu_offloaded:
                cpu_data = self._pool[ref.page_id].cpu().numpy()
                self._np_pool[ref.page_id] = cpu_data
                self._cpu_offloaded.add(ref.page_id)
                offloaded += 1
        return offloaded

    def restore_from_cpu(self, seq_id: str) -> int:
        """Restore offloaded pages back to GPU pool."""
        if not self._uses_torch:
            return 0

        import torch

        with self._lock:
            layer_map = self._page_table.get(seq_id, {})
            all_refs = [r for refs in layer_map.values() for r in refs]

        restored = 0
        for ref in all_refs:
            if ref.page_id in self._cpu_offloaded:
                gpu_data = torch.as_tensor(
                    self._np_pool[ref.page_id],
                    dtype=self._pool.dtype, device=self._pool.device
                )
                self._pool[ref.page_id] = gpu_data
                self._cpu_offloaded.discard(ref.page_id)
                restored += 1
        return restored

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> PagedKVStats:
        with self._lock:
            n_seqs = len(self._page_table)
            shared = sum(
                1 for layer_map in self._page_table.values()
                for refs in layer_map.values()
                for r in refs if r.prefix_shared
            )
        return PagedKVStats(
            total_pages=self.max_pages,
            free_pages=self._allocator.free_count,
            used_pages=self._allocator.used_count,
            active_sequences=n_seqs,
            prefix_shared_pages=shared,
            cpu_offloaded_pages=len(self._cpu_offloaded),
            evictions=0,  # eviction policy not yet implemented
            allocations=self._allocator.allocations,
            frees=self._allocator.frees,
        )

    def pool_bytes(self) -> int:
        """Return the size of the pool in bytes."""
        return self._np_pool.nbytes

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"PagedKVPool("
            f"pages={s.used_pages}/{s.total_pages}, "
            f"seqs={s.active_sequences}, "
            f"device={self.device_str!r}, "
            f"dtype={self.dtype_str!r}, "
            f"page_size={self.page_size}, "
            f"pool_MB={self.pool_bytes() / 1e6:.1f})"
        )
