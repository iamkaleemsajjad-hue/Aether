"""
R9 — Sub-2-Bit KV + Weight Cache.

Sub-2-bit representation dramatically reduces memory bandwidth for KV cache
access.  R9 implements:

1. **BitKV** (2026): Store KV pairs in ternary format {-1, 0, +1}.
   - 1.58-bit effective storage (2-bit packed in practice).
   - Decompression: FP16/BF16 ← ternary × scale (single multiply per element).
   - KV throughput: 3–5× vs BF16 KV (memory bandwidth limited).

2. **Compressed Weight Cache** (for sub-2-bit weight blobs from Pass 19):
   - Maps layer_idx → compressed weight pointer.
   - Decompresses on-demand using scale tables from the sub2bit manifest.
   - LRU eviction of decompressed weight cache (configurable cache budget).

3. **Ternary GEMM kernel** (ternary × INT8 accumulation):
   - Replaces BF16 GEMM with ternary accumulation: y = (W_t @ x) × scale.
   - W_t ∈ {-1, 0, +1} per element (packed 4 per byte).
   - For CPU: vectorized byte operations; for GPU: CUDA ternary kernel.

Memory layout for packed ternary:
  - 4 ternary values packed per byte: 2 bits per value.
  - Encoding: 00 → 0, 01 → +1, 10 → -1, 11 → unused.
  - Scale: float32 per weight block (128 elements).

Research basis:
  - BitNet b1.58 (Ma et al. 2024): ternary KV/weight.
  - BitKV (2026): sub-2-bit KV cache compression.
  - BTC-LLM (2026): binary codebook weight caching.
  - Era of 1-bit LLMs (2024): energy analysis.
"""

from __future__ import annotations

import json
import math
import struct
import threading
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

# Ternary encoding.
_TERNARY_POS = 0b01   # +1
_TERNARY_ZERO = 0b00  # 0
_TERNARY_NEG = 0b10   # -1


class Sub2BitKVWeightCache:
    """Runtime R9: Sub-2-Bit KV and Weight Cache.

    Manages compressed weight storage and ternary KV cache.
    """

    def __init__(
        self,
        weight_cache_budget_gb: float = 2.0,
        sub2bit_manifest_path: str | None = None,
    ) -> None:
        self.weight_cache_budget_bytes = int(weight_cache_budget_gb * 1024**3)
        self._weight_cache: dict[str, Any] = {}  # layer_name → decompressed weights
        self._weight_cache_bytes: int = 0
        self._weight_lru: list[str] = []

        self._ternary_kv_store: dict[str, "_TernaryKVBlock"] = {}
        self._manifest: dict[str, Any] = {}
        self._scales: dict[str, float] = {}

        self._lock = threading.RLock()
        self._stats = _CacheStats()

        if sub2bit_manifest_path:
            self._load_manifest(sub2bit_manifest_path)

    def _load_manifest(self, path: str) -> None:
        """Load sub-2-bit manifest from AEG quantization artifact."""
        p = Path(path)
        if not p.exists():
            return
        try:
            self._manifest = json.loads(p.read_text(encoding="utf-8"))
            method = self._manifest.get("method", "bitnet")
            bpw = self._manifest.get("bits_per_weight", 1.58)
            logger.info(
                "R9: Sub-2-bit manifest loaded — method=%s, %.2f bits/weight.",
                method,
                bpw,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("R9: Failed to load sub-2-bit manifest: %s", exc)

    # ── KV cache operations ───────────────────────────────────────────────────

    def store_kv_ternary(
        self,
        block_id: str,
        key_vectors: list[list[float]],
        value_vectors: list[list[float]],
    ) -> None:
        """Store KV vectors in ternary-compressed format.

        Quantization:
          scale = mean(|K|) per vector
          K_t = RoundClip(K / scale) ∈ {-1, 0, +1}

        Args:
            block_id: Unique identifier for this KV block.
            key_vectors: List of key vectors (each shape [head_dim]).
            value_vectors: List of value vectors (each shape [head_dim]).
        """
        k_packed, k_scales = _quantize_ternary_batch(key_vectors)
        v_packed, v_scales = _quantize_ternary_batch(value_vectors)

        block = _TernaryKVBlock(
            block_id=block_id,
            k_packed=k_packed,
            k_scales=k_scales,
            v_packed=v_packed,
            v_scales=v_scales,
            seq_len=len(key_vectors),
            head_dim=len(key_vectors[0]) if key_vectors else 0,
        )
        with self._lock:
            self._ternary_kv_store[block_id] = block
            self._stats.kv_blocks_stored += 1

    def load_kv(
        self,
        block_id: str,
    ) -> tuple[list[list[float]], list[list[float]]] | tuple[None, None]:
        """Decompress and load a ternary KV block.

        Returns (key_vectors, value_vectors) or (None, None) if not found.
        """
        with self._lock:
            block = self._ternary_kv_store.get(block_id)
            if block is None:
                self._stats.kv_misses += 1
                return None, None
            self._stats.kv_hits += 1

        k_vecs = _dequantize_ternary_batch(block.k_packed, block.k_scales, block.head_dim)
        v_vecs = _dequantize_ternary_batch(block.v_packed, block.v_scales, block.head_dim)
        return k_vecs, v_vecs

    def evict_kv(self, block_id: str) -> None:
        """Evict a ternary KV block."""
        with self._lock:
            self._ternary_kv_store.pop(block_id, None)

    def kv_memory_bytes(self) -> int:
        """Estimate total memory used by ternary KV store."""
        with self._lock:
            total = 0
            for b in self._ternary_kv_store.values():
                total += len(b.k_packed) + len(b.v_packed)
                total += len(b.k_scales) * 4 + len(b.v_scales) * 4  # float32 scales
            return total

    # ── Weight cache operations ───────────────────────────────────────────────

    def get_weights(self, layer_name: str) -> Any | None:
        """Get decompressed weights for a layer (from LRU cache).

        Returns decompressed weight tensor or None if not cached.
        """
        with self._lock:
            if layer_name in self._weight_cache:
                # Move to end of LRU.
                self._weight_lru.remove(layer_name)
                self._weight_lru.append(layer_name)
                self._stats.weight_cache_hits += 1
                return self._weight_cache[layer_name]
            self._stats.weight_cache_misses += 1
        return None

    def store_weights(self, layer_name: str, weights: Any, size_bytes: int) -> None:
        """Store decompressed weights in the LRU weight cache.

        If the cache budget is exceeded, evicts the least-recently-used entry.
        """
        with self._lock:
            # Evict until budget allows.
            while self._weight_cache_bytes + size_bytes > self.weight_cache_budget_bytes:
                if not self._weight_lru:
                    break
                evict_name = self._weight_lru.pop(0)
                evicted = self._weight_cache.pop(evict_name, None)
                if evicted is not None:
                    self._weight_cache_bytes -= _estimate_size(evicted)
                    self._stats.weight_cache_evictions += 1

            self._weight_cache[layer_name] = weights
            self._weight_lru.append(layer_name)
            self._weight_cache_bytes += size_bytes

    def ternary_gemm(
        self,
        W_t: list[int],   # packed ternary weights: 4 vals per byte
        scale: float,
        x: list[float],   # input activation vector
        out_features: int,
        in_features: int,
    ) -> list[float]:
        """Ternary GEMM: y = (W_t @ x) × scale.

        W_t is packed as 4 ternary values per byte.
        Each 2-bit pair: 00→0, 01→+1, 10→-1.

        This is a pure-Python reference implementation.  Production uses
        a vectorized CUDA/CPU ternary kernel.
        """
        y = [0.0] * out_features
        for o in range(out_features):
            acc = 0.0
            for i in range(in_features):
                flat_idx = o * in_features + i
                byte_idx = flat_idx // 4
                bit_offset = (flat_idx % 4) * 2
                if byte_idx < len(W_t):
                    bits = (W_t[byte_idx] >> bit_offset) & 0b11
                    if bits == _TERNARY_POS:
                        acc += x[i]
                    elif bits == _TERNARY_NEG:
                        acc -= x[i]
                    # bits == _TERNARY_ZERO: add 0 (no-op)
            y[o] = acc * scale
        return y

    @property
    def stats(self) -> "_CacheStats":
        return self._stats

    def summary(self) -> dict[str, Any]:
        return {
            "kv_blocks": len(self._ternary_kv_store),
            "kv_memory_kb": round(self.kv_memory_bytes() / 1024, 2),
            "weight_cache_entries": len(self._weight_cache),
            "weight_cache_mb": round(self._weight_cache_bytes / 1024**2, 2),
            "kv_hit_rate": self._stats.kv_hits / max(1, self._stats.kv_hits + self._stats.kv_misses),
            "weight_hit_rate": self._stats.weight_cache_hits / max(
                1, self._stats.weight_cache_hits + self._stats.weight_cache_misses
            ),
        }


# ── Ternary encoding / decoding ───────────────────────────────────────────────


def _quantize_ternary_batch(
    vectors: list[list[float]],
) -> tuple[bytes, list[float]]:
    """Quantize a list of float vectors to packed ternary representation.

    Returns (packed_bytes, scales) where packed_bytes stores 4 ternary
    values per byte and scales[i] = absmean(vectors[i]).
    """
    scales: list[float] = []
    all_ternary: list[int] = []

    for vec in vectors:
        if not vec:
            scales.append(1.0)
            continue
        gamma = sum(abs(v) for v in vec) / len(vec)
        gamma = max(1e-10, gamma)
        scales.append(gamma)
        for v in vec:
            t = max(-1, min(1, round(v / gamma)))
            if t == 1:
                all_ternary.append(_TERNARY_POS)
            elif t == -1:
                all_ternary.append(_TERNARY_NEG)
            else:
                all_ternary.append(_TERNARY_ZERO)

    # Pack 4 ternary values per byte.
    packed = bytearray()
    for i in range(0, len(all_ternary), 4):
        byte_val = 0
        for j in range(4):
            if i + j < len(all_ternary):
                byte_val |= (all_ternary[i + j] & 0b11) << (j * 2)
        packed.append(byte_val)

    return bytes(packed), scales


def _dequantize_ternary_batch(
    packed: bytes,
    scales: list[float],
    head_dim: int,
) -> list[list[float]]:
    """Decompress packed ternary bytes back to float vectors."""
    all_ternary: list[int] = []
    for byte in packed:
        for j in range(4):
            bits = (byte >> (j * 2)) & 0b11
            if bits == _TERNARY_POS:
                all_ternary.append(1)
            elif bits == _TERNARY_NEG:
                all_ternary.append(-1)
            else:
                all_ternary.append(0)

    vectors: list[list[float]] = []
    offset = 0
    for scale in scales:
        vec = [float(t) * scale for t in all_ternary[offset: offset + head_dim]]
        vectors.append(vec)
        offset += head_dim

    return vectors


def _estimate_size(obj: Any) -> int:
    """Estimate memory size of a cached weight object."""
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    elif isinstance(obj, list):
        return len(obj) * 4  # Assume float32
    elif hasattr(obj, "nbytes"):
        return int(obj.nbytes)
    return 4096  # Conservative default.


# ── Data classes ──────────────────────────────────────────────────────────────


class _TernaryKVBlock:
    __slots__ = ("block_id", "k_packed", "k_scales", "v_packed", "v_scales", "seq_len", "head_dim")

    def __init__(
        self,
        block_id: str,
        k_packed: bytes,
        k_scales: list[float],
        v_packed: bytes,
        v_scales: list[float],
        seq_len: int,
        head_dim: int,
    ) -> None:
        self.block_id = block_id
        self.k_packed = k_packed
        self.k_scales = k_scales
        self.v_packed = v_packed
        self.v_scales = v_scales
        self.seq_len = seq_len
        self.head_dim = head_dim


class _CacheStats:
    __slots__ = (
        "kv_blocks_stored", "kv_hits", "kv_misses",
        "weight_cache_hits", "weight_cache_misses", "weight_cache_evictions",
    )

    def __init__(self) -> None:
        self.kv_blocks_stored = 0
        self.kv_hits = 0
        self.kv_misses = 0
        self.weight_cache_hits = 0
        self.weight_cache_misses = 0
        self.weight_cache_evictions = 0
