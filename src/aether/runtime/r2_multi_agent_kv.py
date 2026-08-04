"""
R2 — Multi-Agent KV Coordination Layer.

In multi-agent LLM systems, multiple agents share common prefixes
(system prompt, tool schemas, shared context).  Without coordination, each
agent independently computes and stores KV pairs for the shared prefix,
wasting N× GPU memory and N× compute.

R2 implements a *KV Coordination Registry* that:
  1. Maintains a session registry keyed by prefix hash.
  2. When a new agent session starts with a shared prefix, the registry
     serves cached KV pointers (zero-copy via CUDA IPC / RDMA memory mapping).
  3. Implements copy-on-write (CoW) for private KV divergence after the
     shared prefix — each agent gets a private KV tail appended to the shared head.
  4. Supports cross-agent KV eviction via reference counting: shared KV blocks
     are evicted only when all referencing agents have completed.

KV sharing strategies:
  - **Prefix sharing**: Common system prompt / tool schema prefix.
  - **RadixAttention**: Radix tree-based prefix caching (SGLang 2024).
  - **Cross-request sharing**: Reuse KV across independent requests with
    identical prefixes (via content-addressed KV cache).

Research basis:
  - SGLang (2024): RadixAttention for cross-request KV reuse.
  - vLLM prefix caching (2024): hash-based KV block sharing.
  - Multi-agent KV coordination (2025): CoW for agent divergence.
  - MemServe (2025): disaggregated memory pool for multi-agent KV.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SharedKVBlock:
    """A KV cache block shared across multiple agent sessions.

    Attributes:
        block_id: Unique block identifier (prefix hash).
        ref_count: Number of active agent sessions referencing this block.
        kv_data: Opaque KV data (tensor handle or byte buffer).
        seq_len: Number of tokens in this shared KV block.
        last_access_ts: Timestamp of last access (for LRU eviction).
    """

    block_id: str
    ref_count: int = 0
    kv_data: Any = None
    seq_len: int = 0
    last_access_ts: float = field(default_factory=time.time)
    is_pinned: bool = False  # True = cannot be evicted


@dataclass
class AgentKVSession:
    """Per-agent KV session with a shared head and private tail.

    Attributes:
        session_id: Unique session identifier.
        shared_block_id: Block ID of the shared prefix KV (or None).
        private_kv: Private KV tail for diverged tokens.
        agent_token_offset: Number of shared tokens (divergence point).
    """

    session_id: str
    shared_block_id: str | None = None
    private_kv: Any = None
    agent_token_offset: int = 0
    created_ts: float = field(default_factory=time.time)
    last_update_ts: float = field(default_factory=time.time)


class MultiAgentKVCoordinator:
    """Runtime R2: Multi-Agent KV Coordination Layer.

    Manages shared KV blocks across agent sessions and implements
    copy-on-write for private divergence.
    """

    def __init__(
        self,
        max_shared_blocks: int = 10_000,
        max_sessions: int = 1_000,
        eviction_policy: str = "lru",
    ) -> None:
        self.max_shared_blocks = max_shared_blocks
        self.max_sessions = max_sessions
        self.eviction_policy = eviction_policy

        self._shared_blocks: dict[str, SharedKVBlock] = {}
        self._sessions: dict[str, AgentKVSession] = {}
        self._lock = threading.RLock()
        self._stats = _CoordStats()

    def register_session(
        self,
        session_id: str,
        prefix_tokens: list[int],
        prefix_kv: Any = None,
    ) -> AgentKVSession:
        """Register a new agent session and deduplicate its prefix KV.

        If another session already computed KV for this prefix, the new
        session gets a pointer to the existing shared block (zero-copy).

        Args:
            session_id: Unique ID for this agent session.
            prefix_tokens: Token IDs of the shared prefix.
            prefix_kv: Pre-computed KV for the prefix (or None to request computation).

        Returns:
            AgentKVSession with shared_block_id pointing to the deduped KV.
        """
        prefix_hash = _hash_token_sequence(prefix_tokens)

        with self._lock:
            # Check if shared block exists.
            if prefix_hash in self._shared_blocks:
                block = self._shared_blocks[prefix_hash]
                block.ref_count += 1
                block.last_access_ts = time.time()
                logger.debug(
                    "R2: Session %r reuses shared block %s (ref_count=%d).",
                    session_id[:8],
                    prefix_hash[:8],
                    block.ref_count,
                )
                self._stats.cache_hits += 1
            else:
                # Create new shared block.
                block = SharedKVBlock(
                    block_id=prefix_hash,
                    ref_count=1,
                    kv_data=prefix_kv,
                    seq_len=len(prefix_tokens),
                    last_access_ts=time.time(),
                )
                self._shared_blocks[prefix_hash] = block
                self._stats.blocks_created += 1

                # Evict if over capacity.
                if len(self._shared_blocks) > self.max_shared_blocks:
                    self._evict_one()

            session = AgentKVSession(
                session_id=session_id,
                shared_block_id=prefix_hash,
                private_kv=None,
                agent_token_offset=len(prefix_tokens),
            )
            self._sessions[session_id] = session
            self._stats.sessions_created += 1

        return session

    def append_private_kv(
        self,
        session_id: str,
        new_tokens: list[int],
        new_kv: Any,
    ) -> None:
        """Append private (post-divergence) KV tokens to a session.

        Implements copy-on-write: shared block remains unchanged, only the
        private tail grows.  This enables O(1) divergence with zero copying
        of the shared prefix.

        Args:
            session_id: Session to update.
            new_tokens: Newly generated token IDs (after divergence point).
            new_kv: KV tensors for the new tokens.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning("R2: append_private_kv: unknown session %r.", session_id)
                return

            # CoW: accumulate private KV without touching shared block.
            if session.private_kv is None:
                session.private_kv = _make_kv_buffer(new_kv, capacity=512)
            else:
                _append_to_kv_buffer(session.private_kv, new_kv)

            session.last_update_ts = time.time()

    def get_full_kv(self, session_id: str) -> tuple[Any, Any]:
        """Return (shared_kv, private_kv) for a session.

        The caller concatenates shared + private KV for attention computation.
        This is a zero-copy operation — both are pointers to existing memory.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None, None

            shared_kv = None
            if session.shared_block_id and session.shared_block_id in self._shared_blocks:
                shared_kv = self._shared_blocks[session.shared_block_id].kv_data

            return shared_kv, session.private_kv

    def release_session(self, session_id: str) -> None:
        """Release a session and decrement its shared block ref_count.

        If ref_count reaches 0 and the block is not pinned, it becomes
        eligible for LRU eviction.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return

            if session.shared_block_id and session.shared_block_id in self._shared_blocks:
                block = self._shared_blocks[session.shared_block_id]
                block.ref_count = max(0, block.ref_count - 1)
                if block.ref_count == 0 and not block.is_pinned:
                    logger.debug(
                        "R2: Shared block %s has ref_count=0, eligible for eviction.",
                        session.shared_block_id[:8],
                    )

            self._stats.sessions_released += 1

    def update_shared_kv(self, block_id: str, kv_data: Any) -> None:
        """Update the KV data for an existing shared block (e.g., after computation)."""
        with self._lock:
            if block_id in self._shared_blocks:
                self._shared_blocks[block_id].kv_data = kv_data
                self._shared_blocks[block_id].last_access_ts = time.time()

    def _evict_one(self) -> None:
        """Evict one shared block according to the eviction policy."""
        if self.eviction_policy == "lru":
            # LRU: evict the block with the oldest last_access_ts and ref_count == 0.
            candidates = [
                (b.last_access_ts, bid)
                for bid, b in self._shared_blocks.items()
                if b.ref_count == 0 and not b.is_pinned
            ]
            if candidates:
                candidates.sort()
                _, evict_id = candidates[0]
                del self._shared_blocks[evict_id]
                self._stats.blocks_evicted += 1
                logger.debug("R2: Evicted LRU block %s.", evict_id[:8])

    @property
    def stats(self) -> "_CoordStats":
        """Return coordination statistics."""
        return self._stats

    def summary(self) -> dict[str, Any]:
        """Return a summary of the coordinator state."""
        with self._lock:
            return {
                "shared_blocks": len(self._shared_blocks),
                "active_sessions": len(self._sessions),
                "cache_hit_rate": (
                    self._stats.cache_hits / max(1, self._stats.sessions_created)
                ),
                "blocks_evicted": self._stats.blocks_evicted,
            }


class _CoordStats:
    __slots__ = ("cache_hits", "blocks_created", "blocks_evicted", "sessions_created", "sessions_released")

    def __init__(self) -> None:
        self.cache_hits = 0
        self.blocks_created = 0
        self.blocks_evicted = 0
        self.sessions_created = 0
        self.sessions_released = 0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hash_token_sequence(tokens: list[int]) -> str:
    """Compute a deterministic SHA-256 hash of a token ID sequence."""
    import struct
    packed = struct.pack(f"<{len(tokens)}I", *tokens)
    return hashlib.sha256(packed).hexdigest()


def _make_kv_buffer(kv: Any, capacity: int) -> Any:
    """Create a new KV buffer with the given initial KV data."""
    # In production: allocate a contiguous CUDA tensor buffer.
    # For planning / testing: use a list as a simple accumulator.
    if isinstance(kv, list):
        return list(kv)
    elif hasattr(kv, "tolist"):
        return kv.tolist()
    return [kv]


def _append_to_kv_buffer(buffer: Any, new_kv: Any) -> None:
    """Append new KV data to an existing buffer (in-place)."""
    if isinstance(buffer, list):
        if isinstance(new_kv, list):
            buffer.extend(new_kv)
        elif hasattr(new_kv, "tolist"):
            buffer.extend(new_kv.tolist())
        else:
            buffer.append(new_kv)
