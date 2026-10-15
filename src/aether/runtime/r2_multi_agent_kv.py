"""
R2 — Multi-Agent KV Coordination Layer.

In multi-agent LLM systems, multiple agents share common prefixes
(system prompt, tool schemas, shared context).  Without coordination, each
agent independently computes and stores KV pairs for the shared prefix,
wasting N× GPU memory and N× compute.

R2 implements a *KV Coordination Registry* that:
  1. Maintains a session registry keyed by prefix hash.
  2. When a new agent session starts with a shared prefix, the registry
     serves an in-process reference to compatible cached state. Execution
     adapters clone it before private divergence; no CUDA IPC/RDMA is implied.
  3. Keeps private KV state separate from the shared prefix. The actual copy
     and execution geometry belong to the selected inference adapter.
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
  These papers inform prefix reuse and isolation; this local registry does
  not establish their distributed transport or throughput results.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
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
    execution_variants: dict[str, SharedKVBlock] = field(default_factory=dict)
    """Validated execution-specific caches; never aliases unknown legacy state."""


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

    Tracks local shared references and separate per-agent private values.
    Execution adapters implement their own cache cloning and inference.
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
        self._variant_lru: OrderedDict[tuple[str, str], SharedKVBlock] = OrderedDict()
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

        The shared block remains unchanged while a Python private accumulator
        grows. This bookkeeping does not implement device copy-on-write or
        guarantee constant-time tensor divergence.

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

    def update_shared_kv(
        self,
        block_id: str,
        kv_data: Any,
        *,
        seq_len: int | None = None,
        execution_contract: dict[str, Any] | None = None,
    ) -> None:
        """Publish computed state under its prefix and optional weight contract.

        Bound variants are retained independently of unknown legacy state and
        share the configured variant capacity. If every retained variant is
        pinned, a new one remains request-local rather than exceeding the limit.
        """
        if seq_len is not None and (type(seq_len) is not int or seq_len < 0):
            raise ValueError("shared KV sequence length must be a non-negative integer")
        variant_key = self._execution_key(execution_contract)
        with self._lock:
            if block_id in self._shared_blocks:
                parent = self._shared_blocks[block_id]
                block = parent
                if variant_key is not None:
                    block = parent.execution_variants.get(variant_key)
                    if block is None:
                        if self.max_shared_blocks <= 0:
                            return
                        if len(self._variant_lru) >= self.max_shared_blocks and not self._evict_variant_one():
                            return
                        block = SharedKVBlock(block_id=block_id)
                        parent.execution_variants[variant_key] = block
                        self._variant_lru[block_id, variant_key] = block
                        self._stats.blocks_created += 1
                    self._variant_lru.move_to_end((block_id, variant_key))
                block.kv_data = kv_data
                if seq_len is not None:
                    block.seq_len = seq_len
                block.last_access_ts = time.time()
                parent.last_access_ts = block.last_access_ts

    def get_shared_kv(
        self, block_id: str, *, execution_contract: dict[str, Any] | None = None,
    ) -> tuple[Any, int]:
        """Return only the state bound to the selected execution contract.

        The reference is local; inference adapters own any required cloning or
        device transfer. A contract-bound lookup never consumes unbound state.
        """
        variant_key = self._execution_key(execution_contract)
        with self._lock:
            parent = self._shared_blocks.get(block_id)
            if parent is None:
                return None, 0
            block = parent
            if variant_key is not None:
                block = block.execution_variants.get(variant_key)
                if block is None:
                    return None, 0
                self._variant_lru.move_to_end((block_id, variant_key))
            block.last_access_ts = time.time()
            parent.last_access_ts = block.last_access_ts
            return block.kv_data, block.seq_len

    @staticmethod
    def _execution_key(contract: dict[str, Any] | None) -> str | None:
        if contract is None:
            return None
        if not isinstance(contract, dict) or not contract:
            raise ValueError("shared KV execution contract must be a nonempty object")
        from aether.runtime.session_contract import contract_key

        return contract_key(contract)

    def _evict_variant_one(self) -> bool:
        """Drop the oldest unpinned local variant; held references stay valid."""
        for key in self._variant_lru:
            parent = self._shared_blocks.get(key[0])
            if parent is not None and parent.is_pinned:
                continue
            if parent is not None:
                parent.execution_variants.pop(key[1], None)
            self._variant_lru.pop(key)
            self._stats.blocks_evicted += 1
            return True
        return False

    def hash_prefix(self, prefix_text: str) -> str:
        """Compute a stable hash for a text prefix string.

        Used by Runtime.multi_agent_session() to identify shared system prompts
        without tokenizing. SHA-256 of the normalized UTF-8 prefix.

        Args:
            prefix_text: System prompt or shared context text.

        Returns:
            Hex string content-address of the prefix.
        """
        return hashlib.sha256(prefix_text.strip().encode("utf-8")).hexdigest()

    def create_agent_session(
        self,
        session_id: str,
        prefix_hash: str | None = None,
        prefix_kv: Any = None,
        pin_shared: bool = False,
    ) -> AgentKVSession:
        """Create an agent session optionally sharing a pre-hashed prefix block.

        Unlike register_session() which takes raw token IDs, this accepts an
        already-computed prefix_hash (from hash_prefix()) — useful when the caller
        has hashed the text without tokenizing.

        Args:
            session_id: Unique agent session ID.
            prefix_hash: Pre-computed content hash of the shared prefix.
            prefix_kv: KV data for the prefix (stored in shared block if new).
            pin_shared: If True, pin the shared block so it cannot be evicted.

        Returns:
            AgentKVSession descriptor.
        """
        with self._lock:
            if prefix_hash is not None:
                if prefix_hash in self._shared_blocks:
                    block = self._shared_blocks[prefix_hash]
                    block.ref_count += 1
                    block.last_access_ts = time.time()
                    self._stats.cache_hits += 1
                    logger.debug(
                        "R2: Agent session %r attaches to shared block %s (ref=%d).",
                        session_id[:8], prefix_hash[:8], block.ref_count,
                    )
                else:
                    block = SharedKVBlock(
                        block_id=prefix_hash,
                        ref_count=1,
                        kv_data=prefix_kv,
                        seq_len=0,
                        last_access_ts=time.time(),
                        is_pinned=pin_shared,
                    )
                    self._shared_blocks[prefix_hash] = block
                    self._stats.blocks_created += 1
                    if len(self._shared_blocks) > self.max_shared_blocks:
                        self._evict_one()

                session = AgentKVSession(
                    session_id=session_id,
                    shared_block_id=prefix_hash,
                    private_kv=None,
                    agent_token_offset=block.seq_len,
                )
            else:
                session = AgentKVSession(
                    session_id=session_id,
                    shared_block_id=None,
                    private_kv=None,
                    agent_token_offset=0,
                )

            self._sessions[session_id] = session
            self._stats.sessions_created += 1
            return session

    def pin_block(self, block_id: str, pinned: bool = True) -> bool:
        """Pin or unpin a shared block from eviction.

        Pinned blocks are never evicted regardless of ref_count.
        Used for long-lived shared system prompts.

        Args:
            block_id: Block ID to pin (from hash_prefix()).
            pinned: True to pin, False to unpin.

        Returns:
            True if block was found, False otherwise.
        """
        with self._lock:
            if block_id in self._shared_blocks:
                self._shared_blocks[block_id].is_pinned = pinned
                return True
            return False

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
                for variant_key in self._shared_blocks[evict_id].execution_variants:
                    self._variant_lru.pop((evict_id, variant_key), None)
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
                "execution_variants": len(self._variant_lru),
                "execution_variant_capacity": self.max_shared_blocks,
                "storage_scope": "in_process_reference",
                "active_sessions": len(self._sessions),
                "cache_hit_rate": (
                    self._stats.cache_hits / max(1, self._stats.sessions_created)
                ),
                "blocks_evicted": self._stats.blocks_evicted,
                "research_basis": "SGLang RadixAttention + vLLM execution-bound prefix caching",
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
    # This local accumulator stores Python values, not CUDA/RDMA allocations.
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
