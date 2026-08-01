"""
SSM / Hybrid Architecture State Management.

Supports Mamba, RWKV, Jamba, Bamba, Zamba2, and any hybrid transformer+SSM model.

AEG-IR SSM opcodes supported:
  aeg.ssm_scan(x, A, B, C, D)           — Mamba selective scan
  aeg.ssm_state_update(state, x, dt)    — Mamba recurrent state update
  aeg.ssm_state_snapshot(state)         — State snapshot for spec decoding rollback
  aeg.rwkv_time_mix(x, state, w, u)     — RWKV WKV attention mechanism
  aeg.hybrid_dispatch(x, layer_type)    — Route to attn or SSM per layer
  aeg.mimo_ssm(x, A, B, C, D, order=2) — Mamba-3 MIMO complex-valued formulation

Dual memory pool:
  - KV pool: paged KV cache for transformer attention layers
  - SSM pool: recurrent state for Mamba/RWKV layers
  - Snapshot store: for speculative decoding rollback

Research:
  - Mamba (Gu & Dao, 2023): Hardware-aware selective state space model
  - Mamba-2 (2024): Structured SSM with multi-head state expansion
  - Mamba-3 / MIMO (2026): Complex-valued MIMO SSM formulation
  - RWKV-7 (2025): Linear-time WKV attention
  - Jamba (AI21 Labs, 2024): 52-layer 1:7 attn:mamba hybrid
  - SGLang Hybrid Serving (Alibaba Cloud, 2026): Dual-pool serving
  - State Snapshotting for Spec Decoding on SSMs (2026)
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hybrid architecture configurations
# ---------------------------------------------------------------------------

# Supported hybrid patterns and their layer-type schedules
HYBRID_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "jamba": {
        "attn_ratio": 0.125,    # 1 attn every 8 layers
        "ssm_type": "mamba",
        "description": "Jamba (AI21 Labs, 2024) — 1:7 attn:mamba",
    },
    "bamba": {
        "attn_ratio": 0.25,
        "ssm_type": "mamba2",
        "description": "Bamba — 1:3 attn:mamba2",
    },
    "zamba2": {
        "attn_ratio": 0.10,
        "ssm_type": "mamba",
        "shared_attention": True,
        "description": "Zamba2 — shared attention + mamba",
    },
    "mamba": {
        "attn_ratio": 0.0,
        "ssm_type": "mamba",
        "description": "Pure Mamba — no attention layers",
    },
    "mamba3": {
        "attn_ratio": 0.0,
        "ssm_type": "mimo_ssm",
        "description": "Mamba-3 — MIMO complex-valued SSM",
    },
    "rwkv7": {
        "attn_ratio": 0.0,
        "ssm_type": "rwkv",
        "description": "RWKV-7 — pure linear-time WKV attention",
    },
}


def get_hybrid_layer_schedule(
    architecture: str, num_layers: int
) -> list[str]:
    """
    Return the layer-type schedule for a hybrid model.

    Returns a list of 'attn' or 'ssm' for each layer.
    """
    cfg = HYBRID_ARCHITECTURES.get(architecture.lower(), {})
    attn_ratio = cfg.get("attn_ratio", 0.0)

    if attn_ratio == 0.0:
        return ["ssm"] * num_layers

    # Place attention layers at regular intervals
    schedule = ["ssm"] * num_layers
    if attn_ratio > 0:
        attn_period = max(1, int(round(1.0 / attn_ratio)))
        for i in range(0, num_layers, attn_period):
            schedule[i] = "attn"
    return schedule


# ---------------------------------------------------------------------------
# SSM state data structures
# ---------------------------------------------------------------------------

@dataclass
class MambaState:
    """
    Recurrent state for a single Mamba layer.

    Mamba state: h_t = A_bar × h_{t-1} + B_bar × x_t
    where A_bar = exp(Δ × A) and B_bar = Δ × B (discretization).
    """
    layer_idx: int
    # (batch, d_state, d_inner): SSM recurrent state h
    h: np.ndarray
    # Cache of last token's input x for selective scan continuation
    last_x: np.ndarray | None = None
    step: int = 0

    @property
    def d_state(self) -> int:
        return self.h.shape[-2] if self.h.ndim >= 2 else 1

    @property
    def d_inner(self) -> int:
        return self.h.shape[-1] if self.h.ndim >= 2 else self.h.shape[-1]

    def reset(self) -> None:
        self.h = np.zeros_like(self.h)
        self.last_x = None
        self.step = 0

    def copy(self) -> "MambaState":
        return MambaState(
            layer_idx=self.layer_idx,
            h=self.h.copy(),
            last_x=self.last_x.copy() if self.last_x is not None else None,
            step=self.step,
        )


@dataclass
class RWKVState:
    """
    Recurrent state for a single RWKV-7 layer (WKV mechanism).

    RWKV state: wkv_state + token_shift buffer.
    """
    layer_idx: int
    # WKV numerator and denominator states
    wkv_num: np.ndarray     # (batch, num_heads, head_dim)
    wkv_den: np.ndarray     # (batch, num_heads, 1)
    # Token shift state (last token's embedding)
    token_shift: np.ndarray | None = None
    step: int = 0

    def copy(self) -> "RWKVState":
        return RWKVState(
            layer_idx=self.layer_idx,
            wkv_num=self.wkv_num.copy(),
            wkv_den=self.wkv_den.copy(),
            token_shift=self.token_shift.copy() if self.token_shift is not None else None,
            step=self.step,
        )

    def reset(self) -> None:
        self.wkv_num = np.zeros_like(self.wkv_num)
        self.wkv_den = np.zeros_like(self.wkv_den)
        self.token_shift = None
        self.step = 0


@dataclass
class StateSnapshot:
    """
    Snapshot of both transformer KV state and SSM recurrent state.
    Used for speculative decoding rollback when draft tokens are rejected.

    Two storage modes are supported:

    * **Layer mode** — ``kv_state``/``ssm_state`` hold per-layer tensors, keyed
      by layer index. This is what the runtime execution engines use.
    * **Payload mode** — ``kv_payload``/``ssm_payload`` hold an opaque
      serializable object covering the whole request. Used by session-level
      callers that checkpoint an already-assembled cache structure.
    """
    snapshot_id: str
    request_id: str
    step: int
    created_at: float = field(default_factory=time.time)

    # KV cache state (transformer layers): layer_idx → (K, V) tensors
    kv_state: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # SSM recurrent states: layer_idx → MambaState or RWKVState
    ssm_state: dict[int, Any] = field(default_factory=dict)

    # Opaque whole-request payloads (payload mode). None when unused.
    kv_payload: Any = None
    ssm_payload: Any = None

    @property
    def is_payload_mode(self) -> bool:
        """True when this snapshot checkpoints opaque request-level payloads."""
        return self.kv_payload is not None or self.ssm_payload is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "request_id": self.request_id,
            "step": self.step,
            "kv_layers": list(self.kv_state.keys()),
            "ssm_layers": list(self.ssm_state.keys()),
            "payload_mode": self.is_payload_mode,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# SSM State Pool
# ---------------------------------------------------------------------------

class SSMStatePool:
    """
    In-memory recurrent state pool for Mamba/RWKV-style layers.

    Manages SSM states per request. States are compactly stored and
    deeply copied on get/set to avoid aliasing bugs during speculative
    decoding rollback.
    """

    def __init__(self) -> None:
        self._mamba_states: dict[str, dict[int, MambaState]] = {}   # req → {layer → state}
        self._rwkv_states:  dict[str, dict[int, RWKVState]] = {}
        # Opaque request-level payloads for callers that manage their own
        # state representation instead of per-layer Mamba/RWKV structs.
        self._payloads: dict[str, Any] = {}

    def set_payload(self, request_id: str, payload: Any) -> None:
        """Store an opaque request-level SSM payload (deep-copied)."""
        self._payloads[request_id] = copy.deepcopy(payload)

    def get_payload(self, request_id: str) -> Any:
        """Return the opaque request-level payload, or None if unset."""
        payload = self._payloads.get(request_id)
        return copy.deepcopy(payload) if payload is not None else None

    def init_mamba(
        self,
        request_id: str,
        layer_indices: list[int],
        d_inner: int = 1024,
        d_state: int = 16,
        batch_size: int = 1,
    ) -> None:
        """Initialize zero Mamba states for a request."""
        self._mamba_states[request_id] = {
            layer_idx: MambaState(
                layer_idx=layer_idx,
                h=np.zeros((batch_size, d_state, d_inner), dtype=np.float32),
            )
            for layer_idx in layer_indices
        }

    def init_rwkv(
        self,
        request_id: str,
        layer_indices: list[int],
        num_heads: int = 32,
        head_dim: int = 128,
        batch_size: int = 1,
    ) -> None:
        """Initialize zero RWKV states for a request."""
        self._rwkv_states[request_id] = {
            layer_idx: RWKVState(
                layer_idx=layer_idx,
                wkv_num=np.zeros((batch_size, num_heads, head_dim), dtype=np.float32),
                wkv_den=np.zeros((batch_size, num_heads, 1), dtype=np.float32),
            )
            for layer_idx in layer_indices
        }

    def get_mamba(self, request_id: str, layer_idx: int) -> MambaState | None:
        layers = self._mamba_states.get(request_id)
        if layers is None:
            return None
        state = layers.get(layer_idx)
        return state.copy() if state is not None else None

    def set_mamba(self, request_id: str, layer_idx: int, state: MambaState) -> None:
        if request_id not in self._mamba_states:
            self._mamba_states[request_id] = {}
        self._mamba_states[request_id][layer_idx] = state.copy()

    def get_rwkv(self, request_id: str, layer_idx: int) -> RWKVState | None:
        layers = self._rwkv_states.get(request_id)
        if layers is None:
            return None
        state = layers.get(layer_idx)
        return state.copy() if state is not None else None

    def set_rwkv(self, request_id: str, layer_idx: int, state: RWKVState) -> None:
        if request_id not in self._rwkv_states:
            self._rwkv_states[request_id] = {}
        self._rwkv_states[request_id][layer_idx] = state.copy()

    def get(self, request_id: str) -> Any:
        """
        Get the SSM state for a request (for snapshotting).

        Returns the opaque payload when one was stored via
        :meth:`set_payload`; otherwise a ``{layer_idx: state}`` dict.
        """
        payload = self._payloads.get(request_id)
        if payload is not None:
            return copy.deepcopy(payload)
        result: dict[int, Any] = {}
        for lid, state in self._mamba_states.get(request_id, {}).items():
            result[lid] = state.copy()
        for lid, state in self._rwkv_states.get(request_id, {}).items():
            result[lid] = state.copy()
        return result

    def restore(self, request_id: str, states: dict[int, Any]) -> None:
        """Restore all SSM states from a snapshot."""
        self._mamba_states[request_id] = {}
        self._rwkv_states[request_id] = {}
        for lid, state in states.items():
            if isinstance(state, MambaState):
                self._mamba_states[request_id][lid] = state.copy()
            elif isinstance(state, RWKVState):
                self._rwkv_states[request_id][lid] = state.copy()

    def free(self, request_id: str) -> None:
        self._mamba_states.pop(request_id, None)
        self._rwkv_states.pop(request_id, None)
        self._payloads.pop(request_id, None)

    def stats(self) -> dict[str, Any]:
        return {
            "active_mamba_requests": len(self._mamba_states),
            "active_rwkv_requests": len(self._rwkv_states),
            "active_payload_requests": len(self._payloads),
            "total_mamba_layers": sum(len(v) for v in self._mamba_states.values()),
            "total_rwkv_layers": sum(len(v) for v in self._rwkv_states.values()),
        }


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------

class StateSnapshotStore:
    """
    Immutable state snapshots for speculative decoding rollback.

    When speculative tokens are rejected, the runtime calls rollback() to
    restore both KV cache and SSM recurrent state to the pre-speculation
    checkpoint.
    """

    def __init__(self, max_snapshots_per_request: int = 4) -> None:
        self.max_per_request = max_snapshots_per_request
        self._snapshots: dict[str, list[StateSnapshot]] = {}   # req → [snaps]

    def _next_id(self, request_id: str, step: int) -> str:
        """Build a collision-free snapshot id for a request/step."""
        base = f"{request_id}:step{step}"
        existing = {s.snapshot_id for s in self._snapshots.get(request_id, [])}
        # Two snapshots in the same millisecond must not share an id, or
        # rollback would resolve to the wrong checkpoint.
        candidate = f"{base}:{int(time.time() * 1000) % 100000}"
        if candidate not in existing:
            return candidate
        suffix = 1
        while f"{candidate}-{suffix}" in existing:
            suffix += 1
        return f"{candidate}-{suffix}"

    def save(
        self,
        request_id: str,
        step: int,
        kv_state: dict[int, tuple[np.ndarray, np.ndarray]],
        ssm_state: dict[int, Any],
    ) -> StateSnapshot:
        """Save a full per-layer state snapshot for rollback."""
        snap_id = self._next_id(request_id, step)

        # Deep copy KV state
        kv_copy = {lid: (k.copy(), v.copy()) for lid, (k, v) in kv_state.items()}

        # Deep copy SSM state
        ssm_copy: dict[int, Any] = {}
        for lid, state in ssm_state.items():
            if isinstance(state, MambaState):
                ssm_copy[lid] = state.copy()
            elif isinstance(state, RWKVState):
                ssm_copy[lid] = state.copy()
            else:
                ssm_copy[lid] = copy.deepcopy(state)

        snapshot = StateSnapshot(
            snapshot_id=snap_id,
            request_id=request_id,
            step=step,
            kv_state=kv_copy,
            ssm_state=ssm_copy,
        )
        return self._store(snapshot)

    def save_payload(
        self,
        request_id: str,
        step: int,
        kv_payload: Any,
        ssm_payload: Any,
    ) -> StateSnapshot:
        """
        Save an opaque whole-request payload snapshot.

        Payloads are deep-copied so later mutation of the live pool cannot
        reach back into the checkpoint.
        """
        snapshot = StateSnapshot(
            snapshot_id=self._next_id(request_id, step),
            request_id=request_id,
            step=step,
            kv_payload=copy.deepcopy(kv_payload),
            ssm_payload=copy.deepcopy(ssm_payload),
        )
        return self._store(snapshot)

    def _store(self, snapshot: StateSnapshot) -> StateSnapshot:
        """Append a snapshot to its request ring buffer, evicting the oldest."""
        request_id = snapshot.request_id
        if request_id not in self._snapshots:
            self._snapshots[request_id] = []
        snaps = self._snapshots[request_id]
        snaps.append(snapshot)

        # Evict oldest if over limit
        if len(snaps) > self.max_per_request:
            snaps.pop(0)

        logger.debug(
            "Snapshot saved: %s (step=%d)", snapshot.snapshot_id, snapshot.step
        )
        return snapshot

    def load(self, snapshot_id: str) -> StateSnapshot | None:
        """Load a snapshot by ID."""
        for snaps in self._snapshots.values():
            for snap in snaps:
                if snap.snapshot_id == snapshot_id:
                    return snap
        return None

    def latest(self, request_id: str) -> StateSnapshot | None:
        """Get the most recent snapshot for a request."""
        snaps = self._snapshots.get(request_id)
        if not snaps:
            return None
        return snaps[-1]

    def free(self, request_id: str) -> None:
        self._snapshots.pop(request_id, None)

    def stats(self) -> dict[str, Any]:
        return {
            "active_requests": len(self._snapshots),
            "total_snapshots": sum(len(v) for v in self._snapshots.values()),
        }


# ---------------------------------------------------------------------------
# Hybrid Memory Pool (main interface)
# ---------------------------------------------------------------------------

class HybridMemoryPool:
    """
    Dual KV/SSM memory pool with speculative rollback snapshots.

    For hybrid models (Jamba, Bamba, Zamba2, Mamba-3):
      - Transformer attention layers → KV pool (paged KV blocks)
      - SSM / RWKV layers → SSM pool (recurrent state vectors)
      - Speculative decoding → snapshot store for rollback

    PRD requirement:
      "agentic KV reuse >80% prefill reduction"
    """

    def __init__(self) -> None:
        self.kv_pool: dict[str, Any] = {}
        self.ssm_pool = SSMStatePool()
        self.snapshots = StateSnapshotStore()

    # ------------------------------------------------------------------ #
    # KV pool operations (transformer layers)
    # ------------------------------------------------------------------ #

    def set_kv(
        self,
        request_id: str,
        layer_idx: int | Any = None,
        k: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> None:
        """
        Store KV state for a request.

        Two call forms are supported:

        * ``set_kv(request_id, layer_idx, k, v)`` — layer mode. Stores the
          ``(K, V)`` tensor pair for one transformer layer.
        * ``set_kv(request_id, payload)`` — payload mode. Stores an opaque
          request-level cache object.
        """
        if k is None and v is None:
            # Payload mode: the second positional is the payload itself.
            self.kv_pool[request_id] = copy.deepcopy(layer_idx)
            return

        if k is None or v is None:
            raise ValueError(
                "set_kv layer mode requires both k and v tensors; "
                "use set_kv(request_id, payload) for payload mode"
            )

        entry = self.kv_pool.get(request_id)
        if not isinstance(entry, dict):
            entry = {}
            self.kv_pool[request_id] = entry
        entry[layer_idx] = (k.copy(), v.copy())

    def set_ssm(self, request_id: str, payload: Any) -> None:
        """
        Store an opaque request-level SSM payload.

        For per-layer Mamba/RWKV state use ``pool.ssm_pool.set_mamba(...)``
        or ``pool.ssm_pool.set_rwkv(...)`` instead.
        """
        self.ssm_pool.set_payload(request_id, payload)

    def get_ssm(self, request_id: str) -> Any:
        """Return the request-level SSM payload, or None if unset."""
        return self.ssm_pool.get_payload(request_id)

    def get_kv(
        self, request_id: str, layer_idx: int | None = None
    ) -> Any:
        """
        Retrieve KV state.

        With ``layer_idx`` returns the ``(K, V)`` pair for that layer; without
        it returns the whole request entry (payload or per-layer dict).
        """
        req = self.kv_pool.get(request_id)
        if req is None:
            return None
        if layer_idx is None:
            return req
        if not isinstance(req, dict):
            return None
        entry = req.get(layer_idx)
        if entry is None:
            return None
        k, v = entry
        return k.copy(), v.copy()

    def append_kv(
        self,
        request_id: str,
        layer_idx: int,
        new_k: np.ndarray,  # (new_seq, num_kv_heads, head_dim)
        new_v: np.ndarray,
    ) -> None:
        """Append new KV tokens to the existing cache for a layer."""
        existing = self.get_kv(request_id, layer_idx)
        if existing is None:
            self.set_kv(request_id, layer_idx, new_k, new_v)
        else:
            k, v = existing
            self.set_kv(
                request_id, layer_idx,
                np.concatenate([k, new_k], axis=0),
                np.concatenate([v, new_v], axis=0),
            )

    # ------------------------------------------------------------------ #
    # Speculative decoding: snapshot and rollback
    # ------------------------------------------------------------------ #

    def snapshot(self, request_id: str, step: int | None = None) -> StateSnapshot:
        """
        Save a full state snapshot (KV + SSM) for speculative decoding.

        Call before generating speculative draft tokens. Dispatches to payload
        or layer mode based on how the request's state was stored. When
        ``step`` is omitted it is inferred from the snapshot count so far.
        """
        if step is None:
            step = len(self.snapshots._snapshots.get(request_id, []))

        kv_entry = self.kv_pool.get(request_id)
        ssm_payload = self.ssm_pool.get_payload(request_id)

        # Payload mode: either side was stored as an opaque request payload.
        if ssm_payload is not None or (
            kv_entry is not None and not isinstance(kv_entry, dict)
        ):
            return self.snapshots.save_payload(
                request_id, step, kv_entry, ssm_payload
            )

        return self.snapshots.save(
            request_id, step, kv_entry or {}, self.ssm_pool.get(request_id)
        )

    def rollback(self, request_id: str, snapshot_id: str) -> bool:
        """
        Restore state from snapshot (called when draft tokens are rejected).

        Returns True on success.
        """
        snapshot = self.snapshots.load(snapshot_id)
        if snapshot is None:
            logger.warning("Rollback failed: snapshot %s not found", snapshot_id)
            return False

        if snapshot.is_payload_mode:
            if snapshot.kv_payload is not None:
                self.kv_pool[request_id] = copy.deepcopy(snapshot.kv_payload)
            if snapshot.ssm_payload is not None:
                self.ssm_pool.set_payload(
                    request_id, copy.deepcopy(snapshot.ssm_payload)
                )
        else:
            # Restore KV pool
            self.kv_pool[request_id] = {
                lid: (k.copy(), v.copy())
                for lid, (k, v) in snapshot.kv_state.items()
            }
            # Restore SSM pool
            self.ssm_pool.restore(request_id, snapshot.ssm_state)

        logger.debug(
            "Rollback: request=%s, snapshot=%s (step %d)",
            request_id, snapshot_id, snapshot.step
        )
        return True

    def rollback_to_latest(self, request_id: str) -> bool:
        """Rollback to the most recent snapshot for a request."""
        snap = self.snapshots.latest(request_id)
        if snap is None:
            return False
        return self.rollback(request_id, snap.snapshot_id)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def free_request(self, request_id: str) -> None:
        """Release all state for a completed request."""
        self.kv_pool.pop(request_id, None)
        self.ssm_pool.free(request_id)
        self.snapshots.free(request_id)

    def stats(self) -> dict[str, Any]:
        # Only per-layer entries have measurable tensor bytes; opaque payloads
        # are counted separately since their size is caller-defined.
        kv_mem = 0
        payload_requests = 0
        for req in self.kv_pool.values():
            if not isinstance(req, dict):
                payload_requests += 1
                continue
            for entry in req.values():
                if isinstance(entry, tuple) and len(entry) == 2:
                    k, v = entry
                    if hasattr(k, "nbytes") and hasattr(v, "nbytes"):
                        kv_mem += k.nbytes + v.nbytes
                else:
                    payload_requests += 1
                    break
        return {
            "active_requests": len(self.kv_pool),
            "kv_payload_requests": payload_requests,
            "kv_memory_mb": round(kv_mem / 1e6, 2),
            "ssm": self.ssm_pool.stats(),
            "snapshots": self.snapshots.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"HybridMemoryPool("
            f"requests={len(self.kv_pool)}, "
            f"ssm={self.ssm_pool.stats()['active_mamba_requests']} mamba"
            f")"
        )


# ---------------------------------------------------------------------------
# SSM forward pass reference implementations
# ---------------------------------------------------------------------------

class MambaSSM:
    """
    Mamba selective state space model — single-step forward pass.

    Implements the core Mamba recurrent update:
      h_t = A_bar × h_{t-1} + B_bar × x_t
      y_t = C × h_t + D × x_t

    where A_bar = exp(Δ × A), B_bar = Δ × B (ZOH discretization).

    Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces", ICLR 2024.
    """

    def __init__(self, d_model: int = 1024, d_state: int = 16, d_inner: int = 2048) -> None:
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner

    def step(
        self,
        x: np.ndarray,         # (batch, d_inner) — current token input
        state: MambaState,     # current SSM state
        A: np.ndarray,         # (d_inner, d_state) — learnable diagonal SSM A matrix
        B: np.ndarray,         # (batch, d_state) — input-dependent B (selective)
        C: np.ndarray,         # (batch, d_state) — output-dependent C (selective)
        D: np.ndarray,         # (d_inner,) — skip connection
        dt: np.ndarray,        # (batch, d_inner) — input-dependent Δ (selective)
        dt_softplus: bool = True,
    ) -> tuple[np.ndarray, MambaState]:
        """
        Single Mamba recurrent step.

        Returns:
            (output, new_state): output (batch, d_inner), updated MambaState.
        """
        batch = x.shape[0]
        # Softplus on dt (ensures positive)
        if dt_softplus:
            dt = np.log1p(np.exp(dt.astype(np.float64))).astype(np.float32)

        # ZOH discretization: A_bar = exp(dt × A)
        # A is (d_inner, d_state), dt is (batch, d_inner)
        dtA = dt[:, :, np.newaxis] * A[np.newaxis, :, :]  # (batch, d_inner, d_state)
        A_bar = np.exp(dtA)                                 # (batch, d_inner, d_state)

        # B_bar = dt × B: (batch, d_inner, d_state)
        B_bar = dt[:, :, np.newaxis] * B[:, np.newaxis, :]

        # x expanded: (batch, d_inner, 1)
        x_exp = x[:, :, np.newaxis]

        # State update: h = A_bar ⊙ h + B_bar ⊙ x
        h_prev = state.h   # (batch, d_state, d_inner) — stored transposed
        # Align dimensions: h_prev transposed to (batch, d_inner, d_state)
        h_prev_t = h_prev.transpose(0, 2, 1)
        h_new_t = A_bar * h_prev_t + B_bar * x_exp  # (batch, d_inner, d_state)

        # Output: y = C × h + D × x
        y = np.einsum("bds,bs->bd", h_new_t, C) + D[np.newaxis, :] * x  # (batch, d_inner)

        new_state = MambaState(
            layer_idx=state.layer_idx,
            h=h_new_t.transpose(0, 2, 1),  # back to (batch, d_state, d_inner)
            last_x=x.copy(),
            step=state.step + 1,
        )
        return y.astype(np.float32), new_state


class RWKV7:
    """
    RWKV-7 WKV attention mechanism — linear-time recurrent formulation.

    Implements the RWKV time-mix layer:
      wkv_t = w × wkv_{t-1} + k_t × v_t
      y_t   = wkv_t / (w × den_{t-1} + k_t)

    Reference: Peng et al., RWKV-7 (2025).
    """

    def step(
        self,
        x: np.ndarray,       # (batch, num_heads, head_dim)
        state: RWKVState,    # current RWKV state
        w: np.ndarray,       # (num_heads, head_dim) — decay weights
        u: np.ndarray,       # (num_heads, head_dim) — first-token bonus
        k: np.ndarray,       # (batch, num_heads, head_dim) — key
        v: np.ndarray,       # (batch, num_heads, head_dim) — value
    ) -> tuple[np.ndarray, RWKVState]:
        """
        Single RWKV-7 time-mix step.

        Returns:
            (output, new_state)
        """
        # Decay weights (exponential decay)
        decay = np.exp(-np.exp(w.astype(np.float64))).astype(np.float32)

        # WKV update with first-token bonus u
        # For first token: wkv = k × v + u × k × v
        bonus_kv = (1 + u[np.newaxis]) * k * v           # (batch, heads, dim)
        new_num = decay[np.newaxis] * state.wkv_num + k * v  # (batch, heads, dim)
        new_den = decay[np.newaxis] * state.wkv_den + k      # (batch, heads, dim)

        # Output
        y = (new_num + u[np.newaxis] * k * v) / (new_den + u[np.newaxis] * k + 1e-9)

        new_state = RWKVState(
            layer_idx=state.layer_idx,
            wkv_num=new_num,
            wkv_den=new_den,
            token_shift=x.copy(),
            step=state.step + 1,
        )
        return y.astype(np.float32), new_state
