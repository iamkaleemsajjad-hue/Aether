"""Hybrid SSM and KV memory state handling."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateSnapshot:
    """Snapshot of both transformer KV state and SSM recurrent state."""

    snapshot_id: str
    request_id: str
    kv_state: Any
    ssm_state: Any


class SSMStatePool:
    """In-memory recurrent state pool for Mamba/RWKV-style layers."""

    def __init__(self) -> None:
        self._states: dict[str, Any] = {}

    def set(self, request_id: str, state: Any) -> None:
        self._states[request_id] = copy.deepcopy(state)

    def get(self, request_id: str) -> Any:
        return copy.deepcopy(self._states.get(request_id))

    def restore(self, request_id: str, state: Any) -> None:
        self._states[request_id] = copy.deepcopy(state)


class HybridMemoryPool:
    """Dual KV/SSM memory pool with speculative rollback snapshots."""

    def __init__(self) -> None:
        self.kv_pool: dict[str, Any] = {}
        self.ssm_pool = SSMStatePool()
        self.snapshots: dict[str, StateSnapshot] = {}

    def set_kv(self, request_id: str, state: Any) -> None:
        self.kv_pool[request_id] = copy.deepcopy(state)

    def set_ssm(self, request_id: str, state: Any) -> None:
        self.ssm_pool.set(request_id, state)

    def snapshot(self, request_id: str) -> StateSnapshot:
        payload = repr((request_id, self.kv_pool.get(request_id), self.ssm_pool.get(request_id)))
        snapshot_id = "snap_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            request_id=request_id,
            kv_state=copy.deepcopy(self.kv_pool.get(request_id)),
            ssm_state=self.ssm_pool.get(request_id),
        )
        self.snapshots[snapshot_id] = snapshot
        return snapshot

    def rollback(self, request_id: str, snapshot_id: str) -> None:
        snapshot = self.snapshots[snapshot_id]
        if snapshot.request_id != request_id:
            raise ValueError("snapshot request_id mismatch")
        self.kv_pool[request_id] = copy.deepcopy(snapshot.kv_state)
        self.ssm_pool.restore(request_id, snapshot.ssm_state)
