"""Framework-neutral handles for executable AEG artifacts.

This type lives outside the optional PyTorch backend so the native AEG path has
no import-time relationship with PyTorch.  The handle deliberately contains
only portable Python/NumPy-facing objects; backend-specific tensor objects do
not cross the AEG boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompiledAEGHandle:
    """Loaded executable AEG package and its request-local state."""

    model_id: str
    aeg_path: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    precision_map: dict[str, str] = field(default_factory=dict)
    engine: Any | None = None
    tokenizer: Any | None = None
    lora_adapters: dict[str, dict[tuple[int, str], tuple[Any, Any, float]]] = field(default_factory=dict)
    session_caches: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    batched_engine: Any | None = None
    """Executor used for batched requests, when it differs from ``engine``.

    Populated lazily and at most once per handle.  On a CPU host the AEG loads
    onto the NumPy reference executor, which is sequence-major; batched requests
    are served by promoting the same integrity-checked weights onto the portable
    tensor executor, which does carry a batch axis.  Holding it here means the
    promotion is paid for once rather than per request, and only if a batch is
    actually asked for.
    """
    session_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_target_ids: dict[str, int] = field(default_factory=dict)
    numerical_quality_admitted: bool = False
    """Decision for these loaded weights; a later path replacement cannot promote them."""

    def clear_session_cache(self, session_id: str) -> None:
        """Release the incremental KV state owned by one session."""
        self.session_caches.pop(session_id, None)
        self.session_contracts.pop(session_id, None)
        self.session_target_ids.pop(session_id, None)

    def clear_session_caches(self) -> None:
        """Drop state and its identities together, without retaining side maps."""
        self.session_caches.clear()
        self.session_contracts.clear()
        self.session_target_ids.clear()

    def get_session_cache(self, session_id: str, *, contract: dict[str, Any]) -> tuple[Any, Any] | None:
        """Reuse only state carrying the same validated execution contract."""
        cached = self.session_caches.get(session_id)
        if cached is not None and self.session_contracts.get(session_id) != contract:
            self.clear_session_cache(session_id)
            return None
        return cached

    def save_session_cache(self, session_id: str, token_ids: Any, cache: Any,
                           *, contract: dict[str, Any] | None = None, target_engine: Any = None) -> None:
        """Record only tokens the cache actually consumed, including stop cases."""
        import numpy as np

        if cache is None:
            self.clear_session_cache(session_id)
            return
        ids = np.asarray(token_ids, dtype=np.int64)
        length = getattr(cache, "length", getattr(cache, "logical_length", None))
        if not isinstance(length, int) or not 0 <= length <= ids.size:
            raise ValueError("session cache length does not match its token prefix")
        from aether.runtime.session_contract import session_execution_contract

        if contract is None and self.manifest:
            contract = session_execution_contract(self)
        target = self.engine if target_engine is None else target_engine
        self.session_caches[session_id] = (ids[:length].copy(), cache)
        if contract is not None:
            self.session_contracts[session_id] = contract
            self.session_target_ids[session_id] = id(target)
        else:
            self.session_contracts.pop(session_id, None)
            self.session_target_ids.pop(session_id, None)

    @staticmethod
    def cache_for_engine(cache: Any, engine: Any) -> Any:
        """Retain the exact target state when a request changes its draft policy."""
        from aether.runtime.spec_generation import SpeculativeGenerationAdapter, SpeculativeSessionCache

        if isinstance(cache, SpeculativeSessionCache):
            if not isinstance(engine, SpeculativeGenerationAdapter) or cache.source_identity != engine._source_identity:
                return cache.target_cache
        return cache

    @property
    def architecture_family(self) -> str:
        """Return the family recorded in the integrity-checked manifest."""
        architecture = self.manifest.get("architecture", {})
        return str(architecture.get("family", "unknown"))
