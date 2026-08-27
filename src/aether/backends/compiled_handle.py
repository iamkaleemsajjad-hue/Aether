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
    are served by promoting the same authenticated weights onto the portable
    tensor executor, which does carry a batch axis.  Holding it here means the
    promotion is paid for once rather than per request, and only if a batch is
    actually asked for.
    """

    def clear_session_cache(self, session_id: str) -> None:
        """Release the incremental KV state owned by one session."""
        self.session_caches.pop(session_id, None)

    @property
    def architecture_family(self) -> str:
        """Return the family recorded in the authenticated manifest."""
        architecture = self.manifest.get("architecture", {})
        return str(architecture.get("family", "unknown"))
