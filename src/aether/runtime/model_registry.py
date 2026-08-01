"""
Model registry — tracks loaded models, their backends, and AEG metadata.

Provides a thread-safe registry that maps model IDs to loaded backend handles,
AEG packages, and generation statistics. Supports hot-reload, LRU eviction,
and reference counting for multi-request serving.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.exceptions import ModelNotFoundError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEntry:
    """A single entry in the model registry."""

    model_id: str
    """HuggingFace / local model identifier."""

    aeg_path: str | None
    """Path to the compiled AEG package, or None if not compiled."""

    backend_name: str
    """Name of the backend that loaded this model."""

    handle: Any
    """Backend-specific model handle (e.g., torch.nn.Module or AEGHandle)."""

    loaded_at: float = field(default_factory=time.monotonic)
    """Monotonic timestamp of when the model was loaded."""

    last_used: float = field(default_factory=time.monotonic)
    """Monotonic timestamp of the most recent request."""

    request_count: int = 0
    """Total number of requests served by this model entry."""

    ref_count: int = 0
    """Number of in-flight requests holding a reference."""

    architecture: dict[str, Any] = field(default_factory=dict)
    """Architecture metadata from the AEG manifest or model config."""

    precision_map: dict[str, str] = field(default_factory=dict)
    """Layer → precision string mapping."""

    def touch(self) -> None:
        """Update last_used and increment request_count."""
        self.last_used = time.monotonic()
        self.request_count += 1

    def acquire(self) -> None:
        """Increment reference count (request started)."""
        self.ref_count += 1
        self.touch()

    def release(self) -> None:
        """Decrement reference count (request completed)."""
        self.ref_count = max(0, self.ref_count - 1)

    @property
    def idle_seconds(self) -> float:
        """Seconds since the model was last used."""
        return time.monotonic() - self.last_used

    @property
    def load_seconds(self) -> float:
        """Seconds since the model was first loaded."""
        return time.monotonic() - self.loaded_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "aeg_path": self.aeg_path,
            "backend_name": self.backend_name,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "request_count": self.request_count,
            "ref_count": self.ref_count,
            "idle_seconds": self.idle_seconds,
            "architecture": self.architecture,
            "precision_map": self.precision_map,
        }


class ModelRegistry:
    """
    Thread-safe registry of loaded model handles.

    Features:
    - LRU eviction when ``max_loaded_models`` is reached
    - Reference counting prevents eviction of in-flight models
    - AEG metadata indexing for model info queries
    - Hot-reload: replace a model entry atomically
    """

    def __init__(
        self,
        max_loaded_models: int = 4,
        cache_dir: str | Path | None = None,
    ) -> None:
        from aether.utils.file_io import aether_cache_dir

        self.max_loaded_models = max_loaded_models
        self._entries: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()

        # On-disk side of the registry: compiled AEG packages live under
        # <cache>/models/. Resolved eagerly so models_dir always exists.
        self.cache_dir: Path = aether_cache_dir(cache_dir)
        self.models_dir: Path = self.cache_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Model registry initialized",
            max_loaded_models=max_loaded_models,
            models_dir=str(self.models_dir),
        )

    # ------------------------------------------------------------------
    # On-disk AEG cache
    # ------------------------------------------------------------------

    @staticmethod
    def _slug(model_id: str) -> str:
        """Filesystem-safe name for a model id ('org/model' → 'org__model')."""
        return model_id.replace("/", "__").replace("\\", "__").strip()

    def cached_path(self, model_id: str) -> Path:
        """Return the canonical on-disk AEG path for a model id."""
        return self.models_dir / f"{self._slug(model_id)}.aeg"

    def _cache_candidates(self, model_id: str) -> list[Path]:
        """
        All paths a compiled package for this model may occupy.

        The compiler historically wrote ``<models>/<model_id>.aeg`` verbatim,
        which nests a directory for namespaced ids, so both layouts are probed.
        """
        return [
            self.cached_path(model_id),
            self.models_dir / f"{model_id}.aeg",
        ]

    def is_cached(self, model_id: str) -> bool:
        """Return True if a compiled AEG package for this model is on disk."""
        return any(p.exists() for p in self._cache_candidates(model_id))

    def list_cached(self) -> list[str]:
        """Return sorted model ids that have a compiled package on disk."""
        if not self.models_dir.exists():
            return []
        found = {
            p.name[: -len(".aeg")].replace("__", "/")
            for p in self.models_dir.glob("*.aeg")
        }
        return sorted(found)

    def remove(self, model_id: str) -> bool:
        """
        Delete a model's cached AEG package from disk.

        Also unloads it from memory so a stale handle cannot outlive the
        package it was loaded from. Returns False when nothing was cached.
        """
        import shutil

        removed = False
        for path in self._cache_candidates(model_id):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed = True

        if removed:
            self.unload(model_id)
            logger.info("Model registry: removed cached package", model_id=model_id)
        return removed

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model_id: str,
        handle: Any,
        backend_name: str,
        aeg_path: str | None = None,
        architecture: dict[str, Any] | None = None,
        precision_map: dict[str, str] | None = None,
    ) -> ModelEntry:
        """
        Register a loaded model handle.

        If the registry is full, evicts the least-recently-used non-busy model.

        Args:
            model_id: Model identifier.
            handle: Backend-specific model handle.
            backend_name: Name of the backend.
            aeg_path: Optional path to the compiled AEG package.
            architecture: Optional architecture metadata dict.
            precision_map: Optional layer→precision dict.

        Returns:
            The newly created ModelEntry.
        """
        with self._lock:
            if model_id in self._entries:
                entry = self._entries[model_id]
                entry.handle = handle
                entry.backend_name = backend_name
                if aeg_path is not None:
                    entry.aeg_path = aeg_path
                if architecture:
                    entry.architecture = architecture
                if precision_map:
                    entry.precision_map = precision_map
                logger.info("Model registry: hot-reloaded %s", model_id)
                return entry

            # Evict LRU if needed
            if len(self._entries) >= self.max_loaded_models:
                self._evict_lru()

            # Load AEG metadata if available
            arch: dict[str, Any] = architecture or {}
            pmap: dict[str, str] = precision_map or {}
            if aeg_path and not arch:
                arch, pmap = self._read_aeg_metadata(aeg_path)

            entry = ModelEntry(
                model_id=model_id,
                aeg_path=aeg_path,
                backend_name=backend_name,
                handle=handle,
                architecture=arch,
                precision_map=pmap,
            )
            self._entries[model_id] = entry
            logger.info(
                "Model registry: registered %s (backend=%s, aeg=%s)",
                model_id,
                backend_name,
                aeg_path is not None,
            )
            return entry

    def _read_aeg_metadata(
        self, aeg_path: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Read architecture and precision map from an AEG package path."""
        import json

        arch: dict[str, Any] = {}
        pmap: dict[str, str] = {}
        try:
            root = Path(aeg_path)
            manifest_path = root / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                arch = manifest.get("architecture", {})
            prec_path = root / "weights" / "quantized" / "precision_map.json"
            if prec_path.exists():
                pmap = json.loads(prec_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Could not read AEG metadata from %s: %s", aeg_path, exc)
        return arch, pmap

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> ModelEntry | None:
        """Return the entry for model_id, or None if not loaded."""
        with self._lock:
            return self._entries.get(model_id)

    def require(self, model_id: str) -> ModelEntry:
        """Return the entry for model_id, raising ModelNotFoundError if absent."""
        entry = self.get(model_id)
        if entry is None:
            msg = f"Model '{model_id}' is not loaded"
            raise ModelNotFoundError(msg, model_id=model_id)
        return entry

    def acquire(self, model_id: str) -> ModelEntry:
        """Acquire a reference to a model (increments ref_count)."""
        with self._lock:
            entry = self.require(model_id)
            entry.acquire()
            return entry

    def release(self, model_id: str) -> None:
        """Release a reference to a model (decrements ref_count)."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry:
                entry.release()

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self, model_id: str) -> bool:
        """
        Remove a model from the registry.

        Returns False if the model is in use (ref_count > 0).
        """
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return True
            if entry.ref_count > 0:
                logger.warning(
                    "Cannot unload model %s: %d requests in flight",
                    model_id,
                    entry.ref_count,
                )
                return False
            del self._entries[model_id]
            logger.info("Model registry: unloaded %s", model_id)
            return True

    def _evict_lru(self) -> bool:
        """Evict the least-recently-used idle model. Returns True if evicted."""
        candidates = [e for e in self._entries.values() if e.ref_count == 0]
        if not candidates:
            logger.warning("All models are in use; cannot evict any")
            return False
        lru = min(candidates, key=lambda e: e.last_used)
        del self._entries[lru.model_id]
        logger.info("Model registry: evicted LRU model %s (idle %.1fs)", lru.model_id, lru.idle_seconds)
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_loaded(self, model_id: str) -> bool:
        """Return True if the model is in the registry."""
        with self._lock:
            return model_id in self._entries

    def list_models(self) -> list[str]:
        """Return sorted list of loaded model IDs."""
        with self._lock:
            return sorted(self._entries.keys())

    def stats(self) -> list[dict[str, Any]]:
        """Return per-model statistics."""
        with self._lock:
            return [e.to_dict() for e in self._entries.values()]

    def total_request_count(self) -> int:
        """Return total requests served across all models."""
        with self._lock:
            return sum(e.request_count for e in self._entries.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        with self._lock:
            return f"ModelRegistry(loaded={len(self._entries)}, max={self.max_loaded_models})"
