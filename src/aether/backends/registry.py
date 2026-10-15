"""Discover backend plugins and construct only the execution engine requested."""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
from functools import partial
from typing import Any, Callable

from aether.backends.base import Backend
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def _load_backend(reference: str) -> Backend:
    module_name, class_name = reference.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def _load_entry_point(entry_point: Any) -> Backend:
    return entry_point.load()()


class BackendRegistry:
    """Lazy registry: CPU artifact startup never initializes optional backends."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._factories: dict[str, Callable[[], Backend]] = {}
        self._failed: set[str] = set()
        self._lock = threading.RLock()
        self._discover_backends()

    @staticmethod
    def _normalize_name(name: str) -> str:
        aliases = {
            "torch": "pytorch",
            "llama.cpp": "llamacpp",
            "tensorrt-llm": "trtllm",
            "tensorrt_llm": "trtllm",
            "onnxruntime": "onnx",
        }
        return aliases.get(name, name)

    def _discover_backends(self) -> None:
        # Keep the framework-free native engines first in explicit inventory.
        builtin = {
            "aether_cpu": "aether.backends.native_cpu_backend:NativeCPUBackend",
            "aether_cuda": "aether.backends.native_cuda_backend:NativeCUDABackend",
            "onnx": "aether.backends.onnx_backend:ONNXBackend",
            "llamacpp": "aether.backends.llamacpp_backend:LlamaCppBackend",
            "mlx": "aether.backends.mlx_backend:MLXBackend",
            "vllm": "aether.backends.vllm_backend:vLLMBackend",
            "trtllm": "aether.backends.trtllm_backend:TensorRTLLMBackend",
            "pytorch": "aether.backends.torch_backend:TorchBackend",
        }
        for name, reference in builtin.items():
            self._factories[name] = partial(_load_backend, reference)
        try:
            eps = importlib.metadata.entry_points()
            backend_eps = (
                eps.select(group="aether.backends")
                if hasattr(eps, "select")
                else eps.get("aether.backends", [])
            )
            for ep in backend_eps:
                self._factories[self._normalize_name(ep.name)] = partial(_load_entry_point, ep)
        except Exception as exc:
            logger.debug("Entry point discovery skipped: %s", exc)

    def get_backend(self, name: str) -> Backend | None:
        """Resolve a backend by name or alias, constructing it at most once."""
        name = self._normalize_name(name)
        with self._lock:
            if name not in self._backends and name not in self._failed:
                factory = self._factories.get(name)
                if factory is not None:
                    try:
                        backend = factory()
                        self._backends[name] = backend
                        self._backends[backend.name] = backend
                    except Exception as exc:
                        self._failed.add(name)
                        logger.debug("Could not load backend %s: %s", name, exc)
            return self._backends.get(name)

    def _materialize_all(self) -> None:
        """Explicit inventory may initialize optional backends."""
        for name in self._factories:
            self.get_backend(name)

    @property
    def backend_names(self) -> list[str]:
        """Return canonical names of discoverable backends."""
        self._materialize_all()
        return sorted({backend.name for backend in self._backends.values()})

    def get_available_backends(self) -> list[Backend]:
        """Return installed backends, deduplicating entry-point aliases."""
        self._materialize_all()
        by_name = {backend.name: backend for backend in self._backends.values()}
        return [backend for backend in by_name.values() if backend.is_available()]

    def get_available_backend_names(self) -> list[str]:
        """Return canonical names of installed backends."""
        return sorted(backend.name for backend in self.get_available_backends())

    def register_backend(self, backend: Backend) -> None:
        """Register or replace an application's backend instance."""
        with self._lock:
            self._backends[backend.name] = backend
            self._failed.discard(backend.name)

    def __repr__(self) -> str:
        available = len(self.get_available_backends())
        return f"BackendRegistry({len(self.backend_names)} backends, {available} available)"
