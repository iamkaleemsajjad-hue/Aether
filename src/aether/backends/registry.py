"""
Backend registry — discovers and loads backend plugins.

Backends are registered via entry points (`aether.backends`) or can be loaded
dynamically. The registry provides a stable lookup by name.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

from aether.backends.base import Backend
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class BackendRegistry:
    """Registry of available Aether backends."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._discover_backends()

    def _discover_backends(self) -> None:
        """Discover backends from built-in list and entry points."""
        # PRD §3: Aether is PyTorch-independent at runtime. The priority order
        # ensures that the native CPU engine is always tried first (no external
        # dependencies), followed by optional accelerator runtimes, with the
        # PyTorch backend reserved as a last-resort fallback.
        builtin_backends = [
            # 1. Always available — pure numpy + compiled C++ (no PyTorch needed)
            "aether.backends.native_cpu_backend:NativeCPUBackend",
            # 2. ONNX Runtime — fast CPU/GPU inference without PyTorch
            "aether.backends.onnx_backend:ONNXBackend",
            # 3. llama.cpp — optimised native CPU inference with GGUF models
            "aether.backends.llamacpp_backend:LlamaCppBackend",
            # 4. Apple MLX — Apple Silicon (Metal) without PyTorch
            "aether.backends.mlx_backend:MLXBackend",
            # 5. vLLM — high-throughput GPU serving
            "aether.backends.vllm_backend:vLLMBackend",
            # 6. TensorRT-LLM — NVIDIA optimised GPU serving
            "aether.backends.trtllm_backend:TensorRTLLMBackend",
            # 7. PyTorch — LAST RESORT; requires torch + transformers installed
            "aether.backends.torch_backend:TorchBackend",
        ]

        for backend_ref in builtin_backends:
            try:
                module_name, class_name = backend_ref.split(":")
                module = importlib.import_module(module_name)
                backend_class = getattr(module, class_name)
                backend = backend_class()
                self._backends[backend.name] = backend
                logger.debug(f"Discovered backend: {backend.name}")
            except Exception as exc:
                logger.debug(f"Could not load backend {backend_ref}: {exc}")

        # Try entry points
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                backend_eps = eps.select(group="aether.backends")
            else:
                backend_eps = eps.get("aether.backends", [])
            for ep in backend_eps:
                try:
                    backend_class = ep.load()
                    backend = backend_class()
                    self._backends[backend.name] = backend
                    logger.debug(f"Loaded backend from entry point: {backend.name}")
                except Exception as exc:
                    logger.debug(f"Could not load entry point backend {ep.name}: {exc}")
        except Exception as exc:
            logger.debug(f"Entry point discovery skipped: {exc}")

    @property
    def backend_names(self) -> list[str]:
        """Return all registered backend names."""
        return sorted(self._backends.keys())

    def get_backend(self, name: str) -> Backend | None:
        """Look up a backend by name.

        Args:
            name: Backend name (e.g., 'pytorch', 'vllm').

        Returns:
            Backend instance or None if not found.
        """
        # Normalize aliases
        aliases = {
            "llama.cpp": "llamacpp",
            "tensorrt-llm": "trtllm",
            "tensorrt_llm": "trtllm",
            "onnxruntime": "onnx",
        }
        name = aliases.get(name, name)
        return self._backends.get(name)

    def get_available_backends(self) -> list[Backend]:
        """Return all backends that are currently available (installed)."""
        return [b for b in self._backends.values() if b.is_available()]

    def get_available_backend_names(self) -> list[str]:
        """Return names of available backends."""
        return sorted([b.name for b in self._backends.values() if b.is_available()])

    def register_backend(self, backend: Backend) -> None:
        """Register a backend instance manually."""
        self._backends[backend.name] = backend

    def __repr__(self) -> str:
        available = len(self.get_available_backends())
        return f"BackendRegistry({len(self._backends)} backends, {available} available)"
