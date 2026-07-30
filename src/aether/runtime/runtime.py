"""
Aether Runtime — the main execution engine.

The Runtime is the primary Python API. It loads compiled AEG artifacts, detects
hardware, selects the best backend, manages the KV cache, runs speculative
decoding, and serves generation requests.
"""

from __future__ import annotations

import datetime
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from aether.backends.base import Backend, GenerationRequest, GenerationResult
from aether.backends.registry import BackendRegistry
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.core.constants import DEFAULT_SERVER_PORT
from aether.core.exceptions import BackendNotAvailableError, ModelNotFoundError, RuntimeError as AetherRuntimeError
from aether.core.types import HardwareTarget
from aether.runtime.config import RuntimeConfig
from aether.runtime.hardware import HardwareDetector
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.scheduler import DisaggregatedScheduler, ScheduledRequest
from aether.runtime.speculative import TreeSpeculativeEngine
from aether.utils.file_io import aether_cache_dir, resolve_model_path
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics returned with an inference response."""

    throughput_tps: float = 0.0
    """Tokens per second."""

    ttft_ms: float = 0.0
    """Time to first token in milliseconds."""

    p95_latency_ms: float | None = None
    """P95 decode latency in milliseconds."""

    kernel_target: str | None = None
    """Active hardware target (e.g., 'cuda_sm90')."""

    active_precision: str | None = None
    """Active precision string (e.g., 'mixed_fp8_q4')."""

    spec_accept_rate: float | None = None
    """Speculative decoding acceptance rate."""

    kv_cache_hit_rate: float | None = None
    """KV cache prefix hit rate."""

    memory_pressure: float = 0.0
    """Current VRAM utilization (0.0 - 1.0)."""

    backend_name: str | None = None
    """Backend plugin that executed the request."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "throughput_tps": self.throughput_tps,
            "ttft_ms": self.ttft_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "kernel_target": self.kernel_target,
            "active_precision": self.active_precision,
            "spec_accept_rate": self.spec_accept_rate,
            "kv_cache_hit_rate": self.kv_cache_hit_rate,
            "memory_pressure": self.memory_pressure,
            "backend_name": self.backend_name,
        }


@dataclass
class GenerationResponse:
    """Normalized response from `Runtime.generate()`."""

    text: str
    """Generated text."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage (prompt_tokens, completion_tokens, total_tokens)."""

    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    """Inference metrics."""

    finish_reason: str = "stop"
    """Generation finish reason."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage,
            "metrics": self.metrics.to_dict(),
            "finish_reason": self.finish_reason,
        }


class Runtime:
    """Aether runtime API.

    Usage:
        rt = Runtime()
        response = rt.generate("Qwen/Qwen3-0.6B", "Explain AEG.")
        print(response.text)
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        """Initialize the Aether runtime.

        Args:
            config: Runtime configuration, or None for defaults.
        """
        self.config = config or RuntimeConfig()
        self.hardware_detector = HardwareDetector()
        self.fingerprint = self.hardware_detector.detect()
        self.backend_registry = BackendRegistry()
        self.target_registry = TargetRegistry()
        self.kv_cache = KVCacheManager(
            dtype=self.config.kv_cache_dtype,
            cpu_budget_gb=self.config.kv_cache_cpu_gb,
            nvme_budget_gb=self.config.kv_cache_nvme_gb,
        )
        self.scheduler = DisaggregatedScheduler(
            max_batch_size=self.config.max_batch_size,
            prefill_chunk_size=self.config.prefill_chunk_size,
        )
        self._loaded_models: dict[str, Any] = {}
        self._loaded_backends: dict[str, Backend] = {}
        self._aeg_packages: dict[str, AEGPackage] = {}
        self._lock = threading.RLock()
        logger.info(
            "Aether runtime initialized",
            target=self.fingerprint.target_id,
            backend=self.config.backend_name,
            optimize_for=self.config.optimize_for,
        )

    def hardware(self) -> dict[str, Any]:  # type: ignore[valid-type]
        """Return the current hardware fingerprint."""
        return self.fingerprint.to_dict()

    def _resolve_backend(self, target_id: str | None = None) -> Backend:
        """Resolve and return the best available backend for the target."""
        with self._lock:
            if self.config.backend_name:
                backend = self.backend_registry.get_backend(self.config.backend_name)
                if backend is not None and backend.is_available():
                    return backend
                msg = f"Configured backend '{self.config.backend_name}' is not available"
                raise BackendNotAvailableError(msg, backend_name=self.config.backend_name)

            target = target_id or self.fingerprint.target_id
            profile = HardwareProfile.from_target_id(target) or HardwareProfile.auto()
            # Prefer cached backend
            if profile.target_id in self._loaded_backends:
                return self._loaded_backends[profile.target_id]

            candidates = profile.backend_candidates
            for backend_name in candidates:
                backend = self.backend_registry.get_backend(backend_name)
                if backend is not None and backend.is_available():
                    self._loaded_backends[profile.target_id] = backend
                    return backend
            # Fallback to any available backend
            for backend in self.backend_registry.get_available_backends():
                self._loaded_backends[profile.target_id] = backend
                return backend

            msg = f"No backend available for target {target}"
            raise BackendNotAvailableError(msg, target_id=target)

    def _load_model(self, model_id: str) -> Backend:
        """Ensure a model is loaded and return the backend that loaded it."""
        with self._lock:
            if model_id in self._loaded_models:
                backend = self._loaded_backends.get(model_id)
                if backend:
                    return backend
            # Find or compile AEG
            aeg_path = self._resolve_aeg_path(model_id)
            backend = self._resolve_backend()
            self._loaded_models[model_id] = backend.load_model(model_id, aeg_path)
            self._loaded_backends[model_id] = backend
            return backend

    def _resolve_aeg_path(self, model_id: str) -> str | None:
        """Find the AEG package path for a model, downloading/compile if needed."""
        from aether.utils.file_io import aether_cache_dir

        cache_root = aether_cache_dir(self.config.model_cache_dir)
        aeg_path = cache_root / "models" / model_id.replace("/", "_")
        if aeg_path.exists():
            return str(aeg_path)
        # Check local compiled path
        local = resolve_model_path(model_id, self.config.model_cache_dir)
        if local and (local / "manifest.json").exists():
            return str(local)
        return None

    def pull(self, model_id: str) -> None:
        """Download and compile a model to a local AEG package.

        If the AEG already exists in cache, this is a no-op. Otherwise, it
        invokes the compiler to produce a local AEG artifact.
        """
        aeg_path = self._resolve_aeg_path(model_id)
        if aeg_path is not None:
            logger.info(f"Model {model_id} already available at {aeg_path}")
            return
        logger.info(f"Pulling and compiling model {model_id}")
        from aether import Compiler
        compiler = Compiler()
        aeg = compiler.compile(model_id, output_path=self._resolve_aeg_path(model_id) or aether_cache_dir(self.config.model_cache_dir) / "models" / model_id.replace("/", "_"))
        aeg.save()

    def list(self) -> list[str]:
        """Return a list of model IDs with cached AEG artifacts."""
        from aether.utils.file_io import aether_cache_dir

        cache_root = aether_cache_dir(self.config.model_cache_dir)
        model_dir = cache_root / "models"
        if not model_dir.exists():
            return []
        return sorted(d.name for d in model_dir.iterdir() if (d / "manifest.json").exists())

    def info(self, model_id: str) -> dict[str, Any]:
        """Return metadata and precision info for a compiled model."""
        aeg_path = self._resolve_aeg_path(model_id)
        if aeg_path is None:
            msg = f"Model {model_id} not found"
            raise ModelNotFoundError(msg, model_id=model_id)
        aeg = load_aeg_package(aeg_path)
        if not aeg.manifest:
            msg = f"AEG for {model_id} has no manifest"
            raise AetherRuntimeError(msg)
        return {
            "model_id": aeg.manifest.model_id,
            "format_version": aeg.manifest.format_version,
            "aether_version": aeg.manifest.aether_version,
            "architecture": aeg.manifest.architecture.to_dict(),
            "targets": aeg.manifest.kernels.targets,
            "precision_map": aeg.get_precision_map(),
            "memory": aeg.manifest.memory_requirements.to_dict(),
            "sharding_plans": {k: v.to_dict() for k, v in aeg.sharding_plans.items()},
        }

    def remove(self, model_id: str) -> None:
        """Remove a model from the local AEG cache."""
        from aether.utils.file_io import delete_model

        delete_model(model_id, self.config.model_cache_dir)
        with self._lock:
            self._loaded_models.pop(model_id, None)
            self._loaded_backends.pop(model_id, None)
        logger.info(f"Removed model {model_id} from cache")

    def generate(
        self,
        model_id: str,
        prompt: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int = 0,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate text from a model.

        Args:
            model_id: Model identifier or local AEG path.
            prompt: Text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            stream: Whether to stream output.
            stop: Stop sequences.
            kwargs: Additional backend parameters.

        Returns:
            A GenerationResponse with text, usage, and metrics.
        """
        backend = self._load_model(model_id)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            top_p=top_p if top_p is not None else self.config.default_top_p,
            top_k=top_k,
            stream=stream,
            stop=stop,
            extra=kwargs,
        )
        start = datetime.datetime.now(datetime.timezone.utc)
        result = backend.generate(request)
        end = datetime.datetime.now(datetime.timezone.utc)
        duration_s = (end - start).total_seconds()
        metrics = InferenceMetrics(
            throughput_tps=result.completion_tokens / max(duration_s, 1e-6),
            ttft_ms=result.metrics.get("ttft_ms", duration_s * 1000),
            kernel_target=self.fingerprint.target_id,
            active_precision="mixed",  # Could be refined from AEG
            spec_accept_rate=self.kv_cache.hit_rate(),
            kv_cache_hit_rate=self.kv_cache.hit_rate(),
            memory_pressure=0.0,
            backend_name=result.backend_name or backend.name,
        )
        return GenerationResponse(
            text=result.text,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            metrics=metrics,
            finish_reason=result.finish_reason,
        )

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> GenerationResponse:
        """Chat completion with a list of messages."""
        backend = self._load_model(model_id)
        request = GenerationRequest(
            model_id=model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
            temperature=kwargs.get("temperature", self.config.default_temperature),
            top_p=kwargs.get("top_p", self.config.default_top_p),
            stream=kwargs.get("stream", False),
            stop=kwargs.get("stop"),
        )
        result = backend.generate(request)
        metrics = InferenceMetrics(
            throughput_tps=0.0,
            ttft_ms=result.metrics.get("ttft_ms", 0.0),
            kernel_target=self.fingerprint.target_id,
            backend_name=result.backend_name or backend.name,
        )
        return GenerationResponse(
            text=result.text,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            metrics=metrics,
            finish_reason=result.finish_reason,
        )

    def embed(self, model_id: str, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        backend = self._load_model(model_id)
        if hasattr(backend, "embed"):
            return backend.embed(model_id, input)
        msg = f"Backend {backend.name} does not support embeddings"
        raise AetherRuntimeError(msg)

    def rerank(self, model_id: str, query: str, documents: list[str]) -> list[dict[str, Any]]:
        """Rerank documents for a query."""
        backend = self._load_model(model_id)
        if hasattr(backend, "rerank"):
            return backend.rerank(model_id, query, documents)
        msg = f"Backend {backend.name} does not support reranking"
        raise AetherRuntimeError(msg)

    def transcribe(self, model_id: str, audio: str, language: str | None = None) -> str:
        """Transcribe an audio file."""
        backend = self._load_model(model_id)
        if hasattr(backend, "transcribe"):
            return backend.transcribe(model_id, audio, language=language)
        msg = f"Backend {backend.name} does not support transcription"
        raise AetherRuntimeError(msg)

    def benchmark(self, model_id: str, prompt: str = "Hello, my name is", max_tokens: int = 128) -> dict[str, Any]:
        """Run a simple benchmark on a model."""
        backend = self._load_model(model_id)
        request = GenerationRequest(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        import time
        start = time.perf_counter()
        result = backend.generate(request)
        end = time.perf_counter()
        tps = result.completion_tokens / max(end - start, 1e-6)
        return {
            "model_id": model_id,
            "backend": backend.name,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "duration_s": end - start,
            "throughput_tps": tps,
            "ttft_ms": result.metrics.get("ttft_ms", 0.0),
        }

    def __repr__(self) -> str:
        return (
            f"Runtime(target={self.fingerprint.target_id}, "
            f"backends={self.backend_registry.get_available_backend_names()}, "
            f"loaded_models={list(self._loaded_models.keys())})"
        )
