"""
Backend plugin base interface.

All execution backends (vLLM, llama.cpp, TensorRT-LLM, MLX, ONNX Runtime,
PyTorch) implement the `Backend` abstract class. Aether's runtime selects and
orchestrates backends through this stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendInfo:
    """Metadata about a backend plugin."""

    name: str
    """Short backend name (e.g., 'vllm', 'llama.cpp')."""

    version: str
    """Backend version string."""

    supported_targets: list[str] = field(default_factory=list)
    """List of hardware target IDs supported by this backend."""

    capabilities: list[str] = field(default_factory=list)
    """Capability strings (e.g., 'flash_attention', 'speculative', 'quantization')."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "supported_targets": self.supported_targets,
            "capabilities": self.capabilities,
        }


@dataclass
class GenerationRequest:
    """A normalized generation request passed to a backend."""

    model_id: str
    """Model identifier."""

    prompt: str | None = None
    """Text prompt for completion."""

    messages: list[dict[str, str]] | None = None
    """Chat messages for chat completion."""

    max_tokens: int = 1024
    """Maximum number of tokens to generate."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Top-p sampling parameter."""

    top_k: int = 0
    """Top-k sampling parameter."""

    stream: bool = False
    """Whether to stream output."""

    stop: list[str] | None = None
    """Stop sequences."""

    images: list[str] | None = None
    """Optional image paths for vision models."""

    audio: str | None = None
    """Optional audio path for transcription models."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Backend-specific extra parameters."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt": self.prompt,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.stream,
            "stop": self.stop,
            "images": self.images,
            "audio": self.audio,
            "extra": self.extra,
        }


@dataclass
class GenerationResult:
    """A normalized generation result returned from a backend."""

    text: str
    """Generated text."""

    prompt_tokens: int = 0
    """Number of input tokens."""

    completion_tokens: int = 0
    """Number of generated tokens."""

    finish_reason: str = "stop"
    """Finish reason (e.g., 'stop', 'length')."""

    backend_name: str = "unknown"
    """Name of the backend that produced the result."""

    target_id: str | None = None
    """Hardware target ID used."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Backend-specific metrics (tps, ttft_ms, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "backend_name": self.backend_name,
            "target_id": self.target_id,
            "metrics": self.metrics,
        }


class Backend(ABC):
    """Abstract base class for all Aether execution backends."""

    def __init__(self, info: BackendInfo) -> None:
        self.info = info

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def version(self) -> str:
        return self.info.version

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is installed and can be used."""
        raise NotImplementedError

    def available_for_target(self, target_id: str) -> bool:
        """Return whether this backend can execute on ``target_id`` now.

        Installation availability alone is insufficient for dispatch: an
        installed CPU backend must not satisfy a CUDA, ROCm, or Metal request.
        Vendor backends can override this method when they need a stronger
        driver/device probe.
        """
        return bool(
            self.is_available()
            and (
                not self.info.supported_targets
                or target_id in self.info.supported_targets
            )
        )

    @abstractmethod
    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a model into the backend."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text for a single request."""
        raise NotImplementedError

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Generate text as a stream of chunks.

        Backends may override this; default implementation raises NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Return the backend capabilities."""
        raise NotImplementedError

    def supports(self, capability: str) -> bool:
        """Check if this backend supports a capability."""
        return capability in self.get_capabilities()

    def __repr__(self) -> str:
        return f"Backend({self.name}, {self.version})"
