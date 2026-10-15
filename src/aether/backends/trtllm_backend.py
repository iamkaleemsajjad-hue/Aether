"""
TensorRT-LLM backend for NVIDIA compiled engines.

This interface reserves the TensorRT-LLM integration. Engine loading and the
inference runner are not implemented; an installed SDK or an engine artifact
does not make this backend ready for inference.
"""

from __future__ import annotations

from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import BackendError


class TensorRTLLMBackend(Backend):
    """TensorRT-LLM backend."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="trtllm",
            version=AETHER_VERSION,
            supported_targets=["cuda_sm80", "cuda_sm89", "cuda_sm90", "cuda_sm100"],
            capabilities=[],
        )
        super().__init__(info)
        self._engines: dict[str, Any] = {}
        self.unavailable_reason = (
            "TensorRT-LLM inference is unavailable: the engine loading/runner integration "
            "is not implemented; package or compiled-engine presence is not inference readiness"
        )

    def is_available(self) -> bool:
        return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a TensorRT-LLM engine or build one from the AEG."""
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using TensorRT-LLM."""
        raise BackendError(
            "TensorRT-LLM generation is unavailable until a real engine is loaded",
            backend_name=self.name,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"TensorRTLLMBackend(engines={len(self._engines)})"
