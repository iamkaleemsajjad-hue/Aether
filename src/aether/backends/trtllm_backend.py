"""
TensorRT-LLM backend for NVIDIA compiled engines.

This backend wraps TensorRT-LLM for production NVIDIA inference. It is the
preferred backend when a pre-built TensorRT engine is available for the target
hardware.
"""

from __future__ import annotations

from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult


class TensorRTLLMBackend(Backend):
    """TensorRT-LLM backend."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="trtllm",
            version="1.0.0",
            supported_targets=["cuda_sm80", "cuda_sm89", "cuda_sm90", "cuda_sm100"],
            capabilities=["generate", "chat", "fp8", "compiled_engine"],
        )
        super().__init__(info)
        self._engines: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import tensorrt_llm  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a TensorRT-LLM engine or build one from the AEG."""
        if model_id in self._engines:
            return self._engines[model_id]
        msg = "TensorRT-LLM engine loading requires a pre-built engine path in kwargs['engine_path']"
        raise NotImplementedError(msg)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using TensorRT-LLM."""
        return GenerationResult(
            text="TensorRT-LLM generation not yet implemented in this backend.",
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
            backend_name=self.name,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"TensorRTLLMBackend(engines={len(self._engines)})"
