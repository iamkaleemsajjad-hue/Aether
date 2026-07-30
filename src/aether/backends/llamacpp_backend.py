"""
llama.cpp backend for cross-platform CPU/GPU inference.

This backend wraps llama-cpp-python to run GGUF models on CPU or GPU with
K/I-quants. It is the preferred backend when the model is already quantized
in GGUF format.
"""

from __future__ import annotations

from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult


class LlamaCppBackend(Backend):
    """llama.cpp backend."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="llamacpp",
            version="1.0.0",
            supported_targets=["cpu_avx512", "cpu_neon", "cuda_sm70", "cuda_sm80", "cuda_sm89", "metal_m1", "metal_m3", "rocm_rdna3"],
            capabilities=["generate", "chat", "embed", "gguf", "cpu_offload"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a GGUF model using llama-cpp-python."""
        if model_id in self._models:
            return self._models[model_id]
        try:
            from llama_cpp import Llama
        except ImportError:
            msg = "llama-cpp-python is not installed. Install with: pip install aether-runtime[llamacpp]"
            raise ImportError(msg)

        llama_kwargs: dict[str, Any] = {
            "model_path": model_id,
            "n_ctx": kwargs.get("n_ctx", 2048),
            "n_threads": kwargs.get("n_threads", None),
            "n_gpu_layers": kwargs.get("n_gpu_layers", -1 if kwargs.get("use_gpu", False) else 0),
            "verbose": False,
        }
        llama_kwargs.update(kwargs)
        model = Llama(**llama_kwargs)
        self._models[model_id] = model
        return model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using llama.cpp."""
        model = self._models.get(request.model_id)
        if model is None:
            self.load_model(request.model_id)
            model = self._models[request.model_id]

        text = request.prompt or ""
        if request.messages:
            text = ""
            for msg in request.messages:
                text += f"{msg['role']}: {msg['content']}\n"
            text += "assistant:"

        output = model(
            text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop or [],
        )

        return GenerationResult(
            text=output["choices"][0]["text"],
            prompt_tokens=output["usage"]["prompt_tokens"],
            completion_tokens=output["usage"]["completion_tokens"],
            finish_reason=output["choices"][0].get("finish_reason", "stop"),
            backend_name=self.name,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"LlamaCppBackend(models={len(self._models)})"
