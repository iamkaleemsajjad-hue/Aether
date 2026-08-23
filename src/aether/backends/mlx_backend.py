"""
MLX backend for Apple Silicon.

This backend wraps Apple's MLX framework for native inference on M1-M5 Macs.
"""

from __future__ import annotations

from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult


class MLXBackend(Backend):
    """MLX backend for Apple Silicon."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="mlx",
            version="1.2.3",
            supported_targets=["metal_m1", "metal_m3"],
            capabilities=["generate", "chat", "embed", "unified_memory"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import mlx.core  # noqa: F401
            import platform
            return platform.system() == "Darwin"
        except ImportError:
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a model from HuggingFace or local into MLX."""
        if model_id in self._models:
            return self._models[model_id]
        try:
            from mlx_lm import load
        except ImportError:
            msg = "mlx-lm is not installed. Install with: pip install mlx-lm"
            raise ImportError(msg)
        model, tokenizer = load(model_id)
        self._models[model_id] = (model, tokenizer)
        return (model, tokenizer)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using MLX."""
        try:
            from mlx_lm import generate
        except ImportError:
            msg = "mlx-lm is not installed"
            raise ImportError(msg)

        model, tokenizer = self._models.get(request.model_id) or self.load_model(request.model_id)
        text = request.prompt or ""
        if request.messages:
            text = ""
            for msg in request.messages:
                text += f"{msg['role']}: {msg['content']}\n"
            text += "assistant:"

        output = generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=request.max_tokens,
            temp=request.temperature,
            top_p=request.top_p,
            verbose=False,
        )
        return GenerationResult(
            text=output,
            prompt_tokens=0,
            completion_tokens=request.max_tokens,
            finish_reason="stop",
            backend_name=self.name,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"MLXBackend(models={len(self._models)})"
