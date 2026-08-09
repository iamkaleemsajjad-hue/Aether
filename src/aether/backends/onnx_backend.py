"""
ONNX Runtime backend for cross-platform inference.

This backend wraps ONNX Runtime and is particularly useful for Intel OpenVINO
NPUs, Azure execution providers, and cross-platform CPU deployment.
"""

from __future__ import annotations

from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.exceptions import BackendError


class ONNXBackend(Backend):
    """ONNX Runtime backend."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="onnx",
            version="1.0.0",
            supported_targets=["cpu_avx512", "cpu_neon", "openvino_npu"],
            capabilities=["generate", "chat", "embed", "cross_platform"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load an ONNX model."""
        if model_id in self._models:
            return self._models[model_id]
        try:
            import onnxruntime as ort
        except ImportError:
            msg = "onnxruntime is not installed. Install with: pip install aether-runtime[onnxruntime]"
            raise ImportError(msg)

        providers = kwargs.get("providers", ort.get_available_providers())
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_id, sess_options, providers=providers)
        self._models[model_id] = session
        return session

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using an ONNX autoregressive runtime.

        A plain ONNX Runtime session only exposes tensor inference; it does not
        provide tokenization, sampling, or an autoregressive decode loop. Aether
        therefore fails closed here until an ONNX artifact carries a compatible
        tokenizer and decode adapter instead of returning fabricated text.
        """
        session = self._models.get(request.model_id)
        if session is None:
            session = self.load_model(request.model_id)
        raise BackendError(
            "ONNX Runtime model loaded, but no autoregressive tokenizer/decode "
            "adapter is attached to this artifact; refusing fabricated output.",
            backend_name=self.name,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"ONNXBackend(models={len(self._models)})"
