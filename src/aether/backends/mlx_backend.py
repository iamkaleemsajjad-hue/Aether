"""MLX backend for Apple Silicon.

This backend wraps Apple's MLX framework for native inference on M1-M5 Macs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import BackendError


class MLXBackend(Backend):
    """MLX backend for Apple Silicon."""

    model_formats = frozenset({"source"})

    def __init__(self) -> None:
        info = BackendInfo(
            name="mlx",
            version=AETHER_VERSION,
            supported_targets=["metal_m1", "metal_m3"],
            capabilities=["generate", "chat", "stream", "unified_memory"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}

    def is_available(self) -> bool:
        """Require usable Metal and the actual MLX language-model frontend."""
        try:
            import platform

            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler

            return bool(
                platform.system() == "Darwin"
                and mx.metal.is_available()
                and all(callable(item) for item in (load, stream_generate, make_sampler))
            )
        except (ImportError, OSError, AttributeError, RuntimeError):
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a model from HuggingFace or local into MLX."""
        if not self.available_for_model(model_id, aeg_path):
            raise BackendError(
                "MLX loads source checkpoints; AEG/GGUF/ONNX export is not implicit",
                backend_name=self.name,
            )
        if kwargs.get("execution_devices") not in (None, ["mps"]):
            raise BackendError(
                "MLX requires its Metal device; other execution meshes are unsupported",
                backend_name=self.name,
            )
        if model_id in self._models:
            return self._models[model_id]
        try:
            from mlx_lm import load
        except ImportError as exc:
            msg = "mlx-lm is not installed. Install with: pip install mlx-lm"
            raise ImportError(msg) from exc
        source = model_id
        if kwargs.get("offline") and not Path(source).is_dir():
            from huggingface_hub import snapshot_download

            source = snapshot_download(source, local_files_only=True)
        trust_remote_code = bool(kwargs.get("trust_remote_code", False))
        model, tokenizer = load(source, tokenizer_config={"trust_remote_code": trust_remote_code})
        self._models[model_id] = (model, tokenizer)
        return (model, tokenizer)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using MLX."""
        chunks, result = self._events(request)
        for _ in chunks:
            pass
        return result

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Yield incremental text while withholding possible stop prefixes."""
        chunks, _ = self._events(request)
        yield from chunks

    def _events(self, request: GenerationRequest) -> tuple[Any, GenerationResult]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._models.get(request.model_id) or self.load_model(request.model_id)
        prompt = request.prompt or ""
        if request.messages:
            prompt = tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p, top_k=request.top_k)
        start = time.perf_counter()
        return self._stream_result(
            model, tokenizer, prompt, sampler, request, stream_generate, start
        )

    def _stream_result(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        sampler: Any,
        request: GenerationRequest,
        stream_generate: Any,
        start: float,
    ) -> tuple[Any, GenerationResult]:
        result = GenerationResult(text="", backend_name=self.name, finish_reason="length")

        def chunks():
            first = True
            emitted = 0
            stopped = False
            for event in stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=request.max_tokens, sampler=sampler
            ):
                if first:
                    result.metrics["ttft_ms"] = (time.perf_counter() - start) * 1000
                    first = False
                result.prompt_tokens = int(event.prompt_tokens)
                result.completion_tokens = int(event.generation_tokens)
                delta = str(event.text)
                result.text += delta
                reason = getattr(event, "finish_reason", None)
                if reason is not None:
                    result.finish_reason = reason
                stops = [stop for stop in (request.stop or []) if stop]
                matches = [result.text.find(stop) for stop in stops if stop in result.text]
                if matches:
                    result.text = result.text[: min(matches)]
                    result.finish_reason = "stop"
                    stopped = True
                    boundary = len(result.text)
                else:
                    hold = max(
                        (
                            length
                            for stop in stops
                            for length in range(1, len(stop))
                            if result.text.endswith(stop[:length])
                        ),
                        default=0,
                    )
                    boundary = len(result.text) - hold
                if boundary > emitted:
                    yield result.text[emitted:boundary]
                    emitted = boundary
                if stopped:
                    break
            if not stopped and emitted < len(result.text):
                yield result.text[emitted:]
            elapsed = time.perf_counter() - start
            result.metrics["throughput_tps"] = result.completion_tokens / max(elapsed, 1e-9)

        return chunks(), result

    def get_capabilities(self) -> list[str]:
        """Return implemented frontend operations."""
        return self.info.capabilities

    def unload_model(self, model_id: str) -> None:
        """Release this model and its tokenizer."""
        self._models.pop(model_id, None)

    def __repr__(self) -> str:
        return f"MLXBackend(models={len(self._models)})"
