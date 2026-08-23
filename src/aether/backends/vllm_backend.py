"""
vLLM backend for NVIDIA high-throughput serving.

This backend wraps vLLM's continuous batching and PagedAttention. It is the
preferred backend on NVIDIA CUDA hardware when vLLM is installed.
"""

from __future__ import annotations

from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.constants import AETHER_VERSION


class vLLMBackend(Backend):
    """vLLM backend."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="vllm",
            version=AETHER_VERSION,
            supported_targets=["cuda_sm80", "cuda_sm89", "cuda_sm90", "cuda_sm100", "rocm_cdna3"],
            capabilities=["generate", "chat", "embed", "rerank", "paged_attention", "speculative"],
        )
        super().__init__(info)
        self._engines: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a model using vLLM's LLM engine."""
        if model_id in self._engines:
            return self._engines[model_id]
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            msg = "vllm is not installed. Install with: pip install aether-runtime[vllm]"
            raise ImportError(msg)

        engine_args: dict[str, Any] = {
            "model": model_id,
            "dtype": kwargs.get("dtype", "bfloat16"),
            "max_model_len": kwargs.get("max_model_len", 4096),
            "enforce_eager": kwargs.get("enforce_eager", False),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
        }
        engine_args.update(kwargs)
        llm = LLM(**engine_args)
        self._engines[model_id] = llm
        return llm

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using vLLM."""
        llm = self._engines.get(request.model_id)
        if llm is None:
            self.load_model(request.model_id)
            llm = self._engines[request.model_id]
        try:
            from vllm import SamplingParams
        except ImportError:
            msg = "vllm is not installed"
            raise ImportError(msg)

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else None,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        text = request.prompt or ""
        if request.messages:
            # vLLM handles chat templates automatically for supported models
            text = request.prompt or ""
            for msg in request.messages:
                text += f"{msg['role']}: {msg['content']}\n"
            text += "assistant:"

        outputs = llm.generate([text], sampling_params)
        completion = outputs[0]
        return GenerationResult(
            text=completion.outputs[0].text,
            prompt_tokens=len(completion.prompt_token_ids),
            completion_tokens=len(completion.outputs[0].token_ids),
            finish_reason=completion.outputs[0].finish_reason,
            backend_name=self.name,
            metrics={
                "throughput_tps": completion.outputs[0].metrics.throughput if hasattr(completion.outputs[0], "metrics") else None,
            },
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"vLLMBackend(engines={len(self._engines)})"
