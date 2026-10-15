"""vLLM backend for NVIDIA high-throughput serving.

This backend wraps vLLM's continuous batching and PagedAttention. It is the
preferred backend on NVIDIA CUDA hardware when vLLM is installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.constants import AETHER_VERSION, SUPPORTED_TARGET_IDS
from aether.core.exceptions import BackendError


class vLLMBackend(Backend):  # noqa: N801 - preserve the public backend class name
    """vLLM backend."""

    model_formats = frozenset({"source"})

    def __init__(self) -> None:
        info = BackendInfo(
            name="vllm",
            version=AETHER_VERSION,
            supported_targets=sorted(
                target
                for target in SUPPORTED_TARGET_IDS
                if target.startswith(("cuda_", "rocm_", "amd_mi"))
                and target != "cuda_sm70"
                and not target.endswith("_tee")
            ),
            capabilities=["generate", "chat", "paged_attention", "speculative"],
        )
        super().__init__(info)
        self._engines: dict[str, Any] = {}
        self._model_devices: dict[str, tuple[str, ...]] = {}

    def is_available(self) -> bool:
        """Require vLLM and a compatible framework-visible accelerator."""
        try:
            import vllm  # noqa: F401

            from aether.backends.torch_backend import TorchBackend

            runtime = TorchBackend()
            return any(
                runtime.available_for_target(target) for target in self.info.supported_targets
            )
        except (ImportError, OSError, RuntimeError):
            return False

    def available_for_target(self, target_id: str) -> bool:
        """Check this specific target instead of package presence alone."""
        if target_id not in self.info.supported_targets or not self.is_available():
            return False
        from aether.backends.torch_backend import TorchBackend

        return TorchBackend().available_for_target(target_id)

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a model using vLLM's LLM engine."""
        if not self.available_for_model(model_id, aeg_path):
            raise BackendError(
                "vLLM loads source checkpoints; it does not consume AEG/GGUF/ONNX artifacts",
                backend_name=self.name,
            )
        mesh = tuple(kwargs.get("execution_devices") or ())
        if model_id in self._model_devices and self._model_devices[model_id] != mesh:
            raise BackendError(
                "Model is already loaded on another execution mesh; unload it first",
                backend_name=self.name,
            )
        if model_id in self._engines:
            return self._engines[model_id]
        try:
            from vllm import LLM
        except ImportError as exc:
            msg = "vllm is not installed. Install with: pip install aether-runtime[vllm]"
            raise ImportError(msg) from exc

        options = dict(kwargs)
        offline = options.pop("offline", False)
        options.pop("download_timeout_s", None)
        devices = options.pop("execution_devices", None)
        if offline and not Path(model_id).is_dir():
            from huggingface_hub import snapshot_download

            source = snapshot_download(model_id, local_files_only=True)
        else:
            source = model_id
        if devices:
            # vLLM creates workers in the visible-device namespace. Do not
            # mutate global CUDA_VISIBLE_DEVICES or silently ignore an ordinal.
            import torch

            count = int(torch.cuda.device_count())
            normalized = ["cuda:0" if item == "cuda" else item for item in devices]
            expected = [f"cuda:{index}" for index in range(len(normalized))]
            if normalized != expected or len(normalized) > count:
                raise BackendError(
                    "vLLM explicit meshes must be a contiguous CUDA/HIP prefix "
                    "in the visible-device namespace; set visibility before "
                    "starting Aether for other meshes",
                    backend_name=self.name,
                )
            options.setdefault("tensor_parallel_size", len(normalized))
            if options["tensor_parallel_size"] != len(normalized):
                raise BackendError(
                    "vLLM tensor_parallel_size must match execution_devices", backend_name=self.name
                )
        engine_args: dict[str, Any] = {
            "model": source,
            "dtype": kwargs.get("dtype", "auto"),
            "max_model_len": kwargs.get("max_model_len", 4096),
            "enforce_eager": kwargs.get("enforce_eager", False),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
        }
        engine_args.update(options)
        llm = LLM(**engine_args)
        self._engines[model_id] = llm
        self._model_devices[model_id] = mesh
        return llm

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using vLLM."""
        llm = self._engines.get(request.model_id)
        if llm is None:
            self.load_model(request.model_id)
            llm = self._engines[request.model_id]
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            msg = "vllm is not installed"
            raise ImportError(msg) from exc

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        text = request.prompt or ""
        if request.messages:
            tokenizer = llm.get_tokenizer()
            text = tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )

        start = time.perf_counter()
        outputs = llm.generate([text], sampling_params)
        elapsed = time.perf_counter() - start
        completion = outputs[0]
        return GenerationResult(
            text=completion.outputs[0].text,
            prompt_tokens=len(completion.prompt_token_ids),
            completion_tokens=len(completion.outputs[0].token_ids),
            finish_reason=completion.outputs[0].finish_reason,
            backend_name=self.name,
            metrics={
                "throughput_tps": len(completion.outputs[0].token_ids) / max(elapsed, 1e-9),
                "measurement_source": "token_ids_and_wall_clock",
            },
        )

    def get_capabilities(self) -> list[str]:
        """Return implemented frontend operations."""
        return self.info.capabilities

    def generate_stream(self, request: GenerationRequest) -> Any:
        """The synchronous vLLM frontend returns one buffered completion.

        Incremental token streaming requires vLLM's separate async serving API;
        do not fabricate token chunks or first-token latency here.
        """
        text = self.generate(request).text
        if text:
            yield text

    def unload_model(self, model_id: str) -> None:
        """Release this model's engine and device assignment."""
        self._engines.pop(model_id, None)
        self._model_devices.pop(model_id, None)

    def __repr__(self) -> str:
        return f"vLLMBackend(engines={len(self._engines)})"
