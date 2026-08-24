"""
PyTorch backend — universal fallback for Aether.

This backend loads models directly from HuggingFace or local safetensors using
PyTorch and the `transformers` library. It supports text generation, chat,
embeddings, and vision tasks. It is the default fallback when no specialized
backend (vLLM, llama.cpp, etc.) is available.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import BackendError
from aether.core.hash_utils import compute_file_hash
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class TorchBackend(Backend):
    """PyTorch-based backend for model inference."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="pytorch",
            version=AETHER_VERSION,
            supported_targets=[
                "cuda_sm70", "cuda_sm80", "cuda_sm89", "cuda_sm90", "cuda_sm100",
                "cpu_avx512", "cpu_neon", "rocm_rdna3", "rocm_cdna3", "metal_m1", "metal_m3",
            ],
            capabilities=[
                "generate", "chat", "embed", "rerank", "vision", "transcribe",
                "flash_attention", "cpu_offload", "bitsandbytes",
                "structured_output", "grammar_constraints",
            ],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._device: str = "cpu"
        self._devices: list[str] = ["cpu"]
        self._runtime_family: str = "cpu"
        self._allow_remote_code = False
        self._try_detect_device()

    def _try_detect_device(self) -> None:
        """Auto-detect the best available device for PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
                self._devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
                # ROCm is exposed as torch.cuda in many PyTorch builds.  The
                # tensor device remains ``cuda`` but dispatch must distinguish
                # the vendor runtime.
                self._runtime_family = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
                self._devices = ["mps"]
                self._runtime_family = "metal"
            else:
                self._device = "cpu"
                self._devices = ["cpu"]
                self._runtime_family = "cpu"
        except ImportError:
            self._device = "cpu"
            self._devices = ["cpu"]
            self._runtime_family = "cpu"

    def _should_shard(self, engine: Any) -> bool:
        """Decide whether a dense decoder should be tensor-parallel sharded.

        Tensor parallelism is a memory-capacity tool, not a throughput tool for
        small models: every layer copies activations between devices, and for a
        model that already fits on one GPU those copies serialize the decode
        pipeline and make it *slower* than single-device execution.  A 350M
        model split across two GPUs is the pathological case — it runs at about
        half the single-device rate.

        So sharding is selected only when at least one of these holds:

        * there are ``≥2`` accelerators AND the model's resident size does not
          comfortably fit in the smallest device's free memory (the genuine
          capacity case), or
        * the operator explicitly requested it via
          ``AETHER_FORCE_TENSOR_PARALLEL=1`` (benchmarking / correctness runs).

        Otherwise a single device is used.  A heterogeneous mesh that includes
        CPU is never auto-sharded: mixing a CPU shard into a GPU decode is far
        slower than either alone and must be opted into explicitly.
        """
        accelerators = [device for device in self._devices if device != "cpu"]
        if len(accelerators) < 2:
            return False
        force = os.environ.get("AETHER_FORCE_TENSOR_PARALLEL", "").strip().lower()
        if force in {"1", "true", "yes"}:
            return True
        if len(accelerators) != len(self._devices):
            # Heterogeneous CPU+GPU mesh: never automatic.
            return False
        try:
            required = self._estimated_weight_bytes(engine)
            headroom = self._smallest_free_accelerator_bytes(accelerators)
        except Exception:  # noqa: BLE001 — capacity probing must never block a load
            # If the probe fails, prefer the safe fast path: a single device
            # runs any model that the host could load in the first place.
            return False
        if required <= 0 or headroom <= 0:
            return False
        # Require the whole model plus a working margin for activations and the
        # KV cache (2x) to fit before choosing single-device execution.
        fits_on_one = required * 2.0 <= headroom
        if not fits_on_one:
            logger.info(
                "Model needs ~%.1f GiB but the smallest device has ~%.1f GiB free; "
                "activating tensor-parallel execution.",
                required / 1024**3,
                headroom / 1024**3,
            )
        return not fits_on_one

    @staticmethod
    def _estimated_weight_bytes(engine: Any) -> int:
        """Estimate the resident size of a compiled engine's weights.

        Uses the embedding and per-layer projection shapes already materialized
        by the loader, at two bytes per element (the FP16/BF16 accelerator
        residency), so it needs no second pass over the weight blob.
        """
        import numpy as np

        weights = engine.weights
        total = int(np.asarray(weights.embedding).size)
        if getattr(weights, "lm_head", None) is not None:
            total += int(np.asarray(weights.lm_head).size)
        for layer in weights.layers:
            for name in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    total += int(np.asarray(tensor).size)
            for expert in getattr(layer, "experts", None) or []:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    tensor = getattr(expert, name, None)
                    if tensor is not None:
                        total += int(np.asarray(tensor).size)
        return total * 2  # bytes per element at FP16/BF16 residency

    def _smallest_free_accelerator_bytes(self, accelerators: list[str]) -> int:
        """Return the free memory of the most constrained accelerator, in bytes."""
        import torch

        free_values: list[int] = []
        for device in accelerators:
            if device.startswith("cuda:"):
                index = int(device.split(":", 1)[1])
                free, _total = torch.cuda.mem_get_info(index)
                free_values.append(int(free))
            elif device == "mps":
                # MPS shares host memory and exposes no per-device free query;
                # treat it as ample so a unified-memory Mac never auto-shards.
                free_values.append(1 << 62)
        return min(free_values) if free_values else 0

    def _configure_devices(self, requested: list[str] | None) -> None:
        """Apply an explicit single-copy execution mesh.

        The default mesh contains every detected accelerator.  An explicit
        mesh may also include ``cpu`` for heterogeneous model parallelism.
        This method validates the mesh before any model tensor is materialized;
        it never falls back to a replicated ``device_map='auto'`` load.
        """
        values = requested
        if values is None:
            raw = os.environ.get("AETHER_EXECUTION_DEVICES", "").strip()
            values = [item.strip() for item in raw.split(",") if item.strip()] or None
        if values is None:
            return
        normalized = ["cpu" if item == "cpu:0" else item for item in values]
        if len(set(normalized)) != len(normalized):
            raise BackendError("execution device mesh contains duplicate device IDs", backend_name=self.name)
        for device in normalized:
            if device == "cpu":
                continue
            if device.startswith("cuda:"):
                try:
                    index = int(device.split(":", 1)[1])
                except ValueError as exc:
                    raise BackendError(f"invalid CUDA device ID {device!r}", backend_name=self.name) from exc
                import torch
                if not torch.cuda.is_available() or index < 0 or index >= torch.cuda.device_count():
                    raise BackendError(f"requested CUDA device {device!r} is unavailable", backend_name=self.name)
                continue
            if device == "mps":
                import torch
                if not torch.backends.mps.is_available():
                    raise BackendError("requested MPS device is unavailable", backend_name=self.name)
                continue
            raise BackendError(f"unsupported execution device {device!r}", backend_name=self.name)
        if not normalized:
            raise BackendError("execution device mesh must not be empty", backend_name=self.name)
        self._devices = normalized
        self._device = next((item for item in normalized if item != "cpu"), normalized[0])
        kinds = {"cpu" if item == "cpu" else item.split(":", 1)[0] for item in normalized}
        if len(kinds) > 1:
            self._runtime_family = "heterogeneous"
        elif "cuda" in kinds:
            self._runtime_family = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        elif "mps" in kinds:
            self._runtime_family = "metal"
        else:
            self._runtime_family = "cpu"

    def is_available(self) -> bool:
        """Return True when the optional PyTorch runtime is installed.

        Transformers is only needed for loading an uncompiled Hugging Face
        model.  A self-contained AEG carries its tokenizer and can execute
        with the documented ``[pytorch]`` extra (Torch plus the base
        ``tokenizers`` dependency), so Transformers must not gate accelerator
        backend discovery.
        """
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def available_for_target(self, target_id: str) -> bool:
        """Require an installed PyTorch package and a matching device runtime."""
        if not self.is_available():
            return False
        if target_id.startswith("cpu_"):
            return self._runtime_family == "cpu"
        if target_id.startswith("cuda_"):
            return self._runtime_family in {"cuda", "heterogeneous"} and any(
                device.startswith("cuda:") for device in self._devices
            )
        if target_id.startswith("rocm_"):
            return self._runtime_family in {"rocm", "heterogeneous"} and any(
                device.startswith("cuda:") for device in self._devices
            )
        if target_id.startswith("metal_"):
            return self._runtime_family in {"metal", "heterogeneous"} and any(
                device == "mps" for device in self._devices
            )
        return target_id in self.info.supported_targets

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a model and tokenizer from HuggingFace or local path.

        Args:
            model_id: HuggingFace model ID or local path.
            aeg_path: Optional AEG path (not used by PyTorch backend).
            kwargs: Additional arguments for model loading (e.g., torch_dtype).

        Returns:
            The loaded model instance.
        """
        if model_id in self._models:
            return self._models[model_id]

        self._configure_devices(kwargs.get("execution_devices"))

        compiled_handle = self._try_load_compiled_aeg(model_id, aeg_path)
        if compiled_handle is not None:
            self._models[model_id] = compiled_handle
            return compiled_handle

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict[str, Any] = {
            "torch_dtype": kwargs.get("torch_dtype", torch.float16 if self._device == "cuda" else torch.float32),
            "device_map": kwargs.get("device_map", "auto"),
            "trust_remote_code": bool(kwargs.get("trust_remote_code", False)),
        }
        self._allow_remote_code = bool(kwargs.get("trust_remote_code", False))
        if kwargs.get("offline", False):
            load_kwargs["local_files_only"] = True
        if "low_cpu_mem_usage" not in kwargs:
            load_kwargs["low_cpu_mem_usage"] = True
        # Runtime control arguments are not model-constructor arguments.  In
        # particular, passing ``offline`` or ``download_timeout_s`` through to
        # Transformers can make otherwise valid local/HF loads fail or be
        # interpreted by custom config classes.
        control_keys = {"offline", "download_timeout_s", "trust_remote_code", "execution_devices"}
        load_kwargs.update(
            {key: value for key, value in kwargs.items() if key not in control_keys}
        )

        start = time.perf_counter()
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(float(kwargs.get("download_timeout_s", os.environ.get("AETHER_HF_DOWNLOAD_TIMEOUT_S", "30"))))
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=self._allow_remote_code,
                local_files_only=bool(kwargs.get("offline", False)),
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:
            raise BackendError(
                f"Unable to load model {model_id!r}; no model was loaded and no synthetic fallback is permitted: {exc}",
                backend_name=self.name,
            ) from exc
        finally:
            socket.setdefaulttimeout(previous_timeout)
        load_time = time.perf_counter() - start
        self._models[model_id] = model
        self._tokenizers[model_id] = tokenizer
        return model

    def _try_load_compiled_aeg(self, model_id: str, aeg_path: str | None) -> CompiledAEGHandle | None:
        """Load local AEG metadata without contacting a model registry."""
        if aeg_path is None:
            return None
        root = Path(aeg_path)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_path = root / "graph" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        eval_report_path = root / "observability" / "eval_report.json"
        if eval_report_path.is_file():
            eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
            gate = eval_report.get("gate", {})
            if gate.get("passed") is False:
                raise BackendError(
                    "AEG artifact is rejected by its persisted evaluation gate: "
                    + ", ".join(gate.get("failing_benchmarks", [])),
                    backend_name=self.name,
                )
        precision_path = root / "weights" / "quantized" / "precision_map.json"
        precision_map = {}
        if precision_path.exists():
            precision_map = json.loads(precision_path.read_text(encoding="utf-8"))
        engine = None
        tokenizer = None
        lora_adapters: dict[str, dict[tuple[int, str], tuple[Any, Any, float]]] = {}
        try:
            from aether.runtime.aeg_loader import load_engine_from_path, package_is_runnable
            from aether.adapters.lora import load_compiled_lora_adapters

            from aether.core.aeg_format import AEGPackage

            package = AEGPackage(root)
            package.load()
            # Verify every declared artifact before decoding adapter bytes.
            # Adapter parsing is part of the untrusted AEG boundary.
            package.verify_integrity()
            lora_adapters = load_compiled_lora_adapters(root)
            if package_is_runnable(package):
                engine = load_engine_from_path(root)
                if self._device != "cpu":
                    if not package.supports_portable_backend("pytorch"):
                        raise BackendError(
                            "this AEG has no verified PyTorch portable execution contract "
                            f"for device {self._device!r}; accelerator target plans are not executable kernels",
                            backend_name=self.name,
                        )
                    from aether.runtime.torch_engine import TorchAEGEngine, TorchHybridAEGEngine
                    from aether.runtime.torch_state_engine import (
                        TorchMLAAEGEngine, TorchMambaAEGEngine,
                        TorchMamba2AEGEngine, TorchRWKVAEGEngine,
                    )
                    from aether.runtime.torch_transformer_engine import (
                        TorchEncoderAEGEngine, TorchSeq2SeqAEGEngine,
                    )

                    engine_kind = engine.__class__.__name__
                    # ── Multi-GPU tensor-parallel dispatch ─────────────────────
                    # For standard dense decoders (CPUExecutionEngine is what
                    # load_engine_from_path returns for decoder-only AEGs),
                    # capacity-weighted sharding across every device is
                    # available.  It is only selected when the model cannot be
                    # served from one device: see ``_should_shard``.
                    # Specialised architectures (MLA, SSM, encoder, seq2seq) have
                    # their own cross-device wrappers and are dispatched below.
                    _TP_ELIGIBLE_ENGINES = {"CPUExecutionEngine"}
                    if engine_kind in _TP_ELIGIBLE_ENGINES and self._should_shard(engine):
                        from aether.runtime.torch_tensor_parallel import TorchTensorParallelAEGEngine
                        logger.info(
                            "Activating tensor-parallel execution across %d GPUs: %s",
                            len(self._devices),
                            self._devices,
                        )
                        engine = TorchTensorParallelAEGEngine(engine, self._devices)
                    elif engine_kind == "EncoderExecutionEngine":
                        engine = TorchEncoderAEGEngine(engine, self._device, self._devices)
                    elif engine_kind == "Seq2SeqExecutionEngine":
                        engine = TorchSeq2SeqAEGEngine(engine, self._device, self._devices)
                    elif engine_kind == "HybridExecutionEngine":
                        engine = TorchHybridAEGEngine(engine, self._device, self._devices)
                    elif engine_kind == "MLAExecutionEngine":
                        engine = TorchMLAAEGEngine(engine, self._device, self._devices)
                    elif engine_kind == "MambaExecutionEngine":
                        engine = TorchMambaAEGEngine(engine, self._device, self._devices)
                    elif engine_kind == "Mamba2ExecutionEngine":
                        engine = TorchMamba2AEGEngine(engine, self._device, self._devices)
                    elif engine_kind == "RWKVExecutionEngine":
                        engine = TorchRWKVAEGEngine(engine, self._device, self._devices)
                    else:
                        engine = TorchAEGEngine(engine, self._device)
                tokenizer_root = root / "tokenizer"
                if tokenizer_root.exists():
                    # The tokenizer is part of the authenticated AEG.  Do
                    # not route this path through Transformers: compiled AEG
                    # execution is intentionally supported with only the
                    # optional PyTorch extra, without the HF frontend.
                    from aether.backends.native_cpu_backend import PackagedTokenizer

                    tokenizer = PackagedTokenizer(tokenizer_root / "tokenizer.json")
        except Exception as exc:
            raise BackendError(
                f"AEG artifact {root} failed integrity/load validation before execution: {exc}",
                backend_name=self.name,
            ) from exc
        return CompiledAEGHandle(
            model_id=model_id,
            aeg_path=root,
            manifest=manifest,
            metadata=metadata,
            precision_map=precision_map,
            engine=engine,
            tokenizer=tokenizer,
            lora_adapters=lora_adapters,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def release_session_cache(self, model_id: str, session_id: str) -> None:
        """Release a compiled-AEG session cache after its owner closes."""
        model = self._models.get(model_id)
        if isinstance(model, CompiledAEGHandle):
            model.clear_session_cache(session_id)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using a loaded model."""
        model = self._models.get(request.model_id)
        tokenizer = self._tokenizers.get(request.model_id)
        if isinstance(model, CompiledAEGHandle):
            return self._generate_from_compiled_aeg(model, request)
        if model is None or tokenizer is None:
            self.load_model(request.model_id)
            model = self._models[request.model_id]
            tokenizer = self._tokenizers.get(request.model_id)
        if isinstance(model, CompiledAEGHandle):
            return self._generate_from_compiled_aeg(model, request)
        if tokenizer is None:
            msg = f"Tokenizer for {request.model_id} was not loaded"
            raise ValueError(msg)

        import torch

        # Prepare input text
        if request.messages is not None:
            text = self._apply_chat_template(request.messages, tokenizer)
        elif request.prompt is not None:
            text = request.prompt
        else:
            msg = "Either prompt or messages must be provided"
            raise ValueError(msg)

        inputs = tokenizer(text, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[1]
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k if request.top_k > 0 else None,
            "do_sample": request.temperature > 0.0,
            "stop_strings": request.stop,
            "tokenizer": tokenizer,
        }
        grammar_session = request.extra.get("grammar_session")
        if grammar_session is not None:
            try:
                from transformers import (
                    LogitsProcessor,
                    LogitsProcessorList,
                    StoppingCriteria,
                    StoppingCriteriaList,
                )

                class _GrammarLogitsProcessor(LogitsProcessor):
                    def __init__(self, session: Any, prompt_length: int) -> None:
                        self.session = session
                        self.last_length = prompt_length

                    def __call__(self, input_ids: Any, scores: Any) -> Any:
                        current_length = int(input_ids.shape[-1])
                        if current_length > self.last_length:
                            for token_id in input_ids[0, self.last_length:current_length].tolist():
                                if self.session.advance(int(token_id)) < 0:
                                    raise BackendError(
                                        "The model produced a token rejected by the grammar FSM",
                                        backend_name="pytorch",
                                    )
                            self.last_length = current_length
                        mask = self.session.get_token_mask()
                        if len(mask) * 8 < int(scores.shape[-1]):
                            raise BackendError(
                                "Grammar FSM vocabulary is smaller than model vocabulary",
                                backend_name="pytorch",
                            )
                        invalid = [
                            token_id for token_id in range(int(scores.shape[-1]))
                            if not (mask[token_id // 8] & (1 << (token_id % 8)))
                        ]
                        if len(invalid) == int(scores.shape[-1]):
                            raise BackendError(
                                "Grammar FSM has no valid next token",
                                backend_name="pytorch",
                            )
                        scores[:, invalid] = -float("inf")
                        return scores

                class _GrammarStoppingCriteria(StoppingCriteria):
                    def __init__(self, session: Any) -> None:
                        self.session = session

                    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                        return bool(self.session.is_accepting())

                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    [_GrammarLogitsProcessor(grammar_session, int(input_tokens))]
                )
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_GrammarStoppingCriteria(grammar_session)]
                )
            except ImportError as exc:
                raise BackendError(
                    "Grammar-constrained generation requires transformers logits processors",
                    backend_name=self.name,
                ) from exc
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        end = time.perf_counter()
        generated_ids = outputs[0][input_tokens:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_tokens = len(generated_ids)
        ttft_ms = (end - start) * 1000
        tps = completion_tokens / max(end - start, 1e-6)

        return GenerationResult(
            text=generated_text,
            prompt_tokens=input_tokens,
            completion_tokens=completion_tokens,
            finish_reason="length" if completion_tokens >= request.max_tokens else "stop",
            backend_name=self.name,
            metrics={
                "ttft_ms": ttft_ms,
                "throughput_tps": tps,
                "device": self._device,
            },
        )

    def _generate_from_compiled_aeg(self, handle: CompiledAEGHandle, request: GenerationRequest) -> GenerationResult:
        if handle.engine is not None and handle.tokenizer is not None:
            text = self._request_text(request, handle.tokenizer)
            encoded = handle.tokenizer(text, return_tensors="np")
            prompt_ids = encoded["input_ids"][0]
            start = time.perf_counter()
            execution_engine, reweight_metrics = self._engine_for_task_weights(
                handle, request.extra.get("task_weights")
            )
            adapter_id = request.extra.get("adapter_id", request.extra.get("adapter_name"))
            execution_engine, adapter_metrics = self._engine_for_lora(
                handle, execution_engine, adapter_id
            )
            ttt_state = self._begin_ttt(handle, prompt_ids, request, execution_engine)
            ttt_slots = ttt_state[2] if ttt_state is not None else None
            session_id = request.extra.get("aether_kv_session_id")
            cache_session_id = None if ttt_state is not None else session_id
            reused_tokens = 0
            multi_agent_reused_tokens = 0
            cache = None
            suffix = prompt_ids

            # R2 is connected to normal compiled-CPU generation through an
            # explicit shared-prefix boundary. The coordinator owns one
            # immutable prefix cache; each agent clones it before appending its
            # private suffix, so divergent requests cannot corrupt one another.
            coordinator = request.extra.get("multi_agent_kv_coordinator")
            prefix_text = request.extra.get("multi_agent_prefix")
            prefix_hash = request.extra.get("multi_agent_prefix_hash")
            if (
                ttt_state is None
                and coordinator is not None
                and isinstance(prefix_text, str)
                and prefix_text
                and isinstance(prefix_hash, str)
                and prefix_hash
            ):
                import numpy as np

                if coordinator.hash_prefix(prefix_text) != prefix_hash:
                    raise BackendError(
                        "R2 multi-agent prefix hash does not match its text",
                        backend_name=self.name,
                    )
                prefix_encoded = handle.tokenizer(prefix_text, return_tensors="np")
                prefix_ids = np.asarray(prefix_encoded["input_ids"][0], dtype=np.int64)
                candidate = np.asarray(prompt_ids, dtype=np.int64)
                if (
                    prefix_ids.size == 0
                    or candidate.size < prefix_ids.size
                    or not np.array_equal(candidate[: prefix_ids.size], prefix_ids)
                ):
                    raise BackendError(
                        "R2 multi-agent prefix is not an exact token prefix of the request",
                        backend_name=self.name,
                    )
                shared_cache, shared_length = coordinator.get_shared_kv(prefix_hash)
                if shared_cache is None:
                    _prefix_logits, shared_cache = execution_engine.forward(prefix_ids)
                    coordinator.update_shared_kv(
                        prefix_hash,
                        shared_cache,
                        seq_len=int(prefix_ids.size),
                    )
                    shared_length = int(prefix_ids.size)
                else:
                    multi_agent_reused_tokens = int(shared_length)
                if int(shared_length) != int(prefix_ids.size):
                    raise BackendError(
                        "R2 shared KV length does not match the tokenized prefix",
                        backend_name=self.name,
                    )
                cache = shared_cache.clone()
                suffix = candidate[prefix_ids.size :]

            cached_state = (
                handle.session_caches.get(cache_session_id)
                if cache is None and isinstance(cache_session_id, str)
                else None
            )
            if cached_state is not None:
                import numpy as np

                cached_ids, cached_cache = cached_state
                cached_ids = np.asarray(cached_ids, dtype=np.int64)
                candidate = np.asarray(prompt_ids, dtype=np.int64)
                if candidate.size >= cached_ids.size and np.array_equal(candidate[: cached_ids.size], cached_ids):
                    cache = cached_cache
                    suffix = candidate[cached_ids.size :]
                    reused_tokens = int(cached_ids.size)
                else:
                    # A session may only reuse an exact token prefix.  Drop
                    # stale state instead of silently serving the wrong cache.
                    handle.clear_session_cache(cache_session_id)

            try:
                peagle_engine = request.extra.get("peagle_engine")
                if cache is not None or isinstance(cache_session_id, str):
                    generated_ids, updated_cache = execution_engine.generate_with_cache(
                        suffix,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                        grammar_session=request.extra.get("grammar_session"),
                        cache=cache,
                        ttt_slots=ttt_slots,
                        adapter_id=adapter_id,
                        peagle_engine=peagle_engine,
                    )
                else:
                    generated_ids = execution_engine.generate(
                        prompt_ids,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                        grammar_session=request.extra.get("grammar_session"),
                        ttt_slots=ttt_slots,
                        adapter_id=adapter_id,
                        peagle_engine=peagle_engine,
                    )
                    updated_cache = None
            finally:
                self._end_ttt(ttt_state)

            if isinstance(cache_session_id, str):
                import numpy as np

                full_ids = np.concatenate(
                    [np.asarray(prompt_ids, dtype=np.int64), np.asarray(generated_ids, dtype=np.int64)]
                )
                if updated_cache is not None:
                    handle.session_caches[cache_session_id] = (full_ids, updated_cache)
            generated_text = handle.tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion_tokens = len(generated_ids)
            finish_reason = "length" if len(generated_ids) >= request.max_tokens else "stop"
            if request.stop:
                generated_text, completion_tokens, stopped = self._truncate_stop_text(
                    handle.tokenizer, generated_ids, request.stop
                )
                if stopped:
                    finish_reason = "stop"
            elapsed = time.perf_counter() - start
            speculative_metrics: dict[str, Any] = {}
            stats_fn = getattr(execution_engine, "speculative_stats", None)
            if callable(stats_fn):
                stats = stats_fn()
                if int(stats.get("draft_tokens", 0)) > 0:
                    speculative_metrics["speculative"] = stats
            return GenerationResult(
                text=generated_text,
                prompt_tokens=int(len(prompt_ids)),
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                backend_name=self.name,
                metrics={
                    "ttft_ms": elapsed * 1000.0,
                    "throughput_tps": len(generated_ids) / max(elapsed, 1e-9),
                    "device": str(getattr(execution_engine, "device", self._device)),
                    "kv_reuse": reused_tokens > 0,
                    "kv_reused_tokens": reused_tokens,
                    "multi_agent_kv_reuse": multi_agent_reused_tokens > 0,
                    "multi_agent_kv_reused_tokens": multi_agent_reused_tokens,
                    **(
                        {"ttt_adaptation_loss": ttt_state[3]}
                        if ttt_state is not None
                        else {}
                    ),
                    **(
                        {"task_reweighting": reweight_metrics}
                        if reweight_metrics is not None
                        else {}
                    ),
                    **({"lora_adapter": adapter_metrics} if adapter_metrics is not None else {}),
                    **speculative_metrics,
                },
            )
        raise BackendError(
            f"AEG {handle.aeg_path} contains compiled graph data but no tokenizer-backed "
            "generation adapter for the PyTorch backend. Refusing to return fabricated output.",
            backend_name=self.name,
        )

    @staticmethod
    def _truncate_stop_text(
        tokenizer: Any,
        generated_ids: Any,
        stops: list[str],
    ) -> tuple[str, int, bool]:
        """Apply string stop sequences to compiled CPU output without lying about tokens."""
        full_ids = list(generated_ids)
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        first_cutoff = min(
            (full_text.find(stop) for stop in stops if stop and stop in full_text),
            default=-1,
        )
        if first_cutoff < 0:
            return full_text, len(full_ids), False
        token_count = len(full_ids)
        for count in range(1, len(full_ids) + 1):
            prefix = tokenizer.decode(full_ids[:count], skip_special_tokens=True)
            if any(stop and stop in prefix for stop in stops):
                token_count = count
                break
        return full_text[:first_cutoff], token_count, True

    def _request_text(self, request: GenerationRequest, tokenizer: Any | None = None) -> str:
        """Return the text represented by a generation request."""
        if request.messages is not None:
            if (
                tokenizer is not None
                and hasattr(tokenizer, "apply_chat_template")
                and getattr(tokenizer, "chat_template", None) is not None
            ):
                return str(
                    tokenizer.apply_chat_template(
                        request.messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            return "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in request.messages
            ) + "\nassistant:"
        return request.prompt or ""

    def _begin_ttt(
        self,
        handle: CompiledAEGHandle,
        prompt_ids: Any,
        request: GenerationRequest,
        execution_engine: Any | None = None,
    ) -> tuple[Any, str, list[dict[str, Any]], float] | None:
        """Adapt a persisted R5 slot set from real prompt embeddings."""
        engine = request.extra.get("ttt_engine")
        execution_engine = execution_engine or handle.engine
        if engine is None or execution_engine is None:
            return None
        request_id = str(request.extra.get("ttt_request_id") or uuid.uuid4().hex)
        engine.begin_request(request_id)
        try:
            import numpy as np

            ids = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
            hidden = execution_engine.weights.embedding[ids].astype(np.float32).tolist()
            loss = float(engine.adapt(request_id, hidden))
            slots = []
            for layer_index in range(execution_engine.weights.num_layers):
                slot = engine.get_fast_weights(request_id, layer_index)
                if slot is None:
                    raise BackendError(
                        f"R5 TTT slot {layer_index} was not available after adaptation",
                        backend_name=self.name,
                    )
                slots.append(slot)
            return engine, request_id, slots, loss
        except Exception:
            engine.end_request(request_id)
            raise

    def _engine_for_task_weights(
        self,
        handle: CompiledAEGHandle,
        task_weights: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Load authenticated task deltas into a request-local CPU engine."""
        if not task_weights:
            return handle.engine, None
        if not isinstance(task_weights, dict):
            raise BackendError("task_weights must be a mapping", backend_name=self.name)
        metadata = handle.metadata.get("task_vectors", {})
        vectors = metadata.get("vectors", []) if isinstance(metadata, dict) else []
        if not isinstance(vectors, list) or not vectors:
            raise BackendError(
                "the AEG does not contain executable task-vector payloads",
                backend_name=self.name,
            )
        manifest_artifacts = handle.manifest.get("artifacts", {})
        available = {str(vector.get("name")) for vector in vectors if isinstance(vector, dict)}
        unknown = sorted(set(task_weights) - available)
        if unknown:
            raise BackendError(
                f"task_weights reference unknown vectors {unknown}", backend_name=self.name
            )
        import numpy as np

        deltas: dict[str, np.ndarray] = {}
        applied: list[str] = []
        tensor_count = 0
        for vector in vectors:
            if not isinstance(vector, dict):
                raise BackendError("malformed task-vector descriptor", backend_name=self.name)
            name = str(vector.get("name", ""))
            coefficient = task_weights.get(name, 0.0)
            if not isinstance(coefficient, (int, float)) or coefficient < 0:
                raise BackendError(f"invalid task weight for {name!r}", backend_name=self.name)
            if coefficient == 0:
                continue
            relative_path = vector.get("path")
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise BackendError(f"unsafe task-vector path for {name!r}", backend_name=self.name)
            expected_hash = manifest_artifacts.get(relative_path)
            path = (handle.aeg_path / relative_path).resolve()
            if expected_hash is None or not path.is_file() or compute_file_hash(path) != expected_hash:
                raise BackendError(
                    f"task-vector payload failed AEG integrity validation: {relative_path}",
                    backend_name=self.name,
                )
            try:
                archive = np.load(path, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(
                    f"unable to read task-vector payload {relative_path}",
                    backend_name=self.name,
                ) from exc
            with archive:
                for descriptor in vector.get("tensors", []):
                    tensor_name = descriptor.get("name")
                    key = descriptor.get("key")
                    shape = tuple(int(value) for value in descriptor.get("shape", []))
                    if (
                        not isinstance(tensor_name, str)
                        or not isinstance(key, str)
                        or not shape
                        or key not in archive
                    ):
                        raise BackendError(
                            f"malformed task-vector tensor descriptor in {name!r}",
                            backend_name=self.name,
                        )
                    array = np.asarray(archive[key], dtype=np.float32)
                    if int(np.prod(shape)) != array.size:
                        raise BackendError(
                            f"task-vector tensor shape mismatch for {tensor_name!r}",
                            backend_name=self.name,
                        )
                    contribution = np.ascontiguousarray(
                        array.reshape(shape), dtype=np.float32
                    ) * np.float32(coefficient)
                    if tensor_name in deltas:
                        deltas[tensor_name] += contribution
                    else:
                        deltas[tensor_name] = contribution
                    tensor_count += 1
            applied.append(name)
        if not applied:
            raise BackendError(
                "task_weights selected no non-zero task-vector payload",
                backend_name=self.name,
            )
        try:
            engine = handle.engine.with_task_deltas(deltas)
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"task-vector deltas do not match the compiled model: {exc}",
                backend_name=self.name,
            ) from exc
        return engine, {"vectors": applied, "tensor_count": tensor_count}

    def _engine_for_lora(
        self,
        handle: CompiledAEGHandle,
        engine: Any,
        adapter_id: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Select a verified compiled adapter for this request."""
        if adapter_id is None:
            return engine, None
        if not isinstance(adapter_id, str) or not adapter_id:
            raise BackendError("adapter_id must be a non-empty string", backend_name=self.name)
        if not handle.lora_adapters:
            raise BackendError(
                "adapter_id was requested but the AEG contains no executable adapter artifacts",
                backend_name=self.name,
            )
        try:
            selected = engine.with_lora_adapter(handle.lora_adapters, adapter_id)
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"compiled LoRA adapter {adapter_id!r} cannot be applied: {exc}",
                backend_name=self.name,
            ) from exc
        return selected, {"adapter_id": adapter_id, "targets": len(handle.lora_adapters[adapter_id])}

    @staticmethod
    def _end_ttt(state: tuple[Any, str, list[dict[str, Any]], float] | None) -> None:
        if state is not None:
            state[0].end_request(state[1])

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Stream generated text as the backend produces tokens."""
        model = self._models.get(request.model_id)
        if isinstance(model, CompiledAEGHandle):
            yield from self._generate_compiled_aeg_stream(model, request)
            return

        import threading
        import torch

        tokenizer = self._tokenizers.get(request.model_id)
        if model is None or tokenizer is None:
            self.load_model(request.model_id)
            model = self._models[request.model_id]
            tokenizer = self._tokenizers[request.model_id]

        text = self._apply_chat_template(request.messages, tokenizer) if request.messages else (request.prompt or "")
        inputs = tokenizer(text, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise BackendError(
                "streaming generation requires transformers.TextIteratorStreamer",
                backend_name=self.name,
            ) from exc

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1.0
        )
        failure: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k if request.top_k > 0 else None,
                        do_sample=request.temperature > 0.0,
                        streamer=streamer,
                    )
            except BaseException as exc:  # noqa: BLE001
                failure.append(exc)
                streamer.end()

        worker = threading.Thread(target=run_generation, name="aether-generation", daemon=True)
        worker.start()
        try:
            for chunk in streamer:
                yield str(chunk)
        finally:
            worker.join(timeout=30.0)
        if failure:
            raise BackendError(
                f"streaming generation failed: {failure[0]}", backend_name=self.name
            ) from failure[0]

    def _generate_compiled_aeg_stream(
        self, handle: CompiledAEGHandle, request: GenerationRequest
    ) -> Any:
        """Stream token deltas from the executable CPU AEG engine."""
        if handle.engine is None or handle.tokenizer is None:
            raise BackendError(
                f"AEG {handle.aeg_path} has no executable tokenizer-backed engine",
                backend_name=self.name,
            )
        import numpy as np

        text = self._request_text(request, handle.tokenizer)
        encoded = handle.tokenizer(text, return_tensors="np")
        prompt_ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
        execution_engine, _reweight_metrics = self._engine_for_task_weights(
            handle, request.extra.get("task_weights")
        )
        adapter_id = request.extra.get("adapter_id", request.extra.get("adapter_name"))
        execution_engine, _adapter_metrics = self._engine_for_lora(
            handle, execution_engine, adapter_id
        )
        ttt_state = self._begin_ttt(handle, prompt_ids, request, execution_engine)
        ttt_slots = ttt_state[2] if ttt_state is not None else None
        session_id = request.extra.get("aether_kv_session_id")
        cache_session_id = None if ttt_state is not None else session_id
        cached_state = (
            handle.session_caches.get(cache_session_id)
            if isinstance(cache_session_id, str)
            else None
        )
        cache = None
        suffix = prompt_ids
        if cached_state is not None:
            cached_ids, cached_cache = cached_state
            cached_ids = np.asarray(cached_ids, dtype=np.int64)
            if prompt_ids.size >= cached_ids.size and np.array_equal(prompt_ids[: cached_ids.size], cached_ids):
                cache = cached_cache
                suffix = prompt_ids[cached_ids.size :]
            else:
                handle.clear_session_cache(cache_session_id)

        updated_cache: list[Any] = [None]

        def remember(value: Any) -> None:
            updated_cache[0] = value

        token_ids: list[int] = []
        emitted = False
        previous = ""
        try:
            iterator = execution_engine.generate_iter(
                suffix,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                grammar_session=request.extra.get("grammar_session"),
                cache=cache,
                cache_callback=remember,
                ttt_slots=ttt_slots,
                adapter_id=adapter_id,
            )
            for token_id in iterator:
                token_ids.append(int(token_id))
                decoded = handle.tokenizer.decode(
                    token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                if decoded.startswith(previous):
                    delta = decoded[len(previous) :]
                else:
                    # Some tokenizers normalize preceding whitespace. In that case
                    # preserve the actual decoded text rather than dropping output.
                    delta = decoded
                previous = decoded
                if delta:
                    emitted = True
                    yield delta
        finally:
            self._end_ttt(ttt_state)

        if isinstance(cache_session_id, str) and updated_cache[0] is not None:
            full_ids = np.concatenate([prompt_ids, np.asarray(token_ids, dtype=np.int64)])
            handle.session_caches[cache_session_id] = (full_ids, updated_cache[0])
        if not emitted:
            yield ""

    def _apply_chat_template(self, messages: list[dict[str, str]], tokenizer: Any) -> str:
        """Apply chat template if available; otherwise fallback to concatenation."""
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Fallback: simple concatenation
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def chat(self, messages: list[dict[str, str]], request: GenerationRequest) -> GenerationResult:
        """Chat completion using the chat template."""
        request.messages = messages
        return self.generate(request)

    def embed(self, model_id: str, inputs: list[str]) -> list[list[float]]:
        """Generate embeddings using a sentence-transformers style model."""
        compiled = self._models.get(model_id)
        if isinstance(compiled, CompiledAEGHandle) and hasattr(compiled.engine, "pooled"):
            if compiled.tokenizer is None:
                raise BackendError("encoder AEG has no packaged tokenizer", backend_name=self.name)
            if compiled.tokenizer.pad_token_id is None:
                # A packaged tokenizer is allowed to omit padding (common for
                # minimal WordLevel fixtures and decoder-derived tokenizers).
                # Encode each request independently rather than inventing a
                # pad token that could be a real model input.
                results: list[list[float]] = []
                for text in inputs:
                    encoded = compiled.tokenizer(text, return_tensors="np", truncation=True)
                    results.extend(
                        compiled.engine.embed(
                            encoded["input_ids"],
                            encoded.get("attention_mask"),
                            encoded.get("token_type_ids"),
                        )
                    )
                return results
            encoded = compiled.tokenizer(
                inputs, return_tensors="np", padding=True, truncation=True
            )
            return compiled.engine.embed(
                encoded["input_ids"], encoded.get("attention_mask"), encoded.get("token_type_ids")
            )
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            msg = "transformers is required for embeddings"
            raise ImportError(msg)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self._allow_remote_code)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=self._allow_remote_code)
        import torch
        encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**encodings)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def rerank(self, model_id: str, query: str, documents: list[str]) -> list[dict[str, Any]]:
        """Rerank documents using a cross-encoder style model."""
        compiled = self._models.get(model_id)
        if isinstance(compiled, CompiledAEGHandle) and hasattr(compiled.engine, "pooled"):
            if compiled.tokenizer is None:
                raise BackendError("encoder AEG has no packaged tokenizer", backend_name=self.name)
            query_ids = compiled.tokenizer(query, return_tensors="np", truncation=True)
            query_vector = np.asarray(
                compiled.engine.embed(
                    query_ids["input_ids"], query_ids.get("attention_mask"), query_ids.get("token_type_ids")
                )[0], dtype=np.float32
            )
            scored: list[tuple[int, float]] = []
            for index, document in enumerate(documents):
                encoded = compiled.tokenizer(document, return_tensors="np", truncation=True)
                vector = np.asarray(
                    compiled.engine.embed(
                        encoded["input_ids"], encoded.get("attention_mask"), encoded.get("token_type_ids")
                    )[0], dtype=np.float32
                )
                denominator = float(np.linalg.norm(query_vector) * np.linalg.norm(vector))
                score = float(np.dot(query_vector, vector) / denominator) if denominator else 0.0
                scored.append((index, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            return [
                {"index": index, "document": documents[index], "score": score}
                for index, score in scored
            ]
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            msg = "transformers is required for reranking"
            raise ImportError(msg)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self._allow_remote_code)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=self._allow_remote_code)
        import torch
        scores: list[float] = []
        for doc in documents:
            enc = tokenizer(query, doc, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**enc).logits
            score = logits[0, 0].item() if logits.shape[1] == 1 else logits.softmax(dim=1)[0, 1].item()
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            {"index": idx, "document": documents[idx], "score": score}
            for idx, score in ranked
        ]

    def transcribe(self, model_id: str, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio using a Whisper model."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            msg = "transformers is required for transcription"
            raise ImportError(msg)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self._allow_remote_code)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=self._allow_remote_code)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        result = pipe(audio_path, generate_kwargs={"language": language} if language else {})
        return result["text"]

    def __repr__(self) -> str:
        return f"TorchBackend(device={self._device}, loaded={len(self._models)} models)"


class _SimpleStreamer:
    """Minimal streamer that yields tokens as they are generated."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.tokens: list[int] = []

    def put(self, value: Any) -> None:
        self.tokens.extend(value.tolist() if hasattr(value, "tolist") else [value])

    def end(self) -> None:
        pass

    def __iter__(self) -> Any:
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        yield text
