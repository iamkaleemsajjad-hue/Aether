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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.exceptions import BackendError


@dataclass
class CompiledAEGHandle:
    """Lightweight handle for locally compiled AEG artifacts.

    This handle is used for a compiled artifact with a real executable engine
    and tokenizer.  Session-scoped CPU KV caches are kept here rather than in
    the global Runtime so independent models and tenants cannot accidentally
    share state.
    """

    model_id: str
    aeg_path: Path
    manifest: dict[str, Any]
    precision_map: dict[str, str] = field(default_factory=dict)
    engine: Any | None = None
    tokenizer: Any | None = None
    session_caches: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def clear_session_cache(self, session_id: str) -> None:
        """Release the incremental KV state owned by one agentic session."""
        self.session_caches.pop(session_id, None)

    @property
    def architecture_family(self) -> str:
        architecture = self.manifest.get("architecture", {})
        return str(architecture.get("family", "unknown"))


class TorchBackend(Backend):
    """PyTorch-based backend for model inference."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="pytorch",
            version="1.0.0",
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
        self._allow_remote_code = False
        self._try_detect_device()

    def _try_detect_device(self) -> None:
        """Auto-detect the best available device for PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

    def is_available(self) -> bool:
        """Return True if PyTorch and transformers are installed."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

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
        load_kwargs.update(kwargs)

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
        try:
            from aether.runtime.aeg_loader import load_engine_from_path, package_is_runnable

            from aether.core.aeg_format import AEGPackage

            package = AEGPackage(root)
            package.load()
            if package_is_runnable(package):
                engine = load_engine_from_path(root)
                tokenizer_root = root / "tokenizer"
                if tokenizer_root.exists():
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_root,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
        except Exception as exc:
            raise BackendError(
                f"AEG artifact {root} declares executable weights but could not be loaded: {exc}",
                backend_name=self.name,
            ) from exc
        return CompiledAEGHandle(
            model_id=model_id,
            aeg_path=root,
            manifest=manifest,
            precision_map=precision_map,
            engine=engine,
            tokenizer=tokenizer,
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
                from transformers import LogitsProcessor, LogitsProcessorList

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

                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    [_GrammarLogitsProcessor(grammar_session, int(input_tokens))]
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
            text = self._request_text(request)
            encoded = handle.tokenizer(text, return_tensors="np")
            prompt_ids = encoded["input_ids"][0]
            start = time.perf_counter()
            session_id = request.extra.get("aether_kv_session_id")
            reused_tokens = 0
            cached_state = handle.session_caches.get(session_id) if isinstance(session_id, str) else None
            cache = None
            suffix = prompt_ids
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
                    handle.clear_session_cache(session_id)

            if cache is not None or isinstance(session_id, str):
                generated_ids, updated_cache = handle.engine.generate_with_cache(
                    suffix,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                    grammar_session=request.extra.get("grammar_session"),
                    cache=cache,
                )
            else:
                generated_ids = handle.engine.generate(
                    prompt_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                    grammar_session=request.extra.get("grammar_session"),
                )
                updated_cache = None

            if isinstance(session_id, str):
                import numpy as np

                full_ids = np.concatenate(
                    [np.asarray(prompt_ids, dtype=np.int64), np.asarray(generated_ids, dtype=np.int64)]
                )
                if updated_cache is not None:
                    handle.session_caches[session_id] = (full_ids, updated_cache)
            generated_text = handle.tokenizer.decode(generated_ids, skip_special_tokens=True)
            elapsed = time.perf_counter() - start
            return GenerationResult(
                text=generated_text,
                prompt_tokens=int(len(prompt_ids)),
                completion_tokens=len(generated_ids),
                finish_reason="length" if len(generated_ids) >= request.max_tokens else "stop",
                backend_name=self.name,
                metrics={
                    "ttft_ms": elapsed * 1000.0,
                    "throughput_tps": len(generated_ids) / max(elapsed, 1e-9),
                    "device": "cpu",
                    "kv_reuse": reused_tokens > 0,
                    "kv_reused_tokens": reused_tokens,
                },
            )
        raise BackendError(
            f"AEG {handle.aeg_path} contains compiled graph data but no tokenizer-backed "
            "generation adapter for the PyTorch backend. Refusing to return fabricated output.",
            backend_name=self.name,
        )

    def _request_text(self, request: GenerationRequest) -> str:
        """Return the text represented by a generation request."""
        if request.messages is not None:
            return " ".join(message.get("content", "") for message in request.messages)
        return request.prompt or ""

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

        text = self._request_text(request)
        encoded = handle.tokenizer(text, return_tensors="np")
        prompt_ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
        session_id = request.extra.get("aether_kv_session_id")
        cached_state = handle.session_caches.get(session_id) if isinstance(session_id, str) else None
        cache = None
        suffix = prompt_ids
        if cached_state is not None:
            cached_ids, cached_cache = cached_state
            cached_ids = np.asarray(cached_ids, dtype=np.int64)
            if prompt_ids.size >= cached_ids.size and np.array_equal(prompt_ids[: cached_ids.size], cached_ids):
                cache = cached_cache
                suffix = prompt_ids[cached_ids.size :]
            else:
                handle.clear_session_cache(session_id)

        updated_cache: list[Any] = [None]

        def remember(value: Any) -> None:
            updated_cache[0] = value

        token_ids: list[int] = []
        emitted = False
        previous = ""
        iterator = handle.engine.generate_iter(
            suffix,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
            grammar_session=request.extra.get("grammar_session"),
            cache=cache,
            cache_callback=remember,
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

        if isinstance(session_id, str) and updated_cache[0] is not None:
            full_ids = np.concatenate([prompt_ids, np.asarray(token_ids, dtype=np.int64)])
            handle.session_caches[session_id] = (full_ids, updated_cache[0])
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
