"""
ONNX Runtime backend for cross-platform inference.

This backend wraps ONNX Runtime and is particularly useful for Intel OpenVINO
NPUs, Azure execution providers, and cross-platform CPU deployment.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.exceptions import BackendError


class ONNXBackend(Backend):
    """ONNX Runtime backend."""

    def __init__(self) -> None:
        info = BackendInfo(
            name="onnx",
            version="1.0.0",
            supported_targets=["cpu_avx512", "cpu_neon", "openvino_npu", "openvino_gpu"],
            capabilities=["generate", "chat", "stream", "embed", "cross_platform"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._model_options: dict[str, dict[str, Any]] = {}

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
        try:
            session = ort.InferenceSession(model_id, sess_options, providers=providers)
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"failed to create ONNX Runtime session for {model_id!r}: {exc}",
                backend_name=self.name,
            ) from exc
        self._models[model_id] = session
        self._model_options[model_id] = {
            key: kwargs[key]
            for key in (
                "input_ids_name",
                "attention_mask_name",
                "position_ids_name",
                "logits_output_name",
                "eos_token_id",
            )
            if key in kwargs
        }
        tokenizer = kwargs.get("tokenizer")
        tokenizer_path = kwargs.get("tokenizer_path")
        if tokenizer is not None and tokenizer_path is not None:
            raise ValueError("provide tokenizer or tokenizer_path, not both")
        if tokenizer_path is not None:
            tokenizer = self._load_tokenizer(tokenizer_path)
        if tokenizer is not None:
            self._tokenizers[model_id] = tokenizer
        return session

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text with an explicit tokenizer-backed ONNX decode loop.

        The ONNX model must expose ``input_ids`` plus optional attention and
        position inputs, and return logits shaped ``[batch, sequence, vocab]``
        or ``[batch, vocab]``. KV-cache inputs and arbitrary encoder/decoder
        contracts are rejected until a dedicated adapter is supplied.
        """
        completed: GenerationResult | None = None
        for event in self._decode_events(request):
            if event["type"] == "done":
                completed = event["result"]
        if completed is None:
            raise BackendError(
                "ONNX decode loop ended without a completion result",
                backend_name=self.name,
            )
        return completed

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Yield decoded token deltas from the real ONNX decode loop."""
        for event in self._decode_events(request):
            if event["type"] == "token" and event["text"]:
                yield event["text"]

    def _decode_events(self, request: GenerationRequest) -> Any:
        """Run one validated decode loop and emit token/done events."""
        session = self._models.get(request.model_id)
        if session is None:
            session = self.load_model(request.model_id)
        tokenizer = self._tokenizers.get(request.model_id)
        request_tokenizer = request.extra.get("tokenizer")
        if request_tokenizer is not None:
            tokenizer = request_tokenizer
        if tokenizer is None:
            raise BackendError(
                "ONNX Runtime model loaded, but no tokenizer/decode adapter is "
                "attached; refusing fabricated output.",
                backend_name=self.name,
            )

        try:
            prompt = self._build_prompt(request, tokenizer)
            prompt_ids = self._encode(tokenizer, prompt)
            if not prompt_ids:
                raise ValueError("tokenizer produced no prompt tokens")
            max_tokens = int(request.max_tokens)
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            options = dict(self._model_options.get(request.model_id, {}))
            options.update({
                key: request.extra[key]
                for key in (
                    "input_ids_name",
                    "attention_mask_name",
                    "position_ids_name",
                    "logits_output_name",
                    "eos_token_id",
                )
                if key in request.extra
            })
            input_specs = list(session.get_inputs())
            output_specs = list(session.get_outputs())
            input_ids_name = self._resolve_input_name(
                input_specs, options.get("input_ids_name"), ("input_ids", "input_ids:0", "ids")
            )
            self._reject_unsupported_inputs(input_specs, input_ids_name, options)
            output_name = self._resolve_output_name(output_specs, options.get("logits_output_name"))
            eos_token_id = options.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
            generated: list[int] = []
            emitted_text = ""
            finish_reason = "length"
            start = time.perf_counter()

            for _ in range(max_tokens):
                token_array = np.asarray(
                    [prompt_ids + generated],
                    dtype=self._integer_dtype(input_specs, input_ids_name),
                )
                feeds = {input_ids_name: token_array}
                self._add_optional_inputs(feeds, input_specs, token_array.shape[1], options)
                try:
                    raw_outputs = session.run([output_name], feeds)
                except Exception as exc:  # noqa: BLE001
                    raise BackendError(
                        f"ONNX autoregressive execution failed: {exc}",
                        backend_name=self.name,
                    ) from exc
                logits = self._extract_logits(raw_outputs[0], output_name)
                next_id = self._sample(logits, request)
                generated.append(next_id)
                decoded = self._decode(tokenizer, generated)
                cutoff = min(
                    (decoded.find(stop) for stop in (request.stop or []) if stop and stop in decoded),
                    default=-1,
                )
                if cutoff >= 0:
                    visible = decoded[:cutoff]
                    delta = visible[len(emitted_text):] if visible.startswith(emitted_text) else visible
                    emitted_text = visible
                    if delta:
                        yield {"type": "token", "text": delta, "token_id": next_id}
                    finish_reason = "stop"
                    break
                delta = decoded[len(emitted_text):] if decoded.startswith(emitted_text) else decoded
                emitted_text = decoded
                if delta:
                    yield {"type": "token", "text": delta, "token_id": next_id}
                if eos_token_id is not None and next_id == int(eos_token_id):
                    finish_reason = "stop"
                    break

            text = self._decode(tokenizer, generated)
            if request.stop:
                for stop in request.stop:
                    if stop in text:
                        text = text.split(stop, 1)[0]
            elapsed = time.perf_counter() - start
            yield {
                "type": "done",
                "result": GenerationResult(
                    text=text,
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(generated),
                    finish_reason=finish_reason,
                    backend_name=self.name,
                    metrics={
                        "ttft_ms": elapsed * 1000.0,
                        "throughput_tps": len(generated) / max(elapsed, 1e-9),
                        "device": "onnxruntime",
                        "providers": list(session.get_providers()),
                    },
                ),
            }
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"ONNX autoregressive adapter validation failed: {exc}",
                backend_name=self.name,
            ) from exc

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        """Load a local tokenizer only; remote model code is never executed."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as transformers_exc:
            try:
                from tokenizers import Tokenizer

                return Tokenizer.from_file(tokenizer_path)
            except Exception as tokenizers_exc:
                raise ValueError(
                    f"unable to load local tokenizer {tokenizer_path!r}: "
                    f"transformers={transformers_exc}; tokenizers={tokenizers_exc}"
                ) from tokenizers_exc

    @staticmethod
    def _build_prompt(request: GenerationRequest, tokenizer: Any) -> str:
        if request.messages:
            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                return str(apply_template(request.messages, tokenize=False, add_generation_prompt=True))
            return "\n".join(
                f"<{message.get('role', 'user')}>\n{message.get('content', '')}"
                for message in request.messages
            ) + "\n<assistant>\n"
        return request.prompt or ""

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        ids = getattr(encoded, "ids", encoded)
        return [int(value) for value in ids]

    @staticmethod
    def _decode(tokenizer: Any, token_ids: list[int]) -> str:
        try:
            return str(tokenizer.decode(token_ids, skip_special_tokens=True))
        except TypeError:
            return str(tokenizer.decode(token_ids))

    @staticmethod
    def _resolve_input_name(specs: list[Any], requested: Any, candidates: tuple[str, ...]) -> str:
        names = [str(spec.name) for spec in specs]
        if requested is not None:
            if str(requested) not in names:
                raise ValueError(f"configured ONNX input {requested!r} is not present; inputs={names}")
            return str(requested)
        for candidate in candidates:
            if candidate in names:
                return candidate
        raise ValueError(f"ONNX session has no input_ids input; inputs={names}")

    @staticmethod
    def _resolve_output_name(specs: list[Any], requested: Any) -> str:
        names = [str(spec.name) for spec in specs]
        if requested is not None:
            if str(requested) not in names:
                raise ValueError(f"configured ONNX output {requested!r} is not present; outputs={names}")
            return str(requested)
        for name in names:
            if "logit" in name.lower():
                return name
        if len(names) == 1:
            return names[0]
        raise ValueError(f"cannot identify logits output; outputs={names}")

    @staticmethod
    def _reject_unsupported_inputs(specs: list[Any], input_ids_name: str, options: dict[str, Any]) -> None:
        recognized = {input_ids_name}
        for key in ("attention_mask_name", "position_ids_name"):
            if options.get(key):
                recognized.add(str(options[key]))
        for spec in specs:
            name = str(spec.name)
            lowered = name.lower()
            if name in recognized or lowered in {"attention_mask", "attention_mask:0", "position_ids", "position_ids:0"}:
                continue
            raise ValueError(
                f"ONNX session input {name!r} requires a dedicated adapter; "
                "KV-cache/encoder-decoder contracts are not implicitly guessed"
            )

    @staticmethod
    def _integer_dtype(specs: list[Any], input_ids_name: str) -> Any:
        spec = next(item for item in specs if str(item.name) == input_ids_name)
        type_name = str(getattr(spec, "type", ""))
        if "int32" in type_name:
            return np.int32
        if "int64" in type_name:
            return np.int64
        raise ValueError(f"input_ids must use int32 or int64, got {type_name!r}")

    @staticmethod
    def _add_optional_inputs(
        feeds: dict[str, np.ndarray],
        specs: list[Any],
        sequence_length: int,
        options: dict[str, Any],
    ) -> None:
        names = {str(spec.name): spec for spec in specs}
        attention_name = options.get("attention_mask_name")
        if attention_name is None:
            attention_name = next(
                (name for name in names if name.lower() in {"attention_mask", "attention_mask:0"}),
                None,
            )
        if attention_name is not None:
            spec = names.get(str(attention_name))
            if spec is None:
                raise ValueError(f"configured attention mask input {attention_name!r} is not present")
            dtype = np.int32 if "int32" in str(getattr(spec, "type", "")) else np.int64
            feeds[str(attention_name)] = np.ones((1, sequence_length), dtype=dtype)

        position_name = options.get("position_ids_name")
        if position_name is None:
            position_name = next(
                (name for name in names if name.lower() in {"position_ids", "position_ids:0"}),
                None,
            )
        if position_name is not None:
            if str(position_name) not in names:
                raise ValueError(f"configured position ids input {position_name!r} is not present")
            spec = names[str(position_name)]
            dtype = np.int32 if "int32" in str(getattr(spec, "type", "")) else np.int64
            feeds[str(position_name)] = np.arange(sequence_length, dtype=dtype)[None, :]

    @staticmethod
    def _extract_logits(output: Any, output_name: str) -> np.ndarray:
        logits = np.asarray(output)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0, -1]
        elif logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits[0]
        else:
            raise ValueError(
                f"ONNX logits output {output_name!r} must be [1, sequence, vocab] "
                f"or [1, vocab], got {tuple(logits.shape)}"
            )
        if logits.ndim != 1 or logits.size == 0 or not np.isfinite(logits).all():
            raise ValueError(f"ONNX logits output {output_name!r} is empty or non-finite")
        return logits.astype(np.float32, copy=False)

    @staticmethod
    def _sample(logits: np.ndarray, request: GenerationRequest) -> int:
        if request.temperature <= 0.0:
            return int(np.argmax(logits))
        scaled = logits / max(float(request.temperature), 1e-6)
        if request.top_k > 0 and request.top_k < scaled.size:
            keep = np.argpartition(scaled, -request.top_k)[-request.top_k:]
            filtered = np.full_like(scaled, -np.inf)
            filtered[keep] = scaled[keep]
            scaled = filtered
        shifted = scaled - np.nanmax(scaled)
        probs = np.exp(shifted)
        probs[~np.isfinite(probs)] = 0.0
        if 0.0 < request.top_p < 1.0:
            order = np.argsort(-probs)
            cumulative = np.cumsum(probs[order]) / max(float(probs.sum()), 1e-12)
            remove = cumulative > request.top_p
            if remove.any():
                remove[0] = False
                probs[order[remove]] = 0.0
        total = float(probs.sum())
        if total <= 0.0:
            return int(np.argmax(logits))
        probs /= total
        seed = request.extra.get("seed")
        rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()
        return int(rng.choice(logits.size, p=probs))

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"ONNXBackend(models={len(self._models)})"
