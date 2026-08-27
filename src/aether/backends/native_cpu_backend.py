"""Framework-free execution backend for packaged AEG artifacts.

The native CPU engine is part of Aether's core runtime contract. It must not
be hidden behind the optional PyTorch backend: a user who installs the base
package and receives a compiled AEG must be able to load that artifact without
importing PyTorch or Transformers.

This module owns its compiled-AEG request handling and never imports the
optional PyTorch backend. The forward pass is always performed by
``CPUExecutionEngine``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.backends import batched_generation
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import BackendError
from aether.core.hash_utils import compute_file_hash
from aether.runtime.aeg_loader import load_engine_from_path, package_is_runnable


class PackagedTokenizer:
    """Adapter around a serialized Hugging Face ``tokenizer.json``."""

    def __init__(self, tokenizer_path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - packaging error path
            raise BackendError(
                "The AEG tokenizer requires the 'tokenizers' package; install "
                "aether-runtime with its base dependencies.",
                backend_name="aether_cpu",
            ) from exc
        if not tokenizer_path.is_file():
            raise BackendError(
                f"AEG tokenizer file is missing: {tokenizer_path}",
                backend_name="aether_cpu",
            )
        try:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:  # noqa: BLE001 - normalize untrusted artifact errors
            raise BackendError(
                f"Unable to load packaged tokenizer {tokenizer_path}: {exc}",
                backend_name="aether_cpu",
            ) from exc
        config = self._read_json(tokenizer_path.with_name("tokenizer_config.json"))
        special = self._read_json(tokenizer_path.with_name("special_tokens_map.json"))
        self.eos_token_id = self._special_id(config.get("eos_token"), special.get("eos_token"))
        self.pad_token_id = self._special_id(config.get("pad_token"), special.get("pad_token"))
        self.bos_token_id = self._special_id(config.get("bos_token"), special.get("bos_token"))
        self.chat_template = config.get("chat_template")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _special_id(self, *values: Any) -> int | None:
        for value in values:
            token = value
            if isinstance(value, dict):
                token = value.get("content") or value.get("token")
            if not isinstance(token, str):
                continue
            token_id = self._tokenizer.token_to_id(token)
            if token_id is not None:
                return int(token_id)
        return None

    def __len__(self) -> int:
        """Return the serialized tokenizer vocabulary size.

        ``len(tokenizer)`` is the common Hugging Face/tokenizers contract and
        is also the identity input used by the grammar compiler's tokenizer
        fingerprint.  Keeping this on the adapter ensures a grammar compiled
        before process restart is verified against the exact packaged
        vocabulary rather than being accepted on vocabulary size alone.
        """
        return int(self._tokenizer.get_vocab_size())

    def get_vocab_size(self, *, with_added_tokens: bool = True) -> int:
        """Expose the tokenizers vocabulary-size API used by runtime passes."""
        return int(self._tokenizer.get_vocab_size(with_added_tokens=with_added_tokens))

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "np",
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        if return_tensors not in {"np", "numpy"}:
            raise ValueError("PackagedTokenizer supports return_tensors='np' only")
        texts = text if isinstance(text, list) else [text]
        encodings = [self._tokenizer.encode(value, add_special_tokens=True) for value in texts]
        ids = [encoding.ids for encoding in encodings]
        if truncation and max_length is not None:
            ids = [row[:max_length] for row in ids]
        if padding or len(ids) > 1:
            width = max((len(row) for row in ids), default=0)
            if self.pad_token_id is None:
                raise ValueError("batched packaged-tokenizer input requires a pad token")
            ids = [row + [self.pad_token_id] * (width - len(row)) for row in ids]
        if not ids:
            ids = [[]]
        return {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(
                [[1 if token != self.pad_token_id else 0 for token in row] for row in ids],
                dtype=np.int64,
            ),
        }

    def decode(
        self,
        ids: Any,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        values = np.asarray(ids, dtype=np.int64).reshape(-1).tolist()
        try:
            return self._tokenizer.decode(
                [int(value) for value in values],
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
        except TypeError:
            # ``tokenizers.Tokenizer.decode`` versions before 0.20 do not
            # expose the Transformers-only cleanup keyword.
            return self._tokenizer.decode(
                [int(value) for value in values],
                skip_special_tokens=skip_special_tokens,
            )


class NativeCPUBackend(Backend):
    """Aether-native CPU backend for self-contained AEG execution."""

    def __init__(self) -> None:
        Backend.__init__(
            self,
            BackendInfo(
                name="aether_cpu",
                version=AETHER_VERSION,
                supported_targets=[
                    "cpu_avx2", "cpu_avx512", "cpu_neon",
                    "cpu_avx512_ternary", "cpu_neon_ternary",
                ],
                capabilities=[
                    "generate", "chat", "stream", "kv_cache",
                    "grammar_constraints", "structured_output",
                    "packaged_aeg", "framework_free",
                ],
            ),
        )
        self._models: dict[str, CompiledAEGHandle] = {}
        self._tokenizers: dict[str, PackagedTokenizer] = {}
        self._device = "cpu"
        self._allow_remote_code = False

    def is_available(self) -> bool:
        """The backend is available whenever the base NumPy runtime is usable."""
        return True

    def load_model(self, model_id: str, aeg_path: str | None = None, **_: Any) -> Any:
        """Load and authenticate a self-contained AEG package."""
        if model_id in self._models:
            return self._models[model_id]
        if aeg_path is None:
            raise BackendError(
                "aether_cpu executes compiled .aeg artifacts only; compile the "
                "source model first or select an installed frontend backend",
                backend_name=self.name,
            )
        root = Path(aeg_path).resolve()
        try:
            from aether.core.aeg_format import AEGPackage
            from aether.adapters.lora import load_compiled_lora_adapters

            package = AEGPackage(root).load()
            package.verify_integrity()
            eval_report_path = root / "observability" / "eval_report.json"
            if eval_report_path.is_file():
                eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
                gate = eval_report.get("gate", {})
                if gate.get("passed") is False:
                    failing = gate.get("failing_benchmarks", [])
                    raise BackendError(
                        "AEG artifact is rejected by its persisted evaluation gate: "
                        + ", ".join(str(item) for item in failing),
                        backend_name=self.name,
                    )
            if not package_is_runnable(package):
                raise BackendError(f"AEG artifact {root} is not executable", backend_name=self.name)
            engine = load_engine_from_path(root)
            tokenizer = PackagedTokenizer(root / "tokenizer" / "tokenizer.json")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            metadata_path = root / "graph" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            precision_path = root / "weights" / "quantized" / "precision_map.json"
            precision_map = json.loads(precision_path.read_text(encoding="utf-8")) if precision_path.is_file() else {}
            handle = CompiledAEGHandle(
                model_id=model_id,
                aeg_path=root,
                manifest=manifest,
                metadata=metadata,
                precision_map=precision_map,
                engine=engine,
                tokenizer=tokenizer,
                lora_adapters=load_compiled_lora_adapters(root),
            )
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed at artifact boundary
            raise BackendError(
                f"AEG artifact {root} failed native CPU integrity/load validation: {exc}",
                backend_name=self.name,
            ) from exc
        self._models[model_id] = handle
        self._tokenizers[model_id] = tokenizer
        return handle

    def get_capabilities(self) -> list[str]:
        """Return capabilities of the framework-free packaged-AEG path."""
        return self.info.capabilities

    def _request_text(self, request: GenerationRequest, tokenizer: Any | None = None) -> str:
        """Render a request using only the packaged tokenizer contract."""
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
            )
        if request.prompt is None:
            raise ValueError("Either prompt or messages must be provided")
        return request.prompt

    @staticmethod
    def _stop_text(tokenizer: Any, token_ids: list[int], stops: list[str] | None) -> tuple[str, int, bool]:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        if not stops:
            return text, len(token_ids), False
        cutoff = min((text.find(stop) for stop in stops if stop and stop in text), default=-1)
        if cutoff < 0:
            return text, len(token_ids), False
        count = len(token_ids)
        for index in range(1, len(token_ids) + 1):
            prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=True)
            if any(stop and stop in prefix for stop in stops):
                count = index
                break
        return text[:cutoff], count, True

    def supports_batched_generation(self, model_id: str, batch_size: int = 2) -> bool:
        """Whether ``model_id`` can be served as a real batch of this width.

        A probe: it loads nothing and promotes nothing, so it is safe to call before
        deciding whether to assemble a batch.
        """
        return batched_generation.can_batch(self._models.get(model_id), batch_size)

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Serve several requests in one batched forward pass.

        This backend's own kernels are sequence-major, which is the faster shape for
        a single sequence and the reason they stay that way. A batch is served by
        promoting the same authenticated weights onto the portable tensor executor,
        which carries a batch axis; the shared helper does that once per loaded
        model and refuses outright if it cannot, rather than looping over the
        requests and calling the result a batch.
        """
        if not requests:
            return []
        if len(requests) == 1:
            return [self.generate(requests[0])]

        model_ids = {request.model_id for request in requests}
        if len(model_ids) != 1:
            raise BackendError(
                "every request in a batch must name the same model; got "
                f"{sorted(model_ids)}",
                backend_name=self.name,
            )
        model_id = next(iter(model_ids))
        if model_id not in self._models:
            self.load_model(model_id, aeg_path=requests[0].extra.get("aeg_path"))
        return batched_generation.generate_batch(
            self._models[model_id],
            requests,
            backend_name=self.name,
            request_text=self._request_text,
            truncate_stop_text=lambda tokenizer, ids, stops: self._stop_text(
                tokenizer, list(ids), stops
            ),
            default_device="cpu",
        )

    def release_session_cache(self, model_id: str, session_id: str) -> None:
        """Release request-local KV state without importing PyTorch."""
        model = self._models.get(model_id)
        if isinstance(model, CompiledAEGHandle):
            model.clear_session_cache(session_id)

    def _engine_for_lora(
        self, handle: CompiledAEGHandle, engine: Any, adapter_id: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Select a verified compiled LoRA adapter without a model framework."""
        if adapter_id is None:
            return engine, None
        if not isinstance(adapter_id, str) or not adapter_id:
            raise BackendError("adapter_id must be a non-empty string", backend_name=self.name)
        if adapter_id not in handle.lora_adapters:
            raise BackendError(
                f"compiled LoRA adapter {adapter_id!r} cannot be applied: adapter is absent",
                backend_name=self.name,
            )
        try:
            selected = engine.with_lora_adapter(handle.lora_adapters, adapter_id)
        except Exception as exc:  # noqa: BLE001 - normalize artifact mismatch
            raise BackendError(
                f"compiled LoRA adapter {adapter_id!r} cannot be applied: {exc}",
                backend_name=self.name,
            ) from exc
        return selected, {"adapter_id": adapter_id, "targets": len(handle.lora_adapters[adapter_id])}

    def _engine_for_task_weights(
        self, handle: CompiledAEGHandle, task_weights: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Load authenticated task-vector payloads into a request-local engine."""
        if not task_weights:
            return handle.engine, None
        if not isinstance(task_weights, dict):
            raise BackendError("task_weights must be a mapping", backend_name=self.name)
        metadata = handle.metadata.get("task_vectors", {})
        vectors = metadata.get("vectors", []) if isinstance(metadata, dict) else []
        if not isinstance(vectors, list) or not vectors:
            raise BackendError("the AEG does not contain executable task-vector payloads", backend_name=self.name)
        artifacts = handle.manifest.get("artifacts", {})
        available = {str(item.get("name")) for item in vectors if isinstance(item, dict)}
        unknown = sorted(set(task_weights) - available)
        if unknown:
            raise BackendError(f"task_weights reference unknown vectors {unknown}", backend_name=self.name)
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
            relative = vector.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise BackendError(f"unsafe task-vector path for {name!r}", backend_name=self.name)
            path = (handle.aeg_path / relative).resolve()
            if artifacts.get(relative) is None or not path.is_file() or compute_file_hash(path) != artifacts[relative]:
                raise BackendError(f"task-vector payload failed AEG integrity validation: {relative}", backend_name=self.name)
            try:
                archive = np.load(path, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(f"unable to read task-vector payload {relative}", backend_name=self.name) from exc
            with archive:
                for descriptor in vector.get("tensors", []):
                    tensor_name = descriptor.get("name")
                    key = descriptor.get("key")
                    shape = tuple(int(value) for value in descriptor.get("shape", []))
                    if not isinstance(tensor_name, str) or not isinstance(key, str) or not shape or key not in archive:
                        raise BackendError(f"malformed task-vector tensor descriptor in {name!r}", backend_name=self.name)
                    array = np.asarray(archive[key], dtype=np.float32)
                    if int(np.prod(shape)) != array.size:
                        raise BackendError(f"task-vector tensor shape mismatch for {tensor_name!r}", backend_name=self.name)
                    contribution = np.ascontiguousarray(array.reshape(shape), dtype=np.float32) * np.float32(coefficient)
                    deltas[tensor_name] = deltas.get(tensor_name, 0.0) + contribution
                    tensor_count += 1
            applied.append(name)
        if not applied:
            raise BackendError("task_weights selected no non-zero task-vector payload", backend_name=self.name)
        try:
            return handle.engine.with_task_deltas(deltas), {"vectors": applied, "tensor_count": tensor_count}
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"task-vector deltas do not match the compiled model: {exc}", backend_name=self.name) from exc

    def _begin_ttt(
        self, handle: CompiledAEGHandle, prompt_ids: np.ndarray, request: GenerationRequest, engine: Any,
    ) -> tuple[Any, str, list[dict[str, Any]], float] | None:
        """Adapt persisted R5 slots using the portable engine's embeddings."""
        ttt_engine = request.extra.get("ttt_engine")
        if ttt_engine is None or not hasattr(engine.weights, "embedding"):
            return None
        request_id = str(request.extra.get("ttt_request_id") or uuid.uuid4().hex)
        ttt_engine.begin_request(request_id)
        try:
            hidden = engine.weights.embedding[np.asarray(prompt_ids, dtype=np.int64)].astype(np.float32).tolist()
            loss = float(ttt_engine.adapt(request_id, hidden))
            slots = []
            for layer_index in range(engine.weights.num_layers):
                slot = ttt_engine.get_fast_weights(request_id, layer_index)
                if slot is None:
                    raise BackendError(f"R5 TTT slot {layer_index} was not available after adaptation", backend_name=self.name)
                slots.append(slot)
            return ttt_engine, request_id, slots, loss
        except Exception:
            ttt_engine.end_request(request_id)
            raise

    @staticmethod
    def _end_ttt(state: tuple[Any, str, list[dict[str, Any]], float] | None) -> None:
        if state is not None:
            state[0].end_request(state[1])

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate from an executable AEG without importing a model framework."""
        if request.model_id not in self._models:
            self.load_model(request.model_id, aeg_path=request.extra.get("aeg_path"))
        handle = self._models[request.model_id]
        if handle.engine is None or handle.tokenizer is None:
            raise BackendError("loaded AEG has no native tokenizer-backed engine", backend_name=self.name)
        import time

        text = self._request_text(request, handle.tokenizer)
        encoded = handle.tokenizer(text, return_tensors="np")
        prompt_ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
        engine = handle.engine
        engine, task_metrics = self._engine_for_task_weights(handle, request.extra.get("task_weights"))
        adapter_id = request.extra.get("adapter_id", request.extra.get("adapter_name"))
        engine, adapter_metrics = self._engine_for_lora(handle, engine, adapter_id)
        ttt_state = self._begin_ttt(handle, prompt_ids, request, engine)
        ttt_slots = ttt_state[2] if ttt_state is not None else None

        session_id = request.extra.get("aether_kv_session_id")
        cached = handle.session_caches.get(session_id) if isinstance(session_id, str) else None
        cache = None
        suffix = prompt_ids
        reused_tokens = 0
        multi_agent_reused_tokens = 0
        coordinator = request.extra.get("multi_agent_kv_coordinator")
        prefix_text = request.extra.get("multi_agent_prefix")
        prefix_hash = request.extra.get("multi_agent_prefix_hash")
        if (
            ttt_state is None and coordinator is not None and isinstance(prefix_text, str)
            and prefix_text and isinstance(prefix_hash, str) and prefix_hash
        ):
            if coordinator.hash_prefix(prefix_text) != prefix_hash:
                raise BackendError("R2 multi-agent prefix hash does not match its text", backend_name=self.name)
            prefix_ids = np.asarray(handle.tokenizer(prefix_text, return_tensors="np")["input_ids"][0], dtype=np.int64)
            if prompt_ids.size < prefix_ids.size or not np.array_equal(prompt_ids[:prefix_ids.size], prefix_ids):
                raise BackendError("R2 multi-agent prefix is not an exact token prefix of the request", backend_name=self.name)
            shared_cache, shared_length = coordinator.get_shared_kv(prefix_hash)
            if shared_cache is None:
                _, shared_cache = engine.generate_with_cache(prefix_ids, max_tokens=0, cache=None)
                coordinator.update_shared_kv(prefix_hash, shared_cache, seq_len=int(prefix_ids.size))
                shared_length = int(prefix_ids.size)
            else:
                multi_agent_reused_tokens = int(shared_length)
            if int(shared_length) != int(prefix_ids.size):
                raise BackendError("R2 shared KV length does not match the tokenized prefix", backend_name=self.name)
            cache = shared_cache.clone()
            suffix = prompt_ids[prefix_ids.size:]
        if cached is not None and cache is None:
            cached_ids, cached_cache = cached
            cached_ids = np.asarray(cached_ids, dtype=np.int64)
            if prompt_ids.size >= cached_ids.size and np.array_equal(
                prompt_ids[: cached_ids.size], cached_ids
            ):
                cache = cached_cache
                suffix = prompt_ids[cached_ids.size :]
                reused_tokens = int(cached_ids.size)
            else:
                handle.clear_session_cache(session_id)

        start = time.perf_counter()
        try:
            if hasattr(engine, "generate_with_cache"):
                generated, updated_cache = engine.generate_with_cache(
                    suffix,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                    grammar_session=request.extra.get("grammar_session"),
                    cache=cache,
                    ttt_slots=ttt_slots,
                    adapter_id=str(adapter_id) if adapter_id is not None else None,
                    peagle_engine=request.extra.get("peagle_engine"),
                )
            else:
                generated = engine.generate(
                    prompt_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                    ttt_slots=ttt_slots,
                    peagle_engine=request.extra.get("peagle_engine"),
                )
                updated_cache = None
        except TypeError:
            # Specialised engines can expose a narrower generation signature.
            # Retry only after removing optional orchestration hooks; never
            # fabricate output or silently switch to another backend.
            if not any(value is not None for value in (
                request.extra.get("ttt_slots"), request.extra.get("peagle_engine"),
            )):
                raise
            generated, updated_cache = engine.generate_with_cache(
                suffix,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
                grammar_session=request.extra.get("grammar_session"),
                cache=cache,
            )
        finally:
            self._end_ttt(ttt_state)
        if isinstance(session_id, str):
            full_ids = np.concatenate([prompt_ids, np.asarray(generated, dtype=np.int64)])
            handle.session_caches[session_id] = (full_ids, updated_cache)
        elapsed = time.perf_counter() - start
        generated_text = handle.tokenizer.decode(generated, skip_special_tokens=True)
        completion_tokens = len(generated)
        if request.stop:
            generated_text, completion_tokens, stopped = self._stop_text(
                handle.tokenizer, generated, request.stop
            )
        else:
            stopped = False
        return GenerationResult(
            text=generated_text,
            prompt_tokens=int(prompt_ids.size),
            completion_tokens=completion_tokens,
            finish_reason="stop" if stopped or completion_tokens < request.max_tokens else "length",
            backend_name=self.name,
            metrics={
                "ttft_ms": elapsed * 1000.0,
                "throughput_tps": len(generated) / max(elapsed, 1e-9),
                "device": self._device,
                "framework_free": True,
                "kv_reuse": reused_tokens > 0,
                "kv_reused_tokens": reused_tokens,
                "multi_agent_kv_reuse": multi_agent_reused_tokens > 0,
                "multi_agent_kv_reused_tokens": multi_agent_reused_tokens,
                **({"ttt_adaptation_loss": ttt_state[3]} if ttt_state is not None else {}),
                **({"task_reweighting": task_metrics} if task_metrics is not None else {}),
                **({"lora_adapter": adapter_metrics} if adapter_metrics is not None else {}),
                **(
                    {"speculative": stats}
                    if callable(getattr(engine, "speculative_stats", None))
                    and (stats := engine.speculative_stats()).get("draft_tokens", 0) > 0
                    else {}
                ),
            },
        )

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Stream generated chunks from the compiled AEG path."""
        if request.model_id not in self._models:
            self.load_model(request.model_id, aeg_path=request.extra.get("aeg_path"))
        handle = self._models[request.model_id]
        if handle.engine is None or handle.tokenizer is None:
            raise BackendError("loaded AEG has no native tokenizer-backed engine", backend_name=self.name)
        text = self._request_text(request, handle.tokenizer)
        encoded = handle.tokenizer(text, return_tensors="np")
        prompt_ids = np.asarray(encoded["input_ids"][0], dtype=np.int64)
        engine = handle.engine
        adapter_id = request.extra.get("adapter_id", request.extra.get("adapter_name"))
        if adapter_id is not None and hasattr(engine, "with_lora_adapter"):
            engine = engine.with_lora_adapter(handle.lora_adapters, str(adapter_id))
        session_id = request.extra.get("aether_kv_session_id")
        cached = handle.session_caches.get(session_id) if isinstance(session_id, str) else None
        cache = None
        suffix = prompt_ids
        if cached is not None:
            cached_ids, cached_cache = cached
            cached_ids = np.asarray(cached_ids, dtype=np.int64)
            if prompt_ids.size >= cached_ids.size and np.array_equal(
                prompt_ids[: cached_ids.size], cached_ids
            ):
                cache = cached_cache
                suffix = prompt_ids[cached_ids.size :]
            else:
                handle.clear_session_cache(session_id)

        token_ids: list[int] = []
        previous = ""
        updated_cache: list[Any] = [None]

        def remember(value: Any) -> None:
            updated_cache[0] = value

        iterator = engine.generate_iter(
            suffix,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
            grammar_session=request.extra.get("grammar_session"),
            cache=cache,
            cache_callback=remember,
            adapter_id=str(adapter_id) if adapter_id is not None else None,
        )
        for token_id in iterator:
            token_ids.append(int(token_id))
            decoded = handle.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            delta = decoded[len(previous):] if decoded.startswith(previous) else decoded
            previous = decoded
            if delta:
                yield delta
        if isinstance(session_id, str) and updated_cache[0] is not None:
            full_ids = np.concatenate([prompt_ids, np.asarray(token_ids, dtype=np.int64)])
            handle.session_caches[session_id] = (full_ids, updated_cache[0])
