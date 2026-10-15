"""Executing prefill/decode pools with bounded, typed in-process handoff.

Workers own distinct engine state while factories may share immutable weights.
Decode pulls an owned cache snapshot from a bounded queue. No networking, RDMA,
device isolation or speedup is claimed by this local transport implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from dataclasses import fields, is_dataclass
import queue
import threading
import time
from typing import Any, Protocol
import uuid
import copy

import numpy as np

from aether.runtime.cache_handoff import export_cpu_cache, restore_cpu_cache
from aether.runtime.kv_payload import CachePayload
from aether.runtime.sampling import SamplingConfig, sample_token
from aether.runtime.stopping import stop_token_set


def clone_cpu_engine_for_worker(engine: Any) -> Any:
    """Construct independent known CPU execution state sharing only weights/kernels.

    A constructor rebuilds rotary buffers, lazy dictionaries and counters. A
    shallow engine copy is deliberately unsupported: it would alias request-
    mutable caches and helper engines. Unknown engine families need an explicit
    adapter factory instead of an inferred clone.
    """
    from aether.runtime.cpu_engine import CPUExecutionEngine
    from aether.runtime.hybrid_engine import HybridExecutionEngine
    from aether.runtime.mamba_engine import MambaExecutionEngine
    from aether.runtime.mamba2_engine import Mamba2ExecutionEngine
    from aether.runtime.rwkv_engine import RWKVExecutionEngine
    from aether.runtime.mla_engine import MLAExecutionEngine
    if type(engine) is CPUExecutionEngine:
        # Adapter tensor arrays are read-only model parameters. Copying their
        # dictionaries isolates selection state without duplicating full tensors.
        adapters = {name: dict(values) for name, values in engine.lora_adapters.items()}
        return CPUExecutionEngine(engine.weights, engine.num_heads, engine.num_kv_heads, engine.kernels,
            lora_adapters=adapters, active_lora_adapter=engine.active_lora_adapter,
            sparse_attention_plan=copy.deepcopy(engine.sparse_attention_plan),
            semantic_kv_plan=copy.deepcopy(engine.semantic_kv_plan),
            cross_layer_kv_plan=copy.deepcopy(engine.cross_layer_kv_plan))
    if type(engine) is HybridExecutionEngine:
        mamba = engine._mamba
        return HybridExecutionEngine(engine.weights, layer_types=list(engine.layer_types),
            num_heads=engine.num_heads, num_kv_heads=engine.num_kv_heads,
            state_size=mamba.state_size, inner_size=mamba.inner_size,
            dt_rank=mamba.dt_rank, conv_kernel=mamba.conv_kernel,
            kernels=engine._transformer.kernels)
    if type(engine) is MambaExecutionEngine:
        return MambaExecutionEngine(engine.weights, engine.state_size, engine.inner_size, engine.dt_rank, engine.conv_kernel)
    if type(engine) is Mamba2ExecutionEngine:
        return Mamba2ExecutionEngine(engine.weights, state_size=engine.state_size, inner_size=engine.inner_size,
            num_heads=engine.num_heads, num_groups=engine.num_groups, conv_kernel=engine.conv_kernel, head_dim=engine.head_dim)
    if type(engine) is RWKVExecutionEngine:
        return RWKVExecutionEngine(engine.weights)
    if type(engine) is MLAExecutionEngine:
        return MLAExecutionEngine(engine.weights, copy.deepcopy(engine.config))
    raise ValueError(f"No safe CPU worker clone contract for {type(engine).__name__}")


def make_cpu_adapter_factory(engine: Any, *, model_fingerprint: str, cache_kind: str) -> Callable[[], "CPUDecoderAdapter"]:
    """Return a factory creating owned engine state without duplicating weights."""
    return lambda: CPUDecoderAdapter(clone_cpu_engine_for_worker(engine),
                                    model_fingerprint=model_fingerprint, cache_kind=cache_kind)


class EngineAdapter(Protocol):
    model_fingerprint: str
    cache_kind: str
    device_id: str

    def forward(self, token_ids: np.ndarray, cache: Any | None) -> tuple[Any, Any]: ...
    def export_cache(self, cache: Any) -> CachePayload: ...
    def import_cache(self, payload: CachePayload) -> Any: ...
    def synchronize(self) -> None: ...
    def cache_nbytes(self, cache: Any) -> int: ...
    def close(self) -> None: ...


class CPUDecoderAdapter:
    """Explicit adapter for the existing CPU attention/recurrent cache codecs."""
    device_id = "cpu"

    def __init__(self, engine: Any, *, model_fingerprint: str, cache_kind: str) -> None:
        if not callable(getattr(engine, "forward", None)) or not model_fingerprint or not cache_kind:
            raise ValueError("A decoder adapter requires a real forward engine and model/cache identity")
        self.engine = engine
        self.model_fingerprint = model_fingerprint
        self.cache_kind = cache_kind

    def forward(self, token_ids: np.ndarray, cache: Any | None) -> tuple[Any, Any]:
        return self.engine.forward(token_ids, cache)

    def export_cache(self, cache: Any) -> CachePayload:
        payload = export_cpu_cache(cache, model_fingerprint=self.model_fingerprint)
        if payload.cache_kind != self.cache_kind:
            raise ValueError("Actual engine cache differs from the configured cache contract")
        return payload

    def import_cache(self, payload: CachePayload) -> Any:
        if payload.cache_kind != self.cache_kind:
            raise ValueError("Handoff cache kind is incompatible with the decoder")
        if payload.metadata.get("schema") == "aether_torch_cache_v1" and self.cache_kind == "attention_kv_v1":
            from aether.runtime.torch_cache_handoff import torch_payload_to_cpu_attention
            cache = torch_payload_to_cpu_attention(payload, expected_model_fingerprint=self.model_fingerprint)
        else:
            cache = restore_cpu_cache(payload, expected_model_fingerprint=self.model_fingerprint)
        self._validate_geometry(cache)
        return cache

    def _validate_geometry(self, cache: Any) -> None:
        """A matching fingerprint cannot authorize incompatible tensor geometry."""
        from aether.runtime.cpu_engine import KVCache
        from aether.runtime.hybrid_engine import HybridCache
        from aether.runtime.mamba_engine import MambaCache
        from aether.runtime.mamba2_engine import Mamba2Cache
        from aether.runtime.rwkv_engine import RWKVCache
        from aether.runtime.mla_engine import MLACache
        weights = getattr(self.engine, "weights", None)
        if weights is None:
            # Other causal adapters provide their own model geometry validation.
            return
        layers, vocab = len(weights.layers), int(weights.lm_head.shape[0])
        last = getattr(cache, "last_logits", None)
        if cache.length and (last is None or last.shape != (vocab,)):
            raise ValueError("Cache logits do not match the model vocabulary")
        attention = cache.kv if isinstance(cache, HybridCache) else cache
        if isinstance(attention, KVCache):
            if attention.num_layers != layers:
                raise ValueError("Cache layer count does not match the model")
            engine = self.engine._transformer if isinstance(cache, HybridCache) else self.engine
            schedule = getattr(self.engine, "layer_types", ["attention"] * layers)
            for index, key in enumerate(attention.keys):
                if key is not None and key.shape[1:] != (engine.num_kv_heads, engine.head_dim):
                    raise ValueError("Cache attention head geometry does not match the model")
                if cache.length and schedule[index] == "attention" and (key is None or not attention.stored_lengths[index]):
                    raise ValueError("Populated attention cache is missing active layer state")
        recurrent = cache.ssm if isinstance(cache, HybridCache) else cache
        engine = self.engine._mamba if isinstance(cache, HybridCache) else self.engine
        if isinstance(recurrent, (MambaCache, Mamba2Cache)):
            if len(recurrent.states) != layers:
                raise ValueError("Cache recurrent layer count does not match the model")
            if isinstance(recurrent, MambaCache):
                state_shape = (1, engine.state_size, engine.inner_size)
                channels = engine.inner_size
                states = [state.h for state in recurrent.states]
            else:
                state_shape = (1, engine.num_heads, engine.head_dim, engine.state_size)
                channels = engine.inner_size + 2 * engine.num_groups * engine.state_size
                states = recurrent.states
            if any(state.shape != state_shape for state in states) or any(
                history.shape != (1, channels, max(engine.conv_kernel - 1, 0)) for history in recurrent.conv_history):
                raise ValueError("Cache recurrent geometry does not match the model")
        elif isinstance(recurrent, RWKVCache):
            if len(recurrent.aa) != layers or any(array.shape != (weights.embedding.shape[1],)
                for group in (recurrent.aa, recurrent.bb, recurrent.pp, recurrent.previous) for array in group):
                raise ValueError("Cache RWKV geometry does not match the model")
        elif isinstance(cache, MLACache) and any(token >= vocab for token in cache.token_ids):
            raise ValueError("Cache MLA replay token is outside the model vocabulary")

    def synchronize(self) -> None:
        # NumPy/native CPU calls are synchronous; this is an execution contract,
        # not an assertion that a device transfer or GPU fence occurred.
        return None

    def cache_nbytes(self, cache: Any) -> int:
        owners: dict[int, int] = {}
        def visit(value):
            if isinstance(value, np.ndarray):
                owner = value
                while isinstance(owner.base, np.ndarray):
                    owner = owner.base
                owners[id(owner)] = owner.nbytes
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif is_dataclass(value):
                for item in fields(value):
                    visit(getattr(value, item.name))
        visit(cache)
        return sum(owners.values())

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class GenerationResult:
    tokens: tuple[int, ...]
    cache: CachePayload
    metrics: dict[str, Any]


@dataclass
class _State:
    cache: Any = None
    payload: CachePayload | None = None
    cursor: int = 0
    logits: np.ndarray | None = None
    generated: list[int] = field(default_factory=list)
    phase: str = "queued_prefill"
    started: float = field(default_factory=time.perf_counter)
    queued_at: float = field(default_factory=time.perf_counter)
    metrics: dict[str, Any] = field(default_factory=lambda: {
        "transport": "in_process_typed_snapshot", "prefill_ms": 0.0, "decode_ms": 0.0,
        "handoff_ms": 0.0, "prefill_chunks": 0, "decode_forward_calls": 0,
        "state_bytes_handed_off": 0, "ttft_ms": None,
        "prefill_queue_ms": 0.0, "decode_queue_ms": 0.0, "backpressure_ms": 0.0,
    })


class GenerationHandle:
    """One cancellable request with a bounded incremental token channel."""
    def __init__(self, pool: "DisaggregatedExecutionPool", prompt: np.ndarray, max_tokens: int,
                 sampling: SamplingConfig, stops: frozenset[int], seed: int | None,
                 grammar_session: Any, payload: CachePayload | None) -> None:
        self.request_id = uuid.uuid4().hex
        self._pool = pool
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.sampling = sampling
        self.stops = stops
        self.grammar_session = grammar_session
        self.rng = np.random.default_rng(seed)
        self._state = _State(payload=payload)
        self._output: queue.Queue[int] = queue.Queue(maxsize=pool.output_buffer_tokens)
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._consumer = threading.Lock()
        self._error: BaseException | None = None
        self._result: GenerationResult | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> bool:
        if self.done:
            return False
        self._cancelled.set()
        return True

    def iter_tokens(self, timeout: float | None = None) -> Iterator[int]:
        if timeout is not None and (timeout < 0 or not np.isfinite(timeout)):
            raise ValueError("Generation timeout must be finite and nonnegative")
        if not self._consumer.acquire(blocking=False):
            raise RuntimeError("A generation handle permits one output consumer")
        deadline = None if timeout is None else time.perf_counter() + timeout
        try:
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("Disaggregated generation timed out")
                try:
                    token = self._output.get(timeout=.02)
                except queue.Empty:
                    if self.done:
                        if self._error is not None:
                            raise self._error
                        return
                else:
                    yield token
        finally:
            self._consumer.release()

    def result(self, timeout: float | None = None) -> GenerationResult:
        # Draining prevents a caller asking only for a final result from
        # deadlocking its own bounded output channel.
        for _ in self.iter_tokens(timeout=timeout):
            pass
        if self._result is None:
            raise RuntimeError("Generation finished without a real cache result")
        return self._result

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state.metrics, request_id=self.request_id, phase=self._state.phase)


class DisaggregatedExecutionPool:
    """Independent executing worker pools with bounded admission/backpressure.

    Cancellation is cooperative at engine-call boundaries. Long kernels cannot
    be interrupted by Python; chunk size bounds submitted prefill work. close()
    reports a timeout rather than claiming that a running kernel stopped.
    """
    def __init__(self, prefill_factories: Sequence[Callable[[], EngineAdapter]],
                 decode_factories: Sequence[Callable[[], EngineAdapter]], *,
                 max_pending: int = 8, prefill_chunk_size: int = 2048,
                 output_buffer_tokens: int = 16, max_prompt_tokens: int = 1_000_000,
                 max_new_tokens: int = 1_000_000, max_cache_bytes_per_request: int = 512 * 1024**2) -> None:
        for value in (max_pending, prefill_chunk_size, output_buffer_tokens, max_prompt_tokens,
                      max_new_tokens, max_cache_bytes_per_request):
            if type(value) is not int or value < 1:
                raise ValueError("Pool limits must be positive integers")
        if not prefill_factories or not decode_factories:
            raise ValueError("Separate prefill and decode worker factories are required")
        self.max_pending = max_pending
        self.prefill_chunk_size = prefill_chunk_size
        self.output_buffer_tokens = output_buffer_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.max_cache_bytes_per_request = max_cache_bytes_per_request
        self._slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.RLock()
        self._closed = False
        self._stop = threading.Event()
        self._requests: dict[str, GenerationHandle] = {}
        self._cleanup_errors: list[BaseException] = []
        self._prefill_index = self._decode_index = 0
        self._counts = {"submitted": 0, "completed": 0, "cancelled": 0, "failed": 0}
        adapters: list[EngineAdapter] = []
        try:
            for factory in (*prefill_factories, *decode_factories):
                adapters.append(factory())
            identities = {(adapter.model_fingerprint, adapter.cache_kind) for adapter in adapters}
            if len(identities) != 1:
                raise ValueError("Worker model fingerprints and cache contracts must match")
            if len({id(getattr(adapter, "engine", adapter)) for adapter in adapters}) != len(adapters):
                raise ValueError("Workers must own distinct engine objects; only immutable weights may be shared")
            for adapter in adapters:
                if any(not callable(getattr(adapter, name, None)) for name in
                       ("forward", "export_cache", "import_cache", "synchronize", "cache_nbytes", "close")):
                    raise ValueError("Engine adapter lacks an explicit execution/handoff hook")
            self.model_fingerprint, self.cache_kind = next(iter(identities))
            self._prefill = adapters[:len(prefill_factories)]
            self._decode = adapters[len(prefill_factories):]
            self._prefill_queues = [queue.Queue(max_pending) for _ in self._prefill]
            self._decode_queues = [queue.Queue(max_pending) for _ in self._decode]
            self._threads = []
            for phase, workers, queues in (("prefill", self._prefill, self._prefill_queues),
                                          ("decode", self._decode, self._decode_queues)):
                for index, (adapter, work_queue) in enumerate(zip(workers, queues)):
                    thread = threading.Thread(target=self._worker, args=(phase, adapter, work_queue),
                        name=f"aether-{phase}-{index}-{uuid.uuid4().hex[:6]}", daemon=True)
                    self._threads.append(thread)
                    thread.start()
        except BaseException as original:
            self._stop.set()
            # Started workers own their adapter lifecycle. Do not close a
            # device adapter concurrently with its still-running worker.
            threads = getattr(self, "_threads", [])
            for thread in threads:
                if thread.ident is not None:
                    thread.join()
            started = sum(thread.ident is not None for thread in threads)
            for adapter in adapters[started:]:
                closer = getattr(adapter, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except BaseException as exc:
                        self._cleanup_errors.append(exc)
            if self._cleanup_errors:
                raise RuntimeError(f"Worker initialization failed ({original}); resource cleanup also failed: "
                                   f"{self._cleanup_errors[0]}") from original
            raise

    def submit(self, prompt_ids: Any, *, max_tokens: int = 16,
               temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
               eos_token_id: Any = None, seed: int | None = None,
               grammar_session: Any = None, cache: CachePayload | None = None,
               admission_timeout: float = 0.0) -> GenerationHandle:
        sampling = SamplingConfig(temperature, top_k, top_p)
        ids = np.asarray(prompt_ids)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in "iu" or np.any(ids < 0)
                or np.any(ids > np.iinfo(np.int64).max))):
            raise ValueError("Prompt tokens must be a one-dimensional nonnegative integer array")
        ids = np.ascontiguousarray(ids, dtype=np.int64).copy()
        if ids.size > self.max_prompt_tokens or type(max_tokens) is not int or not 0 <= max_tokens <= self.max_new_tokens:
            raise ValueError("Request exceeds configured prompt/generation token bounds")
        if not ids.size and cache is None:
            raise ValueError("An empty prompt requires an owned populated cache payload")
        if cache is not None:
            self._check_payload(cache)
        if admission_timeout < 0 or not np.isfinite(admission_timeout):
            raise ValueError("Admission timeout must be finite and nonnegative")
        with self._lock:
            if self._closed:
                raise RuntimeError("Disaggregated pool is closed")
        if not self._slots.acquire(timeout=admission_timeout):
            raise queue.Full("Disaggregated pool admission capacity is exhausted")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Disaggregated pool is closed")
                handle = GenerationHandle(self, ids, max_tokens, sampling, stop_token_set(eos_token_id),
                                          seed, grammar_session, cache)
                self._requests[handle.request_id] = handle
                index = self._prefill_index % len(self._prefill)
                self._prefill_index += 1
                self._counts["submitted"] += 1
                self._prefill_queues[index].put_nowait(handle)
                return handle
        except BaseException:
            self._slots.release()
            raise

    def _check_payload(self, payload: CachePayload) -> None:
        if not isinstance(payload, CachePayload) or (payload.model_fingerprint, payload.cache_kind) != (self.model_fingerprint, self.cache_kind):
            raise ValueError("Handoff payload is incompatible with the execution model/cache contract")
        if payload.byte_size > self.max_cache_bytes_per_request:
            raise ValueError("Cache state exceeds the configured per-request byte bound")

    def _finish(self, handle: GenerationHandle, adapter: EngineAdapter,
                error: BaseException | None = None) -> None:
        with handle._lock:
            if handle.done:
                return
            state = handle._state
            if error is None:
                try:
                    payload = adapter.export_cache(state.cache)
                    self._check_payload(payload)
                    state.metrics["total_ms"] = (time.perf_counter() - state.started) * 1000
                    state.metrics["generated_tokens"] = len(state.generated)
                    handle._result = GenerationResult(tuple(state.generated), payload, dict(state.metrics))
                except BaseException as exc:
                    error = exc
            handle._error = error
            state.cache = state.payload = None
            state.phase = "cancelled" if isinstance(error, CancelledError) else "failed" if error else "complete"
            handle._done.set()
        with self._lock:
            self._requests.pop(handle.request_id, None)
            self._counts["cancelled" if isinstance(error, CancelledError) else "failed" if error else "completed"] += 1
        self._slots.release()

    def _worker(self, phase: str, adapter: EngineAdapter, work_queue: queue.Queue) -> None:
        try:
            while not self._stop.is_set() or not work_queue.empty():
                try:
                    handle = work_queue.get(timeout=.02)
                except queue.Empty:
                    continue
                try:
                    handle._state.metrics[f"{phase}_queue_ms"] += (time.perf_counter() - handle._state.queued_at) * 1000
                    if handle._cancelled.is_set() or self._stop.is_set():
                        self._finish(handle, adapter, CancelledError("Generation was cancelled"))
                    elif phase == "prefill":
                        self._prefill_step(handle, adapter, work_queue)
                    else:
                        self._decode_step(handle, adapter, work_queue)
                except BaseException as exc:
                    self._finish(handle, adapter, exc)
                finally:
                    work_queue.task_done()
        finally:
            try:
                adapter.close()
            except BaseException as exc:
                with self._lock:
                    self._cleanup_errors.append(exc)

    def _prefill_step(self, handle: GenerationHandle, adapter: EngineAdapter, work_queue: queue.Queue) -> None:
        state = handle._state
        state.phase = "prefill"
        if state.cache is None and state.payload is not None:
            state.cache = adapter.import_cache(state.payload)
            state.payload = None
        end = min(handle.prompt.size, state.cursor + self.prefill_chunk_size)
        if end > state.cursor:
            began = time.perf_counter()
            logits, state.cache = adapter.forward(handle.prompt[state.cursor:end], state.cache)
            adapter.synchronize()
            self._check_cache(adapter, state.cache)
            state.metrics["prefill_ms"] += (time.perf_counter() - began) * 1000
            state.metrics["prefill_chunks"] += 1
            state.logits = np.asarray(logits)[-1].reshape(-1).copy()
            state.cursor = end
        else:
            state.logits = np.asarray(state.cache.last_logits).reshape(-1).copy()
        if state.cursor < handle.prompt.size:
            state.queued_at = time.perf_counter()
            work_queue.put_nowait(handle)
            return
        if handle._cancelled.is_set():
            self._finish(handle, adapter, CancelledError("Generation was cancelled after prefill"))
            return
        began = time.perf_counter()
        state.payload = adapter.export_cache(state.cache)
        self._check_payload(state.payload)
        state.cache = None
        state.metrics["handoff_ms"] += (time.perf_counter() - began) * 1000
        state.metrics["state_bytes_handed_off"] = state.payload.byte_size
        state.metrics["prefill_device"] = adapter.device_id
        state.phase = "queued_decode"
        with self._lock:
            if self._closed:
                self._finish(handle, adapter, CancelledError("Pool closed before cache handoff"))
                return
            index = self._decode_index % len(self._decode)
            self._decode_index += 1
            state.queued_at = time.perf_counter()
            self._decode_queues[index].put_nowait(handle)

    def _check_cache(self, adapter: EngineAdapter, cache: Any) -> None:
        size = adapter.cache_nbytes(cache)
        if type(size) is not int or size < 0 or size > self.max_cache_bytes_per_request:
            raise ValueError("Allocated inference state exceeds the per-request cache byte bound")

    def _decode_step(self, handle: GenerationHandle, adapter: EngineAdapter, work_queue: queue.Queue) -> None:
        state = handle._state
        state.phase = "decode"
        if state.cache is None:
            began = time.perf_counter()
            state.cache = adapter.import_cache(state.payload)
            state.payload = None
            adapter.synchronize()
            state.metrics["handoff_ms"] += (time.perf_counter() - began) * 1000
            state.metrics["decode_device"] = adapter.device_id
        if len(state.generated) >= handle.max_tokens:
            self._finish(handle, adapter)
            return
        if handle._output.full():
            state.queued_at = time.perf_counter()
            work_queue.put_nowait(handle)
            # The worker remains cancellable and other requests retain access.
            began = time.perf_counter()
            self._stop.wait(.001)
            state.metrics["backpressure_ms"] += (time.perf_counter() - began) * 1000
            return
        token = sample_token(state.logits, handle.sampling, rng=handle.rng, grammar_session=handle.grammar_session)
        stop = token in handle.stops or (handle.grammar_session is not None and
            getattr(handle.grammar_session, "is_accepting", lambda: False)())
        state.generated.append(token)
        if state.metrics["ttft_ms"] is None:
            state.metrics["ttft_ms"] = (time.perf_counter() - state.started) * 1000
        # Publish the token already determined by prefill before computing the
        # following position. A slow next-step kernel must not delay TTFT.
        handle._output.put_nowait(token)
        if not stop:
            began = time.perf_counter()
            logits, state.cache = adapter.forward(np.asarray([token], dtype=np.int64), state.cache)
            adapter.synchronize()
            self._check_cache(adapter, state.cache)
            state.metrics["decode_ms"] += (time.perf_counter() - began) * 1000
            state.metrics["decode_forward_calls"] += 1
            state.logits = np.asarray(logits)[-1].reshape(-1).copy()
        if handle._cancelled.is_set():
            self._finish(handle, adapter, CancelledError("Generation was cancelled during decode"))
            return
        if stop or len(state.generated) >= handle.max_tokens:
            self._finish(handle, adapter)
        else:
            state.queued_at = time.perf_counter()
            work_queue.put_nowait(handle)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self._counts, "active": len(self._requests), "closed": self._closed,
                    "prefill_workers": len(self._prefill), "decode_workers": len(self._decode),
                    "live_workers": sum(thread.is_alive() for thread in self._threads),
                    "cleanup_failures": len(self._cleanup_errors),
                    "transport": "in_process_typed_snapshot"}

    def close(self, *, timeout: float = 10.0) -> None:
        if timeout < 0 or not np.isfinite(timeout):
            raise ValueError("Close timeout must be finite and nonnegative")
        with self._lock:
            self._closed = True
            for handle in self._requests.values():
                handle.cancel()
            self._stop.set()
        deadline = time.perf_counter() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.perf_counter()))
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError("Disaggregated worker is still executing; pool shutdown has not completed")
        with self._lock:
            errors = list(self._cleanup_errors)
        if errors:
            raise RuntimeError(f"Disaggregated worker resource cleanup failed ({len(errors)} error(s)): "
                               f"{errors[0]}") from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PooledGenerationAdapter:
    """Engine-shaped request bridge; completed caches are owned payloads."""
    def __init__(self, pool: DisaggregatedExecutionPool) -> None:
        self.pool = pool
        self._local = threading.local()

    def generate_iter(self, prompt_ids: Any, max_tokens: int = 16, *, cache: CachePayload | None = None,
                      cache_callback: Callable[[CachePayload], Any] | None = None, **kwargs: Any) -> Iterator[int]:
        allowed = {"temperature", "top_k", "top_p", "eos_token_id", "seed", "grammar_session", "admission_timeout"}
        unsupported = {key for key, value in kwargs.items() if key not in allowed and value is not None}
        if unsupported:
            raise ValueError(f"Pooled request has unsupported active execution hooks: {sorted(unsupported)}")
        self._local.metrics = {}
        self._local.cache = None
        handle = self.pool.submit(prompt_ids, max_tokens=max_tokens, cache=cache,
                                  **{key: value for key, value in kwargs.items() if key in allowed})
        try:
            yield from handle.iter_tokens()
            result = handle.result()
            self._local.metrics = result.metrics
            self._local.cache = result.cache
            if cache_callback is not None:
                cache_callback(result.cache)
        finally:
            if not handle.done:
                handle.cancel()

    def generate_with_cache(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> tuple[list[int], CachePayload]:
        generated = list(self.generate_iter(prompt_ids, max_tokens, **kwargs))
        return generated, self._local.cache

    def generate(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens, **kwargs))

    def generation_stats(self) -> dict[str, Any]:
        return dict(getattr(self._local, "metrics", {}))
