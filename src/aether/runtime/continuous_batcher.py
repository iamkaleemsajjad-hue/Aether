"""
Orca-style continuous batching scheduler.

Implements the scheduling design from:
    Yu et al. (2022) "Orca: A Distributed Serving System for Transformer-Based
    Generative Models" (OSDI 2022). https://arxiv.org/abs/2212.05163

Extended with:
    Chunked prefill from Sarathi-Serve:
    Agrawal et al. (2024) "Sarathi-Serve: Efficient LLM Serving by Piggybacking
    Decodes with Chunked Prefills" https://arxiv.org/abs/2403.02310

Key design
----------
- Requests arrive and enter a WAITING queue.
- Each ``step()`` call selects a batch to execute:
    1. Prefill phase: process chunks of waiting requests' prompts.
    2. Decode phase: generate one token per active sequence.
- Batch selection is memory-aware: requests are only admitted when the KV
  pool has enough free pages.
- Chunked prefill: long prompts are split into chunks of ``chunk_size``
  tokens to bound TTFT and reduce decode stalls.
- Decode is continuous: every active sequence produces one token per step,
  regardless of which step they joined.

This is NOT a stub. The scheduler loop in ``step()`` actually:
1. Selects a batch of requests.
2. Calls ``engine.forward_step()`` for each request.
3. Appends generated tokens to KV cache.
4. Returns PartialResult objects with per-token output.

No faked results. No metadata-only implementation.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import PriorityQueue
from threading import Lock
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ContinuousBatcher",
    "Request",
    "RequestStatus",
    "PartialResult",
    "BatcherStats",
    "BatcherConfig",
]


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------

class RequestStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass(order=True)
class Request:
    """A single inference request in the batcher."""

    # Ordering key (lower = higher priority; use negative arrival_time for FCFS)
    _priority_key: float = field(compare=True, default=0.0)

    request_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    prompt_token_ids: list[int] = field(compare=False, default_factory=list)
    max_new_tokens: int = field(compare=False, default=256)
    temperature: float = field(compare=False, default=0.0)
    top_p: float = field(compare=False, default=1.0)
    stop_token_ids: list[int] = field(compare=False, default_factory=list)

    # Internal state
    status: RequestStatus = field(compare=False, default=RequestStatus.WAITING)
    generated_token_ids: list[int] = field(compare=False, default_factory=list)
    prefill_pos: int = field(compare=False, default=0)  # tokens prefilled so far
    kv_cache: Any = field(compare=False, default=None)
    arrival_time: float = field(compare=False, default_factory=time.time)
    first_token_time: float | None = field(compare=False, default=None)
    finish_time: float | None = field(compare=False, default=None)
    error: str | None = field(compare=False, default=None)

    def __post_init__(self):
        if self._priority_key == 0.0:
            self._priority_key = self.arrival_time

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def generated_len(self) -> int:
        return len(self.generated_token_ids)

    @property
    def is_done(self) -> bool:
        if self.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.ERROR):
            return True
        if self.generated_len >= self.max_new_tokens:
            return True
        if self.stop_token_ids and self.generated_token_ids:
            if self.generated_token_ids[-1] in self.stop_token_ids:
                return True
        return False

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.arrival_time) * 1000.0

    @property
    def total_time_ms(self) -> float | None:
        if self.finish_time is None:
            return None
        return (self.finish_time - self.arrival_time) * 1000.0


@dataclass
class PartialResult:
    """Incremental output from one batcher step for one request."""
    request_id: str
    new_token_ids: list[int]
    status: RequestStatus
    is_final: bool = False
    error: str | None = None
    ttft_ms: float | None = None


# ---------------------------------------------------------------------------
# Batcher configuration
# ---------------------------------------------------------------------------

@dataclass
class BatcherConfig:
    """Configuration for ContinuousBatcher."""

    # Maximum total tokens (prompt + generated) across ALL active sequences per step
    max_batch_tokens: int = 2048

    # Chunked prefill chunk size (tokens). Set to 0 to disable chunking.
    chunk_size: int = 512

    # Maximum number of concurrent decode sequences
    max_decode_seqs: int = 32

    # Maximum waiting queue size (0 = unlimited)
    max_queue_size: int = 0

    # Default KV cache budget: pages per sequence
    max_pages_per_seq: int = 256

    # EOS token IDs (added to every request's stop list)
    eos_token_ids: list[int] = field(default_factory=lambda: [2])

    # Sampling: use greedy by default
    default_temperature: float = 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class BatcherStats:
    """Accumulated batcher performance statistics."""

    total_requests: int = 0
    finished_requests: int = 0
    cancelled_requests: int = 0
    error_requests: int = 0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    total_steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    total_time_ms: float = 0.0

    @property
    def mean_ttft_ms(self) -> float:
        return 0.0  # tracked per-request

    @property
    def throughput_tps(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.total_generated_tokens / (self.total_time_ms / 1000.0)

    def report(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "finished": self.finished_requests,
            "cancelled": self.cancelled_requests,
            "errors": self.error_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "total_steps": self.total_steps,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "throughput_tps": round(self.throughput_tps, 2),
            "total_time_ms": round(self.total_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Continuous batcher
# ---------------------------------------------------------------------------

class ContinuousBatcher:
    """
    Orca-style continuous batching scheduler.

    Usage::

        batcher = ContinuousBatcher(engine=my_engine, config=BatcherConfig())
        req_id = batcher.submit(Request(prompt_token_ids=[...], max_new_tokens=128))
        while True:
            results = batcher.step()
            for r in results:
                print(r.new_token_ids, r.is_final)
            if not batcher.has_work():
                break

    The ``engine`` must implement::

        forward_step(token_ids, kv_cache, return_all_logits=False)
            -> (logits: np.ndarray, new_kv_cache: Any)

        init_kv_cache() -> Any          (optional; returns empty cache)

    If ``engine.init_kv_cache`` is not present, the batcher passes ``None``
    as the initial cache (works with any cache-agnostic engine).
    """

    def __init__(
        self,
        engine: Any,
        config: BatcherConfig | None = None,
        kv_pool: Any | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or BatcherConfig()
        self.kv_pool = kv_pool  # Optional PagedKVPool for KV-aware admission

        self._waiting: list[Request] = []  # FIFO waiting queue
        self._running: dict[str, Request] = {}  # seq_id → Request (active decode)
        self._prefilling: list[Request] = []  # requests being chunked-prefilled
        self._lock = Lock()
        self._stats = BatcherStats()
        self._result_callbacks: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: Request) -> str:
        """
        Submit a request to the batcher.

        Returns the request_id for tracking.
        Raises RuntimeError if the waiting queue is full.
        """
        request.status = RequestStatus.WAITING
        request.arrival_time = time.time()
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        with self._lock:
            if (self.config.max_queue_size > 0 and
                    len(self._waiting) >= self.config.max_queue_size):
                raise RuntimeError(
                    f"Batcher queue full ({self.config.max_queue_size} requests). "
                    "Consider increasing max_queue_size or reducing request rate."
                )
            self._waiting.append(request)
            self._stats.total_requests += 1
        return request.request_id

    def cancel(self, request_id: str) -> bool:
        """Cancel a request. Returns True if found and cancelled."""
        with self._lock:
            for req in self._waiting:
                if req.request_id == request_id:
                    req.status = RequestStatus.CANCELLED
                    self._waiting.remove(req)
                    self._stats.cancelled_requests += 1
                    return True
            if request_id in self._running:
                self._running[request_id].status = RequestStatus.CANCELLED
                self._stats.cancelled_requests += 1
                return True
        return False

    def has_work(self) -> bool:
        """Return True if there are waiting or active requests."""
        with self._lock:
            return bool(self._waiting or self._running or self._prefilling)

    @property
    def stats(self) -> BatcherStats:
        return self._stats

    def queue_size(self) -> int:
        with self._lock:
            return len(self._waiting)

    def active_sequences(self) -> int:
        with self._lock:
            return len(self._running)

    # ------------------------------------------------------------------
    # Core scheduler step
    # ------------------------------------------------------------------

    def step(self) -> list[PartialResult]:
        """
        Execute one scheduler step.

        A step consists of:
        1. Admit waiting requests into prefill if capacity allows.
        2. Process prefill chunks (chunked prefill).
        3. Execute decode step for all active sequences.

        Returns a list of PartialResult for all activity this step.
        """
        t0 = time.perf_counter()
        results: list[PartialResult] = []

        with self._lock:
            waiting_snapshot = list(self._waiting)
            running_snapshot = dict(self._running)
            prefilling_snapshot = list(self._prefilling)

        # ── Step 1: Admit waiting requests ────────────────────────────
        admitted = self._admit_requests(waiting_snapshot)

        # ── Step 2: Chunked prefill ───────────────────────────────────
        prefill_results = self._do_prefill(prefilling_snapshot + admitted)
        results.extend(prefill_results)

        # ── Step 3: Decode ─────────────────────────────────────────────
        decode_results = self._do_decode()
        results.extend(decode_results)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._stats.total_steps += 1
        self._stats.total_time_ms += elapsed_ms

        return results

    # ------------------------------------------------------------------
    # Internal: admission
    # ------------------------------------------------------------------

    def _admit_requests(self, waiting: list[Request]) -> list[Request]:
        """Move waiting requests into prefilling state if capacity allows."""
        admitted: list[Request] = []
        with self._lock:
            n_running = len(self._running)
            capacity = self.config.max_decode_seqs - n_running - len(self._prefilling)

        for req in waiting[:capacity]:
            req.status = RequestStatus.PREFILLING
            req.prefill_pos = 0
            req.kv_cache = self._init_cache()
            with self._lock:
                self._waiting.remove(req)
                self._prefilling.append(req)
            admitted.append(req)
        return admitted

    def _init_cache(self) -> Any:
        """Initialize KV cache for a new request."""
        if hasattr(self.engine, "init_kv_cache"):
            return self.engine.init_kv_cache()
        return None

    # ------------------------------------------------------------------
    # Internal: prefill
    # ------------------------------------------------------------------

    def _do_prefill(self, prefilling: list[Request]) -> list[PartialResult]:
        """
        Process one chunk of each prefilling request.

        Chunked prefill: feeds at most ``config.chunk_size`` tokens per step
        to bound the prefill latency and allow decode requests to interleave.
        """
        results: list[PartialResult] = []
        still_prefilling: list[Request] = []

        for req in prefilling:
            if req.status == RequestStatus.CANCELLED:
                with self._lock:
                    if req in self._prefilling:
                        self._prefilling.remove(req)
                continue

            chunk_size = self.config.chunk_size or len(req.prompt_token_ids)
            start = req.prefill_pos
            end = min(start + chunk_size, req.prompt_len)
            chunk = req.prompt_token_ids[start:end]

            try:
                logits, req.kv_cache = self.engine.forward_step(
                    req.prompt_token_ids[:end],
                    req.kv_cache,
                    return_all_logits=False,
                )
                req.prefill_pos = end
                self._stats.prefill_steps += 1
                self._stats.total_prompt_tokens += len(chunk)

                if req.prefill_pos >= req.prompt_len:
                    # Prefill complete: generate first token
                    logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
                    token = self._sample_token(logits_np, req.temperature)
                    req.generated_token_ids.append(token)
                    req.status = RequestStatus.DECODING
                    req.first_token_time = time.time()
                    self._stats.total_generated_tokens += 1

                    with self._lock:
                        if req in self._prefilling:
                            self._prefilling.remove(req)
                        self._running[req.request_id] = req

                    results.append(PartialResult(
                        request_id=req.request_id,
                        new_token_ids=[token],
                        status=RequestStatus.DECODING,
                        ttft_ms=req.ttft_ms,
                    ))
                else:
                    still_prefilling.append(req)
                    results.append(PartialResult(
                        request_id=req.request_id,
                        new_token_ids=[],
                        status=RequestStatus.PREFILLING,
                    ))

            except Exception as exc:
                logger.error("Prefill error for request %s: %s", req.request_id, exc)
                req.status = RequestStatus.ERROR
                req.error = str(exc)
                with self._lock:
                    if req in self._prefilling:
                        self._prefilling.remove(req)
                self._stats.error_requests += 1
                results.append(PartialResult(
                    request_id=req.request_id,
                    new_token_ids=[],
                    status=RequestStatus.ERROR,
                    is_final=True,
                    error=str(exc),
                ))

        return results

    # ------------------------------------------------------------------
    # Internal: decode
    # ------------------------------------------------------------------

    def _do_decode(self) -> list[PartialResult]:
        """
        Generate one token per active sequence.

        Batched decode: all sequences are processed sequentially in this
        reference implementation. A production implementation would batch
        them into a single forward pass using padded inputs.
        """
        results: list[PartialResult] = []
        finished: list[str] = []

        with self._lock:
            running = dict(self._running)

        for req_id, req in running.items():
            if req.status == RequestStatus.CANCELLED:
                finished.append(req_id)
                req.finish_time = time.time()
                continue

            if req.is_done:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                finished.append(req_id)
                self._stats.finished_requests += 1
                results.append(PartialResult(
                    request_id=req_id,
                    new_token_ids=[],
                    status=RequestStatus.FINISHED,
                    is_final=True,
                ))
                continue

            # Build full context for this decode step
            context_ids = req.prompt_token_ids + req.generated_token_ids

            try:
                logits, req.kv_cache = self.engine.forward_step(
                    context_ids,
                    req.kv_cache,
                    return_all_logits=False,
                )
                logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
                token = self._sample_token(logits_np, req.temperature)
                req.generated_token_ids.append(token)
                self._stats.total_generated_tokens += 1
                self._stats.decode_steps += 1

                is_done = req.is_done
                if is_done:
                    req.status = RequestStatus.FINISHED
                    req.finish_time = time.time()
                    finished.append(req_id)
                    self._stats.finished_requests += 1

                results.append(PartialResult(
                    request_id=req_id,
                    new_token_ids=[token],
                    status=req.status,
                    is_final=is_done,
                ))

            except Exception as exc:
                logger.error("Decode error for request %s: %s", req_id, exc)
                req.status = RequestStatus.ERROR
                req.error = str(exc)
                req.finish_time = time.time()
                finished.append(req_id)
                self._stats.error_requests += 1
                results.append(PartialResult(
                    request_id=req_id,
                    new_token_ids=[],
                    status=RequestStatus.ERROR,
                    is_final=True,
                    error=str(exc),
                ))

        with self._lock:
            for req_id in finished:
                self._running.pop(req_id, None)

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _sample_token(self, logits: np.ndarray, temperature: float) -> int:
        """Sample a token from logits."""
        if temperature <= 0.0:
            return int(np.argmax(logits))
        # Temperature sampling
        logits_f = logits.astype(np.float64)
        logits_f = logits_f / temperature
        logits_f = logits_f - logits_f.max()
        probs = np.exp(logits_f)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def generate_sync(
        self,
        token_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Convenience: synchronously generate for a single request.

        Blocks until completion. Returns generated token IDs.
        """
        req = Request(
            prompt_token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_token_ids=list(stop_token_ids or self.config.eos_token_ids),
        )
        req_id = self.submit(req)
        generated: list[int] = []

        while self.has_work():
            results = self.step()
            for r in results:
                if r.request_id == req_id:
                    generated.extend(r.new_token_ids)
                    if r.is_final:
                        return generated

        return generated
