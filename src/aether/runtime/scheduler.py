"""
Disaggregated prefill/decode scheduler.

Separates compute-bound prefill (parallel token processing) from memory-bandwidth-
bound decode (serial token generation). Supports chunked prefill to bound TTFT
and continuous batching for decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScheduledRequest:
    """A request in the scheduler queue."""

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    generated_tokens: list[int] = field(default_factory=list)
    prefill_cursor: int = 0
    """Number of prompt tokens already scheduled through prefill."""

    last_prefill_chunk: tuple[int, int] | None = None
    """Half-open token range most recently scheduled for prefill."""

    phase: str = "prefill"
    """Current phase: 'prefill', 'prefill_scheduled', 'decode', or 'complete'."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": len(self.prompt_tokens),
            "max_tokens": self.max_tokens,
            "generated_tokens": len(self.generated_tokens),
            "prefill_cursor": self.prefill_cursor,
            "last_prefill_chunk": self.last_prefill_chunk,
            "phase": self.phase,
        }


class DisaggregatedScheduler:
    """Scheduler that separates prefill and decode phases."""

    def __init__(self, max_batch_size: int = 256, prefill_chunk_size: int = 2048) -> None:
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self._prefill_queue: list[ScheduledRequest] = []
        self._decode_queue: list[ScheduledRequest] = []
        self._completed: list[ScheduledRequest] = []
        self._request_counter = 0

    @property
    def pending_prefill_count(self) -> int:
        return len(self._prefill_queue)

    @property
    def pending_decode_count(self) -> int:
        return len(self._decode_queue)

    def submit(self, prompt_tokens: list[int], max_tokens: int, temperature: float, top_p: float) -> str:
        """Submit a new request and return a request ID."""
        self._request_counter += 1
        request_id = f"req_{self._request_counter}"
        request = ScheduledRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            phase="prefill",
        )
        self._prefill_queue.append(request)
        logger.info(f"Request {request_id} submitted: {len(prompt_tokens)} prompt tokens")
        return request_id

    def schedule_prefill(self) -> list[ScheduledRequest]:
        """Return the next prefill microbatch.

        Requests are served in FIFO order, but long prompts are chunked so a
        single large prefill cannot monopolize the prefill pool. The returned
        request objects carry `last_prefill_chunk` so an executor can process
        exactly the scheduled token range.
        """
        batch: list[ScheduledRequest] = []
        token_budget = self.max_batch_size * self.prefill_chunk_size
        used_tokens = 0
        remaining_queue: list[ScheduledRequest] = []
        for request in self._prefill_queue:
            if len(batch) >= self.max_batch_size:
                remaining_queue.append(request)
                continue
            remaining_tokens = len(request.prompt_tokens) - request.prefill_cursor
            if remaining_tokens <= 0:
                request.phase = "decode"
                self._decode_queue.append(request)
                continue
            chunk_tokens = min(remaining_tokens, self.prefill_chunk_size)
            if used_tokens and used_tokens + chunk_tokens > token_budget:
                remaining_queue.append(request)
                continue
            chunk_start = request.prefill_cursor
            request.prefill_cursor += chunk_tokens
            request.last_prefill_chunk = (chunk_start, request.prefill_cursor)
            request.phase = "prefill_scheduled"
            batch.append(request)
            used_tokens += chunk_tokens
        self._prefill_queue = remaining_queue
        return batch

    def finish_prefill(self, requests: list[ScheduledRequest]) -> None:
        """Move completed prefills to decode and requeue partial chunks."""
        for request in requests:
            if request.prefill_cursor < len(request.prompt_tokens):
                request.phase = "prefill"
                self._prefill_queue.append(request)
                logger.info(f"Request {request.request_id} requeued for next prefill chunk")
                continue
            request.phase = "decode"
            self._decode_queue.append(request)
            logger.info(f"Request {request.request_id} moved to decode phase")

    def schedule_decode(self) -> list[ScheduledRequest]:
        """Return the next decode batch ordered by remaining work."""
        active = sorted(
            self._decode_queue,
            key=lambda request: (request.max_tokens - len(request.generated_tokens), request.request_id),
        )
        return active[: self.max_batch_size]

    def advance_decode(self, generated_tokens: dict[str, int]) -> list[str]:
        """Advance each decode request by one generated token and return completed IDs."""
        completed: list[str] = []
        active: list[ScheduledRequest] = []
        for request in self._decode_queue:
            token_id = generated_tokens.get(request.request_id)
            if token_id is not None:
                request.generated_tokens.append(token_id)
            if len(request.generated_tokens) >= request.max_tokens:
                request.phase = "complete"
                self._completed.append(request)
                completed.append(request.request_id)
            else:
                active.append(request)
        self._decode_queue = active
        return completed

    def get_request(self, request_id: str) -> ScheduledRequest | None:
        """Look up a request by ID."""
        for queue in (self._prefill_queue, self._decode_queue, self._completed):
            for request in queue:
                if request.request_id == request_id:
                    return request
        return None

    def queue_snapshot(self) -> dict[str, Any]:
        """Return scheduler state useful for metrics and tests."""
        return {
            "pending_prefill": self.pending_prefill_count,
            "pending_decode": self.pending_decode_count,
            "completed": len(self._completed),
            "prefill_tokens_remaining": sum(
                max(0, len(request.prompt_tokens) - request.prefill_cursor)
                for request in self._prefill_queue
            ),
            "decode_tokens_remaining": sum(
                max(0, request.max_tokens - len(request.generated_tokens))
                for request in self._decode_queue
            ),
        }

    def __repr__(self) -> str:
        return (
            f"DisaggregatedScheduler(prefill={self.pending_prefill_count}, "
            f"decode={self.pending_decode_count})"
        )
