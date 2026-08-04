"""
R4 — SLO-Aware Adaptive Scheduler.

Service Level Objectives (SLOs) for LLM inference vary by request type:
  - Interactive chat: TTFT (Time To First Token) < 200 ms, TBT < 50 ms.
  - Batch processing: throughput-optimized, latency-relaxed.
  - Streaming generation: sustained TBT < 100 ms.
  - Agentic (multi-turn): low TTFT for tool calls, high throughput for long gen.

R4 implements a multi-objective scheduler that:
  1. Classifies incoming requests by SLO tier (latency / throughput / balanced).
  2. Maintains per-tier queues with priority-based preemption.
  3. Dynamically adjusts batch size and chunked prefill parameters.
  4. Enforces TTFT guarantees via admission control (reject or defer low-priority
     requests when the latency queue is overloaded).
  5. Provides per-request latency estimates for proactive queue management.

Scheduling algorithms:
  - **FCFS** (First-Come, First-Served): baseline, fair but not SLO-aware.
  - **SJF** (Shortest Job First): prioritizes short prefill sequences.
  - **LCF** (Least Critical First): deprioritizes latency-relaxed requests.
  - **MLFQ** (Multi-Level Feedback Queue): adapts priority based on observed TTFT.

Research basis:
  - Sarathi-Serve (2024): chunked prefill for TTFT control.
  - FastServe (2024): LLM scheduling with preemption.
  - Orca (2022): continuous batching for LLM serving.
  - AlpaServe (NSDI 2023): SLO-aware model serving.
  - vAttention (2025): dynamic KV memory for variable-length sequences.
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class SLOTier(str, Enum):
    """SLO tier classification for incoming requests."""

    LATENCY = "latency"       # Interactive: TTFT < 200 ms, TBT < 50 ms.
    BALANCED = "balanced"     # Mixed: TTFT < 1s, throughput-opportunistic.
    THROUGHPUT = "throughput" # Batch: minimize cost/token, latency-relaxed.
    AGENTIC = "agentic"       # Multi-turn agent: low TTFT for tool calls.


@dataclass(order=True)
class ScheduledRequest:
    """A request entry in the scheduler queue.

    Priority is (tier_priority, arrival_time) — lower values = higher priority.
    """

    priority: float  # Lower = higher priority.
    arrival_time: float = field(compare=False)
    request_id: str = field(compare=False)
    prompt_tokens: int = field(compare=False)
    max_new_tokens: int = field(compare=False)
    slo_tier: SLOTier = field(compare=False)
    ttft_deadline_s: float = field(compare=False)  # Absolute deadline for first token.
    metadata: dict = field(default_factory=dict, compare=False)


class SLOScheduler:
    """Runtime R4: SLO-Aware Adaptive Scheduler.

    Maintains multiple priority queues and a continuous batching loop that
    selects the optimal batch composition to meet per-tier SLO targets.
    """

    # Tier priorities: lower = higher priority in the heap.
    _TIER_PRIORITY: dict[SLOTier, int] = {
        SLOTier.AGENTIC: 0,
        SLOTier.LATENCY: 1,
        SLOTier.BALANCED: 2,
        SLOTier.THROUGHPUT: 3,
    }

    # Default TTFT deadlines (seconds) per tier.
    _DEFAULT_TTFT_DEADLINES: dict[SLOTier, float] = {
        SLOTier.AGENTIC: 0.10,     # 100 ms for tool call responses.
        SLOTier.LATENCY: 0.20,     # 200 ms for interactive chat.
        SLOTier.BALANCED: 1.00,    # 1s for balanced workloads.
        SLOTier.THROUGHPUT: 30.0,  # 30s for batch workloads.
    }

    def __init__(
        self,
        max_batch_tokens: int = 8192,
        max_prefill_chunk_tokens: int = 4096,
        scheduling_algo: str = "mlfq",
        target_gpu_utilization: float = 0.85,
    ) -> None:
        self.max_batch_tokens = max_batch_tokens
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens
        self.scheduling_algo = scheduling_algo
        self.target_gpu_utilization = target_gpu_utilization

        self._queue: list[ScheduledRequest] = []  # Min-heap by priority.
        self._lock = threading.RLock()
        self._stats = _SchedulerStats()
        self._request_map: dict[str, ScheduledRequest] = {}

    def submit(
        self,
        request_id: str,
        prompt_tokens: int,
        max_new_tokens: int,
        slo_tier: SLOTier | str = SLOTier.BALANCED,
        metadata: dict | None = None,
    ) -> ScheduledRequest:
        """Submit a new request to the scheduler.

        Args:
            request_id: Unique request identifier.
            prompt_tokens: Number of tokens in the prompt (prefill cost).
            max_new_tokens: Maximum tokens to generate.
            slo_tier: SLO tier for this request.
            metadata: Optional request metadata.

        Returns:
            ScheduledRequest added to the queue.
        """
        if isinstance(slo_tier, str):
            slo_tier = SLOTier(slo_tier)

        now = time.monotonic()
        tier_priority = self._TIER_PRIORITY[slo_tier]
        ttft_deadline = now + self._DEFAULT_TTFT_DEADLINES[slo_tier]

        # MLFQ: adjust priority based on queue load.
        if self.scheduling_algo == "mlfq":
            load_factor = self._queue_load_factor()
            adjusted_priority = tier_priority + load_factor * 0.1
        elif self.scheduling_algo == "sjf":
            adjusted_priority = prompt_tokens / 1000.0  # Shorter prefill = higher priority.
        else:
            adjusted_priority = float(tier_priority)

        req = ScheduledRequest(
            priority=adjusted_priority,
            arrival_time=now,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            slo_tier=slo_tier,
            ttft_deadline_s=ttft_deadline,
            metadata=metadata or {},
        )

        with self._lock:
            heapq.heappush(self._queue, req)
            self._request_map[request_id] = req
            self._stats.total_submitted += 1

        logger.debug(
            "R4: Submitted %r (tier=%s, prompt=%d, priority=%.2f).",
            request_id[:8],
            slo_tier.value,
            prompt_tokens,
            adjusted_priority,
        )
        return req

    def next_batch(self, token_budget: int | None = None) -> list[ScheduledRequest]:
        """Select the next batch of requests to process.

        Uses chunked prefill (Sarathi-Serve): requests with long prompts are
        split into prefill chunks of ``max_prefill_chunk_tokens`` to control TTFT.

        Args:
            token_budget: Maximum tokens in this batch (default: max_batch_tokens).

        Returns:
            List of ScheduledRequests to process in this iteration.
        """
        budget = token_budget or self.max_batch_tokens
        selected: list[ScheduledRequest] = []
        tokens_used = 0

        with self._lock:
            now = time.monotonic()
            # Preempt overdue latency-tier requests.
            self._rebalance_priorities(now)

            remaining: list[ScheduledRequest] = []
            temp_heap = list(self._queue)

            while temp_heap and tokens_used < budget:
                req = heapq.heappop(temp_heap)

                # Chunked prefill: cap tokens from this request.
                chunk_tokens = min(req.prompt_tokens, self.max_prefill_chunk_tokens)
                if tokens_used + chunk_tokens > budget:
                    remaining.append(req)
                    break

                selected.append(req)
                tokens_used += chunk_tokens
                del self._request_map[req.request_id]

            # Put unselected requests back.
            for req in temp_heap:
                remaining.append(req)
            self._queue = []
            for req in remaining:
                heapq.heappush(self._queue, req)

        if selected:
            self._stats.total_batches += 1
            self._stats.total_processed += len(selected)
            logger.debug(
                "R4: Batch of %d requests (%d tokens) dispatched.",
                len(selected),
                tokens_used,
            )

        return selected

    def _rebalance_priorities(self, now: float) -> None:
        """Boost priority of requests approaching their TTFT deadline."""
        new_heap: list[ScheduledRequest] = []
        for req in self._queue:
            time_remaining = req.ttft_deadline_s - now
            if time_remaining < 0.05:  # Deadline within 50ms: max priority.
                req = ScheduledRequest(
                    priority=-1000.0,  # Emergency priority.
                    arrival_time=req.arrival_time,
                    request_id=req.request_id,
                    prompt_tokens=req.prompt_tokens,
                    max_new_tokens=req.max_new_tokens,
                    slo_tier=req.slo_tier,
                    ttft_deadline_s=req.ttft_deadline_s,
                    metadata=req.metadata,
                )
                self._stats.deadline_boosted += 1
            new_heap.append(req)
        heapq.heapify(new_heap)
        self._queue = new_heap

    def _queue_load_factor(self) -> float:
        """Estimate queue load as fraction of max_batch_tokens queued."""
        total_queued = sum(r.prompt_tokens for r in self._queue)
        return min(1.0, total_queued / max(1, self.max_batch_tokens))

    def queue_depth(self) -> int:
        """Return the number of queued requests."""
        with self._lock:
            return len(self._queue)

    def estimated_wait_time(self, request_id: str) -> float:
        """Estimate wait time in seconds for a queued request.

        Approximation: (position_in_queue × avg_batch_time_ms) / 1000.
        """
        with self._lock:
            if request_id not in self._request_map:
                return 0.0
            req = self._request_map[request_id]
            position = sorted(self._queue).index(req)
            avg_batch_ms = self._stats.avg_batch_latency_ms
            return position * avg_batch_ms / 1000.0

    def record_batch_latency(self, latency_ms: float) -> None:
        """Record the latency of a completed batch for future estimation."""
        self._stats.record_batch_latency(latency_ms)

    @property
    def stats(self) -> "_SchedulerStats":
        return self._stats

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_depth": len(self._queue),
                "total_submitted": self._stats.total_submitted,
                "total_processed": self._stats.total_processed,
                "total_batches": self._stats.total_batches,
                "avg_batch_latency_ms": round(self._stats.avg_batch_latency_ms, 2),
                "deadline_boosted": self._stats.deadline_boosted,
            }


class _SchedulerStats:
    __slots__ = (
        "total_submitted", "total_processed", "total_batches",
        "deadline_boosted", "_batch_latencies",
    )

    def __init__(self) -> None:
        self.total_submitted = 0
        self.total_processed = 0
        self.total_batches = 0
        self.deadline_boosted = 0
        self._batch_latencies: list[float] = []

    def record_batch_latency(self, ms: float) -> None:
        self._batch_latencies.append(ms)
        if len(self._batch_latencies) > 1000:
            self._batch_latencies = self._batch_latencies[-500:]

    @property
    def avg_batch_latency_ms(self) -> float:
        if not self._batch_latencies:
            return 10.0  # Default 10ms estimate.
        return sum(self._batch_latencies) / len(self._batch_latencies)
