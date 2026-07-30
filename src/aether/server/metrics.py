"""
Prometheus-compatible metrics collection for the Aether server.

Hooks into the FastAPI request lifecycle and the runtime to expose:
- request count, latency histograms, active request gauge
- generation throughput and token counts
- memory pressure
- backend and target distribution counters
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ServerMetrics:
    """In-memory metrics accumulator.

    In a production deployment these would be Prometheus histograms, counters,
    and gauges. This implementation provides methods with matching semantics
    that are safe to replace one-for-one with a Prometheus client.
    """

    # Counters
    http_requests_total: int = 0
    http_requests_by_method: dict[str, int] = field(default_factory=dict)
    http_requests_by_status: dict[str, int] = field(default_factory=dict)
    generate_requests_total: int = 0
    generate_tokens_total: int = 0
    embed_requests_total: int = 0
    errors_total: int = 0

    # Histograms (recorded as lists; summary stats are derived)
    request_duration_ms: list[float] = field(default_factory=list)
    generate_latency_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    throughput_tps: list[float] = field(default_factory=list)

    # Gauges
    active_requests: int = 0
    active_sessions: int = 0
    memory_pressure: float = 0.0
    loaded_models: int = 0
    kv_cache_utilization: float = 0.0

    # Backend distribution
    backend_counts: dict[str, int] = field(default_factory=dict)
    target_counts: dict[str, int] = field(default_factory=dict)
    finish_reason_counts: dict[str, int] = field(default_factory=dict)

    def record_http_request(self, method: str, status_code: int, duration_ms: float) -> None:
        """Record an HTTP request."""
        self.http_requests_total += 1
        self.http_requests_by_method[method.upper()] = self.http_requests_by_method.get(method.upper(), 0) + 1
        status_key = f"{status_code // 100}xx"
        self.http_requests_by_status[status_key] = self.http_requests_by_status.get(status_key, 0) + 1
        self.request_duration_ms.append(duration_ms)

    def record_generate(self, tokens: int, latency_ms: float, ttft: float, tps: float, backend: str, target: str, finish_reason: str) -> None:
        """Record a generation request."""
        self.generate_requests_total += 1
        self.generate_tokens_total += tokens
        self.generate_latency_ms.append(latency_ms)
        self.ttft_ms.append(ttft)
        if tps > 0:
            self.throughput_tps.append(tps)
        self.backend_counts[backend] = self.backend_counts.get(backend, 0) + 1
        self.target_counts[target] = self.target_counts.get(target, 0) + 1
        self.finish_reason_counts[finish_reason] = self.finish_reason_counts.get(finish_reason, 0) + 1

    def record_error(self) -> None:
        """Record an error."""
        self.errors_total += 1

    def p50(self, values: list[float]) -> float:
        """Compute the 50th percentile of a sorted list."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        return sorted_vals[len(sorted_vals) // 2]

    def _percentile(self, values: list[float], p: float) -> float:
        """Compute the p-th percentile using the nearest-rank method."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        rank = int(__import__("math").ceil(p * len(sorted_vals)))
        idx = max(0, min(len(sorted_vals) - 1, rank - 1))
        return sorted_vals[idx]

    def p95(self, values: list[float]) -> float:
        """Compute the 95th percentile of a sorted list."""
        return self._percentile(values, 0.95)

    def p99(self, values: list[float]) -> float:
        """Compute the 99th percentile of a sorted list."""
        return self._percentile(values, 0.99)

    def to_dict(self) -> dict[str, Any]:
        """Return a snapshot of all metrics."""
        return {
            "http": {
                "total": self.http_requests_total,
                "by_method": self.http_requests_by_method,
                "by_status": self.http_requests_by_status,
            },
            "generation": {
                "total": self.generate_requests_total,
                "total_tokens": self.generate_tokens_total,
                "p50_latency_ms": self.p50(self.generate_latency_ms),
                "p95_latency_ms": self.p95(self.generate_latency_ms),
                "p99_latency_ms": self.p99(self.generate_latency_ms),
                "p50_ttft_ms": self.p50(self.ttft_ms),
                "p95_ttft_ms": self.p95(self.ttft_ms),
            },
            "gauges": {
                "active_requests": self.active_requests,
                "active_sessions": self.active_sessions,
                "memory_pressure": self.memory_pressure,
                "loaded_models": self.loaded_models,
                "kv_cache_utilization": self.kv_cache_utilization,
            },
            "distribution": {
                "backends": self.backend_counts,
                "targets": self.target_counts,
                "finish_reasons": self.finish_reason_counts,
            },
            "errors_total": self.errors_total,
        }

    def __repr__(self) -> str:
        return f"ServerMetrics(requests={self.http_requests_total}, errors={self.errors_total})"
