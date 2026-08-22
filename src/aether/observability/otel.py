"""OpenTelemetry-native tracing and metrics for Aether Runtime.

Provides per-request spans, P50/P95/P99 latency histograms, and an OTLP-compatible
JSON exporter compatible with Jaeger, Grafana Tempo, and Honeycomb.

Research: OpenTelemetry Specification v1.28 (2024), OTLP over HTTP/gRPC.
"""

from __future__ import annotations

import json
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.constants import AETHER_VERSION


# ---------------------------------------------------------------------------
# Span model
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """A single OpenTelemetry-compatible span."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_ns: int
    end_time_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"  # "OK" | "ERROR"
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1_000_000

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp_ns": time.time_ns(),
            "attributes": attributes or {},
        })

    def set_error(self, message: str) -> None:
        self.status = "ERROR"
        self.attributes["error.message"] = message

    def end(self) -> None:
        self.end_time_ns = time.time_ns()

    def to_otlp_dict(self) -> dict[str, Any]:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "name": self.name,
            "kind": "SPAN_KIND_SERVER",
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(self.end_time_ns),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ],
            "events": self.events,
            "status": {"code": "STATUS_CODE_OK" if self.status == "OK" else "STATUS_CODE_ERROR"},
        }


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class AetherTracer:
    """
    OpenTelemetry-compatible request tracer for Aether inference.

    Creates and finishes spans for every phase of request processing:
      - prefill (tokenization + KV build)
      - decode (token-by-token generation)
      - speculative (EAGLE-3 draft + verify)
      - reasoning (CoT budget phases)
      - retrieval (RAG retrieval + rerank)
    """

    def __init__(self, service_name: str = "aether-runtime") -> None:
        self.service_name = service_name
        self._finished_spans: list[Span] = []

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        span = Span(
            name=name,
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            start_time_ns=time.time_ns(),
            attributes=attributes or {},
        )
        return span

    def finish_span(self, span: Span, attributes: dict[str, Any] | None = None) -> Span:
        span.end()
        if attributes:
            span.attributes.update(attributes)
        self._finished_spans.append(span)
        return span

    def trace_request(
        self,
        request_id: str,
        prompt_tokens: int,
        generated_tokens: int,
        ttft_ms: float,
        total_ms: float,
        model_id: str = "unknown",
        adapter_id: str | None = None,
        actual_start_time_ns: int | None = None,
        actual_end_time_ns: int | None = None,
    ) -> Span:
        """Record a complete inference request as a single span."""
        span = self.start_span(
            name="aether.inference",
            trace_id=uuid.uuid4().hex,
            attributes={
                "request_id": request_id,
                "model_id": model_id,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "tokens_per_second": round(generated_tokens / max(total_ms / 1000, 1e-6), 2),
                "ttft_ms": round(ttft_ms, 3),
                "total_latency_ms": round(total_ms, 3),
                "adapter_id": adapter_id or "base",
            },
        )
        # Callers that have measured wall-clock boundaries can provide them.
        # Without boundaries this helper records an instantaneous event; it
        # never backdates timestamps to manufacture latency.
        span.start_time_ns = actual_start_time_ns or time.time_ns()
        span.end_time_ns = actual_end_time_ns or span.start_time_ns
        span.attributes["duration_source"] = "measured" if actual_start_time_ns and actual_end_time_ns else "event_only"
        span.add_event("prefill_complete", {"ttft_ms": ttft_ms, "prompt_tokens": prompt_tokens})
        span.add_event("decode_complete", {"generated_tokens": generated_tokens})
        self._finished_spans.append(span)
        return span

    def get_finished_spans(self) -> list[Span]:
        return list(self._finished_spans)

    def export_otlp_json(self) -> dict[str, Any]:
        """Export all finished spans as OTLP JSON payload."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                            {"key": "service.version", "value": {"stringValue": AETHER_VERSION}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "aether.tracer", "version": "1.2.0"},
                            "spans": [span.to_otlp_dict() for span in self._finished_spans],
                        }
                    ],
                }
            ]
        }

    def clear(self) -> None:
        self._finished_spans.clear()


# ---------------------------------------------------------------------------
# Metrics Collector — P50/P95/P99 histograms
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects and computes latency percentiles and throughput histograms.

    All statistics are computed with stdlib `statistics` module — no external deps.
    """

    def __init__(self) -> None:
        self._ttft_samples: list[float] = []
        self._tps_samples: list[float] = []
        self._e2e_ms_samples: list[float] = []
        self._eagle_accept_samples: list[float] = []
        self._kv_hit_samples: list[float] = []
        self._request_count: int = 0
        self._error_count: int = 0

    def record(
        self,
        ttft_ms: float,
        tokens_per_second: float,
        e2e_latency_ms: float,
        eagle_accept_rate: float = 0.0,
        kv_hit_rate: float = 0.0,
        is_error: bool = False,
    ) -> None:
        """Record one request's metrics."""
        self._request_count += 1
        self._ttft_samples.append(ttft_ms)
        self._tps_samples.append(tokens_per_second)
        self._e2e_ms_samples.append(e2e_latency_ms)
        if eagle_accept_rate > 0:
            self._eagle_accept_samples.append(eagle_accept_rate)
        if kv_hit_rate > 0:
            self._kv_hit_samples.append(kv_hit_rate)
        if is_error:
            self._error_count += 1

    def _percentiles(self, samples: list[float]) -> dict[str, float]:
        if not samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        n = len(samples)
        if n == 1:
            v = samples[0]
            return {"p50": v, "p95": v, "p99": v, "mean": v, "min": v, "max": v}
        sorted_s = sorted(samples)
        qs = statistics.quantiles(sorted_s, n=100)
        return {
            "p50": round(qs[49], 3),
            "p95": round(qs[94], 3),
            "p99": round(qs[98], 3),
            "mean": round(statistics.fmean(sorted_s), 3),
            "min": round(sorted_s[0], 3),
            "max": round(sorted_s[-1], 3),
        }

    def report(self) -> dict[str, Any]:
        """Return a complete metrics report with all percentile histograms."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / max(self._request_count, 1), 6),
            "ttft_ms": self._percentiles(self._ttft_samples),
            "tokens_per_second": self._percentiles(self._tps_samples),
            "e2e_latency_ms": self._percentiles(self._e2e_ms_samples),
            "eagle_accept_rate": self._percentiles(self._eagle_accept_samples) if self._eagle_accept_samples else None,
            "kv_hit_rate": self._percentiles(self._kv_hit_samples) if self._kv_hit_samples else None,
        }

    def prometheus_text(self) -> str:
        """Render Prometheus text format for scraping."""
        r = self.report()
        lines = [
            "# HELP aether_request_total Total inference requests",
            "# TYPE aether_request_total counter",
            f"aether_request_total {self._request_count}",
            "# HELP aether_error_total Total inference errors",
            "# TYPE aether_error_total counter",
            f"aether_error_total {self._error_count}",
        ]
        for metric, data in [("ttft_ms", r["ttft_ms"]), ("e2e_latency_ms", r["e2e_latency_ms"])]:
            lines.append(f"# HELP aether_{metric} Latency histogram")
            lines.append(f"# TYPE aether_{metric} gauge")
            for pct in ("p50", "p95", "p99", "mean"):
                lines.append(f'aether_{metric}{{quantile="{pct}"}} {data[pct]}')
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        self._ttft_samples.clear()
        self._tps_samples.clear()
        self._e2e_ms_samples.clear()
        self._eagle_accept_samples.clear()
        self._kv_hit_samples.clear()
        self._request_count = 0
        self._error_count = 0


# ---------------------------------------------------------------------------
# OTLP Exporter
# ---------------------------------------------------------------------------

class OTLPExporter:
    """
    Writes OTLP JSON trace files for import into Jaeger / Grafana Tempo.

    In production, swap write_to_file() with an HTTP POST to the OTLP collector endpoint.
    """

    def __init__(self, endpoint: str = "http://localhost:4318/v1/traces") -> None:
        self.endpoint = endpoint

    def export_to_file(self, tracer: AetherTracer, path: str | Path) -> Path:
        """Write OTLP JSON payload to a file for offline ingestion."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = tracer.export_otlp_json()
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def export_to_endpoint(
        self,
        tracer: AetherTracer,
        *,
        timeout_s: float = 5.0,
        headers: dict[str, str] | None = None,
    ) -> int:
        """POST the OTLP/HTTP JSON payload to a collector.

        The exporter intentionally uses the standard library so the CPU-only
        package does not acquire an optional telemetry dependency. A non-2xx
        response and transport failure are surfaced to the caller; telemetry
        is never reported as exported merely because a request was attempted.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        payload = json.dumps(tracer.export_otlp_json()).encode("utf-8")
        request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            request_headers.update({str(key): str(value) for key, value in headers.items()})
        request = Request(self.endpoint, data=payload, headers=request_headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - endpoint is explicit configuration
                status = int(response.status)
                if not 200 <= status < 300:
                    raise RuntimeError(f"OTLP collector returned HTTP {status}")
                return status
        except HTTPError as exc:
            raise RuntimeError(f"OTLP collector returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise ConnectionError(f"could not export OTLP trace to {self.endpoint}: {exc.reason}") from exc

    def export_config(self) -> dict[str, Any]:
        """Return the OTLP exporter configuration dict for the AEG manifest."""
        return {
            "exporter": "otlp",
            "endpoint": self.endpoint,
            "protocol": "http/json",
            "compression": "gzip",
            "headers": {},
            "timeout_ms": 5000,
        }
