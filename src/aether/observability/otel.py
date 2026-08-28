"""OpenTelemetry tracing and metrics for Aether Runtime.

Aether emits OTLP directly rather than through the OpenTelemetry SDK, because the
base runtime is deliberately dependency-light and provenance-grade artifacts must
be producible on a stock CPython install.  That choice only holds if the bytes on
the wire are actually the protocol, so this module implements the OTLP encoding
to specification:

* ``trace_id`` / ``span_id`` are lower-case hex of the correct width, and an
  absent parent is omitted rather than serialized as ``null``;
* attribute values are typed ``AnyValue`` maps — ``intValue`` as a *string*, per
  the protobuf JSON mapping of 64-bit integers — instead of everything being
  stringified into ``stringValue``, which silently destroys numeric aggregation
  in every backend;
* span events use ``timeUnixNano`` and ``KeyValue`` attributes;
* ``kind`` and ``status`` use the enum names OTLP defines, per span;
* ``Content-Encoding: gzip`` is only advertised when the body is really gzipped;
* the standard ``OTEL_*`` environment variables configure the exporter, so Aether
  drops into an existing collector deployment without bespoke configuration;
* W3C Trace Context ``traceparent`` is parsed and emitted, so an Aether span
  joins a trace that started upstream instead of orphaning it;
* sampling is trace-ID-ratio based, so the decision is consistent across
  processes that see the same trace.

When the real OpenTelemetry SDK *is* installed, :mod:`aether.observability.otel_sdk`
bridges these spans into it, so a deployment that already standardizes on the SDK
gets its processors, propagators and exporters. Install with
``pip install "aether-runtime[otel]"``.

References:
  * OpenTelemetry Protocol (OTLP) Specification v1.x — OTLP/HTTP, JSON encoding.
  * ``opentelemetry-proto`` ``trace/v1/trace.proto``, ``metrics/v1/metrics.proto``,
    ``common/v1/common.proto``.
  * W3C Trace Context, Recommendation 2021-11-23 — ``traceparent`` format.
  * OpenTelemetry SDK specification — ``TraceIdRatioBased`` sampler,
    ``OTEL_EXPORTER_OTLP_*`` and ``OTEL_TRACES_SAMPLER`` variables.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aether.core.constants import AETHER_VERSION

__all__ = [
    "Span",
    "SpanKind",
    "StatusCode",
    "TraceContext",
    "TraceIdRatioSampler",
    "AetherTracer",
    "MetricsCollector",
    "OTLPExporter",
    "encode_any_value",
    "encode_attributes",
]

_INT64_MIN, _INT64_MAX = -(2 ** 63), 2 ** 63 - 1


class SpanKind:
    """OTLP ``SpanKind`` enum names (``trace.proto``)."""

    INTERNAL = "SPAN_KIND_INTERNAL"
    SERVER = "SPAN_KIND_SERVER"
    CLIENT = "SPAN_KIND_CLIENT"
    PRODUCER = "SPAN_KIND_PRODUCER"
    CONSUMER = "SPAN_KIND_CONSUMER"


class StatusCode:
    """OTLP ``Status.StatusCode`` enum names."""

    UNSET = "STATUS_CODE_UNSET"
    OK = "STATUS_CODE_OK"
    ERROR = "STATUS_CODE_ERROR"


# ── AnyValue encoding ─────────────────────────────────────────────────────────

def encode_any_value(value: Any) -> dict[str, Any]:
    """Encode one Python value as an OTLP ``AnyValue``.

    The type matters: a latency recorded as ``{"stringValue": "45.2"}`` cannot be
    histogrammed, averaged or alerted on by any backend, and that is what a
    stringify-everything encoder produces.  Note that ``intValue`` is a JSON
    *string* — the protobuf JSON mapping represents 64-bit integers that way
    because a JSON number cannot carry them losslessly.
    """
    # bool first: in Python, True is an int.
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        if _INT64_MIN <= value <= _INT64_MAX:
            return {"intValue": str(value)}
        return {"stringValue": str(value)}
    if isinstance(value, float):
        # Infinity and NaN have no JSON representation; a backend rejecting the
        # whole payload over one metric is worse than losing that one value.
        if value != value or value in (float("inf"), float("-inf")):
            return {"stringValue": repr(value)}
        return {"doubleValue": value}
    if value is None:
        return {}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (bytes, bytearray)):
        import base64

        return {"bytesValue": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple, set)):
        return {"arrayValue": {"values": [encode_any_value(item) for item in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": encode_attributes(value)}}
    return {"stringValue": str(value)}


def encode_attributes(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode a mapping as an OTLP ``repeated KeyValue``."""
    return [
        {"key": str(key), "value": encode_any_value(value)}
        for key, value in attributes.items()
    ]


# ── W3C Trace Context ─────────────────────────────────────────────────────────

_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
_INVALID_TRACE_ID = "0" * 32
_INVALID_SPAN_ID = "0" * 16


@dataclass(frozen=True)
class TraceContext:
    """A parsed W3C ``traceparent``.

    Carrying this across a process boundary is what makes an Aether span a child
    of the caller's span instead of the root of an unrelated trace.
    """

    trace_id: str
    span_id: str
    sampled: bool = True
    trace_state: str = ""

    @classmethod
    def parse(cls, traceparent: str, trace_state: str = "") -> "TraceContext | None":
        """Parse a ``traceparent`` header, or return ``None`` if it is invalid.

        An all-zero trace or span ID is invalid per the specification and must not
        be adopted: doing so would join every unrelated request into one trace.
        """
        match = _TRACEPARENT.match(traceparent.strip().lower())
        if match is None:
            return None
        if match.group("version") == "ff":
            return None
        trace_id = match.group("trace_id")
        span_id = match.group("span_id")
        if trace_id == _INVALID_TRACE_ID or span_id == _INVALID_SPAN_ID:
            return None
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            sampled=bool(int(match.group("flags"), 16) & 0x01),
            trace_state=trace_state,
        )

    def to_header(self) -> str:
        """Render this context as a ``traceparent`` header value."""
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"


class TraceIdRatioSampler:
    """Trace-ID-ratio sampler, matching the OpenTelemetry SDK's definition.

    The decision is a function of the trace ID alone — its low 64 bits compared
    against ``ratio · 2⁶⁴`` — so every process that sees the same trace makes the
    same decision.  A sampler keyed on a local random number would produce traces
    with holes in them, which is worse than not sampling at all.
    """

    def __init__(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"sample ratio must be in [0, 1], got {ratio}")
        self.ratio = float(ratio)
        self._threshold = int(self.ratio * (2 ** 64))

    def should_sample(self, trace_id: str) -> bool:
        if self.ratio >= 1.0:
            return True
        if self.ratio <= 0.0:
            return False
        try:
            low64 = int(trace_id[-16:], 16)
        except ValueError:
            return False
        return low64 < self._threshold


# ── Span ──────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    """One OTLP span.

    ``trace_id`` is 32 hex characters and ``span_id`` is 16, as OTLP requires;
    they are generated from :mod:`random` seeded by the OS, not from UUIDs, so the
    full ID width is random rather than carrying UUID version and variant bits.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_ns: int
    end_time_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"  # "OK" | "ERROR" | "UNSET"
    events: list[dict[str, Any]] = field(default_factory=list)
    kind: str = SpanKind.INTERNAL
    trace_state: str = ""
    sampled: bool = True

    @property
    def duration_ms(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1_000_000

    @property
    def context(self) -> TraceContext:
        """This span's context, for propagation to a downstream service."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            sampled=self.sampled,
            trace_state=self.trace_state,
        )

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": name,
                "time_unix_nano": time.time_ns(),
                "attributes": dict(attributes or {}),
            }
        )

    def set_error(self, message: str) -> None:
        self.status = "ERROR"
        self.attributes["error.message"] = message
        # ``exception.message`` is the OTLP semantic-convention key; the event is
        # what a backend's error view actually reads.
        self.add_event("exception", {"exception.message": message})

    def end(self) -> None:
        self.end_time_ns = time.time_ns()

    def to_otlp_dict(self) -> dict[str, Any]:
        """Render this span in OTLP/JSON form."""
        payload: dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(self.end_time_ns or self.start_time_ns),
            "attributes": encode_attributes(self.attributes),
            "droppedAttributesCount": 0,
            "events": [
                {
                    "timeUnixNano": str(event["time_unix_nano"]),
                    "name": event["name"],
                    "attributes": encode_attributes(event["attributes"]),
                    "droppedAttributesCount": 0,
                }
                for event in self.events
            ],
            "droppedEventsCount": 0,
            "links": [],
            "droppedLinksCount": 0,
            "status": {
                "code": {
                    "OK": StatusCode.OK,
                    "ERROR": StatusCode.ERROR,
                }.get(self.status, StatusCode.UNSET)
            },
            "flags": 1 if self.sampled else 0,
        }
        # An absent parent is omitted. Emitting ``"parentSpanId": null`` is not a
        # valid OTLP string field and collectors reject the whole request.
        if self.parent_span_id:
            payload["parentSpanId"] = self.parent_span_id
        if self.trace_state:
            payload["traceState"] = self.trace_state
        if self.status == "ERROR":
            payload["status"]["message"] = str(
                self.attributes.get("error.message", "")
            )
        return payload


# ── Tracer ────────────────────────────────────────────────────────────────────

_ID_RANDOM = random.SystemRandom()


def _new_trace_id() -> str:
    return f"{_ID_RANDOM.getrandbits(128):032x}"


def _new_span_id() -> str:
    return f"{_ID_RANDOM.getrandbits(64):016x}"


def _resource_attributes_from_env() -> dict[str, Any]:
    """Parse ``OTEL_RESOURCE_ATTRIBUTES`` (``k=v,k=v``)."""
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    parsed: dict[str, Any] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key:
            parsed[key] = value.strip()
    return parsed


def _sample_ratio_from_env(default: float) -> float:
    """Honour ``OTEL_TRACES_SAMPLER`` / ``OTEL_TRACES_SAMPLER_ARG``."""
    sampler = os.environ.get("OTEL_TRACES_SAMPLER", "").strip().lower()
    if not sampler:
        return default
    if sampler in {"always_on", "parentbased_always_on"}:
        return 1.0
    if sampler in {"always_off", "parentbased_always_off"}:
        return 0.0
    if sampler in {"traceidratio", "parentbased_traceidratio"}:
        try:
            return max(0.0, min(1.0, float(os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0"))))
        except ValueError:
            return default
    return default


class AetherTracer:
    """OTLP tracer for Aether inference.

    Spans are created for every phase of request processing — prefill, decode,
    speculative draft/verify, reasoning budget, retrieval — and exported as OTLP.

    Retention is bounded.  An inference server runs for weeks, and an unbounded
    list of finished spans is a memory leak that only shows up in production; the
    oldest spans are dropped and the count of dropped spans is reported so the
    loss is visible rather than silent.
    """

    def __init__(
        self,
        service_name: str | None = None,
        *,
        sample_rate: float = 1.0,
        max_finished_spans: int = 4096,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.service_name = (
            service_name
            or os.environ.get("OTEL_SERVICE_NAME", "").strip()
            or "aether-runtime"
        )
        self.sampler = TraceIdRatioSampler(_sample_ratio_from_env(sample_rate))
        self.resource_attributes: dict[str, Any] = {
            "service.name": self.service_name,
            "service.version": AETHER_VERSION,
            "telemetry.sdk.name": "aether",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": AETHER_VERSION,
            **_resource_attributes_from_env(),
            **(resource_attributes or {}),
        }
        self._finished_spans: deque[Span] = deque(maxlen=max(1, int(max_finished_spans)))
        self._dropped_spans = 0
        self._unsampled_spans = 0

    # ── span lifecycle ────────────────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        *,
        kind: str = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        traceparent: str | None = None,
    ) -> Span:
        """Begin a span, optionally as a child of an upstream trace context.

        ``traceparent`` accepts a raw W3C header, so an HTTP handler can pass what
        it received without parsing it first.  A parent's sampling decision is
        respected rather than re-derived, which is what keeps a distributed trace
        whole.
        """
        if traceparent and parent is None:
            parent = TraceContext.parse(traceparent)
        if parent is not None:
            trace_id = trace_id or parent.trace_id
            parent_span_id = parent_span_id or parent.span_id
        resolved_trace_id = trace_id or _new_trace_id()
        sampled = (
            parent.sampled
            if parent is not None
            else self.sampler.should_sample(resolved_trace_id)
        )
        if not sampled:
            self._unsampled_spans += 1
        return Span(
            name=name,
            trace_id=resolved_trace_id,
            span_id=_new_span_id(),
            parent_span_id=parent_span_id,
            start_time_ns=time.time_ns(),
            attributes=dict(attributes or {}),
            kind=kind,
            trace_state=parent.trace_state if parent is not None else "",
            sampled=sampled,
        )

    def finish_span(self, span: Span, attributes: dict[str, Any] | None = None) -> Span:
        """End a span and retain it for export if it was sampled."""
        span.end()
        if attributes:
            span.attributes.update(attributes)
        self._retain(span)
        return span

    def _retain(self, span: Span) -> None:
        if not span.sampled:
            return
        if len(self._finished_spans) == self._finished_spans.maxlen:
            self._dropped_spans += 1
        self._finished_spans.append(span)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        *,
        kind: str = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        traceparent: str | None = None,
    ) -> Iterator[Span]:
        """Time a block as a span, recording an exception as a span error.

        The span is finished on the way out either way; a span abandoned by an
        exception would otherwise never be exported, hiding exactly the requests
        an operator most needs to see.
        """
        active = self.start_span(
            name, attributes=attributes, kind=kind, parent=parent, traceparent=traceparent
        )
        try:
            yield active
        except BaseException as exc:
            active.set_error(f"{type(exc).__name__}: {exc}")
            self.finish_span(active)
            raise
        else:
            self.finish_span(active)

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
        *,
        traceparent: str | None = None,
    ) -> Span:
        """Record a complete inference request as a single server span."""
        span = self.start_span(
            name="aether.inference",
            kind=SpanKind.SERVER,
            traceparent=traceparent,
            attributes={
                "request_id": request_id,
                "model_id": model_id,
                "prompt_tokens": int(prompt_tokens),
                "generated_tokens": int(generated_tokens),
                "tokens_per_second": round(
                    generated_tokens / max(total_ms / 1000, 1e-6), 2
                ),
                "ttft_ms": round(float(ttft_ms), 3),
                "total_latency_ms": round(float(total_ms), 3),
                "adapter_id": adapter_id or "base",
            },
        )
        # Callers that have measured wall-clock boundaries can provide them.
        # Without boundaries this helper records an instantaneous event; it
        # never backdates timestamps to manufacture latency.
        span.start_time_ns = actual_start_time_ns or time.time_ns()
        span.end_time_ns = actual_end_time_ns or span.start_time_ns
        span.attributes["duration_source"] = (
            "measured" if actual_start_time_ns and actual_end_time_ns else "event_only"
        )
        span.add_event(
            "prefill_complete", {"ttft_ms": ttft_ms, "prompt_tokens": prompt_tokens}
        )
        span.add_event("decode_complete", {"generated_tokens": generated_tokens})
        self._retain(span)
        return span

    # ── export ────────────────────────────────────────────────────────────────

    def get_finished_spans(self) -> list[Span]:
        return list(self._finished_spans)

    @property
    def dropped_spans(self) -> int:
        """Spans evicted by the retention bound, and therefore never exported."""
        return self._dropped_spans

    @property
    def unsampled_spans(self) -> int:
        """Spans the sampler declined to retain."""
        return self._unsampled_spans

    def export_otlp_json(self) -> dict[str, Any]:
        """Render every retained span as an OTLP/JSON ``ExportTraceServiceRequest``."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": encode_attributes(self.resource_attributes),
                        "droppedAttributesCount": 0,
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "aether.observability.otel",
                                "version": AETHER_VERSION,
                                "attributes": [],
                            },
                            "spans": [
                                span.to_otlp_dict() for span in self._finished_spans
                            ],
                        }
                    ],
                }
            ]
        }

    def clear(self) -> None:
        self._finished_spans.clear()


# ── Metrics ───────────────────────────────────────────────────────────────────

_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
    1000.0, 2500.0, 5000.0, 10000.0,
)
"""Explicit histogram bounds, in milliseconds.

Chosen to straddle the range inference latencies actually occupy — sub-millisecond
cache hits through ten-second long-context prefills — because a histogram whose
buckets all fall on one side of the data carries no information.
"""


class MetricsCollector:
    """Latency and throughput metrics, as percentiles and as OTLP histograms.

    Percentiles are computed from retained samples with the standard library, so
    the report is exact for the window rather than an estimate from bucket
    midpoints.  The OTLP export additionally emits real explicit-bucket histograms,
    which is what lets a collector aggregate across replicas — percentiles cannot
    be averaged, and a fleet-wide P99 computed from per-replica P99s is wrong.
    """

    def __init__(self, *, max_samples: int = 100_000) -> None:
        bound = max(1, int(max_samples))
        self._ttft_samples: deque[float] = deque(maxlen=bound)
        self._tps_samples: deque[float] = deque(maxlen=bound)
        self._e2e_ms_samples: deque[float] = deque(maxlen=bound)
        self._eagle_accept_samples: deque[float] = deque(maxlen=bound)
        self._kv_hit_samples: deque[float] = deque(maxlen=bound)
        self._request_count: int = 0
        self._error_count: int = 0
        self._start_time_ns: int = time.time_ns()

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
        self._ttft_samples.append(float(ttft_ms))
        self._tps_samples.append(float(tokens_per_second))
        self._e2e_ms_samples.append(float(e2e_latency_ms))
        if eagle_accept_rate > 0:
            self._eagle_accept_samples.append(float(eagle_accept_rate))
        if kv_hit_rate > 0:
            self._kv_hit_samples.append(float(kv_hit_rate))
        if is_error:
            self._error_count += 1

    def _percentiles(self, samples: "deque[float] | list[float]") -> dict[str, float]:
        values = list(samples)
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        if len(values) == 1:
            single = values[0]
            return {
                "p50": single, "p95": single, "p99": single,
                "mean": single, "min": single, "max": single,
            }
        ordered = sorted(values)
        quantiles = statistics.quantiles(ordered, n=100)
        return {
            "p50": round(quantiles[49], 3),
            "p95": round(quantiles[94], 3),
            "p99": round(quantiles[98], 3),
            "mean": round(statistics.fmean(ordered), 3),
            "min": round(ordered[0], 3),
            "max": round(ordered[-1], 3),
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
            "eagle_accept_rate": (
                self._percentiles(self._eagle_accept_samples)
                if self._eagle_accept_samples else None
            ),
            "kv_hit_rate": (
                self._percentiles(self._kv_hit_samples)
                if self._kv_hit_samples else None
            ),
        }

    def prometheus_text(self) -> str:
        """Render Prometheus text format for scraping."""
        report = self.report()
        lines = [
            "# HELP aether_request_total Total inference requests",
            "# TYPE aether_request_total counter",
            f"aether_request_total {self._request_count}",
            "# HELP aether_error_total Total inference errors",
            "# TYPE aether_error_total counter",
            f"aether_error_total {self._error_count}",
        ]
        for metric, data in (
            ("ttft_ms", report["ttft_ms"]),
            ("e2e_latency_ms", report["e2e_latency_ms"]),
            ("tokens_per_second", report["tokens_per_second"]),
        ):
            lines.append(f"# HELP aether_{metric} Latency histogram")
            lines.append(f"# TYPE aether_{metric} gauge")
            for percentile in ("p50", "p95", "p99", "mean"):
                lines.append(f'aether_{metric}{{quantile="{percentile}"}} {data[percentile]}')
        return "\n".join(lines) + "\n"

    @staticmethod
    def _histogram_point(
        samples: list[float], start_ns: int, now_ns: int
    ) -> dict[str, Any]:
        """Bucket samples into an OTLP ``HistogramDataPoint``.

        ``bucketCounts`` has one more entry than ``explicitBounds``: the trailing
        bucket is the ``(last_bound, +∞)`` overflow. Omitting it is the usual bug
        and makes a collector reject the point.
        """
        counts = [0] * (len(_LATENCY_BUCKETS_MS) + 1)
        for value in samples:
            index = len(_LATENCY_BUCKETS_MS)
            for position, bound in enumerate(_LATENCY_BUCKETS_MS):
                if value <= bound:
                    index = position
                    break
            counts[index] += 1
        point: dict[str, Any] = {
            "startTimeUnixNano": str(start_ns),
            "timeUnixNano": str(now_ns),
            "count": str(len(samples)),
            "sum": float(sum(samples)),
            "bucketCounts": [str(count) for count in counts],
            "explicitBounds": list(_LATENCY_BUCKETS_MS),
            "attributes": [],
        }
        if samples:
            point["min"] = float(min(samples))
            point["max"] = float(max(samples))
        return point

    def export_otlp_metrics_json(
        self,
        *,
        service_name: str = "aether-runtime",
        resource_attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render metrics as an OTLP/JSON ``ExportMetricsServiceRequest``.

        Temporality is ``CUMULATIVE`` (2) because the counters and the sample
        windows here are cumulative since construction. Declaring ``DELTA`` while
        sending cumulative values is the classic mislabel, and it makes a
        collector's rate computations nonsense rather than merely imprecise.
        """
        now_ns = time.time_ns()
        attributes = {
            "service.name": service_name,
            "service.version": AETHER_VERSION,
            **_resource_attributes_from_env(),
            **(resource_attributes or {}),
        }
        metrics: list[dict[str, Any]] = [
            {
                "name": "aether.requests",
                "unit": "{request}",
                "description": "Inference requests handled",
                "sum": {
                    "dataPoints": [
                        {
                            "startTimeUnixNano": str(self._start_time_ns),
                            "timeUnixNano": str(now_ns),
                            "asInt": str(self._request_count),
                            "attributes": [],
                        }
                    ],
                    "aggregationTemporality": 2,
                    "isMonotonic": True,
                },
            },
            {
                "name": "aether.errors",
                "unit": "{request}",
                "description": "Inference requests that failed",
                "sum": {
                    "dataPoints": [
                        {
                            "startTimeUnixNano": str(self._start_time_ns),
                            "timeUnixNano": str(now_ns),
                            "asInt": str(self._error_count),
                            "attributes": [],
                        }
                    ],
                    "aggregationTemporality": 2,
                    "isMonotonic": True,
                },
            },
        ]
        for name, unit, samples in (
            ("aether.ttft", "ms", self._ttft_samples),
            ("aether.request.duration", "ms", self._e2e_ms_samples),
        ):
            metrics.append(
                {
                    "name": name,
                    "unit": unit,
                    "description": f"Distribution of {name}",
                    "histogram": {
                        "dataPoints": [
                            self._histogram_point(
                                list(samples), self._start_time_ns, now_ns
                            )
                        ],
                        "aggregationTemporality": 2,
                    },
                }
            )
        return {
            "resourceMetrics": [
                {
                    "resource": {"attributes": encode_attributes(attributes)},
                    "scopeMetrics": [
                        {
                            "scope": {
                                "name": "aether.observability.otel",
                                "version": AETHER_VERSION,
                            },
                            "metrics": metrics,
                        }
                    ],
                }
            ]
        }

    def reset(self) -> None:
        self._ttft_samples.clear()
        self._tps_samples.clear()
        self._e2e_ms_samples.clear()
        self._eagle_accept_samples.clear()
        self._kv_hit_samples.clear()
        self._request_count = 0
        self._error_count = 0
        self._start_time_ns = time.time_ns()


# ── OTLP/HTTP exporter ────────────────────────────────────────────────────────

_DEFAULT_ENDPOINT = "http://localhost:4318"
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


def _headers_from_env() -> dict[str, str]:
    """Parse ``OTEL_EXPORTER_OTLP_HEADERS`` (``k=v,k=v``).

    This is where collector authentication lives in every standard deployment, so
    an exporter that ignores it cannot be pointed at a hosted backend at all.
    """
    raw = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS") or os.environ.get(
        "OTEL_EXPORTER_OTLP_HEADERS", ""
    )
    headers: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key:
            headers[key] = value.strip()
    return headers


def _endpoint_from_env(signal: str) -> str | None:
    """Resolve the endpoint for ``signal`` from the standard variables.

    The per-signal variable is a *complete* URL; the generic one is a base to
    which the signal path is appended. Conflating the two is a common bug that
    produces ``/v1/traces/v1/traces``.
    """
    specific = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT", "").strip()
    if specific:
        return specific
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if base:
        return f"{base.rstrip('/')}/v1/{signal}"
    return None


class OTLPExporter:
    """Sends OTLP/JSON over HTTP to a collector, or writes it to a file.

    Configuration follows the OpenTelemetry environment variables so this drops
    into an existing collector deployment: ``OTEL_EXPORTER_OTLP_ENDPOINT``,
    ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
    ``OTEL_EXPORTER_OTLP_TIMEOUT`` (milliseconds) and
    ``OTEL_EXPORTER_OTLP_COMPRESSION``.

    Failures are raised, never swallowed. Telemetry that silently stops arriving
    is worse than telemetry that fails loudly, because the absence of spans is
    indistinguishable from the absence of traffic.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
        compression: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.endpoint = (
            endpoint
            or _endpoint_from_env("traces")
            or f"{_DEFAULT_ENDPOINT}/v1/traces"
        )
        self.metrics_endpoint = (
            _endpoint_from_env("metrics") or f"{_DEFAULT_ENDPOINT}/v1/metrics"
        )
        self.headers = {**_headers_from_env(), **(headers or {})}
        if timeout_s is not None:
            self.timeout_s = float(timeout_s)
        else:
            try:
                configured = os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT", "10000")
                self.timeout_s = float(configured) / 1000.0
            except ValueError:
                self.timeout_s = 10.0
        requested = (
            compression
            if compression is not None
            else os.environ.get("OTEL_EXPORTER_OTLP_COMPRESSION", "gzip")
        )
        self.compression = "gzip" if str(requested).strip().lower() == "gzip" else "none"
        self.max_retries = max(0, int(max_retries))

    # ── file export ───────────────────────────────────────────────────────────

    def export_to_file(self, tracer: AetherTracer, path: str | Path) -> Path:
        """Write the OTLP payload to a file for offline ingestion."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tracer.export_otlp_json(), indent=2), encoding="utf-8")
        return out

    # ── HTTP export ───────────────────────────────────────────────────────────

    def _post(self, url: str, payload: dict[str, Any]) -> int:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **{str(key): str(value) for key, value in self.headers.items()},
        }
        if self.compression == "gzip":
            # Only claim gzip when the body is really compressed. Advertising an
            # encoding that was not applied makes a collector fail to decode a
            # payload that is otherwise perfectly valid.
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = Request(url, data=body, headers=headers, method="POST")
            try:
                # noqa: S310 - the endpoint is explicit configuration, not input
                with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                    status = int(response.status)
                    if 200 <= status < 300:
                        return status
                    raise RuntimeError(f"OTLP collector returned HTTP {status}")
            except HTTPError as exc:
                last_error = RuntimeError(f"OTLP collector returned HTTP {exc.code}")
                if exc.code not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise last_error from exc
                self._sleep_before_retry(attempt, exc.headers.get("Retry-After"))
            except URLError as exc:
                last_error = ConnectionError(
                    f"could not export OTLP payload to {url}: {exc.reason}"
                )
                if attempt == self.max_retries:
                    raise last_error from exc
                self._sleep_before_retry(attempt, None)
        raise last_error or RuntimeError("OTLP export failed")

    @staticmethod
    def _sleep_before_retry(attempt: int, retry_after: str | None) -> None:
        """Back off exponentially, honouring ``Retry-After`` when the server sets it.

        Jitter is applied so a fleet of replicas that all lose the collector at
        once does not synchronize its retries into a thundering herd.
        """
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        delay = min(2.0 ** attempt, 8.0)
        time.sleep(delay * (0.5 + random.random() * 0.5))

    def export_to_endpoint(
        self,
        tracer: AetherTracer,
        *,
        timeout_s: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> int:
        """POST the retained spans to the traces endpoint.

        Returns the HTTP status. Raises on a non-2xx response or a transport
        failure; telemetry is never reported as exported merely because a request
        was attempted.
        """
        if timeout_s is not None:
            if timeout_s <= 0:
                raise ValueError("timeout_s must be positive")
            self.timeout_s = float(timeout_s)
        if headers:
            self.headers.update({str(k): str(v) for k, v in headers.items()})
        return self._post(self.endpoint, tracer.export_otlp_json())

    def export_metrics_to_endpoint(
        self,
        collector: MetricsCollector,
        *,
        service_name: str = "aether-runtime",
    ) -> int:
        """POST metrics to the metrics endpoint as OTLP/JSON."""
        return self._post(
            self.metrics_endpoint,
            collector.export_otlp_metrics_json(service_name=service_name),
        )

    def export_config(self) -> dict[str, Any]:
        """Return the exporter configuration recorded in the AEG manifest."""
        return {
            "exporter": "otlp",
            "endpoint": self.endpoint,
            "metrics_endpoint": self.metrics_endpoint,
            "protocol": "http/json",
            "compression": self.compression,
            "headers": {key: "<redacted>" for key in self.headers},
            "timeout_ms": int(self.timeout_s * 1000),
            "max_retries": self.max_retries,
        }
