"""OTLP encoding conformance and trace-context propagation.

The failure these tests exist to prevent is telemetry that *looks* fine — a JSON
payload with plausible field names — but that a collector either rejects or
silently degrades. Every assertion here corresponds to a concrete way the previous
custom encoder diverged from OTLP:

* every attribute stringified, destroying numeric aggregation;
* ``parentSpanId: null``, which is not a valid protobuf-JSON string field;
* span events keyed ``timestamp_ns`` with raw attribute dicts;
* every span reported as ``SPAN_KIND_SERVER``;
* ``Content-Encoding: gzip`` advertised over an uncompressed body;
* no trace-context propagation, so a distributed trace broke at Aether.
"""

from __future__ import annotations

import gzip
import json

import pytest

from aether.observability.otel import (
    AetherTracer,
    MetricsCollector,
    OTLPExporter,
    SpanKind,
    StatusCode,
    TraceContext,
    TraceIdRatioSampler,
    encode_any_value,
    encode_attributes,
)


# ── AnyValue typing ───────────────────────────────────────────────────────────

def test_int_is_encoded_as_a_string_int_value() -> None:
    """protobuf JSON represents 64-bit integers as strings, not numbers."""
    assert encode_any_value(128) == {"intValue": "128"}


def test_float_is_encoded_as_a_double_not_a_string() -> None:
    """A latency in stringValue cannot be histogrammed by any backend."""
    assert encode_any_value(45.2) == {"doubleValue": 45.2}


def test_bool_is_not_encoded_as_an_int() -> None:
    """In Python True is an int; encoding it as one loses the type."""
    assert encode_any_value(True) == {"boolValue": True}
    assert encode_any_value(False) == {"boolValue": False}


def test_string_bytes_list_and_map_use_their_own_any_value_kinds() -> None:
    assert encode_any_value("qwen3") == {"stringValue": "qwen3"}
    assert encode_any_value(b"\x00\x01") == {"bytesValue": "AAE="}
    assert encode_any_value([1, 2]) == {
        "arrayValue": {"values": [{"intValue": "1"}, {"intValue": "2"}]}
    }
    assert encode_any_value({"k": "v"}) == {
        "kvlistValue": {"values": [{"key": "k", "value": {"stringValue": "v"}}]}
    }


def test_non_finite_floats_do_not_produce_invalid_json() -> None:
    """NaN and Infinity have no JSON form; one bad value must not void a batch."""
    for value in (float("nan"), float("inf"), float("-inf")):
        encoded = encode_any_value(value)
        assert "doubleValue" not in encoded
        json.dumps(encoded)


def test_int64_overflow_falls_back_to_a_string() -> None:
    assert encode_any_value(2 ** 64) == {"stringValue": str(2 ** 64)}


def test_encode_attributes_produces_key_value_pairs() -> None:
    assert encode_attributes({"a": 1}) == [{"key": "a", "value": {"intValue": "1"}}]


# ── Span encoding ─────────────────────────────────────────────────────────────

def test_ids_have_the_widths_otlp_requires() -> None:
    tracer = AetherTracer("test")
    span = tracer.start_span("op")
    assert len(span.trace_id) == 32 and int(span.trace_id, 16) >= 0
    assert len(span.span_id) == 16 and int(span.span_id, 16) >= 0


def test_root_span_omits_parent_span_id_rather_than_nulling_it() -> None:
    """``"parentSpanId": null`` is not a valid string field; collectors reject it."""
    tracer = AetherTracer("test")
    payload = tracer.finish_span(tracer.start_span("root")).to_otlp_dict()
    assert "parentSpanId" not in payload
    json.dumps(payload)


def test_child_span_carries_its_parent() -> None:
    tracer = AetherTracer("test")
    parent = tracer.finish_span(tracer.start_span("parent"))
    child = tracer.start_span("child", trace_id=parent.trace_id, parent_span_id=parent.span_id)
    payload = tracer.finish_span(child).to_otlp_dict()
    assert payload["parentSpanId"] == parent.span_id
    assert payload["traceId"] == parent.trace_id


def test_span_kind_is_per_span_not_always_server() -> None:
    tracer = AetherTracer("test")
    internal = tracer.finish_span(tracer.start_span("work"))
    request = tracer.trace_request("r", 1, 1, 1.0, 2.0, "m")
    assert internal.to_otlp_dict()["kind"] == SpanKind.INTERNAL
    assert request.to_otlp_dict()["kind"] == SpanKind.SERVER


def test_events_use_time_unix_nano_and_key_value_attributes() -> None:
    tracer = AetherTracer("test")
    span = tracer.start_span("op")
    span.add_event("prefill_complete", {"prompt_tokens": 128})
    event = tracer.finish_span(span).to_otlp_dict()["events"][0]
    assert set(event) == {"timeUnixNano", "name", "attributes", "droppedAttributesCount"}
    assert isinstance(event["timeUnixNano"], str)
    assert event["attributes"] == [
        {"key": "prompt_tokens", "value": {"intValue": "128"}}
    ]


def test_timestamps_are_strings() -> None:
    """Nanosecond timestamps exceed the range a JSON number carries exactly."""
    tracer = AetherTracer("test")
    payload = tracer.finish_span(tracer.start_span("op")).to_otlp_dict()
    assert isinstance(payload["startTimeUnixNano"], str)
    assert isinstance(payload["endTimeUnixNano"], str)
    assert int(payload["endTimeUnixNano"]) >= int(payload["startTimeUnixNano"])


def test_error_status_uses_the_otlp_enum_and_carries_a_message() -> None:
    tracer = AetherTracer("test")
    span = tracer.start_span("op")
    span.set_error("CUDA OOM")
    payload = tracer.finish_span(span).to_otlp_dict()
    assert payload["status"]["code"] == StatusCode.ERROR
    assert payload["status"]["message"] == "CUDA OOM"
    assert any(event["name"] == "exception" for event in payload["events"])


def test_ok_status_uses_the_otlp_enum() -> None:
    tracer = AetherTracer("test")
    payload = tracer.finish_span(tracer.start_span("op")).to_otlp_dict()
    assert payload["status"]["code"] == StatusCode.OK


def test_request_span_attributes_keep_their_types_through_encoding() -> None:
    tracer = AetherTracer("test")
    span = tracer.trace_request("r", 128, 256, 45.2, 1200.0, "qwen3")
    values = {
        entry["key"]: entry["value"] for entry in span.to_otlp_dict()["attributes"]
    }
    assert values["prompt_tokens"] == {"intValue": "128"}
    assert values["ttft_ms"] == {"doubleValue": 45.2}
    assert values["model_id"] == {"stringValue": "qwen3"}


def test_export_payload_has_the_otlp_envelope_shape() -> None:
    tracer = AetherTracer("aether-test")
    tracer.trace_request("r", 1, 1, 1.0, 2.0, "m")
    payload = tracer.export_otlp_json()
    resource_spans = payload["resourceSpans"]
    assert len(resource_spans) == 1
    scope = resource_spans[0]["scopeSpans"][0]["scope"]
    assert scope["name"] == "aether.observability.otel"
    attributes = {
        entry["key"]: entry["value"]["stringValue"]
        for entry in resource_spans[0]["resource"]["attributes"]
    }
    assert attributes["service.name"] == "aether-test"
    assert attributes["telemetry.sdk.language"] == "python"


# ── W3C Trace Context ─────────────────────────────────────────────────────────

def test_traceparent_round_trip() -> None:
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    context = TraceContext.parse(header)
    assert context is not None
    assert context.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert context.span_id == "00f067aa0ba902b7"
    assert context.sampled is True
    assert context.to_header() == header


@pytest.mark.parametrize(
    "header",
    [
        "",
        "not-a-traceparent",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",           # short
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",        # version ff
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",        # zero trace
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",        # zero span
    ],
)
def test_invalid_traceparent_is_rejected(header: str) -> None:
    """An all-zero ID would join every unrelated request into one trace."""
    assert TraceContext.parse(header) is None


def test_span_started_from_a_traceparent_joins_the_upstream_trace() -> None:
    tracer = AetherTracer("test")
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    span = tracer.start_span("child", traceparent=header)
    assert span.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert span.parent_span_id == "00f067aa0ba902b7"


def test_unsampled_parent_decision_is_respected_not_recomputed() -> None:
    """Re-deciding downstream produces traces with holes in them."""
    tracer = AetherTracer("test", sample_rate=1.0)
    span = tracer.start_span(
        "child", traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
    )
    assert span.sampled is False
    tracer.finish_span(span)
    assert tracer.get_finished_spans() == []


# ── Sampling ──────────────────────────────────────────────────────────────────

def test_ratio_sampler_decision_depends_only_on_the_trace_id() -> None:
    """Consistency across processes is the whole point of an ID-based sampler."""
    sampler = TraceIdRatioSampler(0.5)
    trace_id = "0" * 16 + "1" * 16
    assert sampler.should_sample(trace_id) == sampler.should_sample(trace_id)


def test_ratio_sampler_endpoints() -> None:
    assert TraceIdRatioSampler(1.0).should_sample("f" * 32) is True
    assert TraceIdRatioSampler(0.0).should_sample("0" * 32) is False


def test_ratio_sampler_approximates_the_requested_ratio() -> None:
    sampler = TraceIdRatioSampler(0.25)
    kept = sum(
        sampler.should_sample(f"{value:032x}") for value in range(0, 2 ** 64, 2 ** 64 // 2000)
    )
    assert 0.20 <= kept / 2000 <= 0.30


def test_invalid_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TraceIdRatioSampler(1.5)


def test_sample_rate_is_accepted_by_the_tracer_constructor() -> None:
    """The README documents this keyword; it must exist."""
    tracer = AetherTracer("aether-prod", sample_rate=0.01)
    assert tracer.sampler.ratio == pytest.approx(0.01)


def test_environment_sampler_overrides_the_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.0")
    assert AetherTracer("test", sample_rate=1.0).sampler.ratio == 0.0


def test_service_name_and_resource_attributes_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=prod,region=eu")
    tracer = AetherTracer()
    assert tracer.service_name == "from-env"
    assert tracer.resource_attributes["deployment.environment"] == "prod"
    assert tracer.resource_attributes["region"] == "eu"


# ── Retention ─────────────────────────────────────────────────────────────────

def test_span_retention_is_bounded_and_the_loss_is_visible() -> None:
    """An unbounded span list is a leak that only appears in production."""
    tracer = AetherTracer("test", max_finished_spans=8)
    for index in range(20):
        tracer.finish_span(tracer.start_span(f"op{index}"))
    assert len(tracer.get_finished_spans()) == 8
    assert tracer.dropped_spans == 12


def test_span_context_manager_records_an_exception_and_still_exports() -> None:
    tracer = AetherTracer("test")
    with pytest.raises(ValueError):
        with tracer.span("failing"):
            raise ValueError("boom")
    spans = tracer.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status == "ERROR"
    assert "boom" in spans[0].attributes["error.message"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_export_has_the_otlp_metrics_envelope() -> None:
    collector = MetricsCollector()
    for value in (5.0, 40.0, 300.0, 4000.0):
        collector.record(ttft_ms=value, tokens_per_second=100.0, e2e_latency_ms=value * 2)
    payload = collector.export_otlp_metrics_json(service_name="aether-test")
    metrics = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    names = {metric["name"] for metric in metrics}
    assert {"aether.requests", "aether.errors", "aether.ttft"} <= names
    json.dumps(payload)


def test_histogram_bucket_counts_have_one_more_entry_than_bounds() -> None:
    """The trailing bucket is the (last_bound, +inf) overflow; omitting it is invalid."""
    collector = MetricsCollector()
    collector.record(ttft_ms=99999.0, tokens_per_second=1.0, e2e_latency_ms=1.0)
    metrics = collector.export_otlp_metrics_json()["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    histogram = next(m for m in metrics if m["name"] == "aether.ttft")
    point = histogram["histogram"]["dataPoints"][0]
    assert len(point["bucketCounts"]) == len(point["explicitBounds"]) + 1
    # A value past the last bound lands in the overflow bucket, not nowhere.
    assert point["bucketCounts"][-1] == "1"
    assert point["count"] == "1"


def test_histogram_temporality_matches_the_values_sent() -> None:
    """Labelling cumulative values DELTA makes a collector's rates nonsense."""
    collector = MetricsCollector()
    collector.record(1.0, 1.0, 1.0)
    metrics = collector.export_otlp_metrics_json()["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    for metric in metrics:
        body = metric.get("histogram") or metric.get("sum")
        assert body["aggregationTemporality"] == 2  # CUMULATIVE


def test_counter_values_are_string_ints() -> None:
    collector = MetricsCollector()
    collector.record(1.0, 1.0, 1.0, is_error=True)
    metrics = collector.export_otlp_metrics_json()["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    errors = next(m for m in metrics if m["name"] == "aether.errors")
    assert errors["sum"]["dataPoints"][0]["asInt"] == "1"
    assert errors["sum"]["isMonotonic"] is True


def test_metric_sample_retention_is_bounded() -> None:
    collector = MetricsCollector(max_samples=10)
    for index in range(100):
        collector.record(float(index), 1.0, 1.0)
    assert collector.report()["request_count"] == 100
    assert collector.report()["ttft_ms"]["min"] == 90.0


# ── Exporter configuration ────────────────────────────────────────────────────

def test_endpoint_comes_from_the_standard_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    exporter = OTLPExporter()
    assert exporter.endpoint == "http://collector:4318/v1/traces"
    assert exporter.metrics_endpoint == "http://collector:4318/v1/metrics"


def test_signal_specific_endpoint_is_used_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-signal variable is a full URL; appending to it duplicates the path."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://h/otlp/v1/traces")
    assert OTLPExporter().endpoint == "https://h/otlp/v1/traces"


def test_headers_and_timeout_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer t,x-tenant=acme")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "2500")
    exporter = OTLPExporter()
    assert exporter.headers["authorization"] == "Bearer t"
    assert exporter.headers["x-tenant"] == "acme"
    assert exporter.timeout_s == pytest.approx(2.5)


def test_export_config_does_not_leak_header_values() -> None:
    exporter = OTLPExporter("http://c/v1/traces", headers={"authorization": "Bearer secret"})
    config = exporter.export_config()
    assert config["headers"] == {"authorization": "<redacted>"}
    assert "secret" not in json.dumps(config)


def test_compression_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_COMPRESSION", "none")
    assert OTLPExporter().compression == "none"


# ── HTTP behaviour ────────────────────────────────────────────────────────────

class _Collector:
    """A stub OTLP receiver that records what it was actually sent."""

    def __init__(self, statuses: list[int]) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading

        self.bodies: list[bytes] = []
        self.encodings: list[str] = []
        self.headers: list[dict[str, str]] = []
        remaining = list(statuses)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                encoding = self.headers.get("Content-Encoding", "")
                outer.encodings.append(encoding)
                # HTTP header names are case-insensitive and urllib title-cases
                # what it sends, so compare on a lower-cased copy.
                outer.headers.append(
                    {key.lower(): value for key, value in self.headers.items()}
                )
                outer.bodies.append(gzip.decompress(raw) if encoding == "gzip" else raw)
                self.send_response(remaining.pop(0) if remaining else 200)
                self.end_headers()

            def log_message(self, *_args: object) -> None:
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1/traces"

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2)
        self._server.server_close()


def test_gzip_body_is_really_compressed_when_advertised() -> None:
    """Claiming an encoding that was not applied breaks a valid payload."""
    collector = _Collector([200])
    try:
        tracer = AetherTracer("test")
        tracer.trace_request("r", 4, 2, 1.0, 3.0, "local")
        assert OTLPExporter(collector.url).export_to_endpoint(tracer) == 200
        assert collector.encodings == ["gzip"]
        assert json.loads(collector.bodies[0])["resourceSpans"]
    finally:
        collector.close()


def test_configured_headers_reach_the_collector() -> None:
    collector = _Collector([200])
    try:
        tracer = AetherTracer("test")
        tracer.finish_span(tracer.start_span("op"))
        OTLPExporter(collector.url, headers={"x-tenant": "acme"}).export_to_endpoint(tracer)
        assert collector.headers[0]["x-tenant"] == "acme"
        assert collector.headers[0]["content-type"] == "application/json"
    finally:
        collector.close()


def test_retryable_status_is_retried_then_succeeds() -> None:
    collector = _Collector([503, 200])
    try:
        tracer = AetherTracer("test")
        tracer.finish_span(tracer.start_span("op"))
        exporter = OTLPExporter(collector.url, max_retries=2)
        exporter._sleep_before_retry = staticmethod(lambda *_: None)  # type: ignore[method-assign]
        assert exporter.export_to_endpoint(tracer) == 200
        assert len(collector.bodies) == 2
    finally:
        collector.close()


def test_non_retryable_status_raises_immediately() -> None:
    collector = _Collector([400, 200])
    try:
        tracer = AetherTracer("test")
        tracer.finish_span(tracer.start_span("op"))
        with pytest.raises(RuntimeError, match="HTTP 400"):
            OTLPExporter(collector.url, max_retries=3).export_to_endpoint(tracer)
        assert len(collector.bodies) == 1
    finally:
        collector.close()


def test_unreachable_collector_raises_rather_than_reporting_success() -> None:
    """Telemetry silently not arriving looks identical to no traffic."""
    tracer = AetherTracer("test")
    tracer.finish_span(tracer.start_span("op"))
    exporter = OTLPExporter("http://127.0.0.1:1/v1/traces", max_retries=0, timeout_s=0.5)
    with pytest.raises(ConnectionError):
        exporter.export_to_endpoint(tracer)


def test_metrics_are_posted_to_the_metrics_endpoint() -> None:
    collector = _Collector([200])
    try:
        metrics = MetricsCollector()
        metrics.record(10.0, 100.0, 200.0)
        exporter = OTLPExporter(collector.url)
        exporter.metrics_endpoint = collector.url.replace("/v1/traces", "/v1/metrics")
        assert exporter.export_metrics_to_endpoint(metrics) == 200
        assert json.loads(collector.bodies[0])["resourceMetrics"]
    finally:
        collector.close()


# ── SDK bridge ────────────────────────────────────────────────────────────────

def test_sdk_bridge_reports_availability_without_raising_on_import() -> None:
    from aether.observability import otel_sdk

    assert isinstance(otel_sdk.is_available(), bool)


def test_sdk_bridge_preserves_aether_trace_ids() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from aether.observability.otel_sdk import OpenTelemetryBridge

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))

    tracer = AetherTracer("bridged")
    span = tracer.trace_request("r", 8, 4, 2.0, 6.0, "qwen3")
    OpenTelemetryBridge("bridged", tracer_provider=provider).emit_all(
        tracer.get_finished_spans()
    )
    exported = memory.get_finished_spans()
    assert len(exported) == 1
    assert f"{exported[0].context.trace_id:032x}" == span.trace_id
    assert exported[0].name == "aether.inference"
    assert exported[0].attributes["prompt_tokens"] == 8
