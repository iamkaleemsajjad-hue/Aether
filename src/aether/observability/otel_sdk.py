"""Bridge Aether spans and metrics into the real OpenTelemetry SDK.

The built-in exporter in :mod:`aether.observability.otel` speaks OTLP directly so
the base runtime needs no telemetry dependency.  That is the right default, and it
is not the right answer for a deployment that already standardizes on the
OpenTelemetry SDK: such a deployment has span processors, resource detectors,
context propagators and exporters configured centrally, and telemetry that
bypasses them is telemetry nobody sees.

This module is the other half.  When ``opentelemetry-sdk`` is installed it emits
Aether's spans through the SDK's real tracer, so they pick up whatever pipeline the
host application configured, and Aether's metrics through the SDK's meter.

Install with::

    pip install "aether-runtime[otel]"

The bridge is *additive*.  Nothing here is required for Aether to produce OTLP, and
:func:`is_available` reports honestly rather than raising at import time, so a
caller can degrade to the built-in exporter instead of failing.

References:
  * OpenTelemetry Python SDK — ``opentelemetry.sdk.trace``, ``.metrics``.
  * OpenTelemetry API — ``opentelemetry.trace``, context propagation.
"""

from __future__ import annotations

from typing import Any

from aether.core.constants import AETHER_VERSION
from aether.observability.otel import Span, SpanKind, TraceContext
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "OTelSDKUnavailable",
    "is_available",
    "sdk_version",
    "OpenTelemetryBridge",
]


class OTelSDKUnavailable(RuntimeError):
    """Raised when the SDK bridge is used without the SDK installed."""


def _import_sdk() -> tuple[Any, Any, Any] | None:
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except Exception:  # noqa: BLE001 - an optional integration, never required
        return None
    return trace_api, Resource, TracerProvider


def _import_replay() -> tuple[Any, Any, Any] | None:
    """Import the pieces needed to replay an already-finished span.

    ``Event`` moved between ``opentelemetry.sdk.trace`` and
    ``opentelemetry.trace`` across SDK versions, so both spellings are tried
    rather than pinning one and failing on the other.
    """
    try:
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    except Exception:  # noqa: BLE001
        return None
    event_type: Any = None
    for module_name in ("opentelemetry.sdk.trace", "opentelemetry.trace"):
        try:
            module = __import__(module_name, fromlist=["Event"])
            event_type = module.Event
            break
        except Exception:  # noqa: BLE001
            continue
    if event_type is None:
        return None
    return ReadableSpan, InstrumentationScope, event_type


def is_available() -> bool:
    """Whether the OpenTelemetry SDK can be used in this process."""
    return _import_sdk() is not None


def sdk_version() -> str:
    """Return the installed OpenTelemetry SDK version, or an empty string."""
    try:
        from importlib.metadata import version

        return version("opentelemetry-sdk")
    except Exception:  # noqa: BLE001
        return ""


_KIND_MAP = {
    SpanKind.INTERNAL: "INTERNAL",
    SpanKind.SERVER: "SERVER",
    SpanKind.CLIENT: "CLIENT",
    SpanKind.PRODUCER: "PRODUCER",
    SpanKind.CONSUMER: "CONSUMER",
}


def _flatten(attributes: dict[str, Any]) -> dict[str, Any]:
    """Reduce attributes to the types the OTel attribute contract permits.

    The SDK accepts ``str``, ``bool``, ``int``, ``float`` and homogeneous sequences
    of those.  Anything else is rendered as text here rather than dropped by the
    SDK with a warning, so no attribute silently disappears.
    """
    flat: dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, (str, bool, int, float)):
            flat[str(key)] = value
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(item, type(value[0])) and isinstance(item, (str, bool, int, float))
            for item in value
        ):
            flat[str(key)] = list(value)
        else:
            flat[str(key)] = repr(value)
    return flat


class OpenTelemetryBridge:
    """Emit Aether spans through the OpenTelemetry SDK.

    Spans keep the trace and span IDs Aether already assigned.  That matters: an
    Aether span re-issued with a fresh ID would be a second, disconnected trace of
    the same request, and the correlation an operator is looking for would be
    exactly what got lost.
    """

    def __init__(
        self,
        service_name: str = "aether-runtime",
        *,
        tracer_provider: Any | None = None,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        imported = _import_sdk()
        if imported is None:
            raise OTelSDKUnavailable(
                "the OpenTelemetry SDK is not installed; "
                'install it with pip install "aether-runtime[otel]", or use '
                "aether.observability.otel.OTLPExporter, which needs no dependency"
            )
        trace_api, Resource, TracerProvider = imported
        self._trace_api = trace_api
        self.service_name = service_name
        if tracer_provider is None:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": AETHER_VERSION,
                    **(resource_attributes or {}),
                }
            )
            tracer_provider = TracerProvider(resource=resource)
        self.tracer_provider = tracer_provider
        self._tracer = tracer_provider.get_tracer("aether.observability", AETHER_VERSION)

    def _span_context(self, span: Span) -> Any:
        """Rebuild an SDK ``SpanContext`` from Aether's own IDs."""
        return self._trace_api.SpanContext(
            trace_id=int(span.trace_id, 16),
            span_id=int(span.span_id, 16),
            is_remote=False,
            trace_flags=self._trace_api.TraceFlags(
                self._trace_api.TraceFlags.SAMPLED
                if span.sampled
                else self._trace_api.TraceFlags.DEFAULT
            ),
        )

    def emit(self, span: Span) -> None:
        """Emit one finished Aether span through the SDK.

        The span is *replayed*, not re-started.  Calling ``tracer.start_span``
        would make the SDK mint fresh IDs and time the call rather than the work,
        producing a second disconnected trace of a request that already has one.
        So a :class:`~opentelemetry.sdk.trace.ReadableSpan` is constructed with
        Aether's own IDs, timestamps, attributes and events, and handed to the
        provider's span processor — the same path a span the SDK started takes when
        it ends.
        """
        replay = _import_replay()
        processor = getattr(self.tracer_provider, "_active_span_processor", None)
        if replay is None or processor is None:
            # No supported replay path on this SDK version. Fall back to a
            # freshly-started span so telemetry still flows, and say plainly that
            # the correlation IDs will not match.
            logger.warning(
                "OpenTelemetry SDK %s does not expose a span-replay path; emitting "
                "%s with SDK-generated IDs, so it will not correlate with Aether's "
                "own trace_id %s",
                sdk_version() or "(unknown)",
                span.name,
                span.trace_id,
            )
            self._emit_via_tracer(span)
            return
        ReadableSpan, InstrumentationScope, Event = replay
        events = tuple(
            Event(
                name=event["name"],
                attributes=_flatten(event.get("attributes", {})),
                timestamp=event.get("time_unix_nano"),
            )
            for event in span.events
        )
        status = self._trace_api.Status(
            self._trace_api.StatusCode.ERROR,
            str(span.attributes.get("error.message", "")),
        ) if span.status == "ERROR" else self._trace_api.Status(
            self._trace_api.StatusCode.OK
        )
        readable = ReadableSpan(
            name=span.name,
            context=self._span_context(span),
            parent=self._parent_context(span),
            resource=getattr(self.tracer_provider, "resource", None),
            attributes=_flatten(span.attributes),
            events=events,
            links=(),
            kind=getattr(
                self._trace_api.SpanKind, _KIND_MAP.get(span.kind, "INTERNAL")
            ),
            status=status,
            start_time=span.start_time_ns,
            end_time=span.end_time_ns or span.start_time_ns,
            instrumentation_scope=InstrumentationScope(
                "aether.observability", AETHER_VERSION
            ),
        )
        processor.on_end(readable)

    def _parent_context(self, span: Span) -> Any:
        """Return the parent ``SpanContext``, or ``None`` for a root span."""
        if not span.parent_span_id:
            return None
        return self._trace_api.SpanContext(
            trace_id=int(span.trace_id, 16),
            span_id=int(span.parent_span_id, 16),
            is_remote=True,
            trace_flags=self._trace_api.TraceFlags(self._trace_api.TraceFlags.SAMPLED),
        )

    def _emit_via_tracer(self, span: Span) -> None:
        """Last-resort path: start and end a live SDK span with new IDs."""
        sdk_span = self._tracer.start_span(
            span.name,
            kind=getattr(self._trace_api.SpanKind, _KIND_MAP.get(span.kind, "INTERNAL")),
            attributes=_flatten(span.attributes),
            start_time=span.start_time_ns,
        )
        for event in span.events:
            sdk_span.add_event(
                event["name"],
                attributes=_flatten(event.get("attributes", {})),
                timestamp=event.get("time_unix_nano"),
            )
        if span.status == "ERROR":
            sdk_span.set_status(
                self._trace_api.Status(
                    self._trace_api.StatusCode.ERROR,
                    str(span.attributes.get("error.message", "")),
                )
            )
        sdk_span.end(end_time=span.end_time_ns or span.start_time_ns)

    def emit_all(self, spans: "list[Span]") -> int:
        """Emit a batch of finished spans; returns how many were emitted."""
        emitted = 0
        for span in spans:
            if not span.sampled:
                continue
            self.emit(span)
            emitted += 1
        return emitted

    def current_context(self) -> TraceContext | None:
        """Return the SDK's currently active span as an Aether trace context.

        This is how an Aether span becomes a child of a span the host application
        started — a FastAPI request, for instance — rather than the root of its own
        unrelated trace.
        """
        current = self._trace_api.get_current_span()
        context = current.get_span_context() if current is not None else None
        if context is None or not context.is_valid:
            return None
        return TraceContext(
            trace_id=f"{context.trace_id:032x}",
            span_id=f"{context.span_id:016x}",
            sampled=bool(context.trace_flags & self._trace_api.TraceFlags.SAMPLED),
        )

    def shutdown(self) -> None:
        """Flush and shut down the provider this bridge created."""
        try:
            self.tracer_provider.shutdown()
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask a real error
            logger.debug("OpenTelemetry provider shutdown failed: %s", exc)
