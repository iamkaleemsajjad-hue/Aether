"""
FastAPI middleware for Aether server.

Provides request timing, metrics collection, error tracking, CORS, optional
API key authentication, and request ID injection. Each middleware component is
a standalone Starlette middleware class.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aether.server.metrics import ServerMetrics
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject or forward a unique request ID header."""

    REQUEST_ID_HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        request_id = request.headers.get(self.REQUEST_ID_HEADER, uuid.uuid4().hex[:16])
        response = await call_next(request)
        response.headers[self.REQUEST_ID_HEADER] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Record request timing and pass duration via request state."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        request.state.duration_ms = elapsed_ms
        response.headers["X-Duration-Ms"] = f"{elapsed_ms:.2f}"
        return response


class MetricsMiddleware:
    """Starlette ASGI middleware that records HTTP metrics."""

    def __init__(self, app: ASGIApp, metrics: ServerMetrics | None = None) -> None:
        self.app = app
        self.metrics = metrics or ServerMetrics()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        status_code = [200]

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, _send)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics.record_http_request(method, status_code[0], elapsed_ms)
            logger.debug("HTTP request", method=method, status=status_code[0], duration_ms=f"{elapsed_ms:.1f}")

    @property
    def metrics_store(self) -> ServerMetrics:
        return self.metrics


class CORSMiddleware(BaseHTTPMiddleware):
    """Configurable CORS middleware.

    Security rules (matching the CORS specification's credential model):

    - With a wildcard (``*``) origin policy the ``Origin`` is echoed back but
      ``Access-Control-Allow-Credentials`` is never sent — browsers reject
      credentialed wildcard responses, and reflecting an arbitrary origin
      while allowing credentials would let any site make authenticated
      cross-origin requests.
    - With an explicit origin allow-list, only listed origins are echoed and
      credentials are permitted.
    """

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list[str] | None = None,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if request.method == "OPTIONS":
            response = Response(status_code=204)
        else:
            response = await call_next(request)
        origin = request.headers.get("Origin", "")
        wildcard = "*" in self.allow_origins
        allow_credentials = not wildcard and bool(self.allow_origins)
        if wildcard:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin in self.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            if allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = ",".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ",".join(self.allow_headers)
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication middleware.

    If no API keys are configured, all requests pass through. When keys are set,
    requests must provide a valid key via the ``Authorization: Bearer <key>``
    header.  Key comparison uses ``hmac.compare_digest`` so an attacker cannot
    recover a key byte-by-byte through response timing.
    """

    def __init__(self, app: ASGIApp, api_keys: list[str] | None = None) -> None:
        super().__init__(app)
        self._api_keys = tuple(api_keys) if api_keys else ()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if self._api_keys:
            import hmac

            auth = request.headers.get("Authorization", "")
            supplied = auth[7:] if auth.startswith("Bearer ") else ""
            valid = any(
                hmac.compare_digest(supplied, expected) for expected in self._api_keys
            )
            if not valid:
                return Response(
                    status_code=401,
                    content='{"error":"Unauthorized","message":"Invalid or missing API key"}',
                    headers={"Content-Type": "application/json"},
                )
        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response = await call_next(request)
        duration_ms = getattr(request.state, "duration_ms", 0.0)
        logger.info(
            "HTTP %s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
