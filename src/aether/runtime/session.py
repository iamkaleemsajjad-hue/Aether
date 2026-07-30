"""
Runtime session and request tracking.

A Session groups one or more generation requests for the same model and
records aggregated metrics. Sessions are used by the server to bind HTTP
requests to runtime state and by the CLI for interactive chat.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetrics:
    """Metrics collected for a single request within a session."""

    request_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    ttft_ms: float = 0.0
    backend: str | None = None
    target: str | None = None
    finish_reason: str = "stop"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "duration_ms": self.duration_ms,
            "ttft_ms": self.ttft_ms,
            "backend": self.backend,
            "target": self.target,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@dataclass
class Session:
    """A runtime session grouping requests and state."""

    session_id: str
    model_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    requests: list[RequestMetrics] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _request_counter: int = field(default=0, repr=False)

    def touch(self) -> None:
        """Update the last-activity timestamp."""
        self.last_activity = time.time()

    def new_request_id(self) -> str:
        """Generate a unique request ID within this session."""
        self._request_counter += 1
        return f"{self.session_id}-req-{self._request_counter}"

    def record(self, metrics: RequestMetrics) -> None:
        """Record a completed request."""
        self.requests.append(metrics)
        self.touch()
        logger.info(
            "Request recorded",
            request_id=metrics.request_id,
            tokens=metrics.completion_tokens,
            duration_ms=round(metrics.duration_ms, 2),
        )

    def total_tokens(self) -> int:
        """Return total tokens across all requests."""
        return sum(r.prompt_tokens + r.completion_tokens for r in self.requests)

    def average_tps(self) -> float:
        """Return average throughput tokens per second."""
        total_completion = sum(r.completion_tokens for r in self.requests)
        total_duration_s = sum(r.duration_ms for r in self.requests) / 1000.0
        if total_duration_s <= 0:
            return 0.0
        return total_completion / total_duration_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "request_count": len(self.requests),
            "total_tokens": self.total_tokens(),
            "average_tps": self.average_tps(),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Session({self.session_id}, requests={len(self.requests)})"


class SessionManager:
    """Manages active runtime sessions."""

    def __init__(self, max_sessions: int = 10000, ttl_seconds: float = 3600.0) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}

    def create(self, model_id: str | None = None, metadata: dict[str, Any] | None = None) -> Session:
        """Create a new session."""
        session_id = f"sess-{uuid.uuid4().hex[:16]}"
        session = Session(
            session_id=session_id,
            model_id=model_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        if len(self._sessions) > self.max_sessions:
            self._evict_oldest()
        logger.info("Session created", session_id=session_id, model_id=model_id)
        return session

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID if it exists and is not expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_activity > self.ttl_seconds:
            self._sessions.pop(session_id, None)
            return None
        session.touch()
        return session

    def destroy(self, session_id: str) -> bool:
        """Destroy a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session destroyed", session_id=session_id)
            return True
        return False

    def list_sessions(self) -> list[str]:
        """Return a list of active session IDs."""
        return sorted(self._sessions.keys())

    def _evict_oldest(self) -> None:
        """Evict the least-recently active session."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_activity)
        self._sessions.pop(oldest_id, None)
        logger.info("Evicted oldest session", session_id=oldest_id)

    def cleanup_expired(self) -> int:
        """Remove expired sessions and return the count removed."""
        now = time.time()
        expired = [sid for sid, sess in self._sessions.items() if now - sess.last_activity > self.ttl_seconds]
        for sid in expired:
            self._sessions.pop(sid, None)
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return f"SessionManager(sessions={len(self._sessions)}, max={self.max_sessions})"
