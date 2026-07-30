"""
Anonymous telemetry and usage statistics.

Telemetry is opt-in and collects only high-level usage signals (e.g., which
backend was selected, how many models are cached, compiler pass success rates)
without model prompts, outputs, or personally identifiable information.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.file_io import aether_cache_dir
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TelemetryEvent:
    """A single telemetry event."""

    event_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "payload": self.payload,
        }


class TelemetryClient:
    """In-memory telemetry collector.

    The default implementation buffers events locally. A production deployment
    may override the `flush()` method to send events to a telemetry backend.
    """

    def __init__(self, enabled: bool = True, cache_dir: str | None = None) -> None:
        self.enabled = enabled
        self._events: list[TelemetryEvent] = []
        self._session_id = f"tel-{uuid.uuid4().hex[:16]}"
        self._cache_dir = aether_cache_dir(cache_dir)
        self._log_path = self._cache_dir / "logs" / "telemetry.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Record a telemetry event."""
        if not self.enabled:
            return
        event = TelemetryEvent(
            event_type=event_type,
            session_id=self._session_id,
            payload=payload or {},
        )
        self._events.append(event)
        logger.debug("Telemetry event", event_type=event_type, payload=event.payload)

    def flush(self) -> dict[str, Any]:
        """Flush recorded events to the local log.

        Returns a summary of flushed events.
        """
        if not self.enabled or not self._events:
            return {"flushed": 0, "session_id": self._session_id}
        flushed = 0
        with self._log_path.open("a", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict()) + "\n")
                flushed += 1
        self._events.clear()
        return {"flushed": flushed, "session_id": self._session_id, "log": str(self._log_path)}

    def summary(self) -> dict[str, Any]:
        """Return a summary of buffered events."""
        counts: dict[str, int] = {}
        for event in self._events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return {
            "enabled": self.enabled,
            "session_id": self._session_id,
            "buffered": len(self._events),
            "counts": counts,
        }

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return f"TelemetryClient(enabled={self.enabled}, events={len(self._events)})"


def system_context() -> dict[str, Any]:
    """Return a safe, non-identifying system context dictionary."""
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }
