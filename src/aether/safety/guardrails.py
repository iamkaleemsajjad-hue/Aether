"""AEG safety guardrail layer.

The guardrail layer is intentionally deterministic and dependency-light. It is
suitable for local policy checks, audit logging, and tests; production
installations can replace the policy lists with externally reviewed policies.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SafetyDecision:
    """Result from a guardrail check."""

    allowed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    redacted_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "score": self.score,
            "reasons": self.reasons,
            "redacted_text": self.redacted_text,
        }


class PromptGuard:
    """Prompt-injection detector based on weighted lexical signals."""

    DEFAULT_PATTERNS: dict[str, float] = {
        r"ignore\s+(all\s+)?previous\s+instructions": 0.45,
        r"reveal\s+(the\s+)?system\s+prompt": 0.55,
        r"developer\s+message": 0.35,
        r"exfiltrate|leak|dump\s+secrets": 0.65,
        r"tool\s+output\s+verbatim": 0.30,
        r"disable\s+(safety|guardrails|policy)": 0.40,
    }

    def __init__(self, threshold: float = 0.65, patterns: dict[str, float] | None = None) -> None:
        self.threshold = threshold
        self.patterns = patterns or self.DEFAULT_PATTERNS

    def evaluate(self, prompt: str) -> SafetyDecision:
        score = 0.0
        reasons: list[str] = []
        for pattern, weight in self.patterns.items():
            if re.search(pattern, prompt, flags=re.IGNORECASE):
                score += weight
                reasons.append(pattern)
        score = min(1.0, score)
        return SafetyDecision(allowed=score < self.threshold, score=score, reasons=reasons)


class OutputFilter:
    """Output policy filter with deterministic redaction."""

    DEFAULT_PATTERNS: dict[str, str] = {
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}": "email",
        r"\b(?:\d[ -]*?){13,16}\b": "possible_payment_card",
        r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*\S+": "secret",
    }

    def __init__(self, block_labels: set[str] | None = None) -> None:
        self.block_labels = block_labels or {"secret"}

    def evaluate(self, text: str) -> SafetyDecision:
        reasons: list[str] = []
        redacted = text
        for pattern, label in self.DEFAULT_PATTERNS.items():
            if re.search(pattern, redacted):
                reasons.append(label)
                redacted = re.sub(pattern, f"[{label.upper()}_REDACTED]", redacted)
        allowed = not any(label in self.block_labels for label in reasons)
        score = min(1.0, len(reasons) * 0.35)
        return SafetyDecision(allowed=allowed, score=score, reasons=reasons, redacted_text=redacted)


@dataclass(frozen=True)
class AuditEvent:
    """Immutable audit event stored by the safety layer."""

    event_type: str
    request_hash: str
    decision: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "request_hash": self.request_hash,
            "decision": self.decision,
            "timestamp": self.timestamp,
        }


class AuditLogger:
    """Append-only JSONL audit log with SHA-256 request hashes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, request_text: str, decision: SafetyDecision) -> AuditEvent:
        request_hash = "sha256:" + hashlib.sha256(request_text.encode("utf-8")).hexdigest()
        event = AuditEvent(
            event_type=event_type,
            request_hash=request_hash,
            decision=decision.to_dict(),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return event
