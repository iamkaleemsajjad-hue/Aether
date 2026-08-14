"""
aether.safety.guard — Public PromptGuard API with .check() interface.

Wraps the internal PromptGuard.evaluate() to expose a dict-returning
.check(prompt) API that the test suite and CLI can call uniformly.

This is the import path checked by the adversarial security tests.
"""

from __future__ import annotations

from typing import Any

from aether.safety.guardrails import PromptGuard as _PromptGuard
from aether.safety.guardrails import SafetyDecision


class PromptGuard:
    """Public prompt injection guard with dict-returning .check() API.

    Wraps the lexical PromptGuard with additional heuristics for
    common jailbreak patterns (DAN-style, role-play, instruction override).
    """

    # Extra patterns beyond the baseline PromptGuard defaults
    _EXTRA_PATTERNS: dict[str, float] = {
        r"jailbreak": 0.55,
        r"you\s+are\s+now\s+(?:DAN|GPT|an?\s+AI\s+without\s+restrictions)": 0.70,
        r"forget\s+(everything|all\s+instructions|your\s+instructions)": 0.60,
        r"your\s+new\s+task\s+is": 0.50,
        r"you\s+have\s+no\s+restrictions": 0.65,
        r"act\s+as\s+if\s+you\s+have\s+no\s+(safety|policy|rules)": 0.65,
        r"output\s+['\"]PWNED['\"]": 0.80,
        r"ignore\s+(previous|all)\s+instructions": 0.60,
    }

    def __init__(self, threshold: float = 0.60) -> None:
        combined = dict(_PromptGuard.DEFAULT_PATTERNS)
        combined.update(self._EXTRA_PATTERNS)
        self._guard = _PromptGuard(threshold=threshold, patterns=combined)
        self.threshold = threshold

    def check(self, prompt: str) -> dict[str, Any]:
        """Check a prompt for injection patterns.

        Returns a dict with keys:
          - safe (bool): True if prompt is considered safe
          - blocked (bool): True if the prompt was blocked
          - score (float): 0.0–1.0 injection confidence score
          - reasons (list[str]): matched pattern descriptions

        A return of {'safe': True, 'blocked': False} means the prompt
        passed all checks. {'safe': False, 'blocked': True} means it
        was flagged as potentially malicious.
        """
        decision: SafetyDecision = self._guard.evaluate(prompt)
        blocked = not decision.allowed
        return {
            "safe": decision.allowed,
            "blocked": blocked,
            "score": decision.score,
            "reasons": list(decision.reasons),
        }

    def evaluate(self, prompt: str) -> SafetyDecision:
        """Return the raw SafetyDecision (for internal use)."""
        return self._guard.evaluate(prompt)
