"""
Cascade Router — Complexity-Based Model Tier Selection.

The cascade router classifies each incoming request by complexity and routes
it to the most cost-efficient model tier. Simple requests go to a small fast
model; complex requests escalate to a larger model.

Routing tiers (configurable):
  Tier 0 — Nano  (≤1B params): quick Q&A, simple completions
  Tier 1 — Small (1-7B):       standard conversational tasks
  Tier 2 — Mid   (7-35B):      reasoning, coding, complex Q&A
  Tier 3 — Large (35B+):       hard math, expert-level tasks

Complexity signals used:
  - Token count of the prompt
  - Presence of reasoning/math keywords
  - Presence of code
  - Question complexity heuristic (question word density + negation)
  - Optional PRM (Process Reward Model) head score

Escalation:
  If a lower tier produces a low-confidence answer, the cascade router
  can escalate to the next tier and merge/override the response.

Research:
  - FrugalGPT (Chen et al., 2023) — LLM cascade for cost savings
  - Cascade Inference (Dohan et al., 2022)
  - RouterBench (Hu et al., 2024)
  - Speculative RAG cascades (2025)
"""

from __future__ import annotations

import re
import time
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request complexity scoring
# ---------------------------------------------------------------------------

# Keywords suggesting high complexity (math, reasoning, expert tasks)
_COMPLEX_KEYWORDS = frozenset([
    "prove", "derive", "calculate", "solve", "integrate", "differentiate",
    "algorithm", "optimize", "theorem", "lemma", "hypothesis",
    "implement", "debug", "refactor", "architecture",
    "compare", "contrast", "analyze", "synthesize", "evaluate",
    "explain why", "what if", "how would", "step by step",
    "aime", "competition", "olympiad",
])

# Keywords suggesting reasoning model needed
_REASONING_KEYWORDS = frozenset([
    "think step by step", "chain of thought", "let me think",
    "reasoning", "proof", "formal", "rigorously",
    "aime", "math competition", "putnam",
])

# Code markers
_CODE_MARKERS = frozenset(["```", "def ", "class ", "import ", "function ", "var ", "const "])


@dataclass
class ComplexitySignals:
    """Raw complexity signals extracted from a prompt."""
    prompt_tokens: int = 0
    has_code: bool = False
    has_math: bool = False
    has_reasoning_request: bool = False
    complex_keyword_count: int = 0
    question_word_density: float = 0.0
    negation_count: int = 0
    avg_word_length: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "has_code": self.has_code,
            "has_math": self.has_math,
            "has_reasoning_request": self.has_reasoning_request,
            "complex_keyword_count": self.complex_keyword_count,
            "question_word_density": round(self.question_word_density, 3),
            "negation_count": self.negation_count,
            "avg_word_length": round(self.avg_word_length, 2),
        }


class ComplexityScorer:
    """
    Scores request complexity on a [0, 1] scale.

    Score bands:
      0.0 – 0.25 → simple   (Tier 0-1)
      0.25 – 0.55 → medium  (Tier 1-2)
      0.55 – 0.80 → hard    (Tier 2)
      0.80 – 1.0  → expert  (Tier 3)
    """

    # Weights for each signal
    _W_TOKENS      = 0.15   # prompt length is a weak signal
    _W_CODE        = 0.15
    _W_MATH        = 0.20
    _W_REASONING   = 0.25
    _W_KEYWORDS    = 0.15
    _W_Q_DENSITY   = 0.05
    _W_WORD_LEN    = 0.05

    def extract_signals(self, prompt: str) -> ComplexitySignals:
        """Extract complexity signals from raw prompt text."""
        words = prompt.lower().split()
        num_words = max(len(words), 1)

        tokens = max(1, len(prompt) // 4)  # rough tokenization

        has_code = any(m in prompt for m in _CODE_MARKERS)
        has_math = bool(re.search(r"\$.*?\$|\\[a-z]+{|[∫∑∏√∞≠≤≥]|\b(sin|cos|tan|log|lim)\b", prompt))

        prompt_lower = prompt.lower()
        has_reasoning = any(kw in prompt_lower for kw in _REASONING_KEYWORDS)
        complex_kws = sum(1 for kw in _COMPLEX_KEYWORDS if kw in prompt_lower)

        q_words = {"what", "why", "how", "when", "where", "which", "who", "whose"}
        q_density = sum(1 for w in words if w in q_words) / num_words

        negations = len(re.findall(r"\b(not|no|never|without|cannot|isn't|aren't|don't)\b", prompt_lower))
        avg_word_len = sum(len(w) for w in words) / num_words

        return ComplexitySignals(
            prompt_tokens=tokens,
            has_code=has_code,
            has_math=has_math,
            has_reasoning_request=has_reasoning,
            complex_keyword_count=complex_kws,
            question_word_density=q_density,
            negation_count=negations,
            avg_word_length=avg_word_len,
        )

    def score(self, signals: ComplexitySignals) -> float:
        """Compute scalar complexity score in [0, 1]."""
        # Token count: log-scaled, 0 at 32 tokens, 1 at 4096 tokens
        token_score = min(1.0, math.log(max(signals.prompt_tokens, 1) + 1) / math.log(4097))

        code_score      = 1.0 if signals.has_code else 0.0
        math_score      = 1.0 if signals.has_math else 0.0
        reason_score    = 1.0 if signals.has_reasoning_request else 0.0
        keyword_score   = min(1.0, signals.complex_keyword_count / 5.0)
        q_density_score = min(1.0, signals.question_word_density * 10.0)
        word_len_score  = min(1.0, max(0.0, signals.avg_word_length - 4.0) / 6.0)

        total = (
            self._W_TOKENS    * token_score
            + self._W_CODE    * code_score
            + self._W_MATH    * math_score
            + self._W_REASONING * reason_score
            + self._W_KEYWORDS  * keyword_score
            + self._W_Q_DENSITY * q_density_score
            + self._W_WORD_LEN  * word_len_score
        )
        return float(np.clip(total, 0.0, 1.0))

    def classify(self, score: float) -> str:
        """Map complexity score to named difficulty class."""
        if score < 0.25:
            return "simple"
        elif score < 0.55:
            return "medium"
        elif score < 0.80:
            return "hard"
        else:
            return "very_hard"


# Re-import math (needed for log in score())
import math


# ---------------------------------------------------------------------------
# Model tier registry
# ---------------------------------------------------------------------------

@dataclass
class ModelTier:
    """A registered model tier in the cascade."""
    tier_id: int
    model_id: str
    max_complexity: float     # route here if complexity ≤ this
    max_tokens: int = 4096
    cost_per_token: float = 0.0   # relative cost unit
    supports_reasoning: bool = False
    backend_name: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "model_id": self.model_id,
            "max_complexity": self.max_complexity,
            "max_tokens": self.max_tokens,
            "cost_per_token": self.cost_per_token,
            "supports_reasoning": self.supports_reasoning,
            "backend_name": self.backend_name,
        }


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    """Decision output from the cascade router."""
    tier: ModelTier
    complexity_score: float
    complexity_class: str
    signals: ComplexitySignals
    escalated: bool = False
    reasoning_override: bool = False  # True if reasoning model forced
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "routed_to": self.tier.model_id,
            "tier_id": self.tier.tier_id,
            "complexity_score": round(self.complexity_score, 4),
            "complexity_class": self.complexity_class,
            "escalated": self.escalated,
            "reasoning_override": self.reasoning_override,
            "latency_ms": round(self.latency_ms, 2),
            "signals": self.signals.to_dict(),
        }


# ---------------------------------------------------------------------------
# Cascade Router
# ---------------------------------------------------------------------------

class CascadeRouter:
    """
    Complexity-based cascade router for multi-tier LLM serving.

    Usage:
        router = CascadeRouter()
        router.register_default_tiers(
            nano="path/to/nano.aeg", small="path/to/small.aeg",
            mid="path/to/mid.aeg", large="path/to/large.aeg",
        )

        decision = router.route("What is 2+2?")
        # → routes to qwen3-0.6b (simple)
    """

    def __init__(
        self,
        scorer: ComplexityScorer | None = None,
        escalation_confidence_threshold: float = 0.6,
    ) -> None:
        self._scorer    = scorer or ComplexityScorer()
        self._tiers: list[ModelTier] = []
        self._escalation_threshold = escalation_confidence_threshold
        # Stats
        self._route_counts: dict[int, int] = {}
        self._total_routes = 0
        self._escalation_count = 0

    def register_tier(self, tier: ModelTier) -> None:
        """Register a model tier. Tiers are sorted by tier_id automatically."""
        self._tiers.append(tier)
        self._tiers.sort(key=lambda t: t.tier_id)
        self._route_counts.setdefault(tier.tier_id, 0)

    @property
    def tiers(self) -> tuple[ModelTier, ...]:
        """Return a read-only snapshot of the registered tiers."""
        return tuple(self._tiers)
        logger.debug("Cascade router: registered tier %d → %s", tier.tier_id, tier.model_id)

    def register_default_tiers(
        self,
        nano: str | None = None,
        small: str | None = None,
        mid: str | None = None,
        large: str | None = None,
    ) -> None:
        """Register configured cascade tiers.

        There is deliberately no model-family default.  The compiler/runtime
        cannot know which checkpoints exist on a user's machine, and an
        implicit Qwen registry made this feature appear model-specialized.
        Values may be supplied directly or through the corresponding
        ``AETHER_CASCADE_*_MODEL`` environment variables.
        """
        values = (
            nano or os.environ.get("AETHER_CASCADE_NANO_MODEL"),
            small or os.environ.get("AETHER_CASCADE_SMALL_MODEL"),
            mid or os.environ.get("AETHER_CASCADE_MID_MODEL"),
            large or os.environ.get("AETHER_CASCADE_LARGE_MODEL"),
        )
        if not any(values):
            raise ValueError(
                "cascade tiers require configured model IDs; pass them to "
                "register_default_tiers() or set AETHER_CASCADE_*_MODEL"
            )
        tier_specs = (
            (0, values[0], 0.25, 2048, 0.1, False),
            (1, values[1], 0.55, 8192, 1.0, False),
            (2, values[2], 0.80, 32768, 4.0, True),
            (3, values[3], 1.00, 131072, 16.0, True),
        )
        for tier_id, model_id, limit, max_tokens, cost, reasoning in tier_specs:
            if model_id:
                self.register_tier(ModelTier(
                    tier_id, model_id, max_complexity=limit,
                    max_tokens=max_tokens, cost_per_token=cost,
                    supports_reasoning=reasoning,
                ))

    def route(
        self,
        prompt: str,
        force_tier: int | None = None,
        force_reasoning: bool = False,
    ) -> RouteDecision:
        """
        Route a request to the appropriate model tier.

        Args:
            prompt: The user's input prompt.
            force_tier: Override routing to a specific tier ID.
            force_reasoning: Force routing to a reasoning-capable tier.

        Returns:
            RouteDecision with selected tier and routing metadata.
        """
        t0 = time.perf_counter()

        if not self._tiers:
            raise RuntimeError("No tiers registered. Call register_tier() first.")

        signals = self._scorer.extract_signals(prompt)
        score   = self._scorer.score(signals)
        cls     = self._scorer.classify(score)

        if force_tier is not None:
            tier = self._get_tier(force_tier)
        elif force_reasoning:
            tier = self._get_reasoning_tier()
        else:
            tier = self._select_tier(score, signals)

        reasoning_override = force_reasoning and not tier.supports_reasoning
        if reasoning_override:
            # Escalate to first reasoning-capable tier
            tier = self._get_reasoning_tier()

        self._route_counts[tier.tier_id] = self._route_counts.get(tier.tier_id, 0) + 1
        self._total_routes += 1

        latency = (time.perf_counter() - t0) * 1000
        decision = RouteDecision(
            tier=tier,
            complexity_score=score,
            complexity_class=cls,
            signals=signals,
            reasoning_override=reasoning_override,
            latency_ms=latency,
        )
        logger.debug(
            "Cascade route: %s → tier %d (%s), complexity=%.3f",
            cls, tier.tier_id, tier.model_id, score
        )
        return decision

    def escalate(self, decision: RouteDecision, reason: str = "") -> RouteDecision | None:
        """
        Escalate a routing decision to the next tier.

        Called when the lower-tier model returns a low-confidence answer.
        Returns None if already at the highest tier.
        """
        current_tier_id = decision.tier.tier_id
        next_tiers = [t for t in self._tiers if t.tier_id > current_tier_id]
        if not next_tiers:
            logger.debug("Cascade: already at highest tier, no escalation possible")
            return None

        next_tier = next_tiers[0]
        self._escalation_count += 1
        self._route_counts[next_tier.tier_id] = self._route_counts.get(next_tier.tier_id, 0) + 1
        logger.info(
            "Cascade escalation: tier %d → tier %d (%s), reason=%s",
            current_tier_id, next_tier.tier_id, next_tier.model_id, reason
        )
        return RouteDecision(
            tier=next_tier,
            complexity_score=decision.complexity_score,
            complexity_class=decision.complexity_class,
            signals=decision.signals,
            escalated=True,
            latency_ms=0.0,
        )

    def _select_tier(self, score: float, signals: ComplexitySignals) -> ModelTier:
        """Select the cheapest tier that can handle the request complexity."""
        for tier in self._tiers:
            if score <= tier.max_complexity:
                return tier
        return self._tiers[-1]  # highest tier as fallback

    def _get_tier(self, tier_id: int) -> ModelTier:
        for t in self._tiers:
            if t.tier_id == tier_id:
                return t
        raise ValueError(f"Tier {tier_id} not registered")

    def _get_reasoning_tier(self) -> ModelTier:
        for t in self._tiers:
            if t.supports_reasoning:
                return t
        return self._tiers[-1]

    def stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        if self._total_routes == 0:
            return {"total_routes": 0, "escalations": 0, "tier_distribution": {}}
        tier_dist = {
            f"tier_{tid}": round(count / self._total_routes, 3)
            for tid, count in self._route_counts.items()
        }
        return {
            "total_routes": self._total_routes,
            "escalations": self._escalation_count,
            "escalation_rate": round(self._escalation_count / self._total_routes, 3),
            "tier_distribution": tier_dist,
            "registered_tiers": [t.to_dict() for t in self._tiers],
        }

    def __repr__(self) -> str:
        return f"CascadeRouter(tiers={len(self._tiers)}, routes={self._total_routes})"
