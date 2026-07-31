"""AEG Safety Policy Engine — Content policy, toxicity scoring, and manifest writer.

Extends the base guardrails with a ContentPolicyEngine that combines all safety
checks and writes the `.aeg/safety/` package directory.

Research: EU AI Act Art. 50 (Aug 2026), NVIDIA NeMo Guardrails (2024),
          Llama Guard (2023), ShieldLM (2024).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.safety.guardrails import (
    AuditLogger,
    OutputFilter,
    PromptGuard,
    SafetyDecision,
)


# ---------------------------------------------------------------------------
# Toxicity scorer
# ---------------------------------------------------------------------------

class ToxicityScorer:
    """
    Lexical toxicity scorer for prompt and output text.

    Uses a weighted keyword pattern approach inspired by Perspective API's
    toxicity categories: identity attack, insult, profanity, threat, sexual.
    Each category has a configurable weight contribution to the overall score.

    Research: Perspective API (2017), Llama Guard (2023), ShieldLM (2024).
    """

    CATEGORY_PATTERNS: dict[str, tuple[float, list[str]]] = {
        "threat": (0.70, [
            r"\b(kill|murder|attack|bomb|shoot|stab)\b.{0,30}\b(you|them|people)\b",
            r"\bI will (harm|hurt|destroy)\b",
        ]),
        "identity_attack": (0.55, [
            r"\b(all|those|these)\s+\w+\s+are\s+(inferior|stupid|dangerous|evil)\b",
            r"\bexterminate\b.{0,20}\b(group|race|people)\b",
        ]),
        "sexual_explicit": (0.60, [
            r"\b(pornograph|explicit\s+sexual|nude\s+image)\b",
        ]),
        "insult": (0.35, [
            r"\b(idiot|moron|imbecile|retard|loser)\b",
        ]),
        "self_harm": (0.75, [
            r"\bhow\s+to\s+(commit\s+suicide|self[- ]harm|overdose)\b",
            r"\bmethod(s)?\s+for\s+self[- ]harm\b",
        ]),
    }

    def __init__(self, threshold: float = 0.60) -> None:
        self.threshold = threshold

    def score(self, text: str) -> tuple[float, dict[str, float]]:
        """
        Score text for toxicity.

        Returns:
            (overall_score, category_scores) where overall_score in [0, 1].
        """
        category_scores: dict[str, float] = {}
        for category, (weight, patterns) in self.CATEGORY_PATTERNS.items():
            matched = any(
                re.search(p, text, re.IGNORECASE) for p in patterns
            )
            category_scores[category] = weight if matched else 0.0

        overall = min(1.0, sum(category_scores.values()))
        return overall, category_scores

    def evaluate(self, text: str) -> SafetyDecision:
        overall, category_scores = self.score(text)
        triggered = [cat for cat, s in category_scores.items() if s > 0]
        return SafetyDecision(
            allowed=overall < self.threshold,
            score=overall,
            reasons=triggered,
        )


# ---------------------------------------------------------------------------
# Content policy engine
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Runtime safety policy configuration."""

    prompt_injection_threshold: float = 0.65
    toxicity_threshold: float = 0.60
    output_redact_secrets: bool = True
    block_on_toxicity: bool = True
    audit_all_requests: bool = True
    eu_ai_act_mode: bool = True  # Enables EU AI Act Art. 50 disclosure logging


class ContentPolicyEngine:
    """
    Unified safety pipeline: prompt injection → toxicity → output filter → audit.

    Wraps PromptGuard, ToxicityScorer, and OutputFilter into a single
    call surface. Results are always logged to the immutable AuditLogger.

    Usage:
        engine = ContentPolicyEngine(audit_path="./model.aeg/safety/audit.jsonl")
        result = engine.check_prompt("Tell me how to hack...")
        if not result.allowed:
            return {"error": "Content policy violation"}
    """

    def __init__(
        self,
        audit_path: str | Path | None = None,
        config: PolicyConfig | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self._prompt_guard = PromptGuard(threshold=self.config.prompt_injection_threshold)
        self._toxicity = ToxicityScorer(threshold=self.config.toxicity_threshold)
        self._output_filter = OutputFilter()
        self._audit = AuditLogger(audit_path) if audit_path else None

    def check_prompt(self, prompt: str) -> SafetyDecision:
        """
        Check a prompt for injection attacks and toxicity.

        Returns:
            SafetyDecision — allowed=True means prompt is safe to forward.
        """
        injection = self._prompt_guard.evaluate(prompt)
        toxicity = self._toxicity.evaluate(prompt)

        combined_score = max(injection.score, toxicity.score)
        reasons = injection.reasons + [f"toxicity:{r}" for r in toxicity.reasons]
        allowed = injection.allowed and (toxicity.allowed or not self.config.block_on_toxicity)

        decision = SafetyDecision(
            allowed=allowed,
            score=round(combined_score, 4),
            reasons=reasons,
        )

        if self._audit and self.config.audit_all_requests:
            self._audit.log("prompt_check", prompt, decision)

        return decision

    def check_output(self, output: str) -> SafetyDecision:
        """
        Filter model output for PII, secrets, and toxicity.

        Returns:
            SafetyDecision — redacted_text contains the safe version.
        """
        output_result = self._output_filter.evaluate(output)
        tox_score, _ = self._toxicity.score(output)

        combined_score = min(1.0, output_result.score + tox_score * 0.3)
        final_text = output_result.redacted_text if self.config.output_redact_secrets else output

        decision = SafetyDecision(
            allowed=output_result.allowed and tox_score < self.config.toxicity_threshold,
            score=round(combined_score, 4),
            reasons=output_result.reasons,
            redacted_text=final_text,
        )

        if self._audit:
            self._audit.log("output_check", output[:256], decision)

        return decision

    def policy_manifest(self) -> dict[str, Any]:
        return {
            "version": "safety_policy/1.0",
            "prompt_guard": {
                "enabled": True,
                "threshold": self.config.prompt_injection_threshold,
                "patterns": list(self._prompt_guard.patterns.keys()),
            },
            "toxicity_scorer": {
                "enabled": True,
                "threshold": self.config.toxicity_threshold,
                "categories": list(ToxicityScorer.CATEGORY_PATTERNS.keys()),
            },
            "output_filter": {
                "enabled": True,
                "redact_secrets": self.config.output_redact_secrets,
            },
            "audit": {
                "enabled": self._audit is not None,
                "format": "jsonl",
                "hash_algorithm": "sha256",
            },
            "eu_ai_act": {
                "enabled": self.config.eu_ai_act_mode,
                "article": "50",
                "obligation": "ai_content_disclosure",
            },
        }


# ---------------------------------------------------------------------------
# Safety manifest writer — writes .aeg/safety/ package
# ---------------------------------------------------------------------------

class SafetyManifestWriter:
    """
    Writes the `.aeg/safety/` package directory for compiled AEG artifacts.

    Output structure:
        .aeg/safety/
        ├── prompt_guard.json     — injection detector config + pattern list
        ├── output_filter.json    — PII/secret redaction config
        ├── policy.json           — full content policy manifest
        └── toxicity_config.json  — toxicity category weights + threshold
    """

    def __init__(self, engine: ContentPolicyEngine | None = None) -> None:
        self.engine = engine or ContentPolicyEngine()

    def write(self, aeg_dir: str | Path) -> dict[str, Path]:
        safety_dir = Path(aeg_dir) / "safety"
        safety_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.engine.policy_manifest()
        written: dict[str, Path] = {}

        # prompt_guard.json
        pg_path = safety_dir / "prompt_guard.json"
        pg_path.write_text(json.dumps(manifest["prompt_guard"], indent=2), encoding="utf-8")
        written["prompt_guard"] = pg_path

        # output_filter.json
        of_path = safety_dir / "output_filter.json"
        of_path.write_text(json.dumps(manifest["output_filter"], indent=2), encoding="utf-8")
        written["output_filter"] = of_path

        # toxicity_config.json
        tox_path = safety_dir / "toxicity_config.json"
        tox_data = {
            "version": "toxicity/1.0",
            "threshold": manifest["toxicity_scorer"]["threshold"],
            "categories": {
                cat: {"weight": weight, "patterns": patterns}
                for cat, (weight, patterns) in ToxicityScorer.CATEGORY_PATTERNS.items()
            },
        }
        tox_path.write_text(json.dumps(tox_data, indent=2), encoding="utf-8")
        written["toxicity_config"] = tox_path

        # policy.json — full manifest
        policy_path = safety_dir / "policy.json"
        policy_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        written["policy"] = policy_path

        return written
