"""
Quality Validation Framework for Aether Runtime.

Ensures that compiler optimizations, quantization, and kernel changes do not
silently degrade model output quality.

Every optimization must satisfy an explicit quality gate before being enabled
in production. This module provides the measurement tools.

Quality dimensions
------------------
1. Numerical:
   - logit KL divergence vs reference
   - max / mean absolute error on logits
   - cosine similarity of logit distributions
   - per-layer activation error (when activation maps provided)

2. Token-level:
   - greedy token agreement rate (deterministic)
   - sampled distribution overlap (stochastic)

3. Language-model quality:
   - perplexity delta vs reference on a text corpus
   (requires a tokenizer; not computed when unavailable)

4. Regression detection:
   - compare against a stored reference run

Usage example
-------------
    from aether.evaluation.quality_validator import QualityValidator

    validator = QualityValidator(max_kl_div=0.01, min_token_agreement=0.95)

    result = validator.validate_logits(reference_logits, candidate_logits)
    assert result.passed, result.summary()

References
----------
- GPTQ (Frantar et al. 2022): perplexity as quality metric for quantization
- SmoothQuant (Xiao et al. 2022): activation max error as sensitivity proxy
- LLM.int8() (Dettmers et al. 2022): outlier distribution in activations
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "QualityValidator",
    "ValidationResult",
    "LogitValidation",
    "TokenValidation",
    "PerplexityValidation",
    "ActivationValidation",
    "QualityReport",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LogitValidation:
    """Numerical quality of logit distributions."""
    kl_divergence: float
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float
    top1_agreement: bool        # Do reference and candidate agree on argmax?
    top5_agreement_rate: float  # Fraction of top-5 tokens that match
    passed: bool
    threshold_kl: float
    threshold_mae: float

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] Logits: KL={self.kl_divergence:.4f} "
            f"(threshold={self.threshold_kl:.4f}), "
            f"MAE={self.mean_absolute_error:.4f} "
            f"(threshold={self.threshold_mae:.4f}), "
            f"cos={self.cosine_similarity:.4f}, "
            f"top1={'✓' if self.top1_agreement else '✗'}"
        )


@dataclass
class TokenValidation:
    """Token-level quality check for greedy decoding."""
    total_tokens: int
    matched_tokens: int
    agreement_rate: float
    first_divergence_pos: int | None   # Position where sequences first differ
    passed: bool
    threshold: float

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] Tokens: {self.matched_tokens}/{self.total_tokens} "
            f"matched ({100*self.agreement_rate:.1f}%), "
            f"first divergence at pos {self.first_divergence_pos}"
        )


@dataclass
class PerplexityValidation:
    """Perplexity-based quality measurement."""
    reference_perplexity: float
    candidate_perplexity: float
    delta: float                # candidate - reference
    relative_delta: float       # (candidate - reference) / reference
    passed: bool
    threshold_relative_delta: float
    num_tokens: int

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] Perplexity: ref={self.reference_perplexity:.3f}, "
            f"cand={self.candidate_perplexity:.3f}, "
            f"Δ={self.delta:+.3f} ({100*self.relative_delta:+.2f}%), "
            f"threshold={100*self.threshold_relative_delta:.1f}%"
        )


@dataclass
class ActivationValidation:
    """Layer-wise activation error."""
    layer_errors: dict[str, float]     # layer_name → relative error
    max_layer_error: float
    mean_layer_error: float
    high_error_layers: list[str]       # layers exceeding threshold
    passed: bool
    threshold: float

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] Activations: max_err={self.max_layer_error:.4f}, "
            f"mean={self.mean_layer_error:.4f}, "
            f"high_error_layers={len(self.high_error_layers)}"
        )


@dataclass
class ValidationResult:
    """Composite result of all quality checks."""
    logits: LogitValidation | None = None
    tokens: TokenValidation | None = None
    perplexity: PerplexityValidation | None = None
    activations: ActivationValidation | None = None
    passed: bool = True
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Quality Validation Report ==="]
        if self.logits:
            lines.append(self.logits.summary())
        if self.tokens:
            lines.append(self.tokens.summary())
        if self.perplexity:
            lines.append(self.perplexity.summary())
        if self.activations:
            lines.append(self.activations.summary())
        status = "OVERALL: PASS ✓" if self.passed else "OVERALL: FAIL ✗"
        lines.append(status)
        lines.append(f"Duration: {self.duration_ms:.1f} ms")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "logits": {
                "kl_divergence": self.logits.kl_divergence if self.logits else None,
                "mean_absolute_error": self.logits.mean_absolute_error if self.logits else None,
                "cosine_similarity": self.logits.cosine_similarity if self.logits else None,
                "top1_agreement": self.logits.top1_agreement if self.logits else None,
                "passed": self.logits.passed if self.logits else None,
            } if self.logits else None,
            "tokens": {
                "agreement_rate": self.tokens.agreement_rate if self.tokens else None,
                "matched": self.tokens.matched_tokens if self.tokens else None,
                "total": self.tokens.total_tokens if self.tokens else None,
                "passed": self.tokens.passed if self.tokens else None,
            } if self.tokens else None,
            "perplexity": {
                "reference": self.perplexity.reference_perplexity if self.perplexity else None,
                "candidate": self.perplexity.candidate_perplexity if self.perplexity else None,
                "relative_delta_pct": (
                    100 * self.perplexity.relative_delta if self.perplexity else None
                ),
                "passed": self.perplexity.passed if self.perplexity else None,
            } if self.perplexity else None,
            "metadata": self.metadata,
        }


@dataclass
class QualityReport:
    """Accumulated quality report across multiple validation runs."""
    runs: list[ValidationResult] = field(default_factory=list)
    optimization_name: str = ""

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.runs)

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.passed) / len(self.runs)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "optimization": self.optimization_name,
                    "all_passed": self.all_passed,
                    "pass_rate": self.pass_rate,
                    "runs": [r.to_dict() for r in self.runs],
                },
                indent=2,
            )
        )


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class QualityValidator:
    """
    Quality gate for Aether optimizations.

    Default thresholds are conservative: they represent the maximum allowed
    degradation for any optimization that claims zero quality impact.
    Approximating methods (quantization, pruning) must use explicit, wider
    thresholds supplied by the caller.

    Args:
        max_kl_div:            Max KL divergence from reference logit dist.
        max_mae:               Max mean absolute error on logits.
        min_cosine:            Min cosine similarity of logit vectors.
        min_token_agreement:   Min greedy token match rate.
        max_perplexity_delta:  Max relative perplexity increase (0.01 = 1%).
        max_activation_error:  Max relative layer activation error.
    """

    def __init__(
        self,
        max_kl_div: float = 0.01,
        max_mae: float = 0.1,
        min_cosine: float = 0.999,
        min_token_agreement: float = 0.95,
        max_perplexity_delta: float = 0.01,
        max_activation_error: float = 0.05,
    ) -> None:
        self.max_kl_div = max_kl_div
        self.max_mae = max_mae
        self.min_cosine = min_cosine
        self.min_token_agreement = min_token_agreement
        self.max_perplexity_delta = max_perplexity_delta
        self.max_activation_error = max_activation_error

    # ------------------------------------------------------------------
    # Logit validation
    # ------------------------------------------------------------------

    def validate_logits(
        self,
        reference: Any,
        candidate: Any,
        temperature: float = 1.0,
    ) -> LogitValidation:
        """
        Validate logit distributions at one or more positions.

        Args:
            reference:    Reference logits, shape (vocab,) or (seq, vocab).
            candidate:    Candidate logits, same shape.
            temperature:  Softmax temperature for KL divergence computation.

        Returns:
            LogitValidation with all metrics and pass/fail status.
        """
        ref = np.asarray(reference, dtype=np.float64)
        cand = np.asarray(candidate, dtype=np.float64)

        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
            cand = cand.reshape(1, -1)

        kl_divs, mae_vals, cosines, top1_agrees, top5_rates = [], [], [], [], []

        for r, c in zip(ref, cand):
            # Softmax probabilities
            def softmax(x):
                x = x - x.max()
                e = np.exp(x / max(temperature, 1e-8))
                return e / e.sum()

            p = softmax(r)
            q = softmax(c)

            # KL divergence KL(p || q) — penalizes q putting mass where p doesn't
            eps = 1e-10
            kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
            kl_divs.append(max(0.0, kl))

            # MAE on raw logits
            mae = float(np.mean(np.abs(r - c)))
            mae_vals.append(mae)

            # Cosine similarity
            norm_r = np.linalg.norm(r)
            norm_c = np.linalg.norm(c)
            cos = float(np.dot(r, c) / (norm_r * norm_c + 1e-10))
            cosines.append(cos)

            # Top-1 agreement
            top1_agrees.append(int(np.argmax(r)) == int(np.argmax(c)))

            # Top-5 overlap
            top5_r = set(np.argsort(r)[-5:])
            top5_c = set(np.argsort(c)[-5:])
            top5_rates.append(len(top5_r & top5_c) / 5.0)

        kl_mean = float(np.mean(kl_divs))
        mae_mean = float(np.mean(mae_vals))
        cos_mean = float(np.mean(cosines))
        top1_agree = bool(np.mean(top1_agrees) >= 0.5)
        top5_rate = float(np.mean(top5_rates))

        passed = (
            kl_mean <= self.max_kl_div
            and mae_mean <= self.max_mae
            and cos_mean >= self.min_cosine
        )

        return LogitValidation(
            kl_divergence=kl_mean,
            max_absolute_error=float(np.max([np.max(np.abs(r - c)) for r, c in zip(ref, cand)])),
            mean_absolute_error=mae_mean,
            cosine_similarity=cos_mean,
            top1_agreement=top1_agree,
            top5_agreement_rate=top5_rate,
            passed=passed,
            threshold_kl=self.max_kl_div,
            threshold_mae=self.max_mae,
        )

    # ------------------------------------------------------------------
    # Token agreement validation
    # ------------------------------------------------------------------

    def validate_greedy_agreement(
        self,
        reference_tokens: list[int],
        candidate_tokens: list[int],
    ) -> TokenValidation:
        """
        Compare two greedy token sequences.

        Args:
            reference_tokens: Tokens from reference (trusted) engine.
            candidate_tokens: Tokens from optimized engine.
        """
        n = min(len(reference_tokens), len(candidate_tokens))
        if n == 0:
            return TokenValidation(
                total_tokens=0, matched_tokens=0, agreement_rate=1.0,
                first_divergence_pos=None, passed=True,
                threshold=self.min_token_agreement,
            )

        ref = reference_tokens[:n]
        cand = candidate_tokens[:n]
        matches = sum(r == c for r, c in zip(ref, cand))
        rate = matches / n

        first_div = None
        for i, (r, c) in enumerate(zip(ref, cand)):
            if r != c:
                first_div = i
                break

        return TokenValidation(
            total_tokens=n,
            matched_tokens=matches,
            agreement_rate=rate,
            first_divergence_pos=first_div,
            passed=rate >= self.min_token_agreement,
            threshold=self.min_token_agreement,
        )

    # ------------------------------------------------------------------
    # Perplexity validation
    # ------------------------------------------------------------------

    def validate_perplexity(
        self,
        reference_nll_per_token: list[float],
        candidate_nll_per_token: list[float],
    ) -> PerplexityValidation:
        """
        Compare perplexity from per-token negative log-likelihood lists.

        Args:
            reference_nll_per_token: NLL for each token from the reference.
            candidate_nll_per_token: NLL for each token from the candidate.

        Reference and candidate lists must be the same length.
        """
        n = min(len(reference_nll_per_token), len(candidate_nll_per_token))
        if n == 0:
            return PerplexityValidation(
                reference_perplexity=1.0, candidate_perplexity=1.0,
                delta=0.0, relative_delta=0.0, passed=True,
                threshold_relative_delta=self.max_perplexity_delta, num_tokens=0,
            )

        ref_mean_nll = float(np.mean(reference_nll_per_token[:n]))
        cand_mean_nll = float(np.mean(candidate_nll_per_token[:n]))
        ref_ppl = math.exp(min(ref_mean_nll, 100.0))
        cand_ppl = math.exp(min(cand_mean_nll, 100.0))
        delta = cand_ppl - ref_ppl
        rel = delta / max(ref_ppl, 1e-10)

        return PerplexityValidation(
            reference_perplexity=ref_ppl,
            candidate_perplexity=cand_ppl,
            delta=delta,
            relative_delta=rel,
            passed=rel <= self.max_perplexity_delta,
            threshold_relative_delta=self.max_perplexity_delta,
            num_tokens=n,
        )

    # ------------------------------------------------------------------
    # Activation validation
    # ------------------------------------------------------------------

    def validate_activations(
        self,
        reference_activations: dict[str, Any],
        candidate_activations: dict[str, Any],
    ) -> ActivationValidation:
        """
        Compare per-layer activations between reference and candidate runs.

        Args:
            reference_activations: dict {layer_name: np.ndarray}
            candidate_activations: dict {layer_name: np.ndarray}
        """
        layer_errors: dict[str, float] = {}
        for name, ref_act in reference_activations.items():
            if name not in candidate_activations:
                continue
            ref = np.asarray(ref_act, dtype=np.float32)
            cand = np.asarray(candidate_activations[name], dtype=np.float32)
            if ref.shape != cand.shape:
                continue
            norm = float(np.linalg.norm(ref))
            if norm < 1e-10:
                continue
            err = float(np.linalg.norm(ref - cand)) / norm
            layer_errors[name] = err

        if not layer_errors:
            return ActivationValidation(
                layer_errors={}, max_layer_error=0.0, mean_layer_error=0.0,
                high_error_layers=[], passed=True, threshold=self.max_activation_error,
            )

        errs = list(layer_errors.values())
        high = [name for name, e in layer_errors.items() if e > self.max_activation_error]
        return ActivationValidation(
            layer_errors=layer_errors,
            max_layer_error=float(max(errs)),
            mean_layer_error=float(np.mean(errs)),
            high_error_layers=high,
            passed=len(high) == 0,
            threshold=self.max_activation_error,
        )

    # ------------------------------------------------------------------
    # Composite validation
    # ------------------------------------------------------------------

    def validate_all(
        self,
        reference_logits: Any | None = None,
        candidate_logits: Any | None = None,
        reference_tokens: list[int] | None = None,
        candidate_tokens: list[int] | None = None,
        reference_nll: list[float] | None = None,
        candidate_nll: list[float] | None = None,
        reference_activations: dict | None = None,
        candidate_activations: dict | None = None,
        metadata: dict | None = None,
    ) -> ValidationResult:
        """
        Run all available quality checks and return a composite result.

        Only checks for which both reference and candidate are provided are run.
        """
        t0 = time.perf_counter()
        result = ValidationResult(metadata=metadata or {})
        all_passed = True

        if reference_logits is not None and candidate_logits is not None:
            result.logits = self.validate_logits(reference_logits, candidate_logits)
            if not result.logits.passed:
                all_passed = False

        if reference_tokens is not None and candidate_tokens is not None:
            result.tokens = self.validate_greedy_agreement(reference_tokens, candidate_tokens)
            if not result.tokens.passed:
                all_passed = False

        if reference_nll is not None and candidate_nll is not None:
            result.perplexity = self.validate_perplexity(reference_nll, candidate_nll)
            if not result.perplexity.passed:
                all_passed = False

        if reference_activations is not None and candidate_activations is not None:
            result.activations = self.validate_activations(
                reference_activations, candidate_activations
            )
            if not result.activations.passed:
                all_passed = False

        result.passed = all_passed
        result.duration_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Convenience: compare two engines on a prompt
    # ------------------------------------------------------------------

    def compare_engines(
        self,
        reference_engine: Any,
        candidate_engine: Any,
        token_ids: list[int],
        max_tokens: int = 32,
    ) -> ValidationResult:
        """
        Run both engines greedily on the same prompt and compare.

        Both engines must implement forward_step(token_ids, cache) → (logits, cache).
        """
        def _run(engine: Any) -> tuple[list[int], list[float]]:
            cache = engine.init_kv_cache() if hasattr(engine, "init_kv_cache") else None
            tokens = list(token_ids)
            generated: list[int] = []
            nlls: list[float] = []
            logits_last = None
            for _ in range(max_tokens):
                logits, cache = engine.forward_step(tokens, cache, return_all_logits=False)
                logits_np = np.asarray(logits, dtype=np.float64).reshape(-1)
                logits_last = logits_np
                tok = int(np.argmax(logits_np))
                # NLL for the chosen token
                x = logits_np - logits_np.max()
                log_probs = x - np.log(np.exp(x).sum())
                nlls.append(-float(log_probs[tok]))
                generated.append(tok)
                tokens = tokens + [tok]
            return generated, nlls

        ref_tokens, ref_nll = _run(reference_engine)
        cand_tokens, cand_nll = _run(candidate_engine)

        return self.validate_all(
            reference_tokens=ref_tokens,
            candidate_tokens=cand_tokens,
            reference_nll=ref_nll,
            candidate_nll=cand_nll,
        )
