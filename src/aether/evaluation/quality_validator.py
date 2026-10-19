"""Measure quality from paired evidence; missing evidence never passes.

Checks describe supplied data, not automatic model certification. Compiler
admission must also bind reference/source/candidate/dataset identities.

KL uses the discrete expectation in Blei et al., arXiv:1601.00670v9, Eq. (11),
with reference first. Reference implementation: PyTorch v2.9.1 kl_div:
https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Loss.cpp

Fixed-label likelihood follows Radford et al. (2019), Eq. (1):
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
Reference alignment: Transformers v4.57.3 loss/loss_utils.py::ForCausalLMLoss.
PPL = exp(mean NLL): https://huggingface.co/docs/transformers/en/perplexity
A model's own argmax scores are not corpus perplexity.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "ActivationValidation",
    "LogitValidation",
    "PerplexityValidation",
    "QualityReport",
    "QualityValidator",
    "TokenValidation",
    "ValidationResult",
]

FloatArray: TypeAlias = NDArray[np.float64]


def _scalar(value: float, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its valid range")
    return float(value)


def _array(value: Any, name: str) -> FloatArray:
    if isinstance(value, np.ma.MaskedArray) and np.any(value.mask):
        raise ValueError(f"{name} contains masked, missing evidence")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.size == 0:
        raise ValueError(f"{name} must contain nonempty real numeric evidence")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _tokens(value: list[int], name: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of nonnegative integer token IDs")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional token sequence")
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(token, (bool, np.bool_)) or not isinstance(token, Integral) or token < 0
        for token in value
    ):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return [int(token) for token in value]


def _log_probs(values: FloatArray, temperature: float = 1.0) -> FloatArray:
    # Preserve low-probability evidence, matching PyTorch's log-target KL form.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            shifted = (values - values.max(axis=-1, keepdims=True)) / temperature
            result = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    except FloatingPointError as exc:
        raise ValueError("Logit range cannot be represented for quality measurement") from exc
    if not np.isfinite(result).all():
        raise ValueError("Log probabilities must be finite")
    return np.asarray(result, dtype=np.float64)


def _mean(values: FloatArray) -> float:
    # Divide first so a sum of large finite NLLs does not overflow.
    return math.fsum(float(value) / values.size for value in values.flat)


def _exp(value: float) -> float | None:
    try:
        return math.exp(value)
    except OverflowError:
        return None


def _metric_payload(value: Any, nonfinite: dict[str, str], path: str) -> Any:
    """Encode computed unbounded errors explicitly, without JSON Infinity/NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite[path] = (
            "nan"
            if math.isnan(value)
            else ("positive_infinity" if value > 0 else "negative_infinity")
        )
        return None
    if isinstance(value, dict):
        return {
            key: _metric_payload(item, nonfinite, f"{path}.{key}") for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _metric_payload(item, nonfinite, f"{path}.{index}") for index, item in enumerate(value)
        ]
    return value


@dataclass
class LogitValidation:
    """Logit statistics; gates use worst-position KL/cosine and mean MAE."""

    kl_divergence: float
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float
    top1_agreement: bool
    top5_agreement_rate: float
    passed: bool
    threshold_kl: float
    threshold_mae: float
    max_kl_divergence: float = 0.0
    min_cosine_similarity: float = 0.0
    top1_agreement_rate: float = 0.0
    num_positions: int = 0
    threshold_cosine: float = 0.999

    def summary(self) -> str:
        """Describe measured extrema without claiming other quality dimensions."""
        return (
            f"[{'PASS' if self.passed else 'FAIL'}] Logits: "
            f"KL mean/max={self.kl_divergence:.6g}/{self.max_kl_divergence:.6g}, "
            f"MAE={self.mean_absolute_error:.6g}, cos min={self.min_cosine_similarity:.6g}"
        )


@dataclass
class TokenValidation:
    """Greedy agreement, counting missing/additional suffixes as differences."""

    total_tokens: int
    matched_tokens: int
    agreement_rate: float
    first_divergence_pos: int | None
    passed: bool
    threshold: float
    reference_token_count: int = 0
    candidate_token_count: int = 0

    def summary(self) -> str:
        """Describe the complete token comparison."""
        return (
            f"[{'PASS' if self.passed else 'FAIL'}] Tokens: "
            f"{self.matched_tokens}/{self.total_tokens} matched "
            f"({100 * self.agreement_rate:.1f}%), first divergence={self.first_divergence_pos}"
        )


@dataclass
class PerplexityValidation:
    """Fixed-label PPL; None marks an absolute value outside float64 range."""

    reference_perplexity: float | None
    candidate_perplexity: float | None
    delta: float | None
    relative_delta: float | None
    passed: bool
    threshold_relative_delta: float
    num_tokens: int
    reference_mean_nll: float = 0.0
    candidate_mean_nll: float = 0.0
    log_ppl_ratio: float = 0.0
    overflow_fields: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Preserve log-space evidence when absolute PPL is unrepresentable."""
        return (
            f"[{'PASS' if self.passed else 'FAIL'}] Perplexity: "
            f"ref={self.reference_perplexity}, candidate={self.candidate_perplexity}, "
            f"log ratio={self.log_ppl_ratio:.6g}, overflow={self.overflow_fields}"
        )


@dataclass
class ActivationValidation:
    """Relative L2 errors; changed zero-reference activations have infinite error."""

    layer_errors: dict[str, float]
    max_layer_error: float
    mean_layer_error: float
    high_error_layers: list[str]
    passed: bool
    threshold: float

    def summary(self) -> str:
        """Describe every supplied layer's measured error."""
        return (
            f"[{'PASS' if self.passed else 'FAIL'}] Activations: "
            f"max={self.max_layer_error:.6g}, high-error layers={self.high_error_layers}"
        )


def _real(value: Any, low: float = 0.0, high: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        numeric = float(value)
    except OverflowError:
        return False  # An unrepresentable metric cannot support a passing report.
    return math.isfinite(numeric) and numeric >= low and (high is None or numeric <= high)


def _count(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral) and int(value) > 0


def _same_metric(value: Any, expected: float | None) -> bool:
    if expected is None:
        return value is None
    return _real(value, -math.inf) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def _component_passed(item: Any) -> bool:
    # Public mutable DTOs must not trust stale flags or empty observations.
    if getattr(item, "passed", None) is not True:
        return False
    if isinstance(item, TokenValidation):
        return (
            _count(item.reference_token_count)
            and _count(item.candidate_token_count)
            and _count(item.total_tokens)
            and item.total_tokens == max(item.reference_token_count, item.candidate_token_count)
            and _real(
                item.matched_tokens, 0, min(item.reference_token_count, item.candidate_token_count)
            )
            and isinstance(item.matched_tokens, Integral)
            and _real(item.agreement_rate, 0, 1)
            and _real(item.threshold, 0, 1)
            and math.isclose(item.agreement_rate, item.matched_tokens / item.total_tokens)
            and item.agreement_rate >= item.threshold
            and (
                item.first_divergence_pos is None
                if item.matched_tokens == item.total_tokens
                else (
                    isinstance(item.first_divergence_pos, Integral)
                    and _real(item.first_divergence_pos, 0, item.total_tokens - 1)
                )
            )
        )
    if isinstance(item, LogitValidation):
        return (
            _count(item.num_positions)
            and all(
                _real(value)
                for value in (
                    item.kl_divergence,
                    item.max_kl_divergence,
                    item.mean_absolute_error,
                    item.max_absolute_error,
                    item.threshold_kl,
                    item.threshold_mae,
                )
            )
            and all(
                _real(value, -1, 1)
                for value in (
                    item.cosine_similarity,
                    item.min_cosine_similarity,
                    item.threshold_cosine,
                )
            )
            and _real(item.top1_agreement_rate, 0, 1)
            and _real(item.top5_agreement_rate, 0, 1)
            and item.max_kl_divergence <= item.threshold_kl
            and item.mean_absolute_error <= item.threshold_mae
            and item.min_cosine_similarity >= item.threshold_cosine
            and item.kl_divergence <= item.max_kl_divergence + 1e-12
            and item.mean_absolute_error <= item.max_absolute_error + 1e-12
            and item.min_cosine_similarity <= item.cosine_similarity + 1e-12
            and isinstance(item.top1_agreement, bool)
            and item.top1_agreement == (item.top1_agreement_rate == 1.0)
        )
    if isinstance(item, PerplexityValidation):
        valid = (
            _count(item.num_tokens)
            and _real(item.reference_mean_nll)
            and _real(item.candidate_mean_nll)
            and _real(item.log_ppl_ratio, -float("inf"))
            and _real(item.threshold_relative_delta)
            and math.isclose(
                item.log_ppl_ratio, item.candidate_mean_nll - item.reference_mean_nll, abs_tol=1e-12
            )
            and item.log_ppl_ratio <= math.log1p(item.threshold_relative_delta)
            and (item.reference_perplexity is None or _real(item.reference_perplexity, 1))
            and (item.candidate_perplexity is None or _real(item.candidate_perplexity, 1))
            and (item.delta is None or _real(item.delta, -float("inf")))
            and (item.relative_delta is None or _real(item.relative_delta, -1))
        )
        if not valid:
            return False
        ref_ppl, cand_ppl = _exp(item.reference_mean_nll), _exp(item.candidate_mean_nll)
        delta = cand_ppl - ref_ppl if ref_ppl is not None and cand_ppl is not None else None
        try:
            relative = math.expm1(item.log_ppl_ratio)
        except OverflowError:
            relative = None
        expected = {
            "reference_perplexity": ref_ppl,
            "candidate_perplexity": cand_ppl,
            "delta": delta,
            "relative_delta": relative,
        }
        return (
            all(_same_metric(getattr(item, key), value) for key, value in expected.items())
            and isinstance(item.overflow_fields, list)
            and all(isinstance(name, str) for name in item.overflow_fields)
            and set(item.overflow_fields)
            == {key for key, value in expected.items() if value is None}
        )
    if isinstance(item, ActivationValidation):
        return (
            isinstance(item.layer_errors, dict)
            and bool(item.layer_errors)
            and _real(item.threshold)
            and _real(item.max_layer_error)
            and _real(item.mean_layer_error)
            and item.high_error_layers == []
            and all(_real(error, 0, item.threshold) for error in item.layer_errors.values())
            and item.max_layer_error <= item.threshold
            and _same_metric(item.max_layer_error, max(item.layer_errors.values()))
            and _same_metric(
                item.mean_layer_error,
                math.fsum(error / len(item.layer_errors) for error in item.layer_errors.values()),
            )
        )
    return False


@dataclass
class ValidationResult:
    """Supplied checks' results; an empty object cannot assert success."""

    logits: LogitValidation | None = None
    tokens: TokenValidation | None = None
    perplexity: PerplexityValidation | None = None
    activations: ActivationValidation | None = None
    passed: bool = False
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    required_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.required_checks = tuple(self.required_checks)
        self.passed = self.passed is True and self.evidence_passed

    @property
    def evidence_passed(self) -> bool:
        """Require at least one completed check and no failed check."""
        components = {
            "logits": (self.logits, LogitValidation),
            "tokens": (self.tokens, TokenValidation),
            "perplexity": (self.perplexity, PerplexityValidation),
            "activations": (self.activations, ActivationValidation),
        }
        supplied = {name for name, (item, _) in components.items() if item is not None}
        return (
            bool(supplied)
            and set(self.required_checks) <= supplied
            and all(
                isinstance(item, kind) and _component_passed(item)
                for item, kind in components.values()
                if item is not None
            )
        )

    def summary(self) -> str:
        """Summarize only checks that actually ran."""
        lines = ["Quality validation of supplied evidence"]
        lines.extend(
            item.summary()
            for item in (self.logits, self.tokens, self.perplexity, self.activations)
            if item is not None
        )
        lines.append(
            f"Overall: {'PASS' if self.passed is True and self.evidence_passed else 'FAIL'}"
        )
        lines.append(f"Duration: {self.duration_ms:.1f} ms")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Keep legacy keys and serialize overflow with explicit markers."""
        nonfinite: dict[str, str] = {}
        payload: dict[str, Any] = {
            "passed": self.passed is True and self.evidence_passed,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "required_checks": list(self.required_checks),
        }
        for name in ("logits", "tokens", "perplexity", "activations"):
            item = getattr(self, name)
            payload[name] = _metric_payload(asdict(item), nonfinite, name) if item else None
        if self.tokens is not None:
            payload["tokens"].update(
                matched=self.tokens.matched_tokens, total=self.tokens.total_tokens
            )
        if self.perplexity is not None:
            relative = self.perplexity.relative_delta
            payload["perplexity"].update(
                reference=self.perplexity.reference_perplexity,
                candidate=self.perplexity.candidate_perplexity,
                relative_delta_pct=100 * relative if relative is not None else None,
            )
            payload["perplexity"] = _metric_payload(payload["perplexity"], nonfinite, "perplexity")
        payload["nonfinite_metrics"] = nonfinite
        return payload


@dataclass
class QualityReport:
    """Accumulated supplied-evidence checks; an unexecuted report fails closed."""

    runs: list[ValidationResult] = field(default_factory=list)
    optimization_name: str = ""

    @property
    def all_passed(self) -> bool:
        """Require actual runs whose completed checks passed."""
        return bool(self.runs) and all(
            run.passed is True and run.evidence_passed for run in self.runs
        )

    @property
    def pass_rate(self) -> float:
        """Return the passing fraction, or zero when no runs exist."""
        if not self.runs:
            return 0.0
        return sum(run.passed is True and run.evidence_passed for run in self.runs) / len(self.runs)

    def save(self, path: str | Path) -> None:
        """Write strict JSON; invalid caller metadata raises instead of hiding it."""
        Path(path).write_text(
            json.dumps(
                {
                    "optimization": self.optimization_name,
                    "all_passed": self.all_passed,
                    "pass_rate": self.pass_rate,
                    "runs": [run.to_dict() for run in self.runs],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


class QualityValidator:
    """Validate paired measurements with explicit thresholds and no absent checks."""

    def __init__(
        self,
        max_kl_div: float = 0.01,
        max_mae: float = 0.1,
        min_cosine: float = 0.999,
        min_token_agreement: float = 0.95,
        max_perplexity_delta: float = 0.01,
        max_activation_error: float = 0.05,
    ) -> None:
        self.max_kl_div = _scalar(max_kl_div, "max_kl_div", 0.0)
        self.max_mae = _scalar(max_mae, "max_mae", 0.0)
        self.min_cosine = _scalar(min_cosine, "min_cosine", -1.0, 1.0)
        self.min_token_agreement = _scalar(min_token_agreement, "min_token_agreement", 0.0, 1.0)
        self.max_perplexity_delta = _scalar(max_perplexity_delta, "max_perplexity_delta", 0.0)
        self.max_activation_error = _scalar(max_activation_error, "max_activation_error", 0.0)

    def _validate_policy(self) -> None:
        # Public threshold fields historically remain mutable. Revalidate them
        # at use so a later NaN/Infinity assignment cannot disable a gate.
        for value, name, low, high in (
            (self.max_kl_div, "max_kl_div", 0.0, None),
            (self.max_mae, "max_mae", 0.0, None),
            (self.min_cosine, "min_cosine", -1.0, 1.0),
            (self.min_token_agreement, "min_token_agreement", 0.0, 1.0),
            (self.max_perplexity_delta, "max_perplexity_delta", 0.0, None),
            (self.max_activation_error, "max_activation_error", 0.0, None),
        ):
            _scalar(value, name, low, high)

    def validate_logits(
        self,
        reference: Any,
        candidate: Any,
        temperature: float = 1.0,
    ) -> LogitValidation:
        """Compare equal finite (vocab,) or (positions,vocab) arrays.

        KL(P||Q)=sum(P*(log(P)-log(Q))) follows Blei et al. Eq. (11) and
        PyTorch v2.9.1 kl_div(log_target=True). No epsilon ceiling is used.
        Worst-position KL/cosine gate prevents dilution by unchanged positions.
        Identical zero vectors have cosine 1 by exact-equality convention;
        one zero vector has cosine 0. Nonzero cosine uses scaled vectors.
        """
        self._validate_policy()
        temperature = _scalar(temperature, "temperature", 0.0)
        if temperature == 0:
            raise ValueError("temperature must be positive")
        ref, cand = _array(reference, "reference logits"), _array(candidate, "candidate logits")
        if ref.shape != cand.shape or ref.ndim not in (1, 2):
            raise ValueError("Logits must have identical 1D or 2D shapes")
        if ref.ndim == 1:
            ref, cand = ref[None, :], cand[None, :]
        ref_log, cand_log = _log_probs(ref, temperature), _log_probs(cand, temperature)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                kl = np.maximum(0.0, np.sum(np.exp(ref_log) * (ref_log - cand_log), axis=-1))
                errors = np.abs(ref - cand)
            except FloatingPointError as exc:
                raise ValueError("Logit differences exceed measurement range") from exc
        cosines: list[float] = []
        for r, c in zip(ref, cand, strict=True):
            r_max, c_max = float(np.max(np.abs(r))), float(np.max(np.abs(c)))
            if r_max == 0 or c_max == 0:
                cosines.append(1.0 if r_max == c_max else 0.0)
            else:
                r_scaled, c_scaled = r / r_max, c / c_max
                cosine = np.dot(
                    r_scaled / np.linalg.norm(r_scaled), c_scaled / np.linalg.norm(c_scaled)
                )
                cosines.append(float(np.clip(cosine, -1.0, 1.0)))
        top1 = ref.argmax(axis=-1) == cand.argmax(axis=-1)
        k = min(5, ref.shape[-1])
        top_k = [
            len(set(np.argsort(r)[-k:]) & set(np.argsort(c)[-k:])) / k
            for r, c in zip(ref, cand, strict=True)
        ]
        max_kl, min_cosine, mae = float(kl.max()), min(cosines), _mean(errors)
        return LogitValidation(
            kl_divergence=_mean(kl),
            max_absolute_error=float(errors.max()),
            mean_absolute_error=mae,
            cosine_similarity=math.fsum(cosines) / len(cosines),
            top1_agreement=bool(top1.all()),
            top5_agreement_rate=math.fsum(top_k) / len(top_k),
            passed=max_kl <= self.max_kl_div
            and mae <= self.max_mae
            and min_cosine >= self.min_cosine,
            threshold_kl=self.max_kl_div,
            threshold_mae=self.max_mae,
            max_kl_divergence=max_kl,
            min_cosine_similarity=min_cosine,
            top1_agreement_rate=float(top1.mean()),
            num_positions=len(ref),
            threshold_cosine=self.min_cosine,
        )

    def validate_greedy_agreement(
        self,
        reference_tokens: list[int],
        candidate_tokens: list[int],
    ) -> TokenValidation:
        """Compare complete sequences; absent suffixes are disagreements."""
        self._validate_policy()
        ref = _tokens(reference_tokens, "reference tokens", allow_empty=True)
        cand = _tokens(candidate_tokens, "candidate tokens", allow_empty=True)
        n = max(len(ref), len(cand))
        if n == 0:
            raise ValueError("Greedy evidence must not be empty on both sides")
        matches = sum(r == c for r, c in zip(ref, cand, strict=False))
        first = next(
            (index for index, (r, c) in enumerate(zip(ref, cand, strict=False)) if r != c), None
        )
        if first is None and len(ref) != len(cand):
            first = min(len(ref), len(cand))
        rate = matches / n
        return TokenValidation(
            n,
            matches,
            rate,
            first,
            bool(ref) and bool(cand) and rate >= self.min_token_agreement,
            self.min_token_agreement,
            len(ref),
            len(cand),
        )

    def validate_perplexity(
        self,
        reference_nll_per_token: ArrayLike,
        candidate_nll_per_token: ArrayLike,
    ) -> PerplexityValidation:
        """Gate PPL ratio in log space without capping mean NLL.

        PPL=exp(mean NLL) on the same labels implies relative change equals
        expm1(candidate_mean-reference_mean). Compare log_ratio against
        log1p(allowed_relative_change), deriving from Radford Eq. (1) and HF's
        fixed-label definition. Absolute overflow yields None plus markers;
        finite mean NLLs and the log-ratio gate remain available.
        """
        self._validate_policy()
        ref = _array(reference_nll_per_token, "reference NLL")
        cand = _array(candidate_nll_per_token, "candidate NLL")
        if ref.ndim != 1 or ref.shape != cand.shape or np.any(ref < 0) or np.any(cand < 0):
            raise ValueError("NLL evidence must be equal nonnegative vectors on the same labels")
        ref_mean, cand_mean = _mean(ref), _mean(cand)
        difference = cand_mean - ref_mean
        ref_ppl, cand_ppl = _exp(ref_mean), _exp(cand_mean)
        try:
            relative = math.expm1(difference)
        except OverflowError:
            relative = None
        delta = cand_ppl - ref_ppl if cand_ppl is not None and ref_ppl is not None else None
        values = {
            "reference_perplexity": ref_ppl,
            "candidate_perplexity": cand_ppl,
            "delta": delta,
            "relative_delta": relative,
        }
        return PerplexityValidation(
            reference_perplexity=ref_ppl,
            candidate_perplexity=cand_ppl,
            delta=delta,
            relative_delta=relative,
            passed=difference <= math.log1p(self.max_perplexity_delta),
            threshold_relative_delta=self.max_perplexity_delta,
            num_tokens=ref.size,
            reference_mean_nll=ref_mean,
            candidate_mean_nll=cand_mean,
            log_ppl_ratio=difference,
            overflow_fields=[name for name, value in values.items() if value is None],
        )

    def validate_activations(
        self,
        reference_activations: dict[str, Any],
        candidate_activations: dict[str, Any],
    ) -> ActivationValidation:
        """Compare every layer; missing or invalid evidence is never skipped."""
        self._validate_policy()
        if (
            not reference_activations
            or reference_activations.keys() != candidate_activations.keys()
        ):
            raise ValueError("Activation evidence requires equal nonempty layer sets")
        errors: dict[str, float] = {}
        for name, reference in reference_activations.items():
            ref, cand = (
                _array(reference, f"{name} reference"),
                _array(candidate_activations[name], f"{name} candidate"),
            )
            if ref.shape != cand.shape:
                raise ValueError(f"Activation shape differs for {name}")
            scale = max(float(np.max(np.abs(ref))), float(np.max(np.abs(cand))))
            if scale == 0:
                errors[name] = 0.0
                continue
            normalized_ref, normalized_cand = ref / scale, cand / scale
            norm = float(np.linalg.norm(normalized_ref.ravel()))
            errors[name] = (
                float(np.linalg.norm((normalized_ref - normalized_cand).ravel())) / norm
                if norm > 0
                else math.inf
            )
        high = [name for name, error in errors.items() if error > self.max_activation_error]
        return ActivationValidation(
            errors,
            max(errors.values()),
            math.fsum(error / len(errors) for error in errors.values()),
            high,
            not high,
            self.max_activation_error,
        )

    def validate_all(
        self,
        reference_logits: Any | None = None,
        candidate_logits: Any | None = None,
        reference_tokens: list[int] | None = None,
        candidate_tokens: list[int] | None = None,
        reference_nll: ArrayLike | None = None,
        candidate_nll: ArrayLike | None = None,
        reference_activations: dict[str, Any] | None = None,
        candidate_activations: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        required_checks: tuple[str, ...] = (),
    ) -> ValidationResult:
        """Run paired checks; missing counterparts or required evidence raise.

        Diagnostics can supply a subset. Production admission must explicitly
        require complete evidence and authenticate the measured subjects.
        """
        started = time.perf_counter()
        pairs = {
            "logits": (reference_logits, candidate_logits),
            "tokens": (reference_tokens, candidate_tokens),
            "perplexity": (reference_nll, candidate_nll),
            "activations": (reference_activations, candidate_activations),
        }
        supplied = {name for name, pair in pairs.items() if any(item is not None for item in pair)}
        if not supplied:
            raise ValueError("No quality evidence was supplied")
        if not set(required_checks) <= pairs.keys():
            raise ValueError("Unknown required quality check")
        if not set(required_checks) <= supplied:
            raise ValueError("Required quality evidence is missing")
        for name in supplied:
            if any(item is None for item in pairs[name]):
                raise ValueError(f"{name} requires both reference and candidate evidence")
        result = ValidationResult(metadata=dict(metadata or {}), required_checks=required_checks)
        if reference_logits is not None and candidate_logits is not None:
            result.logits = self.validate_logits(reference_logits, candidate_logits)
        if reference_tokens is not None and candidate_tokens is not None:
            result.tokens = self.validate_greedy_agreement(reference_tokens, candidate_tokens)
        if reference_nll is not None and candidate_nll is not None:
            result.perplexity = self.validate_perplexity(reference_nll, candidate_nll)
        if reference_activations is not None and candidate_activations is not None:
            result.activations = self.validate_activations(
                reference_activations, candidate_activations
            )
        result.passed = result.evidence_passed
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    def compare_engines(
        self,
        reference_engine: Any,
        candidate_engine: Any,
        token_ids: list[int],
        max_tokens: int = 32,
        *,
        continuation_ids: list[int] | None = None,
    ) -> ValidationResult:
        """Compare append-only engines with isolated caches and shared history.

        Prefill once, then append one ID per greedy step. Fixed continuation
        labels score identical histories (Radford Eq. (1), HF next-token loss).
        Without labels, reference greedy history supplies diagnostic logits;
        PPL stays absent. This fixed-count probe does not implement EOS stopping
        or certify corpus quality. Real engine exceptions propagate.
        """
        started = time.perf_counter()
        self._validate_policy()
        prompt = _tokens(token_ids, "prompt")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        labels = _tokens(continuation_ids, "continuation") if continuation_ids is not None else None

        def step(engine: Any, ids: list[int], cache: Any) -> tuple[FloatArray, Any]:
            logits, updated = engine.forward_step(ids, cache, return_all_logits=False)
            row = _array(logits, "engine logits")
            if row.ndim != 1 or updated is None:
                raise ValueError(
                    "forward_step must return a vocabulary vector and append-only cache"
                )
            # Engines may legitimately reuse a float64 output scratch buffer.
            # Each recorded row must retain the values measured at this step.
            return row.copy(), updated

        def decode(engine: Any) -> tuple[list[int], FloatArray]:
            cache = engine.init_kv_cache() if hasattr(engine, "init_kv_cache") else None
            inputs, generated, rows = prompt, [], []
            for _ in range(max_tokens):
                row, cache = step(engine, inputs, cache)
                rows.append(row)
                token = int(np.argmax(row))
                generated.append(token)
                inputs = [token]
            return generated, np.stack(rows)

        def teacher(engine: Any, following: list[int]) -> FloatArray:
            cache = engine.init_kv_cache() if hasattr(engine, "init_kv_cache") else None
            rows = []
            for index in range(len(following)):
                inputs = prompt if index == 0 else [following[index - 1]]
                row, cache = step(engine, inputs, cache)
                rows.append(row)
            return np.stack(rows)

        ref_tokens, ref_greedy = decode(reference_engine)
        cand_tokens, cand_greedy = decode(candidate_engine)
        if ref_greedy.shape != cand_greedy.shape:
            raise ValueError("Compared engines must have the same vocabulary")
        if labels is not None and max(labels) >= ref_greedy.shape[-1]:
            raise ValueError("Fixed labels are outside the engines' vocabulary")
        following = labels if labels is not None else ref_tokens
        ref_rows = teacher(reference_engine, following) if labels is not None else ref_greedy
        cand_rows = teacher(candidate_engine, following)
        ref_nll, cand_nll = None, None
        if labels is not None:
            if ref_rows.shape != cand_rows.shape or max(labels) >= ref_rows.shape[-1]:
                raise ValueError("Fixed-label evidence requires equal vocabulary and valid IDs")
            ref_nll = [
                -float(_log_probs(row)[label]) for row, label in zip(ref_rows, labels, strict=True)
            ]
            cand_nll = [
                -float(_log_probs(row)[label]) for row, label in zip(cand_rows, labels, strict=True)
            ]
        result = self.validate_all(
            reference_logits=ref_rows,
            candidate_logits=cand_rows,
            reference_tokens=ref_tokens,
            candidate_tokens=cand_tokens,
            reference_nll=ref_nll,
            candidate_nll=cand_nll,
            required_checks=("logits", "tokens", "perplexity")
            if labels is not None
            else ("logits", "tokens"),
            metadata={
                "prompt_ids": prompt,
                "continuation_ids": labels,
                "logit_history": "fixed_labels" if labels is not None else "reference_greedy",
                "perplexity_status": "fixed_label_continuation"
                if labels is not None
                else "not_evaluated_no_fixed_labels",
                "greedy_scope": "fixed-count forward_step; no EOS stopping",
            },
        )
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result
