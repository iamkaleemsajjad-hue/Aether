"""Production observability primitives for compiled AEG deployments."""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Request and quality metrics captured by the runtime."""

    tokens_per_second: float
    ttft_ms: float
    spec_accept_rate: float
    kv_hit_rate: float
    mla_compression_ratio: float
    reasoning_budget_used: float
    gpu_vram_utilization: float
    win_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "ttft_ms": self.ttft_ms,
            "spec_accept_rate": self.spec_accept_rate,
            "kv_hit_rate": self.kv_hit_rate,
            "mla_compression_ratio": self.mla_compression_ratio,
            "reasoning_budget_used": self.reasoning_budget_used,
            "gpu_vram_utilization": self.gpu_vram_utilization,
            "win_rate": self.win_rate,
        }


@dataclass(frozen=True)
class EvalResult:
    """A single benchmark outcome for an eval gate."""

    benchmark: str
    baseline_score: float
    candidate_score: float
    higher_is_better: bool = True

    @property
    def regression(self) -> float:
        if self.higher_is_better:
            return max(0.0, self.baseline_score - self.candidate_score)
        return max(0.0, self.candidate_score - self.baseline_score)

    @property
    def relative_regression(self) -> float:
        return self.regression / max(abs(self.baseline_score), 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "higher_is_better": self.higher_is_better,
            "regression": round(self.regression, 6),
            "relative_regression": round(self.relative_regression, 6),
        }


@dataclass(frozen=True)
class EvalGateDecision:
    """Result of an eval gate decision."""

    passed: bool
    max_relative_regression: float
    failing_benchmarks: tuple[str, ...]
    results: tuple[EvalResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_relative_regression": round(self.max_relative_regression, 6),
            "failing_benchmarks": list(self.failing_benchmarks),
            "results": [result.to_dict() for result in self.results],
        }


class EvalGate:
    """Quality gate that blocks rollout when benchmark regressions are too high."""

    def __init__(self, max_relative_regression: float = 0.02, required_benchmarks: tuple[str, ...] | None = None) -> None:
        if max_relative_regression < 0:
            raise ValueError("max_relative_regression must be non-negative")
        self.max_relative_regression = max_relative_regression
        self.required_benchmarks = required_benchmarks or ("hellaswag", "mmlu", "gsm8k", "math-500", "humaneval")

    def evaluate(self, results: list[EvalResult]) -> EvalGateDecision:
        result_by_name = {result.benchmark: result for result in results}
        missing = [name for name in self.required_benchmarks if name not in result_by_name]
        failing = [
            result.benchmark
            for result in results
            if result.relative_regression > self.max_relative_regression
        ]
        failing.extend(f"missing:{name}" for name in missing)
        max_regression = max((result.relative_regression for result in results), default=0.0)
        return EvalGateDecision(
            passed=not failing,
            max_relative_regression=max_regression,
            failing_benchmarks=tuple(failing),
            results=tuple(results),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "max_relative_regression": self.max_relative_regression,
            "required_benchmarks": list(self.required_benchmarks),
            "action": "block_rollout_on_failure",
        }


class DriftMonitor:
    """Detect live quality drift from win-rate and latency telemetry."""

    def __init__(self, baseline_win_rate: float, alert_drop: float = 0.05, min_samples: int = 20) -> None:
        self.baseline_win_rate = baseline_win_rate
        self.alert_drop = alert_drop
        self.min_samples = min_samples
        self._snapshots: list[TelemetrySnapshot] = []

    def record(self, snapshot: TelemetrySnapshot) -> dict[str, Any]:
        self._snapshots.append(snapshot)
        return self.status()

    def status(self) -> dict[str, Any]:
        win_rates = [snapshot.win_rate for snapshot in self._snapshots if snapshot.win_rate is not None]
        live_win_rate = statistics.fmean(win_rates) if win_rates else None
        drift = None if live_win_rate is None else self.baseline_win_rate - live_win_rate
        alert = bool(len(win_rates) >= self.min_samples and drift is not None and drift > self.alert_drop)
        return {
            "sample_count": len(win_rates),
            "baseline_win_rate": self.baseline_win_rate,
            "live_win_rate": live_win_rate,
            "drift": drift,
            "alert": alert,
            "alert_drop": self.alert_drop,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "baseline_win_rate": self.baseline_win_rate,
            "alert_drop": self.alert_drop,
            "min_samples": self.min_samples,
            "signals": ["win_rate", "ttft_ms", "tokens_per_second", "spec_accept_rate"],
        }


class ABRolloutController:
    """Deterministic A/B traffic splitter with safe ramp decisions."""

    def __init__(self, experiment_id: str, candidate_percent: float = 0.01) -> None:
        if not 0.0 <= candidate_percent <= 1.0:
            raise ValueError("candidate_percent must be between 0 and 1")
        self.experiment_id = experiment_id
        self.candidate_percent = candidate_percent

    def assign(self, request_id: str) -> str:
        digest = hashlib.sha256(f"{self.experiment_id}:{request_id}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return "candidate" if bucket < self.candidate_percent else "control"

    def ramp(self, gate_passed: bool, drift_alert: bool) -> float:
        if not gate_passed or drift_alert:
            self.candidate_percent = 0.0
        elif self.candidate_percent < 0.01:
            self.candidate_percent = 0.01
        else:
            self.candidate_percent = min(1.0, self.candidate_percent * 2)
        return self.candidate_percent

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "experiment_id": self.experiment_id,
            "candidate_percent": self.candidate_percent,
            "assignment": "sha256_stable_bucket",
            "rollback_on": ["eval_gate_failure", "quality_drift_alert", "safety_alert"],
        }


# ---------------------------------------------------------------------------
# QualityGate — score-threshold API (wraps EvalGate)
# ---------------------------------------------------------------------------

@dataclass
class QualityGateResult:
    """Result of a QualityGate evaluation."""

    benchmark: str
    score: float
    threshold: float
    blocked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "blocked": self.blocked,
        }


class QualityGate:
    """Score-threshold quality gate.

    Blocks a model when its benchmark score falls below the configured
    threshold.  Provides a simpler interface than EvalGate for cases where
    there is no separate baseline measurement.

    Usage::

        gate = QualityGate(threshold=0.70)
        result = evaluator.run()           # EvalResult from evaluators
        if gate.should_block(result):
            raise ValueError("Quality gate blocked rollout")
    """

    def __init__(self, threshold: float = 0.70, benchmark: str | None = None) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.benchmark = benchmark

    def should_block(self, result: Any) -> bool:
        """Return True if the result fails to meet the quality threshold.

        Args:
            result: Any object with a ``score`` attribute (e.g.,
                    aether.observability.evaluators.EvalResult).
        """
        score = float(getattr(result, "score", 0.0))
        return score < self.threshold

    def evaluate(self, result: Any) -> QualityGateResult:
        """Return a structured gate decision."""
        score = float(getattr(result, "score", 0.0))
        benchmark = getattr(result, "benchmark", self.benchmark or "unknown")
        blocked = score < self.threshold
        return QualityGateResult(
            benchmark=benchmark,
            score=score,
            threshold=self.threshold,
            blocked=blocked,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "threshold": self.threshold,
            "benchmark": self.benchmark,
            "action": "block_rollout_on_failure",
        }

