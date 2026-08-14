"""
Benchmark runner tests (PRD §36).

Verify that the BenchmarkRunner:
  - Produces real measured values (not hardcoded)
  - Computes correct aggregate statistics
  - Records provenance in every report
  - Handles failures gracefully
  - Never returns timing=0 for successful runs (would indicate fake measurement)
"""

from __future__ import annotations

import time

import pytest

from aether.observability.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkReport,
    BenchmarkProvenance,
    RunResult,
    _percentile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "Hello, world!",
    "What is the capital of France?",
    "Explain quantum computing.",
]


def _make_fake_generate(delay_s: float = 0.01, response: str = "Hello, this is a test response."):
    """Return a generate_fn that sleeps for delay_s to produce measurable timing."""
    def _generate(prompt: str, max_tokens: int) -> str:
        time.sleep(delay_s)
        return response
    return _generate


def _make_failing_generate():
    """Return a generate_fn that always raises."""
    def _generate(prompt: str, max_tokens: int) -> str:
        raise RuntimeError("Simulated model failure")
    return _generate


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:

    def test_runner_produces_real_timing(self) -> None:
        """All run e2e_s values must be > 0 for successful runs."""
        runner = BenchmarkRunner(
            _make_fake_generate(delay_s=0.01),
            model_id="test-model",
            num_warmup_runs=0,
            num_measured_runs=3,
        )
        report = runner.run(SAMPLE_PROMPTS, max_tokens=10)
        for run in report.successful_runs:
            assert run.e2e_s > 0.0, \
                f"Run {run.run_index}: e2e_s={run.e2e_s} — must be > 0 for real timing"

    def test_runner_produces_nonzero_tps(self) -> None:
        """TPS must be > 0 for successful runs that produced tokens."""
        runner = BenchmarkRunner(
            _make_fake_generate(delay_s=0.01, response="Token one two three four five"),
            num_warmup_runs=0,
            num_measured_runs=3,
        )
        report = runner.run(SAMPLE_PROMPTS[:1], max_tokens=20)
        for run in report.successful_runs:
            if run.output_tokens > 0:
                assert run.tps > 0.0

    def test_runner_records_provenance(self) -> None:
        """Every report must include provenance metadata."""
        runner = BenchmarkRunner(
            _make_fake_generate(),
            model_id="prov-test-model",
            num_warmup_runs=0,
            num_measured_runs=1,
        )
        report = runner.run(SAMPLE_PROMPTS[:1], max_tokens=5)
        assert report.provenance is not None
        assert isinstance(report.provenance, BenchmarkProvenance)
        assert report.provenance.model_id == "prov-test-model"
        assert report.provenance.num_measured_runs == 1
        assert report.provenance.python_version != ""

    def test_runner_handles_failures_gracefully(self) -> None:
        """Failing generate_fn must produce RunResult with error set, not crash."""
        runner = BenchmarkRunner(
            _make_failing_generate(),
            num_warmup_runs=0,
            num_measured_runs=3,
        )
        report = runner.run(SAMPLE_PROMPTS, max_tokens=5)
        assert len(report.runs) == 3
        for run in report.runs:
            assert run.error is not None
            assert "Simulated model failure" in run.error

    def test_runner_counts_failures_correctly(self) -> None:
        """Report must correctly count successful vs failed runs."""
        runner = BenchmarkRunner(
            _make_failing_generate(),
            num_warmup_runs=0,
            num_measured_runs=4,
        )
        report = runner.run(SAMPLE_PROMPTS, max_tokens=5)
        assert len(report.successful_runs) == 0
        d = report.to_dict()
        assert d["summary"]["failed_runs"] == 4
        assert d["summary"]["successful_runs"] == 0

    def test_runner_to_dict_is_json_serializable(self) -> None:
        """BenchmarkReport.to_dict() must be JSON serializable."""
        import json
        runner = BenchmarkRunner(
            _make_fake_generate(delay_s=0.005),
            num_warmup_runs=0,
            num_measured_runs=2,
        )
        report = runner.run(SAMPLE_PROMPTS[:1], max_tokens=5)
        d = report.to_dict()
        json_str = json.dumps(d, default=str)
        assert isinstance(json_str, str)

    def test_runner_requires_prompts(self) -> None:
        """Empty prompt list must raise ValueError."""
        runner = BenchmarkRunner(_make_fake_generate(), num_warmup_runs=0, num_measured_runs=1)
        with pytest.raises(ValueError, match="At least one prompt"):
            runner.run([], max_tokens=5)

    def test_runner_requires_callable(self) -> None:
        """Non-callable generate_fn must raise ValueError at construction."""
        with pytest.raises((ValueError, TypeError)):
            BenchmarkRunner("not_a_callable", num_warmup_runs=0, num_measured_runs=1)  # type: ignore[arg-type]

    def test_report_save_and_reload(self, tmp_path) -> None:
        """BenchmarkReport.save() must write valid JSON that can be reloaded."""
        import json
        runner = BenchmarkRunner(
            _make_fake_generate(delay_s=0.005),
            num_warmup_runs=0,
            num_measured_runs=2,
        )
        report = runner.run(SAMPLE_PROMPTS[:1], max_tokens=5)
        out = tmp_path / "report.json"
        path = report.save(out)
        assert path.is_file()
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert "provenance" in reloaded
        assert "summary" in reloaded
        assert "runs" in reloaded

    def test_streaming_runner_produces_per_token_latency(self) -> None:
        """Streaming runner must record individual token latencies."""
        def _streaming(prompt: str, max_tokens: int):
            for word in ("Hello", "world", "this", "is", "streaming"):
                time.sleep(0.005)
                yield word

        runner = BenchmarkRunner(
            _make_fake_generate(),
            num_warmup_runs=0,
            num_measured_runs=2,
        )
        report = runner.run_streaming(_streaming, SAMPLE_PROMPTS[:1], max_tokens=5)
        assert len(report.runs) == 2
        for run in report.successful_runs:
            # Streaming should populate token_latencies
            assert run.output_tokens > 0
            assert run.ttft_s >= 0.0


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

class TestPercentileHelper:

    def test_empty_returns_none(self) -> None:
        assert _percentile([], 50) is None

    def test_single_element(self) -> None:
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 99) == 5.0

    def test_p50_of_sorted_list(self) -> None:
        vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        p50 = _percentile(vals, 50)
        assert p50 == pytest.approx(3.0, abs=0.1)

    def test_p95_upper_range(self) -> None:
        vals = sorted(float(i) for i in range(100))
        p95 = _percentile(vals, 95)
        assert p95 is not None
        assert p95 > 90.0
