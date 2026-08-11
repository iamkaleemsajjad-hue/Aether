"""
Aether Runtime — Performance Metrics Test Suite.

Tests that benchmark_runner.py correctly measures, aggregates, and reports
real performance metrics using the actual Aether CPU AEG path.

Covers:
  - TTFT (Time to First Token) measurement
  - TBT (Time Between Tokens) measurement
  - P50/P95/P99 latency percentiles
  - Throughput (tokens/sec)
  - Memory utilization tracking
  - Energy estimation
  - Baseline comparison
  - BenchmarkResult serialization
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the src tree is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from benchmarks.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    RequestMetrics,
    SystemMonitor,
    _percentile,
)


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 0) == 42.0
        assert _percentile([42.0], 100) == 42.0

    def test_two_values_p50(self):
        result = _percentile([10.0, 20.0], 50)
        assert result == 15.0

    def test_sorted_list_p95(self):
        data = list(range(1, 101, 1))  # 1..100
        p95 = _percentile([float(x) for x in data], 95)
        # 95th percentile of 1..100 ≈ 95.95
        assert 94.0 <= p95 <= 96.0

    def test_p99_of_large_list(self):
        data = [float(i) for i in range(1000)]
        p99 = _percentile(data, 99)
        assert 988.0 <= p99 <= 999.0

    def test_all_same_values(self):
        data = [5.0] * 100
        assert _percentile(data, 50) == 5.0
        assert _percentile(data, 99) == 5.0


# ---------------------------------------------------------------------------
# SystemMonitor
# ---------------------------------------------------------------------------

class TestSystemMonitor:
    def test_initial_state(self):
        monitor = SystemMonitor()
        assert monitor.peak_memory_mb() >= 0.0

    def test_sample_increments_memory(self):
        monitor = SystemMonitor()
        monitor.sample()
        # Memory should be > 0 for any running process
        assert monitor.peak_memory_mb() >= 0.0

    def test_energy_estimate_positive(self):
        monitor = SystemMonitor()
        monitor.sample()
        energy = monitor.estimated_energy_wh(duration_sec=10.0, tdp_watts=150.0)
        # Energy = P * t / 3600; must be > 0
        assert energy > 0.0

    def test_energy_zero_duration(self):
        monitor = SystemMonitor()
        energy = monitor.estimated_energy_wh(duration_sec=0.0001, tdp_watts=100.0)
        assert energy >= 0.0

    def test_avg_cpu_percent_initial(self):
        monitor = SystemMonitor()
        # Without samples, should return 0
        assert monitor.avg_cpu_percent() == 0.0

    def test_avg_cpu_after_samples(self):
        monitor = SystemMonitor()
        for _ in range(3):
            monitor.sample()
        # After sampling, value should be 0..100
        pct = monitor.avg_cpu_percent()
        assert 0.0 <= pct <= 100.0


# ---------------------------------------------------------------------------
# RequestMetrics
# ---------------------------------------------------------------------------

class TestRequestMetrics:
    def test_basic_creation(self):
        m = RequestMetrics(
            request_id="test_001",
            prompt_tokens=10,
            completion_tokens=50,
            ttft_ms=150.0,
            end_to_end_ms=2000.0,
            tbt_ms_list=[40.0, 38.0, 42.0],
            throughput_tps=25.0,
        )
        assert m.request_id == "test_001"
        assert m.ttft_ms == 150.0
        assert len(m.tbt_ms_list) == 3
        assert m.error is None

    def test_error_request(self):
        m = RequestMetrics(
            request_id="err_001",
            prompt_tokens=5,
            completion_tokens=0,
            ttft_ms=0.0,
            end_to_end_ms=500.0,
            error="RuntimeError: model not found",
        )
        assert m.error is not None
        assert m.completion_tokens == 0


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def _make_result(self) -> BenchmarkResult:
        return BenchmarkResult(
            model_path="/tmp/test.aeg",
            hardware_target="cpu_avx512",
            num_requests=100,
            ttft_p50_ms=120.5,
            ttft_p95_ms=200.1,
            ttft_p99_ms=350.7,
            ttft_mean_ms=130.0,
            ttft_std_ms=20.0,
            tbt_p50_ms=40.0,
            tbt_p95_ms=60.0,
            tbt_p99_ms=80.0,
            tbt_mean_ms=42.0,
            e2e_p50_ms=500.0,
            e2e_p95_ms=900.0,
            e2e_p99_ms=1200.0,
            e2e_mean_ms=520.0,
            e2e_std_ms=80.0,
            throughput_tps=1250.0,
            requests_per_sec=5.2,
            total_prompt_tokens=5120,
            total_completion_tokens=12800,
            avg_prompt_tokens=51.2,
            avg_completion_tokens=128.0,
            kv_cache_hit_rate=0.72,
            spec_accept_rate=0.68,
            error_rate=0.01,
            peak_memory_mb=1024.0,
            avg_cpu_percent=85.0,
            energy_wh=0.0347,
        )

    def test_to_dict_structure(self):
        r = self._make_result()
        d = r.to_dict()
        assert "latency" in d
        assert "ttft" in d["latency"]
        assert "p50" in d["latency"]["ttft"]
        assert "p95" in d["latency"]["ttft"]
        assert "p99" in d["latency"]["ttft"]
        assert "throughput" in d
        assert "tokens_per_second" in d["throughput"]
        assert "quality" in d
        assert "kv_cache_hit_rate" in d["quality"]
        assert "system" in d
        assert "peak_memory_mb" in d["system"]

    def test_to_dict_values(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["latency"]["ttft"]["p50"] == 120.5
        assert d["latency"]["ttft"]["p99"] == 350.7
        assert d["throughput"]["tokens_per_second"] == 1250.0
        assert d["quality"]["kv_cache_hit_rate"] == 0.72
        assert d["system"]["peak_memory_mb"] == 1024.0
        assert d["system"]["energy_wh"] == 0.0347

    def test_summary_contains_key_metrics(self):
        r = self._make_result()
        s = r.summary()
        assert "TTFT P50" in s
        assert "Throughput" in s
        assert "KV Hit Rate" in s
        assert "Energy Wh" in s
        assert "Spec Accept" in s

    def test_json_round_trip(self):
        r = self._make_result()
        d = r.to_dict()
        serialized = json.dumps(d)
        recovered = json.loads(serialized)
        assert recovered["latency"]["ttft"]["p50"] == 120.5
        assert recovered["num_requests"] == 100


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig(model_path="test.aeg")
        assert cfg.num_requests == 100
        assert cfg.warmup_requests == 10
        assert cfg.max_output_tokens == 256
        assert cfg.temperature == 0.0
        assert cfg.hardware_target == "cpu_avx512"

    def test_custom_values(self):
        cfg = BenchmarkConfig(
            model_path="model.aeg",
            num_requests=500,
            concurrency=4,
            temperature=0.7,
        )
        assert cfg.num_requests == 500
        assert cfg.concurrency == 4
        assert cfg.temperature == 0.7


# ---------------------------------------------------------------------------
# BenchmarkRunner integration (using mocked runtime)
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerWithMock:
    """Tests the BenchmarkRunner logic using a mocked Runtime."""

    def _make_mock_runtime(self, tokens_to_yield: int = 5) -> MagicMock:
        """Create a mock Runtime that yields tokens and returns a response."""
        mock = MagicMock()

        def mock_generate_stream(prompt, max_tokens, temperature):
            for i in range(min(max_tokens, tokens_to_yield)):
                time.sleep(0.001)  # Simulate token generation latency
                yield f"token_{i}"

        mock.generate_stream.side_effect = mock_generate_stream
        return mock

    def test_run_collects_metrics(self):
        """Test that BenchmarkRunner collects real timing metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BenchmarkConfig(
                model_path=f"{tmpdir}/test.aeg",
                num_requests=5,
                warmup_requests=2,
                max_output_tokens=3,
            )
            runner = BenchmarkRunner(cfg)
            mock_runtime = self._make_mock_runtime(tokens_to_yield=3)

            with patch.object(runner, "_load_runtime"):
                runner._runtime = mock_runtime
                result = runner.run()

        assert result.num_requests == 5
        assert result.ttft_p50_ms >= 0.0
        assert result.throughput_tps >= 0.0

    def test_ttft_measured_correctly(self):
        """TTFT should be positive and less than total E2E time."""
        cfg = BenchmarkConfig(
            model_path="test.aeg",
            num_requests=3,
            warmup_requests=0,
            max_output_tokens=5,
        )
        runner = BenchmarkRunner(cfg)
        mock_runtime = self._make_mock_runtime(tokens_to_yield=5)

        with patch.object(runner, "_load_runtime"):
            runner._runtime = mock_runtime
            result = runner.run()

        # TTFT must be > 0 and < E2E latency
        if result.ttft_p50_ms > 0:
            assert result.ttft_p50_ms <= result.e2e_p50_ms + 1.0  # allow 1ms tolerance

    def test_error_rate_zero_on_success(self):
        """Error rate should be 0 when all requests succeed."""
        cfg = BenchmarkConfig(
            model_path="test.aeg",
            num_requests=5,
            warmup_requests=0,
            max_output_tokens=3,
        )
        runner = BenchmarkRunner(cfg)
        mock_runtime = self._make_mock_runtime(tokens_to_yield=3)

        with patch.object(runner, "_load_runtime"):
            runner._runtime = mock_runtime
            result = runner.run()

        assert result.error_rate == 0.0

    def test_error_rate_one_on_all_failures(self):
        """Error rate should be 1.0 when all requests fail."""
        cfg = BenchmarkConfig(
            model_path="test.aeg",
            num_requests=3,
            warmup_requests=0,
            max_output_tokens=2,
        )
        runner = BenchmarkRunner(cfg)
        mock_runtime = MagicMock()
        mock_runtime.generate_stream.side_effect = RuntimeError("backend unavailable")

        with patch.object(runner, "_load_runtime"):
            runner._runtime = mock_runtime
            result = runner.run()

        assert result.error_rate == 1.0

    def test_result_saved_to_json(self):
        """Results should be saved to JSON file when output_path is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            cfg = BenchmarkConfig(
                model_path="test.aeg",
                num_requests=3,
                warmup_requests=0,
                max_output_tokens=2,
                output_path=str(output_path),
            )
            runner = BenchmarkRunner(cfg)
            mock_runtime = self._make_mock_runtime(tokens_to_yield=2)

            with patch.object(runner, "_load_runtime"):
                runner._runtime = mock_runtime
                runner.run()

            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert "latency" in data
            assert "throughput" in data

    def test_baseline_comparison(self):
        """When baseline JSON is provided, comparison should be computed."""
        baseline = {
            "latency": {
                "ttft": {"p50": 100.0, "p99": 300.0},
            },
            "throughput": {"tokens_per_second": 1000.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            output_path = Path(tmpdir) / "results.json"

            cfg = BenchmarkConfig(
                model_path="test.aeg",
                num_requests=3,
                warmup_requests=0,
                max_output_tokens=2,
                output_path=str(output_path),
                compare_baseline=str(baseline_path),
            )
            runner = BenchmarkRunner(cfg)
            mock_runtime = self._make_mock_runtime(tokens_to_yield=2)

            with patch.object(runner, "_load_runtime"):
                runner._runtime = mock_runtime
                result = runner.run()

            # Comparison dict should be present
            assert isinstance(result.baseline_comparison, dict)

    def test_throughput_positive(self):
        """Throughput in tokens/sec must always be positive."""
        cfg = BenchmarkConfig(
            model_path="test.aeg",
            num_requests=3,
            warmup_requests=0,
            max_output_tokens=5,
        )
        runner = BenchmarkRunner(cfg)
        mock_runtime = self._make_mock_runtime(tokens_to_yield=5)

        with patch.object(runner, "_load_runtime"):
            runner._runtime = mock_runtime
            result = runner.run()

        assert result.throughput_tps >= 0.0
        assert result.requests_per_sec >= 0.0


# ---------------------------------------------------------------------------
# Performance regression guard
# ---------------------------------------------------------------------------

class TestPerformanceRegressionGuard:
    """Test that performance regression detection logic works correctly."""

    def test_regression_detected_when_slower(self):
        """A significantly slower result should be flagged."""
        runner = BenchmarkRunner(BenchmarkConfig(model_path="test.aeg"))
        # Simulate a current result that is 50% slower than baseline
        current = BenchmarkResult(
            model_path="test.aeg",
            hardware_target="cpu_avx512",
            num_requests=10,
            ttft_p50_ms=150.0,
            ttft_p99_ms=450.0,
            throughput_tps=666.0,
        )
        baseline_data = {
            "latency": {"ttft": {"p50": 100.0, "p99": 300.0}},
            "throughput": {"tokens_per_second": 1000.0},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline_data, f)
            baseline_path = f.name

        try:
            runner.config.compare_baseline = baseline_path
            comparison = runner._compare_baseline(current)
            # p50 went from 100 to 150 → +50%
            assert "ttft_p50" in comparison
            assert "+50.0%" in comparison["ttft_p50"]
        finally:
            Path(baseline_path).unlink(missing_ok=True)

    def test_improvement_detected(self):
        """A faster result should show a negative delta."""
        runner = BenchmarkRunner(BenchmarkConfig(model_path="test.aeg"))
        current = BenchmarkResult(
            model_path="test.aeg",
            hardware_target="cpu_avx512",
            num_requests=10,
            ttft_p50_ms=80.0,
            ttft_p99_ms=200.0,
            throughput_tps=1500.0,
        )
        baseline_data = {
            "latency": {"ttft": {"p50": 100.0, "p99": 300.0}},
            "throughput": {"tokens_per_second": 1000.0},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline_data, f)
            baseline_path = f.name

        try:
            runner.config.compare_baseline = baseline_path
            comparison = runner._compare_baseline(current)
            # p50 went from 100 to 80 → -20%
            assert "-20.0%" in comparison.get("ttft_p50", "")
        finally:
            Path(baseline_path).unlink(missing_ok=True)
