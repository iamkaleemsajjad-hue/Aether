"""Tests for the server metrics and middleware."""

from __future__ import annotations

import pytest

from aether.server.metrics import ServerMetrics


class TestServerMetrics:
    def test_record_http_request(self) -> None:
        metrics = ServerMetrics()
        metrics.record_http_request("GET", 200, 10.5)
        metrics.record_http_request("POST", 404, 5.0)
        assert metrics.http_requests_total == 2
        assert metrics.http_requests_by_method["GET"] == 1
        assert metrics.http_requests_by_method["POST"] == 1
        assert metrics.http_requests_by_status["2xx"] == 1
        assert metrics.http_requests_by_status["4xx"] == 1

    def test_record_generate(self) -> None:
        metrics = ServerMetrics()
        metrics.record_generate(
            tokens=100,
            latency_ms=500.0,
            ttft=50.0,
            tps=200.0,
            backend="vllm",
            target="cuda_sm90",
            finish_reason="stop",
        )
        assert metrics.generate_requests_total == 1
        assert metrics.generate_tokens_total == 100
        assert metrics.backend_counts["vllm"] == 1
        assert metrics.target_counts["cuda_sm90"] == 1
        assert metrics.finish_reason_counts["stop"] == 1

    def test_percentiles(self) -> None:
        metrics = ServerMetrics()
        assert metrics.p50([]) == 0.0
        assert metrics.p95([]) == 0.0
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert metrics.p50(values) == 30.0
        assert metrics.p95(values) == 50.0
        assert metrics.p99(values) == 50.0

    def test_to_dict(self) -> None:
        metrics = ServerMetrics()
        metrics.record_http_request("GET", 200, 5.0)
        result = metrics.to_dict()
        assert "http" in result
        assert "generation" in result
        assert "gauges" in result
        assert "distribution" in result
        assert result["http"]["total"] == 1

    def test_error_count(self) -> None:
        metrics = ServerMetrics()
        metrics.record_error()
        metrics.record_error()
        assert metrics.errors_total == 2
