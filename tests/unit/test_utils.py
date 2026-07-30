"""Tests for utility submodules."""

from __future__ import annotations

import numpy as np
import pytest

from aether.utils.memory import (
    available_system_memory_gb,
    estimate_kv_cache_memory_gb,
    estimate_model_memory_gb,
    fit_in_memory,
    memory_pressure,
    recommend_batch_size,
)
from aether.utils.profiling import OpProfile, Profiler, Timer
from aether.utils.telemetry import TelemetryClient
from aether.utils.threading import BackgroundWorker, RateLimiter, run_in_thread


class TestProfiling:
    def test_op_profile(self) -> None:
        profile = OpProfile(op_name="matmul", duration_ms=1.5, memory_bytes=4096)
        assert profile.op_name == "matmul"
        assert profile.duration_ms == 1.5

    def test_profiler_disabled(self) -> None:
        profiler = Profiler(enabled=False)
        profiler.record(OpProfile(op_name="test", duration_ms=1.0))
        assert len(profiler) == 0

    def test_profiler_enabled(self) -> None:
        profiler = Profiler(enabled=True)
        profiler.record(OpProfile(op_name="add", duration_ms=0.5))
        assert len(profiler) == 1

    def test_summary(self) -> None:
        profiler = Profiler(enabled=True)
        profiler.record(OpProfile(op_name="add", duration_ms=1.0, memory_bytes=100))
        profiler.record(OpProfile(op_name="matmul", duration_ms=2.0, memory_bytes=200))
        summary = profiler.summary()
        assert summary["count"] == 2
        assert summary["total_duration_ms"] == 3.0
        assert "add" in summary["op_summary"]

    def test_timer(self) -> None:
        with Timer("test_block") as timer:
            pass
        assert timer.elapsed_ms >= 0.0

    def test_timer_with_profiler(self) -> None:
        profiler = Profiler(enabled=True)
        with Timer("test", profiler):
            pass
        assert len(profiler) == 1


class TestMemory:
    def test_estimate_model_memory(self) -> None:
        from aether.core.types import ModelArchitecture
        arch = ModelArchitecture(family="test", layers=4, hidden_size=256, num_attention_heads=4, params_billion=0.1)
        mem_gb = estimate_model_memory_gb(arch, {"default": "BF16"}, kv_cache_gb=1.0)
        assert mem_gb > 0

    def test_estimate_kv_cache(self) -> None:
        from aether.core.types import ModelArchitecture
        arch = ModelArchitecture(family="test", layers=4, hidden_size=256, num_attention_heads=4, num_kv_heads=2, params_billion=0.01)
        mem_gb = estimate_kv_cache_memory_gb(arch, sequence_length=1024, batch_size=1)
        assert mem_gb > 0

    def test_memory_pressure(self) -> None:
        assert memory_pressure(5.0, 10.0) == 0.5
        assert memory_pressure(0.0, 10.0) == 0.0
        assert memory_pressure(20.0, 10.0) == 1.0
        assert memory_pressure(-1.0, 10.0) == 0.0

    def test_fit_in_memory(self) -> None:
        from aether.core.types import ModelArchitecture
        arch = ModelArchitecture(family="test", layers=1, hidden_size=64, num_attention_heads=2, params_billion=0.001)
        result = fit_in_memory(arch, available_gb=100.0, precision_map={"default": "Q4_K_M"})
        assert result["fits"]
        assert result["estimated_gb"] > 0

    def test_available_system_memory(self) -> None:
        mem = available_system_memory_gb()
        assert mem > 0

    def test_recommend_batch_size(self) -> None:
        bs = recommend_batch_size(available_gb=10.0, model_memory_gb=2.0, kv_per_sequence_gb=0.1)
        assert 1 <= bs <= 256


class TestTelemetry:
    def test_disabled(self) -> None:
        client = TelemetryClient(enabled=False)
        client.record("test_event", {"key": "val"})
        assert client.summary()["buffered"] == 0

    def test_enabled(self) -> None:
        client = TelemetryClient(enabled=True)
        client.record("compile_success", {"model": "test"})
        assert client.summary()["buffered"] == 1

    def test_flush(self, tmp_path: pytest.TestPath) -> None:
        client = TelemetryClient(enabled=True, cache_dir=str(tmp_path))
        client.record("event1", {"a": 1})
        result = client.flush()
        assert result["flushed"] == 1
        assert (tmp_path / "logs" / "telemetry.jsonl").exists()


class TestThreading:
    def test_background_worker(self) -> None:
        worker = BackgroundWorker(max_workers=2)
        fut = worker.submit(lambda: 42)
        result = fut.result()
        assert result == 42
        worker.shutdown()

    def test_rate_limiter_acquire(self) -> None:
        limiter = RateLimiter(rate=1000.0)
        wait = limiter.acquire(tokens=1.0)
        assert wait == 0.0

    def test_rate_limiter_try_acquire(self) -> None:
        limiter = RateLimiter(rate=1000.0)
        assert limiter.try_acquire(tokens=1.0) is True
