"""Tests for runtime submodules."""

from __future__ import annotations

import time

import pytest

from aether.core.types import TensorLayout, TensorShape, DType, MemoryLayout
from aether.runtime.executor import Executor
from aether.runtime.model_registry import ModelRegistry
from aether.runtime.precision_manager import PrecisionManager
from aether.runtime.session import RequestMetrics, SessionManager


class TestExecutor:
    def test_init(self) -> None:
        executor = Executor()
        assert executor.hardware is not None

    def test_dispatch_embedding(self) -> None:
        executor = Executor()
        from aether.runtime.executor import ExecutionContext
        from aether.core.types import HardwareTarget
        ctx = ExecutionContext(
            request_id="test-1",
            model_id="test-model",
            target=HardwareTarget.CPU_AVX512,
            precision="BF16",
        )
        result = executor.dispatch(ctx, "aeg.embedding", {"vocab_size": 100, "hidden_size": 64}, {})
        assert result is not None

    def test_reference_ops_compute_concrete_values(self) -> None:
        import numpy as np

        from aether.core.types import HardwareTarget
        from aether.runtime.executor import ExecutionContext

        executor = Executor()
        ctx = ExecutionContext(
            request_id="test-values",
            model_id="test-model",
            target=HardwareTarget.CPU_AVX512,
            precision="FP32",
        )
        layout = TensorLayout.from_dict(
            {"shape": {"dims": [2, 2]}, "dtype": "fp32", "layout": "dense"}
        )
        left = executor._allocate("left", layout, value=np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        right = executor._allocate("right", layout, value=np.asarray([[5.0, 6.0], [7.0, 8.0]]))
        indices = executor._allocate(
            "indices",
            TensorLayout.from_dict(
                {"shape": {"dims": [2]}, "dtype": "int32", "layout": "dense"}
            ),
            value=np.asarray([0, 1], dtype=np.int32),
        )

        embedded = executor.dispatch(
            ctx,
            "aeg.embedding",
            {"weight": np.asarray([[10.0, 11.0], [12.0, 13.0]])},
            {"indices": indices},
        )
        linear = executor.dispatch(
            ctx,
            "aeg.linear",
            {"weight": np.asarray([[2.0, 0.0], [0.0, 3.0]]), "bias": np.asarray([1.0, -1.0])},
            {"left": left},
        )
        added = executor.dispatch(ctx, "aeg.add", {}, {"left": left, "right": right})
        multiplied = executor.dispatch(ctx, "aeg.mul", {}, {"left": left, "right": right})
        product = executor.dispatch(ctx, "aeg.matmul", {}, {"left": left, "right": right})
        probabilities = executor.dispatch(ctx, "aeg.softmax", {}, {"left": left})
        activated = executor.dispatch(ctx, "aeg.silu", {}, {"left": left})

        np.testing.assert_allclose(embedded.value, [[10.0, 11.0], [12.0, 13.0]])
        np.testing.assert_allclose(linear.value, [[3.0, 5.0], [7.0, 11.0]])
        np.testing.assert_allclose(added.value, [[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_allclose(multiplied.value, [[5.0, 12.0], [21.0, 32.0]])
        np.testing.assert_allclose(product.value, [[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_allclose(np.sum(probabilities.value, axis=-1), [1.0, 1.0])
        np.testing.assert_allclose(activated.value, np.asarray(left.value) / (1.0 + np.exp(-left.value)))

    def test_session_context_manager(self) -> None:
        executor = Executor()
        with executor.session("req-1", "test-model") as ctx:
            assert ctx.request_id == "req-1"
        assert len(executor._allocations) == 0


class TestSessionManager:
    def test_create_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create(model_id="test-model")
        assert session.session_id is not None
        assert session.model_id == "test-model"

    def test_get_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create()
        retrieved = mgr.get(session.session_id)
        assert retrieved is session

    def test_destroy_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create()
        assert mgr.destroy(session.session_id) is True
        assert mgr.get(session.session_id) is None

    def test_record_metrics(self) -> None:
        mgr = SessionManager()
        session = mgr.create()
        metrics = RequestMetrics(
            request_id=session.new_request_id(),
            prompt_tokens=10,
            completion_tokens=20,
            duration_ms=100.0,
            ttft_ms=50.0,
        )
        session.record(metrics)
        assert session.total_tokens() == 30
        assert session.average_tps() > 0


class TestModelRegistry:
    def test_init(self, tmp_path: pytest.TestPath) -> None:
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert registry.models_dir.exists()

    def test_list_empty(self, tmp_path: pytest.TestPath) -> None:
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert registry.list_models() == []

    def test_is_cached_false(self, tmp_path: pytest.TestPath) -> None:
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert not registry.is_cached("test-model")

    def test_remove(self, tmp_path: pytest.TestPath) -> None:
        registry = ModelRegistry(cache_dir=str(tmp_path))
        assert not registry.remove("nonexistent")


class TestPrecisionManager:
    def test_register(self) -> None:
        mgr = PrecisionManager()
        precision_map = {"layer_0": "BF16", "layer_1": "Q4_K_M"}
        state = mgr.register("test-model", precision_map)
        assert state.model_id == "test-model"
        assert state.active_map == precision_map

    def test_update_memory_pressure(self) -> None:
        mgr = PrecisionManager()
        mgr.register("test-model", {"layer_0": "BF16"})
        mgr.update_memory_pressure("test-model", 0.95)
        state = mgr.get_state("test-model")
        assert state is not None
        assert state.memory_pressure == 0.95

    def test_cooldown(self) -> None:
        mgr = PrecisionManager(cooldown_seconds=60.0)
        mgr.register("test-model", {"layer_0": "BF16"})
        mgr.update_memory_pressure("test-model", 0.95)
        result = mgr.adjust("test-model")
        assert result["action"] == "cooldown"

    def test_adjust_after_cooldown(self) -> None:
        mgr = PrecisionManager(cooldown_seconds=0.0)
        mgr.register("test-model", {"layer_0": "BF16", "layer_1": "Q4_K_M"})
        mgr.update_memory_pressure("test-model", 0.95)
        result = mgr.adjust("test-model")
        assert result["action"] in ("downgrade", "downgrade_selective", "none")

    def test_reset(self) -> None:
        mgr = PrecisionManager(cooldown_seconds=0.0)
        mgr.register("test-model", {"layer_0": "BF16"})
        mgr.reset("test-model", {"layer_0": "Q4_K_M"})
        state = mgr.get_state("test-model")
        assert state is not None
        assert state.active_map["layer_0"] == "Q4_K_M"
