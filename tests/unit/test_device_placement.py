"""Placement policy for multi-accelerator hosts.

Tensor parallelism is a memory-capacity mechanism.  Splitting a model that
already fits on one device adds a cross-device copy to every layer, which
serializes the decode pipeline and makes generation slower than single-device
execution — measurably so for small models.  These tests pin the policy that
decides between the two.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from aether.backends.torch_backend import TorchBackend


def _engine(hidden: int = 64, layers: int = 2, vocab: int = 128) -> SimpleNamespace:
    """A weights-shaped stand-in; only tensor sizes matter for the estimate."""
    def layer() -> SimpleNamespace:
        return SimpleNamespace(
            q_proj=np.zeros((hidden, hidden), dtype=np.float32),
            k_proj=np.zeros((hidden, hidden), dtype=np.float32),
            v_proj=np.zeros((hidden, hidden), dtype=np.float32),
            o_proj=np.zeros((hidden, hidden), dtype=np.float32),
            gate_proj=np.zeros((hidden * 2, hidden), dtype=np.float32),
            up_proj=np.zeros((hidden * 2, hidden), dtype=np.float32),
            down_proj=np.zeros((hidden, hidden * 2), dtype=np.float32),
            experts=[],
        )

    embedding = np.zeros((vocab, hidden), dtype=np.float32)
    return SimpleNamespace(
        weights=SimpleNamespace(
            embedding=embedding, lm_head=embedding, layers=[layer() for _ in range(layers)]
        )
    )


class TestWeightEstimate:
    def test_counts_every_projection_at_two_bytes(self) -> None:
        engine = _engine(hidden=64, layers=2, vocab=128)
        # 2 x (128x64) embedding/lm_head + 2 layers x (4x64x64 + 3x2x64x64)
        expected = (2 * 128 * 64 + 2 * (4 * 64 * 64 + 3 * 2 * 64 * 64)) * 2
        assert TorchBackend._estimated_weight_bytes(engine) == expected

    def test_includes_routed_experts(self) -> None:
        engine = _engine()
        dense = TorchBackend._estimated_weight_bytes(engine)
        expert = SimpleNamespace(
            gate_proj=np.zeros((16, 8), dtype=np.float32),
            up_proj=np.zeros((16, 8), dtype=np.float32),
            down_proj=np.zeros((8, 16), dtype=np.float32),
        )
        engine.weights.layers[0].experts = [expert, expert]
        assert TorchBackend._estimated_weight_bytes(engine) > dense


class TestShardingPolicy:
    def test_single_accelerator_never_shards(self) -> None:
        backend = TorchBackend()
        backend._devices = ["cuda:0"]
        assert backend._should_shard(_engine()) is False

    def test_heterogeneous_cpu_mesh_is_never_automatic(self) -> None:
        # A CPU shard inside a GPU decode is slower than either device alone.
        backend = TorchBackend()
        backend._devices = ["cuda:0", "cpu"]
        assert backend._should_shard(_engine()) is False

    def test_model_that_fits_stays_on_one_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = TorchBackend()
        backend._devices = ["cuda:0", "cuda:1"]
        monkeypatch.setattr(
            TorchBackend, "_smallest_free_accelerator_bytes",
            lambda self, devices: 16 * 1024**3,
        )
        monkeypatch.setattr(
            TorchBackend, "_estimated_weight_bytes",
            staticmethod(lambda engine: 1 * 1024**3),
        )
        assert backend._should_shard(_engine()) is False

    def test_model_that_does_not_fit_is_sharded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = TorchBackend()
        backend._devices = ["cuda:0", "cuda:1"]
        monkeypatch.setattr(
            TorchBackend, "_smallest_free_accelerator_bytes",
            lambda self, devices: 16 * 1024**3,
        )
        monkeypatch.setattr(
            TorchBackend, "_estimated_weight_bytes",
            staticmethod(lambda engine: 30 * 1024**3),
        )
        assert backend._should_shard(_engine()) is True

    def test_operator_can_force_sharding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AETHER_FORCE_TENSOR_PARALLEL", "1")
        backend = TorchBackend()
        backend._devices = ["cuda:0", "cuda:1"]
        assert backend._should_shard(_engine()) is True

    def test_a_failed_capacity_probe_prefers_single_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Probing must never block a load; the single-device path runs anything
        # the host could load at all.
        backend = TorchBackend()
        backend._devices = ["cuda:0", "cuda:1"]

        def explode(self: object, devices: list[str]) -> int:
            raise RuntimeError("no NVML")

        monkeypatch.setattr(TorchBackend, "_smallest_free_accelerator_bytes", explode)
        assert backend._should_shard(_engine()) is False


class TestDeviceResolution:
    """An index-less accelerator device breaks every device-equality cache."""

    def test_cuda_spec_gains_an_explicit_index(self) -> None:
        torch = pytest.importorskip("torch")
        from aether.runtime.torch_engine import _resolve_device

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device on this host")
        resolved = _resolve_device(torch, "cuda")
        assert resolved.index is not None
        # The whole point: a tensor placed here must compare equal to it.
        assert torch.empty(1, device=resolved).device == resolved

    def test_index_less_cuda_would_not_compare_equal(self) -> None:
        torch = pytest.importorskip("torch")
        # Documents the trap this resolution exists to avoid.  It holds whether
        # or not CUDA is present, since it is a property of torch.device.
        assert torch.device("cuda") != torch.device("cuda:0")

    def test_cpu_is_unchanged_and_compares_equal(self) -> None:
        torch = pytest.importorskip("torch")
        from aether.runtime.torch_engine import _resolve_device

        resolved = _resolve_device(torch, "cpu")
        assert torch.empty(1, device=resolved).device == resolved


class TestRotaryTableGrowth:
    """Position-indexed tables must not grow without bound.

    They are sized by sequence length, so a multiplicative growth policy keeps
    reserving positions the model can never reach.  Combined with a device
    comparison that never matches, that turned a 112-token generation into a
    multi-gigabyte allocation.
    """

    def _engine(self, context_length: int | None = None):
        pytest.importorskip("torch")
        import numpy as np

        from aether.runtime.cpu_engine import (
            CPUExecutionEngine, LayerWeights, ModelWeights,
        )
        from aether.runtime.torch_engine import TorchAEGEngine

        hidden, heads, vocab = 8, 2, 16
        layer = LayerWeights(
            attention_norm=np.ones(hidden, dtype=np.float32),
            q_proj=np.eye(hidden, dtype=np.float32),
            k_proj=np.eye(hidden, dtype=np.float32),
            v_proj=np.eye(hidden, dtype=np.float32),
            o_proj=np.eye(hidden, dtype=np.float32),
            ffn_norm=np.ones(hidden, dtype=np.float32),
            gate_proj=np.zeros((hidden, hidden), dtype=np.float32),
            up_proj=np.zeros((hidden, hidden), dtype=np.float32),
            down_proj=np.zeros((hidden, hidden), dtype=np.float32),
        )
        weights = ModelWeights(
            embedding=np.eye(vocab, hidden, dtype=np.float32),
            layers=[layer],
            final_norm=np.ones(hidden, dtype=np.float32),
            lm_head=np.eye(vocab, hidden, dtype=np.float32),
            context_length=context_length,
        )
        return TorchAEGEngine(CPUExecutionEngine(weights, num_heads=heads), "cpu")

    def test_repeated_calls_do_not_grow_the_table(self) -> None:
        engine = self._engine()
        engine._ensure_rope(16)
        first = int(engine._cos.shape[0])
        for _ in range(40):
            engine._ensure_rope(16)
        assert int(engine._cos.shape[0]) == first

    def test_growth_is_additive_not_multiplicative(self) -> None:
        engine = self._engine()
        engine._ensure_rope(16)
        engine._ensure_rope(int(engine._cos.shape[0]) + 1)
        # Additive headroom, so height stays proportional to sequence length
        # rather than doubling on every extension.
        assert int(engine._cos.shape[0]) <= 16 + 2 * engine._ROPE_HEADROOM + 2

    def test_generation_keeps_the_table_proportional_to_length(self) -> None:
        import numpy as np

        engine = self._engine()
        engine.generate(np.asarray([1, 2, 3], dtype=np.int64), max_tokens=40, temperature=0.0)
        assert int(engine._cos.shape[0]) <= 43 + engine._ROPE_HEADROOM + 1

    def test_a_declared_context_length_caps_the_table(self) -> None:
        engine = self._engine(context_length=64)
        engine._ensure_rope(8)
        assert int(engine._cos.shape[0]) <= 64

    def test_exceeding_the_context_length_is_reported(self) -> None:
        engine = self._engine(context_length=32)
        with pytest.raises(ValueError, match="exceeds the compiled context length"):
            engine._ensure_rope(64)


def test_every_runtime_engine_module_imports_cleanly() -> None:
    """Import each accelerator engine and touch its module-level names.

    Ruff's undefined-name check does not flag a global referenced only inside a
    method body, so a missing import in one of these modules surfaces as a
    ``NameError`` at model-load time — on whichever architecture happens to use
    that engine, which may be none of the ones a given test run exercises.
    Importing every engine and resolving the shared helpers they reference makes
    that failure mode cheap to catch here instead.
    """
    import importlib

    pytest.importorskip("torch")
    engines = (
        "aether.runtime.torch_engine",
        "aether.runtime.torch_tensor_parallel",
        "aether.runtime.torch_state_engine",
        "aether.runtime.torch_transformer_engine",
    )
    for name in engines:
        module = importlib.import_module(name)
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        # Any shared helper a module references must actually resolve in it.
        for helper in ("_resolve_device", "execution_numerics", "TorchKVCache"):
            if helper in source:
                assert hasattr(module, helper), f"{name} references {helper} without importing it"
