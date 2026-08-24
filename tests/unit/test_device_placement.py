"""Placement policy for multi-accelerator hosts.

Tensor parallelism is a memory-capacity mechanism.  Splitting a model that
already fits on one device adds a cross-device copy to every layer, which
serializes the decode pipeline and makes generation slower than single-device
execution — measurably so for small models.  These tests pin the policy that
decides between the two.
"""

from __future__ import annotations

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
