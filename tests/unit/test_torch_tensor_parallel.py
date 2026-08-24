"""Correctness and ownership tests for the local tensor-parallel executor."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
from aether.parallelism.sharding import capacity_weighted_partition
from aether.runtime.torch_engine import TorchAEGEngine
from aether.runtime.torch_tensor_parallel import TorchTensorParallelAEGEngine


def _toy_engine(
    *, heads: int = 4, kv_heads: int = 2, head_dim: int = 2
) -> CPUExecutionEngine:
    rng = np.random.default_rng(11)
    vocab, hidden, intermediate = 17, 8, 12
    layer = LayerWeights(
        attention_norm=np.ones(hidden, dtype=np.float32),
        q_proj=rng.normal(size=(heads * head_dim, hidden)).astype(np.float32),
        k_proj=rng.normal(size=(kv_heads * head_dim, hidden)).astype(np.float32),
        v_proj=rng.normal(size=(kv_heads * head_dim, hidden)).astype(np.float32),
        o_proj=rng.normal(size=(hidden, heads * head_dim)).astype(np.float32),
        ffn_norm=np.ones(hidden, dtype=np.float32),
        gate_proj=rng.normal(size=(intermediate, hidden)).astype(np.float32),
        up_proj=rng.normal(size=(intermediate, hidden)).astype(np.float32),
        down_proj=rng.normal(size=(hidden, intermediate)).astype(np.float32),
    )
    weights = ModelWeights(
        embedding=rng.normal(size=(vocab, hidden)).astype(np.float32),
        layers=[layer],
        final_norm=np.ones(hidden, dtype=np.float32),
        lm_head=rng.normal(size=(vocab, hidden)).astype(np.float32),
    )
    return CPUExecutionEngine(weights, heads, kv_heads)


def _conv1d_engine() -> CPUExecutionEngine:
    """Build a valid engine whose projections use source Conv1D layout."""
    source = _toy_engine(heads=3, kv_heads=1, head_dim=2)
    projection_names = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    for layer in source.weights.layers:
        for name in projection_names:
            value = getattr(layer, name)
            if value is not None:
                setattr(layer, name, value.T.copy())
    source.weights.lm_head = source.weights.lm_head.T.copy()
    return source


def test_tensor_parallel_matches_single_device_and_shards_weights() -> None:
    source = _toy_engine()
    single = TorchAEGEngine(source, "cpu")
    parallel = TorchTensorParallelAEGEngine(source, ["cpu:0", "cpu:1"])

    token_ids = np.asarray([1, 3, 4], dtype=np.int64)
    single_logits, single_cache = single.forward(token_ids)
    parallel_logits, parallel_cache = parallel.forward(token_ids)
    np.testing.assert_allclose(parallel_logits, single_logits, rtol=2e-5, atol=2e-5)

    single_next, _ = single.forward(np.asarray([5], dtype=np.int64), single_cache)
    parallel_next, _ = parallel.forward(np.asarray([5], dtype=np.int64), parallel_cache)
    np.testing.assert_allclose(parallel_next, single_next, rtol=2e-5, atol=2e-5)

    assert [tuple(shard.shape) for shard in parallel.embedding] == [(8, 8), (9, 8)]
    assert [tuple(shard.shape) for shard in parallel.lm_head] == [(8, 8), (9, 8)]
    assert all(shard.shape != source.weights.embedding.shape for shard in parallel.embedding)
    assert all(shard.shape != source.weights.lm_head.shape for shard in parallel.lm_head)


def test_tensor_parallel_mesh_wider_than_kv_heads_is_lossless() -> None:
    """MQA/GQA KV projections may have empty local shards on a wide mesh."""
    source = _toy_engine()
    single = TorchAEGEngine(source, "cpu")
    parallel = TorchTensorParallelAEGEngine(
        source, ["cpu:0", "cpu:1", "cpu:2", "cpu:3", "cpu:4"]
    )
    token_ids = np.asarray([1, 3], dtype=np.int64)
    expected, _ = single.forward(token_ids)
    actual, _ = parallel.forward(token_ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert sum(int(shard.shape[0]) for shard in parallel.layers[0]["k_proj"]) == source.num_kv_heads * source.head_dim
    assert any(int(shard.shape[0]) == 0 for shard in parallel.layers[0]["k_proj"])


def test_linear_accepts_both_linear_and_conv1d_weight_layouts() -> None:
    """Weight orientation is inferred from the contraction dimension."""
    source = _toy_engine()
    engine = TorchAEGEngine(source, "cpu")
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    canonical = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected = torch.nn.functional.linear(x, canonical)

    torch.testing.assert_close(engine._linear(x, canonical), expected)
    torch.testing.assert_close(engine._linear(x, canonical.transpose(0, 1)), expected)

    class NumpyGemm:
        @staticmethod
        def sgemm(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return left @ right

    source.kernels = NumpyGemm()
    np_expected = expected.numpy()
    np.testing.assert_allclose(
        source._linear(x.numpy(), canonical.numpy()), np_expected
    )
    np.testing.assert_allclose(
        source._linear(x.numpy(), canonical.transpose(0, 1).numpy()), np_expected
    )


def test_tensor_parallel_canonicalizes_conv1d_projection_axes() -> None:
    """Conv1D matrices must be normalized before row/column sharding."""
    source = _conv1d_engine()
    single = TorchAEGEngine(source, "cpu")
    parallel = TorchTensorParallelAEGEngine(source, ["cpu:0", "cpu:1"])

    token_ids = np.asarray([1, 3, 4], dtype=np.int64)
    expected, _ = single.forward(token_ids)
    actual, _ = parallel.forward(token_ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_capacity_partition_uses_physical_memory_ratio() -> None:
    """A 36/64 capacity mesh receives 36%/64% of a divisible tensor."""
    ranges = capacity_weighted_partition(100, [36.0, 64.0])
    assert ranges == [(0, 36), (36, 100)]


def test_tensor_parallel_keeps_persisted_optimizations_advisory() -> None:
    """Optional CPU cache plans must not make a GPU artifact unloadable."""
    source = _toy_engine()
    source.sparse_attention_plan = {"enabled": True}
    source.semantic_kv_plan = {"format": "aether_kv_compression_v1"}
    source.cross_layer_kv_plan = {"format": "aether_cross_layer_kv_v1"}

    parallel = TorchTensorParallelAEGEngine(source, ["cpu:0", "cpu:1"])
    assert sorted(parallel.persisted_optimization_plans) == [
        "cross_layer_kv", "semantic_kv", "sparse_attention"
    ]


def test_tensor_parallel_generate_matches_single_device() -> None:
    """The inherited generation loop must run on the sharded engine.

    ``forward()`` alone does not exercise ``generate_iter``, which calls
    ``_forward_device`` with its full keyword contract.  A sharded override that
    drifts from the base signature fails only here, and only on a host with two
    or more devices — exactly the case a single-GPU or CPU run never reaches.
    """
    source = _toy_engine()
    single = TorchAEGEngine(source, "cpu")
    parallel = TorchTensorParallelAEGEngine(source, ["cpu:0", "cpu:1"])

    prompt = np.asarray([1, 3, 4], dtype=np.int64)
    expected = single.generate(prompt, max_tokens=6, temperature=0.0)
    actual = parallel.generate(prompt, max_tokens=6, temperature=0.0)
    assert actual == expected

    # generate_with_cache returns the cache the session layer stores and reuses.
    resumed, cache = parallel.generate_with_cache(prompt, max_tokens=3, temperature=0.0)
    assert len(resumed) == 3
    assert cache.length == len(prompt) + 3


def test_tensor_parallel_streaming_matches_single_device() -> None:
    """Streaming decode must agree token-for-token with the single-device engine."""
    source = _toy_engine()
    single = TorchAEGEngine(source, "cpu")
    parallel = TorchTensorParallelAEGEngine(source, ["cpu:0", "cpu:1"])

    prompt = np.asarray([2, 5], dtype=np.int64)
    expected = list(single.generate_iter(prompt, max_tokens=5, temperature=0.0))
    actual = list(parallel.generate_iter(prompt, max_tokens=5, temperature=0.0))
    assert actual == expected


def test_sharded_overrides_keep_the_base_signature() -> None:
    """A sharded override must accept exactly what the base loop passes.

    The generation loop lives on the base class and calls these methods
    directly, so a drifted signature raises only when a multi-device mesh is
    actually present.  Comparing signatures catches it on any host.
    """
    import inspect

    drifted: list[str] = []
    for name, override in vars(TorchTensorParallelAEGEngine).items():
        if name.startswith("__") or not callable(override):
            continue
        inherited = getattr(TorchAEGEngine, name, None)
        if inherited is None:
            continue
        if inspect.signature(inherited) != inspect.signature(override):
            drifted.append(
                f"{name}: base{inspect.signature(inherited)} "
                f"vs sharded{inspect.signature(override)}"
            )
    assert not drifted, "sharded overrides drifted from the base contract: " + "; ".join(drifted)
