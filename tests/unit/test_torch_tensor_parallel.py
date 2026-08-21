"""Correctness and ownership tests for the local tensor-parallel executor."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
from aether.runtime.torch_engine import TorchAEGEngine
from aether.runtime.torch_tensor_parallel import TorchTensorParallelAEGEngine


def _toy_engine() -> CPUExecutionEngine:
    rng = np.random.default_rng(11)
    vocab, hidden, heads, kv_heads, head_dim, intermediate = 17, 8, 4, 2, 2, 12
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
