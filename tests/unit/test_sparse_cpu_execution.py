"""Runtime consumption tests for persisted Pass 8 sparse attention plans."""

from __future__ import annotations

import numpy as np

from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights


def _engine(
    plan: dict | None = None,
    semantic_plan: dict | None = None,
    cross_layer_plan: dict | None = None,
    num_layers: int = 1,
) -> CPUExecutionEngine:
    rng = np.random.default_rng(123)
    hidden = 8
    intermediate = 16
    layer = LayerWeights(
        attention_norm=np.ones(hidden, dtype=np.float32),
        q_proj=rng.normal(size=(hidden, hidden)).astype(np.float32),
        k_proj=rng.normal(size=(hidden // 2, hidden)).astype(np.float32),
        v_proj=rng.normal(size=(hidden // 2, hidden)).astype(np.float32),
        o_proj=rng.normal(size=(hidden, hidden)).astype(np.float32),
        ffn_norm=np.ones(hidden, dtype=np.float32),
        gate_proj=rng.normal(size=(intermediate, hidden)).astype(np.float32),
        up_proj=rng.normal(size=(intermediate, hidden)).astype(np.float32),
        down_proj=rng.normal(size=(hidden, intermediate)).astype(np.float32),
    )
    return CPUExecutionEngine(
        ModelWeights(
            embedding=rng.normal(size=(32, hidden)).astype(np.float32),
            layers=[layer for _ in range(num_layers)],
            final_norm=np.ones(hidden, dtype=np.float32),
            lm_head=rng.normal(size=(32, hidden)).astype(np.float32),
        ),
        num_heads=2,
        num_kv_heads=1,
        sparse_attention_plan=plan,
        semantic_kv_plan=semantic_plan,
        cross_layer_kv_plan=cross_layer_plan,
    )


def test_sparse_plan_builds_per_head_causal_masks_and_changes_attention() -> None:
    plan = {
        "version": "sparse_attention/1.0",
        "enabled": True,
        "patterns": [
            {"head": 0, "pattern": "a_shape", "local_window": 1, "sink_tokens": 1},
            {"head": 1, "pattern": "block_sparse", "block_size": 2, "block_stride": 4},
        ],
    }
    sparse = _engine(plan)
    dense = _engine()
    q = np.random.default_rng(5).normal(size=(8, 2, 4)).astype(np.float32)
    k = np.random.default_rng(6).normal(size=(8, 1, 4)).astype(np.float32)
    v = np.random.default_rng(7).normal(size=(8, 1, 4)).astype(np.float32)

    allowed = sparse._sparse_allowed_mask(seq_len=8, total=8, causal_offset=0, heads=2)
    assert allowed is not None
    for head in range(2):
        assert np.all(np.diagonal(allowed[head]))
        assert not np.any(np.triu(allowed[head], k=1))
    sparse_out = sparse._attention(q, k, v, causal_offset=0)
    dense_out = dense._attention(q, k, v, causal_offset=0)
    assert np.isfinite(sparse_out).all()
    assert not np.allclose(sparse_out, dense_out)


def test_disabled_or_missing_plan_preserves_dense_attention() -> None:
    dense = _engine()
    disabled = _engine({"enabled": False, "patterns": []})
    q = np.random.default_rng(8).normal(size=(5, 2, 4)).astype(np.float32)
    k = np.random.default_rng(9).normal(size=(5, 1, 4)).astype(np.float32)
    v = np.random.default_rng(10).normal(size=(5, 1, 4)).astype(np.float32)
    np.testing.assert_allclose(
        disabled._attention(q, k, v, causal_offset=0),
        dense._attention(q, k, v, causal_offset=0),
    )


def test_semantic_kv_plan_compresses_real_cache_and_preserves_positions() -> None:
    plan = {
        "format": "aether_kv_compression_v1",
        "strategy": "chunk",
        "layers": [{"retention_ratio": 0.5, "chunk_size": 2}],
    }
    engine = _engine(semantic_plan=plan)
    # Identical embeddings produce repeated real K/V rows, making the
    # compressor's cosine clustering deterministic and observable.
    engine.weights.embedding[:] = 1.0

    _, cache = engine.forward(np.arange(1, 9, dtype=np.int64))
    assert cache.length == 8
    assert cache.stored_length < cache.length
    assert cache.positions[0] is not None
    assert np.array_equal(cache.positions[0], np.array([0, 7], dtype=np.int64))

    logits, cache = engine.forward(np.array([9], dtype=np.int64), cache)
    assert np.isfinite(logits).all()
    assert cache.length == 9
    assert cache.positions[0] is not None
    assert int(cache.positions[0][-1]) == 8


def test_semantic_kv_plan_rejects_unsupported_sentence_runtime_contract() -> None:
    plan = {
        "format": "aether_kv_compression_v1",
        "strategy": "sentence",
        "layers": [{"retention_ratio": 0.5, "chunk_size": 2}],
    }
    try:
        _engine(semantic_plan=plan)
    except ValueError as exc:
        assert "tokenizer boundary metadata" in str(exc)
    else:
        raise AssertionError("sentence semantic-KV plans must fail closed without boundaries")


def test_cross_layer_kv_plan_aliases_real_cache_arrays() -> None:
    plan = {
        "format": "aether_cross_layer_kv_v1",
        "strategy": "middle_outward",
        "threshold": 0.0,
        "n_layers": 2,
        "sharing_groups": [{"src_layer": 0, "shared_with": [1]}],
    }
    engine = _engine(cross_layer_plan=plan, num_layers=2)
    _, cache = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    assert cache.keys[0] is cache.keys[1]
    assert cache.values[0] is cache.values[1]
    assert cache.positions[0] is cache.positions[1]
    assert cache.length == 3
