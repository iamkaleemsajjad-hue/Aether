"""Tests for the real graph/weight portable tensor executor."""

from __future__ import annotations

import numpy as np
import pytest

from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights


def _tiny_engine() -> CPUExecutionEngine:
    rng = np.random.default_rng(7)
    vocab, hidden, heads, kv_heads, intermediate, layers = 17, 8, 2, 1, 12, 2

    def matrix(out: int, inn: int) -> np.ndarray:
        return rng.normal(0.0, 0.08, (out, inn)).astype(np.float32)

    blocks = [
        LayerWeights(
            attention_norm=np.ones(hidden, dtype=np.float32),
            q_proj=matrix(hidden, hidden),
            k_proj=matrix(hidden // 2, hidden),
            v_proj=matrix(hidden // 2, hidden),
            o_proj=matrix(hidden, hidden),
            ffn_norm=np.ones(hidden, dtype=np.float32),
            gate_proj=matrix(intermediate, hidden),
            up_proj=matrix(intermediate, hidden),
            down_proj=matrix(hidden, intermediate),
        )
        for _ in range(layers)
    ]
    weights = ModelWeights(
        embedding=matrix(vocab, hidden),
        layers=blocks,
        final_norm=np.ones(hidden, dtype=np.float32),
        lm_head=matrix(vocab, hidden),
        rope_theta=10000.0,
        norm_eps=1e-5,
    )
    return CPUExecutionEngine(weights, num_heads=heads, num_kv_heads=kv_heads)


def test_torch_portable_executor_matches_reference_cpu() -> None:
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    reference = _tiny_engine()
    portable = TorchAEGEngine(reference, "cpu")
    ids = np.asarray([1, 3, 5, 2], dtype=np.int64)
    expected, _ = reference.forward(ids)
    actual, _ = portable.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    expected_ids = reference.generate(ids, max_tokens=2, temperature=0.0)
    actual_ids = portable.generate(ids, max_tokens=2, temperature=0.0)
    assert actual_ids == expected_ids


def test_torch_portable_executor_preserves_qk_norm_and_local_attention() -> None:
    """Qwen-style head norms and GPT-Neo-style windows share one contract."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    reference = _tiny_engine()
    for layer in reference.weights.layers:
        layer.q_norm = np.ones(reference.head_dim, dtype=np.float32)
        layer.k_norm = np.ones(reference.head_dim, dtype=np.float32)
    reference.weights.attention_layers = ["global", "local"]
    reference.weights.attention_window = 2
    portable = TorchAEGEngine(reference, "cpu")
    ids = np.asarray([1, 3, 5, 2], dtype=np.int64)
    expected, _ = reference.forward(ids)
    actual, _ = portable.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_torch_portable_executor_applies_grammar_token_mask() -> None:
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    class Session:
        def __init__(self) -> None:
            self.advanced: list[int] = []

        def get_token_mask(self) -> bytearray:
            mask = bytearray(3)
            mask[7 // 8] |= 1 << (7 % 8)
            return mask

        def advance(self, token_id: int) -> int:
            self.advanced.append(token_id)
            return 0 if token_id == 7 else -1

    session = Session()
    tokens = TorchAEGEngine(_tiny_engine(), "cpu").generate(
        np.asarray([1], dtype=np.int64),
        max_tokens=3,
        temperature=0.0,
        grammar_session=session,
    )
    assert tokens == [7, 7, 7]
    assert session.advanced == tokens


def test_torch_portable_executor_rejects_unsupported_optimized_cache_plans() -> None:
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    reference = _tiny_engine()
    reference.semantic_kv_plan = {"format": "aether_kv_compression_v1"}
    with pytest.raises(ValueError, match="persisted sparse/KV alias plans"):
        TorchAEGEngine(reference, "cpu")
