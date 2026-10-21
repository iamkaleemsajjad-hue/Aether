"""Bound the hybrid's causal history using an independent full convolution.

The active SSM block is Mamba-1's depthwise causal convolution, with zero B/C
and a nonzero D skip to isolate it through real forwards. Attention/FFN output
projections are zero, so their residual block is the identity. The FP64 oracle
uses numpy.convolve, never the runtime's cache update. See Gu & Dao (2023),
section 3.4, and state-spaces/mamba v2.2.4 mamba_simple.py:
https://arxiv.org/html/2312.00752v2
https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py
This isolates convolution correctness; it is not whole-model HF parity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_mamba_convolution_history import _engine, _oracle

from aether.kernels.native_cpu import NativeCPUKernels
from aether.runtime.cpu_engine import LayerWeights
from aether.runtime.hybrid_engine import (
    HybridExecutionEngine,
    HybridLayerWeights,
    HybridModelWeights,
)


def hybrid_fixture(width: int, with_bias: bool) -> tuple[HybridExecutionEngine, Any]:
    """Create one real SSM and one identity attention/FFN residual block."""
    source = _engine("mamba", width, with_bias, 3)
    hidden = source.weights.embedding.shape[1]
    zero = np.zeros((hidden, hidden), np.float32)
    attention = LayerWeights(
        attention_norm=np.ones(hidden, np.float32),
        q_proj=zero,
        k_proj=zero,
        v_proj=zero,
        o_proj=zero,
        ffn_norm=np.ones(hidden, np.float32),
        gate_proj=zero,
        up_proj=zero,
        down_proj=zero,
    )
    weights = HybridModelWeights(
        embedding=source.weights.embedding,
        layers=[
            HybridLayerWeights(kind=kind, mamba=source.weights.layers[0], transformer=attention)
            for kind in ("ssm", "attention")
        ],
        final_norm=source.weights.final_norm,
        lm_head=source.weights.lm_head,
    )
    return HybridExecutionEngine(
        weights,
        layer_types=["ssm", "attention"],
        num_heads=2,
        num_kv_heads=2,
        state_size=source.state_size,
        inner_size=source.inner_size,
        dt_rank=source.dt_rank,
        conv_kernel=width,
        kernels=NativeCPUKernels.reference(),
    ), source


@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("with_bias", [False, True])
def test_hybrid_forward_and_cached_partitions_match_full_convolution(
    backend: str, width: int, with_bias: bool
) -> None:
    """Every active recurrent history must retain exactly k-1 input columns."""
    engine: Any
    engine, source = hybrid_fixture(width, with_bias)
    if backend == "torch_cpu":
        pytest.importorskip("torch")
        from aether.runtime.torch_engine import TorchHybridAEGEngine

        engine = TorchHybridAEGEngine(engine, "cpu")
    ids = np.random.default_rng(7).integers(0, 17, 65)
    expected = _oracle(source, ids)
    actual, cache = engine.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    recurrent = cache.ssm if backend == "cpu" else cache
    assert all(history.shape[-1] == width - 1 for history in recurrent.conv_history)

    rows, cache, cursor = [], None, 0
    for length in (1, 3, 1, 8, 1, 17, 2, 32):
        logits, cache = engine.forward(ids[cursor : cursor + length], cache)
        rows.append(logits)
        cursor += length
        recurrent = cache.ssm if backend == "cpu" else cache
        assert all(history.shape[-1] == width - 1 for history in recurrent.conv_history)
    assert cursor == len(ids)
    np.testing.assert_allclose(np.concatenate(rows), expected, rtol=2e-5, atol=2e-6)

    _, prefix = engine.forward(ids[:3])
    assert prefix.last_logits is not None
    saved_logits = (
        prefix.last_logits.clone() if backend == "torch_cpu" else prefix.last_logits.copy()
    )
    suffix, continued = engine.forward(ids[3:], prefix.clone())
    assert prefix.length == 3 and continued.length == len(ids)
    np.testing.assert_allclose(suffix, expected[3:], rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(prefix.last_logits, saved_logits)
