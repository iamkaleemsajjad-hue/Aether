"""Check causal convolution independently of either recurrent implementation.

For channel c, u[t,c] = bias[c] + sum_j w[c,j]*x[t-k+1+j,c],
with zero input before t=0. This is the depthwise Conv1d used by the
Mamba-2 reference (Dao & Gu, 2024, section 7.1), before its SSD recurrence:
https://arxiv.org/abs/2405.21060 and
https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2_simple.py

Zero B/C projections isolate that convolution through real public forwards:
the recurrent contribution is zero and the learned D skip remains active.
The oracle uses a full-sequence NumPy convolution, never the cache update.
This does not establish whole-model parity with the original HF model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from aether.runtime.mamba2_engine import (
    Mamba2ExecutionEngine,
    Mamba2LayerWeights,
    Mamba2ModelWeights,
)
from aether.runtime.mamba_engine import (
    MambaExecutionEngine,
    MambaLayerWeights,
    MambaModelWeights,
)


def _engine(family: str, width: int, with_bias: bool, rank: int) -> Any:
    rng = np.random.default_rng(20261020)
    hidden, inner, vocab, state, heads, groups = 8, 8, 17, 2, 2, 1

    def matrix(*shape: int) -> np.ndarray:
        return rng.normal(0, 0.3, shape).astype(np.float32)

    embedding, head = matrix(vocab, hidden), matrix(vocab, hidden)
    norm, final_norm = rng.uniform(0.5, 1.5, (2, hidden)).astype(np.float32)
    projection, gate = matrix(inner, hidden), matrix(inner, hidden)
    kernel = matrix(inner, width)
    bias = matrix(inner) if with_bias else np.zeros(inner, np.float32)
    out, skip = matrix(hidden, inner), rng.uniform(0.5, 1.5, inner).astype(np.float32)
    if family == "mamba":
        layer = MambaLayerWeights(
            norm=norm,
            in_proj=np.concatenate((projection, gate)),
            conv1d=kernel if rank == 2 else kernel[:, None, :],
            conv_bias=bias if with_bias else None,
            x_proj=np.zeros((1 + 2 * state, inner), np.float32),
            dt_proj=np.zeros((inner, 1), np.float32),
            a_log=np.zeros((inner, state), np.float32),
            d=skip,
            out_proj=out,
        )
        return MambaExecutionEngine(
            MambaModelWeights(embedding, [layer], final_norm, head),
            state_size=state,
            inner_size=inner,
            dt_rank=1,
            conv_kernel=width,
        )
    channels = inner + 2 * groups * state
    full_kernel = np.concatenate((kernel, np.zeros((channels - inner, width), np.float32)))
    full_bias = np.concatenate((bias, np.zeros(channels - inner, np.float32)))
    layer2 = Mamba2LayerWeights(
        norm=norm,
        in_proj=np.concatenate(
            (gate, projection, np.zeros((2 * groups * state + heads, hidden), np.float32))
        ),
        conv1d=full_kernel if rank == 2 else full_kernel[:, None, :],
        conv_bias=full_bias if with_bias else None,
        a_log=np.zeros(heads, np.float32),
        d=np.ones(heads, np.float32),
        dt=np.zeros(heads, np.float32),
        out_proj=out,
    )
    return Mamba2ExecutionEngine(
        Mamba2ModelWeights(embedding, [layer2], final_norm, head),
        state_size=state,
        inner_size=inner,
        num_heads=heads,
        num_groups=groups,
        conv_kernel=width,
    )


def _oracle(engine: Any, token_ids: np.ndarray) -> np.ndarray:
    """Evaluate the isolated block using full-sequence convolution in FP64."""
    weights, layer = engine.weights, engine.weights.layers[0]

    def norm(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (
            values / np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + weights.norm_eps) * scale
        )

    hidden = weights.embedding[token_ids].astype(np.float64)
    projected = norm(hidden, layer.norm) @ layer.in_proj.astype(np.float64).T
    if (bias := getattr(layer, "in_proj_bias", None)) is not None:
        projected += bias
    if isinstance(engine, MambaExecutionEngine):
        x, gate = np.split(projected, 2, axis=-1)
        skip = layer.d
    else:
        gate, x = (
            projected[:, : engine.inner_size],
            projected[:, engine.inner_size : 2 * engine.inner_size],
        )
        skip = np.repeat(layer.d, engine.head_dim)
    kernel = layer.conv1d.reshape(-1, engine.conv_kernel)[: engine.inner_size]
    convolved = np.column_stack(
        [
            np.convolve(x[:, channel], kernel[channel, ::-1], mode="full")[: len(token_ids)]
            for channel in range(engine.inner_size)
        ]
    )
    if layer.conv_bias is not None:
        convolved += layer.conv_bias[: engine.inner_size]
    mixed = convolved / (1 + np.exp(-convolved)) * skip * gate / (1 + np.exp(-gate))
    hidden += mixed @ layer.out_proj.astype(np.float64).T
    if (bias := getattr(layer, "out_proj_bias", None)) is not None:
        hidden += bias
    return norm(hidden, weights.final_norm) @ weights.lm_head.astype(np.float64).T


def _portable(engine: Any, backend: str) -> Any:
    if backend == "cpu":
        return engine
    pytest.importorskip("torch")
    from aether.runtime.torch_state_engine import TorchMamba2AEGEngine, TorchMambaAEGEngine

    adapter = (
        TorchMambaAEGEngine if isinstance(engine, MambaExecutionEngine) else TorchMamba2AEGEngine
    )
    return adapter(engine, "cpu")


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("rank", [2, 3])
def test_forward_matches_independent_causal_convolution(
    family: str,
    backend: str,
    width: int,
    with_bias: bool,
    rank: int,
) -> None:
    """Keep causal outputs and history bounds across prefill/decode partitions."""
    source = _engine(family, width, with_bias, rank)
    engine = _portable(source, backend)
    ids = np.random.default_rng(7).integers(0, 17, 65)
    expected = _oracle(source, ids)
    actual, cache = engine.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(history.shape[-1] == width - 1 for history in cache.conv_history)

    # Token decode and uneven prefill partitions must retain the same history.
    parts, cache, cursor = [], None, 0
    for length in (1, 3, 1, 8, 1, 17, 2, 32):
        logits, cache = engine.forward(ids[cursor : cursor + length], cache)
        parts.append(logits)
        cursor += length
        assert all(history.shape[-1] == width - 1 for history in cache.conv_history)
    assert cursor == len(ids)
    np.testing.assert_allclose(np.concatenate(parts), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("width", [1, 4])
def test_cpu_history_handoff_preserves_continuation(family: str, width: int) -> None:
    """Restore a convolution history accepted by the real worker adapter."""
    from aether.runtime.disaggregated_execution import CPUDecoderAdapter

    engine = _engine(family, width, True, 3)
    adapter = CPUDecoderAdapter(
        engine, model_fingerprint="convolution-oracle", cache_kind=f"{family}_v1"
    )
    ids = np.asarray([1, 3, 2, 7, 4, 5, 1], dtype=np.int64)
    expected = _oracle(engine, ids)
    _, prefix = engine.forward(ids[:3])
    restored = adapter.import_cache(adapter.export_cache(prefix))
    suffix, restored = engine.forward(ids[3:], restored)
    np.testing.assert_allclose(suffix, expected[3:], rtol=2e-5, atol=2e-6)
    assert restored.length == len(ids)
    assert all(history.shape[-1] == width - 1 for history in restored.conv_history)
