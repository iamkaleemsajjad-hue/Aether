"""The dispatch roof's operation count must come from the executor, not from reading it.

The three-roof cost model charges ``n_ops x t_dispatch`` for the host, and that term is
the only one that explains why splitting a small model across two devices makes it
slower.  It was 2.6x low, because the count had been enumerated by naming the
operations a decoder block appears to perform -- norm, q, k, v, rope, append, attend,
project, add -- and a dispatch is not a line of arithmetic.  Slices, unsqueezes,
reshapes, transposes and index_selects are each a launch, and the executor issues
sixteen slices and seven unsqueezes per block that no reading of the block predicts.

So the count is measured here, at the granularity the roof charges for, and each
structural adjustment is measured as a difference between two engines that differ only
in that feature.  This file is what stops the constants from going stale: if the
executor's dispatch sequence changes, the roof changes with it or this fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.placement.model_profile import (
    DECODE_OPS_FIXED,
    DECODE_OPS_PER_BLOCK,
    _ops_per_layer,
)

HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB = 32, 4, 4, 64, 40
HEAD_DIM = 8

def _matrix(rng, out: int, inn: int) -> np.ndarray:
    return rng.normal(0.0, 0.05, (out, inn)).astype(np.float32)


def _ones(width: int = HIDDEN) -> np.ndarray:
    return np.ones(width, dtype=np.float32)


def _layer(rng, *, gated=True, qk_norm=False, sandwich=False, experts=0, top_k=1):
    from aether.runtime.cpu_engine import ExpertWeights, LayerWeights

    fields = dict(
        attention_norm=_ones(),
        q_proj=_matrix(rng, HEADS * HEAD_DIM, HIDDEN),
        k_proj=_matrix(rng, KV_HEADS * HEAD_DIM, HIDDEN),
        v_proj=_matrix(rng, KV_HEADS * HEAD_DIM, HIDDEN),
        o_proj=_matrix(rng, HIDDEN, HEADS * HEAD_DIM),
        ffn_norm=_ones(),
        gate_proj=_matrix(rng, INTERMEDIATE, HIDDEN),
        up_proj=_matrix(rng, INTERMEDIATE, HIDDEN) if gated else None,
        down_proj=_matrix(rng, HIDDEN, INTERMEDIATE),
    )
    if qk_norm:
        fields["q_norm"] = _ones(HEAD_DIM)
        fields["k_norm"] = _ones(HEAD_DIM)
    if sandwich:
        fields["post_attention_norm"] = _ones()
        fields["post_ffn_norm"] = _ones()
    if experts:
        fields["router"] = _matrix(rng, experts, HIDDEN)
        fields["experts"] = [
            ExpertWeights(
                gate_proj=_matrix(rng, INTERMEDIATE, HIDDEN),
                up_proj=_matrix(rng, INTERMEDIATE, HIDDEN) if gated else None,
                down_proj=_matrix(rng, HIDDEN, INTERMEDIATE),
            )
            for _ in range(experts)
        ]
        fields["num_activated_experts"] = top_k
    return LayerWeights(**fields)


def _engine(layers: int, *, ffn="SwiGLU", parallel=False, placement="pre", **layer_kw):
    pytest.importorskip("torch")
    from aether.runtime.cpu_engine import CPUExecutionEngine, ModelWeights
    from aether.runtime.torch_engine import TorchAEGEngine

    rng = np.random.default_rng(5)
    weights = ModelWeights(
        embedding=_matrix(rng, VOCAB, HIDDEN),
        layers=[_layer(rng, **layer_kw) for _ in range(layers)],
        final_norm=_ones(),
        lm_head=_matrix(rng, VOCAB, HIDDEN),
        ffn_type=ffn,
        parallel_residual=parallel,
        norm_placement=placement,
    )
    engine = TorchAEGEngine(
        CPUExecutionEngine(weights, num_heads=HEADS, num_kv_heads=KV_HEADS), "cpu"
    )
    engine._ensure_rope(8)
    return engine

def _counter():
    from torch.utils._python_dispatch import TorchDispatchMode

    class Counter(TorchDispatchMode):
        def __init__(self) -> None:
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.count += 1
            return func(*args, **(kwargs or {}))

    return Counter()


def _steady_state(engine):
    """A cache with two tokens already in it, so nothing is being allocated."""
    ids = np.zeros(1, dtype=np.int64)
    _, cache = engine._forward_device(ids, None, reserve=8, logits="last")
    _, cache = engine._forward_device(ids, cache, reserve=8, logits="last")
    return cache


def _decode_ops(engine, *, sample: bool = False) -> int:
    """Every ``aten`` call one decode step makes."""
    cache = _steady_state(engine)
    ids = np.zeros(1, dtype=np.int64)
    counter = _counter()
    with counter:
        logits, cache = engine._forward_device(ids, cache, reserve=8, logits="last")
        if sample:
            token = engine._sample_device(logits, 0.0, 0, 0.0)
            int(token.reshape(1)[0])
    return counter.count


def _measure(**kw) -> tuple[int, int]:
    """``(per_block, fixed)``, differenced so the two separate cleanly."""
    one, three = _decode_ops(_engine(1, **kw)), _decode_ops(_engine(3, **kw))
    per_block = (three - one) // 2
    return per_block, one - per_block


@pytest.fixture(autouse=True)
def _no_kernel_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strategy probes dispatch their own work; the roof counts the model's."""
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")


def test_the_block_count_is_the_one_the_executor_dispatches() -> None:
    per_block, _ = _measure()
    assert per_block == DECODE_OPS_PER_BLOCK


def test_the_fixed_count_covers_the_graph_and_the_token_handoff() -> None:
    """A token costs the host the graph plus sampling plus the readback.

    All three are per-token host launches on the same critical path, so all three
    belong to the roof.
    """
    per_block, graph_fixed = _measure()
    with_sampling = _decode_ops(_engine(1), sample=True)
    assert with_sampling - per_block == DECODE_OPS_FIXED
    assert graph_fixed < DECODE_OPS_FIXED


@pytest.mark.parametrize(
    ("feature", "kwargs"),
    [
        ("gated_ffn", dict(ffn="GELU", gated=False)),
        ("qk_norm", dict(qk_norm=True)),
        ("parallel_residual", dict(parallel=True)),
        ("post_norm", dict(placement="post")),
        ("sandwich", dict(placement="sandwich", sandwich=True)),
        ("moe_top_1", dict(experts=4, top_k=1)),
        ("moe_top_2", dict(experts=4, top_k=2)),
    ],
)
def test_every_structural_adjustment_matches_a_measured_difference(
    feature: str, kwargs: dict
) -> None:
    """Each addend is a measurement, so none of them can be a guess.

    ``parallel_residual`` and ``post`` normalization are in the list precisely because
    they measure as *zero*: the earlier model deducted a launch for the first and the
    executor does not, and asserting the zero is what keeps a plausible-sounding
    deduction from coming back.
    """
    per_block, _ = _measure(**kwargs)
    predicted = _ops_per_layer(
        gated_ffn=kwargs.get("gated", True),
        qk_norm=kwargs.get("qk_norm", False),
        parallel_residual=kwargs.get("parallel", False),
        norm_placement=kwargs.get("placement", "pre"),
        is_moe=bool(kwargs.get("experts", 0)),
        experts_per_token=kwargs.get("top_k", 0),
    )
    assert per_block == predicted, feature


def test_the_roof_is_now_high_enough_to_change_a_decision() -> None:
    """Why the correction matters rather than merely being more accurate.

    A 32-layer decoder dispatches over fourteen hundred operations per token.  At the
    ledger's dispatch cost that is tens of milliseconds of pure host time -- the same
    order as the bandwidth roof on a mid-range accelerator -- so understating it by
    2.6x is what makes a launch-bound model look bandwidth-bound and hides the host
    from the plan entirely.
    """
    from aether.placement.model_profile import ModelProfile

    profile = ModelProfile(
        model_id="phi-3.5-mini-shaped",
        layers=32,
        ops_per_layer=_ops_per_layer(
            gated_ffn=True, qk_norm=False, parallel_residual=False,
            norm_placement="pre", is_moe=False, experts_per_token=0,
        ),
        ops_fixed=DECODE_OPS_FIXED,
    )
    assert profile.ops_per_token == 32 * DECODE_OPS_PER_BLOCK + DECODE_OPS_FIXED
    assert profile.ops_per_token > 1400


def test_a_routed_expert_is_gated_whatever_the_dense_ffn_declares() -> None:
    """The one adjustment that cannot be measured, held in place structurally.

    Every other addend in ``_ops_per_layer`` is a difference between two engines that
    differ in one feature.  A non-gated MoE is the exception: the executable
    representation has no such layer to run, because ``ExpertWeights.up_proj`` is a
    required field and ``_moe_ffn`` projects gate and up unconditionally.  So the
    non-gated deduction must not reach the expert triple, and that has to be asserted
    against the representation rather than against a measurement.

    Without this, the branch is a plausible-sounding claim in a comment: exactly the
    shape of the enumeration mistake the rest of this file exists to prevent.
    """
    import inspect

    from aether.runtime.cpu_engine import ExpertWeights

    annotations = ExpertWeights.__annotations__
    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert "None" not in str(annotations[projection]), (
            f"{projection} became optional; the MoE dispatch count must be re-derived"
        )

    from aether.runtime import torch_engine

    body = inspect.getsource(torch_engine.TorchAEGEngine._moe_ffn)
    assert 'expert["up_proj"]' in body and "if" not in body.split('expert["up_proj"]')[0].split("\n")[-1]

    gated = _ops_per_layer(
        gated_ffn=True, qk_norm=False, parallel_residual=False,
        norm_placement="pre", is_moe=True, experts_per_token=2,
    )
    plain = _ops_per_layer(
        gated_ffn=False, qk_norm=False, parallel_residual=False,
        norm_placement="pre", is_moe=True, experts_per_token=2,
    )
    assert gated == plain, "a routed layer's launch count does not follow the dense FFN"
