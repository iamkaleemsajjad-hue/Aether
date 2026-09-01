"""A hybrid checkpoint's attention layers must rotate the way the checkpoint says.

The Jamba-shaped contract carries a block schedule, and its attention blocks are
ordinary rotary attention.  The rotary transform, however, never reached them: the
hybrid weight record had no ``rope_scaling`` field, the reference engine built its
inner transformer without one, and the tensor executor derived plain
``theta ** -exponent`` frequencies of its own.  A context-extended hybrid checkpoint
therefore rotated at the unscaled rate on every attention layer in the schedule.

Nothing fails when that happens.  The transform acts on the *slow* rotary
dimensions, so short prompts look fine and quality falls away as the position grows
-- the same failure mode, and the same root cause, as the dense executors deriving
their tables from a table height instead of the length being evaluated
(``tests/unit/test_rope_table_length.py``).  Two executors implement this contract
and both are checked here, because a divergence between them is a silent numerical
difference rather than a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.runtime.rope_scaling import scaled_inverse_frequencies

HEAD_DIM = 8
HEADS = KV_HEADS = 2
HIDDEN = HEAD_DIM * HEADS
VOCAB, INTERMEDIATE = 16, 12
INNER, STATE, DT_RANK, CONV = 8, 2, 1, 3

#: Small enough to cross inside a test, wide enough that the two factor tables differ.
TRAINED = 32
EXTENDED = 512

LONGROPE = {
    "rope_type": "longrope",
    "short_factor": [1.0 + 0.05 * index for index in range(HEAD_DIM // 2)],
    "long_factor": [1.0 + 0.60 * index for index in range(HEAD_DIM // 2)],
}

TABLE_TOLERANCE = 1e-6


def _weights(scaling: dict | None):
    from aether.runtime.hybrid_engine import HybridLayerWeights, HybridModelWeights
    from aether.runtime.cpu_engine import LayerWeights
    from aether.runtime.mamba_engine import MambaLayerWeights

    rng = np.random.default_rng(3)

    def matrix(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def ones(size: int) -> np.ndarray:
        return np.ones(size, dtype=np.float32)

    transformer = LayerWeights(
        attention_norm=ones(HIDDEN),
        q_proj=matrix(HEADS * HEAD_DIM, HIDDEN),
        k_proj=matrix(KV_HEADS * HEAD_DIM, HIDDEN),
        v_proj=matrix(KV_HEADS * HEAD_DIM, HIDDEN),
        o_proj=matrix(HIDDEN, HEADS * HEAD_DIM),
        ffn_norm=ones(HIDDEN),
        gate_proj=matrix(INTERMEDIATE, HIDDEN),
        up_proj=matrix(INTERMEDIATE, HIDDEN),
        down_proj=matrix(HIDDEN, INTERMEDIATE),
    )
    mamba = MambaLayerWeights(
        norm=ones(HIDDEN),
        in_proj=matrix(2 * INNER, HIDDEN),
        conv1d=matrix(INNER, CONV),
        x_proj=matrix(DT_RANK + 2 * STATE, INNER),
        dt_proj=matrix(INNER, DT_RANK),
        a_log=matrix(INNER, STATE),
        d=ones(INNER),
        out_proj=matrix(HIDDEN, INNER),
        conv_bias=np.zeros(INNER, dtype=np.float32),
        dt_bias=np.zeros(INNER, dtype=np.float32),
    )
    return HybridModelWeights(
        embedding=matrix(VOCAB, HIDDEN),
        layers=[
            HybridLayerWeights(kind="ssm", transformer=transformer, mamba=mamba),
            HybridLayerWeights(kind="attention", transformer=transformer, mamba=mamba),
        ],
        final_norm=ones(HIDDEN),
        lm_head=matrix(VOCAB, HIDDEN),
        rope_theta=10000.0,
        rope_scaling=scaling,
        original_context_length=TRAINED if scaling else None,
        context_length=EXTENDED if scaling else None,
    )


def _reference(scaling: dict | None):
    from aether.runtime.hybrid_engine import HybridExecutionEngine

    return HybridExecutionEngine(
        _weights(scaling),
        layer_types=["ssm", "attention"],
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        state_size=STATE,
        inner_size=INNER,
        dt_rank=DT_RANK,
        conv_kernel=CONV,
    )


def _tensor(scaling: dict | None):
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchHybridAEGEngine

    return TorchHybridAEGEngine(_reference(scaling), "cpu")


def _expected(length: int, position: int = 1) -> np.ndarray:
    """Row ``position`` of the cos table the contract calls for at ``length``.

    ``length`` is what the transform is *derived* from and ``position`` is the row
    the angle is evaluated at.  They are separate on purpose: the defect this file
    guards against confused a table's height with the length being evaluated, so a
    test that could only look at one row would not be able to tell the two apart.
    """
    from aether.runtime.rope_scaling import parse_rope_scaling

    spec = parse_rope_scaling(
        LONGROPE, context_length=EXTENDED, original_context_length=TRAINED
    )
    inverse, attention = scaled_inverse_frequencies(
        10000.0, HEAD_DIM, spec, sequence_length=int(length)
    )
    return np.cos(inverse * float(position)) * attention


def test_the_declared_transform_reaches_the_hybrid_weight_record() -> None:
    """The field has to exist before either executor can honour it."""
    weights = _weights(LONGROPE)
    assert weights.rope_scaling == LONGROPE
    assert weights.original_context_length == TRAINED
    assert weights.context_length == EXTENDED


def test_the_reference_hybrid_hands_the_transform_to_its_attention_half() -> None:
    engine = _reference(LONGROPE)
    inner = engine._transformer.weights
    assert inner.rope_scaling == LONGROPE
    assert inner.original_context_length == TRAINED
    assert inner.context_length == EXTENDED
    assert engine._transformer._rope_scaling_spec() is not None


@pytest.mark.parametrize("length", [4, TRAINED, TRAINED + 1, 200])
def test_the_tensor_hybrid_builds_the_table_the_transform_calls_for(
    length: int,
) -> None:
    engine = _tensor(LONGROPE)
    engine._ensure_rope(length)
    built = engine._cos[1, : HEAD_DIM // 2].numpy().astype(np.float64)
    np.testing.assert_allclose(built, _expected(length), rtol=0, atol=TABLE_TOLERANCE)


def test_the_unscaled_frequencies_are_not_what_the_transform_asks_for() -> None:
    """The witness.  Without this the fix could be asserting a coincidence.

    ``theta ** -exponent`` is exactly what the executor used to compute, so showing
    that it differs from the derived table is what makes the parametrized check above
    a real constraint rather than a restatement of the old behaviour.
    """
    from aether.runtime.rope_scaling import base_inverse_frequencies

    plain = base_inverse_frequencies(10000.0, HEAD_DIM)
    for length in (4, TRAINED + 1):
        derived = np.arccos(np.clip(_expected(length), -1.0, 1.0))
        assert np.max(np.abs(np.cos(plain) - np.cos(derived))) > 1e-3, length


def test_crossing_the_trained_length_switches_the_factor_table() -> None:
    """LongRoPE's two tables must both be reachable, in both directions.

    The reference switches on ``max(position_ids) + 1``, so a step at the boundary
    reads the short table and the next step reads the long one.  A hybrid executor
    advances one position at a time, which makes it the case most likely to carry a
    stale table forward.
    """
    #: How far apart the contract itself puts the two tables, read at a row whose
    #: angle is large enough to separate them.  Derived, not chosen: a hand-picked
    #: threshold would only say the tables differ by more than someone's guess.
    row = TRAINED
    separation = float(
        np.max(np.abs(_expected(TRAINED, row) - _expected(TRAINED + 1, row)))
    )
    assert separation > TABLE_TOLERANCE, "the fixture's two factor tables must differ"

    engine = _tensor(LONGROPE)
    engine._ensure_rope(TRAINED)
    short = engine._cos[row, : HEAD_DIM // 2].numpy().astype(np.float64)
    engine._ensure_rope(TRAINED + 1)
    long = engine._cos[row, : HEAD_DIM // 2].numpy().astype(np.float64)
    assert float(np.max(np.abs(short - long))) == pytest.approx(
        separation, abs=TABLE_TOLERANCE
    )

    np.testing.assert_allclose(
        short, _expected(TRAINED, row), rtol=0, atol=TABLE_TOLERANCE
    )
    np.testing.assert_allclose(
        long, _expected(TRAINED + 1, row), rtol=0, atol=TABLE_TOLERANCE
    )

    # And back: a shorter evaluation must not keep the long table.
    engine._ensure_rope(4)
    back = engine._cos[row, : HEAD_DIM // 2].numpy().astype(np.float64)
    np.testing.assert_allclose(back, _expected(4, row), rtol=0, atol=TABLE_TOLERANCE)


def test_a_taller_table_is_not_evidence_that_its_frequencies_are_right() -> None:
    """Headroom must not decide the derivation, only how many rows are materialized."""
    engine = _tensor(LONGROPE)
    engine._ensure_rope(4)
    assert int(engine._cos.shape[0]) > TRAINED, "the slab must cross the boundary"
    built = engine._cos[1, : HEAD_DIM // 2].numpy().astype(np.float64)
    np.testing.assert_allclose(built, _expected(4), rtol=0, atol=TABLE_TOLERANCE)


def test_an_unscaled_hybrid_reuses_its_table_instead_of_rebuilding_per_token() -> None:
    """The cost side.  The old code sized the table at exactly the length asked for.

    ``required`` grows by one per decode step, so every step rebuilt the whole table
    and reallocated it on the device.  With headroom the table is built once and
    reused, and an unscaled model never re-derives at all.
    """
    engine = _tensor(None)
    engine._ensure_rope(1)
    first = engine._cos
    for position in range(2, 40):
        engine._ensure_rope(position)
        assert engine._cos is first, position


def test_a_scaled_hybrid_reuses_its_table_while_the_factors_hold() -> None:
    """Re-deriving is not rebuilding: the same frequencies keep the same tables."""
    engine = _tensor(LONGROPE)
    engine._ensure_rope(4)
    first = engine._cos
    for position in range(5, TRAINED + 1):
        engine._ensure_rope(position)
        assert engine._cos is first, position


def test_both_executors_agree_on_the_logits_of_a_scaled_hybrid() -> None:
    """The two implementations of one contract must not diverge numerically."""
    torch = pytest.importorskip("torch")
    reference = _reference(LONGROPE)
    tensor = _tensor(LONGROPE)
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    expected, _ = reference.forward(ids)
    logits, _ = tensor.forward(ids)
    actual = logits.numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64).reshape(-1)[-VOCAB:],
        np.asarray(expected, dtype=np.float64).reshape(-1)[-VOCAB:],
        rtol=2e-4, atol=2e-4,
    )
