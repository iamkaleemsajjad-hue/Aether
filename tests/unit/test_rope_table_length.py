"""The rotary tables must be derived for the length the request evaluates.

``tests/unit/test_rope_scaling_wiring.py`` proves the transform is applied.  This
file proves it is applied *at the right length*, which is a separate claim and the
one that was false.

Two of the five rotary schemes choose their frequencies from the sequence length:
``dynamic`` rescales the base with it, and ``longrope`` switches between a short and
a long per-dimension factor table at the length the checkpoint was trained to.  Both
executors build their tables taller than the current length so the next few steps
need no rebuild -- and both used to derive the frequencies from that *height*.

For Phi-3.5-mini the trained length is 4096 and the tensor executor's headroom is
512, so any request evaluating 3585..4096 positions built its tables past the
boundary and got the long factor table the reference reserves for longer contexts.
Nothing fails; the model simply rotates its slow dimensions at a fraction of the
correct rate, and the answer degrades.  The reference executor had the same defect
with a much wider band, because its growth policy doubles rather than adding a slab.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.runtime.rope_scaling import scaled_inverse_frequencies

#: Phi-3.5-mini's rotary width and trained context.
HEAD_DIM = 96
HALF = HEAD_DIM // 2
TRAINED = 4096

LONGROPE = {
    "rope_type": "longrope",
    "short_factor": [1.0 + 0.02 * index for index in range(HALF)],
    "long_factor": [1.0 + 0.15 * index for index in range(HALF)],
}

#: FP32 table rounding; the derivation itself is FP64.
TABLE_TOLERANCE = 1e-6

def _weights(scaling: dict | None, original: int):
    from aether.runtime.cpu_engine import LayerWeights, ModelWeights

    heads, vocab, intermediate = 2, 17, 12
    hidden = HEAD_DIM * heads
    rng = np.random.default_rng(11)

    def matrix(out: int, inn: int) -> np.ndarray:
        return rng.normal(0.0, 0.05, (out, inn)).astype(np.float32)

    layer = LayerWeights(
        attention_norm=np.ones(hidden, dtype=np.float32),
        q_proj=matrix(hidden, hidden), k_proj=matrix(hidden, hidden),
        v_proj=matrix(hidden, hidden), o_proj=matrix(hidden, hidden),
        ffn_norm=np.ones(hidden, dtype=np.float32),
        gate_proj=matrix(intermediate, hidden), up_proj=matrix(intermediate, hidden),
        down_proj=matrix(hidden, intermediate),
    )
    return ModelWeights(
        embedding=matrix(vocab, hidden), layers=[layer],
        final_norm=np.ones(hidden, dtype=np.float32), lm_head=matrix(vocab, hidden),
        rope_theta=10000.0, norm_eps=1e-5,
        rope_scaling=scaling, original_context_length=original,
        context_length=131072 if scaling else None,
    )


def _reference_engine(scaling: dict | None, original: int = TRAINED):
    from aether.runtime.cpu_engine import CPUExecutionEngine

    return CPUExecutionEngine(_weights(scaling, original), num_heads=2, num_kv_heads=2)


def _tensor_engine(scaling: dict | None, original: int = TRAINED):
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    return TorchAEGEngine(_reference_engine(scaling, original), "cpu")


def _expected(length: int, rotary_dim: int, spec) -> np.ndarray:
    """The first row of the cos table the reference would build for ``length``."""
    inverse, attention = scaled_inverse_frequencies(
        10000.0, rotary_dim, spec, sequence_length=int(length)
    )
    return np.cos(inverse) * attention

@pytest.mark.parametrize("required", [200, 3585, 3700, TRAINED, TRAINED + 1, 9000])
def test_tensor_tables_hold_the_frequencies_for_the_evaluated_length(
    required: int,
) -> None:
    engine = _tensor_engine(LONGROPE)
    engine._ensure_rope(required)
    assert int(engine._cos.shape[0]) >= required
    expected = _expected(required, engine.rotary_dim, engine.rope_scaling_spec)
    built = engine._cos[1, : expected.size].numpy().astype(np.float64)
    np.testing.assert_allclose(built, expected, rtol=0, atol=TABLE_TOLERANCE)


def test_the_boundary_band_is_where_the_height_and_the_length_disagree() -> None:
    """The bug had a witness: a request whose height crosses the trained context.

    Asserting the fix without asserting that the two answers *differ* here would pass
    just as well on a model where the height happened to be harmless, and would stop
    protecting anything.
    """
    engine = _tensor_engine(LONGROPE)
    engine._ensure_rope(3700)
    height = int(engine._cos.shape[0])
    assert height > TRAINED >= 3700

    for_length = _expected(3700, engine.rotary_dim, engine.rope_scaling_spec)
    for_height = _expected(height, engine.rotary_dim, engine.rope_scaling_spec)
    assert np.max(np.abs(for_length - for_height)) > 1e-2

    built = engine._cos[1, : for_length.size].numpy().astype(np.float64)
    np.testing.assert_allclose(built, for_length, rtol=0, atol=TABLE_TOLERANCE)


def test_a_short_request_after_a_boundary_request_is_not_poisoned() -> None:
    """The defect outlived the request that caused it.

    ``_ensure_rope`` reuses its tables when the recorded frequencies match the ones
    the new length needs.  With the tables built from the height and the record kept
    from the length, the two disagreed -- the record said *short* while the tables
    held *long* -- so every following shorter request matched the record and silently
    reused long-factor tables for the rest of the process.
    """
    engine = _tensor_engine(LONGROPE)
    engine._ensure_rope(3700)
    engine._ensure_rope(200)
    expected = _expected(200, engine.rotary_dim, engine.rope_scaling_spec)
    built = engine._cos[1, : expected.size].numpy().astype(np.float64)
    np.testing.assert_allclose(built, expected, rtol=0, atol=TABLE_TOLERANCE)


def test_the_recorded_frequencies_are_the_ones_the_tables_were_built_from() -> None:
    engine = _tensor_engine(LONGROPE)
    engine._ensure_rope(3700)
    inverse, attention = scaled_inverse_frequencies(
        10000.0, engine.rotary_dim, engine.rope_scaling_spec, sequence_length=3700
    )
    np.testing.assert_allclose(engine._rope_inv_freq, inverse, rtol=0, atol=0)
    assert engine.rope_attention_scaling == pytest.approx(attention)
    built = engine._cos[1, : inverse.size].numpy().astype(np.float64)
    np.testing.assert_allclose(
        built, np.cos(inverse) * attention, rtol=0, atol=TABLE_TOLERANCE
    )


def test_an_unscaled_model_still_takes_the_cheap_reuse_path() -> None:
    """No scheme, no per-pass re-derivation: the cost stays where it was."""
    engine = _tensor_engine(None)
    engine._ensure_rope(64)
    assert not engine._rope_is_length_sensitive()
    first = engine._cos
    engine._ensure_rope(64)
    assert engine._cos is first

@pytest.mark.parametrize("required", [500, 3700, TRAINED, TRAINED + 1, 9000])
def test_reference_tables_hold_the_frequencies_for_the_evaluated_length(
    required: int,
) -> None:
    engine = _reference_engine(LONGROPE)
    engine._ensure_rope_capacity(required)
    expected = _expected(required, engine._rotary_dim, engine._rope_scaling_spec())
    built = engine._cos[1, : expected.size].astype(np.float64)
    np.testing.assert_allclose(built, expected, rtol=0, atol=TABLE_TOLERANCE)


def test_reference_rebuilds_when_the_factors_change_under_a_tall_table() -> None:
    """Height is not evidence of correctness for a length-sensitive scheme.

    A checkpoint trained to 32768 and served to 131072 has a doubling band tens of
    thousands of positions wide: growing a 16384-row table for a 20000-position
    request derives at 32768, and one more growth derives past the boundary while the
    request is still inside it.
    """
    engine = _reference_engine(LONGROPE, original=32768)
    engine._ensure_rope_capacity(20000)
    engine._ensure_rope_capacity(30000)
    expected = _expected(30000, engine._rotary_dim, engine._rope_scaling_spec())
    built = engine._cos[1, : expected.size].astype(np.float64)
    np.testing.assert_allclose(built, expected, rtol=0, atol=TABLE_TOLERANCE)


def test_reference_does_not_grow_the_table_for_a_shorter_request() -> None:
    """Rebuilding for a factor change must not reserve positions nobody asked for.

    The growth policy doubles, which is right when the table is too short and wrong
    when it is being rebuilt for a length that already fits -- otherwise a short
    request following a long one would double a table it never reads.
    """
    engine = _reference_engine(LONGROPE, original=32768)
    engine._ensure_rope_capacity(30000)
    height = int(engine._cos.shape[0])
    engine._ensure_rope_capacity(500)
    assert int(engine._cos.shape[0]) == height


def test_an_unscaled_reference_model_keeps_the_pure_height_check() -> None:
    engine = _reference_engine(None)
    assert not engine._rope_is_length_sensitive()
    first = engine._cos
    engine._ensure_rope_capacity(int(first.shape[0]))
    assert engine._cos is first
