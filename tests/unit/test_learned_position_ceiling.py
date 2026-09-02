"""A learned position table is a hard ceiling, and every executor must say so.

A trained absolute-position table has exactly as many rows as the model has
positions.  Unlike the rotary tables -- which are *derived*, and which
``_ensure_rope`` rebuilds taller on demand -- there is no arithmetic that produces
the next row, so a step past the last one has no position to read.

The reference and hybrid executors refused that step by name.  The tensor executor
handed the index straight to ``index_select``, and the tensor-parallel executor did
something worse: it writes gathered rows into a ``torch.empty`` buffer masked by
shard range, so a position no shard claims leaves uninitialised memory in place and
adds it to the hidden state as though it were an embedding.

On CPU the unguarded lookup raises from inside the gather.  On an accelerator it is
an asynchronous device-side assert -- ``indexSelectSmallIndex`` failing its bounds
check -- which poisons the CUDA context, so the process dies later at whatever
unrelated call touches it next, with a stack that points at cleanup rather than at
the cause.  That is what took the benchmark's GPT-Neo cell down: the placement
bootstrap probe runs a prefill at the planner's ceiling *and one decode step on it*,
and with the planner asking for 2048 positions of a 2048-row table, that step read
row 2048.

See also ``tests/unit/test_rope_table_length.py``: the same distinction between a
table's height and the length a request actually evaluates, from the other side.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
from aether.runtime.torch_engine import TorchAEGEngine
from aether.runtime.torch_tensor_parallel import TorchTensorParallelAEGEngine

#: Stands in for GPT-Neo's 2048.  Small enough to run on CPU, and the only property
#: under test is that the ceiling is *the table's own height*, whatever it is.
ROWS = 16

HEADS = 2
HEAD_DIM = 4
HIDDEN = HEADS * HEAD_DIM

#: The message both executors owe the caller, keyed on the number that is wrong.
REFUSAL = r"sequence end 17 exceeds learned position embedding capacity 16"


def _weights(rows: int | None = ROWS) -> ModelWeights:
    """A GPT-Neo-shaped block: learned absolute positions, non-gated GELU FFN."""
    rng = np.random.default_rng(7)
    vocab, intermediate = 19, 12

    def matrix(out: int, inn: int) -> np.ndarray:
        return rng.normal(0.0, 0.05, (out, inn)).astype(np.float32)

    layer = LayerWeights(
        attention_norm=np.ones(HIDDEN, dtype=np.float32),
        q_proj=matrix(HIDDEN, HIDDEN), k_proj=matrix(HIDDEN, HIDDEN),
        v_proj=matrix(HIDDEN, HIDDEN), o_proj=matrix(HIDDEN, HIDDEN),
        ffn_norm=np.ones(HIDDEN, dtype=np.float32),
        # One intermediate projection, as a classic GPT block has: ``gate_proj`` is
        # the up projection and ``up_proj`` stays absent.
        gate_proj=matrix(intermediate, HIDDEN), up_proj=None,
        down_proj=matrix(HIDDEN, intermediate),
    )
    return ModelWeights(
        embedding=matrix(vocab, HIDDEN), layers=[layer],
        final_norm=np.ones(HIDDEN, dtype=np.float32), lm_head=matrix(vocab, HIDDEN),
        position_embedding=None if rows is None else matrix(rows, HIDDEN),
        position_type="learned" if rows is not None else "RoPE",
        ffn_type="gelu", norm_eps=1e-5,
    )


def _reference(rows: int | None = ROWS) -> CPUExecutionEngine:
    return CPUExecutionEngine(_weights(rows), num_heads=HEADS, num_kv_heads=HEADS)


def _tensor(rows: int | None = ROWS) -> TorchAEGEngine:
    return TorchAEGEngine(_reference(rows), "cpu")


def _sharded(rows: int | None = ROWS) -> TorchTensorParallelAEGEngine:
    return TorchTensorParallelAEGEngine(_reference(rows), ["cpu:0", "cpu:1"])


def _prefill_then_decode(engine, tokens: int) -> None:
    """The bootstrap probe's two steps, which together evaluate ``tokens + 1``."""
    _, cache = engine._forward_device(
        np.zeros(tokens, dtype=np.int64), None, reserve=tokens + 1, logits="last"
    )
    engine._forward_device(
        np.zeros(1, dtype=np.int64), cache, reserve=tokens + 1, logits="last"
    )


# -- the ceiling is the table's height ----------------------------------------

def test_the_tensor_executor_derives_the_ceiling_from_the_table() -> None:
    engine = _tensor()
    assert engine.max_positions == ROWS
    assert int(engine.position_embedding.shape[0]) == ROWS


def test_sharding_does_not_lose_the_ceiling() -> None:
    """The constraint has to survive the weights being taken apart.

    The sharded executor keeps only scalar graph metadata, deliberately holding no
    weight container -- so the base class's derivation found no table and fell back
    to the generous cap meant for rotary models.  A million-position ceiling on a
    sixteen-row table is not a bound at all.
    """
    engine = _sharded()
    assert engine.max_positions == ROWS
    assert engine.max_positions != TorchAEGEngine._DEFAULT_MAX_POSITIONS


def test_a_rotary_model_keeps_the_generous_cap() -> None:
    """No learned table, no hard ceiling: this must not become a global limit."""
    engine = _tensor(rows=None)
    assert engine.position_embedding is None
    assert engine.max_positions == TorchAEGEngine._DEFAULT_MAX_POSITIONS

# -- the step past the last row is refused, by name ---------------------------

def test_the_tensor_executor_refuses_the_step_past_the_last_row() -> None:
    with pytest.raises(ValueError, match=REFUSAL):
        _prefill_then_decode(_tensor(), ROWS)


def test_the_reference_executor_refuses_it_the_same_way() -> None:
    """Parity is the point: two executors, one model, one answer."""
    engine = _reference()
    _, cache = engine.forward(np.zeros(ROWS, dtype=np.int64), None)
    with pytest.raises(ValueError, match=REFUSAL):
        engine.forward(np.zeros(1, dtype=np.int64), cache)


def test_the_sharded_executor_refuses_it_rather_than_reading_empty_memory() -> None:
    with pytest.raises(ValueError, match=REFUSAL):
        _prefill_then_decode(_sharded(), ROWS)


def test_a_batched_prefill_past_the_ceiling_is_refused_too() -> None:
    """The batched branch shares the arithmetic, so it must share the guard."""
    engine = _tensor()
    with pytest.raises(ValueError, match=r"exceeds learned position embedding"):
        engine._forward_device(
            np.zeros((2, ROWS + 1), dtype=np.int64), None,
            reserve=ROWS + 1, batched=True, logits="last",
        )


def test_it_is_a_refusal_and_not_a_lookup_failure() -> None:
    """An ``IndexError`` from inside the gather is the symptom, not the diagnosis.

    The caller cannot act on it, and its accelerator equivalent does not surface at
    the offending call at all.
    """
    with pytest.raises(ValueError) as raised:
        _prefill_then_decode(_tensor(), ROWS)
    assert not isinstance(raised.value, IndexError)
    assert "learned position embedding capacity" in str(raised.value)

# -- a request the model can serve is still served ----------------------------

@pytest.mark.parametrize("tokens", [1, ROWS // 2, ROWS - 1])
def test_a_sequence_inside_the_ceiling_is_not_refused(tokens: int) -> None:
    _prefill_then_decode(_tensor(), tokens)
    _prefill_then_decode(_sharded(), tokens)


def test_the_last_representable_position_is_reachable() -> None:
    """Off-by-one in the guard would cost the model its final position."""
    engine = _tensor()
    logits, cache = engine._forward_device(
        np.zeros(ROWS, dtype=np.int64), None, reserve=ROWS, logits="last"
    )
    assert int(cache.length) == ROWS
    assert logits is not None


# -- the probe measures a step the model can take -----------------------------

def test_the_calibration_probe_at_the_planners_ceiling_is_representable() -> None:
    """The crash, in one line.

    ``bootstrap_placement`` calls this with the planner's context target, which
    defaults to 2048 -- exactly GPT-Neo's table height.  The probe's decode step then
    asks for one position past the end.
    """
    _tensor().calibration_pass(1, ROWS)


def test_the_probe_derives_its_length_from_the_model_not_the_caller() -> None:
    """Any caller's ceiling, however large, must still yield a runnable probe."""
    _tensor().calibration_pass(1, 4096)
    _sharded().calibration_pass(1, 4096)


def test_the_probe_still_measures_the_requested_shape_when_it_fits() -> None:
    """The clamp must not quietly shrink a probe that was already representable.

    The measurement's whole value is that it measures the pass a request will make.
    """
    engine = _tensor()
    seen: list[int] = []
    original = engine._forward_device

    def record(ids, cache, **kwargs):
        seen.append(int(np.asarray(ids).shape[-1]))
        return original(ids, cache, **kwargs)

    engine._forward_device = record  # type: ignore[method-assign]
    engine.calibration_pass(1, ROWS - 1)
    assert seen == [ROWS - 1, 1]


# -- the plan describes a request the model can serve --------------------------

def test_the_planner_envelope_stays_representable(monkeypatch) -> None:
    """A plan is a promise about one request's positions.

    Sizing memory for a request that cannot exist is the harmless half; the harmful
    half is that the probe then measures at that impossible ceiling.
    """
    from aether.backends.torch_backend import TorchBackend

    for name in ("AETHER_PLAN_CONTEXT", "AETHER_PLAN_GENERATE", "AETHER_PLAN_BATCH"):
        monkeypatch.delenv(name, raising=False)
    engine = _tensor()
    envelope = TorchBackend()._placement_workload(engine)
    assert envelope.context_target + envelope.generate_target <= ROWS
    assert envelope.context_target >= 1
    assert envelope.generate_target >= 1
    assert envelope.context_floor <= envelope.context_target
    assert envelope.generate_floor <= envelope.generate_target


def test_the_envelope_is_untouched_when_the_model_can_represent_it(monkeypatch) -> None:
    from aether.backends.torch_backend import TorchBackend

    for name in ("AETHER_PLAN_CONTEXT", "AETHER_PLAN_GENERATE", "AETHER_PLAN_BATCH"):
        monkeypatch.delenv(name, raising=False)
    backend = TorchBackend()
    assert backend._placement_workload(_tensor(rows=None)) == backend._placement_workload()
