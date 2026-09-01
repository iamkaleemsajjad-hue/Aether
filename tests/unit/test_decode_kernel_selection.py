"""A measured kernel choice must reach the shape it was measured on, and only it.

Three claims are checked here, all of them about the same mistake made at three
levels.  Which GEMM formulation a backend runs fastest is measured, not assumed, and
:class:`~aether.runtime.kernel_strategy.ShapeClass` keys that measurement on
``(phase, M bucket, K*N bucket, dtype, device)``.  Every part of that key has to
survive the trip to the call site:

* **Per weight shape.**  A decoder block holds several weight magnitudes -- for
  Phi-3.5-mini ``9216x3072``, ``3072x3072``, ``16384x3072`` and ``3072x8192``.  The
  engine used to measure one of them and apply the winner to all of them, which asks
  the backend for a layout chosen for a different GEMM.
* **Per phase.**  The forward pass resolved on the row count alone, and a batch-8
  decode step and an 8-token prefill both have eight rows, so whichever ran first
  supplied the other's formulation.
* **Before the first request.**  A measurement charged to the first token is
  time-to-first-token; the calibrator's own docstring calls itself "a bound on a load
  path rather than a per-request cost", and only a warm-up makes that true.

The calibrator is stubbed throughout: real candidate timing is neither deterministic
nor quick, and none of these claims is about which candidate wins.  What is under
test is that the answer for a shape class is applied to that class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

#: Magnitudes chosen so the block holds several *distinct* weight shapes, which is
#: the condition the per-shape table exists to handle.
HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, VOCAB = 32, 4, 4, 128, 64
HEAD_DIM = 8


def _matrix(rng, out: int, inn: int) -> np.ndarray:
    return rng.normal(0.0, 0.05, (out, inn)).astype(np.float32)


def _ones(width: int = HIDDEN) -> np.ndarray:
    return np.ones(width, dtype=np.float32)


def _engine():
    pytest.importorskip("torch")
    from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
    from aether.runtime.torch_engine import TorchAEGEngine

    rng = np.random.default_rng(7)
    layer = LayerWeights(
        attention_norm=_ones(),
        q_proj=_matrix(rng, HEADS * HEAD_DIM, HIDDEN),
        k_proj=_matrix(rng, KV_HEADS * HEAD_DIM, HIDDEN),
        v_proj=_matrix(rng, KV_HEADS * HEAD_DIM, HIDDEN),
        o_proj=_matrix(rng, HIDDEN, HEADS * HEAD_DIM),
        ffn_norm=_ones(),
        gate_proj=_matrix(rng, INTERMEDIATE, HIDDEN),
        up_proj=_matrix(rng, INTERMEDIATE, HIDDEN),
        down_proj=_matrix(rng, HIDDEN, INTERMEDIATE),
    )
    weights = ModelWeights(
        embedding=_matrix(rng, VOCAB, HIDDEN),
        layers=[layer, layer],
        final_norm=_ones(),
        lm_head=_matrix(rng, VOCAB, HIDDEN),
    )
    engine = TorchAEGEngine(
        CPUExecutionEngine(weights, num_heads=HEADS, num_kv_heads=KV_HEADS), "cpu"
    )
    engine._ensure_rope(16)
    return engine


class _StubCalibrator:
    """Answers per shape class with a tagged callable, and records what it was asked.

    Deliberately gives a *different* answer to every magnitude: a table that
    collapsed to one winner would return the same tag everywhere and the per-shape
    assertions would not be able to tell the two apart.
    """

    def __init__(self) -> None:
        self.asked: list[tuple[str, int, int, int]] = []

    def choose(self, *, phase, rows, in_features, out_features, dtype, probe=None):
        from aether.runtime.kernel_strategy import ShapeClass, StrategyChoice

        self.asked.append((phase, int(rows), int(in_features), int(out_features)))
        if probe is not None:
            probe()
        shape = ShapeClass.of(
            phase=phase, rows=rows, in_features=in_features,
            out_features=out_features, dtype=str(dtype).replace("torch.", ""),
            device_kind="cpu",
        )
        # ``linear`` is the reference and the engine special-cases it, so name the
        # choice something else to prove the table holds the *calibrated* callable.
        return StrategyChoice(shape=shape, name="transposed", source="measured")

    def strategy(self, choice) -> Any:
        """A real projection that also records which weight shapes reached it."""
        import torch

        magnitude = choice.shape.weight_magnitude
        linear = torch.nn.functional.linear

        def apply(x, weight, bias=None):
            apply.calls.append(tuple(weight.shape))
            return linear(x, weight, bias)

        apply.calls = []
        apply.magnitude = magnitude
        return apply

    def report(self) -> dict:
        return {"enabled": True, "stub": True}


@pytest.fixture(autouse=True)
def _no_real_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """No candidate timing anywhere in this file; the stub supplies the answers."""
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")


def _with_stub(engine) -> _StubCalibrator:
    stub = _StubCalibrator()
    engine._strategies = stub
    return stub


def test_every_distinct_projection_shape_is_measured_once() -> None:
    """One probe per shape, not one probe per projection and not one per block."""
    engine = _engine()
    shapes = [tuple(weight.shape) for weight, _ in engine._projection_probe_shapes()]
    assert len(shapes) == len(set(shapes))
    block = engine.layers[0]
    present = {
        tuple(block[name].shape)
        for name in engine._PROBE_NAMES
        if block.get(name) is not None
    }
    present.add(tuple(engine.lm_head.shape))
    assert set(shapes) == present
    assert len(present) > 1


def test_the_probes_are_ordered_largest_first() -> None:
    """If the calibration budget runs out, it runs out on the cheapest projection."""
    engine = _engine()
    sizes = [
        int(weight.shape[0]) * int(weight.shape[1])
        for weight, _ in engine._projection_probe_shapes()
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_each_shape_holds_the_formulation_measured_for_that_shape() -> None:
    """The regression for the defect: one winner applied to every magnitude.

    The block's magnitudes fall in three buckets, so a table that carried a single
    winner would show one tag against all of them.
    """
    engine = _engine()
    _with_stub(engine)
    engine._resolve_projection(1, "decode")

    from aether.runtime.kernel_strategy import ShapeClass

    assert engine._projection_table
    for shape, projection in engine._projection_table.items():
        expected = ShapeClass.of(
            phase="decode", rows=1, in_features=int(shape[1]),
            out_features=int(shape[0]), dtype="float32", device_kind="cpu",
        ).weight_magnitude
        assert projection.magnitude == expected, shape
    buckets = {p.magnitude for p in engine._projection_table.values()}
    assert len(buckets) > 1


def test_a_projection_is_dispatched_by_its_own_weight_shape() -> None:
    """The table is only worth building if the lookup uses the weight, not the pass."""
    engine = _engine()
    _with_stub(engine)
    engine._resolve_projection(1, "decode")
    torch = engine.torch
    for shape, projection in engine._projection_table.items():
        weight = torch.zeros(shape, dtype=torch.float32)
        activation = torch.zeros((1, int(shape[1])), dtype=torch.float32)
        engine._matmul(activation, weight)
        assert projection.calls == [shape], shape
        for other, entry in engine._projection_table.items():
            if other != shape:
                assert entry.calls == [], (shape, other)
        projection.calls.clear()


def test_an_unmeasured_shape_falls_back_to_the_largest_winner() -> None:
    """A shape the block does not hold still runs; it does not raise or skip."""
    engine = _engine()
    _with_stub(engine)
    engine._resolve_projection(1, "decode")
    torch = engine.torch
    weight = torch.zeros((7, HIDDEN), dtype=torch.float32)
    activation = torch.zeros((1, HIDDEN), dtype=torch.float32)
    engine._matmul(activation, weight)
    assert engine._projection.calls == [(7, HIDDEN)]
    largest = max(engine._projection_table, key=lambda shape: shape[0] * shape[1])
    assert engine._projection is engine._projection_table[largest]


def test_the_report_names_a_choice_per_shape() -> None:
    engine = _engine()
    _with_stub(engine)
    engine._resolve_projection(1, "decode")
    report = engine.projection_report()
    assert set(report["per_shape"]) == {
        str(shape) for shape in engine._projection_table
    }
    assert report["active"]["shape"]["phase"] == "decode"


def test_an_uncalibrated_engine_uses_the_reference_kernel() -> None:
    """No calibrator, no table: the path behaves as it did before the mechanism."""
    engine = _engine()
    engine._strategies = None
    engine._projection_table = {}
    engine._resolve_projection(1, "decode")
    assert engine._projection_table == {}
    torch = engine.torch
    weight = torch.eye(HIDDEN, dtype=torch.float32)
    activation = torch.ones((1, HIDDEN), dtype=torch.float32)
    assert torch.allclose(engine._matmul(activation, weight), activation)


def test_a_prefill_and_a_decode_step_of_the_same_width_resolve_separately() -> None:
    """The phase gate.  Eight rows can mean either phase, and they differ.

    Keying the gate on the row count alone let an eight-token prefill and a batch-8
    decode step share one resolution, so one of them ran a formulation measured in
    the other's regime.  Asserting both phases were asked for is what stops the gate
    from silently narrowing back to ``rows``.
    """
    engine = _engine()
    stub = _with_stub(engine)

    engine._forward_device(np.zeros(8, dtype=np.int64), None, reserve=16, logits="last")
    prefill = {(phase, rows) for phase, rows, _, _ in stub.asked}
    assert prefill == {("prefill", 8)}

    stub.asked.clear()
    ids = np.zeros((8, 1), dtype=np.int64)
    engine._forward_device(ids, None, reserve=4, batched=True, logits="last")
    decode = {(phase, rows) for phase, rows, _, _ in stub.asked}
    assert decode == {("decode", 8)}


def test_the_resolution_gate_does_not_re_resolve_a_repeated_pass() -> None:
    """One resolution per pass shape, so the hot path stays a dictionary lookup."""
    engine = _engine()
    stub = _with_stub(engine)
    _, cache = engine._forward_device(
        np.zeros(1, dtype=np.int64), None, reserve=16, logits="last"
    )
    first = len(stub.asked)
    assert first > 0
    for _ in range(3):
        _, cache = engine._forward_device(
            np.zeros(1, dtype=np.int64), cache, reserve=16, logits="last"
        )
    assert len(stub.asked) == first


def test_the_warm_up_resolves_the_decode_class_before_any_request() -> None:
    """TTFT does not pay for a measurement that a load can pay for instead."""
    engine = _engine()
    stub = _with_stub(engine)
    assert engine.warmup_kernels() is True
    assert ("decode", 1) in {(phase, rows) for phase, rows, _, _ in stub.asked}
    assert engine._projection_rows == (1, "decode")

    before = len(stub.asked)
    engine._forward_device(np.zeros(1, dtype=np.int64), None, reserve=8, logits="last")
    assert len(stub.asked) == before


def test_the_warm_up_resolves_the_class_a_request_spends_its_time_in() -> None:
    """Both warm-up forwards are one token wide, which *is* the decode class.

    The prefill width varies per request and is deliberately not guessed at: a
    prefill resolves its own class on first use, and it is also the pass that
    amortizes a probe best.
    """
    engine = _engine()
    stub = _with_stub(engine)
    engine.warmup_kernels()
    assert {phase for phase, _, _, _ in stub.asked} == {"decode"}
    assert all(rows == 1 for _, rows, _, _ in stub.asked)


def test_the_warm_up_leaves_no_cache_behind() -> None:
    engine = _engine()
    _with_stub(engine)
    engine.warmup_kernels()
    assert getattr(engine, "_cache", None) is None


def test_the_warm_up_can_be_switched_off_to_reproduce_a_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator measuring first-token cost must be able to put it back."""
    monkeypatch.setenv("AETHER_KERNEL_WARMUP", "0")
    engine = _engine()
    stub = _with_stub(engine)
    assert engine.warmup_kernels() is False
    assert stub.asked == []
    assert engine._projection_rows is None


def test_a_failing_warm_up_never_fails_a_load() -> None:
    """Warming is an optimisation; a load that cannot warm still serves."""
    engine = _engine()
    _with_stub(engine)

    def explode(*args: Any, **kwargs: Any):
        raise RuntimeError("no kernel for this device")

    engine._forward_device = explode  # type: ignore[method-assign]
    assert engine.warmup_kernels() is False


def test_the_backend_warms_the_engine_it_just_loaded() -> None:
    """The wiring, not the method: a load calls it, and does not require it."""
    import inspect

    from aether.backends import torch_backend

    source = inspect.getsource(torch_backend)
    assert "warmup_kernels" in source
