"""Decode kernel-strategy calibration, profiling, and the sliding-window fast path.

These are the CPU-verifiable half of the batch-scaling work.  They pin the
*architecture* — that a strategy is measured once per shape class, validated for
numerical equivalence before it can be selected, cached by hardware signature, never
re-measured on the hot path, and always falls back to the reference kernel — without
depending on any accelerator being present.

What they deliberately do **not** claim is that any particular candidate is fastest.
That is a property of the device and is decided by measurement at run time; a test that
asserted a winner would be asserting the hardware it happened to run on.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aether.placement.ledger import CalibrationLedger  # noqa: E402
from aether.runtime.cpu_engine import (  # noqa: E402
    CPUExecutionEngine, LayerWeights, ModelWeights,
)
from aether.runtime.decode_profile import (  # noqa: E402
    DecodeProfile, profile_batch_scaling, profile_engine,
)
from aether.runtime.kernel_strategy import (  # noqa: E402
    CALIBRATION_VERSION, ShapeClass, StrategyCalibrator, calibration_enabled,
)
from aether.runtime.torch_engine import TorchAEGEngine  # noqa: E402


def engine(layers: int = 3, hidden: int = 128, heads: int = 4, kv_heads: int = 2,
           inter: int = 256, vocab: int = 256, window: int | None = None) -> TorchAEGEngine:
    rng = np.random.default_rng(0)
    head_dim = hidden // heads

    def w(out: int, inp: int) -> np.ndarray:
        return rng.standard_normal((out, inp)).astype(np.float32) * 0.05

    stack = [
        LayerWeights(
            attention_norm=np.ones(hidden, dtype=np.float32),
            q_proj=w(heads * head_dim, hidden), k_proj=w(kv_heads * head_dim, hidden),
            v_proj=w(kv_heads * head_dim, hidden), o_proj=w(hidden, heads * head_dim),
            ffn_norm=np.ones(hidden, dtype=np.float32),
            gate_proj=w(inter, hidden), up_proj=w(inter, hidden),
            down_proj=w(hidden, inter),
        )
        for _ in range(layers)
    ]
    weights = ModelWeights(
        embedding=w(vocab, hidden), layers=stack,
        final_norm=np.ones(hidden, dtype=np.float32), lm_head=w(vocab, hidden),
        context_length=4096,
    )
    built = TorchAEGEngine(
        CPUExecutionEngine(weights, num_heads=heads, num_kv_heads=kv_heads), "cpu"
    )
    if window is not None:
        built.layer_plan = [(True, window, plan[2], plan[3]) for plan in built.layer_plan]
    return built


def probe_for(rows: int, k: int, n: int):
    def make():
        return torch.randn(rows, k), torch.randn(n, k), torch.randn(n)
    return make


# ── shape classes ─────────────────────────────────────────────────────────────

def test_the_shape_class_buckets_rows_by_powers_of_two() -> None:
    """A GEMM tile sees a power-of-two row count, so that is the calibration unit."""
    keys = {
        rows: ShapeClass.of(
            phase="decode", rows=rows, in_features=1024, out_features=1024,
            dtype="float16", device_kind="cuda-sm80",
        ).key
        for rows in (1, 2, 3, 4, 5, 8)
    }
    assert keys[2] == keys[3], "3 rows shares 2's tile and must share its calibration"
    assert keys[4] == keys[5]
    assert len({keys[1], keys[2], keys[4], keys[8]}) == 4, "distinct tiles, distinct keys"


def test_prefill_and_decode_are_calibrated_separately() -> None:
    """They sit in different regimes, so one measurement cannot serve both."""
    common = dict(rows=1, in_features=1024, out_features=1024,
                  dtype="float16", device_kind="cpu")
    assert (
        ShapeClass.of(phase="decode", **common).key
        != ShapeClass.of(phase="prefill", **common).key
    )


def test_the_key_carries_the_calibration_version() -> None:
    """Changing a candidate's semantics must not silently reuse an old winner."""
    key = ShapeClass.of(phase="decode", rows=1, in_features=8, out_features=8,
                        dtype="float32", device_kind="cpu").key
    assert key.startswith(CALIBRATION_VERSION)


def test_dtype_and_device_are_part_of_the_class() -> None:
    """The best formulation differs by precision and by architecture."""
    base = dict(phase="decode", rows=1, in_features=512, out_features=512)
    assert (
        ShapeClass.of(**base, dtype="float16", device_kind="cuda-sm80").key
        != ShapeClass.of(**base, dtype="float32", device_kind="cuda-sm80").key
    )
    assert (
        ShapeClass.of(**base, dtype="float16", device_kind="cuda-sm80").key
        != ShapeClass.of(**base, dtype="float16", device_kind="cuda-sm75").key
    )


# ── selection ─────────────────────────────────────────────────────────────────

def test_a_candidate_is_only_selected_after_measurement_and_a_correctness_check() -> None:
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=2, in_features=128, out_features=256,
        dtype=torch.float32, probe=probe_for(2, 128, 256),
    )
    assert choice.source == "measured"
    assert choice.candidates, "the field must be recorded, not just the winner"
    assert all(c.equivalent or c.error for c in choice.candidates)
    chosen = next(c for c in choice.candidates if c.name == choice.name)
    assert chosen.equivalent and chosen.seconds > 0


def test_the_winner_is_numerically_equivalent_to_the_reference() -> None:
    """Selection may change how the arithmetic is asked for, never what it produces."""
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    x, weight, bias = probe_for(4, 96, 192)()
    choice = calibrator.choose(
        phase="decode", rows=4, in_features=96, out_features=192,
        dtype=torch.float32, probe=lambda: (x, weight, bias),
    )
    applied = calibrator.strategy(choice)(x, weight, bias)
    reference = torch.nn.functional.linear(x, weight, bias)
    assert applied.shape == reference.shape
    assert torch.allclose(applied, reference, rtol=1e-4, atol=1e-4)


def test_a_wrong_candidate_can_never_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """The correctness gate is the thing that makes this safe to enable everywhere."""
    from aether.runtime import kernel_strategy

    def sabotage(torch_mod, x, weight, bias):
        return torch.zeros(
            (*x.shape[:-1], int(weight.shape[0])), dtype=x.dtype, device=x.device
        )

    monkeypatch.setitem(kernel_strategy._CANDIDATES, "sabotaged", sabotage)
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=2, in_features=64, out_features=64,
        dtype=torch.float32, probe=probe_for(2, 64, 64),
    )
    assert choice.name != "sabotaged"
    bad = next(c for c in choice.candidates if c.name == "sabotaged")
    assert not bad.equivalent and bad.error == "numeric mismatch"


def test_an_unsupported_candidate_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend with an incomplete operator set must lose a candidate, not a load."""
    from aether.runtime import kernel_strategy

    def unsupported(torch_mod, x, weight, bias):
        raise RuntimeError("operator not implemented for this backend")

    monkeypatch.setitem(kernel_strategy._CANDIDATES, "unsupported", unsupported)
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=1, in_features=64, out_features=64,
        dtype=torch.float32, probe=probe_for(1, 64, 64),
    )
    assert choice.source == "measured"
    failed = next(c for c in choice.candidates if c.name == "unsupported")
    assert not failed.eligible and "not implemented" in failed.error


def test_a_win_inside_the_noise_band_is_not_a_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two formulations within measurement noise resolve to the reference, always."""
    from aether.runtime import kernel_strategy

    monkeypatch.setattr(kernel_strategy, "_MIN_MARGIN", 0.5)
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=2, in_features=64, out_features=64,
        dtype=torch.float32, probe=probe_for(2, 64, 64),
    )
    assert choice.name == "linear"
    assert choice.speedup == pytest.approx(1.0)


# ── bounded, deferrable, disableable ──────────────────────────────────────────

def test_calibration_can_be_switched_off_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator reproducing a measurement must be able to remove the mechanism."""
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    assert not calibration_enabled()
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=1, in_features=64, out_features=64,
        dtype=torch.float32, probe=probe_for(1, 64, 64),
    )
    assert choice.source == "disabled" and choice.name == "linear"


def test_an_exhausted_budget_defers_rather_than_ranking_a_partial_field() -> None:
    calibrator = StrategyCalibrator(torch, torch.device("cpu"), budget_seconds=0.0)
    choice = calibrator.choose(
        phase="decode", rows=1, in_features=64, out_features=64,
        dtype=torch.float32, probe=probe_for(1, 64, 64),
    )
    assert choice.source == "deferred" and choice.name == "linear"
    assert "budget" in choice.detail


def test_no_probe_means_no_guess() -> None:
    """Without representative tensors the honest answer is the reference kernel."""
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=1, in_features=64, out_features=64,
        dtype=torch.float32, probe=None,
    )
    assert choice.source == "deferred" and choice.name == "linear"


def test_a_broken_probe_never_breaks_the_caller() -> None:
    def explode():
        raise MemoryError("device out of memory allocating probe tensors")

    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    choice = calibrator.choose(
        phase="decode", rows=1, in_features=64, out_features=64,
        dtype=torch.float32, probe=explode,
    )
    assert choice.source == "deferred" and choice.name == "linear"


# ── caching ───────────────────────────────────────────────────────────────────

def test_a_class_is_measured_once_per_process() -> None:
    calibrator = StrategyCalibrator(torch, torch.device("cpu"))
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return probe_for(2, 64, 64)()

    for _ in range(5):
        calibrator.choose(phase="decode", rows=2, in_features=64, out_features=64,
                          dtype=torch.float32, probe=counted)
    assert calls["n"] == 1, "the hot path must never re-benchmark itself"


def test_a_calibration_survives_into_a_second_process(tmp_path) -> None:
    """The second load on a machine performs no measurement at all."""
    store = CalibrationLedger(tmp_path / "cal.json", autosave=True)
    first = StrategyCalibrator(torch, torch.device("cpu"), store=store,
                               signature="cpu:Host:1GiB", backend_build="build-a")
    chosen = first.choose(phase="decode", rows=2, in_features=128, out_features=128,
                          dtype=torch.float32, probe=probe_for(2, 128, 128))
    assert chosen.source == "measured"

    reopened = CalibrationLedger(tmp_path / "cal.json", autosave=False)
    second = StrategyCalibrator(torch, torch.device("cpu"), store=reopened,
                                signature="cpu:Host:1GiB", backend_build="build-a")
    reused = second.choose(phase="decode", rows=2, in_features=128, out_features=128,
                           dtype=torch.float32, probe=None)
    assert reused.source == "cached"
    assert reused.name == chosen.name


def test_a_different_backend_build_does_not_reuse_the_calibration(tmp_path) -> None:
    """The kernel a formulation maps to is a property of the runtime build."""
    store = CalibrationLedger(tmp_path / "cal.json", autosave=False)
    StrategyCalibrator(torch, torch.device("cpu"), store=store,
                       signature="s", backend_build="build-a").choose(
        phase="decode", rows=2, in_features=64, out_features=64,
        dtype=torch.float32, probe=probe_for(2, 64, 64))
    other = StrategyCalibrator(torch, torch.device("cpu"), store=store,
                               signature="s", backend_build="build-b")
    assert other.choose(phase="decode", rows=2, in_features=64, out_features=64,
                        dtype=torch.float32, probe=None).source == "deferred"


def test_notes_from_two_namespaces_do_not_overwrite_each_other(tmp_path) -> None:
    store = CalibrationLedger(tmp_path / "cal.json", autosave=False)
    store.record_notes("s", "b", {"decode_strategies": {"k1": "addmm"}})
    store.record_notes("s", "b", {"decode_strategies": {"k2": "linear"}})
    store.record_notes("s", "b", {"other_namespace": {"x": 1}})
    notes = store.get("s", "b").notes
    assert notes["decode_strategies"] == {"k1": "addmm", "k2": "linear"}
    assert notes["other_namespace"] == {"x": 1}


def test_a_store_that_cannot_persist_is_not_fatal() -> None:
    class Hostile:
        def get(self, *_a, **_k):
            raise OSError("read-only filesystem")

    calibrator = StrategyCalibrator(torch, torch.device("cpu"), store=Hostile(),
                                    signature="s", backend_build="b")
    choice = calibrator.choose(phase="decode", rows=2, in_features=64, out_features=64,
                               dtype=torch.float32, probe=probe_for(2, 64, 64))
    assert choice.source == "measured"


# ── engine integration ────────────────────────────────────────────────────────

def _expected_classes(eng, phase: str, rows: int) -> set:
    """The shape classes a pass of this width covers, as the calibrator keys them."""
    from aether.runtime.kernel_strategy import ShapeClass

    return {
        ShapeClass.of(
            phase=phase, rows=rows,
            in_features=int(weight.shape[1]), out_features=int(weight.shape[0]),
            dtype=str(weight.dtype).replace("torch.", ""), device_kind="cpu",
        ).key
        for weight, _ in eng._projection_probe_shapes()
    }


def test_the_engine_resolves_one_strategy_per_class_per_pass_and_none_per_step() -> None:
    """A pass resolves each class it covers once; a step after the first resolves none.

    The granularity is per *shape*, not per pass: a block holds several weight
    magnitudes and :class:`ShapeClass` keys on the magnitude, so one winner cannot
    stand for all of them.  Two shapes can still share a class -- the magnitude is
    bucketed by powers of two -- which is why the expectation is derived from the
    keys rather than from the number of shapes.  What must not happen is a
    measurement on the hot path, and that is what the decode loop asserts.
    """
    eng = engine()
    shapes = len(eng._projection_probe_shapes())
    assert shapes > 1, "the fixture must hold more than one projection magnitude"

    ids = np.arange(16, dtype=np.int64) % 256
    _, cache = eng._forward_device(ids, None, reserve=64, logits="last")
    prefill_classes = _expected_classes(eng, "prefill", 16)
    assert set(eng._strategies._choices) == prefill_classes
    assert len(eng._projection_table) == shapes, "the table is keyed per shape"

    token = torch.zeros(1, dtype=torch.long)
    _, cache = eng._forward_device(token, cache, logits="last")
    decode_classes = _expected_classes(eng, "decode", 1)
    assert set(eng._strategies._choices) == prefill_classes | decode_classes
    resolved = len(eng._strategies._choices)

    for _ in range(10):
        _, cache = eng._forward_device(token, cache, logits="last")
    assert len(eng._strategies._choices) == resolved, (
        "nothing is measured per step after the first"
    )


def test_the_engine_keeps_the_reference_when_calibration_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    eng = engine()
    assert eng._strategies is None
    ids = np.arange(8, dtype=np.int64) % 256
    logits, _ = eng._forward_device(ids, None, reserve=16, logits="last")
    assert torch.isfinite(logits).all()


def test_selection_does_not_change_what_the_model_computes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same ids must produce the same logits with calibration on and off."""
    ids = np.arange(12, dtype=np.int64) % 256
    token = torch.zeros(1, dtype=torch.long)
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    plain = engine()
    reference, cache = plain._forward_device(ids, None, reserve=32, logits="last")
    reference_step, _ = plain._forward_device(token, cache, logits="last")
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "1")
    tuned = engine()
    produced, cache2 = tuned._forward_device(ids, None, reserve=32, logits="last")
    produced_step, _ = tuned._forward_device(token, cache2, logits="last")
    assert torch.allclose(reference, produced, rtol=2e-2, atol=2e-2)
    assert torch.allclose(reference_step, produced_step, rtol=2e-2, atol=2e-2)


def test_the_engine_reports_which_strategy_is_in_force() -> None:
    eng = engine()
    ids = np.arange(8, dtype=np.int64) % 256
    eng._forward_device(ids, None, reserve=16, logits="last")
    report = eng.projection_report()
    assert report["enabled"] is not False
    assert report["active"]["shape"]["device_kind"] == "cpu"
    assert report["active"]["name"] in {"linear", "transposed", "padded_rows", "addmm"}


# ── the profiler ──────────────────────────────────────────────────────────────

def test_the_profiler_attributes_a_decode_step_to_phases() -> None:
    totals = profile_engine(engine(), batch=1, context=16, steps=4)
    assert totals.steps == 4
    assert totals.wall_seconds > 0
    per_step = totals.per_step()
    for phase in ("projections", "attention", "kv", "norm", "logits"):
        assert per_step.get(phase, 0.0) > 0, f"{phase} was not attributed"
    assert "other" in per_step, "dispatch overhead must be reported as a residual"


def test_the_residual_is_never_negative() -> None:
    """Unattributed time is wall minus phases; a negative would mean double counting."""
    totals = profile_engine(engine(), batch=2, context=16, steps=4)
    assert totals.unattributed >= 0.0
    assert totals.attributed <= totals.wall_seconds + 1e-9


def test_instrumentation_is_removed_when_the_block_exits() -> None:
    """The decode path must carry no profiling cost when nobody is profiling."""
    eng = engine()
    assert "_matmul" not in eng.__dict__, "the engine starts uninstrumented"
    with DecodeProfile(eng):
        assert "_matmul" in eng.__dict__, "the wrapper is an instance attribute"
    assert "_matmul" not in eng.__dict__, "and it is gone again on exit"


def test_profiling_one_engine_does_not_touch_another() -> None:
    """The harness keeps two backends resident, so instrumentation is per instance."""
    first, second = engine(), engine()
    with DecodeProfile(first):
        assert "_attention" in first.__dict__
        assert "_attention" not in second.__dict__
    assert "_attention" not in first.__dict__


def test_batch_scaling_reports_efficiency_and_names_the_growing_phase() -> None:
    report = profile_batch_scaling(engine, batches=(1, 2), context=16, steps=3)
    assert [cell["batch"] for cell in report["cells"]] == [1, 2]
    for cell in report["cells"]:
        assert cell["ms_per_step"] > 0
        assert cell["tokens_per_s"] > 0
        assert 0.0 < cell["scaling_efficiency"] <= 4.0
    assert report["cells"][0]["scaling_efficiency"] == pytest.approx(1.0, abs=1e-6)
    assert report["largest_absolute_growth"]["phase"]


def test_the_profiler_works_for_a_batched_engine() -> None:
    totals = profile_engine(engine(), batch=4, context=16, steps=4)
    assert totals.steps == 4
    assert totals.per_step()["projections"] > 0


# ── the sliding-window decode fast path ───────────────────────────────────────

def masked_reference(engine_, q, k, v, qpos, kpos, window, *, batched):
    """Full key range, explicit window mask, float64 — no slicing anywhere.

    Written out here rather than taken from another engine path: both engine paths now
    apply the slice, so comparing them would only prove they agree with each other.
    """
    reps = engine_.num_heads // engine_.num_kv_heads
    scale = engine_.base_attention_scale
    q64, k64, v64 = q.double(), k.double(), v.double()
    if reps > 1:
        k64 = k64.repeat_interleave(reps, dim=-2)
        v64 = v64.repeat_interleave(reps, dim=-2)
    if batched:
        scores = torch.einsum("bqhd,bkhd->bhqk", q64, k64) * scale
        allowed = (kpos.unsqueeze(1) <= qpos.unsqueeze(-1)) & (
            kpos.unsqueeze(1) >= qpos.unsqueeze(-1) - window + 1
        )
        scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, -1), v64)
    scores = torch.einsum("qhd,khd->hqk", q64, k64) * scale
    allowed = (kpos.unsqueeze(0) <= qpos.unsqueeze(-1)) & (
        kpos.unsqueeze(0) >= qpos.unsqueeze(-1) - window + 1
    )
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), v64)


@pytest.mark.parametrize(
    ("keys", "window"), [(40, 8), (100, 32), (17, 16), (9, 32)]
)
@pytest.mark.parametrize("batched", [False, True])
def test_a_windowed_decode_step_equals_the_masked_reference(
    keys: int, window: int, batched: bool
) -> None:
    """Slicing the cache to the window must not change a single number.

    Covers a window smaller than the context, one larger than it (where the constraint
    is vacuous), and an exact boundary — the three cases a suffix slice can get wrong.
    """
    eng = engine(window=window)
    heads, kv, dim = eng.num_heads, eng.num_kv_heads, eng.head_dim
    torch.manual_seed(7)
    if batched:
        q = torch.randn(3, 1, heads, dim)
        k = torch.randn(3, keys, kv, dim)
        v = torch.randn(3, keys, kv, dim)
        positions = torch.full((3, 1), keys - 1, dtype=torch.long)
        key_positions = torch.arange(keys).unsqueeze(0).expand(3, keys)
    else:
        q = torch.randn(1, heads, dim)
        k = torch.randn(keys, kv, dim)
        v = torch.randn(keys, kv, dim)
        positions = torch.tensor([keys - 1])
        key_positions = torch.arange(keys)
    produced = eng._attention(q, k, v, positions, key_positions, window, None, live=None)
    expected = masked_reference(
        eng, q, k, v, positions, key_positions, window, batched=batched
    )
    assert produced.shape == expected.shape
    assert float((produced.double() - expected).abs().max()) < 1e-5


def test_a_windowed_decode_step_stops_growing_with_context() -> None:
    """The whole point: attention reads the window, not the history."""
    costs = []
    for context in (64, 512):
        eng = engine(window=32)
        totals = profile_engine(eng, batch=1, context=context, steps=6)
        costs.append(totals.per_step().get("attention", 0.0))
    assert costs[1] < costs[0] * 3.0, (
        f"windowed attention cost grew from {costs[0] * 1e3:.3f} to "
        f"{costs[1] * 1e3:.3f} ms/step over an 8x longer context"
    )


def test_a_ragged_batch_still_uses_the_mask() -> None:
    """With per-row pad counts one slice cannot express the window, so it must not try."""
    eng = engine(window=8)
    heads, kv, dim, keys = eng.num_heads, eng.num_kv_heads, eng.head_dim, 40
    torch.manual_seed(2)
    q = torch.randn(2, 1, heads, dim)
    k = torch.randn(2, keys, kv, dim)
    v = torch.randn(2, keys, kv, dim)
    positions = torch.tensor([[keys - 1], [keys - 5]])
    key_positions = torch.arange(keys).unsqueeze(0).expand(2, keys)
    live = torch.ones(2, keys, dtype=torch.bool)
    live[1, -4:] = False
    out = eng._attention(q, k, v, positions, key_positions, 8, None, live=live)
    assert out.shape == (2, 1, heads, dim)
    assert torch.isfinite(out).all()
