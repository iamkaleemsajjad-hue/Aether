"""Time-to-first-token contracts for the accelerator decode path.

Every test here pins a *cause* of first-token latency rather than a duration, because
a duration is a property of the host it was measured on and a cause is a property of
the code. The causes, in the order a request meets them: the artifact path resolved
again per request; the prompt's ids validated by copying two scalars back off the
device; the rotary frequencies re-derived on the host at the head of every pass; and —
the decisive one — the first token read only *after* the next decode step had been
queued in front of it, so a token the prefill had already determined arrived one whole
step late.

Speed and output quality are held fixed by the same tests rather than by a separate
timing suite. Throughput is the count of forward passes and the order they are queued
in, so both are asserted; quality is the token ids, so those are asserted to be
identical to the ones the previous ordering produced.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import numpy as np
import pytest

from test_memory_efficiency import VOCAB, _cpu_engine

PROMPT = np.asarray([1, 3, 5, 2, 7, 4], dtype=np.int64)


def _engine(**kwargs):
    from aether.runtime.torch_engine import TorchAEGEngine

    return TorchAEGEngine(_cpu_engine(**kwargs), "cpu")


def _count_forward_passes(engine) -> list[tuple[int, str | None]]:
    """Record ``(token count, logits mode)`` for each device forward pass."""
    seen: list[tuple[int, str | None]] = []
    real = engine._forward_device

    def traced(token_ids, cache=None, **kwargs):
        size = int(token_ids.numel()) if hasattr(token_ids, "numel") else int(np.asarray(token_ids).size)
        seen.append((size, kwargs.get("logits")))
        return real(token_ids, cache, **kwargs)

    engine._forward_device = traced  # type: ignore[method-assign]
    return seen


# ── The first token does not wait for the next step ──────────────────────────


def test_the_first_token_costs_exactly_one_forward_pass() -> None:
    """The prefill determines the first token; nothing after it may be waited on.

    The pipelined loop queues step N+1 before collecting step N's token, which is
    what hides host launch cost behind device execution. Reading the token with
    ``.item()`` after that queueing turned the overlap into a delay: the copy is
    blocking and in-order, so it completed only once the whole lookahead step had
    run. Two forward passes therefore stood between the prompt and the first token.
    """
    pytest.importorskip("torch")
    engine = _engine()
    passes = _count_forward_passes(engine)

    first = next(engine.generate_iter(PROMPT, max_tokens=8, temperature=0.0))

    assert isinstance(first, int)
    assert len(passes) == 1, f"first token waited for {len(passes)} forward passes"
    assert passes[0] == (PROMPT.size, "last")


def test_the_whole_generation_still_costs_one_pass_per_token() -> None:
    """Latency was moved, not work: the pass count per token is unchanged."""
    pytest.importorskip("torch")
    engine = _engine()
    passes = _count_forward_passes(engine)

    tokens = list(engine.generate_iter(PROMPT, max_tokens=8, temperature=0.0))

    # One prompt pass plus one per decode step.  This prompt reaches the budget
    # without emitting a stop token, so every step runs.
    assert len(tokens) == 8
    assert len(passes) == 1 + len(tokens)
    assert all(size == 1 for size, _ in passes[1:])


def test_stopping_early_does_not_pay_for_a_step_whose_rows_are_discarded() -> None:
    """The lookahead is speculative; a stop must not still be charged for it.

    On a device that executes as it is called there is no queue for the lookahead to
    overlap with, so running one before yielding would buy nothing and cost a whole
    step — including, at the end, a step whose KV rows are immediately rewound away.
    """
    pytest.importorskip("torch")
    engine = _engine()
    passes = _count_forward_passes(engine)

    first = next(engine.generate_iter(PROMPT, max_tokens=8, temperature=0.0))
    tokens = list(engine.generate_iter(PROMPT, max_tokens=8, temperature=0.0, eos_token_id=first))

    assert tokens == [first]
    # The prompt pass of each of the two generations, and nothing else: the stop
    # token was the first thing sampled, so no decode step was ever needed.
    assert len(passes) == 2


def test_reading_the_token_early_does_not_change_the_tokens() -> None:
    """Quality is the token ids, so the ids are what is held fixed.

    The reference is the ordering this fix replaced: sample, queue the next step,
    *then* read. Reproduced here directly from the engine's own primitives, so the
    comparison is against the old behaviour rather than against a stored list.
    """
    pytest.importorskip("torch")
    engine = _engine()
    fast = list(engine.generate_iter(PROMPT, max_tokens=8, temperature=0.0))

    reference_engine = _engine()
    _, cache = reference_engine._forward_device(PROMPT, None, logits="last")
    token_device = reference_engine._sample_device(cache.last_logits, 0.0, 0, 1.0)
    late: list[int] = []
    for _ in range(8):
        _, cache = reference_engine._forward_device(
            token_device.reshape(1), cache, logits="last"
        )
        following = reference_engine._sample_device(cache.last_logits, 0.0, 0, 1.0)
        late.append(int(token_device.item()))
        token_device = following

    assert fast == late[: len(fast)]


# ── The handoff that lets it happen ──────────────────────────────────────────
#
# ``token.item()`` is a blocking, in-order copy: it completes only after every
# kernel already queued ahead of it. Staging the copy the moment the token is
# sampled, and waiting on an event recorded right after that copy, separates "the
# token is on the host" from "the queue has drained". The three routes below are
# the three answers a device can give, and each one has to be reachable.


class _FakeEvent:
    """Records once, waits once, and refuses to be waited on before recording."""

    def __init__(self) -> None:
        self.recorded = False
        self.waited = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        assert self.recorded, "waited on an event that was never recorded"
        self.waited = True


class _FakeBuffer:
    """A pinned host slot: the copy into it must not block the stream."""

    def __init__(self) -> None:
        self.value: int | None = None

    def copy_(self, source, non_blocking: bool = False) -> None:
        assert non_blocking, "a staged readback must not block the stream"
        self.value = int(source[0])

    def __getitem__(self, index: int) -> int:
        assert self.value is not None, "read a slot before its copy was issued"
        return self.value


class _FakeToken:
    """A device-resident scalar: readable only by an explicit copy."""

    def __init__(self, value: int) -> None:
        self.value = value
        self.items = 0

    def reshape(self, *shape):
        return [self.value]

    def item(self) -> int:
        self.items += 1
        return self.value


def _handoff_stub(device_type: str, *, staging: bool):
    """An engine reduced to what the handoff actually reads.

    Built with ``__new__`` on purpose: the readback state has to live as class-level
    defaults, because the tensor-parallel executor never runs the parent constructor.
    """
    from aether.runtime.torch_engine import TorchAEGEngine

    stub = TorchAEGEngine.__new__(TorchAEGEngine)
    stub.device = types.SimpleNamespace(type=device_type)
    allocated: list[_FakeBuffer] = []

    def empty(*shape, dtype=None, pin_memory=False):
        assert pin_memory, "an asynchronous copy needs a pinned destination"
        allocated.append(_FakeBuffer())
        return allocated[-1]

    namespace = types.SimpleNamespace(Event=_FakeEvent) if staging else types.SimpleNamespace()
    stub.torch = types.SimpleNamespace(long=object(), empty=empty, **{device_type: namespace})
    return stub, allocated


def test_a_synchronous_device_reads_the_token_immediately() -> None:
    """On CPU there is no queue to skip ahead of, so reading now is free."""
    pytest.importorskip("torch")
    engine = _engine()
    token = _FakeToken(11)

    handoff = engine._begin_readback(token)

    assert handoff[0] == 11
    assert handoff[1] is None and handoff[2] is None and handoff[3] is None
    assert engine._finish_readback(handoff) == 11


def test_an_asynchronous_device_stages_the_copy_and_waits_only_on_it() -> None:
    """The event must be recorded at ``begin`` and awaited at ``finish``."""
    pytest.importorskip("torch")
    stub, allocated = _handoff_stub("cuda", staging=True)
    token = _FakeToken(23)

    value, buffer, event, tensor = stub._begin_readback(token)

    assert value is None and tensor is None
    assert buffer is allocated[0] and buffer.value == 23
    assert event.recorded and not event.waited
    assert token.items == 0, "staging must not read the token synchronously"

    assert stub._finish_readback((value, buffer, event, tensor)) == 23
    assert event.waited


def test_two_slots_keep_the_in_flight_token_from_being_overwritten() -> None:
    """One token is being read while the next is sampled, so one slot is not enough."""
    pytest.importorskip("torch")
    stub, allocated = _handoff_stub("cuda", staging=True)

    first = stub._begin_readback(_FakeToken(5))
    second = stub._begin_readback(_FakeToken(6))

    assert first[1] is not second[1]
    assert len(allocated) == 2, "the slots are allocated once, not per token"
    assert stub._finish_readback(first) == 5
    assert stub._finish_readback(second) == 6

    third = stub._begin_readback(_FakeToken(7))
    assert third[1] is first[1], "the slots must be reused, not grown per token"
    assert len(allocated) == 2


def test_a_device_without_events_falls_back_to_the_late_read() -> None:
    """No staging is a valid answer, and it must reproduce the old behaviour."""
    pytest.importorskip("torch")
    stub, allocated = _handoff_stub("xpu", staging=False)
    token = _FakeToken(31)

    handoff = stub._begin_readback(token)

    assert handoff == (None, None, None, token)
    assert allocated == []
    assert stub._finish_readback(handoff) == 31
    assert token.items == 1

    # Probed once, and the negative answer is remembered rather than re-derived.
    assert stub._readback_slots is None
    stub._begin_readback(_FakeToken(32))
    assert allocated == []


def test_the_pipelined_loop_queues_the_next_step_before_it_reads_the_token() -> None:
    """The overlap is the throughput property; asserting it structurally keeps it.

    Reading the token earlier is only correct if the lookahead is still queued
    first — otherwise the host's launch cost stops overlapping device execution and
    decode throughput falls. So the source order is part of the contract.
    """
    from aether.runtime.torch_engine import TorchAEGEngine

    body = inspect.getsource(TorchAEGEngine._generate_pipelined)
    forward = body.index("self._forward_device(")
    queued = body.index("_begin_readback(following)")
    collected = body.index("self._finish_readback(handoff)")
    assert forward < queued < collected


# ── The prompt's ids are a host question ─────────────────────────────────────


def test_validating_a_numpy_prompt_never_touches_the_device() -> None:
    """Ids arrive from a tokenizer as host memory, so their bounds are host-side.

    Asking the device instead means uploading them, then copying two scalars back —
    two blocking round trips at the very head of the prompt pass, to learn something
    the host already knew. Proved by making the upload itself impossible: the error
    that surfaces is the vocabulary error, so validation ran before any upload.
    """
    pytest.importorskip("torch")
    engine = _engine()
    uploads: list[int] = []
    real = engine.torch.as_tensor

    def refuse(*args, **kwargs):
        uploads.append(1)
        raise AssertionError("the prompt was uploaded before it was validated")

    engine.torch = types.SimpleNamespace(**{
        name: getattr(engine.torch, name) for name in dir(engine.torch) if not name.startswith("__")
    })
    engine.torch.as_tensor = refuse
    with pytest.raises(ValueError, match="outside the compiled vocabulary"):
        engine._forward_device(
            np.asarray([1, VOCAB], dtype=np.int64), None, validate_ids=True, logits="last"
        )
    assert uploads == []

    engine.torch.as_tensor = real
    with pytest.raises(ValueError, match="outside the compiled vocabulary"):
        engine._forward_device(
            np.asarray([-1, 1], dtype=np.int64), None, validate_ids=True, logits="last"
        )


def test_validating_a_tensor_prompt_costs_one_readback_not_two() -> None:
    """Both bounds come back in one copy, because each copy is a full stream stall."""
    torch = pytest.importorskip("torch")
    engine = _engine()
    counts = {"item": 0, "tolist": 0}
    real_item, real_tolist = torch.Tensor.item, torch.Tensor.tolist

    def item(self):
        counts["item"] += 1
        return real_item(self)

    def tolist(self):
        counts["tolist"] += 1
        return real_tolist(self)

    torch.Tensor.item, torch.Tensor.tolist = item, tolist
    try:
        with pytest.raises(ValueError, match="outside the compiled vocabulary"):
            engine._forward_device(
                torch.tensor([1, VOCAB], dtype=torch.long),
                None,
                validate_ids=True,
                logits="last",
            )
    finally:
        torch.Tensor.item, torch.Tensor.tolist = real_item, real_tolist

    assert counts == {"item": 0, "tolist": 1}


def test_a_valid_prompt_is_still_accepted_from_either_container() -> None:
    """Moving the check must not narrow what it accepts."""
    torch = pytest.importorskip("torch")
    engine = _engine()
    from_numpy, _ = engine.forward(PROMPT)
    from_tensor, _ = engine.forward(torch.as_tensor(PROMPT, dtype=torch.long))
    np.testing.assert_array_equal(from_tensor, from_numpy)


# ── The rotary frequencies are a property of the checkpoint ──────────────────


def _count_frequency_derivations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count host-side inverse-frequency derivations inside the executor."""
    import aether.runtime.torch_engine as module

    calls = [0]
    real = module.scaled_inverse_frequencies

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "scaled_inverse_frequencies", counted)
    return calls


def test_a_length_independent_model_derives_its_frequencies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recomputing them per pass could only prove they had not changed.

    The proof costs a double-precision power over the head half-width and a full
    array comparison, on the host, at the head of every forward pass — including the
    prompt pass, where it is the first thing time-to-first-token pays for.
    """
    pytest.importorskip("torch")
    engine = _engine()
    engine.forward(PROMPT)  # build the tables

    calls = _count_frequency_derivations(monkeypatch)
    list(engine.generate_iter(PROMPT, max_tokens=8, temperature=0.0))

    assert calls[0] == 0
    assert engine._rope_is_length_sensitive() is False


def test_a_length_dependent_model_still_re_derives_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dynamic`` rescales the base with the current length, so it must not be cached."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    cpu = _cpu_engine()
    cpu.weights.rope_scaling = {
        "rope_type": "dynamic",
        "factor": 2.0,
        "original_max_position_embeddings": 8,
    }
    cpu.weights.context_length = 16
    engine = TorchAEGEngine(cpu, "cpu")
    engine.forward(PROMPT)

    assert engine._rope_is_length_sensitive() is True
    calls = _count_frequency_derivations(monkeypatch)
    list(engine.generate_iter(PROMPT, max_tokens=4, temperature=0.0))
    assert calls[0] > 0, "a length-dependent scheme must be re-derived per length"


def test_skipping_the_derivation_does_not_change_the_logits() -> None:
    """Memoization is a latency change; it must not be an arithmetic one."""
    pytest.importorskip("torch")
    memoized = _engine()
    always = _engine()
    # Forcing the flag makes ``always`` take the pre-memoization path on every pass.
    always._rope_length_sensitive = True

    for length in (1, 3, PROMPT.size):
        prompt = PROMPT[:length]
        np.testing.assert_array_equal(
            memoized.forward(prompt)[0], always.forward(prompt)[0]
        )
    assert list(memoized.generate_iter(PROMPT, max_tokens=8, temperature=0.0)) == list(
        always.generate_iter(PROMPT, max_tokens=8, temperature=0.0)
    )


def test_growing_the_tables_is_still_triggered_by_a_longer_sequence() -> None:
    """The memo must short-circuit only when the tables already cover the length."""
    pytest.importorskip("torch")
    engine = _engine()
    engine._ensure_rope(4)
    assert int(engine._cos.shape[0]) >= 4
    small = int(engine._cos.shape[0])

    engine._ensure_rope(small + 1)
    assert int(engine._cos.shape[0]) >= small + 1


# ── The artifact is found once, not once per request ─────────────────────────


def _runtime_with_artifact(tmp_path: Path, model_id: str):
    """A runtime whose cache holds a directory the resolver accepts as a package."""
    import json

    from aether.runtime.config import RuntimeConfig
    from aether.runtime.runtime import Runtime
    from aether.utils.file_io import aeg_cache_path

    package = tmp_path / "models" / aeg_cache_path(model_id, tmp_path).name
    package.mkdir(parents=True, exist_ok=True)
    (package / "manifest.json").write_text(
        json.dumps({"model_id": model_id}), encoding="utf-8"
    )
    runtime = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path), hf_offline=True))
    return runtime, package


def test_a_loaded_model_answers_the_path_question_from_the_memo(tmp_path: Path) -> None:
    """Loading is what establishes the path, so loading is what fills the memo.

    Resolution walks several cache layouts with an ``is_file`` probe each and then a
    ``Path.resolve``. That is host-filesystem work, ahead of the prompt, repeated per
    request — and an overlay or network mount charges far more for a stat than local
    disk does.
    """
    model_id = "Qwen/Qwen3-0.6B"
    runtime, package = _runtime_with_artifact(tmp_path, model_id)

    resolved = runtime._resolve_aeg_path(model_id)
    assert resolved is not None
    runtime._aeg_paths[model_id] = (resolved, str(Path(resolved).resolve()))

    calls = [0]
    real = runtime._resolve_aeg_path

    def counted(value: str):
        calls[0] += 1
        return real(value)

    runtime._resolve_aeg_path = counted  # type: ignore[method-assign]
    for _ in range(5):
        path, key = runtime._loaded_aeg_path(model_id)
        assert Path(path) == package
        assert key == str(package.resolve())
    assert calls[0] == 0


def test_an_unloaded_model_still_resolves_from_the_filesystem(tmp_path: Path) -> None:
    """The memo is filled by loading, never consulted in place of resolving.

    Discovery has to stay a property of the filesystem: an artifact compiled by
    another process must be found by this one.
    """
    model_id = "Qwen/Qwen3-0.6B"
    runtime, package = _runtime_with_artifact(tmp_path, model_id)

    assert runtime._aeg_paths == {}
    path, key = runtime._loaded_aeg_path(model_id)
    assert Path(path) == package
    assert key == str(package.resolve())
    assert runtime._loaded_aeg_path("nobody/never-compiled") == (None, None)


def test_the_memo_is_written_and_cleared_where_the_load_is(tmp_path: Path) -> None:
    """An entry must exist exactly while the model it describes is loaded."""
    from aether.runtime.runtime import Runtime

    load = inspect.getsource(Runtime._load_model)
    assert "self._aeg_paths[model_id] = (" in load
    assert load.index("self._loaded_backends[model_id] = backend") < load.index(
        "self._aeg_paths[model_id] = ("
    )
    assert "self._aeg_paths.pop(model_id, None)" in inspect.getsource(Runtime.remove)


def test_neither_request_path_resolves_the_artifact_itself() -> None:
    """Both ``generate`` and ``generate_stream`` must read through the memo."""
    from aether.runtime.runtime import Runtime

    for method in (Runtime.generate, Runtime.generate_stream):
        body = inspect.getsource(method)
        assert "_loaded_aeg_path" in body, method.__name__
        assert "_resolve_aeg_path" not in body, method.__name__
