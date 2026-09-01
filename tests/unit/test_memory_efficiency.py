"""Memory-efficiency contracts for the accelerator load path.

Each test here pins a *cause* of the load-time footprint rather than a total, because
a total is a property of the host it was measured on and a cause is a property of the
code. The causes, in the order the load meets them: a tied ``lm_head`` stored and
materialized twice; ``torch.cat`` holding both the parts and the packed result live;
the whole host weight set surviving until every device tensor exists; a calibration
probe that projects logits at every position; and the load's transient segments never
being handed back to the driver.

Speed and output quality are held fixed by the same tests rather than by a separate
timing suite: fusion must still happen (that is the speed property), the packed
tensor must equal the concatenation it replaced bit for bit, and greedy decoding must
produce the same token ids with the new path as without it.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights

VOCAB, HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, LAYERS = 17, 8, 2, 1, 12, 2


def _weights(*, tied: bool, shared_layer: bool = False) -> ModelWeights:
    """A small but structurally complete decoder, tied or untied."""
    rng = np.random.default_rng(7)

    def matrix(out: int, inn: int) -> np.ndarray:
        return rng.normal(0.0, 0.08, (out, inn)).astype(np.float32)

    def block() -> LayerWeights:
        return LayerWeights(
            attention_norm=np.ones(HIDDEN, dtype=np.float32),
            q_proj=matrix(HIDDEN, HIDDEN),
            k_proj=matrix(HIDDEN // 2, HIDDEN),
            v_proj=matrix(HIDDEN // 2, HIDDEN),
            o_proj=matrix(HIDDEN, HIDDEN),
            ffn_norm=np.ones(HIDDEN, dtype=np.float32),
            gate_proj=matrix(INTERMEDIATE, HIDDEN),
            up_proj=matrix(INTERMEDIATE, HIDDEN),
            down_proj=matrix(HIDDEN, INTERMEDIATE),
        )

    if shared_layer:
        one = block()
        blocks = [one] * LAYERS
    else:
        blocks = [block() for _ in range(LAYERS)]
    embedding = matrix(VOCAB, HIDDEN)
    return ModelWeights(
        embedding=embedding,
        layers=blocks,
        final_norm=np.ones(HIDDEN, dtype=np.float32),
        # A tied artifact is one where the loader has already proved the two
        # payloads identical and aliased them, which is exactly this identity.
        lm_head=embedding if tied else matrix(VOCAB, HIDDEN),
        rope_theta=10000.0,
        norm_eps=1e-5,
    )


def _cpu_engine(*, tied: bool = False, shared_layer: bool = False) -> CPUExecutionEngine:
    return CPUExecutionEngine(
        _weights(tied=tied, shared_layer=shared_layer),
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
    )


def _device_engine(monkeypatch: pytest.MonkeyPatch, *, dtype: str = "fp32", **kwargs):
    """Build the accelerator executor at a chosen compute dtype.

    ``fp16`` is the interesting case even on a CPU device: it makes the device
    tensors genuine copies rather than views of the host arrays, which is the
    condition every real accelerator load is in.
    """
    from aether.runtime.torch_engine import TorchAEGEngine

    monkeypatch.setenv("AETHER_TORCH_DTYPE", dtype)
    return TorchAEGEngine(_cpu_engine(**kwargs), "cpu")


PROMPT = np.asarray([1, 3, 5, 2], dtype=np.int64)


# ── A tied head is one matrix, not two ───────────────────────────────────────


def test_a_tied_head_shares_one_device_storage() -> None:
    """The saving is a shared pointer, so a pointer is what the test asserts."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    tied = TorchAEGEngine(_cpu_engine(tied=True), "cpu")
    assert tied.lm_head.untyped_storage().data_ptr() == (
        tied.embedding.untyped_storage().data_ptr()
    )

    untied = TorchAEGEngine(_cpu_engine(tied=False), "cpu")
    assert untied.lm_head.untyped_storage().data_ptr() != (
        untied.embedding.untyped_storage().data_ptr()
    )


def test_aliasing_the_head_does_not_change_the_logits_it_projects() -> None:
    """Sharing storage is a memory change; it must not be an arithmetic one."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    reference = _cpu_engine(tied=True)
    expected, _ = reference.forward(PROMPT)
    actual, _ = TorchAEGEngine(reference, "cpu").forward(PROMPT)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_the_census_counts_an_aliased_head_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deduplication by storage pointer is what makes the census attributable."""
    pytest.importorskip("torch")

    tied = _device_engine(monkeypatch, dtype="fp16", tied=True).memory_census()
    untied = _device_engine(monkeypatch, dtype="fp16", tied=False).memory_census()
    assert tied["device_storages"] == untied["device_storages"] - 1
    saved = untied["device_weight_bytes"] - tied["device_weight_bytes"]
    assert saved == VOCAB * HIDDEN * 2  # one FP16 vocabulary matrix


# ── Fused packing without a concatenation ────────────────────────────────────


def test_fusion_still_happens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fusion is the *speed* property, so removing ``cat`` must not remove it."""
    pytest.importorskip("torch")

    layer = _device_engine(monkeypatch).layers[0]
    assert layer["qkv_weight"] is not None
    assert layer["gate_up_weight"] is not None
    assert layer["q_proj"] is None and layer["gate_proj"] is None
    assert layer["q_width"] == HIDDEN
    assert layer["k_width"] == layer["v_width"] == HIDDEN // 2
    assert layer["gate_width"] == layer["up_width"] == INTERMEDIATE


def test_the_packed_tensor_equals_the_concatenation_it_replaced() -> None:
    """Bit-for-bit, because a copy into a slice is a copy either way."""
    torch = pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(_cpu_engine(), "cpu")
    parts = [
        np.random.default_rng(11).normal(0, 0.1, (rows, HIDDEN)).astype(np.float32)
        for rows in (5, 3, 3)
    ]
    packed = engine._packed_tensor(parts)
    expected = torch.cat([engine._tensor(part) for part in parts], dim=0)
    assert torch.equal(packed, expected)
    assert packed.untyped_storage().nbytes() == expected.untyped_storage().nbytes()


def test_the_load_path_never_calls_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    """The absence of the transient is the fix, so its absence is the assertion.

    ``cat`` allocates the result while every part is still live, which is both twice
    the transient bytes and the small-blocks-then-one-large-block order the caching
    allocator cannot satisfy from what the parts freed -- segments never merge. A
    guard that raises proves the packing no longer takes that route, which no
    measurement of a total would show.
    """
    torch = pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the load path must not concatenate device tensors")

    monkeypatch.setattr(torch, "cat", refuse)
    engine = TorchAEGEngine(_cpu_engine(), "cpu")
    assert engine.layers[0]["qkv_weight"] is not None


def test_fused_and_unfused_layers_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same numbers whichever way the projections are laid out."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    fused, _ = TorchAEGEngine(_cpu_engine(), "cpu").forward(PROMPT)
    monkeypatch.setattr(TorchAEGEngine, "_fusible", lambda *_a, **_k: None)
    unfused, _ = TorchAEGEngine(_cpu_engine(), "cpu").forward(PROMPT)
    np.testing.assert_allclose(fused, unfused, rtol=1e-5, atol=1e-5)


# ── The host copy shrinks while the device copy grows ─────────────────────────


def test_layers_are_freed_during_the_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak is what decides whether a load fits, so the release must be inside it."""
    pytest.importorskip("torch")

    engine = _device_engine(monkeypatch, dtype="fp16")
    assert engine.host_bytes_streamed > 0
    for layer in engine.weights.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"):
            assert getattr(layer, name) is None
    # Model-level matrices survive the upload: the alias check reads the embedding,
    # and the sweep afterwards is still the authority on the host set.
    assert isinstance(engine.weights.embedding, np.ndarray)
    assert not engine.host_weights_released


def test_the_reported_release_total_includes_what_streaming_already_freed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    engine = _device_engine(monkeypatch, dtype="fp16")
    streamed = engine.host_bytes_streamed
    freed = engine.release_host_weights()
    assert streamed > 0
    assert freed > streamed  # the layers, plus the model-level matrices
    assert engine.host_weights_released
    assert engine.release_host_weights() == 0


def test_keeping_host_weights_disables_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """One switch, one meaning: an operator who asks to keep them keeps all of them."""
    pytest.importorskip("torch")

    monkeypatch.setenv("AETHER_KEEP_HOST_WEIGHTS", "1")
    engine = _device_engine(monkeypatch, dtype="fp16")
    assert engine.host_bytes_streamed == 0
    assert isinstance(engine.weights.layers[0].q_proj, np.ndarray)


def test_aliased_weights_are_never_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    """On CPU at FP32 the device tensors *are* the host arrays; freeing would dangle."""
    pytest.importorskip("torch")

    engine = _device_engine(monkeypatch, dtype="fp32")
    assert engine.device_tensors_alias_host()
    assert engine.host_bytes_streamed == 0
    assert isinstance(engine.weights.layers[0].o_proj, np.ndarray)


def test_shared_layer_objects_are_never_freed_early(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeing a layer that another entry still points at would empty it mid-load."""
    pytest.importorskip("torch")

    engine = _device_engine(monkeypatch, dtype="fp16", shared_layer=True)
    assert engine.host_bytes_streamed == 0
    assert isinstance(engine.weights.layers[0].q_proj, np.ndarray)
    assert engine.layers[0]["qkv_weight"] is not None
    assert engine.layers[1]["qkv_weight"] is not None


def test_streaming_does_not_change_the_tokens_that_come_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The quality contract: identical greedy ids with the release and without it."""
    pytest.importorskip("torch")

    monkeypatch.setenv("AETHER_KEEP_HOST_WEIGHTS", "1")
    kept = _device_engine(monkeypatch, dtype="fp16")
    baseline = kept.generate(PROMPT, max_tokens=4, temperature=0.0)
    monkeypatch.delenv("AETHER_KEEP_HOST_WEIGHTS")
    streamed = _device_engine(monkeypatch, dtype="fp16")
    assert streamed.host_bytes_streamed > 0
    assert streamed.generate(PROMPT, max_tokens=4, temperature=0.0) == baseline


# ── Calibrating against the pass that actually runs ──────────────────────────


def test_the_probe_projects_one_position_where_forward_projects_all() -> None:
    """The probe's whole purpose: measure the shape a generation step really takes.

    ``forward`` is the inspection entry point -- logits at every position, widened to
    FP32 and copied to the host. At the planner's 2048-token default ceiling that is
    a tensor no generation step ever allocates, and calibrating the memory margin
    through it both inflates the load's high-water mark and teaches the ledger a
    residual for a pass that does not exist.
    """
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(_cpu_engine(), "cpu")
    inspected, _ = engine.forward(PROMPT)
    assert inspected.shape == (PROMPT.size, VOCAB)

    projected, _ = engine._forward_device(PROMPT, None, logits="last")
    assert tuple(projected.shape) == (1, VOCAB)
    assert engine.calibration_pass(1, PROMPT.size) is None


def test_the_bootstrap_prefers_the_probe_over_the_inspection_forward() -> None:
    pytest.importorskip("torch")
    from aether.backends.torch_backend import TorchBackend

    calls: list[tuple[int, int]] = []

    class Engine:
        def calibration_pass(self, batch: int, tokens: int) -> None:
            calls.append((batch, tokens))

        def forward(self, _ids: object) -> None:
            raise AssertionError("the bootstrap must not probe through forward()")

    class Planner:
        @staticmethod
        def needs_bootstrap(_decision: object) -> bool:
            return True

        @staticmethod
        def calibrate(_decision: object, one_pass: object) -> object:
            one_pass(1, 8)  # type: ignore[operator]
            return types.SimpleNamespace(calibrated=True)

    backend = TorchBackend()
    backend._placement = object()
    backend._placement_planner = Planner()
    assert backend.bootstrap_placement(Engine()).calibrated
    assert calls == [(1, 8)]


def test_the_bootstrap_still_works_for_an_engine_without_a_probe() -> None:
    """Older engine classes keep the old route rather than losing calibration."""
    pytest.importorskip("torch")
    from aether.backends.torch_backend import TorchBackend

    seen: list[int] = []

    class Engine:
        def forward(self, ids: np.ndarray) -> None:
            seen.append(int(np.asarray(ids).size))

    class Planner:
        @staticmethod
        def needs_bootstrap(_decision: object) -> bool:
            return True

        @staticmethod
        def calibrate(_decision: object, one_pass: object) -> object:
            one_pass(1, 6)  # type: ignore[operator]
            return types.SimpleNamespace(calibrated=True)

    backend = TorchBackend()
    backend._placement = object()
    backend._placement_planner = Planner()
    backend.bootstrap_placement(Engine())
    assert seen == [6]


# ── Giving the load's segments back to the driver ────────────────────────────


def test_the_reclaim_returns_cuda_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A segment stays reserved for the process's life unless it is *asked* for back.

    Which matters twice over: reserved bytes are what a capacity check sees, and
    ``reset_peak_memory_stats`` sets the peak to currently-reserved rather than to
    zero, so an unreleased load transient is folded into every peak reported
    afterwards.
    """
    torch = pytest.importorskip("torch")
    from aether.backends.torch_backend import _release_load_reservations

    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    _release_load_reservations("cuda:0")
    assert calls == ["sync", "empty"]


def test_the_reclaim_is_silent_on_cpu_and_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from aether.backends.torch_backend import _release_load_reservations

    _release_load_reservations("cpu")
    _release_load_reservations(None)

    def explode() -> None:
        raise RuntimeError("driver said no")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", explode)
    _release_load_reservations("cuda")  # must not raise: reclaiming is best-effort


def _fake_torch(*, available: bool, initialized: bool, hip: object = None) -> object:
    return types.SimpleNamespace(
        version=types.SimpleNamespace(hip=hip),
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            is_initialized=lambda: initialized,
        ),
    )


def test_expandable_segments_is_enabled_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One growable virtual range, so freed neighbours can coalesce at all.

    This codebase already prices the difference: ``aether.placement.ledger`` carries
    a fragmentation prior of 1.25 for the default allocator against 1.08 for
    expandable segments.
    """
    from aether.backends.torch_backend import _prefer_expandable_segments
    from aether.placement.ledger import _expandable_segments_enabled

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert _prefer_expandable_segments(_fake_torch(available=True, initialized=False))
    assert _expandable_segments_enabled()


@pytest.mark.parametrize(
    "torch_module",
    [
        _fake_torch(available=False, initialized=False),
        _fake_torch(available=True, initialized=True),
        _fake_torch(available=True, initialized=False, hip="6.0"),
    ],
    ids=["no-cuda", "allocator-already-running", "rocm"],
)
def test_expandable_segments_is_not_claimed_where_it_would_not_take_effect(
    monkeypatch: pytest.MonkeyPatch, torch_module: object
) -> None:
    from aether.backends.torch_backend import _prefer_expandable_segments

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert not _prefer_expandable_segments(torch_module)
    import os

    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_an_explicit_allocator_configuration_always_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from aether.backends.torch_backend import _prefer_expandable_segments

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert not _prefer_expandable_segments(_fake_torch(available=True, initialized=False))
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


# ── Proving a tie from the stored bytes, cheaply ──────────────────────────────


def _store(directory: object, payloads: dict[str, np.ndarray]):
    from aether.core.weight_store import WeightStore
    from aether.quantization.formats import QuantizedTensor

    written = WeightStore(directory)
    written.save(
        {
            name: QuantizedTensor(precision="bf16", shape=data.shape, data=data)
            for name, data in payloads.items()
        }
    )
    reopened = WeightStore(directory)
    reopened.load_index()
    return reopened


def test_a_tensor_survives_the_single_copy_read(tmp_path) -> None:
    """``readinto`` a pre-allocated array: one host copy where there were two."""
    payload = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    store = _store(tmp_path, {"embedding": payload})
    loaded = store.load_tensor("embedding")
    np.testing.assert_array_equal(np.asarray(loaded.data).reshape(64, 64), payload)


def test_identical_payloads_are_proved_identical(tmp_path) -> None:
    payload = np.arange(4096, dtype=np.uint16)
    store = _store(tmp_path, {"embedding": payload, "lm_head": payload.copy()})
    assert store.payloads_identical("embedding", "lm_head")


def test_a_single_differing_byte_refuses_the_alias(tmp_path) -> None:
    """A manifest flag is a claim; the bytes are the evidence.

    Aliasing on the strength of the flag alone would silently replace an untied head
    with the embedding matrix -- a memory saving that changes every token the model
    emits. So the flag is necessary and the comparison is decisive.
    """
    payload = np.arange(4096, dtype=np.uint16)
    other = payload.copy()
    other[-1] += 1
    store = _store(tmp_path, {"embedding": payload, "lm_head": other})
    assert not store.payloads_identical("embedding", "lm_head")


def test_the_comparison_spans_more_chunks_than_one(tmp_path, monkeypatch) -> None:
    """Bounded buffers mean the loop, not the buffer, has to cover the tensor."""
    from aether.core.weight_store import WeightStore

    payload = np.arange(8192, dtype=np.uint16)
    other = payload.copy()
    other[5000] += 1
    store = _store(tmp_path, {"a": payload, "b": payload.copy(), "c": other})
    monkeypatch.setattr(WeightStore, "_COMPARE_CHUNK_BYTES", 512)
    assert store.payloads_identical("a", "b")
    assert not store.payloads_identical("a", "c")


def test_the_comparison_leaves_no_mapping_resident(tmp_path) -> None:
    """Touching a mapped page charges it to the resident set, which is the metric."""
    payload = np.arange(4096, dtype=np.uint16)
    store = _store(tmp_path, {"embedding": payload, "lm_head": payload.copy()})
    assert store.payloads_identical("embedding", "lm_head")
    assert store._mmap is None and store._mmap_file is None


def test_mismatched_shapes_are_rejected_before_any_bytes_are_read(tmp_path) -> None:
    store = _store(
        tmp_path,
        {
            "embedding": np.zeros(64, dtype=np.uint16),
            "lm_head": np.zeros((8, 8), dtype=np.uint16),
        },
    )
    assert not store.payloads_identical("embedding", "lm_head")
    assert not store.payloads_identical("embedding", "absent")


def test_a_zero_copy_view_still_shares_the_mapped_bytes(tmp_path) -> None:
    """``raw_view`` remains available for callers that want the pages themselves."""
    payload = np.arange(256, dtype=np.uint16)
    store = _store(tmp_path, {"embedding": payload})
    try:
        view = store.raw_view("embedding")
        np.testing.assert_array_equal(view, payload)
        assert not view.flags.writeable
    finally:
        store.close()


def _package(*, tied: bool, payloads: dict[str, np.ndarray], directory: object) -> object:
    store = _store(directory, payloads)
    return types.SimpleNamespace(
        manifest=types.SimpleNamespace(architecture={"tie_word_embeddings": tied}),
        weight_store=lambda: store,
    )


def test_the_loader_aliases_a_tie_it_has_verified(tmp_path) -> None:
    from aether.runtime.aeg_loader import _tied_head_is_duplicate

    payload = np.arange(4096, dtype=np.uint16)
    package = _package(
        tied=True,
        payloads={"embedding": payload, "lm_head": payload.copy()},
        directory=tmp_path,
    )
    assert _tied_head_is_duplicate(package)


def test_the_loader_refuses_to_alias_a_tie_the_bytes_contradict(tmp_path) -> None:
    """Manifest says tied, blob says otherwise: keep both matrices and warn."""
    from aether.runtime.aeg_loader import _tied_head_is_duplicate

    payload = np.arange(4096, dtype=np.uint16)
    other = payload.copy()
    other[0] += 7
    package = _package(
        tied=True,
        payloads={"embedding": payload, "lm_head": other},
        directory=tmp_path,
    )
    assert not _tied_head_is_duplicate(package)


def test_the_loader_leaves_an_untied_head_alone(tmp_path) -> None:
    from aether.runtime.aeg_loader import _tied_head_is_duplicate

    payload = np.arange(4096, dtype=np.uint16)
    package = _package(
        tied=False,
        payloads={"embedding": payload, "lm_head": payload.copy()},
        directory=tmp_path,
    )
    assert not _tied_head_is_duplicate(package)


# ── Speed is a memory property too ───────────────────────────────────────────


def test_decoding_never_projects_more_than_the_position_it_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The already-lean part of the path, pinned so a memory change cannot undo it.

    Projecting every position costs ``tokens × vocab`` device bytes and the matmul
    that fills them, so ``logits="last"`` is simultaneously the memory choice and the
    speed choice. A regression here would look like a memory regression *and* a
    throughput one, which is why it is asserted rather than assumed.
    """
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    modes: list[str] = []
    original = TorchAEGEngine._forward_device

    def record(self, token_ids, cache=None, **kwargs):  # type: ignore[no-untyped-def]
        modes.append(str(kwargs.get("logits", "all")))
        return original(self, token_ids, cache, **kwargs)

    monkeypatch.setattr(TorchAEGEngine, "_forward_device", record)
    engine = TorchAEGEngine(_cpu_engine(), "cpu")
    produced = list(engine.generate_iter(PROMPT, max_tokens=3, temperature=0.0))
    assert len(produced) == 3
    assert modes and set(modes) == {"last"}


def test_the_kv_cache_is_reserved_once_for_the_whole_request() -> None:
    """Growth by reallocation would double the cache's peak; the reserve avoids it."""
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(_cpu_engine(), "cpu")
    _, cache = engine._forward_device(PROMPT, None, reserve=PROMPT.size + 8, logits="last")
    assert cache.keys[0].shape[0] >= PROMPT.size + 8
    pointer = cache.keys[0].untyped_storage().data_ptr()
    for step in range(8):
        token = np.asarray([step % VOCAB], dtype=np.int64)
        _, cache = engine._forward_device(token, cache, logits="last")
    # No reallocation: the same storage absorbed every appended step.
    assert cache.keys[0].untyped_storage().data_ptr() == pointer


def _self_calls(cls: type) -> set[str]:
    """Every ``self.<name>(...)`` the class's own source calls."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    called: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            called.add(node.func.attr)
    return called


def _self_assignments(cls: type) -> set[str]:
    """Every ``self.<name> = ...`` the class's own source performs."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    assigned: set[str] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    return assigned


@pytest.mark.parametrize("class_name", ["TorchAEGEngine", "TorchHybridAEGEngine"])
def test_every_helper_an_engine_calls_on_itself_exists(class_name: str) -> None:
    """A standalone engine must carry the helpers it calls, not borrow them by luck.

    ``TorchHybridAEGEngine`` re-implements the same contract as ``TorchAEGEngine``
    without inheriting from it, so a helper added to one is *not* available on the
    other. Adding the tied-embedding rule taught this the hard way: the call was
    added to both constructors and the method to only one, and the hybrid path
    raised ``AttributeError`` at load time. Nothing about the shape of that mistake
    was visible until a Jamba checkpoint was actually loaded, so it is checked here
    on the source instead - no device, no checkpoint, no import of torch required.
    """
    pytest.importorskip("torch")
    from aether.runtime import torch_engine

    cls = getattr(torch_engine, class_name)
    missing = sorted(
        name
        for name in _self_calls(cls) - _self_assignments(cls)
        if not hasattr(cls, name)
    )
    assert missing == [], f"{class_name} calls self.{{{','.join(missing)}}} but defines neither"


def test_the_hybrid_path_shares_a_tied_matrix_instead_of_uploading_it_twice() -> None:
    """The hybrid path must get the saving, not merely stop raising.

    The regression the test above guards was found by a load failing; this pins the
    behaviour that load was supposed to have. Only the rule is exercised here - the
    real construction, with a real Jamba artifact, is covered by
    ``tests/integration/test_jamba_hybrid_execution.py`` - so this stays a unit test
    and still fails if the hybrid engine ever uploads a tied matrix twice.
    """
    torch = pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchHybridAEGEngine

    engine = TorchHybridAEGEngine.__new__(TorchHybridAEGEngine)
    engine.torch = torch
    engine.device = torch.device("cpu")

    embedding_host = np.arange(VOCAB * HIDDEN, dtype=np.float32).reshape(VOCAB, HIDDEN)
    resident = engine._tensor(embedding_host)

    tied = engine._tied_tensor(embedding_host, embedding_host, resident)
    assert tied is resident, "a tied lm_head must reuse the embedding already uploaded"

    distinct_host = embedding_host + 1.0
    separate = engine._tied_tensor(distinct_host, embedding_host, resident)
    assert separate is not resident, "an untied lm_head must keep its own storage"
    assert torch.allclose(separate, torch.from_numpy(distinct_host), atol=1e-6)
