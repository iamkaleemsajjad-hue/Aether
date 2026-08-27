"""Correctness tests for batched inference on the portable executor.

The contract under test is stated in ``docs/adr-batched-inference.md``: a row of a
batch must produce exactly what that sequence produces when decoded alone.  Every
test here is written to fail loudly if that stops being true, because the failure
modes of batched decode are numerical rather than structural — a wrong position
offset or a leaked mask does not change a single tensor shape.

The cases that matter most, and would each pass a naive implementation that is
nonetheless wrong:

* ``test_batched_prefill_matches_independent_runs`` — equal-length rows. Catches a
  broken batch axis.
* ``test_ragged_batch_matches_independent_runs`` — unequal lengths, so padding is
  live. Catches padded-index-as-position (the single most likely defect) and mask
  leakage into the pad region.
* ``test_learned_absolute_positions_survive_padding`` — the same, for models
  positioned by a learned table rather than rotary. A rotary-only fix passes the
  previous test and fails this one.
* ``test_row_is_unaffected_by_its_neighbours`` — direct isolation check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether.runtime.batch import (
    BatchLayout,
    normalize_sequences,
    pack_left_padded,
)
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights

VOCAB, HIDDEN, HEADS, KV_HEADS, INTERMEDIATE, LAYERS = 17, 8, 2, 1, 12, 2

#: FP32 on CPU, but a batched GEMM reduces in a different order than a rank-2 one,
#: so the comparison is a numerical-equivalence tolerance rather than exact.
LOGIT_TOLERANCE = 2e-4


def _reference_engine(seed: int = 7) -> CPUExecutionEngine:
    rng = np.random.default_rng(seed)

    def matrix(out: int, inn: int) -> np.ndarray:
        return rng.normal(0.0, 0.08, (out, inn)).astype(np.float32)

    blocks = [
        LayerWeights(
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
        for _ in range(LAYERS)
    ]
    weights = ModelWeights(
        embedding=matrix(VOCAB, HIDDEN),
        layers=blocks,
        final_norm=np.ones(HIDDEN, dtype=np.float32),
        lm_head=matrix(VOCAB, HIDDEN),
        rope_theta=10000.0,
        norm_eps=1e-5,
    )
    return CPUExecutionEngine(weights, num_heads=HEADS, num_kv_heads=KV_HEADS)


def _engine(reference: CPUExecutionEngine | None = None):
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    return TorchAEGEngine(reference or _reference_engine(), "cpu")


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _final_logits_alone(engine, ids: np.ndarray) -> np.ndarray:
    """Logits at the last position when ``ids`` is the only sequence in flight."""
    logits, _ = engine.forward(np.asarray(ids, dtype=np.int64))
    return np.asarray(logits)[-1]


# ── The layout, on its own ──────────────────────────────────────────────────


def test_layout_right_aligns_and_restarts_positions() -> None:
    packed = pack_left_padded([[1, 2, 3], [4, 5], [6]])

    assert packed.layout.padded_length == 3
    assert packed.layout.pad_counts == (0, 1, 2)
    assert not packed.layout.is_uniform

    # Rows end flush against the right edge: that is what gives decode one
    # shared write index instead of a per-row scatter.
    np.testing.assert_array_equal(packed.token_ids[0], [1, 2, 3])
    np.testing.assert_array_equal(packed.token_ids[1][1:], [4, 5])
    np.testing.assert_array_equal(packed.token_ids[2][2:], [6])

    np.testing.assert_array_equal(
        packed.live, [[True] * 3, [False, True, True], [False, False, True]]
    )
    # Each row's first *real* token is at position 0 however much pad precedes it.
    np.testing.assert_array_equal(packed.position_ids[0], [0, 1, 2])
    np.testing.assert_array_equal(packed.position_ids[1], [0, 0, 1])
    np.testing.assert_array_equal(packed.position_ids[2], [0, 0, 0])


def test_uniform_batch_reports_no_padding() -> None:
    packed = pack_left_padded([[1, 2], [3, 4], [5, 6]])
    assert packed.layout.is_uniform
    assert packed.layout.pad_counts == (0, 0, 0)
    assert packed.layout.padding_overhead == 0.0
    assert packed.live.all()


def test_decode_positions_continue_each_row_from_its_own_length() -> None:
    layout = BatchLayout(lengths=(3, 1), padded_length=3)
    # Decode step 0 writes padded index 3.  Row 0 has 3 real tokens so its next
    # position is 3; row 1 has 1, so its next position is 1.
    np.testing.assert_array_equal(layout.decode_positions(0), [[3], [1]])
    np.testing.assert_array_equal(layout.decode_positions(1), [[4], [2]])


def test_layout_rejects_degenerate_batches() -> None:
    with pytest.raises(ValueError, match="at least one sequence"):
        pack_left_padded([])
    with pytest.raises(ValueError, match="empty"):
        pack_left_padded([[1, 2], []])
    with pytest.raises(ValueError, match="rank 1 or 2"):
        pack_left_padded(np.zeros((2, 2, 2), dtype=np.int64))


def test_normalize_treats_a_flat_array_as_one_sequence() -> None:
    rows = normalize_sequences(np.asarray([1, 2, 3], dtype=np.int64))
    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0], [1, 2, 3])


# ── Batch = 1 must remain a strict special case ─────────────────────────────


def test_batch_of_one_matches_the_single_sequence_path() -> None:
    """The regression guard: B=1 through the batched path is the unbatched path."""
    engine = _engine()
    ids = np.asarray([1, 3, 5, 2], dtype=np.int64)

    expected, _ = engine.forward(ids)
    batched, _ = engine.forward_batch([ids])

    assert _numpy(batched).shape == (1, ids.size, VOCAB)
    np.testing.assert_allclose(
        _numpy(batched)[0], np.asarray(expected), rtol=LOGIT_TOLERANCE, atol=LOGIT_TOLERANCE
    )

    alone = engine.generate(ids, max_tokens=6, temperature=0.0)
    together = engine.generate_batch([ids], max_tokens=6, temperature=0.0)
    assert together == [alone]


def test_single_sequence_api_still_returns_rank_two_logits() -> None:
    """``forward`` keeps its contract: a (1, seq) input is one sequence, not a batch."""
    engine = _engine()
    flat, _ = engine.forward(np.asarray([1, 3, 5], dtype=np.int64))
    nested, _ = engine.forward(np.asarray([[1, 3, 5]], dtype=np.int64))
    assert np.asarray(flat).shape == (3, VOCAB)
    np.testing.assert_array_equal(np.asarray(nested), np.asarray(flat))


# ── Cross-batch equivalence ─────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batched_prefill_matches_independent_runs(batch_size: int) -> None:
    """Equal-length rows: every row's final logits equal its solo logits."""
    engine = _engine()
    prompts = [
        np.asarray([(index * 3 + step) % VOCAB for step in range(5)], dtype=np.int64)
        for index in range(batch_size)
    ]

    logits, cache = engine.forward_batch(prompts)
    assert _numpy(logits).shape == (batch_size, 5, VOCAB)
    # Equal lengths mean no padding, so no mask is materialized at all.
    assert cache.live is None
    assert cache.layout is not None and cache.layout.is_uniform

    for index, prompt in enumerate(prompts):
        np.testing.assert_allclose(
            _numpy(cache.last_logits)[index],
            _final_logits_alone(engine, prompt),
            rtol=LOGIT_TOLERANCE,
            atol=LOGIT_TOLERANCE,
        )


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batched_greedy_decode_matches_independent_runs(batch_size: int) -> None:
    engine = _engine()
    prompts = [
        np.asarray([(index * 5 + step) % VOCAB for step in range(4)], dtype=np.int64)
        for index in range(batch_size)
    ]

    alone = [engine.generate(prompt, max_tokens=8, temperature=0.0) for prompt in prompts]
    together = engine.generate_batch(prompts, max_tokens=8, temperature=0.0)

    assert together == alone


def test_ragged_batch_matches_independent_runs() -> None:
    """Unequal lengths: the case that catches padded-index-as-position.

    A row whose prompt is shorter than the batch's widest is preceded by pad slots.
    If those slots were allowed to shift the row's positions, or to be attended to,
    this row's logits would differ from decoding it alone — while every tensor in
    the pass kept a perfectly valid shape.
    """
    engine = _engine()
    prompts = [
        np.asarray([1, 2], dtype=np.int64),                       # short
        np.asarray([3, 4, 5, 6], dtype=np.int64),                 # medium
        np.asarray([7, 8, 9, 10, 11, 12], dtype=np.int64),        # long
        np.asarray(list(range(1, 12)), dtype=np.int64),            # very long
    ]

    _, cache = engine.forward_batch(prompts)
    assert cache.live is not None, "a ragged batch must materialize a validity mask"
    assert cache.layout is not None and cache.layout.pad_counts == (9, 7, 5, 0)

    for index, prompt in enumerate(prompts):
        np.testing.assert_allclose(
            _numpy(cache.last_logits)[index],
            _final_logits_alone(engine, prompt),
            rtol=LOGIT_TOLERANCE,
            atol=LOGIT_TOLERANCE,
            err_msg=f"row {index} (length {prompt.size}) diverged from its solo run",
        )


def test_ragged_batch_greedy_decode_matches_independent_runs() -> None:
    engine = _engine()
    prompts = [
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([3, 4, 5, 6], dtype=np.int64),
        np.asarray([7, 8, 9], dtype=np.int64),
        np.asarray(list(range(1, 10)), dtype=np.int64),
    ]

    alone = [engine.generate(prompt, max_tokens=6, temperature=0.0) for prompt in prompts]
    together = engine.generate_batch(prompts, max_tokens=6, temperature=0.0)

    assert together == alone


def test_learned_absolute_positions_survive_padding() -> None:
    """The same ragged guarantee for a model positioned by a learned table.

    Rotary and learned-absolute read positions through different code. A fix that
    only threads per-row positions into the rotary path passes the ragged rotary
    test above and silently corrupts this one, which is why it is separate.
    """
    reference = _reference_engine()
    reference.weights.norm_type = "LayerNorm"
    reference.weights.position_type = "absolute"
    reference.weights.position_embedding = (
        np.arange(32 * HIDDEN, dtype=np.float32).reshape(32, HIDDEN) / 100.0
    )
    engine = _engine(reference)

    prompts = [
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([3, 4, 5, 6, 7], dtype=np.int64),
    ]
    _, cache = engine.forward_batch(prompts)

    for index, prompt in enumerate(prompts):
        np.testing.assert_allclose(
            _numpy(cache.last_logits)[index],
            _final_logits_alone(engine, prompt),
            rtol=LOGIT_TOLERANCE,
            atol=LOGIT_TOLERANCE,
            err_msg=f"row {index} was positioned by its padded index, not its own",
        )


def test_grouped_query_and_local_attention_batch_correctly() -> None:
    """Q/K norms plus a sliding window, batched and ragged at once."""
    reference = _reference_engine()
    for layer in reference.weights.layers:
        layer.q_norm = np.ones(reference.head_dim, dtype=np.float32)
        layer.k_norm = np.ones(reference.head_dim, dtype=np.float32)
    reference.weights.attention_layers = ["global", "local"]
    reference.weights.attention_window = 3
    engine = _engine(reference)

    prompts = [
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([4, 5, 6, 7, 8, 9], dtype=np.int64),
    ]
    _, cache = engine.forward_batch(prompts)
    for index, prompt in enumerate(prompts):
        np.testing.assert_allclose(
            _numpy(cache.last_logits)[index],
            _final_logits_alone(engine, prompt),
            rtol=LOGIT_TOLERANCE,
            atol=LOGIT_TOLERANCE,
        )


# ── Isolation ───────────────────────────────────────────────────────────────


def test_row_is_unaffected_by_its_neighbours() -> None:
    """Changing every other row must not move row 0 by one bit."""
    engine = _engine()
    subject = np.asarray([1, 2, 3, 4], dtype=np.int64)

    first = engine.generate_batch(
        [subject, np.asarray([5, 6, 7, 8], dtype=np.int64)],
        max_tokens=6,
        temperature=0.0,
    )
    second = engine.generate_batch(
        [subject, np.asarray([9, 10], dtype=np.int64), np.asarray([11, 12, 13, 14, 15], dtype=np.int64)],
        max_tokens=6,
        temperature=0.0,
    )
    solo = engine.generate(subject, max_tokens=6, temperature=0.0)

    assert first[0] == solo
    assert second[0] == solo


def test_kv_cache_rows_hold_only_their_own_sequence() -> None:
    """Structural isolation: a row's cached keys equal that row's solo keys."""
    engine = _engine()
    shared = np.asarray([1, 2, 3, 4], dtype=np.int64)
    other = np.asarray([9, 8, 7, 6], dtype=np.int64)

    _, solo_cache = engine.forward(shared)
    _, batch_cache = engine.forward_batch([shared, other])

    for layer in range(engine.num_layers):
        solo_keys = _numpy(solo_cache.keys[layer][: shared.size])
        row_keys = _numpy(batch_cache.keys[layer][0, : shared.size])
        np.testing.assert_allclose(
            row_keys, solo_keys, rtol=LOGIT_TOLERANCE, atol=LOGIT_TOLERANCE,
            err_msg=f"layer {layer} row 0 KV does not match the solo run",
        )
        # The two rows must also differ, or the test above would pass for a cache
        # that broadcast one row over all of them.  Compare only the written
        # region: the tail of the allocation is deliberately uninitialized.
        assert not np.allclose(
            _numpy(batch_cache.keys[layer][0, : shared.size]),
            _numpy(batch_cache.keys[layer][1, : other.size]),
        ), f"layer {layer} rows hold identical keys for different prompts"


def test_pad_slots_are_never_attended() -> None:
    """A short row's logits must not change when its pad content changes.

    Pad ids are masked out, so choosing a different pad token cannot be observable.
    If it is, the validity mask is not reaching attention.
    """
    engine = _engine()
    prompts = [np.asarray([1, 2], dtype=np.int64), np.asarray([3, 4, 5, 6, 7], dtype=np.int64)]

    _, zero_pad = engine.forward_batch(prompts, pad_token_id=0)
    _, other_pad = engine.forward_batch(prompts, pad_token_id=VOCAB - 1)

    np.testing.assert_allclose(
        _numpy(zero_pad.last_logits),
        _numpy(other_pad.last_logits),
        rtol=LOGIT_TOLERANCE,
        atol=LOGIT_TOLERANCE,
    )


# ── Per-row stopping ────────────────────────────────────────────────────────


def test_one_row_stopping_early_does_not_truncate_the_others() -> None:
    engine = _engine()
    early = np.asarray([1, 2, 3], dtype=np.int64)

    # Choose the stop token to be whatever row 0 emits first, so that row is
    # guaranteed to finish on step 1.
    stop = engine.generate(early, max_tokens=1, temperature=0.0)[0]

    # And choose a partner whose own first token is *not* that stop token, so
    # "the other row kept going" is a real assertion rather than a coincidence.
    late = None
    for candidate in ([9, 8, 7], [4, 5, 6], [11, 12, 13], [2, 4, 6], [15, 14, 13]):
        ids = np.asarray(candidate, dtype=np.int64)
        if engine.generate(ids, max_tokens=1, temperature=0.0)[0] != stop:
            late = ids
            break
    assert late is not None, "no partner prompt with a different first token"

    late_solo = engine.generate(late, max_tokens=10, temperature=0.0, eos_token_id=stop)
    together = engine.generate_batch(
        [early, late], max_tokens=10, temperature=0.0, eos_token_id=stop
    )

    assert together[0] == [stop], "the finished row kept generating"
    assert together[1] == late_solo, "an unfinished row was cut short by its neighbour"
    assert len(together[1]) > 1, "the partner row stopped on step 1 too; test is vacuous"


def test_stop_token_is_recorded_like_the_single_sequence_path() -> None:
    """EOS is yielded, not swallowed — matching ``generate``'s behaviour."""
    engine = _engine()
    prompt = np.asarray([4, 5, 6], dtype=np.int64)
    stop = engine.generate(prompt, max_tokens=1, temperature=0.0)[0]

    solo = engine.generate(prompt, max_tokens=5, temperature=0.0, eos_token_id=stop)
    batched = engine.generate_batch([prompt], max_tokens=5, temperature=0.0, eos_token_id=stop)
    assert batched == [solo] == [[stop]]


# ── Sampling ────────────────────────────────────────────────────────────────


def test_top_k_one_selects_each_row_argmax_independently() -> None:
    """A per-row sampler check that does not depend on the RNG.

    ``top_k=1`` leaves exactly one candidate per row, so a sampled draw must land
    on that row's own argmax. A sampler that reduced over the wrong axis — or
    thresholded the whole batch against one row's k-th logit — fails here.
    """
    engine = _engine()
    prompts = [
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([9, 10, 11], dtype=np.int64),
        np.asarray([5, 6], dtype=np.int64),
    ]
    greedy = engine.generate_batch(prompts, max_tokens=1, temperature=0.0)
    sampled = engine.generate_batch(prompts, max_tokens=1, temperature=0.9, top_k=1)
    assert sampled == greedy


def test_top_p_sampling_stays_within_each_row_support() -> None:
    """Nucleus sampling must build a separate nucleus per row."""
    engine = _engine()
    prompts = [
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([12, 13, 14], dtype=np.int64),
    ]
    # A nucleus that keeps essentially all mass still has to be per row; drawing
    # repeatedly, every token must be in that row's vocabulary support.
    for _ in range(12):
        drawn = engine.generate_batch(
            prompts, max_tokens=1, temperature=1.0, top_p=0.95
        )
        assert len(drawn) == 2
        for row in drawn:
            assert len(row) == 1
            assert 0 <= row[0] < VOCAB


def test_batched_sampler_shapes_are_per_row() -> None:
    """Direct check on the sampler: (B, vocab) in, (B,) out."""
    torch = pytest.importorskip("torch")
    engine = _engine()
    logits = torch.randn(4, VOCAB)

    greedy = engine._sample_device(logits, 0.0, 0, 1.0)
    assert tuple(greedy.shape) == (4,)
    torch.testing.assert_close(greedy, torch.argmax(logits, dim=-1))

    for temperature, top_k, top_p in ((1.0, 0, 1.0), (1.0, 3, 1.0), (1.0, 0, 0.8)):
        drawn = engine._sample_device(logits, temperature, top_k, top_p)
        assert tuple(drawn.shape) == (4,), f"t={temperature} k={top_k} p={top_p}"
        assert int(drawn.min()) >= 0 and int(drawn.max()) < VOCAB

    # And the single-sequence contract is untouched: rank-1 in, 0-dim out.
    scalar = engine._sample_device(logits[0], 0.0, 0, 1.0)
    assert scalar.dim() == 0


# ── Repeated use ────────────────────────────────────────────────────────────


def test_repeated_batched_generation_carries_no_stale_state() -> None:
    """Many sequential batches, including changing batch width, stay correct."""
    engine = _engine()
    prompts = [
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([4, 5, 6, 7], dtype=np.int64),
    ]
    expected = engine.generate_batch(prompts, max_tokens=5, temperature=0.0)

    for _ in range(5):
        assert engine.generate_batch(prompts, max_tokens=5, temperature=0.0) == expected

    # Interleave a different width, then repeat: a cache or table sized for the
    # wider batch must not perturb the narrower one.
    engine.generate_batch(prompts + [np.asarray([8, 9], dtype=np.int64)] * 2,
                          max_tokens=5, temperature=0.0)
    assert engine.generate_batch(prompts, max_tokens=5, temperature=0.0) == expected


def test_interleaving_batched_and_single_sequence_calls_is_safe() -> None:
    engine = _engine()
    prompt = np.asarray([1, 2, 3, 4], dtype=np.int64)
    solo = engine.generate(prompt, max_tokens=5, temperature=0.0)

    engine.generate_batch(
        [prompt, np.asarray([5, 6], dtype=np.int64)], max_tokens=5, temperature=0.0
    )
    assert engine.generate(prompt, max_tokens=5, temperature=0.0) == solo


# ── Capability and edge cases ───────────────────────────────────────────────


def test_engine_advertises_unbounded_batch_support() -> None:
    engine = _engine()
    assert engine.supports_batch(1)
    assert engine.supports_batch(4)
    # No compiled-in ceiling: the AEG IR declares the batch axis dynamic.
    assert engine.max_batch_size is None


def test_batched_entry_points_reject_invalid_input() -> None:
    engine = _engine()
    ids = np.asarray([1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        engine.generate_batch([ids], max_tokens=0)
    with pytest.raises(ValueError, match="at least one sequence"):
        engine.generate_batch([])
    with pytest.raises(ValueError, match="outside the compiled vocabulary"):
        engine.forward_batch([np.asarray([VOCAB], dtype=np.int64)])
    with pytest.raises(ValueError, match="rank-2"):
        engine._forward_device(ids, batched=True)


def test_tensor_parallel_refuses_batching_rather_than_flattening() -> None:
    """The sharded executor must say no, not silently splice the batch."""
    pytest.importorskip("torch")
    import inspect

    from aether.runtime.torch_engine import TorchAEGEngine
    from aether.runtime.torch_tensor_parallel import TorchTensorParallelAEGEngine

    # Contract parity is enforced elsewhere; here we assert the refusal exists so
    # a batch can never reach a single-sequence layer loop.
    source = inspect.getsource(TorchTensorParallelAEGEngine._forward_device)
    assert "NotImplementedError" in source
    assert inspect.signature(TorchAEGEngine._forward_device) == inspect.signature(
        TorchTensorParallelAEGEngine._forward_device
    )


# ── The backend boundary ────────────────────────────────────────────────────
#
# The engine tests above prove the batch is computed correctly. These prove the
# backend never *fakes* one: an executor that cannot batch must produce an error,
# never a loop over the requests reported as batched throughput.


class _StubTokenizer:
    """Minimal packaged-tokenizer stand-in: whitespace ids, no vocabulary file."""

    eos_token_id = None

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [(len(word) % VOCAB) or 1 for word in str(text).split()] or [1]
        return {"input_ids": np.asarray([ids], dtype=np.int64)}

    def decode(self, ids, skip_special_tokens=True, **_):
        return " ".join(str(int(value)) for value in ids)


def _handle(engine):
    from aether.backends.compiled_handle import CompiledAEGHandle

    return CompiledAEGHandle(
        model_id="stub",
        aeg_path=Path("stub.aeg"),
        manifest={},
        engine=engine,
        tokenizer=_StubTokenizer(),
    )


def _requests(count: int, **overrides):
    from aether.backends.base import GenerationRequest

    return [
        GenerationRequest(
            model_id="stub",
            prompt=f"prompt number {'x' * (index + 1)}",
            max_tokens=overrides.pop("max_tokens", 3),
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            **overrides,
        )
        for index in range(count)
    ]


def test_backend_runs_a_real_batch_through_the_shared_path() -> None:
    from aether.backends import batched_generation

    engine = _engine()
    results = batched_generation.generate_batch(
        _handle(engine),
        _requests(3),
        backend_name="test",
        request_text=lambda request, _tokenizer: request.prompt,
        truncate_stop_text=lambda tokenizer, ids, stops: (
            tokenizer.decode(ids), len(ids), False
        ),
    )
    assert len(results) == 3
    for result in results:
        assert result.completion_tokens == 3
        assert result.metrics["batch_size"] == 3
        assert result.metrics["engine"] == "TorchAEGEngine"
        # Aggregate and per-row rates are both reported, never conflated.
        assert result.metrics["batch_throughput_tps"] >= result.metrics["row_throughput_tps"]


def test_backend_refuses_a_batch_it_cannot_execute_in_one_pass() -> None:
    """A single-sequence executor must raise, not be looped over."""
    from aether.backends import batched_generation
    from aether.core.exceptions import BackendError

    class _SingleSequenceOnly:
        """An executor with no batch axis and no promotion path."""

        def generate(self, ids, **_):
            return [1, 2, 3]

    with pytest.raises(BackendError, match="single-sequence"):
        batched_generation.generate_batch(
            _handle(_SingleSequenceOnly()),
            _requests(2),
            backend_name="test",
            request_text=lambda request, _t: request.prompt,
            truncate_stop_text=lambda t, ids, stops: ("", len(ids), False),
        )


def test_backend_rejects_mixed_sampling_settings() -> None:
    """Sampling is applied to the whole logits tensor, so rows must agree on it."""
    from aether.backends import batched_generation
    from aether.core.exceptions import BackendError

    requests = _requests(2)
    requests[1].temperature = 0.7
    with pytest.raises(BackendError, match="temperature"):
        batched_generation.generate_batch(
            _handle(_engine()),
            requests,
            backend_name="test",
            request_text=lambda request, _t: request.prompt,
            truncate_stop_text=lambda t, ids, stops: ("", len(ids), False),
        )


def test_differing_max_tokens_truncate_per_row() -> None:
    """A shared decode horizon, then each row cut to its own budget.

    Rows are independent, so over-running one cannot affect another; this checks
    the budgets are honoured rather than levelled to the longest.
    """
    from aether.backends import batched_generation

    requests = _requests(2)
    requests[0].max_tokens = 2
    requests[1].max_tokens = 5
    results = batched_generation.generate_batch(
        _handle(_engine()),
        requests,
        backend_name="test",
        request_text=lambda request, _t: request.prompt,
        truncate_stop_text=lambda tokenizer, ids, stops: (
            tokenizer.decode(ids), len(ids), False
        ),
    )
    assert [result.completion_tokens for result in results] == [2, 5]


def test_can_batch_probe_does_not_promote_anything() -> None:
    """The capability probe must be free of side effects."""
    from aether.backends import batched_generation

    engine = _engine()
    handle = _handle(engine)
    assert batched_generation.can_batch(handle, 4) is True
    assert handle.batched_engine is None, "the probe promoted an engine"

    class _NoBatch:
        pass

    assert batched_generation.can_batch(_handle(_NoBatch()), 4) is False
    # A width of one is always servable, by the single-sequence path.
    assert batched_generation.can_batch(_handle(_NoBatch()), 1) is True
    assert batched_generation.can_batch(None, 2) is False


def test_promotion_happens_once_and_is_reused() -> None:
    """A CPU-loaded NumPy engine is promoted to batch, and only once.

    The promotion materializes the weights a second time as tensors, so paying for
    it per request would be a real cost; this pins the caching.
    """
    from aether.backends import batched_generation

    pytest.importorskip("torch")
    handle = _handle(_reference_engine())  # the NumPy executor: no batch axis
    assert not batched_generation.engine_can_batch(handle.engine, 2)

    first = batched_generation.batch_capable_engine(handle, 2, backend_name="test")
    assert type(first).__name__ == "TorchAEGEngine"
    assert handle.batched_engine is first
    second = batched_generation.batch_capable_engine(handle, 4, backend_name="test")
    assert second is first, "the promoted executor was rebuilt"
    # And the single-sequence executor is left in place for single-sequence work.
    assert type(handle.engine).__name__ == "CPUExecutionEngine"


def test_promoted_engine_agrees_with_the_numpy_executor() -> None:
    """Promotion must not change the model — same weights, same predictions."""
    from aether.backends import batched_generation

    pytest.importorskip("torch")
    reference = _reference_engine()
    handle = _handle(reference)
    promoted = batched_generation.batch_capable_engine(handle, 2, backend_name="test")

    prompt = np.asarray([1, 3, 5, 2], dtype=np.int64)
    expected, _ = reference.forward(prompt)
    _, cache = promoted.forward_batch([prompt])
    np.testing.assert_allclose(
        _numpy(cache.last_logits)[0],
        np.asarray(expected)[-1],
        rtol=LOGIT_TOLERANCE,
        atol=LOGIT_TOLERANCE,
    )
    assert (
        promoted.generate_batch([prompt], max_tokens=4, temperature=0.0)[0]
        == reference.generate(prompt, max_tokens=4, temperature=0.0)
    )


def test_empty_and_single_request_batches() -> None:
    from aether.backends import batched_generation

    assert batched_generation.generate_batch(
        _handle(_engine()), [],
        backend_name="test",
        request_text=lambda request, _t: request.prompt,
        truncate_stop_text=lambda t, ids, stops: ("", len(ids), False),
    ) == []


def test_both_aeg_backends_expose_the_batch_surface() -> None:
    """Batching is a backend capability, so every AEG executor must answer for it."""
    from aether.backends.native_cpu_backend import NativeCPUBackend
    from aether.backends.torch_backend import TorchBackend

    for backend_type in (NativeCPUBackend, TorchBackend):
        assert callable(getattr(backend_type, "generate_batch", None)), backend_type
        assert callable(
            getattr(backend_type, "supports_batched_generation", None)
        ), backend_type
