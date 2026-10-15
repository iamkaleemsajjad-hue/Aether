"""Actual CPU forward/state handoff, worker lifecycle and admission contracts."""
from __future__ import annotations

from concurrent.futures import CancelledError
import hashlib
import queue
import threading

import numpy as np
import pytest

from aether.kernels.native_cpu import NativeCPUKernels
from aether.runtime.cpu_engine import CPUExecutionEngine
from aether.runtime.cache_handoff import export_cpu_cache, restore_cpu_cache
from aether.runtime.disaggregated_execution import (
    CPUDecoderAdapter, DisaggregatedExecutionPool, PooledGenerationAdapter,
)
from aether.runtime.kv_payload import CachePayload
from aether.runtime.sampling import SamplingConfig, sample_token, token_probabilities
from test_memory_efficiency import HEADS, KV_HEADS, _weights

PROMPT = np.asarray([1, 3, 5, 2], dtype=np.int64)


def _setup():
    weights = _weights(tied=False)
    fingerprint = "sha256:" + hashlib.sha256(weights.embedding.tobytes() + weights.lm_head.tobytes()).hexdigest()
    def engine():
        return CPUExecutionEngine(weights, num_heads=HEADS, num_kv_heads=KV_HEADS,
                                  kernels=NativeCPUKernels.reference())
    def adapter():
        return CPUDecoderAdapter(engine(), model_fingerprint=fingerprint, cache_kind="attention_kv_v1")
    return engine, adapter, fingerprint


def test_attention_handoff_preserves_actual_cached_decode_and_owned_arrays():
    engine, _, fingerprint = _setup()
    source, target = engine(), engine()
    _, cache = source.forward(PROMPT)
    cache.reserve(1024)
    payload = export_cpu_cache(cache, model_fingerprint=fingerprint)
    wire = payload.serialize()
    restored = restore_cpu_cache(CachePayload.deserialize(wire, expected_model_fingerprint=fingerprint),
                                 expected_model_fingerprint=fingerprint)
    expected, cache = source.forward(np.asarray([7, 4]), cache)
    actual, resumed = target.forward(np.asarray([7, 4]), restored)
    np.testing.assert_array_equal(actual, expected)
    assert resumed.length == cache.length == 6
    assert not np.shares_memory(cache.keys[0], restored.keys[0])
    assert payload.byte_size > 0
    assert payload.logical_length == 4


def test_cache_aliases_and_compressed_positions_survive_without_spare_capacity():
    from aether.runtime.cpu_engine import KVCache
    keys = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    cache = KVCache(2, keys=[keys, keys], values=[keys, keys],
        positions=[np.asarray([0, 4, 8]), np.asarray([0, 4, 8])], logical_length=9,
        stored_lengths=[3, 3], last_logits=np.asarray([1., 2.], dtype=np.float32))
    payload = export_cpu_cache(cache, model_fingerprint="cache-fixture")
    restored = restore_cpu_cache(payload, expected_model_fingerprint="cache-fixture")
    assert restored.keys[0] is restored.keys[1] is restored.values[0]
    assert restored.stored_length == 3 and restored.length == 9
    np.testing.assert_array_equal(restored.positions[0], [0, 4, 8])
    with pytest.raises(ValueError, match="fingerprint"):
        restore_cpu_cache(payload, expected_model_fingerprint="other")


def test_real_worker_pool_matches_uninterrupted_tokens_cache_and_metrics():
    engine, adapter, fingerprint = _setup()
    expected, expected_cache = engine().generate_with_cache(PROMPT, max_tokens=5, temperature=0)
    with DisaggregatedExecutionPool([adapter], [adapter], prefill_chunk_size=2) as pool:
        result = pool.submit(PROMPT, max_tokens=5).result(timeout=10)
        assert list(result.tokens) == expected
        restored = restore_cpu_cache(result.cache, expected_model_fingerprint=fingerprint)
        np.testing.assert_allclose(restored.last_logits, expected_cache.last_logits, rtol=0, atol=3e-7)
        assert restored.length == expected_cache.length
        assert result.metrics["prefill_chunks"] == 2
        assert result.metrics["decode_forward_calls"] == 5
        assert result.metrics["state_bytes_handed_off"] > 0
        assert result.metrics["transport"] == "in_process_typed_snapshot"
        assert result.metrics["ttft_ms"] is not None
        assert pool.snapshot()["completed"] == 1
    assert pool.snapshot()["live_workers"] == 0


def test_engine_shaped_bridge_resumes_owned_cache_without_reprefilling():
    _, adapter, fingerprint = _setup()
    with DisaggregatedExecutionPool([adapter], [adapter], prefill_chunk_size=2) as pool:
        bridge = PooledGenerationAdapter(pool)
        tokens, cache = bridge.generate_with_cache(PROMPT, 2, temperature=0)
        resumed, updated = bridge.generate_with_cache(np.asarray([], dtype=np.int64), 2,
                                                     cache=cache, temperature=0)
        assert len(tokens) == len(resumed) == 2
        assert updated.logical_length == cache.logical_length + 2
        assert updated.model_fingerprint == fingerprint
        assert bridge.generation_stats()["prefill_chunks"] == 0
        with pytest.raises(ValueError, match="unsupported active"):
            bridge.generate(PROMPT, ttt_slots=[{}])


def test_first_ready_token_is_observable_before_next_decode_returns():
    engine, _, fingerprint = _setup()
    entered, release = threading.Event(), threading.Event()
    class BlockingDecode(CPUDecoderAdapter):
        def forward(self, ids, cache):
            entered.set()
            assert release.wait(10)
            return super().forward(ids, cache)
    prefill = lambda: CPUDecoderAdapter(engine(), model_fingerprint=fingerprint, cache_kind="attention_kv_v1")
    decode = lambda: BlockingDecode(engine(), model_fingerprint=fingerprint, cache_kind="attention_kv_v1")
    pool = DisaggregatedExecutionPool([prefill], [decode])
    try:
        handle = pool.submit(PROMPT, max_tokens=2)
        iterator = handle.iter_tokens(timeout=5)
        first = next(iterator)
        assert first >= 0
        assert entered.wait(5) and not release.is_set()
        release.set()
        assert len([first, *iterator]) == 2
        handle.result(timeout=5)
    finally:
        release.set()
        pool.close()


def test_bounded_admission_backpressure_cancellation_and_shutdown():
    _, adapter, _ = _setup()
    with DisaggregatedExecutionPool([adapter], [adapter], max_pending=1, output_buffer_tokens=1) as pool:
        handle = pool.submit(PROMPT, max_tokens=100)
        assert handle._output.get(timeout=5) >= 0
        with pytest.raises(queue.Full, match="capacity"):
            pool.submit(PROMPT)
        assert handle.cancel()
        with pytest.raises(CancelledError):
            handle.result(timeout=5)
        assert pool.snapshot()["active"] == 0
        assert pool.submit(PROMPT, max_tokens=1).result(timeout=5).tokens
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit(PROMPT)


def test_distinct_engines_required_and_failed_factory_closes_constructed_adapter():
    _, adapter, _ = _setup()
    single = adapter()
    with pytest.raises(ValueError, match="distinct engine"):
        DisaggregatedExecutionPool([lambda: single], [lambda: single])
    closed = []
    first = adapter()
    first.close = lambda: closed.append(True)
    def failure():
        raise RuntimeError("factory failed")
    with pytest.raises(RuntimeError, match="factory failed"):
        DisaggregatedExecutionPool([lambda: first], [failure])
    assert closed == [True]


def test_error_and_cache_budget_do_not_leave_admitted_requests():
    _, adapter, _ = _setup()
    with DisaggregatedExecutionPool([adapter], [adapter], max_cache_bytes_per_request=1) as pool:
        with pytest.raises(ValueError, match="byte bound"):
            pool.submit(PROMPT).result(timeout=5)
        assert pool.snapshot()["failed"] == 1
        assert pool.snapshot()["active"] == 0


def test_sampling_masks_topk_nucleus_and_grammar_advance():
    logits = np.asarray([0., 1., 3., 2.])
    probs = token_probabilities(logits, SamplingConfig(1., 2, .5))
    np.testing.assert_array_equal(probs, [0, 0, 1, 0])
    class Grammar:
        advanced = []
        def get_token_mask(self): return bytes([0b0010])
        def advance(self, token): self.advanced.append(token); return 0
    grammar = Grammar()
    assert sample_token(logits, SamplingConfig(), grammar_session=grammar) == 1
    assert grammar.advanced == [1]


def test_actual_hybrid_attention_and_recurrent_handoff_equivalence():
    from cpu_family_fixtures import engine_for_family
    source, target = engine_for_family("hybrid"), engine_for_family("hybrid")
    _, cache = source.forward(np.asarray([1, 3, 5], dtype=np.int64))
    payload = export_cpu_cache(cache, model_fingerprint="hybrid-state-fixture")
    restored = restore_cpu_cache(CachePayload.deserialize(payload.serialize()),
                                 expected_model_fingerprint="hybrid-state-fixture")
    expected, expected_cache = source.forward(np.asarray([2, 4]), cache)
    actual, actual_cache = target.forward(np.asarray([2, 4]), restored)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_cache.ssm.states[0].h, expected_cache.ssm.states[0].h)
    assert actual_cache.length == actual_cache.kv.length == actual_cache.ssm.length == 5


@pytest.mark.parametrize("family,kind", [("dense", "attention_kv_v1"), ("hybrid", "hybrid_v1"),
    ("mamba", "mamba_v1"), ("mamba2", "mamba2_v1"), ("rwkv", "rwkv_v1"), ("mla", "mla_replay_v1")])
def test_actual_family_handoff_pools_and_worker_state_isolation(family, kind):
    from cpu_family_fixtures import engine_for_family
    from aether.runtime.disaggregated_execution import clone_cpu_engine_for_worker, make_cpu_adapter_factory
    source = engine_for_family(family)
    worker = clone_cpu_engine_for_worker(source)
    assert worker is not source and worker.weights is source.weights
    if family == "dense":
        assert worker.kernels is source.kernels
        assert worker._offset_norm_cache is not source._offset_norm_cache
        assert not np.shares_memory(worker._cos, source._cos)
    if family == "hybrid":
        assert worker._transformer is not source._transformer
        assert worker._mamba is not source._mamba
        assert worker._transformer.weights.embedding is source.weights.embedding
    _, cache = source.forward(PROMPT)
    payload = export_cpu_cache(cache, model_fingerprint="actual-" + family)
    restored = restore_cpu_cache(CachePayload.deserialize(payload.serialize()), expected_model_fingerprint="actual-" + family)
    expected, _ = source.forward(np.asarray([7, 4]), cache)
    actual, _ = worker.forward(np.asarray([7, 4]), restored)
    np.testing.assert_array_equal(actual, expected)
    factory = make_cpu_adapter_factory(source, model_fingerprint="actual-" + family, cache_kind=kind)
    reference = clone_cpu_engine_for_worker(source)
    logits, expected_cache = reference.forward(PROMPT)
    tokens = []
    for _ in range(4):
        token = int(np.argmax(logits[-1])); tokens.append(token)
        logits, expected_cache = reference.forward(np.asarray([token]), expected_cache)
    with DisaggregatedExecutionPool([factory], [factory], prefill_chunk_size=2) as pool:
        result = pool.submit(PROMPT, max_tokens=4).result(timeout=10)
        assert list(result.tokens) == tokens
        resumed = restore_cpu_cache(result.cache, expected_model_fingerprint="actual-" + family)
        np.testing.assert_allclose(resumed.last_logits, expected_cache.last_logits, rtol=0, atol=5e-7)
        assert resumed.length == expected_cache.length == len(PROMPT) + 4


def test_sampling_tiny_temperature_and_prompt_overflow_are_not_silent():
    np.testing.assert_array_equal(token_probabilities([1000., 999.], SamplingConfig(1e-320)), [1, 0])
    _, factory, _ = _setup()
    with DisaggregatedExecutionPool([factory], [factory]) as pool:
        with pytest.raises(ValueError, match="integer array"):
            pool.submit(np.asarray([2**63], dtype=np.uint64))
        first = pool.submit(PROMPT, max_tokens=0).result(timeout=5)
        resumed = pool.submit([], cache=first.cache, max_tokens=1).result(timeout=5)
        assert len(resumed.tokens) == 1


def test_cache_identity_does_not_allow_wrong_model_geometry():
    _, factory, fingerprint = _setup()
    adapter = factory()
    _, cache = adapter.engine.forward(PROMPT)
    for index in range(cache.num_layers):
        cache.keys[index] = cache.keys[index][:, :, :2]
        cache.values[index] = cache.values[index][:, :, :2]
    payload = export_cpu_cache(cache, model_fingerprint=fingerprint)
    with pytest.raises(ValueError, match="head geometry"):
        adapter.import_cache(payload)


def test_real_worker_shutdown_timeout_does_not_claim_kernel_cancellation():
    _, factory, _ = _setup()
    entered, release = threading.Event(), threading.Event()
    class Blocking(CPUDecoderAdapter):
        def forward(self, ids, cache):
            entered.set()
            if not release.wait(5): raise TimeoutError("Fixture kernel was not released")
            return super().forward(ids, cache)
    base = factory()
    blocked = lambda: Blocking(base.engine, model_fingerprint=base.model_fingerprint, cache_kind=base.cache_kind)
    pool = DisaggregatedExecutionPool([blocked], [factory])
    handle = pool.submit(PROMPT)
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="shutdown has not completed"):
            pool.close(timeout=.01)
        assert pool.snapshot()["live_workers"] > 0
    finally:
        release.set()
        pool.close(timeout=5)
    with pytest.raises(CancelledError): handle.result(timeout=1)
    assert pool.snapshot()["live_workers"] == pool.snapshot()["active"] == 0


def test_owned_worker_cleanup_failure_is_reported_after_threads_stop():
    _, factory, _ = _setup()
    base = factory()
    class FailedClose(CPUDecoderAdapter):
        def close(self): raise OSError("device resource cleanup failed")
    failed = lambda: FailedClose(base.engine, model_fingerprint=base.model_fingerprint, cache_kind=base.cache_kind)
    pool = DisaggregatedExecutionPool([failed], [factory])
    assert pool.submit(PROMPT, max_tokens=1).result(timeout=5).tokens
    with pytest.raises(RuntimeError, match="resource cleanup failed"):
        pool.close(timeout=5)
    assert pool.snapshot()["live_workers"] == 0
    assert pool.snapshot()["cleanup_failures"] == 1


def test_initialization_reports_factory_and_cleanup_failures():
    _, factory, _ = _setup()
    adapter = factory()
    def failed_close(): raise OSError("cleanup failed")
    adapter.close = failed_close
    def failed_factory(): raise ValueError("construction failed")
    with pytest.raises(RuntimeError, match="construction failed.*resource cleanup also failed.*cleanup failed"):
        DisaggregatedExecutionPool([lambda: adapter], [failed_factory])
