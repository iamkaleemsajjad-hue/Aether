"""Real Torch CPU forwards verify device-adapter cache rollback contracts."""
import numpy as np
import pytest

from aether.runtime.spec_decode import numpy_logits
from aether.runtime.speculative_engine import SpeculativeGenerationAdapter
from cpu_family_fixtures import engine_for_family


def test_real_torch_stream_cross_thread_close_releases_borrowed_engine_locks(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    pytest.importorskip("torch")
    monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
    from aether.runtime.torch_engine import TorchAEGEngine
    target, draft = (TorchAEGEngine(engine_for_family("dense"), "cpu") for _ in range(2))
    adapter = SpeculativeGenerationAdapter(target, draft, gamma=2)
    stream = adapter.generate_iter(np.asarray([1, 3, 5]), 5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert isinstance(executor.submit(next, stream).result(timeout=10), int)
    stream.close()  # A different thread closes the paused real inference.
    assert all(not lock.locked() for lock in adapter._execution_locks)
    assert len(adapter.generate(np.asarray([1, 3, 5]), 2)) == 2


def test_real_torch_streams_can_interleave_without_holding_a_lock_across_yield(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
    from aether.runtime.torch_engine import TorchAEGEngine
    target, draft = (TorchAEGEngine(engine_for_family("dense"), "cpu") for _ in range(2))
    adapter = SpeculativeGenerationAdapter(target, draft, gamma=2)
    prompt = np.asarray([1, 3, 5])
    expected = adapter.generate(prompt, 6)
    first, second = adapter.generate_iter(prompt, 6), adapter.generate_iter(prompt, 6)
    a, b = [], []
    try:
        for _ in range(6):
            a.append(next(first))
            b.append(next(second))
            assert all(not lock.locked() for lock in adapter._execution_locks)
        assert a == b == expected
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("family", ["dense", "hybrid", "mamba", "mamba2", "rwkv", "mla"])
def test_actual_torch_cpu_speculation_preserves_greedy_tokens_and_owned_cache(monkeypatch, family):
    pytest.importorskip("torch")
    monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
    from aether.runtime.torch_engine import TorchAEGEngine, TorchHybridAEGEngine
    from aether.runtime.torch_state_engine import (
        TorchMambaAEGEngine, TorchMamba2AEGEngine, TorchRWKVAEGEngine, TorchMLAAEGEngine,
    )
    engine_type = {"dense": TorchAEGEngine, "hybrid": TorchHybridAEGEngine,
        "mamba": TorchMambaAEGEngine, "mamba2": TorchMamba2AEGEngine,
        "rwkv": TorchRWKVAEGEngine, "mla": TorchMLAAEGEngine}[family]
    target = engine_type(engine_for_family(family), "cpu")
    draft_source = engine_for_family(family)
    draft_source.weights.lm_head *= -1
    draft = engine_type(draft_source, "cpu")
    prompt = np.asarray([1, 3, 5, 2])
    logits, expected_cache = target.forward(prompt)
    expected = []
    for _ in range(7):
        token = int(np.argmax(numpy_logits(logits)[-1])); expected.append(token)
        logits, expected_cache = target.forward(np.asarray([token]), expected_cache)
    adapter = SpeculativeGenerationAdapter(target, draft, gamma=4, min_acceptance_rate=0)
    first, session = adapter.generate_with_cache(prompt, 4)
    before = numpy_logits(session.last_logits).copy()
    second, updated = adapter.generate_with_cache([], 3, cache=session)
    assert first + second == expected
    assert updated.length == len(prompt) + 7
    np.testing.assert_array_equal(numpy_logits(session.last_logits), before)
    np.testing.assert_allclose(numpy_logits(updated.last_logits), numpy_logits(expected_cache.last_logits), rtol=0, atol=1e-6)
    assert adapter.speculative_stats()["target_forward_passes"] > 0
