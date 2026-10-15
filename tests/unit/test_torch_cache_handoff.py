"""Real CPU Torch execution/typed byte transfers; GPU tests remain optional."""

import numpy as np
import pytest

from aether.runtime.kv_payload import CachePayload


def make_engine(monkeypatch, hybrid=False):
    """Create a real deterministic CPU Torch executor with a dense or hybrid source."""
    pytest.importorskip("torch")
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    monkeypatch.setenv("AETHER_TORCH_DTYPE", "float32")
    from aether.runtime.torch_engine import TorchAEGEngine, TorchHybridAEGEngine

    if hybrid:
        from test_hybrid_rope_scaling import _reference

        return TorchHybridAEGEngine(_reference(None), "cpu")
    from test_aeg_onnx_export import make_engine as cpu_engine

    return TorchAEGEngine(cpu_engine(), "cpu")


@pytest.mark.parametrize("hybrid", [False, True])
def test_actual_torch_prefill_snapshot_roundtrip_cached_decode_and_alias_ownership(
    monkeypatch, hybrid
):
    """Restored typed snapshots preserve execution while owning destination storage."""
    from aether.runtime.torch_cache_handoff import (
        export_torch_cache,
        make_torch_adapter_factory,
        restore_torch_cache,
    )

    source = make_engine(monkeypatch, hybrid)
    factory = make_torch_adapter_factory(source, model_fingerprint="actual-shared-artifact")
    destination = factory()
    assert destination.engine is not source
    assert destination.engine.embedding.data_ptr() == source.embedding.data_ptr()
    assert destination.engine.layers is not source.layers
    assert destination.engine._projection_choices is not getattr(
        source, "_projection_choices", None
    )
    _, cache = source.forward(np.asarray([1, 3, 2]))
    snapshot = export_torch_cache(cache, engine=source, model_fingerprint="actual-shared-artifact")
    decoded = CachePayload.deserialize(
        snapshot.serialize(), expected_model_fingerprint="actual-shared-artifact"
    )
    restored = restore_torch_cache(
        decoded, engine=destination.engine, expected_model_fingerprint="actual-shared-artifact"
    )
    expected, _ = source.forward(np.asarray([4, 5]), cache.clone())
    actual, resumed = destination.forward(np.asarray([4, 5]), restored)
    np.testing.assert_array_equal(actual, expected)
    assert resumed.length == 5
    active = next(index for index, value in enumerate(cache.keys) if value is not None)
    assert restored.keys[active].data_ptr() != cache.keys[active].data_ptr()
    assert snapshot.byte_size > 0
    assert destination.cache_nbytes(cache) >= snapshot.byte_size
    with pytest.raises(ValueError, match="fingerprint"):
        restore_torch_cache(
            decoded, engine=destination.engine, expected_model_fingerprint="wrong-model"
        )


def test_actual_cpu_attention_interop_and_compressed_cache_rejection(monkeypatch):
    """Resume dense CPU/Torch coordinates and reject incompatible cache layouts."""
    from aether.runtime.cache_handoff import export_cpu_cache
    from aether.runtime.torch_cache_handoff import (
        export_torch_cache,
        restore_torch_cache,
        torch_payload_to_cpu_attention,
    )

    destination = make_engine(monkeypatch)
    source = destination.source_engine
    prompt = np.asarray([1, 3, 2])
    _, cpu_cache = source.forward(prompt)
    exported_cpu = export_cpu_cache(cpu_cache, model_fingerprint="same-model")
    restored = restore_torch_cache(
        exported_cpu, engine=destination, expected_model_fingerprint="same-model"
    )
    expected, _ = destination.forward(prompt)
    expected, _ = destination.forward(np.asarray([4]), restored.clone())
    actual, torch_cache = destination.forward(np.asarray([4]), restored)
    np.testing.assert_array_equal(actual, expected)
    snapshot = export_torch_cache(torch_cache, engine=destination, model_fingerprint="same-model")
    resumed_cpu = torch_payload_to_cpu_attention(snapshot, expected_model_fingerprint="same-model")
    torch_logits, _ = destination.forward(np.asarray([5]), torch_cache.clone())
    cpu_logits, _ = source.forward(np.asarray([5]), resumed_cpu)
    np.testing.assert_allclose(torch_logits, cpu_logits, rtol=4e-5, atol=3e-6)
    cpu_cache.positions[0][1] = 2
    with pytest.raises(ValueError):
        restore_torch_cache(
            export_cpu_cache(cpu_cache, model_fingerprint="same-model"),
            engine=destination,
            expected_model_fingerprint="same-model",
        )
    with pytest.raises(ValueError, match="contiguous attention KV"):
        restore_torch_cache(
            exported_cpu,
            engine=make_engine(monkeypatch, hybrid=True),
            expected_model_fingerprint="same-model",
        )


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float8_e4m3fn"])
def test_exact_dtype_bytes_survive_and_cpu_interop_never_converts(monkeypatch, dtype_name):
    """Preserve real BF16/FP8 tensor bytes without claiming model execution support."""
    torch = pytest.importorskip("torch")
    from aether.runtime.torch_cache_handoff import (
        export_torch_cache,
        restore_torch_cache,
        torch_payload_to_cpu_attention,
    )
    from aether.runtime.torch_engine import TorchKVCache

    engine = make_engine(monkeypatch)
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        pytest.skip(f"Torch build has no {dtype_name}")
    engine.compute_dtype = dtype
    keys = [
        torch.ones((2, engine.num_kv_heads, engine.head_dim)).to(dtype)
        for _ in range(engine.num_layers)
    ]
    cache = TorchKVCache(
        keys=keys,
        values=keys,
        length=2,
        last_logits=torch.ones(engine.embedding.shape[0]).to(dtype),
    )
    payload = export_torch_cache(cache, engine=engine, model_fingerprint="dtype-fixture")
    restored = restore_torch_cache(
        CachePayload.deserialize(payload.serialize()),
        engine=engine,
        expected_model_fingerprint="dtype-fixture",
    )
    assert restored.keys[0].dtype == dtype
    assert restored.keys[0] is restored.values[0]
    assert (
        payload.serialize()
        == export_torch_cache(
            restored, engine=engine, model_fingerprint="dtype-fixture"
        ).serialize()
    )
    with pytest.raises(ValueError, match="lossless NumPy representation"):
        torch_payload_to_cpu_attention(payload, expected_model_fingerprint="dtype-fixture")


@pytest.mark.parametrize("hybrid", [False, True])
def test_real_torch_worker_pool_matches_uninterrupted_generation(monkeypatch, hybrid):
    """Separate Torch workers hand off exact state and reproduce real generation."""
    from aether.runtime.disaggregated_execution import DisaggregatedExecutionPool
    from aether.runtime.torch_cache_handoff import make_torch_adapter_factory

    engine = make_engine(monkeypatch, hybrid)
    prompt = np.asarray([1, 3, 2])
    expected = engine.generate(prompt, max_tokens=3, temperature=0.0)
    factory = make_torch_adapter_factory(engine, model_fingerprint="pool-model")
    with DisaggregatedExecutionPool([factory], [factory], prefill_chunk_size=2) as pool:
        result = pool.submit(prompt, max_tokens=3).result(timeout=30)
        assert list(result.tokens) == expected
        assert result.metrics["prefill_chunks"] == 2
        assert result.metrics["state_bytes_handed_off"] > 0
        assert result.cache.model_fingerprint == "pool-model"
    assert pool.snapshot()["live_workers"] == 0


def test_sharded_or_destination_dtype_mismatch_fails_before_inference(monkeypatch):
    """Reject unsupported collective ownership and implicit precision changes."""
    from aether.runtime.torch_cache_handoff import (
        export_torch_cache,
        make_torch_adapter_factory,
        restore_torch_cache,
    )

    engine = make_engine(monkeypatch)
    _, cache = engine.forward(np.asarray([1, 2]))
    payload = export_torch_cache(cache, engine=engine, model_fingerprint="model")
    engine.compute_dtype = engine.torch.float16
    with pytest.raises(ValueError, match="dtype"):
        restore_torch_cache(payload, engine=engine, expected_model_fingerprint="model")
    engine.devices = [engine.torch.device("cpu"), engine.torch.device("cpu")]
    with pytest.raises(ValueError, match="nonsharded"):
        make_torch_adapter_factory(engine, model_fingerprint="model")


@pytest.mark.parametrize("torch_prefill", [False, True])
def test_real_mixed_cpu_torch_pools_resume_actual_cache(monkeypatch, torch_prefill):
    """Transfer actual FP32 attention state in both CPU/Torch worker directions."""
    from aether.runtime.disaggregated_execution import (
        DisaggregatedExecutionPool,
        make_cpu_adapter_factory,
    )
    from aether.runtime.torch_cache_handoff import make_torch_adapter_factory

    engine = make_engine(monkeypatch)
    prompt = np.asarray([1, 3, 2])
    expected = engine.generate(prompt, max_tokens=3, temperature=0.0)
    assert engine.source_engine.generate(prompt, max_tokens=3, temperature=0.0) == expected
    cpu_factory = make_cpu_adapter_factory(
        engine.source_engine,
        model_fingerprint="mixed-model",
        cache_kind="attention_kv_v1",
    )
    torch_factory = make_torch_adapter_factory(engine, model_fingerprint="mixed-model")
    prefill, decode = (
        (torch_factory, cpu_factory) if torch_prefill else (cpu_factory, torch_factory)
    )
    with DisaggregatedExecutionPool([prefill], [decode], prefill_chunk_size=2) as pool:
        result = pool.submit(prompt, max_tokens=3).result(timeout=30)
        assert list(result.tokens) == expected
        assert result.metrics["state_bytes_handed_off"] > 0
        assert result.cache.model_fingerprint == "mixed-model"
    assert pool.snapshot()["live_workers"] == 0
