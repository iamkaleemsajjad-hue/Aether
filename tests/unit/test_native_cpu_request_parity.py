"""Actual CPU request weights, streaming and request-local adaptation parity."""

from dataclasses import fields

import numpy as np
import pytest

from aether.backends.base import GenerationRequest
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.backends.native_cpu_backend import NativeCPUBackend
from aether.core.exceptions import BackendError
from aether.core.hash_utils import compute_file_hash
from aether.kernels.native_cpu import NativeCPUKernels
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
from aether.runtime.r5_ttt_engine import TTTFastWeightEngine


class IntegerTokenizer:
    """Lossless deterministic tokens; inference still uses actual decoder kernels."""

    eos_token_id = None
    pad_token_id = None

    def __call__(self, text, **kwargs):
        """Encode the literal integer sequence."""
        return {"input_ids": np.array([[int(value) for value in text.split()]], dtype=np.int64)}

    def decode(self, ids, **kwargs):
        """Decode without obscuring incremental token differences."""
        return "".join(chr(ord("a") + int(value)) for value in ids)


@pytest.fixture
def native_model(tmp_path):
    """Construct a real decoder and hash-bound task-vector payload."""
    rng = np.random.default_rng(82)

    def matrix(out, features):
        return (rng.normal(size=(out, features)) * 0.15).astype(np.float32)

    layer = LayerWeights(
        attention_norm=np.ones(8, np.float32),
        q_proj=matrix(8, 8),
        k_proj=matrix(4, 8),
        v_proj=matrix(4, 8),
        o_proj=matrix(8, 8),
        ffn_norm=np.ones(8, np.float32),
        gate_proj=matrix(12, 8),
        up_proj=matrix(12, 8),
        down_proj=matrix(8, 12),
    )
    weights = ModelWeights(
        embedding=matrix(17, 8),
        layers=[layer],
        final_norm=np.ones(8, np.float32),
        lm_head=matrix(17, 8),
        context_length=32,
    )
    engine = CPUExecutionEngine(weights, 2, 1, kernels=NativeCPUKernels.reference())
    vector = tmp_path / "task.npz"
    np.savez(vector, head=-2.0 * weights.lm_head)
    handle = CompiledAEGHandle(
        "tiny",
        tmp_path,
        {"artifacts": {"task.npz": compute_file_hash(vector)}},
        metadata={
            "task_vectors": {
                "vectors": [
                    {
                        "name": "reverse",
                        "path": "task.npz",
                        "tensors": [{"name": "lm_head", "key": "head", "shape": [17, 8]}],
                    }
                ]
            }
        },
        engine=engine,
        tokenizer=IntegerTokenizer(),
        lora_adapters={"projection": {(0, "v_proj"): (matrix(2, 8), matrix(4, 2), 2.0)}},
    )
    backend = NativeCPUBackend()
    backend._models["tiny"] = handle
    return backend, handle


def request(prompt="1 2 3", **extra):
    """Create a deterministic greedy request."""
    return GenerationRequest("tiny", prompt=prompt, max_tokens=3, temperature=0.0, extra=extra)


def ttt_engine():
    """Construct actual request-local fast-weight adaptation."""
    return TTTFastWeightEngine(n_layers=1, hidden_size=8, rank=2, learning_rate=0.3)


def execute(backend, mode, req):
    """Consume the same request through either public backend method."""
    if mode == "stream":
        return "".join(backend.generate_stream(req))
    return backend.generate(req)


@pytest.mark.parametrize("with_lora", [False, True])
def test_actual_task_weight_stream_matches_selected_engine(native_model, with_lora):
    """Task and adapter selection must reach both real forward paths."""
    backend, handle = native_model
    original_head = handle.engine.weights.lm_head.copy()
    extras = {"task_weights": {"reverse": 1.0}}
    selected = handle.engine.with_task_deltas({"lm_head": -2.0 * original_head})
    if with_lora:
        extras["adapter_name"] = "projection"
        selected = selected.with_lora_adapter(handle.lora_adapters, "projection")
    expected = handle.tokenizer.decode(selected.generate(np.array([1, 2, 3]), max_tokens=3))
    plain = backend.generate(request()).text
    actual = backend.generate(request(**extras))
    assert actual.text == expected
    assert "".join(backend.generate_stream(request(**extras))) == expected
    assert expected != plain
    np.testing.assert_array_equal(handle.engine.weights.lm_head, original_head)


@pytest.mark.parametrize("mode", ["blocking", "stream"])
@pytest.mark.parametrize("adapter", ["missing", 3])
def test_stream_uses_same_lora_validation(native_model, mode, adapter):
    """Malformed adapters must fail consistently before inference."""
    backend, _ = native_model
    with pytest.raises(BackendError, match=r"adapter is absent|non-empty string"):
        execute(backend, mode, request(adapter_id=adapter))


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_stream_authenticates_task_payload(native_model, mode):
    """Streaming must verify the actual requested task-vector bytes."""
    backend, handle = native_model
    (handle.aeg_path / "task.npz").write_bytes(b"changed after authenticated load")
    with pytest.raises(BackendError, match="integrity validation"):
        execute(backend, mode, request(task_weights={"reverse": 1.0}))


@pytest.mark.parametrize("mode", ["blocking", "stream"])
@pytest.mark.parametrize(
    "coefficient", [float("nan"), float("inf"), -float("inf"), True, -1.0, 1e308]
)
def test_task_coefficients_fail_before_request_arithmetic(native_model, mode, coefficient):
    """Neither execution path may introduce invalid coefficients into real weights."""
    backend, _ = native_model
    with pytest.raises(BackendError, match=r"finite|non-negative|overflow"):
        execute(backend, mode, request(task_weights={"reverse": coefficient}))


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_authenticated_nonfinite_task_payload_is_rejected(native_model, mode):
    """A valid artifact hash cannot certify nonfinite numerical task-vector values."""
    backend, handle = native_model
    path = handle.aeg_path / "task.npz"
    values = handle.engine.weights.lm_head.copy()
    values[0, 0] = np.nan
    np.savez(path, head=values)
    handle.manifest["artifacts"]["task.npz"] = compute_file_hash(path)
    with pytest.raises(BackendError, match="not finite"):
        execute(backend, mode, request(task_weights={"reverse": 1.0}))


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_unknown_zero_task_coordinate_is_validated(native_model, mode):
    """Dropping zero coordinates must not bypass known-vector validation."""
    backend, _ = native_model
    with pytest.raises(BackendError, match="unknown vector"):
        execute(backend, mode, request(task_weights={"unknown": 0.0}))


@pytest.mark.parametrize("mode", ["blocking", "stream"])
@pytest.mark.parametrize("variant", ["task", "lora"])
@pytest.mark.parametrize("restore_base", [False, True])
def test_static_weight_session_switch_recomputes_real_prefix(
    native_model, mode, variant, restore_base
):
    """Cached logits and KV must match the selected task/adapter contract."""
    backend, handle = native_model
    adapted = (
        {"task_weights": {"reverse": 1.0}} if variant == "task" else {"adapter_id": "projection"}
    )
    before, after = (adapted, {}) if restore_base else ({}, adapted)
    prefill = request(aether_kv_session_id="switch", **before)
    prefill.max_tokens = 0
    backend.generate(prefill)
    old_cache = handle.session_caches["switch"][1]
    expected = backend.generate(request(**after)).text
    actual = execute(backend, mode, request(aether_kv_session_id="switch", **after))
    assert (actual if mode == "stream" else actual.text) == expected
    assert handle.session_caches["switch"][1] is not old_cache
    if mode == "blocking":
        assert actual.metrics["kv_reuse"] is False


def test_equivalent_task_sessions_reuse_real_logits_across_selected_clones(native_model):
    """Static payload coefficients, rather than temporary engine object IDs, govern reuse."""
    backend, handle = native_model
    first = request(aether_kv_session_id="same", task_weights={"reverse": 1.0})
    first.max_tokens = 0
    backend.generate(first)
    old_cache = handle.session_caches["same"][1]
    actual = backend.generate(request(aether_kv_session_id="same", task_weights={"reverse": 1.0}))
    assert actual.text == backend.generate(request(task_weights={"reverse": 1.0})).text
    assert actual.metrics["kv_reuse"] is True
    assert actual.metrics["kv_reused_tokens"] == 3
    assert handle.session_caches["same"][1] is old_cache


def test_multi_agent_prefix_partitions_actual_static_weight_contracts(native_model):
    """Shared raw prefix KV must preserve separate base, task and adapter arithmetic."""
    from aether.runtime.r2_multi_agent_kv import MultiAgentKVCoordinator

    backend, _ = native_model
    coordinator = MultiAgentKVCoordinator(max_shared_blocks=8)
    prefix_hash = coordinator.hash_prefix("1 2")
    coordinator.create_agent_session("agent", prefix_hash=prefix_hash)
    shared = {
        "multi_agent_kv_coordinator": coordinator,
        "multi_agent_prefix": "1 2",
        "multi_agent_prefix_hash": prefix_hash,
    }
    for variant in ({}, {"task_weights": {"reverse": 1.0}}, {"adapter_id": "projection"}):
        expected = backend.generate(request(**variant)).text
        first = backend.generate(request(**shared, **variant))
        second = backend.generate(request(**shared, **variant))
        assert first.text == second.text == expected
        assert first.metrics["multi_agent_kv_reuse"] is False
        assert second.metrics["multi_agent_kv_reused_tokens"] == 2


def test_actual_ttt_slots_affect_logits_and_stream_matches_blocking(native_model):
    """Adapted logits and generated tokens agree with an independent request."""
    backend, handle = native_model
    manual = ttt_engine()
    manual.begin_request("manual")
    manual.adapt("manual", handle.engine.weights.embedding[[1, 2, 3]].tolist())
    slots = [manual.get_fast_weights("manual", 0)]
    baseline, _ = handle.engine.forward(np.array([1, 2, 3]))
    adapted, _ = handle.engine.forward(np.array([1, 2, 3]), ttt_slots=slots)
    assert not np.allclose(baseline, adapted, atol=1e-7)
    expected = handle.tokenizer.decode(
        handle.engine.generate(np.array([1, 2, 3]), max_tokens=3, ttt_slots=slots)
    )
    manual.end_request("manual")
    active = ttt_engine()
    blocking = backend.generate(request(ttt_engine=active, ttt_request_id="blocking"))
    streamed = "".join(backend.generate_stream(request(ttt_engine=active, ttt_request_id="stream")))
    assert blocking.text == streamed == expected
    assert blocking.metrics["ttt_adaptation_loss"] > 0
    assert active.stats.total_adapt_steps == active.stats.total_requests == 2
    assert not active._active_weights


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_ttt_does_not_consume_and_releases_base_session(native_model, mode):
    """Request-local TTT cannot reuse or leave persistable stale base state."""
    backend, handle = native_model
    backend.generate(request(aether_kv_session_id="session"))
    old_ids, old_cache = handle.session_caches["session"]
    old_keys = [value.copy() for value in old_cache.keys]
    old_length = old_cache.length
    prompt = " ".join(str(int(value)) for value in old_ids) + " 4"
    active = ttt_engine()
    actual = execute(
        backend, mode, request(prompt, aether_kv_session_id="session", ttt_engine=active)
    )
    if mode == "blocking":
        assert actual.metrics["kv_reuse"] is False
    assert "session" not in handle.session_caches
    assert old_cache.length == old_length
    for before, after in zip(old_keys, old_cache.keys, strict=True):
        np.testing.assert_array_equal(before, after)
    assert not active._active_weights


@pytest.mark.parametrize("hook", ["task", "lora", "ttt"])
def test_adapted_continuous_request_uses_actual_selected_engine(native_model, hook):
    """Unsupported batch hooks retain real request arithmetic via standard decode."""
    backend, _ = native_model
    extras = (
        {"task_weights": {"reverse": 1.0}}
        if hook == "task"
        else ({"adapter_id": "projection"} if hook == "lora" else {"ttt_engine": ttt_engine()})
    )
    expected = backend.generate(request(**extras))
    actual = backend.generate(request(use_continuous_batching=True, **extras))
    assert actual.text == expected.text
    assert actual.metrics.get("scheduler") != "continuous_batching"
    assert backend.get_batcher("tiny") is None


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_legacy_draft_does_not_silently_ignore_active_ttt(native_model, mode):
    """A draft adapter without fast-weight hooks must reject active adaptation."""
    backend, handle = native_model
    active = ttt_engine()
    with pytest.raises(BackendError, match="does not support active TTT"):
        execute(backend, mode, request(ttt_engine=active, spec_draft_engine=handle.engine))
    assert not active._active_weights
    assert active.stats.total_adapt_steps == 0


def test_stream_close_releases_real_adapted_slots(native_model):
    """Closing a partially consumed stream releases request-local fast weights."""
    backend, handle = native_model
    active = ttt_engine()
    stream = backend.generate_stream(
        request(ttt_engine=active, ttt_request_id="cancelled", aether_kv_session_id="session")
    )
    assert next(stream)
    assert "cancelled" in active._active_weights
    stream.close()
    assert not active._active_weights
    assert active.stats.total_adapt_steps == active.stats.total_requests == 1
    assert "session" not in handle.session_caches


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_ttt_setup_rejection_releases_slots(native_model, tmp_path, mode):
    """Feature setup failures must also complete adaptation cleanup."""
    from aether.runtime.config import RuntimeConfig
    from aether.runtime.execution_features import RuntimeExecutionFeatures

    backend, _ = native_model
    active = ttt_engine()
    features = RuntimeExecutionFeatures(
        RuntimeConfig(disaggregate_prefill_decode=True), tmp_path / "features"
    )
    try:
        with pytest.raises(ValueError, match="active weight/adapter contract"):
            execute(backend, mode, request(ttt_engine=active, _execution_features=features))
    finally:
        features.close()
    assert not active._active_weights
    assert active.stats.total_requests == 1


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_generation_error_releases_ttt_slots(native_model, mode):
    """A real grammar failure after forward execution releases TTT state."""

    class RejectingGrammar:
        def get_token_mask(self):
            return bytes(3)

    backend, _ = native_model
    active = ttt_engine()
    with pytest.raises(ValueError, match="no valid next token"):
        execute(backend, mode, request(ttt_engine=active, grammar_session=RejectingGrammar()))
    assert not active._active_weights
    assert active.stats.total_requests == 1


@pytest.mark.parametrize("mode", ["blocking", "stream"])
def test_adaptation_interrupt_releases_ttt_slots(native_model, mode):
    """Cancellation must unwind completed mathematical adaptation."""

    class InterruptedTTT(TTTFastWeightEngine):
        def adapt(self, *args, **kwargs):
            super().adapt(*args, **kwargs)
            raise KeyboardInterrupt("actual adaptation interrupted")

    backend, _ = native_model
    active = InterruptedTTT(n_layers=1, hidden_size=8, rank=2, learning_rate=0.3)
    with pytest.raises(KeyboardInterrupt, match="adaptation interrupted"):
        execute(backend, mode, request(ttt_engine=active))
    assert not active._active_weights
    assert active.stats.total_requests == 1


def test_public_runtime_set_task_weights_changes_stream_and_blocking(tmp_path):
    """The Compiler and public Runtime apply task weights to actual streamed AEG output."""
    from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
    from aether.backends.hardware_validation import _write_probe_checkpoint
    from aether.core.aeg_format import AEGPackage

    source = tmp_path / "source"
    _write_probe_checkpoint(source)
    config = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "compiler"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    artifact = tmp_path / "task-model.aeg"
    Compiler(config).compile(str(source), output_path=artifact)
    package = AEGPackage(artifact).load()
    head = package.weight_store().dequantize_all()["lm_head"]
    vector = artifact / "merging" / "reverse.npz"
    vector.parent.mkdir(exist_ok=True)
    np.savez(vector, head=-2.0 * head)
    package.metadata["task_vectors"] = {
        "vectors": [
            {
                "name": "reverse",
                "path": "merging/reverse.npz",
                "tensors": [{"name": "lm_head", "key": "head", "shape": list(head.shape)}],
            }
        ]
    }
    package.save()
    with Runtime(
        RuntimeConfig(
            allow_uncertified=True,
            backend_name="aether_cpu",
            execution_devices=["cpu"],
            model_cache_dir=str(tmp_path / "runtime"),
            enable_semantic_cache=False,
        )
    ) as runtime:
        model = str(artifact)
        baseline = runtime.generate(model, "hello world", max_tokens=3, temperature=0.0)
        runtime.set_task_weights(model, reverse=1.0)
        blocking = runtime.generate(model, "hello world", max_tokens=3, temperature=0.0)
        streaming = "".join(
            runtime.generate_stream(model, "hello world", max_tokens=3, temperature=0.0)
        )
        assert blocking.text == streaming
        assert blocking.text != baseline.text
        assert blocking.metrics.extra["task_reweighting"]["vectors"] == ["reverse"]
