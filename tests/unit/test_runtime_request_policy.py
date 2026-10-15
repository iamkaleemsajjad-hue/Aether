"""Task-weight admission, immutable request snapshots and local cache isolation."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aether.backends.base import GenerationRequest
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.core.exceptions import BackendError
from aether.runtime.lifecycle import runtime_lifecycle
from aether.runtime.r2_multi_agent_kv import MultiAgentKVCoordinator
from aether.runtime.runtime import Runtime


def owner():
    runtime = Runtime.__new__(Runtime)
    runtime._lock = threading.RLock()
    return runtime


@pytest.mark.parametrize("weights", [
    None, [], 1, True, {}, {"a": 0}, {"a": True}, {"a": False},
    {"a": float("nan")}, {"a": float("inf")}, {"a": -float("inf")},
    {"a": -1}, {"a": "1"}, {1: 1}, {"": 1}, {"a\0b": 1},
    {"a" * 257: 1}, {"a": 10**1000},
])
def test_invalid_setter_leaves_previous_configuration_unchanged(weights):
    runtime = owner()
    runtime.set_task_weights({"a": 1, "b": 2})
    before = runtime._effective_task_weights("model")
    with pytest.raises(ValueError):
        runtime.set_task_weights(weights)
    assert runtime._effective_task_weights("model") == before


def test_global_defaults_normalize_without_overflow_and_model_override(monkeypatch):
    runtime = owner()
    assert runtime.set_task_weights({"a": 1e308, "b": 1e308}) == {"a": .5, "b": .5}
    monkeypatch.setattr(runtime, "_resolve_aeg_path", lambda model: "existing.aeg")
    monkeypatch.setattr("aether.runtime.runtime.load_aeg_package", lambda path: SimpleNamespace(
        metadata={"task_vectors": {"vectors": [{"name": "a"}, {"name": "b"}]}}
    ))
    assert runtime.set_task_weights("specific", a=3, b=1) == {"a": .75, "b": .25}
    assert runtime._effective_task_weights("specific") == {"a": .75, "b": .25}
    assert runtime._effective_task_weights("other") == {"a": .5, "b": .5}
    assert runtime._effective_task_weights("specific", {"task_weights": {"b": 4}}) == {"b": 4}
    assert runtime._effective_task_weights("specific", {"task_weights": None}) is None
    copied = runtime._effective_task_weights("specific")
    copied["a"] = 100
    assert runtime._effective_task_weights("specific")["a"] == .75


def test_per_model_mapping_accepts_parameter_names_as_data(monkeypatch):
    runtime = owner()
    names = ["weights", "self", "model_id", "task_weights", "ordinary"]
    monkeypatch.setattr(runtime, "_resolve_aeg_path", lambda model: "existing.aeg")
    monkeypatch.setattr("aether.runtime.runtime.load_aeg_package", lambda path: SimpleNamespace(
        metadata={"task_vectors": {"vectors": [{"name": name} for name in names]}}
    ))
    selected = runtime.set_model_task_weights("model", {name: 1e308 for name in names})
    assert selected == {name: .2 for name in names}
    assert runtime._effective_task_weights("model") == selected
    assert runtime.set_model_task_weights("model", {"ordinary": 3}) == runtime.set_task_weights("model", ordinary=3)
    before = runtime._effective_task_weights("model")
    with pytest.raises(ValueError, match="finite"):
        runtime.set_model_task_weights("model", {"weights": float("nan")})
    assert runtime._effective_task_weights("model") == before
    lifecycle = runtime_lifecycle(runtime)
    claimed, errors = lifecycle.begin_close(1)
    assert claimed and not errors
    lifecycle.finish_close(True)
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.set_model_task_weights("model", {"self": 1})


def test_unknown_zero_coordinate_is_rejected_before_model_defaults_change(monkeypatch):
    runtime = owner()
    monkeypatch.setattr(runtime, "_resolve_aeg_path", lambda model: "existing.aeg")
    monkeypatch.setattr("aether.runtime.runtime.load_aeg_package", lambda path: SimpleNamespace(
        metadata={"task_vectors": {"vectors": [{"name": "a"}]}}
    ))
    runtime.set_task_weights("model", a=1)
    with pytest.raises(ValueError, match="unknown"):
        runtime.set_task_weights("model", a=1, absent=0)
    assert runtime._effective_task_weights("model") == {"a": 1}


def test_task_defaults_are_atomic_snapshots_during_concurrent_updates():
    runtime = owner()
    runtime.set_task_weights({"a": 1, "b": 3})
    entered, release = threading.Event(), threading.Event()

    def admitted_request():
        snapshot = runtime._effective_task_weights("model")
        entered.set()
        assert release.wait(5)
        return snapshot

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(admitted_request)
        assert entered.wait(5)
        try:
            runtime.set_task_weights({"a": 3, "b": 1})
        finally:
            release.set()
        assert pending.result(5) == {"a": .25, "b": .75}
    assert runtime._effective_task_weights("model") == {"a": .75, "b": .25}


def test_closed_runtime_rejects_setter_before_mutation():
    runtime = owner()
    runtime.set_task_weights({"a": 1})
    lifecycle = runtime_lifecycle(runtime)
    claimed, errors = lifecycle.begin_close(1)
    assert claimed and not errors
    lifecycle.finish_close(True)
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.set_task_weights({"a": 2, "b": 1})
    assert runtime._task_weights == {"a": 1}


def test_prefix_partitions_never_consume_unbound_or_other_execution_state():
    coordinator = MultiAgentKVCoordinator(max_shared_blocks=4)
    prefix = coordinator.hash_prefix("same tokens")
    legacy = np.asarray([1, 2], dtype=np.float32)
    coordinator.create_agent_session("a", prefix_hash=prefix, prefix_kv=legacy)
    first = {"model": "a", "adapter": "bc", "task": {"x": .5}}
    second = {"model": "ab", "adapter": "c", "task": {"x": .5}}
    assert coordinator.get_shared_kv(prefix)[0] is legacy
    assert coordinator.get_shared_kv(prefix, execution_contract=first) == (None, 0)
    selected = np.asarray([5, 6], dtype=np.float32)
    coordinator.update_shared_kv(prefix, selected, seq_len=2, execution_contract=first)
    assert coordinator.get_shared_kv(prefix, execution_contract=first)[0] is selected
    assert coordinator.get_shared_kv(prefix, execution_contract=second) == (None, 0)
    assert coordinator.get_shared_kv(prefix)[0] is legacy
    assert coordinator._shared_blocks[prefix].ref_count == 1


def test_prefix_variants_are_bounded_lru_and_pins_prevent_retention_overflow():
    coordinator = MultiAgentKVCoordinator(max_shared_blocks=2)
    prefix = coordinator.hash_prefix("same")
    coordinator.create_agent_session("a", prefix_hash=prefix, pin_shared=True)
    contracts = [{"model": str(index)} for index in range(3)]
    for index, contract in enumerate(contracts):
        coordinator.update_shared_kv(prefix, np.asarray([index]), seq_len=1, execution_contract=contract)
    assert coordinator.summary()["execution_variants"] == 2
    assert coordinator.get_shared_kv(prefix, execution_contract=contracts[2]) == (None, 0)
    retained = coordinator.get_shared_kv(prefix, execution_contract=contracts[0])[0]
    coordinator.pin_block(prefix, False)
    coordinator.update_shared_kv(prefix, np.asarray([2]), seq_len=1, execution_contract=contracts[2])
    assert coordinator.summary()["execution_variants"] == 2
    assert coordinator.get_shared_kv(prefix, execution_contract=contracts[1]) == (None, 0)
    np.testing.assert_array_equal(retained, [0])  # Eviction never invalidates held references.


def test_prefix_cleanup_removes_all_bound_tensor_variants():
    coordinator = MultiAgentKVCoordinator(max_shared_blocks=1)
    prefix = coordinator.hash_prefix("first")
    coordinator.create_agent_session("first", prefix_hash=prefix)
    coordinator.update_shared_kv(prefix, np.ones(4), seq_len=4, execution_contract={"model": "a"})
    coordinator.release_session("first")
    coordinator.create_agent_session("second", prefix_hash=coordinator.hash_prefix("second"))
    assert prefix not in coordinator._shared_blocks
    assert coordinator.summary()["execution_variants"] == 0
    assert coordinator.get_shared_kv(prefix, execution_contract={"model": "a"}) == (None, 0)


def test_invalid_prefix_publication_cannot_mutate_previous_state():
    coordinator = MultiAgentKVCoordinator()
    prefix = coordinator.hash_prefix("same")
    coordinator.create_agent_session("a", prefix_hash=prefix, prefix_kv="original")
    with pytest.raises(ValueError):
        coordinator.update_shared_kv(prefix, "replacement", seq_len=-1)
    assert coordinator.get_shared_kv(prefix) == ("original", 0)
    with pytest.raises(ValueError):
        coordinator.update_shared_kv(prefix, "invalid", execution_contract={"value": float("nan")})
    assert coordinator.summary()["execution_variants"] == 0


def frontend_owner(capabilities):
    runtime = owner()
    runtime.config = SimpleNamespace(
        backend_name="frontend", disaggregate_prefill_decode=False, draft_model_id=None,
    )
    runtime._loaded_models = {"model": object()}
    runtime._loaded_backends = {"model": SimpleNamespace(name="frontend", get_capabilities=lambda: capabilities)}
    return runtime


@pytest.mark.parametrize("extras", [
    {"task_weights": {"task": 1}}, {"adapter_id": "demo"}, {"adapter_name": "demo"},
])
def test_raw_frontend_cannot_silently_ignore_requested_weights(extras):
    runtime = frontend_owner(["generate"])
    with pytest.raises(BackendError, match="frontend.*request-state execution contract"):
        runtime._prepare_execution_features(GenerationRequest(model_id="model", prompt="hello", extra=extras))


def test_custom_frontend_explicit_contract_opt_in_preserves_native_adapter_paths():
    runtime = frontend_owner(["generate", "task_reweighting", "lora_adapter_selection"])
    request = GenerationRequest(model_id="model", prompt="hello", extra={"task_weights": {"task": 1}, "adapter_id": "demo"})
    assert runtime._prepare_execution_features(request) is runtime._loaded_models["model"]
    ordinary = frontend_owner(["generate"])
    # This policy covers compiled adapter identifiers; frontend-native paths
    # remain the implementing backend's own API and are never rewritten here.
    request = GenerationRequest(model_id="model", prompt="hello", extra={"adapter_path": "native-adapter"})
    assert ordinary._prepare_execution_features(request) is ordinary._loaded_models["model"]


@pytest.mark.parametrize("capabilities", [None, True, {"task_reweighting": True}, [1]])
def test_malformed_frontend_capability_contract_cannot_upgrade_readiness(capabilities):
    runtime = frontend_owner(capabilities)
    with pytest.raises(BackendError, match="invalid request-state capability"):
        runtime._prepare_execution_features(GenerationRequest(model_id="model", extra={"task_weights": {"task": 1}}))


def test_compiled_task_catalog_and_engine_hook_are_both_required():
    from pathlib import Path

    runtime = frontend_owner(["generate"])
    handle = CompiledAEGHandle("model", Path("model.aeg"), {"architecture": {}}, engine=object())
    runtime._loaded_models["model"] = handle
    request = GenerationRequest(model_id="model", extra={"task_weights": {"task": 1}})
    with pytest.raises(BackendError, match="unknown vector"):
        runtime._prepare_execution_features(request)
    handle.metadata["task_vectors"] = {"vectors": [{"name": "task"}]}
    with pytest.raises(BackendError, match="no task-vector application contract"):
        runtime._prepare_execution_features(request)


def test_conflicting_adapter_aliases_fail_before_frontend_dispatch():
    runtime = frontend_owner(["lora_adapter_selection"])
    with pytest.raises(BackendError, match="different adapters"):
        runtime._prepare_execution_features(GenerationRequest(
            model_id="model", extra={"adapter_id": "one", "adapter_name": "two"},
        ))


def test_sdk_forwards_task_mapping_without_keyword_expansion(monkeypatch):
    from aether.sdk import AetherClient

    client = AetherClient("model.aeg", enable_safety=False)
    calls = []
    monkeypatch.setattr(client, "_ensure_loaded", lambda: None)
    client._runtime = SimpleNamespace(set_model_task_weights=lambda model, weights: calls.append((model, dict(weights))) or dict(weights))
    try:
        assert client.set_task_weights(weights=2) == {"weights": 2}
        assert client.set_task_weights_mapping({"self": 1, "weights": 2, "model_id": 3}) == {"self": 1, "weights": 2, "model_id": 3}
        assert calls == [("model.aeg", {"weights": 2}), ("model.aeg", {"self": 1, "weights": 2, "model_id": 3})]
    finally:
        client._runtime = None
