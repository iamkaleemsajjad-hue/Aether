"""Actual static weight changes must never reuse another execution's cache."""
import json
from dataclasses import fields

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.core.aeg_format import AEGPackage
from aether.runtime.session_contract import validate_task_weights
from aether.runtime.spec_decode import numpy_logits


def _config(tmp_path, backend):
    return RuntimeConfig(backend_name=backend, execution_devices=["cpu"],
                         speculative_decoding=False, enable_semantic_cache=False,
                         dynamic_precision=False, model_cache_dir=str(tmp_path / "runtime-cache"),
                         kv_cache_path=str(tmp_path / "stored"))


@pytest.fixture(scope="module")
def adapted_artifact(tmp_path_factory):
    """Compile real LoRA K/V and publish actual task vectors through Runtime.merge."""
    from safetensors.numpy import save_file
    root = tmp_path_factory.mktemp("weight-contract-artifact")
    source = root / "checkpoint"
    _write_probe_checkpoint(source)
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8")
    rng = np.random.default_rng(18)
    tensors = {}
    for projection in ("k_proj", "v_proj"):
        prefix = f"base_model.model.model.layers.0.self_attn.{projection}"
        tensors[prefix + ".lora_A.weight"] = rng.normal(0, .7, (2, 16)).astype(np.float32)
        tensors[prefix + ".lora_B.weight"] = rng.normal(0, .7, (8, 2)).astype(np.float32)
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    config = CompilerConfig(targets=["cpu_generic"], precision="FP32", optimization_level=0,
                            cache_dir=str(root / "compiler-cache"), peft_adapter_paths=[str(adapter)])
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name in {"enable_precision_assignment", "enable_advanced_peft"})
    base = Compiler(config).compile(str(source), output_path=root / "base.aeg")
    weights = base.weight_store().dequantize_all()
    task = root / "task.safetensors"
    save_file({"layer_0_v_proj": weights["layer_0_v_proj"] + rng.normal(0, .5, weights["layer_0_v_proj"].shape).astype(np.float32),
               "lm_head": -weights["lm_head"]}, str(task))
    head = root / "head.safetensors"
    save_file({"lm_head": -weights["lm_head"]}, str(head))
    with Runtime(_config(root, "aether_cpu")) as runtime:
        merged = runtime.merge(str(base.root), [
            {"name": "changed", "path": str(task), "coefficient": .1},
            {"name": "head", "path": str(head), "coefficient": .1},
        ], method="task_arithmetic")
    package = AEGPackage(merged["output_model"]).load()
    package.verify_integrity()
    return str(package.root)


@pytest.fixture(params=["aether_cpu", "pytorch"], ids=["cpu", "torch"])
def backend(request, monkeypatch):
    if request.param == "pytorch":
        pytest.importorskip("torch")
        monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
        monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    return request.param


def _selection(mode, coefficient=1):
    return {"adapter_id": "adapter"} if mode == "adapter" else {"task_weights": {"changed": coefficient}}


@pytest.mark.parametrize("mode", ["adapter", "task"])
def test_weight_switch_recomputes_prefix_and_same_contract_reuses(adapted_artifact, backend, mode, tmp_path):
    with Runtime(_config(tmp_path, backend)) as runtime:
        model = adapted_artifact
        runtime.generate(model, "hello world", max_tokens=0, temperature=0, aether_kv_session_id="mixed")
        selected = _selection(mode)
        runtime.generate(model, "hello world hello", max_tokens=0, temperature=0,
                         aether_kv_session_id="fresh", **selected)
        changed = runtime.generate(model, "hello world hello", max_tokens=0, temperature=0,
                                   aether_kv_session_id="mixed", **selected)
        handle = runtime._loaded_models[model]
        assert changed.metrics.extra["kv_reused_tokens"] == 0
        np.testing.assert_array_equal(numpy_logits(handle.session_caches["mixed"][1].last_logits),
                                      numpy_logits(handle.session_caches["fresh"][1].last_logits))
        expected = runtime.generate(model, "hello world hello", max_tokens=4, temperature=0,
                                    aether_kv_session_id="fresh", **selected)
        actual = runtime.generate(model, "hello world hello", max_tokens=4, temperature=0,
                                  aether_kv_session_id="mixed", **selected)
        assert actual.text == expected.text
        assert actual.metrics.extra["kv_reused_tokens"] == 3


@pytest.mark.parametrize("mode", ["adapter", "task"])
def test_adapted_persistence_restart_binds_explicit_request_contract(adapted_artifact, backend, mode, tmp_path):
    from tests.integration.test_speculative_session_persistence import _snapshot
    config = _config(tmp_path, backend)
    selected = _selection(mode)
    with Runtime(config) as runtime:
        model = adapted_artifact
        runtime.generate(model, "hello world hello", max_tokens=0, temperature=0,
                         aether_kv_session_id="adapted", **selected)
        handle = runtime._loaded_models[model]
        before = _snapshot(handle.session_caches["adapted"][1], handle).serialize()
        expected = runtime.generate(model, "hello world hello", max_tokens=4, temperature=0, **selected)
        saved = runtime.persist_session(model, "adapted")
        assert saved["execution_contract"]["adapter_id"] == selected.get("adapter_id")
        assert saved["execution_contract"]["task_weights"] == selected.get("task_weights", {})
    with Runtime(config) as restarted:
        restarted.restore_session(model, "adapted", saved["key"])
        handle = restarted._loaded_models[model]
        assert _snapshot(handle.session_caches["adapted"][1], handle).serialize() == before
        actual = restarted.generate(model, "hello world hello", max_tokens=4, temperature=0,
                                    aether_kv_session_id="adapted", **selected)
        assert actual.text == expected.text
        assert actual.metrics.extra["kv_reused_tokens"] == 3


@pytest.mark.parametrize("mode", ["adapter", "task"])
def test_restored_contract_never_silently_activates_saved_weights(adapted_artifact, backend, mode, tmp_path):
    config = _config(tmp_path, backend)
    selected = _selection(mode)
    next_selection = {} if mode == "adapter" else _selection(mode, .5)
    with Runtime(config) as runtime:
        model = adapted_artifact
        runtime.generate(model, "hello world hello", max_tokens=0, temperature=0,
                         aether_kv_session_id="adapted", **selected)
        saved = runtime.persist_session(model, "adapted")
        expected = runtime.generate(model, "hello world hello", max_tokens=4, temperature=0, **next_selection)
    with Runtime(config) as restarted:
        restarted.restore_session(model, "adapted", saved["key"])
        actual = restarted.generate(model, "hello world hello", max_tokens=4, temperature=0,
                                    aether_kv_session_id="adapted", **next_selection)
        assert actual.text == expected.text
        assert actual.metrics.extra["kv_reused_tokens"] == 0


@pytest.mark.parametrize("mode", ["adapter", "task"])
def test_streaming_weight_switch_matches_fresh_adapted_execution(adapted_artifact, backend, mode, tmp_path):
    with Runtime(_config(tmp_path, backend)) as runtime:
        model = adapted_artifact
        selected = _selection(mode)
        runtime.generate(model, "hello world", max_tokens=0, temperature=0, aether_kv_session_id="mixed")
        expected = runtime.generate(model, "hello world hello", max_tokens=4, temperature=0, **selected)
        actual = "".join(runtime.generate_stream(model, "hello world hello", max_tokens=4,
                                                  temperature=0, aether_kv_session_id="mixed", **selected))
        assert actual == expected.text
        handle = runtime._loaded_models[model]
        assert handle.session_contracts["mixed"]["adapter_id"] == selected.get("adapter_id")
        assert handle.session_contracts["mixed"]["task_weights"] == selected.get("task_weights", {})


def test_head_only_task_switch_invalidates_cached_last_logits(adapted_artifact, backend, tmp_path):
    with Runtime(_config(tmp_path, backend)) as runtime:
        model = adapted_artifact
        runtime.generate(model, "hello world", max_tokens=0, temperature=0, aether_kv_session_id="mixed")
        fresh = runtime.generate(model, "hello world", max_tokens=4, temperature=0, task_weights={"head": 1})
        changed = runtime.generate(model, "hello world", max_tokens=4, temperature=0,
                                   aether_kv_session_id="mixed", task_weights={"head": 1})
        assert changed.metrics.extra["kv_reused_tokens"] == 0
        assert changed.text == fresh.text


def test_unknown_legacy_cache_contract_recomputes_and_side_maps_clear(adapted_artifact, backend, tmp_path):
    with Runtime(_config(tmp_path, backend)) as runtime:
        model = adapted_artifact
        runtime.generate(model, "hello world", max_tokens=0, temperature=0, aether_kv_session_id="legacy")
        handle = runtime._loaded_models[model]
        handle.session_contracts.pop("legacy")
        actual = runtime.generate(model, "hello world", max_tokens=2, temperature=0, aether_kv_session_id="legacy")
        assert actual.metrics.extra["kv_reused_tokens"] == 0
        handle.clear_session_caches()
        assert not handle.session_caches and not handle.session_contracts and not handle.session_target_ids


@pytest.mark.parametrize("weights", [False, {"head": True}, {"head": float("nan")},
                                     {"head": float("inf")}, {"head": -1}, {"missing": 0}])
def test_finite_task_validation_checks_even_zero_coordinates(weights):
    with pytest.raises(ValueError):
        validate_task_weights(weights, available={"head"})


def test_task_normalization_is_finite_order_independent_and_overflow_safe():
    assert validate_task_weights({"b": 1e308, "a": 1e308}, normalize=True) == {"a": .5, "b": .5}
    assert validate_task_weights({"a": 0, "b": 2}) == {"b": 2}


@pytest.mark.parametrize("mode", ["adapter", "task"])
def test_direct_backend_batch_refuses_unsupported_active_weight_hooks(
    adapted_artifact, backend, mode, tmp_path
):
    from aether.backends.base import GenerationRequest
    from aether.core.exceptions import BackendError

    with Runtime(_config(tmp_path, backend)) as runtime:
        model = adapted_artifact
        runtime._load_model(model)
        selected_backend = runtime._loaded_backends[model]
        requests = [GenerationRequest(model_id=model, prompt="hello world", max_tokens=1,
                                      temperature=0, extra=_selection(mode)) for _ in range(2)]
        with pytest.raises(BackendError, match="active.*contract|unsupported.*hook"):
            selected_backend.generate_batch(requests)
