"""DirectML integration contracts and separately gated physical workload tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from aether.backends.onnx_backend import ONNXBackend, validate_directml_execution
from aether.core.exceptions import BackendError


def ort_fixture(monkeypatch, *, registered=None, configured=None, profile_path=None):
    registered = (
        registered if registered is not None else ["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    options = SimpleNamespace(
        enable_mem_pattern=True,
        execution_mode="parallel",
        add_session_config_entry=Mock(),
    )
    session = SimpleNamespace(
        get_providers=Mock(return_value=configured if configured is not None else registered),
        disable_fallback=Mock(),
        end_profiling=Mock(return_value=str(profile_path) if profile_path else ""),
    )
    module = SimpleNamespace(
        get_available_providers=Mock(return_value=registered),
        SessionOptions=Mock(return_value=options),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=Mock(return_value=session),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return module, options, session


def test_explicit_directml_configures_required_options_without_profiling_overhead(monkeypatch):
    module, options, session = ort_fixture(monkeypatch)
    backend = ONNXBackend()
    assert backend.load_model("model.onnx", providers=["DmlExecutionProvider"]) is session
    assert options.enable_mem_pattern is False
    assert options.execution_mode == "sequential"
    options.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
    session.disable_fallback.assert_called_once_with()
    assert options.enable_profiling is False
    module.InferenceSession.assert_called_once()


def test_onnx_target_readiness_requires_matching_provider_and_initialized_directml_session(
    monkeypatch,
):
    _, _, _ = ort_fixture(monkeypatch)
    monkeypatch.setattr(
        "aether.backends.hardware_inventory.probe_gpu_inventory", lambda: {"devices": []}
    )
    backend = ONNXBackend()
    assert backend.available_for_target("cpu_generic")
    assert backend.available_for_target("cpu_avx2")
    assert not backend.available_for_target("directml")
    assert not backend.available_for_target("openvino_npu")
    backend.load_model("diagnostic.onnx", providers=["DmlExecutionProvider"])
    assert backend.available_for_target("directml")
    backend.unload_model("diagnostic.onnx")
    assert not backend.available_for_target("directml")


def test_directml_registered_provider_and_observed_device_are_loading_candidate_only(monkeypatch):
    ort_fixture(monkeypatch)
    monkeypatch.setattr(
        "aether.backends.hardware_inventory.probe_gpu_inventory",
        lambda: {"devices": [{"device": "observed"}]},
    )
    backend = ONNXBackend()
    assert backend.available_for_target("directml")
    assert not backend._models


def test_directml_requested_but_not_registered_fails_before_session_creation(monkeypatch):
    module, _, _ = ort_fixture(monkeypatch, registered=["CPUExecutionProvider"])
    backend = ONNXBackend()
    with pytest.raises(BackendError, match="not registered"):
        backend.load_model("model.onnx", providers=["DmlExecutionProvider"])
    module.InferenceSession.assert_not_called()
    assert not backend._models


def test_directml_session_creation_cpu_fallback_is_rejected(monkeypatch):
    _, _, _ = ort_fixture(monkeypatch, configured=["CPUExecutionProvider"])
    backend = ONNXBackend()
    with pytest.raises(BackendError, match="fell back"):
        backend.load_model("model.onnx", providers=["DmlExecutionProvider"])
    assert not backend._models


def test_explicit_mixed_provider_fallback_remains_opt_in_and_is_not_validation(monkeypatch):
    _, options, session = ort_fixture(monkeypatch)
    backend = ONNXBackend()
    backend.load_model(
        "model.onnx",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        allow_provider_fallback=True,
    )
    assert options.enable_mem_pattern is False
    options.add_session_config_entry.assert_not_called()
    session.disable_fallback.assert_not_called()
    with pytest.raises(BackendError, match="explicitly profiled"):
        backend.provider_execution_evidence("model.onnx", "DmlExecutionProvider")


def test_cached_cpu_model_cannot_silently_satisfy_a_later_directml_request(monkeypatch):
    module, _, _ = ort_fixture(monkeypatch)
    backend = ONNXBackend()
    backend.load_model("model.onnx", providers=["CPUExecutionProvider"])
    with pytest.raises(BackendError, match="different providers"):
        backend.load_model("model.onnx", providers=["DmlExecutionProvider"])
    assert module.InferenceSession.call_count == 1
    backend.unload_model("model.onnx")
    backend.load_model("model.onnx", providers=["DmlExecutionProvider"])
    assert module.InferenceSession.call_count == 2


@pytest.mark.parametrize(
    "provider", ["CUDAExecutionProvider", "ROCMExecutionProvider", "OpenVINOExecutionProvider"]
)
def test_explicit_accelerator_providers_forbid_cpu_fallback(monkeypatch, provider):
    _, options, session = ort_fixture(monkeypatch, registered=[provider, "CPUExecutionProvider"])
    ONNXBackend().load_model("model.onnx", providers=[provider])
    options.add_session_config_entry.assert_called_once_with("session.disable_cpu_ep_fallback", "1")
    session.disable_fallback.assert_called_once()


def test_cached_provider_cannot_ignore_changed_fallback_policy(monkeypatch):
    ort_fixture(monkeypatch)
    backend = ONNXBackend()
    backend.load_model(
        "model.onnx", providers=["CPUExecutionProvider"], allow_provider_fallback=True
    )
    with pytest.raises(BackendError, match="different fallback policy"):
        backend.load_model(
            "model.onnx", providers=["CPUExecutionProvider"], allow_provider_fallback=False
        )


def test_onnx_does_not_implicitly_load_aeg(monkeypatch):
    module, _, _ = ort_fixture(monkeypatch)
    with pytest.raises(BackendError, match="AEG export"):
        ONNXBackend().load_model("logical-model", aeg_path="artifact.aeg")
    module.InferenceSession.assert_not_called()


@pytest.mark.parametrize(
    "providers", [[], "DmlExecutionProvider", [5], ["MissingExecutionProvider"]]
)
def test_invalid_provider_contract_is_not_replaced_with_cpu(monkeypatch, providers):
    module, _, _ = ort_fixture(monkeypatch)
    with pytest.raises(BackendError):
        ONNXBackend().load_model("model.onnx", providers=providers)
    module.InferenceSession.assert_not_called()


def test_invalid_tokenizer_options_do_not_leave_a_half_loaded_model(monkeypatch):
    module, _, _ = ort_fixture(monkeypatch)
    backend = ONNXBackend()
    with pytest.raises(ValueError, match="tokenizer or tokenizer_path"):
        backend.load_model("model.onnx", tokenizer=object(), tokenizer_path="tokenizer.json")
    module.InferenceSession.assert_not_called()
    assert not backend._models


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"cat": "Session", "name": "model_loading", "args": {}}],
        [
            {
                "cat": "Node",
                "name": "MatMul_kernel_time",
                "args": {"provider": "CPUExecutionProvider"},
            }
        ],
        [
            {
                "cat": "Node",
                "name": "MatMul_kernel_time",
                "args": {"provider": "DmlExecutionProvider"},
            },
            {
                "cat": "Node",
                "name": "Add_kernel_time",
                "args": {"provider": "CPUExecutionProvider"},
            },
        ],
    ],
)
def test_provider_listing_cannot_replace_real_node_attribution(monkeypatch, tmp_path, events):
    profile = tmp_path / "diagnostic-profile.json"
    profile.write_text(json.dumps(events), encoding="utf-8")
    _, _, session = ort_fixture(monkeypatch, profile_path=profile)
    backend = ONNXBackend()
    backend.load_model("diagnostic.onnx", providers=["DmlExecutionProvider"], enable_profiling=True)
    # These mocked diagnostic events deliberately fail. They are not a
    # physical execution result and cannot promote hardware validation state.
    with pytest.raises(BackendError, match="unconfirmed|outside"):
        backend.provider_execution_evidence("diagnostic.onnx", "DmlExecutionProvider")
    session.end_profiling.assert_called_once()


def test_directml_validation_missing_provider_does_not_execute_a_cpu_probe(monkeypatch):
    module, _, _ = ort_fixture(monkeypatch, registered=["CPUExecutionProvider"])
    with pytest.raises(BackendError, match="not registered"):
        validate_directml_execution()
    module.InferenceSession.assert_not_called()


@pytest.mark.parametrize("index", [-1, 1.5, True, "0"])
def test_directml_validation_rejects_invalid_adapter_indices(index):
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_directml_execution(index)


def test_real_cpu_onnx_profiling_verifies_the_executing_provider(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import helper, numpy_helper

    values = np.asarray([[1.0, 2.0]], dtype=np.float32)
    weights = np.asarray([[2.0], [-3.0]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "cpu_provider_validation",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1])],
        initializer=[numpy_helper.from_array(weights, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "cpu-matmul.onnx"
    onnx.save(model, str(path))
    backend = ONNXBackend()
    session = backend.load_model(
        str(path),
        providers=["CPUExecutionProvider"],
        enable_profiling=True,
        profile_file_prefix=str(tmp_path / "cpu-profile"),
    )
    np.testing.assert_allclose(session.run(["output"], {"input": values})[0], values @ weights)
    evidence = backend.provider_execution_evidence(str(path), "CPUExecutionProvider")
    assert evidence["execution_provider"] == "CPUExecutionProvider"
    assert evidence["executed_node_count"] >= 1


def test_real_independent_onnx_profiles_keep_unique_session_files(tmp_path):
    """Real sessions sharing a user prefix retain separate parseable execution evidence."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import helper, numpy_helper

    values = np.eye(2, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="real_matmul")],
        "independent_profile_validation",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])],
        initializer=[numpy_helper.from_array(values, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    backend = ONNXBackend()
    paths = [tmp_path / "first.onnx", tmp_path / "second.onnx"]
    sessions = []
    try:
        for path in paths:
            onnx.save(model, str(path))
            session = backend.load_model(
                str(path),
                providers=["CPUExecutionProvider"],
                allow_provider_fallback=False,
                enable_profiling=True,
                profile_file_prefix=str(tmp_path / "shared-prefix"),
            )
            np.testing.assert_array_equal(session.run(None, {"input": values})[0], values)
            sessions.append(session)
        profile_paths = [session.end_profiling() for session in sessions]
        assert len(set(profile_paths)) == len(sessions)
        for path in profile_paths:
            events = json.loads(Path(path).read_text(encoding="utf-8"))
            providers = {
                event["args"]["provider"]
                for event in events
                if event.get("cat") == "Node" and event.get("args", {}).get("provider")
            }
            assert providers == {"CPUExecutionProvider"}
    finally:
        for path in paths:
            backend.unload_model(str(path))


@pytest.mark.gpu
def test_real_directml_offline_workload_requires_gpu_node_execution():
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("physical DirectML runtime is not installed on this host")
    evidence = validate_directml_execution()
    assert evidence["execution_scope"] == "onnxruntime_directml_inference"
    assert evidence["execution_provider"] == "DmlExecutionProvider"
    assert evidence["executed_node_count"] >= 1
    assert evidence["maximum_absolute_error"] <= 1e-5


@pytest.mark.gpu
def test_real_directml_runtime_routes_generation_and_stream_to_gpu(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("physical DirectML runtime is not installed on this host")
    from onnx import helper, numpy_helper

    from aether import Runtime, RuntimeConfig

    monkeypatch.delenv("AETHER_EXECUTION_DEVICES", raising=False)
    weight = np.asarray([[0, 1, -1, 0]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input_ids"], ["floats"], to=onnx.TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["floats"], ["sum"], axes=[1], keepdims=1),
            helper.make_node("MatMul", ["sum", "weight"], ["logits"]),
        ],
        "input_dependent_generation",
        [helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [1, "sequence"])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 4])],
        initializer=[numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    path = tmp_path / "dependent.onnx"
    onnx.save(model, path)

    class Tokenizer:
        eos_token_id = None

        def encode(self, text, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "".join("A" if int(value) == 1 else "?" for value in ids)

    runtime = Runtime(
        RuntimeConfig(
            backend_name="onnx",
            hf_offline=True,
            speculative_decoding=False,
            model_cache_dir=str(tmp_path / "cache"),
        )
    )
    backend = runtime.backend_registry.get_backend("onnx")
    session = backend.load_model(
        str(path),
        providers=[("DmlExecutionProvider", {"device_id": 0})],
        allow_provider_fallback=False,
        tokenizer=Tokenizer(),
        enable_profiling=True,
        profile_file_prefix=str(tmp_path / "profile"),
    )
    values = np.asarray([[1, 2]], dtype=np.int32)
    np.testing.assert_array_equal(
        session.run(["logits"], {"input_ids": values})[0],
        values.astype(np.float32).sum(axis=1, keepdims=True) @ weight,
    )
    result = runtime.generate(str(path), "offline", max_tokens=2, temperature=0)
    assert result.text == "AA" and result.usage["completion_tokens"] == 2
    assert result.metrics.kernel_target == "directml"
    assert (
        "".join(runtime.generate_stream(str(path), "offline", max_tokens=2, temperature=0)) == "AA"
    )
    evidence = backend.provider_execution_evidence(str(path), "DmlExecutionProvider")
    assert evidence["executed_node_count"] >= 9
    assert all(item["provider"] == "DmlExecutionProvider" for item in evidence["executed_nodes"])
    assert not any(
        item["execution_tested"]
        for item in runtime.hardware()["capabilities"]
        if item["target_id"] == "directml"
    )
