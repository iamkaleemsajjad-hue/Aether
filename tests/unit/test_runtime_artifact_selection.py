"""Artifact contracts determine execution, including on mixed hardware hosts."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.core.aeg_format import AEGPackage
from aether.core.exceptions import BackendError, BackendNotAvailableError
from aether.runtime.config import RuntimeConfig
from aether.runtime.runtime import Runtime


def _runtime(tmp_path, host_target, *, backend_name=None, execution_devices=None):
    """Inspect mocked dispatch for deliberately uncertified manual artifacts."""
    runtime = object.__new__(Runtime)
    runtime.config = RuntimeConfig(
        backend_name=backend_name,
        execution_devices=execution_devices,
        model_cache_dir=str(tmp_path),
        hf_offline=True,
        allow_uncertified=True,
    )
    runtime.fingerprint = SimpleNamespace(
        target_id=host_target,
        attributes={"available_targets": [host_target, "cpu_avx2"]},
        capabilities=[],
    )
    runtime._lock = threading.RLock()
    runtime._loaded_models = {}
    runtime._loaded_backends = {}
    runtime._model_targets = {}
    runtime._aeg_paths = {}
    runtime._init_v4_layers = lambda _: None
    runtime._init_v5_layers = lambda _: None
    native = Mock(name="native_cpu")
    native.name = "aether_cpu"
    native.is_available.return_value = True
    native.available_for_target.side_effect = lambda target: target.startswith("cpu_")
    runtime.backend_registry = Mock()
    runtime.backend_registry.get_backend.side_effect = lambda name: (
        native if name == "aether_cpu" else None
    )
    runtime.backend_registry.get_available_backends.return_value = [native]
    return runtime, native


def _package(tmp_path, targets, *, portable=True, status=None):
    package = AEGPackage.create(
        tmp_path / "model.aeg",
        model_id="any-model-family",
        aether_version="1.3.0",
    )
    package.manifest.kernels.targets = targets
    package.manifest.kernels.variant_status = (
        status if status is not None else {target: "executable" for target in targets}
    )
    package.manifest.kernels.portable_backends = ["aether_cpu", "pytorch"] if portable else []
    package.save()
    return package


@pytest.mark.parametrize(
    "host_target",
    [
        "cuda_sm80",
        "cuda_sm90",
        "rocm_cdna3",
        "amd_mi350x",
        "rocm_cdna5_mi455x",
        "metal",
        "metal_m3",
        "openvino_npu",
    ],
)
def test_cpu_artifact_loads_on_a_host_with_an_accelerator(tmp_path, host_target):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, host_target)
    runtime._resolve_aeg_path = lambda _: str(package.root)
    assert runtime._load_model("model") is native
    native.load_model.assert_called_once()
    assert native.load_model.call_args.kwargs["execution_devices"] == ["cpu"]
    assert runtime._model_targets["model"] == "cpu_avx2"
    assert runtime.fingerprint.target_id == host_target


def test_explicit_native_cpu_backend_works_on_a_gpu_host(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, "cuda_sm80", backend_name="aether_cpu")
    runtime._resolve_aeg_path = lambda _: str(package.root)
    assert runtime._load_model("model") is native


@pytest.mark.parametrize("host_target", ["cuda_sm80", "rocm_cdna3", "amd_mi350x", "rocm_cdna5_mi455x", "metal", "metal_m3"])
def test_multi_target_artifact_prefers_its_available_accelerator_variant(tmp_path, host_target):
    package = _package(
        tmp_path,
        ["cpu_avx2", host_target],
        status={
            "cpu_avx2": "executable",
            host_target: "portable",
        },
    )
    runtime, native = _runtime(tmp_path, host_target)
    accelerator = Mock()
    accelerator.name = "pytorch"
    accelerator.available_for_target.side_effect = lambda target: target == host_target
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native,
        "pytorch": accelerator,
    }.get(name)
    backend, target, _ = runtime._resolve_aeg_backend(package, "model")
    assert backend is accelerator
    assert target == host_target


def test_multi_target_artifact_uses_cpu_when_accelerator_backend_is_absent(tmp_path):
    package = _package(
        tmp_path,
        ["cpu_avx2", "cuda_sm80"],
        status={
            "cpu_avx2": "executable",
            "cuda_sm80": "portable",
        },
    )
    runtime, native = _runtime(tmp_path, "cuda_sm80")
    backend, target, devices = runtime._resolve_aeg_backend(package, "model")
    assert backend is native
    assert target == "cpu_avx2"
    assert devices == ["cpu"]


def test_explicit_accelerator_mesh_never_falls_back_to_cpu(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, _ = _runtime(tmp_path, "cuda_sm80", execution_devices=["cuda:0"])
    with pytest.raises(BackendNotAvailableError, match="pytorch.*cuda_sm80"):
        runtime._resolve_aeg_backend(package, "model")


def test_explicit_cpu_mesh_overrides_accelerator_preference(tmp_path):
    package = _package(tmp_path, ["cpu_avx2", "cuda_sm80"])
    runtime, native = _runtime(tmp_path, "cuda_sm80", execution_devices=["cpu"])
    backend, target, devices = runtime._resolve_aeg_backend(package, "model")
    assert (backend, target, devices) == (native, "cpu_avx2", ["cpu"])


def test_environment_mesh_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_EXECUTION_DEVICES", "cpu")
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, "cuda_sm80")
    assert runtime._resolve_aeg_backend(package, "model") == (native, "cpu_avx2", ["cpu"])


def test_unavailable_explicit_device_is_reported(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, _ = _runtime(tmp_path, "cpu_avx2", execution_devices=["mps"])
    with pytest.raises(BackendNotAvailableError, match="Requested execution devices"):
        runtime._resolve_aeg_backend(package, "model")


def test_plan_only_artifact_without_portable_contract_is_rejected(tmp_path):
    package = _package(tmp_path, ["cuda_sm80"], portable=False, status={"cuda_sm80": "plan_only"})
    runtime, _ = _runtime(tmp_path, "cuda_sm80")
    with pytest.raises(BackendNotAvailableError, match="no executable variant"):
        runtime._resolve_aeg_backend(package, "model")


def test_accelerator_artifact_with_portable_cpu_contract_can_run_on_cpu(tmp_path):
    package = _package(tmp_path, ["cuda_sm80"], status={"cuda_sm80": "portable"})
    runtime, native = _runtime(tmp_path, "cpu_avx2")
    assert runtime._resolve_aeg_backend(package, "model") == (native, "cpu_avx2", ["cpu"])


@pytest.mark.parametrize(
    "artifact_target,host_target",
    [
        ("cpu_avx512", "cpu_avx2"),
        ("cpu_avx2", "cpu_neon"),
        ("cpu_neon", "cpu_avx512"),
    ],
)
def test_cpu_reference_contract_is_portable_across_cpu_variants(
    tmp_path, artifact_target, host_target
):
    package = _package(tmp_path, [artifact_target])
    runtime, native = _runtime(tmp_path, host_target)
    backend, target, _ = runtime._resolve_aeg_backend(package, "model")
    assert backend is native
    assert target == host_target


@pytest.mark.parametrize("target", ["cuda_sm89", "rocm_cdna3", "amd_mi350x", "metal", "metal_m3"])
def test_authorized_portable_artifact_prefers_a_ready_accelerator(tmp_path, target):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, target)
    accelerator = Mock()
    accelerator.name = "pytorch"
    accelerator.available_for_target.side_effect = lambda requested: requested == target
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native,
        "pytorch": accelerator,
    }.get(name)
    backend, selected, _ = runtime._resolve_aeg_backend(package, "model")
    assert backend is accelerator
    assert selected == target


def test_cpu_only_artifact_remains_cpu_even_when_an_accelerator_is_ready(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"], portable=False)
    runtime, native = _runtime(tmp_path, "cuda_sm89")
    accelerator = Mock()
    accelerator.name = "pytorch"
    accelerator.available_for_target.return_value = True
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native, "pytorch": accelerator,
    }.get(name)
    assert runtime._resolve_aeg_backend(package, "model") == (native, "cpu_avx2", ["cpu"])


def test_explicit_cpu_backend_overrides_ready_portable_accelerator(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, "cuda_sm89", backend_name="aether_cpu")
    assert runtime._resolve_aeg_backend(package, "model") == (native, "cpu_avx2", ["cpu"])


def test_explicit_cpu_mesh_normalizes_cpu_zero(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, "cuda_sm89", execution_devices=["cpu:0"])
    assert runtime._resolve_aeg_backend(package, "model") == (native, "cpu_avx2", ["cpu"])


def test_requested_gpu_ordinal_selects_its_actual_target(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, "cuda_sm89", execution_devices=["cuda:1"])
    runtime.fingerprint.attributes["available_targets"] = ["cuda_sm89", "cuda_sm80", "cpu_avx2"]
    runtime.fingerprint.capabilities = [
        {"target_id": target, "available": True, "extra": {"execution_device": f"cuda:{index}"}}
        for index, target in enumerate(["cuda_sm89", "cuda_sm80"])
    ]
    accelerator = Mock()
    accelerator.name = "pytorch"
    accelerator.available_for_target.return_value = True
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native, "pytorch": accelerator,
    }.get(name)
    assert runtime._resolve_aeg_backend(package, "model") == (accelerator, "cuda_sm80", ["cuda:1"])


def test_requested_missing_ordinal_fails_before_model_load(tmp_path):
    runtime, _ = _runtime(tmp_path, "cuda_sm89", execution_devices=["cuda:2"])
    runtime.fingerprint.capabilities = [
        {"target_id": "cuda_sm89", "available": True, "extra": {"execution_device": "cuda:0"}},
    ]
    with pytest.raises(BackendNotAvailableError, match="cuda:2.*unavailable"):
        runtime._execution_candidates()


@pytest.mark.parametrize("devices", [["cuda"], ["cuda:-1"], ["cpu", "cpu:0"], ["xpu:0"]])
def test_invalid_device_constraints_cannot_be_ignored(tmp_path, devices):
    runtime, _ = _runtime(tmp_path, "cuda_sm89", execution_devices=devices)
    with pytest.raises(BackendNotAvailableError, match="Invalid execution devices"):
        runtime._execution_candidates()


def test_raw_model_cpu_fallback_records_the_execution_target(tmp_path):
    runtime, native = _runtime(tmp_path, "cuda_sm89")
    runtime._resolve_aeg_path = lambda _: None
    assert runtime._load_model("organization/real-model") is native
    assert runtime._model_targets["organization/real-model"] == "cpu_avx2"
    assert native.load_model.call_args.kwargs["execution_devices"] == ["cpu"]


def test_raw_model_explicit_gpu_does_not_fall_back_to_cpu(tmp_path):
    runtime, _ = _runtime(tmp_path, "cuda_sm89", execution_devices=["cuda:0"])
    with pytest.raises(BackendNotAvailableError, match="No compatible model execution backend"):
        runtime._resolve_model_backend("organization/real-model")


def test_raw_model_local_source_is_not_mistaken_for_an_aeg(tmp_path):
    source = tmp_path / "raw-model"
    source.mkdir()
    (source / "config.json").write_text('{}', encoding="utf-8")
    runtime, native = _runtime(tmp_path, "cpu_avx2")
    runtime._resolve_aeg_path = lambda _: None
    assert runtime._load_model(str(source)) is native


def test_model_format_applicability_is_checked_after_target_readiness(tmp_path):
    runtime, native = _runtime(tmp_path, "cuda_sm89")
    unavailable_frontend = Mock()
    unavailable_frontend.name = "vllm"
    unavailable_frontend.available_for_target.return_value = True
    unavailable_frontend.available_for_model.return_value = False
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native, "vllm": unavailable_frontend,
    }.get(name)
    runtime.backend_registry.get_available_backends.return_value = [unavailable_frontend, native]
    assert runtime._resolve_model_backend("organization/real-model") == (native, "cpu_avx2", ["cpu"])


def _onnx_runtime(tmp_path, target, providers, *, devices=None, backend_name=None):
    runtime, _ = _runtime(tmp_path, target, execution_devices=devices, backend_name=backend_name)
    runtime._resolve_aeg_path = lambda _: None
    backend = Mock()
    backend.name = "onnx"
    backend.is_available.return_value = True
    backend.available_providers.return_value = providers
    runtime.backend_registry.get_backend.side_effect = lambda name: backend if name == "onnx" else None
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"dispatch contract fixture, not an executable ONNX model")
    return runtime, backend, str(model_path)


def test_raw_onnx_cpu_request_cannot_be_overridden_by_gpu_provider_priority(tmp_path):
    runtime, backend, model_id = _onnx_runtime(
        tmp_path, "cuda_sm89", ["CUDAExecutionProvider", "CPUExecutionProvider"], devices=["cpu"],
    )
    assert runtime._load_model(model_id) is backend
    assert backend.load_model.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert backend.load_model.call_args.kwargs["allow_provider_fallback"] is False
    assert runtime._model_targets[model_id] == "cpu_avx2"


def test_raw_onnx_automatic_cpu_fallback_reports_cpu_target(tmp_path):
    runtime, backend, model_id = _onnx_runtime(tmp_path, "cuda_sm89", ["CPUExecutionProvider"])
    runtime._load_model(model_id)
    assert runtime._model_targets[model_id] == "cpu_avx2"
    assert backend.load_model.call_args.kwargs["providers"] == ["CPUExecutionProvider"]


def test_raw_onnx_gpu_request_passes_the_requested_ordinal(tmp_path):
    runtime, backend, model_id = _onnx_runtime(
        tmp_path, "cuda_sm89", ["CUDAExecutionProvider", "CPUExecutionProvider"], devices=["cuda:2"],
    )
    runtime._load_model(model_id)
    assert backend.load_model.call_args.kwargs["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 2}),
    ]
    assert backend.load_model.call_args.kwargs["allow_provider_fallback"] is False
    assert runtime._model_targets[model_id] == "cuda_sm89"


def test_raw_onnx_automatic_failed_gpu_initialization_can_try_cpu(tmp_path):
    runtime, backend, model_id = _onnx_runtime(
        tmp_path, "cuda_sm89", ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    backend.load_model.side_effect = [BackendError("CUDA dependencies unavailable"), object()]
    runtime._load_model(model_id)
    assert backend.load_model.call_count == 2
    assert runtime._model_targets[model_id] == "cpu_avx2"


def test_raw_onnx_explicit_gpu_initialization_failure_does_not_try_cpu(tmp_path):
    runtime, backend, model_id = _onnx_runtime(
        tmp_path, "cuda_sm89", ["CUDAExecutionProvider", "CPUExecutionProvider"], devices=["cuda:0"],
    )
    backend.load_model.side_effect = BackendError("CUDA dependencies unavailable")
    with pytest.raises(BackendError, match="CUDA dependencies"):
        runtime._load_model(model_id)
    backend.load_model.assert_called_once()
    assert not runtime._loaded_models


def test_raw_onnx_respects_configured_backend_constraint(tmp_path):
    runtime, backend, model_id = _onnx_runtime(
        tmp_path, "cpu_avx2", ["CPUExecutionProvider"], backend_name="pytorch",
    )
    with pytest.raises(BackendNotAvailableError, match="cannot execute a raw .onnx"):
        runtime._load_model(model_id)
    backend.load_model.assert_not_called()


def test_raw_onnx_directml_session_readiness_is_not_execution_validation(tmp_path):
    runtime, backend, model_id = _onnx_runtime(
        tmp_path, "cpu_avx2", ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    evidence = {"target_id": "directml", "available": False, "execution_tested": False, "extra": {}}
    runtime.fingerprint.capabilities = [evidence]
    runtime._load_model(model_id)
    assert backend.load_model.call_args.kwargs["providers"] == [
        ("DmlExecutionProvider", {"device_id": 0}),
    ]
    assert runtime._model_targets[model_id] == "directml"
    assert evidence["execution_tested"] is False
    assert evidence["available"] is False


@pytest.mark.parametrize("gfx,expected_target", [("gfx942:sramecc+:xnack-", "rocm_cdna3"), ("", "cpu_avx2")])
def test_unknown_hip_dispatch_uses_actual_framework_architecture_not_device_name(tmp_path, monkeypatch, gfx, expected_target):
    import sys

    package = _package(tmp_path, ["cpu_avx2"])
    runtime, native = _runtime(tmp_path, "cpu_avx2")
    detected = {
        "target_id": "rocm_unknown", "runtime_available": True, "hardware_present": True,
        "available": False, "execution_tested": False, "extra": {},
    }
    runtime.fingerprint.capabilities = [detected]
    module = SimpleNamespace(
        version=SimpleNamespace(hip="6.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda _: "AMD Instinct MI300X",
            get_device_properties=lambda _: SimpleNamespace(gcnArchName=gfx, total_memory=0, warp_size=64),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", module)
    accelerator = Mock()
    accelerator.name = "pytorch"
    accelerator.is_available.return_value = True
    accelerator.available_for_target.side_effect = lambda target: target == "rocm_cdna3"
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native, "pytorch": accelerator,
    }.get(name)
    backend, target, _ = runtime._resolve_aeg_backend(package, "model")
    assert target == expected_target
    assert backend is (accelerator if gfx else native)
    assert detected["target_id"] == "rocm_unknown"
    assert detected["execution_tested"] is False


def test_cpu_only_artifact_does_not_probe_framework_for_unknown_hip(tmp_path):
    package = _package(tmp_path, ["cpu_avx2"], portable=False)
    runtime, native = _runtime(tmp_path, "cpu_avx2")
    runtime.fingerprint.capabilities = [{
        "target_id": "rocm_unknown", "runtime_available": True, "hardware_present": True,
        "available": False, "extra": {},
    }]
    assert runtime._resolve_aeg_backend(package, "model") == (native, "cpu_avx2", ["cpu"])
    assert all(call.args[0] != "pytorch" for call in runtime.backend_registry.get_backend.call_args_list)


def test_executable_accelerator_plan_precedes_optional_portable_materialization(tmp_path):
    package = _package(tmp_path, ["cuda_sm80", "cpu_avx2"])
    package.set_backend_plan("cuda_sm80", "aether_cuda")
    runtime, native = _runtime(tmp_path, "cuda_sm80")
    accelerator = Mock()
    accelerator.name = "aether_cuda"
    accelerator.available_for_target.return_value = True
    runtime.backend_registry.get_backend.side_effect = lambda name: {
        "aether_cpu": native,
        "aether_cuda": accelerator,
    }.get(name)
    assert runtime._resolve_aeg_backend(package, "model")[0] is accelerator
