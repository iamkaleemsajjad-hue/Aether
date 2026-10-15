"""Detection mocks test evidence rules; they never certify physical execution."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from aether.backends import capabilities, cuda_detection, hardware_detector
from aether.backends.capabilities import HardwareCapabilities


@pytest.mark.parametrize(("features", "target", "fp16", "bf16"), [
    ({}, "cpu_generic", False, False),
    ({"avx2": True, "f16c": True}, "cpu_avx2", False, False),
    ({"avx512f": True, "avx2": True}, "cpu_avx512", False, False),
    ({"avx512f": True, "avx512fp16": False, "avx512bf16": False}, "cpu_avx512", False, False),
    ({"avx512f": True, "avx512fp16": True}, "cpu_avx512", True, False),
    ({"avx512f": True, "avx512bf16": True}, "cpu_avx512", False, True),
    ({"neon": True, "asimdhp": True}, "cpu_neon", True, False),
    ({"avx512f": False, "avx2": False, "avx": False}, "cpu_generic", False, False),
    ({"avx512f": False, "avx512fp16": True, "avx512bf16": True}, "cpu_generic", False, False),
])
def test_cpu_requires_separate_precision_evidence(monkeypatch, features, target, fp16, bf16):
    monkeypatch.setattr(capabilities, "cpu_runtime_features", lambda: features)
    monkeypatch.setattr(capabilities.platform, "processor", lambda: "AMD64 Family 25 Model 117")
    cpu = hardware_detector.detect_cpu()
    assert cpu.target_id == target
    assert cpu.supports_fp16 is fp16
    assert cpu.supports_bf16 is bf16
    assert not cpu.execution_tested and not cpu.compile_tested and not cpu.production_validated
    assert cpu.extra["precision_execution"]["bf16"] == "storage_converted_to_fp32"
    assert cpu.extra["precision_evidence"]["fp16"]["extension_confirmed"] is bool(
        features.get("avx512fp16") or features.get("asimdhp")
    )
    assert cpu.extra["precision_evidence"]["bf16"]["extension_confirmed"] is bool(
        features.get("avx512bf16") or features.get("bf16")
    )


def _fake_driver(major=8, minor=9, init_error=0, attribute_error=None, memory_error=0):
    calls = []
    def init(_):
        calls.append("init")
        return init_error
    def output(pointer, value):
        pointer._obj.value = value
        return 0
    def attribute(pointer, key, device):
        calls.append(("attribute", key, device))
        if key == attribute_error:
            return 999
        return output(pointer, {75: major, 76: minor, 10: 32, 83: 1}.get(key, 0))
    def name(buffer, length, device):
        buffer.value = b"NVIDIA GeForce RTX 4070 Laptop GPU"
        return 0
    def memory(pointer, device):
        if memory_error:
            return memory_error
        return output(pointer, 8 * 1024 ** 3)
    return SimpleNamespace(
        cuInit=init, cuDeviceGetCount=lambda pointer: output(pointer, 1),
        cuDeviceGet=lambda pointer, index: output(pointer, index),
        cuDriverGetVersion=lambda pointer: output(pointer, 12040),
        cuDeviceGetAttribute=attribute, cuDeviceGetName=name,
        cuDeviceTotalMem_v2=memory, calls=calls,
    )


def _install_driver(monkeypatch, driver, telemetry=None):
    monkeypatch.setattr(cuda_detection, "_nvml_evidence", lambda: (telemetry or [], "unknown"))
    monkeypatch.setattr(cuda_detection, "_driver_release", lambda _: ("unknown", None))
    monkeypatch.setattr(cuda_detection.ctypes, "CDLL", lambda *_: driver)


def test_cuda_driver_query_is_not_inference_or_driver_release(monkeypatch):
    driver = _fake_driver()
    _install_driver(monkeypatch, driver)
    cap, = hardware_detector.detect_cuda_devices()
    assert "init" in driver.calls
    assert cap.target_id == "cuda_sm89"
    assert cap.device == "NVIDIA GeForce RTX 4070 Laptop GPU"
    assert cap.memory_bytes == 8 * 1024 ** 3
    assert cap.runtime_version == "CUDA Driver API 12.4"
    assert cap.driver_version == "unknown"  # CUDA API version is not driver release
    assert cap.extra["cuda_driver_api_version"] == "12.4"
    assert cap.hardware_present and cap.runtime_available and cap.available
    assert cap.backend_available is (None if cap.backend_dependencies["torch"] else False)
    assert cap.supports_fp8 and cap.supports_int4
    assert not cap.supports_peer_access  # a single GPU has no confirmed peer
    assert not cap.execution_tested and not cap.production_validated


def test_nvml_inventory_does_not_prove_cuda_ready(monkeypatch):
    driver = _fake_driver(init_error=803)
    _install_driver(monkeypatch, driver, [{"name": "RTX 4070 Laptop", "memory": 8 * 1024 ** 3, "cc": (8, 9)}])
    cap, = hardware_detector.detect_cuda_devices()
    assert cap.hardware_present
    assert not cap.available and not cap.runtime_available and not cap.backend_available
    assert "cuInit failed" in cap.unavailable_reason
    assert not cap.execution_tested


def test_failed_compute_capability_query_is_not_sm0_success(monkeypatch):
    _install_driver(monkeypatch, _fake_driver(attribute_error=75))
    cap, = hardware_detector.detect_cuda_devices()
    assert cap.target_id == "cuda_unknown"
    assert not cap.available and not cap.implemented
    assert not cap.supports_fp16 and not cap.supports_bf16
    assert "query failed" in cap.unavailable_reason


def test_cuda_memory_query_failure_is_explicit(monkeypatch):
    _install_driver(monkeypatch, _fake_driver(memory_error=999))
    cap, = hardware_detector.detect_cuda_devices()
    assert cap.available
    assert cap.memory_bytes == 0
    assert not cap.to_dict()["memory_bytes_known"]
    assert cap.memory_source is None


def test_cuda_identity_joins_telemetry_without_reusing_its_physical_ordinal(monkeypatch):
    from unittest.mock import Mock
    driver = _fake_driver()
    def bus(buffer, length, device):
        buffer.value = b"0000:41:00.0"
        return 0
    def identifier(buffer, device):
        buffer[:] = bytes(range(16))
        return 0
    driver.cuDeviceGetPCIBusId = Mock(side_effect=bus)
    driver.cuDeviceGetUuid_v2 = Mock(side_effect=identifier)
    _install_driver(monkeypatch, driver, [{"name": "physical NVIDIA GPU", "telemetry_index": 4,
        "pci_bus_id": "00000000:41:00.0", "uuid": "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f", "nvlink_active": True}])
    cap, = hardware_detector.detect_cuda_devices()
    assert cap.extra["device_index"] == 0 and cap.extra["execution_device"] == "cuda:0"
    assert cap.extra["pci_bus_id"] == "0000:41:00.0"
    assert cap.extra["uuid"] == "00010203-0405-0607-0809-0a0b0c0d0e0f"
    assert cap.supports_nvlink and not cap.execution_tested


def test_hidden_nvml_device_has_no_assumed_compute_ordinal(monkeypatch):
    driver = _fake_driver()
    _install_driver(monkeypatch, driver, [{"name": "hidden NVIDIA GPU", "telemetry_index": 1,
                                        "pci_bus_id": "0000:42:00.0", "cc": (8, 9)}])
    active, hidden = hardware_detector.detect_cuda_devices()
    assert active.available and active.extra["execution_device"] == "cuda:0"
    assert hidden.hardware_present and not hidden.available and not hidden.runtime_available
    assert hidden.extra["device_index"] is None and hidden.extra["execution_device"] is None
    assert hidden.extra["telemetry_index"] == 1
    assert not hidden.execution_tested


def test_older_confirmed_cuda_fp32_is_not_conflated_with_half_arithmetic():
    assert cuda_detection._precision(5, 2)["supports_fp32"]
    assert not cuda_detection._precision(5, 2)["supports_fp16"]
    assert cuda_detection._precision(5, 3)["supports_fp16"]


@pytest.mark.parametrize(("major", "minor", "fp8", "fp4", "int4"), [
    (8, 0, False, False, True), (8, 9, True, False, True),
    (9, 0, True, False, False), (10, 0, True, True, False),
    (12, 0, True, True, False), (13, 0, False, False, False),
])
def test_cuda_precision_formats_are_architecture_specific(major, minor, fp8, fp4, int4):
    features = cuda_detection._precision(major, minor)
    assert features["supports_fp8"] is fp8
    assert features["supports_fp4"] is fp4
    assert features["supports_int4"] is int4


def test_unknown_future_cuda_major_is_not_downgraded():
    assert cuda_detection._target(11, 0) == "cuda_sm110"
    assert cuda_detection._target(8, 6) == "cuda_sm80"
    assert cuda_detection._target(8, 9) == "cuda_sm89"


def test_directml_provider_registration_does_not_enumerate_device(monkeypatch):
    from aether.backends import auxiliary_detection as auxiliary
    monkeypatch.setattr(auxiliary.platform, "system", lambda: "Windows")
    monkeypatch.setattr("aether.backends.hardware_inventory.probe_gpu_inventory", lambda: {"devices": [], "complete": False})
    ort = SimpleNamespace(__version__="test", get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(auxiliary.importlib, "import_module", lambda _: ort)
    cap = auxiliary.detect_directml()
    assert cap.implemented
    assert not cap.available and cap.hardware_present is None
    assert cap.runtime_available is None
    assert cap.backend_available is None
    assert cap.backend_dependencies["DmlExecutionProvider"]
    assert not cap.execution_tested and not cap.compile_tested and not cap.production_validated
    assert not cap.supports_fp16 and not cap.supports_fp32
    assert cap.memory_bytes == 0 and not cap.to_dict()["memory_bytes_known"]
    assert "AEG exports" in cap.extra["implementation_scope"]
    assert "ONNX Runtime" in cap.extra["implementation_scope"]
    # Provider registration and implemented export support are distinct from
    # device initialization and physical execution evidence.
    assert "require explicit validation" in cap.unavailable_reason


def test_directml_os_dll_presence_is_not_gpu_evidence(monkeypatch):
    from aether.backends import auxiliary_detection as auxiliary
    monkeypatch.setattr(auxiliary.platform, "system", lambda: "Windows")
    monkeypatch.setattr("aether.backends.hardware_inventory.probe_gpu_inventory", lambda: {"devices": [], "complete": False})
    def missing(_):
        raise ImportError("no provider")
    monkeypatch.setattr(auxiliary.importlib, "import_module", missing)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_, **__: object(), raising=False)
    cap = auxiliary.detect_directml()
    assert cap.extra["os_dlls_present"]
    assert not cap.available and not cap.backend_available
    assert cap.hardware_present is None
    assert not cap.supports_fp32 and not cap.supports_fp16


def test_qnn_sdk_does_not_confirm_npu(monkeypatch):
    from aether.backends import auxiliary_detection as auxiliary
    monkeypatch.setattr(auxiliary.importlib, "import_module", lambda _: SimpleNamespace(__version__="2.0"))
    cap = auxiliary.detect_qualcomm_npu()
    assert cap.backend_dependencies["qnn"]
    assert cap.hardware_present is None and cap.runtime_available is None
    assert not cap.implemented and not cap.available and not cap.execution_tested
    assert not cap.supports_fp16 and not cap.supports_int8


def test_ane_private_api_failure_is_unknown_not_success(monkeypatch):
    from aether.backends import auxiliary_detection as auxiliary
    monkeypatch.setattr(auxiliary.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(auxiliary.platform, "machine", lambda: "arm64")
    def module(name):
        if name.endswith("utils"):
            return SimpleNamespace()  # private enumeration API absent
        return SimpleNamespace(__version__="test")
    monkeypatch.setattr(auxiliary.importlib, "import_module", module)
    cap = auxiliary.detect_apple_ane()
    assert cap.hardware_present is None
    assert not cap.available and not cap.implemented and not cap.execution_tested
    assert cap.extra["enumeration_error"]


def test_validation_rejects_incompatible_cuda_target_before_backend(monkeypatch):
    cap = HardwareCapabilities(vendor="NVIDIA", device="RTX4070", architecture="sm89", target_id="cuda_sm89",
                               available=True, implemented=True, extra={"device_index": 0, "compute_capability": "8.9"})
    monkeypatch.setattr(hardware_detector, "detect_cuda_devices", lambda: [cap])
    result = hardware_detector.validate_backend_environment("cuda_sm90", execute=True)
    assert not result.all_passed and not result.available and not result.execution_tested
    assert "compatible" in result.checks_failed[0]


def test_validation_does_not_claim_unknown_gpu_target_backend_ready(monkeypatch):
    cap = HardwareCapabilities(vendor="NVIDIA", device="future GPU", architecture="sm110", target_id="cuda_sm110",
                               available=True, implemented=False, extra={"device_index": 0, "compute_capability": "11.0"})
    monkeypatch.setattr(hardware_detector, "detect_cuda_devices", lambda: [cap])
    result = hardware_detector.validate_backend_environment("cuda_sm110")
    assert result.available  # physical driver evidence is preserved
    assert not result.all_passed and not result.backend_available and not result.execution_tested
    assert "no compatible implemented" in result.checks_failed[0]


def test_validation_environment_only_never_marks_execution():
    result = hardware_detector.validate_backend_environment("cpu_generic")
    assert result.all_passed
    assert result.backend_available
    assert not result.execution_tested and result.execution_scope is None
    assert not result.production_validated


def test_execution_failure_is_visible_and_not_tested(monkeypatch):
    from aether.backends import hardware_validation
    def fail(*_):
        raise RuntimeError("workload execution failed")
    monkeypatch.setattr(hardware_validation, "_execute_aeg_probe", fail)
    result = hardware_detector.validate_backend_environment("cpu_generic", execute=True)
    assert result.available and result.backend_available
    assert not result.all_passed and not result.execution_tested
    assert "workload execution failed" in result.checks_failed[0]


def test_real_cpu_numpy_fallback_is_reported_and_executes(monkeypatch):
    from aether.kernels import native_cpu
    reference = native_cpu.NativeCPUKernels.reference()
    monkeypatch.setattr(native_cpu, "_INSTANCE", reference)
    monkeypatch.setattr(native_cpu, "get_native_kernels", lambda: reference)
    result = hardware_detector.validate_backend_environment("cpu_generic", execute=True)
    assert result.all_passed, result.to_dict()
    assert result.execution_tested
    assert result.details["native_cpu"]["compiler_optional"]
    assert result.details["native_cpu"]["fallback"] == "NumPy FP32 reference"
    assert result.details["workload"]["native_cpu"] is False
    assert any("compilation failed" in warning and "optional" in warning for warning in result.warnings)


def test_directml_execution_failure_never_upgrades_provider_registration(monkeypatch):
    from aether.backends.onnx_backend import validate_directml_execution
    assert callable(validate_directml_execution)
    cap = HardwareCapabilities(vendor="Microsoft", device="not enumerated", architecture="directml", target_id="directml",
                               implemented=True, available=False, backend_dependencies={"DmlExecutionProvider": True})
    monkeypatch.setattr(hardware_detector, "detect_directml", lambda: cap)
    def fail(_):
        raise RuntimeError("No attributed DirectML execution nodes")
    monkeypatch.setattr("aether.backends.onnx_backend.validate_directml_execution", fail)
    result = hardware_detector.validate_backend_environment("directml", execute=True)
    assert not result.all_passed and not result.available and not result.execution_tested
    assert "No attributed" in result.checks_failed[0]


def test_real_cpu_aeg_validation_runs_prefill_and_decode():
    result = hardware_detector.validate_backend_environment("cpu_generic", execute=True)
    assert result.all_passed, result.to_dict()
    assert result.execution_tested
    assert result.execution_scope == "aether_aeg_compile_load_prefill_decode"
    workload = result.details["workload"]
    assert workload["device"] == "cpu" and workload["kv_cache_length"] == 4
    assert workload["max_absolute_errors"] and max(workload["max_absolute_errors"]) < 2e-4
    assert not result.production_validated


@pytest.mark.gpu
def test_real_cuda_aeg_execution_never_uses_cpu_fallback():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("No physical NVIDIA CUDA device and matching PyTorch runtime")
    devices = [cap for cap in hardware_detector.detect_cuda_devices() if cap.available]
    if not devices:
        pytest.fail("PyTorch CUDA initialized but Aether driver detection found no usable device")
    result = hardware_detector.validate_backend_environment(devices[0].target_id, execute=True,
                                                            device_index=devices[0].extra["device_index"])
    assert result.all_passed, result.to_dict()
    assert result.execution_tested
    assert result.details["workload"]["device"].startswith("cuda:")
    assert not result.production_validated
