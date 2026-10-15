"""Vendor discovery mocks test evidence handling, never physical execution validation."""

from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.backends import hardware_detector as detector


def _assign(pointer, ctype, value):
    ctypes.cast(pointer, ctypes.POINTER(ctype)).contents.value = value
    return 0


def _hip(names, *, init_status=0, memory_status=0):
    def name(buffer, length, index):
        buffer.value = names[index].encode()
        return 0

    def memory(pointer, index):
        return memory_status or _assign(pointer, ctypes.c_size_t, (index + 1) * 1024**3)

    return SimpleNamespace(
        hipInit=Mock(return_value=init_status),
        hipGetDeviceCount=Mock(side_effect=lambda p: _assign(p, ctypes.c_int, len(names))),
        hipDeviceGet=Mock(side_effect=lambda p, i: _assign(p, ctypes.c_int, i)),
        hipDeviceGetName=Mock(side_effect=name),
        hipDeviceTotalMem=Mock(side_effect=memory),
        hipRuntimeGetVersion=Mock(side_effect=lambda p: _assign(p, ctypes.c_int, 60241134)),
    )


def test_hip_enumeration_preserves_each_device_without_framework_or_execution(monkeypatch):
    hip = _hip(["AMD Instinct MI300X gfx942", "AMD unknown future device"])
    monkeypatch.setattr(detector, "_load_hip_runtime", lambda: hip)
    monkeypatch.setattr(detector, "_vendor_package_present", lambda _: False)
    caps = detector.detect_rocm_devices()
    assert len(caps) == 2
    assert [cap.extra["device_index"] for cap in caps] == [0, 1]
    assert caps[0].target_id == "rocm_cdna3"
    assert caps[1].target_id == "rocm_unknown"
    assert caps[1].architecture == "gfx_unknown"
    assert caps[1].implemented is False
    assert [cap.extra["execution_device"] for cap in caps] == ["cuda:0", "cuda:1"]
    assert [cap.memory_bytes for cap in caps] == [1024**3, 2 * 1024**3]
    for cap in caps:
        assert cap.hardware_present is True and cap.runtime_available is True
        assert cap.available is True and cap.backend_available is False
        assert cap.execution_tested is False and cap.compile_tested is False
        assert cap.production_validated is False
        assert cap.supports_bf16 is False and cap.supports_fp8 is False
        assert cap.memory_source == "hipDeviceTotalMem"


def test_hip_init_failure_is_not_execution_success(monkeypatch):
    hip = _hip(["AMD GPU"], init_status=35)
    monkeypatch.setattr(detector, "_load_hip_runtime", lambda: hip)
    caps, reason = detector._detect_hip_runtime_devices()
    assert caps == []
    assert "status 35" in reason
    hip.hipGetDeviceCount.assert_not_called()


def test_hip_memory_query_failure_remains_unknown(monkeypatch):
    monkeypatch.setattr(
        detector, "_load_hip_runtime", lambda: _hip(["AMD MI300X"], memory_status=1)
    )
    cap = detector._detect_hip_runtime_devices()[0][0]
    assert cap.memory_bytes == 0
    assert cap.memory_source is None
    assert cap.memory_kind is None


def test_hip_product_name_is_only_an_architecture_hint(monkeypatch):
    monkeypatch.setattr(detector, "_load_hip_runtime", lambda: _hip(["AMD Instinct MI300X"]))
    cap, = detector._detect_hip_runtime_devices()[0]
    assert cap.target_id == "rocm_unknown" and cap.architecture == "gfx_unknown"
    assert cap.extra["inferred_target_id"] == "rocm_cdna3"
    assert cap.extra["architecture_source"] == "unknown"
    assert cap.implemented is False and cap.backend_available is False
    assert cap.runtime_available is True and cap.execution_tested is False


def test_hip_stable_identity_does_not_use_a_versioned_property_struct(monkeypatch):
    hip = _hip(["AMD GPU gfx942"])
    def bus(buffer, length, index):
        buffer.value = b"00000000:41:00.0"
        return 0
    def identifier(buffer, index):
        buffer[:] = bytes(range(16))
        return 0
    hip.hipDeviceGetPCIBusId = Mock(side_effect=bus)
    hip.hipDeviceGetUuid = Mock(side_effect=identifier)
    monkeypatch.setattr(detector, "_load_hip_runtime", lambda: hip)
    cap, = detector._detect_hip_runtime_devices()[0]
    assert cap.extra["pci_bus_id"] == "0000:41:00.0"
    assert cap.extra["uuid"] == "00010203-0405-0607-0809-0a0b0c0d0e0f"
    assert cap.target_id == "rocm_cdna3" and cap.implemented
    assert not cap.execution_tested


def test_rocm_detection_does_not_import_execution_frameworks(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "mlx"}:
            raise AssertionError("hardware discovery imported an execution framework")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(detector, "_load_hip_runtime", lambda: _hip(["AMD MI300X"]))
    assert detector.detect_rocm_devices()[0].execution_tested is False


def test_amdsmi_monitoring_is_not_a_usable_hip_runtime(monkeypatch):
    monkeypatch.setattr(
        detector, "_detect_hip_runtime_devices", lambda: ([], "HIP library missing")
    )
    smi = SimpleNamespace(
        amdsmi_init=Mock(),
        amdsmi_shut_down=Mock(),
        amdsmi_get_processor_handles=lambda: [1],
        amdsmi_get_gpu_board_info=lambda _: {"product_name": "AMD Instinct MI300X"},
        amdsmi_get_gpu_memory_total=lambda *a: 8 * 1024**3,
        AmdSmiMemoryType=SimpleNamespace(VRAM=0),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", smi)
    cap = detector.detect_rocm_devices()[0]
    assert cap.hardware_present is True
    assert cap.runtime_available is False and cap.available is False
    assert cap.memory_bytes == 8 * 1024**3
    assert cap.extra["monitoring_only"] is True
    assert cap.execution_tested is False
    assert cap.unavailable_reason == "HIP library missing"
    smi.amdsmi_shut_down.assert_called_once()


@pytest.mark.parametrize(
    "device,gfx,target",
    [
        ("AMD Instinct MI250", "gfx90a", "rocm_gfx90a"),
        ("AMD Radeon RX 6900 XT", "gfx1030", "rocm_gfx1030"),
        ("gfx942:sramecc+:xnack-", "gfx942", "rocm_cdna3"),
        ("unknown AMD adapter", "gfx_unknown", "rocm_unknown"),
    ],
)
def test_amd_architecture_is_not_invented_as_cdna3(device, gfx, target):
    assert detector._infer_gfx_arch(device) == gfx
    assert detector._rocm_target_id(device) == target


def test_legacy_hip_helper_does_not_create_devices_when_enumeration_fails():
    torch = SimpleNamespace(
        version=SimpleNamespace(hip="6.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=Mock(side_effect=RuntimeError("driver error"))
        ),
    )
    assert detector._detect_rocm_via_hip(torch) == []
    assert detector.inspect_torch_hip_devices(torch) == []


def test_caller_supplied_hip_properties_are_required_for_device_availability():
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=Mock(
            return_value=SimpleNamespace(gcnArchName="gfx90a:xnack-", total_memory=1024**3)
        ),
    )
    torch = SimpleNamespace(version=SimpleNamespace(hip="6.2"), cuda=cuda)
    invalid = detector._rocm_cap_from_torch(1, "AMD GPU", torch)
    assert invalid.available is False
    cuda.get_device_properties.assert_not_called()
    valid = detector._rocm_cap_from_torch(0, "AMD MI250", torch)
    assert valid.available is True and valid.target_id == "rocm_gfx90a"
    assert valid.warp_or_wavefront_size == 0
    assert valid.extra["wavefront_size_known"] is False
    assert valid.execution_tested is False


def test_public_hip_inspection_confirms_gfx_from_framework_not_name():
    torch = SimpleNamespace(version=SimpleNamespace(hip="6.2"), cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda _: "AMD GPU product name intentionally unclassified",
        get_device_properties=lambda _: SimpleNamespace(gcnArchName="gfx942:xnack-", total_memory=1024**3)))
    cap, = detector.inspect_torch_hip_devices(torch)
    assert cap.target_id == "rocm_cdna3" and cap.implemented and cap.backend_available
    assert cap.extra["execution_device"] == "cuda:0"
    assert cap.extra["raw_architecture"] == "gfx942:xnack-"
    assert not cap.execution_tested and not cap.production_validated


def _metal(monkeypatch, chipset, *, machine="arm64", native=True):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(detector.platform, "machine", lambda: machine)
    monkeypatch.setattr(detector.platform, "mac_ver", lambda: ("15.4", (), ""))
    probe = Mock(return_value={"devices": [{"device": chipset, "registry_id": 1234,
        "device_id": "metal_registry:1234", "system_default": True,
        "unified_memory": machine == "arm64", "recommended_working_set_bytes": 10 * 1024**3}]
        if native else [], "errors": [] if native else ["native device enumeration failed"], "complete": native})
    monkeypatch.setattr(detector, "_query_native_metal_devices", probe)
    monkeypatch.setattr(detector, "_host_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(detector, "_vendor_package_present", lambda _: False)
    return probe


@pytest.mark.parametrize(
    "chipset,target",
    [("Apple M1 Pro", "metal_m1"), ("Apple M2", "metal_m1"), ("Apple M4 Max", "metal_m3")],
)
def test_metal_uses_real_chipset_and_only_reports_confirmed_formats(monkeypatch, chipset, target):
    probe = _metal(monkeypatch, chipset)
    cap = detector.detect_metal()
    assert cap.device == chipset and cap.target_id == target
    assert cap.available is True and cap.runtime_available is True
    assert cap.supports_unified_memory is True
    assert cap.memory_bytes == 16 * 1024**3 and cap.memory_kind == "shared_host"
    assert cap.supports_bf16 is False
    assert cap.execution_tested is False and cap.production_validated is False
    assert cap.extra["registry_id"] == 1234
    assert cap.extra["execution_device"] == "mps"
    assert cap.extra["recommended_working_set_bytes"] == 10 * 1024**3
    probe.assert_called_once_with()


def test_non_apple_metal_device_is_not_relabelled_m3_or_unified_memory(monkeypatch):
    _metal(monkeypatch, "AMD Radeon Pro 5500M", machine="x86_64")
    cap = detector.detect_metal()
    assert cap.target_id == "metal"
    assert cap.supports_unified_memory is False
    assert cap.memory_bytes == 0 and cap.memory_source is None
    assert cap.device == "AMD Radeon Pro 5500M"


def test_failed_native_metal_enumeration_is_unknown_not_available(monkeypatch):
    _metal(monkeypatch, "Apple M3", native=False)
    cap = detector.detect_metal()
    assert cap.hardware_present is None
    assert cap.available is False
    assert cap.execution_tested is False


def test_metal_all_devices_keep_registry_identity_and_only_default_exposes_mps(monkeypatch):
    _metal(monkeypatch, "AMD Radeon Pro 5500M", machine="x86_64")
    monkeypatch.setattr(detector, "_query_native_metal_devices", lambda: {"devices": [
        {"device": "Intel Iris Plus Graphics", "registry_id": 10, "system_default": False,
         "unified_memory": False, "recommended_working_set_bytes": 6 * 1024**3},
        {"device": "AMD Radeon Pro 5500M", "registry_id": 20, "system_default": True,
         "unified_memory": False, "recommended_working_set_bytes": 3 * 1024**3},
    ], "errors": [], "complete": True})
    caps = detector.detect_metal_devices()
    assert [cap.extra["registry_id"] for cap in caps] == [20, 10]
    assert [cap.extra["device_index"] for cap in caps] == [1, 0]
    assert [cap.extra["execution_device"] for cap in caps] == ["mps", None]
    assert all(cap.target_id == "metal" and cap.memory_bytes == 0 for cap in caps)
    assert all(not cap.execution_tested for cap in caps)
    assert caps[1].backend_available is False


def _openvino(monkeypatch, devices, properties):
    core = SimpleNamespace(
        available_devices=devices,
        get_property=Mock(side_effect=lambda device, prop: properties[(device, prop)]),
    )
    module = SimpleNamespace(Core=lambda: core, __version__="2026.test")
    monkeypatch.setitem(sys.modules, "openvino", module)
    return core


def test_openvino_precision_flags_are_queried_and_not_hardcoded(monkeypatch):
    _openvino(
        monkeypatch,
        ["CPU", "GPU.0"],
        {
            ("GPU.0", "FULL_DEVICE_NAME"): "Actual Intel integrated GPU",
            ("GPU.0", "OPTIMIZATION_CAPABILITIES"): ["FP32", "FP16", "INT8"],
            ("GPU.0", "DEVICE_TYPE"): "INTEGRATED",
            ("GPU.0", "GPU_DEVICE_TOTAL_MEM_SIZE"): 12 * 1024**3,
        },
    )
    cap = detector.detect_openvino_gpu()
    assert cap.device == "Actual Intel integrated GPU"
    assert cap.available is True and cap.hardware_present is True
    assert cap.supports_fp32 is True and cap.supports_fp16 is True
    assert cap.supports_int8 is True
    assert cap.supports_bf16 is False and cap.supports_int4 is False
    assert cap.memory_kind == "shared_allocation_limit" and cap.memory_bytes == 12 * 1024**3
    assert cap.implemented is False and cap.backend_available is False
    assert cap.execution_tested is False


def test_openvino_unknown_precision_and_memory_are_not_confirmation(monkeypatch):
    _openvino(monkeypatch, ["NPU"], {})
    cap = detector.detect_openvino()
    assert cap.available is True
    assert cap.supports_fp32 is False and cap.supports_bf16 is False
    assert cap.memory_bytes == 0 and cap.memory_source is None
    assert cap.implemented is False


def test_openvino_package_without_device_is_unavailable(monkeypatch):
    _openvino(monkeypatch, ["CPU"], {})
    cap = detector.detect_openvino_gpu()
    assert cap.available is False
    assert cap.implemented is False
    assert "contains no GPU" in cap.unavailable_reason


def test_openvino_enumerates_all_gpu_npu_ids_from_one_core(monkeypatch):
    core = _openvino(monkeypatch, ["CPU", "GPU.0", "GPU.2", "NPU.0", "NPU.1"], {
        ("GPU.0", "FULL_DEVICE_NAME"): "Intel integrated GPU",
        ("GPU.2", "FULL_DEVICE_NAME"): "Intel discrete GPU",
        ("GPU.2", "DEVICE_LUID"): "0011223344556677",
        ("NPU.0", "NPU_DRIVER_VERSION"): "actual-driver",
        ("NPU.0", "NPU_DEVICE_TOTAL_MEM_SIZE"): 1024**3,
    })
    module = sys.modules["openvino"]
    module.Core = Mock(return_value=core)
    caps = detector.detect_openvino_devices()
    assert [cap.extra["device_id"] for cap in caps] == ["GPU.0", "GPU.2", "NPU.0", "NPU.1"]
    assert [cap.extra["device_index"] for cap in caps] == [0, 2, 0, 1]
    assert caps[1].extra["luid"] == "0011223344556677"
    assert caps[2].driver_version == "actual-driver" and caps[2].memory_bytes == 1024**3
    assert all(cap.available and not cap.implemented and not cap.execution_tested for cap in caps)
    module.Core.assert_called_once_with()
