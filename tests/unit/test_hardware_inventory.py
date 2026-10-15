"""OS inventories test presence and identity only; mocks never prove GPU execution."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.backends import hardware_inventory as inventory
from aether.backends.capabilities import HardwareCapabilities


def _cap(vendor="NVIDIA", source="cuda", **identity):
    target = {"NVIDIA": "cuda_sm89", "AMD": "rocm_cdna3", "Intel": "openvino_gpu"}.get(vendor, "gpu_unknown")
    return HardwareCapabilities(vendor=vendor, device=vendor + " GPU", architecture="unknown",
        target_id=target, hardware_present=True, extra={"detection_method": source, **identity})


@pytest.mark.parametrize("driver_status", [0, -2147467262])
def test_windows_dxgi_enumerates_physical_adapters_and_releases_com_objects(monkeypatch, driver_status):
    released = []
    def create(identifier, pointer):
        pointer._obj.value = 100
        return 0
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: SimpleNamespace(CreateDXGIFactory1=Mock(side_effect=create)), raising=False)
    def method(pointer, slot, *signature):
        if slot == 12:
            def enumerate_adapter(factory, index, output):
                if index >= 2:
                    return -2005270526  # DXGI_ERROR_NOT_FOUND
                output._obj.value = index + 200
                return 0
            return enumerate_adapter
        if slot == 10:
            def describe(adapter, output):
                description = output._obj
                description.name = "Intel integrated GPU" if adapter.value == 200 else "Software WARP"
                description.vendor = 0x8086
                description.device = 0xA721
                description.dedicated_video = 128 * 1024**2
                description.shared_system = 4 * 1024**3
                description.luid.low = adapter.value
                description.flags = 2 if adapter.value == 201 else 0
                return 0
            return describe
        if slot == 9:
            def driver_version(adapter, interface, output):
                output._obj.value = (32 << 48) | (101 << 16) | 7085
                return driver_status
            return driver_version
        assert slot == 2
        return lambda pointer: released.append(pointer.value) or 0
    monkeypatch.setattr(inventory, "_com_method", method)
    result = inventory._windows_inventory()
    assert result["complete"] and not result["errors"]
    record, = result["devices"]
    assert record["vendor"] == "Intel" and record["device"] == "Intel integrated GPU"
    assert record["directml_device_index"] == 0
    assert record["memory_bytes"] == 128 * 1024**2
    assert record["memory_kind"] == "dedicated_video"
    assert record["shared_system_memory_limit_bytes"] == 4 * 1024**3
    assert record["driver_version"] == ("32.0.101.7085" if driver_status == 0 else "unknown")
    assert record["driver_version_source"] == ("IDXGIAdapter.CheckInterfaceSupport(IDXGIDevice)" if driver_status == 0 else None)
    assert record["driver_version_query_hresult"] == f"0x{driver_status & 0xffffffff:08x}"
    assert set(released) == {100, 200, 201}
    assert "execution_device" not in record and "supports_fp16" not in record


def test_windows_inventory_failure_is_unknown_and_still_releases_factory(monkeypatch):
    def create(identifier, pointer):
        pointer._obj.value = 100
        return 0
    released = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: SimpleNamespace(CreateDXGIFactory1=Mock(side_effect=create)), raising=False)
    monkeypatch.setattr(inventory, "_com_method", lambda pointer, slot, *args:
        (lambda *args: -2147024891) if slot == 12 else (lambda pointer: released.append(pointer.value)))
    result = inventory._windows_inventory()
    assert result["complete"] is False and result["devices"] == []
    assert "EnumAdapters1" in result["errors"][0]
    assert released == [100]


def test_linux_sysfs_lists_graphics_devices_without_optional_runtimes(tmp_path):
    entries = []
    class Entry:
        def __init__(self, name, path):
            self.name, self.path = name, path
        def __truediv__(self, field):
            return self.path / field
        def resolve(self):
            return self.path.resolve()
    class Root:
        def __truediv__(self, path):
            # PCI BDF contains ':' which Windows forbids as a filename. The
            # entry facade preserves real sysfs names while reading actual
            # fixture files on every test host.
            return SimpleNamespace(is_dir=lambda: True, iterdir=lambda: iter(entries)) if path == "bus/pci/devices" else tmp_path / path
    for name, kind, vendor, device in (("0000:01:00.0", "0x030000", "0x10de", "0x2786"),
                                       ("0000:00:02.0", "0x030000", "0x8086", "0xa721"),
                                       ("0000:00:1f.0", "0x060100", "0x8086", "0x1234")):
        path = tmp_path / name.replace(":", "_")
        path.mkdir(parents=True)
        entries.append(Entry(name, path))
        for field, value in (("class", kind), ("vendor", vendor), ("device", device)):
            (path / field).write_text(value)
    result = inventory._linux_inventory(Root())
    assert result["complete"] and not result["errors"]
    assert {record["vendor"] for record in result["devices"]} == {"Intel", "NVIDIA"}
    assert {record["pci_bus_id"] for record in result["devices"]} == {"0000:00:02.0", "0000:01:00.0"}
    assert all("memory_bytes" not in record for record in result["devices"])


def test_linux_platform_drm_gpu_can_exist_without_a_pci_bus(tmp_path):
    device = tmp_path / "class/drm/card0/device"
    device.mkdir(parents=True)
    result = inventory._linux_inventory(tmp_path)
    assert result["complete"] and len(result["devices"]) == 1
    assert result["devices"][0]["pci_bus_id"] is None
    assert result["devices"][0]["device_id"].startswith("sysfs:")


def test_inventory_exact_identity_enriches_without_creating_duplicate_gpu():
    cap = _cap(luid="0011223344556677", device_index=0, execution_device="cuda:0")
    record = {"vendor": "NVIDIA", "device": "Actual NVIDIA adapter", "source": "windows_dxgi",
              "luid": "0011223344556677", "device_id": "luid:0011223344556677", "directml_device_index": 1}
    caps = inventory.merge_inventory_capabilities([cap], {"devices": [record], "complete": True})
    assert caps == [cap]
    assert cap.extra["directml_device_index"] == 1
    assert cap.extra["device_index"] == 0 and cap.extra["execution_device"] == "cuda:0"
    assert cap.extra["os_inventory"] == record
    assert not cap.execution_tested


def test_inventory_only_gpu_has_no_guessed_execution_ordinal_or_precision():
    record = {"vendor": "Intel", "device": "Intel UHD Graphics", "source": "windows_dxgi",
              "luid": "0011223344556677", "memory_bytes": 128 * 1024**2,
              "memory_kind": "dedicated_video", "memory_source": "DXGI"}
    cap, = inventory.merge_inventory_capabilities([], {"devices": [record], "complete": True})
    assert cap.hardware_present is True and cap.runtime_available is None
    assert not cap.available and not cap.implemented and not cap.backend_available
    assert cap.extra["device_index"] is None and cap.extra["execution_device"] is None
    assert cap.memory_bytes == 128 * 1024**2 and cap.memory_kind == "dedicated_video"
    assert not cap.supports_fp32 and not cap.execution_tested


def test_gpu_count_deduplicates_strong_identity_across_providers():
    caps = [_cap(source="cuda", pci_bus_id="0000:01:00.0", uuid="GPU-00112233-4455-6677-8899-aabbccddeeff"),
            _cap(source="dxgi", luid="0102030405060708"),
            _cap(source="bridge", pci_bus_id="00000000:01:00.0", luid="0102030405060708")]
    summary = inventory.summarize_gpu_devices(caps)
    assert summary["count"] == summary["maximum_count"] == 1
    assert summary["count_known"] is True
    keys = inventory.device_identity_keys(caps[0].extra)
    assert "uuid:00112233-4455-6677-8899-aabbccddeeff" in keys


def test_gpu_count_keeps_ambiguous_same_vendor_names_unconfirmed():
    caps = [_cap(source="cuda"), _cap(source="dxgi")]
    summary = inventory.summarize_gpu_devices(caps)
    assert summary["count"] == 1 and summary["maximum_count"] == 2
    assert summary["count_known"] is False


def test_zero_and_placeholder_identifiers_do_not_confirm_a_physical_match():
    for evidence in ({"luid": "0000000000000000", "device_id": "luid:0000000000000000"},
                     {"uuid": "GPU-00000000-0000-0000-0000-000000000000"},
                     {"uuid": "unknown"}, {"registry_id": 0, "device_id": "metal_registry:0"}):
        assert inventory.device_identity_keys(evidence) == set()
    caps = [_cap(source="openvino", luid="0000000000000000"),
            _cap(source="openvino", luid="0000000000000000")]
    summary = inventory.summarize_gpu_devices(caps)
    assert summary["count"] == 1 and summary["maximum_count"] == 2
    assert not summary["count_known"]


def test_gpu_count_cannot_merge_known_different_vendors_by_name():
    summary = inventory.summarize_gpu_devices([_cap("NVIDIA", "cuda"), _cap("AMD", "hip")])
    assert summary["count"] == 2 and summary["count_known"]


def test_denied_inventory_does_not_confirm_absence():
    cpu = HardwareCapabilities.cpu_host()
    cpu.extra["hardware_inventory"] = {"complete": False, "errors": ["access denied"]}
    assert inventory.summarize_gpu_devices([cpu])["count_known"] is False
    cpu.extra["hardware_inventory"]["complete"] = True
    assert inventory.summarize_gpu_devices([cpu])["count_known"] is True


def test_cached_probe_runs_once_and_returned_evidence_is_independent(monkeypatch):
    monkeypatch.setattr(inventory, "_CACHE", None)
    monkeypatch.setattr(inventory.platform, "system", lambda: "Windows")
    probe = Mock(return_value={"devices": [{"vendor": "Intel"}], "complete": True, "errors": [], "source": "test"})
    monkeypatch.setattr(inventory, "_windows_inventory", probe)
    first = inventory.probe_gpu_inventory()
    first["devices"].clear()
    assert len(inventory.probe_gpu_inventory()["devices"]) == 1
    probe.assert_called_once_with()


def test_runtime_fingerprint_counts_inventory_gpu_while_selecting_available_cpu(monkeypatch):
    from aether.runtime.hardware import HardwareDetector
    cpu = HardwareCapabilities.cpu_host()
    record = {"vendor": "Intel", "device": "Intel UHD Graphics", "source": "windows_dxgi",
              "device_id": "luid:0011223344556677", "memory_bytes": 128 * 1024**2}
    caps = inventory.merge_inventory_capabilities([cpu], {"devices": [record], "complete": True})
    monkeypatch.setattr("aether.backends.hardware_detector.detect_all_capabilities", lambda: caps)
    fingerprint = HardwareDetector().detect()
    assert fingerprint.target_id.startswith("cpu_")
    assert fingerprint.gpu_count == 1 and fingerprint.gpu_name == "Intel UHD Graphics"
    assert fingerprint.attributes["gpu_count_known"] is True
    assert fingerprint.attributes["available_accelerator_count"] == 0
