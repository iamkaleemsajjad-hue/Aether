"""Bounded OS GPU inventory, independent of optional inference runtimes.

These records establish devices, identities and scoped memory measurements.
They never establish CUDA/HIP ordinals, supported arithmetic or execution.
"""
from __future__ import annotations

import copy
import ctypes
import json
import platform
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from aether.backends.capabilities import HardwareCapabilities

_CACHE: dict[str, Any] | None = None
_CACHE_AT = 0.0
_CACHE_LOCK = threading.Lock()
_VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel", 0x17CB: "Qualcomm"}


def normalize_pci_bus_id(value: Any) -> str | None:
    match = re.fullmatch(r"([0-9a-f]{1,8}):([0-9a-f]{1,2}):([0-9a-f]{1,2})\.([0-7])", str(value or "").lower())
    if match is None:
        return None
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def _com_method(pointer: ctypes.c_void_p, slot: int, result: Any, *arguments: Any) -> Any:
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[slot])


def _windows_inventory() -> dict[str, Any]:
    """Public DXGI 1.1 enumeration; hardware and DirectML indices share this order."""
    class Guid(ctypes.Structure):
        _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                    ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    class Luid(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]

    class Description(ctypes.Structure):
        _fields_ = [("name", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint32),
                    ("device", ctypes.c_uint32), ("subsystem", ctypes.c_uint32),
                    ("revision", ctypes.c_uint32), ("dedicated_video", ctypes.c_size_t),
                    ("dedicated_system", ctypes.c_size_t), ("shared_system", ctypes.c_size_t),
                    ("luid", Luid), ("flags", ctypes.c_uint32)]

    result: dict[str, Any] = {"devices": [], "complete": False, "source": "windows_dxgi", "errors": []}
    factory = ctypes.c_void_p()
    try:
        dxgi = ctypes.WinDLL("dxgi", use_last_error=True)
        create = dxgi.CreateDXGIFactory1
        create.argtypes = [ctypes.POINTER(Guid), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        identifier = Guid.from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
        code = create(ctypes.byref(identifier), ctypes.byref(factory))
        if code != 0 or not factory:
            raise RuntimeError(f"CreateDXGIFactory1 failed (HRESULT 0x{code & 0xffffffff:08x})")
        enumerate_adapter = _com_method(factory, 12, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(256):
            adapter = ctypes.c_void_p()
            code = enumerate_adapter(factory, index, ctypes.byref(adapter))
            if code & 0xffffffff == 0x887A0002:  # DXGI_ERROR_NOT_FOUND
                result["complete"] = True
                break
            if code != 0 or not adapter:
                raise RuntimeError(f"EnumAdapters1({index}) failed (HRESULT 0x{code & 0xffffffff:08x})")
            try:
                description = Description()
                code = _com_method(adapter, 10, ctypes.c_int32, ctypes.POINTER(Description))(adapter, ctypes.byref(description))
                if code != 0:
                    result["errors"].append(f"GetDesc1({index}) failed (HRESULT 0x{code & 0xffffffff:08x})")
                    continue
                if description.flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE (e.g. WARP)
                    continue
                luid = ctypes.string_at(ctypes.byref(description.luid), ctypes.sizeof(Luid)).hex()
                # This is the queried UMD package version, not proof that a
                # D3D12/DirectML workload can execute. Failed optional queries
                # leave the version unknown without losing the adapter.
                driver = ctypes.c_int64()
                device_interface = Guid.from_buffer_copy(uuid.UUID("54ec77fa-1377-44e6-8c32-88fd5f44c84c").bytes_le)
                driver_code = _com_method(adapter, 9, ctypes.c_int32, ctypes.POINTER(Guid), ctypes.POINTER(ctypes.c_int64))(adapter, ctypes.byref(device_interface), ctypes.byref(driver))
                driver_version = ".".join(str((driver.value >> shift) & 0xffff) for shift in (48, 32, 16, 0)) if driver_code == 0 and driver.value else "unknown"
                result["devices"].append({
                    "vendor": _VENDORS.get(description.vendor, "unknown"),
                    "device": str(description.name) or "DXGI hardware adapter (name unavailable)",
                    "device_id": f"luid:{luid}", "luid": luid,
                    "dxgi_adapter_index": index, "directml_device_index": index,
                    "vendor_id": f"{description.vendor:04x}", "pci_device_id": f"{description.device:04x}",
                    "memory_bytes": int(description.dedicated_video), "memory_kind": "dedicated_video",
                    "memory_source": "DXGI_ADAPTER_DESC1.DedicatedVideoMemory",
                    "dedicated_system_memory_bytes": int(description.dedicated_system),
                    "shared_system_memory_limit_bytes": int(description.shared_system),
                    "source": "windows_dxgi", "device_kind": "gpu",
                    "driver_version": driver_version,
                    "driver_version_source": "IDXGIAdapter.CheckInterfaceSupport(IDXGIDevice)" if driver_version != "unknown" else None,
                    "driver_version_query_hresult": f"0x{driver_code & 0xffffffff:08x}",
                })
            finally:
                _com_method(adapter, 2, ctypes.c_uint32)(adapter)
        else:
            result["errors"].append("DXGI enumeration exceeded the bounded adapter count")
        if result["errors"]:
            result["complete"] = False
    except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        result["errors"].append(str(exc))
    finally:
        if factory:
            _com_method(factory, 2, ctypes.c_uint32)(factory)
    return result


def _linux_inventory(root: Path = Path("/sys")) -> dict[str, Any]:
    result: dict[str, Any] = {"devices": [], "complete": False, "source": "linux_sysfs", "errors": []}
    pci = root / "bus/pci/devices"
    drm = root / "class/drm"
    seen: set[str] = set()
    try:
        paths = list(pci.iterdir()) if pci.is_dir() else []
        if not pci.is_dir() and not drm.is_dir():
            raise OSError("PCI and DRM sysfs inventories are unavailable")
        if len(paths) > 4096:
            raise OSError("PCI sysfs inventory exceeds the bounded device count")
        for path in paths:
            try:
                kind = int((path / "class").read_text().strip(), 16)
                if kind >> 16 != 3:
                    continue
                vendor = int((path / "vendor").read_text().strip(), 16)
                device = int((path / "device").read_text().strip(), 16)
                bus = normalize_pci_bus_id(path.name)
                identity = f"pci:{bus}" if bus else f"sysfs:{path.resolve()}"
                seen.add(identity)
                result["devices"].append({"vendor": _VENDORS.get(vendor, "unknown"),
                    "device": f"{_VENDORS.get(vendor, 'Unknown vendor')} PCI graphics device {vendor:04x}:{device:04x}",
                    "device_id": identity, "pci_bus_id": bus, "vendor_id": f"{vendor:04x}",
                    "pci_device_id": f"{device:04x}", "source": "linux_sysfs_pci", "device_kind": "gpu"})
            except (OSError, ValueError) as exc:
                result["errors"].append(f"{path.name}: {exc}")
        if drm.is_dir():
            cards = list(drm.iterdir())
            if len(cards) > 4096:
                raise OSError("DRM sysfs inventory exceeds the bounded entry count")
            for card in cards:
                if not re.fullmatch(r"card\d+", card.name):
                    continue
                try:
                    device = (card / "device").resolve(strict=True)
                    bus = normalize_pci_bus_id(device.name)
                    identity = f"pci:{bus}" if bus else f"sysfs:{device}"
                    if identity in seen:
                        continue
                    seen.add(identity)
                    result["devices"].append({"vendor": "unknown", "device": f"DRM graphics device {card.name}",
                        "device_id": identity, "pci_bus_id": bus, "source": "linux_sysfs_drm", "device_kind": "gpu"})
                except OSError as exc:
                    result["errors"].append(f"{card.name}: {exc}")
        result["complete"] = not result["errors"]
    except OSError as exc:
        result["errors"].append(str(exc))
    return result


def _mac_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {"devices": [], "complete": False, "source": "macos_system_profiler", "errors": []}
    try:
        queried = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                                 capture_output=True, text=True, timeout=4, check=False)
        if queried.returncode != 0:
            raise RuntimeError(queried.stderr.strip() or f"system_profiler exited {queried.returncode}")
        displays = json.loads(queried.stdout).get("SPDisplaysDataType")
        if not isinstance(displays, list):
            raise ValueError("system_profiler returned no display inventory array")
        for entry in displays[:256]:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("sppci_model", entry.get("_name", "macOS GPU (identity unknown)")))
            vendor = next((brand for brand in ("Apple", "AMD", "Intel", "NVIDIA") if brand.lower() in name.lower()), "unknown")
            result["devices"].append({"vendor": vendor, "device": name,
                "source": "macos_system_profiler", "device_kind": "gpu", "properties": entry})
        result["complete"] = len(displays) <= 256
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


def probe_gpu_inventory(*, refresh: bool = False) -> dict[str, Any]:
    """Cached, read-only OS enumeration; unavailable enumeration remains unknown."""
    global _CACHE, _CACHE_AT
    with _CACHE_LOCK:
        now = time.monotonic()
        if not refresh and _CACHE is not None and now - _CACHE_AT < 5:
            return copy.deepcopy(_CACHE)
        system = platform.system()
        if system == "Windows":
            result = _windows_inventory()
        elif system == "Linux":
            result = _linux_inventory()
        elif system == "Darwin":
            result = _mac_inventory()
        else:
            result = {"devices": [], "complete": False, "source": "unsupported_os_inventory",
                      "errors": [f"OS inventory is not implemented for {system}"]}
        result["scope"] = "OS graphics-device inventory; no compute runtime or execution validation"
        result["elapsed_seconds"] = time.monotonic() - now
        _CACHE, _CACHE_AT = copy.deepcopy(result), now
        return copy.deepcopy(result)


def device_identity_keys(evidence: dict[str, Any]) -> set[str]:
    def normalized(field: str, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).lower()
        if field == "uuid":
            raw = text.removeprefix("gpu-").removeprefix("mig-")
            try:
                parsed = uuid.UUID(raw)
                return str(parsed) if parsed.int else None
            except ValueError:
                # Older MIG monitoring IDs include the parent UUID plus GPU
                # and compute-instance IDs. Keep their entire scoped identity.
                return text if re.fullmatch(r"mig-gpu-[0-9a-f-]{36}/\d+/\d+", text) else None
        if field == "luid":
            return text if re.fullmatch(r"[0-9a-f]{16}", text) and int(text, 16) else None
        if field == "registry_id":
            return text if text.isdigit() and int(text) > 0 else None
        return None
    keys: set[str] = set()
    bus = normalize_pci_bus_id(evidence.get("pci_bus_id"))
    if bus:
        keys.add("pci:" + bus)
    for field, prefix in (("luid", "luid:"), ("uuid", "uuid:"), ("registry_id", "metal_registry:")):
        value = normalized(field, evidence.get(field))
        if value is not None:
            keys.add(prefix + value)
    identity = evidence.get("device_id")
    if isinstance(identity, str) and identity.startswith(("pci:", "luid:", "uuid:", "metal_registry:", "sysfs:")):
        prefix, _, value = identity.partition(":")
        if prefix == "pci":
            value = normalize_pci_bus_id(value)
        elif prefix in {"luid", "uuid", "metal_registry"}:
            value = normalized("registry_id" if prefix == "metal_registry" else prefix, value)
        if value:
            keys.add(prefix + ":" + value.lower())
    return keys


def merge_inventory_capabilities(caps: list[HardwareCapabilities], inventory: dict[str, Any]) -> list[HardwareCapabilities]:
    """Enrich exact physical matches; retain unidentified inventory separately."""
    cpu = next((cap for cap in caps if cap.vendor == "CPU"), None)
    if cpu is not None:
        cpu.extra["hardware_inventory"] = {key: value for key, value in inventory.items() if key != "devices"}
    for record in inventory.get("devices", []):
        keys = device_identity_keys(record)
        matches = [cap for cap in caps if keys and keys.intersection(device_identity_keys(cap.extra))]
        if matches:
            for cap in matches:
                cap.extra["os_inventory"] = record
                if cap.driver_version in {"unknown", "", None} and record.get("driver_version", "unknown") != "unknown":
                    cap.driver_version = record["driver_version"]
                    cap.extra["driver_version_source"] = record.get("driver_version_source")
                if cap.device.lower().startswith(("unknown", "gpu ", "intel pci", "nvidia pci", "amd pci")):
                    cap.device = record["device"]
                cap.extra.update({key: record[key] for key in ("device_id", "luid", "pci_bus_id", "directml_device_index") if key in record})
            continue
        target = {"NVIDIA": "cuda_unknown", "AMD": "rocm_unknown", "Intel": "openvino_gpu"}.get(record["vendor"], "gpu_unknown")
        caps.append(HardwareCapabilities(vendor=record["vendor"], device=record["device"], architecture="unknown",
            target_id=target, driver_version=record.get("driver_version", "unknown"),
            memory_bytes=int(record.get("memory_bytes", 0)), memory_kind=record.get("memory_kind"),
            memory_source=record.get("memory_source"), hardware_present=True,
            runtime_available=None, implemented=False, available=False, backend_available=False,
            unavailable_reason="GPU hardware detected by the OS; no matching usable Aether execution runtime was confirmed",
            extra={**record, "detection_method": record["source"], "inventory_only": True,
                   "device_index": None, "execution_device": None, "raw_architecture": None,
                   "memory_known": "memory_bytes" in record,
                   "precision_evidence": "unconfirmed"}))
    return caps


def summarize_gpu_devices(caps: list[HardwareCapabilities]) -> dict[str, Any]:
    """Count strong identities; cross-provider ambiguous counts stay unconfirmed."""
    gpu_prefixes = ("cuda", "rocm", "amd_mi", "metal", "openvino_gpu", "gpu_")
    gpu_caps = [cap for cap in caps if cap.target_id.startswith(gpu_prefixes)
                and (cap.hardware_present is True or cap.available) and not cap.extra.get("catalog_only")]
    groups: list[dict[str, Any]] = []
    for cap in gpu_caps:
        keys = device_identity_keys(cap.extra)
        matches = [group for group in groups if keys.intersection(group["keys"])]
        if not matches:
            groups.append({"keys": set(keys), "caps": [cap]})
        else:
            match = matches[0]
            match["keys"].update(keys)
            match["caps"].append(cap)
            for other in matches[1:]:
                match["keys"].update(other["keys"])
                match["caps"].extend(other["caps"])
                groups.remove(other)
    # Exact identities can deduplicate providers. If identities are missing or
    # incomparable, use the largest single-provider count as an honest lower
    # bound rather than declaring every duplicate provider row a separate GPU.
    def physical_enumeration(group: dict[str, Any]) -> bool:
        return any(cap.extra.get("inventory_only") or cap.extra.get("detection_method") == "nvml_telemetry_only"
                   for cap in group["caps"])
    source_counts: dict[str, int] = {}
    for group in groups:
        if not physical_enumeration(group):
            continue
        source = str(group["caps"][0].extra.get("detection_method", "device_evidence"))
        source_counts[source] = source_counts.get(source, 0) + 1
    def count_vendor(vendor_groups: list[dict[str, Any]]) -> tuple[int, bool]:
        sources: dict[str, int] = {}
        for group in vendor_groups:
            if not physical_enumeration(group):
                continue
            source = str(group["caps"][0].extra.get("detection_method", "device_evidence"))
            sources[source] = sources.get(source, 0) + 1
        known = len(vendor_groups) <= 1 or (all(physical_enumeration(group) for group in vendor_groups) and len(sources) == 1)
        if len(vendor_groups) > 1 and all(group["keys"] for group in vendor_groups):
            namespaces = [set(key.split(":", 1)[0] for key in group["keys"]) & {"pci", "luid", "metal_registry", "sysfs"}
                          for group in vendor_groups]
            known = known or bool(set.intersection(*namespaces))
        # OpenVINO may expose one GPU through multiple OpenCL platforms and
        # CUDA/HIP may enumerate logical compute partitions. Runtime ordinals
        # or UUIDs alone therefore do not establish multiple physical GPUs.
        return (len(vendor_groups) if known else max(1, max(sources.values(), default=0))), known
    vendors: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        named = {cap.vendor for cap in group["caps"] if cap.vendor in {"NVIDIA", "AMD", "Intel", "Apple", "Qualcomm"}}
        vendor = next(iter(named)) if len(named) == 1 else "unknown"
        vendors.setdefault(vendor, []).append(group)
    counts = {vendor: count_vendor(rows) for vendor, rows in vendors.items()}
    named_minimum = sum(count for vendor, (count, _) in counts.items() if vendor != "unknown")
    minimum = max(named_minimum, max(source_counts.values(), default=0), counts.get("unknown", (0, False))[0])
    known = all(confirmed for _, confirmed in counts.values()) and not ("unknown" in vendors and len(vendors) > 1)
    if not groups:
        inventory = next((cap.extra.get("hardware_inventory") for cap in caps if "hardware_inventory" in cap.extra), None)
        known = bool(inventory and inventory.get("complete"))
    elif known:
        minimum = len(groups)
    return {"count": minimum, "count_known": known, "maximum_count": len(groups),
            "devices": [group["caps"][0].to_dict() for group in groups]}
