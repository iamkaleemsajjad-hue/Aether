"""
Aether Runtime — Real Hardware Detection Pipeline.

Implements the detection chain required by PRD §41:

  detect_host()
      ↓
  detect_accelerator()
      ↓
  detect_driver()
      ↓
  detect_runtime()
      ↓
  detect_memory()
      ↓
  detect_precision()
      ↓
  select_backend()
      ↓
  HardwareCapabilities

Key rules (PRD §4, §57):
  - Never return available=True without real detection evidence.
  - Never fabricate driver versions, memory sizes, or capability flags.
  - CPU host is always available. GPU/TEE/NPU require runtime confirmation.
"""

from __future__ import annotations

import platform
import time
from typing import Any

from aether.backends.capabilities import (
    HardwareCapabilities,
    ValidationResult,
    MemoryInfo,
    PowerInfo,
    DeviceInfo,
    _host_memory_bytes,
)
from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_all_capabilities() -> list[HardwareCapabilities]:
    """Detect capabilities for every hardware backend on this host.

    Returns a list of HardwareCapabilities objects, one per detected or
    registered backend. Backends that are not available are included with
    available=False and an unavailable_reason.

    This is the master detection function called by ``aether hardware detect``.
    Detection order matches backend priority: best accelerator first, then
    specialized NPU/ANE pipelines, then CPU fallback.
    """
    caps: list[HardwareCapabilities] = []

    # 1. CPU (always available — never raises)
    caps.append(detect_cpu())

    # 2. CUDA (NVIDIA GPUs) — one capability per physical device
    caps.extend(detect_cuda_devices())

    # 3. ROCm (AMD GPUs) — one capability per physical device
    caps.extend(detect_rocm_devices())

    # 4. Metal (all native devices, with independent identities)
    caps.extend(detect_metal_devices())

    # 5. Apple Neural Engine (ANE) — separate from Metal GPU, always present
    #    on Apple Silicon but requires Core ML for any execution.
    caps.append(detect_apple_ane())

    # 6. OpenVINO / Intel NPU and iGPU
    caps.extend(detect_openvino_devices())

    # 7. DirectML — Windows vendor-agnostic GPU (any GPU on Windows)
    caps.append(detect_directml())

    # 8. Qualcomm NPU via QNN / SNPE (Snapdragon ARM Windows + Android)
    caps.append(detect_qualcomm_npu())

    # 9. Keep planning profiles separate from measured device evidence.
    caps.extend(_unavailable_vendor_targets())
    from aether.backends.hardware_inventory import merge_inventory_capabilities, probe_gpu_inventory
    return merge_inventory_capabilities(caps, probe_gpu_inventory())


def detect_cpu() -> HardwareCapabilities:
    """Detect CPU capabilities for the current host."""
    return HardwareCapabilities.cpu_host()


def detect_cuda_devices() -> list[HardwareCapabilities]:
    """Checked CUDA Driver API enumeration; telemetry alone is insufficient."""
    from aether.backends.cuda_detection import detect_cuda_devices as probe
    return probe()


def detect_rocm_devices() -> list[HardwareCapabilities]:
    """Enumerate HIP compute devices; monitoring APIs alone are not a runtime."""
    import sys

    devices, hip_error = _detect_hip_runtime_devices()
    if devices:
        return devices
    dependency = _vendor_package_present("torch")
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        try:
            for index, processor in enumerate(amdsmi.amdsmi_get_processor_handles()):
                try:
                    name = str(amdsmi.amdsmi_get_gpu_board_info(processor).get("product_name", "AMD GPU"))
                    memory = int(amdsmi.amdsmi_get_gpu_memory_total(processor, amdsmi.AmdSmiMemoryType.VRAM))
                except Exception:
                    name, memory = "AMD GPU", 0
                gfx = _reported_gfx_arch(name)
                target = _rocm_target_id(gfx)
                devices.append(HardwareCapabilities(
                    vendor="AMD", device=name, architecture=gfx,
                    target_id=target, memory_bytes=max(0, memory),
                    supports_fp32=False, implemented=_known_target(target), available=False,
                    hardware_present=True, runtime_available=False, backend_available=False,
                    backend_dependencies={"torch": dependency},
                    memory_kind="device_vram" if memory > 0 else None,
                    memory_source="amdsmi" if memory > 0 else None,
                    unavailable_reason=hip_error,
                    extra={"device_index": None, "telemetry_index": index, "execution_device": None,
                           "detection_method": "amdsmi", "monitoring_only": True,
                           "inferred_target_id": _rocm_target_id(name), "raw_architecture": gfx},
                ))
        finally:
            amdsmi.amdsmi_shut_down()
        if devices:
            return devices
    except Exception:
        pass
    if sys.platform == "linux":
        try:
            import subprocess
            import json
            result = subprocess.run(["rocm-smi", "--showproductname", "--json"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for index, info in enumerate(json.loads(result.stdout).values()):
                    if not isinstance(info, dict):
                        continue
                    name = str(info.get("Card Series", "AMD GPU"))
                    gfx = _reported_gfx_arch(str(info.get("GFX Version", "")))
                    target = _rocm_target_id(gfx)
                    devices.append(HardwareCapabilities(
                        vendor="AMD", device=name, architecture=gfx,
                        target_id=target,
                        supports_fp32=False, implemented=_known_target(target), available=False,
                        hardware_present=True, runtime_available=False, backend_available=False,
                        backend_dependencies={"torch": dependency}, unavailable_reason=hip_error,
                        extra={"device_index": None, "telemetry_index": index, "execution_device": None,
                               "detection_method": "rocm_smi", "raw_architecture": gfx,
                               "inferred_target_id": _rocm_target_id(name),
                               "monitoring_only": True, "monitoring_properties": info},
                    ))
                if devices:
                    return devices
        except Exception:
            pass
        from pathlib import Path
        if Path("/dev/kfd").exists():
            return [HardwareCapabilities(
                vendor="AMD", device="AMD KFD device (identity not enumerated)",
                architecture="gfx_unknown", target_id="rocm_unknown", supports_fp32=False,
                implemented=False, available=False, hardware_present=None,
                runtime_available=False, backend_available=False, unavailable_reason=hip_error,
                extra={"detection_method": "kfd_node", "monitoring_only": True},
            )]
    return [HardwareCapabilities.unavailable(
        vendor="AMD", device="unknown", architecture="unknown", target_id="rocm_unknown",
        reason=hip_error, implemented=False,
    )]


def _vendor_package_present(package: str) -> bool:
    """Inspect optional package metadata without importing an execution framework."""
    import importlib.util
    import sys
    if package in sys.modules:
        return sys.modules[package] is not None
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _known_target(target: str) -> bool:
    from aether.core.constants import SUPPORTED_TARGET_IDS
    return target in SUPPORTED_TARGET_IDS


def _load_hip_runtime() -> Any:
    """Load AMD HIP through its stable public C ABI."""
    import ctypes
    import ctypes.util
    import os
    import sys
    from pathlib import Path
    name = "amdhip64.dll" if sys.platform == "win32" else "libamdhip64.so"
    candidates = [name]
    discovered = ctypes.util.find_library("amdhip64")
    if discovered:
        candidates.append(discovered)
    for variable in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(variable)
        if root:
            candidates.append(str(Path(root) / ("bin" if sys.platform == "win32" else "lib") / name))
    if sys.platform == "linux":
        candidates.append("/opt/rocm/lib/libamdhip64.so")
    for candidate in dict.fromkeys(candidates):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            pass
    raise OSError("AMD HIP runtime library could not be loaded")


def _detect_hip_runtime_devices() -> tuple[list[HardwareCapabilities], str]:
    """Query typed C APIs without guessing a versioned hipDeviceProp_t layout."""
    import ctypes
    devices: list[HardwareCapabilities] = []
    try:
        hip = _load_hip_runtime()
        signatures = {
            "hipInit": [ctypes.c_uint],
            "hipGetDeviceCount": [ctypes.POINTER(ctypes.c_int)],
            "hipDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "hipDeviceGetName": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
            "hipDeviceTotalMem": [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int],
        }
        for name, signature in signatures.items():
            function = getattr(hip, name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        status = hip.hipInit(0)
        if status != 0:
            return devices, f"HIP initialization failed (status {status}); monitoring is not execution readiness"
        count = ctypes.c_int()
        status = hip.hipGetDeviceCount(ctypes.byref(count))
        if status != 0 or count.value <= 0:
            return devices, f"HIP did not enumerate a compute device (status {status}, count {count.value})"
        runtime_version = "unknown"
        raw_version = None
        try:
            hip.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
            hip.hipRuntimeGetVersion.restype = ctypes.c_int
            version = ctypes.c_int()
            if hip.hipRuntimeGetVersion(ctypes.byref(version)) == 0:
                raw_version = version.value
                runtime_version = f"HIP API version {version.value}"
        except AttributeError:
            pass
        dependency = _vendor_package_present("torch")
        if count.value > 256:
            return [], f"HIP reported {count.value} devices; bounded enumeration permits at most 256"
        for index in range(count.value):
            device = ctypes.c_int()
            device_status = hip.hipDeviceGet(ctypes.byref(device), index)
            if device_status != 0:
                continue
            buffer = ctypes.create_string_buffer(256)
            name = "AMD HIP compute device (identity unknown)"
            name_status = hip.hipDeviceGetName(buffer, len(buffer), device.value)
            if name_status == 0:
                name = buffer.value.decode(errors="replace") or name
            memory = ctypes.c_size_t()
            memory_status = hip.hipDeviceTotalMem(ctypes.byref(memory), device.value)
            known_memory = memory_status == 0 and memory.value > 0
            identity = {}
            try:
                hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
                bus = ctypes.create_string_buffer(32)
                if hip.hipDeviceGetPCIBusId(bus, len(bus), index) == 0:
                    from aether.backends.hardware_inventory import normalize_pci_bus_id
                    identity["pci_bus_id"] = normalize_pci_bus_id(bus.value.decode(errors="replace"))
            except AttributeError:
                pass
            try:
                hip.hipDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
                hip.hipDeviceGetUuid.restype = ctypes.c_int
                identifier = (ctypes.c_ubyte * 16)()
                if hip.hipDeviceGetUuid(identifier, index) == 0:
                    import uuid
                    identity["uuid"] = str(uuid.UUID(bytes=bytes(identifier)))
            except AttributeError:
                pass
            gfx = _reported_gfx_arch(name)
            target = _rocm_target_id(gfx)
            devices.append(HardwareCapabilities(
                vendor="AMD", device=name, architecture=gfx, target_id=target,
                runtime_version=runtime_version, memory_bytes=int(memory.value) if known_memory else 0,
                memory_kind="device_vram" if known_memory else None,
                memory_source="hipDeviceTotalMem" if known_memory else None,
                supports_fp32=True, warp_or_wavefront_size=0, implemented=_known_target(target), available=True,
                hardware_present=True, runtime_available=True,
                backend_dependencies={"torch": dependency}, backend_available=None if dependency and _known_target(target) else False,
                extra={**identity, "device_index": index, "execution_device": f"cuda:{index}",
                       "device_index_scope": "HIP compute-runtime visible ordinal", "detection_method": "hip_runtime_api",
                       "device_handle": device.value, "name_query_status": name_status,
                       "memory_query_status": memory_status,
                       "hip_runtime_version_raw": raw_version, "wavefront_size_known": False,
                       "raw_architecture": gfx, "inferred_architecture": _infer_gfx_arch(name),
                       "inferred_target_id": _rocm_target_id(name),
                       "architecture_source": "reported_gfx_identifier" if gfx != "gfx_unknown" else "unknown",
                       "precision_execution": {"fp32": "optional PyTorch HIP AEG adapter",
                                               "fp16": "backend dependent; not validated",
                                               "bf16": "backend dependent; not validated"}},
            ))
        return devices, "" if devices else "HIP enumerated devices, but no device handle could be queried"
    except Exception as exc:
        return devices, f"HIP runtime/device enumeration unavailable: {exc}"


def detect_metal_devices() -> list[HardwareCapabilities]:
    """Enumerate native Metal devices without treating PyTorch's default as all GPUs."""
    import re
    import sys
    if sys.platform != "darwin":
        return [HardwareCapabilities.unavailable(
            vendor="Metal", device="Metal GPU", architecture="unknown", target_id="metal",
            reason="Metal requires macOS; current platform is not macOS", implemented=True,
        )]
    evidence = _query_native_metal_devices()
    dependency = {"torch": _vendor_package_present("torch"), "mlx": _vendor_package_present("mlx")}
    caps = []
    for index, record in enumerate(evidence["devices"]):
        name = record["device"]
        match = re.search(r"\bApple\s+M(\d+)\b", name, flags=re.IGNORECASE)
        apple_silicon = bool(match and platform.machine().lower() in {"arm64", "aarch64"})
        target = ("metal_m1" if int(match.group(1)) <= 2 else "metal_m3") if match and apple_silicon else "metal"
        unified = record.get("unified_memory") is True
        # Host RAM is physical shared memory, not a dedicated GPU allocation.
        # recommendedMaxWorkingSetSize is a recommendation, never total VRAM.
        memory = _host_memory_bytes() if unified else 0
        default = record.get("system_default") is True
        caps.append(HardwareCapabilities(
            vendor="Apple" if apple_silicon else next((vendor for vendor in ("AMD", "Intel", "NVIDIA") if vendor.lower() in name.lower()), "Metal"),
            device=name, architecture=name, target_id=target,
            runtime_version=f"Metal (macOS {platform.mac_ver()[0]})",
            memory_bytes=memory, memory_kind="shared_host" if memory else None,
            memory_source="host_physical_memory" if memory else None,
            supports_fp32=True, supports_fp16=True, supports_bf16=False,
            supports_unified_memory=unified, implemented=True, available=True,
            hardware_present=True, runtime_available=True, backend_dependencies=dependency,
            backend_available=None if default and any(dependency.values()) else False,
            extra={**record, "device_index": index, "execution_device": "mps" if default else None,
                   "device_index_scope": "MTLCopyAllDevices enumeration; MPS exposes the system default only",
                   "detection_method": "metal_device_api", "apple_silicon": apple_silicon,
                   "precision_scope": "runtime_formats", "simd_group_size_known": False,
                   "precision_evidence": "Metal shading language float/half; BF16 not queried",
                   "precision_execution": "Optional MPS/MLX adapter; discovery does not execute inference",
                   "enumeration_errors": evidence["errors"]},
        ))
    if caps:
        # Compatibility singular APIs and automatic MPS selection choose its
        # default device, preserving the native enumeration index in metadata.
        caps.sort(key=lambda cap: not cap.extra.get("system_default", False))
        return caps
    return [HardwareCapabilities.unavailable(
        vendor="Metal", device="Metal GPU (not enumerated)", architecture="unknown", target_id="metal",
        reason="Metal device enumeration failed: " + "; ".join(evidence["errors"]), implemented=True,
    )]


def _query_native_metal_devices() -> dict[str, Any]:
    """Typed public Metal/Objective-C queries with owned-array lifetime handling."""
    import ctypes
    result: dict[str, Any] = {"devices": [], "errors": [], "complete": False}
    array = None
    default = None
    core = None
    message = None
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        def message(receiver: Any, selector: bytes, output: Any) -> Any:
            method = ctypes.CFUNCTYPE(output, ctypes.c_void_p, ctypes.c_void_p)(address)
            return method(receiver, objc.sel_registerName(selector))
        responds = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
        def supports(receiver: Any, selector: bytes) -> bool:
            return bool(receiver and responds(receiver, objc.sel_registerName(b"respondsToSelector:"), objc.sel_registerName(selector)))
        metal.MTLCopyAllDevices.argtypes = []
        metal.MTLCopyAllDevices.restype = ctypes.c_void_p
        metal.MTLCreateSystemDefaultDevice.argtypes = []
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        core.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        core.CFArrayGetCount.restype = ctypes.c_long
        core.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
        core.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        core.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_uint32]
        core.CFStringGetCString.restype = ctypes.c_bool
        core.CFRelease.argtypes = [ctypes.c_void_p]
        core.CFRelease.restype = None
        array = metal.MTLCopyAllDevices()
        default = metal.MTLCreateSystemDefaultDevice()
        if not array:
            raise RuntimeError("MTLCopyAllDevices returned no device array")
        count = int(core.CFArrayGetCount(array))
        if count < 0 or count > 256:
            raise RuntimeError(f"Metal device count {count} exceeds the bounded enumeration")
        default_id = int(message(default, b"registryID", ctypes.c_uint64)) if supports(default, b"registryID") else None
        for index in range(count):
            device = core.CFArrayGetValueAtIndex(array, index)
            if not device:
                result["errors"].append(f"Metal device {index} returned a null object")
                continue
            identifier = int(message(device, b"registryID", ctypes.c_uint64)) if supports(device, b"registryID") else None
            name_buffer = ctypes.create_string_buffer(1024)
            native_name = message(device, b"name", ctypes.c_void_p)
            name = (name_buffer.value.decode("utf-8", errors="replace")
                    if native_name and core.CFStringGetCString(native_name, name_buffer, len(name_buffer), 0x08000100)
                    else f"Metal GPU {index} (name query unavailable)")
            unified = None
            working_set = None
            # hasUnifiedMemory was introduced on macOS 10.15. Check selector
            # availability before invoking it on older supported Metal hosts.
            for selector, field, output in ((b"hasUnifiedMemory", "unified", ctypes.c_bool),
                                             (b"recommendedMaxWorkingSetSize", "working_set", ctypes.c_uint64)):
                if supports(device, selector):
                    value = message(device, selector, output)
                    if field == "unified":
                        unified = bool(value)
                    else:
                        working_set = int(value)
            result["devices"].append({"device": name, "registry_id": identifier,
                "device_id": f"metal_registry:{identifier}" if identifier is not None else None,
                "system_default": (identifier == default_id if identifier is not None and default_id is not None else device == default),
                "unified_memory": unified, "recommended_working_set_bytes": working_set,
                "recommended_working_set_scope": "Metal allocation recommendation; not physical VRAM"})
        result["complete"] = not result["errors"]
    except (OSError, AttributeError, ValueError, TypeError, RuntimeError) as exc:
        result["errors"].append(str(exc))
    finally:
        if array and core is not None:
            core.CFRelease(array)
        if default and message is not None:
            message(default, b"release", None)
    return result


def detect_metal() -> HardwareCapabilities:
    """Compatibility view of the actual system-default Metal device."""
    return detect_metal_devices()[0]


def detect_openvino() -> HardwareCapabilities:
    """Query the NPU plugin; emitted-plan support is not an Aether generation adapter."""
    return _detect_openvino_device("NPU", "openvino_npu")


def detect_openvino_gpu() -> HardwareCapabilities:
    """Query the GPU plugin's actual precision and memory properties."""
    return _detect_openvino_device("GPU", "openvino_gpu")


def detect_openvino_devices() -> list[HardwareCapabilities]:
    """Enumerate every Intel runtime GPU/NPU using its exact plugin device ID."""
    try:
        import openvino
        core = openvino.Core()
    except Exception:
        return _detect_openvino_devices("GPU", "openvino_gpu") + _detect_openvino_devices("NPU", "openvino_npu")
    return (_detect_openvino_devices("GPU", "openvino_gpu", core=core, module=openvino)
            + _detect_openvino_devices("NPU", "openvino_npu", core=core, module=openvino))


def _detect_openvino_device(device_class: str, target_id: str) -> HardwareCapabilities:
    """Compatibility wrapper returning the first device, or unavailable evidence."""
    return _detect_openvino_devices(device_class, target_id)[0]


def _detect_openvino_devices(device_class: str, target_id: str, *, core: Any = None, module: Any = None) -> list[HardwareCapabilities]:
    try:
        if core is None:
            import openvino
            module = openvino
            core = openvino.Core()
        devices = [str(device) for device in core.available_devices]
        if len(devices) > 256:
            raise RuntimeError(f"OpenVINO reported {len(devices)} devices; bounded enumeration permits at most 256")
        selected_devices = [device for device in devices if device.split(".", 1)[0] == device_class]
        if not selected_devices:
            return [HardwareCapabilities.unavailable(
                vendor="Intel", device=f"Intel {device_class}", architecture="unknown", target_id=target_id,
                reason=f"OpenVINO is installed, but its device list contains no {device_class}; available devices: {devices}",
                implemented=False,
            )]
        caps = []
        for ordinal, selected in enumerate(selected_devices):
            properties = {}
            def query(name: str, default: Any = None) -> Any:
                try:
                    value = core.get_property(selected, name)
                    properties[name] = _property_evidence(value)
                    return value
                except Exception:
                    return default
            name = str(query("FULL_DEVICE_NAME", f"Intel {device_class} (identity unknown)"))
            architecture = str(query("DEVICE_ARCHITECTURE", "unknown"))
            driver = str(query("DRIVER_VERSION", query("GPU_DRIVER_VERSION" if device_class == "GPU" else "NPU_DRIVER_VERSION", "unknown")))
            advertised = query("OPTIMIZATION_CAPABILITIES", [])
            precisions = {str(value).upper() for value in advertised} if isinstance(advertised, (list, tuple, set)) else set()
            device_type = str(query("DEVICE_TYPE", "unknown")).lower()
            memory = query("GPU_DEVICE_TOTAL_MEM_SIZE") if device_class == "GPU" else query("NPU_DEVICE_TOTAL_MEM_SIZE", query("DEVICE_TOTAL_MEM_SIZE"))
            try:
                memory = max(0, int(memory)) if memory is not None and not isinstance(memory, bool) else 0
            except (ValueError, TypeError, OverflowError):
                memory = 0
            integrated = "integrated" in device_type
            suffix = selected.partition(".")[2]
            device_index = int(suffix) if suffix.isdigit() else ordinal
            identity = {}
            for prop, field, length in (("DEVICE_LUID", "luid", 8), ("DEVICE_UUID", "uuid", 16)):
                raw = query(prop)
                value = getattr(raw, field, raw)
                if isinstance(value, (list, tuple, bytes, bytearray)):
                    try:
                        value = bytes(value).hex()
                    except (ValueError, TypeError):
                        value = None
                if isinstance(value, str):
                    import re
                    if re.fullmatch(r"[0-9a-fA-F]{" + str(length * 2) + r"}", value):
                        identity[field] = value.lower()
            pci = query("DEVICE_PCI_INFO")
            if pci is not None:
                try:
                    values = [pci[key] if isinstance(pci, dict) else getattr(pci, key) for key in ("domain", "bus", "device", "function")]
                    from aether.backends.hardware_inventory import normalize_pci_bus_id
                    identity["pci_bus_id"] = normalize_pci_bus_id("{:04x}:{:02x}:{:02x}.{:x}".format(*map(int, values)))
                except (KeyError, AttributeError, TypeError, ValueError):
                    pass
            vendor = next((brand for brand in ("Intel", "AMD", "NVIDIA") if brand.lower() in name.lower()), "OpenVINO")
            caps.append(HardwareCapabilities(
            vendor=vendor, device=name, architecture=architecture, target_id=target_id,
            driver_version=driver, runtime_version=str(getattr(module, "__version__", "unknown")),
            memory_bytes=memory, memory_kind=("shared_allocation_limit" if integrated else "device_memory") if memory else None,
            memory_source="openvino_device_property" if memory else None,
            supports_fp32="FP32" in precisions, supports_fp16="FP16" in precisions,
            supports_bf16="BF16" in precisions, supports_fp8="FP8" in precisions,
            supports_fp4="FP4" in precisions, supports_int8="INT8" in precisions,
            supports_int4="INT4" in precisions, supports_unified_memory=integrated,
            implemented=False, available=True, hardware_present=True, runtime_available=True,
            backend_dependencies={"openvino": True}, backend_available=False,
            extra={**identity, "available_devices": devices, "device_id": selected,
                   "device_index": device_index, "execution_device": f"openvino:{selected}",
                   "device_index_scope": "OpenVINO plugin device ID", "raw_architecture": architecture,
                   "detection_method": "openvino_core", "precision_scope": "runtime_formats",
                   "queried_properties": properties, "optimization_capabilities": sorted(precisions),
                   "precision_evidence": "OpenVINO OPTIMIZATION_CAPABILITIES",
                   "precision_execution": {"status": "vendor runtime formats; Aether generation runner unavailable"},
                   "execution_adapter_status": "OpenVINO graph compilation exists; task-specific generation runner is not implemented"},
            ))
        return caps
    except ImportError:
        reason = "OpenVINO package not installed"
    except Exception as exc:
        reason = f"OpenVINO {device_class} enumeration failed: {exc}"
    return [HardwareCapabilities.unavailable(
        vendor="Intel", device=f"Intel {device_class}", architecture="unknown", target_id=target_id,
        reason=reason, implemented=False,
    )]


def _property_evidence(value: Any) -> Any:
    """Keep vendor-query evidence JSON serializable without inventing fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _property_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_property_evidence(item) for item in value]
    return str(value)


def get_memory_info(device_type: str = "cpu", device_index: int = 0) -> MemoryInfo:
    """Get current memory information for a device.

    Args:
        device_type: "cpu", "cuda", "mps", "hip"
        device_index: GPU device index (for CUDA/HIP)

    Returns:
        MemoryInfo with real measurements where available.
    """
    if device_type == "cuda":
        # Use pynvml (no torch required)
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return MemoryInfo(
                total_bytes=mem_info.total,
                free_bytes=mem_info.free,
                used_bytes=mem_info.used,
                device_type="cuda",
            )
        except Exception:  # noqa: BLE001
            pass

    if device_type == "cpu":
        total = _host_memory_bytes()
        try:
            import psutil
            vm = psutil.virtual_memory()
            return MemoryInfo(
                total_bytes=vm.total,
                free_bytes=vm.available,
                used_bytes=vm.used,
                device_type="cpu",
            )
        except ImportError:
            pass
        return MemoryInfo(
            total_bytes=total,
            free_bytes=0,
            used_bytes=0,
            device_type="cpu",
        )

    return MemoryInfo(total_bytes=0, free_bytes=0, used_bytes=0, device_type=device_type)


def get_power_info(device_type: str = "cpu", device_index: int = 0) -> PowerInfo:
    """Get power/energy information where hardware telemetry is available.

    CUDA: uses NVML (pynvml / nvidia-ml-py)
    ROCm: uses rocm_smi
    CPU: not supported at this level (uses TDP estimate elsewhere)
    """
    if device_type == "cuda":
        # Try nvidia-ml-py
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            return PowerInfo(
                power_draw_watts=mw / 1000.0,
                power_limit_watts=limit_mw / 1000.0,
                energy_source="nvml",
            )
        except Exception:  # noqa: BLE001
            pass

    return PowerInfo(
        power_draw_watts=None,
        power_limit_watts=None,
        energy_source="unavailable",
    )


def validate_backend_environment(
    target_id: str, *, execute: bool = False, backend_name: str | None = None,
    device_index: int = 0,
) -> ValidationResult:
    """Check environment; optionally run and verify a real offline AEG."""
    from aether.backends.hardware_validation import validate_backend_environment as validate
    return validate(target_id, execute=execute, backend_name=backend_name, device_index=device_index)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cuda_target_id(major: int, minor: int) -> str:
    """Select a compatible baseline target within the same CUDA major CC."""
    from aether.backends.cuda_detection import _target
    return _target(major, minor)


def _cuda_driver_version() -> str:
    """Return CUDA driver version string via pynvml (no torch required)."""
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        ver = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        if isinstance(ver, bytes):
            ver = ver.decode()
        return f"driver/{ver}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _has_nvlink(device_index: int) -> bool:
    """Check NVLink availability by device index via pynvml."""
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        result = _has_nvlink_nvml(handle)
        pynvml.nvmlShutdown()
        return result
    except Exception:  # noqa: BLE001
        pass
    return False


def _has_nvlink_nvml(handle: Any) -> bool:
    """Check NVLink availability given an already-opened pynvml device handle."""
    try:
        import pynvml  # type: ignore[import]  # noqa: PLC0415
        for link in range(6):
            try:
                state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                if state == pynvml.NVML_FEATURE_ENABLED:
                    return True
            except pynvml.NVMLError:
                break
    except Exception:  # noqa: BLE001
        pass
    return False


def inspect_torch_hip_devices(torch: Any) -> list[HardwareCapabilities]:
    """Inspect a caller-supplied framework's visible HIP devices, without execution.

    Automatic hardware discovery never imports PyTorch. Runtime selection and
    explicit validation may supply it to obtain the versioned GFX properties
    safely through PyTorch's own binding.
    """
    return _detect_rocm_via_hip(torch)


def _detect_rocm_via_hip(torch: Any) -> list[HardwareCapabilities]:
    """Inspect an already supplied HIP framework without fabricated fallback devices."""
    caps = []
    try:
        cuda = getattr(torch, "cuda", None)
        if not getattr(getattr(torch, "version", None), "hip", None) or cuda is None or not cuda.is_available():
            return caps
        device_count = int(cuda.device_count())
        if device_count < 0 or device_count > 256:
            return caps
        for i in range(device_count):
            cap = _rocm_cap_from_torch(i, str(cuda.get_device_name(i)), torch)
            if cap.available:
                caps.append(cap)
    except Exception:
        return []
    return caps


def _rocm_cap_from_torch(i: int, name: str, torch: Any) -> HardwareCapabilities:
    """Inspect a caller-supplied HIP framework; this helper never imports it."""
    cuda = getattr(torch, "cuda", None)
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    properties = None
    try:
        if 0 <= i < int(cuda.device_count()):
            properties = cuda.get_device_properties(i)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    raw_architecture = str(getattr(properties, "gcnArchName", "") or "")
    gfx = _reported_gfx_arch(raw_architecture)
    target = _rocm_target_id(gfx)
    try:
        available = bool(hip_version and properties is not None and cuda is not None and cuda.is_available())
    except (AttributeError, RuntimeError):
        available = False
    try:
        memory = max(0, int(getattr(properties, "total_memory", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        memory = 0
    try:
        warp_size = max(0, int(getattr(properties, "warp_size", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        warp_size = 0
    identity = {}
    raw_uuid = getattr(properties, "uuid", None)
    if raw_uuid is not None:
        identity["uuid"] = str(raw_uuid)
    raw_bus = getattr(properties, "pci_bus_id", None)
    if raw_bus is not None:
        from aether.backends.hardware_inventory import normalize_pci_bus_id
        normalized = normalize_pci_bus_id(raw_bus)
        if normalized:
            identity["pci_bus_id"] = normalized
    return HardwareCapabilities(
        vendor="AMD", device=name, architecture=gfx,
        target_id=target,
        runtime_version=str(hip_version or "unknown"), memory_bytes=memory,
        memory_kind="device_vram" if memory else None,
        memory_source="torch_device_properties" if memory else None,
        supports_fp32=available, warp_or_wavefront_size=warp_size,
        implemented=_known_target(target), available=available, hardware_present=True if properties else None,
        runtime_available=available, backend_dependencies={"torch": True}, backend_available=available and _known_target(target),
        unavailable_reason=None if available else "Caller-supplied PyTorch module has no available HIP runtime",
        extra={**identity, "device_index": i, "execution_device": f"cuda:{i}" if available else None,
               "device_index_scope": "PyTorch HIP visible ordinal", "raw_architecture": raw_architecture or None,
               "inferred_target_id": _rocm_target_id(name), "detection_method": "caller_supplied_torch_hip",
               "wavefront_size_known": warp_size > 0,
               "precision_execution": {"status": "PyTorch HIP available; arithmetic was not execution tested"}},
    )


def _infer_gfx_arch(device_name: str) -> str:
    """Preserve reported GFX IDs; device-name inference is explicitly tagged by callers."""
    import re
    raw = re.search(r"\bgfx[0-9a-f]+\b", device_name, flags=re.IGNORECASE)
    if raw:
        return raw.group(0).lower()
    normalized = device_name.upper()
    for identifier, gfx in (("MI300", "gfx942"), ("MI250", "gfx90a"), ("MI100", "gfx908"),
                            ("7900", "gfx1100"), ("6900", "gfx1030")):
        if identifier in normalized:
            return gfx
    return "gfx_unknown"


def _reported_gfx_arch(text: str) -> str:
    """A queried GFX identifier is evidence; product-name mapping is only a hint."""
    import re
    raw = re.search(r"\bgfx[0-9a-f]+\b", text, flags=re.IGNORECASE)
    return raw.group(0).lower() if raw else "gfx_unknown"


def _rocm_target_id(device_name: str) -> str:
    """Map recognized devices without relabeling unknown/older GPUs as CDNA3."""
    normalized = device_name.upper()
    gfx = _infer_gfx_arch(device_name)
    if "MI455" in normalized or "CDNA5" in normalized:
        return "rocm_cdna5_mi455x"
    if "MI350" in normalized or "CDNA4" in normalized or gfx == "gfx950":
        return "amd_mi350x"
    if "MI300" in normalized or "CDNA3" in normalized or gfx == "gfx942":
        return "rocm_cdna3"
    if "7900" in normalized or "RDNA3" in normalized or gfx.startswith("gfx11"):
        return "rocm_rdna3"
    if gfx != "gfx_unknown":
        return f"rocm_{gfx}"
    if "MI200" in normalized or "CDNA2" in normalized:
        return "rocm_cdna2"
    return "rocm_unknown"


def _unavailable_vendor_targets() -> list[HardwareCapabilities]:
    """Return stub capabilities for targets that have no runtime on this host."""
    stubs = [
        ("SiFive", "X160", "riscv", "riscv_sifive_x160"),
        ("MIPS/Wave", "S8200", "riscv", "riscv_mips_s8200"),
        ("XuanTie", "C930", "riscv", "riscv_xuantie_c930"),
        ("Cervell", "NPU", "riscv", "riscv_cervell"),
        ("Xilinx", "VU9P FPGA", "fpga", "fpga_xilinx_vu9p"),
        ("BitNet", "Ternary FPGA", "fpga", "fpga_ternary"),
        ("AMD", "MI350X", "gfx_mi350", "amd_mi350x"),
        ("AMD", "MI455X CDNA5", "gfx_cdna5", "rocm_cdna5_mi455x"),
        ("NVIDIA", "B200 TEE", "sm100_tee", "cuda_sm100_tee"),
        ("NVIDIA", "GB300", "sm100_gb300", "cuda_sm100_gb300"),
        ("NVIDIA", "Rubin R100", "sm120", "cuda_sm120"),
        ("NVIDIA", "Rubin Ultra", "sm130", "cuda_sm130"),
    ]
    result = []
    for vendor, device, arch, target_id in stubs:
        result.append(HardwareCapabilities.unavailable(
            vendor=vendor,
            device=device,
            architecture=arch,
            target_id=target_id,
            reason=f"Planning profile; no executable {vendor} adapter or device evidence is established by this entry",
            # These entries are capability placeholders only.  Marking them
            # implemented made the detector overstate support even though no
            # executable backend or vendor SDK was present.
            implemented=False,
        ))
        result[-1].extra["catalog_only"] = True
    return result


def detect_directml() -> HardwareCapabilities:
    """Report provider/SDK evidence without assuming successful execution."""
    from aether.backends.auxiliary_detection import detect_directml as probe
    return probe()


def detect_qualcomm_npu() -> HardwareCapabilities:
    """Report provider/SDK evidence without assuming successful execution."""
    from aether.backends.auxiliary_detection import detect_qualcomm_npu as probe
    return probe()


def detect_apple_ane() -> HardwareCapabilities:
    """Report provider/SDK evidence without assuming successful execution."""
    from aether.backends.auxiliary_detection import detect_apple_ane as probe
    return probe()
