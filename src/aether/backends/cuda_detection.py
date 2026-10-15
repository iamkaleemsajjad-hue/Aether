"""CUDA hardware evidence without importing an execution framework.

NVML is telemetry, not proof that the CUDA driver can initialize a device.
Driver API calls below are checked individually. No kernels are executed here.
"""
from __future__ import annotations

import ctypes
import importlib.util
import platform
import subprocess
from typing import Any

from aether.backends.capabilities import HardwareCapabilities
from aether.core.constants import SUPPORTED_TARGET_IDS
from aether.backends.hardware_inventory import device_identity_keys, normalize_pci_bus_id


def _dependency(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _nvml_evidence() -> tuple[list[dict[str, Any]], str]:
    devices: list[dict[str, Any]] = []
    driver = "unknown"
    initialized = False
    try:
        import pynvml
        pynvml.nvmlInit()
        initialized = True
        try:
            raw = pynvml.nvmlSystemGetDriverVersion()
            driver = raw.decode() if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass
        for index in range(min(256, pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            row: dict[str, Any] = {"telemetry_index": index}
            for field, method in (("name", "nvmlDeviceGetName"), ("uuid", "nvmlDeviceGetUUID")):
                try:
                    raw = getattr(pynvml, method)(handle)
                    row[field] = raw.decode() if isinstance(raw, bytes) else str(raw)
                except Exception:
                    pass
            try:
                row["memory"] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
            except Exception:
                pass
            try:
                row["cc"] = tuple(pynvml.nvmlDeviceGetCudaComputeCapability(handle))
            except Exception:
                pass
            try:
                bus = pynvml.nvmlDeviceGetPciInfo(handle).busId
                row["pci_bus_id"] = bus.decode() if isinstance(bus, bytes) else str(bus)
            except Exception:
                pass
            link_states = []
            try:
                for link in range(32):
                    try:
                        link_states.append(bool(pynvml.nvmlDeviceGetNvLinkState(handle, link)))
                    except pynvml.NVMLError:
                        break
            except (AttributeError, TypeError):
                pass
            if link_states:
                row["nvlink_active"] = any(link_states)
            devices.append(row)
    except Exception:
        pass
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    return devices, driver


def _driver_release(driver: str) -> tuple[str, str | None]:
    if driver != "unknown":
        return driver, "nvml"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip(), "nvidia-smi"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown", None


def _bind(lib: Any, name: str, args: list[Any]) -> Any:
    function = getattr(lib, name)
    function.argtypes = args
    function.restype = ctypes.c_int
    return function


def _target(major: int, minor: int) -> str:
    # A baseline cubin is compatible only within its major CC, at an equal or
    # newer minor CC. Do not fold an unknown future major into an older target.
    exact = f"cuda_sm{major}{minor}"
    if exact in SUPPORTED_TARGET_IDS:
        return exact
    candidates = [str(t) for t in SUPPORTED_TARGET_IDS if str(t).startswith(f"cuda_sm{major}")]
    compatible = [t for t in candidates if t.removeprefix("cuda_sm").isdigit()
                  and int(t.removeprefix("cuda_sm")) <= major * 10 + minor]
    return max(compatible, key=lambda t: int(t.removeprefix("cuda_sm"))) if compatible else exact


def _precision(major: int, minor: int) -> dict[str, bool]:
    """Confirmed arithmetic, including architecture-specific tensor formats.

    NVIDIA CUDA Programming Guide, Tensor Core input type table; future
    architectures are deliberately unconfirmed rather than extrapolated.
    """
    cc = (major, minor)
    documented = cc in {(3, 0), (3, 2), (3, 5), (3, 7), (5, 0), (5, 2), (5, 3),
                        (6, 0), (6, 1), (6, 2), (7, 0), (7, 2), (7, 5),
                        (8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (10, 0),
                        (10, 3), (10, 7), (11, 0), (12, 0), (12, 1)}
    return {
        "supports_fp32": documented,
        "supports_fp16": documented and cc >= (5, 3),
        "supports_bf16": documented and major >= 8,
        "supports_fp8": cc in {(8, 9), (9, 0), (10, 0), (10, 3), (10, 7), (11, 0), (12, 0), (12, 1)},
        "supports_fp4": cc in {(10, 0), (10, 3), (10, 7), (11, 0), (12, 0), (12, 1)},
        "supports_int8": documented and cc >= (6, 1),  # DP4A / INT8 tensor arithmetic
        "supports_int4": cc in {(7, 5), (8, 0), (8, 6), (8, 7), (8, 9)},
    }


def detect_cuda_devices() -> list[HardwareCapabilities]:
    telemetry, driver = _nvml_evidence()
    driver, driver_source = _driver_release(driver)
    dependencies = {"torch": _dependency("torch")}
    error = "CUDA driver library not found"
    devices: list[HardwareCapabilities] = []
    try:
        lib = ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")
        init = _bind(lib, "cuInit", [ctypes.c_uint])
        status = int(init(0))
        if status:
            raise RuntimeError(f"cuInit failed (CUDA error {status})")
        count = ctypes.c_int()
        get_count = _bind(lib, "cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
        status = int(get_count(ctypes.byref(count)))
        if status:
            raise RuntimeError(f"cuDeviceGetCount failed (CUDA error {status})")
        if count.value < 0 or count.value > 256:
            raise RuntimeError(f"CUDA device count {count.value} exceeds the bounded enumeration")
        version = ctypes.c_int()
        try:
            status = _bind(lib, "cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)])(ctypes.byref(version))
            api_version = f"{version.value // 1000}.{(version.value % 1000) // 10}" if status == 0 else "unknown"
        except AttributeError:
            api_version = "unknown"
        get_device = _bind(lib, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
        get_attr = _bind(lib, "cuDeviceGetAttribute", [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int])
        get_name = _bind(lib, "cuDeviceGetName", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int])
        get_memory = _bind(lib, "cuDeviceTotalMem_v2", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int])
        for index in range(count.value):
            device = ctypes.c_int()
            status = int(get_device(ctypes.byref(device), index))
            if status:
                continue
            errors: list[str] = []
            def attribute(key: int) -> int | None:
                value = ctypes.c_int()
                code = int(get_attr(ctypes.byref(value), key, device.value))
                if code:
                    errors.append(f"cuDeviceGetAttribute({key}) failed (CUDA error {code})")
                    return None
                return value.value
            major, minor = attribute(75), attribute(76)
            cc_known = major is not None and minor is not None and major > 0
            name = ctypes.create_string_buffer(256)
            name_status = int(get_name(name, len(name), device.value))
            memory = ctypes.c_size_t()
            memory_status = int(get_memory(ctypes.byref(memory), device.value))
            pci_bus_id = None
            try:
                bus = ctypes.create_string_buffer(32)
                pci = _bind(lib, "cuDeviceGetPCIBusId", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int])
                if pci(bus, len(bus), device.value) == 0:
                    pci_bus_id = normalize_pci_bus_id(bus.value.decode(errors="replace"))
            except AttributeError:
                pass
            identity: dict[str, Any] = {"pci_bus_id": pci_bus_id}
            try:
                raw_uuid = (ctypes.c_ubyte * 16)()
                uuid_api = "cuDeviceGetUuid_v2" if hasattr(lib, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
                query_uuid = _bind(lib, uuid_api, [ctypes.c_void_p, ctypes.c_int])
                if query_uuid(raw_uuid, device.value) == 0:
                    import uuid
                    identity["uuid"] = str(uuid.UUID(bytes=bytes(raw_uuid)))
                    identity["uuid_source"] = uuid_api
            except AttributeError:
                pass
            if platform.system() == "Windows":
                try:
                    luid = (ctypes.c_ubyte * 8)()
                    node_mask = ctypes.c_uint()
                    query_luid = _bind(lib, "cuDeviceGetLuid", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int])
                    if query_luid(luid, ctypes.byref(node_mask), device.value) == 0:
                        identity.update(luid=bytes(luid).hex(), luid_node_mask=node_mask.value)
                except AttributeError:
                    pass
            # Match by PCI identity rather than assuming visibility-filtered
            # CUDA ordinals equal NVML indices. Ambiguous MIG entries are not
            # used as device-specific telemetry.
            identity_matches = [row for row in telemetry if device_identity_keys(identity).intersection(device_identity_keys(row))]
            device_telemetry = identity_matches[0] if len(identity_matches) == 1 else {}
            # NVML indices need not equal CUDA ordinals (visibility/MIG).
            # The CUDA API is authoritative for device identity and memory.
            unified = attribute(83)  # CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
            peers: list[int] = []
            if count.value > 1:
                try:
                    peer_fn = _bind(lib, "cuDeviceCanAccessPeer", [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int])
                    for other in range(count.value):
                        if other == index:
                            continue
                        peer_device, access = ctypes.c_int(), ctypes.c_int()
                        if get_device(ctypes.byref(peer_device), other) == 0 and peer_fn(ctypes.byref(access), device.value, peer_device.value) == 0 and access.value:
                            peers.append(other)
                except AttributeError:
                    pass
            target = _target(major, minor) if cc_known else "cuda_unknown"
            caps = HardwareCapabilities(
                vendor="NVIDIA", device=name.value.decode(errors="replace") if name_status == 0 and name.value else f"NVIDIA GPU {index}",
                architecture=f"sm{major}{minor}" if cc_known else "unknown", target_id=target,
                driver_version=driver, runtime_version=f"CUDA Driver API {api_version}",
                memory_bytes=int(memory.value) if memory_status == 0 else 0,
                memory_kind="device_global", memory_source="cuDeviceTotalMem_v2" if memory_status == 0 else None,
                **(_precision(major, minor) if cc_known else {}),
                supports_cuda_graph=version.value >= 10000,
                supports_unified_memory=bool(unified), supports_peer_access=bool(peers),
                supports_nvlink=bool(device_telemetry.get("nvlink_active")),
                warp_or_wavefront_size=attribute(10) or 0,
                hardware_present=True, runtime_available=True,
                backend_dependencies=dependencies, backend_available=None if any(dependencies.values()) else False,
                implemented=target in SUPPORTED_TARGET_IDS, available=cc_known,
                unavailable_reason=None if cc_known else "CUDA device compute capability query failed",
                extra={**identity, "device_index": index, "device_count": count.value,
                       "execution_device": f"cuda:{index}", "device_index_scope": "CUDA compute-driver visible ordinal",
                       "raw_architecture": f"sm{major}{minor}" if cc_known else None,
                       "compute_capability": f"{major}.{minor}" if cc_known else None,
                       "detection_method": "ctypes_cuda_driver", "driver_version_source": driver_source,
                       "cuda_driver_api_version": api_version, "memory_known": memory_status == 0,
                       "peer_device_indices": peers, "query_errors": errors,
                       "nvlink_confirmed": "nvlink_active" in device_telemetry,
                       "precision_evidence": "CUDA compute capability specification" if cc_known else "unconfirmed",
                       "precision_execution": "depends on the selected execution backend; not tested by detection"},
            )
            devices.append(caps)
        error = "Device is reported by NVML but not exposed as a compute device by the CUDA Driver API" if devices else "CUDA initialized but exposed no usable device"
    except (OSError, AttributeError, RuntimeError, ValueError) as exc:
        error = str(exc)
    # Preserve independent hardware evidence if the driver cannot initialize.
    for row in telemetry:
        if any(device_identity_keys(row).intersection(device_identity_keys(cap.extra)) for cap in devices):
            continue
        cc = row.get("cc")
        target = _target(*cc) if cc else "cuda"
        devices.append(HardwareCapabilities(
            vendor="NVIDIA", device=row.get("name", "NVIDIA GPU"),
            architecture=f"sm{cc[0]}{cc[1]}" if cc else "unknown", target_id=target,
            driver_version=driver, memory_bytes=row.get("memory", 0),
            memory_kind="device_global", memory_source="nvml" if "memory" in row else None,
            **(_precision(*cc) if cc else {}), hardware_present=True,
            runtime_available=False, backend_dependencies=dependencies, backend_available=False,
            implemented=target in SUPPORTED_TARGET_IDS, available=False, unavailable_reason=error,
            extra={**row, "device_index": None, "execution_device": None,
                   "device_index_scope": "NVML physical telemetry index; no execution ordinal established",
                   "raw_architecture": f"sm{cc[0]}{cc[1]}" if cc else None,
                   "detection_method": "nvml_telemetry_only", "driver_version_source": driver_source,
                   "memory_known": "memory" in row, "cuda_driver_error": error},
        ))
    if not devices:
        devices.append(HardwareCapabilities(
            vendor="NVIDIA", device="unknown", architecture="unknown", target_id="cuda",
            driver_version=driver, hardware_present=None, runtime_available=False,
            backend_dependencies=dependencies, backend_available=False, implemented=False,
            available=False, unavailable_reason=error,
        ))
    return devices
