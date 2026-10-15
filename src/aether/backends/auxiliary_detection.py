"""SDK/provider evidence for accelerators without confirmed device execution."""
from __future__ import annotations

import importlib
import platform
import subprocess

from aether.backends.capabilities import HardwareCapabilities


def detect_directml() -> HardwareCapabilities:
    cap = HardwareCapabilities(
        vendor="Microsoft", device="DirectML adapter (not enumerated)", architecture="directml",
        target_id="directml", implemented=True, available=False,
        backend_available=None, unavailable_reason="A DirectML device has not been initialized",
        extra={"detection_method": "provider_registration", "precision_evidence": "unconfirmed",
               "implementation_scope": "ONNX models and explicitly authorized dense-decoder AEG exports via ONNX Runtime"},
    )
    if platform.system() != "Windows":
        cap.hardware_present = False
        cap.runtime_available = False
        cap.backend_available = False
        cap.unavailable_reason = "DirectML requires Windows"
        return cap
    from aether.backends.hardware_inventory import probe_gpu_inventory
    inventory = probe_gpu_inventory()
    cap.extra["physical_adapters"] = inventory["devices"]
    cap.extra["hardware_inventory"] = {key: value for key, value in inventory.items() if key != "devices"}
    cap.hardware_present = True if inventory["devices"] else (False if inventory["complete"] else None)
    try:
        ort = importlib.import_module("onnxruntime")
        providers = ort.get_available_providers()
        registered = "DmlExecutionProvider" in providers
        cap.runtime_version = str(ort.__version__)
        cap.backend_dependencies = {"onnxruntime": True, "DmlExecutionProvider": registered}
        cap.extra["ort_providers"] = providers
        # Provider registration is not adapter enumeration or device creation.
        cap.runtime_available = None if registered else False
        cap.backend_available = None if registered else False
        cap.unavailable_reason = ("DmlExecutionProvider is registered; device initialization and execution require explicit validation"
                                  if registered else "Install onnxruntime-directml; this ONNX Runtime has no DmlExecutionProvider")
        return cap
    except (ImportError, OSError, RuntimeError) as exc:
        cap.extra["provider_error"] = str(exc)
        cap.backend_dependencies = {"onnxruntime": False, "DmlExecutionProvider": False}
        cap.runtime_available = False
        cap.backend_available = False
    try:
        import ctypes
        ctypes.WinDLL("d3d12", use_last_error=True)
        ctypes.WinDLL("dxgi", use_last_error=True)
        cap.extra["detection_method"] = "d3d12_dxgi_dll_probe"
        cap.extra["os_dlls_present"] = True
        cap.unavailable_reason = "D3D12/DXGI DLLs found; they do not prove GPU presence or a functioning DirectML runtime"
    except (OSError, AttributeError):
        cap.unavailable_reason = "No DirectML provider or usable device was detected"
    return cap


def detect_qualcomm_npu() -> HardwareCapabilities:
    cap = HardwareCapabilities(
        vendor="Qualcomm", device="Qualcomm NPU (not enumerated)", architecture="qnn",
        target_id="qualcomm_qnn", implemented=False, available=False, backend_available=False,
        unavailable_reason="No executable Aether QNN/SNPE adapter or initialized NPU was detected",
    )
    for name in ("qnn", "snpe"):
        try:
            sdk = importlib.import_module(name)
            cap.runtime_version = str(getattr(sdk, "__version__", "unknown"))
            cap.backend_dependencies[name] = True
            cap.extra = {"detection_method": f"{name}_sdk_import", "sdk_present": True,
                         "precision_evidence": "unconfirmed"}
            return cap
        except (ImportError, OSError, RuntimeError) as exc:
            cap.backend_dependencies[name] = False
            cap.extra[f"{name}_error"] = str(exc)
    try:
        result = subprocess.run(["qnn-net-run", "--version"], capture_output=True, text=True, timeout=3, check=False)
        if result.returncode == 0:
            lines = (result.stdout or result.stderr).strip().splitlines()
            cap.runtime_version = lines[0] if lines else "unknown"
            cap.extra = {"detection_method": "qnn_net_run_version", "sdk_present": True,
                         "precision_evidence": "unconfirmed"}
    except (OSError, subprocess.SubprocessError):
        pass
    return cap


def detect_apple_ane() -> HardwareCapabilities:
    cap = HardwareCapabilities(
        vendor="Apple", device="Apple Neural Engine", architecture="ane", target_id="apple_ane",
        implemented=False, available=False, backend_available=False,
        unavailable_reason="Aether has no executable ANE inference adapter; Core ML may schedule operations on other devices",
        extra={"precision_evidence": "unconfirmed"},
    )
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        cap.hardware_present = False
        cap.runtime_available = False
        cap.unavailable_reason = "Apple Neural Engine requires macOS on Apple Silicon"
        return cap
    # ARM architecture alone is not device enumeration (VMs exist).
    try:
        coremltools = importlib.import_module("coremltools")
        cap.runtime_version = f"coremltools {getattr(coremltools, '__version__', 'unknown')}"
        cap.backend_dependencies["coremltools"] = True
        cap.extra["detection_method"] = "coremltools_import"
        try:
            utils = importlib.import_module("coremltools.models.utils")
            units = utils._get_available_compute_units()
            cap.extra["compute_units"] = [str(unit) for unit in units]
            if any("neural_engine" in str(unit).lower() for unit in units):
                cap.hardware_present = True
                cap.runtime_available = True
        except Exception as exc:
            cap.extra["enumeration_error"] = str(exc)
            # Failure or missing private API is unknown, never assumed success.
    except (ImportError, OSError, RuntimeError):
        cap.backend_dependencies["coremltools"] = False
        cap.runtime_available = False
        cap.unavailable_reason += "; coremltools is not installed"
    return cap
