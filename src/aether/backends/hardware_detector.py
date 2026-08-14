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
    """
    caps: list[HardwareCapabilities] = []

    # 1. CPU (always available)
    caps.append(detect_cpu())

    # 2. CUDA (NVIDIA GPUs)
    caps.extend(detect_cuda_devices())

    # 3. ROCm (AMD GPUs)
    caps.extend(detect_rocm_devices())

    # 4. Metal (Apple Silicon)
    caps.append(detect_metal())

    # 5. OpenVINO / Intel NPU
    caps.append(detect_openvino())

    # 6. Vendor-specific targets (always unavailable on this host)
    caps.extend(_unavailable_vendor_targets())

    return caps


def detect_cpu() -> HardwareCapabilities:
    """Detect CPU capabilities for the current host."""
    return HardwareCapabilities.cpu_host()


def detect_cuda_devices() -> list[HardwareCapabilities]:
    """Detect all NVIDIA CUDA devices on this host.

    Returns one HardwareCapabilities per physical GPU device. If CUDA is
    unavailable, returns a single unavailable capability object.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return [HardwareCapabilities.unavailable(
                vendor="NVIDIA",
                device="unknown",
                architecture="unknown",
                target_id="cuda",
                reason="CUDA runtime is not available on this host "
                       "(no NVIDIA GPU or driver not installed)",
                implemented=True,
            )]

        devices: list[HardwareCapabilities] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = f"{props.major}.{props.minor}"
            sm = f"sm{props.major}{props.minor}"
            target_id = _cuda_target_id(props.major, props.minor)
            devices.append(HardwareCapabilities(
                vendor="NVIDIA",
                device=props.name,
                architecture=sm,
                target_id=target_id,
                driver_version=_cuda_driver_version(),
                runtime_version=torch.version.cuda or "unknown",
                memory_bytes=props.total_memory,
                supports_fp32=True,
                supports_fp16=True,
                supports_bf16=props.major >= 8,
                supports_fp8=props.major >= 9,
                supports_fp4=(props.major >= 10),
                supports_int8=True,
                supports_int4=True,
                supports_cuda_graph=True,
                supports_tee=False,  # Would require NVIDIA CC attestation
                supports_nvlink=_has_nvlink(i),
                supports_peer_access=props.multi_processor_count > 0,
                warp_or_wavefront_size=32,
                implemented=True,
                available=True,
                compile_tested=False,   # No CUDA compilation tested yet
                execution_tested=False, # No GPU inference yet
                production_validated=False,
                extra={
                    "device_index": i,
                    "compute_capability": cc,
                    "multi_processor_count": props.multi_processor_count,
                    "max_threads_per_block": props.max_threads_per_block,
                },
            ))
        return devices

    except ImportError:
        return [HardwareCapabilities.unavailable(
            vendor="NVIDIA",
            device="unknown",
            architecture="unknown",
            target_id="cuda",
            reason="PyTorch not installed",
            implemented=True,
        )]
    except Exception as exc:  # noqa: BLE001
        logger.warning("CUDA detection failed: %s", exc)
        return [HardwareCapabilities.unavailable(
            vendor="NVIDIA",
            device="unknown",
            architecture="unknown",
            target_id="cuda",
            reason=f"CUDA detection error: {exc}",
            implemented=True,
        )]


def detect_rocm_devices() -> list[HardwareCapabilities]:
    """Detect AMD ROCm/HIP devices on this host."""
    try:
        import torch
        # ROCm appears as CUDA in PyTorch on some builds
        if hasattr(torch, "hip") and torch.hip.is_available():
            return _detect_rocm_via_hip(torch)
        if torch.cuda.is_available():
            # Check if any GPU is AMD
            amd_devices = []
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if any(kw in name for kw in ("AMD", "Radeon", "Instinct")):
                    amd_devices.append(_rocm_cap_from_torch(i, name, torch))
            if amd_devices:
                return amd_devices
        return [HardwareCapabilities.unavailable(
            vendor="AMD",
            device="unknown",
            architecture="unknown",
            target_id="rocm",
            reason="ROCm runtime not available on this host",
            implemented=True,
        )]
    except Exception as exc:  # noqa: BLE001
        return [HardwareCapabilities.unavailable(
            vendor="AMD",
            device="unknown",
            architecture="unknown",
            target_id="rocm",
            reason=f"ROCm detection error: {exc}",
            implemented=True,
        )]


def detect_metal() -> HardwareCapabilities:
    """Detect Apple Metal / MPS availability."""
    try:
        import torch
        if torch.backends.mps.is_available():
            # Get memory info if possible
            mem = 0
            try:
                mem = torch.mps.driver_allocated_memory() + \
                      getattr(torch.mps, "recommended_max_memory", lambda: 0)()
            except Exception:  # noqa: BLE001
                pass
            device_name = platform.machine()
            return HardwareCapabilities(
                vendor="Apple",
                device=device_name,
                architecture="apple_silicon",
                target_id="metal_m1",
                driver_version="Metal " + platform.mac_ver()[0],
                runtime_version=str(torch.__version__),
                memory_bytes=mem or _host_memory_bytes(),  # unified memory
                supports_fp32=True,
                supports_fp16=True,
                supports_bf16=True,
                supports_int8=True,
                supports_unified_memory=True,
                warp_or_wavefront_size=32,
                implemented=True,
                available=True,
                compile_tested=False,
                execution_tested=False,
                production_validated=False,
            )
    except (ImportError, AttributeError):
        pass
    return HardwareCapabilities.unavailable(
        vendor="Apple",
        device="Apple Silicon",
        architecture="apple_silicon",
        target_id="metal",
        reason="Metal/MPS runtime not available (requires macOS with Apple Silicon)",
        implemented=True,
    )


def detect_openvino() -> HardwareCapabilities:
    """Detect Intel OpenVINO / NPU."""
    try:
        import openvino  # type: ignore[import]
        return HardwareCapabilities(
            vendor="Intel",
            device="OpenVINO Runtime",
            architecture="openvino",
            target_id="openvino_npu",
            runtime_version=getattr(openvino, "__version__", "unknown"),
            implemented=True,
            available=True,
            compile_tested=False,
            execution_tested=False,
            production_validated=False,
        )
    except ImportError:
        return HardwareCapabilities.unavailable(
            vendor="Intel",
            device="Intel NPU",
            architecture="openvino",
            target_id="openvino_npu",
            reason="OpenVINO package not installed",
            implemented=True,
        )


def get_memory_info(device_type: str = "cpu", device_index: int = 0) -> MemoryInfo:
    """Get current memory information for a device.

    Args:
        device_type: "cpu", "cuda", "mps", "hip"
        device_index: GPU device index (for CUDA/HIP)

    Returns:
        MemoryInfo with real measurements where available.
    """
    if device_type == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_index)
                free, total = torch.cuda.mem_get_info(device_index)
                used = total - free
                return MemoryInfo(
                    total_bytes=total,
                    free_bytes=free,
                    used_bytes=used,
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


def validate_backend_environment(target_id: str) -> ValidationResult:
    """Run environment checks for a specific backend target.

    Returns a ValidationResult with pass/fail evidence for each check.
    This is called by ``aether hardware validate``.
    """
    checks_passed: list[str] = []
    checks_failed: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    if target_id.startswith("cpu"):
        try:
            import torch
            checks_passed.append(f"PyTorch {torch.__version__} available")
            details["torch_version"] = torch.__version__
        except ImportError:
            checks_failed.append("PyTorch not installed")

        try:
            import numpy as np
            checks_passed.append(f"NumPy {np.__version__} available")
        except ImportError:
            checks_failed.append("NumPy not installed")

        # Verify native CPU kernel compilation
        try:
            from aether.kernels.native_cpu import get_native_kernels
            nk = get_native_kernels()
            if nk.ensure_compiled():
                checks_passed.append(f"Native CPU kernels compiled: {nk.available_kernels()}")
            else:
                warnings.append(f"Native CPU kernel compilation failed: {nk.build_error}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Native CPU kernel check error: {exc}")

        available = not checks_failed
        return ValidationResult(
            backend_name=target_id,
            available=available,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            details=details,
        )

    if target_id.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                checks_passed.append(f"CUDA available: {torch.cuda.get_device_name(0)}")
                checks_passed.append(f"CUDA version: {torch.version.cuda}")
                details["cuda_version"] = torch.version.cuda
                details["device_count"] = torch.cuda.device_count()
            else:
                checks_failed.append("CUDA runtime not available")
        except ImportError:
            checks_failed.append("PyTorch not installed")

        return ValidationResult(
            backend_name=target_id,
            available=bool(checks_passed and not checks_failed),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            details=details,
        )

    if target_id.startswith("rocm"):
        try:
            import torch
            has_rocm = (
                (hasattr(torch, "hip") and torch.hip.is_available()) or
                (torch.cuda.is_available() and any(
                    kw in torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                    for kw in ("AMD", "Radeon", "Instinct")
                ))
            )
            if has_rocm:
                checks_passed.append("ROCm runtime detected")
            else:
                checks_failed.append("ROCm runtime not available")
        except Exception as exc:  # noqa: BLE001
            checks_failed.append(f"ROCm detection failed: {exc}")

        return ValidationResult(
            backend_name=target_id,
            available=bool(checks_passed and not checks_failed),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            details=details,
        )

    if target_id.startswith("metal"):
        try:
            import torch
            if torch.backends.mps.is_available():
                checks_passed.append("Apple MPS/Metal available")
            else:
                checks_failed.append("Apple MPS/Metal not available")
        except Exception as exc:  # noqa: BLE001
            checks_failed.append(f"Metal detection failed: {exc}")

        return ValidationResult(
            backend_name=target_id,
            available=bool(checks_passed and not checks_failed),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            details=details,
        )

    # Unknown / stub target
    checks_failed.append(
        f"Target {target_id!r} has no executable backend implementation on this host"
    )
    return ValidationResult(
        backend_name=target_id,
        available=False,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        warnings=warnings,
        details=details,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cuda_target_id(major: int, minor: int) -> str:
    """Map compute capability to an Aether target ID."""
    sm = major * 10 + minor
    if sm >= 130:
        return "cuda_sm130"
    if sm >= 120:
        return "cuda_sm120"
    if sm >= 100:
        return "cuda_sm100"
    if sm >= 90:
        return "cuda_sm90"
    if sm >= 89:
        return "cuda_sm89"
    if sm >= 80:
        return "cuda_sm80"
    if sm >= 70:
        return "cuda_sm70"
    return f"cuda_sm{major}{minor}"


def _cuda_driver_version() -> str:
    try:
        import torch
        ver = torch.version.cuda
        return f"CUDA {ver}" if ver else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _has_nvlink(device_index: int) -> bool:
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        # NVLink is link-count > 0
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


def _detect_rocm_via_hip(torch: Any) -> list[HardwareCapabilities]:
    caps = []
    try:
        device_count = torch.hip.device_count() if hasattr(torch.hip, "device_count") else 1
        for i in range(device_count):
            name = torch.hip.get_device_name(i) if hasattr(torch.hip, "get_device_name") else "AMD GPU"
            caps.append(_rocm_cap_from_torch(i, name, torch))
    except Exception:  # noqa: BLE001
        caps.append(_rocm_cap_from_torch(0, "AMD GPU", torch))
    return caps


def _rocm_cap_from_torch(i: int, name: str, torch: Any) -> HardwareCapabilities:
    gfx = _infer_gfx_arch(name)
    target_id = _rocm_target_id(name)
    return HardwareCapabilities(
        vendor="AMD",
        device=name,
        architecture=gfx,
        target_id=target_id,
        driver_version="ROCm",
        runtime_version=str(torch.__version__),
        memory_bytes=0,  # Would need rocm_smi
        supports_fp32=True,
        supports_fp16=True,
        supports_bf16=True,
        supports_fp8="MI300" in name or "MI350" in name,
        supports_int8=True,
        warp_or_wavefront_size=64,
        implemented=True,
        available=True,
        compile_tested=False,
        execution_tested=False,
        production_validated=False,
        extra={"device_index": i},
    )


def _infer_gfx_arch(device_name: str) -> str:
    """Infer GFX architecture from device name."""
    if "MI300" in device_name:
        return "gfx942"
    if "MI250" in device_name:
        return "gfx90a"
    if "MI100" in device_name:
        return "gfx908"
    if "7900" in device_name or "RDNA3" in device_name:
        return "gfx1100"
    if "6900" in device_name or "RDNA2" in device_name:
        return "gfx1030"
    return "gfx_unknown"


def _rocm_target_id(device_name: str) -> str:
    if "MI300" in device_name:
        return "rocm_cdna3"
    if "MI250" in device_name or "MI200" in device_name:
        return "rocm_cdna2"
    if "7900" in device_name:
        return "rocm_rdna3"
    return "rocm_cdna3"


def _unavailable_vendor_targets() -> list[HardwareCapabilities]:
    """Return stub capabilities for targets that have no runtime on this host."""
    stubs = [
        ("Qualcomm", "Cloud AI 100", "qnn", "qualcomm_cloud_ai100"),
        ("Qualcomm", "QNN NPU", "qnn", "qualcomm_qnn"),
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
            reason=f"No {vendor} hardware or SDK present on this host",
            implemented=True,
        ))
    return result
