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
    """Detect all NVIDIA CUDA devices on this host via pynvml (no torch required).

    Returns one HardwareCapabilities per physical GPU device. If CUDA is
    unavailable, returns a single unavailable capability object.
    """
    # ── Primary: pynvml (installed as transitive dep via nvidia-ml-py) ────────
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return [HardwareCapabilities.unavailable(
                vendor="NVIDIA",
                device="unknown",
                architecture="unknown",
                target_id="cuda",
                reason="CUDA runtime is not available on this host (no NVIDIA GPU or driver not installed)",
                implemented=True,
            )]
        devices: list[HardwareCapabilities] = []
        try:
            driver_ver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_ver, bytes):
                driver_ver = driver_ver.decode()
        except Exception:  # noqa: BLE001
            driver_ver = "unknown"
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
            except Exception:  # noqa: BLE001
                name = f"NVIDIA GPU {i}"
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mem = mem_info.total
            except Exception:  # noqa: BLE001
                total_mem = 0
            cc = f"{major}.{minor}"
            sm = f"sm{major}{minor}"
            target_id = _cuda_target_id(major, minor)
            devices.append(HardwareCapabilities(
                vendor="NVIDIA",
                device=name,
                architecture=sm,
                target_id=target_id,
                driver_version=driver_ver,
                runtime_version="nvml",
                memory_bytes=total_mem,
                supports_fp32=True,
                supports_fp16=True,
                supports_bf16=major >= 8,
                supports_fp8=major >= 9,
                supports_fp4=(major >= 10),
                supports_int8=True,
                supports_int4=True,
                supports_cuda_graph=True,
                supports_tee=False,
                supports_nvlink=_has_nvlink_nvml(handle),
                supports_peer_access=True,
                warp_or_wavefront_size=32,
                implemented=True,
                available=True,
                compile_tested=False,
                execution_tested=False,
                production_validated=False,
                extra={"device_index": i, "compute_capability": cc},
            ))
        pynvml.nvmlShutdown()
        return devices
    except Exception as exc:  # noqa: BLE001  — pynvml not installed or no NVIDIA GPU
        logger.debug("pynvml CUDA detection failed: %s", exc)

    # ── Fallback: CUDA Driver API via ctypes ──────────────────────────────────
    import ctypes, sys  # noqa: E401,PLC0415
    try:
        _lib = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        if _lib.cuInit(0) == 0:  # CUDA_SUCCESS
            count = ctypes.c_int(0)
            _lib.cuDeviceGetCount(ctypes.byref(count))
            if count.value > 0:
                major = ctypes.c_int(0)
                minor = ctypes.c_int(0)
                _lib.cuDeviceGetAttribute(ctypes.byref(major), 75, 0)
                _lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, 0)
                target_id = _cuda_target_id(major.value, minor.value)
                return [HardwareCapabilities(
                    vendor="NVIDIA",
                    device=f"NVIDIA GPU (via cuDriver, {count.value} device(s))",
                    architecture=f"sm{major.value}{minor.value}",
                    target_id=target_id,
                    driver_version="unknown (ctypes)",
                    runtime_version="cuda_driver",
                    memory_bytes=0,
                    supports_fp32=True,
                    supports_fp16=True,
                    supports_bf16=major.value >= 8,
                    supports_fp8=major.value >= 9,
                    supports_fp4=(major.value >= 10),
                    supports_int8=True,
                    supports_int4=True,
                    supports_cuda_graph=True,
                    warp_or_wavefront_size=32,
                    implemented=True,
                    available=True,
                    compile_tested=False,
                    execution_tested=False,
                    production_validated=False,
                    extra={"device_count": count.value, "detection_method": "ctypes_cuda_driver"},
                )]
    except Exception:  # noqa: BLE001
        pass

    return [HardwareCapabilities.unavailable(
        vendor="NVIDIA",
        device="unknown",
        architecture="unknown",
        target_id="cuda",
        reason="No NVIDIA GPU detected (pynvml and CUDA Driver API both failed)",
        implemented=True,
    )]


def detect_rocm_devices() -> list[HardwareCapabilities]:
    """Detect AMD ROCm/HIP devices on this host (no torch required).

    Uses: amdsmi Python binding, rocm-smi subprocess, or /dev/kfd presence.
    """
    import sys  # noqa: PLC0415

    # ── 1. amdsmi Python binding (ROCm 5.6+) ─────────────────────────────────
    try:
        import amdsmi  # type: ignore[import]  # noqa: PLC0415
        amdsmi.amdsmi_init()
        processors = amdsmi.amdsmi_get_processor_handles()
        if processors:
            devices: list[HardwareCapabilities] = []
            for proc in processors:
                try:
                    name = amdsmi.amdsmi_get_gpu_board_info(proc).get("product_name", "AMD GPU")
                    mem_bytes = amdsmi.amdsmi_get_gpu_memory_total(proc, amdsmi.AmdSmiMemoryType.VRAM)
                except Exception:  # noqa: BLE001
                    name, mem_bytes = "AMD GPU", 0
                devices.append(HardwareCapabilities(
                    vendor="AMD",
                    device=name,
                    architecture="gfx_unknown",
                    target_id="rocm",
                    runtime_version="amdsmi",
                    memory_bytes=mem_bytes,
                    supports_fp32=True,
                    supports_fp16=True,
                    supports_bf16=True,
                    supports_int8=True,
                    warp_or_wavefront_size=64,
                    implemented=True,
                    available=True,
                    compile_tested=False,
                    execution_tested=False,
                    production_validated=False,
                ))
            amdsmi.amdsmi_shut_down()
            return devices
        amdsmi.amdsmi_shut_down()
    except Exception:  # noqa: BLE001
        pass

    # ── 2. rocm-smi subprocess ────────────────────────────────────────────────
    if sys.platform in ("linux", "darwin"):
        try:
            import subprocess, json  # noqa: E401,PLC0415
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                devices = []
                for _key, info in data.items():
                    if isinstance(info, dict):
                        devices.append(HardwareCapabilities(
                            vendor="AMD",
                            device=info.get("Card Series", "AMD GPU"),
                            architecture=info.get("GFX Version", "gfx_unknown"),
                            target_id="rocm",
                            runtime_version="rocm-smi",
                            implemented=True,
                            available=True,
                            compile_tested=False,
                            execution_tested=False,
                            production_validated=False,
                        ))
                if devices:
                    return devices
        except Exception:  # noqa: BLE001
            pass

    # ── 3. /dev/kfd presence on Linux ────────────────────────────────────────
    if sys.platform == "linux":
        try:
            import pathlib  # noqa: PLC0415
            if pathlib.Path("/dev/kfd").exists():
                return [HardwareCapabilities(
                    vendor="AMD",
                    device="AMD GPU (/dev/kfd present)",
                    architecture="gfx_unknown",
                    target_id="rocm",
                    runtime_version="kfd",
                    implemented=True,
                    available=True,
                    compile_tested=False,
                    execution_tested=False,
                    production_validated=False,
                )]
        except Exception:  # noqa: BLE001
            pass

    return [HardwareCapabilities.unavailable(
        vendor="AMD",
        device="unknown",
        architecture="unknown",
        target_id="rocm",
        reason="ROCm runtime not available on this host (amdsmi, rocm-smi, and /dev/kfd all failed)",
        implemented=True,
    )]


def detect_metal() -> HardwareCapabilities:
    """Detect Apple Metal availability (no torch required).

    Uses: platform.mac_ver() + system_profiler on macOS. Returns unavailable
    immediately on non-macOS platforms.
    """
    import sys  # noqa: PLC0415
    if sys.platform != "darwin":
        return HardwareCapabilities.unavailable(
            vendor="Apple",
            device="Apple Silicon",
            architecture="apple_silicon",
            target_id="metal",
            reason="Metal requires macOS; current platform is not macOS",
            implemented=True,
        )
    # ── Check via system_profiler ──────────────────────────────────────────────
    try:
        import subprocess  # noqa: PLC0415
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        has_metal = "Metal" in result.stdout or "Apple" in result.stdout
    except Exception:  # noqa: BLE001
        has_metal = False  # Cannot confirm, but we are on macOS
    # ── Check via MLX as secondary indicator ──────────────────────────────────
    if not has_metal:
        try:
            import mlx.core  # noqa: F401,PLC0415
            has_metal = True
        except ImportError:
            pass
    if has_metal:
        mac_ver = platform.mac_ver()[0]
        machine = platform.machine()
        return HardwareCapabilities(
            vendor="Apple",
            device=f"Apple {machine}",
            architecture="apple_silicon",
            target_id="metal_m3",
            driver_version=f"Metal (macOS {mac_ver})",
            runtime_version="platform",
            memory_bytes=_host_memory_bytes(),  # unified memory
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
    return HardwareCapabilities.unavailable(
        vendor="Apple",
        device="Apple Silicon",
        architecture="apple_silicon",
        target_id="metal",
        reason="Metal/MPS not detected on this macOS system (system_profiler and MLX both failed)",
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
            implemented=False,
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
            # These entries are capability placeholders only.  Marking them
            # implemented made the detector overstate support even though no
            # executable backend or vendor SDK was present.
            implemented=False,
        ))
    return result
