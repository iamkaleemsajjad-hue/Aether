"""
Aether Runtime — Formal Hardware Capability Model.

This module defines the canonical ``HardwareCapabilities`` dataclass required
by PRD §12.  The compiler uses it to validate that compiled artifacts only
contain features supported by the target.  The hardware validation manifest
uses it to record evidence for each target.

Key principle (PRD §4, §57):
  - ``implemented=True`` means backend code exists.
  - ``available=True`` means the hardware/runtime is present on this host.
  - ``execution_tested=True`` means real inference ran on this hardware.
  - ``production_validated=True`` means multi-model, multi-workload testing
    passed on real target hardware with measured results.
  These are NOT equivalent. Never upgrade a status without real evidence.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a backend environment validation check."""

    backend_name: str
    available: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    validated_at: float = field(default_factory=time.time)

    @property
    def all_passed(self) -> bool:
        return self.available and not self.checks_failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "available": self.available,
            "all_passed": self.all_passed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "details": self.details,
            "validated_at": self.validated_at,
        }


# ---------------------------------------------------------------------------
# Memory / device / power info
# ---------------------------------------------------------------------------

@dataclass
class MemoryInfo:
    """Current memory state on a device."""

    total_bytes: int
    free_bytes: int
    used_bytes: int
    device_type: str  # "cuda", "mps", "cpu", "hip", etc.

    @property
    def utilization_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return 100.0 * self.used_bytes / self.total_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "utilization_pct": round(self.utilization_pct, 2),
            "device_type": self.device_type,
        }


@dataclass
class PowerInfo:
    """Power consumption information where hardware telemetry is available."""

    power_draw_watts: float | None  # None = not measurable
    power_limit_watts: float | None
    energy_source: str  # "nvml", "rocm_smi", "tdp_estimate", "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "power_draw_watts": self.power_draw_watts,
            "power_limit_watts": self.power_limit_watts,
            "energy_source": self.energy_source,
            "measurement_available": self.power_draw_watts is not None,
        }


@dataclass
class DeviceInfo:
    """Static device properties."""

    vendor: str
    device_name: str
    architecture: str  # e.g. "sm90", "gfx942", "apple9", "x86_64"
    driver_version: str
    runtime_version: str
    device_index: int = 0
    pcie_bus_id: str | None = None
    compute_capability: str | None = None  # CUDA-specific: "9.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "device_name": self.device_name,
            "architecture": self.architecture,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "device_index": self.device_index,
            "pcie_bus_id": self.pcie_bus_id,
            "compute_capability": self.compute_capability,
        }


# ---------------------------------------------------------------------------
# Core capability object (PRD §12)
# ---------------------------------------------------------------------------

@dataclass
class HardwareCapabilities:
    """
    Canonical hardware capability object (PRD §12).

    The compiler uses this to validate that compiled artifacts do not contain
    features that the target cannot execute.  The hardware validation manifest
    stores one entry per target using this schema.

    Validation levels (must not be conflated — PRD §11, §57):
      implemented         — backend code exists in this repository
      available           — the hardware/runtime is present on this host
      compile_tested      — artifact generation was tested without hardware
      execution_tested    — real inference ran on the physical target
      production_validated — multi-model, multi-workload, measured, reproducible
    """

    # Identity
    vendor: str                    # "NVIDIA", "AMD", "Apple", "Intel", "CPU", …
    device: str                    # "H100", "MI300X", "M4", "i9-13900K", …
    architecture: str              # "sm90", "gfx942", "apple9", "x86_64", …
    target_id: str                 # canonical Aether target ID, e.g. "cuda_sm90"

    # Driver / runtime
    driver_version: str = "unknown"
    runtime_version: str = "unknown"

    # Memory
    memory_bytes: int = 0

    # Precision support
    supports_fp32: bool = True
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_fp4: bool = False
    supports_int8: bool = False
    supports_int4: bool = False

    # Feature support
    supports_cuda_graph: bool = False
    supports_tee: bool = False
    supports_nvlink: bool = False
    supports_peer_access: bool = False
    supports_unified_memory: bool = False
    warp_or_wavefront_size: int = 1  # CPU=1, CUDA=32, ROCm=64

    # Validation level (PRD §11)
    implemented: bool = False
    available: bool = False
    compile_tested: bool = False
    execution_tested: bool = False
    production_validated: bool = False

    # Reason when unavailable
    unavailable_reason: str | None = None

    # Timestamp of last detection
    detected_at: float = field(default_factory=time.time)

    # Free-form metadata
    extra: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------
    # Compiler validation helpers
    # -------------------------------------------------------------------

    def validate_precision(self, precision: str) -> bool:
        """Return True if this target supports the requested precision.

        The compiler calls this before emitting a precision annotation to
        prevent creating artifacts that cannot run on the target.
        """
        prec = precision.upper()
        mapping = {
            "FP32": self.supports_fp32,
            "FP16": self.supports_fp16,
            "BF16": self.supports_bf16,
            "FP8": self.supports_fp8,
            "FP4": self.supports_fp4,
            "NVFP4": self.supports_fp4,
            "MXFP4": self.supports_fp4,
            "INT8": self.supports_int8,
            "INT4": self.supports_int4,
        }
        return mapping.get(prec, False)

    def validate_feature(self, feature: str) -> bool:
        """Return True if the target supports a named feature."""
        feat = feature.lower()
        if feat in ("cuda_graph", "cuda_graphs"):
            return self.supports_cuda_graph
        if feat in ("tee", "confidential_computing"):
            return self.supports_tee
        if feat in ("nvlink",):
            return self.supports_nvlink
        if feat in ("peer_access",):
            return self.supports_peer_access
        if feat in ("unified_memory",):
            return self.supports_unified_memory
        return False

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["precision_summary"] = {
            "fp32": self.supports_fp32,
            "fp16": self.supports_fp16,
            "bf16": self.supports_bf16,
            "fp8": self.supports_fp8,
            "fp4": self.supports_fp4,
            "int8": self.supports_int8,
            "int4": self.supports_int4,
        }
        return d

    @classmethod
    def unavailable(
        cls,
        vendor: str,
        device: str,
        architecture: str,
        target_id: str,
        reason: str,
        implemented: bool = True,
    ) -> "HardwareCapabilities":
        """Construct a capability object for an unavailable backend."""
        return cls(
            vendor=vendor,
            device=device,
            architecture=architecture,
            target_id=target_id,
            implemented=implemented,
            available=False,
            unavailable_reason=reason,
        )

    @classmethod
    def cpu_host(cls) -> "HardwareCapabilities":
        """Construct capability for the current CPU host."""
        import os
        arch = platform.machine()
        proc = platform.processor() or arch
        uname = platform.uname()
        identifier = " ".join(
            value for value in (
                proc,
                uname.processor,
                os.environ.get("PROCESSOR_IDENTIFIER", ""),
            ) if value
        )
        normalized_identifier = identifier.upper()
        if "INTEL" in normalized_identifier:
            cpu_vendor = "Intel"
        elif "AMD" in normalized_identifier or "RYZEN" in normalized_identifier:
            cpu_vendor = "AMD"
        elif "APPLE" in normalized_identifier:
            cpu_vendor = "Apple"
        elif "QUALCOMM" in normalized_identifier or "SNAPDRAGON" in normalized_identifier:
            cpu_vendor = "Qualcomm"
        elif arch.lower() in {"arm64", "aarch64"}:
            cpu_vendor = "ARM"
        else:
            cpu_vendor = "unknown"
        cpu_count = os.cpu_count() or 1
        # Detect the actual instruction set on every supported host.  The old
        # implementation only inspected Linux /proc/cpuinfo, which made
        # Windows report a generic CPU even when NumPy exposed the real CPU
        # feature table (and made AVX-512 claims depend on the target string).
        has_avx512 = False
        try:
            from numpy.core import _multiarray_umath

            features = getattr(_multiarray_umath, "__cpu_features__", {})
            has_avx512 = any(
                bool(features.get(name))
                for name in ("AVX512F", "AVX512BW", "AVX512VL")
            )
        except (ImportError, AttributeError):
            pass
        features: dict[str, bool] = {}
        try:
            from numpy.core import _multiarray_umath
            features = {
                str(name).lower(): bool(value)
                for name, value in getattr(_multiarray_umath, "__cpu_features__", {}).items()
            }
        except (ImportError, AttributeError):
            pass
        if not has_avx512 and platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    has_avx512 = "avx512" in f.read().lower()
            except OSError:
                pass
        return cls(
            vendor="CPU",
            device=proc,
            architecture=arch,
            target_id=(
                "cpu_avx512"
                if has_avx512
                else ("cpu_neon" if arch.lower() in {"arm64", "aarch64"} else "cpu_avx2")
            ),
            driver_version="n/a",
            runtime_version=platform.python_version(),
            memory_bytes=_host_memory_bytes(),
            supports_fp32=True,
            supports_fp16=True,  # PyTorch emulates on x86
            supports_bf16=True,  # PyTorch emulates on x86
            supports_int8=True,
            supports_int4=True,
            warp_or_wavefront_size=1,
            implemented=True,
            available=True,
            compile_tested=True,
            execution_tested=True,  # CPU inference proven
            production_validated=False,  # not yet multi-model validated
            extra={
                "cpu_count": cpu_count,
                "cpu_vendor": cpu_vendor,
                "cpu_model": proc,
                "machine": arch,
                "has_avx512": has_avx512,
                "cpu_features": features,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _host_memory_bytes() -> int:
    """Return total physical memory in bytes (best-effort)."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0
