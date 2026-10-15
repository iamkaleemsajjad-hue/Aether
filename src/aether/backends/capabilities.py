"""
Aether Runtime — Formal Hardware Capability Model.

This module defines the canonical ``HardwareCapabilities`` dataclass required
by PRD §12.  The compiler uses it to validate that compiled artifacts only
contain features supported by the target.  The hardware validation manifest
uses it to record evidence for each target.

Key principle (PRD §4, §57):
  - ``implemented=True`` means an inference adapter exists for the reported scope.
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
    backend_available: bool | None = None
    execution_tested: bool = False
    execution_scope: str | None = None
    production_validated: bool = False

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
            "backend_available": self.backend_available,
            "execution_tested": self.execution_tested,
            "execution_scope": self.execution_scope,
            "production_validated": self.production_validated,
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
    supports_fp32: bool = False
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
    warp_or_wavefront_size: int = 0  # 0=unconfirmed; CPU=1, queried GPU width

    # Validation level (PRD §11)
    implemented: bool = False
    available: bool = False
    compile_tested: bool = False
    execution_tested: bool = False
    production_validated: bool = False

    # Detection evidence is separate from Aether backend readiness. None means
    # the detector could not confirm that stage; it is never a successful probe.
    hardware_present: bool | None = None
    runtime_available: bool | None = None
    backend_dependencies: dict[str, bool] = field(default_factory=dict)
    backend_available: bool | None = None
    memory_kind: str | None = None
    memory_source: str | None = None

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
        """Return whether native device arithmetic has been confirmed.

        False also covers unconfirmed formats. Storage conversion and reference
        execution are separate compiler/backend contracts; this method does
        not reject an FP16/BF16/INT4 artifact executed with FP32 accumulation.
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
        d["memory_bytes_known"] = bool(self.extra.get("memory_known", self.memory_bytes > 0))
        d["precision_semantics"] = (
            "queried vendor runtime formats; native arithmetic and Aether execution require separate validation"
            if self.extra.get("precision_scope") == "runtime_formats"
            else "confirmed device arithmetic; Aether execution modes are separate"
        )
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
        features = cpu_runtime_features()
        # NumPy's runtime table checks OSXSAVE/XGETBV, not merely CPUID or a
        # model name. A hardware flag without OS register-state support is not
        # usable by the process. Unknown flags never enable an optimized ISA.
        has_avx512 = bool(features.get("avx512f"))
        has_avx2 = bool(features.get("avx2"))
        has_neon = bool(features.get("neon") or features.get("asimd"))
        fp16 = bool((has_avx512 and features.get("avx512fp16")) or (has_neon and features.get("asimdhp")))
        bf16 = bool((has_avx512 and features.get("avx512bf16")) or (has_neon and features.get("bf16")))
        try:
            import numpy
            numpy_available = bool(getattr(numpy, "__version__", None))
        except (ImportError, OSError):
            numpy_available = False
        return cls(
            vendor="CPU",
            device=proc,
            architecture=arch,
            target_id=(
                "cpu_avx512"
                if has_avx512
                else ("cpu_avx2" if has_avx2 else ("cpu_neon" if has_neon else "cpu_generic"))
            ),
            driver_version="n/a",
            runtime_version=platform.python_version(),
            memory_bytes=_host_memory_bytes(),
            supports_fp32=True,
            supports_fp16=fp16,
            supports_bf16=bf16,
            supports_int8=True,
            supports_int4=False,  # Packed INT4 weights are decoded, not an ISA.
            warp_or_wavefront_size=1,
            implemented=True,
            available=numpy_available,
            compile_tested=False,
            execution_tested=False,
            production_validated=False,
            hardware_present=True,
            runtime_available=numpy_available,
            backend_dependencies={"numpy": numpy_available},
            backend_available=numpy_available,
            unavailable_reason=None if numpy_available else "CPU hardware present but the NumPy execution dependency could not be imported",
            memory_kind="host_physical",
            memory_source="host_memory_query",
            extra={
                "cpu_count": cpu_count,
                "cpu_vendor": cpu_vendor,
                "cpu_model": proc,
                "machine": arch,
                "has_avx512": has_avx512,
                "cpu_features": features,
                "cpu_features_source": "numpy_os_usable_runtime" if features else "unconfirmed",
                "precision_evidence": {
                    "fp16": {"native_arithmetic": fp16, "extension_confirmed": bool(features.get("avx512fp16") or features.get("asimdhp"))},
                    "bf16": {"native_arithmetic": bf16, "extension_confirmed": bool(features.get("avx512bf16") or features.get("bf16"))},
                    "f16c_conversion": bool(features.get("f16c")),
                },
                "precision_execution": {
                    "fp32": "native_cpu_or_numpy_reference",
                    "fp16": "storage_converted_to_fp32",
                    "bf16": "storage_converted_to_fp32",
                    "int8": "quantized_weights_with_fp32_accumulation",
                    "int4": "packed_weights_decoded_for_fp32_accumulation",
                },
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cpu_runtime_features() -> dict[str, bool]:
    """Return process-usable CPU flags, conservatively empty if unconfirmed.

    NumPy gates x86 SIMD on OS register-state support. Reading /proc/cpuinfo
    alone cannot establish that execution is safe. Prefer the NumPy 2 module
    name and retain compatibility with NumPy 1 without deprecation warnings.
    """
    import importlib

    for name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            module = importlib.import_module(name)
            return {str(key).lower(): bool(value) for key, value in getattr(module, "__cpu_features__", {}).items()}
        except (ImportError, AttributeError):
            continue
    return {}


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
