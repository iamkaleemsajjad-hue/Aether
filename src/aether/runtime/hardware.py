"""
Hardware detection and fingerprinting for Aether runtime.

The runtime fingerprints the current machine to determine which hardware target
and backend to use. This module supports CUDA, Apple Metal, ROCm, and CPU
backends.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any

from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.types import HardwareTarget


@dataclass
class HardwareFingerprint:
    """Structured fingerprint of the current machine."""

    target_id: str
    """Selected hardware target identifier."""

    target_name: str
    """Human-readable target name."""

    os_name: str
    """Operating system name."""

    cpu_arch: str
    """CPU architecture."""

    cpu_count: int
    """Number of CPU cores."""

    total_ram_gb: float
    """Total system RAM in GB."""

    gpu_name: str | None = None
    """GPU name if available."""

    gpu_count: int = 0
    """Number of GPUs detected."""

    gpu_memory_gb: float = 0.0
    """Total GPU memory per GPU in GB."""

    driver_version: str | None = None
    """GPU driver or framework version."""

    compute_capability: str | None = None
    """CUDA compute capability or equivalent."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional attributes."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_name": self.target_name,
            "os_name": self.os_name,
            "cpu_arch": self.cpu_arch,
            "cpu_count": self.cpu_count,
            "total_ram_gb": self.total_ram_gb,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory_gb,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "attributes": self.attributes,
        }


class HardwareDetector:
    """Detects hardware capabilities and produces a hardware fingerprint."""

    def __init__(self) -> None:
        self.target = HardwareTarget.auto()
        self.profile = HardwareProfile.from_target_id(self.target.value)

    def detect(self) -> HardwareFingerprint:
        """Detect hardware and produce a fingerprint."""
        os_name = platform.system()
        cpu_arch = platform.machine()
        cpu_count = os.cpu_count() or 1
        total_ram_gb = self._get_total_ram_gb()

        gpu_name = None
        gpu_count = 0
        gpu_memory_gb = 0.0
        driver_version = None
        compute_capability = None

        if self.target.value.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    compute_capability = f"{torch.cuda.get_device_capability(0)}"
                    driver_version = torch.version.cuda
            except ImportError:
                pass
        elif self.target.value.startswith("metal"):
            try:
                import platform as pf
                gpu_name = pf.processor()
                gpu_count = 1
                gpu_memory_gb = total_ram_gb  # Unified memory on Apple
                compute_capability = "metal"
            except Exception:
                pass

        profile_name = self.profile.name if self.profile else self.target.value
        return HardwareFingerprint(
            target_id=self.target.value,
            target_name=profile_name,
            os_name=os_name,
            cpu_arch=cpu_arch,
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            driver_version=driver_version,
            compute_capability=compute_capability,
        )

    @staticmethod
    def _get_total_ram_gb() -> float:
        """Get total system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            try:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
            except (ValueError, AttributeError):
                return 16.0

    def __repr__(self) -> str:
        return f"HardwareDetector(target={self.target.value})"
