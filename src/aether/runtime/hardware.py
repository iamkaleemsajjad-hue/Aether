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

    capabilities: list[dict[str, Any]] = field(default_factory=list)
    """Complete detector evidence for every vendor/backend pipeline."""

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
            "capabilities": self.capabilities,
        }


class HardwareDetector:
    """Detects hardware capabilities and produces a hardware fingerprint."""

    def __init__(self) -> None:
        self.target = HardwareTarget.auto()
        self.profile = HardwareProfile.from_target_id(self.target.value)

    def detect(self) -> HardwareFingerprint:
        """Detect hardware and produce a fingerprint."""
        from aether.backends.hardware_detector import detect_all_capabilities

        capabilities = detect_all_capabilities()
        available = [item for item in capabilities if item.available]
        primary = next(
            (
                item for item in available
                if item.vendor in {"NVIDIA", "AMD", "Apple", "Intel"}
            ),
            next((item for item in available if item.vendor == "CPU"), None),
        )
        if primary is not None:
            try:
                self.target = HardwareTarget.from_string(primary.target_id)
            except ValueError:
                # Keep the safe enum-selected target until a new vendor target
                # is added to the canonical target registry.
                self.target = HardwareTarget.auto()
            self.profile = HardwareProfile.from_target_id(self.target.value)
        os_name = platform.system()
        cpu_arch = platform.machine()
        cpu_count = os.cpu_count() or 1
        total_ram_gb = self._get_total_ram_gb()

        gpu_name = None
        gpu_count = 0
        gpu_memory_gb = 0.0
        driver_version = None
        compute_capability = None

        accelerator_caps = [
            item for item in available if item.vendor in {"NVIDIA", "AMD", "Apple"}
        ]
        if accelerator_caps:
            first = accelerator_caps[0]
            gpu_count = len(accelerator_caps)
            gpu_name = first.device
            gpu_memory_gb = first.memory_bytes / (1024 ** 3) if first.memory_bytes else 0.0
            compute_capability = str(first.extra.get("compute_capability", first.architecture))
            driver_version = first.driver_version

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
            attributes={
                "available_targets": [item.target_id for item in available],
                "available_accelerator_count": len(accelerator_caps),
            },
            capabilities=[item.to_dict() for item in capabilities],
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
