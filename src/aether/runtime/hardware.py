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

    total_ram_gb: float | None
    """Measured total system RAM in GiB, or None when unknown."""

    gpu_name: str | None = None
    """GPU name if available."""

    gpu_count: int = 0
    """Number of GPUs detected."""

    gpu_memory_gb: float | None = None
    """Measured memory of the first GPU in GiB, or None when unknown."""

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
        # Construction does not probe hardware. The complete detector below
        # is the single source of device evidence and target selection.
        self.target: HardwareTarget | None = None
        self.profile: HardwareProfile | None = None

    def detect(self) -> HardwareFingerprint:
        """Detect hardware and produce a fingerprint."""
        from aether.backends.hardware_detector import detect_all_capabilities

        capabilities = detect_all_capabilities()
        known_targets = {target.value for target in HardwareTarget}
        available = [
            item for item in capabilities if item.available and item.target_id in known_targets
        ]
        gpu_prefixes = ("cuda_", "rocm_", "amd_mi", "metal", "openvino_gpu")
        primary = next(
            (
                item
                for item in available
                if item.implemented and item.target_id.startswith(gpu_prefixes)
            ),
            next(
                (
                    item
                    for item in available
                    if item.implemented and item.target_id.startswith("cpu_")
                ),
                None,
            ),
        )
        if primary is None:
            raise RuntimeError("Hardware detection found no available canonical CPU/GPU target")
        self.target = HardwareTarget.from_string(primary.target_id)
        self.profile = HardwareProfile.from_target_id(self.target.value)
        os_name = platform.system()
        cpu_arch = platform.machine()
        cpu_count = os.cpu_count() or 1
        cpu_capability = next((item for item in capabilities if item.vendor == "CPU"), None)
        total_ram_gb = (
            cpu_capability.memory_bytes / (1024**3)
            if cpu_capability is not None and cpu_capability.memory_bytes > 0
            else self._get_total_ram_gb()
        )

        gpu_name = None
        gpu_count = 0
        gpu_memory_gb = None
        driver_version = None
        compute_capability = None

        gpu_caps = [
            item
            for item in capabilities
            if item.target_id.startswith(("cuda", "rocm", "amd_mi", "metal", "openvino_gpu", "gpu_"))
            and (getattr(item, "hardware_present", None) is True or item.available)
            and not item.extra.get("catalog_only")
        ]
        from aether.backends.hardware_inventory import summarize_gpu_devices
        gpu_summary = summarize_gpu_devices(capabilities)
        if gpu_caps:
            first = primary if primary in gpu_caps else gpu_caps[0]
            gpu_count = gpu_summary["count"]
            gpu_name = first.device
            gpu_memory_gb = first.memory_bytes / (1024**3) if first.memory_bytes > 0 else None
            compute_capability = str(first.extra.get("compute_capability", first.architecture))
            driver_version = (
                first.driver_version
                if first.driver_version and not first.driver_version.lower().startswith("unknown")
                else None
            )

        return HardwareFingerprint(
            target_id=self.target.value,
            target_name=primary.device,
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
                "available_targets": list(dict.fromkeys(item.target_id for item in available)),
                "available_accelerator_count": sum(item.available for item in gpu_caps),
                "gpu_count_known": gpu_summary["count_known"],
                "gpu_count_maximum": gpu_summary["maximum_count"],
                "gpu_count_scope": "physical device identities; ambiguous cross-provider matches report a lower bound",
                "hardware_inventory": cpu_capability.extra.get("hardware_inventory") if cpu_capability else None,
                "compiler_profile_name": self.profile.name if self.profile else None,
                "gpu_memory_known": gpu_memory_gb is not None,
                "ram_memory_known": total_ram_gb is not None,
            },
            capabilities=[item.to_dict() for item in capabilities],
        )

    @staticmethod
    def _get_total_ram_gb() -> float | None:
        """Get measured system memory; never substitute an assumed capacity."""
        from aether.backends.capabilities import _host_memory_bytes

        memory_bytes = _host_memory_bytes()
        return memory_bytes / (1024**3) if memory_bytes > 0 else None

    def __repr__(self) -> str:
        return f"HardwareDetector(target={self.target.value if self.target else 'not detected'})"
