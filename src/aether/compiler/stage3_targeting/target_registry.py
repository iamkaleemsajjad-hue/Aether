"""
Target registry — maps AEG graphs and hardware targets to hardware profiles and backend plans.
"""

from __future__ import annotations

from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.constants import SUPPORTED_TARGET_IDS
from aether.core.exceptions import UnsupportedTargetError
from aether.core.types import ModelArchitecture
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class TargetRegistry:
    """Registry of supported hardware targets and their profiles."""

    def __init__(self) -> None:
        """Initialize the target registry with built-in profiles."""
        self._profiles: dict[str, HardwareProfile] = {}
        for target_id in SUPPORTED_TARGET_IDS:
            profile = HardwareProfile.from_target_id(target_id)
            if profile:
                self._profiles[target_id] = profile
        logger.info(f"Target registry loaded with {len(self._profiles)} profiles")

    @property
    def supported_targets(self) -> list[str]:
        """Return a list of supported target IDs."""
        return sorted(self._profiles.keys())

    def is_supported(self, target_id: str) -> bool:
        """Check if a target is supported."""
        return target_id in self._profiles

    def get_profile(self, target_id: str) -> HardwareProfile:
        """Get a hardware profile by ID.

        Raises:
            UnsupportedTargetError: If the target is not supported.
        """
        if target_id not in self._profiles:
            msg = f"Unsupported target: {target_id}. Supported targets: {self.supported_targets}"
            raise UnsupportedTargetError(msg)
        return self._profiles[target_id]

    def create_profiles(
        self,
        graph: Any,
        architecture: ModelArchitecture,
        target_ids: list[str],
    ) -> list[HardwareProfile]:
        """Create hardware profiles for a list of targets.

        Args:
            graph: Optimized computation graph.
            architecture: Model architecture.
            target_ids: List of target IDs.

        Returns:
            List of HardwareProfile objects.
        """
        profiles: list[HardwareProfile] = []
        for target_id in target_ids:
            profile = self.get_profile(target_id)
            # Adjust memory requirements based on model size
            required_gb = architecture.params_billion * 2.0
            if profile.memory_gb < required_gb:
                logger.warning(
                    f"Target {target_id} has {profile.memory_gb}GB, but model needs ~{required_gb}GB; may require parallelism"
                )
            profiles.append(profile)
        return profiles

    def recommend_targets(self, architecture: ModelArchitecture) -> list[str]:
        """Recommend a list of targets based on model size and common hardware.

        Args:
            architecture: Model architecture.

        Returns:
            List of recommended target IDs.
        """
        if architecture.is_moe:
            return ["cuda_sm90", "cuda_sm100", "rocm_cdna3", "cpu_avx512"]
        if architecture.params_billion <= 1.0:
            return ["cuda_sm89", "cuda_sm90", "metal_m3", "cpu_avx512"]
        if architecture.params_billion <= 8.0:
            return ["cuda_sm90", "cuda_sm89", "metal_m3", "rocm_rdna3", "cpu_avx512"]
        return ["cuda_sm90", "cuda_sm100", "rocm_cdna3", "metal_m3", "cpu_avx512"]

    def __repr__(self) -> str:
        return f"TargetRegistry({len(self._profiles)} targets)"
