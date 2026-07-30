"""
ROCm targeting pass.

Selects ROCm-specific kernels and backend preferences for AMD targets.
"""

from __future__ import annotations

from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
from aether.targets.rocm_kernels import ROCmTargetKernels
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ROCmTarget:
    """ROCm-specific targeting configuration."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.kernels = ROCmTargetKernels(target_id)
        self.emitter = KernelEmitter(target_id)

    def flags(self) -> dict[str, object]:
        """Return ROCm-specific compiler flags."""
        return self.kernels.recommended_flags()

    def __repr__(self) -> str:
        return f"ROCmTarget({self.target_id})"
