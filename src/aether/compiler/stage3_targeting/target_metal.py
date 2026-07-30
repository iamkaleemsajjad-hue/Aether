"""
Metal targeting pass.

Selects Metal-specific kernels and backend preferences for Apple Silicon targets.
"""

from __future__ import annotations

from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
from aether.targets.metal_kernels import MetalTargetKernels
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class MetalTarget:
    """Metal-specific targeting configuration."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.kernels = MetalTargetKernels(target_id)
        self.emitter = KernelEmitter(target_id)

    def flags(self) -> dict[str, object]:
        """Return Metal-specific compiler flags."""
        return self.kernels.recommended_flags()

    def __repr__(self) -> str:
        return f"MetalTarget({self.target_id})"
