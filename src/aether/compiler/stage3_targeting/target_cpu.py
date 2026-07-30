"""
CPU targeting pass.

Selects CPU-specific kernels and backend preferences for x86 and ARM targets.
"""

from __future__ import annotations

from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
from aether.targets.cpu_kernels import CPUTargetKernels
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class CPUTarget:
    """CPU-specific targeting configuration."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.kernels = CPUTargetKernels(target_id)
        self.emitter = KernelEmitter(target_id)

    def flags(self) -> dict[str, object]:
        """Return CPU-specific compiler flags."""
        return self.kernels.recommended_flags()

    def __repr__(self) -> str:
        return f"CPUTarget({self.target_id})"
