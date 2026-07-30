"""
CUDA targeting pass.

Selects CUDA-specific kernels and backend preferences for NVIDIA targets.
"""

from __future__ import annotations

from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
from aether.targets.cuda_kernels import CUDATargetKernels
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class CUDATarget:
    """CUDA-specific targeting configuration."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.kernels = CUDATargetKernels(target_id)
        self.emitter = KernelEmitter(target_id)

    def flags(self) -> dict[str, object]:
        """Return CUDA-specific compiler flags."""
        return self.kernels.recommended_flags()

    def __repr__(self) -> str:
        return f"CUDATarget({self.target_id})"
