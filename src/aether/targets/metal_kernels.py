"""
Metal target kernels.

Defines Metal-specific kernel contracts and backend preferences for Apple Silicon.
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class MetalTargetKernels:
    """Metal kernel contracts for Apple Silicon targets."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.generation = 1 if "m1" in target_id else 3

    @property
    def preferred_attention(self) -> str:
        """Return preferred attention implementation."""
        if self.generation >= 3:
            return "mlx_flash_attention"
        return "mlx_attention"

    @property
    def supports_fp16(self) -> bool:
        """Return True if the target supports FP16."""
        return True

    @property
    def supports_int8(self) -> bool:
        """Return True if the target supports INT8 quantization."""
        return True

    def recommended_flags(self) -> dict[str, Any]:
        """Return recommended compiler flags for Metal."""
        return {
            "use_metal_performance_shaders": True,
            "use_mlx": True,
            "preferred_attention": self.preferred_attention,
            "supports_int4": False,
            "supports_int8": self.supports_int8,
        }

    def __repr__(self) -> str:
        return f"MetalTargetKernels({self.target_id})"
