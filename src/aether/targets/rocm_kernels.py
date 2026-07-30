"""
ROCm target kernels.

Defines AMD ROCm-specific kernel contracts and backend preferences.
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ROCmTargetKernels:
    """ROCm kernel contracts for RDNA3 and CDNA3."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.arch = "cdna3" if "cdna" in target_id else "rdna3"

    @property
    def preferred_attention(self) -> str:
        """Return preferred attention implementation."""
        if self.arch == "cdna3":
            return "flash_attention_2"
        return "vanilla"

    @property
    def supports_fp16(self) -> bool:
        return True

    @property
    def supports_int8(self) -> bool:
        return True

    def recommended_flags(self) -> dict[str, Any]:
        """Return recommended compiler flags for ROCm."""
        return {
            "use_hip": True,
            "preferred_attention": self.preferred_attention,
            "supports_fp16": self.supports_fp16,
            "supports_int8": self.supports_int8,
            "supports_int4": False,
        }

    def __repr__(self) -> str:
        return f"ROCmTargetKernels({self.target_id})"
