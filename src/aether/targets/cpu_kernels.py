"""
CPU target kernels.

Defines CPU-specific kernel contracts and backend preferences (AVX-512, NEON).
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class CPUTargetKernels:
    """CPU kernel contracts for AVX-512 and NEON targets."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.is_arm = "neon" in target_id
        self.is_x86 = "avx" in target_id

    @property
    def preferred_attention(self) -> str:
        return "cpu_attention"

    @property
    def supports_int8(self) -> bool:
        return True

    @property
    def supports_int4(self) -> bool:
        return True

    def recommended_flags(self) -> dict[str, Any]:
        """Return recommended compiler flags for CPU."""
        return {
            "use_openblas": True,
            "use_avx512": self.is_x86 and "512" in self.target_id,
            "use_neon": self.is_arm,
            "preferred_attention": self.preferred_attention,
            "supports_int4": self.supports_int4,
            "supports_int8": self.supports_int8,
        }

    def __repr__(self) -> str:
        return f"CPUTargetKernels({self.target_id})"
