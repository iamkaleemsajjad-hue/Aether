"""
CUDA target kernels.

Defines CUDA-specific kernel descriptors and backend preferences. The actual
CUDA kernels are provided by the backend plugin (e.g., vLLM, TensorRT-LLM, or
Triton). This module describes the target contract.
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class CUDATargetKernels:
    """CUDA kernel contracts for SM70 through SM100."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.sm_version = int(target_id.split("_sm")[1]) if "_sm" in target_id else 80

    @property
    def preferred_attention(self) -> str:
        """Return preferred attention implementation."""
        if self.sm_version >= 80:
            return "flash_attention_2"
        return "vanilla"

    @property
    def supports_fp8(self) -> bool:
        """Return True if the target supports FP8."""
        return self.sm_version >= 90

    @property
    def supports_int4_gemm(self) -> bool:
        """Return True if the target supports INT4 GEMM."""
        return self.sm_version >= 75

    def recommended_flags(self) -> dict[str, Any]:
        """Return recommended compiler flags for CUDA."""
        return {
            "target_sm": self.sm_version,
            "use_flash_attention": self.sm_version >= 80,
            "use_fp8": self.supports_fp8,
            "use_int4_gemm": self.supports_int4_gemm,
            "enable_cuda_graphs": True,
        }

    def __repr__(self) -> str:
        return f"CUDATargetKernels({self.target_id})"
