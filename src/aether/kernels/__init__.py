"""
Kernels package initialization.
"""

from __future__ import annotations

from aether.kernels.base import Kernel
from aether.kernels.attention import AttentionKernel
from aether.kernels.gemm import GEMMKernel
from aether.kernels.ffn import FFNKernel
from aether.kernels.rope import RoPEKernel
from aether.kernels.norm import RMSNormKernel, LayerNormKernel
from aether.kernels.native_cpu import (
    CompilerToolchain,
    NativeCPUKernels,
    NativeKernelError,
    detect_toolchain,
    get_native_kernels,
)

__all__ = [
    "Kernel",
    "AttentionKernel",
    "GEMMKernel",
    "FFNKernel",
    "RoPEKernel",
    "RMSNormKernel",
    "LayerNormKernel",
    # Natively compiled CPU kernels
    "NativeCPUKernels",
    "NativeKernelError",
    "CompilerToolchain",
    "detect_toolchain",
    "get_native_kernels",
]
