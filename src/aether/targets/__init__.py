"""
Targets package initialization.
"""

from __future__ import annotations

from aether.targets.registry import TargetRegistry, TargetInfo
from aether.targets.templates import TemplateLibrary
from aether.targets.cuda_kernels import CUDATargetKernels
from aether.targets.metal_kernels import MetalTargetKernels
from aether.targets.rocm_kernels import ROCmTargetKernels
from aether.targets.cpu_kernels import CPUTargetKernels

__all__ = [
    "TargetRegistry",
    "TargetInfo",
    "TemplateLibrary",
    "CUDATargetKernels",
    "MetalTargetKernels",
    "ROCmTargetKernels",
    "CPUTargetKernels",
]
