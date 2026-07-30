"""
Stage 3: Hardware Targeting and Backend Selection.

This package provides the target registry, hardware profiles, backend selector,
kernel emitter, and per-target configurations.
"""

from __future__ import annotations

from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.compiler.stage3_targeting.backend_selector import BackendSelector
from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter, KernelPlan
from aether.compiler.stage3_targeting.target_cuda import CUDATarget
from aether.compiler.stage3_targeting.target_metal import MetalTarget
from aether.compiler.stage3_targeting.target_rocm import ROCmTarget
from aether.compiler.stage3_targeting.target_openvino import OpenVINOTarget
from aether.compiler.stage3_targeting.target_cpu import CPUTarget

__all__ = [
    "HardwareProfile",
    "TargetRegistry",
    "BackendSelector",
    "KernelEmitter",
    "KernelPlan",
    "CUDATarget",
    "MetalTarget",
    "ROCmTarget",
    "OpenVINOTarget",
    "CPUTarget",
]
