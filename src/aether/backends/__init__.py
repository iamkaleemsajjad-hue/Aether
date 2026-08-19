"""
Aether backend plugin package.

Backends are pluggable execution engines that run AEG artifacts on specific
hardware. Aether's value is in selecting and orchestrating the best backend,
not in writing custom kernels for every accelerator.
"""

from __future__ import annotations

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.backends.capabilities import (
    HardwareCapabilities,
    ValidationResult,
    MemoryInfo,
    PowerInfo,
    DeviceInfo,
)
from aether.backends.hardware_detector import (
    detect_all_capabilities,
    detect_cpu,
    detect_cuda_devices,
    detect_rocm_devices,
    detect_metal,
    detect_openvino,
    get_memory_info,
    get_power_info,
    validate_backend_environment,
)
from aether.backends.registry import BackendRegistry
from aether.backends.native_cpu_backend import NativeCPUBackend

__all__ = [
    # Base
    "Backend",
    "BackendInfo",
    "BackendRegistry",
    "NativeCPUBackend",
    "GenerationRequest",
    "GenerationResult",
    # Capability model (PRD §12)
    "HardwareCapabilities",
    "ValidationResult",
    "MemoryInfo",
    "PowerInfo",
    "DeviceInfo",
    # Hardware detection (PRD §41)
    "detect_all_capabilities",
    "detect_cpu",
    "detect_cuda_devices",
    "detect_rocm_devices",
    "detect_metal",
    "detect_openvino",
    "get_memory_info",
    "get_power_info",
    "validate_backend_environment",
]
