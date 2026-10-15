"""
Aether backend plugin package.

Backends are pluggable execution engines that run AEG artifacts on specific
hardware. Native kernels execute the compiler's graph contract; optional
backend plugins provide additional hardware and frontend support.
"""

from __future__ import annotations

from functools import lru_cache

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
from aether.core.exceptions import BackendNotAvailableError


@lru_cache(maxsize=1)
def _default_registry() -> BackendRegistry:
    """Discover the public convenience registry on first use."""
    return BackendRegistry()


def get_backend(name: str) -> Backend:
    """Return an installed backend by name using the registry's aliases.

    Applications needing custom plugins or isolated backend instances can
    construct their own :class:`BackendRegistry` instead.
    """
    backend = _default_registry().get_backend(name)
    if backend is None or not backend.is_available():
        raise BackendNotAvailableError(
            f"Backend {name!r} is not installed or available", backend_name=name,
        )
    return backend

__all__ = [
    # Base
    "Backend",
    "BackendInfo",
    "BackendRegistry",
    "NativeCPUBackend",
    "get_backend",
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
