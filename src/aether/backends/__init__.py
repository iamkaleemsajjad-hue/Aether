"""
Aether backend plugin package.

Backends are pluggable execution engines that run AEG artifacts on specific
hardware. Aether's value is in selecting and orchestrating the best backend,
not in writing custom kernels for every accelerator.
"""

from __future__ import annotations

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.backends.registry import BackendRegistry

__all__ = [
    "Backend",
    "BackendInfo",
    "BackendRegistry",
    "GenerationRequest",
    "GenerationResult",
]
