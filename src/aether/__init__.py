"""
Aether Runtime — the compiler for AI models.

Compile any model into a portable Aether Execution Graph (AEG)
and run it on any hardware with maximum performance.
"""

from __future__ import annotations

from aether.core.types import (
    AEGVersion,
    DType,
    HardwareTarget,
    MemoryLayout,
    Precision,
    PrecisionTier,
    TensorLayout,
    TensorShape,
)
from aether.core.constants import (
    AEG_FORMAT_VERSION,
    AETHER_VERSION,
    DEFAULT_CACHE_DIR,
    DEFAULT_HUB_URL,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_TARGETS,
)
from aether.runtime.runtime import Runtime
from aether.runtime.config import RuntimeConfig
from aether.compiler.compiler import Compiler
from aether.compiler.config import CompilerConfig
from aether.compiler.report import QualityReport

__version__ = AETHER_VERSION

__all__ = [
    # Version
    "__version__",
    # Core types
    "AEGVersion",
    "DType",
    "HardwareTarget",
    "MemoryLayout",
    "Precision",
    "PrecisionTier",
    "TensorLayout",
    "TensorShape",
    # Constants
    "AEG_FORMAT_VERSION",
    "AETHER_VERSION",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_HUB_URL",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_TARGETS",
    # Public API
    "Runtime",
    "RuntimeConfig",
    "Compiler",
    "CompilerConfig",
    "QualityReport",
]
