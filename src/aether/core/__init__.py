"""
Aether core package — AEG format, IR, graph, types, and hashing.

This package provides the foundational data structures that every other
package depends on. It defines the AEG format specification, the AEG-IR
operator graph, tensor types and layouts, content-addressed hashing, and
version constants.
"""

from __future__ import annotations

from aether.core.aeg_format import AEGManifest, AEGPackage, ModelArchitecture, OptimizationMetadata
from aether.core.aeg_ir import (
    AEGInstruction,
    AEGIRModule,
    AEGOpCode,
    AEGOperand,
    AEGVariable,
    AttributeDict,
    Block,
    Function,
)
from aether.core.constants import (
    AEG_FORMAT_VERSION,
    AETHER_VERSION,
    DEFAULT_CACHE_DIR,
    DEFAULT_HUB_URL,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_TARGETS,
)
from aether.core.exceptions import (
    AEGError,
    AEGFormatError,
    AEGIntegrityError,
    AEGVersionError,
    BackendError,
    CompilationError,
    CompilerConfigError,
    HubError,
    IngestionError,
    KernelError,
    ModelNotFoundError,
    QuantizationError,
    RuntimeConfigError,
    RuntimeError as AetherRuntimeError,
    SchedulingError,
    TargetingError,
)
from aether.core.graph import (
    AEGGraph,
    AEGGraphEdge,
    AEGGraphNode,
    AEGGraphNodeType,
    GraphValidationResult,
)
from aether.core.hash_utils import (
    compute_content_hash,
    compute_file_hash,
    compute_graph_hash,
    verify_content_hash,
    verify_file_hash,
)
from aether.core.types import (
    AEGVersion,
    DType,
    HardwareTarget,
    MemoryLayout,
    MemoryTier,
    Precision,
    PrecisionTier,
    ShardingPlan,
    TensorLayout,
    TensorShape,
)

__all__ = [
    # AEG Format
    "AEGManifest",
    "AEGPackage",
    "ModelArchitecture",
    "OptimizationMetadata",
    # AEG-IR
    "AEGInstruction",
    "AEGIRModule",
    "AEGOpCode",
    "AEGOperand",
    "AEGVariable",
    "AttributeDict",
    "Block",
    "Function",
    # Types
    "AEGVersion",
    "DType",
    "HardwareTarget",
    "MemoryLayout",
    "MemoryTier",
    "Precision",
    "PrecisionTier",
    "ShardingPlan",
    "TensorLayout",
    "TensorShape",
    # Graph
    "AEGGraph",
    "AEGGraphEdge",
    "AEGGraphNode",
    "AEGGraphNodeType",
    "GraphValidationResult",
    # Exceptions
    "AEGError",
    "AEGFormatError",
    "AEGIntegrityError",
    "AEGVersionError",
    "BackendError",
    "CompilationError",
    "CompilerConfigError",
    "HubError",
    "IngestionError",
    "KernelError",
    "ModelNotFoundError",
    "QuantizationError",
    "RuntimeConfigError",
    "AetherRuntimeError",
    "SchedulingError",
    "TargetingError",
    # Hash utils
    "compute_content_hash",
    "compute_file_hash",
    "compute_graph_hash",
    "verify_content_hash",
    "verify_file_hash",
    # Constants
    "AEG_FORMAT_VERSION",
    "AETHER_VERSION",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_HUB_URL",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_TARGETS",
]
