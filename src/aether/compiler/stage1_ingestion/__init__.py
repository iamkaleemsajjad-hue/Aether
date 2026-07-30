"""
Stage 1: Model Ingestion and Graph Extraction.

This package provides the loaders and architecture detectors that convert
model artifacts (SafeTensors, GGUF, ONNX, MLX, PyTorch) into AEG-IR.
"""

from __future__ import annotations

from aether.compiler.stage1_ingestion.architecture_detector import (
    ARCHITECTURE_PATTERNS,
    ArchitectureDetector,
)
from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline

__all__ = [
    "ARCHITECTURE_PATTERNS",
    "ArchitectureDetector",
    "IngestionPipeline",
]
