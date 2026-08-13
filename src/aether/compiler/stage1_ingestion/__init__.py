"""
Stage 1: Model Ingestion and Graph Extraction.

This package provides the loaders and architecture detectors that convert
model artifacts (SafeTensors, GGUF, ONNX, MLX, PyTorch, VLM, Video, MLA,
MoE, SSM) into AEG-IR computation graphs.
"""

from __future__ import annotations

from aether.compiler.stage1_ingestion.architecture_detector import (
    ARCHITECTURE_PATTERNS,
    ArchitectureDetector,
)
from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
from aether.compiler.stage1_ingestion.video_loader import (
    VideoModelLoader,
    VideoArchitecture,
    load_video_model,
    detect_video_architecture,
)
from aether.compiler.stage1_ingestion.mla_loader import (
    MLALoader,
    MLAArchitecture,
    load_mla_model,
    is_mla_model,
)
from aether.compiler.stage1_ingestion.moe_loader import (
    MoELoader,
    MoEArchitecture,
    load_moe_model,
    is_moe_model,
)

__all__ = [
    # Core dispatcher
    "ARCHITECTURE_PATTERNS",
    "ArchitectureDetector",
    "IngestionPipeline",
    # Specialised loaders
    "VideoModelLoader",
    "VideoArchitecture",
    "load_video_model",
    "detect_video_architecture",
    "MLALoader",
    "MLAArchitecture",
    "load_mla_model",
    "is_mla_model",
    "MoELoader",
    "MoEArchitecture",
    "load_moe_model",
    "is_moe_model",
]
