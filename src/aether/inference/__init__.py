"""Compiled inference workflow helpers."""

from aether.inference.rag import RAGPipeline, Document, RetrievalResult
from aether.inference.multimodal import MultiModalGraphDispatcher, VLMConfig

# Backward-compat stubs for old import surface
RAGPipelinePlan = RAGPipeline

class RetrievalSource:
    VECTOR = "vector"
    BM25   = "bm25"
    GRAPH  = "graph"

ModalityEncoder = VLMConfig
MultiModalGraphPlan = VLMConfig
default_multimodal_plan = VLMConfig()

__all__ = [
    "RAGPipeline",
    "RAGPipelinePlan",
    "RetrievalSource",
    "Document",
    "RetrievalResult",
    "MultiModalGraphDispatcher",
    "VLMConfig",
    "ModalityEncoder",
    "MultiModalGraphPlan",
    "default_multimodal_plan",
]
