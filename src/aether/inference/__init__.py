"""
Compiled inference workflow helpers.

Two layers are exported:

* **Plan layer** — compile-time graph plans serialized into the ``.aeg``
  package (:class:`RAGPipelinePlan`, :class:`MultiModalGraphPlan`).
* **Runtime layer** — executable engines that run those graphs
  (:class:`RAGPipeline`, :class:`MultiModalGraphDispatcher`).
"""

from aether.inference.multimodal import (
    ImagePreprocessor,
    ModalConnector,
    ModalityEncoder,
    MultiModalGraphDispatcher,
    MultiModalGraphPlan,
    VisualTokenCompressor,
    ViTEncoder,
    VLMArchitecture,
    VLMConfig,
    default_multimodal_plan,
)
from aether.inference.rag import (
    BM25Retriever,
    ContextAssembler,
    CrossEncoderReranker,
    Document,
    EmbeddingEncoder,
    RAGPipeline,
    RAGPipelinePlan,
    RetrievalResult,
    RetrievalSource,
    VectorStore,
)

__all__ = [
    # RAG — plan layer
    "RAGPipelinePlan",
    "RetrievalSource",
    # RAG — runtime layer
    "RAGPipeline",
    "Document",
    "RetrievalResult",
    "EmbeddingEncoder",
    "VectorStore",
    "BM25Retriever",
    "CrossEncoderReranker",
    "ContextAssembler",
    # Multi-modal — plan layer
    "MultiModalGraphPlan",
    "ModalityEncoder",
    "default_multimodal_plan",
    # Multi-modal — runtime layer
    "MultiModalGraphDispatcher",
    "VLMConfig",
    "VLMArchitecture",
    "ImagePreprocessor",
    "ViTEncoder",
    "VisualTokenCompressor",
    "ModalConnector",
]
