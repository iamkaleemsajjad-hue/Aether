"""RAG-native compilation plan helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalSource:
    """One retrieval source in a compiled RAG workflow."""

    source_id: str
    source_type: str
    top_k: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "source_type": self.source_type, "top_k": self.top_k}


@dataclass
class RAGPipelinePlan:
    """Compiled RAG workflow plan emitted into AEG metadata."""

    embedding_model: str
    reranker_model: str
    llm_model: str
    sources: list[RetrievalSource] = field(default_factory=list)
    max_context_tokens: int = 4096

    def to_graph(self) -> dict[str, Any]:
        return {
            "version": "rag_pipeline/1.0",
            "models": {
                "embedding": self.embedding_model,
                "reranker": self.reranker_model,
                "llm": self.llm_model,
            },
            "stages": [
                {"id": "query_encode", "op": "aeg.embedding_encode"},
                {"id": "parallel_retrieve", "op": "aeg.async_retrieve", "sources": [s.to_dict() for s in self.sources]},
                {"id": "rerank", "op": "aeg.cross_encoder_rerank", "top_k": 5},
                {"id": "context_pack", "op": "aeg.context_pack", "max_tokens": self.max_context_tokens},
                {"id": "generate", "op": "aeg.generate"},
            ],
            "optimizations": ["parallel_retrieval", "hot_document_kv_pin", "shared_prompt_cache"],
        }
