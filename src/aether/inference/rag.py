"""
RAG-Native Compilation — Full RAG Pipeline as Compiled AEG Graph.

Aether v3.1 treats the entire RAG pipeline as a compiled execution graph:
LLM + retriever + reranker + embedding model are compiled into one AEG workflow.

AEG RAG Pipeline stages:
  Stage 1: Query encoding (embedding model)
  Stage 2: Parallel retrieval (vector + BM25 + graph sources simultaneously)
  Stage 3: Cross-encoder reranking
  Stage 4: Context assembly + LLM generation

Compile-time optimizations:
  - Embedding + reranker compiled to same hardware target as LLM
  - Async parallel retrieval (all sources simultaneously)
  - Common system prompts pre-computed in KV L2 cache
  - Hot document pre-caching (85% TTFT reduction for frequently retrieved docs)

Research:
  - FlashRAG (2024): modular RAG framework
  - ColBERT v2 (2022): late-interaction retrieval
  - Cross-encoder reranking (Monobert, 2019)
  - RAG (Lewis et al., 2020): retrieval-augmented generation
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compiled RAG pipeline plan (PRD §34.2)
# ---------------------------------------------------------------------------

#: Retrieval kind → AEG-IR search opcode (PRD §34.2 Stage 2).
RETRIEVAL_OPS: dict[str, str] = {
    "vector": "aeg.vector_search",
    "bm25":   "aeg.bm25_search",
    "graph":  "aeg.graph_retrieve",
}


@dataclass
class RetrievalSource:
    """
    One retrieval source in the compiled RAG graph (PRD §34.2).

    All sources in a plan are dispatched concurrently by ``aeg.async_retrieve``,
    so total retrieval latency is the slowest source rather than the sum.

    Args:
        name: Source identifier, e.g. ``docs`` — names the backing index.
        kind: One of ``vector``, ``bm25``, ``graph``.
        top_k: Candidates this source contributes to the rerank pool.
        index: Index artifact URI. Defaults to ``@<name>_index``.
        hops: Graph traversal depth. Only meaningful for ``kind="graph"``.
        weight: Score multiplier applied when fusing across sources.
    """

    name: str
    kind: str = "vector"
    top_k: int = 50
    index: str = ""
    hops: int = 2
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RetrievalSource.name must be a non-empty string")
        kind = self.kind.lower().strip()
        if kind not in RETRIEVAL_OPS:
            raise ValueError(
                f"Unknown retrieval kind {self.kind!r}. "
                f"Choose from: {sorted(RETRIEVAL_OPS)}"
            )
        self.kind = kind
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if not self.index:
            self.index = f"@{self.name}_index"

    @property
    def op(self) -> str:
        """AEG-IR opcode used to query this source."""
        return RETRIEVAL_OPS[self.kind]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": self.op,
            "name": self.name,
            "kind": self.kind,
            "index": self.index,
            "top_k": self.top_k,
            "weight": self.weight,
        }
        if self.kind == "graph":
            d["hops"] = self.hops
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RetrievalSource":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RAGPipelinePlan:
    """
    The whole RAG pipeline compiled as a single AEG graph (PRD §34.1).

    Embedding model, retrievers, reranker, and LLM are compiled together and
    targeted at the same hardware, so a query never crosses a process boundary
    between retrieval and generation. Serialized to
    ``.aeg/graph/rag_pipeline.aeg-ir``.

    Args:
        embedding_model: Compiled query-encoder ``.aeg`` artifact.
        reranker_model: Compiled cross-encoder ``.aeg`` artifact.
        llm_model: Compiled generator ``.aeg`` artifact.
        sources: Retrieval sources queried in parallel.
        rerank_top_k: Documents surviving reranking into the context.
        max_context_tokens: Token budget for ``aeg.context_pack``.
        pin_system_prompt_kv: Pin shared RAG system prompts in the L2 KV cache.
        precache_hot_documents: Pre-cache frequently retrieved docs — PRD §34.2
            reports 85% TTFT reduction for hot documents.
    """

    embedding_model: str
    llm_model: str
    reranker_model: str = ""
    sources: list[RetrievalSource] = field(default_factory=list)
    rerank_top_k: int = 5
    max_context_tokens: int = 4096
    pin_system_prompt_kv: bool = True
    precache_hot_documents: bool = True
    version: str = "rag_pipeline/1.0"

    def __post_init__(self) -> None:
        self.sources = list(self.sources)
        for src in self.sources:
            if not isinstance(src, RetrievalSource):
                raise TypeError(
                    f"sources must contain RetrievalSource instances, "
                    f"got {type(src).__name__}"
                )
        seen: set[str] = set()
        for src in self.sources:
            if src.name in seen:
                raise ValueError(f"duplicate retrieval source: {src.name!r}")
            seen.add(src.name)

    @property
    def candidate_pool_size(self) -> int:
        """Total candidates entering the rerank stage across all sources."""
        return sum(s.top_k for s in self.sources)

    def to_graph(self) -> dict[str, Any]:
        """
        Lower the plan to the four-stage AEG-IR graph of PRD §34.2.

        Stages: encode → async_retrieve → cross_encoder_rerank →
        context_pack + generate.
        """
        stages: list[dict[str, Any]] = [
            {
                "op": "aeg.embedding_encode",
                "model": self.embedding_model,
                "input": "query",
                "output": "query_vector",
            },
            {
                # All sources dispatched concurrently — latency is the slowest
                # source, not the sum.
                "op": "aeg.async_retrieve",
                "sources": [s.to_dict() for s in self.sources],
                "output": f"candidates[{self.candidate_pool_size}]",
            },
            {
                "op": "aeg.cross_encoder_rerank",
                "model": self.reranker_model,
                "input": "candidates",
                "top_k": self.rerank_top_k,
                "output": "ranked_docs",
            },
            {
                "op": "aeg.context_pack",
                "input": "ranked_docs",
                "max_tokens": self.max_context_tokens,
                "output": "context",
            },
            {
                "op": "aeg.generate",
                "model": self.llm_model,
                "inputs": ["query", "context"],
                "output": "response",
            },
        ]

        return {
            "version": self.version,
            "stages": stages,
            "models": {
                "embedding": self.embedding_model,
                "reranker": self.reranker_model,
                "llm": self.llm_model,
            },
            "optimizations": {
                "parallel_retrieval": True,
                "co_compiled_targets": True,
                "pin_system_prompt_kv": self.pin_system_prompt_kv,
                "precache_hot_documents": self.precache_hot_documents,
                # PRD §34.2: hot-document pre-caching cuts TTFT by 85%.
                "hot_document_ttft_reduction": 0.85 if self.precache_hot_documents else 0.0,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "llm_model": self.llm_model,
            "sources": [s.to_dict() for s in self.sources],
            "rerank_top_k": self.rerank_top_k,
            "max_context_tokens": self.max_context_tokens,
            "candidate_pool_size": self.candidate_pool_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RAGPipelinePlan":
        sources = [RetrievalSource.from_dict(s) for s in d.get("sources", [])]
        fields = {
            k: v for k, v in d.items()
            if k in cls.__dataclass_fields__ and k != "sources"
        }
        return cls(sources=sources, **fields)

    def save(self, aeg_dir: str | Path) -> Path:
        """Write the compiled graph to ``.aeg/graph/rag_pipeline.aeg-ir``."""
        out = Path(aeg_dir) / "graph" / "rag_pipeline.aeg-ir"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_graph(), indent=2), encoding="utf-8")
        logger.info("RAG pipeline graph saved", path=str(out), sources=len(self.sources))
        return out

    @classmethod
    def load(cls, aeg_dir: str | Path) -> "RAGPipelinePlan":
        p = Path(aeg_dir) / "graph" / "rag_pipeline.aeg-ir"
        if not p.exists():
            raise FileNotFoundError(f"RAG pipeline graph not found at {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        models = d.get("models", {})
        sources: list[RetrievalSource] = []
        rerank_top_k = 5
        max_context_tokens = 4096
        for stage in d.get("stages", []):
            if stage.get("op") == "aeg.async_retrieve":
                sources = [RetrievalSource.from_dict(s) for s in stage.get("sources", [])]
            elif stage.get("op") == "aeg.cross_encoder_rerank":
                rerank_top_k = stage.get("top_k", 5)
            elif stage.get("op") == "aeg.context_pack":
                max_context_tokens = stage.get("max_tokens", 4096)
        opts = d.get("optimizations", {})
        return cls(
            embedding_model=models.get("embedding", ""),
            reranker_model=models.get("reranker", ""),
            llm_model=models.get("llm", ""),
            sources=sources,
            rerank_top_k=rerank_top_k,
            max_context_tokens=max_context_tokens,
            pin_system_prompt_kv=opts.get("pin_system_prompt_kv", True),
            precache_hot_documents=opts.get("precache_hot_documents", True),
        )


# ---------------------------------------------------------------------------
# Document and retrieval result types
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A retrieved document with metadata."""
    doc_id: str
    text: str
    source: str = ""          # "vector" | "bm25" | "graph"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text[:200] + ("..." if len(self.text) > 200 else ""),
            "source": self.source,
            "score": round(self.score, 4),
            "content_hash": self.content_hash,
        }


@dataclass
class RetrievalResult:
    """Aggregated retrieval result from all sources."""
    query: str
    documents: list[Document]
    retrieval_latency_ms: float = 0.0
    sources_used: list[str] = field(default_factory=list)

    def top_k(self, k: int) -> list[Document]:
        return sorted(self.documents, key=lambda d: d.score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Embedding encoder
# ---------------------------------------------------------------------------

class EmbeddingEncoder:
    """
    Query and document encoder for dense retrieval.

    Production: dispatches to a compiled embedding model AEG.
    This reference implementation uses a simple TF-IDF-style sparse encoding
    that produces a dense vector via random projection.
    """

    def __init__(self, dim: int = 768, rng_seed: int = 42) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(rng_seed)
        # Random projection matrix for BoW → dense embedding
        self._vocab_size = 10000
        self._proj = self._rng.normal(0, 1 / dim ** 0.5, (self._vocab_size, dim)).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a unit-norm dense vector."""
        vec = self._bow_encode(text)
        emb = (vec @ self._proj).astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return (emb / norm)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts, returning (N, dim) matrix."""
        return np.stack([self.encode(t) for t in texts], axis=0)

    def _bow_encode(self, text: str) -> np.ndarray:
        """Simple bag-of-words feature vector."""
        vec = np.zeros(self._vocab_size, dtype=np.float32)
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        for w in words:
            idx = hash(w) % self._vocab_size
            vec[idx] += 1.0
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec


# ---------------------------------------------------------------------------
# Vector store (FAISS-like interface, numpy reference)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    In-memory vector store with approximate nearest-neighbor search.

    Uses brute-force cosine similarity (production: FAISS IVF-PQ or ScaNN).
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def add(self, documents: list[Document], encoder: EmbeddingEncoder) -> None:
        """Index documents with their embeddings."""
        texts = [d.text for d in documents]
        new_embs = encoder.encode_batch(texts)
        self._docs.extend(documents)
        if self._embeddings is None:
            self._embeddings = new_embs
        else:
            self._embeddings = np.concatenate([self._embeddings, new_embs], axis=0)

    def search(
        self, query_emb: np.ndarray, top_k: int = 50
    ) -> list[Document]:
        """Return top-k documents by cosine similarity."""
        if self._embeddings is None or len(self._docs) == 0:
            return []
        scores = (self._embeddings @ query_emb).astype(np.float32)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            doc = Document(
                doc_id=self._docs[i].doc_id,
                text=self._docs[i].text,
                source="vector",
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
            )
            results.append(doc)
        return results

    def __len__(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# BM25 retriever (Okapi BM25)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Okapi BM25 sparse retrieval.

    BM25 score:
      score(q, d) = Σ_{t∈q} IDF(t) × (tf(t,d) × (k1+1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))

    Reference: Robertson & Zaragoza, 2009.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Document] = []
        self._tf: list[dict[str, int]] = []       # term frequency per doc
        self._df: dict[str, int] = {}             # document frequency
        self._avgdl: float = 0.0

    def index(self, documents: list[Document]) -> None:
        self._docs = list(documents)
        self._tf = []
        self._df = {}
        total_len = 0
        for doc in documents:
            terms = self._tokenize(doc.text)
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in set(terms):
                self._df[t] = self._df.get(t, 0) + 1
            total_len += len(terms)
        self._avgdl = total_len / max(len(documents), 1)

    def search(self, query: str, top_k: int = 50) -> list[Document]:
        if not self._docs:
            return []
        query_terms = self._tokenize(query)
        N = len(self._docs)
        scores = np.zeros(N, dtype=np.float64)

        for t in query_terms:
            df = self._df.get(t, 0)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for i, tf_doc in enumerate(self._tf):
                tf = tf_doc.get(t, 0)
                if tf == 0:
                    continue
                dl = sum(tf_doc.values())
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            Document(
                doc_id=self._docs[i].doc_id,
                text=self._docs[i].text,
                source="bm25",
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
            )
            for i in top_idx if scores[i] > 0
        ]

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def __len__(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross-encoder reranker: scores each (query, document) pair jointly.

    Production: compiled cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2).
    Reference: simple lexical overlap + length scoring as approximation.
    """

    def __init__(self, score_fn: Callable[[str, str], float] | None = None) -> None:
        self._score_fn = score_fn or self._lexical_overlap_score

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Rerank documents by cross-encoder score, return top-k.
        """
        scored = []
        for doc in documents:
            score = self._score_fn(query, doc.text)
            scored.append(Document(
                doc_id=doc.doc_id,
                text=doc.text,
                source=doc.source,
                score=score,
                metadata=doc.metadata,
            ))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]

    def _lexical_overlap_score(self, query: str, doc_text: str) -> float:
        """
        Approximate cross-encoder score via lexical overlap.

        Score = |query_terms ∩ doc_terms| / |query_terms| × length_bonus
        """
        q_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        d_terms = set(re.sub(r"[^\w\s]", "", doc_text.lower()).split())
        if not q_terms:
            return 0.0
        overlap = len(q_terms & d_terms) / len(q_terms)
        # Length bonus: prefer medium-length docs (too short = fragmentary)
        length_factor = min(1.0, len(doc_text) / 500) * min(1.0, 2000 / max(len(doc_text), 1))
        return float(overlap * 0.7 + length_factor * 0.3)


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """
    Assembles retrieved documents into a context string for the LLM.

    Handles:
    - Token budget management (max_tokens)
    - Deduplication (same content_hash → keep once)
    - Source attribution formatting
    """

    def __init__(self, max_tokens: int = 4096, tokens_per_char: float = 0.25) -> None:
        self.max_tokens = max_tokens
        self.tokens_per_char = tokens_per_char

    def assemble(
        self,
        query: str,
        documents: list[Document],
        system_prefix: str = "",
    ) -> str:
        """
        Assemble documents into context string within token budget.
        """
        max_chars = int(self.max_tokens / self.tokens_per_char)
        seen_hashes: set[str] = set()
        parts: list[str] = []
        total_chars = 0

        if system_prefix:
            parts.append(system_prefix)
            total_chars += len(system_prefix)

        parts.append(f"Query: {query}\n\nRelevant context:")
        total_chars += len(parts[-1])

        for i, doc in enumerate(documents, 1):
            h = doc.content_hash
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            snippet = f"\n\n[Document {i}] (source: {doc.source}, score: {doc.score:.3f})\n{doc.text}"
            if total_chars + len(snippet) > max_chars:
                # Truncate last document to fit
                remaining = max_chars - total_chars - 50
                if remaining > 100:
                    snippet = snippet[:remaining] + "..."
                    parts.append(snippet)
                break
            parts.append(snippet)
            total_chars += len(snippet)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# RAG Pipeline (full compiled graph)
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Full compiled RAG pipeline: embed → retrieve → rerank → generate.

    Implements the AEG RAG Pipeline Graph from PRD §34:
      Stage 1: Query encoding
      Stage 2: Parallel retrieval (vector + BM25 simultaneously)
      Stage 3: Cross-encoder reranking
      Stage 4: Context assembly + LLM generation

    Compile-time optimizations:
    - KV cache for common system prompts (85% TTFT reduction for hot docs)
    - Parallel retrieval (all sources simultaneously via threading)
    - Pre-computed document embeddings baked into AEG
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        vector_store: VectorStore | None = None,
        bm25: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        assembler: ContextAssembler | None = None,
        generate_fn: Callable[[str], str] | None = None,
        top_k_retrieve: int = 50,
        top_k_rerank: int = 5,
    ) -> None:
        self.encoder     = encoder or EmbeddingEncoder()
        self.vector_store = vector_store or VectorStore()
        self.bm25        = bm25 or BM25Retriever()
        self.reranker    = reranker or CrossEncoderReranker()
        self.assembler   = assembler or ContextAssembler()
        self.generate_fn = generate_fn
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank   = top_k_rerank

        # Hot document KV cache (hash → precomputed context)
        self._context_cache: dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0

    def index_documents(self, documents: list[Document]) -> None:
        """Index documents into both vector store and BM25."""
        self.vector_store.add(documents, self.encoder)
        self.bm25.index(documents)
        logger.info(
            "RAG: indexed %d documents (vector=%d, bm25=%d)",
            len(documents), len(self.vector_store), len(self.bm25)
        )

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Stage 2: Parallel retrieval from all sources.

        In production: parallel async retrieval via asyncio.
        Reference: sequential for simplicity.
        """
        t0 = time.perf_counter()

        # Stage 1: encode query
        query_emb = self.encoder.encode(query)

        # Stage 2: parallel retrieval
        vector_docs = self.vector_store.search(query_emb, top_k=self.top_k_retrieve)
        bm25_docs   = self.bm25.search(query, top_k=self.top_k_retrieve)

        # Merge and deduplicate by doc_id
        seen: set[str] = set()
        all_docs: list[Document] = []
        for doc in vector_docs + bm25_docs:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                all_docs.append(doc)

        latency = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            query=query,
            documents=all_docs,
            retrieval_latency_ms=latency,
            sources_used=["vector", "bm25"],
        )

    def run(
        self,
        query: str,
        system_prompt: str = "",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Full RAG pipeline: encode → retrieve → rerank → assemble → generate.

        Returns dict with response, retrieved_docs, latencies.
        """
        t0 = time.perf_counter()
        self._total_requests += 1

        # Check context cache for hot queries
        cache_key = hashlib.sha256((query + system_prompt).encode()).hexdigest()
        if use_cache and cache_key in self._context_cache:
            context = self._context_cache[cache_key]
            self._cache_hits += 1
        else:
            # Stage 1+2: retrieve
            retrieval = self.retrieve(query)

            # Stage 3: cross-encoder rerank
            reranked = self.reranker.rerank(
                query, retrieval.documents, top_k=self.top_k_rerank
            )

            # Stage 4a: assemble context
            context = self.assembler.assemble(query, reranked, system_prompt)

            if use_cache:
                self._context_cache[cache_key] = context
                self._cache_misses += 1

        # Stage 4b: LLM generation
        t_gen = time.perf_counter()
        if self.generate_fn is not None:
            response = self.generate_fn(context)
        else:
            response = f"[Generated response for: {query[:80]}]"
        gen_latency = (time.perf_counter() - t_gen) * 1000

        total_latency = (time.perf_counter() - t0) * 1000
        return {
            "query": query,
            "response": response,
            "context_length": len(context),
            "cache_hit": cache_key in self._context_cache and self._cache_hits > 0,
            "total_latency_ms": round(total_latency, 2),
            "generation_latency_ms": round(gen_latency, 2),
        }

    def save_to_aeg(self, aeg_dir: str | Path) -> Path:
        """Save RAG pipeline configuration to AEG package."""
        out = Path(aeg_dir) / "graph" / "rag_pipeline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": "rag/1.0",
            "stages": [
                "embedding_encode",
                "parallel_retrieve",
                "cross_encoder_rerank",
                "context_pack",
                "generate",
            ],
            "vector_store_size": len(self.vector_store),
            "bm25_size": len(self.bm25),
            "top_k_retrieve": self.top_k_retrieve,
            "top_k_rerank": self.top_k_rerank,
            "encoder_dim": self.encoder.dim,
            "context_cache_size": len(self._context_cache),
        }
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("RAG pipeline manifest saved", path=str(out))
        return out

    def stats(self) -> dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total, 1)
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(hit_rate, 4),
            "estimated_ttft_reduction": round(hit_rate * 0.85, 4),
            "vector_store_docs": len(self.vector_store),
            "bm25_docs": len(self.bm25),
            "context_cache_entries": len(self._context_cache),
        }
