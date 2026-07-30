"""
REST API routes for Aether.

Implements:
- POST /v1/generate — text completion
- POST /v1/chat — chat completion (OpenAI-compatible)
- POST /v1/embeddings — embedding generation
- POST /v1/rerank — document reranking
- POST /v1/transcribe — audio transcription
- POST /v1/compile — compile a model (async job)
- GET /v1/compile/{job_id} — compilation job status
- GET /v1/models — list compiled models
- POST /v1/models/pull — download and compile
- DELETE /v1/models/{name} — remove compiled model
- GET /v1/models/{name} — model info
- POST /v1/metrics — Prometheus endpoint
"""

from __future__ import annotations

import time
from typing import Any

from aether.runtime import Runtime


def create_router(runtime: Runtime) -> Any:
    """Create a FastAPI router with all endpoints."""
    try:
        from fastapi import APIRouter, HTTPException, Depends, Body, Path as FAPath
        from pydantic import BaseModel
    except ImportError:
        msg = "fastapi and pydantic are required for the server"
        raise ImportError(msg)

    router = APIRouter()

    # ── Request/Response models ──────────────────────────────────────────────

    class GenerateRequest(BaseModel):
        model: str
        prompt: str
        max_tokens: int = 1024
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 0
        stream: bool = False
        stop: list[str] | None = None

    class GenerateResponse(BaseModel):
        text: str
        usage: dict[str, int]
        metrics: dict[str, Any]

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str
        messages: list[ChatMessage]
        max_tokens: int = 1024
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

    class EmbedRequest(BaseModel):
        model: str
        input: list[str]

    class RerankRequest(BaseModel):
        model: str
        query: str
        documents: list[str]

    class TranscribeRequest(BaseModel):
        model: str
        audio: str
        language: str | None = None

    class PullRequest(BaseModel):
        model: str

    # ── Routes ───────────────────────────────────────────────────────────────

    @router.post("/generate", tags=["Generation"])
    async def generate(req: GenerateRequest):
        """Text completion."""
        try:
            response = runtime.generate(
                model_id=req.model,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                stop=req.stop,
            )
            return GenerateResponse(
                text=response.text,
                usage=response.usage,
                metrics=response.metrics.to_dict(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/chat", tags=["Chat"])
    async def chat(req: ChatRequest):
        """Chat completion."""
        try:
            messages = [m.model_dump() for m in req.messages]
            response = runtime.chat(
                model_id=req.model,
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            return {
                "model": req.model,
                "text": response.text,
                "usage": response.usage,
                "metrics": response.metrics.to_dict(),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/embeddings", tags=["Embeddings"])
    async def embeddings(req: EmbedRequest):
        """Embedding generation."""
        try:
            vectors = runtime.embed(req.model, req.input)
            return {
                "model": req.model,
                "vectors": vectors,
                "usage": {"prompt_tokens": 0},
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/rerank", tags=["Rerank"])
    async def rerank(req: RerankRequest):
        """Document reranking."""
        try:
            results = runtime.rerank(req.model, req.query, req.documents)
            return {"results": results}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/transcribe", tags=["Transcription"])
    async def transcribe(req: TranscribeRequest):
        """Audio transcription."""
        try:
            text = runtime.transcribe(req.model, req.audio, language=req.language)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/models", tags=["Model Management"])
    async def list_models():
        """List compiled models."""
        models = runtime.list()
        return {"models": models}

    @router.post("/models/pull", tags=["Model Management"])
    async def pull_model(req: PullRequest):
        """Download and compile a model."""
        try:
            runtime.pull(req.model)
            return {"status": "success", "model": req.model}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/models/{name:path}", tags=["Model Management"])
    async def get_model(name: str):
        """Get model info."""
        try:
            info = runtime.info(name)
            return info
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.delete("/models/{name:path}", tags=["Model Management"])
    async def delete_model(name: str):
        """Remove a compiled model."""
        try:
            runtime.remove(name)
            return {"status": "success", "model": name}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/hardware", tags=["System"])
    async def hardware():
        """Return hardware fingerprint."""
        return runtime.hardware()

    @router.get("/kernels", tags=["System"])
    async def kernels():
        """Return active kernel targets."""
        return {"target": runtime.fingerprint.target_id}

    @router.get("/metrics", tags=["System"])
    async def metrics():
        """Return Prometheus-compatible metrics."""
        return {
            "runtime_up": 1,
            "loaded_models": len(runtime._loaded_models),  # type: ignore
            "kv_cache_blocks": runtime.kv_cache.block_count,
            "kv_cache_hit_rate": runtime.kv_cache.hit_rate(),
        }

    @router.get("/health", tags=["System"])
    async def health():
        return {"status": "healthy"}

    return router
