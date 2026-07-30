"""
Aether REST server application.

Uses FastAPI to serve OpenAI-compatible endpoints, model management, hardware
diagnostics, and Prometheus metrics.
"""

from __future__ import annotations

from typing import Any

from aether.runtime import Runtime
from aether.runtime.config import RuntimeConfig
from aether.server.routes import create_router
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(config: RuntimeConfig | None = None) -> Any:
    """Create and return a configured FastAPI application.

    Args:
        config: Runtime configuration.

    Returns:
        A configured FastAPI app instance.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        msg = "fastapi is required for the server. Install with: pip install aether-runtime"
        raise ImportError(msg)

    runtime = Runtime(config)
    app = FastAPI(
        title="Aether Runtime API",
        version="0.1.0",
        description="Aether Runtime — compile any model, run on any hardware.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = create_router(runtime)
    app.include_router(router, prefix="/v1")

    @app.get("/health", tags=["System"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "target": runtime.fingerprint.target_id}

    @app.get("/", tags=["System"])
    async def root():
        return {
            "service": "Aether Runtime API",
            "version": "0.1.0",
            "target": runtime.fingerprint.target_id,
            "docs": "/docs",
        }

    logger.info("Aether REST server created")
    return app
