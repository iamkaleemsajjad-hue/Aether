"""
Aether REST server application.

Uses FastAPI to serve OpenAI-compatible endpoints, model management, hardware
diagnostics, and Prometheus metrics.
"""

from __future__ import annotations

import os
from typing import Any

from aether.runtime import Runtime
from aether.runtime.config import RuntimeConfig
from aether.server.routes import create_router
from aether.core.constants import AETHER_VERSION
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
        from aether.server.middleware import AuthMiddleware
    except ImportError:
        msg = "fastapi is required for the server. Install with: pip install aether-runtime"
        raise ImportError(msg)

    runtime = Runtime(config)
    app = FastAPI(
        title="Aether Runtime API",
        version=AETHER_VERSION,
        description="Aether Runtime — compile any model, run on any hardware.",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    # Expose the single runtime instance to embedded transports (gRPC, tests,
    # and process supervisors) so REST and non-HTTP APIs share model/cache
    # state rather than silently constructing independent runtimes.
    app.state.aether_runtime = runtime

    configured_origins = [origin.strip() for origin in os.environ.get(
        "AETHER_CORS_ORIGINS", "http://localhost,http://127.0.0.1"
    ).split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_keys = [key.strip() for key in os.environ.get("AETHER_API_KEYS", "").split(",") if key.strip()]
    app.add_middleware(AuthMiddleware, api_keys=api_keys)

    router = create_router(runtime)
    app.include_router(router, prefix="/v1")

    @app.get("/health", tags=["System"])
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": AETHER_VERSION,
            "target": runtime.fingerprint.target_id,
            "loaded_models": len(runtime._loaded_models),
        }

    @app.get("/", tags=["System"])
    async def root():
        return {
            "service": "Aether Runtime API",
            "version": AETHER_VERSION,
            "target": runtime.fingerprint.target_id,
            "docs": "/docs",
        }

    logger.info("Aether REST server created")
    return app
