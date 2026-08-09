"""
Aether REST server package.

Provides OpenAI-compatible routes (`/v1/chat`, `/v1/generate`, etc.) and
Prometheus-compatible metrics.
"""

from __future__ import annotations

from aether.server.app import create_app
from aether.server.grpc_service import AetherGrpcClient, add_grpc_service, start_grpc_server

__all__ = ["AetherGrpcClient", "add_grpc_service", "create_app", "start_grpc_server"]
