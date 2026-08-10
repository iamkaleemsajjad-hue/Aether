"""Typed protobuf bindings for the Aether gRPC API."""

from .aether_pb2 import (
    GenerateChunk,
    GenerateRequest,
    GenerateResponse,
    HealthRequest,
    HealthResponse,
)

__all__ = [
    "GenerateChunk",
    "GenerateRequest",
    "GenerateResponse",
    "HealthRequest",
    "HealthResponse",
]
