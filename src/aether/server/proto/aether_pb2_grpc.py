"""gRPC service bindings for the typed Aether protobuf schema."""

from __future__ import annotations

from typing import Any

import grpc

from . import aether_pb2 as aether__pb2


class AetherRuntimeStub:
    """Typed client stub generated from ``service AetherRuntime``."""

    def __init__(self, channel: grpc.Channel) -> None:
        self.Generate = channel.unary_unary(
            "/aether.AetherRuntime/Generate",
            request_serializer=aether__pb2.GenerateRequest.SerializeToString,
            response_deserializer=aether__pb2.GenerateResponse.FromString,
        )
        self.GenerateStream = channel.unary_stream(
            "/aether.AetherRuntime/GenerateStream",
            request_serializer=aether__pb2.GenerateRequest.SerializeToString,
            response_deserializer=aether__pb2.GenerateChunk.FromString,
        )
        self.Health = channel.unary_unary(
            "/aether.AetherRuntime/Health",
            request_serializer=aether__pb2.HealthRequest.SerializeToString,
            response_deserializer=aether__pb2.HealthResponse.FromString,
        )


class AetherRuntimeServicer:
    """Base typed service implementation."""

    def Generate(self, request: Any, context: Any) -> Any:  # noqa: N802
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented")
        raise NotImplementedError("Generate")

    def GenerateStream(self, request: Any, context: Any) -> Any:  # noqa: N802
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented")
        raise NotImplementedError("GenerateStream")

    def Health(self, request: Any, context: Any) -> Any:  # noqa: N802
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented")
        raise NotImplementedError("Health")


def add_AetherRuntimeServicer_to_server(servicer: AetherRuntimeServicer, server: grpc.Server) -> None:
    """Register the typed service on a gRPC server."""
    rpc_method_handlers = {
        "Generate": grpc.unary_unary_rpc_method_handler(
            servicer.Generate,
            request_deserializer=aether__pb2.GenerateRequest.FromString,
            response_serializer=aether__pb2.GenerateResponse.SerializeToString,
        ),
        "GenerateStream": grpc.unary_stream_rpc_method_handler(
            servicer.GenerateStream,
            request_deserializer=aether__pb2.GenerateRequest.FromString,
            response_serializer=aether__pb2.GenerateChunk.SerializeToString,
        ),
        "Health": grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=aether__pb2.HealthRequest.FromString,
            response_serializer=aether__pb2.HealthResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("aether.AetherRuntime", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


__all__ = ["AetherRuntimeServicer", "AetherRuntimeStub", "add_AetherRuntimeServicer_to_server"]
