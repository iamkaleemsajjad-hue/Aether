"""gRPC transport for the real Aether Runtime API.

The service deliberately uses ``google.protobuf.Struct`` messages so clients
can evolve request/response metadata without regenerating code for every
minor SDK release.  The canonical wire methods and their serializers are
provided here, alongside the checked-in ``proto/aether.proto`` definition.
"""

from __future__ import annotations

import json
import threading
from concurrent import futures
from typing import Any, Iterable, Mapping

from aether.core.exceptions import AetherError

try:  # Optional import keeps CPU-only library imports usable without grpcio.
    import grpc
    from google.protobuf.json_format import MessageToDict, ParseDict
    from google.protobuf.struct_pb2 import Struct
except ImportError:  # pragma: no cover - exercised only in minimal installs
    grpc = None  # type: ignore[assignment]
    MessageToDict = ParseDict = Struct = None  # type: ignore[assignment,misc]

SERVICE_NAME = "aether.AetherRuntime"


def _require_grpc() -> None:
    if grpc is None or Struct is None:
        raise RuntimeError("gRPC support requires grpcio and protobuf; install aether-runtime[gRPC]")


def _decode(payload: bytes) -> dict[str, Any]:
    _require_grpc()
    message = Struct()
    message.ParseFromString(payload)
    return dict(MessageToDict(message, preserving_proto_field_name=True))


def _encode(value: Mapping[str, Any]) -> bytes:
    _require_grpc()
    message = Struct()
    ParseDict(json.loads(json.dumps(dict(value), default=str)), message)
    return message.SerializeToString()


def _authorize(context: Any, token: str | None) -> None:
    if not token:
        return
    metadata = dict(context.invocation_metadata())
    supplied = metadata.get("authorization", "")
    if supplied != f"Bearer {token}":
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")


class AetherGrpcService:
    """Bind gRPC requests to a real :class:`aether.runtime.Runtime` instance."""

    def __init__(self, runtime: Any, auth_token: str | None = None) -> None:
        _require_grpc()
        self.runtime = runtime
        self.auth_token = auth_token

    def generate(self, request: bytes, context: Any) -> bytes:
        _authorize(context, self.auth_token)
        try:
            data = _decode(request)
            model_id = str(data.get("model_id", data.get("model", "")))
            prompt = data.get("prompt")
            if not model_id or prompt is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_id and prompt are required")
            kwargs: dict[str, Any] = {}
            if "max_tokens" in data:
                kwargs["max_tokens"] = int(data["max_tokens"])
            if "top_k" in data:
                kwargs["top_k"] = int(data["top_k"])
            if "temperature" in data:
                kwargs["temperature"] = float(data["temperature"])
            if "top_p" in data:
                kwargs["top_p"] = float(data["top_p"])
            if "stop" in data:
                kwargs["stop"] = [str(value) for value in data["stop"]]
            response = self.runtime.generate(model_id, str(prompt), **kwargs)
            return _encode(response.to_dict())
        except AetherError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        raise AssertionError("context.abort must terminate the RPC")

    def generate_stream(self, request: bytes, context: Any) -> Iterable[bytes]:
        _authorize(context, self.auth_token)
        try:
            data = _decode(request)
            model_id = str(data.get("model_id", data.get("model", "")))
            prompt = data.get("prompt")
            if not model_id or prompt is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_id and prompt are required")
            kwargs: dict[str, Any] = {}
            if "max_tokens" in data:
                kwargs["max_tokens"] = int(data["max_tokens"])
            if "top_k" in data:
                kwargs["top_k"] = int(data["top_k"])
            if "temperature" in data:
                kwargs["temperature"] = float(data["temperature"])
            if "top_p" in data:
                kwargs["top_p"] = float(data["top_p"])
            if "stop" in data:
                kwargs["stop"] = [str(value) for value in data["stop"]]
            last_index = -1
            for index, chunk in enumerate(
                self.runtime.generate_stream(model_id, str(prompt), **kwargs)
            ):
                last_index = index
                yield _encode({"text": str(chunk), "index": index, "final": False})
            # The terminal marker is separate from text so the server does not
            # need to buffer the final token just to label it as final.
            yield _encode({"text": "", "index": last_index + 1, "final": True})
        except AetherError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return

    def health(self, request: bytes, context: Any) -> bytes:
        _authorize(context, self.auth_token)
        return _encode({"status": "ok", "service": SERVICE_NAME})


def add_grpc_service(server: Any, runtime: Any, auth_token: str | None = None) -> Any:
    """Register Aether RPC handlers on a synchronous ``grpc.Server``."""
    _require_grpc()
    service = AetherGrpcService(runtime, auth_token=auth_token)
    handlers = {
        "Generate": grpc.unary_unary_rpc_method_handler(
            service.generate, request_deserializer=lambda value: value, response_serializer=lambda value: value
        ),
        "GenerateStream": grpc.unary_stream_rpc_method_handler(
            service.generate_stream, request_deserializer=lambda value: value, response_serializer=lambda value: value
        ),
        "Health": grpc.unary_unary_rpc_method_handler(
            service.health, request_deserializer=lambda value: value, response_serializer=lambda value: value
        ),
    }
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE_NAME, handlers),))
    return service


def start_grpc_server(
    runtime: Any,
    host: str = "127.0.0.1",
    port: int = 50051,
    auth_token: str | None = None,
    max_workers: int = 8,
) -> Any:
    """Start and return a real gRPC server; caller owns ``stop()``."""
    _require_grpc()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_grpc_service(server, runtime, auth_token=auth_token)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise OSError(f"could not bind gRPC listener on {host}:{port}")
    server.start()
    # Expose the resolved ephemeral port for tests and embedded callers using
    # port=0 without depending on grpc's private server state.
    setattr(server, "aether_port", bound_port)
    return server


class AetherGrpcClient:
    """Small typed client for the Aether Struct-based gRPC contract."""

    def __init__(self, target: str, auth_token: str | None = None) -> None:
        _require_grpc()
        self.channel = grpc.insecure_channel(target)
        self.auth_token = auth_token

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self.auth_token}"),) if self.auth_token else ()

    def health(self) -> dict[str, Any]:
        call = self.channel.unary_unary(f"/{SERVICE_NAME}/Health", request_serializer=_encode, response_deserializer=_decode)
        return call({}, metadata=self._metadata())

    def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        call = self.channel.unary_unary(f"/{SERVICE_NAME}/Generate", request_serializer=_encode, response_deserializer=_decode)
        return call(dict(request), metadata=self._metadata())

    def generate_stream(self, request: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
        call = self.channel.unary_stream(f"/{SERVICE_NAME}/GenerateStream", request_serializer=_encode, response_deserializer=_decode)
        return call(dict(request), metadata=self._metadata())

    def close(self) -> None:
        self.channel.close()


__all__ = ["AetherGrpcClient", "AetherGrpcService", "add_grpc_service", "start_grpc_server"]
