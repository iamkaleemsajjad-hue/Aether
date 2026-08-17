"""Typed gRPC transport for the Aether Runtime API.

The wire contract is defined in ``proto/aether.proto`` and checked-in typed
protobuf bindings live under ``aether.server.proto``.  The service shares the
same Runtime instance as REST and streams text produced by the backend's
incremental decoder.
"""

from __future__ import annotations

from concurrent import futures
from pathlib import Path
from typing import Any, Iterable, Mapping

from aether.core.exceptions import AetherError

try:  # Keep CPU-only imports usable when grpcio is intentionally omitted.
    import grpc
    from google.protobuf.json_format import MessageToDict, ParseDict

    from aether.server.proto import aether_pb2, aether_pb2_grpc
except ImportError:  # pragma: no cover - exercised only in minimal installs
    grpc = None  # type: ignore[assignment]
    MessageToDict = None  # type: ignore[assignment]
    ParseDict = None  # type: ignore[assignment]
    aether_pb2 = None  # type: ignore[assignment]
    aether_pb2_grpc = None  # type: ignore[assignment]

SERVICE_NAME = "aether.AetherRuntime"


def _credential_bytes(value: bytes | bytearray | str | Path | None, name: str) -> bytes | None:
    """Resolve an inline credential or a credential-file path safely."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{name} credential file does not exist: {path}")
    return path.read_bytes()


def _require_grpc() -> None:
    if grpc is None or aether_pb2 is None or aether_pb2_grpc is None:
        raise RuntimeError("gRPC support requires grpcio and protobuf; install aether-runtime")


def _authorize(context: Any, token: str | None) -> None:
    """Enforce the configured bearer token before decoding request content."""
    if not token:
        return
    import hmac

    metadata = dict(context.invocation_metadata())
    supplied = metadata.get("authorization", "")
    # Constant-time comparison: a plain ``!=`` leaks key material through
    # response timing on a remote gRPC channel.
    if not hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")


def _request_kwargs(request: Any) -> dict[str, Any]:
    """Convert typed optional protobuf fields to Runtime generation kwargs."""
    kwargs: dict[str, Any] = {}
    for field_name in ("max_tokens", "temperature", "top_p", "top_k"):
        if request.HasField(field_name):
            kwargs[field_name] = getattr(request, field_name)
    if request.stop:
        kwargs["stop"] = list(request.stop)
    return kwargs


def _response_message(response: Any) -> Any:
    """Convert a Runtime GenerationResponse into a typed protobuf response."""
    _require_grpc()
    message = aether_pb2.GenerateResponse(
        text=response.text,
        prompt_tokens=int(response.usage.get("prompt_tokens", 0)),
        completion_tokens=int(response.usage.get("completion_tokens", 0)),
        total_tokens=int(response.usage.get("total_tokens", 0)),
        finish_reason=str(response.finish_reason),
        backend_name=str(response.metrics.backend_name),
    )
    ParseDict(response.metrics.to_dict(), message.metrics)
    return message


_TypedServicer = aether_pb2_grpc.AetherRuntimeServicer if aether_pb2_grpc else object


class AetherGrpcService(_TypedServicer):
    """Bind typed gRPC requests to a real :class:`aether.runtime.Runtime`."""

    def __init__(self, runtime: Any, auth_token: str | None = None) -> None:
        _require_grpc()
        self.runtime = runtime
        self.auth_token = auth_token

    def Generate(self, request: Any, context: Any) -> Any:  # noqa: N802
        _authorize(context, self.auth_token)
        try:
            if not request.model_id:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_id is required")
            response = self.runtime.generate(
                request.model_id,
                request.prompt,
                **_request_kwargs(request),
            )
            return _response_message(response)
        except AetherError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        raise AssertionError("context.abort must terminate the RPC")

    def GenerateStream(self, request: Any, context: Any) -> Iterable[Any]:  # noqa: N802
        _authorize(context, self.auth_token)
        try:
            if not request.model_id:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_id is required")
            last_index = -1
            for index, chunk in enumerate(
                self.runtime.generate_stream(
                    request.model_id,
                    request.prompt,
                    **_request_kwargs(request),
                )
            ):
                last_index = index
                yield aether_pb2.GenerateChunk(text=str(chunk), index=index, final=False)
            # Keep the terminal marker separate so the server can flush each
            # token as soon as it is produced instead of buffering the last one.
            yield aether_pb2.GenerateChunk(text="", index=last_index + 1, final=True)
        except AetherError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return

    def Health(self, request: Any, context: Any) -> Any:  # noqa: N802
        _authorize(context, self.auth_token)
        return aether_pb2.HealthResponse(status="ok", service=SERVICE_NAME)


def add_grpc_service(server: Any, runtime: Any, auth_token: str | None = None) -> Any:
    """Register the typed Aether service on a synchronous gRPC server."""
    _require_grpc()
    service = AetherGrpcService(runtime, auth_token=auth_token)
    aether_pb2_grpc.add_AetherRuntimeServicer_to_server(service, server)
    return service


def start_grpc_server(
    runtime: Any,
    host: str = "127.0.0.1",
    port: int = 50051,
    auth_token: str | None = None,
    max_workers: int = 8,
    server_private_key: bytes | bytearray | str | Path | None = None,
    server_certificate_chain: bytes | bytearray | str | Path | None = None,
    client_ca: bytes | bytearray | str | Path | None = None,
    require_client_auth: bool = False,
) -> Any:
    """Start and return a real authenticated typed gRPC server.

    With no TLS material the function preserves the local-development
    ``insecure`` transport. Supplying both ``server_private_key`` and
    ``server_certificate_chain`` enables TLS. ``client_ca`` plus
    ``require_client_auth`` enables mutual TLS; a bearer token can still be
    required as an application-level credential.
    """
    _require_grpc()
    private_key = _credential_bytes(server_private_key, "server_private_key")
    certificate_chain = _credential_bytes(server_certificate_chain, "server_certificate_chain")
    ca = _credential_bytes(client_ca, "client_ca")
    if (private_key is None) != (certificate_chain is None):
        raise ValueError("server_private_key and server_certificate_chain must be supplied together")
    if require_client_auth and ca is None:
        raise ValueError("client_ca is required when require_client_auth=True")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_grpc_service(server, runtime, auth_token=auth_token)
    if private_key is not None and certificate_chain is not None:
        credentials = grpc.ssl_server_credentials(
            ((private_key, certificate_chain),),
            root_certificates=ca,
            require_client_auth=require_client_auth,
        )
        bound_port = server.add_secure_port(f"{host}:{port}", credentials)
        transport = "tls" if not require_client_auth else "mtls"
    else:
        bound_port = server.add_insecure_port(f"{host}:{port}")
        transport = "insecure"
    if bound_port == 0:
        raise OSError(f"could not bind gRPC listener on {host}:{port}")
    server.start()
    setattr(server, "aether_port", bound_port)
    setattr(server, "aether_transport", transport)
    return server


class AetherGrpcClient:
    """Typed client for the Aether gRPC contract."""

    def __init__(
        self,
        target: str,
        auth_token: str | None = None,
        root_certificates: bytes | bytearray | str | Path | None = None,
        client_private_key: bytes | bytearray | str | Path | None = None,
        client_certificate_chain: bytes | bytearray | str | Path | None = None,
    ) -> None:
        _require_grpc()
        roots = _credential_bytes(root_certificates, "root_certificates")
        private_key = _credential_bytes(client_private_key, "client_private_key")
        certificate_chain = _credential_bytes(client_certificate_chain, "client_certificate_chain")
        if (private_key is None) != (certificate_chain is None):
            raise ValueError("client_private_key and client_certificate_chain must be supplied together")
        if roots is not None or private_key is not None:
            self.channel = grpc.secure_channel(
                target,
                grpc.ssl_channel_credentials(
                    root_certificates=roots,
                    private_key=private_key,
                    certificate_chain=certificate_chain,
                ),
            )
            self.transport = "tls"
        else:
            self.channel = grpc.insecure_channel(target)
            self.transport = "insecure"
        self.auth_token = auth_token
        self.stub = aether_pb2_grpc.AetherRuntimeStub(self.channel)

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self.auth_token}"),) if self.auth_token else ()

    @staticmethod
    def _request_message(request: Mapping[str, Any]) -> Any:
        _require_grpc()
        model_id = str(request.get("model_id", request.get("model", "")))
        message = aether_pb2.GenerateRequest(
            model_id=model_id,
            prompt=str(request.get("prompt", "")),
        )
        for field_name in ("max_tokens", "temperature", "top_p", "top_k"):
            if field_name in request and request[field_name] is not None:
                setattr(message, field_name, request[field_name])
        if request.get("stop") is not None:
            message.stop.extend(str(value) for value in request["stop"])
        return message

    def health(self) -> dict[str, Any]:
        response = self.stub.Health(aether_pb2.HealthRequest(), metadata=self._metadata())
        return {"status": response.status, "service": response.service}

    def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self.stub.Generate(self._request_message(request), metadata=self._metadata())
        return {
            "text": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "finish_reason": response.finish_reason,
            "backend_name": response.backend_name,
            "metrics": MessageToDict(response.metrics, preserving_proto_field_name=True),
        }

    def generate_stream(self, request: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
        call = self.stub.GenerateStream(self._request_message(request), metadata=self._metadata())
        for chunk in call:
            yield {"text": chunk.text, "index": chunk.index, "final": chunk.final}

    def close(self) -> None:
        self.channel.close()


__all__ = [
    "AetherGrpcClient",
    "AetherGrpcService",
    "add_grpc_service",
    "start_grpc_server",
]
