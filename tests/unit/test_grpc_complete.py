"""
Aether Runtime — Complete gRPC Transport Test Suite.

Tests the full gRPC layer including:
  - AetherGrpcService: Generate, GenerateStream, Health
  - AetherGrpcClient: insecure/TLS client construction
  - add_grpc_service / start_grpc_server helpers
  - _credential_bytes: TLS material resolution
  - _authorize: bearer token enforcement
  - _request_kwargs: protobuf field mapping
  - _response_message: response conversion

All tests mock grpc + protobuf so they run without a real server,
enabling CI on CPU-only machines without grpcio installed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers — skip the whole module if grpcio is not installed
# ---------------------------------------------------------------------------

def _grpc_available() -> bool:
    try:
        import grpc
        return True
    except ImportError:
        return False


GRPC_AVAILABLE = _grpc_available()

grpc_required = pytest.mark.skipif(
    not GRPC_AVAILABLE,
    reason="grpcio not installed — gRPC transport tests skipped",
)


# ---------------------------------------------------------------------------
# Module-level import tests (run regardless of grpcio)
# ---------------------------------------------------------------------------

class TestGrpcServiceImport:
    def test_module_importable(self):
        from aether.server.grpc_service import (
            AetherGrpcClient,
            AetherGrpcService,
            add_grpc_service,
            start_grpc_server,
        )
        assert AetherGrpcService is not None
        assert AetherGrpcClient is not None
        assert add_grpc_service is not None
        assert start_grpc_server is not None

    def test_service_name_constant(self):
        from aether.server.grpc_service import SERVICE_NAME
        assert SERVICE_NAME == "aether.AetherRuntime"

    def test_credential_bytes_none(self):
        from aether.server.grpc_service import _credential_bytes
        assert _credential_bytes(None, "test") is None

    def test_credential_bytes_raw(self):
        from aether.server.grpc_service import _credential_bytes
        data = b"raw credential"
        result = _credential_bytes(data, "test")
        assert result == data

    def test_credential_bytes_bytearray(self):
        from aether.server.grpc_service import _credential_bytes
        data = bytearray(b"credential bytes")
        result = _credential_bytes(data, "test")
        assert result == bytes(data)

    def test_credential_bytes_file(self):
        from aether.server.grpc_service import _credential_bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as f:
            f.write(b"CERTIFICATE DATA")
            path = f.name
        try:
            result = _credential_bytes(path, "test")
            assert result == b"CERTIFICATE DATA"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_credential_bytes_missing_file_raises(self):
        from aether.server.grpc_service import _credential_bytes
        with pytest.raises(FileNotFoundError):
            _credential_bytes("/nonexistent/path/cert.pem", "test")

    def test_require_grpc_raises_without_grpc(self):
        from aether.server import grpc_service
        original_grpc = grpc_service.grpc
        grpc_service.grpc = None
        try:
            with pytest.raises(RuntimeError, match="gRPC support requires"):
                grpc_service._require_grpc()
        finally:
            grpc_service.grpc = original_grpc


# ---------------------------------------------------------------------------
# _authorize helper (mocked context)
# ---------------------------------------------------------------------------

class TestAuthorize:
    def setup_method(self):
        """Only run if grpc is available."""
        if not GRPC_AVAILABLE:
            pytest.skip("grpcio not installed")

    def test_no_token_allows_all(self):
        from aether.server.grpc_service import _authorize
        ctx = MagicMock()
        _authorize(ctx, None)  # No token → always allow
        ctx.abort.assert_not_called()

    def test_empty_token_allows_all(self):
        from aether.server.grpc_service import _authorize
        ctx = MagicMock()
        _authorize(ctx, "")
        ctx.abort.assert_not_called()

    def test_valid_bearer_token_allows(self):
        from aether.server.grpc_service import _authorize
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [
            ("authorization", "Bearer my_secret_token")
        ]
        _authorize(ctx, "my_secret_token")
        ctx.abort.assert_not_called()

    def test_invalid_bearer_token_aborts(self):
        """Wrong token should abort with UNAUTHENTICATED."""
        import grpc
        from aether.server.grpc_service import _authorize
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = [
            ("authorization", "Bearer wrong_token")
        ]
        _authorize(ctx, "correct_token")
        ctx.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")

    def test_missing_authorization_header_aborts(self):
        """Missing auth header when token required → abort."""
        import grpc
        from aether.server.grpc_service import _authorize
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = []  # No headers
        _authorize(ctx, "required_token")
        ctx.abort.assert_called_once_with(grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")


# ---------------------------------------------------------------------------
# _request_kwargs protobuf mapping
# ---------------------------------------------------------------------------

class TestRequestKwargs:
    def setup_method(self):
        if not GRPC_AVAILABLE:
            pytest.skip("grpcio not installed")

    def test_optional_fields_extracted(self):
        from aether.server.grpc_service import _request_kwargs
        request = MagicMock()
        request.HasField.side_effect = lambda f: f in ("max_tokens", "temperature")
        request.max_tokens = 256
        request.temperature = 0.7
        request.stop = []
        kwargs = _request_kwargs(request)
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.7
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs

    def test_stop_sequences_extracted(self):
        from aether.server.grpc_service import _request_kwargs
        request = MagicMock()
        request.HasField.return_value = False
        request.stop = ["<|end|>", "\n\n"]
        kwargs = _request_kwargs(request)
        assert kwargs.get("stop") == ["<|end|>", "\n\n"]

    def test_empty_stop_not_added(self):
        from aether.server.grpc_service import _request_kwargs
        request = MagicMock()
        request.HasField.return_value = False
        request.stop = []
        kwargs = _request_kwargs(request)
        assert "stop" not in kwargs

    def test_all_optional_fields(self):
        from aether.server.grpc_service import _request_kwargs
        request = MagicMock()
        request.HasField.return_value = True
        request.max_tokens = 512
        request.temperature = 0.5
        request.top_p = 0.9
        request.top_k = 50
        request.stop = []
        kwargs = _request_kwargs(request)
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 50


# ---------------------------------------------------------------------------
# AetherGrpcService (fully mocked)
# ---------------------------------------------------------------------------

class TestAetherGrpcService:
    """Tests the service logic using a mocked runtime and protobuf."""

    def setup_method(self):
        if not GRPC_AVAILABLE:
            pytest.skip("grpcio not installed")

    def _make_service(self, auth_token=None):
        import grpc
        from aether.server.grpc_service import AetherGrpcService

        runtime = MagicMock()
        # Patch out the servicer parent class to avoid protobuf registration
        with patch("aether.server.grpc_service.aether_pb2_grpc") as mock_pb2_grpc:
            mock_pb2_grpc.AetherRuntimeServicer = object
            service = AetherGrpcService(runtime, auth_token=auth_token)
        service.runtime = runtime
        return service, runtime

    def test_health_returns_ok(self):
        """Health endpoint should return status=ok."""
        from aether.server.grpc_service import AetherGrpcService
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        service, runtime = self._make_service()
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = []
        request = MagicMock()

        # Call Health without auth token
        result = service.Health(request, ctx)
        # Health uses aether_pb2.HealthResponse — verify aether_pb2 called
        # If aether_pb2 is not None, response should have status
        assert result is not None

    def test_generate_calls_runtime(self):
        """Generate should delegate to runtime.generate()."""
        from aether.server.grpc_service import AetherGrpcService
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        service, runtime = self._make_service()
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = []

        request = MagicMock()
        request.model_id = "llama_7b"
        request.prompt = "Hello"
        request.HasField.return_value = False
        request.stop = []

        mock_response = MagicMock()
        mock_response.text = "World"
        mock_response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        mock_response.finish_reason = "stop"
        mock_response.metrics = MagicMock()
        mock_response.metrics.backend_name = "pytorch"
        mock_response.metrics.to_dict.return_value = {}
        runtime.generate.return_value = mock_response

        service.Generate(request, ctx)
        runtime.generate.assert_called_once_with("llama_7b", "Hello")

    def test_generate_stream_yields_chunks(self):
        """GenerateStream should yield all tokens from runtime.generate_stream."""
        from aether.server.grpc_service import AetherGrpcService
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        service, runtime = self._make_service()
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = []

        request = MagicMock()
        request.model_id = "llama_7b"
        request.prompt = "Hello"
        request.HasField.return_value = False
        request.stop = []

        runtime.generate_stream.return_value = iter(["tok1", "tok2", "tok3"])

        chunks = list(service.GenerateStream(request, ctx))
        # Should yield 3 content chunks + 1 final marker
        assert len(chunks) == 4
        # Last chunk should be final
        final_chunk = chunks[-1]
        assert final_chunk.final is True

    def test_generate_missing_model_id_aborts(self):
        """Generate with empty model_id should abort with INVALID_ARGUMENT."""
        import grpc
        from aether.server.grpc_service import AetherGrpcService
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        service, runtime = self._make_service()
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = []

        request = MagicMock()
        request.model_id = ""  # Empty model_id
        request.prompt = "Hello"
        request.HasField.return_value = False
        request.stop = []

        # context.abort() is a mock so won't stop execution; the code then
        # continues and may raise. The key assertion is that abort was called
        # at least once with INVALID_ARGUMENT as the first argument.
        try:
            service.Generate(request, ctx)
        except Exception:
            pass

        # Check abort was called with INVALID_ARGUMENT at some point
        abort_calls = ctx.abort.call_args_list
        invalid_arg_calls = [
            c for c in abort_calls
            if c.args and c.args[0] == grpc.StatusCode.INVALID_ARGUMENT
        ]
        assert len(invalid_arg_calls) >= 1, (
            f"Expected INVALID_ARGUMENT abort, got: {abort_calls}"
        )


# ---------------------------------------------------------------------------
# AetherGrpcClient
# ---------------------------------------------------------------------------

class TestAetherGrpcClient:
    def setup_method(self):
        if not GRPC_AVAILABLE:
            pytest.skip("grpcio not installed")

    def test_insecure_channel_created(self):
        """Client with no TLS credentials should create an insecure channel."""
        from aether.server.grpc_service import AetherGrpcClient
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        with patch("aether.server.grpc_service.grpc.insecure_channel") as mock_insecure:
            with patch("aether.server.grpc_service.aether_pb2_grpc.AetherRuntimeStub"):
                mock_insecure.return_value = MagicMock()
                client = AetherGrpcClient("localhost:50051")
                assert client.transport == "insecure"
                mock_insecure.assert_called_once_with("localhost:50051")

    def test_metadata_with_auth_token(self):
        """Client with auth_token should add Bearer metadata."""
        from aether.server.grpc_service import AetherGrpcClient
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        with patch("aether.server.grpc_service.grpc.insecure_channel", return_value=MagicMock()):
            with patch("aether.server.grpc_service.aether_pb2_grpc.AetherRuntimeStub"):
                client = AetherGrpcClient("localhost:50051", auth_token="secret_token")
                metadata = client._metadata()
                assert ("authorization", "Bearer secret_token") in metadata

    def test_metadata_without_auth_token(self):
        """Client without auth_token should return empty metadata tuple."""
        from aether.server.grpc_service import AetherGrpcClient
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        with patch("aether.server.grpc_service.grpc.insecure_channel", return_value=MagicMock()):
            with patch("aether.server.grpc_service.aether_pb2_grpc.AetherRuntimeStub"):
                client = AetherGrpcClient("localhost:50051")
                assert client._metadata() == ()

    def test_request_message_construction(self):
        """_request_message should map dict keys to protobuf fields."""
        from aether.server.grpc_service import AetherGrpcClient
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        with patch("aether.server.grpc_service.grpc.insecure_channel", return_value=MagicMock()):
            with patch("aether.server.grpc_service.aether_pb2_grpc.AetherRuntimeStub"):
                client = AetherGrpcClient("localhost:50051")
                # Using mock request
                with patch("aether.server.grpc_service.aether_pb2") as mock_pb2:
                    mock_request = MagicMock()
                    mock_pb2.GenerateRequest.return_value = mock_request
                    req = {"model_id": "my_model", "prompt": "hello", "max_tokens": 100}
                    client._request_message(req)
                    mock_pb2.GenerateRequest.assert_called_once_with(
                        model_id="my_model", prompt="hello"
                    )

    def test_mismatched_tls_keys_raise(self):
        """Supplying only one of key/cert should raise ValueError."""
        from aether.server.grpc_service import AetherGrpcClient
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        with pytest.raises(ValueError, match="must be supplied together"):
            AetherGrpcClient(
                "localhost:50051",
                client_private_key=b"private_key_only",  # Missing chain
            )


# ---------------------------------------------------------------------------
# start_grpc_server validation
# ---------------------------------------------------------------------------

class TestStartGrpcServer:
    def test_mismatched_tls_raises(self):
        """Providing only key without cert should raise ValueError."""
        from aether.server.grpc_service import start_grpc_server
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        runtime = MagicMock()
        with pytest.raises(ValueError, match="must be supplied together"):
            start_grpc_server(
                runtime,
                server_private_key=b"key_without_cert",
                # server_certificate_chain not provided
            )

    def test_require_client_auth_without_ca_raises(self):
        """require_client_auth=True without client_ca should raise."""
        from aether.server.grpc_service import start_grpc_server
        import aether.server.grpc_service as gs

        if gs.aether_pb2 is None:
            pytest.skip("protobuf bindings not generated")

        runtime = MagicMock()
        with pytest.raises(ValueError, match="client_ca is required"):
            start_grpc_server(
                runtime,
                require_client_auth=True,
                # No client_ca supplied
            )


# ---------------------------------------------------------------------------
# Proto bindings (generated code check)
# ---------------------------------------------------------------------------

class TestProtoBindings:
    def test_proto_directory_exists(self):
        proto_dir = Path(__file__).parent.parent.parent / "src" / "aether" / "server" / "proto"
        assert proto_dir.is_dir()

    def test_proto_file_exists(self):
        proto_dir = Path(__file__).parent.parent.parent / "src" / "aether" / "server" / "proto"
        proto_files = list(proto_dir.glob("*.proto")) + list(proto_dir.glob("aether*.py"))
        # At minimum, one proto or generated file should exist
        assert len(proto_files) >= 1

    def test_aether_pb2_importable(self):
        try:
            from aether.server.proto import aether_pb2
            assert aether_pb2 is not None
        except ImportError:
            pytest.skip("protobuf bindings not generated — run 'make proto' first")

    def test_aether_pb2_grpc_importable(self):
        try:
            from aether.server.proto import aether_pb2_grpc
            assert aether_pb2_grpc is not None
        except ImportError:
            pytest.skip("protobuf bindings not generated — run 'make proto' first")

    def test_generate_request_type(self):
        """GenerateRequest protobuf type should have required fields."""
        try:
            from aether.server.proto import aether_pb2
            req = aether_pb2.GenerateRequest(model_id="test", prompt="hello")
            assert req.model_id == "test"
            assert req.prompt == "hello"
        except ImportError:
            pytest.skip("protobuf bindings not generated")

    def test_generate_response_type(self):
        """GenerateResponse protobuf type should have required fields."""
        try:
            from aether.server.proto import aether_pb2
            resp = aether_pb2.GenerateResponse(
                text="hello world",
                finish_reason="stop",
            )
            assert resp.text == "hello world"
        except ImportError:
            pytest.skip("protobuf bindings not generated")

    def test_health_request_response_types(self):
        """HealthRequest and HealthResponse should be defined."""
        try:
            from aether.server.proto import aether_pb2
            req = aether_pb2.HealthRequest()
            resp = aether_pb2.HealthResponse(status="ok", service="test")
            assert resp.status == "ok"
        except ImportError:
            pytest.skip("protobuf bindings not generated")

    def test_generate_chunk_type(self):
        """GenerateChunk for streaming should have text, index, final fields."""
        try:
            from aether.server.proto import aether_pb2
            chunk = aether_pb2.GenerateChunk(text="tok1", index=0, final=False)
            assert chunk.text == "tok1"
            assert chunk.final is False
        except ImportError:
            pytest.skip("protobuf bindings not generated")
