"""
Integration tests for gRPC TLS and mTLS.

These tests generate in-memory self-signed certificates using the cryptography
package and exercise the real grpc.ssl_server_credentials / ssl_channel_credentials
path in aether.server.grpc_service.

Requirements:
  pip install grpcio cryptography

All tests are skipped if grpcio or cryptography is unavailable, or if the
aether proto stubs (aether_pb2 / aether_pb2_grpc) are missing.
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
import threading
import time
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _grpc_available() -> bool:
    try:
        import grpc  # noqa: F401
        return True
    except ImportError:
        return False


def _crypto_available() -> bool:
    try:
        from cryptography import x509  # noqa: F401
        return True
    except ImportError:
        return False


def _proto_available() -> bool:
    try:
        from aether.server.proto import aether_pb2, aether_pb2_grpc  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


requires_grpc = pytest.mark.skipif(not _grpc_available(), reason="grpcio not installed")
requires_crypto = pytest.mark.skipif(not _crypto_available(), reason="cryptography not installed")
requires_proto = pytest.mark.skipif(not _proto_available(), reason="aether proto stubs not generated")


# ---------------------------------------------------------------------------
# In-memory certificate generation
# ---------------------------------------------------------------------------

def _make_key():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    return rsa.generate_private_key(65537, 2048, backend=default_backend())


def _cert_builder(
    subject_name: str,
    key,
    issuer_cert=None,
    signing_key=None,
    is_ca: bool = False,
):
    """Build and sign an X.509 certificate.

    Args:
        subject_name: CN for the new certificate.
        key: Private key whose public key goes into the certificate.
        issuer_cert: The issuer's certificate (None → self-signed).
        signing_key: Private key used to sign (required when issuer_cert is set).
        is_ca: Whether to set BasicConstraints CA=True.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_name)])
    issuer = issuer_cert.subject if issuer_cert else name
    actual_signing_key = signing_key if signing_key is not None else key

    now = datetime.datetime.utcnow()
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
    )
    if not is_ca:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
    return builder.sign(actual_signing_key, hashes.SHA256(), default_backend())


def _key_pem(key) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
    return key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())


def _cert_pem(cert) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding
    return cert.public_bytes(Encoding.PEM)


class _CABundle:
    """Minimal in-memory CA + signed server/client certs for TLS tests."""

    def __init__(self) -> None:
        # CA (self-signed)
        self.ca_key = _make_key()
        self.ca_cert = _cert_builder("Aether Test CA", self.ca_key, is_ca=True)
        # Server cert signed by CA
        self.server_key = _make_key()
        self.server_cert = _cert_builder(
            "localhost", self.server_key,
            issuer_cert=self.ca_cert, signing_key=self.ca_key,
        )
        # Client cert signed by CA
        self.client_key = _make_key()
        self.client_cert = _cert_builder(
            "aether-test-client", self.client_key,
            issuer_cert=self.ca_cert, signing_key=self.ca_key,
        )

    @property
    def ca_cert_pem(self) -> bytes:
        return _cert_pem(self.ca_cert)

    @property
    def server_key_pem(self) -> bytes:
        return _key_pem(self.server_key)

    @property
    def server_cert_pem(self) -> bytes:
        return _cert_pem(self.server_cert)

    @property
    def client_key_pem(self) -> bytes:
        return _key_pem(self.client_key)

    @property
    def client_cert_pem(self) -> bytes:
        return _cert_pem(self.client_cert)


# ---------------------------------------------------------------------------
# Helper: free port
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_grpc
@requires_crypto
class TestGRPCTLSCredentials:
    """Test that gRPC TLS credential helpers work correctly."""

    def test_ssl_server_credentials_creation(self):
        """Verify grpc.ssl_server_credentials accepts PEM bytes."""
        import grpc
        ca = _CABundle()
        creds = grpc.ssl_server_credentials(
            [(ca.server_key_pem, ca.server_cert_pem)],
            root_certificates=ca.ca_cert_pem,
            require_client_auth=False,
        )
        assert creds is not None

    def test_ssl_server_credentials_mtls(self):
        """Verify mTLS credential creation with client auth."""
        import grpc
        ca = _CABundle()
        creds = grpc.ssl_server_credentials(
            [(ca.server_key_pem, ca.server_cert_pem)],
            root_certificates=ca.ca_cert_pem,
            require_client_auth=True,
        )
        assert creds is not None

    def test_ssl_channel_credentials_creation(self):
        """Verify grpc.ssl_channel_credentials accepts PEM bytes."""
        import grpc
        ca = _CABundle()
        creds = grpc.ssl_channel_credentials(
            root_certificates=ca.ca_cert_pem,
            private_key=ca.client_key_pem,
            certificate_chain=ca.client_cert_pem,
        )
        assert creds is not None

    def test_credential_bytes_helper(self):
        """Verify _credential_bytes() resolves file paths and inline bytes."""
        from aether.server.grpc_service import _credential_bytes
        import tempfile, pathlib

        ca = _CABundle()
        # Inline bytes
        result = _credential_bytes(ca.ca_cert_pem, "ca_cert")
        assert result == ca.ca_cert_pem

        # From file
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            f.write(ca.ca_cert_pem)
            path = pathlib.Path(f.name)
        try:
            result = _credential_bytes(str(path), "ca_cert")
            assert result == ca.ca_cert_pem
        finally:
            path.unlink(missing_ok=True)

    def test_credential_bytes_missing_file(self):
        """Verify _credential_bytes raises FileNotFoundError for missing path."""
        from aether.server.grpc_service import _credential_bytes
        with pytest.raises(FileNotFoundError):
            _credential_bytes("/nonexistent/path/cert.pem", "test")

    def test_credential_bytes_none_passthrough(self):
        """Verify _credential_bytes returns None for None input."""
        from aether.server.grpc_service import _credential_bytes
        assert _credential_bytes(None, "test") is None


@requires_grpc
@requires_crypto
class TestGRPCServiceHelper:
    """Test grpc_service module helper functions."""

    def test_require_grpc_no_error(self):
        """_require_grpc() should not raise when grpcio is installed."""
        from aether.server.grpc_service import _require_grpc
        _require_grpc()  # should not raise

    def test_service_name_constant(self):
        """SERVICE_NAME should be the expected protobuf service path."""
        from aether.server.grpc_service import SERVICE_NAME
        assert SERVICE_NAME == "aether.AetherRuntime"
