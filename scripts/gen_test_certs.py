"""
Generate self-signed TLS certificates for Aether gRPC integration testing.

This script creates a minimal CA + server certificate pair for use in
integration tests.  NEVER use these certificates in production.

Requires: cryptography package (pip install cryptography)

Usage:
    python scripts/gen_test_certs.py --out certs/
    python scripts/gen_test_certs.py --out certs/ --days 365

Output files:
    ca.crt       — self-signed CA certificate
    server.key   — server private key (RSA 2048)
    server.crt   — server certificate signed by the CA
    client.key   — client private key (for mTLS tests)
    client.crt   — client certificate signed by the CA
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import sys
from pathlib import Path


def _require_cryptography() -> None:
    try:
        import cryptography  # noqa: F401
    except ImportError:
        print("Error: 'cryptography' package required — pip install cryptography", file=sys.stderr)
        sys.exit(1)


def _make_key():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )


def _make_ca_cert(key, days: int):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aether Test CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Aether Test Root CA"),
    ])
    now = datetime.datetime.utcnow()
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256(), default_backend())
    )


def _make_server_cert(server_key, ca_key, ca_cert, days: int, hostname: str = "localhost"):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aether Test Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, hostname),
    ])
    now = datetime.datetime.utcnow()
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(ca_key, hashes.SHA256(), default_backend())
    )


def _make_client_cert(client_key, ca_key, ca_cert, days: int):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aether Test Client"),
        x509.NameAttribute(NameOID.COMMON_NAME, "aether-test-client"),
    ])
    now = datetime.datetime.utcnow()
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256(), default_backend())
    )


def _write_pem(path: Path, obj) -> None:
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
    if hasattr(obj, "private_bytes"):
        path.write_bytes(obj.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()))
    else:
        path.write_bytes(obj.public_bytes(Encoding.PEM))
    print(f"  Written: {path}")


def main() -> int:
    _require_cryptography()

    parser = argparse.ArgumentParser(
        description="Generate self-signed TLS certificates for Aether gRPC integration tests."
    )
    parser.add_argument("--out", default="certs/test", help="Output directory for certificates.")
    parser.add_argument("--days", type=int, default=90, help="Certificate validity in days.")
    parser.add_argument("--hostname", default="localhost", help="Server hostname for SAN.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test TLS certificates in {out_dir}/")
    print("WARNING: These are TEST certificates only. Never use in production.\n")

    # CA
    ca_key = _make_key()
    ca_cert = _make_ca_cert(ca_key, args.days)
    _write_pem(out_dir / "ca.key", ca_key)
    _write_pem(out_dir / "ca.crt", ca_cert)

    # Server
    server_key = _make_key()
    server_cert = _make_server_cert(server_key, ca_key, ca_cert, args.days, args.hostname)
    _write_pem(out_dir / "server.key", server_key)
    _write_pem(out_dir / "server.crt", server_cert)

    # Client (for mTLS)
    client_key = _make_key()
    client_cert = _make_client_cert(client_key, ca_key, ca_cert, args.days)
    _write_pem(out_dir / "client.key", client_key)
    _write_pem(out_dir / "client.crt", client_cert)

    print(f"\nAll certificates written to {out_dir}/")
    print("\nUsage in tests:")
    print(f"  CA cert:     {out_dir}/ca.crt")
    print(f"  Server key:  {out_dir}/server.key")
    print(f"  Server cert: {out_dir}/server.crt")
    print(f"  Client key:  {out_dir}/client.key")
    print(f"  Client cert: {out_dir}/client.crt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
