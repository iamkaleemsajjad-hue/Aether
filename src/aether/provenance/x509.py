"""Minimal DER and X.509 support for C2PA claim-signing certificates.

C2PA carries the signer's certificate chain in the COSE protected header
(``x5chain``, label 33), and a validator resolves the signing key from the leaf
certificate rather than from a bare public key.  To emit a spec-shaped manifest
without making ``cryptography`` a hard dependency, this module implements the
narrow slice of DER and X.509 that a claim-signing certificate needs:

* DER encoding of the types that appear in a v3 certificate;
* generation of a self-signed Ed25519 certificate with the key usage and
  extended key usage C2PA requires of a claim signer;
* extraction of the ``SubjectPublicKeyInfo`` key from a certificate, so
  verification can proceed from the chain alone.

What this module deliberately does *not* do is decide trust.  It can prove that
a claim was signed by the key in the leaf certificate; it cannot prove that the
certificate belongs to anyone in particular.  Chain validation against the C2PA
trust list is a separate concern, and
:class:`aether.provenance.c2pa.VerificationResult` reports the distinction
explicitly rather than letting a self-signed certificate read as "verified".

References:
  * ITU-T X.690 (DER), ISO/IEC 8825-1.
  * RFC 5280, "Internet X.509 Public Key Infrastructure Certificate and CRL
    Profile", §4.1 (certificate structure), §4.2.1.3 (key usage),
    §4.2.1.12 (extended key usage).
  * RFC 8410, "Algorithm Identifiers for Ed25519 ... in the Internet X.509
    Public Key Infrastructure" — the ``id-Ed25519`` OID 1.3.101.112 and the
    absent-parameters rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from aether.provenance import ed25519

__all__ = [
    "DERError",
    "OID_ED25519",
    "CertificateInfo",
    "encode_der_length",
    "generate_self_signed_ed25519",
    "public_key_from_certificate",
    "certificate_info",
]

OID_ED25519 = "1.3.101.112"
OID_COMMON_NAME = "2.5.4.3"
OID_ORGANIZATION = "2.5.4.10"
OID_KEY_USAGE = "2.5.29.15"
OID_EXT_KEY_USAGE = "2.5.29.37"
OID_BASIC_CONSTRAINTS = "2.5.29.19"
OID_EKU_EMAIL_PROTECTION = "1.3.6.1.5.5.7.3.4"


class DERError(ValueError):
    """Raised when DER input is malformed or uses an unsupported construct."""


# ── DER writer ────────────────────────────────────────────────────────────────

def encode_der_length(length: int) -> bytes:
    """Encode a DER definite length, always in the shortest permitted form."""
    if length < 0:
        raise DERError("negative DER length")
    if length < 0x80:
        return bytes((length,))
    raw = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(raw),)) + raw


def _tlv(tag: int, payload: bytes) -> bytes:
    return bytes((tag,)) + encode_der_length(len(payload)) + payload


def _sequence(*items: bytes) -> bytes:
    return _tlv(0x30, b"".join(items))


def _set(*items: bytes) -> bytes:
    return _tlv(0x31, b"".join(items))


def _integer(value: int) -> bytes:
    if value == 0:
        return _tlv(0x02, b"\x00")
    if value < 0:
        raise DERError("negative INTEGERs are not needed here")
    raw = value.to_bytes((value.bit_length() + 8) // 8, "big")
    return _tlv(0x02, raw)


def _bit_string(payload: bytes, unused_bits: int = 0) -> bytes:
    return _tlv(0x03, bytes((unused_bits,)) + payload)


def _octet_string(payload: bytes) -> bytes:
    return _tlv(0x04, payload)


def _boolean(value: bool) -> bytes:
    return _tlv(0x01, b"\xff" if value else b"\x00")


def _utf8_string(value: str) -> bytes:
    return _tlv(0x0C, value.encode("utf-8"))


def _oid(dotted: str) -> bytes:
    """Encode an OBJECT IDENTIFIER from its dotted-decimal form."""
    parts = [int(component) for component in dotted.split(".")]
    if len(parts) < 2 or parts[0] > 2 or (parts[0] < 2 and parts[1] >= 40):
        raise DERError(f"invalid OID {dotted!r}")
    body = bytearray([parts[0] * 40 + parts[1]])
    for value in parts[2:]:
        chunk = bytearray()
        chunk.append(value & 0x7F)
        value >>= 7
        while value:
            chunk.append((value & 0x7F) | 0x80)
            value >>= 7
        body += bytes(reversed(chunk))
    return _tlv(0x06, bytes(body))


def _utc_time(epoch_seconds: float) -> bytes:
    """Encode a UTCTime. RFC 5280 mandates UTCTime for years before 2050."""
    stamp = time.strftime("%y%m%d%H%M%SZ", time.gmtime(epoch_seconds))
    return _tlv(0x17, stamp.encode("ascii"))


# ── DER reader ────────────────────────────────────────────────────────────────

def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    """Read one TLV at ``offset``; return ``(tag, value, next_offset)``."""
    if offset + 2 > len(data):
        raise DERError("truncated DER element")
    tag = data[offset]
    length_byte = data[offset + 1]
    cursor = offset + 2
    if length_byte < 0x80:
        length = length_byte
    elif length_byte == 0x80:
        raise DERError("indefinite lengths are not valid DER")
    else:
        count = length_byte & 0x7F
        if cursor + count > len(data):
            raise DERError("truncated DER length")
        length = int.from_bytes(data[cursor:cursor + count], "big")
        cursor += count
    end = cursor + length
    if end > len(data):
        raise DERError("DER element extends past the end of input")
    return tag, data[cursor:end], end


def _children(value: bytes) -> list[tuple[int, bytes]]:
    result: list[tuple[int, bytes]] = []
    offset = 0
    while offset < len(value):
        tag, payload, offset = _read_tlv(value, offset)
        result.append((tag, payload))
    return result


def _decode_oid(payload: bytes) -> str:
    if not payload:
        raise DERError("empty OID")
    first = payload[0]
    components = [str(first // 40), str(first % 40)]
    value = 0
    for byte in payload[1:]:
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            components.append(str(value))
            value = 0
    return ".".join(components)


@dataclass(frozen=True)
class CertificateInfo:
    """The fields of a claim-signing certificate a validator reports."""

    subject: str
    issuer: str
    serial_number: int
    not_before: str
    not_after: str
    public_key_algorithm: str
    public_key: bytes
    self_signed: bool
    public_key_info: bytes = b""
    """The complete ``SubjectPublicKeyInfo`` DER.

    Kept alongside the raw key because the two consumers want different forms:
    Ed25519 verification takes the 32 raw bytes, while ``cryptography``'s
    ``load_der_public_key`` — used for the ECDSA algorithms — needs the SPKI.
    """

    @property
    def is_ed25519(self) -> bool:
        return self.public_key_algorithm == OID_ED25519


def _name(common_name: str, organization: str) -> bytes:
    return _sequence(
        _set(_sequence(_oid(OID_ORGANIZATION), _utf8_string(organization))),
        _set(_sequence(_oid(OID_COMMON_NAME), _utf8_string(common_name))),
    )


def _spki_ed25519(public_key: bytes) -> bytes:
    # RFC 8410 §3: the AlgorithmIdentifier for id-Ed25519 has absent parameters.
    return _sequence(_sequence(_oid(OID_ED25519)), _bit_string(public_key))


def _extensions_for_claim_signer() -> bytes:
    key_usage = _octet_string(_bit_string(b"\x80", unused_bits=7))
    """keyUsage = digitalSignature only: bit 0 set, 7 unused bits."""
    eku = _octet_string(_sequence(_oid(OID_EKU_EMAIL_PROTECTION)))
    basic = _octet_string(_sequence())  # basicConstraints with cA absent (FALSE)
    return _tlv(
        0xA3,
        _sequence(
            _sequence(_oid(OID_BASIC_CONSTRAINTS), _boolean(True), basic),
            _sequence(_oid(OID_KEY_USAGE), _boolean(True), key_usage),
            _sequence(_oid(OID_EXT_KEY_USAGE), _boolean(True), eku),
        ),
    )


def generate_self_signed_ed25519(
    seed: bytes,
    *,
    common_name: str = "Aether Runtime Claim Signer",
    organization: str = "Aether Runtime",
    valid_from: float | None = None,
    valid_days: int = 3650,
    serial_number: int | None = None,
) -> bytes:
    """Build a self-signed v3 Ed25519 certificate in DER form.

    The certificate carries ``keyUsage: digitalSignature`` and
    ``extendedKeyUsage: emailProtection``, which is the profile C2PA requires of
    a claim signer.  It is *self-signed*: it makes the signing key verifiable
    from the manifest alone, and asserts nothing about who holds it.  Production
    deployments pass their own chain to
    :func:`aether.provenance.c2pa.sign_artifact` instead.
    """
    public_key = ed25519.public_key_from_seed(seed)
    start = time.time() if valid_from is None else valid_from
    serial = serial_number if serial_number is not None else (
        int.from_bytes(public_key[:8], "big") | 1
    )
    name = _name(common_name, organization)
    algorithm = _sequence(_oid(OID_ED25519))
    tbs = _sequence(
        _tlv(0xA0, _integer(2)),                     # version v3
        _integer(serial),
        algorithm,
        name,                                        # issuer == subject
        _sequence(_utc_time(start), _utc_time(start + valid_days * 86400.0)),
        name,
        _spki_ed25519(public_key),
        _extensions_for_claim_signer(),
    )
    signature = ed25519.sign(tbs, seed, public_key)
    return _sequence(tbs, algorithm, _bit_string(signature))


def _parse_name(value: bytes) -> str:
    parts: list[str] = []
    for tag, rdn in _children(value):
        if tag != 0x31:
            continue
        for attribute_tag, attribute in _children(rdn):
            if attribute_tag != 0x30:
                continue
            fields = _children(attribute)
            if len(fields) != 2:
                continue
            oid = _decode_oid(fields[0][1])
            text = fields[1][1].decode("utf-8", "replace")
            label = {OID_COMMON_NAME: "CN", OID_ORGANIZATION: "O"}.get(oid, oid)
            parts.append(f"{label}={text}")
    return ", ".join(parts)


def certificate_info(certificate: bytes) -> CertificateInfo:
    """Parse the fields of a DER certificate needed to report a signer.

    Only v1/v3 certificates whose ``SubjectPublicKeyInfo`` this module
    understands are accepted.  An unparseable certificate raises rather than
    yielding a partially-populated record, so a caller can never mistake a
    failed parse for a certificate that simply lacked fields.
    """
    tag, body, end = _read_tlv(certificate, 0)
    if tag != 0x30 or end != len(certificate):
        raise DERError("certificate is not a single DER SEQUENCE")
    top = _children(body)
    if len(top) != 3:
        raise DERError("certificate must hold tbsCertificate, algorithm, signature")
    tbs_bytes = top[0][1]
    tbs = _children(tbs_bytes)
    index = 0
    if tbs and tbs[0][0] == 0xA0:
        index = 1  # explicit version
    try:
        serial = int.from_bytes(tbs[index][1], "big")
        issuer = _parse_name(tbs[index + 2][1])
        validity = _children(tbs[index + 3][1])
        subject = _parse_name(tbs[index + 4][1])
        spki_der = _tlv(0x30, tbs[index + 5][1])
        spki = _children(tbs[index + 5][1])
    except IndexError as exc:
        raise DERError("tbsCertificate is missing required fields") from exc
    algorithm_children = _children(spki[0][1])
    if not algorithm_children:
        raise DERError("SubjectPublicKeyInfo has no algorithm OID")
    algorithm_oid = _decode_oid(algorithm_children[0][1])
    key_bits = spki[1][1]
    if not key_bits or key_bits[0] != 0:
        raise DERError("subjectPublicKey BIT STRING must have zero unused bits")
    public_key = key_bits[1:]
    return CertificateInfo(
        subject=subject,
        issuer=issuer,
        serial_number=serial,
        not_before=validity[0][1].decode("ascii", "replace"),
        not_after=validity[1][1].decode("ascii", "replace"),
        public_key_algorithm=algorithm_oid,
        public_key=public_key,
        self_signed=subject == issuer,
        public_key_info=spki_der,
    )


def public_key_from_certificate(certificate: bytes) -> tuple[str, bytes]:
    """Return ``(algorithm_oid, public_key_bytes)`` from a DER certificate."""
    info = certificate_info(certificate)
    return info.public_key_algorithm, info.public_key
