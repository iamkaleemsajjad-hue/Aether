"""COSE_Sign1 signing and verification (RFC 9052) for C2PA claim signatures.

A C2PA claim signature is a ``COSE_Sign1_Tagged`` structure whose payload is the
serialized claim, carried in *detached* form: the claim bytes live in the
manifest's claim box, and the signature box holds only the COSE envelope with a
``nil`` payload.  Verification therefore has to reconstruct the exact
``Sig_structure`` the signer built — which is why the deterministic CBOR encoder
in :mod:`aether.provenance.cbor` matters as much as the curve arithmetic.

Structure signed (RFC 9052 §4.4):

    Sig_structure = [
        context         : "Signature1",
        body_protected  : bstr,          ; the protected header, as encoded
        external_aad    : bstr,
        payload         : bstr           ; the claim bytes, even when detached
    ]

The protected header is signed *as the bytes that appear in the envelope*, not
re-encoded from its decoded map.  Re-encoding is the classic COSE verification
bug: a signer and a verifier that serialize the same header differently will
disagree about a signature that is in fact valid.

Two backends are supported and produce identical bytes: ``cryptography`` when it
is installed, and :mod:`aether.provenance.ed25519` otherwise.  Ed25519 signing is
deterministic, so a signature does not depend on which backend produced it —
``tests/unit/test_c2pa_signing.py`` asserts exactly that when both are present.

References:
  * RFC 9052, "CBOR Object Signing and Encryption (COSE): Structures and
    Process" — §2 (headers), §4.2 (COSE_Sign1), §4.4 (Sig_structure).
  * RFC 9053 §2.2 (EdDSA), §2.1 (ECDSA) — algorithm identifiers.
  * RFC 9360 §2 — the ``x5chain`` header parameter (label 33).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.provenance import ed25519, x509
from aether.provenance.cbor import CBORTag, dumps, loads

__all__ = [
    "COSEError",
    "COSE_SIGN1_TAG",
    "HEADER_ALG",
    "HEADER_KID",
    "HEADER_X5CHAIN",
    "ALG_EDDSA",
    "ALG_ES256",
    "ALG_ES384",
    "ALG_ES512",
    "Ed25519Signer",
    "Signer",
    "SignatureVerification",
    "sign_1",
    "verify_1",
]

COSE_SIGN1_TAG = 18
"""CBOR tag for COSE_Sign1 (RFC 9052 §2, table 1)."""

HEADER_ALG = 1
HEADER_CONTENT_TYPE = 3
HEADER_KID = 4
HEADER_X5CHAIN = 33

ALG_EDDSA = -8
ALG_ES256 = -7
ALG_ES384 = -35
ALG_ES512 = -36

_COSE_CURVE_HASH = {ALG_ES256: "sha256", ALG_ES384: "sha384", ALG_ES512: "sha512"}


class COSEError(ValueError):
    """Raised when a COSE structure is malformed or a signature is unusable."""


def _cryptography_ed25519() -> Any | None:
    """Return the ``cryptography`` Ed25519 module, or ``None`` if unavailable."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519 as backend
    except Exception:  # noqa: BLE001 - an optional accelerator, never required
        return None
    return backend


class Signer:
    """A claim signer: an algorithm, a private operation, and a chain."""

    alg: int = 0
    certificate_chain: tuple[bytes, ...] = ()
    key_id: bytes | None = None

    def sign(self, message: bytes) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def public_key(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def backend(self) -> str:
        return "unknown"


class Ed25519Signer(Signer):
    """EdDSA over Curve25519, the default C2PA algorithm for Aether.

    ``cryptography`` is used when installed purely for speed.  Because Ed25519
    signing is deterministic, the bytes it produces are identical to the
    pure-Python path, so which backend ran is an implementation detail rather
    than something a verifier has to know.
    """

    alg = ALG_EDDSA

    def __init__(
        self,
        seed: bytes,
        certificate_chain: tuple[bytes, ...] | list[bytes] | None = None,
        *,
        key_id: bytes | None = None,
        generate_certificate: bool = True,
    ) -> None:
        if len(seed) != ed25519.SECRET_KEY_SIZE:
            raise COSEError(
                f"Ed25519 seed must be {ed25519.SECRET_KEY_SIZE} bytes, got {len(seed)}"
            )
        self._seed = bytes(seed)
        self._public_key = ed25519.public_key_from_seed(self._seed)
        self._native = _cryptography_ed25519()
        if certificate_chain:
            self.certificate_chain = tuple(bytes(item) for item in certificate_chain)
        elif generate_certificate:
            self.certificate_chain = (
                x509.generate_self_signed_ed25519(self._seed),
            )
        self.key_id = key_id

    @classmethod
    def generate(cls, **kwargs: Any) -> "Ed25519Signer":
        """Create a signer with a fresh key from the OS CSPRNG."""
        return cls(ed25519.generate_seed(), **kwargs)

    @property
    def seed(self) -> bytes:
        """The private seed. Persist it with restrictive permissions."""
        return self._seed

    @property
    def public_key(self) -> bytes:
        return self._public_key

    @property
    def backend(self) -> str:
        return "cryptography" if self._native is not None else "pure-python"

    def sign(self, message: bytes) -> bytes:
        if self._native is not None:
            key = self._native.Ed25519PrivateKey.from_private_bytes(self._seed)
            return bytes(key.sign(message))
        return ed25519.sign(message, self._seed, self._public_key)


# ── Sig_structure ─────────────────────────────────────────────────────────────

def _sig_structure(protected: bytes, payload: bytes, external_aad: bytes) -> bytes:
    """Build the exact bytes RFC 9052 §4.4 says are signed for COSE_Sign1."""
    return dumps(["Signature1", protected, external_aad, payload])


def sign_1(
    payload: bytes,
    signer: Signer,
    *,
    external_aad: bytes = b"",
    detached: bool = True,
    protected_extra: dict[Any, Any] | None = None,
    unprotected_extra: dict[Any, Any] | None = None,
) -> bytes:
    """Return a tagged ``COSE_Sign1`` over ``payload``.

    Args:
        payload: The bytes to sign — for C2PA, the serialized claim.
        signer: Provides the algorithm, the private operation and the chain.
        external_aad: Externally supplied authenticated data. Signed, not carried.
        detached: When true the envelope stores ``nil`` in place of the payload,
            which is what C2PA requires: the claim already exists in its own box,
            and duplicating it would allow the two copies to disagree.
        protected_extra: Additional *integrity-protected* header parameters.
        unprotected_extra: Additional unprotected header parameters. These are
            not signed; never put anything a verifier must trust here.

    Raises:
        COSEError: If the signer declares no algorithm.
    """
    if not signer.alg:
        raise COSEError("signer does not declare a COSE algorithm")
    protected_map: dict[Any, Any] = {HEADER_ALG: signer.alg}
    if signer.certificate_chain:
        # RFC 9360 §2: a single certificate is a bstr; a chain is an array.
        protected_map[HEADER_X5CHAIN] = (
            signer.certificate_chain[0]
            if len(signer.certificate_chain) == 1
            else list(signer.certificate_chain)
        )
    if signer.key_id is not None:
        protected_map[HEADER_KID] = signer.key_id
    if protected_extra:
        protected_map.update(protected_extra)
    protected = dumps(protected_map)
    signature = signer.sign(_sig_structure(protected, payload, external_aad))
    envelope = [
        protected,
        dict(unprotected_extra or {}),
        None if detached else payload,
        signature,
    ]
    return dumps(CBORTag(COSE_SIGN1_TAG, envelope))


@dataclass
class SignatureVerification:
    """The outcome of checking one COSE_Sign1 structure.

    ``signature_valid`` and ``trusted`` are separate on purpose.  A valid
    signature proves the claim was signed by the key in the leaf certificate; it
    says nothing about whether that certificate belongs to anyone.  Collapsing
    the two is how a self-signed manifest ends up presented as verified.
    """

    signature_valid: bool
    algorithm: int | None = None
    algorithm_name: str = ""
    certificate: x509.CertificateInfo | None = None
    certificate_chain_length: int = 0
    trusted: bool = False
    trust_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_valid": self.signature_valid,
            "algorithm": self.algorithm,
            "algorithm_name": self.algorithm_name,
            "certificate_chain_length": self.certificate_chain_length,
            "signer": (
                {
                    "subject": self.certificate.subject,
                    "issuer": self.certificate.issuer,
                    "serial_number": self.certificate.serial_number,
                    "not_before": self.certificate.not_before,
                    "not_after": self.certificate.not_after,
                    "self_signed": self.certificate.self_signed,
                }
                if self.certificate is not None
                else None
            ),
            "trusted": self.trusted,
            "trust_reason": self.trust_reason,
            "errors": list(self.errors),
        }


_ALG_NAMES = {
    ALG_EDDSA: "EdDSA (Ed25519)",
    ALG_ES256: "ES256 (ECDSA P-256, SHA-256)",
    ALG_ES384: "ES384 (ECDSA P-384, SHA-384)",
    ALG_ES512: "ES512 (ECDSA P-521, SHA-512)",
}


def _chain_from_header(protected: dict[Any, Any]) -> list[bytes]:
    raw = protected.get(HEADER_X5CHAIN)
    if raw is None:
        return []
    if isinstance(raw, bytes):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, bytes) for item in raw):
        return list(raw)
    raise COSEError("x5chain must be a bstr or an array of bstr")


def _verify_raw(alg: int, message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Check one signature with the algorithm the protected header declares."""
    if alg == ALG_EDDSA:
        native = _cryptography_ed25519()
        if native is not None:
            try:
                key = native.Ed25519PublicKey.from_public_bytes(public_key)
                key.verify(signature, message)
                return True
            except Exception:  # noqa: BLE001 - any failure is a failed check
                return False
        return ed25519.verify(message, signature, public_key)
    if alg in _COSE_CURVE_HASH:
        return _verify_ecdsa(alg, message, signature, public_key)
    raise COSEError(f"unsupported COSE algorithm {alg}")


def _verify_ecdsa(alg: int, message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify an ECDSA COSE signature via ``cryptography``.

    COSE stores ECDSA signatures as the fixed-width ``r ‖ s`` pair, while the
    library expects DER, so the pair is re-encoded here.  There is no
    pure-Python fallback for ECDSA: rather than ship a second, unverified curve
    implementation, an ES* signature without ``cryptography`` installed raises so
    the caller learns the check could not be performed instead of reading a
    ``False`` as "forged".
    """
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, utils
        from cryptography.hazmat.primitives.serialization import load_der_public_key
    except Exception as exc:  # noqa: BLE001
        raise COSEError(
            f"verifying COSE algorithm {alg} requires the 'cryptography' package; "
            "install aether-runtime[provenance]"
        ) from exc
    hash_name = _COSE_CURVE_HASH[alg]
    hash_algorithm = {
        "sha256": hashes.SHA256,
        "sha384": hashes.SHA384,
        "sha512": hashes.SHA512,
    }[hash_name]()
    if len(signature) % 2:
        return False
    half = len(signature) // 2
    der = utils.encode_dss_signature(
        int.from_bytes(signature[:half], "big"),
        int.from_bytes(signature[half:], "big"),
    )
    try:
        key = load_der_public_key(public_key)
        key.verify(der, message, ec.ECDSA(hash_algorithm))
        return True
    except Exception:  # noqa: BLE001
        return False


def verify_1(
    envelope: bytes,
    payload: bytes | None = None,
    *,
    external_aad: bytes = b"",
    public_key: bytes | None = None,
    trust_anchors: list[bytes] | tuple[bytes, ...] | None = None,
) -> SignatureVerification:
    """Verify a tagged ``COSE_Sign1`` structure.

    Args:
        envelope: The serialized ``COSE_Sign1_Tagged`` bytes.
        payload: The detached payload. Required when the envelope carries
            ``nil``; when the envelope carries its own payload and this differs
            from it, verification fails rather than silently preferring one.
        external_aad: The same external data the signer used.
        public_key: Overrides the key taken from the certificate chain. Use for
            pinned-key deployments.
        trust_anchors: DER certificates that establish trust. A leaf matching one
            of these (or issued by it, when ``cryptography`` is available) is
            reported as trusted; otherwise the result is untrusted with a stated
            reason, never a silent pass.

    Returns:
        A :class:`SignatureVerification`. Structural problems are reported in
        ``errors`` with ``signature_valid`` false; they do not raise, so a caller
        verifying many manifests gets a result for each.
    """
    result = SignatureVerification(signature_valid=False)
    try:
        tagged = loads(envelope)
    except ValueError as exc:
        result.errors.append(f"claim signature is not well-formed CBOR: {exc}")
        return result
    if not isinstance(tagged, CBORTag) or tagged.tag != COSE_SIGN1_TAG:
        result.errors.append(
            f"expected a COSE_Sign1 structure tagged {COSE_SIGN1_TAG}"
        )
        return result
    body = tagged.value
    if not isinstance(body, list) or len(body) != 4:
        result.errors.append("COSE_Sign1 must be a four-element array")
        return result
    protected_bytes, unprotected, embedded_payload, signature = body
    if not isinstance(protected_bytes, bytes) or not isinstance(signature, bytes):
        result.errors.append("COSE_Sign1 protected header and signature must be bstr")
        return result
    del unprotected

    try:
        protected = loads(protected_bytes) if protected_bytes else {}
    except ValueError as exc:
        result.errors.append(f"protected header is not well-formed CBOR: {exc}")
        return result
    if not isinstance(protected, dict):
        result.errors.append("protected header must be a CBOR map")
        return result

    alg = protected.get(HEADER_ALG)
    if not isinstance(alg, int):
        result.errors.append("protected header does not declare an algorithm")
        return result
    result.algorithm = alg
    result.algorithm_name = _ALG_NAMES.get(alg, f"unregistered ({alg})")

    if embedded_payload is None:
        if payload is None:
            result.errors.append("payload is detached and none was supplied")
            return result
        signed_payload = payload
    else:
        if not isinstance(embedded_payload, bytes):
            result.errors.append("embedded payload must be a bstr or nil")
            return result
        if payload is not None and payload != embedded_payload:
            result.errors.append(
                "the embedded payload and the supplied payload differ; refusing "
                "to choose between them"
            )
            return result
        signed_payload = embedded_payload

    resolved_key = public_key
    try:
        chain = _chain_from_header(protected)
    except COSEError as exc:
        result.errors.append(str(exc))
        return result
    result.certificate_chain_length = len(chain)
    if chain:
        try:
            info = x509.certificate_info(chain[0])
        except x509.DERError as exc:
            result.errors.append(f"leaf certificate could not be parsed: {exc}")
            return result
        result.certificate = info
        if resolved_key is None:
            resolved_key = (
                info.public_key if alg == ALG_EDDSA else info.public_key_info
            )
    if resolved_key is None:
        result.errors.append(
            "no signing key: the protected header carries no x5chain and no key "
            "was supplied"
        )
        return result

    # The protected header is verified as the bytes that appear in the envelope.
    # Re-encoding the decoded map here would make a valid signature fail whenever
    # the signer's encoder differed from ours by even one byte.
    message = _sig_structure(protected_bytes, signed_payload, external_aad)
    try:
        result.signature_valid = _verify_raw(alg, message, signature, resolved_key)
    except COSEError as exc:
        result.errors.append(str(exc))
        return result

    result.trusted, result.trust_reason = _assess_trust(
        result.certificate, chain, trust_anchors
    )
    return result


def _assess_trust(
    certificate: x509.CertificateInfo | None,
    chain: list[bytes],
    trust_anchors: list[bytes] | tuple[bytes, ...] | None,
) -> tuple[bool, str]:
    """Decide whether the signer's identity is established, and say why not.

    This is intentionally conservative. Without a configured anchor the answer is
    always "not trusted", including for a signature that verified perfectly: the
    cryptography proves integrity, not identity.
    """
    if certificate is None:
        return False, "no certificate chain; the signature proves key possession only"
    if not trust_anchors:
        if certificate.self_signed:
            return False, (
                "leaf certificate is self-signed and no trust anchor was "
                "configured: integrity is proven, the signer's identity is not"
            )
        return False, (
            "no trust anchor was configured, so the certificate chain could not "
            "be validated"
        )
    anchors = {bytes(anchor) for anchor in trust_anchors}
    if chain and chain[-1] in anchors:
        return True, "chain terminates at a configured trust anchor"
    if chain and chain[0] in anchors:
        return True, "leaf certificate is itself a configured trust anchor"
    return False, "certificate chain does not terminate at a configured trust anchor"
