"""Ed25519 signatures (RFC 8032) in pure Python.

C2PA permits EdDSA over Curve25519 as a claim-signature algorithm, and a signed
artifact is worth nothing if signing requires an optional dependency that the
base runtime does not have.  This module therefore implements the full scheme —
key derivation, signing and verification — with exact integer arithmetic, so
``aether sign`` works on a stock CPython install.  When ``cryptography`` is
present :mod:`aether.provenance.cose` uses its C implementation instead; this
one is the reference and the fallback, and the two are cross-checked in
``tests/unit/test_c2pa_signing.py``.

The curve is the twisted Edwards curve

    -x² + y² = 1 + d·x²·y²   over  GF(q),  q = 2²⁵⁵ - 19,
    d = -121665 / 121666,

with base point B of prime order L = 2²⁵² + 27742317777372353535851937790883648493.

Signing is deterministic (there is no per-signature nonce to leak), which also
means two signatures over the same claim are byte-identical — a property the
provenance tests rely on.

Reference: S. Josefsson and I. Liusvaara, "Edwards-Curve Digital Signature
Algorithm (EdDSA)", RFC 8032, January 2017, §5.1 (Ed25519).
"""

from __future__ import annotations

import hashlib
import os

__all__ = [
    "Ed25519Error",
    "PUBLIC_KEY_SIZE",
    "SECRET_KEY_SIZE",
    "SIGNATURE_SIZE",
    "generate_seed",
    "public_key_from_seed",
    "sign",
    "verify",
]

SECRET_KEY_SIZE = 32
PUBLIC_KEY_SIZE = 32
SIGNATURE_SIZE = 64

# ── Field and curve constants (RFC 8032 §5.1) ────────────────────────────────

_Q = 2 ** 255 - 19
"""Field prime q."""

_L = 2 ** 252 + 27742317777372353535851937790883648493
"""Order of the base point."""

_D = -121665 * pow(121666, _Q - 2, _Q) % _Q
"""Curve parameter d = -121665/121666 mod q."""

_I = pow(2, (_Q - 1) // 4, _Q)
"""A square root of -1 mod q, used to recover x from y."""


class Ed25519Error(ValueError):
    """Raised when a key or signature is malformed, or verification fails."""


# ── Group arithmetic in extended homogeneous coordinates ─────────────────────
# A point is (X, Y, Z, T) with x = X/Z, y = Y/Z and x·y = T/Z.  Extended
# coordinates keep addition and doubling free of modular inversion, which is
# what makes a pure-Python scalar multiplication fast enough to be practical.


def _point_add(
    p: tuple[int, int, int, int], q: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Unified Edwards addition (RFC 8032 §5.1.4)."""
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % _Q
    b = (y1 + x1) * (y2 + x2) % _Q
    c = 2 * t1 * t2 * _D % _Q
    d = 2 * z1 * z2 % _Q
    e, f, g, h = b - a, d - c, d + c, b + a
    return e * f % _Q, g * h % _Q, f * g % _Q, e * h % _Q


def _point_double(p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Doubling specialized from the addition law (saves one multiplication)."""
    x1, y1, z1, _ = p
    a = x1 * x1 % _Q
    b = y1 * y1 % _Q
    c = 2 * z1 * z1 % _Q
    h = a + b
    e = h - (x1 + y1) * (x1 + y1) % _Q
    g = a - b
    f = c + g
    return e * f % _Q, g * h % _Q, f * g % _Q, e * h % _Q


def _recover_x(y: int, sign: int) -> int:
    """Solve the curve equation for x given y and the sign bit.

    x² = (y² - 1) / (d·y² + 1).  A y with no square root is not a curve point,
    and rejecting it here is what stops a forged public key from being accepted.
    """
    if y >= _Q:
        raise Ed25519Error("y coordinate is not a field element")
    numerator = (y * y - 1) % _Q
    denominator = (_D * y * y + 1) % _Q
    if denominator == 0:
        raise Ed25519Error("point is not on the curve")
    x2 = numerator * pow(denominator, _Q - 2, _Q) % _Q
    if x2 == 0:
        if sign:
            raise Ed25519Error("non-canonical encoding of the identity x")
        return 0
    x = pow(x2, (_Q + 3) // 8, _Q)
    if x * x % _Q != x2:
        x = x * _I % _Q
    if x * x % _Q != x2:
        raise Ed25519Error("point is not on the curve")
    if x & 1 != sign:
        x = _Q - x
    return x


_BY = 4 * pow(5, _Q - 2, _Q) % _Q
_BX = _recover_x(_BY, 0)
_B: tuple[int, int, int, int] = (_BX, _BY, 1, _BX * _BY % _Q)
"""Base point B in extended coordinates."""


def _scalar_mult(scalar: int, point: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Compute ``scalar · point`` by fixed-order double-and-add.

    The loop runs over a fixed 255-bit width and performs the same operations
    for every bit, so the operation count does not depend on the secret scalar.
    """
    result = (0, 1, 1, 0)  # neutral element
    addend = point
    for _ in range(256):
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_double(addend)
        scalar >>= 1
    return result


def _encode_point(point: tuple[int, int, int, int]) -> bytes:
    """Compress a point to 32 bytes: little-endian y with x's low bit on top."""
    x, y, z, _ = point
    inverse = pow(z, _Q - 2, _Q)
    x = x * inverse % _Q
    y = y * inverse % _Q
    return int(y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode_point(data: bytes) -> tuple[int, int, int, int]:
    if len(data) != 32:
        raise Ed25519Error(f"point must be 32 bytes, got {len(data)}")
    value = int.from_bytes(data, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return x, y, 1, x * y % _Q


def _on_curve(point: tuple[int, int, int, int]) -> bool:
    """Check -x² + y² - z² - d·t² == 0 and x·y == z·t projectively."""
    x, y, z, t = point
    return (
        (-x * x + y * y - z * z - _D * t * t) % _Q == 0
        and (x * y - z * t) % _Q == 0
    )


def _sha512(*chunks: bytes) -> bytes:
    digest = hashlib.sha512()
    for chunk in chunks:
        digest.update(chunk)
    return digest.digest()


def _clamp(digest: bytes) -> int:
    """Prune the scalar as RFC 8032 §5.1.5 requires.

    Clearing the low three bits forces a multiple of the cofactor 8; setting bit
    254 and clearing bit 255 fixes the scalar's bit length.  Both are what make
    the fixed-width ladder above safe.
    """
    scalar = bytearray(digest[:32])
    scalar[0] &= 0xF8
    scalar[31] &= 0x7F
    scalar[31] |= 0x40
    return int.from_bytes(scalar, "little")


# ── Public interface ──────────────────────────────────────────────────────────

def generate_seed() -> bytes:
    """Return a fresh 32-byte private seed from the OS CSPRNG."""
    return os.urandom(SECRET_KEY_SIZE)


def public_key_from_seed(seed: bytes) -> bytes:
    """Derive the 32-byte public key A = [s]B from a private seed."""
    if len(seed) != SECRET_KEY_SIZE:
        raise Ed25519Error(f"seed must be {SECRET_KEY_SIZE} bytes, got {len(seed)}")
    scalar = _clamp(_sha512(seed))
    return _encode_point(_scalar_mult(scalar, _B))


def sign(message: bytes, seed: bytes, public_key: bytes | None = None) -> bytes:
    """Produce the 64-byte Ed25519 signature of ``message`` under ``seed``.

    The nonce r = H(prefix ‖ message) is derived from the key and the message,
    never from a random source, so the signature is deterministic and there is
    no nonce to reuse.
    """
    if len(seed) != SECRET_KEY_SIZE:
        raise Ed25519Error(f"seed must be {SECRET_KEY_SIZE} bytes, got {len(seed)}")
    digest = _sha512(seed)
    scalar = _clamp(digest)
    prefix = digest[32:]
    encoded_public = public_key if public_key is not None else _encode_point(
        _scalar_mult(scalar, _B)
    )
    if len(encoded_public) != PUBLIC_KEY_SIZE:
        raise Ed25519Error("public key must be 32 bytes")

    r = int.from_bytes(_sha512(prefix, message), "little") % _L
    encoded_r = _encode_point(_scalar_mult(r, _B))
    k = int.from_bytes(_sha512(encoded_r, encoded_public, message), "little") % _L
    s = (r + k * scalar) % _L
    return encoded_r + s.to_bytes(32, "little")


def verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Return whether ``signature`` is a valid Ed25519 signature.

    Uses the cofactored equation [8][S]B = [8]R + [8][k]A of RFC 8032 §5.1.7,
    which is the check the specification defines; ``S ≥ L`` is rejected outright
    so a malleable re-encoding of a valid signature does not verify.
    """
    if len(signature) != SIGNATURE_SIZE:
        return False
    if len(public_key) != PUBLIC_KEY_SIZE:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    try:
        point_r = _decode_point(signature[:32])
        point_a = _decode_point(public_key)
    except Ed25519Error:
        return False
    if not (_on_curve(point_r) and _on_curve(point_a)):
        return False
    k = int.from_bytes(_sha512(signature[:32], public_key, message), "little") % _L
    left = _scalar_mult(s, _B)
    right = _point_add(point_r, _scalar_mult(k, point_a))
    # Compare projectively after multiplying both sides by the cofactor 8.
    for _ in range(3):
        left = _point_double(left)
        right = _point_double(right)
    x1, y1, z1, _ = left
    x2, y2, z2, _ = right
    return (x1 * z2 - x2 * z1) % _Q == 0 and (y1 * z2 - y2 * z1) % _Q == 0
