"""Deterministic CBOR (RFC 8949) encoder and decoder.

C2PA requires that a claim be serialized with the *Core Deterministic Encoding
Requirements* of RFC 8949 §4.2.1, because the claim bytes are what gets signed.
Any freedom left in the encoding is a way for two implementations to produce
different bytes for the same claim and therefore disagree about a signature.
This module removes that freedom:

* integers and lengths use the shortest form that represents the value
  (preferred serialization, §4.2.1 item 1);
* only definite-length strings, arrays and maps are emitted (item 2);
* map keys are sorted by the bytewise lexicographic order of their own
  deterministic encodings (item 3);
* floats use the shortest of binary16 / binary32 / binary64 that round-trips
  exactly, and an integer-valued float stays a float — narrowing it to an
  integer would change the data model, not just the bytes.

It is written against the standard library alone so the base runtime does not
acquire a dependency in order to produce a signed artifact.  When ``cbor2`` is
installed it is *not* used: a second encoder would reintroduce exactly the
ambiguity this module exists to remove.

Reference: C. Bormann and P. Hoffman, "Concise Binary Object Representation
(CBOR)", RFC 8949, December 2020.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CBORDecodeError",
    "CBOREncodeError",
    "CBORTag",
    "Undefined",
    "dumps",
    "loads",
]


class CBOREncodeError(ValueError):
    """Raised when a value has no deterministic CBOR representation."""


class CBORDecodeError(ValueError):
    """Raised when a byte string is not well-formed CBOR."""


class _Undefined:
    """The CBOR ``undefined`` simple value (major type 7, value 23)."""

    _instance: "_Undefined | None" = None

    def __new__(cls) -> "_Undefined":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Undefined"

    def __bool__(self) -> bool:
        return False


Undefined = _Undefined()


@dataclass(frozen=True)
class CBORTag:
    """A CBOR tagged value (major type 6).

    COSE relies on these: ``COSE_Sign1_Tagged`` is tag 18 wrapping the
    four-element signature array.
    """

    tag: int
    value: Any

    def __post_init__(self) -> None:
        if self.tag < 0 or self.tag > 0xFFFF_FFFF_FFFF_FFFF:
            raise CBOREncodeError(f"tag {self.tag} out of range for CBOR")


# ── Encoding ──────────────────────────────────────────────────────────────────

_UINT64_MAX = 0xFFFF_FFFF_FFFF_FFFF


def _head(major: int, argument: int, out: bytearray) -> None:
    """Emit an initial byte plus the shortest argument encoding for it.

    This single function is what makes preferred serialization unavoidable: the
    argument width is chosen from the value, never from the caller.
    """
    if argument < 0 or argument > _UINT64_MAX:
        raise CBOREncodeError(f"argument {argument} out of range for CBOR head")
    prefix = major << 5
    if argument < 24:
        out.append(prefix | argument)
    elif argument <= 0xFF:
        out.append(prefix | 24)
        out.append(argument)
    elif argument <= 0xFFFF:
        out.append(prefix | 25)
        out += struct.pack(">H", argument)
    elif argument <= 0xFFFF_FFFF:
        out.append(prefix | 26)
        out += struct.pack(">I", argument)
    else:
        out.append(prefix | 27)
        out += struct.pack(">Q", argument)


def _encode_float(value: float, out: bytearray) -> None:
    """Emit the shortest float width that reproduces ``value`` exactly.

    NaN is encoded as the canonical binary16 quiet NaN (0xf97e00), as RFC 8949
    §4.2.2 requires, so two NaNs never produce different signed bytes.
    """
    if value != value:  # NaN
        out += b"\xf9\x7e\x00"
        return
    for width, code, fmt in ((2, 0xF9, ">e"), (4, 0xFA, ">f")):
        try:
            packed = struct.pack(fmt, value)
        except (OverflowError, struct.error):
            continue
        if struct.unpack(fmt, packed)[0] == value:
            out.append(code)
            out += packed
            return
        del width
    out.append(0xFB)
    out += struct.pack(">d", value)


def _encode(value: Any, out: bytearray, depth: int) -> None:
    if depth > 256:
        raise CBOREncodeError("CBOR nesting deeper than 256 levels")
    # bool must precede int: in Python, True is an int with value 1.
    if value is True:
        out.append(0xF5)
    elif value is False:
        out.append(0xF4)
    elif value is None:
        out.append(0xF6)
    elif value is Undefined:
        out.append(0xF7)
    elif isinstance(value, int):
        if value >= 0:
            _head(0, value, out)
        else:
            # Major type 1 encodes -1-n, so n = -1-value.
            _head(1, -1 - value, out)
    elif isinstance(value, float):
        _encode_float(value, out)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        _head(2, len(raw), out)
        out += raw
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        _head(3, len(raw), out)
        out += raw
    elif isinstance(value, (list, tuple)):
        _head(4, len(value), out)
        for item in value:
            _encode(item, out, depth + 1)
    elif isinstance(value, dict):
        _encode_map(value, out, depth)
    elif isinstance(value, CBORTag):
        _head(6, value.tag, out)
        _encode(value.value, out, depth + 1)
    else:
        raise CBOREncodeError(
            f"no deterministic CBOR encoding for {type(value).__name__}"
        )


def _encode_map(mapping: dict[Any, Any], out: bytearray, depth: int) -> None:
    """Emit a map with keys in RFC 8949 §4.2.1 bytewise lexicographic order.

    Sorting the *encoded* keys rather than the Python values is what makes the
    order well defined across mixed key types, which C2PA assertions do use
    (COSE headers mix negative integers and text strings).
    """
    encoded: list[tuple[bytes, Any]] = []
    for key, item in mapping.items():
        key_buffer = bytearray()
        _encode(key, key_buffer, depth + 1)
        encoded.append((bytes(key_buffer), item))
    keys = [entry[0] for entry in encoded]
    if len(set(keys)) != len(keys):
        raise CBOREncodeError("duplicate map keys have no deterministic encoding")
    encoded.sort(key=lambda entry: entry[0])
    _head(5, len(encoded), out)
    for key_bytes, item in encoded:
        out += key_bytes
        _encode(item, out, depth + 1)


def dumps(value: Any) -> bytes:
    """Serialize ``value`` using RFC 8949 core deterministic encoding."""
    out = bytearray()
    _encode(value, out, 0)
    return bytes(out)


# ── Decoding ──────────────────────────────────────────────────────────────────

class _Decoder:
    """Strict CBOR reader.

    Indefinite-length items are rejected rather than accepted, because a
    deterministic profile cannot contain them: silently accepting one would let
    a verifier validate bytes an encoder is forbidden to produce.
    """

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _take(self, count: int) -> bytes:
        end = self.pos + count
        if count < 0 or end > len(self.data):
            raise CBORDecodeError("truncated CBOR item")
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def _argument(self, info: int) -> int:
        if info < 24:
            return info
        if info == 24:
            return self._take(1)[0]
        if info == 25:
            return int(struct.unpack(">H", self._take(2))[0])
        if info == 26:
            return int(struct.unpack(">I", self._take(4))[0])
        if info == 27:
            return int(struct.unpack(">Q", self._take(8))[0])
        if info == 31:
            raise CBORDecodeError("indefinite-length items are not deterministic CBOR")
        raise CBORDecodeError(f"reserved additional-information value {info}")

    def decode(self, depth: int = 0) -> Any:
        if depth > 256:
            raise CBORDecodeError("CBOR nesting deeper than 256 levels")
        initial = self._take(1)[0]
        major, info = initial >> 5, initial & 0x1F
        if major == 0:
            return self._argument(info)
        if major == 1:
            return -1 - self._argument(info)
        if major == 2:
            return self._take(self._argument(info))
        if major == 3:
            raw = self._take(self._argument(info))
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CBORDecodeError("text string is not valid UTF-8") from exc
        if major == 4:
            return [self.decode(depth + 1) for _ in range(self._argument(info))]
        if major == 5:
            return self._decode_map(self._argument(info), depth)
        if major == 6:
            tag = self._argument(info)
            return CBORTag(tag, self.decode(depth + 1))
        return self._decode_simple(info)

    def _decode_map(self, count: int, depth: int) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for _ in range(count):
            key = self.decode(depth + 1)
            if isinstance(key, (bytes, bytearray)):
                key = bytes(key)
            elif isinstance(key, (list, dict)):
                raise CBORDecodeError("composite map keys are not supported")
            if key in result:
                raise CBORDecodeError("duplicate map key")
            result[key] = self.decode(depth + 1)
        return result

    def _decode_simple(self, info: int) -> Any:
        if info == 20:
            return False
        if info == 21:
            return True
        if info == 22:
            return None
        if info == 23:
            return Undefined
        if info == 25:
            return float(struct.unpack(">e", self._take(2))[0])
        if info == 26:
            return float(struct.unpack(">f", self._take(4))[0])
        if info == 27:
            return float(struct.unpack(">d", self._take(8))[0])
        if info == 31:
            raise CBORDecodeError("unexpected break code")
        raise CBORDecodeError(f"unsupported simple value {info}")


def loads(data: bytes, *, allow_trailing: bool = False) -> Any:
    """Decode one deterministic CBOR item from ``data``.

    Trailing bytes are an error by default.  A signature payload that decodes
    while leaving bytes unread is a classic way to smuggle content past a
    verifier, so the strict reading is the default and the permissive one has to
    be asked for.
    """
    decoder = _Decoder(bytes(data))
    value = decoder.decode()
    if not allow_trailing and decoder.pos != len(decoder.data):
        raise CBORDecodeError(
            f"{len(decoder.data) - decoder.pos} trailing byte(s) after CBOR item"
        )
    return value
