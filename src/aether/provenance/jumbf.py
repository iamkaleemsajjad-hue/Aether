"""JUMBF box serialization (ISO/IEC 19566-5) for the C2PA manifest store.

A C2PA manifest store is not JSON — it is a tree of JUMBF superboxes.  Getting
that binary layout right is what makes an Aether-signed artifact readable by
other C2PA tooling, and it is also load-bearing for the signature: a hashed URI
inside a claim covers a box's *description box plus its content boxes, excluding
the superbox header* (C2PA §11.1), so a writer that lays the boxes out
differently produces hashes a validator cannot reproduce.

Box layout::

    superbox:  LBox(4) TBox(4)='jumb' <description box> <content box>...
    jumd box:  LBox(4) TBox(4)='jumd' UUID(16) toggles(1) [label\\0] [id(4)] [sig(32)]
    cbor box:  LBox(4) TBox(4)='cbor' <payload>

All lengths are big-endian and include the LBox and TBox themselves.  The
extended 8-byte XLBox form (signalled by ``LBox == 1``) is read but never
written: an AEG manifest store is orders of magnitude below 4 GiB, and emitting
only one length form keeps the bytes deterministic.

References:
  * ISO/IEC 19566-5, "JPEG universal metadata box format (JUMBF)".
  * C2PA Technical Specification §8.4 (manifest boxes) and §11.1 (use of JUMBF,
    box hashing, and ``self#jumbf`` URI resolution).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

__all__ = [
    "JUMBFError",
    "CONTENT_TYPE_CBOR",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_C2PA_STORE",
    "CONTENT_TYPE_C2PA_MANIFEST",
    "CONTENT_TYPE_C2PA_UPDATE_MANIFEST",
    "CONTENT_TYPE_C2PA_ASSERTION_STORE",
    "CONTENT_TYPE_C2PA_CLAIM",
    "CONTENT_TYPE_C2PA_SIGNATURE",
    "JUMBFBox",
    "cbor_box",
    "content_box",
    "json_box",
    "content_payload",
    "parse_superbox",
]


class JUMBFError(ValueError):
    """Raised when a JUMBF structure is malformed or unsupported."""


def _content_type(four_cc: str) -> bytes:
    """Build a JUMBF content-type UUID from its 4CC.

    ISO/IEC 19566-5 derives these by placing the ASCII 4CC in the first four
    bytes of the fixed suffix ``0011-0010-8000-00AA00389B71``, which is why every
    C2PA box UUID below is mechanically the same shape.
    """
    if len(four_cc) != 4:
        raise JUMBFError(f"a JUMBF 4CC must be four characters, got {four_cc!r}")
    return four_cc.encode("ascii") + bytes.fromhex("00110010800000AA00389B71")


CONTENT_TYPE_CBOR = _content_type("cbor")
CONTENT_TYPE_JSON = _content_type("json")
CONTENT_TYPE_C2PA_STORE = _content_type("c2pa")
CONTENT_TYPE_C2PA_MANIFEST = _content_type("c2ma")
CONTENT_TYPE_C2PA_UPDATE_MANIFEST = _content_type("c2um")
CONTENT_TYPE_C2PA_ASSERTION_STORE = _content_type("c2as")
CONTENT_TYPE_C2PA_CLAIM = _content_type("c2cl")
CONTENT_TYPE_C2PA_SIGNATURE = _content_type("c2cs")

TOGGLE_REQUESTABLE = 0x01
TOGGLE_LABEL = 0x02
TOGGLE_ID = 0x04
TOGGLE_SIGNATURE = 0x08
TOGGLE_PRIVATE = 0x10


def _box(four_cc: str, payload: bytes) -> bytes:
    """Serialize one non-super JUMBF box with a 4-byte LBox."""
    total = len(payload) + 8
    if total > 0xFFFF_FFFF:
        raise JUMBFError("box exceeds the 4-byte LBox limit")
    return struct.pack(">I", total) + four_cc.encode("ascii") + payload


@dataclass
class JUMBFBox:
    """A JUMBF superbox: one description box and any number of content boxes.

    ``label`` is what ``self#jumbf`` URIs resolve against, so it is required for
    every box a claim references.  Children are kept in insertion order because
    C2PA hashes a box as a byte range: reordering them would change every hashed
    URI that points into this subtree.
    """

    content_type: bytes
    label: str = ""
    requestable: bool = True
    payload: bytes = b""
    """Raw content-box bytes, used for leaf boxes such as ``cbor``."""

    children: list["JUMBFBox"] = field(default_factory=list)
    box_id: int | None = None

    def __post_init__(self) -> None:
        if len(self.content_type) != 16:
            raise JUMBFError("a JUMBF content type must be a 16-byte UUID")
        if self.payload and self.children:
            raise JUMBFError("a JUMBF box holds either raw content or children")

    def add(self, child: "JUMBFBox") -> "JUMBFBox":
        """Append a child box and return it, for fluent construction."""
        if self.payload:
            raise JUMBFError("cannot add children to a box that has raw content")
        self.children.append(child)
        return child

    def description_box(self) -> bytes:
        """Serialize the ``jumd`` description box."""
        toggles = 0
        if self.requestable:
            toggles |= TOGGLE_REQUESTABLE
        if self.label:
            toggles |= TOGGLE_LABEL
        if self.box_id is not None:
            toggles |= TOGGLE_ID
        body = bytearray(self.content_type)
        body.append(toggles)
        if self.label:
            body += self.label.encode("utf-8") + b"\x00"
        if self.box_id is not None:
            body += struct.pack(">I", self.box_id)
        return _box("jumd", bytes(body))

    def content(self) -> bytes:
        """Serialize this box's content boxes, without the description box."""
        if self.payload:
            return self.payload
        return b"".join(child.to_bytes() for child in self.children)

    def hashed_content(self) -> bytes:
        """The byte range a hashed URI covers for this box.

        C2PA §11.1: hashing a JUMBF box covers the description box plus all
        content boxes and *excludes* the superbox header.  That is exactly the
        superbox payload, which is what this returns.
        """
        return self.description_box() + self.content()

    def digest(self, algorithm: str = "sha256") -> bytes:
        """Hash this box the way a claim's hashed URI does."""
        try:
            hasher = hashlib.new(algorithm)
        except ValueError as exc:
            raise JUMBFError(f"unsupported hash algorithm {algorithm!r}") from exc
        hasher.update(self.hashed_content())
        return hasher.digest()

    def to_bytes(self) -> bytes:
        """Serialize the complete superbox, header included."""
        return _box("jumb", self.hashed_content())

    def find(self, label: str) -> "JUMBFBox | None":
        """Return the direct child with ``label``, or ``None``.

        Only direct children are searched, and a duplicate label returns
        ``None``: C2PA requires a validator to treat an ambiguous reference as
        unresolved rather than picking one of the candidates.
        """
        matches = [child for child in self.children if child.label == label]
        return matches[0] if len(matches) == 1 else None

    def resolve(self, path: str) -> "JUMBFBox | None":
        """Resolve a ``/``-separated label path relative to this box."""
        current: JUMBFBox | None = self
        for segment in [part for part in path.split("/") if part]:
            if current is None:
                return None
            current = current.find(segment)
        return current


def content_box(four_cc: str, payload: bytes) -> bytes:
    """Serialize a bare content box (``cbor``, ``json``, …) for use as a payload."""
    return _box(four_cc, payload)


def cbor_box(
    label: str,
    payload: bytes,
    *,
    content_type: bytes | None = None,
    requestable: bool = True,
) -> JUMBFBox:
    """Build a CBOR-content JUMBF superbox.

    The result is the shape C2PA uses for every CBOR-valued box — a claim, a
    claim signature, or a CBOR assertion: a description box carrying the label,
    followed by a single ``cbor`` content box.  ``content_type`` overrides the
    description box's UUID for the boxes C2PA gives their own type (``c2cl`` for a
    claim, ``c2cs`` for a claim signature); assertions use the plain CBOR UUID.
    The label is what ``self#jumbf`` URIs resolve against.
    """
    return JUMBFBox(
        content_type=CONTENT_TYPE_CBOR if content_type is None else content_type,
        label=label,
        requestable=requestable,
        payload=_box("cbor", payload),
    )


def json_box(label: str, payload: bytes, *, requestable: bool = True) -> JUMBFBox:
    """Build a JSON-content JUMBF superbox, for JSON-valued assertions."""
    return JUMBFBox(
        content_type=CONTENT_TYPE_JSON,
        label=label,
        requestable=requestable,
        payload=_box("json", payload),
    )


# ── Reading ───────────────────────────────────────────────────────────────────

def _read_box(data: bytes, offset: int) -> tuple[str, bytes, int]:
    """Read one box; return ``(four_cc, payload, next_offset)``."""
    if offset + 8 > len(data):
        raise JUMBFError("truncated JUMBF box header")
    length = int(struct.unpack(">I", data[offset:offset + 4])[0])
    four_cc = data[offset + 4:offset + 8].decode("ascii", "replace")
    body_start = offset + 8
    if length == 1:
        # XLBox: an 8-byte length follows the TBox.
        if body_start + 8 > len(data):
            raise JUMBFError("truncated XLBox length")
        length = int(struct.unpack(">Q", data[body_start:body_start + 8])[0])
        body_start += 8
        end = offset + length
    elif length == 0:
        end = len(data)  # "to end of file", permitted for the last box
    else:
        end = offset + length
    if end > len(data) or end < body_start:
        raise JUMBFError(f"JUMBF box {four_cc!r} declares a length past end of input")
    return four_cc, data[body_start:end], end


def _parse_description(payload: bytes) -> tuple[bytes, str, bool, int | None]:
    if len(payload) < 17:
        raise JUMBFError("description box is shorter than UUID + toggles")
    content_type = payload[:16]
    toggles = payload[16]
    cursor = 17
    label = ""
    if toggles & TOGGLE_LABEL:
        end = payload.find(b"\x00", cursor)
        if end < 0:
            raise JUMBFError("description box label is not null-terminated")
        label = payload[cursor:end].decode("utf-8", "replace")
        cursor = end + 1
    box_id: int | None = None
    if toggles & TOGGLE_ID:
        if cursor + 4 > len(payload):
            raise JUMBFError("description box declares an ID but is truncated")
        box_id = int(struct.unpack(">I", payload[cursor:cursor + 4])[0])
    return content_type, label, bool(toggles & TOGGLE_REQUESTABLE), box_id


def parse_superbox(data: bytes, offset: int = 0) -> tuple[JUMBFBox, int]:
    """Parse one ``jumb`` superbox; return the box and the next offset.

    Children are parsed recursively when they are themselves superboxes, and kept
    as raw content otherwise.  A round trip through :meth:`JUMBFBox.to_bytes`
    reproduces the input byte for byte — asserted by
    ``tests/unit/test_c2pa_signing.py``, because a lossy round trip would break
    every hashed URI in a re-serialized manifest.
    """
    four_cc, payload, end = _read_box(data, offset)
    if four_cc != "jumb":
        raise JUMBFError(f"expected a 'jumb' superbox, found {four_cc!r}")
    description_cc, description, cursor = _read_box(payload, 0)
    if description_cc != "jumd":
        raise JUMBFError("a JUMBF superbox must begin with a 'jumd' description box")
    content_type, label, requestable, box_id = _parse_description(description)
    box = JUMBFBox(
        content_type=content_type, label=label, requestable=requestable, box_id=box_id
    )
    remainder = payload[cursor:]
    if remainder[:8] and remainder[4:8] == b"jumb":
        position = 0
        while position < len(remainder):
            child, position = parse_superbox(remainder, position)
            box.children.append(child)
    else:
        box.payload = remainder
    return box, end


def content_payload(box: JUMBFBox, four_cc: str = "cbor") -> bytes:
    """Return the payload of ``box``'s single ``four_cc`` content box."""
    if not box.payload:
        raise JUMBFError(f"box {box.label!r} has no raw content box")
    found, payload, end = _read_box(box.payload, 0)
    if found != four_cc:
        raise JUMBFError(f"box {box.label!r} holds a {found!r} box, expected {four_cc!r}")
    if end != len(box.payload):
        raise JUMBFError(f"box {box.label!r} holds more than one content box")
    return payload
