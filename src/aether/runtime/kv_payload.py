"""Versioned lossless cache snapshots, independent of any inference engine.

Tensor coordinates and cache semantics belong to the exporting adapter. This
module preserves typed bytes, validates framing/identity and never unpickles
objects or silently changes precision. Device tensors are explicitly copied to
host memory before storage; this is not GPUDirect or RDMA.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Mapping

import numpy as np

MAGIC = b"AEKVC01\0"
MAX_PAYLOAD_BYTES = 2 * 1024**3
MAX_HEADER_BYTES = 4 * 1024**2
_TORCH_WIDTHS = {
    "torch.bool": 1, "torch.uint8": 1, "torch.int8": 1,
    "torch.uint16": 2, "torch.uint32": 4, "torch.uint64": 8,
    "torch.int16": 2, "torch.int32": 4, "torch.int64": 8,
    "torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4,
    "torch.float64": 8, "torch.complex64": 8, "torch.complex128": 16,
    "torch.float8_e4m3fn": 1, "torch.float8_e4m3fnuz": 1,
    "torch.float8_e5m2": 1, "torch.float8_e5m2fnuz": 1,
}


def _integer(value: Any, name: str, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a nonnegative bounded integer")
    return value


def _text(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\0" in value:
        raise ValueError(f"{name} must be nonempty bounded text")
    return value


@dataclass(frozen=True)
class TensorPayload:
    """Contiguous tensor coordinates and their exact dtype-preserving bytes."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if len(shape) > 16:
            raise ValueError("Tensor rank exceeds the cache payload limit")
        for dimension in shape:
            _integer(dimension, "tensor dimension", 2**31 - 1)
        if self.dtype.startswith("torch."):
            if self.dtype not in _TORCH_WIDTHS:
                raise ValueError(f"Unsupported tensor dtype: {self.dtype}")
            width = _TORCH_WIDTHS[self.dtype]
        else:
            try:
                dtype = np.dtype(self.dtype)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Unsupported tensor dtype: {self.dtype}") from exc
            if dtype.hasobject or dtype.fields is not None or dtype.kind not in "buifc":
                raise ValueError("Cache tensors require primitive numeric dtypes, without objects")
            width = dtype.itemsize
            object.__setattr__(self, "dtype", dtype.str)
        count = math.prod(shape)
        if count * width > MAX_PAYLOAD_BYTES:
            raise ValueError("Tensor exceeds the cache payload byte limit")
        if not isinstance(self.data, bytes) or len(self.data) != count * width:
            raise ValueError("Tensor byte count does not match dtype and shape")
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_array(cls, value: np.ndarray) -> "TensorPayload":
        array = np.asarray(value)
        return cls(array.dtype.str, tuple(array.shape), array.tobytes(order="C"))

    @classmethod
    def from_tensor(cls, value: Any) -> "TensorPayload":
        """Snapshot NumPy or an explicit Torch tensor; import Torch lazily."""
        if isinstance(value, np.ndarray):
            return cls.from_array(value)
        if type(value).__module__.startswith("torch"):
            import torch

            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise ValueError("Only dense strided Torch cache tensors are supported")
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return cls(str(tensor.dtype), tuple(tensor.shape), raw)
        return cls.from_array(np.asarray(value))

    @property
    def byte_size(self) -> int:
        return len(self.data)

    def array(self, *, copy: bool = True) -> np.ndarray:
        if self.dtype.startswith("torch."):
            equivalents = {
                "torch.bool": "?", "torch.uint8": "u1", "torch.int8": "i1",
                "torch.uint16": "<u2", "torch.uint32": "<u4", "torch.uint64": "<u8",
                "torch.int16": "<i2", "torch.int32": "<i4", "torch.int64": "<i8",
                "torch.float16": "<f2", "torch.float32": "<f4", "torch.float64": "<f8",
                "torch.complex64": "<c8", "torch.complex128": "<c16",
            }
            dtype = equivalents.get(self.dtype)
            if dtype is None:
                raise ValueError(f"{self.dtype} has no lossless NumPy representation; use to_tensor")
        else:
            dtype = self.dtype
        array = np.frombuffer(self.data, dtype=dtype).reshape(self.shape)
        return array.copy() if copy else array

    def to_tensor(self, device: Any = "cpu") -> Any:
        import torch

        if self.dtype.startswith("torch."):
            dtype = getattr(torch, self.dtype.removeprefix("torch."), None)
            if dtype is None:
                raise ValueError(f"This Torch build cannot restore {self.dtype}")
            raw = torch.from_numpy(np.frombuffer(self.data, dtype=np.uint8).copy())
            return raw.view(dtype).reshape(self.shape).to(device)
        return torch.from_numpy(self.array()).to(device)


@dataclass
class CachePayload:
    """An adapter-specific cache snapshot bound to a model and logical length.

    Adapter metadata records positions, aliases, layout and family geometry.
    Restoration must check the expected model fingerprint and cache kind before
    using these arrays. ``array`` returns owned writable storage by default.
    """

    model_fingerprint: str
    cache_kind: str
    logical_length: int
    tensors: Mapping[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.model_fingerprint, "model fingerprint")
        _text(self.cache_kind, "cache kind", 128)
        _integer(self.logical_length, "logical length")
        if not isinstance(self.tensors, Mapping) or len(self.tensors) > 4096:
            raise ValueError("Cache tensors must be a bounded named mapping")
        normalized = {}
        for name, value in self.tensors.items():
            _text(name, "tensor name")
            normalized[name] = value if isinstance(value, TensorPayload) else TensorPayload.from_tensor(value)
        if sum(value.byte_size for value in normalized.values()) > MAX_PAYLOAD_BYTES:
            raise ValueError("Cache tensors exceed the payload byte limit")
        if self.logical_length and not normalized:
            raise ValueError("Nonempty cache length requires actual tensor state")
        if not isinstance(self.metadata, dict):
            raise ValueError("Cache metadata must be a JSON object")
        try:
            metadata = json.dumps(self.metadata, allow_nan=False, ensure_ascii=False)
            if len(metadata.encode("utf-8")) > MAX_HEADER_BYTES // 2:
                raise ValueError("Cache metadata exceeds the header limit")
            self.metadata = json.loads(metadata)
        except (TypeError, OverflowError, RecursionError) as exc:
            raise ValueError("Cache metadata must be bounded JSON data") from exc
        self.tensors = normalized

    @property
    def byte_size(self) -> int:
        return sum(tensor.byte_size for tensor in self.tensors.values())

    def array(self, name: str, *, copy: bool = True) -> np.ndarray:
        return self.tensors[name].array(copy=copy)

    def numpy_tensors(self, *, copy: bool = True) -> dict[str, np.ndarray]:
        return {name: tensor.array(copy=copy) for name, tensor in self.tensors.items()}

    def _chunks(self):
        # Revalidate public dataclass fields before emitting an archive. This
        # also snapshots the mapping so concurrent caller mutation cannot
        # create descriptors for different bytes than those being written.
        snapshot = CachePayload(self.model_fingerprint, self.cache_kind, self.logical_length,
                                dict(self.tensors), dict(self.metadata))
        descriptors = []
        offset = 0
        for name in sorted(snapshot.tensors):
            tensor = snapshot.tensors[name]
            descriptors.append({"name": name, "dtype": tensor.dtype, "shape": list(tensor.shape),
                                "offset": offset, "size": tensor.byte_size,
                                "sha256": hashlib.sha256(tensor.data).hexdigest()})
            offset += tensor.byte_size
        header = json.dumps({"format": "aether_cache_payload_v1", "model_fingerprint": snapshot.model_fingerprint,
                             "cache_kind": snapshot.cache_kind, "logical_length": snapshot.logical_length,
                             "metadata": snapshot.metadata, "tensors": descriptors},
                            sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")
        if len(header) > MAX_HEADER_BYTES:
            raise ValueError("Cache header exceeds the configured bound")
        if 48 + len(header) + offset > MAX_PAYLOAD_BYTES:
            raise ValueError("Serialized cache payload exceeds the configured byte bound")
        yield MAGIC + struct.pack("<Q", len(header)) + header
        for descriptor in descriptors:
            yield snapshot.tensors[descriptor["name"]].data

    def write_to(self, stream: BinaryIO) -> int:
        digest = hashlib.sha256()
        size = 0
        for chunk in self._chunks():
            digest.update(chunk)
            stream.write(chunk)
            size += len(chunk)
        stream.write(digest.digest())
        return size + digest.digest_size

    def serialize(self) -> bytes:
        buffer = io.BytesIO()
        self.write_to(buffer)
        return buffer.getvalue()

    def content_hash(self) -> str:
        """Hash the exact framed snapshot without allocating a concatenated copy."""
        message = hashlib.sha256()
        whole = hashlib.sha256()
        for chunk in self._chunks():
            message.update(chunk)
            whole.update(chunk)
        whole.update(message.digest())
        return whole.hexdigest()

    @classmethod
    def deserialize(cls, data: bytes, *, expected_model_fingerprint: str | None = None,
                    expected_cache_kind: str | None = None, max_bytes: int = MAX_PAYLOAD_BYTES) -> "CachePayload":
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PAYLOAD_BYTES:
            raise ValueError("max_bytes must be a positive bounded integer")
        if not isinstance(data, bytes) or len(data) > max_bytes or len(data) < 48 or data[:8] != MAGIC:
            raise ValueError("Invalid or oversized cache payload framing")
        if hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise ValueError("Cache payload checksum mismatch")
        header_length = struct.unpack("<Q", data[8:16])[0]
        if header_length > MAX_HEADER_BYTES or 16 + header_length > len(data) - 32:
            raise ValueError("Invalid cache payload header length")
        try:
            header = json.loads(data[16:16 + header_length].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Invalid cache payload JSON header") from exc
        if not isinstance(header, dict) or header.get("format") != "aether_cache_payload_v1":
            raise ValueError("Unsupported cache payload schema")
        if expected_model_fingerprint is not None and header.get("model_fingerprint") != expected_model_fingerprint:
            raise ValueError("Cache model fingerprint does not match the destination model")
        if expected_cache_kind is not None and header.get("cache_kind") != expected_cache_kind:
            raise ValueError("Cache kind does not match the destination adapter")
        descriptors = header.get("tensors")
        if not isinstance(descriptors, list) or len(descriptors) > 4096:
            raise ValueError("Invalid cache tensor descriptors")
        tensors = {}
        offset = 0
        start = 16 + header_length
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise ValueError("Invalid cache tensor descriptor")
            name = _text(descriptor.get("name"), "tensor name")
            size = _integer(descriptor.get("size"), "tensor size", max_bytes)
            if name in tensors or descriptor.get("offset") != offset or start + offset + size > len(data) - 32:
                raise ValueError("Duplicate, overlapping or out-of-bounds cache tensor")
            raw = data[start + offset:start + offset + size]
            if hashlib.sha256(raw).hexdigest() != descriptor.get("sha256"):
                raise ValueError(f"Cache tensor checksum mismatch: {name}")
            try:
                tensors[name] = TensorPayload(descriptor["dtype"], tuple(descriptor["shape"]), raw)
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError("Invalid cache tensor shape or dtype") from exc
            offset += size
        if start + offset != len(data) - 32:
            raise ValueError("Cache payload contains undeclared trailing bytes")
        return cls(header.get("model_fingerprint"), header.get("cache_kind"), header.get("logical_length"), tensors, header.get("metadata", {}))

