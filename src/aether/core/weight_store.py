"""
Weight persistence for AEG packages.

A compiled ``.aeg`` package records the graph, precision map and plans, but to
actually run inference it must also carry the weights. This module stores
quantized tensors in a single ``.aeg-quant`` blob plus a JSON index, so the
package stays self-contained and portable across machines.

Layout
------
``weights/quantized/model.aeg-quant``
    Concatenated little-endian payloads: for each tensor, the packed codes
    followed by its block scales and (for affine formats) zero points.
``weights/quantized/weight_index.json``
    Per-tensor precision, shape, block size, bit width, and byte offsets into the
    blob, which is what makes the blob readable without loading it all.

Storing offsets rather than one file per tensor matters at model scale: a 7B model
has thousands of tensors, and thousands of small files is markedly slower to read
and much less friendly to page cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import AEGFormatError
from aether.quantization.formats import QuantizedTensor
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["WeightEntry", "WeightStore", "WEIGHT_BLOB_FILENAME", "WEIGHT_INDEX_FILENAME"]

WEIGHT_BLOB_FILENAME = "model.aeg-quant"
WEIGHT_INDEX_FILENAME = "weight_index.json"

#: Index format version, bumped when the on-disk layout changes.
WEIGHT_INDEX_VERSION = "aeg-weights/1.0"


@dataclass
class WeightEntry:
    """Index record locating one tensor inside the weight blob."""

    name: str
    precision: str
    shape: tuple[int, ...]
    bits: int
    block_size: int
    packed: bool
    num_elements: int
    codes_offset: int
    codes_bytes: int
    codes_dtype: str
    #: Shape of the stored payload itself, which differs from ``shape`` for
    #: bit-packed formats (flat bytes) and dense float formats (2-D storage).
    codes_shape: tuple[int, ...] = ()
    scales_offset: int = 0
    scales_bytes: int = 0
    scales_dtype: str = ""
    zero_points_offset: int = 0
    zero_points_bytes: int = 0
    zero_points_dtype: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "precision": self.precision,
            "shape": list(self.shape),
            "bits": self.bits,
            "block_size": self.block_size,
            "packed": self.packed,
            "num_elements": self.num_elements,
            "codes": {
                "offset": self.codes_offset,
                "bytes": self.codes_bytes,
                "dtype": self.codes_dtype,
                "shape": list(self.codes_shape),
            },
            "scales": {
                "offset": self.scales_offset,
                "bytes": self.scales_bytes,
                "dtype": self.scales_dtype,
            },
            "zero_points": {
                "offset": self.zero_points_offset,
                "bytes": self.zero_points_bytes,
                "dtype": self.zero_points_dtype,
            },
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WeightEntry:
        codes = data.get("codes", {})
        scales = data.get("scales", {})
        zeros = data.get("zero_points", {})
        return WeightEntry(
            name=data["name"],
            precision=data["precision"],
            shape=tuple(data["shape"]),
            bits=int(data["bits"]),
            block_size=int(data["block_size"]),
            packed=bool(data["packed"]),
            num_elements=int(data["num_elements"]),
            codes_offset=int(codes.get("offset", 0)),
            codes_bytes=int(codes.get("bytes", 0)),
            codes_dtype=codes.get("dtype", "uint8"),
            codes_shape=tuple(codes.get("shape", ())),
            scales_offset=int(scales.get("offset", 0)),
            scales_bytes=int(scales.get("bytes", 0)),
            scales_dtype=scales.get("dtype", ""),
            zero_points_offset=int(zeros.get("offset", 0)),
            zero_points_bytes=int(zeros.get("bytes", 0)),
            zero_points_dtype=zeros.get("dtype", ""),
        )


class WeightStore:
    """Reads and writes the quantized weight blob of an AEG package.

    Args:
        directory: The ``weights/quantized`` directory inside a package.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.entries: dict[str, WeightEntry] = {}
        #: Lazily opened read-only mapping of the weight blob, plus the file object
        #: keeping it valid.  Both stay ``None`` until a zero-copy view is asked for.
        self._mmap: Any = None
        self._mmap_file: Any = None

    @property
    def blob_path(self) -> Path:
        return self.directory / WEIGHT_BLOB_FILENAME

    @property
    def index_path(self) -> Path:
        return self.directory / WEIGHT_INDEX_FILENAME

    @property
    def exists(self) -> bool:
        """True when a persisted weight blob and index are both present."""
        return self.blob_path.exists() and self.index_path.exists()

    @property
    def total_bytes(self) -> int:
        """Size of the weight blob on disk."""
        return self.blob_path.stat().st_size if self.blob_path.exists() else 0

    # ── Write ────────────────────────────────────────────────────────────────

    def save(self, tensors: dict[str, QuantizedTensor]) -> int:
        """Write quantized tensors to the blob and index.

        Args:
            tensors: Quantized tensors keyed by weight name.

        Returns:
            Total bytes written.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        offset = 0

        with self.blob_path.open("wb") as blob:
            for name, tensor in tensors.items():
                codes = np.ascontiguousarray(tensor.data)
                entry = WeightEntry(
                    name=name,
                    precision=tensor.precision,
                    shape=tuple(tensor.shape),
                    bits=tensor.bits,
                    block_size=tensor.block_size,
                    packed=tensor.packed,
                    num_elements=tensor.num_elements,
                    codes_offset=offset,
                    codes_bytes=codes.nbytes,
                    codes_dtype=str(codes.dtype),
                    codes_shape=tuple(codes.shape),
                )
                blob.write(codes.tobytes())
                offset += codes.nbytes

                if tensor.scales is not None:
                    scales = np.ascontiguousarray(tensor.scales)
                    entry.scales_offset = offset
                    entry.scales_bytes = scales.nbytes
                    entry.scales_dtype = str(scales.dtype)
                    blob.write(scales.tobytes())
                    offset += scales.nbytes

                if tensor.zero_points is not None:
                    zeros = np.ascontiguousarray(tensor.zero_points)
                    entry.zero_points_offset = offset
                    entry.zero_points_bytes = zeros.nbytes
                    entry.zero_points_dtype = str(zeros.dtype)
                    blob.write(zeros.tobytes())
                    offset += zeros.nbytes

                self.entries[name] = entry

        self.index_path.write_text(
            json.dumps(
                {
                    "version": WEIGHT_INDEX_VERSION,
                    "tensor_count": len(self.entries),
                    "total_bytes": offset,
                    "tensors": [entry.to_dict() for entry in self.entries.values()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %d weight tensors (%d bytes) to %s", len(tensors), offset, self.blob_path.name)
        return offset

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_index(self) -> dict[str, WeightEntry]:
        """Read the weight index without touching the blob.

        Returns:
            Entries keyed by weight name; empty when no weights are stored.

        Raises:
            AEGFormatError: If the index is present but unreadable.
        """
        if not self.index_path.exists():
            self.entries = {}
            return self.entries
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.entries = {
                entry["name"]: WeightEntry.from_dict(entry) for entry in data.get("tensors", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            msg = f"Malformed weight index at {self.index_path}: {exc}"
            raise AEGFormatError(msg) from exc
        return self.entries

    def load_tensor(self, name: str) -> QuantizedTensor:
        """Read one tensor from the blob.

        Args:
            name: Weight name as stored in the index.

        Returns:
            The reconstructed :class:`QuantizedTensor`.

        Raises:
            KeyError: If the name is not in the index.
            AEGFormatError: If the blob is missing or truncated.
        """
        if not self.entries:
            self.load_index()
        entry = self.entries.get(name)
        if entry is None:
            msg = f"weight '{name}' is not in the index ({len(self.entries)} tensors available)"
            raise KeyError(msg)
        if not self.blob_path.exists():
            msg = f"weight blob missing: {self.blob_path}"
            raise AEGFormatError(msg)

        with self.blob_path.open("rb") as blob:
            codes = self._read_array(blob, entry.codes_offset, entry.codes_bytes, entry.codes_dtype)
            if entry.codes_shape:
                codes = codes.reshape(entry.codes_shape)
            scales = (
                self._read_array(blob, entry.scales_offset, entry.scales_bytes, entry.scales_dtype)
                if entry.scales_bytes
                else None
            )
            zeros = (
                self._read_array(
                    blob, entry.zero_points_offset, entry.zero_points_bytes, entry.zero_points_dtype
                )
                if entry.zero_points_bytes
                else None
            )

        return QuantizedTensor(
            precision=entry.precision,
            shape=entry.shape,
            data=codes,
            scales=scales,
            zero_points=zeros,
            block_size=entry.block_size,
            bits=entry.bits,
            packed=entry.packed,
            num_elements=entry.num_elements,
        )

    def _read_array(self, blob: Any, offset: int, size: int, dtype: str) -> np.ndarray:
        """Read ``size`` bytes at ``offset`` into one freshly allocated array.

        Filling a destination array with ``readinto`` costs *one* host copy. The
        obvious spelling — ``np.frombuffer(blob.read(size)).copy()`` — costs two,
        because the ``bytes`` object is a full materialization of the tensor that
        exists only to be copied out of. At load time every tensor pays that, so the
        transient host high-water mark carries an extra copy of the largest tensor in
        the model for no benefit. On a 151936x1024 embedding that is 297 MiB of pure
        overhead per read.
        """
        dt = np.dtype(dtype)
        if size % dt.itemsize:
            msg = (
                f"weight blob entry is not a whole number of {dt} items: "
                f"{size} bytes at offset {offset}"
            )
            raise AEGFormatError(msg)
        out = np.empty(size // dt.itemsize, dtype=dt)
        blob.seek(offset)
        read = blob.readinto(memoryview(out).cast("B")) if hasattr(blob, "readinto") else None
        if read is None:  # a file object without readinto (BytesIO always has one)
            raw = blob.read(size)
            if len(raw) != size:
                msg = f"weight blob truncated: expected {size} bytes at offset {offset}, got {len(raw)}"
                raise AEGFormatError(msg)
            return np.frombuffer(raw, dtype=dt).copy()
        if read != size:
            msg = f"weight blob truncated: expected {size} bytes at offset {offset}, got {read}"
            raise AEGFormatError(msg)
        return out

    # ── Zero-copy blob access ────────────────────────────────────────────────

    def _blob_mmap(self) -> Any:
        """Map the weight blob read-only, once, and keep the mapping alive.

        A mapping lets a caller compare or reinterpret stored bytes without
        allocating anything: the pages are the OS page cache, shared rather than
        copied. It is opened lazily so a store that is only indexed never maps.
        """
        import mmap

        existing = getattr(self, "_mmap", None)
        if existing is not None:
            return existing
        if not self.blob_path.exists():
            msg = f"weight blob missing: {self.blob_path}"
            raise AEGFormatError(msg)
        handle = self.blob_path.open("rb")
        try:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):  # empty file, or a platform that refuses
            handle.close()
            raise
        self._mmap_file = handle
        self._mmap = mapped
        return mapped

    def raw_view(self, name: str, *, part: str = "codes") -> np.ndarray:
        """Return a read-only, zero-copy view of one stored payload.

        Args:
            name: Weight name as stored in the index.
            part: ``"codes"``, ``"scales"`` or ``"zero_points"``.

        Raises:
            KeyError: If the name is not in the index.
            AEGFormatError: If the blob is missing or the region is truncated.
        """
        if not self.entries:
            self.load_index()
        entry = self.entries.get(name)
        if entry is None:
            msg = f"weight '{name}' is not in the index ({len(self.entries)} tensors available)"
            raise KeyError(msg)
        offset, size, dtype = {
            "codes": (entry.codes_offset, entry.codes_bytes, entry.codes_dtype),
            "scales": (entry.scales_offset, entry.scales_bytes, entry.scales_dtype),
            "zero_points": (
                entry.zero_points_offset, entry.zero_points_bytes, entry.zero_points_dtype,
            ),
        }[part]
        if not size:
            return np.empty(0, dtype=np.uint8)
        mapped = self._blob_mmap()
        if offset + size > len(mapped):
            msg = (
                f"weight blob truncated: expected {size} bytes at offset {offset}, "
                f"blob is {len(mapped)} bytes"
            )
            raise AEGFormatError(msg)
        dt = np.dtype(dtype)
        view = np.frombuffer(mapped, dtype=dt, count=size // dt.itemsize, offset=offset)
        view.flags.writeable = False
        return view

    def payloads_identical(self, first: str, second: str) -> bool:
        """Whether two stored tensors hold byte-identical payloads.

        This is what lets the loader prove that a tied ``lm_head`` really is the
        embedding matrix before aliasing the two, instead of trusting a manifest flag
        about a tensor it has not looked at.

        Compared in bounded chunks rather than through :meth:`raw_view`. A mapped
        comparison allocates nothing, but touching a mapped page charges it to the
        process's resident set, so proving two 300 MiB matrices equal would add
        600 MiB of resident file pages -- and leave the mapping open behind it. A
        fixed pair of buffers keeps the proof O(1) in memory instead, which matters
        because the whole point of the check is to *save* memory.
        """
        if not self.entries:
            self.load_index()
        left, right = self.entries.get(first), self.entries.get(second)
        if left is None or right is None:
            return False
        if (left.shape, left.precision, left.bits, left.block_size, left.packed) != (
            right.shape, right.precision, right.bits, right.block_size, right.packed
        ):
            return False
        regions = [
            ((left.codes_offset, left.codes_bytes), (right.codes_offset, right.codes_bytes)),
            ((left.scales_offset, left.scales_bytes), (right.scales_offset, right.scales_bytes)),
            (
                (left.zero_points_offset, left.zero_points_bytes),
                (right.zero_points_offset, right.zero_points_bytes),
            ),
        ]
        if any(a[1] != b[1] for a, b in regions):
            return False
        try:
            with self.blob_path.open("rb") as blob:
                return all(
                    self._regions_identical(blob, a[0], b[0], a[1]) for a, b in regions
                )
        except OSError:
            return False

    #: Comparison buffer size.  The comparison's footprint is this constant rather
    #: than the tensor size, so the only thing the value trades is syscall count
    #: against a footprint that is already negligible beside any tensor worth
    #: comparing; a few MiB puts both sides of that trade in the noise.
    _COMPARE_CHUNK_BYTES = 8 * 1024 * 1024

    def _regions_identical(self, blob: Any, first: int, second: int, size: int) -> bool:
        """Whether two equal-length regions of one open blob hold the same bytes."""
        if size <= 0:
            return True
        chunk = min(size, self._COMPARE_CHUNK_BYTES)
        left, right = bytearray(chunk), bytearray(chunk)
        left_view, right_view = memoryview(left), memoryview(right)
        done = 0
        while done < size:
            width = min(chunk, size - done)
            blob.seek(first + done)
            if blob.readinto(left_view[:width]) != width:
                return False
            blob.seek(second + done)
            if blob.readinto(right_view[:width]) != width:
                return False
            if left_view[:width] != right_view[:width]:
                return False
            done += width
        return True

    def close(self) -> None:
        """Release the blob mapping, if one was opened.

        A mapping with live ``numpy`` views cannot be unmapped — those views point
        into it — so the attempt is allowed to fail and the mapping is simply
        dropped, leaving the OS to reclaim it when the last view dies. Refusing to
        close is correct here: closing it out from under a view would hand the caller
        a dangling pointer.
        """
        mapped = getattr(self, "_mmap", None)
        if mapped is not None:
            try:
                mapped.close()
            except BufferError:
                return  # views are still live; keep the mapping valid for them
            finally:
                if getattr(self, "_mmap", None) is mapped and mapped.closed:
                    self._mmap = None
        handle = getattr(self, "_mmap_file", None)
        if handle is not None and getattr(self, "_mmap", None) is None:
            try:
                handle.close()
            finally:
                self._mmap_file = None

    def load_all(self) -> dict[str, QuantizedTensor]:
        """Read every tensor in the index."""
        self.load_index()
        return {name: self.load_tensor(name) for name in self.entries}

    def dequantize_all(self) -> dict[str, np.ndarray]:
        """Read and dequantize every tensor to float32."""
        from aether.quantization.formats import dequantize_tensor

        return {name: dequantize_tensor(tensor) for name, tensor in self.load_all().items()}

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        if not self.entries:
            self.load_index()
        return name in self.entries

    def __repr__(self) -> str:
        return f"WeightStore({self.directory}, tensors={len(self.entries)}, bytes={self.total_bytes})"
