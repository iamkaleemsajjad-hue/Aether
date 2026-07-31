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
        """Read ``size`` bytes at ``offset`` and view them as ``dtype``."""
        blob.seek(offset)
        raw = blob.read(size)
        if len(raw) != size:
            msg = f"weight blob truncated: expected {size} bytes at offset {offset}, got {len(raw)}"
            raise AEGFormatError(msg)
        return np.frombuffer(raw, dtype=np.dtype(dtype)).copy()

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
