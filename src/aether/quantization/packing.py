"""
Bit-packing utilities for quantized weights.

Provides routines for packing and unpacking quantized integers into compact
bit-level representations. Used to store Q3_K, Q4_K_M, and other block-level
quantized formats efficiently.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class BitPacker:
    """Packs and unpacks arrays of small integers into bit-packed representations.

    Supports arbitrary bit widths from 1 to 8. The packed format stores multiple
    elements per byte, enabling efficient storage for Q3_K (3-bit), Q4_K_M (4-bit),
    Q2_K (2-bit), and similar formats.
    """

    def __init__(self, bit_width: int) -> None:
        self.bit_width = bit_width
        self.elements_per_byte = 8 // bit_width if bit_width > 0 else 1

    def pack(self, values: np.ndarray) -> np.ndarray:
        """Pack an array of integers into a compact bit-packed representation.

        Element ``i`` occupies bits ``[i * bit_width, (i + 1) * bit_width)`` of the
        stream, little-endian within each byte. Fully vectorized: a 7B-parameter
        tensor packs in seconds rather than hours.

        Args:
            values: Integer array with values in ``[0, 2^bit_width)``. Values wider
                than ``bit_width`` are masked to their low bits.

        Returns:
            Bit-packed uint8 array of length ``ceil(n * bit_width / 8)``.
        """
        flat = np.ascontiguousarray(values).ravel().astype(np.uint8)
        if flat.size == 0:
            return np.zeros(0, dtype=np.uint8)
        if self.bit_width >= 8:
            return flat.copy()
        # Expand each byte to its 8 little-endian bits, keep the low bit_width.
        bits = np.unpackbits(flat[:, None], axis=1, bitorder="little")[:, : self.bit_width]
        return np.packbits(bits.reshape(-1), bitorder="little")

    def unpack(self, packed: np.ndarray, count: int) -> np.ndarray:
        """Unpack a bit-packed representation back to integers.

        Inverse of :meth:`pack`. Fully vectorized.

        Args:
            packed: Packed uint8 array.
            count: Number of elements to extract.

        Returns:
            Unpacked uint8 array of shape ``(count,)``.

        Raises:
            ValueError: If ``packed`` holds fewer bits than ``count`` requires.
        """
        if count <= 0:
            return np.zeros(0, dtype=np.uint8)
        buf = np.ascontiguousarray(packed).ravel().astype(np.uint8)
        if self.bit_width >= 8:
            if buf.size < count:
                msg = f"packed buffer holds {buf.size} elements, need {count}"
                raise ValueError(msg)
            return buf[:count].copy()

        needed_bits = count * self.bit_width
        if buf.size * 8 < needed_bits:
            msg = (
                f"packed buffer holds {buf.size * 8} bits, need {needed_bits} "
                f"for {count} elements at {self.bit_width}-bit"
            )
            raise ValueError(msg)
        bits = np.unpackbits(buf, bitorder="little")[:needed_bits].reshape(count, self.bit_width)
        # Zero-extend each group back to 8 bits before repacking into bytes.
        padded = np.zeros((count, 8), dtype=np.uint8)
        padded[:, : self.bit_width] = bits
        return np.packbits(padded, axis=1, bitorder="little").reshape(-1)

    def __repr__(self) -> str:
        return f"BitPacker(bit_width={self.bit_width}, elems_per_byte={self.elements_per_byte})"
