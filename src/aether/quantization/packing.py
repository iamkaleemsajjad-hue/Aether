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
        
        # For large arrays, pack in 1M element chunks to keep memory footprint minimal
        chunk_size = 1_048_576
        if flat.size > chunk_size:
            out_chunks = []
            for i in range(0, flat.size, chunk_size):
                chunk = flat[i : i + chunk_size]
                bits = np.unpackbits(chunk[:, None], axis=1, bitorder="little")[:, : self.bit_width]
                out_chunks.append(np.packbits(bits.reshape(-1), bitorder="little"))
            return np.concatenate(out_chunks)

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

        if self.bit_width == 4:
            needed_bytes = (count + 1) // 2
            slice_buf = buf[:needed_bytes]
            low = slice_buf & np.uint8(0x0F)
            high = (slice_buf >> np.uint8(4)) & np.uint8(0x0F)
            return np.column_stack([low, high]).ravel()[:count]

        if self.bit_width == 2:
            needed_bytes = (count + 3) // 4
            slice_buf = buf[:needed_bytes]
            c0 = slice_buf & np.uint8(0x03)
            c1 = (slice_buf >> np.uint8(2)) & np.uint8(0x03)
            c2 = (slice_buf >> np.uint8(4)) & np.uint8(0x03)
            c3 = (slice_buf >> np.uint8(6)) & np.uint8(0x03)
            return np.column_stack([c0, c1, c2, c3]).ravel()[:count]

        chunk_size = 1048576
        if count > chunk_size:
            out_chunks = []
            for i in range(0, count, chunk_size):
                curr_count = min(chunk_size, count - i)
                curr_needed_bits = curr_count * self.bit_width
                curr_byte_start = (i * self.bit_width) // 8
                curr_byte_end = ((i + curr_count) * self.bit_width + 7) // 8
                curr_buf = buf[curr_byte_start:curr_byte_end]
                bits = np.unpackbits(curr_buf, bitorder="little")[:curr_needed_bits].reshape(curr_count, self.bit_width)
                padded = np.zeros((curr_count, 8), dtype=np.uint8)
                padded[:, : self.bit_width] = bits
                out_chunks.append(np.packbits(padded, axis=1, bitorder="little").reshape(-1))
            return np.concatenate(out_chunks)

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
