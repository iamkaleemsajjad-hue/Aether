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

        Args:
            values: Integer array with values in [0, 2^bit_width).

        Returns:
            Bit-packed uint8 array.
        """
        flat = values.ravel().astype(np.uint64)
        n = len(flat)
        out_size = (n * self.bit_width + 7) // 8
        mask = (1 << self.bit_width) - 1
        packed = np.zeros(out_size, dtype=np.uint8)
        for i in range(n):
            bit_pos = i * self.bit_width
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            val = int(flat[i]) & mask
            packed[byte_idx] |= (val << bit_offset) & 0xFF
            if bit_offset + self.bit_width > 8 and byte_idx + 1 < out_size:
                packed[byte_idx + 1] |= (val >> (8 - bit_offset)) & 0xFF
        return packed

    def unpack(self, packed: np.ndarray, count: int) -> np.ndarray:
        """Unpack a bit-packed representation back to integers.

        Args:
            packed: Packed uint8 array.
            count: Number of elements to extract.

        Returns:
            Unpacked uint8 array of shape (count,).
        """
        result = np.zeros(count, dtype=np.uint8)
        for i in range(count):
            bit_pos = i * self.bit_width
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            val = packed[byte_idx] >> bit_offset
            if bit_offset + self.bit_width > 8 and byte_idx + 1 < len(packed):
                val |= packed[byte_idx + 1] << (8 - bit_offset)
            result[i] = val & ((1 << self.bit_width) - 1)
        return result

    def __repr__(self) -> str:
        return f"BitPacker(bit_width={self.bit_width}, elems_per_byte={self.elements_per_byte})"
