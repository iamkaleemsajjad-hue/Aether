"""Tests for vectorized bit-packing.

The packer was rewritten from a per-element Python loop to a vectorized
implementation (4 hours -> 5 minutes for a 7B-parameter tensor). These tests pin
the exact bit layout against an independent scalar reference so the rewrite
cannot silently change the on-disk format.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.quantization.packing import BitPacker

SUB_BYTE_WIDTHS = [1, 2, 3, 4, 5, 6, 7]
ELEMENT_COUNTS = [1, 2, 5, 8, 17, 32, 64, 257]


def scalar_pack_reference(values: np.ndarray, bit_width: int) -> np.ndarray:
    """Independent scalar implementation of the intended bit layout.

    Deliberately written as a plain loop so it shares no code with the
    vectorized implementation under test.
    """
    flat = values.ravel().astype(np.uint64)
    out_size = (flat.size * bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    packed = np.zeros(out_size, dtype=np.uint8)
    for i in range(flat.size):
        bit_pos = i * bit_width
        byte_idx, bit_offset = bit_pos // 8, bit_pos % 8
        value = int(flat[i]) & mask
        packed[byte_idx] |= (value << bit_offset) & 0xFF
        if bit_offset + bit_width > 8 and byte_idx + 1 < out_size:
            packed[byte_idx + 1] |= (value >> (8 - bit_offset)) & 0xFF
    return packed


class TestBitLayout:
    @pytest.mark.parametrize("bit_width", SUB_BYTE_WIDTHS)
    @pytest.mark.parametrize("count", ELEMENT_COUNTS)
    def test_matches_scalar_reference_byte_for_byte(self, bit_width: int, count: int) -> None:
        values = np.random.RandomState(bit_width * 100 + count).randint(
            0, 1 << bit_width, size=count
        ).astype(np.uint8)
        np.testing.assert_array_equal(
            BitPacker(bit_width).pack(values), scalar_pack_reference(values, bit_width)
        )

    def test_known_layout_is_little_endian_within_bytes(self) -> None:
        """Two 4-bit values 0x1, 0x2 pack into one byte as 0x21, low nibble first."""
        packed = BitPacker(4).pack(np.array([0x1, 0x2], dtype=np.uint8))
        assert packed.tolist() == [0x21]

    def test_two_bit_layout_packs_four_per_byte(self) -> None:
        """Values 1,2,3,0 at 2 bits each -> 0b00_11_10_01 = 0x39."""
        packed = BitPacker(2).pack(np.array([1, 2, 3, 0], dtype=np.uint8))
        assert packed.tolist() == [0x39]

    def test_three_bit_value_spans_a_byte_boundary(self) -> None:
        """At 3 bits, element 2 straddles bytes 0 and 1 — the tricky carry case."""
        values = np.array([0b111, 0b111, 0b111], dtype=np.uint8)
        packed = BitPacker(3).pack(values)
        np.testing.assert_array_equal(packed, scalar_pack_reference(values, 3))
        np.testing.assert_array_equal(BitPacker(3).unpack(packed, 3), values)


class TestRoundtrip:
    @pytest.mark.parametrize("bit_width", SUB_BYTE_WIDTHS + [8])
    @pytest.mark.parametrize("count", ELEMENT_COUNTS)
    def test_pack_unpack_is_lossless(self, bit_width: int, count: int) -> None:
        values = np.random.RandomState(bit_width + count).randint(
            0, 1 << min(bit_width, 8), size=count
        ).astype(np.uint8)
        packer = BitPacker(bit_width)
        np.testing.assert_array_equal(packer.unpack(packer.pack(values), count), values)

    @pytest.mark.parametrize("bit_width", SUB_BYTE_WIDTHS)
    def test_full_code_range_roundtrips(self, bit_width: int) -> None:
        values = np.arange(1 << bit_width, dtype=np.uint8)
        packer = BitPacker(bit_width)
        np.testing.assert_array_equal(packer.unpack(packer.pack(values), values.size), values)

    @pytest.mark.parametrize("bit_width", SUB_BYTE_WIDTHS)
    def test_all_zeros_and_all_max(self, bit_width: int) -> None:
        packer = BitPacker(bit_width)
        for fill in (0, (1 << bit_width) - 1):
            values = np.full(64, fill, dtype=np.uint8)
            np.testing.assert_array_equal(packer.unpack(packer.pack(values), 64), values)

    def test_multidimensional_input_is_flattened(self) -> None:
        values = np.random.RandomState(0).randint(0, 16, size=(8, 8)).astype(np.uint8)
        packer = BitPacker(4)
        unpacked = packer.unpack(packer.pack(values), values.size)
        np.testing.assert_array_equal(unpacked, values.ravel())


class TestCompression:
    @pytest.mark.parametrize(
        ("bit_width", "count", "expected_bytes"),
        [(4, 256, 128), (2, 256, 64), (3, 256, 96), (1, 256, 32), (8, 256, 256)],
    )
    def test_packed_size_reflects_bit_width(
        self, bit_width: int, count: int, expected_bytes: int
    ) -> None:
        values = np.zeros(count, dtype=np.uint8)
        assert BitPacker(bit_width).pack(values).nbytes == expected_bytes

    def test_four_bit_halves_the_payload(self) -> None:
        values = np.random.RandomState(1).randint(0, 16, size=1024).astype(np.uint8)
        assert BitPacker(4).pack(values).nbytes == values.nbytes // 2


class TestEdgeCases:
    def test_empty_input_returns_empty(self) -> None:
        assert BitPacker(4).pack(np.array([], dtype=np.uint8)).size == 0

    def test_unpack_zero_count_returns_empty(self) -> None:
        assert BitPacker(4).unpack(np.array([0xFF], dtype=np.uint8), 0).size == 0

    def test_unpack_beyond_buffer_raises(self) -> None:
        packed = BitPacker(4).pack(np.array([1, 2], dtype=np.uint8))
        with pytest.raises(ValueError, match="need"):
            BitPacker(4).unpack(packed, 100)

    def test_byte_width_is_passthrough(self) -> None:
        values = np.arange(256, dtype=np.uint8)
        np.testing.assert_array_equal(BitPacker(8).pack(values), values)

    def test_pack_does_not_mutate_input(self) -> None:
        values = np.random.RandomState(2).randint(0, 16, size=32).astype(np.uint8)
        original = values.copy()
        BitPacker(4).pack(values)
        np.testing.assert_array_equal(values, original)

    def test_elements_per_byte_metadata(self) -> None:
        assert BitPacker(4).elements_per_byte == 2
        assert BitPacker(2).elements_per_byte == 4
        assert BitPacker(8).elements_per_byte == 1


class TestPerformance:
    def test_large_tensor_packs_quickly(self) -> None:
        """Guards against regressing to the per-element Python loop."""
        import time

        values = np.random.RandomState(3).randint(0, 16, size=2_000_000).astype(np.uint8)
        packer = BitPacker(4)
        start = time.perf_counter()
        packed = packer.pack(values)
        unpacked = packer.unpack(packed, values.size)
        elapsed = time.perf_counter() - start
        np.testing.assert_array_equal(unpacked, values)
        # The scalar loop needed ~4 s for this; vectorized runs in well under 1 s.
        assert elapsed < 2.0, f"packing 2M elements took {elapsed:.2f}s"
