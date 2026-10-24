"""Protect benchmark byte accounting against alias inflation and undercounting."""

from dataclasses import dataclass

import numpy as np
import pytest
from tools.validate_dense_hf import _array_bytes


def test_overlapping_views_count_only_their_union() -> None:
    """Count shared subranges once while retaining independent allocations."""

    @dataclass
    class Weights:
        projection: np.ndarray

    storage = np.zeros(32, dtype=np.float32)
    separate = np.zeros(4, dtype=np.float32)
    observed = {
        "weights": Weights(storage[4:20]),
        "cache": [storage[16:24], storage[4:20], None],
        "other": (separate,),
    }
    # The first allocation contributes elements [4, 24), the second all four.
    assert _array_bytes(observed) == (20 + 4) * storage.dtype.itemsize


def test_fortran_contiguous_aliases_share_the_same_span() -> None:
    """A transposed contiguous view contributes no second payload."""
    storage = np.arange(24, dtype=np.float32).reshape(4, 6)
    assert storage.T.flags.f_contiguous
    assert _array_bytes([storage, storage.T]) == storage.nbytes


def test_equal_independent_arrays_are_both_counted() -> None:
    """Equal values do not imply shared storage; empty arrays add no bytes."""
    first = np.ones(16, dtype=np.float32)
    assert _array_bytes([first, first.copy()]) == 2 * first.nbytes
    assert _array_bytes([np.empty(0), None]) == 0


def test_strided_storage_refuses_misleading_contiguous_accounting() -> None:
    """Refuse layouts whose allocation cannot be described by this span method."""
    storage = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError, match="contiguous array spans"):
        _array_bytes(storage[::2])
