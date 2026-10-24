"""Protect benchmark byte accounting against alias inflation and undercounting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from tools.validate_dense_hf import PASSES, _array_bytes, _compiler_config, _load_profile


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


def test_dense_benchmark_config_accepts_fp32_and_disables_unrelated_passes(tmp_path: Path) -> None:
    """An explicit precision needs P3; validate with the actual compiler config."""
    config = _compiler_config(
        {"diagnostic": False, "dataset": str(tmp_path / "probe.jsonl")}, tmp_path
    )
    assert [entry.number for entry in PASSES if getattr(config, entry.flag)] == [3]
    assert config.precision == "FP32"
    assert config.precision_assignment_mode == "uniform"
    assert config.quality_mode == "require_hf"


def test_load_warms_the_actual_shared_provider(
    tmp_path: Path, tiny_local_safetensors_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual artifact loading must not discover another compiler after warmup."""
    from aether import Compiler
    from aether.kernels import native_cpu

    job = {"diagnostic": True, "dataset": None}
    destination = tmp_path / "model.aeg"
    Compiler(_compiler_config(job, tmp_path)).compile(
        str(tiny_local_safetensors_model), output_path=destination
    )
    discoveries = 0

    def no_compiler() -> None:
        nonlocal discoveries
        discoveries += 1

    # A fresh worker starts with no shared provider. Simulate a host without a
    # compiler; the artifact and real loader still execute their normal paths.
    monkeypatch.setattr(native_cpu, "_INSTANCE", None)
    monkeypatch.setattr(native_cpu, "detect_toolchain", no_compiler)
    work = tmp_path / "load"
    work.mkdir()
    result = _load_profile(
        {"artifact": str(destination), "diagnostic": True, "exercise": False}, work
    )
    assert result["status"] == "ok"
    assert discoveries == 1
