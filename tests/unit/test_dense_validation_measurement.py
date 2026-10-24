"""Protect benchmark byte accounting against alias inflation and undercounting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import py_compile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tools.pass_audit import PASSES
from tools.validate_dense_hf import (
    _array_bytes,
    _compiler_config,
    _load_profile,
    _prepare_source_bytecode,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_overlapping_views_count_only_their_union() -> None:
    """Count shared subranges once while retaining independent allocations."""

    @dataclass
    class Weights:
        projection: NDArray[np.generic]

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
    assert result["loaded_artifact"] == str(destination)
    assert (
        result["loaded_manifest_sha256"]
        == hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()
    )
    assert discoveries == 1


def test_both_reader_caches_are_warm_even_when_workers_cannot_write(tmp_path: Path) -> None:
    """Reproduce -B reading one parent's cache and compiling the cold baseline."""
    roots = [tmp_path / "current", tmp_path / "baseline"]
    for source in roots:
        package = source / "aether"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "probe.py").write_text("VALUE = 1729\n", encoding="utf-8")
    py_compile.compile(str(roots[0] / "aether/probe.py"), doraise=True)
    script = """
import importlib.util, json, sys
from pathlib import Path
source = Path(sys.argv[1])
target = (source / 'aether/probe.py').resolve()
events = []
def audit(event, args):
    if event == 'compile' and len(args) > 1 and Path(args[1]).resolve() == target:
        events.append(event)
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location('cache_probe', target)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    'compilations': len(events), 'value': module.VALUE,
    'dont_write': sys.dont_write_bytecode,
}))
"""

    def observed(source: Path) -> dict[str, int]:
        result = subprocess.check_output(
            [sys.executable, "-B", "-c", script, str(source)], text=True
        )
        value = json.loads(result)
        assert isinstance(value, dict)
        return value

    before = [observed(source) for source in roots]
    assert [row["compilations"] for row in before] == [0, 1]
    policy = _prepare_source_bytecode(roots)
    assert [tree["prepared_modules"] for tree in policy["trees"]] == [2, 2]
    after = [observed(source) for source in roots]
    assert [row["compilations"] for row in after] == [0, 0]
    assert all(row["value"] == 1729 and row["dont_write"] for row in after)
    assert all(
        Path(importlib.util.cache_from_source(str(source / "aether/probe.py"))).is_file()
        for source in roots
    )
