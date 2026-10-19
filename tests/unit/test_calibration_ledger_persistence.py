"""Calibration persistence must not stall inference or damage existing evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aether.placement import ledger as ledger_module
from aether.placement.ledger import CalibrationLedger


class _WindowsTempfileOS:
    """Exercise CPython's Windows permission branch without changing os.name."""

    name = "nt"

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


def _existing_ledger(tmp_path: Path) -> tuple[CalibrationLedger, bytes]:
    store = CalibrationLedger(tmp_path / "calibration.json")
    store.record_notes("cpu", "test-build", {"old": {"measurement": 17}})
    original = store.path.read_bytes()
    store.autosave = False
    store.record_notes("cpu", "test-build", {"new": {"measurement": 23}})
    return store, original


def test_permission_denial_is_not_retried_as_a_filename_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denied write retains evidence and makes exactly one creation attempt."""
    store, original = _existing_ledger(tmp_path)
    opened: list[Path] = []
    real_open = os.open
    logger = Mock(wraps=ledger_module.logger)
    warning = logger.warning
    monkeypatch.setattr(ledger_module, "logger", logger)

    def denied(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        candidate = Path(path)
        if candidate.parent == tmp_path and candidate.name.startswith(".calibration-"):
            opened.append(candidate)
            raise PermissionError("sandbox denies creation despite writable ACL")
        return real_open(path, flags, mode, **kwargs)

    # The old Windows helper retries this PermissionError up to TMP_MAX times.
    # Cap that old failure during the regression rather than waiting minutes.
    monkeypatch.setattr(tempfile, "TMP_MAX", 4)
    monkeypatch.setattr(tempfile, "_os", _WindowsTempfileOS())
    monkeypatch.setattr(os, "open", denied)
    store.save()

    assert len(opened) == 1
    assert store.path.read_bytes() == original
    assert store.get("cpu", "test-build").notes["new"]["measurement"] == 23
    assert store._dirty
    assert not list(tmp_path.glob(".calibration-*.tmp"))
    warning.assert_called_once()
    assert warning.call_args.args[0] == "could not persist placement ledger: %s"
    assert isinstance(warning.call_args.args[1], PermissionError)


def test_failed_replace_cleans_temp_file_and_preserves_previous_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a fully published replacement may displace the previous ledger."""
    store, original = _existing_ledger(tmp_path)

    def denied_replace(source: Any, destination: Any) -> None:
        assert Path(destination) == store.path
        assert json.loads(Path(source).read_text(encoding="utf-8"))["entries"]
        raise PermissionError("ledger is currently read-only")

    monkeypatch.setattr(os, "replace", denied_replace)
    store.save()

    assert store.path.read_bytes() == original
    assert store._dirty
    assert not list(tmp_path.glob(".calibration-*.tmp"))


def test_serialization_error_propagates_and_cleans_partial_file(tmp_path: Path) -> None:
    """A caller's non-JSON notes are an error, with no partial file retained."""
    store, original = _existing_ledger(tmp_path)
    store.record_notes("cpu", "test-build", {"invalid": object()})

    with pytest.raises(TypeError, match="JSON serializable"):
        store.save()

    assert store.path.read_bytes() == original
    assert store._dirty
    assert not list(tmp_path.glob(".calibration-*.tmp"))


def test_filename_collisions_have_a_finite_retry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even genuine name collisions cannot cause an unbounded retry loop."""
    store, original = _existing_ledger(tmp_path)
    real_open = os.open
    attempts: list[Path] = []

    def collision(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        candidate = Path(path)
        if candidate.parent == tmp_path and candidate.name.startswith(".calibration-"):
            attempts.append(candidate)
            raise FileExistsError("another writer owns this name")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(tempfile, "TMP_MAX", 32)
    monkeypatch.setattr(os, "open", collision)
    store.save()

    assert 1 < len(attempts) <= 16
    assert store.path.read_bytes() == original
    assert store._dirty
    assert not list(tmp_path.glob(".calibration-*.tmp"))


def test_collisions_do_not_truncate_other_writers_and_can_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusive creation protects colliding files while allowing later success."""
    store, _ = _existing_ledger(tmp_path)
    real_open = os.open
    other_files: list[Path] = []

    def collided_twice(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        candidate = Path(path)
        if (
            candidate.parent == tmp_path
            and candidate.name.startswith(".calibration-")
            and len(other_files) < 2
        ):
            candidate.write_bytes(b"another writer's evidence")
            other_files.append(candidate)
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", collided_twice)
    store.save()

    assert len(other_files) == 2
    assert all(path.read_bytes() == b"another writer's evidence" for path in other_files)
    assert set(tmp_path.glob(".calibration-*.tmp")) == set(other_files)
    assert not store._dirty
    assert CalibrationLedger(store.path).get("cpu", "test-build").notes == {
        "old": {"measurement": 17}, "new": {"measurement": 23},
    }
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600


def test_fdopen_failure_closes_descriptor_and_removes_owned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership of the raw descriptor transfers only after fdopen succeeds."""
    store, original = _existing_ledger(tmp_path)
    descriptors: list[int] = []

    def failed_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
        descriptors.append(descriptor)
        raise OSError("cannot create text wrapper")

    monkeypatch.setattr(os, "fdopen", failed_fdopen)
    store.save()

    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert store.path.read_bytes() == original
    assert store._dirty
    assert not list(tmp_path.glob(".calibration-*.tmp"))


def test_torch_prefill_and_decode_keep_working_with_denied_calibration_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Torch arithmetic remains equivalent when its advisory cache is denied."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
    from aether.runtime.torch_engine import TorchAEGEngine

    rng = np.random.default_rng(7)

    def weight(rows: int, columns: int) -> Any:
        return rng.normal(0, 0.05, (rows, columns)).astype(np.float32)

    weights = ModelWeights(
        embedding=weight(32, 16),
        layers=[LayerWeights(
            attention_norm=np.ones(16, dtype=np.float32),
            q_proj=weight(16, 16), k_proj=weight(8, 16), v_proj=weight(8, 16),
            o_proj=weight(16, 16), ffn_norm=np.ones(16, dtype=np.float32),
            gate_proj=weight(32, 16), up_proj=weight(32, 16), down_proj=weight(16, 32),
        )],
        final_norm=np.ones(16, dtype=np.float32), lm_head=weight(32, 16),
        context_length=32,
    )
    monkeypatch.setenv("AETHER_TORCH_DTYPE", "fp32")
    monkeypatch.setenv("AETHER_PLACEMENT_LEDGER", str(tmp_path / "calibration.json"))

    def run() -> tuple[Any, Any, Any]:
        engine = TorchAEGEngine(
            CPUExecutionEngine(weights, num_heads=2, num_kv_heads=1),
            "cpu", retain_host_weights=True,
        )
        prefill, cache = engine._forward_device(
            np.array([1, 2]), None, reserve=8, logits="last",
        )
        decode, _ = engine._forward_device(np.array([3]), cache, logits="last")
        return engine, prefill, decode

    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "0")
    _, reference_prefill, reference_decode = run()
    real_open = os.open
    attempts: list[Path] = []

    def denied(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        candidate = Path(path)
        if candidate.parent == tmp_path and candidate.name.startswith(".calibration-"):
            attempts.append(candidate)
            raise PermissionError("read-only calibration cache")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(tempfile, "TMP_MAX", 4)
    monkeypatch.setattr(tempfile, "_os", _WindowsTempfileOS())
    monkeypatch.setattr(os, "open", denied)
    monkeypatch.setenv("AETHER_DECODE_CALIBRATION", "1")
    tuned, produced_prefill, produced_decode = run()

    torch.testing.assert_close(produced_prefill, reference_prefill, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(produced_decode, reference_decode, rtol=1e-5, atol=1e-6)
    assert tuned._strategies is not None
    choices = tuned._strategies.report()["choices"]
    assert choices and all(choice["source"] == "measured" for choice in choices)
    assert len(attempts) == len(choices)
    assert not (tmp_path / "calibration.json").exists()
    assert not list(tmp_path.glob(".calibration-*.tmp"))
