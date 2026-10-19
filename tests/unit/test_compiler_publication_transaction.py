"""Filesystem and native-lock contracts for compiler candidate publication."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from aether.compiler import publication
from aether.core.exceptions import CompilationError
from aether.utils.artifact_publication import publish_staged_directory


def _old(root: Path) -> Path:
    target = root / "model.aeg"
    target.mkdir()
    (target / "old.bin").write_bytes(b"previous complete bytes")
    return target


def _candidate(transaction: publication.CompilationPublication) -> None:
    transaction.candidate.mkdir()
    (transaction.candidate / "new.bin").write_bytes(b"accepted candidate bytes")


def test_accepted_overwrite_moves_complete_directory_and_removes_backup(tmp_path: Path) -> None:
    """The old artifact exists throughout candidate construction and acceptance."""
    target = _old(tmp_path)
    with publication.compilation_publication(target, overwrite=True) as transaction:
        _candidate(transaction)
        assert (target / "old.bin").read_bytes() == b"previous complete bytes"
        transaction.publish()
        assert (target / "new.bin").read_bytes() == b"accepted candidate bytes"
        assert transaction.previous.is_dir()
        assert not transaction.candidate.exists()
    assert not list(tmp_path.glob(".aether-compile-*"))
    assert list(tmp_path.glob(".aether-publication-*.lock"))


def test_failed_promotion_restores_previous_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed second rename restores the old destination before returning an error."""
    target = _old(tmp_path)
    real_publish = publish_staged_directory
    with (
        pytest.raises(OSError, match="promotion denied"),
        publication.compilation_publication(target, overwrite=True) as transaction,
    ):
        _candidate(transaction)

        def fail_candidate(source: Any, destination: Any) -> Path:
            if Path(source) == transaction.candidate:
                raise PermissionError("promotion denied")
            return real_publish(source, destination)

        monkeypatch.setattr(publication, "publish_staged_directory", fail_candidate)
        transaction.publish()
    assert (target / "old.bin").read_bytes() == b"previous complete bytes"
    assert not (target / "new.bin").exists()
    assert not list(tmp_path.glob(".aether-compile-*"))


def test_failed_rollback_retains_recoverable_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup cannot delete the only remaining copy if rollback itself is denied."""
    target = _old(tmp_path)
    real_publish = publish_staged_directory
    with (
        pytest.raises(CompilationError, match="previous artifact is retained") as error,
        publication.compilation_publication(target, overwrite=True) as transaction,
    ):
        _candidate(transaction)

        def fail_promotion_and_rollback(source: Any, destination: Any) -> Path:
            if Path(source) in (transaction.candidate, transaction.previous):
                raise PermissionError("controlled rename denial")
            return real_publish(source, destination)

        monkeypatch.setattr(publication, "publish_staged_directory", fail_promotion_and_rollback)
        transaction.publish()
    recovery = Path(error.value.details["recovery_path"])
    assert (recovery / "old.bin").read_bytes() == b"previous complete bytes"
    assert not target.exists()
    assert error.value.__cause__ is not None
    assert recovery.parent in list(tmp_path.glob(".aether-compile-*"))


def test_cancellation_during_promotion_also_restores_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyboard interruption must not leave a moved old artifact unpublished."""
    target = _old(tmp_path)
    real_publish = publish_staged_directory
    with (
        pytest.raises(KeyboardInterrupt),
        publication.compilation_publication(target, overwrite=True) as transaction,
    ):
        _candidate(transaction)

        def cancel_candidate(source: Any, destination: Any) -> Path:
            if Path(source) == transaction.candidate:
                raise KeyboardInterrupt
            return real_publish(source, destination)

        monkeypatch.setattr(publication, "publish_staged_directory", cancel_candidate)
        transaction.publish()
    assert (target / "old.bin").read_bytes() == b"previous complete bytes"
    assert not list(tmp_path.glob(".aether-compile-*"))


def test_cancellation_after_backup_rename_restores_previous_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed OS rename must be recovered before any Python flag is updated."""
    target = _old(tmp_path)
    with (
        pytest.raises(KeyboardInterrupt),
        publication.compilation_publication(target, overwrite=True) as transaction,
    ):
        _candidate(transaction)

        def move_then_cancel(source: Any, destination: Any) -> Path:
            result = publish_staged_directory(source, destination)
            if Path(source) == target:
                raise KeyboardInterrupt
            return result

        monkeypatch.setattr(publication, "publish_staged_directory", move_then_cancel)
        transaction.publish()
    assert (target / "old.bin").read_bytes() == b"previous complete bytes"
    assert not list(tmp_path.glob(".aether-compile-*"))


def test_second_process_cannot_enter_same_destination_until_lock_released(tmp_path: Path) -> None:
    """A real child process is refused by the held native writer lock."""
    target = _old(tmp_path)
    script = """
import sys
from aether.compiler.publication import compilation_publication
from aether.core.exceptions import CompilationError
try:
    with compilation_publication(sys.argv[1], overwrite=True):
        raise AssertionError('second process acquired an already held lock')
except CompilationError as error:
    assert error.stage == 'publication'
    assert 'publication lock' in str(error)
    print('native lock refused the competing compiler')
"""
    with publication.compilation_publication(target, overwrite=True):
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(target)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "refused the competing compiler" in result.stdout
    # A retained lock filename is not a stale lock: the kernel lock was released.
    with publication.compilation_publication(target, overwrite=True):
        assert (target / "old.bin").read_bytes() == b"previous complete bytes"


@pytest.mark.parametrize(
    "error_type,expected_attempts", [(PermissionError, 1), (FileExistsError, 8)]
)
def test_stage_creation_retries_only_bounded_name_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
    expected_attempts: int,
) -> None:
    """Denied staging creation is immediate and even collisions have a fixed bound."""
    target = tmp_path / "model.aeg"
    real_mkdir = Path.mkdir
    attempts = []

    def fail_stage(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name.startswith(".aether-compile-"):
            attempts.append(path)
            raise error_type("controlled staging creation error")
        real_mkdir(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", fail_stage)
        with (
            pytest.raises(error_type),
            publication.compilation_publication(target, overwrite=True),
        ):
            pytest.fail("failed stage creation must not enter the body")
    assert len(attempts) == expected_attempts
    assert not target.exists()
    with publication.compilation_publication(target, overwrite=True):
        pass


def test_overwrite_cannot_replace_a_source_directory_or_parent(tmp_path: Path) -> None:
    """Output validation protects an explicitly supplied local source from deletion."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_bytes(b"source config bytes")
    for target in (source, tmp_path):
        with (
            pytest.raises(ValueError, match="source checkpoint"),
            publication.compilation_publication(target, overwrite=True, source_path=source),
        ):
            pytest.fail("overlapping source must be refused")
    assert (source / "config.json").read_bytes() == b"source config bytes"
    assert not list(tmp_path.glob(".aether-publication-*.lock"))


def test_existing_file_is_preserved_even_with_overwrite(tmp_path: Path) -> None:
    """A package-directory overwrite must not silently replace a regular file."""
    target = tmp_path / "model.aeg"
    target.write_bytes(b"unrelated file")
    with (
        pytest.raises(ValueError, match="ordinary directory"),
        publication.compilation_publication(target, overwrite=True),
    ):
        pytest.fail("an existing file must be refused")
    assert target.read_bytes() == b"unrelated file"


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows sharing semantics")
def test_real_windows_old_member_lock_preserves_published_output(tmp_path: Path) -> None:
    """A live file handle blocks replacement without deleting the previous directory."""
    from test_artifact_publication import _deny_delete_sharing

    target = _old(tmp_path)
    with (
        _deny_delete_sharing(target / "old.bin"),
        pytest.raises(PermissionError),
        publication.compilation_publication(target, overwrite=True) as transaction,
    ):
        _candidate(transaction)
        transaction.publish()
    assert os.path.isdir(target)
    assert (target / "old.bin").read_bytes() == b"previous complete bytes"
    assert not list(tmp_path.glob(".aether-compile-*"))


def test_publication_import_has_a_bounded_python_allocation_budget() -> None:
    """Publication must not add a general locking framework's measured 1.30 MiB."""
    script = """
import json
import tracemalloc
from aether.compiler.compiler import Compiler
tracemalloc.start()
before = tracemalloc.take_snapshot()
import aether.compiler.publication
allocated = sum(x.size_diff for x in tracemalloc.take_snapshot().compare_to(before, 'filename'))
print(json.dumps({'allocated_python_bytes': allocated}))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["allocated_python_bytes"] <= 256 * 1024


@pytest.mark.parametrize("fail_acquire", [True, False])
def test_failed_native_lock_operation_closes_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_acquire: bool,
) -> None:
    """Unsupported locking or failed explicit unlock cannot leak a file descriptor."""
    lock_path = tmp_path / "writer.lock"
    real_lock = publication._lock_descriptor
    descriptors: list[int] = []

    def fail_operation(descriptor: int, *, acquire: bool) -> None:
        descriptors.append(descriptor)
        if acquire == fail_acquire:
            raise OSError(errno.ENOSYS, "controlled unsupported native operation")
        real_lock(descriptor, acquire=acquire)

    error_type = CompilationError if fail_acquire else OSError
    with monkeypatch.context() as patch:
        patch.setattr(publication, "_lock_descriptor", fail_operation)
        with (
            pytest.raises(error_type, match="controlled unsupported native operation"),
            publication._publication_lock(lock_path),
        ):
            assert not fail_acquire
    assert descriptors
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    with publication._publication_lock(lock_path):
        assert lock_path.is_file()


def test_process_exit_releases_kernel_lock_without_python_cleanup(tmp_path: Path) -> None:
    """A real child exits inside the lock; no stale marker lease blocks the next writer."""
    lock_path, marker = tmp_path / "writer.lock", tmp_path / "acquired.bin"
    script = """
import os
import sys
from pathlib import Path
from aether.compiler.publication import _publication_lock
with _publication_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_bytes(b'acquired before immediate process exit')
    os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(lock_path), str(marker)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_bytes() == b"acquired before immediate process exit"
    with publication._publication_lock(lock_path):
        assert lock_path.is_file()
