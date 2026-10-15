"""Publication policy and actual Windows handle sharing, with no hardware claims."""

import ctypes
import errno
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from aether.utils import artifact_publication as publication


def _stage(tmp_path):
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"verified complete artifact")
    return source, tmp_path / "published"


@contextmanager
def _deny_delete_sharing(path):
    """Own a real Win32 handle until explicitly closed; no fabricated lock."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, OPEN_EXISTING.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    closed = False

    def close():
        nonlocal closed
        if not closed:
            if not kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            closed = True

    try:
        yield close
    finally:
        close()


def test_immediate_publication_never_sleeps(tmp_path, monkeypatch):
    """Ordinary publication performs one atomic rename without backoff."""
    source, destination = _stage(tmp_path)
    waits = []
    monkeypatch.setattr(publication.time, "sleep", waits.append)
    assert publication.publish_staged_directory(source, destination) == destination
    assert (destination / "payload.bin").read_bytes() == b"verified complete artifact"
    assert not source.exists()
    assert not waits


def test_preexisting_empty_destination_is_preserved(tmp_path):
    """Even an empty existing destination belongs to its current owner."""
    source, destination = _stage(tmp_path)
    destination.mkdir()
    with pytest.raises(FileExistsError):
        publication.publish_staged_directory(source, destination)
    assert source.is_dir() and destination.is_dir()


def test_preexisting_dangling_destination_link_is_preserved(tmp_path):
    """A dangling link is an existing destination, not permission to overwrite."""
    source, destination = _stage(tmp_path)
    try:
        destination.symlink_to(tmp_path / "absent", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"OS does not permit unprivileged symlink creation: {exc}")
    with pytest.raises(FileExistsError):
        publication.publish_staged_directory(source, destination)
    assert source.is_dir() and destination.is_symlink()


@pytest.mark.skipif(sys.platform != "win32", reason="requires actual Windows file sharing")
def test_real_member_handle_blocks_rename_then_retry_publishes(tmp_path, monkeypatch):
    """A genuine Windows sharing failure recovers after its handle is released."""
    source, destination = _stage(tmp_path)
    waits = []
    with _deny_delete_sharing(source / "payload.bin") as close:
        with pytest.raises(PermissionError) as blocked:
            source.rename(destination)
        assert blocked.value.winerror in {5, 32, 33}

        def release_on_retry(delay):
            assert source.exists() and not destination.exists()
            waits.append(delay)
            close()

        monkeypatch.setattr(publication.time, "sleep", release_on_retry)
        publication.publish_staged_directory(source, destination)
    assert waits == [publication._WINDOWS_RETRY_DELAYS[0]]
    assert not source.exists()
    assert (destination / "payload.bin").read_bytes() == b"verified complete artifact"


@pytest.mark.skipif(sys.platform != "win32", reason="requires actual Windows file sharing")
def test_real_persistent_member_lock_fails_after_bounded_attempts(tmp_path, monkeypatch):
    """A retained handle never turns persistent denial into a false success."""
    source, destination = _stage(tmp_path)
    waits = []
    monkeypatch.setattr(publication.time, "sleep", waits.append)
    with _deny_delete_sharing(source / "payload.bin"), pytest.raises(PermissionError) as failed:
        publication.publish_staged_directory(source, destination)
    assert failed.value.winerror in {5, 32, 33}
    assert isinstance(failed.value.__cause__, PermissionError)
    assert failed.value.__cause__.winerror in {5, 32, 33}
    assert waits == list(publication._WINDOWS_RETRY_DELAYS)
    assert source.exists() and not destination.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="requires actual Windows file sharing")
def test_destination_created_during_real_lock_retry_is_preserved(tmp_path, monkeypatch):
    """A competing publication during backoff must retain its complete output."""
    source, destination = _stage(tmp_path)
    waits = []
    with _deny_delete_sharing(source / "payload.bin") as close:

        def competing_publisher(delay):
            waits.append(delay)
            close()
            destination.mkdir()
            (destination / "other.bin").write_bytes(b"another publisher")

        monkeypatch.setattr(publication.time, "sleep", competing_publisher)
        with pytest.raises(FileExistsError) as failed:
            publication.publish_staged_directory(source, destination)
    assert isinstance(failed.value.__cause__, PermissionError)
    assert waits == [publication._WINDOWS_RETRY_DELAYS[0]]
    assert (destination / "other.bin").read_bytes() == b"another publisher"
    assert not (destination / "payload.bin").exists()
    assert source.is_dir()


@pytest.mark.parametrize("windows,error", [(False, 5), (True, 17), (True, 2)])
def test_other_platform_or_error_never_retries(tmp_path, monkeypatch, windows, error):
    """Deterministic error-policy test; separate from real Windows handle evidence."""
    source, destination = _stage(tmp_path)
    waits = []
    failure = OSError(errno.EACCES, "policy sentinel")
    failure.winerror = error
    monkeypatch.setattr(publication, "_WINDOWS", windows)
    monkeypatch.setattr(publication.time, "sleep", waits.append)

    def fail_rename(self, target):
        raise failure

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError) as failed:
        publication.publish_staged_directory(source, destination)
    assert failed.value is failure
    assert not waits
