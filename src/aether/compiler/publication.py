"""Stage compiler output before publishing; retain old bytes until acceptance.

Native descriptor locks serialize cooperating writers to the same destination.
Windows uses msvcrt.locking; POSIX uses fcntl.flock, without soft-lock fallback.
Overwrite uses two directory renames with rollback, not a crash-atomic exchange.
Readers can briefly observe a missing destination during successful overwrite.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from aether.core.exceptions import CompilationError
from aether.utils.artifact_publication import publish_staged_directory
from aether.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


def _lock_descriptor(descriptor: int, *, acquire: bool) -> None:
    """Lock/unlock byte zero on Windows or the descriptor's file on POSIX.

    Windows nonblocking and matching-range guarantees:
    https://docs.python.org/3.10/library/msvcrt.html#msvcrt.locking
    https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking
    CPython v3.10.11 PC/msvcrtmodule.c::msvcrt_locking_impl calls that CRT API.
    POSIX: https://docs.python.org/3.10/library/fcntl.html#fcntl.flock
    A failed native call propagates; no marker lease or retry substitutes for it.
    """
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        operation = fcntl.LOCK_EX | fcntl.LOCK_NB if acquire else fcntl.LOCK_UN
        fcntl.flock(descriptor, operation)


@contextmanager
def _publication_lock(path: Path) -> Iterator[None]:
    """Hold one native writer lock, keeping its stable filename after release."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        opened, named = os.fstat(descriptor), path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or not os.path.samestat(opened, named)
        ):
            raise ValueError("Publication lock must be the named regular file, not a link")
        try:
            _lock_descriptor(descriptor, acquire=True)
        except OSError as exc:
            raise CompilationError(
                f"Could not acquire the native AEG publication lock at {path}: {exc}",
                stage="publication",
            ) from exc
        try:
            yield
        finally:
            _lock_descriptor(descriptor, acquire=False)
    finally:
        os.close(descriptor)


class CompilationPublication:
    """One compiler-owned candidate and optional previous artifact, on one filesystem."""

    def __init__(self, destination: Path, *, overwrite: bool) -> None:
        self.destination = destination
        self.overwrite = overwrite
        self._root: Path | None = None
        self._published = False

    def check_destination(self) -> None:
        """Refuse links, filesystem roots and unrequested replacement."""
        target = self.destination
        if target == Path(target.anchor):
            raise ValueError("A filesystem root cannot be an AEG output directory")
        if os.path.lexists(target):
            if target.is_symlink() or target.resolve() != target or not target.is_dir():
                raise ValueError("AEG output must be an ordinary directory, not a link or file")
            if not self.overwrite:
                raise FileExistsError(
                    f"AEG output path already exists: {target}. Use overwrite=True to replace."
                )

    def create_stage(self) -> None:
        """Create an exclusive sibling directory with bounded collision retries."""
        for _ in range(8):
            root = self.destination.parent / f".aether-compile-{secrets.token_hex(16)}"
            try:
                root.mkdir(mode=0o700)
            except FileExistsError:
                continue
            self._root = root
            return
        raise FileExistsError("Could not allocate a compiler stage after eight name collisions")

    @property
    def candidate(self) -> Path:
        """Return the unpublished candidate path."""
        if self._root is None:
            raise RuntimeError("Compiler stage has not been created")
        return self._root / "candidate.aeg"

    @property
    def previous(self) -> Path:
        """Return the rollback location, never a recursive-delete target on failure."""
        if self._root is None:
            raise RuntimeError("Compiler stage has not been created")
        return self._root / "previous.aeg"

    def publish(self) -> None:
        """Promote accepted bytes; restore the previous directory if promotion fails."""
        if self._published:
            raise RuntimeError("Compiler candidate was already published")
        self.check_destination()
        try:
            if os.path.lexists(self.destination):
                publish_staged_directory(self.destination, self.previous)
            publish_staged_directory(self.candidate, self.destination)
        except BaseException as promotion_error:
            # A rename can finish before a Python signal handler raises. Use
            # actual recovery bytes, not a flag assigned after the OS call.
            if self.previous.exists():
                try:
                    publish_staged_directory(self.previous, self.destination)
                except OSError as rollback_error:
                    raise CompilationError(
                        "AEG promotion and rollback failed; the previous artifact is retained "
                        f"at {self.previous}",
                        stage="publication",
                        details={
                            "recovery_path": str(self.previous),
                            "promotion_error": str(promotion_error),
                            "rollback_error": str(rollback_error),
                        },
                    ) from rollback_error
            raise
        self._published = True

    def cleanup(self) -> None:
        """Remove owned staging, preserving any previous artifact needing recovery."""
        if self._root is None:
            return
        if self.previous.exists() and not self._published:
            logger.error("Previous AEG retained for recovery at %s", self.previous)
            return
        if self._root.parent != self.destination.parent or not self._root.name.startswith(
            ".aether-compile-"
        ):
            raise RuntimeError("Compiler cleanup path is outside its owned staging directory")
        try:
            shutil.rmtree(self._root)
        except OSError as exc:
            # Publication may already have succeeded. Keep the truthful outcome
            # and report the retained files instead of hiding the cleanup error.
            logger.warning("Compiler staging cleanup failed at %s: %s", self._root, exc)


@contextmanager
def compilation_publication(
    destination: str | Path,
    *,
    overwrite: bool,
    source_path: str | Path | None = None,
) -> Iterator[CompilationPublication]:
    """Hold a native nonblocking writer lock through packaging, gating and promotion.

    A stable lock file remains beside the artifact. Native locking is required;
    unsupported filesystem/platform locks do not silently become marker leases.
    """
    raw = Path(destination).expanduser().absolute()
    target = raw.parent.resolve() / raw.name
    publication = CompilationPublication(target, overwrite=overwrite)
    publication.check_destination()
    if source_path is not None and Path(source_path).exists():
        source = Path(source_path).resolve()
        if source == target or source.is_relative_to(target):
            raise ValueError("AEG output must not contain or replace its source checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(os.path.normcase(str(target)).encode("utf-8")).hexdigest()
    lock_path = target.parent / f".aether-publication-{identity}.lock"
    with _publication_lock(lock_path):
        publication.check_destination()
        publication.create_stage()
        try:
            yield publication
        finally:
            publication.cleanup()
