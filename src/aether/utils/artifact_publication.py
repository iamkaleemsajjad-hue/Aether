"""Atomic publication of verified staged directories with bounded Windows retries.

Windows can deny a directory rename while a member has an open handle. See
https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
and the FILE_SHARE_DELETE contract in CreateFileW. This helper never copies
partial output, changes permissions, or infers which process holds a handle.

On POSIX the destination precheck retains Path.rename semantics: it is not an
atomic RENAME_NOREPLACE guarantee against concurrent creation of an empty
destination directory. Callers must own their publication destination.
"""

from __future__ import annotations

import errno
import os
import sys
import time
from pathlib import Path

_WINDOWS = sys.platform == "win32"
_RETRYABLE_WINERRORS = frozenset({5, 32, 33})
_WINDOWS_RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.5, 0.5)


def publish_staged_directory(source: str | Path, destination: str | Path) -> Path:
    """Rename a verified stage, with at most 1.63s of Windows error backoff.

    The ordinary path attempts one immediate rename and never sleeps. Existing
    destinations, including dangling links, are rejected before every attempt.
    Windows also rejects a destination created concurrently with its rename.
    Other platforms and other error codes fail immediately. Persistent errors
    preserve the concrete OSError with the initial failure as their cause.
    """
    source, destination = Path(source), Path(destination)
    first_error: OSError | None = None
    retries = 0
    while True:
        if os.path.lexists(destination):
            raise FileExistsError(
                errno.EEXIST, "Artifact publication destination already exists", str(destination)
            ) from first_error
        try:
            return source.rename(destination)
        except OSError as exc:
            if not _WINDOWS or getattr(exc, "winerror", None) not in _RETRYABLE_WINERRORS:
                if first_error is not None:
                    raise exc from first_error
                raise
            if first_error is None:
                first_error = exc
            if retries >= len(_WINDOWS_RETRY_DELAYS):
                if exc is not first_error:
                    raise exc from first_error
                raise
            time.sleep(_WINDOWS_RETRY_DELAYS[retries])
            retries += 1
