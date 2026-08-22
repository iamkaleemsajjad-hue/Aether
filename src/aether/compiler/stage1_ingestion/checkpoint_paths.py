"""Safe, portable resolution of checkpoint index shard paths.

Hugging Face checkpoint indexes store shard names relative to the checkpoint
directory.  They are not paths relative to the process working directory and
must not be resolved as such.  This helper is shared by the SafeTensors and
PyTorch loaders so both formats obey the same portability and safety contract.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_checkpoint_shard(checkpoint_dir: str | Path, shard_name: object) -> Path:
    """Resolve one index ``weight_map`` value to an existing checkpoint file.

    Index values are checkpoint-relative POSIX names in the Hugging Face
    format.  Accepting both slash styles makes locally materialized indexes
    portable between Windows and Linux.  The returned path intentionally does
    not call ``resolve()``: Hugging Face's cache stores snapshot files as
    symlinks into its sibling ``blobs`` directory, and following those links
    would incorrectly make valid files appear to escape the snapshot.

    Absolute paths, drive-qualified Windows paths, NUL bytes, empty names, and
    parent traversal are rejected before filesystem access.  A missing or
    non-file target is also rejected so an index cannot silently produce a
    graph-only compilation.
    """
    if not isinstance(shard_name, str) or not shard_name.strip():
        raise ValueError(f"invalid shard path {shard_name!r}")
    if "\x00" in shard_name:
        raise ValueError(f"invalid shard path {shard_name!r}: NUL byte")

    # SafeTensors indexes use POSIX separators even on Windows.  Treat
    # backslashes as separators too, including when a Linux process receives
    # an index authored on Windows.  PureWindowsPath catches drive and UNC
    # forms that pathlib.Path on Linux would otherwise treat as plain names.
    portable_name = shard_name.replace("\\", "/")
    posix_name = PurePosixPath(portable_name)
    windows_name = PureWindowsPath(shard_name)
    if (
        posix_name.is_absolute()
        or windows_name.is_absolute()
        or bool(windows_name.drive)
        or windows_name.root in {"\\", "/"}
    ):
        raise ValueError(f"unsafe shard path {shard_name!r}: absolute paths are not allowed")

    parts = tuple(part for part in posix_name.parts if part not in {"", "."})
    if not parts or ".." in parts:
        raise ValueError(
            f"shard path escapes checkpoint directory: {shard_name!r}"
        )

    root = Path(checkpoint_dir).absolute()
    shard = root.joinpath(*parts)
    if not shard.is_file():
        raise ValueError(f"shard file not found: {shard}")
    return shard
