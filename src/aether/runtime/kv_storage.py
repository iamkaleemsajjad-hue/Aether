"""Transactional, bounded local-disk storage for real typed cache snapshots.

The directory may reside on NVMe, SATA, or another filesystem; this module
does not infer the physical medium or claim GPU-direct/network execution.
SQLite serializes writers and records committed references to immutable,
checksummed blobs. A failed write never replaces a previously committed key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aether.runtime.kv_payload import CachePayload, MAX_PAYLOAD_BYTES


class CacheQuotaError(ValueError):
    """A cache operation would exceed the configured storage bound."""


def _safe_path(path: Path, *, directory: bool = False) -> None:
    """Reject symlinks, Windows junctions/reparse points and special files."""
    if not path.exists() and not path.is_symlink():
        return
    attributes = path.lstat()
    if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Cache storage cannot use linked/reparse paths: {path}")
    if directory and not stat.S_ISDIR(attributes.st_mode):
        raise ValueError(f"Cache storage directory is not a directory: {path}")
    if not directory and not stat.S_ISREG(attributes.st_mode):
        raise ValueError(f"Cache storage requires regular files: {path}")


class KVPayloadStore:
    """Persistent opaque-key cache snapshots with a physical blob byte quota.

    ``put`` accepts an adapter snapshot, ``get`` checks destination identity,
    ``delete`` removes unreferenced bytes, and ``entries`` survives restarts.
    Identical bytes share one blob. Quotas count serialized blobs, including
    their framing, not an estimate from token count or head geometry.
    """

    def __init__(self, root: str | Path, *, max_bytes: int,
                 max_entry_bytes: int = MAX_PAYLOAD_BYTES, max_entries: int = 65536) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if type(max_entry_bytes) is not int or max_entry_bytes <= 0:
            raise ValueError("max_entry_bytes must be a positive integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        supplied = Path(root).absolute()
        for component in (supplied, *supplied.parents):
            _safe_path(component, directory=True)
        supplied.mkdir(parents=True, exist_ok=True)
        self.root = supplied.resolve()
        self.blobs_dir = self.root / "payloads"
        _safe_path(self.blobs_dir, directory=True)
        self.blobs_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "cache.sqlite3"
        _safe_path(self.db_path)
        self.max_bytes = max_bytes
        self.max_entry_bytes = min(max_entry_bytes, MAX_PAYLOAD_BYTES)
        self.max_entries = max_entries
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS entries (cache_key TEXT PRIMARY KEY, "
                               "content_hash TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
                               "tensor_bytes INTEGER NOT NULL, model_fingerprint TEXT NOT NULL, "
                               "cache_kind TEXT NOT NULL, logical_length INTEGER NOT NULL, "
                               "metadata TEXT NOT NULL, created_at REAL NOT NULL, accessed_at REAL NOT NULL)")
        self.cleanup()

    @contextmanager
    def _connect(self):
        _safe_path(self.db_path)
        _safe_path(self.root, directory=True)
        _safe_path(self.blobs_dir, directory=True)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _key(key: str) -> str:
        if not isinstance(key, str) or not key or len(key) > 1024 or "\0" in key:
            raise ValueError("Cache key must be nonempty bounded text")
        return key

    def _blob(self, content_hash: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
            raise ValueError("Invalid cache blob content hash")
        path = self.blobs_dir / f"{content_hash}.blob"
        _safe_path(path)
        return path

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        result["storage_scope"] = "local_disk"
        return result

    def put(self, key: str, payload: CachePayload, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Atomically publish a snapshot; reject quota/IO failures explicitly."""
        if not isinstance(payload, CachePayload):
            raise TypeError("put requires an actual CachePayload")
        return self._put(key, payload, metadata=metadata)

    def put_bytes(self, key: str, data: bytes, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and persist an already framed snapshot (e.g. HTTP upload)."""
        payload = CachePayload.deserialize(data, max_bytes=self.max_entry_bytes)
        return self._put(key, payload, metadata=metadata, encoded=data)

    def _put(self, key: str, payload: CachePayload, *, metadata: dict[str, Any] | None,
             encoded: bytes | None = None) -> dict[str, Any]:
        self._key(key)
        metadata_json = json.dumps(metadata or {}, allow_nan=False, sort_keys=True)
        if not isinstance(metadata or {}, dict) or len(metadata_json.encode("utf-8")) > 64 * 1024:
            raise ValueError("Store metadata must be a bounded JSON object")
        staged: Path | None = None
        new_blob: Path | None = None
        committed = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                descriptor, filename = tempfile.mkstemp(prefix=".pending-", dir=self.blobs_dir)
                staged = Path(filename)
                with os.fdopen(descriptor, "wb") as stream:
                    size = payload.write_to(stream) if encoded is None else stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                if size > self.max_entry_bytes:
                    raise CacheQuotaError("Cache entry exceeds max_entry_bytes")
                digest = hashlib.sha256()
                with staged.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024**2), b""):
                        digest.update(chunk)
                content_hash = digest.hexdigest()
                destination = self._blob(content_hash)
                rows = connection.execute("SELECT cache_key,content_hash,size_bytes FROM entries").fetchall()
                if len(rows) >= self.max_entries and not any(row["cache_key"] == key for row in rows):
                    raise CacheQuotaError("Cache entry-count quota would be exceeded")
                retained = {row["content_hash"]: row["size_bytes"] for row in rows if row["cache_key"] != key}
                retained[content_hash] = size
                if sum(retained.values()) > self.max_bytes:
                    raise CacheQuotaError("Cache storage quota would be exceeded")
                if destination.exists():
                    existing = destination.read_bytes()
                    if len(existing) != size or hashlib.sha256(existing).hexdigest() != content_hash:
                        raise ValueError("Existing cache blob checksum mismatch")
                else:
                    os.replace(staged, destination)
                    staged = None
                    new_blob = destination
                now = time.time()
                connection.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)",
                                   (key, content_hash, size, payload.byte_size, payload.model_fingerprint,
                                    payload.cache_kind, payload.logical_length, metadata_json, now, now))
                connection.commit()
                committed = True
                # Publication succeeded. Old unreferenced blobs can now be removed.
                connection.execute("BEGIN IMMEDIATE")
                self._remove_orphans(connection)
                row = connection.execute("SELECT * FROM entries WHERE cache_key=?", (key,)).fetchone()
                return self._row(row)
            finally:
                if not committed:
                    connection.rollback()
                    if new_blob is not None:
                        new_blob.unlink(missing_ok=True)
                if staged is not None:
                    staged.unlink(missing_ok=True)

    def get_bytes(self, key: str) -> bytes | None:
        """Read and verify committed bytes while preventing concurrent deletion."""
        self._key(key)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM entries WHERE cache_key=?", (key,)).fetchone()
            if row is None:
                return None
            if row["size_bytes"] > self.max_entry_bytes:
                raise CacheQuotaError("Stored cache entry exceeds the configured read bound")
            path = self._blob(row["content_hash"])
            if not path.exists():
                raise ValueError("Committed cache payload is missing from disk")
            with path.open("rb") as stream:
                data = stream.read(self.max_entry_bytes + 1)
            if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["content_hash"]:
                raise ValueError("Stored cache payload checksum mismatch")
            payload = CachePayload.deserialize(data, expected_model_fingerprint=row["model_fingerprint"],
                                               expected_cache_kind=row["cache_kind"], max_bytes=self.max_entry_bytes)
            if payload.byte_size != row["tensor_bytes"] or payload.logical_length != row["logical_length"]:
                raise ValueError("Stored cache metadata disagrees with its payload")
            connection.execute("UPDATE entries SET accessed_at=? WHERE cache_key=?", (time.time(), key))
            return data

    def get(self, key: str, *, expected_model_fingerprint: str | None = None,
            expected_cache_kind: str | None = None) -> CachePayload | None:
        data = self.get_bytes(key)
        if data is None:
            return None
        return CachePayload.deserialize(data, expected_model_fingerprint=expected_model_fingerprint,
                                        expected_cache_kind=expected_cache_kind, max_bytes=self.max_entry_bytes)

    def entries(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [self._row(row) for row in connection.execute("SELECT * FROM entries ORDER BY cache_key")]

    def delete(self, key: str) -> bool:
        self._key(key)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute("DELETE FROM entries WHERE cache_key=?", (key,)).rowcount > 0
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            self._remove_orphans(connection)
            return deleted

    def _remove_orphans(self, connection: sqlite3.Connection) -> int:
        retained = {row[0] for row in connection.execute("SELECT DISTINCT content_hash FROM entries")}
        removed = 0
        for path in self.blobs_dir.iterdir():
            _safe_path(path)
            match = re.fullmatch(r"([0-9a-f]{64})\.blob", path.name)
            if (match is not None and match.group(1) not in retained) or path.name.startswith(".pending-"):
                path.unlink()
                removed += 1
        return removed

    def cleanup(self) -> int:
        """Recover uncommitted temporary/orphan blobs after a process restart."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._remove_orphans(connection)

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            entries = connection.execute("SELECT COUNT(*), COALESCE(SUM(tensor_bytes),0) FROM entries").fetchone()
            physical = connection.execute("SELECT COALESCE(SUM(size_bytes),0) FROM "
                                          "(SELECT content_hash,MAX(size_bytes) AS size_bytes FROM entries GROUP BY content_hash)").fetchone()[0]
        return {"entries": entries[0], "physical_bytes": physical, "referenced_tensor_bytes": entries[1],
                "max_bytes": self.max_bytes, "max_entry_bytes": self.max_entry_bytes,
                "max_entries": self.max_entries,
                "storage_scope": "local_disk", "physical_medium_verified": False}
