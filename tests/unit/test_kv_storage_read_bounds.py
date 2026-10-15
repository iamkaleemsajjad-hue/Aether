"""Committed blob sizes bound actual reads independently of configured quotas."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from aether.runtime.kv_payload import MAX_PAYLOAD_BYTES, CachePayload
from aether.runtime.kv_storage import CacheQuotaError, KVPayloadStore


def _stored(tmp_path):
    payload = CachePayload("model", "test_layout", 1, {"key": np.arange(4, dtype=np.float32)})
    store = KVPayloadStore(tmp_path, max_bytes=MAX_PAYLOAD_BYTES)
    row = store.put("key", payload)
    return store, row, payload.serialize()


def _guard_blob_reads(monkeypatch, blob, committed_size, before_read=None):
    """Intercept the requested size before Python can allocate a quota-sized buffer."""
    original_open = Path.open
    requested = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            requested.append(size)
            assert 0 <= size <= committed_size + 1, (
                f"read requested {size} bytes for {committed_size} committed bytes"
            )
            if before_read is not None:
                before_read(original_open)
            return self.stream.read(size)

    @contextmanager
    def guarded_open(path, mode="r", *args, **kwargs):
        with original_open(path, mode, *args, **kwargs) as stream:
            yield BoundedReader(stream) if path == blob and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    return requested


def test_tiny_blob_read_uses_committed_size_not_two_gib_quota(tmp_path, monkeypatch):
    """Reproduce the former 2GiB read request without performing that allocation."""
    store, row, encoded = _stored(tmp_path)
    assert store.max_entry_bytes == MAX_PAYLOAD_BYTES
    blob = store.blobs_dir / f"{row['content_hash']}.blob"
    requested = _guard_blob_reads(monkeypatch, blob, row["size_bytes"])
    assert store.get_bytes("key") == encoded
    assert requested == [row["size_bytes"] + 1]


def test_duplicate_write_reads_existing_blob_with_payload_sized_bound(tmp_path, monkeypatch):
    """Deduplication verifies existing bytes without an unbounded read."""
    store, row, encoded = _stored(tmp_path)
    blob = store.blobs_dir / f"{row['content_hash']}.blob"
    requested = _guard_blob_reads(monkeypatch, blob, row["size_bytes"])
    repeated = store.put("duplicate", CachePayload.deserialize(encoded))
    assert repeated["content_hash"] == row["content_hash"]
    assert requested == [row["size_bytes"] + 1]


def test_duplicate_write_rejects_corrupt_existing_blob(tmp_path, monkeypatch):
    """A deduplication hit cannot reuse same-size data with an invalid hash."""
    store, row, encoded = _stored(tmp_path)
    blob = store.blobs_dir / f"{row['content_hash']}.blob"
    blob.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    requested = _guard_blob_reads(monkeypatch, blob, row["size_bytes"])
    with pytest.raises(ValueError, match="checksum"):
        store.put("duplicate", CachePayload.deserialize(encoded))
    assert requested == [row["size_bytes"] + 1]
    assert [entry["cache_key"] for entry in store.entries()] == ["key"]


@pytest.mark.parametrize("damage", ["truncate", "append", "same_size_corruption"])
def test_blob_size_and_checksum_corruption_still_fail_bounded_reads(tmp_path, monkeypatch, damage):
    """Both framing length and equal-size corruption remain integrity errors."""
    store, row, encoded = _stored(tmp_path)
    blob = store.blobs_dir / f"{row['content_hash']}.blob"
    damaged = encoded[:-1] if damage == "truncate" else encoded + b"x"
    if damage == "same_size_corruption":
        damaged = encoded[:-1] + bytes([encoded[-1] ^ 1])
    blob.write_bytes(damaged)
    _guard_blob_reads(monkeypatch, blob, row["size_bytes"])
    with pytest.raises(ValueError, match="checksum"):
        store.get_bytes("key")


@pytest.mark.parametrize("declared", [-1, "invalid", 1.5, MAX_PAYLOAD_BYTES])
def test_invalid_or_truncated_committed_size_fails_before_allocating(
    tmp_path, monkeypatch, declared
):
    """A corrupt database size cannot request an unbounded or oversized read."""
    store, row, _ = _stored(tmp_path)
    with store._connect() as connection:
        connection.execute("UPDATE entries SET size_bytes=? WHERE cache_key='key'", (declared,))
    blob = store.blobs_dir / f"{row['content_hash']}.blob"
    requested = _guard_blob_reads(monkeypatch, blob, row["size_bytes"])
    with pytest.raises(ValueError):
        store.get_bytes("key")
    assert requested == []


def test_committed_size_above_quota_is_rejected_before_read(tmp_path, monkeypatch):
    """The smaller read request does not relax the configured entry quota."""
    store, row, _ = _stored(tmp_path)
    with store._connect() as connection:
        connection.execute(
            "UPDATE entries SET size_bytes=? WHERE cache_key='key'", (MAX_PAYLOAD_BYTES + 1,)
        )
    blob = store.blobs_dir / f"{row['content_hash']}.blob"
    requested = _guard_blob_reads(monkeypatch, blob, row["size_bytes"])
    with pytest.raises(CacheQuotaError, match="read bound"):
        store.get_bytes("key")
    assert requested == []


def test_one_byte_guard_detects_growth_between_stat_and_read(tmp_path, monkeypatch):
    """Trailing bytes added after the size check must still be rejected."""
    store, row, _ = _stored(tmp_path)
    blob = store.blobs_dir / f"{row['content_hash']}.blob"

    def append_after_stat(original_open):
        with original_open(blob, "ab") as stream:
            stream.write(b"x")

    requested = _guard_blob_reads(monkeypatch, blob, row["size_bytes"], append_after_stat)
    with pytest.raises(ValueError, match="checksum"):
        store.get_bytes("key")
    assert requested == [row["size_bytes"] + 1]
