"""Owned file-backed pool/thread shutdown; no physical CXL hardware claim."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from aether.runtime.r12_cxl_kv_pool import CXLRackScaleKVPool


def test_real_mapped_pool_close_wakes_long_interval_worker_and_releases_file(tmp_path):
    """Real mmap bytes and the default sleeping worker are both owned by close()."""
    path = tmp_path / "pool.bin"
    pool = CXLRackScaleKVPool(pool_size_gb=0.002, emulated_path=str(path))
    worker = pool._defrag_thread
    mapping, file = pool._backend._mmap, pool._backend._file
    try:
        assert pool._backend.mode == "emulated"
        assert worker is not None and worker.is_alive()
        payload = b"k" * 2048
        assert pool.store_kv("actual-session", [0], payload, seq_len=1) is not None
        assert pool.load_kv("actual-session", [0]) == payload
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(pool.close).result(timeout=5)
        assert not worker.is_alive()
        assert mapping.closed and file.closed
        assert pool._backend._mmap is None and pool._backend._file is None
        assert pool.get_stats()["closed"] is True
        pool.close()
        path.unlink()
        assert not path.exists()
    finally:
        pool.close()


def test_real_mapped_pool_close_waits_active_callback_before_closing_backend(tmp_path, monkeypatch):
    """Controlled worker pacing still performs actual mmap reads before shutdown."""
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    pool = CXLRackScaleKVPool(
        pool_size_gb=0.008,
        emulated_path=str(tmp_path / "active.bin"),
        enable_defrag_daemon=False,
    )
    payload = b"v" * (3400 * 2048)
    block = pool.store_kv("active", [0], payload, seq_len=3400)
    assert block is not None and pool._stats.utilization > 0.80
    observed = []

    def active_callback():
        observed.append(pool._backend.read(block.cxl_offset, 2048))
        entered.set()
        assert release.wait(5)
        observed.append(pool._backend.read(block.cxl_offset, 2048))

    def close_owner():
        pool.close()
        closed.set()

    monkeypatch.setattr(pool, "defragment", active_callback)
    pool._start_defrag_daemon(0.01)
    closer = threading.Thread(target=close_owner)
    try:
        assert entered.wait(5)
        closer.start()
        assert not closed.wait(0.05)
        assert pool._backend._mmap is not None and not pool._backend._mmap.closed
        release.set()
        closer.join(timeout=5)
        assert not closer.is_alive() and closed.is_set()
        assert observed == [payload[:2048], payload[:2048]]
        assert not pool._defrag_thread.is_alive()
        assert pool._backend._mmap is None
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=5)
        pool.close()


def test_pool_concurrent_close_is_idempotent_and_disabled_daemon_closes(tmp_path):
    """Concurrent owners can close one real backend exactly once without deadlock."""
    pool = CXLRackScaleKVPool(
        pool_size_gb=0.002,
        emulated_path=str(tmp_path / "concurrent.bin"),
        enable_defrag_daemon=False,
    )
    assert pool._defrag_thread is None
    file = pool._backend._file
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pool.close) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert file.closed and pool._backend._mmap is None
    assert pool.get_stats()["closed"] is True


def test_closed_pool_rejects_storage_and_defragmentation(tmp_path):
    """Closed mappings cannot be treated as successful pool reads or writes."""
    pool = CXLRackScaleKVPool(
        pool_size_gb=0.002,
        emulated_path=str(tmp_path / "closed.bin"),
        enable_defrag_daemon=False,
    )
    pool.close()
    operations = [
        lambda: pool.store_kv("session", [0], b"k" * 2048),
        lambda: pool.load_kv("session", [0]),
        lambda: pool.migrate_session("source", "destination", "session"),
        lambda: pool.release_session("session"),
        pool.defragment,
    ]
    for operation in operations:
        with pytest.raises(RuntimeError, match="CXL pool is closed"):
            operation()


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_daemon_interval_fails_before_opening_backing_file(tmp_path, interval):
    """Reject worker intervals before allocating a mapped pool resource."""
    path = tmp_path / "invalid.bin"
    with pytest.raises(ValueError, match="finite and positive"):
        CXLRackScaleKVPool(pool_size_gb=0.002, emulated_path=str(path), defrag_interval_s=interval)
    assert not path.exists()
