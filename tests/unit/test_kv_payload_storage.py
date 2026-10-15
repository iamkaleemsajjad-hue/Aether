"""Real byte round trips, fault atomicity and cache residency accounting."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import struct
import subprocess
import sys
import weakref

import numpy as np
import pytest

from aether.core.types import MemoryTier
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.kv_payload import CachePayload, TensorPayload
from aether.runtime.kv_storage import CacheQuotaError, KVPayloadStore


def snapshot(value=1, *, length=4):
    return CachePayload("model-weights-sha256", "test_attention_v1", length,
                        {"key": np.full((2, 4, 8), value, dtype=np.float32),
                         "position": np.arange(4, dtype=np.int64)},
                        {"layout": "heads,tokens,dim"})


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64, ">f4", np.complex64, np.bool_])
def test_tensor_snapshot_preserves_noncontiguous_dtype_and_bits(dtype):
    array = np.arange(60).reshape(3, 4, 5).astype(dtype)[:, ::2, ::-1]
    payload = CachePayload("model", "layout", 2, {"state": array})
    decoded = CachePayload.deserialize(payload.serialize(), expected_model_fingerprint="model", expected_cache_kind="layout")
    restored = decoded.array("state")
    assert restored.dtype.str == array.dtype.str
    assert restored.shape == array.shape
    assert restored.tobytes() == array.tobytes()
    assert restored.flags.writeable and restored.flags.owndata
    restored.flat[0] = 0
    assert decoded.tensors["state"].data == array.tobytes()


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_torch_cache_dtype_bytes_roundtrip(dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    source = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4).to(dtype).transpose(0, 1)
    payload = CachePayload("model", "torch_layout", 4, {"state": source})
    decoded = CachePayload.deserialize(payload.serialize())
    restored = decoded.tensors["state"].to_tensor()
    assert restored.dtype == dtype and tuple(restored.shape) == tuple(source.shape)
    assert torch.equal(restored.view(torch.uint8), source.contiguous().view(torch.uint8))
    if dtype_name == "bfloat16":
        with pytest.raises(ValueError, match="lossless NumPy"):
            decoded.array("state")


def test_framework_free_payload_import_does_not_import_torch():
    code = "import sys; from aether.runtime.kv_payload import CachePayload; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_identity_and_corruption_reject_before_restore():
    data = snapshot().serialize()
    with pytest.raises(ValueError, match="fingerprint"):
        CachePayload.deserialize(data, expected_model_fingerprint="other")
    with pytest.raises(ValueError, match="kind"):
        CachePayload.deserialize(data, expected_cache_kind="recurrent")
    damaged = bytearray(data)
    damaged[-40] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        CachePayload.deserialize(bytes(damaged))
    with pytest.raises(ValueError, match="oversized"):
        CachePayload.deserialize(data, max_bytes=len(data) - 1)


@pytest.mark.parametrize("change", ["overlap", "object", "shape", "duplicate", "trailing"])
def test_authenticated_but_invalid_tensor_descriptors_rejected(change):
    data = snapshot().serialize()
    size = struct.unpack("<Q", data[8:16])[0]
    header = json.loads(data[16:16 + size])
    raw = data[16 + size:-32]
    if change == "overlap":
        header["tensors"][1]["offset"] = 0
    elif change == "object":
        header["tensors"][0]["dtype"] = "object"
    elif change == "shape":
        header["tensors"][0]["shape"] = [1]
    elif change == "duplicate":
        header["tensors"][1]["name"] = header["tensors"][0]["name"]
    else:
        raw += b"undeclared"
    encoded = json.dumps(header).encode()
    message = data[:8] + struct.pack("<Q", len(encoded)) + encoded + raw
    with pytest.raises(ValueError):
        CachePayload.deserialize(message + hashlib.sha256(message).digest())


def test_store_restart_dedup_and_cleanup(tmp_path):
    payload = snapshot()
    encoded_size = len(payload.serialize())
    store = KVPayloadStore(tmp_path / "cache", max_bytes=encoded_size)
    first = store.put("opaque/key", payload, metadata={"request": 7})
    store.put("another", payload)
    assert store.stats()["physical_bytes"] == encoded_size
    assert store.stats()["referenced_tensor_bytes"] == payload.byte_size * 2
    restarted = KVPayloadStore(tmp_path / "cache", max_bytes=encoded_size)
    assert restarted.get("opaque/key", expected_model_fingerprint=payload.model_fingerprint).serialize() == payload.serialize()
    assert restarted.entries()[1]["metadata"] == {"request": 7}
    assert restarted.delete("opaque/key")
    assert (store.blobs_dir / f"{first['content_hash']}.blob").exists()
    assert restarted.delete("another")
    assert restarted.stats()["physical_bytes"] == 0
    assert not list(store.blobs_dir.iterdir())


def test_quota_failure_preserves_existing_entry_and_cleans_staging(tmp_path):
    original = snapshot()
    store = KVPayloadStore(tmp_path, max_bytes=len(original.serialize()))
    store.put("current", original)
    with pytest.raises(CacheQuotaError):
        store.put("different", snapshot(2))
    assert store.get("current").serialize() == original.serialize()
    assert store.get("different") is None
    assert len(list(store.blobs_dir.iterdir())) == 1


def test_deduplicated_keys_and_store_metadata_remain_bounded(tmp_path):
    store = KVPayloadStore(tmp_path, max_bytes=100000, max_entries=1)
    payload = snapshot()
    store.put("first", payload)
    with pytest.raises(CacheQuotaError, match="entry-count"):
        store.put("second", payload)
    with pytest.raises(ValueError, match="bounded JSON"):
        store.put("first", payload, metadata={"unbounded": "x" * 65536})
    assert len(store.entries()) == 1
    store.put("first", snapshot(2))
    assert store.get("first").array("key")[0, 0, 0] == 2


def test_database_publication_failure_rolls_back_blob_and_old_key(tmp_path):
    store = KVPayloadStore(tmp_path, max_bytes=100000)
    original = snapshot()
    store.put("current", original)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("CREATE TRIGGER reject_test BEFORE INSERT ON entries WHEN NEW.logical_length=999 "
                           "BEGIN SELECT RAISE(ABORT,'injected publication failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.put("current", snapshot(2, length=999))
    assert store.get("current").serialize() == original.serialize()
    assert len(list(store.blobs_dir.iterdir())) == 1


def test_interrupted_write_cleanup_and_restart_orphan_recovery(tmp_path, monkeypatch):
    store = KVPayloadStore(tmp_path, max_bytes=100000)
    payload = snapshot()
    store.put("current", payload)
    def interrupted(stream):
        stream.write(b"partial cache state")
        raise OSError("disk write interrupted")
    monkeypatch.setattr(payload, "write_to", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        store.put("new", payload)
    assert store.get("new") is None
    assert not list(store.blobs_dir.glob(".pending-*"))
    orphan = store.blobs_dir / f"{'0' * 64}.blob"
    orphan.write_bytes(b"uncommitted")
    (store.blobs_dir / ".pending-crash").write_bytes(b"partial")
    recovered = KVPayloadStore(tmp_path, max_bytes=100000)
    assert len(recovered.entries()) == 1 and not orphan.exists()
    assert not list(store.blobs_dir.glob(".pending-*"))


def test_disk_corruption_is_never_reloaded_or_marked_resident(tmp_path):
    store = KVPayloadStore(tmp_path, max_bytes=100000)
    manager = KVCacheManager(payload_store=store)
    block = manager.import_payload(snapshot(), prefix_hash="prefix")
    assert manager.swap(block.block_id, MemoryTier.L3_NVME)
    row = store.entries()[0]
    path = store.blobs_dir / f"{row['content_hash']}.blob"
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum"):
        manager.reload(block.block_id)
    assert block.tier == MemoryTier.L3_NVME
    assert manager.get_transfer_stats()["resident_tensor_bytes"] == 0


def test_parallel_writers_deduplicate_one_actual_blob(tmp_path):
    payload = snapshot()
    store = KVPayloadStore(tmp_path, max_bytes=len(payload.serialize()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda key: store.put(str(key), payload), range(12)))
    assert len({row["content_hash"] for row in rows}) == 1
    assert store.stats()["entries"] == 12
    assert len(list(store.blobs_dir.iterdir())) == 1


def test_manager_real_swap_restart_reload_remove_and_descriptor_scope(tmp_path):
    store = KVPayloadStore(tmp_path, max_bytes=100000)
    payload = snapshot()
    manager = KVCacheManager(payload_store=store)
    block = manager.import_payload(payload, prefix_hash="shared-prefix")
    assert block.tier == MemoryTier.L2_CPU_DRAM
    assert manager.swap(block.block_id, MemoryTier.L3_NVME)
    stats = manager.get_transfer_stats()
    assert stats["payload_transfers"] == 1 and stats["transferred_tensor_bytes"] == payload.byte_size
    assert stats["transferred_serialized_bytes"] == len(payload.serialize())
    assert stats["resident_tensor_bytes"] == 0 and stats["local_tier_transfers"] == 0
    restarted = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000))
    found = restarted.find_prefix("shared-prefix")
    assert found is not None and found.tier == MemoryTier.L3_NVME
    restored = restarted.export_payload(found.block_id, expected_model_fingerprint=payload.model_fingerprint)
    assert restored.serialize() == payload.serialize() and found.tier == MemoryTier.L2_CPU_DRAM
    assert restarted.remove(found.block_id)
    assert restarted.find_prefix("shared-prefix") is None and store.stats()["entries"] == 0
    descriptor = restarted.allocate_block(0, 0, 16)
    restarted.evict_to_tier(str(descriptor.block_id), MemoryTier.L2_CPU_DRAM)
    assert restarted.get_transfer_stats()["legacy_transfer_scope"] == "descriptor_transitions"
    with pytest.raises(ValueError, match="descriptor-only"):
        restarted.export_payload(descriptor.block_id)


def test_failed_swap_retains_host_payload_and_tier(tmp_path):
    payload = snapshot()
    manager = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=1))
    block = manager.import_payload(payload)
    with pytest.raises(CacheQuotaError):
        manager.swap(block.block_id, MemoryTier.L3_NVME)
    assert block.tier == MemoryTier.L2_CPU_DRAM
    assert manager.export_payload(block.block_id).serialize() == payload.serialize()
    assert manager.get_transfer_stats()["payload_transfers"] == 0
    with pytest.raises(ValueError, match="execution adapter"):
        manager.swap(block.block_id, MemoryTier.L1_GPU_HBM)


@pytest.mark.parametrize("family", ["attention", "hybrid"])
def test_actual_prefill_disk_restart_reload_decode_matches_uninterrupted(tmp_path, family):
    from aether.runtime.cache_handoff import export_cpu_cache, restore_cpu_cache
    if family == "attention":
        from test_aeg_onnx_export import make_engine
        source, target = make_engine(), make_engine()
    else:
        from test_hybrid_rope_scaling import _reference
        source, target = _reference(None), _reference(None)
    _, cache = source.forward(np.asarray([1, 3, 5], dtype=np.int64))
    payload = export_cpu_cache(cache, model_fingerprint=f"actual-{family}-weights")
    manager = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=1024**2))
    block = manager.import_payload(payload, prefix_hash="actual-prefill-prefix")
    key = manager.payload_key(block.block_id)
    manager.swap(block.block_id, MemoryTier.L3_NVME)
    restarted = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=1024**2))
    recovered = restarted.block_for_payload_key(key)
    assert recovered is not None
    loaded = restarted.export_payload(recovered.block_id, expected_model_fingerprint=payload.model_fingerprint)
    resumed = restore_cpu_cache(loaded, expected_model_fingerprint=payload.model_fingerprint)
    expected, expected_cache = source.forward(np.asarray([2, 4], dtype=np.int64), cache)
    actual, actual_cache = target.forward(np.asarray([2, 4], dtype=np.int64), resumed)
    np.testing.assert_array_equal(actual, expected)
    assert actual_cache.length == expected_cache.length == 5


def test_failed_remote_swap_from_disk_preserves_committed_source(tmp_path):
    from aether.core.exceptions import HubError
    class Offline:
        def put_kv(self, *args):
            raise HubError("offline")
    manager = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000),
                             hub_client=Offline(), hub_namespace="alice")
    payload = snapshot()
    block = manager.import_payload(payload)
    manager.swap(block.block_id, MemoryTier.L3_NVME)
    with pytest.raises(HubError, match="offline"):
        manager.swap(block.block_id, MemoryTier.L4_HUB)
    assert block.tier == MemoryTier.L3_NVME
    assert manager.get_transfer_stats()["resident_tensor_bytes"] == 0
    assert manager.payload_store.get(manager.payload_key(block.block_id)).serialize() == payload.serialize()


def test_manager_attachment_does_not_share_mutable_payload_metadata():
    payload = snapshot()
    manager = KVCacheManager()
    block = manager.import_payload(payload)
    payload.metadata["layout"] = "wrong"
    payload.model_fingerprint = "different-model"
    restored = manager.export_payload(block.block_id, expected_model_fingerprint="model-weights-sha256")
    assert restored.metadata["layout"] == "heads,tokens,dim"


def test_release_residency_requires_verified_unchanged_backing_and_no_rewrite(tmp_path, monkeypatch):
    manager = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000))
    block = manager.import_payload(snapshot())
    with pytest.raises(ValueError, match="uncommitted"):
        manager.release_payload_residency(block.block_id)
    manager.swap(block.block_id, MemoryTier.L3_NVME)
    payload = manager.reload(block.block_id)
    def no_write(*args, **kwargs):
        raise AssertionError("residency release must not rewrite committed bytes")
    monkeypatch.setattr(manager.payload_store, "put", no_write)
    assert manager.release_payload_residency(block.block_id)
    assert block.tier == MemoryTier.L3_NVME
    assert manager.get_transfer_stats()["resident_tensor_bytes"] == 0
    assert manager.reload(block.block_id).serialize() == payload.serialize()
    modified = manager.export_payload(block.block_id)
    modified.metadata["layout"] = "different"
    with pytest.raises(ValueError, match="modified"):
        manager.release_payload_residency(block.block_id)
    assert block.tier == MemoryTier.L2_CPU_DRAM


def test_terminal_discard_releases_owned_bytes_without_storage_operations(tmp_path, monkeypatch):
    store = KVPayloadStore(tmp_path, max_bytes=100000)
    manager = KVCacheManager(payload_store=store)
    original = snapshot()
    block = manager.import_payload(original, prefix_hash="committed-prefix")
    key = manager.payload_key(block.block_id)
    manager.swap(block.block_id, MemoryTier.L3_NVME)
    resident = manager.reload(block.block_id)
    reference = weakref.ref(resident)
    del resident
    uncommitted = manager.import_payload(snapshot(2), prefix_hash="uncommitted-prefix")
    uncommitted_key = manager.payload_key(uncommitted.block_id)
    before = store.stats()

    def forbidden_io(*args, **kwargs):
        raise AssertionError("terminal payload cleanup must perform no storage I/O")

    monkeypatch.setattr(store, "put", forbidden_io)
    monkeypatch.setattr(store, "get", forbidden_io)
    monkeypatch.setattr(store, "delete", forbidden_io)
    assert manager.discard_resident_payloads() == original.byte_size * 2
    assert reference() is None  # The retained manager no longer owns this snapshot.
    assert manager.get_transfer_stats()["resident_tensor_bytes"] == 0
    assert block.tier == MemoryTier.L3_NVME
    assert manager.payload_key(block.block_id) == key
    assert manager.payload_key(uncommitted.block_id) == uncommitted_key
    assert manager.discard_resident_payloads() == 0
    assert store.stats() == before

    restarted = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000))
    recovered = restarted.block_for_payload_key(key)
    assert recovered is not None
    assert restarted.export_payload(recovered.block_id).serialize() == original.serialize()
    assert restarted.block_for_payload_key(uncommitted_key) is None


def test_terminal_discard_preserves_explicit_hub_backing_without_network():
    class Hub:
        def get_kv(self, *args, **kwargs):
            return snapshot()

        last_operation = {"downloaded_bytes": len(snapshot().serialize())}

    manager = KVCacheManager(hub_client=Hub(), hub_namespace="alice")
    block = manager.import_from_hub("stored-key", expected_model_fingerprint="model-weights-sha256")
    reference = weakref.ref(manager.export_payload(block.block_id))

    def forbidden_network(*args, **kwargs):
        raise AssertionError("terminal cleanup must not upload, download or delete remote bytes")

    manager.hub_client.get_kv = forbidden_network
    manager.hub_client.put_kv = forbidden_network
    manager.hub_client.delete_kv = forbidden_network
    assert manager.discard_resident_payloads() == snapshot().byte_size
    assert reference() is None
    assert block.tier == MemoryTier.L4_HUB
    assert manager.block_for_payload_key("stored-key") is block
    assert manager.payload_key(block.block_id) == "stored-key"
    assert "stored-key" in manager._hub_keys


def test_closed_retained_runtime_releases_payloads_and_preserves_disk_restart(tmp_path):
    from aether import Runtime, RuntimeConfig

    root = tmp_path / "persistent-state"
    runtime = Runtime(RuntimeConfig(
        backend_name="aether_cpu", execution_devices=["cpu"],
        enable_semantic_cache=False, speculative_decoding=False,
        model_cache_dir=str(tmp_path / "models"), kv_cache_path=str(root),
    ))
    payload = snapshot()
    manager = runtime.kv_cache
    block = manager.import_payload(payload)
    key = manager.payload_key(block.block_id)
    manager.swap(block.block_id, MemoryTier.L3_NVME)
    resident = manager.reload(block.block_id)
    reference = weakref.ref(resident)
    del resident
    assert manager.get_transfer_stats()["resident_tensor_bytes"] == payload.byte_size
    try:
        runtime.close()
        assert reference() is None  # Retaining runtime/manager must not retain tensor bytes.
        assert runtime.kv_cache is manager
        assert manager.get_transfer_stats()["resident_tensor_bytes"] == 0
        restarted = KVCacheManager(payload_store=KVPayloadStore(root, max_bytes=100000))
        recovered = restarted.block_for_payload_key(key)
        assert recovered is not None
        assert restarted.export_payload(recovered.block_id).serialize() == payload.serialize()
    finally:
        runtime.close()


def test_restart_preserves_keys_from_managers_with_colliding_local_block_ids(tmp_path):
    first = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000))
    second = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000))
    a = first.import_payload(snapshot(1))
    b = second.import_payload(snapshot(2))
    assert a.block_id == b.block_id == 0
    keys = first.payload_key(a.block_id), second.payload_key(b.block_id)
    assert keys[0] != keys[1]
    first.swap(a.block_id, MemoryTier.L3_NVME)
    second.swap(b.block_id, MemoryTier.L3_NVME)
    for _ in range(2):
        restarted = KVCacheManager(payload_store=KVPayloadStore(tmp_path, max_bytes=100000))
        assert restarted.block_count == 2
        restored_blocks = [restarted.block_for_payload_key(key) for key in keys]
        assert restored_blocks[0].block_id != restored_blocks[1].block_id
        for expected, block in enumerate(restored_blocks, 1):
            state = restarted.export_payload(block.block_id)
            assert state.array("key")[0, 0, 0] == expected
            # Re-publishing a recovered descriptor must retain its UUID key.
            restarted.swap(block.block_id, MemoryTier.L3_NVME)


def test_linked_storage_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode/privilege")
    with pytest.raises(ValueError, match="linked/reparse"):
        KVPayloadStore(link, max_bytes=100000)
