"""Actual localhost tensor upload/download with scoped access and persistence."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import socket
import threading
import time

import numpy as np
import pytest

from aether.core.exceptions import AuthenticationError, HubError
from aether.core.types import MemoryTier
from aether.hub.client import HubClient
from aether.hub.server import AetherHubServer, create_hub_app
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.kv_payload import CachePayload


@contextmanager
def running_hub(hub, **kwargs):
    uvicorn = pytest.importorskip("uvicorn")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_hub_app(hub, **kwargs), host="127.0.0.1",
                            port=sock.getsockname()[1], log_level="critical", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    try:
        while not server.started and time.monotonic() < deadline:
            assert thread.is_alive()
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive()


def payload(value=1):
    return CachePayload("weight+graph-fingerprint", "attention_kv_v1", 4,
                        {"keys": np.full((4, 2, 8), value, dtype=np.float32),
                         "values": np.arange(64, dtype=np.float16).reshape(4, 2, 8)},
                        {"positions": [0, 1, 2, 3], "layers": 1})


def test_actual_hub_bytes_restart_and_manager_handoff(tmp_path):
    root = tmp_path / "registry"
    server = AetherHubServer(root)
    alice = server.create_user("alice", "alice@example.test", namespaces=["alice"])
    source = payload()
    with running_hub(server) as url:
        client = HubClient(url, auth_token=alice.api_key, allow_local_cache=True)
        manager = KVCacheManager(hub_client=client, hub_namespace="alice")
        block = manager.import_payload(source)
        key = manager.payload_key(block.block_id)
        assert manager.swap(block.block_id, MemoryTier.L4_HUB)
        assert manager.get_transfer_stats()["resident_tensor_bytes"] == 0
        assert client.list_kv("alice")[0]["content_hash"] == hashlib.sha256(source.serialize()).hexdigest()
        recovered = manager.reload(block.block_id, expected_model_fingerprint=source.model_fingerprint)
        assert recovered.serialize() == source.serialize()
    # New server/client/manager share no process-local cache from the first run.
    with running_hub(AetherHubServer(root)) as url:
        client = HubClient(url, auth_token=alice.api_key, allow_local_cache=False)
        manager = KVCacheManager(hub_client=client, hub_namespace="alice")
        imported = manager.import_from_hub(key, expected_model_fingerprint=source.model_fingerprint)
        restored = manager.export_payload(imported.block_id)
        assert restored.array("keys").tobytes() == source.array("keys").tobytes()
        assert restored.array("values").dtype == np.float16
        assert manager.get_transfer_stats()["transferred_tensor_bytes"] == source.byte_size
        assert manager.remove(imported.block_id)
        assert client.list_kv("alice") == []
        assert client.delete_kv("alice", key) is False
        with pytest.raises(HubError, match="404"):
            client.get_kv("alice", key)


def test_hub_kv_auth_private_visibility_hash_body_quota_and_corruption(tmp_path):
    httpx = pytest.importorskip("httpx")
    server = AetherHubServer(tmp_path / "registry")
    alice = server.create_user("alice", "alice@example.test", namespaces=["alice"])
    bob = server.create_user("bob", "bob@example.test", namespaces=["bob"])
    viewer = server.create_user("viewer", "viewer@example.test", role="viewer", namespaces=["alice"])
    encoded = payload().serialize()
    headers = {"Authorization": f"Bearer {alice.api_key}", "X-Content-Hash": hashlib.sha256(encoded).hexdigest()}
    with running_hub(server, max_upload_bytes=len(encoded) + 100, max_kv_storage_bytes=len(encoded)) as url:
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/v1/kv/alice/current").status_code == 401
            assert client.put("/v1/kv/alice/current", content=encoded, headers={"Authorization": f"Bearer {bob.api_key}"}).status_code == 403
            assert client.put("/v1/kv/alice/current", content=encoded, headers={**headers, "Authorization": f"Bearer {viewer.api_key}"}).status_code == 403
            assert client.put("/v1/kv/alice/current", content=encoded, headers={**headers, "X-Content-Hash": "wrong"}).status_code == 400
            invalid = b"not a tensor cache"
            assert client.put("/v1/kv/alice/current", content=invalid,
                              headers={**headers, "X-Content-Hash": hashlib.sha256(invalid).hexdigest()}).status_code == 400
            assert client.put("/v1/kv/alice/current", content=b"x" * (len(encoded) + 101), headers=headers).status_code == 413
            assert client.put("/v1/kv/alice/current", content=encoded, headers=headers).status_code == 200
            assert client.get("/v1/kv/alice/current", headers={"Authorization": f"Bearer {bob.api_key}"}).status_code == 404
            assert client.get("/v1/kv/alice", headers={"Authorization": f"Bearer {bob.api_key}"}).status_code == 404
            read = client.get("/v1/kv/alice/current", headers={"Authorization": f"Bearer {viewer.api_key}"})
            assert read.content == encoded and read.headers["cache-control"] == "private, no-store"
            different = payload(2).serialize()
            assert client.put("/v1/kv/alice/new", content=different,
                              headers={**headers, "X-Content-Hash": hashlib.sha256(different).hexdigest()}).status_code == 413
            assert client.get("/v1/kv/alice/current", headers=headers).content == encoded
            stored = next((server.storage.storage_root / "kv_cache" / "payloads").glob("*.blob"))
            damaged = bytearray(stored.read_bytes())
            damaged[-1] ^= 1
            stored.write_bytes(damaged)
            assert client.get("/v1/kv/alice/current", headers=headers).status_code == 500


def test_hub_cache_no_hidden_fallback_and_strict_ack(monkeypatch):
    client = HubClient("http://127.0.0.1:1", auth_token="token", allow_local_cache=True)
    def offline(*args, **kwargs):
        raise HubError("offline")
    monkeypatch.setattr(client, "_request", offline)
    with pytest.raises(HubError, match="offline"):
        client.put_kv("alice", "key", payload())
    with pytest.raises(HubError, match="offline"):
        client.delete_kv("alice", "key")
    with pytest.raises(HubError, match="offline"):
        client.list_kv("alice")
    monkeypatch.setattr("urllib.request.urlopen", offline)
    with pytest.raises(HubError, match="offline"):
        client.get_kv("alice", "key")
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {"status": "ok"})
    with pytest.raises(HubError, match="acknowledgement"):
        client.put_kv("alice", "key", payload())
    manager = KVCacheManager(hub_client=client, hub_namespace="alice")
    block = manager.import_payload(payload())
    with pytest.raises(HubError, match="acknowledgement"):
        manager.swap(block.block_id, MemoryTier.L4_HUB)
    assert block.tier == MemoryTier.L2_CPU_DRAM
    assert manager.get_transfer_stats()["payload_transfers"] == 0


def test_remote_model_fingerprint_mismatch_is_rejected(tmp_path):
    server = AetherHubServer(tmp_path / "registry")
    user = server.create_user("alice", "alice@example.test", namespaces=["alice"])
    with running_hub(server) as url:
        client = HubClient(url, auth_token=user.api_key)
        client.put_kv("alice", "key", payload())
        with pytest.raises(HubError, match="fingerprint"):
            client.get_kv("alice", "key", expected_model_fingerprint="other-model")
        with pytest.raises(HubError, match="read bound"):
            client.get_kv("alice", "key", max_bytes=10)
        client.auth_token = None
        with pytest.raises(AuthenticationError):
            client.get_kv("alice", "key")


def test_actual_decoder_prefill_hub_restart_reload_decode_parity(tmp_path):
    from aether.kernels.native_cpu import NativeCPUKernels
    from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
    from aether.runtime.cache_handoff import export_cpu_cache, restore_cpu_cache
    rng = np.random.default_rng(71)
    def matrix(rows, columns):
        return rng.normal(0., .1, (rows, columns)).astype(np.float32)
    layer = LayerWeights(attention_norm=np.ones(8, np.float32), q_proj=matrix(8, 8), k_proj=matrix(4, 8),
                         v_proj=matrix(4, 8), o_proj=matrix(8, 8), ffn_norm=np.ones(8, np.float32),
                         gate_proj=matrix(12, 8), up_proj=matrix(12, 8), down_proj=matrix(8, 12))
    weights = ModelWeights(embedding=matrix(17, 8), layers=[layer], final_norm=np.ones(8, np.float32),
                           lm_head=matrix(17, 8), context_length=32)
    source = CPUExecutionEngine(weights, 2, 1, kernels=NativeCPUKernels.reference())
    target = CPUExecutionEngine(weights, 2, 1, kernels=NativeCPUKernels.reference())
    _, cache = source.forward(np.asarray([1, 3, 5], dtype=np.int64))
    fingerprint = hashlib.sha256(weights.embedding.tobytes() + weights.lm_head.tobytes()).hexdigest()
    snapshot = export_cpu_cache(cache, model_fingerprint=fingerprint)
    root = tmp_path / "registry"
    server = AetherHubServer(root)
    user = server.create_user("alice", "alice@example.test", namespaces=["alice"])
    with running_hub(server) as url:
        client = HubClient(url, auth_token=user.api_key, allow_local_cache=False)
        client.put_kv("alice", "prefill", snapshot)
    with running_hub(AetherHubServer(root)) as url:
        client = HubClient(url, auth_token=user.api_key, allow_local_cache=False)
        loaded = client.get_kv("alice", "prefill", expected_model_fingerprint=fingerprint)
        resumed = restore_cpu_cache(loaded, expected_model_fingerprint=fingerprint)
        expected, expected_cache = source.forward(np.asarray([2, 4], dtype=np.int64), cache)
        actual, actual_cache = target.forward(np.asarray([2, 4], dtype=np.int64), resumed)
        np.testing.assert_array_equal(actual, expected)
        assert actual_cache.length == expected_cache.length == 5
