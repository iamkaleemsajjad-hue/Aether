"""
Aether Runtime — Complete Hub System Test Suite.

Tests the full Hub model registry including:
  - HubStorageBackend: content-addressed blob storage, deduplication
  - HubModel / HubModelVersion: data models and serialization
  - HubUser: user accounts and API key generation
  - AetherHubServer: push/pull/search/delete/auth/versioning
  - FastAPI integration: HTTP routes (when fastapi available)
  - Path-traversal protection in ZIP artifacts

Research basis:
  - OCI Distribution Spec (Docker Registry HTTP API V2)
  - HuggingFace Hub API design (2024)
  - Aether Hub specification (PRD v3.1, v4.0)
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aether.hub.server import (
    AetherHubServer,
    HubModel,
    HubModelVersion,
    HubStorageBackend,
    HubUser,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_aeg_zip(manifest: dict | None = None) -> bytes:
    """Create a minimal, valid AEG ZIP artifact."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.bin", b"\x00" * 64)
        zf.writestr("tokenizer.json", json.dumps({"version": "1.0"}))
        if manifest is not None:
            zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def _make_full_aeg_zip() -> bytes:
    """Create a full AEG ZIP with a manifest.json."""
    manifest = {
        "format_version": "aeg/2.0",
        "architecture": {"family": "llama", "params_b": 7},
        "kernels": {"cpu_avx512": {}, "cpu_avx2": {}},
    }
    return _make_aeg_zip(manifest)


# ---------------------------------------------------------------------------
# HubStorageBackend
# ---------------------------------------------------------------------------

class TestHubStorageBackend:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.storage = HubStorageBackend(Path(self._tmpdir))

    def test_init_creates_dirs(self):
        assert (Path(self._tmpdir) / "blobs").is_dir()
        assert (Path(self._tmpdir) / "metadata").is_dir()

    def test_store_and_load_blob(self):
        data = b"Hello AEG artifact content"
        content_hash = self.storage.store_blob(data)
        assert len(content_hash) == 64  # SHA-256 hex
        recovered = self.storage.load_blob(content_hash)
        assert recovered == data

    def test_load_nonexistent_blob_returns_none(self):
        result = self.storage.load_blob("a" * 64)
        assert result is None

    def test_has_blob_true_after_store(self):
        data = b"test content"
        h = self.storage.store_blob(data)
        assert self.storage.has_blob(h) is True

    def test_has_blob_false_for_unknown(self):
        assert self.storage.has_blob("b" * 64) is False

    def test_deduplication(self):
        """Storing same content twice should produce the same hash."""
        data = b"identical content"
        h1 = self.storage.store_blob(data)
        h2 = self.storage.store_blob(data)
        assert h1 == h2
        # Only one blob file should exist
        blob_files = list((Path(self._tmpdir) / "blobs").rglob("*"))
        blob_files = [f for f in blob_files if f.is_file()]
        assert len(blob_files) == 1

    def test_store_and_load_model(self):
        model = HubModel(
            model_id="model_001",
            namespace="testns",
            name="my_model",
            created_at=time.time(),
            created_by="user_x",
            description="A test model",
            visibility="public",
            tags=["llm", "7b"],
        )
        self.storage.store_model(model)
        recovered = self.storage.load_model("model_001")
        assert recovered is not None
        assert recovered.model_id == "model_001"
        assert recovered.name == "my_model"
        assert recovered.namespace == "testns"
        assert "llm" in recovered.tags

    def test_load_nonexistent_model_returns_none(self):
        result = self.storage.load_model("nonexistent_id")
        assert result is None

    def test_find_model_by_name(self):
        model = HubModel(
            model_id="find_001",
            namespace="corp",
            name="llama7b",
            created_at=time.time(),
            created_by="admin",
        )
        self.storage.store_model(model)
        found = self.storage.find_model_by_name("corp", "llama7b")
        assert found is not None
        assert found.model_id == "find_001"

    def test_find_nonexistent_model_returns_none(self):
        result = self.storage.find_model_by_name("nobody", "nothing")
        assert result is None

    def test_search_by_namespace(self):
        for i in range(3):
            m = HubModel(
                model_id=f"ns_model_{i}",
                namespace="myns",
                name=f"model_{i}",
                created_at=time.time(),
                created_by="dev",
                visibility="public",
            )
            self.storage.store_model(m)
        results = self.storage.search_models(namespace="myns")
        assert len(results) == 3

    def test_search_by_visibility(self):
        for i, vis in enumerate(["public", "private", "public"]):
            m = HubModel(
                model_id=f"vis_model_{i}",
                namespace="vis_ns",
                name=f"model_{i}",
                created_at=time.time(),
                created_by="dev",
                visibility=vis,
            )
            self.storage.store_model(m)
        public = self.storage.search_models(visibility="public")
        private = self.storage.search_models(visibility="private")
        assert len(public) == 2
        assert len(private) == 1

    def test_index_persists_across_instances(self):
        model = HubModel(
            model_id="persist_001",
            namespace="persist_ns",
            name="persist_model",
            created_at=time.time(),
            created_by="dev",
        )
        self.storage.store_model(model)
        # Create a new storage instance pointing to same directory
        new_storage = HubStorageBackend(Path(self._tmpdir))
        recovered = new_storage.load_model("persist_001")
        assert recovered is not None
        assert recovered.name == "persist_model"

    def test_store_and_load_user(self):
        user = HubUser(
            user_id="user_001",
            username="alice",
            email="alice@example.com",
            api_key="aether_abc123",
            created_at=time.time(),
            role="user",
            namespaces=["alice"],
        )
        self.storage.store_user(user)
        found = self.storage.find_user_by_api_key("aether_abc123")
        assert found is not None
        assert found.username == "alice"
        assert found.email == "alice@example.com"
        assert found.role == "user"

    def test_find_user_by_wrong_key_returns_none(self):
        result = self.storage.find_user_by_api_key("invalid_key_xxx")
        assert result is None


# ---------------------------------------------------------------------------
# HubModel
# ---------------------------------------------------------------------------

class TestHubModel:
    def test_full_name_property(self):
        model = HubModel(
            model_id="m1", namespace="org", name="model",
            created_at=0, created_by="dev",
        )
        assert model.full_name == "org/model"

    def test_latest_version_none_when_empty(self):
        model = HubModel(
            model_id="m2", namespace="ns", name="n",
            created_at=0, created_by="dev",
        )
        assert model.latest_version is None

    def test_latest_version_most_recent(self):
        model = HubModel(
            model_id="m3", namespace="ns", name="n",
            created_at=0, created_by="dev",
        )
        v1 = HubModelVersion("v1", "v1.0", "hash1", 100, 1000.0, "dev")
        v2 = HubModelVersion("v2", "v2.0", "hash2", 200, 2000.0, "dev")
        model.versions = [v1, v2]
        assert model.latest_version.version_id == "v2"

    def test_to_dict_structure(self):
        model = HubModel(
            model_id="m4", namespace="corp", name="llama",
            created_at=12345.0, created_by="admin",
            description="Test model",
            visibility="public",
            tags=["llm"],
        )
        d = model.to_dict()
        assert d["model_id"] == "m4"
        assert d["namespace"] == "corp"
        assert d["full_name"] == "corp/llama"
        assert d["visibility"] == "public"
        assert "llm" in d["tags"]
        assert d["latest_version"] is None

    def test_version_to_dict(self):
        v = HubModelVersion(
            version_id="v1", tag="v1.0", content_hash="abc123",
            size_bytes=1024, created_at=9999.0, pushed_by="alice",
            architecture="llama", hardware_targets=["cpu_avx512"],
        )
        d = v.to_dict()
        assert d["tag"] == "v1.0"
        assert d["architecture"] == "llama"
        assert "cpu_avx512" in d["hardware_targets"]


# ---------------------------------------------------------------------------
# AetherHubServer
# ---------------------------------------------------------------------------

class TestAetherHubServer:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.hub = AetherHubServer(storage_root=self._tmpdir)

    def test_default_admin_created(self):
        """Default admin user should be created on init."""
        admin = self.hub.storage.find_user_by_api_key(
            "aether_admin_" + hashlib.sha256(b"default_admin").hexdigest()[:32]
        )
        assert admin is not None
        assert admin.role == "admin"

    def test_push_and_pull_model(self):
        """Push an artifact and then pull it back — should be identical."""
        artifact = _make_aeg_zip()
        version = self.hub.push_model(
            namespace="test_ns",
            name="llama7b",
            tag="v1.0",
            artifact_data=artifact,
            pushed_by="alice",
            description="Test LLaMA 7B",
        )
        assert version.tag == "v1.0"
        assert version.size_bytes == len(artifact)

        pulled = self.hub.pull_model("test_ns", "llama7b", "v1.0")
        assert pulled == artifact

    def test_pull_latest_tag(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("ns1", "model1", "v1", artifact, "dev")
        pulled = self.hub.pull_model("ns1", "model1", "latest")
        assert pulled == artifact

    def test_pull_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.hub.pull_model("nobody", "nothing", "latest")

    def test_pull_wrong_tag_raises(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("ns2", "model2", "v1", artifact, "dev")
        with pytest.raises(ValueError, match="not found"):
            self.hub.pull_model("ns2", "model2", "v999")

    def test_push_invalid_zip_raises(self):
        with pytest.raises(ValueError, match="Invalid artifact"):
            self.hub.push_model("ns3", "m3", "v1", b"not a zip file", "dev")

    def test_path_traversal_in_zip_rejected(self):
        """ZIP with path-traversal entries should be rejected."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../../etc/passwd", "root:x:0:0")
        with pytest.raises(ValueError, match="path traversal"):
            self.hub.push_model("ns", "evil", "v1", buf.getvalue(), "attacker")

    def test_content_deduplication(self):
        """Pushing same artifact twice should deduplicate storage."""
        artifact = _make_aeg_zip()
        self.hub.push_model("ns4", "model_a", "v1", artifact, "dev")
        self.hub.push_model("ns4", "model_b", "v1", artifact, "dev")
        # Both should be pullable
        assert self.hub.pull_model("ns4", "model_a") == artifact
        assert self.hub.pull_model("ns4", "model_b") == artifact
        # But only one blob should be stored
        blobs = list((Path(self._tmpdir) / "blobs").rglob("*"))
        blob_files = [b for b in blobs if b.is_file()]
        assert len(blob_files) == 1

    def test_multiple_versions(self):
        """A model can have multiple versions with different tags."""
        artifact_v1 = _make_aeg_zip()
        artifact_v2 = _make_aeg_zip({"version": "2"})
        self.hub.push_model("mv_ns", "multi_v", "v1.0", artifact_v1, "dev")
        self.hub.push_model("mv_ns", "multi_v", "v2.0", artifact_v2, "dev")

        pulled_v1 = self.hub.pull_model("mv_ns", "multi_v", "v1.0")
        pulled_v2 = self.hub.pull_model("mv_ns", "multi_v", "v2.0")
        assert pulled_v1 == artifact_v1
        assert pulled_v2 == artifact_v2
        assert pulled_v1 != pulled_v2

    def test_get_model_info(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("info_ns", "info_model", "v1", artifact, "dev", description="A test")
        info = self.hub.get_model_info("info_ns", "info_model")
        assert info is not None
        assert info["namespace"] == "info_ns"
        assert info["name"] == "info_model"
        assert info["description"] == "A test"
        assert len(info["versions"]) == 1

    def test_get_model_info_nonexistent_returns_none(self):
        assert self.hub.get_model_info("nobody", "nothing") is None

    def test_search_by_namespace(self):
        for i in range(3):
            artifact = _make_aeg_zip()
            self.hub.push_model("search_ns", f"model_{i}", "v1", artifact, "dev")
        results = self.hub.search(namespace="search_ns")
        assert len(results) == 3

    def test_search_by_query(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("q_ns", "llama_7b", "v1", artifact, "dev", description="LLaMA 7B model")
        self.hub.push_model("q_ns", "mistral", "v1", artifact, "dev", description="Mistral model")
        results = self.hub.search(query="llama")
        names = [r["name"] for r in results]
        assert "llama_7b" in names
        assert "mistral" not in names

    def test_delete_model(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("del_ns", "to_delete", "v1", artifact, "dev")
        assert self.hub.get_model_info("del_ns", "to_delete") is not None
        result = self.hub.delete_model("del_ns", "to_delete")
        assert result is True
        assert self.hub.get_model_info("del_ns", "to_delete") is None

    def test_delete_nonexistent_returns_false(self):
        assert self.hub.delete_model("nobody", "nothing") is False

    def test_delete_specific_version(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("dv_ns", "dv_model", "v1", artifact, "dev")
        self.hub.push_model("dv_ns", "dv_model", "v2", artifact, "dev")
        result = self.hub.delete_model("dv_ns", "dv_model", tag="v1")
        assert result is True
        info = self.hub.get_model_info("dv_ns", "dv_model")
        tags = [v["tag"] for v in info["versions"]]
        assert "v1" not in tags
        assert "v2" in tags

    def test_create_user(self):
        user = self.hub.create_user("bob", "bob@example.com", role="user")
        assert user.username == "bob"
        assert user.email == "bob@example.com"
        assert user.api_key.startswith("aether_")
        assert len(user.api_key) > 10

    def test_authenticate_with_api_key(self):
        user = self.hub.create_user("charlie", "charlie@test.com")
        authenticated = self.hub.authenticate(user.api_key)
        assert authenticated is not None
        assert authenticated.username == "charlie"

    def test_authenticate_invalid_key_returns_none(self):
        assert self.hub.authenticate("invalid_key_xxx") is None

    def test_download_count_incremented(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("dc_ns", "dc_model", "v1", artifact, "dev")
        info_before = self.hub.get_model_info("dc_ns", "dc_model")
        assert info_before["download_count"] == 0
        self.hub.pull_model("dc_ns", "dc_model")
        info_after = self.hub.get_model_info("dc_ns", "dc_model")
        assert info_after["download_count"] == 1

    def test_get_stats(self):
        artifact = _make_aeg_zip()
        self.hub.push_model("stats_ns", "s_model", "v1", artifact, "dev")
        stats = self.hub.get_stats()
        assert "total_models" in stats
        assert "total_users" in stats
        assert "total_blobs" in stats
        assert "storage_bytes" in stats
        assert stats["total_models"] >= 1
        assert stats["total_users"] >= 1

    def test_manifest_metadata_extracted(self):
        """When manifest.json is in the ZIP, metadata should be extracted."""
        artifact = _make_full_aeg_zip()
        version = self.hub.push_model(
            "meta_ns", "meta_model", "v1", artifact, "dev"
        )
        assert version.architecture == "llama"
        assert "cpu_avx512" in version.hardware_targets

    def test_content_integrity_verified_on_pull(self):
        """If blob is corrupted, pull should raise."""
        artifact = _make_aeg_zip()
        version = self.hub.push_model("corrupt_ns", "corrupt_m", "v1", artifact, "dev")
        # Directly corrupt the blob
        content_hash = version.content_hash
        blob_path = self.hub.storage.blobs_dir / content_hash[:2] / content_hash[2:]
        blob_path.write_bytes(b"CORRUPTED")
        with pytest.raises(ValueError, match="hash mismatch"):
            self.hub.pull_model("corrupt_ns", "corrupt_m")


# ---------------------------------------------------------------------------
# Hub client integration
# ---------------------------------------------------------------------------

class TestHubClient:
    def test_hub_client_importable(self):
        from aether.hub.client import HubClient
        assert HubClient is not None

    def test_hub_client_init(self):
        # Real signature: HubClient(hub_url, auth_token, timeout_s, allow_local_cache)
        from aether.hub.client import HubClient
        client = HubClient(
            hub_url="http://localhost:8765",
            auth_token="test_key",
            allow_local_cache=True,
        )
        assert client is not None

    def test_hub_auth_importable(self):
        # Real classes: AuthCredentials, TokenManager (not HubAuth)
        from aether.hub.auth import AuthCredentials, TokenManager
        assert AuthCredentials is not None
        assert TokenManager is not None


# ---------------------------------------------------------------------------
# E2E: push → pull round-trip via server API
# ---------------------------------------------------------------------------

class TestHubE2E:
    def test_push_pull_round_trip_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = AetherHubServer(storage_root=tmpdir)
            artifact = _make_full_aeg_zip()
            hub.push_model("e2e", "model", "v1.0", artifact, "alice")
            recovered = hub.pull_model("e2e", "model", "v1.0")
            assert recovered == artifact

    def test_multiple_tenants_isolated(self):
        """Two different namespaces should not interfere."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = AetherHubServer(storage_root=tmpdir)
            artifact_a = _make_aeg_zip({"tenant": "a"})
            artifact_b = _make_aeg_zip({"tenant": "b"})
            hub.push_model("tenant_a", "shared_name", "v1", artifact_a, "alice")
            hub.push_model("tenant_b", "shared_name", "v1", artifact_b, "bob")
            pulled_a = hub.pull_model("tenant_a", "shared_name")
            pulled_b = hub.pull_model("tenant_b", "shared_name")
            assert pulled_a == artifact_a
            assert pulled_b == artifact_b
            assert pulled_a != pulled_b

    def test_search_returns_only_owned_namespace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub = AetherHubServer(storage_root=tmpdir)
            artifact = _make_aeg_zip()
            hub.push_model("ns_a", "m1", "v1", artifact, "alice")
            hub.push_model("ns_b", "m2", "v1", artifact, "bob")
            results_a = hub.search(namespace="ns_a")
            assert all(r["namespace"] == "ns_a" for r in results_a)
            assert len(results_a) == 1
