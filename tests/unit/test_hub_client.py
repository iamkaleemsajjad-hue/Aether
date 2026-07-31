"""
Tests for the Hub client — local cache, HTTP fallback, auth, upload/download.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aether.hub.client import HubClient, HubManifest


class TestHubManifest:
    def test_to_dict_roundtrip(self):
        m = HubManifest(
            model_id="org/model",
            aether_version="1.0.0",
            aeg_version="AEG/1.0",
            targets=["cuda_sm90", "cpu_avx512"],
            architecture={"family": "llama", "layers": 32},
            file_size_bytes=1024,
            content_hash="abc123",
            uploaded_at="2025-01-01T00:00:00Z",
            downloads=42,
        )
        d = m.to_dict()
        m2 = HubManifest.from_dict(d)
        assert m2.model_id == m.model_id
        assert m2.targets == m.targets
        assert m2.downloads == 42

    def test_from_dict_minimal(self):
        d = {"model_id": "test/model"}
        m = HubManifest.from_dict(d)
        assert m.model_id == "test/model"
        assert m.targets == []
        assert m.aether_version == ""


class TestHubClientLocalCache:
    def _client(self) -> HubClient:
        return HubClient(hub_url="http://localhost:9999")  # unreachable → local mode

    def test_login_stores_token(self):
        client = self._client()
        result = client.login("my-secret-token")
        assert client.auth_token == "my-secret-token"
        assert result["status"] == "ok"

    def test_logout_clears_token(self):
        client = self._client()
        client.login("token")
        client.logout()
        assert client.auth_token is None

    def test_upload_requires_auth(self):
        from aether.core.exceptions import AuthenticationError
        client = self._client()
        with pytest.raises(AuthenticationError):
            client.upload("model/id", "/some/path")

    def test_upload_missing_path(self):
        from aether.core.exceptions import HubError
        client = self._client()
        client.login("token")
        with pytest.raises(HubError):
            client.upload("model/id", "/nonexistent/path")

    def test_upload_directory(self):
        client = self._client()
        client.login("test-token")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal AEG structure
            Path(tmpdir, "manifest.json").write_text(
                json.dumps({
                    "model_id": "test/model",
                    "architecture": {"family": "llama"},
                    "kernels": {"targets": ["cpu_avx512"]},
                })
            )
            manifest = client.upload("test/model", tmpdir)
            assert manifest.model_id == "test/model"
            assert manifest.file_size_bytes > 0
            assert manifest.content_hash != ""

    def test_upload_stores_in_local_cache(self):
        client = self._client()
        client.login("token")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "manifest.json").write_text('{"model_id": "m", "kernels": {}}')
            client.upload("m", tmpdir)
            assert client.get_manifest("m") is not None

    def test_download_missing_raises(self):
        from unittest.mock import patch
        from aether.core.exceptions import HubError
        client = self._client()
        # Patch _is_hub_available so we don't block on a real TCP connection
        with patch.object(client, "_is_hub_available", return_value=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(HubError):
                    client.download("missing/model", tmpdir)

    def test_download_from_local_cache(self):
        client = self._client()
        client.login("token")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "manifest.json").write_text('{"model_id": "local", "kernels": {}}')
            client.upload("local/model", tmpdir)
        with tempfile.TemporaryDirectory() as outdir:
            path = client.download("local/model", outdir)
            assert path.exists()

    def test_search_local_cache(self):
        client = self._client()
        client.login("token")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "manifest.json").write_text('{"model_id": "llama/7b", "kernels": {}}')
            client.upload("llama/7b", tmpdir)
        results = client.search("llama")
        assert any(m.model_id == "llama/7b" for m in results)

    def test_search_no_match(self):
        client = self._client()
        results = client.search("nonexistent_prefix_xyz")
        assert results == []

    def test_list_models(self):
        client = self._client()
        client.login("t")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "manifest.json").write_text('{"model_id": "A", "kernels": {}}')
            client.upload("A", d)
        with tempfile.TemporaryDirectory() as d:
            Path(d, "manifest.json").write_text('{"model_id": "B", "kernels": {}}')
            client.upload("B", d)
        models = client.list_models()
        ids = {m.model_id for m in models}
        assert "A" in ids
        assert "B" in ids

    def test_get_manifest_miss(self):
        client = self._client()
        assert client.get_manifest("missing") is None

    def test_delete_removes_from_cache(self):
        client = self._client()
        client.login("token")
        with tempfile.TemporaryDirectory() as d:
            Path(d, "manifest.json").write_text('{"model_id": "M", "kernels": {}}')
            client.upload("M", d)
        assert client.get_manifest("M") is not None
        client.delete("M")
        assert client.get_manifest("M") is None

    def test_delete_requires_auth(self):
        from aether.core.exceptions import AuthenticationError
        client = self._client()
        with pytest.raises(AuthenticationError):
            client.delete("M")

    def test_stats(self):
        client = self._client()
        stats = client.stats()
        assert "hub_url" in stats
        assert "local_cache_count" in stats
        assert "hub_available" in stats

    def test_repr(self):
        client = self._client()
        assert "HubClient" in repr(client)

    def test_content_hash_deterministic(self):
        client = self._client()
        data = b"hello world"
        h1 = client._content_hash(data)
        h2 = client._content_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_zip_directory(self):
        import zipfile, io
        client = self._client()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.json").write_text('{"x": 1}')
            Path(d, "b.bin").write_bytes(b"\x00\x01\x02")
            zipped = client._zip_directory(Path(d))
            assert len(zipped) > 0
            with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
                names = zf.namelist()
                assert any("a.json" in n for n in names)
