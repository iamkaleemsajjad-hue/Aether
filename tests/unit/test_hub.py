"""Tests for the hub package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether.hub import AuthCredentials, HubClient, KernelCache, TokenManager


class TestHubClient:
    def test_init_defaults(self) -> None:
        client = HubClient()
        assert "hub.aether.dev" in client.hub_url

    def test_login_logout(self) -> None:
        client = HubClient()
        client.login("test-token")
        assert client.auth_token == "test-token"
        result = client.logout()
        assert result["status"] == "ok"
        assert client.auth_token is None

    def test_search_empty(self) -> None:
        client = HubClient()
        results = client.search("llama")
        assert isinstance(results, list)

    def test_upload_requires_auth(self) -> None:
        client = HubClient()
        with pytest.raises(Exception):
            client.upload("test-model", ".")


class TestKernelCache:
    def test_store_and_lookup(self, tmp_path: Path) -> None:
        cache = KernelCache(cache_dir=str(tmp_path))
        data = b"kernel_binary_data"
        path = cache.store("hash1", "cuda_sm90", "0.1.0", data, {"precision": "FP8"})
        assert path.exists()
        result = cache.lookup("hash1", "cuda_sm90", "0.1.0")
        assert result == data

    def test_lookup_miss(self, tmp_path: Path) -> None:
        cache = KernelCache(cache_dir=str(tmp_path))
        result = cache.lookup("nonexistent", "cuda_sm90", "0.1.0")
        assert result is None

    def test_exists(self, tmp_path: Path) -> None:
        cache = KernelCache(cache_dir=str(tmp_path))
        assert not cache.exists("hash1", "cuda_sm90", "0.1.0")
        cache.store("hash1", "cuda_sm90", "0.1.0", b"data")
        assert cache.exists("hash1", "cuda_sm90", "0.1.0")


class TestAuthCredentials:
    def test_sign_request(self) -> None:
        creds = AuthCredentials(access_key="ak1", secret_key="sk1")
        sig = creds.sign_request("GET", "/api/v1/models")
        assert sig.startswith("ak1")

    def test_to_dict(self) -> None:
        creds = AuthCredentials(access_key="ak1", secret_key="sk1")
        d = creds.to_dict()
        assert d["access_key"] == "ak1"
        assert "secret_key" not in d


class TestTokenManager:
    def test_add_and_validate(self) -> None:
        mgr = TokenManager()
        mgr.add_token("dev", "tok123", ["read", "write"])
        name = mgr.validate_token("tok123")
        assert name == "dev"

    def test_revoke(self) -> None:
        mgr = TokenManager()
        mgr.add_token("dev", "tok123")
        assert mgr.revoke_token("dev") is True
        assert mgr.revoke_token("dev") is False
        name = mgr.validate_token("tok123")
        assert name is None
