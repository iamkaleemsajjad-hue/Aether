"""Remote CLI operations must persist remotely or fail honestly."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import socket
import stat
import threading
import time
import zipfile

from click.testing import CliRunner
import pytest

from aether.cli import cli
from aether.core.exceptions import AuthenticationError, HubError
from aether.hub.client import HubClient, HubManifest
from aether.hub.server import AetherHubServer, create_hub_app


def _runner():
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


@pytest.mark.parametrize("operation", ["login", "search", "list", "manifest", "upload", "download", "delete"])
def test_strict_remote_mode_does_not_fall_back_to_memory(operation, monkeypatch, tmp_path):
    client = HubClient(auth_token="test", allow_local_cache=False)
    package = tmp_path / "source.aeg"
    package.mkdir()
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    client._local_cache["org/model"] = HubManifest("org/model", "0", "AEG/1.0", [])
    monkeypatch.setattr(client, "_is_hub_available", lambda: False)
    def offline(*args, **kwargs):
        raise HubError("remote service is unavailable")
    monkeypatch.setattr(client, "_request", offline)
    monkeypatch.setattr("urllib.request.urlopen", offline)
    operations = {
        "login": lambda: client.login("new-token"),
        "search": lambda: client.search("model"),
        "list": client.list_models,
        "manifest": lambda: client.get_manifest("org/model"),
        "upload": lambda: client.upload("org/model", package),
        "download": lambda: client.download("org/model", tmp_path / "out"),
        "delete": lambda: client.delete("org/model"),
    }
    with pytest.raises(HubError, match="unavailable"):
        operations[operation]()
    assert "org/model" in client._local_cache
    assert client.auth_token == "test"


def test_authentication_rejection_never_becomes_local_login_success(monkeypatch):
    for local in (False, True):
        client = HubClient(auth_token="previous", allow_local_cache=local)
        monkeypatch.setattr(client, "_is_hub_available", lambda: True)
        def rejected(*args, **kwargs):
            raise AuthenticationError("invalid token")
        monkeypatch.setattr(client, "_request", rejected)
        with pytest.raises(AuthenticationError):
            client.login("bad-token")
        assert client.auth_token == "previous"


@pytest.mark.parametrize("response", [{}, {"status": "ok"}, {"authentication_validated": 1}, {"authentication_validated": False}])
def test_login_requires_explicit_authentication_evidence(response, monkeypatch):
    client = HubClient(auth_token="previous", allow_local_cache=False)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises(AuthenticationError, match="explicitly validate"):
        client.login("new-token")
    assert client.auth_token == "previous"
    assert not client.last_operation.get("authentication_validated", False)


@pytest.mark.parametrize("field,value", [("manifest", None), ("model_id", "other/model"), ("content_hash", "wrong"), ("file_size_bytes", -1), ("file_size_bytes", "7")])
def test_upload_requires_acknowledgement_of_exact_payload(field, value, monkeypatch, tmp_path):
    payload = b"payload"
    source = tmp_path / "package.aeg"
    source.write_bytes(payload)
    acknowledgement = {
        "model_id": "org/model", "aether_version": "0", "aeg_version": "AEG/2.0",
        "targets": [], "content_hash": hashlib.sha256(payload).hexdigest(),
        "file_size_bytes": len(payload),
    }
    response = {"manifest": acknowledgement}
    if field == "manifest":
        response[field] = value
    else:
        acknowledgement[field] = value
    client = HubClient(auth_token="token", allow_local_cache=False)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises(HubError, match="acknowledge the exact"):
        client.upload("org/model", source)
    assert client._local_cache == {}
    assert client._local_packages == {}


def test_explicit_local_cache_reports_memory_only_scope(monkeypatch, tmp_path):
    client = HubClient(allow_local_cache=True)
    monkeypatch.setattr(client, "_is_hub_available", lambda: False)
    login = client.login("token")
    assert login["authentication_validated"] is False
    assert login["persistence"] == "client_instance_memory"
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = client.upload("org/model", tmp_path)
    assert manifest.source == "local_cache"
    assert manifest.persistence == "client_instance_memory"
    assert HubClient(allow_local_cache=True)._local_cache == {}


def test_archive_rejects_external_symlink_before_reading_payload(tmp_path):
    source = tmp_path / "package"
    source.mkdir()
    outside = tmp_path / "outside-secret"
    outside.write_text("never upload this", encoding="utf-8")
    try:
        (source / "extra").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"This host cannot create file symlinks: {exc}")
    with pytest.raises(HubError, match="symlinks, junctions"):
        HubClient()._zip_directory(source)


def test_archive_rejects_windows_reparse_points_before_traversal(monkeypatch, tmp_path):
    source = tmp_path / "package"
    source.mkdir()
    child = source / "junction"
    child.mkdir()
    original = Path.lstat
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", reparse_flag, raising=False)
    def lstat(path):
        info = original(path)
        if path == child:
            return type("ReparseStat", (), {"st_mode": info.st_mode, "st_file_attributes": reparse_flag})()
        return info
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(HubError, match="symlinks, junctions"):
        HubClient()._zip_directory(source)


@contextmanager
def _running_hub(hub, **kwargs):
    uvicorn = pytest.importorskip("uvicorn")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_hub_app(hub, **kwargs), host="127.0.0.1", port=port, log_level="critical", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    try:
        while not server.started and time.monotonic() < deadline:
            assert thread.is_alive(), "Hub server failed during startup"
            time.sleep(0.01)
        assert server.started, "Hub server did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive(), "Hub server failed to shut down"


def test_cli_real_local_http_upload_download_and_server_restart(minimal_aeg_package, tmp_path):
    storage = tmp_path / "registry"
    hub = AetherHubServer(storage)
    # This test authenticates a generated scoped account, not the legacy
    # development bootstrap administrator credential.
    user = hub.create_user("alice", "alice@example.test", namespaces=["alice"])
    cache = tmp_path / "cli-cache"
    runner = _runner()
    with _running_hub(hub) as url:
        prefix = ["--cache-dir", str(cache), "hub", "--url", url]
        logged_in = runner.invoke(cli, prefix + ["login", "--token", user.api_key])
        assert logged_in.exit_code == 0, logged_in.output
        assert json.loads(logged_in.stdout)["authentication_validated"] is True
        assert (cache / "hub_token").read_text(encoding="utf-8") == user.api_key
        uploaded = runner.invoke(cli, prefix + ["push", "alice/model", str(minimal_aeg_package.root)])
        assert uploaded.exit_code == 0, uploaded.output
        manifest = json.loads(uploaded.stdout)
        assert manifest["source"] == "server"
        assert manifest["persistence"] == "remote_service"
        assert manifest["targets"] == minimal_aeg_package.manifest.kernels.targets
        found = runner.invoke(cli, prefix + ["search", "alice/model"])
        assert found.exit_code == 0, found.output
        assert json.loads(found.stdout)[0]["model_id"] == "alice/model"
        downloaded = runner.invoke(cli, prefix + ["pull", "alice/model", str(tmp_path / "first-download")])
        assert downloaded.exit_code == 0, downloaded.output
    # A new server and client must recover from persisted storage, with no
    # shared in-memory upload cache from the first instance.
    restarted = AetherHubServer(storage)
    with _running_hub(restarted) as url:
        client = HubClient(url, auth_token=user.api_key, allow_local_cache=False)
        assert client.get_manifest("alice/model").content_hash == manifest["content_hash"]
        recovered = client.download("alice/model", tmp_path / "after-restart")
        from aether.core.aeg_format import load_aeg_package
        package = load_aeg_package(recovered)
        package.verify_integrity()
        assert (recovered / "graph" / "metadata.json").read_bytes() == (minimal_aeg_package.root / "graph" / "metadata.json").read_bytes()


def test_http_aliases_authorization_archive_validation_and_body_limit(minimal_aeg_package, tmp_path):
    httpx = pytest.importorskip("httpx")
    hub = AetherHubServer(tmp_path / "registry")
    alice = hub.create_user("alice", "alice@example.test", namespaces=["alice"])
    bob = hub.create_user("bob", "bob@example.test", namespaces=["bob"])
    viewer = hub.create_user("viewer", "viewer@example.test", role="viewer", namespaces=["alice"])
    archive = HubClient()._zip_directory(minimal_aeg_package.root)
    alice_headers = {"Authorization": f"Bearer {alice.api_key}"}
    bob_headers = {"Authorization": f"Bearer {bob.api_key}"}
    with _running_hub(hub, max_upload_bytes=len(archive) + 1000) as url:
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/v1/models").status_code == 401
            bad_hash = client.post("/v1/models/alice/model/upload", content=archive, headers={**alice_headers, "X-Content-Hash": "wrong"})
            assert bad_hash.status_code == 400
            invalid = client.post("/v1/models/alice/model/upload", content=b"invalid", headers=alice_headers)
            assert invalid.status_code == 400
            oversized = client.post("/v1/models/alice/model/upload", content=b"x" * (len(archive) + 1001), headers=alice_headers)
            assert oversized.status_code == 413
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as malicious:
                malicious.writestr("manifest.json", "{}")
                malicious.writestr("../escape.txt", "bad")
            assert client.post("/v1/models/alice/model/upload", content=buf.getvalue(), headers=alice_headers).status_code == 400
            forbidden = client.post("/v1/models/alice/model/upload", content=archive, headers=bob_headers)
            assert forbidden.status_code == 403
            assert client.post("/v1/models/alice/model/upload", content=archive, headers={"Authorization": f"Bearer {viewer.api_key}"}).status_code == 403
            legacy = client.post("/v1/hub/models/alice/model/push", content=archive, headers=alice_headers)
            assert legacy.status_code == 200, legacy.text
            assert client.get("/v1/hub/models/alice/model/pull", headers=alice_headers).content == archive
            assert client.get("/v1/models/alice/model/download", headers=bob_headers).status_code == 404
            assert client.get("/v1/models/search?q=model", headers=bob_headers).json()["models"] == []
            assert client.get("/v1/hub/models", headers=bob_headers).json()["models"] == []
            public = client.post("/v1/models/alice/model/upload?visibility=public", content=archive, headers=alice_headers)
            assert public.status_code == 200
            response = client.get("/v1/models/alice/model/download", headers=bob_headers)
            assert response.status_code == 200
            assert response.headers["x-content-hash"] == hashlib.sha256(response.content).hexdigest()
            assert client.delete("/v1/models/alice/model", headers=bob_headers).status_code == 403
            assert client.delete("/v1/hub/models/alice/model", headers=alice_headers).status_code == 200


def test_cli_failed_hub_login_preserves_existing_credentials(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    token_path = cache / "hub_token"
    token_path.write_text("previous-token", encoding="utf-8")
    def rejected(self, token):
        assert self.allow_local_cache is False
        assert self.auth_token == "previous-token"
        raise AuthenticationError("invalid token")
    monkeypatch.setattr(HubClient, "login", rejected)
    result = _runner().invoke(cli, ["--cache-dir", str(cache), "hub", "login", "--token", "invalid"])
    assert result.exit_code == 1
    assert token_path.read_text(encoding="utf-8") == "previous-token"


def test_new_hub_admin_uses_random_local_secret_and_preserves_accounts(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_HUB_ADMIN_TOKEN", raising=False)
    first = AetherHubServer(tmp_path / "first")
    second = AetherHubServer(tmp_path / "second")
    first_data = json.loads((first.storage.metadata_dir / "user_admin_001.json").read_text(encoding="utf-8"))
    second_data = json.loads((second.storage.metadata_dir / "user_admin_001.json").read_text(encoding="utf-8"))
    known_default = "aether_admin_" + hashlib.sha256(b"default_admin").hexdigest()[:32]
    assert first.authenticate(known_default) is None
    assert second.authenticate(known_default) is None
    assert first_data["api_key"] != second_data["api_key"]
    assert first.authenticate(first_data["api_key"]).role == "admin"
    monkeypatch.setenv("AETHER_HUB_ADMIN_TOKEN", "explicit-provisioned-token")
    restarted = AetherHubServer(tmp_path / "first")
    assert restarted.authenticate(first_data["api_key"]) is not None
    assert restarted.authenticate("explicit-provisioned-token") is None
    provisioned = AetherHubServer(tmp_path / "provisioned")
    assert provisioned.authenticate("explicit-provisioned-token").role == "admin"


def test_known_bootstrap_admin_token_does_not_authenticate_over_http(tmp_path, monkeypatch):
    monkeypatch.delenv("AETHER_HUB_ADMIN_TOKEN", raising=False)
    httpx = pytest.importorskip("httpx")
    hub = AetherHubServer(tmp_path / "registry")
    known_default = "aether_admin_" + hashlib.sha256(b"default_admin").hexdigest()[:32]
    with _running_hub(hub) as url:
        with httpx.Client(base_url=url, trust_env=False) as client:
            rejected = client.post("/auth/validate", json={"token": known_default}, headers={"Authorization": f"Bearer {known_default}"})
            assert rejected.status_code == 403
            assert known_default not in client.get("/health").text
