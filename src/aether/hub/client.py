"""
Aether Hub client — real HTTP with retry/backoff + local cache simulation.

Two operational modes:
  1. **Real HTTP** — when ``hub_url`` resolves to a reachable server.
  2. **Local simulation** — when the server is unreachable or not deployed.
     The client transparently falls back to the in-memory local cache so
     upload/download/search work during development without a live Hub.

All HTTP calls use only stdlib ``urllib`` (no requests dependency).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.constants import AETHER_VERSION, DEFAULT_HUB_URL, HUB_RETRY_ATTEMPTS, HUB_RETRY_BACKOFF_S
from aether.core.exceptions import AuthenticationError, HubError
from aether.utils.file_io import safe_model_id_path
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a Hub archive without allowing traversal or symlink escapes."""
    root = destination.resolve()
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        candidate = (root / normalized).resolve()
        try:
            inside = candidate == root or candidate.is_relative_to(root)
        except AttributeError:  # Python 3.8 compatibility
            inside = str(candidate).startswith(str(root) + str(Path("/"))) or candidate == root
        if not inside or normalized.startswith("/"):
            raise HubError(f"unsafe archive member path: {info.filename!r}")
        # ZIP symlinks encode a Unix symlink mode in the external attributes.
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise HubError(f"symlink archive members are not permitted: {info.filename!r}")
    archive.extractall(root)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract a TAR archive without allowing path traversal, absolute paths, or symlinks."""
    root = destination.resolve()
    for member in archive.getmembers():
        # Reject absolute paths
        if member.name.startswith("/") or member.name.startswith("\\"):
            raise HubError(f"unsafe archive member (absolute path): {member.name!r}")
        # Reject path traversal
        normalized = member.name.replace("\\", "/")
        candidate = (root / normalized).resolve()
        try:
            inside = candidate == root or candidate.is_relative_to(root)
        except AttributeError:  # Python 3.8 compat
            inside = str(candidate).startswith(str(root)) or candidate == root
        if not inside:
            raise HubError(f"unsafe archive member (path traversal): {member.name!r}")
        # Reject symlinks and hardlinks
        if member.issym() or member.islnk():
            raise HubError(f"symlink/hardlink archive members are not permitted: {member.name!r}")
        # Reject device files
        if member.isdev():
            raise HubError(f"device file archive members are not permitted: {member.name!r}")
    archive.extractall(root)


@dataclass
class HubManifest:
    """Manifest for an AEG artifact on the Hub."""

    model_id: str
    aether_version: str
    aeg_version: str
    targets: list[str]
    architecture: dict[str, Any] = field(default_factory=dict)
    file_size_bytes: int = 0
    content_hash: str = ""
    uploaded_at: str = ""
    downloads: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "aether_version": self.aether_version,
            "aeg_version": self.aeg_version,
            "targets": self.targets,
            "architecture": self.architecture,
            "file_size_bytes": self.file_size_bytes,
            "content_hash": self.content_hash,
            "uploaded_at": self.uploaded_at,
            "downloads": self.downloads,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "HubManifest":
        return HubManifest(
            model_id=data["model_id"],
            aether_version=data.get("aether_version", ""),
            aeg_version=data.get("aeg_version", ""),
            targets=data.get("targets", []),
            architecture=data.get("architecture", {}),
            file_size_bytes=data.get("file_size_bytes", 0),
            content_hash=data.get("content_hash", ""),
            uploaded_at=data.get("uploaded_at", ""),
            downloads=data.get("downloads", 0),
        )


class HubClient:
    """
    Client for the Aether Hub API.

    HTTP retry logic: exponential back-off with jitter, up to
    ``HUB_RETRY_ATTEMPTS`` attempts (default 3).
    """

    def __init__(
        self,
        hub_url: str = DEFAULT_HUB_URL,
        auth_token: str | None = None,
        timeout_s: int = 30,
        allow_local_cache: bool = True,
    ) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout_s = timeout_s
        self.allow_local_cache = allow_local_cache
        self._local_cache: dict[str, HubManifest] = {}
        self._local_packages: dict[str, bytes] = {}
        logger.info("Hub client initialized", hub_url=self.hub_url, auth=auth_token is not None)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"aether-runtime/{AETHER_VERSION}",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request to the Hub API with exponential retry.

        Returns the parsed JSON response body.
        Raises HubError on non-2xx responses.
        """
        url = f"{self.hub_url}{path}"
        headers = self._headers()
        if extra_headers:
            headers.update(extra_headers)

        last_exc: Exception | None = None
        for attempt in range(HUB_RETRY_ATTEMPTS):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    expected_hash = resp.headers.get("X-Content-Hash")
                    if expected_hash and self._content_hash(raw) != expected_hash:
                        raise HubError(
                            f"Hub response hash mismatch for {path!r}: "
                            f"expected {expected_hash}, got {self._content_hash(raw)}"
                        )
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    msg = "Hub authentication failed: invalid or missing token"
                    raise AuthenticationError(msg) from exc
                if exc.code == 404:
                    msg = f"Hub resource not found: {path}"
                    raise HubError(msg) from exc
                if exc.code in (429, 503) and attempt < HUB_RETRY_ATTEMPTS - 1:
                    # Rate-limited or service unavailable: back off
                    backoff = HUB_RETRY_BACKOFF_S * (2 ** attempt)
                    logger.warning("Hub rate-limited (%d), backing off %.1fs", exc.code, backoff)
                    time.sleep(backoff)
                    last_exc = exc
                    continue
                msg = f"Hub API error {exc.code}: {exc.reason}"
                raise HubError(msg) from exc
            except (urllib.error.URLError, OSError) as exc:
                if attempt < HUB_RETRY_ATTEMPTS - 1:
                    backoff = HUB_RETRY_BACKOFF_S * (2 ** attempt)
                    logger.warning("Hub unreachable (%s), retry in %.1fs", exc, backoff)
                    time.sleep(backoff)
                    last_exc = exc
                    continue
                last_exc = exc
                break

        # All retries exhausted — return None to trigger local fallback
        logger.warning("Hub unreachable after %d attempts: %s", HUB_RETRY_ATTEMPTS, last_exc)
        raise HubError(f"Hub unreachable: {last_exc}", url=url) from last_exc

    def _is_hub_available(self) -> bool:
        """Return True if the Hub health endpoint responds."""
        try:
            req = urllib.request.Request(
                f"{self.hub_url}/health",
                headers={"User-Agent": f"aether-runtime/{AETHER_VERSION}"},
                method="GET",
            )
            # Health probing is only a capability check; it must not block
            # normal local-cache operations for the full artifact timeout.
            with urllib.request.urlopen(req, timeout=min(1.0, max(0.1, self.timeout_s))):
                return True
        except Exception:
            return False

    @staticmethod
    def _content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, token: str) -> dict[str, Any]:
        """
        Authenticate with the Hub.

        When Hub is reachable, validates the token against /auth/validate.
        Otherwise stores the token locally.
        """
        self.auth_token = token
        if self._is_hub_available():
            try:
                result = self._request("POST", "/auth/validate", body=json.dumps({"token": token}).encode())
                logger.info("Hub login validated by server")
                return result or {"status": "ok", "source": "server"}
            except Exception as exc:
                logger.debug("Hub login validation failed: %s", exc)
        if self.allow_local_cache:
            logger.info("Hub login stored locally (Hub offline)")
            return {"status": "ok", "message": "Token stored locally", "source": "local_cache"}
        raise HubError("Hub is unavailable; token was not validated", url=self.hub_url)

    def logout(self) -> dict[str, Any]:
        """Clear the authentication token."""
        self.auth_token = None
        return {"status": "ok", "message": "Logged out"}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[HubManifest]:
        """
        Search the Hub for models matching a query.

        Tries remote API first; falls back to local cache search.
        """
        if self._is_hub_available():
            try:
                params = urllib.parse.urlencode({"q": query, "limit": limit})
                data = self._request("GET", f"/v1/models/search?{params}")
                if data.get("models"):
                    return [HubManifest.from_dict(m) for m in data["models"]]
            except HubError:
                pass

        if not self.allow_local_cache:
            return []

        # Explicit local cache mode.
        query_lower = query.lower()
        results = [
            m for m in self._local_cache.values()
            if query_lower in m.model_id.lower()
        ]
        return sorted(results, key=lambda m: m.downloads, reverse=True)[:limit]

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(self, model_id: str, package_path: str | Path) -> HubManifest:
        """
        Upload an AEG package to the Hub.

        The package directory is ZIP-archived before upload. If the Hub is
        unreachable, the manifest is stored in the local cache only.
        """
        if not self.auth_token:
            msg = "Authentication required to upload to the Hub"
            raise AuthenticationError(msg)
        path = Path(package_path)
        if not path.exists():
            msg = f"Package path does not exist: {path}"
            raise HubError(msg, url=str(path))

        # Collect binary data
        if path.is_dir():
            data = self._zip_directory(path)
        else:
            data = path.read_bytes()

        content_hash = self._content_hash(data)
        arch: dict[str, Any] = {}
        targets: list[str] = []

        # Read manifest for metadata
        manifest_json = path / "manifest.json" if path.is_dir() else None
        if manifest_json and manifest_json.exists():
            try:
                mdata = json.loads(manifest_json.read_text(encoding="utf-8"))
                arch = mdata.get("architecture", {})
                targets = mdata.get("kernels", {}).get("targets", [])
            except Exception:
                pass

        aether_version = "unknown"
        aeg_version = "unknown"
        if manifest_json and manifest_json.exists():
            try:
                mdata = json.loads(manifest_json.read_text(encoding="utf-8"))
                aether_version = str(mdata.get("aether_version") or mdata.get("compiler_version") or "unknown")
                aeg_version = str(mdata.get("format_version") or "unknown")
            except Exception:
                pass

        manifest = HubManifest(
            model_id=model_id,
            aether_version=aether_version,
            aeg_version=aeg_version,
            targets=targets,
            architecture=arch,
            file_size_bytes=len(data),
            content_hash=content_hash,
            uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Try remote upload
        if self._is_hub_available():
            try:
                upload_headers = {
                    "Content-Type": "application/octet-stream",
                    "X-Model-Id": model_id,
                    "X-Content-Hash": content_hash,
                }
                resp = self._request(
                    "POST",
                    f"/v1/models/{urllib.parse.quote(model_id, safe='')}/upload",
                    body=data,
                    extra_headers=upload_headers,
                )
                if resp.get("manifest"):
                    manifest = HubManifest.from_dict(resp["manifest"])
                logger.info("Uploaded model to Hub (remote)", model_id=model_id)
            except HubError as exc:
                logger.warning("Remote Hub upload failed: %s — stored locally", exc)

        self._local_cache[model_id] = manifest
        self._local_packages[model_id] = data
        logger.info("Model cached locally", model_id=model_id, size_bytes=len(data))
        return manifest

    def _zip_directory(self, directory: Path) -> bytes:
        """Create a ZIP archive of an AEG package directory in memory."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(directory.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(directory)
                    zf.write(file_path, arcname)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, model_id: str, output_dir: str | Path) -> Path:
        """
        Download an AEG package from the Hub to a local directory.

        When Hub is reachable, fetches the ZIP archive and extracts it.
        Falls back to the local cache manifest if unreachable.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_id = safe_model_id_path(model_id)
        dest_dir = output_path / safe_id

        if self._is_hub_available():
            try:
                encoded_id = urllib.parse.quote(model_id, safe="")
                req = urllib.request.Request(
                    f"{self.hub_url}/v1/models/{encoded_id}/download",
                    headers=self._headers(),
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    # Try to unzip; otherwise write as a raw file
                    try:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                            _safe_extract_zip(zf, dest_dir)
                        logger.info("Downloaded & extracted AEG from Hub", model_id=model_id, dst=str(dest_dir))
                        return dest_dir
                    except zipfile.BadZipFile:
                        # Write raw bytes
                        raw_path = output_path / f"{safe_id}.aeg"
                        raw_path.write_bytes(raw)
                        logger.info("Downloaded AEG from Hub (raw)", model_id=model_id, dst=str(raw_path))
                        return raw_path
            except Exception as exc:
                logger.warning("Hub download failed: %s — falling back to local cache", exc)

        # Local cache fallback.  Metadata alone is not a model artifact: only
        # a retained uploaded package archive may be downloaded locally.
        manifest = self._local_cache.get(model_id)
        package_data = self._local_packages.get(model_id)
        if manifest is None or package_data is None:
            msg = f"Model '{model_id}' not found on Hub or in local cache"
            raise HubError(msg)
        try:
            if manifest.content_hash and self._content_hash(package_data) != manifest.content_hash:
                raise HubError(f"local Hub artifact hash mismatch for {model_id!r}")
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(package_data)) as zf:
                _safe_extract_zip(zf, dest_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            raise HubError(f"local Hub payload for {model_id!r} is not a valid AEG archive") from exc
        logger.info("Downloaded model package from local cache", model_id=model_id, dst=str(dest_dir))
        return dest_dir

    # ------------------------------------------------------------------
    # Listing & manifest queries
    # ------------------------------------------------------------------

    def list_models(self) -> list[HubManifest]:
        """List all models available on the Hub (remote first, then local)."""
        if self._is_hub_available():
            try:
                data = self._request("GET", "/v1/models")
                if data.get("models"):
                    return [HubManifest.from_dict(m) for m in data["models"]]
            except HubError:
                pass
        return list(self._local_cache.values()) if self.allow_local_cache else []

    def get_manifest(self, model_id: str) -> HubManifest | None:
        """Get the manifest for a specific model (remote then local)."""
        if self._is_hub_available():
            try:
                encoded_id = urllib.parse.quote(model_id, safe="")
                data = self._request("GET", f"/v1/models/{encoded_id}")
                if data.get("model_id"):
                    return HubManifest.from_dict(data)
            except HubError:
                pass
        return self._local_cache.get(model_id) if self.allow_local_cache else None

    def delete(self, model_id: str) -> dict[str, Any]:
        """Delete a model from the Hub (requires auth)."""
        if not self.auth_token:
            msg = "Authentication required to delete from Hub"
            raise AuthenticationError(msg)
        self._local_cache.pop(model_id, None)
        self._local_packages.pop(model_id, None)
        if self._is_hub_available():
            try:
                encoded_id = urllib.parse.quote(model_id, safe="")
                return self._request("DELETE", f"/v1/models/{encoded_id}")
            except HubError as exc:
                logger.warning("Remote Hub delete failed: %s", exc)
        return {"status": "ok", "model": model_id}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return Hub client statistics."""
        return {
            "hub_url": self.hub_url,
            "auth_configured": self.auth_token is not None,
            "local_cache_count": len(self._local_cache),
            "hub_available": self._is_hub_available(),
        }

    def __repr__(self) -> str:
        return f"HubClient(hub_url={self.hub_url}, auth={self.auth_token is not None})"
