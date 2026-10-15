"""
Aether Hub client — real HTTP with retry/backoff + local cache simulation.

Two operational modes:
  1. **Real HTTP** — when ``hub_url`` resolves to a reachable server.
  2. **Explicit local-cache fallback** — ``allow_local_cache=True`` preserves
     development workflows in this client instance. It is memory-only and is
     never evidence that an artifact was published remotely.

All HTTP calls use only stdlib ``urllib`` (no requests dependency).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
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
    source: str = "unknown"
    persistence: str = "unknown"

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
            "source": self.source,
            "persistence": self.persistence,
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
            source=data.get("source", "unknown"),
            persistence=data.get("persistence", "unknown"),
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
        self.last_operation: dict[str, Any] = {}
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
                    result = json.loads(raw.decode("utf-8"))
                    if not isinstance(result, dict):
                        raise HubError(f"Hub response for {path!r} must be a JSON object", url=url)
                    return result
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
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
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise HubError(f"invalid Hub JSON response for {path!r}: {exc}", url=url) from exc

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
        previous_token = self.auth_token
        self.auth_token = token
        if not self.allow_local_cache or self._is_hub_available():
            try:
                result = self._request("POST", "/auth/validate", body=json.dumps({"token": token}).encode())
                if result.get("status") in {"failed", "error", "denied"} or result.get("authentication_validated") is not True:
                    raise AuthenticationError("Hub did not explicitly validate the authentication token")
                logger.info("Hub login validated by server")
                self.last_operation = {"source": "server", "authentication_validated": True}
                return {**result, "status": "ok", **self.last_operation}
            except AuthenticationError:
                self.auth_token = previous_token
                raise
            except HubError as exc:
                if not self.allow_local_cache:
                    self.auth_token = previous_token
                    raise
                logger.debug("Hub login validation failed: %s", exc)
        if self.allow_local_cache:
            logger.info("Hub login stored locally (Hub offline)")
            self.last_operation = {"source": "local_cache", "persistence": "client_instance_memory", "authentication_validated": False}
            return {"status": "ok", "message": "Token stored in this client only; remote authentication was not validated", **self.last_operation}
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
        if not self.allow_local_cache or self._is_hub_available():
            try:
                params = urllib.parse.urlencode({"q": query, "limit": limit})
                data = self._request("GET", f"/v1/models/search?{params}")
                rows = data.get("models", [])
                if not isinstance(rows, list) or not all(isinstance(row, dict) and "model_id" in row for row in rows):
                    raise HubError("Hub search response has malformed models")
                self.last_operation = {"source": "server", "persistence": "remote_service"}
                return [HubManifest.from_dict({**m, **self.last_operation}) for m in rows]
            except AuthenticationError:
                raise
            except HubError:
                if not self.allow_local_cache:
                    raise

        # Explicit local cache mode.
        query_lower = query.lower()
        self.last_operation = {"source": "local_cache", "persistence": "client_instance_memory"}
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
        if not self.allow_local_cache or self._is_hub_available():
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
                if resp.get("status") in {"failed", "error", "denied"}:
                    raise HubError("Hub rejected the artifact upload")
                response_manifest = resp.get("manifest")
                if (
                    not isinstance(response_manifest, dict)
                    or response_manifest.get("model_id") != model_id
                    or response_manifest.get("content_hash") != content_hash
                    or type(response_manifest.get("file_size_bytes")) is not int
                    or response_manifest["file_size_bytes"] != len(data)
                ):
                    raise HubError("Hub upload did not acknowledge the exact model ID, content hash and byte count")
                manifest = HubManifest.from_dict(response_manifest)
                manifest.source = "server"
                manifest.persistence = "remote_service"
                self.last_operation = {"source": manifest.source, "persistence": manifest.persistence}
                self._local_cache[model_id] = manifest
                self._local_packages[model_id] = data
                logger.info("Uploaded model to Hub (remote)", model_id=model_id)
                return manifest
            except AuthenticationError:
                raise
            except HubError as exc:
                if not self.allow_local_cache:
                    raise
                logger.warning("Remote Hub upload failed: %s — stored locally", exc)

        manifest.source = "local_cache"
        manifest.persistence = "client_instance_memory"
        self.last_operation = {"source": manifest.source, "persistence": manifest.persistence}
        self._local_cache[model_id] = manifest
        self._local_packages[model_id] = data
        logger.info("Model cached locally", model_id=model_id, size_bytes=len(data))
        return manifest

    def _zip_directory(self, directory: Path) -> bytes:
        """Create a ZIP archive of an AEG package directory in memory."""
        root = directory.resolve()
        files: list[Path] = []
        pending = [directory]
        while pending:
            path = pending.pop()
            attributes = path.lstat()
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if (
                path.is_symlink()
                or getattr(attributes, "st_file_attributes", 0) & reparse_flag
                or not path.resolve().is_relative_to(root)
            ):
                raise HubError(f"Package archive cannot include symlinks, junctions or paths outside its root: {path}")
            if stat.S_ISDIR(attributes.st_mode):
                pending.extend(path.iterdir())
            elif stat.S_ISREG(attributes.st_mode):
                files.append(path)
            else:
                raise HubError(f"Package archive cannot include special files: {path}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(files):
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

        if not self.allow_local_cache or self._is_hub_available():
            try:
                encoded_id = urllib.parse.quote(model_id, safe="")
                req = urllib.request.Request(
                    f"{self.hub_url}/v1/models/{encoded_id}/download",
                    headers=self._headers(),
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    expected_hash = resp.headers.get("X-Content-Hash")
                    if expected_hash and self._content_hash(raw) != expected_hash:
                        raise HubError(f"Hub artifact hash mismatch for {model_id!r}")
                    # Try to unzip; otherwise write as a raw file
                    try:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                            _safe_extract_zip(zf, dest_dir)
                        logger.info("Downloaded & extracted AEG from Hub", model_id=model_id, dst=str(dest_dir))
                        self.last_operation = {"source": "server", "persistence": "local_filesystem", "path": str(dest_dir)}
                        return dest_dir
                    except zipfile.BadZipFile:
                        # Write raw bytes
                        raw_path = output_path / f"{safe_id}.aeg"
                        raw_path.write_bytes(raw)
                        logger.info("Downloaded AEG from Hub (raw)", model_id=model_id, dst=str(raw_path))
                        self.last_operation = {"source": "server", "persistence": "local_filesystem", "path": str(raw_path)}
                        return raw_path
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
                    raise AuthenticationError("Hub authentication failed: invalid or missing token") from exc
                if not self.allow_local_cache:
                    if isinstance(exc, HubError):
                        raise
                    raise HubError(f"Hub download failed: {exc}", url=self.hub_url) from exc
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
        self.last_operation = {"source": "local_cache", "persistence": "local_filesystem", "path": str(dest_dir)}
        return dest_dir

    # ------------------------------------------------------------------
    # Listing & manifest queries
    # ------------------------------------------------------------------

    def list_models(self) -> list[HubManifest]:
        """List all models available on the Hub (remote first, then local)."""
        if not self.allow_local_cache or self._is_hub_available():
            try:
                data = self._request("GET", "/v1/models")
                rows = data.get("models", [])
                if not isinstance(rows, list) or not all(isinstance(row, dict) and "model_id" in row for row in rows):
                    raise HubError("Hub model listing has malformed models")
                self.last_operation = {"source": "server", "persistence": "remote_service"}
                return [HubManifest.from_dict({**m, **self.last_operation}) for m in rows]
            except AuthenticationError:
                raise
            except HubError:
                if not self.allow_local_cache:
                    raise
        return list(self._local_cache.values()) if self.allow_local_cache else []

    def get_manifest(self, model_id: str) -> HubManifest | None:
        """Get the manifest for a specific model (remote then local)."""
        if not self.allow_local_cache or self._is_hub_available():
            try:
                encoded_id = urllib.parse.quote(model_id, safe="")
                data = self._request("GET", f"/v1/models/{encoded_id}")
                if data.get("model_id"):
                    self.last_operation = {"source": "server", "persistence": "remote_service"}
                    return HubManifest.from_dict({**data, **self.last_operation})
                raise HubError("Hub model metadata lacks model_id")
            except AuthenticationError:
                raise
            except HubError:
                if not self.allow_local_cache:
                    raise
        return self._local_cache.get(model_id) if self.allow_local_cache else None

    def delete(self, model_id: str) -> dict[str, Any]:
        """Delete a model from the Hub (requires auth)."""
        if not self.auth_token:
            msg = "Authentication required to delete from Hub"
            raise AuthenticationError(msg)
        if not self.allow_local_cache or self._is_hub_available():
            try:
                encoded_id = urllib.parse.quote(model_id, safe="")
                result = self._request("DELETE", f"/v1/models/{encoded_id}")
                self._local_cache.pop(model_id, None)
                self._local_packages.pop(model_id, None)
                self.last_operation = {"source": "server", "persistence": "remote_service"}
                return {**result, **self.last_operation}
            except AuthenticationError:
                raise
            except HubError as exc:
                if not self.allow_local_cache:
                    raise
                logger.warning("Remote Hub delete failed: %s", exc)
        self._local_cache.pop(model_id, None)
        self._local_packages.pop(model_id, None)
        self.last_operation = {"source": "local_cache", "persistence": "client_instance_memory"}
        return {"status": "ok", "model": model_id, **self.last_operation}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _kv_path(namespace: str, key: str) -> str:
        import re
        if not all(isinstance(part, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part)
                   for part in (namespace, key)):
            raise ValueError("Cache namespace/key require bounded letters, digits, dots, underscores or hyphens")
        return f"/v1/kv/{namespace}/{key}"

    def put_kv(self, namespace: str, key: str, payload: Any) -> dict[str, Any]:
        """Explicitly upload real cache bytes; never use the model-cache fallback."""
        from aether.runtime.kv_payload import CachePayload
        if not self.auth_token:
            raise AuthenticationError("Authentication required for private Hub cache snapshots")
        if not isinstance(payload, CachePayload):
            raise TypeError("put_kv requires a CachePayload")
        data = payload.serialize()
        digest = self._content_hash(data)
        result = self._request("PUT", self._kv_path(namespace, key), data,
                               {"Content-Type": "application/vnd.aether.cache", "X-Content-Hash": digest})
        expected = {"namespace": namespace, "key": key, "content_hash": digest, "size_bytes": len(data),
                    "tensor_bytes": payload.byte_size, "model_fingerprint": payload.model_fingerprint,
                    "cache_kind": payload.cache_kind, "logical_length": payload.logical_length}
        if any(result.get(field) != value or (isinstance(value, int) and type(result.get(field)) is not int)
               for field, value in expected.items()):
            raise HubError("Hub cache acknowledgement does not match the uploaded snapshot")
        self.last_operation = {"source": "server", "persistence": "remote_service", "transport": "http",
                               "uploaded_bytes": len(data), "tensor_bytes": payload.byte_size}
        return {**result, **self.last_operation}

    def get_kv(self, namespace: str, key: str, *, expected_model_fingerprint: str | None = None,
               expected_cache_kind: str | None = None, max_bytes: int = 2 * 1024**3) -> Any:
        """Download and verify a private typed snapshot, bounded before allocation."""
        from aether.runtime.kv_payload import CachePayload, MAX_PAYLOAD_BYTES
        if not self.auth_token:
            raise AuthenticationError("Authentication required for private Hub cache snapshots")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PAYLOAD_BYTES:
            raise ValueError("max_bytes must be a positive bounded integer")
        path = self._kv_path(namespace, key)
        try:
            request = urllib.request.Request(f"{self.hub_url}{path}", headers=self._headers(), method="GET")
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                length = response.headers.get("Content-Length")
                if length is not None and (int(length) < 0 or int(length) > max_bytes):
                    raise HubError("Hub cache response exceeds the configured read bound")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise HubError("Hub cache response exceeds the configured read bound")
                digest = response.headers.get("X-Content-Hash")
                if digest is None or self._content_hash(data) != digest:
                    raise HubError("Hub cache response checksum is missing or invalid")
            payload = CachePayload.deserialize(data, expected_model_fingerprint=expected_model_fingerprint,
                                                expected_cache_kind=expected_cache_kind, max_bytes=max_bytes)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AuthenticationError("Hub cache authentication failed") from exc
            raise HubError(f"Hub cache download failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise HubError(f"Hub cache download/integrity validation failed: {exc}") from exc
        self.last_operation = {"source": "server", "persistence": "remote_service", "transport": "http",
                               "downloaded_bytes": len(data), "tensor_bytes": payload.byte_size}
        return payload

    def delete_kv(self, namespace: str, key: str) -> bool:
        if not self.auth_token:
            raise AuthenticationError("Authentication required for private Hub cache snapshots")
        result = self._request("DELETE", self._kv_path(namespace, key))
        if result.get("namespace") != namespace or result.get("key") != key or type(result.get("deleted")) is not bool:
            raise HubError("Hub cache deletion acknowledgement is malformed")
        return result["deleted"]

    def list_kv(self, namespace: str) -> list[dict[str, Any]]:
        if not self.auth_token:
            raise AuthenticationError("Authentication required for private Hub cache snapshots")
        path = self._kv_path(namespace, "listing").rsplit("/", 1)[0]
        result = self._request("GET", path)
        entries = result.get("entries")
        if result.get("namespace") != namespace or not isinstance(entries, list) or not all(isinstance(row, dict) for row in entries):
            raise HubError("Hub cache listing is malformed")
        return entries

    def stats(self) -> dict[str, Any]:
        """Return Hub client statistics."""
        return {
            "hub_url": self.hub_url,
            "auth_configured": self.auth_token is not None,
            "local_cache_count": len(self._local_cache),
            "hub_available": self._is_hub_available(),
            "local_cache_persistence": "client_instance_memory",
            "allow_local_cache": self.allow_local_cache,
            "last_operation": dict(self.last_operation),
        }

    def __repr__(self) -> str:
        return f"HubClient(hub_url={self.hub_url}, auth={self.auth_token is not None})"
