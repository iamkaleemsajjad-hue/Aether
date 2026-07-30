"""
Aether Hub client for uploading, downloading, and discovering AEG artifacts.

The Hub is a (future) public registry of compiled AEG artifacts. This client
provides the local end of that protocol: uploading compiled models, downloading
pre-compiled ones, and searching the catalog.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.constants import DEFAULT_HUB_URL, HUB_RETRY_ATTEMPTS, HUB_RETRY_BACKOFF_S
from aether.core.exceptions import AuthenticationError, HubError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


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
    def from_dict(data: dict[str, Any]) -> HubManifest:
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
    """Client for the Aether Hub API.

    The Hub is a remote registry of compiled AEG artifacts. This implementation
    provides a full API surface with local file-based simulation for development.
    """

    def __init__(self, hub_url: str = DEFAULT_HUB_URL, auth_token: str | None = None) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.auth_token = auth_token
        self._local_cache: dict[str, HubManifest] = {}
        logger.info("Hub client initialized", hub_url=self.hub_url, auth=auth_token is not None)

    def _headers(self) -> dict[str, str]:
        """Return HTTP headers including auth if configured."""
        headers: dict[str, str] = {"Content-Type": "application/json", "User-Agent": "aether-runtime/0.1.0"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _content_hash(self, data: bytes) -> str:
        """Compute the SHA-256 content hash of data."""
        return hashlib.sha256(data).hexdigest()

    def login(self, token: str) -> dict[str, Any]:
        """Authenticate with the Hub.

        In production this would call the Hub API. Local development stores the
        token in memory.
        """
        self.auth_token = token
        logger.info("Hub login successful")
        return {"status": "ok", "message": "Authentication token accepted"}

    def logout(self) -> dict[str, Any]:
        """Clear the authentication token."""
        self.auth_token = None
        return {"status": "ok", "message": "Logged out"}

    def search(self, query: str, limit: int = 20) -> list[HubManifest]:
        """Search the Hub for models matching a query."""
        query_lower = query.lower()
        results = [
            manifest for manifest in self._local_cache.values()
            if query_lower in manifest.model_id.lower()
        ]
        return sorted(results, key=lambda m: m.downloads, reverse=True)[:limit]

    def upload(self, model_id: str, package_path: str | Path) -> HubManifest:
        """Upload an AEG package to the Hub."""
        if not self.auth_token:
            msg = "Authentication required to upload to the Hub"
            raise AuthenticationError(msg)
        path = Path(package_path)
        if not path.exists():
            msg = f"Package path does not exist: {path}"
            raise HubError(msg)
        data = path.read_bytes() if path.is_file() else json.dumps({"dir": str(path)}).encode()
        content_hash = self._content_hash(data)
        manifest_path = path / "manifest.json" if path.is_dir() else None
        arch: dict[str, Any] = {}
        targets: list[str] = []
        if manifest_path and manifest_path.exists():
            try:
                mdata = json.loads(manifest_path.read_text())
                arch = mdata.get("architecture", {})
                targets = mdata.get("kernels", {}).get("targets", [])
            except Exception:
                pass
        manifest = HubManifest(
            model_id=model_id,
            aether_version="0.1.0",
            aeg_version="AEG/1.0",
            targets=targets,
            architecture=arch,
            file_size_bytes=len(data),
            content_hash=content_hash,
            uploaded_at="",  # Would be set by server
        )
        self._local_cache[model_id] = manifest
        logger.info("Uploaded model to Hub", model_id=model_id, size_bytes=len(data))
        return manifest

    def download(self, model_id: str, output_dir: str | Path) -> Path:
        """Download an AEG package from the Hub to a local directory.

        Falls back to a cached entry if the model was uploaded in-process.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest = self._local_cache.get(model_id)
        if manifest is None:
            msg = f"Model '{model_id}' not found on Hub"
            raise HubError(msg, model_id=model_id)
        package_file = output_path / f"{model_id.replace('/', '_')}.aeg"
        package_file.write_text(json.dumps(manifest.to_dict(), indent=2))
        logger.info("Downloaded model from Hub", model_id=model_id, dst=str(package_file))
        return package_file

    def list_models(self) -> list[HubManifest]:
        """List all models available on the Hub (local cache)."""
        return list(self._local_cache.values())

    def get_manifest(self, model_id: str) -> HubManifest | None:
        """Get the manifest for a specific model."""
        return self._local_cache.get(model_id)

    def __repr__(self) -> str:
        return f"HubClient(hub_url={self.hub_url}, auth={self.auth_token is not None})"
