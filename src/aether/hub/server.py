"""
Aether Runtime — Production Hub Server.

A real AEG model registry with HTTP API for model storage, retrieval,
versioning, access control, and content-addressed deduplication.

Features:
  - Content-addressed artifact storage (SHA-256 keyed)
  - Multi-tenant namespaces with role-based access control
  - Model versioning with semantic version tags
  - AEG integrity verification on push/pull
  - Deduplication: identical content stores only one copy
  - Search and filtering by architecture, precision, target
  - Real HTTP server with FastAPI

Research basis:
  - Docker Registry HTTP API V2 (OCI Distribution Spec)
  - HuggingFace Hub API design (2024)
  - Aether Hub specification (PRD v3.1, v4.0)
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.constants import AETHER_VERSION


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HubModelVersion:
    """A specific version of a model in the Hub."""

    version_id: str
    tag: str
    content_hash: str  # SHA-256 of the artifact ZIP
    size_bytes: int
    created_at: float
    pushed_by: str
    description: str = ""
    architecture: str = ""
    format_version: str = ""
    hardware_targets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "tag": self.tag,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "pushed_by": self.pushed_by,
            "description": self.description,
            "architecture": self.architecture,
            "format_version": self.format_version,
            "hardware_targets": self.hardware_targets,
            "metadata": self.metadata,
        }


@dataclass
class HubModel:
    """A model entry in the Hub."""

    model_id: str
    namespace: str
    name: str
    created_at: float
    created_by: str
    description: str = ""
    visibility: str = "private"  # "public" | "private" | "team"
    tags: list[str] = field(default_factory=list)
    versions: list[HubModelVersion] = field(default_factory=list)
    download_count: int = 0
    like_count: int = 0

    @property
    def full_name(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def latest_version(self) -> HubModelVersion | None:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "namespace": self.namespace,
            "name": self.name,
            "full_name": self.full_name,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "description": self.description,
            "visibility": self.visibility,
            "tags": self.tags,
            "versions": [v.to_dict() for v in self.versions],
            "download_count": self.download_count,
            "like_count": self.like_count,
            "latest_version": self.latest_version.to_dict() if self.latest_version else None,
        }


@dataclass
class HubUser:
    """A Hub user account."""

    user_id: str
    username: str
    email: str
    api_key: str
    created_at: float
    role: str = "user"  # "admin" | "user" | "viewer"
    namespaces: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "role": self.role,
            "namespaces": self.namespaces,
        }


# ---------------------------------------------------------------------------
# Hub storage backend
# ---------------------------------------------------------------------------

class HubStorageBackend:
    """
    Content-addressed storage backend for AEG artifacts.

    Implements content deduplication: if two models have identical content,
    only one copy is stored. This is the same approach used by Docker Registry
    (OCI Distribution Spec) and git's object store.
    """

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self.blobs_dir = storage_root / "blobs"
        self.metadata_dir = storage_root / "metadata"
        self.index_path = storage_root / "index.json"

        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, Any] = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text())
            except Exception:  # noqa: BLE001
                pass
        return {"models": {}, "content_hashes": {}, "users": {}}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def store_blob(self, data: bytes) -> str:
        """Store a blob and return its content hash."""
        content_hash = hashlib.sha256(data).hexdigest()
        blob_path = self.blobs_dir / content_hash[:2] / content_hash[2:]
        if not blob_path.exists():
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(data)
        return content_hash

    def load_blob(self, content_hash: str) -> bytes | None:
        """Load a blob by its content hash."""
        blob_path = self.blobs_dir / content_hash[:2] / content_hash[2:]
        if blob_path.exists():
            return blob_path.read_bytes()
        return None

    def has_blob(self, content_hash: str) -> bool:
        """Check if a blob exists."""
        blob_path = self.blobs_dir / content_hash[:2] / content_hash[2:]
        return blob_path.exists()

    def store_model(self, model: HubModel) -> None:
        """Persist model metadata."""
        meta_path = self.metadata_dir / f"{model.model_id}.json"
        meta_path.write_text(json.dumps(model.to_dict(), indent=2))
        self._index.setdefault("models", {})[model.model_id] = {
            "namespace": model.namespace,
            "name": model.name,
            "full_name": model.full_name,
            "visibility": model.visibility,
            "tags": model.tags,
        }
        self._save_index()

    def load_model(self, model_id: str) -> HubModel | None:
        """Load model metadata by ID."""
        meta_path = self.metadata_dir / f"{model_id}.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        model = HubModel(
            model_id=data["model_id"],
            namespace=data["namespace"],
            name=data["name"],
            created_at=data["created_at"],
            created_by=data["created_by"],
            description=data.get("description", ""),
            visibility=data.get("visibility", "private"),
            tags=data.get("tags", []),
            download_count=data.get("download_count", 0),
            like_count=data.get("like_count", 0),
        )
        for vdata in data.get("versions", []):
            version = HubModelVersion(
                version_id=vdata["version_id"],
                tag=vdata["tag"],
                content_hash=vdata["content_hash"],
                size_bytes=vdata["size_bytes"],
                created_at=vdata["created_at"],
                pushed_by=vdata["pushed_by"],
                description=vdata.get("description", ""),
                architecture=vdata.get("architecture", ""),
                format_version=vdata.get("format_version", ""),
                hardware_targets=vdata.get("hardware_targets", []),
                metadata=vdata.get("metadata", {}),
            )
            model.versions.append(version)
        return model

    def find_model_by_name(self, namespace: str, name: str) -> HubModel | None:
        """Find a model by namespace/name."""
        for model_id, info in self._index.get("models", {}).items():
            if info.get("namespace") == namespace and info.get("name") == name:
                return self.load_model(model_id)
        return None

    def search_models(
        self,
        query: str | None = None,
        namespace: str | None = None,
        tags: list[str] | None = None,
        architecture: str | None = None,
        visibility: str | None = None,
        limit: int = 50,
    ) -> list[HubModel]:
        """Search models by various criteria."""
        results: list[HubModel] = []
        for model_id, info in self._index.get("models", {}).items():
            if visibility and info.get("visibility") != visibility:
                continue
            if namespace and info.get("namespace") != namespace:
                continue
            model = self.load_model(model_id)
            if model is None:
                continue
            if query and query.lower() not in f"{model.full_name} {model.description}".lower():
                continue
            if tags and not any(t in model.tags for t in tags):
                continue
            results.append(model)
        return results[:limit]

    def store_user(self, user: HubUser) -> None:
        """Persist user account."""
        user_path = self.metadata_dir / f"user_{user.user_id}.json"
        user_data = {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "api_key": user.api_key,
            "created_at": user.created_at,
            "role": user.role,
            "namespaces": user.namespaces,
        }
        # API keys belong only in the administrator-controlled storage tree.
        # POSIX creates this file with owner access; Windows access control is
        # inherited from that tree and must be configured by its administrator.
        descriptor = os.open(user_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(user_data, stream, indent=2)
        if os.name != "nt":
            user_path.chmod(0o600)
        self._index.setdefault("users", {})[user.user_id] = {
            "username": user.username,
            "api_key_prefix": user.api_key[:8],
        }
        self._save_index()

    def find_user_by_api_key(self, api_key: str) -> HubUser | None:
        """Find a user by their API key."""
        for user_id in self._index.get("users", {}):
            user_path = self.metadata_dir / f"user_{user_id}.json"
            if user_path.exists():
                data = json.loads(user_path.read_text())
                if data.get("api_key") == api_key:
                    return HubUser(
                        user_id=data["user_id"],
                        username=data["username"],
                        email=data["email"],
                        api_key=data["api_key"],
                        created_at=data["created_at"],
                        role=data.get("role", "user"),
                        namespaces=data.get("namespaces", []),
                    )
        return None


# ---------------------------------------------------------------------------
# Hub server
# ---------------------------------------------------------------------------

class AetherHubServer:
    """
    Production Aether Hub server.

    Manages model registry, authentication, versioning, and content-addressed
    storage. Designed to support the full PRD Hub spec including:
    - Push/pull AEG artifacts
    - Model discovery and search
    - Content deduplication
    - Multi-tenant namespaces
    - Role-based access control
    """

    def __init__(self, storage_root: str | Path | None = None) -> None:
        if storage_root is None:
            default = Path.home() / ".aether" / "hub_storage"
            storage_root = default
        self.storage = HubStorageBackend(Path(storage_root))
        self._initialize_default_admin()

    def _initialize_default_admin(self) -> None:
        """Bootstrap a locally stored administrator only for new storage.

        An explicit AETHER_HUB_ADMIN_TOKEN can provision the secret. Otherwise
        a random key is persisted in metadata/user_admin_001.json for local
        administrator retrieval. Existing accounts are never replaced and the
        credential is never logged or included in public HTTP responses.
        """
        if not self.storage._index.get("users"):
            configured_token = os.environ.get("AETHER_HUB_ADMIN_TOKEN")
            if configured_token is not None and not configured_token.strip():
                raise ValueError("AETHER_HUB_ADMIN_TOKEN must not be empty")
            admin = HubUser(
                user_id="admin_001",
                username="admin",
                email="admin@aether.local",
                api_key=configured_token or "aether_admin_" + secrets.token_urlsafe(32),
                created_at=time.time(),
                role="admin",
                namespaces=["public", "admin"],
            )
            self.storage.store_user(admin)

    def authenticate(self, api_key: str) -> HubUser | None:
        """Authenticate a request by API key."""
        return self.storage.find_user_by_api_key(api_key)

    def push_model(
        self,
        namespace: str,
        name: str,
        tag: str,
        artifact_data: bytes,
        pushed_by: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> HubModelVersion:
        """
        Push an AEG artifact to the Hub.

        Validates the artifact, computes its content hash, deduplicates storage,
        and registers the version.
        """
        # Validate artifact (ensure it's a valid ZIP/AEG)
        import zipfile
        import io
        try:
            with zipfile.ZipFile(io.BytesIO(artifact_data)) as zf:
                names = zf.namelist()
        except Exception as exc:
            msg = f"Invalid artifact: not a valid ZIP/AEG archive: {exc}"
            raise ValueError(msg) from exc

        # Check for path traversal in ZIP
        for name_in_zip in names:
            norm = Path(name_in_zip)
            parts = norm.parts
            if ".." in parts or name_in_zip.startswith("/"):
                msg = f"Archive contains path traversal entry: {name_in_zip}"
                raise ValueError(msg)

        # Store content (deduplicated)
        content_hash = self.storage.store_blob(artifact_data)

        # Extract metadata from manifest if present
        arch = ""
        fmt_version = ""
        hw_targets: list[str] = []
        meta = metadata or {}

        try:
            with zipfile.ZipFile(io.BytesIO(artifact_data)) as zf:
                if "manifest.json" in zf.namelist():
                    manifest_data = json.loads(zf.read("manifest.json").decode())
                    arch = manifest_data.get("architecture", {}).get("family", "")
                    fmt_version = manifest_data.get("format_version", "")
                    kernel_data = manifest_data.get("kernels", {})
                    hw_targets = list(kernel_data.get("targets", [])) if isinstance(kernel_data.get("targets"), list) else list(kernel_data.keys())
                    meta.update({
                        "manifest_keys": list(manifest_data.keys()),
                        "manifest_architecture": manifest_data.get("architecture", {}),
                        "aether_version": manifest_data.get("aether_version", "unknown"),
                    })
        except Exception:  # noqa: BLE001
            pass

        # Find or create model entry
        model = self.storage.find_model_by_name(namespace, name)
        if model is None:
            model = HubModel(
                model_id=str(uuid.uuid4()),
                namespace=namespace,
                name=name,
                created_at=time.time(),
                created_by=pushed_by,
                description=description,
            )

        # Create version
        version = HubModelVersion(
            version_id=str(uuid.uuid4()),
            tag=tag,
            content_hash=content_hash,
            size_bytes=len(artifact_data),
            created_at=time.time(),
            pushed_by=pushed_by,
            description=description,
            architecture=arch,
            format_version=fmt_version,
            hardware_targets=hw_targets,
            metadata=meta,
        )
        model.versions.append(version)
        self.storage.store_model(model)
        return version

    def pull_model(
        self,
        namespace: str,
        name: str,
        tag: str = "latest",
    ) -> bytes:
        """
        Pull an AEG artifact from the Hub.

        Returns the raw artifact bytes. Raises ValueError if not found.
        """
        model = self.storage.find_model_by_name(namespace, name)
        if model is None:
            msg = f"Model {namespace}/{name} not found"
            raise ValueError(msg)

        version: HubModelVersion | None = None
        if tag == "latest":
            version = model.latest_version
        else:
            for v in model.versions:
                if v.tag == tag:
                    version = v
                    break

        if version is None:
            msg = f"Version {tag!r} not found for {namespace}/{name}"
            raise ValueError(msg)

        data = self.storage.load_blob(version.content_hash)
        if data is None:
            msg = f"Artifact data missing for {namespace}/{name}:{tag} (content_hash={version.content_hash})"
            raise ValueError(msg)

        # Verify content integrity
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != version.content_hash:
            msg = f"Content hash mismatch: expected {version.content_hash}, got {actual_hash}"
            raise ValueError(msg)

        model.download_count += 1
        self.storage.store_model(model)

        return data

    def get_model_info(self, namespace: str, name: str) -> dict[str, Any] | None:
        """Get model metadata without downloading the artifact."""
        model = self.storage.find_model_by_name(namespace, name)
        if model is None:
            return None
        return model.to_dict()

    def search(
        self,
        query: str | None = None,
        namespace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search the Hub for models."""
        models = self.storage.search_models(
            query=query,
            namespace=namespace,
            tags=tags,
            limit=limit,
        )
        return [m.to_dict() for m in models]

    def delete_model(self, namespace: str, name: str, tag: str | None = None) -> bool:
        """Delete a model or a specific version."""
        model = self.storage.find_model_by_name(namespace, name)
        if model is None:
            return False
        if tag is None:
            # Delete entire model (all versions)
            meta_path = self.storage.metadata_dir / f"{model.model_id}.json"
            if meta_path.exists():
                meta_path.unlink()
            del self.storage._index["models"][model.model_id]
            self.storage._save_index()
            return True
        # Delete specific version
        model.versions = [v for v in model.versions if v.tag != tag]
        self.storage.store_model(model)
        return True

    def create_user(
        self,
        username: str,
        email: str,
        role: str = "user",
        namespaces: list[str] | None = None,
    ) -> HubUser:
        """Create a new Hub user account with a generated API key."""
        api_key = "aether_" + hashlib.sha256(
            f"{username}_{email}_{time.time()}".encode()
        ).hexdigest()[:40]
        user = HubUser(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            api_key=api_key,
            created_at=time.time(),
            role=role,
            namespaces=namespaces or [username],
        )
        self.storage.store_user(user)
        return user

    def get_stats(self) -> dict[str, Any]:
        """Return Hub server statistics."""
        total_models = len(self.storage._index.get("models", {}))
        total_users = len(self.storage._index.get("users", {}))
        total_blobs = sum(
            1 for _ in self.storage.blobs_dir.rglob("*")
            if _.is_file()
        )
        storage_bytes = sum(
            f.stat().st_size for f in self.storage.blobs_dir.rglob("*") if f.is_file()
        )
        return {
            "total_models": total_models,
            "total_users": total_users,
            "total_blobs": total_blobs,
            "storage_bytes": storage_bytes,
            "storage_gb": round(storage_bytes / (1024 ** 3), 3),
        }


# ---------------------------------------------------------------------------
# FastAPI integration (optional — only if fastapi is installed)
# ---------------------------------------------------------------------------

def create_hub_app(
    hub: AetherHubServer | None = None,
    *,
    max_upload_bytes: int = 2 * 1024**3,
    max_kv_storage_bytes: int = 200 * 1024**3,
) -> Any:
    """Expose the Hub storage API and compatible HubClient HTTP routes.

    Uploads contain binary ZIP AEG artifacts, with bounded body and expanded
    size, verified package hashes, and namespace access checks. Older
    ``/v1/hub`` endpoints remain available. This factory creates no external
    connections and makes no claim of production or hardware validation.
    """
    try:
        from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
        from fastapi.responses import Response
    except ImportError as exc:
        raise ImportError("fastapi is required for Hub server mode") from exc
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")
    hub = hub or AetherHubServer()
    app = FastAPI(title="Aether Hub", version=AETHER_VERSION)
    app.state.aether_hub = hub
    app.state.max_upload_bytes = max_upload_bytes
    from aether.runtime.kv_storage import KVPayloadStore, CacheQuotaError
    kv_store = KVPayloadStore(hub.storage.storage_root / "kv_cache", max_bytes=max_kv_storage_bytes,
                              max_entry_bytes=max_upload_bytes)
    app.state.kv_store = kv_store

    def _auth(authorization: str | None = Header(default=None)) -> HubUser:
        if authorization is None:
            raise HTTPException(401, "Authorization header required")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Bearer authentication required")
        user = hub.authenticate(token)
        if user is None:
            raise HTTPException(403, "Invalid API key")
        return user

    def _parts(model_id: str) -> tuple[str, str]:
        import re
        parts = model_id.split("/")
        if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) for part in parts):
            raise HTTPException(400, "model ID must be NAMESPACE/NAME using letters, digits, dots, underscores or hyphens")
        return parts[0], parts[1]

    def _can_read(user: HubUser, model: HubModel) -> bool:
        return (
            user.role == "admin" or model.visibility == "public"
            or model.created_by == user.user_id or model.namespace in user.namespaces
        )

    def _write_access(user: HubUser, namespace: str) -> None:
        if user.role != "admin" and (user.role != "user" or namespace not in user.namespaces):
            raise HTTPException(403, "write access to this namespace is required")

    def _read_model(model_id: str, user: HubUser) -> HubModel:
        namespace, name = _parts(model_id)
        model = hub.storage.find_model_by_name(namespace, name)
        if model is None or not _can_read(user, model):
            raise HTTPException(404, "model not found")
        return model

    def _manifest(model: HubModel, version: HubModelVersion | None = None) -> dict[str, Any]:
        version = version or model.latest_version
        return {
            "model_id": model.full_name,
            "aether_version": version.metadata.get("aether_version", "unknown") if version else "unknown",
            "aeg_version": version.format_version if version else "unknown",
            "targets": version.hardware_targets if version else [],
            "architecture": version.metadata.get("manifest_architecture", {}) if version else {},
            "file_size_bytes": version.size_bytes if version else 0,
            "content_hash": version.content_hash if version else "",
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(version.created_at)) if version else "",
            "downloads": model.download_count,
            "source": "server", "persistence": "remote_service",
        }

    async def _body(request: Any, limit: int) -> bytes:
        length = request.headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
            except ValueError as exc:
                raise HTTPException(400, "invalid Content-Length") from exc
            if size < 0:
                raise HTTPException(400, "invalid Content-Length")
            if size > limit:
                raise HTTPException(413, "request exceeds configured body limit")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                raise HTTPException(413, "request exceeds configured body limit")
            body.extend(chunk)
        return bytes(body)

    def _kv_access(namespace: str, key: str, user: HubUser, *, write: bool = False) -> str:
        _parts(f"{namespace}/{key}")
        if write:
            _write_access(user, namespace)
        elif user.role != "admin" and namespace not in user.namespaces:
            # Cache tensors may contain private prompt state; never public-by-default.
            raise HTTPException(404, "cache snapshot not found")
        return f"{namespace}/{key}"

    async def put_kv(request: Any, namespace: str, key: str, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        cache_key = _kv_access(namespace, key, user, write=True)
        data = await _body(request, min(max_upload_bytes, kv_store.max_entry_bytes))
        claimed = request.headers.get("x-content-hash")
        if claimed is None or claimed != hashlib.sha256(data).hexdigest():
            raise HTTPException(400, "cache upload requires its matching X-Content-Hash")
        try:
            entry = kv_store.put_bytes(cache_key, data, metadata={"namespace": namespace, "key": key,
                                                                "uploaded_by": user.user_id})
        except CacheQuotaError as exc:
            raise HTTPException(413, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {**entry, "namespace": namespace, "key": key, "source": "server", "persistence": "remote_service"}

    @app.get("/v1/kv/{namespace}/{key}")
    async def get_kv(namespace: str, key: str, user: HubUser = Depends(_auth)) -> Any:
        cache_key = _kv_access(namespace, key, user)
        try:
            data = kv_store.get_bytes(cache_key)
        except (ValueError, OSError) as exc:
            raise HTTPException(500, f"cache snapshot integrity/read failure: {exc}") from exc
        if data is None:
            raise HTTPException(404, "cache snapshot not found")
        return Response(content=data, media_type="application/vnd.aether.cache",
                        headers={"X-Content-Hash": hashlib.sha256(data).hexdigest(),
                                 "Cache-Control": "private, no-store"})

    @app.delete("/v1/kv/{namespace}/{key}")
    async def delete_kv(namespace: str, key: str, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        cache_key = _kv_access(namespace, key, user, write=True)
        return {"deleted": kv_store.delete(cache_key), "namespace": namespace, "key": key,
                "source": "server", "persistence": "remote_service"}

    @app.get("/v1/kv/{namespace}")
    async def list_kv(namespace: str, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        _kv_access(namespace, "listing", user)
        return {"entries": [entry for entry in kv_store.entries() if entry["metadata"].get("namespace") == namespace],
                "namespace": namespace, "source": "server", "persistence": "remote_service"}

    def _validate_artifact(data: bytes) -> None:
        import io
        import zipfile
        from aether.core.aeg_format import load_aeg_package
        from aether.hub.client import _safe_extract_zip
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 100000 or sum(item.file_size for item in entries) > max_upload_bytes * 4:
                    raise ValueError("archive exceeds expanded artifact limit")
                names = [item.filename.replace("\\", "/") for item in entries]
                if len(set(names)) != len(names):
                    raise ValueError("archive contains duplicate member names")
                if "manifest.json" not in names:
                    raise ValueError("AEG archive requires manifest.json at its root")
                with tempfile.TemporaryDirectory(prefix="aether-hub-verify-") as staging:
                    _safe_extract_zip(archive, Path(staging))
                    package = load_aeg_package(staging)
                    package.verify_integrity()
        except Exception as exc:
            raise HTTPException(400, f"invalid AEG archive: {exc}") from exc

    async def _store_upload(
        request: Any, model_id: str, user: HubUser,
        tag: str, description: str, visibility: str | None,
    ) -> dict[str, Any]:
        namespace, name = _parts(model_id)
        _write_access(user, namespace)
        if visibility is not None and visibility not in {"public", "private", "team"}:
            raise HTTPException(400, "visibility must be public, private or team")
        header_id = request.headers.get("x-model-id")
        if header_id is not None and header_id != model_id:
            raise HTTPException(400, "X-Model-Id does not match requested model")
        data = await _body(request, max_upload_bytes)
        expected_hash = request.headers.get("x-content-hash")
        digest = hashlib.sha256(data).hexdigest()
        if expected_hash is not None and expected_hash != digest:
            raise HTTPException(400, "artifact content hash mismatch")
        _validate_artifact(data)
        version = hub.push_model(namespace, name, tag, data, user.user_id, description=description)
        model = hub.storage.find_model_by_name(namespace, name)
        if model is None:
            raise HTTPException(500, "stored model metadata is unavailable")
        if visibility is not None:
            model.visibility = visibility
            hub.storage.store_model(model)
        return {"status": "ok", "manifest": _manifest(model, version)}

    # Request is imported lazily to keep the Hub data models importable without
    # FastAPI. Bind its concrete annotation before registering these handlers;
    # postponed annotations otherwise resolve against module globals.
    def _request_route(path: str, handler: Any, method: str) -> None:
        handler.__annotations__["request"] = Request
        app.add_api_route(path, handler, methods=[method])

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "aether_hub", "version": AETHER_VERSION}

    async def login(request: Any, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        try:
            data = json.loads(await _body(request, 65536))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise HTTPException(400, "invalid authentication JSON") from exc
        if not isinstance(data, dict) or data.get("token") != user.api_key:
            raise HTTPException(401, "token does not match authenticated request")
        return {"status": "ok", "authentication_validated": True, "user": user.to_dict()}
    _request_route("/auth/validate", login, "POST")
    _request_route("/v1/kv/{namespace}/{key}", put_kv, "PUT")

    def _visible_models(user: HubUser, query: str | None, namespace: str | None, limit: int) -> list[HubModel]:
        models = hub.storage.search_models(query=query, namespace=namespace, limit=len(hub.storage._index.get("models", {})))
        return [model for model in models if _can_read(user, model)][:limit]

    @app.get("/v1/models/search")
    @app.get("/v1/models")
    async def search(
        q: str | None = None, limit: int = Query(default=20, ge=1, le=1000),
        user: HubUser = Depends(_auth),
    ) -> dict[str, Any]:
        rows = [_manifest(model) for model in _visible_models(user, q, None, limit)]
        return {"models": rows, "count": len(rows)}

    async def upload(
        request: Any, model_id: str, tag: str = "latest", description: str = "",
        visibility: str | None = None, user: HubUser = Depends(_auth),
    ) -> dict[str, Any]:
        return await _store_upload(request, model_id, user, tag, description, visibility)
    _request_route("/v1/models/{model_id:path}/upload", upload, "POST")

    def _download(model_id: str, user: HubUser, tag: str) -> Any:
        model = _read_model(model_id, user)
        try:
            data = hub.pull_model(model.namespace, model.name, tag)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(content=data, media_type="application/zip", headers={"X-Content-Hash": hashlib.sha256(data).hexdigest()})

    @app.get("/v1/models/{model_id:path}/download")
    async def download(model_id: str, tag: str = "latest", user: HubUser = Depends(_auth)) -> Any:
        return _download(model_id, user, tag)

    @app.get("/v1/models/{model_id:path}")
    async def metadata(model_id: str, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        return _manifest(_read_model(model_id, user))

    def _delete(model_id: str, user: HubUser, tag: str | None) -> dict[str, Any]:
        namespace, name = _parts(model_id)
        _write_access(user, namespace)
        if not hub.delete_model(namespace, name, tag):
            raise HTTPException(404, "model not found")
        return {"status": "ok", "model": model_id}

    @app.delete("/v1/models/{model_id:path}")
    async def delete(model_id: str, tag: str | None = None, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        return _delete(model_id, user, tag)

    async def legacy_push(
        request: Any, namespace: str, name: str, tag: str = "latest", description: str = "",
        visibility: str | None = None, user: HubUser = Depends(_auth),
    ) -> dict[str, Any]:
        return await _store_upload(request, f"{namespace}/{name}", user, tag, description, visibility)
    _request_route("/v1/hub/models/{namespace}/{name}/push", legacy_push, "POST")

    @app.get("/v1/hub/models/{namespace}/{name}/pull")
    async def legacy_pull(namespace: str, name: str, tag: str = "latest", user: HubUser = Depends(_auth)) -> Any:
        return _download(f"{namespace}/{name}", user, tag)

    @app.get("/v1/hub/models/{namespace}/{name}")
    async def legacy_metadata(namespace: str, name: str, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        return _read_model(f"{namespace}/{name}", user).to_dict()

    @app.delete("/v1/hub/models/{namespace}/{name}")
    async def legacy_delete(namespace: str, name: str, tag: str | None = None, user: HubUser = Depends(_auth)) -> dict[str, Any]:
        return _delete(f"{namespace}/{name}", user, tag)

    @app.get("/v1/hub/models")
    async def legacy_search(query: str | None = None, namespace: str | None = None, limit: int = Query(default=50, ge=1, le=1000), user: HubUser = Depends(_auth)) -> dict[str, Any]:
        rows = [model.to_dict() for model in _visible_models(user, query, namespace, limit)]
        return {"models": rows, "count": len(rows)}

    @app.get("/v1/hub/stats")
    async def stats(user: HubUser = Depends(_auth)) -> dict[str, Any]:
        if user.role != "admin":
            raise HTTPException(403, "Admin access required")
        return hub.get_stats()

    @app.post("/v1/hub/users")
    async def create_user(username: str, email: str, role: str = "user", user: HubUser = Depends(_auth)) -> dict[str, Any]:
        if user.role != "admin":
            raise HTTPException(403, "Admin access required")
        if role not in {"admin", "user", "viewer"}:
            raise HTTPException(400, "invalid user role")
        new_user = hub.create_user(username=username, email=email, role=role)
        return {"user_id": new_user.user_id, "username": new_user.username, "api_key": new_user.api_key, "role": new_user.role}

    return app
