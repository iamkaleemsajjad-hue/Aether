"""
Local content-addressed kernel cache.

Stores compiled kernel blobs keyed by (graph_hash, target_id, aether_version)
so that recompiling the same model for the same target reuses previously
compiled kernels.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from aether.core.constants import KERNEL_CACHE_KEY_FMT
from aether.core.exceptions import KernelCacheError
from aether.utils.file_io import aether_cache_dir
from aether.utils.logging import get_logger

logger = get_logger(__name__)

KERNEL_CACHE_VERSION = 1


class KernelCache:
    """Content-addressed local kernel cache on disk."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = aether_cache_dir(cache_dir)
        self.kernel_dir = self.cache_dir / "kernels"
        self.kernel_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Kernel cache initialized", cache_dir=str(self.kernel_dir))

    def _key_path(self, graph_hash: str, target_id: str, version: str) -> Path:
        """Return the file path for a cache key."""
        key = KERNEL_CACHE_KEY_FMT.format(graph_hash=graph_hash, target_id=target_id, aether_version=version)
        hashed = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.kernel_dir / f"{hashed}.kernel"

    def _metadata_path(self, graph_hash: str, target_id: str, version: str) -> Path:
        key_file = self._key_path(graph_hash, target_id, version)
        return key_file.with_suffix(".meta.json")

    def lookup(self, graph_hash: str, target_id: str, version: str) -> bytes | None:
        """Look up a cached kernel blob.

        Returns the kernel bytes if found, None otherwise.
        """
        path = self._key_path(graph_hash, target_id, version)
        if not path.exists():
            return None
        if not self._verify(path, graph_hash, target_id, version):
            logger.warning("Kernel cache integrity check failed", path=str(path))
            path.unlink(missing_ok=True)
            return None
        return path.read_bytes()

    def store(self, graph_hash: str, target_id: str, version: str, data: bytes, metadata: dict[str, Any] | None = None) -> Path:
        """Store a kernel blob in the cache."""
        path = self._key_path(graph_hash, target_id, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        meta = {
            "cache_version": KERNEL_CACHE_VERSION,
            "graph_hash": graph_hash,
            "target_id": target_id,
            "aether_version": version,
            "size_bytes": len(data),
            "stored_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            **(metadata or {}),
        }
        with self._metadata_path(graph_hash, target_id, version).open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Kernel cached", key=path.stem, size_bytes=len(data))
        return path

    def exists(self, graph_hash: str, target_id: str, version: str) -> bool:
        """Check whether a cached kernel exists."""
        path = self._key_path(graph_hash, target_id, version)
        return path.exists()

    def _verify(self, path: Path, graph_hash: str, target_id: str, version: str) -> bool:
        """Verify that a cached kernel matches the requested key."""
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return True
        try:
            meta = json.loads(meta_path.read_text())
            return (
                meta.get("graph_hash") == graph_hash
                and meta.get("target_id") == target_id
                and meta.get("aether_version") == version
            )
        except Exception:
            return False

    def clear(self) -> int:
        """Remove all cached kernels. Returns the count removed."""
        count = 0
        if self.kernel_dir.exists():
            for f in self.kernel_dir.iterdir():
                if f.is_file() and f.suffix in (".kernel", ".meta.json"):
                    f.unlink(missing_ok=True)
                    count += 1
        logger.info("Kernel cache cleared", count=count)
        return count

    def disk_usage_bytes(self) -> int:
        """Return total disk usage in bytes."""
        total = 0
        if self.kernel_dir.exists():
            for f in self.kernel_dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def __repr__(self) -> str:
        return f"KernelCache(cache_dir={self.kernel_dir})"
