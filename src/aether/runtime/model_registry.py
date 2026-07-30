"""
Local AEG model registry and cache manager.

The model registry manages compiled AEG artifacts on local disk. It supports
listing, deleting, resolving, and basic metadata retrieval. It also provides
stubs for integration with HuggingFace and the Aether Hub.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.core.exceptions import ModelNotFoundError
from aether.utils.file_io import aether_cache_dir, safe_model_id_path
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Local cache registry for compiled AEG models."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = aether_cache_dir(cache_dir)
        self.models_dir = self.cache_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model registry initialized", cache_dir=str(self.cache_dir))

    def _model_path(self, model_id: str) -> Path:
        """Return the local directory path for a model ID."""
        return self.models_dir / safe_model_id_path(model_id)

    def is_cached(self, model_id: str) -> bool:
        """Return True if a compiled AEG exists for the model."""
        path = self._model_path(model_id)
        return path.exists() and (path / "manifest.json").is_file()

    def list_models(self) -> list[str]:
        """Return all model IDs with cached AEG artifacts."""
        if not self.models_dir.exists():
            return []
        models: list[str] = []
        for entry in self.models_dir.iterdir():
            if entry.is_dir() and (entry / "manifest.json").is_file():
                models.append(entry.name)
        return sorted(models)

    def remove(self, model_id: str) -> bool:
        """Remove a cached model."""
        path = self._model_path(model_id)
        if not path.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        logger.info("Removed model from cache", model_id=model_id)
        return True

    def resolve(self, model_id: str) -> Path | None:
        """Resolve a model ID to a local AEG path if cached."""
        path = self._model_path(model_id)
        if (path / "manifest.json").is_file():
            return path
        return None

    def load(self, model_id: str) -> AEGPackage:
        """Load the AEG package for a cached model."""
        path = self.resolve(model_id)
        if path is None:
            msg = f"Model {model_id} not found in local cache"
            raise ModelNotFoundError(msg, model_id=model_id)
        return load_aeg_package(str(path))

    def metadata(self, model_id: str) -> dict[str, Any]:
        """Return metadata for a cached model."""
        aeg = self.load(model_id)
        if aeg.manifest is None:
            return {"model_id": model_id, "cached": True}
        return {
            "model_id": aeg.manifest.model_id,
            "format_version": aeg.manifest.format_version,
            "aether_version": aeg.manifest.aether_version,
            "architecture": aeg.manifest.architecture.to_dict(),
            "targets": aeg.manifest.kernels.targets,
            "precision_map": aeg.get_precision_map(),
        }

    def register(self, model_id: str, package_path: Path | str) -> Path:
        """Register an external AEG package into the local cache."""
        src = Path(package_path)
        if not src.exists():
            msg = f"Package path does not exist: {src}"
            raise ModelNotFoundError(msg, model_id=model_id)
        dst = self._model_path(model_id)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / "package.aeg")
        logger.info("Registered model package", model_id=model_id, dst=str(dst))
        return dst

    def clear_cache(self) -> int:
        """Remove all cached models. Returns the number removed."""
        count = 0
        if self.models_dir.exists():
            for entry in self.models_dir.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                    count += 1
        logger.info("Cleared model cache", count=count)
        return count

    def disk_usage_bytes(self) -> int:
        """Return total disk usage of the model cache in bytes."""
        total = 0
        if not self.models_dir.exists():
            return 0
        for path in self.models_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def __repr__(self) -> str:
        return f"ModelRegistry(cache={self.cache_dir}, models={len(self.list_models())})"


class HubModelRegistry(ModelRegistry):
    """Registry backed by the Aether Hub.

    This is a stub that extends the local registry with Hub-aware lookup. In a
    full implementation it would query the Hub API, download missing artifacts,
    and authenticate uploads.
    """

    def __init__(self, cache_dir: str | None = None, hub_url: str | None = None) -> None:
        super().__init__(cache_dir=cache_dir)
        self.hub_url = hub_url

    def hub_search(self, query: str) -> list[dict[str, Any]]:
        """Stub: search the Hub for models."""
        return [
            {
                "model_id": query,
                "available": False,
                "hub_url": self.hub_url,
            }
        ]

    def __repr__(self) -> str:
        return f"HubModelRegistry(cache={self.cache_dir}, hub={self.hub_url})"
