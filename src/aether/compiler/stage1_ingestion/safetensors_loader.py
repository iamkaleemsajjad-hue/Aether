"""
SafeTensors model loader.

Loads HuggingFace models stored in the SafeTensors format. This is the preferred
format for Aether ingestion because it provides zero-copy, safe weight access
and is standard for the HuggingFace ecosystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from aether.core.exceptions import IngestionError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class SafeTensorsLoader:
    """Loads model weights and metadata from SafeTensors files."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._tensors: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def discover_files(self) -> list[Path]:
        """Return all SafeTensors files in the model path."""
        if self.model_path.is_file() and self.model_path.suffix == ".safetensors":
            return [self.model_path]
        if self.model_path.is_dir():
            return sorted(self.model_path.glob("*.safetensors"))
        msg = f"No SafeTensors files found at {self.model_path}"
        raise IngestionError(msg)

    def load(self) -> dict[str, Any]:
        """Load all tensors and metadata from discovered SafeTensors files."""
        tensors: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for st_file in self.discover_files():
            with safe_open(st_file, framework="pt", device="cpu") as f:
                metadata.update(f.metadata() or {})
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        self._tensors = tensors
        self._metadata = metadata
        logger.info("Loaded SafeTensors", path=str(self.model_path), tensors=len(tensors))
        return tensors

    def load_config(self) -> dict[str, Any]:
        """Load the model config.json if present."""
        config_path = self.model_path / "config.json" if self.model_path.is_dir() else self.model_path.parent / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def __repr__(self) -> str:
        return f"SafeTensorsLoader({self.model_path}, tensors={len(self._tensors)})"
