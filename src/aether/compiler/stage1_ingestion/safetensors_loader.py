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
        """Return all SafeTensors files in the model path.

        Handles three layouts:
        1. A single ``.safetensors`` file path.
        2. A directory with a ``model.safetensors.index.json`` shard index —
           the canonical HuggingFace multi-shard layout.  Only the shards
           listed in ``weight_map`` are returned (in sorted order) so that
           optimizer-state or other non-weight shards are excluded.
        3. A directory containing one or more ``*.safetensors`` files without
           an index (single-shard or legacy layout).

        Raises:
            IngestionError: When no SafeTensors files can be found.
        """
        if self.model_path.is_file() and self.model_path.suffix == ".safetensors":
            return [self.model_path]

        if self.model_path.is_dir():
            # --- Multi-shard layout: index.json takes precedence ---
            index_path = self.model_path / "model.safetensors.index.json"
            if index_path.exists():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                    weight_map = index.get("weight_map", {})
                    if not isinstance(weight_map, dict) or not weight_map:
                        raise ValueError("index.json has no weight_map")
                    # Collect unique shard filenames in sorted order.
                    shard_names = sorted(set(str(v) for v in weight_map.values()))
                    root = self.model_path.resolve()
                    shard_files: list[Path] = []
                    for shard_name in shard_names:
                        relative = Path(shard_name)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError(f"unsafe shard path {shard_name!r}")
                        shard = (self.model_path / relative).resolve()
                        if not shard.is_relative_to(root):
                            raise ValueError(
                                f"shard path escapes checkpoint directory: {shard_name!r}"
                            )
                        if not shard.exists():
                            raise ValueError(f"shard file not found: {shard}")
                        shard_files.append(shard)
                    if shard_files:
                        return shard_files
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    raise IngestionError(
                        f"invalid SafeTensors shard index {index_path}: {exc}"
                    ) from exc

            # --- Single-shard / legacy layout: glob *.safetensors ---
            files = sorted(self.model_path.glob("*.safetensors"))
            if files:
                return files

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
