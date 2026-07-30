"""
PyTorch model loader.

Loads PyTorch state dicts and pickled models for ingestion. This is the fallback
loader when other formats are not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from aether.core.exceptions import IngestionError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class PyTorchLoader:
    """Loads PyTorch model weights and config."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """Load PyTorch weights from a file or directory."""
        if not self.model_path.exists():
            msg = f"PyTorch model path not found: {self.model_path}"
            raise IngestionError(msg)
        weights: dict[str, Any] = {}
        if self.model_path.is_file():
            weights = torch.load(self.model_path, map_location="cpu", weights_only=True)
        elif self.model_path.is_dir():
            for pt_file in sorted(self.model_path.glob("*.bin")) + sorted(self.model_path.glob("*.pt")):
                partial = torch.load(pt_file, map_location="cpu", weights_only=True)
                if isinstance(partial, dict):
                    weights.update(partial)
                else:
                    weights[pt_file.stem] = partial
        logger.info("Loaded PyTorch model", path=str(self.model_path), weights=len(weights) if isinstance(weights, dict) else 0)
        return {"weights": weights, "config": {}}

    def __repr__(self) -> str:
        return f"PyTorchLoader({self.model_path})"
