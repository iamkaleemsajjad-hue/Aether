"""
MLX model loader.

Loads MLX-formatted models on Apple Silicon. The loader extracts weights and
configuration from the MLX model directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class MLXLoader:
    """Loads MLX model weights and configuration."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """Load MLX weights and config."""
        if not self.model_path.exists():
            msg = f"MLX model path not found: {self.model_path}"
            raise IngestionError(msg)
        weights: dict[str, Any] = {}
        weight_files = sorted(self.model_path.glob("*.safetensors")) + sorted(self.model_path.glob("*.npz"))
        for wf in weight_files:
            if wf.suffix == ".npz":
                with np.load(wf) as data:
                    for key in data:
                        weights[key] = data[key]
            elif wf.suffix == ".safetensors":
                try:
                    from safetensors import safe_open
                    with safe_open(wf, framework="np", device="cpu") as f:
                        for key in f.keys():
                            weights[key] = f.get_tensor(key)
                except ImportError as exc:
                    msg = "safetensors is required to load MLX safetensors weights"
                    raise UnsupportedFormatError(msg) from exc
        config = {}
        config_path = self.model_path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
        logger.info("Loaded MLX model", path=str(self.model_path), weights=len(weights))
        return {"weights": weights, "config": config}

    def __repr__(self) -> str:
        return f"MLXLoader({self.model_path})"
