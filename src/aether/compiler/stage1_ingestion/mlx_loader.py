"""
MLX model loader.

Loads Apple MLX model checkpoints (which are stored as safetensors with an
optional mlx-specific config). MLX models are structurally identical to
HuggingFace safetensors models; this loader provides the mlx-specific weight
name normalization and bfloat16 conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class MLXLoader:
    """
    Loads MLX model checkpoints as float32 numpy arrays.

    MLX models are distributed as:
    - ``*.safetensors`` shards (primary format)
    - ``weights.npz`` (older format)
    - ``model.safetensors`` (single file)

    Weight names follow the same conventions as HuggingFace; no name
    remapping is required.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """
        Load MLX model weights.

        Returns:
            dict with:
              - ``weights``:  name → float32 numpy array
              - ``tensors``:  alias for weights
              - ``keys``:     sorted weight names
              - ``format``:   "safetensors" | "npz" | "mlx_native"
        """
        path = self.model_path
        if not path.exists():
            msg = f"MLX model path not found: {path}"
            raise IngestionError(msg)

        # Strategy 1: safetensors files in directory
        if path.is_dir():
            st_files = sorted(path.glob("*.safetensors"))
            if st_files:
                return self._load_safetensors_shards(st_files)
            npz_files = sorted(path.glob("*.npz"))
            if npz_files:
                return self._load_npz_shards(npz_files)
            # Check for a flat weights directory
            msg = f"No safetensors or npz files found in MLX model directory: {path}"
            raise IngestionError(msg)

        # Strategy 2: single .safetensors file
        if path.suffix == ".safetensors":
            return self._load_safetensors_shards([path])

        # Strategy 3: .npz file
        if path.suffix == ".npz":
            return self._load_npz_shards([path])

        # Strategy 4: try mlx native format (requires mlx package)
        return self._load_mlx_native(path)

    def _load_safetensors_shards(self, files: list[Path]) -> dict[str, Any]:
        """Load one or more safetensors shards."""
        try:
            from safetensors.numpy import load_file
        except ImportError:
            msg = "safetensors package required for MLX safetensors loading"
            raise UnsupportedFormatError(msg)

        weights: dict[str, np.ndarray] = {}
        for f in files:
            try:
                shard = load_file(str(f))
                for name, arr in shard.items():
                    weights[name] = self._to_float32(arr)
            except Exception as exc:
                logger.warning("Could not load MLX shard %s: %s", f, exc)

        logger.info(
            "Loaded MLX model (safetensors)",
            files=len(files),
            tensors=len(weights),
        )
        return {
            "weights": weights,
            "tensors": weights,
            "keys": sorted(weights.keys()),
            "format": "safetensors",
        }

    def _load_npz_shards(self, files: list[Path]) -> dict[str, Any]:
        """Load one or more .npz weight files."""
        weights: dict[str, np.ndarray] = {}
        for f in files:
            try:
                npz = np.load(str(f), allow_pickle=False)
                for name in npz.files:
                    weights[name] = self._to_float32(npz[name])
            except Exception as exc:
                logger.warning("Could not load MLX npz shard %s: %s", f, exc)

        logger.info(
            "Loaded MLX model (npz)",
            files=len(files),
            tensors=len(weights),
        )
        return {
            "weights": weights,
            "tensors": weights,
            "keys": sorted(weights.keys()),
            "format": "npz",
        }

    def _load_mlx_native(self, path: Path) -> dict[str, Any]:
        """Load via the mlx.core package if available."""
        try:
            import mlx.core as mx
        except ImportError:
            msg = "mlx package required for native MLX format loading"
            raise UnsupportedFormatError(msg)

        try:
            loaded = mx.load(str(path))
            weights: dict[str, np.ndarray] = {}
            for name, tensor in loaded.items():
                arr = np.array(tensor, dtype=np.float32)
                weights[name] = arr
            logger.info(
                "Loaded MLX model (mlx_native)",
                path=str(path),
                tensors=len(weights),
            )
            return {
                "weights": weights,
                "tensors": weights,
                "keys": sorted(weights.keys()),
                "format": "mlx_native",
            }
        except Exception as exc:
            msg = f"Failed to load MLX model from {path}: {exc}"
            raise IngestionError(msg) from exc

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        """Convert any array to float32, handling bfloat16 specially."""
        if arr.dtype == np.dtype("bfloat16") or str(arr.dtype) in ("bfloat16", "bf16"):
            # NumPy has no native bfloat16; reinterpret as uint16 and shift
            u16 = arr.view(np.uint16) if arr.itemsize == 2 else arr.astype(np.uint16)
            u32 = u16.astype(np.uint32) << 16
            return u32.view(np.float32)
        return arr.astype(np.float32)

    def __repr__(self) -> str:
        return f"MLXLoader({self.model_path})"
