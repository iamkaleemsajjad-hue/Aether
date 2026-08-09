"""
PyTorch model loader.

Loads PyTorch checkpoint files (.pt, .pth, .bin) and extracts weight tensors
as float32 numpy arrays. Works with both torch.save checkpoints and the
HuggingFace ``pytorch_model.bin`` format (which is also a torch.save file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def _tensor_to_numpy(tensor: Any) -> np.ndarray | None:
    """Convert a PyTorch tensor to a float32 numpy array without requiring torch."""
    if hasattr(tensor, "numpy"):
        try:
            t = tensor
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            return t.numpy().astype(np.float32)
        except Exception:
            pass
    if hasattr(tensor, "__array__"):
        try:
            return np.asarray(tensor, dtype=np.float32)
        except Exception:
            pass
    return None


def _flatten_state_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten a nested state dict or OrderedDict into a flat
    name → tensor mapping (matching HuggingFace checkpoint conventions).
    """
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if _tensor_to_numpy(value) is not None:
                result[full_key] = value
            elif isinstance(value, dict):
                result.update(_flatten_state_dict(value, full_key))
    return result


class PyTorchLoader:
    """
    Loads PyTorch checkpoint files and extracts weight tensors.

    Supports:
    - ``torch.save`` state dicts (.pt, .pth, .bin)
    - HuggingFace ``pytorch_model.bin`` sharded checkpoints
    - Full model saves (``model.state_dict()`` wrappers)
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """
        Load a PyTorch checkpoint and return extracted weights.

        Returns:
            dict with keys:
              - ``weights``: name → float32 numpy array
              - ``tensors``: alias for weights
              - ``keys``: sorted list of weight names
              - ``format``: "state_dict" | "full_model" | "sharded"
        """
        path = self.model_path
        if not path.exists():
            msg = f"PyTorch checkpoint not found: {path}"
            raise IngestionError(msg)

        try:
            import torch
        except ImportError:
            msg = "torch is required to load PyTorch checkpoints"
            raise UnsupportedFormatError(msg)

        if path.is_dir():
            return self._load_sharded(path)

        return self._load_single_file(path)

    def _load_single_file(self, path: Path) -> dict[str, Any]:
        """Load a single .pt / .pth / .bin file."""
        import torch

        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            # weights_only not supported in older torch
            checkpoint = torch.load(str(path), map_location="cpu")
        except Exception as exc:
            msg = f"Failed to load PyTorch checkpoint {path}: {exc}"
            raise IngestionError(msg) from exc

        weights, fmt = self._extract_weights(checkpoint)
        logger.info(
            "Loaded PyTorch checkpoint",
            path=str(path),
            format=fmt,
            tensors=len(weights),
        )
        return {
            "weights": weights,
            "tensors": weights,
            "keys": sorted(weights.keys()),
            "format": fmt,
        }

    def _load_sharded(self, directory: Path) -> dict[str, Any]:
        """Load all .bin shards from a HuggingFace model directory."""
        import torch

        shard_files = sorted(directory.glob("pytorch_model*.bin")) or sorted(
            directory.glob("model*.bin")
        )
        if not shard_files:
            shard_files = sorted(directory.glob("*.pt")) + sorted(
                directory.glob("*.pth")
            )
        if not shard_files:
            msg = f"No PyTorch weight files found in {directory}"
            raise IngestionError(msg)

        weights: dict[str, np.ndarray] = {}
        for shard in shard_files:
            try:
                try:
                    ckpt = torch.load(str(shard), map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(str(shard), map_location="cpu")
                shard_weights, _ = self._extract_weights(ckpt)
                if not shard_weights:
                    raise IngestionError(f"PyTorch shard contains no tensor weights: {shard}")
                weights.update(shard_weights)
            except Exception as exc:
                raise IngestionError(f"Failed to load PyTorch shard {shard}: {exc}") from exc

        if not weights:
            raise IngestionError(f"No readable tensor weights found in {directory}")

        logger.info(
            "Loaded sharded PyTorch checkpoint",
            directory=str(directory),
            shards=len(shard_files),
            tensors=len(weights),
        )
        return {
            "weights": weights,
            "tensors": weights,
            "keys": sorted(weights.keys()),
            "format": "sharded",
        }

    def _extract_weights(self, checkpoint: Any) -> tuple[dict[str, np.ndarray], str]:
        """
        Extract weight tensors from various checkpoint formats.

        Handles:
        - Raw state dict (dict of tensors)
        - Wrapped state dict ({\"state_dict\": {...}})
        - Full model save (has .state_dict() method)
        """
        fmt = "state_dict"
        state_dict: dict[str, Any] = {}

        if hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
            fmt = "full_model"
        elif isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "module"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    fmt = "state_dict"
                    break
            else:
                state_dict = checkpoint  # raw state dict

        flat = _flatten_state_dict(state_dict)
        weights: dict[str, np.ndarray] = {}
        for name, tensor in flat.items():
            arr = _tensor_to_numpy(tensor)
            if arr is not None:
                weights[name] = arr

        return weights, fmt

    def __repr__(self) -> str:
        return f"PyTorchLoader({self.model_path})"
