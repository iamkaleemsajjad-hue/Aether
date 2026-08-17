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
        """Load all tensors and metadata from discovered SafeTensors files.

        Torch-free by default: the numpy framework covers every dtype the
        CPU pipeline consumes. Shards numpy cannot represent (e.g. BF16) are
        retried through the optional PyTorch frontend; without torch such a
        checkpoint fails with an explicit error instead of silently importing
        the framework.
        """
        tensors: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        try:
            for st_file in self.discover_files():
                with safe_open(st_file, framework="numpy", device="cpu") as f:
                    metadata.update(f.metadata() or {})
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
        except Exception as numpy_exc:  # noqa: BLE001 — retry via torch only when installed
            try:
                import torch  # noqa: F401 — optional PyTorch frontend
            except ImportError:
                raise IngestionError(
                    f"SafeTensors file at {self.model_path} contains a dtype numpy "
                    f"cannot represent (e.g. BF16) and torch is not installed. "
                    f"Install it with: pip install 'aether-runtime[pytorch]'"
                ) from numpy_exc
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

    def validate_weights(self) -> dict[str, Any]:
        """Validate loaded weights for completeness and integrity.

        Returns:
            Validation report with warnings and errors.
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        }

        if not self._tensors:
            report["valid"] = False
            report["errors"].append("No tensors loaded")
            return report

        # Load config to validate against expected architecture
        config = self.load_config()
        if not config:
            report["warnings"].append("No config.json found - skipping architecture validation")
            return report

        # Expected tensor patterns for common architectures
        expected_patterns = self._get_expected_tensor_patterns(config)

        # Check for missing critical tensors
        loaded_keys = set(self._tensors.keys())
        for pattern_type, patterns in expected_patterns.items():
            found = any(any(p in key for p in patterns) for key in loaded_keys)
            if not found and pattern_type in ["embed", "lm_head"]:
                report["errors"].append(f"Missing critical tensors: {pattern_type}")
                report["valid"] = False

        # Validate tensor shapes against config
        num_layers = config.get("num_hidden_layers", config.get("num_layers", 0))
        if num_layers > 0:
            layer_count = sum(1 for key in loaded_keys if ".layers." in key or ".h." in key)
            expected_keys_per_layer = 8  # typical: attn.q,k,v,o + mlp.gate,up,down + norm
            actual_layers = layer_count // expected_keys_per_layer
            if actual_layers < num_layers * 0.9:  # Allow 10% tolerance
                report["warnings"].append(
                    f"Expected {num_layers} layers but found ~{actual_layers} based on tensor count"
                )

        # Check for NaN or Inf values in a sample of tensors (numpy; no torch).
        import numpy as _np

        def _to_numpy(tensor: Any) -> Any:
            return tensor if isinstance(tensor, _np.ndarray) else _np.asarray(tensor)

        sample_keys = list(loaded_keys)[:10]  # Check first 10 tensors
        for key in sample_keys:
            array = _to_numpy(self._tensors[key]).astype(_np.float32, copy=False)
            if _np.isnan(array).any():
                report["errors"].append(f"NaN values detected in {key}")
                report["valid"] = False
            if _np.isinf(array).any():
                report["warnings"].append(f"Inf values detected in {key}")

        logger.info("Weight validation complete", valid=report["valid"],
                   warnings=len(report["warnings"]), errors=len(report["errors"]))
        return report

    def _get_expected_tensor_patterns(self, config: dict[str, Any]) -> dict[str, list[str]]:
        """Get expected tensor key patterns based on model architecture."""
        patterns = {
            "embed": ["embed_tokens", "wte", "token_embed", "word_embeddings"],
            "lm_head": ["lm_head", "output", "embed_out"],
            "attention": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "self_attn.q_proj",
                         "attention.query", "attention.key", "attention.value"],
            "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                   "ffn.gate", "ffn.up", "ffn.down", "mlp.fc1", "mlp.fc2"],
            "norm": ["norm", "ln_", "layernorm", "rms_norm"],
        }

        # Add MoE patterns if applicable
        if config.get("num_local_experts", 0) > 0 or config.get("num_experts", 0) > 0:
            patterns["moe"] = ["experts.", "expert_", "router", "gate"]

        return patterns

    def compute_sha256(self) -> str:
        """Compute SHA-256 hash of all loaded weights for integrity checking.

        Returns:
            Hex string of SHA-256 hash.
        """
        import hashlib

        import numpy as _np

        hasher = hashlib.sha256()

        # Sort keys for deterministic hashing
        for key in sorted(self._tensors.keys()):
            tensor = self._tensors[key]
            # Hash tensor metadata
            hasher.update(key.encode('utf-8'))
            hasher.update(str(tensor.shape).encode('utf-8'))
            hasher.update(str(tensor.dtype).encode('utf-8'))
            # Hash tensor data (numpy; no torch dependency)
            array = tensor if isinstance(tensor, _np.ndarray) else _np.asarray(tensor)
            hasher.update(_np.ascontiguousarray(array).tobytes())

        return hasher.hexdigest()

    def __repr__(self) -> str:
        return f"SafeTensorsLoader({self.model_path}, tensors={len(self._tensors)})"
