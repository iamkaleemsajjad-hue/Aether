"""
SafeTensors model loader.

Loads HuggingFace models stored in the SafeTensors format. This is the preferred
format for Aether ingestion because it provides zero-copy, safe weight access
and is standard for the HuggingFace ecosystem.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from safetensors import safe_open

from aether.compiler.stage1_ingestion.checkpoint_paths import resolve_checkpoint_shard
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
                    # Index values are relative to the checkpoint directory,
                    # not to the process working directory.  Resolve them via
                    # the shared cross-platform helper so normal Hugging Face
                    # names and cache-backed symlinks work on every machine.
                    # Sort by representation so malformed mixed-type values
                    # still reach the resolver and produce a useful
                    # IngestionError instead of an uncaught TypeError.
                    shard_names = sorted(weight_map.values(), key=repr)
                    shard_files: list[Path] = []
                    seen_files: set[Path] = set()
                    for shard_name in shard_names:
                        shard = resolve_checkpoint_shard(self.model_path, shard_name)
                        if shard not in seen_files:
                            seen_files.add(shard)
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

    def _load_with_optional_torch(self) -> dict[str, Any]:
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

    def load(self) -> dict[str, Any]:
        """Load tensors without requiring PyTorch, including BF16 files."""
        tensors: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        try:
            for st_file in self.discover_files():
                with safe_open(st_file, framework="numpy", device="cpu") as handle:
                    metadata.update(handle.metadata() or {})
                    for key in handle.keys():
                        tensors[key] = handle.get_tensor(key)
        except Exception as numpy_exc:  # noqa: BLE001 - use framework-free decoder
            try:
                for st_file in self.discover_files():
                    raw_tensors, raw_metadata = self._load_raw_safetensors(st_file)
                    tensors.update(raw_tensors)
                    metadata.update(raw_metadata)
            except Exception as raw_exc:  # noqa: BLE001 - preserve source error
                raise IngestionError(
                    f"Unable to load SafeTensors without a framework: {raw_exc}"
                ) from numpy_exc
        self._tensors = tensors
        self._metadata = metadata
        logger.info("Loaded SafeTensors", path=str(self.model_path), tensors=len(tensors))
        return tensors

    @staticmethod
    def _load_raw_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Decode SafeTensors with only the stdlib and NumPy.

        BF16 payloads are expanded by moving their 16 bits into the high half
        of an IEEE FP32 word. This keeps local SafeTensors compilation
        independent of PyTorch on NumPy versions without native BF16 support.
        """
        blob = path.read_bytes()
        if len(blob) < 8:
            raise ValueError(f"SafeTensors file is truncated: {path}")
        header_length = struct.unpack_from("<Q", blob, 0)[0]
        header_start = 8
        data_start = header_start + header_length
        if data_start > len(blob):
            raise ValueError(f"SafeTensors header exceeds file size: {path}")
        header = json.loads(blob[header_start:data_start].decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("SafeTensors header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict):
            metadata = {}
        dtype_map: dict[str, np.dtype[Any]] = {
            "BOOL": np.dtype(np.bool_), "U8": np.dtype("<u1"), "I8": np.dtype("<i1"),
            "U16": np.dtype("<u2"), "I16": np.dtype("<i2"), "U32": np.dtype("<u4"),
            "I32": np.dtype("<i4"), "U64": np.dtype("<u8"), "I64": np.dtype("<i8"),
            "F16": np.dtype("<f2"), "F32": np.dtype("<f4"), "F64": np.dtype("<f8"),
        }
        tensors: dict[str, np.ndarray] = {}
        for name, spec in header.items():
            if not isinstance(spec, dict):
                raise ValueError(f"invalid tensor header for {name!r}")
            dtype_name = str(spec.get("dtype", ""))
            shape = tuple(int(value) for value in spec.get("shape", []))
            offsets = spec.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data offsets for {name!r}")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or data_start + end > len(blob):
                raise ValueError(f"tensor {name!r} has invalid data bounds")
            payload = memoryview(blob)[data_start + start:data_start + end]
            if dtype_name == "BF16":
                values = np.frombuffer(payload, dtype="<u2").astype("<u4") << 16
                tensor = values.view("<f4")
            else:
                dtype = dtype_map.get(dtype_name)
                if dtype is None:
                    raise ValueError(f"unsupported SafeTensors dtype {dtype_name!r}")
                tensor = np.frombuffer(payload, dtype=dtype)
            expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if tensor.size != expected:
                raise ValueError(f"tensor {name!r} element count does not match its shape")
            tensors[name] = np.array(tensor.reshape(shape), copy=True)
        return tensors, metadata

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

        loaded_keys = set(self._tensors.keys())

        # Load config to validate against expected architecture if available
        config = self.load_config()
        if not config:
            report["warnings"].append("No config.json found - skipping architecture validation")
        else:
            # Expected tensor patterns for common architectures
            expected_patterns = self._get_expected_tensor_patterns(config)

            # Check for missing critical tensors
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
            if isinstance(tensor, _np.ndarray):
                return tensor
            if hasattr(tensor, "float") and getattr(tensor, "dtype", None) is not None and str(tensor.dtype).endswith("bfloat16"):
                tensor = tensor.float()
            return tensor.numpy() if hasattr(tensor, "numpy") else _np.asarray(tensor)

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
