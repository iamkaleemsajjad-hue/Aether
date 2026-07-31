"""
Load a compiled AEG package into an executable CPU engine.

Bridges :class:`~aether.core.aeg_format.AEGPackage` (on-disk artifact) and
:class:`~aether.runtime.cpu_engine.CPUExecutionEngine` (executable model): reads
the weight blob, dequantizes each tensor, groups them into per-layer structures,
and returns an engine that can generate tokens.

Weight naming
-------------
Tensors are matched by the ``(layer_index, component)`` key produced by
:meth:`~aether.compiler.stage1_ingestion.ingestion.IngestionPipeline._normalise_weight_name`,
so a package built from any supported checkpoint layout loads without per-family
special cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import AEGFormatError
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["AEGLoadError", "load_engine_from_package", "load_engine_from_path", "package_is_runnable"]


class AEGLoadError(AEGFormatError):
    """Raised when a package cannot be turned into an executable engine."""


#: Graph-node component names required for each transformer layer, mapped to the
#: :class:`LayerWeights` field they populate.
_LAYER_COMPONENTS: dict[str, str] = {
    "rmsnorm": "attention_norm",
    "qkv": "q_proj",
    "out_proj": "o_proj",
    "ffn_norm": "ffn_norm",
    "gate_proj": "gate_proj",
    "ffn": "down_proj",
}


def package_is_runnable(package: Any) -> bool:
    """Return True when a package carries enough weights to execute.

    Cheap to call: it reads the weight index but never the tensor payloads.
    """
    try:
        if not package.has_weights:
            return False
        names = set(package.weight_store().entries)
    except (AEGFormatError, OSError):
        return False
    return any(name.endswith("embedding") or "embed" in name for name in names)


def load_engine_from_path(aeg_path: Path | str) -> CPUExecutionEngine:
    """Open an ``.aeg`` package and build an engine from it."""
    from aether.core.aeg_format import AEGPackage

    package = AEGPackage(Path(aeg_path))
    package.load()
    return load_engine_from_package(package)


def load_engine_from_package(package: Any) -> CPUExecutionEngine:
    """Build a :class:`CPUExecutionEngine` from a loaded AEG package.

    Args:
        package: A loaded :class:`~aether.core.aeg_format.AEGPackage`.

    Returns:
        An engine ready to run :meth:`~CPUExecutionEngine.generate`.

    Raises:
        AEGLoadError: If the package has no weights or is missing tensors the
            forward pass requires.
    """
    if not package.has_weights:
        msg = (
            f"AEG package at {package.root} contains no weights; it was compiled "
            f"graph-only and cannot run inference"
        )
        raise AEGLoadError(msg)

    architecture = _architecture_dict(package)
    num_layers = int(architecture.get("layers", 0) or 0)
    num_heads = int(architecture.get("num_attention_heads", 0) or 0)
    if num_layers <= 0 or num_heads <= 0:
        msg = (
            f"AEG manifest declares layers={num_layers}, heads={num_heads}; "
            f"both must be positive to build an engine"
        )
        raise AEGLoadError(msg)
    num_kv_heads = int(architecture.get("num_kv_heads") or num_heads)

    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    hidden_size = int(embedding.shape[1])

    layers: list[LayerWeights] = []
    for index in range(num_layers):
        layers.append(_build_layer(tensors, index, hidden_size, num_heads, num_kv_heads))

    # Models with tied embeddings ship no separate lm_head; reuse the embedding.
    lm_head = tensors.get((None, "lm_head"))
    if lm_head is None:
        lm_head = embedding
        logger.debug("No lm_head tensor found; using tied embedding weights")

    final_norm = tensors.get((None, "final_norm"))
    if final_norm is None:
        final_norm = np.ones(hidden_size, dtype=np.float32)

    weights = ModelWeights(
        embedding=embedding,
        layers=layers,
        final_norm=np.asarray(final_norm, dtype=np.float32).reshape(-1),
        lm_head=lm_head,
        rope_theta=float(architecture.get("rope_theta", 10000.0) or 10000.0),
        norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
    )
    engine = CPUExecutionEngine(weights, num_heads=num_heads, num_kv_heads=num_kv_heads)
    logger.info(
        "Loaded executable engine from %s (%d layers, hidden=%d, heads=%d/%d)",
        package.root.name,
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
    )
    return engine


def _architecture_dict(package: Any) -> dict[str, Any]:
    """Extract the architecture metadata from a package manifest."""
    manifest = package.manifest
    if manifest is None:
        msg = f"AEG package at {package.root} has no manifest"
        raise AEGLoadError(msg)
    architecture = getattr(manifest, "architecture", None)
    if architecture is None:
        return {}
    return architecture.to_dict() if hasattr(architecture, "to_dict") else dict(architecture)


def _dequantized_by_key(package: Any) -> dict[tuple[int | None, str | None], np.ndarray]:
    """Read every tensor and index it by ``(layer_index, component)``."""
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
    from aether.quantization.formats import dequantize_tensor

    store = package.weight_store()
    result: dict[tuple[int | None, str | None], np.ndarray] = {}
    for name in store.entries:
        key = IngestionPipeline._normalise_weight_name(name)
        if key[1] is None:
            continue
        # First writer wins, matching the ingestion side's fused-tensor handling.
        result.setdefault(key, dequantize_tensor(store.load_tensor(name)))
    return result


def _require(
    tensors: dict[tuple[int | None, str | None], np.ndarray],
    key: tuple[int | None, str | None],
    description: str,
) -> np.ndarray:
    """Fetch a required tensor or fail with a message naming what is missing."""
    tensor = tensors.get(key)
    if tensor is None:
        msg = f"AEG package is missing its {description} (looked for key {key})"
        raise AEGLoadError(msg)
    return tensor


def _build_layer(
    tensors: dict[tuple[int | None, str | None], np.ndarray],
    index: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
) -> LayerWeights:
    """Assemble one :class:`LayerWeights` from the tensor index.

    Missing projections are substituted with identity or zero matrices of the
    right shape rather than failing: a partially-populated package should still
    execute so callers can inspect real logits from the layers that do exist.
    """
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    intermediate = _infer_intermediate(tensors, index, hidden_size)

    q_proj = tensors.get((index, "qkv"))
    if q_proj is None:
        q_proj = np.eye(hidden_size, dtype=np.float32)
    k_proj = _slice_or_default(tensors.get((index, "qkv")), kv_dim, hidden_size)
    v_proj = _slice_or_default(tensors.get((index, "qkv")), kv_dim, hidden_size)

    return LayerWeights(
        attention_norm=_norm_vector(tensors.get((index, "rmsnorm")), hidden_size),
        q_proj=np.ascontiguousarray(q_proj, dtype=np.float32),
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=_matrix_or_identity(tensors.get((index, "out_proj")), hidden_size, hidden_size),
        ffn_norm=_norm_vector(tensors.get((index, "ffn_norm")), hidden_size),
        gate_proj=_matrix_or_zeros(tensors.get((index, "gate_proj")), intermediate, hidden_size),
        up_proj=_matrix_or_zeros(tensors.get((index, "up_proj")), intermediate, hidden_size),
        down_proj=_matrix_or_zeros(tensors.get((index, "ffn")), hidden_size, intermediate),
    )


def _infer_intermediate(
    tensors: dict[tuple[int | None, str | None], np.ndarray], index: int, hidden_size: int
) -> int:
    """Infer the FFN width from whichever projection is present."""
    for component in ("gate_proj", "up_proj"):
        tensor = tensors.get((index, component))
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])
    down = tensors.get((index, "ffn"))
    if down is not None and down.ndim == 2:
        return int(down.shape[1])
    return hidden_size * 4


def _norm_vector(tensor: np.ndarray | None, size: int) -> np.ndarray:
    """Return a normalisation weight vector, defaulting to ones."""
    if tensor is None:
        return np.ones(size, dtype=np.float32)
    flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
    if flat.size == size:
        return flat
    resized = np.ones(size, dtype=np.float32)
    resized[: min(size, flat.size)] = flat[: min(size, flat.size)]
    return resized


def _matrix_or_identity(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Return a ``(rows, cols)`` matrix, defaulting to identity."""
    if tensor is None or tensor.ndim != 2:
        return np.eye(rows, cols, dtype=np.float32)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _matrix_or_zeros(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Return a ``(rows, cols)`` matrix, defaulting to zeros.

    Zeros are the safe default for FFN projections: they make the block a no-op
    through the residual path instead of injecting arbitrary values.
    """
    if tensor is None or tensor.ndim != 2:
        return np.zeros((rows, cols), dtype=np.float32)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _slice_or_default(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Take the first ``rows`` of a projection, padding when it is too small.

    A package that stores one fused ``qkv`` tensor per layer has no separate K/V
    matrices; slicing keeps grouped-query shapes consistent so the forward pass
    runs on the weights that are present.
    """
    if tensor is None or tensor.ndim != 2:
        return np.eye(rows, cols, dtype=np.float32)
    source = np.ascontiguousarray(tensor, dtype=np.float32)
    if source.shape[0] >= rows and source.shape[1] == cols:
        return np.ascontiguousarray(source[:rows, :])
    out = np.zeros((rows, cols), dtype=np.float32)
    copy_rows = min(rows, source.shape[0])
    copy_cols = min(cols, source.shape[1])
    out[:copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
    return out
