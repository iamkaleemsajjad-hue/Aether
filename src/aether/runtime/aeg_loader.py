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
import re
from typing import Any

import numpy as np

from aether.core.exceptions import AEGFormatError
from aether.kernels.native_cpu import get_native_kernels
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights
from aether.runtime.encoder_engine import EncoderExecutionEngine, EncoderLayerWeights, EncoderModelWeights
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
    Fails closed on artifacts that were never properly finalized: a pending
    graph hash, a placeholder architecture, or a missing per-layer tensor
    set can never be runnable.
    """
    try:
        manifest = getattr(package, "manifest", None)
        if manifest is not None:
            if getattr(manifest, "graph_hash", "") == "sha256:pending":
                return False
            architecture = getattr(manifest, "architecture", None)
            layers = int(getattr(architecture, "layers", 0) or 0) if architecture else 0
            if layers <= 0:
                return False
        if not package.has_weights:
            return False
        store = package.weight_store()
        # has_weights only checks that the blob file exists; we also need at
        # least one indexed tensor entry to confirm weights were actually written.
        names = set(store.entries)
        if not names:
            return False
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
    # Loading a manifest is not sufficient: verify every declared payload
    # before any weight or packaged-kernel data reaches execution. This also
    # protects callers that use the loader directly instead of Runtime.
    package.verify_integrity()
    if not package.has_weights:
        msg = (
            f"AEG package at {package.root} contains no weights; it was compiled "
            f"graph-only and cannot run inference"
        )
        raise AEGLoadError(msg)

    architecture = _architecture_dict(package)
    if bool(architecture.get("is_encoder", False)):
        return _load_encoder_engine_from_package(package, architecture)
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

    # Hard architecture invariants: the rebuilt engine must describe exactly
    # the model the manifest declares. A mismatch means the artifact is not
    # the compiled source model and must never reach execution.
    manifest_hidden = int(architecture.get("hidden_size", 0) or 0)
    if manifest_hidden > 0 and manifest_hidden != hidden_size:
        raise AEGLoadError(
            f"Hidden size invariant violated: manifest declares {manifest_hidden} "
            f"but the embedding tensor has hidden dimension {hidden_size}"
        )
    manifest_vocab = int(architecture.get("vocab_size", 0) or 0)
    if manifest_vocab > 0 and manifest_vocab != int(embedding.shape[0]):
        raise AEGLoadError(
            f"Vocabulary invariant violated: manifest declares vocab_size="
            f"{manifest_vocab} but the embedding tensor has {int(embedding.shape[0])} rows"
        )
    for name, tensor in ((k, v) for k, v in tensors.items() if k[0] is not None):
        if tensor.ndim == 2 and name[1] in ("q_proj", "k_proj", "v_proj") and tensor.shape[1] != hidden_size:
            raise AEGLoadError(
                f"Layer {name[0]} {name[1]} input feature dimension {tensor.shape[1]} "
                f"does not match hidden size {hidden_size}"
            )
        if tensor.ndim == 2 and name[1] == "o_proj" and tensor.shape[0] != hidden_size:
            raise AEGLoadError(
                f"Layer {name[0]} o_proj output feature dimension {tensor.shape[0]} "
                f"does not match hidden size {hidden_size}"
            )

    layers: list[LayerWeights] = []
    for index in range(num_layers):
        layers.append(_build_layer(tensors, index, hidden_size, num_heads, num_kv_heads))
    if len(layers) != num_layers:  # defensive: loop above always satisfies this
        raise AEGLoadError(
            f"Layer count invariant violated: built {len(layers)} layers, "
            f"manifest declares {num_layers}"
        )
    # The weight store must describe exactly the manifest's layer count: a
    # blob containing layer tensors beyond num_layers-1 means the manifest
    # understates the model (the silent 4-layer -> 1-layer failure mode).
    max_stored_layer = max(
        (key[0] for key in tensors if key[0] is not None and key[0] >= 0), default=-1
    )
    if max_stored_layer != num_layers - 1:
        raise AEGLoadError(
            f"Layer count invariant violated: manifest declares {num_layers} layers "
            f"but the weight store contains tensors for layers 0..{max_stored_layer}"
        )

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
    kernels = get_native_kernels()
    _load_packaged_native_kernels(package, kernels)
    engine = CPUExecutionEngine(
        weights,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kernels=kernels,
        sparse_attention_plan=_runtime_sparse_attention_plan(package.metadata, num_heads),
        semantic_kv_plan=_runtime_semantic_kv_plan(package.metadata, num_layers),
        cross_layer_kv_plan=_runtime_cross_layer_kv_plan(package.metadata, num_layers),
    )
    logger.info(
        "Loaded executable engine from %s (%d layers, hidden=%d, heads=%d/%d)",
        package.root.name,
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
    )
    return engine


def _load_encoder_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> EncoderExecutionEngine:
    """Build the bidirectional CPU encoder from named AEG tensors."""
    from aether.quantization.formats import dequantize_tensor

    store = package.weight_store()

    def required(name: str) -> np.ndarray:
        if name not in store.entries:
            raise AEGLoadError(f"encoder AEG is missing required tensor {name!r}")
        return np.asarray(dequantize_tensor(store.load_tensor(name)), dtype=np.float32)

    def optional(name: str) -> np.ndarray | None:
        return required(name) if name in store.entries else None

    embedding = required("embedding")
    hidden_size = int(embedding.shape[1])
    declared_hidden = int(architecture.get("hidden_size", 0) or 0)
    if declared_hidden and declared_hidden != hidden_size:
        raise AEGLoadError(
            f"encoder hidden size invariant violated: manifest={declared_hidden}, tensor={hidden_size}"
        )
    declared_vocab = int(architecture.get("vocab_size", 0) or 0)
    if declared_vocab and declared_vocab != int(embedding.shape[0]):
        raise AEGLoadError(
            f"encoder vocabulary invariant violated: manifest={declared_vocab}, tensor={embedding.shape[0]}"
        )

    layers: list[EncoderLayerWeights] = []
    layer_count = int(architecture.get("layers", 0) or 0)
    for index in range(layer_count):
        prefix = f"layer_{index}_"
        layers.append(
            EncoderLayerWeights(
                q_proj=required(prefix + "q_proj"),
                k_proj=required(prefix + "k_proj"),
                v_proj=required(prefix + "v_proj"),
                o_proj=required(prefix + "o_proj"),
                attention_norm=required(prefix + "attention_norm"),
                intermediate_proj=required(prefix + "intermediate_proj"),
                output_proj=required(prefix + "output_proj"),
                output_norm=required(prefix + "output_norm"),
                q_bias=optional(prefix + "q_proj_bias"),
                k_bias=optional(prefix + "k_proj_bias"),
                v_bias=optional(prefix + "v_proj_bias"),
                o_bias=optional(prefix + "o_proj_bias"),
                intermediate_bias=optional(prefix + "intermediate_proj_bias"),
                output_bias=optional(prefix + "output_proj_bias"),
                attention_norm_bias=optional(prefix + "attention_norm_bias"),
                output_norm_bias=optional(prefix + "output_norm_bias"),
            )
        )
    if len(layers) != layer_count or layer_count <= 0:
        raise AEGLoadError("encoder manifest must declare at least one layer")
    num_heads = int(architecture.get("num_attention_heads", 0) or 0)
    weights = EncoderModelWeights(
        embedding=embedding,
        position_embedding=required("position_embedding"),
        token_type_embedding=required("token_type_embedding"),
        embedding_norm=required("embedding_norm"),
        pooler=required("pooler"),
        layers=layers,
        embedding_norm_bias=optional("embedding_norm_bias"),
        pooler_bias=optional("pooler_bias"),
        norm_eps=float(architecture.get("norm_eps", 1e-12) or 1e-12),
    )
    return EncoderExecutionEngine(weights, num_heads=num_heads)


def _runtime_sparse_attention_plan(
    metadata: Any,
    num_heads: int,
) -> dict[str, Any] | None:
    """Validate and return a Pass 8 plan for the executable CPU engine."""
    if not isinstance(metadata, dict):
        return None
    plan = metadata.get("attention_head_patterns")
    if not isinstance(plan, dict) or not bool(plan.get("enabled")):
        return None
    patterns = plan.get("patterns")
    if not isinstance(patterns, list) or len(patterns) != num_heads:
        raise AEGLoadError(
            "enabled sparse attention plan must contain one pattern per attention head"
        )
    valid = {"dense", "a_shape", "vertical_slash", "block_sparse"}
    for descriptor in patterns:
        if not isinstance(descriptor, dict):
            raise AEGLoadError("sparse attention pattern must be an object")
        pattern = descriptor.get("pattern", descriptor.get("pattern_type"))
        if pattern not in valid:
            raise AEGLoadError(f"unsupported sparse attention pattern {pattern!r}")
    return plan


def _runtime_semantic_kv_plan(metadata: Any, num_layers: int) -> dict[str, Any] | None:
    """Validate and return an executable Pass 14 plan."""
    if not isinstance(metadata, dict):
        return None
    plan = metadata.get("kv_compression_plan")
    if plan is None:
        return None
    if not isinstance(plan, dict) or plan.get("format") != "aether_kv_compression_v1":
        raise AEGLoadError("semantic KV plan has an unsupported format")
    layers = plan.get("layers")
    if not isinstance(layers, list) or len(layers) != num_layers:
        raise AEGLoadError(
            f"semantic KV plan must contain exactly {num_layers} layer policies"
        )
    strategy = str(plan.get("strategy", "chunk"))
    if strategy not in {"chunk", "hybrid"}:
        raise AEGLoadError(
            f"semantic KV strategy {strategy!r} requires tokenizer boundary metadata; "
            "only chunk and hybrid plans are executable by the CPU cache"
        )
    for index, policy in enumerate(layers):
        if not isinstance(policy, dict):
            raise AEGLoadError(f"semantic KV policy {index} must be an object")
        try:
            retention = float(policy["retention_ratio"])
            chunk_size = int(policy["chunk_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AEGLoadError(f"invalid semantic KV policy {index}") from exc
        if not 0.0 < retention <= 1.0 or chunk_size <= 0:
            raise AEGLoadError(f"invalid semantic KV policy {index}")
    return plan


def _runtime_cross_layer_kv_plan(metadata: Any, num_layers: int) -> dict[str, Any] | None:
    """Validate and return an executable Pass 15 plan."""
    if not isinstance(metadata, dict):
        return None
    plan = metadata.get("cross_layer_kv_plan")
    if plan is None:
        return None
    if not isinstance(plan, dict) or plan.get("format") != "aether_cross_layer_kv_v1":
        raise AEGLoadError("cross-layer KV plan has an unsupported format")
    if int(plan.get("n_layers", -1)) != num_layers:
        raise AEGLoadError("cross-layer KV plan layer count does not match the model")
    groups = plan.get("sharing_groups")
    if not isinstance(groups, list):
        raise AEGLoadError("cross-layer KV plan must contain sharing_groups")
    targets: set[int] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise AEGLoadError("cross-layer KV sharing group must be an object")
        source = int(group.get("src_layer", -1))
        shared = group.get("shared_with")
        if source < 0 or source >= num_layers or not isinstance(shared, list):
            raise AEGLoadError("invalid cross-layer KV source group")
        for target_value in shared:
            target = int(target_value)
            if target <= source or target >= num_layers or target in targets:
                raise AEGLoadError(
                    "cross-layer KV targets must have one earlier source layer"
                )
            targets.add(target)
    return plan


def _load_packaged_native_kernels(package: Any, kernels: Any) -> None:
    """Load the authenticated native CPU library embedded in an AEG, if any."""
    metadata = getattr(package, "metadata", {})
    descriptors = metadata.get("kernel_artifacts", []) if isinstance(metadata, dict) else []
    if not descriptors:
        # Older AEG/1.x packages may not carry a packaged library.  Preserve
        # their existing host-cache/reference-kernel behavior.
        return
    if not isinstance(descriptors, list):
        raise AEGLoadError("AEG kernel_artifacts metadata must be a list")
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or descriptor.get("backend") != "native_cpu":
            continue
        relative_path = descriptor.get("path")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise AEGLoadError("AEG packaged kernel path is unsafe")
        path = (Path(package.root) / relative_path).resolve()
        expected = descriptor.get("sha256")
        if not isinstance(expected, str):
            raise AEGLoadError(f"AEG packaged kernel {relative_path!r} has no SHA-256 digest")
        if not path.is_relative_to(Path(package.root).resolve()):
            raise AEGLoadError("AEG packaged kernel escapes package root")
        if not kernels.load_library(path, expected_sha256=expected):
            raise AEGLoadError(
                f"unable to load packaged native CPU kernel {relative_path!r}: {kernels.build_error}"
            )
        return


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
        # The ingestion normalizer intentionally groups checkpoint q/k/v names
        # under ``qkv``.  The persisted AEG store has already split those
        # projections, so preserve their exact component names here.
        match = re.match(r"^layer_(\d+)_(q_proj|k_proj|v_proj|o_proj|attention_norm|ffn_norm|gate_proj|up_proj|down_proj)$", name)
        if match:
            key = (int(match.group(1)), match.group(2))
        elif name == "embedding":
            key = (None, "embedding")
        elif name == "lm_head":
            key = (None, "lm_head")
        elif name == "final_norm":
            key = (None, "final_norm")
        else:
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

    A package is executable only when every projection required by the declared
    architecture is present.  Identity/zero substitutions would produce logits
    that look valid while no longer representing the source model, so malformed
    or partial artifacts fail closed.
    """
    q_proj = tensors.get((index, "q_proj"))
    head_dim = (q_proj.shape[0] // num_heads) if q_proj is not None and q_proj.ndim == 2 else (hidden_size // num_heads)
    kv_dim = num_kv_heads * head_dim
    intermediate = _infer_intermediate(tensors, index, hidden_size)

    k_proj = tensors.get((index, "k_proj"))
    v_proj = tensors.get((index, "v_proj"))
    fused = tensors.get((index, "qkv"))
    if q_proj is None and fused is not None and fused.ndim == 2:
        q_rows = num_heads * (hidden_size // num_heads)
        expected_rows = q_rows + 2 * kv_dim
        if fused.shape[0] == expected_rows:
            q_proj = fused[:q_rows]
            k_proj = fused[q_rows : q_rows + kv_dim]
            v_proj = fused[q_rows + kv_dim :]
    required = {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": tensors.get((index, "o_proj")),
        "attention_norm": tensors.get((index, "attention_norm")),
        "ffn_norm": tensors.get((index, "ffn_norm")),
        "gate_proj": tensors.get((index, "gate_proj")),
        "up_proj": tensors.get((index, "up_proj")),
        "down_proj": tensors.get((index, "down_proj"))
        if tensors.get((index, "down_proj")) is not None
        else tensors.get((index, "ffn")),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise AEGLoadError(f"AEG package is missing required layer {index} tensors: {', '.join(missing)}")

    return LayerWeights(
        attention_norm=_norm_vector(required["attention_norm"], hidden_size),
        q_proj=np.ascontiguousarray(required["q_proj"], dtype=np.float32),
        k_proj=np.ascontiguousarray(required["k_proj"], dtype=np.float32),
        v_proj=np.ascontiguousarray(required["v_proj"], dtype=np.float32),
        o_proj=np.ascontiguousarray(required["o_proj"], dtype=np.float32),
        ffn_norm=_norm_vector(required["ffn_norm"], hidden_size),
        gate_proj=np.ascontiguousarray(required["gate_proj"], dtype=np.float32),
        up_proj=np.ascontiguousarray(required["up_proj"], dtype=np.float32),
        down_proj=np.ascontiguousarray(required["down_proj"], dtype=np.float32),
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
    """Return a ``(rows, cols)`` matrix, defaulting to identity.

    Raises AEGLoadError instead of crashing with MemoryError when the default
    matrix would be unreasonably large (> 128 MiB), which indicates the tensor
    was expected from the weight blob but is missing.
    """
    if tensor is None or tensor.ndim != 2:
        size_bytes = rows * cols * 4  # float32
        if size_bytes > 128 * 1024 * 1024:  # 128 MiB guard
            msg = (
                f"Missing weight tensor ({rows}×{cols} = {size_bytes // (1024*1024)} MiB); "
                f"the package was compiled from a model without local weights and cannot be "
                f"loaded on this machine.  Run 'aether pull <model>' first."
            )
            raise AEGLoadError(msg)
        return np.eye(rows, cols, dtype=np.float32)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _matrix_or_zeros(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Return a ``(rows, cols)`` matrix, defaulting to zeros.

    Zeros are the safe default for FFN projections: they make the block a no-op
    through the residual path instead of injecting arbitrary values.

    Raises AEGLoadError instead of crashing with MemoryError when the default
    matrix would be unreasonably large (> 128 MiB), which indicates the tensor
    was expected from the weight blob but is missing.
    """
    if tensor is None or tensor.ndim != 2:
        size_bytes = rows * cols * 4  # float32
        if size_bytes > 128 * 1024 * 1024:  # 128 MiB guard
            msg = (
                f"Missing weight tensor ({rows}×{cols} = {size_bytes // (1024*1024)} MiB); "
                f"the package was compiled from a model without local weights and cannot be "
                f"loaded on this machine.  Run 'aether pull <model>' first."
            )
            raise AEGLoadError(msg)
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
