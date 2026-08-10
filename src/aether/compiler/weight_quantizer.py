"""
Graph weight quantization bridge.

This module is the missing link between the optimizer pipeline and the
``AEGPackage.weights`` dict.  After the optimizer runs and assigns a
per-layer precision map, ``GraphWeightQuantizer`` walks every graph node,
reads the ``weight`` numpy array that was attached during ingestion, and
quantizes it with the codec matching the node's assigned precision.

The resulting ``dict[str, QuantizedTensor]`` is written directly into
``AEGPackage.weights`` so that the subsequent ``package.save()`` call
writes a real ``model.aeg-quant`` blob and ``weight_index.json`` rather
than an empty package.

Weight name convention
----------------------
The names stored in the weight blob must match the keys that
``ModelWeightsLoader`` (in ``aether.runtime.aeg_loader``) later reads
back and assembles into ``LayerWeights`` / ``ModelWeights``.  Both sides
agree on the following scheme::

    embedding                        — token embedding table
    layer_{i}_attention_norm         — pre-attention RMSNorm weight
    layer_{i}_q_proj                 — query projection
    layer_{i}_k_proj                 — key projection
    layer_{i}_v_proj                 — value projection
    layer_{i}_o_proj                 — output / down projection
    layer_{i}_ffn_norm               — pre-FFN RMSNorm weight
    layer_{i}_gate_proj              — SwiGLU gate projection
    layer_{i}_up_proj                — SwiGLU up projection
    layer_{i}_down_proj              — SwiGLU down projection
    final_norm                       — final RMSNorm weight
    lm_head                          — language-model head
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from aether.quantization.formats import QuantizedTensor, quantize_tensor
from aether.utils.logging import get_logger

if TYPE_CHECKING:
    from aether.core.aeg_format import AEGPackage
    from aether.core.graph import AEGGraph

logger = get_logger(__name__)

__all__ = ["GraphWeightQuantizer", "quantize_graph_weights"]

#: Graph node op_types that never carry a weight tensor.
_STRUCTURAL_OPS = frozenset({
    "input", "output", "add", "rope", "gqa", "moe_router",
    "expert_ffn", "residual",
})

#: How an AEGGraph node id maps to a canonical weight-store key.
#: The node id is already close to what we want; the table below covers
#: the cases where op_type adds more information than the node id alone.
_NODE_ID_TO_WEIGHT_KEY: dict[str, str] = {
    "embedding": "embedding",
    "lm_head": "lm_head",
    "final_norm": "final_norm",
}

#: Regex that parses "layer_{i}_{suffix}" node ids.
_LAYER_NODE_RE = re.compile(r"^layer_(\d+)_(.+)$")

#: Maps the per-layer node id suffix to the weight-store key suffix.
_LAYER_SUFFIX_MAP: dict[str, str] = {
    "rmsnorm": "attention_norm",
    "qkv": None,            # fused QKV → expanded to q/k/v below
    "out_proj": "o_proj",
    "ffn_norm": "ffn_norm",
    "gate_proj": "gate_proj",
    "ffn": "down_proj",
    # Dense SwiGLU up projection is stored separately by the ingestion pipeline.
    "up_proj": "up_proj",
}


@dataclass
class QuantizationStats:
    """Counters collected while quantizing a graph."""

    total_nodes: int = 0
    nodes_with_weights: int = 0
    tensors_written: int = 0
    bytes_written: int = 0
    skipped_no_weight: int = 0
    skipped_structural: int = 0
    precision_counts: dict[str, int] = field(default_factory=dict)

    def record(self, precision: str, tensor: QuantizedTensor) -> None:
        self.tensors_written += 1
        self.bytes_written += tensor.compressed_size_bytes
        self.precision_counts[precision] = self.precision_counts.get(precision, 0) + 1

    def __str__(self) -> str:
        return (
            f"QuantizationStats(tensors={self.tensors_written}, "
            f"bytes={self.bytes_written:,}, "
            f"precision={self.precision_counts})"
        )


class GraphWeightQuantizer:
    """Quantizes weight arrays attached to graph nodes and stores them in
    an :class:`~aether.core.aeg_format.AEGPackage`.

    Args:
        precision_map: Maps layer identifier (e.g. ``"layer_0"``) to a
            precision string (e.g. ``"Q4_K_M"``).  The quantizer falls back
            to ``default_precision`` when a node has no explicit entry.
        default_precision: Precision used for nodes not in ``precision_map``.
        block_size: Number of elements per quantization block for block-
            scaled formats.
    """

    def __init__(
        self,
        precision_map: dict[str, str] | None = None,
        default_precision: str = "Q4_K_M",
        block_size: int = 32,
    ) -> None:
        self.precision_map: dict[str, str] = precision_map or {}
        self.default_precision = default_precision
        self.block_size = block_size

    # ── Public API ────────────────────────────────────────────────────────────

    def quantize(
        self,
        graph: AEGGraph,
        package: AEGPackage,
    ) -> QuantizationStats:
        """Quantize all weight-bearing nodes in ``graph`` and store them.

        Populates ``package.weights`` with ``QuantizedTensor`` objects.
        Returns a :class:`QuantizationStats` summary.

        Calling ``package.save()`` afterwards will write the weight blob.
        """
        stats = QuantizationStats()
        quantized: dict[str, QuantizedTensor] = {}

        # Pass 1 retains fused source nodes for inspection but excludes them
        # from the executable topological order. Their checkpoint tensors are
        # still required by the backend, so quantization must visit the full
        # node store rather than silently dropping weights hidden by fusion.
        nodes = graph._nodes.values() if hasattr(graph, "_nodes") else graph
        for node in nodes:
            stats.total_nodes += 1

            # Skip structural ops that never carry parameters.
            op_type = getattr(node, "op_type", "")
            if op_type in _STRUCTURAL_OPS:
                stats.skipped_structural += 1
                continue

            weight = self._extract_weight(node)
            if weight is None:
                stats.skipped_no_weight += 1
                continue

            # Pass 9's verified mask must affect the payload, not only the
            # metadata report. A malformed mask fails compilation instead of
            # being silently ignored.
            weight = self._apply_pruning_mask(node, weight)

            stats.nodes_with_weights += 1
            node_id = getattr(node, "id", "")
            layer_index = getattr(node, "layer_index", None)
            precision = self._resolve_precision(node_id, layer_index)

            # Fused QKV nodes split back into three tensors.
            if op_type == "qkv_proj" or node_id.endswith("_qkv"):
                sub_tensors = self._split_qkv(node, weight, precision, layer_index)
                for name, qt in sub_tensors.items():
                    quantized[name] = qt
                    stats.record(precision, qt)
            else:
                name = self._weight_name(node_id, op_type, layer_index)
                if name is None:
                    logger.debug("Skipping unrecognised node '%s' (op=%s)", node_id, op_type)
                    continue
                qt = quantize_tensor(weight, precision, self.block_size)
                quantized[name] = qt
                stats.record(precision, qt)
                # The ingestion graph models SwiGLU's gate and up projections
                # as one logical node, but they are distinct checkpoint tensors.
                # Persist both real tensors so the CPU engine can reconstruct
                # the original block without substituting zeros.
                up_weight = getattr(node, "attributes", {}).get("up_weight")
                if up_weight is not None and node_id.endswith("_gate_proj"):
                    up_name = f"layer_{layer_index}_up_proj"
                    up_qt = quantize_tensor(np.asarray(up_weight, dtype=np.float32), precision, self.block_size)
                    quantized[up_name] = up_qt
                    stats.record(precision, up_qt)

        package.weights = quantized
        logger.info(
            "GraphWeightQuantizer: %d tensors quantized, %d bytes",
            stats.tensors_written,
            stats.bytes_written,
        )
        return stats

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_weight(self, node: Any) -> np.ndarray | None:
        """Return the numpy weight array attached to ``node``, or ``None``."""
        attrs = getattr(node, "attributes", {}) or {}

        # Preferred path: ingestion attached a real numpy array.
        weight = attrs.get("weight")
        if weight is not None:
            arr = np.asarray(weight, dtype=np.float32)
            if arr.ndim >= 1 and arr.size > 0:
                return arr

        # Never manufacture model parameters. A graph without attached
        # checkpoint tensors is useful for planning, but it must not become a
        # runnable AEG artifact containing random weights.
        return None

    def _apply_pruning_mask(self, node: Any, weight: np.ndarray) -> np.ndarray:
        """Apply a real Pass 9 keep-mask before quantization."""
        attrs = getattr(node, "attributes", {}) or {}
        pruning_mask = attrs.get("pruning_mask")
        if pruning_mask is None:
            return weight
        raw_mask = getattr(pruning_mask, "mask", None)
        if raw_mask is None:
            raise ValueError(
                f"node {getattr(node, 'id', '<unknown>')} has an invalid pruning mask"
            )
        mask = np.asarray(raw_mask, dtype=bool)
        if tuple(mask.shape) != tuple(weight.shape):
            raise ValueError(
                f"pruning mask shape {tuple(mask.shape)} does not match weight shape "
                f"{tuple(weight.shape)} for node {getattr(node, 'id', '<unknown>')}"
            )
        return np.where(mask, weight, 0.0).astype(np.float32, copy=False)

    def _infer_weight_shape(self, node: Any) -> tuple[int, ...] | None:
        """Infer a plausible weight shape from node attributes."""
        attrs = getattr(node, "attributes", {}) or {}
        op = getattr(node, "op_type", "")

        if op in ("embedding", "lm_head"):
            v = attrs.get("vocab_size", 0)
            h = attrs.get("hidden_size", 0)
            if v > 0 and h > 0:
                return (v, h) if op == "embedding" else (v, h)

        if op in ("rmsnorm",):
            h = attrs.get("hidden_size", 0)
            if h > 0:
                return (h,)

        if op in ("linear", "qkv_proj"):
            out_f = attrs.get("out_features", 0)
            in_f = attrs.get("in_features", 0)
            # QKV projection: output is Q+K+V concatenated.
            if op == "qkv_proj":
                h = attrs.get("num_heads", 0)
                kv = attrs.get("num_kv_heads", h)
                hd = attrs.get("head_dim", 0)
                if h > 0 and hd > 0:
                    out_f = (h + kv + kv) * hd
                    in_f = h * hd
            if out_f > 0 and in_f > 0:
                return (out_f, in_f)

        if op in ("gate_proj", "swiglu_ffn", "ffn"):
            out_f = attrs.get("out_features", attrs.get("intermediate_size", 0))
            in_f = attrs.get("in_features", attrs.get("hidden_size", 0))
            if out_f > 0 and in_f > 0:
                return (out_f, in_f)

        return None

    def _resolve_precision(self, node_id: str, layer_index: int | None) -> str:
        """Look up the quantization precision for a node."""
        if layer_index is not None:
            key = f"layer_{layer_index}"
            if key in self.precision_map:
                return self.precision_map[key]
        if node_id in self.precision_map:
            return self.precision_map[node_id]
        # Embedding and LM head often stay in BF16 for quality.
        if node_id in ("embedding", "lm_head", "final_norm"):
            return self.precision_map.get(node_id, "BF16")
        return self.default_precision

    def _weight_name(
        self, node_id: str, op_type: str, layer_index: int | None
    ) -> str | None:
        """Convert a graph node id to a canonical weight-store key name."""
        # Global nodes.
        if node_id in _NODE_ID_TO_WEIGHT_KEY:
            return _NODE_ID_TO_WEIGHT_KEY[node_id]

        # Layer-scoped nodes.
        m = _LAYER_NODE_RE.match(node_id)
        if m:
            idx = int(m.group(1))
            suffix = m.group(2)
            mapped = _LAYER_SUFFIX_MAP.get(suffix)
            if mapped is None:
                # Suffix unknown → return a safe fallback.
                return f"layer_{idx}_{suffix}"
            return f"layer_{idx}_{mapped}"

        # op_type-based fallback.
        if op_type == "embedding":
            return "embedding"
        if op_type == "lm_head":
            return "lm_head"
        return None

    def _split_qkv(
        self,
        node: Any,
        weight: np.ndarray,
        precision: str,
        layer_index: int | None,
    ) -> dict[str, QuantizedTensor]:
        """Split a fused QKV weight matrix into separate q / k / v projections.

        The fused layout assumed here is Q stacked over K over V along axis 0,
        which is what the ingestion pipeline produces when it encounters a
        single ``qkv_proj`` node.  If the shapes don't divide cleanly (e.g.
        GQA where K/V heads != Q heads) we fall back to storing the whole
        matrix under a generic key.
        """
        attrs = getattr(node, "attributes", {}) or {}
        n_q = attrs.get("num_heads", 0)
        n_kv = attrs.get("num_kv_heads", n_q)
        hd = attrs.get("head_dim", 0)
        prefix = f"layer_{layer_index}" if layer_index is not None else "layer_0"

        if n_q > 0 and n_kv > 0 and hd > 0:
            q_rows = n_q * hd
            kv_rows = n_kv * hd
            total_rows = q_rows + kv_rows + kv_rows
            if weight.shape[0] == total_rows:
                q_w = weight[:q_rows]
                k_w = weight[q_rows : q_rows + kv_rows]
                v_w = weight[q_rows + kv_rows :]
                return {
                    f"{prefix}_q_proj": quantize_tensor(q_w, precision, self.block_size),
                    f"{prefix}_k_proj": quantize_tensor(k_w, precision, self.block_size),
                    f"{prefix}_v_proj": quantize_tensor(v_w, precision, self.block_size),
                }

        # Fallback: store as one monolithic weight.
        node_id = getattr(node, "id", f"{prefix}_qkv")
        logger.debug("QKV split skipped for '%s'; storing as fused matrix", node_id)
        return {f"{prefix}_qkv_fused": quantize_tensor(weight, precision, self.block_size)}


# ── Convenience function ──────────────────────────────────────────────────────

def quantize_graph_weights(
    graph: AEGGraph,
    package: AEGPackage,
    precision_map: dict[str, str] | None = None,
    default_precision: str = "Q4_K_M",
    block_size: int = 32,
) -> QuantizationStats:
    """Convenience wrapper around :class:`GraphWeightQuantizer`.

    Args:
        graph: The optimized AEG graph whose nodes carry weight arrays.
        package: The AEG package to populate with quantized weights.
        precision_map: Per-layer precision overrides.
        default_precision: Precision for nodes not in the map.
        block_size: Quantization block size.

    Returns:
        :class:`QuantizationStats` summary.
    """
    quantizer = GraphWeightQuantizer(
        precision_map=precision_map,
        default_precision=default_precision,
        block_size=block_size,
    )
    return quantizer.quantize(graph, package)
