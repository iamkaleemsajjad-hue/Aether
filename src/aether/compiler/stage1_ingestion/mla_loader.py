"""
Aether Runtime Ã¢â‚¬â€ MLA (Multi-head Latent Attention) Native Loader.

Implements graph extraction for the DeepSeek family of models that use
Multi-head Latent Attention (MLA) for KV-cache compression. MLA replaces
the standard multi-head attention mechanism by compressing Keys and Values
into a low-dimensional latent vector at each layer, reducing KV cache by
5-13Ãƒâ€” compared to MHA.

Supported models:
  - DeepSeek-V2 (236B MoE, MLA + 21B active params)
  - DeepSeek-V3 (685B MoE, MLA + FP8 training)
  - DeepSeek-R1 (671B MoE, MLA + chain-of-thought RLHF)
  - DeepSeek-Coder-V2 (MoE coding model)
  - Any custom model with ``kv_lora_rank`` in config

Research basis:
  - DeepSeek-V2: DeepSeek-AI (2024) Ã¢â‚¬â€ https://arxiv.org/abs/2405.04434
  - DeepSeek-V3: DeepSeek-AI (2025) Ã¢â‚¬â€ https://arxiv.org/abs/2412.19437
  - DeepSeek-R1: DeepSeek-AI (2025) Ã¢â‚¬â€ https://arxiv.org/abs/2501.12948
  - PRD v4.0 Ã‚Â§4.3 Ã¢â‚¬â€ MLA Native Loader
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MLA architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class MLAArchitecture:
    """Describes the MLA-specific attention configuration."""

    # Model identity
    model_type: str       # "deepseek_v2", "deepseek_v3", "deepseek_r1"
    family: str           # "deepseek"

    # Standard transformer dims
    hidden_size: int      # e.g., 5120
    num_layers: int       # e.g., 60
    num_heads: int        # Query heads (e.g., 128)
    vocab_size: int       # e.g., 102400

    # MLA-specific dimensions
    kv_lora_rank: int     # Latent KV rank (e.g., 512)
    q_lora_rank: int      # Query compression rank (0 = no Q compression)
    qk_rope_head_dim: int # RoPE head dimension (e.g., 64)
    qk_nope_head_dim: int # No-position-embedding head dim (e.g., 128)
    v_head_dim: int       # Value head dimension (e.g., 128)

    # MoE configuration
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_token: int = 1
    moe_intermediate_size: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 1  # First N layers use standard MHA (not MLA+MoE)

    # Generation config
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] = field(default_factory=dict)

    @property
    def kv_compression_ratio(self) -> float:
        """KV cache compression ratio vs standard GQA."""
        # Standard MHA KV per layer = num_heads * head_dim * 2 (K+V) floats
        standard_kv = self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        # MLA KV per layer = kv_lora_rank + qk_rope_head_dim (rope cache only)
        mla_kv = self.kv_lora_rank + self.qk_rope_head_dim
        return standard_kv / max(mla_kv, 1)

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


# ---------------------------------------------------------------------------
# Config parsing and architecture detection
# ---------------------------------------------------------------------------

_MODEL_TYPE_REGISTRY = {
    "deepseek_v2": {
        "hidden_size": 5120,
        "num_layers": 60,
        "num_heads": 128,
        "vocab_size": 102400,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "is_moe": True,
        "num_experts": 160,
        "num_experts_per_token": 6,
        "moe_intermediate_size": 1536,
        "num_shared_experts": 2,
        "first_k_dense_replace": 1,
        "max_position_embeddings": 163840,
    },
    "deepseek_v3": {
        "hidden_size": 7168,
        "num_layers": 61,
        "num_heads": 128,
        "vocab_size": 129280,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "is_moe": True,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "moe_intermediate_size": 2048,
        "num_shared_experts": 1,
        "first_k_dense_replace": 3,
        "max_position_embeddings": 163840,
    },
    "deepseek_r1": {
        "hidden_size": 7168,
        "num_layers": 61,
        "num_heads": 128,
        "vocab_size": 129280,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "is_moe": True,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "moe_intermediate_size": 2048,
        "num_shared_experts": 1,
        "first_k_dense_replace": 3,
        "max_position_embeddings": 163840,
        "rope_theta": 500000.0,
    },
}

_MODEL_TYPE_ALIASES = {
    "deepseekv2": "deepseek_v2",
    "deepseekv3": "deepseek_v3",
    "deepseek_coder_v2": "deepseek_v2",
    "deepseek-r1": "deepseek_r1",
}


def _parse_mla_config(config: dict[str, Any]) -> MLAArchitecture:
    """Parse a HuggingFace config.json into an MLAArchitecture."""
    raw_type = (config.get("model_type") or "").lower().replace("-", "_")
    canonical = _MODEL_TYPE_ALIASES.get(raw_type, raw_type)
    defaults = _MODEL_TYPE_REGISTRY.get(canonical, {})

    def _get(key: str, fallback: Any = 0) -> Any:
        return config.get(key, defaults.get(key, fallback))

    return MLAArchitecture(
        model_type=canonical or "deepseek_v2",
        family="deepseek",
        hidden_size=int(_get("hidden_size", 5120)),
        num_layers=int(_get("num_hidden_layers", 60)),
        num_heads=int(_get("num_attention_heads", 128)),
        vocab_size=int(_get("vocab_size", 102400)),
        kv_lora_rank=int(_get("kv_lora_rank", 512)),
        q_lora_rank=int(_get("q_lora_rank", 0)),
        qk_rope_head_dim=int(_get("qk_rope_head_dim", 64)),
        qk_nope_head_dim=int(_get("qk_nope_head_dim", 128)),
        v_head_dim=int(_get("v_head_dim", 128)),
        is_moe=bool(_get("n_routed_experts", 0)) or bool(_get("num_experts", 0)),
        num_experts=int(_get("n_routed_experts", _get("num_experts", 1))),
        num_experts_per_token=int(_get("num_experts_per_tok", _get("num_experts_per_token", 1))),
        moe_intermediate_size=int(_get("moe_intermediate_size", 0)),
        num_shared_experts=int(_get("n_shared_experts", _get("num_shared_experts", 0))),
        first_k_dense_replace=int(_get("first_k_dense_replace", 1)),
        max_position_embeddings=int(_get("max_position_embeddings", 163840)),
        rope_theta=float(_get("rope_theta", 10000.0)),
        rope_scaling=dict(_get("rope_scaling", {})),
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class MLALoader:
    """
    Loads and extracts an AEG graph from a DeepSeek MLA model.

    Key differentiators from standard loaders:
    - Emits ``aeg.mla_attention`` nodes instead of ``aeg.attention``
    - Records KV compression metadata for the runtime KV manager
    - Handles hybrid dense (first K layers) + MLA+MoE (remaining layers)
    - Emits expert routing plans for MoE layers
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """
        Load the MLA model and return an AEG graph.

        Returns:
            dict with keys:
              - ``graph``: AEGGraph with all nodes and edges
              - ``architecture``: MLAArchitecture descriptor
              - ``kv_compression_ratio``: measured KV reduction vs standard MHA
              - ``format``: "mla_model"
        """
        path = self.model_path
        if not path.exists():
            from aether.core.exceptions import IngestionError
            raise IngestionError(f"MLA model not found: {path}")

        config = self._load_config(path)
        arch = _parse_mla_config(config)

        logger.info(
            "Detected MLA architecture",
            model_type=arch.model_type,
            kv_lora_rank=arch.kv_lora_rank,
            q_lora_rank=arch.q_lora_rank,
            kv_compression_ratio=f"{arch.kv_compression_ratio:.1f}x",
            is_moe=arch.is_moe,
            num_experts=arch.num_experts if arch.is_moe else 0,
        )

        graph = self._build_mla_graph(arch, config, path)

        logger.info(
            "MLA graph extracted",
            layers=arch.num_layers,
            dense_layers=arch.first_k_dense_replace,
            mla_layers=arch.num_layers - arch.first_k_dense_replace,
            kv_compression_ratio=f"{arch.kv_compression_ratio:.1f}x",
        )

        return {
            "graph": graph,
            "architecture": arch,
            "kv_compression_ratio": arch.kv_compression_ratio,
            "format": "mla_model",
        }

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> dict[str, Any]:
        for name in ("config.json", "model_config.json"):
            candidate = path / name
            if candidate.is_file():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        return {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_mla_graph(
        self,
        arch: MLAArchitecture,
        config: dict[str, Any],
        path: Path,
    ) -> Any:
        """Build full AEG graph with hybrid dense + MLA + MoE layers."""
        try:
            from aether.core.graph import AEGGraph, AEGGraphEdge, AEGGraphNode, AEGGraphNodeType
        except ImportError:
            return _FallbackGraph()

        graph = AEGGraph()

        # === Embedding ===
        embed_node = AEGGraphNode(
            id="embedding",
            node_type=AEGGraphNodeType.OPERATION,
            name="token_embedding",
            op_type="aeg.embedding",
            attributes={"vocab_size": arch.vocab_size, "hidden_size": arch.hidden_size},
        )
        graph.add_node(embed_node)

        prev_id = embed_node.id

        # === Transformer layers (hybrid dense + MLA + MoE) ===
        for i in range(arch.num_layers):
            is_dense = i < arch.first_k_dense_replace
            layer_nodes = self._build_transformer_layer(arch, i, is_dense)
            for node in layer_nodes:
                graph.add_node(node)
            # Chain the first node from the previous layer
            graph.add_edge(AEGGraphEdge(prev_id, layer_nodes[0].id))
            # Chain within the layer
            for j in range(len(layer_nodes) - 1):
                graph.add_edge(AEGGraphEdge(layer_nodes[j].id, layer_nodes[j + 1].id))
            prev_id = layer_nodes[-1].id

        # === LM head ===
        lm_head = AEGGraphNode(
            id="lm_head",
            node_type=AEGGraphNodeType.OUTPUT,
            name="lm_head",
            op_type="aeg.lm_head",
            attributes={"vocab_size": arch.vocab_size},
        )
        graph.add_node(lm_head)
        graph.add_edge(AEGGraphEdge(prev_id, lm_head.id))

        # === MLA metadata for KV manager ===
        if hasattr(graph, "set_metadata"):
            graph.set_metadata("mla_config", {
                "kv_lora_rank": arch.kv_lora_rank,
                "q_lora_rank": arch.q_lora_rank,
                "qk_rope_head_dim": arch.qk_rope_head_dim,
                "qk_nope_head_dim": arch.qk_nope_head_dim,
                "v_head_dim": arch.v_head_dim,
                "kv_compression_ratio": arch.kv_compression_ratio,
                "dense_layers": arch.first_k_dense_replace,
                "mla_layers": arch.num_layers - arch.first_k_dense_replace,
            })
            if arch.is_moe:
                graph.set_metadata("moe_config", {
                    "num_experts": arch.num_experts,
                    "num_experts_per_token": arch.num_experts_per_token,
                    "num_shared_experts": arch.num_shared_experts,
                    "moe_intermediate_size": arch.moe_intermediate_size,
                    "first_k_dense_replace": arch.first_k_dense_replace,
                })

        return graph

    def _build_transformer_layer(
        self,
        arch: MLAArchitecture,
        layer_idx: int,
        is_dense: bool,
    ) -> list[Any]:
        """Build nodes for one transformer layer (dense MHA or MLA+MoE)."""
        from aether.core.graph import AEGGraphNode, AEGGraphNodeType

        nodes = []

        # RMSNorm pre-attention
        pre_norm = AEGGraphNode(
            id=f"layer_{layer_idx}_pre_attn_norm",
            node_type=AEGGraphNodeType.OPERATION,
            name="RMSNorm",
            op_type="aeg.rmsnorm",
            layer_index=layer_idx,
            attributes={"hidden_size": arch.hidden_size},
        )
        nodes.append(pre_norm)

        # Attention: standard MHA for first K layers, MLA for the rest
        if is_dense:
            attn = AEGGraphNode(
                id=f"layer_{layer_idx}_attention",
                node_type=AEGGraphNodeType.OPERATION,
                name="MultiHeadAttention",
                op_type="aeg.attention",
                layer_index=layer_idx,
                attributes={
                    "num_heads": arch.num_heads,
                    "head_dim": arch.q_head_dim,
                    "attention_type": "mha",
                    "is_dense_layer": True,
                },
            )
        else:
            attn = AEGGraphNode(
                id=f"layer_{layer_idx}_mla_attention",
                node_type=AEGGraphNodeType.OPERATION,
                name="MLAAttention",
                op_type="aeg.mla_attention",
                layer_index=layer_idx,
                attributes={
                    "num_heads": arch.num_heads,
                    "kv_lora_rank": arch.kv_lora_rank,
                    "q_lora_rank": arch.q_lora_rank,
                    "qk_rope_head_dim": arch.qk_rope_head_dim,
                    "qk_nope_head_dim": arch.qk_nope_head_dim,
                    "v_head_dim": arch.v_head_dim,
                    "kv_compression_ratio": arch.kv_compression_ratio,
                    "attention_type": "mla",
                    "is_dense_layer": False,
                },
            )
        nodes.append(attn)

        # FFN: dense or MoE
        pre_ffn_norm = AEGGraphNode(
            id=f"layer_{layer_idx}_pre_ffn_norm",
            node_type=AEGGraphNodeType.OPERATION,
            name="RMSNorm",
            op_type="aeg.rmsnorm",
            layer_index=layer_idx,
            attributes={"hidden_size": arch.hidden_size},
        )
        nodes.append(pre_ffn_norm)

        if is_dense or not arch.is_moe:
            ffn = AEGGraphNode(
                id=f"layer_{layer_idx}_ffn",
                node_type=AEGGraphNodeType.OPERATION,
                name="SwiGLUFFN",
                op_type="aeg.swiglu_ffn",
                layer_index=layer_idx,
                attributes={"hidden_size": arch.hidden_size},
            )
        else:
            # Shared experts + routed experts
            ffn = AEGGraphNode(
                id=f"layer_{layer_idx}_moe",
                node_type=AEGGraphNodeType.OPERATION
                if hasattr(AEGGraphNodeType, "MOE_LAYER")
                else AEGGraphNodeType.OPERATION,
                name=f"MoE_{arch.num_experts}x",
                op_type="aeg.moe_layer",
                layer_index=layer_idx,
                attributes={
                    "num_experts": arch.num_experts,
                    "num_experts_per_token": arch.num_experts_per_token,
                    "num_shared_experts": arch.num_shared_experts,
                    "expert_intermediate_size": arch.moe_intermediate_size,
                },
            )
        nodes.append(ffn)

        return nodes


# ---------------------------------------------------------------------------
# Fallback graph
# ---------------------------------------------------------------------------

class _FallbackGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    @property
    def nodes(self) -> dict[str, Any]:
        return self._nodes

    def add_node(self, node: Any) -> None:
        self._nodes[getattr(node, "id", str(len(self._nodes)))] = node

    def add_edge(self, src: str, dst: str) -> None:
        pass

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_mla_model(model_path: str | Path) -> dict[str, Any]:
    """Load a DeepSeek MLA model and return the extracted AEG graph."""
    return MLALoader(model_path).load()


def is_mla_model(config: dict[str, Any]) -> bool:
    """Return True if the config describes an MLA model."""
    return (
        "kv_lora_rank" in config
        or (config.get("model_type") or "").lower().startswith("deepseek")
    )
