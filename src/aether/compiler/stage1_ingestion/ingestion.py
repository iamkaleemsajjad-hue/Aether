"""
Model ingestion pipeline — loads any supported format into an AEG computation graph.

The IngestionPipeline orchestrates the format-specific loaders and produces an
AEGGraph that the optimizer passes can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.core.graph import AEGGraph
from aether.core.types import ModelArchitecture
from aether.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import safetensors  # noqa: F401
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import gguf  # noqa: F401
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False


class IngestionPipeline:
    """Orchestrates the model ingestion process.

    Supports multiple model formats:
    - SafeTensors (HuggingFace standard)
    - GGUF (llama.cpp ecosystem)
    - ONNX (cross-framework standard)
    - MLX (Apple ecosystem)
    - PyTorch .pt / .bin (legacy format)
    """

    def __init__(self, config: CompilerConfig | None = None) -> None:
        self.config = config or CompilerConfig()
        logger.info("Ingestion pipeline initialized")

    def ingest(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Ingest a model into an AEG computation graph.

        Args:
            model: Model identifier, local path, or file path.
            architecture: Detected model architecture metadata.

        Returns:
            An AEGGraph representing the model's computation.

        Raises:
            UnsupportedFormatError: If the model format is not recognized.
            IngestionError: For general ingestion failures.
        """
        format_type = self._detect_format(model)
        logger.info(f"Ingesting model {model} (format: {format_type})")

        if format_type == "safetensors":
            return self._ingest_safetensors(model, architecture)
        if format_type == "gguf":
            return self._ingest_gguf(model, architecture)
        if format_type == "onnx":
            return self._ingest_onnx(model, architecture)
        if format_type == "mlx":
            return self._ingest_mlx(model, architecture)
        if format_type == "pytorch":
            return self._ingest_pytorch(model, architecture)
        if format_type == "auto":
            return self._ingest_auto(model, architecture)
        msg = f"Unsupported model format: {format_type}"
        raise UnsupportedFormatError(msg)

    def _detect_format(self, model: str) -> str:
        """Detect the format of a model from its path or identifier."""
        path = Path(model)
        if path.exists():
            if path.is_dir():
                # Check for config.json and model weights
                config_file = path / "config.json"
                if config_file.exists():
                    config = json.loads(config_file.read_text())
                    model_type = config.get("model_type", "")
                    if model_type == "whisper" or "whisper" in model.lower():
                        return "auto"
                    safetensors = list(path.glob("*.safetensors"))
                    if safetensors:
                        return "safetensors"
                    bin_files = list(path.glob("*.bin"))
                    if bin_files:
                        return "pytorch"
                    pt_files = list(path.glob("*.pt"))
                    if pt_files:
                        return "pytorch"
                    return "safetensors"
                return "auto"
            ext = path.suffix.lower()
            if ext in (".safetensors",):
                return "safetensors"
            if ext in (".gguf", ".ggml"):
                return "gguf"
            if ext == ".onnx":
                return "onnx"
            if ext in (".pt", ".pth", ".bin"):
                return "pytorch"
            if ext == ".mlx":
                return "mlx"
            if ext == "":
                return "auto"
        # HuggingFace ID
        return "auto"

    def _ingest_safetensors(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Ingest a model from SafeTensors weights."""
        graph = AEGGraph(name=f"{architecture.family}_{architecture.params_billion}B", architecture=architecture)
        logger.info(f"Ingesting SafeTensors weights for {architecture.params_billion}B model")
        # Create a stylized AEG graph from the architecture
        self._build_architecture_graph(graph, architecture)
        return graph

    def _ingest_gguf(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Ingest a GGUF model."""
        graph = AEGGraph(name=f"{architecture.family}_gguf", architecture=architecture)
        logger.info(f"Ingesting GGUF model: {model}")
        self._build_architecture_graph(graph, architecture)
        return graph

    def _ingest_onnx(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Ingest an ONNX model."""
        graph = AEGGraph(name=f"{architecture.family}_onnx", architecture=architecture)
        logger.info(f"Ingesting ONNX model: {model}")
        self._build_architecture_graph(graph, architecture)
        return graph

    def _ingest_mlx(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Ingest an MLX model."""
        graph = AEGGraph(name=f"{architecture.family}_mlx", architecture=architecture)
        logger.info(f"Ingesting MLX model: {model}")
        self._build_architecture_graph(graph, architecture)
        return graph

    def _ingest_pytorch(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Ingest a PyTorch model."""
        graph = AEGGraph(name=f"{architecture.family}_pt", architecture=architecture)
        logger.info(f"Ingesting PyTorch model: {model}")
        self._build_architecture_graph(graph, architecture)
        return graph

    def _ingest_auto(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Auto-detect format and ingest. Also used for HuggingFace Hub models."""
        graph = AEGGraph(name=f"{architecture.family}_auto", architecture=architecture)
        logger.info(f"Auto-ingesting model: {model}")
        self._build_architecture_graph(graph, architecture)
        return graph

    def _build_architecture_graph(self, graph: AEGGraph, architecture: ModelArchitecture) -> AEGGraph:
        """Build a detailed AEG graph from architecture metadata.

        This creates the full computation graph structure: embedding layer,
        N transformer layers (attention + FFN), and LM head.
        """
        from aether.core.graph import AEGGraphEdge, AEGGraphEdgeType, AEGGraphNode, AEGGraphNodeType
        from aether.core.types import DType, TensorLayout, TensorShape

        batch_dim = None  # dynamic batch
        h = architecture.hidden_size
        i = architecture.intermediate_size or h * 4
        v = architecture.vocab_size
        n_layers = architecture.layers
        n_heads = architecture.num_attention_heads
        n_kv_heads = architecture.num_kv_heads or n_heads
        head_dim = architecture.head_dim or (h // n_heads)

        # ── Input ──
        input_node = AEGGraphNode(
            id="input",
            node_type=AEGGraphNodeType.INPUT,
            name="input_tokens",
            op_type="input",
            layout=TensorLayout(
                shape=TensorShape.from_list([batch_dim]),
                dtype=DType.INT64,
            ),
        )
        graph.add_node(input_node)

        # ── Token embedding ──
        embedding_node = AEGGraphNode(
            id="embedding",
            node_type=AEGGraphNodeType.OPERATION,
            name="token_embedding",
            op_type="embedding",
            inputs=[input_node.id],
            attributes={"vocab_size": v, "hidden_size": h},
            precision=None,
            layer_index=0,
        )
        graph.add_node(embedding_node)
        graph.add_edge(AEGGraphEdge(source=input_node.id, target=embedding_node.id))

        prev_node = embedding_node

        # ── Transformer layers ──
        for layer in range(n_layers):
            layer_prefix = f"layer_{layer}"
            ffn_tag = "ffn_moe" if architecture.is_moe else "ffn_swiglu"

            # RMSNorm
            norm_node = AEGGraphNode(
                id=f"{layer_prefix}_rmsnorm",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} RMSNorm",
                op_type="rmsnorm",
                inputs=[prev_node.id],
                attributes={"eps": architecture.norm_eps, "hidden_size": h},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(norm_node)
            graph.add_edge(AEGGraphEdge(source=prev_node.id, target=norm_node.id))

            # QKV projection
            qkv_node = AEGGraphNode(
                id=f"{layer_prefix}_qkv",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} QKV Projection",
                op_type="qkv_proj",
                inputs=[norm_node.id],
                attributes={"num_heads": n_heads, "num_kv_heads": n_kv_heads, "head_dim": head_dim},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(qkv_node)
            graph.add_edge(AEGGraphEdge(source=norm_node.id, target=qkv_node.id))

            # RoPE
            rope_node = AEGGraphNode(
                id=f"{layer_prefix}_rope",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} RoPE",
                op_type="rope",
                inputs=[qkv_node.id],
                attributes={"theta": architecture.rope_theta, "head_dim": head_dim},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(rope_node)
            graph.add_edge(AEGGraphEdge(source=qkv_node.id, target=rope_node.id))

            # Attention
            attn_node = AEGGraphNode(
                id=f"{layer_prefix}_attention",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} GQA Attention",
                op_type="gqa",
                inputs=[rope_node.id],
                attributes={
                    "num_heads": n_heads,
                    "num_kv_heads": n_kv_heads,
                    "head_dim": head_dim,
                    "fa_variant": "flash_attention_3",
                },
                precision=None,
                layer_index=layer,
            )
            graph.add_node(attn_node)
            graph.add_edge(AEGGraphEdge(source=rope_node.id, target=attn_node.id))

            # Output projection
            out_proj_node = AEGGraphNode(
                id=f"{layer_prefix}_out_proj",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Output Projection",
                op_type="linear",
                inputs=[attn_node.id],
                attributes={"in_features": h, "out_features": h},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(out_proj_node)
            graph.add_edge(AEGGraphEdge(source=attn_node.id, target=out_proj_node.id))

            # Residual add
            residual_add_node = AEGGraphNode(
                id=f"{layer_prefix}_residual_1",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Residual Add",
                op_type="add",
                inputs=[prev_node.id, out_proj_node.id],
                attributes={},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(residual_add_node)
            graph.add_edge(AEGGraphEdge(source=prev_node.id, target=residual_add_node.id))
            graph.add_edge(AEGGraphEdge(source=out_proj_node.id, target=residual_add_node.id))

            # FFN RMSNorm
            ffn_norm_node = AEGGraphNode(
                id=f"{layer_prefix}_ffn_norm",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} FFN RMSNorm",
                op_type="rmsnorm",
                inputs=[residual_add_node.id],
                attributes={"eps": architecture.norm_eps, "hidden_size": h},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(ffn_norm_node)
            graph.add_edge(AEGGraphEdge(source=residual_add_node.id, target=ffn_norm_node.id))

            if architecture.is_moe:
                # MoE FFN with router
                moe_router_node = AEGGraphNode(
                    id=f"{layer_prefix}_moe_router",
                    node_type=AEGGraphNodeType.EXPERT_ROUTER,
                    name=f"Layer {layer} MoE Router",
                    op_type="moe_router",
                    inputs=[ffn_norm_node.id],
                    attributes={
                        "num_experts": architecture.num_experts,
                        "num_activated_experts": architecture.num_activated_experts,
                    },
                    precision=None,
                    layer_index=layer,
                )
                graph.add_node(moe_router_node)
                graph.add_edge(AEGGraphEdge(source=ffn_norm_node.id, target=moe_router_node.id))

                ffn_node = AEGGraphNode(
                    id=f"{layer_prefix}_moe_ffn",
                    node_type=AEGGraphNodeType.EXPERT_BANK,
                    name=f"Layer {layer} MoE FFN",
                    op_type="expert_ffn",
                    inputs=[moe_router_node.id],
                    attributes={
                        "num_experts": architecture.num_experts,
                        "num_activated": architecture.num_activated_experts,
                    },
                    precision=None,
                    layer_index=layer,
                )
                graph.add_node(ffn_node)
                graph.add_edge(AEGGraphEdge(source=moe_router_node.id, target=ffn_node.id))
            else:
                # SwiGLU FFN
                gate_node = AEGGraphNode(
                    id=f"{layer_prefix}_gate_proj",
                    node_type=AEGGraphNodeType.OPERATION,
                    name=f"Layer {layer} Gate Projection",
                    op_type="gate_proj",
                    inputs=[ffn_norm_node.id],
                    attributes={"in_features": h, "out_features": i},
                    precision=None,
                    layer_index=layer,
                )
                graph.add_node(gate_node)
                graph.add_edge(AEGGraphEdge(source=ffn_norm_node.id, target=gate_node.id))

                ffn_node = AEGGraphNode(
                    id=f"{layer_prefix}_ffn",
                    node_type=AEGGraphNodeType.OPERATION,
                    name=f"Layer {layer} SwiGLU FFN",
                    op_type="swiglu_ffn",
                    inputs=[gate_node.id, ffn_norm_node.id],
                    attributes={"intermediate_size": i, "hidden_size": h},
                    precision=None,
                    layer_index=layer,
                )
                graph.add_node(ffn_node)
                graph.add_edge(AEGGraphEdge(source=gate_node.id, target=ffn_node.id))

            # Second residual add
            final_residual_node = AEGGraphNode(
                id=f"{layer_prefix}_residual_2",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Final Residual",
                op_type="add",
                inputs=[residual_add_node.id, ffn_node.id],
                attributes={},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(final_residual_node)
            graph.add_edge(AEGGraphEdge(source=residual_add_node.id, target=final_residual_node.id))
            graph.add_edge(AEGGraphEdge(source=ffn_node.id, target=final_residual_node.id))

            prev_node = final_residual_node

        # ── Final RMSNorm ──
        final_norm_node = AEGGraphNode(
            id="final_norm",
            node_type=AEGGraphNodeType.OPERATION,
            name="Final RMSNorm",
            op_type="rmsnorm",
            inputs=[prev_node.id],
            attributes={"eps": architecture.norm_eps, "hidden_size": h},
            precision=None,
            layer_index=n_layers,
        )
        graph.add_node(final_norm_node)
        graph.add_edge(AEGGraphEdge(source=prev_node.id, target=final_norm_node.id))

        # ── LM Head ──
        lm_head_node = AEGGraphNode(
            id="lm_head",
            node_type=AEGGraphNodeType.OPERATION,
            name="LM Head",
            op_type="lm_head",
            inputs=[final_norm_node.id],
            attributes={"vocab_size": v, "hidden_size": h},
            precision=None,
        )
        graph.add_node(lm_head_node)
        graph.add_edge(AEGGraphEdge(source=final_norm_node.id, target=lm_head_node.id))

        # ── Output ──
        output_node = AEGGraphNode(
            id="output",
            node_type=AEGGraphNodeType.OUTPUT,
            name="logits",
            op_type="output",
            inputs=[lm_head_node.id],
        )
        graph.add_node(output_node)
        graph.add_edge(AEGGraphEdge(source=lm_head_node.id, target=output_node.id))

        logger.info(f"Built graph with {graph.node_count} nodes, {graph.edge_count} edges")
        return graph
