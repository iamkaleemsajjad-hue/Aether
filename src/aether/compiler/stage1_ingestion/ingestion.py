"""
Model ingestion pipeline — loads any supported format into an AEG computation graph.

The IngestionPipeline orchestrates the format-specific loaders and produces an
AEGGraph that the optimizer passes can consume.
"""

from __future__ import annotations

import json
import os
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
        # The low-level ingestion API is deterministic and local-only by
        # default.  Network materialization belongs to Compiler, which passes
        # its explicit ``skip_download`` policy through this object.
        self.config = config or CompilerConfig(skip_download=True)
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
            graph = self._ingest_safetensors(model, architecture)
        elif format_type == "gguf":
            graph = self._ingest_gguf(model, architecture)
        elif format_type == "onnx":
            graph = self._ingest_onnx(model, architecture)
        elif format_type == "mlx":
            graph = self._ingest_mlx(model, architecture)
        elif format_type == "pytorch":
            graph = self._ingest_pytorch(model, architecture)
        elif format_type == "auto":
            graph = self._ingest_auto(model, architecture)
        else:
            msg = f"Unsupported model format: {format_type}"
            raise UnsupportedFormatError(msg)
        local_path = Path(model)
        if local_path.exists() and (
            local_path.is_dir() or local_path.suffix.lower() in {".gguf", ".ggml"}
        ):
            graph.set_metadata("source_model_path", str(local_path.resolve()))
            graph.set_metadata("source_format", format_type)
        return graph

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
        """Ingest a model from SafeTensors weights.

        Builds the architecture skeleton, then attaches the real weight tensors
        read off disk so downstream passes (sensitivity, pruning, quantization)
        operate on actual values rather than metadata alone.
        """
        graph = AEGGraph(name=f"{architecture.family}_{architecture.params_billion}B", architecture=architecture)
        logger.info(f"Ingesting SafeTensors weights for {architecture.params_billion}B model")
        self._build_architecture_graph(graph, architecture)
        self._attach_safetensors_weights(graph, model)
        return graph

    def _attach_safetensors_weights(self, graph: AEGGraph, model: str) -> int:
        """Load weights from disk and bind them to matching graph nodes.

        Returns the number of nodes that received a weight tensor. Missing files or
        an unavailable ``safetensors`` package are logged and skipped rather than
        raised: the architecture graph is still valid and useful without weights.
        """
        if not HAS_SAFETENSORS:
            logger.warning("safetensors not installed; graph built without weights")
            graph.set_metadata("weights_attached", 0)
            return 0

        path = Path(model)
        if not path.exists():
            logger.info("Model path %s is not local; graph built without weights", model)
            graph.set_metadata("weights_attached", 0)
            return 0

        from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

        try:
            tensors = SafeTensorsLoader(path).load()
        except Exception as exc:
            logger.warning("Could not read SafeTensors weights from %s: %s", model, exc)
            graph.set_metadata("weights_attached", 0)
            return 0

        attached = self._bind_weights(graph, tensors)
        graph.set_metadata("weights_attached", attached)
        graph.set_metadata("weight_tensor_count", len(tensors))
        logger.info("Attached %d/%d weight tensors to graph nodes", attached, len(tensors))
        return attached

    def _bind_weights(self, graph: AEGGraph, tensors: dict[str, Any]) -> int:
        """Match checkpoint tensor names to graph node ids and attach the arrays.

        Checkpoint layouts vary across model families, so matching is done by
        normalising both sides to a comparable suffix rather than by exact name.
        """
        import numpy as np

        lookup: dict[tuple[int | None, str | None], list[tuple[str, Any]]] = {}
        for name, tensor in tensors.items():
            key = self._normalise_weight_name(name)
            # A key without a component identifies nothing: several unrelated
            # names reduce to (layer, None), which would match each other.
            if key[1] is None:
                continue
            value = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            lookup.setdefault(key, []).append((name, value))

        attached = 0
        for node in graph:
            if not hasattr(node, "add_attribute"):
                continue
            key = self._normalise_weight_name(getattr(node, "id", ""))
            if key[1] is None:
                continue
            candidates = lookup.get(key)
            if not candidates:
                continue
            # A node may stand in for several checkpoint tensors.  Preserve the
            # real projection semantics instead of silently dropping parameters:
            # fused QKV nodes receive Q/K/V stacked in that order, while the
            # single SwiGLU node keeps the separate up projection as metadata for
            # the weight packer.
            if key[1] == "qkv" and len(candidates) > 1:
                def projection_order(item: tuple[str, Any]) -> int:
                    name = item[0].lower()
                    return (
                        0
                        if "q_proj" in name or "attn_q" in name
                        else 1
                        if "k_proj" in name or "attn_k" in name
                        else 2
                    )

                ordered = sorted(candidates, key=projection_order)
                source_name = "+".join(name for name, _ in ordered)
                import numpy as np

                value = np.concatenate([np.asarray(array) for _, array in ordered], axis=0)
            else:
                source_name, value = candidates[0]
            node.add_attribute("weight", value)
            node.add_attribute("weight_shape", list(value.shape))
            node.add_attribute("weight_source", source_name)
            if key[1] == "gate_proj":
                up_candidates = [
                    item
                    for item in candidates
                    if "up_proj" in item[0].lower() or "ffn_up" in item[0].lower()
                ]
                gate_candidates = [
                    item
                    for item in candidates
                    if "gate_proj" in item[0].lower() or "ffn_gate" in item[0].lower()
                ]
                if up_candidates and gate_candidates:
                    node.add_attribute("up_weight", up_candidates[0][1])
                    node.add_attribute("up_weight_source", up_candidates[0][0])
            if len(candidates) > 1:
                node.add_attribute("fused_weight_sources", [n for n, _ in candidates])
            attached += 1
        return attached

    #: Checkpoint component names mapped to the graph's node-id vocabulary. Node
    #: ids are coarser than checkpoint names (one ``qkv`` node stands in for
    #: separate q/k/v projections), so several tensors can map to one node. Every
    #: canonical name also maps to itself so node ids resolve through the same
    #: table as checkpoint names.
    _COMPONENT_ALIASES: dict[str, str] = {
        # Checkpoint spellings.
        "q_proj": "qkv",
        "k_proj": "qkv",
        "v_proj": "qkv",
        "qkv_proj": "qkv",
        # Standard llama.cpp/GGUF spellings.
        "attn_q": "qkv",
        "attn_k": "qkv",
        "attn_v": "qkv",
        "o_proj": "out_proj",
        "attn_output": "out_proj",
        "embed_tokens": "embedding",
        "token_embd": "embedding",
        "wte": "embedding",
        "input_layernorm": "rmsnorm",
        "attn_norm": "rmsnorm",
        "post_attention_layernorm": "ffn_norm",
        "ffn_norm": "ffn_norm",
        "down_proj": "ffn",
        "ffn_down": "ffn",
        "up_proj": "gate_proj",
        "ffn_up": "gate_proj",
        "ffn_gate": "gate_proj",
        # Canonical names, as used in graph node ids.
        "qkv": "qkv",
        "out_proj": "out_proj",
        "embedding": "embedding",
        "rmsnorm": "rmsnorm",
        "ffn_norm": "ffn_norm",
        "ffn": "ffn",
        "gate_proj": "gate_proj",
        "lm_head": "lm_head",
        "output_norm": "final_norm",
        "final_norm": "final_norm",
    }

    #: Structural path segments that carry no identifying information.
    _IGNORED_SEGMENTS = frozenset({"model", "transformer", "module", "self_attn", "mlp", "blk", "weight"})

    @classmethod
    def _normalise_weight_name(cls, name: str) -> tuple[int | None, str | None]:
        """Reduce a tensor or node name to a ``(layer_index, component)`` key.

        Checkpoints and graph nodes disagree on both separators (``.`` vs ``_``)
        and component naming (``self_attn.q_proj`` vs ``qkv``), so names are parsed
        into a structured key instead of string-matched. Returns ``(None, None)``
        when nothing identifiable can be extracted.
        """
        import re

        normalized = name.lower().replace(".", "_")
        mtp_match = re.search(
            r"(?:^|_)(?:mtp_head|mtp_heads|mtp_lm_head|mtp_lm_heads|lmtp_head|linear_mtp)_?(\d+)(?:_|$)",
            normalized,
        )
        if mtp_match:
            return None, f"mtp_head_{int(mtp_match.group(1))}"

        tokens = [t for t in re.split(r"[._]", name.lower()) if t]
        layer_index: int | None = None
        component: str | None = None

        # GGUF uses ``output.weight`` for the LM head.  The bare graph node
        # ``output`` is structural and must remain unidentifiable, otherwise
        # it would be counted as a second attachment to ``lm_head``.
        if tokens == ["output", "weight"]:
            return None, "lm_head"

        for position, token in enumerate(tokens):
            # "layer"/"layers" followed by its index.
            if token in ("layer", "layers", "blocks", "blk", "h") and position + 1 < len(tokens):
                if tokens[position + 1].isdigit():
                    layer_index = int(tokens[position + 1])
                continue
            if token.isdigit() or token in cls._IGNORED_SEGMENTS:
                continue
            # Rebuild multi-token component names such as "q" + "proj".
            for width in (3, 2, 1):
                candidate = "_".join(tokens[position : position + width])
                if candidate in cls._COMPONENT_ALIASES:
                    component = cls._COMPONENT_ALIASES[candidate]
                    break
            if component is not None:
                break

        return layer_index, component

    def _ingest_gguf(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """
        Ingest a GGUF model.

        Parses the GGUF binary, dequantizes weight tensors to float32, and
        binds them to matching graph nodes using the standard weight-binding
        pipeline.
        """
        graph = AEGGraph(name=f"{architecture.family}_gguf", architecture=architecture)
        logger.info(f"Ingesting GGUF model: {model}")
        self._build_architecture_graph(graph, architecture)
        self._attach_gguf_weights(graph, model)
        return graph

    def _attach_gguf_weights(self, graph: "AEGGraph", model: str) -> int:  # type: ignore[name-defined]
        """Load GGUF tensor data and bind to graph nodes."""
        from pathlib import Path as _Path

        path = _Path(model)
        if not path.exists() or path.suffix.lower() not in (".gguf", ".ggml"):
            graph.set_metadata("weights_attached", 0)
            return 0
        try:
            from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader
            import numpy as np

            reader = GGUFReader(path)
            tensors: dict[str, Any] = {}
            for name, info in reader.tensors.items():
                try:
                    tensors[name] = reader.dequantize(name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not dequantize GGUF tensor %s: %s", name, exc)
            attached = self._bind_weights(graph, tensors)
            graph.set_metadata("weights_attached", attached)
            graph.set_metadata("weight_tensor_count", len(tensors))
            graph.set_metadata("gguf_architecture", reader.architecture)
            logger.info("Attached %d/%d GGUF tensors to graph nodes", attached, len(tensors))
            return attached
        except Exception as exc:
            logger.warning("Could not load GGUF weights from %s: %s", model, exc)
            graph.set_metadata("weights_attached", 0)
            return 0

    def _ingest_onnx(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """
        Ingest an ONNX model.

        Lowers ONNX op nodes into AEG graph nodes, extracts initializer
        tensors, and binds weight tensors to matching graph nodes.
        """
        graph = AEGGraph(name=f"{architecture.family}_onnx", architecture=architecture)
        logger.info(f"Ingesting ONNX model: {model}")
        self._build_architecture_graph(graph, architecture)
        self._attach_onnx_weights(graph, model)
        return graph

    def _attach_onnx_weights(self, graph: "AEGGraph", model: str) -> int:  # type: ignore[name-defined]
        """Load ONNX initializer tensors and bind to graph nodes."""
        from pathlib import Path as _Path

        path = _Path(model)
        if not path.exists() or path.suffix.lower() != ".onnx":
            graph.set_metadata("weights_attached", 0)
            return 0
        try:
            from aether.compiler.stage1_ingestion.onnx_loader import ONNXLoader

            data = ONNXLoader(path).load()
            tensors = data.get("initializers", {})
            attached = self._bind_weights(graph, tensors)
            graph.set_metadata("weights_attached", attached)
            graph.set_metadata("weight_tensor_count", len(tensors))
            graph.set_metadata("onnx_opset", data.get("opset", 17))
            logger.info("Attached %d/%d ONNX initializers to graph nodes", attached, len(tensors))
            return attached
        except Exception as exc:
            logger.warning("Could not load ONNX weights from %s: %s", model, exc)
            graph.set_metadata("weights_attached", 0)
            return 0

    def _ingest_mlx(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """
        Ingest an MLX model.

        Loads safetensors or npz weights from an MLX checkpoint directory
        and binds them to the architecture graph.
        """
        graph = AEGGraph(name=f"{architecture.family}_mlx", architecture=architecture)
        logger.info(f"Ingesting MLX model: {model}")
        self._build_architecture_graph(graph, architecture)
        self._attach_mlx_weights(graph, model)
        return graph

    def _attach_mlx_weights(self, graph: "AEGGraph", model: str) -> int:  # type: ignore[name-defined]
        """Load MLX weight tensors and bind to graph nodes."""
        from pathlib import Path as _Path

        path = _Path(model)
        if not path.exists():
            graph.set_metadata("weights_attached", 0)
            return 0
        try:
            from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader

            data = MLXLoader(path).load()
            tensors = data.get("weights", {})
            attached = self._bind_weights(graph, tensors)
            graph.set_metadata("weights_attached", attached)
            graph.set_metadata("weight_tensor_count", len(tensors))
            graph.set_metadata("mlx_format", data.get("format", "unknown"))
            logger.info("Attached %d/%d MLX tensors to graph nodes", attached, len(tensors))
            return attached
        except Exception as exc:
            logger.warning("Could not load MLX weights from %s: %s", model, exc)
            graph.set_metadata("weights_attached", 0)
            return 0

    def _ingest_pytorch(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """
        Ingest a PyTorch model.

        Loads state dict tensors from .pt/.pth/.bin checkpoints and binds
        them to the architecture graph.
        """
        graph = AEGGraph(name=f"{architecture.family}_pt", architecture=architecture)
        logger.info(f"Ingesting PyTorch model: {model}")
        self._build_architecture_graph(graph, architecture)
        self._attach_pytorch_weights(graph, model)
        return graph

    def _attach_pytorch_weights(self, graph: "AEGGraph", model: str) -> int:  # type: ignore[name-defined]
        """Load PyTorch checkpoint tensors and bind to graph nodes."""
        from pathlib import Path as _Path

        path = _Path(model)
        if not path.exists():
            graph.set_metadata("weights_attached", 0)
            return 0
        try:
            from aether.compiler.stage1_ingestion.pytorch_loader import PyTorchLoader

            data = PyTorchLoader(path).load()
            tensors = data.get("weights", {})
            attached = self._bind_weights(graph, tensors)
            graph.set_metadata("weights_attached", attached)
            graph.set_metadata("weight_tensor_count", len(tensors))
            graph.set_metadata("pytorch_format", data.get("format", "unknown"))
            logger.info("Attached %d/%d PyTorch tensors to graph nodes", attached, len(tensors))
            return attached
        except Exception as exc:
            logger.warning("Could not load PyTorch weights from %s: %s", model, exc)
            graph.set_metadata("weights_attached", 0)
            return 0

    def _ingest_auto(self, model: str, architecture: ModelArchitecture) -> AEGGraph:
        """Auto-detect format and ingest. Also used for HuggingFace Hub models."""
        graph = AEGGraph(name=f"{architecture.family}_auto", architecture=architecture)
        logger.info(f"Auto-ingesting model: {model}")
        self._build_architecture_graph(graph, architecture)
        source = Path(model)
        downloaded = False
        if not source.exists():
            if self.config.skip_download:
                graph.set_metadata("weights_attached", 0)
                graph.set_metadata("source_model_id", model)
                logger.warning("Skipping model download for %s by compiler configuration", model)
                return graph
            source = Path(self._download_hf_snapshot(model))
            downloaded = True
        graph.set_metadata("source_model_path", str(source.resolve()))
        # A local directory may still hold SafeTensors shards even when format
        # detection fell through to "auto"; attach them when they are there.
        attached = 0
        if list(source.glob("*.safetensors")) or list(source.glob("*.safetensors.index.json")):
            attached = self._attach_safetensors_weights(graph, str(source))
        elif list(source.glob("*.gguf")):
            attached = self._attach_gguf_weights(graph, str(next(source.glob("*.gguf"))))
        elif list(source.glob("*.onnx")):
            attached = self._attach_onnx_weights(graph, str(next(source.glob("*.onnx"))))
        elif list(source.glob("*.bin")) or list(source.glob("*.pt")):
            attached = self._attach_pytorch_weights(graph, str(source))
        graph.set_metadata("weights_attached", attached)
        if downloaded and attached == 0:
            raise IngestionError(
                f"No supported model weights were found in materialized model {source}. "
                "A graph-only artifact is not a runnable compilation."
            )
        return graph

    def _download_hf_snapshot(self, model: str) -> str:
        """Materialize a real Hugging Face snapshot for compiler ingestion.

        Downloading is explicit and bounded.  A failed or incomplete snapshot
        raises instead of allowing the compiler to continue with fabricated
        parameters.
        """
        try:
            from huggingface_hub import snapshot_download
            from aether.utils.file_io import aether_cache_dir

            cache_dir = aether_cache_dir(self.config.cache_dir) / "hf_snapshots"
            cache_dir.mkdir(parents=True, exist_ok=True)
            timeout = float(os.environ.get("AETHER_HF_ETAG_TIMEOUT_S", "10"))
            path = snapshot_download(
                repo_id=model,
                cache_dir=cache_dir,
                etag_timeout=timeout,
                local_files_only=os.environ.get("AETHER_HF_OFFLINE", "").lower() in {"1", "true", "yes"},
                allow_patterns=[
                    "config.json", "generation_config.json", "tokenizer.*", "*.json",
                    "*.safetensors", "*.safetensors.index.json", "*.bin", "*.pt", "*.pth",
                    "*.gguf", "*.onnx", "*.model", "*.txt", "*.py",
                ],
            )
            return path
        except Exception as exc:
            raise IngestionError(
                f"Unable to materialize Hugging Face model {model!r}; no weights were loaded: {exc}"
            ) from exc

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

        # Materialize declared native MTP classifier heads so their real
        # checkpoint tensors can be bound and Pass 10 can emit executable
        # speculation blobs.
        mtp_heads = int(getattr(architecture, "mtp_heads", 0) or 0)
        for head_index in range(mtp_heads):
            mtp_node = AEGGraphNode(
                id=f"mtp_head_{head_index}",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"mtp_head_{head_index}",
                op_type="mtp_head",
                inputs=[final_norm_node.id],
                attributes={
                    "head_index": head_index,
                    "vocab_size": v,
                    "hidden_size": h,
                },
                precision=None,
            )
            graph.add_node(mtp_node)
            graph.add_edge(AEGGraphEdge(source=final_norm_node.id, target=mtp_node.id))

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
