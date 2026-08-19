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
    # Probe torch WITHOUT importing it: importing the full package here would
    # make every framework-free path (AEG loading, CPU execution) pull torch
    # into sys.modules. Only the PyTorch loader imports it, lazily.
    from importlib.util import find_spec as _find_spec

    HAS_TORCH = _find_spec("torch") is not None
except (ImportError, ValueError):
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
        # Metadata stashed by a specialised loader whose graph could not host
        # the checkpoint weights.  The generic graph inherits these keys so a
        # fallback still preserves MLA/MoE/VLM/SSM structure information.
        self._specialised_fallback_metadata: dict[str, Any] = {}
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

        # --- Specialised sub-loaders (run before generic format dispatch) ---
        # These produce a richer graph than the generic architecture builder.
        graph = self._try_specialised_loader(model, architecture, format_type)
        if graph is None:
            # Generic format dispatch
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
        # A specialised loader may have produced structure metadata that the
        # generic graph cannot recompute (MLA ranks, MoE layout, ...).  Merge
        # it without clobbering anything the generic path already recorded.
        if self._specialised_fallback_metadata:
            for key, value in self._specialised_fallback_metadata.items():
                if graph.get_metadata(key) is None:
                    graph.set_metadata(key, value)
            self._specialised_fallback_metadata = {}
        return graph

    def _try_specialised_loader(
        self,
        model: str,
        architecture: ModelArchitecture,
        format_type: str,
    ) -> "AEGGraph | None":
        """Attempt to load using a specialised sub-loader.

        Returns a populated AEGGraph when a specialised loader succeeds, or
        None to fall through to the generic format dispatch.  All failures are
        caught and logged so the caller can fall back gracefully.
        """
        path = Path(model)

        # Load config.json to detect specialised architectures
        config: dict[str, Any] = {}
        config_path = (path / "config.json") if path.is_dir() else None
        if config_path and config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        arch_family = (getattr(architecture, "family", "") or "").lower()
        model_type = (config.get("model_type") or arch_family).lower()

        def _qualified(result: dict[str, Any], loader_name: str) -> AEGGraph | None:
            """Wrap a specialised result and verify it can host the checkpoint.

            Returns the graph when it is runnable, otherwise ``None`` so the
            generic dispatch path takes over.  Structure metadata from the
            specialised loader is stashed for the generic graph either way,
            guaranteeing compile-time invariants (every required tensor bound)
            never trade away the specialised loader's architectural insight.
            """
            candidate = self._wrap_specialised_result(result, architecture)
            attached = self._attach_checkpoint_weights(candidate, model, format_type)
            unbound = candidate.get_metadata("unbound_weight_names") or []
            has_checkpoint = self._local_checkpoint_exists(model)
            if attached > 0 and not unbound:
                return candidate
            if not has_checkpoint:
                # No weights anywhere: keep the richer specialised graph and
                # let the packaging invariant fail loudly downstream.
                return candidate
            self._specialised_fallback_metadata.update(
                {
                    key: value
                    for key, value in (candidate.metadata or {}).items()
                    if key not in {
                        "weights_attached", "weight_tensor_count",
                        "bound_weight_count", "source_tensor_count",
                        "unbound_weight_names", "source_model_path", "source_format",
                        # The generic graph is NOT specialised-loader output;
                        # claiming so would misreport the ingestion path.
                        "specialised_loader_arch", "specialised_loader_format",
                    }
                }
            )
            logger.info(
                "%s graph cannot host the local checkpoint (%d tensors attached, "
                "%d unbound); falling back to generic ingestion with %s metadata",
                loader_name,
                attached,
                len(unbound),
                loader_name,
            )
            return None

        # --- MLA (DeepSeek V2/V3/R1) ---
        if "deepseek" in model_type or "kv_lora_rank" in config:
            try:
                from aether.compiler.stage1_ingestion.mla_loader import MLALoader
                if path.exists():
                    graph = _qualified(MLALoader(path).load(), "MLALoader")
                    if graph is not None:
                        logger.info("Ingested via MLALoader (DeepSeek MLA)")
                        return graph
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"MLALoader failed, falling back: {exc}")

        # --- MoE (Mixtral, Qwen MoE, Jamba, DBRX, OLMoE) ---
        moe_keys = {"num_local_experts", "n_routed_experts", "moe_layer_frequency"}
        if moe_keys & set(config.keys()) or getattr(architecture, "is_moe", False):
            try:
                from aether.compiler.stage1_ingestion.moe_loader import MoELoader
                if path.exists():
                    graph = _qualified(MoELoader(path).load(), "MoELoader")
                    if graph is not None:
                        logger.info("Ingested via MoELoader")
                        return graph
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"MoELoader failed, falling back: {exc}")

        # --- Video models ---
        video_keys = {"max_frames", "video_config", "num_video_query_token"}
        if (video_keys & set(config.keys())
                or any(k in model_type for k in ("video_llama", "videochat", "llava_video"))):
            try:
                from aether.compiler.stage1_ingestion.video_loader import VideoModelLoader
                if path.exists():
                    graph = _qualified(VideoModelLoader(path).load(), "VideoModelLoader")
                    if graph is not None:
                        logger.info("Ingested via VideoModelLoader")
                        return graph
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"VideoModelLoader failed, falling back: {exc}")

        # --- VLM (LLaVA, Qwen-VL, InternVL, PaliGemma, etc.) ---
        vlm_types = {"llava", "qwen2_vl", "internvl", "paligemma", "phi3_v", "pixtral"}
        has_vision_config = "vision_config" in config or "visual_config" in config
        if any(vt in model_type for vt in vlm_types) or has_vision_config:
            try:
                from aether.compiler.stage1_ingestion.vlm_loader import VLMLoader
                if path.exists():
                    graph = _qualified(VLMLoader(path).load(), "VLMLoader")
                    if graph is not None:
                        logger.info("Ingested via VLMLoader")
                        return graph
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"VLMLoader failed, falling back: {exc}")

        # --- SSM / Mamba / Jamba ---
        ssm_types = {"mamba", "jamba", "rwkv", "retnet", "ssm"}
        if any(st in model_type for st in ssm_types):
            try:
                from aether.compiler.stage1_ingestion.ssm_loader import SSMLoader
                if path.exists():
                    graph = _qualified(SSMLoader(path).load(), "SSMLoader")
                    if graph is not None:
                        logger.info("Ingested via SSMLoader")
                        return graph
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"SSMLoader failed, falling back: {exc}")

        # --- BERT / RoBERTa / DeBERTa / ELECTRA / ALBERT (encoder-only) ---
        encoder_families = {
            "bert", "roberta", "deberta", "electra", "albert",
            "bert_family", "roberta_family", "deberta_family",
            "electra_family", "albert_family",
        }
        is_encoder = (
            getattr(architecture, "is_encoder", False)
            or any(ef in model_type for ef in encoder_families)
            or any(ef in arch_family for ef in encoder_families)
        )
        if is_encoder:
            # Build encoder graph directly without a separate loader module;
            # BERT-family models are architecturally simple enough that the
            # structure can be synthesised from architecture metadata alone.
            try:
                from aether.core.graph import AEGGraph
                enc_graph = AEGGraph(
                    name=f"{arch_family or 'bert'}",
                    architecture=architecture,
                )
                self._build_encoder_graph(enc_graph, architecture)
                enc_graph.set_metadata("is_encoder", True)
                enc_graph.set_metadata("encoder_family", arch_family or model_type)
                # Attach weights from local checkpoint if available
                if path.exists():
                    attached = self._attach_checkpoint_weights(enc_graph, model, format_type)
                    enc_graph.set_metadata("weights_attached", attached > 0)
                logger.info(
                    "Ingested encoder model %s (%s family, %d layers, %dD hidden)",
                    model,
                    arch_family,
                    architecture.layers,
                    architecture.hidden_size,
                )
                return enc_graph
            except Exception as exc:  # noqa: BLE001
                logger.warning("Encoder graph build failed for %s: %s", model, exc)

        return None  # No specialised loader matched — use generic dispatch

    def _attach_checkpoint_weights(
        self,
        graph: "AEGGraph",
        model: str,
        format_type: str,
    ) -> int:
        """Bind local checkpoint tensors onto a specialised-loader graph.

        Specialised loaders build structure-rich graphs but historically never
        attached weights, which silently produced un-runnable artifacts.  This
        reuses the exact per-format attachers the generic dispatch path uses so
        accounting metadata (``weights_attached``, ``unbound_weight_names``)
        is populated identically for both paths.
        """
        try:
            if format_type == "safetensors":
                return self._attach_safetensors_weights(graph, model)
            if format_type == "gguf":
                return self._attach_gguf_weights(graph, model)
            if format_type == "onnx":
                return self._attach_onnx_weights(graph, model)
            if format_type == "mlx":
                return self._attach_mlx_weights(graph, model)
            if format_type == "pytorch":
                return self._attach_pytorch_weights(graph, model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Checkpoint weight attachment failed for %s: %s", model, exc)
        return 0

    def _local_checkpoint_exists(self, model: str) -> bool:
        """Return True when a local directory or file carries model weights."""
        path = Path(model)
        if not path.exists():
            return False
        if path.is_file():
            return path.suffix.lower() in {
                ".safetensors", ".gguf", ".ggml", ".onnx", ".mlx", ".pt", ".bin",
            }
        weight_patterns = ("*.safetensors", "*.gguf", "*.onnx", "*.pt", "*.bin")
        return any(
            entry
            for pattern in weight_patterns
            for entry in path.glob(pattern)
        )

    def _wrap_specialised_result(
        self,
        result: dict[str, Any],
        architecture: ModelArchitecture,
    ) -> AEGGraph:
        """Convert a specialised loader result dict into an AEGGraph.

        If the result already contains an AEGGraph (keyed ``"graph"``), it is
        returned directly after attaching architecture metadata.  Otherwise a
        skeleton AEGGraph is built from the result weights.
        """
        if "graph" in result and isinstance(result["graph"], AEGGraph):
            graph = result["graph"]
        else:
            # Fallback: construct a generic architecture graph
            graph = AEGGraph(
                name=f"{getattr(architecture, 'family', 'model')}",
                architecture=architecture,
            )
            self._build_architecture_graph(graph, architecture)
            # Attach weights if present
            if "weights" in result:
                self._attach_weight_dict(graph, result["weights"])

        # Record specialised loader metadata
        if hasattr(graph, "set_metadata"):
            if "architecture" in result:
                loader_arch = result["architecture"]
                graph.set_metadata("specialised_loader_arch", str(type(loader_arch).__name__))
            if "format" in result:
                graph.set_metadata("specialised_loader_format", result["format"])

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

        Records architecture-aware accounting metadata on the graph:
        ``source_tensor_count`` (logical checkpoint tensors), ``bound_weight_count``
        (logical tensors consumed by a node, counting every member of a fused
        group), and ``unbound_weight_names`` (logical tensors with no valid
        graph mapping). Compilation treats a non-empty unbound list as a defect
        rather than silently dropping parameters.
        """
        import numpy as np

        lookup: dict[tuple[int | None, str | None], list[tuple[str, Any]]] = {}
        unresolvable: list[str] = []
        for name, tensor in tensors.items():
            key = self._normalise_weight_name(name)
            if key[1] is None:
                unresolvable.append(name)
                continue
            if hasattr(tensor, "float") and getattr(tensor, "dtype", None) is not None and str(tensor.dtype).endswith("bfloat16"):
                tensor = tensor.float()
            value = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            lookup.setdefault(key, []).append((name, value))

        attached = 0
        bound_names: list[str] = []
        unbound_names: list[str] = []
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
                # A BERT-style checkpoint stores Q/K/V weights and biases
                # under the same normalized component.  Only matrices may be
                # fused into QKV; treating one-dimensional biases as matrices
                # corrupts the projection layout and can silently bind a bias
                # as a model weight.
                matrix_candidates = [
                    item for item in candidates if np.asarray(item[1]).ndim >= 2
                ]
                bias_candidates = [
                    item for item in candidates
                    if np.asarray(item[1]).ndim == 1 and "bias" in item[0].lower()
                ]
                candidates_for_projection = matrix_candidates or candidates

                def projection_order(item: tuple[str, Any]) -> int:
                    name = item[0].lower()
                    return (
                        0
                        if "q_proj" in name or "attn_q" in name or name.endswith("_q")
                        else 1
                        if "k_proj" in name or "attn_k" in name or name.endswith("_k")
                        else 2
                    )

                ordered = sorted(candidates_for_projection, key=projection_order)
                source_name = "+".join(name for name, _ in ordered)
                import numpy as np

                arrays = [np.asarray(arr) for _, arr in ordered]
                # GQA models (Llama-3, Qwen2, Mistral) have Q with more heads
                # than K/V, so shapes differ on axis=0 and concatenation fails.
                # In that case, store each projection separately so the weight
                # packer can handle them individually.
                try:
                    # Only fuse when all shapes match on axis=0 (full MHA).
                    if len({a.shape[0] for a in arrays}) == 1:
                        value = np.concatenate(arrays, axis=0)
                    else:
                        # GQA: store Q, K, V arrays as separate attributes.
                        q_arr = next((a for (n, _), a in zip(ordered, arrays) if "q" in n.lower().split(".")[-1].split("_")), arrays[0])
                        k_arr = next((a for (n, _), a in zip(ordered, arrays) if "k" in n.lower().split(".")[-1].split("_")), arrays[1] if len(arrays) > 1 else arrays[0])
                        v_arr = next((a for (n, _), a in zip(ordered, arrays) if "v" in n.lower().split(".")[-1].split("_")), arrays[2] if len(arrays) > 2 else arrays[-1])
                        node.add_attribute("q_weight", q_arr)
                        node.add_attribute("k_weight", k_arr)
                        node.add_attribute("v_weight", v_arr)
                        node.add_attribute("weight_source", source_name)
                        node.add_attribute("fused_weight_sources", [n for n, _ in ordered])
                        node.add_attribute("is_gqa", True)
                        bound_names.extend(name for name, _ in candidates)
                        self._attach_projection_biases(node, bias_candidates)
                        attached += 1
                        continue  # Skip the generic node.add_attribute("weight", ...) below
                except (ValueError, Exception) as qkv_exc:
                    # Last resort: store first available array and log the issue.
                    logger.debug(
                        "QKV fusion failed for node %s (%s); storing Q projection only: %s",
                        getattr(node, "id", "?"),
                        source_name,
                        qkv_exc,
                    )
                    value = arrays[0]

            else:
                # Prefer the matrix/scale tensor over a same-key bias.  A
                # normalized key is intentionally shared by a parameter and
                # its bias, so selecting candidates[0] is not deterministic
                # across checkpoint writers.
                value_candidates = [
                    item for item in candidates
                    if np.asarray(item[1]).ndim >= 2
                ] or sorted(
                    candidates,
                    key=lambda item: (
                        0 if item[0].lower().endswith(".weight") else 1,
                        item[0],
                    ),
                )
                source_name, value = value_candidates[0]
            node.add_attribute("weight", value)
            node.add_attribute("weight_shape", list(value.shape))
            node.add_attribute("weight_source", source_name)
            bias_candidates = [
                item for item in candidates
                if np.asarray(item[1]).ndim == 1 and "bias" in item[0].lower()
            ]
            if key[1] == "qkv":
                # Full-MHA fusion reaches this common path; GQA reaches the
                # early branch above.  Both forms must retain all three real
                # bias vectors rather than selecting the first one.
                self._attach_projection_biases(node, bias_candidates)
            elif bias_candidates:
                node.add_attribute("bias", bias_candidates[0][1])
                node.add_attribute("bias_source", bias_candidates[0][0])
            # Every candidate consumed by this node counts as a bound logical
            # tensor, even when several are fused into one physical array.
            bound_names.extend(name for name, _ in candidates)
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

        bound_set = set(bound_names)
        for key_group in lookup.values():
            for name, _ in key_group:
                if name not in bound_set:
                    unbound_names.append(name)
        graph.set_metadata("source_tensor_count", len(tensors))
        graph.set_metadata("bound_weight_count", len(bound_set))
        graph.set_metadata(
            "unbound_weight_names", sorted(unbound_names + unresolvable)
        )
        return attached

    @staticmethod
    def _attach_projection_biases(node: Any, candidates: list[tuple[str, Any]]) -> None:
        """Attach Q/K/V bias vectors to a fused projection node.

        Biases are kept separate from the fused matrices because grouped-query
        attention may have different K/V row counts.  The quantizer can then
        persist them under their canonical per-projection names.
        """
        for name, value in candidates:
            lowered = name.lower()
            projection = "q" if ".query." in lowered or "q_proj" in lowered else (
                "k" if ".key." in lowered or "k_proj" in lowered else (
                    "v" if ".value." in lowered or "v_proj" in lowered else None
                )
            )
            if projection is not None:
                node.add_attribute(f"{projection}_bias", value)

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
        # GPT-2 / DialoGPT layernorm spellings.
        "ln_1": "rmsnorm",
        "ln_2": "ffn_norm",
        "ln_f": "final_norm",
        "lnf": "final_norm",
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
        # BERT / RoBERTa / DeBERTa / ELECTRA spellings:
        "query": "qkv",
        "key": "qkv",
        "value": "qkv",
        "word_embeddings": "embedding",
        "position_embeddings": "position_embedding",
        "token_type_embeddings": "token_type_embedding",
        "token_embeddings": "embedding",
        # BERT intermediate (FFN first layer) and output (FFN second layer):
        "intermediate_dense": "gate_proj",
        "intermediate": "gate_proj",
        "output_dense": "ffn",
        # BERT layernorm spellings:
        "layernorm": "rmsnorm",
        "layer_norm": "rmsnorm",
        "attention_output_layernorm": "rmsnorm",
        "output_layernorm": "ffn_norm",
        "embedding_layernorm": "rmsnorm",
        # BERT attention output projection:
        "attention_output_dense": "out_proj",
        "ffn_intermediate": "gate_proj",
        "ffn_output": "ffn",
        # Pooler (sentence-transformers / BERT CLS):
        "pooler_dense": "pooler",
        "pooler": "pooler",
        # T5 / Flan-T5 spellings:
        "densereluredense_wi": "gate_proj",
        "densereluredense_wo": "ffn",
        "denseactdense_wi_0": "gate_proj",
        "denseactdense_wi_1": "gate_proj",
        "denseactdense_wo": "ffn",
        "selfattention_q": "qkv",
        "selfattention_k": "qkv",
        "selfattention_v": "qkv",
        "selfattention_o": "out_proj",
        "encderattenion_q": "qkv",
        "encderattenion_k": "qkv",
        "encderattenion_v": "qkv",
        "encderattenion_o": "out_proj",
        "layer_0": "rmsnorm",
        # OPT spellings:
        "q_proj_weight": "qkv",
        "k_proj_weight": "qkv",
        "v_proj_weight": "qkv",
        "fc1": "gate_proj",
        "fc2": "ffn",
        "self_attn_layer_norm": "rmsnorm",
        "final_layer_norm": "final_norm",
        # Falcon / RWKV spellings:
        "query_key_value": "qkv",
        "dense": "out_proj",
        "dense_h_to_4h": "gate_proj",
        "dense_4h_to_h": "ffn",
        "ln_attn": "rmsnorm",
        "ln_mlp": "ffn_norm",
        # GPT-Neo / GPT-J:
        "q_proj": "qkv",
        "k_proj": "qkv",
        "v_proj": "qkv",
        "out_proj": "out_proj",
        "c_attn": "qkv",
        "c_proj": "out_proj",
        "c_fc": "gate_proj",
        "mlp_c_proj": "ffn",
        # Phi-2 / Phi-3:
        "Wqkv": "qkv",
        "wqkv": "qkv",
        "out_proj": "out_proj",
        "fc1": "gate_proj",
        "fc2": "ffn",
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

        Dispatches to the encoder-specific builder for BERT-family models
        (is_encoder=True) and uses the decoder-only builder for causal LMs.
        """
        if getattr(architecture, "is_encoder", False):
            return self._build_encoder_graph(graph, architecture)
        return self._build_decoder_graph(graph, architecture)

    def _build_encoder_graph(self, graph: AEGGraph, architecture: ModelArchitecture) -> AEGGraph:
        """Build an encoder-only (BERT-style) AEG graph.

        Encoder graphs differ from decoder graphs in several key ways:
        - Bidirectional attention (no causal mask, no RoPE by default)
        - LayerNorm instead of RMSNorm
        - GELU activation in FFN (not SwiGLU)
        - Positional embeddings + token-type embeddings
        - Pooler (CLS-token extraction) instead of LM head
        - No KV cache
        """
        from aether.core.graph import AEGGraphEdge, AEGGraphNode, AEGGraphNodeType
        from aether.core.types import DType, TensorLayout, TensorShape

        h = architecture.hidden_size
        i = architecture.intermediate_size or (h * 4)
        v = architecture.vocab_size
        n_layers = architecture.layers
        n_heads = architecture.num_attention_heads
        head_dim = architecture.head_dim or (h // n_heads)
        max_pos = architecture.context_length or 512
        batch_dim = None  # dynamic

        # ── Inputs ──
        token_ids_node = AEGGraphNode(
            id="input_ids",
            node_type=AEGGraphNodeType.INPUT,
            name="input_ids",
            op_type="input",
            layout=TensorLayout(
                shape=TensorShape.from_list([batch_dim]),
                dtype=DType.INT64,
            ),
        )
        graph.add_node(token_ids_node)

        # ── Token embeddings ──
        tok_emb_node = AEGGraphNode(
            id="token_embeddings",
            node_type=AEGGraphNodeType.OPERATION,
            name="token_embeddings",
            op_type="embedding",
            inputs=[token_ids_node.id],
            attributes={"vocab_size": v, "hidden_size": h},
            layer_index=0,
        )
        graph.add_node(tok_emb_node)
        graph.add_edge(AEGGraphEdge(source=token_ids_node.id, target=tok_emb_node.id))

        # ── Positional embeddings ──
        pos_emb_node = AEGGraphNode(
            id="position_embeddings",
            node_type=AEGGraphNodeType.OPERATION,
            name="position_embeddings",
            op_type="embedding",
            inputs=[token_ids_node.id],
            attributes={"vocab_size": max_pos, "hidden_size": h, "embedding_type": "position"},
            layer_index=0,
        )
        graph.add_node(pos_emb_node)
        graph.add_edge(AEGGraphEdge(source=token_ids_node.id, target=pos_emb_node.id))

        # ── Token-type embeddings ──
        seg_emb_node = AEGGraphNode(
            id="token_type_embeddings",
            node_type=AEGGraphNodeType.OPERATION,
            name="token_type_embeddings",
            op_type="embedding",
            inputs=[token_ids_node.id],
            attributes={"vocab_size": 2, "hidden_size": h, "embedding_type": "token_type"},
            layer_index=0,
        )
        graph.add_node(seg_emb_node)
        graph.add_edge(AEGGraphEdge(source=token_ids_node.id, target=seg_emb_node.id))

        # ── Embedding sum + LayerNorm ──
        emb_sum_node = AEGGraphNode(
            id="embedding_sum",
            node_type=AEGGraphNodeType.OPERATION,
            name="embedding_sum",
            op_type="add",
            inputs=[tok_emb_node.id, pos_emb_node.id, seg_emb_node.id],
            attributes={},
            layer_index=0,
        )
        graph.add_node(emb_sum_node)
        graph.add_edge(AEGGraphEdge(source=tok_emb_node.id, target=emb_sum_node.id))
        graph.add_edge(AEGGraphEdge(source=pos_emb_node.id, target=emb_sum_node.id))
        graph.add_edge(AEGGraphEdge(source=seg_emb_node.id, target=emb_sum_node.id))

        emb_ln_node = AEGGraphNode(
            id="embedding_layernorm",
            node_type=AEGGraphNodeType.OPERATION,
            name="embedding_layernorm",
            op_type="layernorm",
            inputs=[emb_sum_node.id],
            attributes={"eps": architecture.norm_eps or 1e-12, "hidden_size": h},
            layer_index=0,
        )
        graph.add_node(emb_ln_node)
        graph.add_edge(AEGGraphEdge(source=emb_sum_node.id, target=emb_ln_node.id))

        prev_node = emb_ln_node

        # ── Encoder layers ──
        for layer in range(n_layers):
            lp = f"layer_{layer}"

            # Self-attention: QKV projection
            qkv_node = AEGGraphNode(
                id=f"{lp}_qkv",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} QKV Projection",
                op_type="qkv_proj",
                inputs=[prev_node.id],
                attributes={"num_heads": n_heads, "num_kv_heads": n_heads, "head_dim": head_dim,
                            "bidirectional": True},
                layer_index=layer,
            )
            graph.add_node(qkv_node)
            graph.add_edge(AEGGraphEdge(source=prev_node.id, target=qkv_node.id))

            # Bidirectional full attention (no causal mask)
            attn_node = AEGGraphNode(
                id=f"{lp}_attention",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Bidirectional Attention",
                op_type="mha",
                inputs=[qkv_node.id],
                attributes={"num_heads": n_heads, "head_dim": head_dim,
                            "causal": False, "bidirectional": True},
                layer_index=layer,
            )
            graph.add_node(attn_node)
            graph.add_edge(AEGGraphEdge(source=qkv_node.id, target=attn_node.id))

            # Attention output projection
            attn_out_node = AEGGraphNode(
                id=f"{lp}_attn_output",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Attention Output",
                op_type="linear",
                inputs=[attn_node.id],
                attributes={"in_features": h, "out_features": h},
                layer_index=layer,
            )
            graph.add_node(attn_out_node)
            graph.add_edge(AEGGraphEdge(source=attn_node.id, target=attn_out_node.id))

            # Residual + LayerNorm (post-attention)
            attn_res_node = AEGGraphNode(
                id=f"{lp}_attn_residual",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Attention Residual",
                op_type="add",
                inputs=[prev_node.id, attn_out_node.id],
                attributes={},
                layer_index=layer,
            )
            graph.add_node(attn_res_node)
            graph.add_edge(AEGGraphEdge(source=prev_node.id, target=attn_res_node.id))
            graph.add_edge(AEGGraphEdge(source=attn_out_node.id, target=attn_res_node.id))

            attn_ln_node = AEGGraphNode(
                id=f"{lp}_attn_layernorm",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Attention LayerNorm",
                op_type="layernorm",
                inputs=[attn_res_node.id],
                attributes={"eps": architecture.norm_eps or 1e-12, "hidden_size": h},
                layer_index=layer,
            )
            graph.add_node(attn_ln_node)
            graph.add_edge(AEGGraphEdge(source=attn_res_node.id, target=attn_ln_node.id))

            # FFN: intermediate (GELU) + output
            ffn_int_node = AEGGraphNode(
                id=f"{lp}_ffn_intermediate",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} FFN Intermediate",
                op_type="linear",
                inputs=[attn_ln_node.id],
                attributes={"in_features": h, "out_features": i, "activation": "gelu"},
                layer_index=layer,
            )
            graph.add_node(ffn_int_node)
            graph.add_edge(AEGGraphEdge(source=attn_ln_node.id, target=ffn_int_node.id))

            ffn_out_node = AEGGraphNode(
                id=f"{lp}_ffn_output",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} FFN Output",
                op_type="linear",
                inputs=[ffn_int_node.id],
                attributes={"in_features": i, "out_features": h},
                layer_index=layer,
            )
            graph.add_node(ffn_out_node)
            graph.add_edge(AEGGraphEdge(source=ffn_int_node.id, target=ffn_out_node.id))

            # Residual + LayerNorm (post-FFN)
            ffn_res_node = AEGGraphNode(
                id=f"{lp}_ffn_residual",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} FFN Residual",
                op_type="add",
                inputs=[attn_ln_node.id, ffn_out_node.id],
                attributes={},
                layer_index=layer,
            )
            graph.add_node(ffn_res_node)
            graph.add_edge(AEGGraphEdge(source=attn_ln_node.id, target=ffn_res_node.id))
            graph.add_edge(AEGGraphEdge(source=ffn_out_node.id, target=ffn_res_node.id))

            ffn_ln_node = AEGGraphNode(
                id=f"{lp}_output_layernorm",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} Output LayerNorm",
                op_type="layernorm",
                inputs=[ffn_res_node.id],
                attributes={"eps": architecture.norm_eps or 1e-12, "hidden_size": h},
                layer_index=layer,
            )
            graph.add_node(ffn_ln_node)
            graph.add_edge(AEGGraphEdge(source=ffn_res_node.id, target=ffn_ln_node.id))

            prev_node = ffn_ln_node

        # ── Pooler (CLS token extraction for sentence-level tasks / SBERT) ──
        pooler_node = AEGGraphNode(
            id="pooler",
            node_type=AEGGraphNodeType.OPERATION,
            name="pooler",
            op_type="pooler",
            inputs=[prev_node.id],
            attributes={"hidden_size": h, "pool_type": "cls", "activation": "tanh"},
        )
        graph.add_node(pooler_node)
        graph.add_edge(AEGGraphEdge(source=prev_node.id, target=pooler_node.id))

        # ── Output (sentence embedding or logits depending on task head) ──
        output_node = AEGGraphNode(
            id="output",
            node_type=AEGGraphNodeType.OUTPUT,
            name="sentence_embedding",
            op_type="output",
            inputs=[pooler_node.id],
        )
        graph.add_node(output_node)
        graph.add_edge(AEGGraphEdge(source=pooler_node.id, target=output_node.id))

        # Mark as encoder
        graph.set_metadata("is_encoder", True)
        graph.set_metadata("encoder_arch", "bert_style")
        graph.set_metadata("encoder_layers", n_layers)
        graph.set_metadata("encoder_heads", n_heads)
        graph.set_metadata("encoder_hidden_size", h)

        logger.info(
            "Built BERT/encoder graph: %d nodes, %d edges, %d layers, %dD hidden",
            graph.node_count, graph.edge_count, n_layers, h,
        )
        return graph

    def _build_decoder_graph(self, graph: AEGGraph, architecture: ModelArchitecture) -> AEGGraph:
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
