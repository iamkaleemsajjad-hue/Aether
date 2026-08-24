"""
Model ingestion pipeline — loads any supported format into an AEG computation graph.

The IngestionPipeline orchestrates the format-specific loaders and produces an
AEGGraph that the optimizer passes can consume.
"""

from __future__ import annotations

import json
import os
import re
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

        def _qualified(result: Any, loader_name: str) -> AEGGraph | None:
            """Wrap a specialised result and verify it can host the checkpoint.

            Returns the graph when it is runnable, otherwise ``None`` so the
            generic dispatch path takes over.  Structure metadata from the
            specialised loader is stashed for the generic graph either way,
            guaranteeing compile-time invariants (every required tensor bound)
            never trade away the specialised loader's architectural insight.
            """
            # Specialised loaders predate the AEGGraph API and a few of them
            # still return a typed tuple (SSM) rather than the dictionary used
            # by the graph-producing loaders.  Normalize that boundary here;
            # otherwise a valid Mamba/RWKV config is silently treated as a
            # loader failure and loses its capability metadata.
            if isinstance(result, tuple) and len(result) == 2:
                result = {
                    "architecture": result[0],
                    "nodes": result[1],
                    "format": "ssm_model",
                }
            if not isinstance(result, dict):
                raise IngestionError(
                    f"{loader_name} returned an unsupported result type "
                    f"{type(result).__name__}"
                )
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
                    graph = _qualified(SSMLoader().load(path, config), "SSMLoader")
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
            # The specialised graph builders intentionally parse their own
            # descriptor, but the compiler's source of truth is the
            # architecture detected from the checkpoint config.  Preserve it
            # on the graph so invariant checks, optimization and packaging do
            # not operate on an architecture-less graph.
            if getattr(graph, "architecture", None) is None:
                graph.architecture = architecture
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

        # The architecture's normalization placement disambiguates the
        # ``*_layernorm`` spellings shared by pre-, post-, and sandwich-normed
        # blocks.  See ``_normalise_weight_name``.
        placement = str(
            getattr(getattr(graph, "architecture", None), "norm_placement", "pre") or "pre"
        )
        lookup: dict[tuple[int | None, str | None], list[tuple[str, Any]]] = {}
        unresolvable: list[str] = []
        for name, tensor in tensors.items():
            key = self._normalise_weight_name(name, placement)
            if key[1] is None:
                unresolvable.append(name)
                continue
            if hasattr(tensor, "float") and getattr(tensor, "dtype", None) is not None and str(tensor.dtype).endswith("bfloat16"):
                tensor = tensor.float()
            value = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            value = self._canonicalize_checkpoint_tensor(name, value, key[1], graph)
            lookup.setdefault(key, []).append((name, value))

        attached = 0
        bound_names: list[str] = []
        unbound_names: list[str] = []
        for node in graph:
            if not hasattr(node, "add_attribute"):
                continue
            key = self._normalise_weight_name(getattr(node, "id", ""), placement)
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
            packed_source = source_name.lower()
            packed_shape = np.asarray(value).shape
            expected_intermediate = int(
                getattr(getattr(graph, "architecture", None), "intermediate_size", 0) or 0
            )
            packed_gate_up = (
                key[1] == "gate_proj"
                and np.asarray(value).ndim == 2
                and packed_shape[0] % 2 == 0
                and (
                    "gate_up_proj" in packed_source
                    or (
                        # OLMo/OLMoE use ``ff_proj`` for the fused gate+up
                        # projection.  Match the configured intermediate
                        # width so a classic single ``ff_proj`` is not
                        # accidentally halved merely because its row count
                        # happens to be even.
                        "ff_proj" in packed_source
                        and expected_intermediate > 0
                        and packed_shape[0] == 2 * expected_intermediate
                    )
                )
            )
            if packed_gate_up:
                packed = np.asarray(value)
                split = packed.shape[0] // 2
                value = packed[:split]
                node.add_attribute("up_weight", packed[split:])
                node.add_attribute("up_weight_source", source_name)
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
                    if (
                        "up_proj" in item[0].lower()
                        or "ffn_up" in item[0].lower()
                        or re.search(r"(?:^|[._])w3(?:[._]|$)", item[0].lower())
                    )
                ]
                gate_candidates = [
                    item
                    for item in candidates
                    if (
                        "gate_proj" in item[0].lower()
                        or "ffn_gate" in item[0].lower()
                        or re.search(r"(?:^|[._])w1(?:[._]|$)", item[0].lower())
                    )
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
    def _canonicalize_checkpoint_tensor(
        name: str,
        value: Any,
        component: str | None,
        graph: "AEGGraph",
    ) -> Any:
        """Convert known framework layouts to Aether's row-major convention.

        Hugging Face's GPT-2 family stores ``Conv1D`` weights as
        ``(in_features, out_features)`` while the rest of the Aether runtime
        consumes linear weights as ``(out_features, in_features)``.  This is a
        structural property of the checkpoint tensor name, not a model-name
        special case, and is applied only to the GPT ``c_*`` layout.

        Fused QKV projections are also normalized here.  GPT-NeoX, BLOOM and
        Falcon interleave Q/K/V *within* each head or KV group rather than
        stacking them, so reading the tensor as three contiguous blocks
        permutes every attention head.  Rewriting it once at ingestion keeps
        every downstream pass and every runtime on one layout.
        """
        import numpy as np

        array = np.asarray(value)
        lowered = name.lower()
        if component == "qkv":
            architecture = getattr(graph, "architecture", None)
            layout = str(getattr(architecture, "fused_qkv_layout", "contiguous") or "contiguous")
            if layout != "contiguous" and architecture is not None:
                rewritten = IngestionPipeline._deinterleave_fused_qkv(array, architecture)
                if rewritten is not None:
                    return rewritten
        if array.ndim != 2 or not any(marker in lowered for marker in ("c_attn", "c_fc", "c_proj")):
            return value
        if component in {"qkv", "gate_proj", "out_proj", "ffn"}:
            return np.ascontiguousarray(array.T)
        return value

    @staticmethod
    def _deinterleave_fused_qkv(array: Any, architecture: Any) -> Any | None:
        """Restack an interleaved fused QKV tensor as ``[all Q | all K | all V]``.

        GPT-NeoX and BLOOM lay the projection out per head as
        ``[q, k, v][q, k, v]...``; Falcon's new decoder architecture lays it out
        per KV group as ``[q x heads_per_group, k, v]...``.  Both are the same
        operation with different group widths: view the rows as
        ``(groups, heads_per_group + 2, head_dim, ...)`` and gather the query
        rows, then the key row, then the value row.

        Returns ``None`` when the tensor does not match the declared geometry,
        so a checkpoint that already uses the contiguous layout is left alone
        rather than silently permuted.
        """
        import numpy as np

        source = np.asarray(array)
        if source.ndim not in (1, 2):
            return None
        heads = int(getattr(architecture, "num_attention_heads", 0) or 0)
        kv_heads = int(getattr(architecture, "num_kv_heads", 0) or heads)
        head_dim = int(getattr(architecture, "head_dim", 0) or 0)
        if heads <= 0 or kv_heads <= 0 or head_dim <= 0 or heads % kv_heads:
            return None
        per_group = heads // kv_heads
        expected_rows = (heads + 2 * kv_heads) * head_dim
        if int(source.shape[0]) != expected_rows:
            return None
        trailing = source.shape[1:]
        grouped = source.reshape(kv_heads, per_group + 2, head_dim, *trailing)
        query = grouped[:, :per_group].reshape(heads * head_dim, *trailing)
        key = grouped[:, per_group].reshape(kv_heads * head_dim, *trailing)
        value = grouped[:, per_group + 1].reshape(kv_heads * head_dim, *trailing)
        return np.ascontiguousarray(np.concatenate((query, key, value), axis=0))

    @staticmethod
    def _attach_projection_biases(node: Any, candidates: list[tuple[str, Any]]) -> None:
        """Attach Q/K/V bias vectors to a fused projection node.

        Biases are kept separate from the fused matrices because grouped-query
        attention may have different K/V row counts.  The quantizer can then
        persist them under their canonical per-projection names.
        """
        for name, value in candidates:
            lowered = name.lower()
            if "c_attn" in lowered or "query_key_value" in lowered or "qkv" in lowered:
                import numpy as np
                vector = np.asarray(value, dtype=np.float32).reshape(-1)
                attrs = getattr(node, "attributes", {}) or {}
                heads = int(attrs.get("num_heads", 0) or 0)
                kv_heads = int(attrs.get("num_kv_heads", heads) or heads)
                head_dim = int(attrs.get("head_dim", 0) or 0)
                expected = (heads + 2 * kv_heads) * head_dim
                if expected == vector.size and heads > 0 and kv_heads > 0 and head_dim > 0:
                    q_width = heads * head_dim
                    kv_width = kv_heads * head_dim
                    node.add_attribute("q_bias", vector[:q_width])
                    node.add_attribute("k_bias", vector[q_width : q_width + kv_width])
                    node.add_attribute("v_bias", vector[q_width + kv_width :])
                    continue
                if vector.size % 3 == 0:
                    width = vector.size // 3
                    node.add_attribute("q_bias", vector[:width])
                    node.add_attribute("k_bias", vector[width : 2 * width])
                    node.add_attribute("v_bias", vector[2 * width :])
                    continue
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
        "q_norm": "q_norm",
        "k_norm": "k_norm",
        "qkv_proj": "qkv",
        # Standard llama.cpp/GGUF spellings.
        "attn_q": "qkv",
        "attn_k": "qkv",
        "attn_v": "qkv",
        # OLMo / OLMoE checkpoints use a fused attention projection and a
        # short FFN vocabulary rather than the Llama-style names.
        "att_proj": "qkv",
        "attn_proj": "qkv",
        "attn_out": "out_proj",
        "ff_proj": "gate_proj",
        "ff_out": "ffn",
        "ff_norm": "ffn_norm",
        # GPT-2 / DialoGPT layernorm spellings.
        "ln_1": "rmsnorm",
        "ln_2": "ffn_norm",
        "ln_f": "final_norm",
        "norm_f": "final_norm",
        "ln_out": "final_norm",
        "lnf": "final_norm",
        "o_proj": "out_proj",
        "attn_output": "out_proj",
        "embed_tokens": "embedding",
        "embed_in": "embedding",
        "word_embeddings": "embedding",
        "embed_positions": "position_embedding",
        "embed_out": "lm_head",
        "head": "lm_head",
        "tok_embeddings": "embedding",
        "embeddings": "embedding",
        "token_embd": "embedding",
        "wte": "embedding",
        "wpe": "position_embedding",
        "position_embedding": "position_embedding",
        "input_layernorm": "rmsnorm",
        "ln1": "rmsnorm",
        "ln2": "ffn_norm",
        "attn_norm": "rmsnorm",
        "post_attention_layernorm": "ffn_norm",
        "ffn_norm": "ffn_norm",
        "down_proj": "ffn",
        "ffn_down": "ffn",
        "up_proj": "gate_proj",
        "gate_up_proj": "gate_proj",
        "ffn_up": "gate_proj",
        "ffn_gate": "gate_proj",
        "w1": "gate_proj",
        "w2": "ffn",
        "w3": "gate_proj",
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
        # The global encoder embedding LayerNorm has a distinct runtime slot.
        # Decoder blocks use ``rmsnorm``/``ffn_norm``; treating this name as a
        # generic RMSNorm causes the real checkpoint tensor to collide with
        # the embedding table and leaves the executable encoder artifact
        # incomplete.
        "embedding_layernorm": "embedding_norm",
        "embedding_norm": "embedding_norm",
        # BERT attention output projection:
        "attention_output_dense": "out_proj",
        "attn_output": "out_proj",
        "attn_layernorm": "attention_norm",
        "attention_norm": "attention_norm",
        "intermediate_proj": "intermediate_proj",
        "output_proj": "output_proj",
        "output_norm": "output_norm",
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
        "wqkv": "qkv",
        "wo": "out_proj",
        "qkv_proj": "qkv",
        "norm_1": "rmsnorm",
        "norm_2": "ffn_norm",
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
        "fc_in": "gate_proj",
        "fc_out": "ffn",
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
        # Decoder-only checkpoints commonly name the terminal norm simply
        # ``model.norm.weight`` (Qwen3, Llama, Mistral, ...).
        "norm": "final_norm",
        "ffn_norm": "ffn_norm",
        "ffn": "ffn",
        "gate_proj": "gate_proj",
        "lm_head": "lm_head",
        # Encoder post-FFN LayerNorm has its own runtime slot; a decoder
        # terminal ``output_norm`` is handled by the explicit global-name
        # branch below, so it must not overwrite this component alias.
        "output_norm": "output_norm",
        "final_norm": "final_norm",
    }


    #: Structural path segments that carry no identifying information.
    _IGNORED_SEGMENTS = frozenset({"model", "transformer", "module", "self_attn", "mlp", "blk", "weight"})

    @classmethod
    def _normalise_weight_name(
        cls, name: str, placement: str = "pre"
    ) -> tuple[int | None, str | None]:
        """Reduce a tensor or node name to a ``(layer_index, component)`` key.

        Checkpoints and graph nodes disagree on both separators (``.`` vs ``_``)
        and component naming (``self_attn.q_proj`` vs ``qkv``), so names are parsed
        into a structured key instead of string-matched. Returns ``(None, None)``
        when nothing identifiable can be extracted.

        ``placement`` resolves a genuine ambiguity in the ecosystem's naming.
        ``post_attention_layernorm`` is the *pre-FFN* norm in Llama-style blocks
        (the name is historical), but in OLMo-2 it really is the attention
        *output* norm, and in Gemma-2/3 it is one of four norms per block.  The
        architecture's declared normalization placement is what distinguishes
        them; nothing in the tensor name alone can.
        """
        import re

        lowered_placement = str(placement or "pre").lower()
        if lowered_placement in {"post", "sandwich"}:
            layer_scope = re.search(
                r"(?:^|_)(?:layers?|blocks?|blk|h)[._](\d+)(?:[._]|$)",
                name.lower().replace(".", "_"),
            )
            if layer_scope is not None:
                scoped = name.lower().replace(".", "_")
                index = int(layer_scope.group(1))
                if lowered_placement == "post":
                    # OLMo-2: the two stored norms are sublayer *output* norms
                    # and fill the two standard slots.
                    if re.search(r"(?:^|_)post_attention_layernorm(?:_|$)", scoped):
                        return index, "rmsnorm"
                    if re.search(r"(?:^|_)post_feedforward_layernorm(?:_|$)", scoped):
                        return index, "ffn_norm"
                else:
                    # Gemma-2/3, EXAONE-4: four norms per block.
                    if re.search(r"(?:^|_)post_attention_layernorm(?:_|$)", scoped):
                        return index, "post_attention_norm"
                    if re.search(r"(?:^|_)pre_feedforward_layernorm(?:_|$)", scoped):
                        return index, "ffn_norm"
                    if re.search(r"(?:^|_)post_feedforward_layernorm(?:_|$)", scoped):
                        return index, "post_ffn_norm"
                    if re.search(r"(?:^|_)post_attention_norm(?:_|$)", scoped):
                        return index, "post_attention_norm"
                    if re.search(r"(?:^|_)post_ffn_norm(?:_|$)", scoped):
                        return index, "post_ffn_norm"

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

        # GPT-2 uses the same ``c_proj`` spelling for attention output and
        # MLP output.  Preserve the surrounding structural path before the
        # generic alias table is consulted.
        lowered_name = name.lower().replace(".", "_")
        # Multi-head latent attention uses compressed query/KV projections
        # rather than ordinary q/k/v matrices.  Normalize the framework
        # spellings (including DeepSeek's ``kv_a_proj_with_mqa``) to a stable
        # AEG vocabulary so the artifact can be executed without knowing the
        # originating model class.
        mla_layer_match = re.search(
            r"(?:^|_)(?:layers?|blocks?|blk|h)[_](\d+)(?:_|$)", lowered_name
        )
        if mla_layer_match:
            mla_components = (
                ("q_a_proj", "q_a_proj"),
                ("q_b_proj", "q_b_proj"),
                ("kv_a_proj_with_mqa", "kv_a_proj"),
                ("kv_a_proj", "kv_a_proj"),
                ("kv_b_proj", "kv_b_proj"),
                ("k_rope_proj", "k_rope_proj"),
                ("k_pe_proj", "k_rope_proj"),
                ("q_a_layernorm", "q_a_norm"),
                ("kv_a_layernorm", "kv_a_norm"),
                ("q_a_norm", "q_a_norm"),
                ("kv_a_norm", "kv_a_norm"),
            )
            for marker, component_name in mla_components:
                if re.search(rf"(?:^|_){re.escape(marker)}(?:_|$)", lowered_name):
                    return int(mla_layer_match.group(1)), component_name
            ssm_components = (
                ("in_proj", "ssm_in_proj"),
                ("conv1d", "ssm_conv1d"),
                ("x_proj", "ssm_x_proj"),
                ("dt_proj", "ssm_dt_proj"),
                ("dt_bias", "ssm_dt"),
                ("a_log", "ssm_a_log"),
                ("d", "ssm_d"),
                ("out_proj", "ssm_out_proj"),
            )
            rwkv_components = (
                ("time_decay", "ssm_time_decay"),
                ("time_first", "ssm_time_first"),
                ("time_mix_k", "ssm_time_mix_k"),
                ("time_mix_v", "ssm_time_mix_v"),
                ("time_mix_r", "ssm_time_mix_r"),
                ("time_mix_k", "ssm_ffn_time_mix_k"),
                ("time_mix_r", "ssm_ffn_time_mix_r"),
                ("key", "ssm_key"),
                ("value", "ssm_value"),
                ("receptance", "ssm_receptance"),
                ("output", "ssm_output"),
            )
            is_rwkv_tensor = bool(re.search(r"(?:^|_)(?:att|ffn|ln[12])(?:_|$)", lowered_name))
            if is_rwkv_tensor and re.search(r"(?:^|_)ln1(?:_|$)", lowered_name):
                return int(mla_layer_match.group(1)), "ssm_norm"
            if is_rwkv_tensor and re.search(r"(?:^|_)ln2(?:_|$)", lowered_name):
                return int(mla_layer_match.group(1)), "ssm_ffn_norm"
            for marker, component_name in (("time_mix_k", "ssm_ffn_time_mix_k"), ("time_mix_r", "ssm_ffn_time_mix_r")):
                if is_rwkv_tensor and re.search(rf"(?:^|_)ffn_{re.escape(marker)}(?:_|$)", lowered_name):
                    return int(mla_layer_match.group(1)), component_name
            for marker, component_name in rwkv_components:
                if is_rwkv_tensor and re.search(rf"(?:^|_)(?:att|attention)_{re.escape(marker)}(?:_|$)", lowered_name):
                    return int(mla_layer_match.group(1)), component_name
            for marker, component_name in (("key", "ssm_ffn_key"), ("value", "ssm_ffn_value"), ("receptance", "ssm_ffn_receptance")):
                if is_rwkv_tensor and re.search(rf"(?:^|_)ffn_{re.escape(marker)}(?:_|$)", lowered_name):
                    return int(mla_layer_match.group(1)), component_name
            for marker, component_name in ssm_components:
                if re.search(rf"(?:^|_)(?:mixer|mamba)_(?:{re.escape(marker)})(?:_|$)", lowered_name) or (
                    marker != "out_proj" and re.search(rf"(?:^|_){re.escape(marker)}(?:_|$)", lowered_name)
                ):
                    return int(mla_layer_match.group(1)), component_name
        # Generic routed-MoE checkpoint spellings.  Mixtral uses w1/w2/w3,
        # while Qwen/OLMoE-style checkpoints expose gate/up/down_proj.  Both
        # describe the same three affine projections and are normalized to a
        # model-independent expert key.
        layer_match = re.search(r"(?:^|_)(?:layers?|blocks?|blk|h)[_](\d+)(?:_|$)", lowered_name)
        # Canonical graph IDs must win over source-layout context rules below.
        # In particular, ``attention_norm`` is a valid encoder runtime slot;
        # it must not be rewritten to the InternLM ``rmsnorm`` alias.
        canonical_layer_component = re.match(
            r"^layer_(\d+)_(attention_norm|ffn_norm|output_norm)$", lowered_name
        )
        if canonical_layer_component:
            return int(canonical_layer_component.group(1)), canonical_layer_component.group(2)
        # OPT stores the post-attention norm as ``final_layer_norm`` inside
        # each decoder block, while the same spelling at model scope means
        # the terminal norm.  Resolve the scoped form before the alias scan.
        if layer_match and re.search(r"(?:^|_)final_layer_norm(?:_|$)", lowered_name):
            return int(layer_match.group(1)), "ffn_norm"
        if layer_match and re.search(r"(?:^|_)(?:attention_norm|attn_norm)(?:_|$)", lowered_name):
            return int(layer_match.group(1)), "rmsnorm"
        expert_match = re.search(r"(?:^|_)experts[_](\d+)(?:_|$)", lowered_name)
        if layer_match and expert_match:
            expert_projection = None
            if re.search(r"(?:^|_)(?:w1|gate_proj|ffn_gate)(?:_|$)", lowered_name):
                expert_projection = "gate_proj"
            elif re.search(r"(?:^|_)(?:w2|down_proj|ffn_down)(?:_|$)", lowered_name):
                expert_projection = "down_proj"
            elif re.search(r"(?:^|_)(?:w3|up_proj|ffn_up)(?:_|$)", lowered_name):
                expert_projection = "up_proj"
            if expert_projection is not None:
                return (
                    int(layer_match.group(1)),
                    f"expert_{int(expert_match.group(1))}_{expert_projection}",
                )
        if layer_match and (
            "router" in lowered_name
            or re.search(r"(?:^|_)(?:router|router_gate)[_]weight$", lowered_name)
            or ("moe" in lowered_name and re.search(r"(?:^|_)gate[_]weight$", lowered_name))
            # DeepSeek names the routed gate ``mlp.gate.weight``; it is
            # distinct from an expert ``gate_proj`` and must be retained as
            # the router rather than reported as an unbound source tensor.
            or re.search(r"(?:^|_)(?:mlp|feed_forward)_gate_weight$", lowered_name)
        ) and not expert_match:
            return int(layer_match.group(1)), "moe_router"
        canonical_moe = re.match(
            r"^layer_(\d+)_(?:moe_router|expert_(\d+)_(gate_proj|up_proj|down_proj))$",
            lowered_name,
        )
        if canonical_moe:
            layer_index = int(canonical_moe.group(1))
            if lowered_name.endswith("moe_router"):
                return layer_index, "moe_router"
            return layer_index, (
                f"expert_{int(canonical_moe.group(2))}_{canonical_moe.group(3)}"
            )
        canonical_ssm = re.match(
            r"^layer_(\d+)_((?:ssm_norm|ssm_in_proj|ssm_conv1d|ssm_x_proj|"
            r"ssm_dt_proj|ssm_dt|ssm_a_log|ssm_d|ssm_out_proj|ssm_ffn_norm|"
            r"ssm_time_decay|ssm_time_first|ssm_time_mix_k|ssm_time_mix_v|"
            r"ssm_time_mix_r|ssm_ffn_time_mix_k|ssm_ffn_time_mix_r|ssm_key|"
            r"ssm_value|ssm_receptance|ssm_output|ssm_ffn_key|ssm_ffn_value|"
            r"ssm_ffn_receptance))(?:_bias)?$",
            lowered_name,
        )
        if canonical_ssm:
            return int(canonical_ssm.group(1)), canonical_ssm.group(2)
        if layer_match and (
            re.search(r"(?:^|_)backbone_layers_\d+_norm(?:_|$)", lowered_name)
            or re.search(r"(?:^|_)layers_\d+_(?:mixer|mamba)_norm(?:_|$)", lowered_name)
        ):
            return int(layer_match.group(1)), "ssm_norm"
        # BERT-family checkpoints expose semantic encoder paths rather than
        # the canonical AEG node names.  Normalize them before the generic
        # token scan so ``encoder.layer.0`` is never confused with the
        # encoder namespace used by T5 (which intentionally uses negatives).
        bert_match = re.match(
            r"^(?:.*_)?encoder_layer_(\d+)_"
            r"(attention_self_(?:query|key|value)|attention_output_(?:dense|layer_?norm)|"
            r"intermediate_dense|output_(?:dense|layer_?norm))_(?:weight|bias)$",
            lowered_name,
        )
        if bert_match:
            index = int(bert_match.group(1))
            suffix = bert_match.group(2)
            component = {
                "attention_self_query": "qkv",
                "attention_self_key": "qkv",
                "attention_self_value": "qkv",
                "attention_output_dense": "out_proj",
                "attention_output_layernorm": "attention_norm",
                "attention_output_layer_norm": "attention_norm",
                "intermediate_dense": "intermediate_proj",
                "output_dense": "output_proj",
                "output_layernorm": "output_norm",
                "output_layer_norm": "output_norm",
            }[suffix]
            return index, component
        # T5-family checkpoints use separate encoder/decoder blocks and may
        # contain two attention modules in each decoder block. Encode the
        # namespace in the component key rather than conflating them with a
        # decoder-only layer. Encoder indices use -(index+1), while decoder
        # indices remain non-negative; this keeps the existing tuple contract
        # stable for all decoder-only callers.
        seq_match = re.search(r"(?:^|_)((?:encoder|decoder))_block_(\d+)_layer_(\d+)_(.*)", lowered_name)
        if seq_match:
            namespace, block_index_text, sublayer_text, suffix = seq_match.groups()
            block_index = int(block_index_text)
            layer_index = -(block_index + 1) if namespace == "encoder" else block_index
            if "relative_attention_bias" in suffix:
                component = (
                    "encoder_relative_attention_bias"
                    if namespace == "encoder"
                    else "decoder_self_relative_attention_bias"
                )
            elif namespace == "encoder" and sublayer_text == "0":
                component = (
                    "encoder_norm1" if "layer_norm" in suffix else
                    next((f"{part}_proj" for part in ("q", "k", "v", "o") if re.search(rf"(?:^|_){part}(?:_|$)", suffix)), None)
                )
            elif namespace == "encoder" and sublayer_text == "1":
                component = "encoder_norm2" if "layer_norm" in suffix else (
                    "encoder_ffn_in_0" if "wi_0" in suffix else
                    "encoder_ffn_in_1" if "wi_1" in suffix else
                    "encoder_ffn_in" if "wi" in suffix or "fc1" in suffix else "encoder_ffn_out"
                )
            elif namespace == "decoder" and sublayer_text == "0":
                component = (
                    "decoder_self_norm" if "layer_norm" in suffix else
                    next((f"self_{part}_proj" for part in ("q", "k", "v", "o") if re.search(rf"(?:^|_){part}(?:_|$)", suffix)), None)
                )
            elif namespace == "decoder" and sublayer_text == "1":
                component = (
                    "decoder_cross_norm" if "layer_norm" in suffix else
                    next((f"cross_{part}_proj" for part in ("q", "k", "v", "o") if re.search(rf"(?:^|_){part}(?:_|$)", suffix)), None)
                )
            elif namespace == "decoder" and sublayer_text == "2":
                component = "decoder_ffn_norm" if "layer_norm" in suffix else (
                    "decoder_ffn_in_0" if "wi_0" in suffix else
                    "decoder_ffn_in_1" if "wi_1" in suffix else
                    "decoder_ffn_in" if "wi" in suffix or "fc1" in suffix else "decoder_ffn_out"
                )
            if component is not None:
                return layer_index, component
        if "encoder_final_layer_norm" in lowered_name or lowered_name.endswith("encoder_final_layer_norm_weight"):
            return None, "encoder_final_norm"
        if "decoder_final_layer_norm" in lowered_name or lowered_name.endswith("decoder_final_layer_norm_weight"):
            return None, "final_norm"
        if lowered_name in {"shared_weight", "encoder_embed_tokens_weight", "decoder_embed_tokens_weight"}:
            return None, "embedding"
        if lowered_name in {"lm_head_weight", "output_weight"}:
            return None, "lm_head"
        # Canonical seq2seq graph node IDs use the same explicit namespaces.
        canonical = re.match(r"^(encoder|decoder)_layer_(\d+)_(.+)$", lowered_name)
        if canonical:
            namespace, index_text, suffix = canonical.groups()
            index = int(index_text)
            if namespace == "encoder":
                index = -(index + 1)
                suffix = {
                    "norm1": "encoder_norm1",
                    "norm2": "encoder_norm2",
                    "ffn_in": "encoder_ffn_in",
                    "ffn_in_0": "encoder_ffn_in_0",
                    "ffn_in_1": "encoder_ffn_in_1",
                    "ffn_out": "encoder_ffn_out",
                    "relative_attention_bias": "encoder_relative_attention_bias",
                }.get(suffix, suffix)
            else:
                suffix = {
                    "self_norm": "decoder_self_norm",
                    "cross_norm": "decoder_cross_norm",
                    "ffn_norm": "decoder_ffn_norm",
                    "ffn_in": "decoder_ffn_in",
                    "ffn_in_0": "decoder_ffn_in_0",
                    "ffn_in_1": "decoder_ffn_in_1",
                    "ffn_out": "decoder_ffn_out",
                    "self_relative_attention_bias": "decoder_self_relative_attention_bias",
                }.get(suffix, suffix)
            return index, suffix
        if lowered_name in {"encoder_final_norm", "final_norm", "embedding", "lm_head"}:
            return None, lowered_name
        # Global names must be resolved before the token alias scan.  For
        # example ``embeddings.position_embeddings.weight`` contains the
        # generic ``embeddings`` token, but it is a position table, not the
        # token embedding table.  The same rule makes GGUF's
        # ``output_norm.weight`` map to the decoder terminal norm.
        if re.search(r"(?:^|_)position_embeddings?_(?:weight|bias)$", lowered_name):
            return None, "position_embedding"
        if re.search(r"(?:^|_)token_type_embeddings?_(?:weight|bias)$", lowered_name):
            return None, "token_type_embedding"
        if re.search(r"(?:^|_)(?:word_embeddings?|token_embeddings?)_(?:weight|bias)$", lowered_name):
            return None, "embedding"
        if re.search(r"(?:^|_)emb_weight$", lowered_name):
            return None, "embedding"
        if re.search(r"(?:^|_)embeddings?_(?:layernorm|layer_norm)_(?:weight|bias)$", lowered_name):
            return None, "embedding_norm"
        if lowered_name in {"output_norm_weight", "output_norm_bias"}:
            return None, "final_norm"
        if "mlp_c_proj" in lowered_name or "mlp_c_proj_weight" in lowered_name:
            component = "ffn"
        elif "attn_c_proj" in lowered_name:
            component = "out_proj"

        # Resolve Qwen3's norms before aliases such as ``attn_q``.  In a
        # checkpoint path like ``self_attn.q_norm`` the earlier ``attn_q``
        # prefix is also a valid alias, but it is not the tensor we need.
        qk_norm_match = re.search(r"(?:^|[._])([qk]_norm)(?:[._]|$)", name.lower())
        if qk_norm_match:
            layer_match = re.search(r"(?:^|[._])(layers?|blocks?|blk|h)[._](\d+)(?:[._]|$)", name.lower())
            if layer_match:
                layer_index = int(layer_match.group(2))
            return layer_index, qk_norm_match.group(1)

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
            # Qwen3 uses ``q_norm``/``k_norm``.  Check these two-token
            # components before the broader ``attn_q`` alias, otherwise the
            # shared ``self_attn.q_norm`` prefix is misclassified as QKV.
            if token in ("q", "k") and position + 1 < len(tokens) and tokens[position + 1] == "norm":
                component = f"{token}_norm"
                break
            # Rebuild multi-token component names such as "q" + "proj".
            # A structural GPT-2 disambiguation above is authoritative; do
            # not let the later generic ``c_proj`` alias overwrite it.
            if component is None:
                # Some decoder families encode the projection role as a
                # four-token path (dense_h_to_4h).  Check the longest alias
                # first so its leading ``dense`` token is not mistaken for
                # an attention output projection.
                for width in (4, 3, 2, 1):
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
        logger.info(f"Auto-ingesting model: {model}")
        source = Path(model)
        downloaded = False
        if not source.exists():
            if self.config.skip_download:
                graph = AEGGraph(name=f"{architecture.family}_auto", architecture=architecture)
                self._build_architecture_graph(graph, architecture)
                graph.set_metadata("weights_attached", 0)
                graph.set_metadata("source_model_id", model)
                logger.warning("Skipping model download for %s by compiler configuration", model)
                return graph
            source = Path(self._download_hf_snapshot(model))
            downloaded = True

        # A Hub identifier can be recognized from a name before its snapshot
        # exists.  That preliminary recognition is only a routing hint: the
        # materialized checkpoint's config is the authoritative architecture
        # contract.  Refresh the existing object in place so every later
        # compiler stage (graph, optimizer, targeting, manifest, and runtime)
        # observes the same dimensions.  This is deliberately family-neutral;
        # it fixes stale name tables without adding model-specific exceptions.
        architecture = self._refresh_architecture_from_snapshot(source, architecture)
        graph = AEGGraph(name=f"{architecture.family}_auto", architecture=architecture)
        self._build_architecture_graph(graph, architecture)
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

    @staticmethod
    def _refresh_architecture_from_snapshot(
        source: Path, architecture: ModelArchitecture
    ) -> ModelArchitecture:
        """Use a materialized checkpoint's config as the architecture source.

        Name-only model references may have been classified with a convenience
        table before the Hub snapshot was downloaded.  Such tables cannot be
        trusted for exact tensor geometry: model revisions, tokenizer sizes,
        layer schedules, and attention constants can change independently of a
        repository name.  If the snapshot has a config, parse it now and copy
        the result into the caller's object so existing graph references remain
        valid.  A config parsing error is surfaced instead of silently compiling
        against stale or guessed metadata.
        """
        if not source.is_dir() or not (source / "config.json").is_file():
            return architecture

        try:
            from aether.compiler.stage1_ingestion.architecture_detector import (
                ArchitectureDetector,
            )

            refreshed = ArchitectureDetector().detect(str(source))
        except Exception as exc:  # noqa: BLE001 - normalize at ingestion boundary
            raise IngestionError(
                f"Unable to read authoritative architecture from materialized "
                f"checkpoint {source}: {exc}"
            ) from exc

        # ModelArchitecture is intentionally mutable.  Updating in place is
        # important because the compiler received this same object before the
        # download and later stages still hold that reference.
        architecture.__dict__.clear()
        architecture.__dict__.update(refreshed.__dict__)
        return architecture

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
        if getattr(architecture, "is_encoder_decoder", False) or getattr(
            architecture, "family", ""
        ) == "encoder_decoder_family":
            return self._build_encoder_decoder_graph(graph, architecture)
        if getattr(architecture, "is_encoder", False):
            return self._build_encoder_graph(graph, architecture)
        if getattr(architecture, "ssm_variant", None) == "hybrid_selective_scan":
            return self._build_hybrid_decoder_graph(graph, architecture)
        if getattr(architecture, "ssm_variant", None) in {"selective_scan", "ssd", "rwkv_time_mix"}:
            return self._build_ssm_decoder_graph(graph, architecture)
        if str(getattr(architecture, "attention_type", "") or "").upper() == "MLA":
            return self._build_mla_decoder_graph(graph, architecture)
        return self._build_decoder_graph(graph, architecture)

    def _build_mla_decoder_graph(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> AEGGraph:
        """Build the standard transformer contract plus MLA parameter slots.

        MLA keeps the decoder residual/FFN structure identical to a causal
        transformer, but replaces Q/K/V with compressed projections.  Reuse
        the ordinary graph for residual and FFN nodes, then add explicit
        parameter slots for the compressed attention tensors.  This keeps
        ingestion model-generic and lets the artifact/runtime select MLA from
        capability metadata rather than a DeepSeek name check.
        """
        from aether.core.graph import AEGGraphNode, AEGGraphNodeType

        self._build_decoder_graph(graph, architecture)
        hidden = int(architecture.hidden_size)
        heads = int(architecture.num_attention_heads)
        kv_rank = int(architecture.mla_kv_lora_rank or 0)
        q_rank = int(architecture.mla_q_lora_rank or 0)
        nope = int(architecture.mla_qk_nope_head_dim or architecture.head_dim or 1)
        rope = int(architecture.mla_qk_rope_head_dim or max(16, nope // 2))
        value = int(architecture.mla_v_head_dim or architecture.head_dim or 1)

        slots: tuple[tuple[str, int, int, str], ...] = (
            ("q_a_proj", q_rank, hidden, "linear"),
            ("q_b_proj", heads * (nope + rope), q_rank, "linear"),
            ("kv_a_proj", kv_rank, hidden, "linear"),
            ("kv_b_proj", heads * (nope + value), kv_rank, "linear"),
            ("k_rope_proj", heads * rope, hidden, "linear"),
            ("q_a_norm", q_rank, q_rank, "rmsnorm"),
            ("kv_a_norm", kv_rank, kv_rank, "rmsnorm"),
        )
        for layer in range(int(architecture.layers)):
            for suffix, out_features, in_features, op_type in slots:
                node = AEGGraphNode(
                    id=f"layer_{layer}_{suffix}",
                    node_type=AEGGraphNodeType.PARAMETER,
                    name=suffix,
                    op_type=op_type,
                    layer_index=layer,
                    attributes={
                        "in_features": in_features,
                        "out_features": out_features,
                        "hidden_size": hidden,
                    },
                )
                graph.add_node(node)
        graph.set_metadata("mla_runtime_contract", "aether.mla.v1")
        graph.set_metadata("mla_geometry", {
            "kv_lora_rank": kv_rank,
            "q_lora_rank": q_rank,
            "qk_nope_head_dim": nope,
            "qk_rope_head_dim": rope,
            "v_head_dim": value,
            "num_heads": heads,
        })
        return graph

    def _build_ssm_decoder_graph(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> AEGGraph:
        """Build the canonical Mamba selective-scan parameter contract."""
        from aether.core.graph import AEGGraphNode, AEGGraphNodeType

        hidden = int(architecture.hidden_size)
        variant = str(getattr(architecture, "ssm_variant", "selective_scan") or "selective_scan")
        inner = int(architecture.ssm_inner_size or hidden * 2)
        state = int(architecture.ssm_state_size or 16)
        dt_rank = int(architecture.ssm_dt_rank or max(1, (hidden + 15) // 16))
        conv = int(architecture.ssm_conv_kernel or 4)
        self._build_decoder_graph(graph, architecture)
        # Remove ordinary attention/FFN parameter slots from the structural
        # contract; the state-space runtime consumes these learned tensors.
        for layer in range(int(architecture.layers)):
            for suffix in (
                "qkv", "out_proj", "rmsnorm", "ffn_norm", "gate_proj", "up_proj", "ffn",
            ):
                # Use the graph API so incident edges and its reverse-edge
                # indexes stay consistent.  Directly popping nodes leaves
                # dangling edges in the serialized AEG-IR, which is
                # especially harmful for validators and alternate backends.
                graph.remove_node(f"layer_{layer}_{suffix}")
            if variant == "ssd":
                heads = int(getattr(architecture, "ssm_num_heads", None) or 1)
                groups = int(getattr(architecture, "ssm_num_groups", None) or 1)
                slot_channels = inner + 2 * groups * state
                slots = (
                    ("ssm_norm", hidden, hidden, "rmsnorm"),
                    ("ssm_ffn_norm", hidden, hidden, "rmsnorm"),
                    ("ssm_in_proj", 2 * inner + 2 * groups * state + heads, hidden, "linear"),
                    ("ssm_conv1d", slot_channels, conv, "linear"),
                    ("ssm_a_log", heads, 1, "parameter"),
                    ("ssm_d", heads, 1, "parameter"),
                    ("ssm_dt", heads, 1, "parameter"),
                    ("ssm_out_proj", hidden, inner, "linear"),
                )
            elif variant == "rwkv_time_mix":
                slots = (
                    ("ssm_norm", hidden, hidden, "rmsnorm"),
                    ("ssm_ffn_norm", hidden, hidden, "rmsnorm"),
                    ("ssm_time_decay", hidden, 1, "parameter"),
                    ("ssm_time_first", hidden, 1, "parameter"),
                    ("ssm_time_mix_k", hidden, 1, "parameter"),
                    ("ssm_time_mix_v", hidden, 1, "parameter"),
                    ("ssm_time_mix_r", hidden, 1, "parameter"),
                    ("ssm_ffn_time_mix_k", hidden, 1, "parameter"),
                    ("ssm_ffn_time_mix_r", hidden, 1, "parameter"),
                    ("ssm_key", hidden, hidden, "linear"),
                    ("ssm_value", hidden, hidden, "linear"),
                    ("ssm_receptance", hidden, hidden, "linear"),
                    ("ssm_output", hidden, hidden, "linear"),
                    ("ssm_ffn_key", hidden * 4, hidden, "linear"),
                    ("ssm_ffn_value", hidden, hidden * 4, "linear"),
                    ("ssm_ffn_receptance", hidden, hidden, "linear"),
                )
            else:
                slots = (
                    ("ssm_norm", hidden, hidden, "rmsnorm"),
                    ("ssm_in_proj", inner * 2, hidden, "linear"),
                    ("ssm_conv1d", inner, conv, "linear"),
                    ("ssm_x_proj", dt_rank + 2 * state, inner, "linear"),
                    ("ssm_dt_proj", inner, dt_rank, "linear"),
                    ("ssm_a_log", inner, state, "parameter"),
                    ("ssm_d", inner, 1, "parameter"),
                    ("ssm_out_proj", hidden, inner, "linear"),
                )
            for suffix, out_features, in_features, op_type in slots:
                node = AEGGraphNode(
                    id=f"layer_{layer}_{suffix}",
                    node_type=AEGGraphNodeType.PARAMETER,
                    name=suffix,
                    op_type=op_type,
                    layer_index=layer,
                    attributes={
                        "in_features": in_features,
                        "out_features": out_features,
                        "hidden_size": hidden,
                        "d_state": state,
                        "dt_rank": dt_rank,
                        "conv_kernel": conv,
                    },
                )
                graph.add_node(node)
        graph.set_metadata(
            "ssm_runtime_contract",
            {
                "selective_scan": "aether.mamba.selective_scan.v1",
                "ssd": "aether.mamba.ssd.v1",
                "rwkv_time_mix": "aether.rwkv.time_mix.v1",
            }.get(variant, "aether.ssm.v1"),
        )
        return graph

    def _build_hybrid_decoder_graph(
        self, graph: AEGGraph, architecture: ModelArchitecture
    ) -> AEGGraph:
        """Build a mixed transformer/Mamba graph from an explicit schedule."""
        hybrid_types = getattr(architecture, "hybrid_layer_types", None)
        if not isinstance(hybrid_types, list) or len(hybrid_types) != int(architecture.layers):
            raise IngestionError(
                "hybrid_selective_scan requires one explicit hybrid_layer_types entry per layer"
            )
        self._build_decoder_graph(graph, architecture)
        # Keep ordinary transformer slots for attention layers and replace
        # only state-space layers with the canonical selective-scan slots.
        from aether.core.graph import AEGGraphNode, AEGGraphNodeType
        hidden = int(architecture.hidden_size)
        inner = int(architecture.ssm_inner_size or hidden * 2)
        state = int(architecture.ssm_state_size or 16)
        dt_rank = int(architecture.ssm_dt_rank or max(1, (hidden + 15) // 16))
        conv = int(architecture.ssm_conv_kernel or 4)
        for layer, layer_type in enumerate(hybrid_types):
            if str(layer_type).lower() != "ssm":
                continue
            for suffix in ("qkv", "out_proj", "rmsnorm", "ffn_norm", "gate_proj", "up_proj", "ffn"):
                graph.remove_node(f"layer_{layer}_{suffix}")
            slots = (
                ("ssm_norm", hidden, hidden, "rmsnorm"),
                ("ssm_in_proj", inner * 2, hidden, "linear"),
                ("ssm_conv1d", inner, conv, "linear"),
                ("ssm_x_proj", dt_rank + 2 * state, inner, "linear"),
                ("ssm_dt_proj", inner, dt_rank, "linear"),
                ("ssm_a_log", inner, state, "parameter"),
                ("ssm_d", inner, 1, "parameter"),
                ("ssm_out_proj", hidden, inner, "linear"),
            )
            for suffix, out_features, in_features, op_type in slots:
                graph.add_node(AEGGraphNode(
                    id=f"layer_{layer}_{suffix}",
                    node_type=AEGGraphNodeType.PARAMETER,
                    name=suffix,
                    op_type=op_type,
                    layer_index=layer,
                    attributes={
                        "in_features": in_features, "out_features": out_features,
                        "hidden_size": hidden, "d_state": state,
                        "dt_rank": dt_rank, "conv_kernel": conv,
                    },
                ))
        graph.set_metadata("ssm_runtime_contract", "aether.hybrid.mamba_attention.v1")
        graph.set_metadata("hybrid_layer_types", list(hybrid_types))
        return graph

    def _build_encoder_decoder_graph(self, graph: AEGGraph, architecture: ModelArchitecture) -> AEGGraph:
        """Build the canonical T5-style encoder/decoder graph contract.

        The graph deliberately uses family-neutral encoder/decoder attention
        and FFN nodes.  T5, FLAN-T5, mT5, ByT5, and UL2 all expose this
        encoder-decoder contract even though their tokenizer and activation
        details differ.  The actual tensor names are normalized at ingestion;
        dimensions always come from the source config.
        """
        from aether.core.graph import AEGGraphEdge, AEGGraphNode, AEGGraphNodeType

        enc_layers = int(getattr(architecture, "encoder_layers", None) or architecture.layers)
        dec_layers = int(getattr(architecture, "decoder_layers", None) or architecture.layers)
        hidden = int(architecture.hidden_size)
        intermediate = int(architecture.intermediate_size or hidden * 4)
        heads = int(architecture.num_attention_heads)
        head_dim = int(architecture.head_dim or hidden // max(heads, 1))

        def add(node_id: str, op_type: str, layer_index: int | None = None, **attributes: Any) -> None:
            node = AEGGraphNode(
                id=node_id,
                node_type=AEGGraphNodeType.OPERATION,
                name=node_id,
                op_type=op_type,
                attributes=attributes,
                layer_index=layer_index,
            )
            graph.add_node(node)

        # The graph is a structural IR for the package and runtime. Inputs
        # are represented explicitly so graph validation can distinguish the
        # encoder source sequence from the decoder autoregressive sequence.
        graph.add_node(AEGGraphNode(
            id="encoder_input", node_type=AEGGraphNodeType.INPUT,
            name="encoder_input_ids", op_type="input",
        ))
        graph.add_node(AEGGraphNode(
            id="decoder_input", node_type=AEGGraphNodeType.INPUT,
            name="decoder_input_ids", op_type="input",
        ))
        add("embedding", "embedding", vocab_size=architecture.vocab_size, hidden_size=hidden)
        add("encoder_final_norm", "rmsnorm", hidden_size=hidden, eps=architecture.norm_eps)
        add("final_norm", "rmsnorm", hidden_size=hidden, eps=architecture.norm_eps)
        add("lm_head", "lm_head", vocab_size=architecture.vocab_size, hidden_size=hidden)

        for i in range(enc_layers):
            # Negative layer indices namespace encoder blocks without
            # changing the existing decoder-only layer-index contract.  The
            # compiler invariant and the weight normalizer use this same
            # representation for encoder-decoder artifacts.
            encoder_index = -(i + 1)
            add(f"encoder_layer_{i}_norm1", "rmsnorm", encoder_index, hidden_size=hidden, eps=architecture.norm_eps)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add(f"encoder_layer_{i}_{projection}", "linear", encoder_index, in_features=hidden, out_features=heads * head_dim)
            add(f"encoder_layer_{i}_relative_attention_bias", "relative_attention_bias", encoder_index, num_heads=heads)
            add(f"encoder_layer_{i}_norm2", "rmsnorm", encoder_index, hidden_size=hidden, eps=architecture.norm_eps)
            add(f"encoder_layer_{i}_ffn_in", "linear", encoder_index, in_features=hidden, out_features=intermediate)
            add(f"encoder_layer_{i}_ffn_in_0", "linear", encoder_index, in_features=hidden, out_features=intermediate)
            add(f"encoder_layer_{i}_ffn_in_1", "linear", encoder_index, in_features=hidden, out_features=intermediate)
            add(f"encoder_layer_{i}_ffn_out", "linear", encoder_index, in_features=intermediate, out_features=hidden)

        for i in range(dec_layers):
            add(f"decoder_layer_{i}_self_norm", "rmsnorm", i, hidden_size=hidden, eps=architecture.norm_eps)
            for projection in ("self_q_proj", "self_k_proj", "self_v_proj", "self_o_proj"):
                add(f"decoder_layer_{i}_{projection}", "linear", i, in_features=hidden, out_features=heads * head_dim)
            add(f"decoder_layer_{i}_self_relative_attention_bias", "relative_attention_bias", i, num_heads=heads)
            add(f"decoder_layer_{i}_cross_norm", "rmsnorm", i, hidden_size=hidden, eps=architecture.norm_eps)
            for projection in ("cross_q_proj", "cross_k_proj", "cross_v_proj", "cross_o_proj"):
                add(f"decoder_layer_{i}_{projection}", "linear", i, in_features=hidden, out_features=heads * head_dim)
            add(f"decoder_layer_{i}_ffn_norm", "rmsnorm", i, hidden_size=hidden, eps=architecture.norm_eps)
            add(f"decoder_layer_{i}_ffn_in", "linear", i, in_features=hidden, out_features=intermediate)
            add(f"decoder_layer_{i}_ffn_in_0", "linear", i, in_features=hidden, out_features=intermediate)
            add(f"decoder_layer_{i}_ffn_in_1", "linear", i, in_features=hidden, out_features=intermediate)
            add(f"decoder_layer_{i}_ffn_out", "linear", i, in_features=intermediate, out_features=hidden)

        graph.set_metadata("is_encoder_decoder", True)
        graph.set_metadata("encoder_layers", enc_layers)
        graph.set_metadata("decoder_layers", dec_layers)
        graph.set_metadata("seq2seq_family", architecture.family)
        logger.info(
            "Built encoder-decoder graph: encoder_layers=%d decoder_layers=%d hidden=%d",
            enc_layers, dec_layers, hidden,
        )
        return graph

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
                id=f"{lp}_o_proj",
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
                id=f"{lp}_attention_norm",
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
                id=f"{lp}_intermediate_proj",
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
                id=f"{lp}_output_proj",
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
                id=f"{lp}_output_norm",
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
        if bool(getattr(architecture, "embedding_norm", False)):
            embedding_norm_node = AEGGraphNode(
                id="embedding_norm",
                node_type=AEGGraphNodeType.OPERATION,
                name="embedding_output_norm",
                op_type="layernorm" if str(getattr(architecture, "norm_type", "RMSNorm")).lower() == "layernorm" else "rmsnorm",
                inputs=[prev_node.id],
                attributes={"eps": architecture.norm_eps, "hidden_size": h},
                precision=None,
                layer_index=0,
            )
            graph.add_node(embedding_norm_node)
            graph.add_edge(AEGGraphEdge(source=prev_node.id, target=embedding_norm_node.id))
            prev_node = embedding_norm_node
        position_type = str(getattr(architecture, "position_type", "RoPE") or "RoPE").lower()
        if position_type in {"absolute", "learned", "learned_absolute"}:
            position_node = AEGGraphNode(
                id="position_embedding",
                node_type=AEGGraphNodeType.OPERATION,
                name="learned_position_embedding",
                op_type="embedding",
                inputs=[input_node.id],
                attributes={
                    "vocab_size": architecture.context_length,
                    "hidden_size": h,
                    "embedding_type": "position",
                },
                precision=None,
                layer_index=0,
            )
            graph.add_node(position_node)
            graph.add_edge(AEGGraphEdge(source=input_node.id, target=position_node.id))
            embedding_sum = AEGGraphNode(
                id="embedding_with_position",
                node_type=AEGGraphNodeType.OPERATION,
                name="token_plus_position_embedding",
                op_type="add",
                inputs=[embedding_node.id, position_node.id],
                attributes={},
                precision=None,
                layer_index=0,
            )
            graph.add_node(embedding_sum)
            graph.add_edge(AEGGraphEdge(source=embedding_node.id, target=embedding_sum.id))
            graph.add_edge(AEGGraphEdge(source=position_node.id, target=embedding_sum.id))
            prev_node = embedding_sum

        # ── Transformer layers ──
        for layer in range(n_layers):
            layer_prefix = f"layer_{layer}"
            moe_layers = getattr(architecture, "moe_layer_indices", None)
            is_moe_layer = bool(
                architecture.is_moe
                and (moe_layers is None or layer in moe_layers)
            )

            # RMSNorm
            norm_node = AEGGraphNode(
                id=f"{layer_prefix}_rmsnorm",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} RMSNorm",
                op_type="layernorm" if str(getattr(architecture, "norm_type", "RMSNorm")).lower() == "layernorm" else "rmsnorm",
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
                attributes={
                    "num_heads": n_heads,
                    "num_kv_heads": n_kv_heads,
                    "head_dim": head_dim,
                    "qk_norm": architecture.qk_norm,
                },
                precision=None,
                layer_index=layer,
            )
            graph.add_node(qkv_node)
            graph.add_edge(AEGGraphEdge(source=norm_node.id, target=qkv_node.id))

            # Qwen3 normalizes each query/key head after projection and before
            # RoPE.  Keep these as real parameter-bearing nodes so the source
            # weights survive graph binding and AEG quantization.  Models that
            # do not declare qk_norm retain the ordinary QKV -> RoPE path.
            rope_input_ids = [qkv_node.id]
            if architecture.qk_norm:
                for component in ("q_norm", "k_norm"):
                    qk_norm_node = AEGGraphNode(
                        id=f"{layer_prefix}_{component}",
                        node_type=AEGGraphNodeType.OPERATION,
                        name=f"Layer {layer} {component.upper()}",
                        # Keep a distinct graph op type so Pass 1 does not
                        # fuse this parameter node into the pre-attention
                        # RMSNorm.  AEG-IR lowers it to RMS_NORM below.
                        op_type="qk_norm",
                        inputs=[qkv_node.id],
                        attributes={"eps": architecture.norm_eps, "head_dim": head_dim},
                        precision=None,
                        layer_index=layer,
                    )
                    graph.add_node(qk_norm_node)
                    graph.add_edge(AEGGraphEdge(source=qkv_node.id, target=qk_norm_node.id))
                    rope_input_ids.append(qk_norm_node.id)

            # Rotary position encoding is only one attention position scheme.
            # Learned/absolute-position decoders (GPT-2/OPT-style) must not
            # acquire a synthetic RoPE operation during lowering.
            uses_rope = str(getattr(architecture, "position_type", "RoPE") or "RoPE").lower() in {
                "rope", "rotary", "rotary_embedding"
            }
            attention_input_id = qkv_node.id
            if uses_rope:
                rope_node = AEGGraphNode(
                    id=f"{layer_prefix}_rope",
                    node_type=AEGGraphNodeType.OPERATION,
                    name=f"Layer {layer} RoPE",
                    op_type="rope",
                    inputs=rope_input_ids,
                    attributes={"theta": architecture.rope_theta, "head_dim": head_dim},
                    precision=None,
                    layer_index=layer,
                )
                graph.add_node(rope_node)
                if architecture.qk_norm:
                    graph.add_edge(AEGGraphEdge(source=f"{layer_prefix}_q_norm", target=rope_node.id))
                    graph.add_edge(AEGGraphEdge(source=f"{layer_prefix}_k_norm", target=rope_node.id))
                else:
                    graph.add_edge(AEGGraphEdge(source=qkv_node.id, target=rope_node.id))
                attention_input_id = rope_node.id

            # Attention
            attn_node = AEGGraphNode(
                id=f"{layer_prefix}_attention",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"Layer {layer} GQA Attention",
                op_type="gqa",
                inputs=[attention_input_id],
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
            graph.add_edge(AEGGraphEdge(source=attention_input_id, target=attn_node.id))

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

            # Residual add. Parallel-residual architectures retain this node
            # for graph readability, but the final block add below combines
            # both branches directly with the original residual.
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
                op_type="layernorm" if str(getattr(architecture, "norm_type", "RMSNorm")).lower() == "layernorm" else "rmsnorm",
                inputs=[norm_node.id] if bool(getattr(architecture, "parallel_residual", False)) else [residual_add_node.id],
                attributes={"eps": architecture.norm_eps, "hidden_size": h},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(ffn_norm_node)
            graph.add_edge(AEGGraphEdge(
                source=(norm_node.id if bool(getattr(architecture, "parallel_residual", False)) else residual_add_node.id),
                target=ffn_norm_node.id,
            ))

            # Sandwich-normalized blocks (Gemma-2/3, EXAONE-4) carry two extra
            # learned norms per layer, applied to each sublayer's output before
            # the residual add.  They must exist as parameter-bearing nodes or
            # the source tensors would be dropped during binding.
            if str(getattr(architecture, "norm_placement", "pre") or "pre").lower() == "sandwich":
                for component, source_id in (
                    ("post_attention_norm", out_proj_node.id),
                    ("post_ffn_norm", ffn_norm_node.id),
                ):
                    sandwich_node = AEGGraphNode(
                        id=f"{layer_prefix}_{component}",
                        node_type=AEGGraphNodeType.OPERATION,
                        name=f"Layer {layer} {component}",
                        op_type=(
                            "layernorm"
                            if str(getattr(architecture, "norm_type", "RMSNorm")).lower() == "layernorm"
                            else "rmsnorm"
                        ),
                        inputs=[source_id],
                        attributes={"eps": architecture.norm_eps, "hidden_size": h},
                        precision=None,
                        layer_index=layer,
                    )
                    graph.add_node(sandwich_node)
                    graph.add_edge(AEGGraphEdge(source=source_id, target=sandwich_node.id))

            if is_moe_layer:
                # MoE FFN with a real router parameter and separate expert
                # projections. The artifact must retain every expert tensor.
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

                for expert_index in range(int(architecture.num_experts)):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        graph.add_node(AEGGraphNode(
                            id=f"{layer_prefix}_expert_{expert_index}_{projection}",
                            node_type=AEGGraphNodeType.PARAMETER,
                            name=(
                                f"Layer {layer} Expert {expert_index} "
                                f"{projection}"
                            ),
                            op_type="expert_weight",
                            attributes={
                                "expert_index": expert_index,
                                "projection": projection,
                                "num_experts": architecture.num_experts,
                                "intermediate_size": i,
                                "hidden_size": h,
                            },
                            layer_index=layer,
                        ))

                ffn_node = AEGGraphNode(
                    id=f"{layer_prefix}_moe_ffn",
                    node_type=AEGGraphNodeType.EXPERT_BANK,
                    name=f"Layer {layer} MoE FFN",
                    op_type="expert_ffn",
                    inputs=[moe_router_node.id],
                    attributes={
                        "num_experts": architecture.num_experts,
                        "num_activated": architecture.num_activated_experts,
                        "expert_intermediate_size": i,
                    },
                    precision=None,
                    layer_index=layer,
                )
                graph.add_node(ffn_node)
                graph.add_edge(AEGGraphEdge(source=moe_router_node.id, target=ffn_node.id))
            else:
                # GLU families use gate+up projections.  GPT-style GELU
                # blocks use a single intermediate projection (mapped to the
                # canonical gate_proj slot) and the down/output projection.
                ffn_type = str(getattr(architecture, "ffn_type", "SwiGLU") or "SwiGLU")
                classic_gelu = ffn_type.lower() in {"gelu", "relu", "relu2"}
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
                    name=f"Layer {layer} {ffn_type} FFN",
                    op_type="gelu_ffn" if classic_gelu else "swiglu_ffn",
                    inputs=[gate_node.id] if classic_gelu else [gate_node.id, ffn_norm_node.id],
                    attributes={
                        "intermediate_size": i,
                        "hidden_size": h,
                        "ffn_type": ffn_type,
                        "requires_up_projection": not classic_gelu,
                    },
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
                inputs=(
                    [prev_node.id, out_proj_node.id, ffn_node.id]
                    if bool(getattr(architecture, "parallel_residual", False))
                    else [residual_add_node.id, ffn_node.id]
                ),
                attributes={},
                precision=None,
                layer_index=layer,
            )
            graph.add_node(final_residual_node)
            if bool(getattr(architecture, "parallel_residual", False)):
                graph.add_edge(AEGGraphEdge(source=prev_node.id, target=final_residual_node.id))
                graph.add_edge(AEGGraphEdge(source=out_proj_node.id, target=final_residual_node.id))
            else:
                graph.add_edge(AEGGraphEdge(source=residual_add_node.id, target=final_residual_node.id))
            graph.add_edge(AEGGraphEdge(source=ffn_node.id, target=final_residual_node.id))

            prev_node = final_residual_node

        # ── Final RMSNorm ──
        final_norm_node = AEGGraphNode(
            id="final_norm",
            node_type=AEGGraphNodeType.OPERATION,
            name="Final RMSNorm",
            op_type="layernorm" if str(getattr(architecture, "norm_type", "RMSNorm")).lower() == "layernorm" else "rmsnorm",
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
