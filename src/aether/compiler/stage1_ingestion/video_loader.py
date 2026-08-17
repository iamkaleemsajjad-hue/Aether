"""
Aether Runtime Ã¢â‚¬â€ Video Model Loader.

Implements graph extraction for video understanding models including:
  - Video-LLaMA2 (video + audio + language)
  - VideoChat2 (Q-Former video-to-language bridge)
  - LLaVA-Video (temporal frame ViT + LLM)
  - InternVideo2 (dual-encoder video-text)
  - Video-R1 / Video-o1 (reasoning video models)

Each loader:
1. Detects the video model architecture from config.json / model_type
2. Extracts the visual encoder subgraph (ViT / video encoder)
3. Extracts the temporal aggregation subgraph (Q-Former / temporal attn)
4. Extracts the language backbone subgraph
5. Builds the projection/adapter connection nodes
6. Returns an AEGGraph with fully typed nodes and edges

Research basis:
  - Video-LLaMA2: Cheng et al. (2024) Ã¢â‚¬â€ https://arxiv.org/abs/2406.07476
  - VideoChat2: Li et al. (2024) Ã¢â‚¬â€ https://arxiv.org/abs/2311.17005
  - LLaVA-Video: Zhang et al. (2024) Ã¢â‚¬â€ https://arxiv.org/abs/2410.02713
  - InternVideo2: Wang et al. (2024) Ã¢â‚¬â€ https://arxiv.org/abs/2403.15377
  - PRD v4.0 Ã‚Â§4.2 Ã¢â‚¬â€ Video Graph Extraction
  - PRD v5.0 Ã‚Â§5.1 Ã¢â‚¬â€ Temporal KV Compression
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Video architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class VideoArchitecture:
    """Describes the architecture of a video understanding model."""

    model_type: str          # e.g., "video_llama2", "videochat2", "llava_video"
    video_encoder: str       # e.g., "eva_vit_g", "clip_vit_l14", "internvideo2"
    temporal_aggregator: str  # "qformer", "temporal_attn", "spatial_pool", "mean_pool"
    language_backbone: str   # e.g., "llama3_8b", "mistral_7b", "internlm2_7b"
    projection_type: str     # "mlp", "qformer_proj", "perceiver", "identity"
    frame_resolution: int    # Per-frame resolution (e.g., 224, 336)
    patch_size: int          # ViT patch size (e.g., 14, 16)
    max_frames: int          # Maximum frames sampled per video (e.g., 8, 16, 32)
    tokens_per_frame: int    # Visual tokens per frame (e.g., 256, 576)
    has_audio: bool = False  # Whether audio encoder is present
    supports_streaming: bool = False  # Online/streaming video processing
    temporal_compression: int = 1    # Temporal token compression ratio

    @property
    def total_visual_tokens(self) -> int:
        return self.max_frames * self.tokens_per_frame // self.temporal_compression


@dataclass
class VideoGraphMetadata:
    """Graph-level metadata for a video model AEG."""

    architecture: VideoArchitecture
    num_video_encoder_nodes: int = 0
    num_temporal_nodes: int = 0
    num_language_nodes: int = 0
    num_projection_nodes: int = 0
    frame_kv_budget: int = 0  # Max KV cache tokens for all frames


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_VIDEO_ARCH_REGISTRY: dict[str, VideoArchitecture] = {
    "video_llama": VideoArchitecture(
        model_type="video_llama",
        video_encoder="eva_vit_g_1b",
        temporal_aggregator="qformer",
        language_backbone="llama2_7b",
        projection_type="qformer_proj",
        frame_resolution=224,
        patch_size=14,
        max_frames=8,
        tokens_per_frame=32,  # Q-Former compresses to 32 tokens/frame
        has_audio=True,
    ),
    "video_llama2": VideoArchitecture(
        model_type="video_llama2",
        video_encoder="clip_vit_l14",
        temporal_aggregator="temporal_attn",
        language_backbone="llama2_13b",
        projection_type="mlp",
        frame_resolution=224,
        patch_size=14,
        max_frames=16,
        tokens_per_frame=256,
        has_audio=False,
        temporal_compression=4,
    ),
    "videochat2": VideoArchitecture(
        model_type="videochat2",
        video_encoder="eva_vit_g_1b",
        temporal_aggregator="qformer",
        language_backbone="mistral_7b",
        projection_type="qformer_proj",
        frame_resolution=224,
        patch_size=14,
        max_frames=16,
        tokens_per_frame=32,
        has_audio=False,
    ),
    "llava_video": VideoArchitecture(
        model_type="llava_video",
        video_encoder="siglip_so400m",
        temporal_aggregator="spatial_pool",
        language_backbone="qwen2_72b",
        projection_type="mlp",
        frame_resolution=384,
        patch_size=14,
        max_frames=32,
        tokens_per_frame=144,  # 2x2 spatial pooling Ã¢â€ â€™ 144 from 576
        temporal_compression=1,
    ),
    "internvideo2": VideoArchitecture(
        model_type="internvideo2",
        video_encoder="internvideo2_6b",
        temporal_aggregator="mean_pool",
        language_backbone="internlm2_7b",
        projection_type="mlp",
        frame_resolution=224,
        patch_size=14,
        max_frames=8,
        tokens_per_frame=256,
    ),
}

# Aliases for common HF model_type strings
_VIDEO_TYPE_ALIASES = {
    "video_llava": "llava_video",
    "llava_next_video": "llava_video",
    "video-chatgpt": "video_llama",
    "videollama": "video_llama",
    "videollama2": "video_llama2",
}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class VideoModelLoader:
    """
    Loads and extracts an AEG graph from a video understanding model.

    Supports:
    - Video-LLaMA / Video-LLaMA2
    - VideoChat2
    - LLaVA-Video / LLaVA-NeXT-Video
    - InternVideo2
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """
        Load the video model and return an AEG graph.

        Returns:
            dict with keys:
              - ``graph``: AEGGraph with all nodes and edges
              - ``architecture``: VideoArchitecture descriptor
              - ``metadata``: VideoGraphMetadata
              - ``format``: "video_model"
        """
        path = self.model_path
        if not path.exists():
            from aether.core.exceptions import IngestionError
            raise IngestionError(f"Video model not found: {path}")

        config = self._load_config(path)
        arch = self._detect_architecture(config, path)

        logger.info(
            "Detected video model architecture",
            model_type=arch.model_type,
            video_encoder=arch.video_encoder,
            temporal_aggregator=arch.temporal_aggregator,
            language_backbone=arch.language_backbone,
            max_frames=arch.max_frames,
            total_visual_tokens=arch.total_visual_tokens,
        )

        graph = self._build_video_graph(arch, config, path)
        metadata = self._build_metadata(arch, graph)

        logger.info(
            "Video model graph extracted",
            video_nodes=metadata.num_video_encoder_nodes,
            temporal_nodes=metadata.num_temporal_nodes,
            language_nodes=metadata.num_language_nodes,
        )

        return {
            "graph": graph,
            "architecture": arch,
            "metadata": metadata,
            "format": "video_model",
        }

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Load config.json from the model directory."""
        candidates = [path / "config.json", path / "model_config.json"]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        return {}

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    def _detect_architecture(
        self,
        config: dict[str, Any],
        path: Path,
    ) -> VideoArchitecture:
        """Detect which video model architecture this is."""
        raw_type = (config.get("model_type") or "").lower().replace("-", "_")
        canonical = _VIDEO_TYPE_ALIASES.get(raw_type, raw_type)

        if canonical in _VIDEO_ARCH_REGISTRY:
            arch = _VIDEO_ARCH_REGISTRY[canonical]
        else:
            # Heuristic detection from config fields
            arch = self._detect_from_config_fields(config, canonical)

        return self._apply_config_overrides(arch, config)

    def _detect_from_config_fields(
        self,
        config: dict[str, Any],
        model_type: str,
    ) -> VideoArchitecture:
        """Detect architecture from config fields when model_type is unknown."""
        vision_cfg = config.get("vision_config", {})
        text_cfg = config.get("text_config", {})

        # Detect video encoder
        encoder_type = vision_cfg.get("model_type", "clip_vit_l14")
        # Detect temporal method from config
        has_qformer = "qformer" in str(config).lower()
        temporal = "qformer" if has_qformer else "spatial_pool"
        # Language backbone
        lang_type = text_cfg.get("model_type", "llama")

        return VideoArchitecture(
            model_type=model_type or "video_unknown",
            video_encoder=encoder_type,
            temporal_aggregator=temporal,
            language_backbone=lang_type,
            projection_type="qformer_proj" if has_qformer else "mlp",
            frame_resolution=vision_cfg.get("image_size", 224),
            patch_size=vision_cfg.get("patch_size", 14),
            max_frames=config.get("max_frames", 8),
            tokens_per_frame=config.get("num_query_tokens", 32) if has_qformer else 256,
        )

    def _apply_config_overrides(
        self,
        arch: VideoArchitecture,
        config: dict[str, Any],
    ) -> VideoArchitecture:
        """Override arch defaults with explicit config values."""
        import dataclasses
        overrides: dict[str, Any] = {}
        vision_cfg = config.get("vision_config", {})

        if "max_frames" in config:
            overrides["max_frames"] = int(config["max_frames"])
        if "image_size" in vision_cfg:
            overrides["frame_resolution"] = int(vision_cfg["image_size"])
        if "patch_size" in vision_cfg:
            overrides["patch_size"] = int(vision_cfg["patch_size"])
        if "num_query_tokens" in config and arch.temporal_aggregator == "qformer":
            overrides["tokens_per_frame"] = int(config["num_query_tokens"])

        return dataclasses.replace(arch, **overrides) if overrides else arch

    # ------------------------------------------------------------------
    # AEG graph construction
    # ------------------------------------------------------------------

    def _build_video_graph(
        self,
        arch: VideoArchitecture,
        config: dict[str, Any],
        path: Path,
    ) -> Any:
        """Build the full AEG graph with video, temporal, and language subgraphs."""
        try:
            from aether.core.graph import AEGGraph, AEGGraphEdge, AEGGraphNode, AEGGraphNodeType
        except ImportError:
            return _FallbackGraph(arch)

        graph = AEGGraph()

        # === Video encoder subgraph ===
        ve_node = AEGGraphNode(
            id="video_encoder",
            node_type=AEGGraphNodeType.OPERATION,
            name=arch.video_encoder,
            op_type="aeg.video_encoder",
            attributes={
                "encoder_type": arch.video_encoder,
                "frame_resolution": arch.frame_resolution,
                "patch_size": arch.patch_size,
                "max_frames": arch.max_frames,
                "tokens_per_frame_raw": (arch.frame_resolution // arch.patch_size) ** 2,
                "has_audio": arch.has_audio,
                "supports_streaming": arch.supports_streaming,
            },
        )
        graph.add_node(ve_node)

        # === Temporal aggregation subgraph ===
        ta_node = self._build_temporal_node(arch)
        graph.add_node(ta_node)
        graph.add_edge(AEGGraphEdge(ve_node.id, ta_node.id))

        # === Audio encoder (optional) ===
        if arch.has_audio:
            audio_node = AEGGraphNode(
                id="audio_encoder",
                node_type=AEGGraphNodeType.OPERATION
                if hasattr(AEGGraphNodeType, "AUDIO_ENCODER")
                else AEGGraphNodeType.OPERATION,
                name="whisper_small",
                op_type="aeg.audio_encoder",
                attributes={
                    "sample_rate": 16000,
                    "hop_length": 160,
                    "n_mels": 80,
                },
            )
            graph.add_node(audio_node)
            graph.add_edge(AEGGraphEdge(audio_node.id, ta_node.id))

        # === Projection layer ===
        proj_node = AEGGraphNode(
            id="video_projection",
            node_type=AEGGraphNodeType.OPERATION,
            name=f"{arch.projection_type}_projection",
            op_type="aeg.multimodal_projection",
            attributes={
                "projection_type": arch.projection_type,
                "output_tokens": arch.total_visual_tokens,
                "temporal_compression": arch.temporal_compression,
            },
        )
        graph.add_node(proj_node)
        graph.add_edge(AEGGraphEdge(ta_node.id, proj_node.id))

        # === Language backbone (transformer layers) ===
        llm_layers = config.get("num_hidden_layers",
                                config.get("text_config", {}).get("num_hidden_layers", 32))
        for i in range(max(1, min(llm_layers, 128))):
            layer_node = AEGGraphNode(
                id=f"llm_layer_{i}",
                node_type=AEGGraphNodeType.OPERATION,
                name=f"transformer_layer_{i}",
                op_type="aeg.transformer_layer",
                layer_index=i,
                attributes={
                    "backbone": arch.language_backbone,
                    "receives_visual_tokens": i == 0,
                },
            )
            graph.add_node(layer_node)
            if i == 0:
                graph.add_edge(AEGGraphEdge(proj_node.id, layer_node.id))
            else:
                graph.add_edge(AEGGraphEdge(f"llm_layer_{i - 1}", layer_node.id))

        # === LM head ===
        lm_head = AEGGraphNode(
            id="lm_head",
            node_type=AEGGraphNodeType.OUTPUT,
            name="lm_head",
            op_type="aeg.lm_head",
            attributes={"vocab_size": config.get("vocab_size", 32000)},
        )
        graph.add_node(lm_head)
        graph.add_edge(AEGGraphEdge(f"llm_layer_{max(0, min(llm_layers, 128) - 1)}", lm_head.id))

        # === Frame KV budget metadata ===
        if hasattr(graph, "set_metadata"):
            graph.set_metadata("video_architecture", {
                "model_type": arch.model_type,
                "max_frames": arch.max_frames,
                "tokens_per_frame": arch.tokens_per_frame,
                "total_visual_tokens": arch.total_visual_tokens,
                "temporal_compression": arch.temporal_compression,
                "frame_kv_budget": arch.total_visual_tokens * 2,  # K + V
            })

        return graph

    def _build_temporal_node(self, arch: VideoArchitecture) -> Any:
        """Build the temporal aggregation node based on the aggregator type."""
        from aether.core.graph import AEGGraphNode, AEGGraphNodeType

        if arch.temporal_aggregator == "qformer":
            return AEGGraphNode(
                id="temporal_qformer",
                node_type=AEGGraphNodeType.OPERATION,
                name="QFormer_Temporal",
                op_type="aeg.video_qformer",
                attributes={
                    "num_query_tokens": arch.tokens_per_frame,
                    "num_frames": arch.max_frames,
                    "total_output_tokens": arch.max_frames * arch.tokens_per_frame,
                    "aggregator": "qformer",
                },
            )
        elif arch.temporal_aggregator == "spatial_pool":
            pool_factor = 4  # 2x2 spatial pooling
            return AEGGraphNode(
                id="temporal_spatial_pool",
                node_type=AEGGraphNodeType.OPERATION,
                name="SpatialPool_Temporal",
                op_type="aeg.spatial_temporal_pool",
                attributes={
                    "pool_factor": pool_factor,
                    "tokens_per_frame_out": arch.tokens_per_frame,
                    "num_frames": arch.max_frames,
                    "aggregator": "spatial_pool",
                },
            )
        elif arch.temporal_aggregator == "mean_pool":
            return AEGGraphNode(
                id="temporal_mean_pool",
                node_type=AEGGraphNodeType.OPERATION,
                name="MeanPool_Temporal",
                op_type="aeg.temporal_mean_pool",
                attributes={
                    "pool_frames": arch.max_frames,
                    "tokens_out": arch.tokens_per_frame,
                    "aggregator": "mean_pool",
                },
            )
        else:  # temporal_attn
            return AEGGraphNode(
                id="temporal_attn",
                node_type=AEGGraphNodeType.OPERATION,
                name="TemporalAttention",
                op_type="aeg.temporal_self_attention",
                attributes={
                    "num_frames": arch.max_frames,
                    "tokens_per_frame": arch.tokens_per_frame,
                    "compression_ratio": arch.temporal_compression,
                    "aggregator": "temporal_attn",
                },
            )

    def _build_metadata(
        self, arch: VideoArchitecture, graph: Any
    ) -> VideoGraphMetadata:
        """Compute graph-level metadata after graph construction."""
        if hasattr(graph, "nodes"):
            nodes = list(graph.nodes.values()) if hasattr(graph.nodes, "values") else []
        elif hasattr(graph, "_nodes"):
            nodes = list(getattr(graph, "_nodes", {}).values())
        else:
            nodes = []

        video_nodes = sum(
            1 for n in nodes if getattr(n, "op_type", "").startswith("aeg.video")
        )
        temporal_nodes = sum(
            1 for n in nodes
            if getattr(n, "op_type", "").startswith("aeg.temporal")
            or getattr(n, "op_type", "").startswith("aeg.spatial")
            or getattr(n, "op_type", "") == "aeg.video_qformer"
        )
        lang_nodes = sum(
            1 for n in nodes if getattr(n, "op_type", "") == "aeg.transformer_layer"
        )
        proj_nodes = sum(
            1 for n in nodes if "projection" in getattr(n, "op_type", "")
        )

        return VideoGraphMetadata(
            architecture=arch,
            num_video_encoder_nodes=video_nodes,
            num_temporal_nodes=temporal_nodes,
            num_language_nodes=lang_nodes,
            num_projection_nodes=proj_nodes,
            frame_kv_budget=arch.total_visual_tokens * 2,
        )


# ---------------------------------------------------------------------------
# Fallback graph (when aether.core.graph is not available)
# ---------------------------------------------------------------------------

class _FallbackGraph:
    """Minimal graph for environments without aether.core.graph."""

    def __init__(self, arch: VideoArchitecture) -> None:
        self.arch = arch
        self._nodes: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self.edges: list[tuple[str, str]] = []

    @property
    def nodes(self) -> dict[str, Any]:
        return self._nodes

    def add_node(self, node: Any) -> None:
        self._nodes[getattr(node, "id", str(len(self._nodes)))] = node

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.append((src, dst))

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        return self._metadata.get(key)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_video_model(model_path: str | Path) -> dict[str, Any]:
    """
    Load a video model and return the extracted AEG graph.

    Convenience wrapper around ``VideoModelLoader``.
    """
    return VideoModelLoader(model_path).load()


def detect_video_architecture(config: dict[str, Any]) -> VideoArchitecture | None:
    """
    Detect the video architecture from a HuggingFace config dict.

    Returns None if the config does not describe a video model.
    """
    raw_type = (config.get("model_type") or "").lower().replace("-", "_")
    canonical = _VIDEO_TYPE_ALIASES.get(raw_type, raw_type)

    if canonical in _VIDEO_ARCH_REGISTRY:
        return _VIDEO_ARCH_REGISTRY[canonical]

    # Heuristic: look for video-specific keys
    video_keys = {"max_frames", "video_config", "num_video_query_token", "video_encoder"}
    if video_keys & set(config.keys()):
        loader = VideoModelLoader.__new__(VideoModelLoader)
        return loader._detect_from_config_fields(config, canonical)

    return None
