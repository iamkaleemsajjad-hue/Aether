"""
Aether Runtime — VLM & Multimodal Model Loader.

Implements graph extraction for Vision-Language Models (VLMs) including:
  - LLaVA / LLaVA-NeXT (ViT + language backbone)
  - Qwen-VL / Qwen2-VL (dynamic resolution ViT)
  - InternVL2 (InternViT + InternLM)
  - Phi-3 Vision (CLIP + Phi-3 backbone)
  - Gemma3 Vision (SigLIP + Gemma)
  - PaliGemma2 (SigLIP + Gemma2)
  - Pixtral (ViT-400M + Mistral)
  - Video models: Video-LLaMA2, VideoChat2 (temporal frame processing)

Each loader:
1. Detects the VLM architecture from config.json/model_type
2. Extracts the visual encoder subgraph (ViT/CLIP/SigLIP)
3. Extracts the language backbone subgraph (LLM)
4. Builds the projection/adapter connection nodes
5. Returns an AEGGraph with fully typed nodes and edges

Research basis:
  - LLaVA: Liu et al. (2023)
  - Qwen-VL: Bai et al. (2023)
  - InternVL2: Chen et al. (2024)
  - Phi-3 Vision: Microsoft Research (2024)
  - PaliGemma2: Google DeepMind (2024)
  - PRD v4.0 §4.1 — VLM Graph Extraction
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# VLM architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class VLMArchitecture:
    """Describes the architecture of a Vision-Language Model."""

    model_type: str  # e.g., "llava", "qwen2_vl", "internvl2", "phi3_v"
    vision_encoder: str  # e.g., "clip_vit_l14", "siglip_so400m", "internvit_6b"
    language_backbone: str  # e.g., "llama3_8b", "mistral_7b", "phi3_mini"
    projection_type: str  # "mlp", "qformer", "perceiver", "identity"
    image_resolution: int  # native image resolution (e.g., 336, 448)
    patch_size: int  # ViT patch size (e.g., 14, 16)
    num_image_tokens: int  # tokens per image (e.g., 576, 1024)
    supports_video: bool = False
    supports_multiple_images: bool = False
    dynamic_resolution: bool = False
    max_num_tiles: int = 1  # For dynamic resolution (InternVL, Qwen2-VL)
    num_video_frames: int = 1
    family: str = "vlm"
    layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    vocab_size: int = 32000
    num_kv_heads: int | None = None
    head_dim: int | None = None
    is_moe: bool = False
    num_experts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "vision_encoder": self.vision_encoder,
            "language_backbone": self.language_backbone,
            "projection_type": self.projection_type,
            "image_resolution": self.image_resolution,
            "patch_size": self.patch_size,
            "num_image_tokens": self.num_image_tokens,
            "supports_video": self.supports_video,
            "supports_multiple_images": self.supports_multiple_images,
            "dynamic_resolution": self.dynamic_resolution,
            "max_num_tiles": self.max_num_tiles,
            "family": self.family,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
        }


# ---------------------------------------------------------------------------
# VLM architecture detector
# ---------------------------------------------------------------------------

_VLM_TYPE_MAP: dict[str, dict[str, Any]] = {
    "llava": {
        "vision_encoder": "clip_vit_l14",
        "language_backbone": "llama3",
        "projection_type": "mlp",
        "image_resolution": 336,
        "patch_size": 14,
        "num_image_tokens": 576,
        "supports_multiple_images": True,
    },
    "llava_next": {
        "vision_encoder": "clip_vit_l14",
        "language_backbone": "llama3",
        "projection_type": "mlp",
        "image_resolution": 672,
        "patch_size": 14,
        "num_image_tokens": 2304,
        "dynamic_resolution": True,
        "max_num_tiles": 4,
    },
    "qwen2_vl": {
        "vision_encoder": "qwen_vit",
        "language_backbone": "qwen2",
        "projection_type": "mlp",
        "image_resolution": 448,
        "patch_size": 14,
        "num_image_tokens": 1024,
        "dynamic_resolution": True,
        "max_num_tiles": 4,
        "supports_video": True,
        "supports_multiple_images": True,
    },
    "internvl2": {
        "vision_encoder": "internvit_6b",
        "language_backbone": "internlm2",
        "projection_type": "mlp",
        "image_resolution": 448,
        "patch_size": 14,
        "num_image_tokens": 1024,
        "dynamic_resolution": True,
        "max_num_tiles": 6,
        "supports_video": True,
    },
    "phi3_v": {
        "vision_encoder": "clip_vit_l14_336px",
        "language_backbone": "phi3_mini",
        "projection_type": "mlp",
        "image_resolution": 336,
        "patch_size": 14,
        "num_image_tokens": 576,
        "dynamic_resolution": True,
        "max_num_tiles": 4,
    },
    "paligemma": {
        "vision_encoder": "siglip_so400m",
        "language_backbone": "gemma",
        "projection_type": "identity",
        "image_resolution": 224,
        "patch_size": 14,
        "num_image_tokens": 256,
    },
    "paligemma2": {
        "vision_encoder": "siglip_so400m_448px",
        "language_backbone": "gemma2",
        "projection_type": "identity",
        "image_resolution": 448,
        "patch_size": 14,
        "num_image_tokens": 1024,
        "dynamic_resolution": True,
    },
    "pixtral": {
        "vision_encoder": "vit_400m",
        "language_backbone": "mistral",
        "projection_type": "mlp",
        "image_resolution": 1024,
        "patch_size": 16,
        "num_image_tokens": 4096,
        "dynamic_resolution": True,
    },
    "gemma3": {
        "vision_encoder": "siglip_so400m",
        "language_backbone": "gemma3",
        "projection_type": "identity",
        "image_resolution": 896,
        "patch_size": 14,
        "num_image_tokens": 4096,
        "supports_multiple_images": True,
    },
    "mistral3": {
        "vision_encoder": "vit_400m",
        "language_backbone": "mistral3",
        "projection_type": "mlp",
        "image_resolution": 1024,
        "patch_size": 16,
        "num_image_tokens": 4096,
    },
    "deepseek_vl2": {
        "vision_encoder": "siglip_so400m",
        "language_backbone": "deepseek_v3",
        "projection_type": "mlp",
        "image_resolution": 448,
        "patch_size": 14,
        "num_image_tokens": 1024,
        "supports_multiple_images": True,
    },
    "videollama2": {
        "vision_encoder": "clip_vit_l14",
        "language_backbone": "llama3",
        "projection_type": "qformer",
        "image_resolution": 336,
        "patch_size": 14,
        "num_image_tokens": 576,
        "supports_video": True,
        "num_video_frames": 32,
    },
}


def detect_vlm_architecture(model_path: str | Path) -> VLMArchitecture | None:
    """
    Detect the VLM architecture from a model directory.

    Reads config.json to determine model_type and architecture details.
    Returns None if not a recognized VLM architecture.
    """
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except Exception:  # noqa: BLE001
        return None

    model_type = config.get("model_type", "").lower()

    # Normalize aliases
    aliases = {
        "llava_llama3": "llava",
        "llava_mistral": "llava",
        "llava-next": "llava_next",
        "qwen_vl": "qwen2_vl",
        "phi-3-vision": "phi3_v",
        "phi3vision": "phi3_v",
        "paligemma_v2": "paligemma2",
    }
    model_type = aliases.get(model_type, model_type)

    if model_type not in _VLM_TYPE_MAP:
        return None

    spec = _VLM_TYPE_MAP[model_type].copy()

    # Override with actual config values where available
    llm_config = config.get("text_config", config.get("llm_config", config))
    arch = VLMArchitecture(
        model_type=model_type,
        vision_encoder=spec["vision_encoder"],
        language_backbone=spec["language_backbone"],
        projection_type=spec.get("projection_type", "mlp"),
        image_resolution=config.get("image_size", spec["image_resolution"]),
        patch_size=spec.get("patch_size", 14),
        num_image_tokens=spec.get("num_image_tokens", 576),
        supports_video=spec.get("supports_video", False),
        supports_multiple_images=spec.get("supports_multiple_images", False),
        dynamic_resolution=spec.get("dynamic_resolution", False),
        max_num_tiles=spec.get("max_num_tiles", 1),
        num_video_frames=spec.get("num_video_frames", 1),
        family="vlm",
        layers=llm_config.get("num_hidden_layers", 32),
        hidden_size=llm_config.get("hidden_size", 4096),
        num_attention_heads=llm_config.get("num_attention_heads", 32),
        intermediate_size=llm_config.get("intermediate_size", 11008),
        vocab_size=llm_config.get("vocab_size", 32000),
        num_kv_heads=llm_config.get("num_key_value_heads"),
    )
    return arch


# ---------------------------------------------------------------------------
# VLM graph builder (produces AEG-compatible nodes)
# ---------------------------------------------------------------------------

@dataclass
class VLMGraphNode:
    """A node in a VLM computation graph."""

    node_id: str
    node_type: str  # "vision_encoder", "projection", "language_model", "input_merge"
    op: str
    attrs: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


class VLMGraphBuilder:
    """
    Builds a computation graph for a VLM architecture.

    The graph has this topology:
      pixel_values → vision_encoder → patch_features
      patch_features → projection → image_tokens
      text_tokens → embedding_lookup → text_embeddings
      [image_tokens, text_embeddings] → merge_modalities → unified_sequence
      unified_sequence → language_model → logits
    """

    def build(self, arch: VLMArchitecture) -> list[VLMGraphNode]:
        nodes: list[VLMGraphNode] = []

        # 1. Image input node
        nodes.append(VLMGraphNode(
            node_id="pixel_input",
            node_type="input",
            op="aeg.image_input",
            attrs={
                "resolution": arch.image_resolution,
                "patch_size": arch.patch_size,
                "channels": 3,
                "dynamic_resolution": arch.dynamic_resolution,
                "max_num_tiles": arch.max_num_tiles,
            },
        ))

        # 2. Vision encoder
        nodes.append(VLMGraphNode(
            node_id="vision_encoder",
            node_type="vision_encoder",
            op=f"aeg.vision_encoder.{arch.vision_encoder}",
            attrs={
                "encoder_type": arch.vision_encoder,
                "output_tokens": arch.num_image_tokens,
                "hidden_size": self._encoder_hidden_size(arch.vision_encoder),
                "patch_size": arch.patch_size,
                "image_resolution": arch.image_resolution,
            },
            inputs=["pixel_input"],
        ))

        # 3. Video frame pooler (if video model)
        if arch.supports_video:
            nodes.append(VLMGraphNode(
                node_id="temporal_pooler",
                node_type="temporal_pooler",
                op="aeg.temporal_avg_pool",
                attrs={
                    "num_frames": arch.num_video_frames,
                    "pooling": "average",
                },
                inputs=["vision_encoder"],
            ))
            last_visual = "temporal_pooler"
        else:
            last_visual = "vision_encoder"

        # 4. Projection layer
        if arch.projection_type != "identity":
            nodes.append(VLMGraphNode(
                node_id="projection",
                node_type="projection",
                op=f"aeg.projection.{arch.projection_type}",
                attrs={
                    "projection_type": arch.projection_type,
                    "in_features": self._encoder_hidden_size(arch.vision_encoder),
                    "out_features": arch.hidden_size,
                },
                inputs=[last_visual],
            ))
            last_visual = "projection"

        # 5. Text embedding lookup
        nodes.append(VLMGraphNode(
            node_id="text_embedding",
            node_type="embedding",
            op="aeg.embedding_lookup",
            attrs={
                "vocab_size": arch.vocab_size,
                "hidden_size": arch.hidden_size,
            },
            inputs=["input_ids"],
        ))

        # 6. Merge modalities
        nodes.append(VLMGraphNode(
            node_id="modality_merge",
            node_type="modality_merge",
            op="aeg.merge_modalities",
            attrs={
                "image_token_id": -200,  # Special token marking image positions
                "merge_strategy": "replace",  # Replace <image> tokens with visual features
            },
            inputs=[last_visual, "text_embedding"],
        ))

        # 7. Language model transformer layers
        nodes.append(VLMGraphNode(
            node_id="language_model",
            node_type="language_model",
            op=f"aeg.language_model.{arch.language_backbone}",
            attrs={
                "backbone": arch.language_backbone,
                "num_layers": arch.layers,
                "hidden_size": arch.hidden_size,
                "num_heads": arch.num_attention_heads,
                "intermediate_size": arch.intermediate_size,
            },
            inputs=["modality_merge"],
        ))

        # 8. LM head
        nodes.append(VLMGraphNode(
            node_id="lm_head",
            node_type="output",
            op="aeg.lm_head",
            attrs={"vocab_size": arch.vocab_size, "hidden_size": arch.hidden_size},
            inputs=["language_model"],
        ))

        return nodes

    def _encoder_hidden_size(self, encoder_type: str) -> int:
        """Return the hidden size of a known vision encoder."""
        sizes = {
            "clip_vit_l14": 1024,
            "clip_vit_l14_336px": 1024,
            "siglip_so400m": 1152,
            "siglip_so400m_448px": 1152,
            "internvit_6b": 3200,
            "qwen_vit": 1280,
            "vit_400m": 1024,
        }
        return sizes.get(encoder_type, 1024)


# ---------------------------------------------------------------------------
# VLM loader entry point
# ---------------------------------------------------------------------------

class VLMLoader:
    """
    Complete VLM/Multimodal model loader for Aether Runtime.

    Supports all major VLM architectures. Integrates with the AEG ingestion
    pipeline to produce a typed graph suitable for compiler optimization.
    """

    def __init__(self) -> None:
        self._builder = VLMGraphBuilder()

    def load(
        self,
        model_path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> tuple[VLMArchitecture, list[VLMGraphNode]] | None:
        """
        Load a VLM from a model directory.

        Returns:
            (architecture, graph_nodes) if successful, None if not a VLM.
        """
        arch = detect_vlm_architecture(model_path)
        if arch is None:
            return None

        logger.info(
            f"Loading VLM: {arch.model_type} | "
            f"vision={arch.vision_encoder} + language={arch.language_backbone} | "
            f"image_tokens={arch.num_image_tokens}"
        )

        nodes = self._builder.build(arch)
        return arch, nodes

    def is_vlm(self, model_path: str | Path) -> bool:
        """Check if a model directory contains a VLM."""
        return detect_vlm_architecture(model_path) is not None

    @staticmethod
    def list_supported_types() -> list[str]:
        """Return a list of all supported VLM architecture types."""
        return sorted(_VLM_TYPE_MAP.keys())


# ---------------------------------------------------------------------------
# Video model loader
# ---------------------------------------------------------------------------

@dataclass
class VideoModelConfig:
    """Configuration for video model processing."""

    max_frames: int = 32
    frame_sample_rate: int = 1  # 1 = every frame, 2 = every other frame
    temporal_aggregation: str = "average"  # "average", "max", "concat"
    num_image_tokens: int = 576
    vision_encoder: str = "clip_vit_l14"
    language_backbone: str = "llama3"


class VideoModelLoader(VLMLoader):
    """
    Specialized loader for video understanding models.

    Handles:
    - Video-LLaMA2 (temporal pooling + Q-Former)
    - VideoChat2 (MQ-Former for multi-scale video)
    - Qwen2-VL in video mode (dynamic frame processing)
    - InternVL2-Video (split frame encoding)

    Adds temporal processing nodes to the VLM graph.
    """

    def load_video_model(
        self,
        model_path: str | Path,
        video_config: VideoModelConfig | None = None,
    ) -> tuple[VLMArchitecture, list[VLMGraphNode]] | None:
        """
        Load a video model, forcing video=True even if not detected.
        """
        vcfg = video_config or VideoModelConfig()
        model_path = Path(model_path)

        # Try standard VLM detection first
        result = self.load(model_path)

        if result is None:
            # Check for dedicated video model configs
            config_path = model_path / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                model_type = config.get("model_type", "").lower()
                if "video" in model_type or model_type in ("videollama", "videochat"):
                    # Force video VLM construction
                    arch = VLMArchitecture(
                        model_type=model_type,
                        vision_encoder=vcfg.vision_encoder,
                        language_backbone=vcfg.language_backbone,
                        projection_type="qformer",
                        image_resolution=336,
                        patch_size=14,
                        num_image_tokens=vcfg.num_image_tokens,
                        supports_video=True,
                        num_video_frames=vcfg.max_frames,
                        family="vlm",
                    )
                    nodes = self._builder.build(arch)
                    return arch, nodes
            return None

        arch, nodes = result
        # Upgrade existing VLM to video mode
        arch.supports_video = True
        arch.num_video_frames = vcfg.max_frames
        # Rebuild with video nodes
        nodes = self._builder.build(arch)
        return arch, nodes
