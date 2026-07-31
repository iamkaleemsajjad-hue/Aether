"""
Multi-Modal Graph Dispatch — VLM Vision-Language Model Support.

Handles three VLM architectures as defined in PRD §7.2:
  1. Qwen3-VL / InternVL  — Early fusion (ViT embedded at token level)
  2. LLaVA family         — Late fusion (ViT → MLP connector → LLM)
  3. InternVL2            — Late fusion with high-resolution tiling

Pipeline:
  Stage 1: Image preprocessing (resize, normalize, patch)
  Stage 2: ViT encoder (visual feature extraction)
  Stage 3: Token compression / dynamic token reduction
  Stage 4: Cross-modal connector (MLP or cross-attention)
  Stage 5: LLM forward (combined text + visual tokens)

Research:
  - Fast-VLM (2025): dynamic token compression — 8.3x speedup
  - AttentionPack (2025): pack visual tokens efficiently
  - Qwen3-VL early-fusion (2025)
  - InternVL2 dynamic resolution (2024)
  - LLaVA (2023): visual instruction tuning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# VLM architecture configs
# ---------------------------------------------------------------------------

class VLMArchitecture:
    EARLY_FUSION  = "early_fusion"   # Qwen3-VL, GLM-4V
    LATE_FUSION   = "late_fusion"    # LLaVA, InternVL
    CROSS_ATTN    = "cross_attn"     # Flamingo-style


@dataclass
class VLMConfig:
    """Configuration for a Vision-Language Model."""
    architecture: str = VLMArchitecture.LATE_FUSION
    vit_model: str = "clip-vit-large-patch14"
    image_size: int = 336
    patch_size: int = 14
    num_image_tokens: int = 576          # standard LLaVA-1.5 tokens
    dynamic_resolution: bool = False     # InternVL dynamic tiling
    max_tiles: int = 12                  # InternVL max tiles
    connector_type: str = "mlp"          # "mlp" | "qformer" | "cross_attn"
    connector_hidden_dim: int = 4096
    compression_ratio: float = 1.0       # Fast-VLM token compression
    image_rope: bool = False             # Qwen3-VL 2D RoPE for images

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "vit_model": self.vit_model,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_image_tokens": self.num_image_tokens,
            "dynamic_resolution": self.dynamic_resolution,
            "max_tiles": self.max_tiles,
            "connector_type": self.connector_type,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def llava_15(cls) -> "VLMConfig":
        return cls(
            architecture=VLMArchitecture.LATE_FUSION,
            vit_model="clip-vit-large-patch14-336",
            image_size=336, patch_size=14, num_image_tokens=576,
            connector_type="mlp",
        )

    @classmethod
    def qwen3_vl(cls) -> "VLMConfig":
        return cls(
            architecture=VLMArchitecture.EARLY_FUSION,
            vit_model="qwen-vit-dynamic",
            image_size=448, patch_size=14, num_image_tokens=256,
            dynamic_resolution=True, max_tiles=16,
            connector_type="mlp", image_rope=True,
        )

    @classmethod
    def internvl2(cls) -> "VLMConfig":
        return cls(
            architecture=VLMArchitecture.LATE_FUSION,
            vit_model="intern-vit-6b",
            image_size=448, patch_size=14, num_image_tokens=256,
            dynamic_resolution=True, max_tiles=12,
            connector_type="mlp", compression_ratio=0.5,
        )


# ---------------------------------------------------------------------------
# Image preprocessor
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Preprocesses images for VLM input.

    Steps:
    1. Resize to target resolution (with dynamic tiling for high-res)
    2. Normalize with mean/std
    3. Split into patches (patch_size × patch_size)
    4. Dynamic token compression (Fast-VLM style)
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    OPENAI_MEAN   = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    OPENAI_STD    = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._use_openai_norm = "clip" in config.vit_model.lower()

    def preprocess(self, image: np.ndarray) -> dict[str, Any]:
        """
        Preprocess a raw image (H, W, 3) uint8 or float32.

        Returns:
            Dict with 'pixel_values' (tiles, C, H, W) and 'num_tiles'.
        """
        cfg = self.config
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        if cfg.dynamic_resolution:
            tiles, num_tiles = self._dynamic_tile(img, cfg.max_tiles, cfg.image_size)
        else:
            resized = self._resize(img, cfg.image_size, cfg.image_size)
            tiles = resized[np.newaxis]   # (1, H, W, 3)
            num_tiles = 1

        # Normalize
        mean = self.OPENAI_MEAN if self._use_openai_norm else self.IMAGENET_MEAN
        std  = self.OPENAI_STD  if self._use_openai_norm else self.IMAGENET_STD
        tiles = (tiles - mean[np.newaxis, np.newaxis, np.newaxis, :]) / (
            std[np.newaxis, np.newaxis, np.newaxis, :] + 1e-6
        )

        # (tiles, H, W, C) → (tiles, C, H, W)
        pixel_values = tiles.transpose(0, 3, 1, 2).astype(np.float32)
        return {"pixel_values": pixel_values, "num_tiles": num_tiles}

    def _resize(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinear resize using numpy (production: PIL/cv2)."""
        src_h, src_w = img.shape[:2]
        row_idx = np.linspace(0, src_h - 1, h).astype(int)
        col_idx = np.linspace(0, src_w - 1, w).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    def _dynamic_tile(
        self, img: np.ndarray, max_tiles: int, tile_size: int
    ) -> tuple[np.ndarray, int]:
        """
        Dynamic resolution tiling (InternVL / Qwen3-VL style).

        Splits high-res image into up to max_tiles tiles of tile_size × tile_size.
        """
        src_h, src_w = img.shape[:2]
        # Compute optimal tile grid
        aspect = src_w / max(src_h, 1)
        grid_w = min(max_tiles, max(1, int(np.sqrt(max_tiles * aspect))))
        grid_h = min(max_tiles // grid_w, max(1, max_tiles // max(grid_w, 1)))
        num_tiles = grid_w * grid_h

        tiles = []
        th = tile_size // grid_h
        tw = tile_size // grid_w
        resized = self._resize(img, tile_size, tile_size)

        for ri in range(grid_h):
            for ci in range(grid_w):
                tile = resized[ri*th:(ri+1)*th, ci*tw:(ci+1)*tw]
                # Resize tile to full tile_size
                tile_full = self._resize(tile, tile_size, tile_size)
                tiles.append(tile_full)

        # Also add full-image thumbnail as context tile
        thumbnail = self._resize(img, tile_size, tile_size)
        tiles.append(thumbnail)
        num_tiles = len(tiles)

        return np.stack(tiles, axis=0), num_tiles


# ---------------------------------------------------------------------------
# ViT Encoder (reference)
# ---------------------------------------------------------------------------

class ViTEncoder:
    """
    Vision Transformer encoder — reference implementation.

    Produces visual feature tokens from preprocessed image patches.
    Production: dispatches to compiled ViT AEG artifact.
    """

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._hidden_dim = config.connector_hidden_dim
        # Fake projection for reference implementation
        self._rng = np.random.default_rng(42)

    def encode(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Encode preprocessed image tiles into visual tokens.

        Args:
            pixel_values: (num_tiles, C, H, W)

        Returns:
            visual_tokens: (num_tiles * num_patches, hidden_dim)
        """
        cfg = self.config
        num_tiles, C, H, W = pixel_values.shape
        num_patches_per_tile = (H // cfg.patch_size) * (W // cfg.patch_size)
        total_patches = num_tiles * num_patches_per_tile

        # Reference: random projection of flattened patches
        flat_dim = C * cfg.patch_size * cfg.patch_size
        proj = self._rng.normal(0, 1 / flat_dim ** 0.5,
                                (flat_dim, self._hidden_dim)).astype(np.float32)

        visual_tokens = np.zeros((total_patches, self._hidden_dim), dtype=np.float32)
        idx = 0
        for t in range(num_tiles):
            tile = pixel_values[t]  # (C, H, W)
            for ph in range(H // cfg.patch_size):
                for pw in range(W // cfg.patch_size):
                    patch = tile[
                        :,
                        ph * cfg.patch_size:(ph+1) * cfg.patch_size,
                        pw * cfg.patch_size:(pw+1) * cfg.patch_size,
                    ].ravel()
                    visual_tokens[idx] = patch[:flat_dim] @ proj
                    idx += 1

        # Normalize
        norms = np.linalg.norm(visual_tokens, axis=1, keepdims=True) + 1e-9
        return (visual_tokens / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Token compressor (Fast-VLM style)
# ---------------------------------------------------------------------------

class VisualTokenCompressor:
    """
    Dynamic visual token compression for efficiency.

    Fast-VLM (2025): reduces visual tokens by keeping only the most
    informative tokens based on attention salience scores.
    Achieves 2-4x fewer visual tokens with <1% accuracy loss.
    """

    def __init__(self, compression_ratio: float = 0.5) -> None:
        if not 0.0 < compression_ratio <= 1.0:
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")
        self.compression_ratio = compression_ratio

    def compress(self, visual_tokens: np.ndarray) -> np.ndarray:
        """
        Compress visual tokens by importance scoring.

        Importance = L2 norm of each token (high-norm tokens carry more signal).

        Args:
            visual_tokens: (num_tokens, hidden_dim)

        Returns:
            compressed: (num_keep, hidden_dim)
        """
        if self.compression_ratio >= 1.0:
            return visual_tokens

        num_tokens = visual_tokens.shape[0]
        num_keep = max(1, int(num_tokens * self.compression_ratio))

        # Score by L2 norm
        norms = np.linalg.norm(visual_tokens, axis=1)
        top_idx = np.argsort(norms)[::-1][:num_keep]
        top_idx_sorted = np.sort(top_idx)  # preserve spatial order

        return visual_tokens[top_idx_sorted]


# ---------------------------------------------------------------------------
# Modal connector
# ---------------------------------------------------------------------------

class ModalConnector:
    """
    Connects visual tokens to the LLM's embedding space.

    Modes:
    - MLP:         2-layer MLP projection (LLaVA style)
    - QFormer:     query-based cross-attention (BLIP-2 style)
    - Cross-Attn:  cross-attention from LLM layers to ViT features
    """

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        hidden = config.connector_hidden_dim
        vit_hidden = hidden  # assume same dim for reference
        self._rng = np.random.default_rng(0)

        if config.connector_type == "mlp":
            # Two-layer MLP: visual_dim → hidden → llm_dim
            self._W1 = self._rng.normal(0, 0.02, (vit_hidden, hidden)).astype(np.float32)
            self._W2 = self._rng.normal(0, 0.02, (hidden, hidden)).astype(np.float32)

    def project(self, visual_tokens: np.ndarray) -> np.ndarray:
        """
        Project visual tokens into LLM embedding space.

        Args:
            visual_tokens: (num_tokens, vit_hidden)

        Returns:
            projected: (num_tokens, llm_hidden)
        """
        if self.config.connector_type == "mlp":
            h = np.maximum(0, visual_tokens @ self._W1)   # GELU approx with ReLU
            return h @ self._W2
        else:
            # Passthrough for unsupported connector types
            return visual_tokens


# ---------------------------------------------------------------------------
# Multimodal Graph Dispatcher (main)
# ---------------------------------------------------------------------------

class MultiModalGraphDispatcher:
    """
    Full multimodal graph dispatch engine for VLMs.

    Orchestrates the 5-stage VLM pipeline:
    1. Preprocess images → pixel_values
    2. ViT encode → visual_tokens
    3. Token compress → compressed_visual_tokens
    4. Modal connect → llm_visual_embeddings
    5. Merge with text embeddings → combined input for LLM

    Supports early-fusion (Qwen3-VL) and late-fusion (LLaVA, InternVL2).
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.vit           = ViTEncoder(self.config)
        self.compressor    = VisualTokenCompressor(self.config.compression_ratio)
        self.connector     = ModalConnector(self.config)
        self._requests_processed = 0

    def process_image(self, image: np.ndarray) -> dict[str, Any]:
        """
        Full image processing pipeline: preprocess → encode → compress → project.

        Returns:
            Dict with visual_tokens, num_visual_tokens, pipeline stats.
        """
        t0 = __import__("time").perf_counter()

        # Stage 1: preprocess
        preprocessed = self.preprocessor.preprocess(image)
        pixel_values = preprocessed["pixel_values"]
        num_tiles = preprocessed["num_tiles"]

        # Stage 2: ViT encode
        visual_tokens = self.vit.encode(pixel_values)
        raw_tokens = visual_tokens.shape[0]

        # Stage 3: compress
        if self.config.compression_ratio < 1.0:
            visual_tokens = self.compressor.compress(visual_tokens)

        compressed_tokens = visual_tokens.shape[0]

        # Stage 4: modal connector projection
        projected = self.connector.project(visual_tokens)

        latency = (__import__("time").perf_counter() - t0) * 1000
        self._requests_processed += 1

        return {
            "visual_embeddings": projected,      # (num_tokens, hidden_dim)
            "num_visual_tokens": compressed_tokens,
            "raw_visual_tokens": raw_tokens,
            "num_tiles": num_tiles,
            "compression_ratio": round(compressed_tokens / max(raw_tokens, 1), 3),
            "latency_ms": round(latency, 2),
        }

    def merge_embeddings(
        self,
        text_embeddings: np.ndarray,   # (seq, hidden)
        visual_embeddings: np.ndarray, # (num_visual, hidden)
        image_token_id: int = 32000,
        text_token_ids: list[int] | None = None,
    ) -> np.ndarray:
        """
        Merge text and visual embeddings for LLM input.

        For early-fusion: visual tokens replace <image> placeholder tokens.
        For late-fusion: visual tokens prepended before text.

        Args:
            text_embeddings: (seq_len, hidden_dim)
            visual_embeddings: (num_visual_tokens, hidden_dim)
            image_token_id: Token ID of <image> placeholder.
            text_token_ids: Token IDs for the text sequence (needed for early fusion).

        Returns:
            merged: (total_tokens, hidden_dim)
        """
        if self.config.architecture == VLMArchitecture.EARLY_FUSION:
            if text_token_ids is not None:
                # Find <image> placeholder positions
                img_positions = [
                    i for i, tid in enumerate(text_token_ids)
                    if tid == image_token_id
                ]
                if img_positions:
                    # Replace first occurrence with visual tokens
                    pos = img_positions[0]
                    before = text_embeddings[:pos]
                    after  = text_embeddings[pos+1:]
                    return np.concatenate([before, visual_embeddings, after], axis=0)
            # Fallback: insert visual tokens at position 1 (after BOS)
            return np.concatenate([
                text_embeddings[:1],
                visual_embeddings,
                text_embeddings[1:],
            ], axis=0)
        else:
            # Late fusion: prepend visual tokens
            return np.concatenate([visual_embeddings, text_embeddings], axis=0)

    def detect_vlm_architecture(
        self, model_config: dict[str, Any]
    ) -> VLMConfig | None:
        """
        Detect VLM architecture from model config.

        Returns VLMConfig if VLM detected, None otherwise.
        """
        arch = " ".join(model_config.get("architectures", [])).lower()
        model_type = model_config.get("model_type", "").lower()

        is_vlm = any(k in arch or k in model_type for k in [
            "llava", "qwen_vl", "internvl", "blip", "flamingo", "idefics",
            "fuyu", "paligemma", "mistral_vl", "phi_vision"
        ])
        if not is_vlm:
            return None

        if "qwen" in arch or "qwen" in model_type:
            return VLMConfig.qwen3_vl()
        elif "intern" in arch or "intern" in model_type:
            return VLMConfig.internvl2()
        else:
            return VLMConfig.llava_15()

    def save_to_aeg(self, aeg_dir: str | Path) -> Path:
        """Save multimodal config to AEG package."""
        out = Path(aeg_dir) / "graph" / "multimodal_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": "multimodal/1.0",
            "vlm_config": self.config.to_dict(),
            "pipeline_stages": [
                "image_preprocess",
                "vit_encode",
                "token_compress",
                "modal_connect",
                "embedding_merge",
            ],
        }
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Multimodal config saved", path=str(out))
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "requests_processed": self._requests_processed,
            "architecture": self.config.architecture,
            "num_image_tokens": self.config.num_image_tokens,
            "compression_ratio": self.config.compression_ratio,
            "dynamic_resolution": self.config.dynamic_resolution,
        }
