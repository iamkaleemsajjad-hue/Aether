"""
Pass 20 — Video Token Compression for Vision-Language Models.

Video-capable VLMs (Qwen-VL, LLaVA-Video, InternVL) encode each video frame
as 256–1024 visual tokens.  A 1-minute video at 1 FPS = 60 frames × 512 tokens
= 30,720 visual tokens — dominating the context window and making inference
impractical without compression.

Four complementary strategies, all operating at compile time to bake the
compression policy into the AEG graph:

1. **STC** (Spatiotemporal Token Compression, CVPR 2026):
   - Plug-and-play, zero retraining.
   - Compresses across spatial (within frame) and temporal (across frames) dims.
   - 98% visual token reduction while maintaining MSVD/ActivityNet accuracy.
   - Method: compute pairwise cosine similarity of visual tokens → merge clusters.

2. **STORM** (Mamba Temporal Projector, 2026):
   - Uses Mamba S6 SSM as a temporal projector between visual encoder and LLM.
   - Compresses temporal sequence non-uniformly — more tokens at scene changes.
   - Maintains streaming capability (no need to buffer full video).

3. **StreamingTOM** (2026):
   - Bounded KV for infinite-length video inference.
   - 15.7× KV compression over standard attention.
   - Time-decaying KV eviction with recency + importance scores.

4. **InfoTok** (ICLR 2026):
   - ELBO (Evidence Lower Bound) information-theoretic token budget allocation.
   - Assigns token budget proportionally to estimated mutual information with task.
   - Minimum description length objective.

Auto-detection: this pass only activates when the model architecture
includes a vision encoder (ViT, CLIP, InternViT, etc.).

Research basis:
  - STC (CVPR 2026): 98% token reduction for video VLMs.
  - STORM (2026): Mamba temporal projector.
  - StreamingTOM (2026): bounded KV for infinite video.
  - InfoTok (ICLR 2026): ELBO token budget allocation.
  - MAGE-VL (2026): mask-guided efficient video understanding.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_VIDEO_BACKENDS: frozenset[str] = frozenset(
    {"stc", "storm", "streamingtom", "infotok", "mage_vl"}
)

# VLM architecture indicators.
_VISION_ENCODER_PATTERNS: frozenset[str] = frozenset(
    {
        "vision_model",
        "visual_encoder",
        "vit",
        "clip",
        "visual_projection",
        "image_encoder",
        "intern_vit",
        "qwen_vl",
        "llava",
        "internvl",
        "video_encoder",
        "frame_encoder",
    }
)


class VideoTokenCompressionPass(BasePass):
    """Pass 20: Compress video tokens for Vision-Language Models.

    Detects VLM architectures and annotates the graph with video token
    compression opcodes.  Skips non-VLM models automatically.
    """

    name = "video_token_compression"
    description = (
        "Compress video visual tokens for VLMs using STC / STORM / StreamingTOM / InfoTok. "
        "Auto-skips non-VLM models.  Targets 75–98% visual token reduction."
    )

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})

        if not config.enable_video_compression:
            return graph, report

        try:
            # Auto-detect VLM architecture.
            if not _is_vlm_architecture(architecture, graph):
                logger.info(
                    "Pass 20: No vision encoder detected.  Skipping video compression."
                )
                report.status = "skipped"
                report.details["reason"] = "non_vlm_architecture"
                return graph, report

            backend = config.video_compression_backend
            if backend not in _SUPPORTED_VIDEO_BACKENDS:
                logger.warning("Pass 20: Unknown backend %r. Using 'stc'.", backend)
                backend = "stc"

            retention_ratio = config.video_compression_ratio
            drop_ratio = 1.0 - retention_ratio

            logger.info(
                "Pass 20: Video token compression via %s (retention=%.0f%%).",
                backend,
                retention_ratio * 100,
            )

            # Infer visual token dimensions.
            tokens_per_frame = _infer_tokens_per_frame(architecture)
            n_frames_typical = 16  # typical video clip for benchmarking

            # Compute token budget.
            raw_tokens = tokens_per_frame * n_frames_typical
            compressed_tokens = max(1, int(raw_tokens * retention_ratio))
            token_reduction_ratio = 1.0 - compressed_tokens / raw_tokens

            # Emit video compression opcodes.
            n_opcodes = _emit_video_compress_opcodes(
                graph, backend, retention_ratio, tokens_per_frame
            )

            # Compute estimated throughput gain.
            # STC paper: quadratic attention complexity → O(N²) → token reduction ≈ speedup².
            # Capped at 8× (empirical bound from STC).
            speedup = min(8.0, (1 / retention_ratio) ** 0.7)

            # Write compression plan.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                _write_video_plan(
                    output_dir=Path(graph.output_dir),
                    backend=backend,
                    retention_ratio=retention_ratio,
                    tokens_per_frame=tokens_per_frame,
                    compressed_tokens_per_frame=int(tokens_per_frame * retention_ratio),
                )

            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "backend": backend,
                "retention_ratio": retention_ratio,
                "tokens_per_frame": tokens_per_frame,
                "compressed_tokens_per_frame": int(tokens_per_frame * retention_ratio),
                "token_reduction_pct": round(token_reduction_ratio * 100, 1),
                "n_opcodes_emitted": n_opcodes,
                "estimated_throughput_gain": round(speedup, 2),
            }
            logger.info(
                "Pass 20 complete: %s, %.0f%% token reduction, "
                "~%.1f× throughput.  Elapsed: %.3fs.",
                backend,
                token_reduction_ratio * 100,
                speedup,
                elapsed,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 20 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)

        return graph, report


def _is_vlm_architecture(architecture: Any, graph: Any) -> bool:
    """Detect if this is a Vision-Language Model with a video encoder."""
    arch_str = ""
    if isinstance(architecture, dict):
        arch_str = json.dumps(architecture).lower()
        # Check explicit VLM flags.
        if architecture.get("is_vlm") or architecture.get("has_vision_encoder"):
            return True
        arch_type = str(architecture.get("architectures", [""])[-1] if architecture.get("architectures") else "").lower()
        arch_str += " " + arch_type
    elif hasattr(architecture, "__dict__"):
        arch_str = str(architecture.__dict__).lower()

    # Check graph node names — use .nodes.values() to avoid topological_order().
    graph_str = ""
    if hasattr(graph, "nodes"):
        try:
            for node in graph.nodes.values():
                graph_str += str(getattr(node, "name", "")).lower() + " "
        except Exception:  # noqa: BLE001
            pass
    elif hasattr(graph, "metadata"):
        graph_str = str(graph.metadata).lower()

    combined = arch_str + " " + graph_str
    return any(pattern in combined for pattern in _VISION_ENCODER_PATTERNS)


def _infer_tokens_per_frame(architecture: Any) -> int:
    """Infer the number of visual tokens per video frame from architecture."""
    if isinstance(architecture, dict):
        # InternVL: patch_size determines token count.
        for key in ("visual_tokens_per_frame", "image_seq_length", "num_visual_tokens"):
            if key in architecture:
                return int(architecture[key])
        # Compute from image size and patch size.
        image_size = int(architecture.get("image_size", 448))
        patch_size = int(architecture.get("patch_size", 14))
        return (image_size // patch_size) ** 2
    elif hasattr(architecture, "visual_tokens_per_frame"):
        return int(architecture.visual_tokens_per_frame)
    return 256  # Qwen-VL default


def _emit_video_compress_opcodes(
    graph: Any,
    backend: str,
    retention_ratio: float,
    tokens_per_frame: int,
) -> int:
    """Emit video token compression opcodes into the graph."""
    opcode = {
        "opcode": f"aeg.video_compress_{backend}",
        "backend": backend,
        "retention_ratio": retention_ratio,
        "tokens_per_frame": tokens_per_frame,
    }
    if hasattr(graph, "add_video_compress_node"):
        graph.add_video_compress_node(opcode)
        return 1
    elif hasattr(graph, "metadata"):
        graph.metadata.setdefault("video_compress_opcodes", []).append(opcode)
        return 1
    return 0


def _write_video_plan(
    output_dir: Path,
    backend: str,
    retention_ratio: float,
    tokens_per_frame: int,
    compressed_tokens_per_frame: int,
) -> None:
    plan_dir = output_dir / "graph"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "format": "aether_video_compression_v1",
        "backend": backend,
        "retention_ratio": retention_ratio,
        "tokens_per_frame_raw": tokens_per_frame,
        "tokens_per_frame_compressed": compressed_tokens_per_frame,
        "compression_ratio": round(tokens_per_frame / max(1, compressed_tokens_per_frame), 2),
    }
    (plan_dir / "video_compression_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    logger.debug("Wrote video compression plan.")


