"""
R10 — Video Frame KV Manager.

Video inference over long video clips (minutes to hours) requires efficient
management of the visual token KV cache.  Without specialized management,
a 10-minute video at 1 FPS with 256 tokens/frame = 153,600 visual tokens,
vastly exceeding typical context windows.

R10 implements the runtime counterpart to Pass 20's compile-time compression
plan:

1. **Scene-Adaptive Frame Sampling**: Track scene change scores using
   inter-frame cosine similarity.  High-change frames get more visual tokens
   (fine granularity); low-change (static) frames get fewer.

2. **StreamingTOM KV Eviction**: Time-decaying KV eviction with importance
   scoring.  Importance = recency weight × motion magnitude × task relevance.
   Implements the StreamingTOM bounded KV window (PRD v4.0).

3. **STC Runtime Execution**: Apply the ChunkKV/SentenceKV compression plan
   loaded from ``.aeg/graph/video_compression_plan.json`` at frame-by-frame
   encoding time.

4. **Temporal Attention Routing**: Route different video segments to different
   attention windows:
   - Recent frames: full attention context.
   - Mid-term frames: compressed STC representation.
   - Long-term frames: summary token (single mean-pooled token per scene).

Memory model:
  - ``VideoKVSlot``: A compressed KV representation for one video segment.
  - ``VideoKVManager``: Maintains a priority queue of slots with eviction policy.
  - Maximum memory: ``max_kv_slots × avg_slot_size``.

Research basis:
  - StreamingTOM (2026): bounded KV for infinite video VLMs.
  - STC (CVPR 2026): spatial-temporal token compression.
  - STORM (2026): Mamba-based temporal projector.
  - Video-SALMONN (2024): audio-visual temporal alignment.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VideoKVSlot:
    """A compressed KV slot representing one video segment.

    Attributes:
        slot_id: Unique identifier (e.g., ``frame_{t}``, ``scene_{k}``).
        frame_range: (start_frame, end_frame) inclusive.
        kv_data: Compressed KV representation (STC-compressed or mean-pooled).
        n_tokens: Number of visual tokens retained.
        importance: Priority score for eviction decisions (higher = keep).
        is_scene_boundary: True if this slot starts at a scene cut.
        created_ts: Timestamp of slot creation.
    """

    slot_id: str
    frame_range: tuple[int, int]
    kv_data: Any
    n_tokens: int
    importance: float = 1.0
    is_scene_boundary: bool = False
    created_ts: float = field(default_factory=time.time)


class VideoFrameKVManager:
    """Runtime R10: Video Frame KV Manager.

    Manages the visual token KV cache for video VLMs with scene-adaptive
    compression and time-decaying importance scoring.
    """

    def __init__(
        self,
        max_kv_slots: int = 512,
        tokens_per_frame_raw: int = 256,
        compression_ratio: float = 0.25,  # Retain 25% of visual tokens.
        scene_change_threshold: float = 0.3,
        decay_rate: float = 0.01,  # Importance decay per second.
        compression_plan_path: str | None = None,
    ) -> None:
        self.max_kv_slots = max_kv_slots
        self.tokens_per_frame_raw = tokens_per_frame_raw
        self.tokens_per_frame_compressed = max(1, int(tokens_per_frame_raw * compression_ratio))
        self.scene_change_threshold = scene_change_threshold
        self.decay_rate = decay_rate

        self._slots: dict[str, VideoKVSlot] = {}
        self._frame_count: int = 0
        self._scene_count: int = 0
        self._lock = threading.RLock()
        self._stats = _VideoKVStats()

        # Load compression plan from AEG artifact.
        self._plan: dict[str, Any] = {}
        if compression_plan_path:
            self._load_plan(compression_plan_path)

    def _load_plan(self, path: str) -> None:
        """Load video compression plan from AEG artifact."""
        p = Path(path)
        if not p.exists():
            return
        try:
            self._plan = json.loads(p.read_text(encoding="utf-8"))
            self.tokens_per_frame_raw = int(
                self._plan.get("tokens_per_frame_raw", self.tokens_per_frame_raw)
            )
            self.tokens_per_frame_compressed = int(
                self._plan.get("tokens_per_frame_compressed", self.tokens_per_frame_compressed)
            )
            logger.info(
                "R10: Video compression plan loaded — %d→%d tokens/frame (%.0f%% reduction).",
                self.tokens_per_frame_raw,
                self.tokens_per_frame_compressed,
                (1 - self.tokens_per_frame_compressed / self.tokens_per_frame_raw) * 100,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("R10: Failed to load video compression plan: %s", exc)

    def ingest_frame(
        self,
        frame_idx: int,
        frame_tokens: list[Any],
        prev_frame_tokens: list[Any] | None = None,
        task_relevance: float = 1.0,
    ) -> VideoKVSlot:
        """Ingest a new video frame and create/update a KV slot.

        Args:
            frame_idx: Global frame index.
            frame_tokens: Visual tokens for this frame (pre-encoded by ViT).
            prev_frame_tokens: Tokens from the previous frame (for scene change detection).
            task_relevance: Task-specific relevance score for this frame (0-1).

        Returns:
            Created VideoKVSlot.
        """
        # Scene change detection.
        is_scene_boundary = False
        motion_score = 0.0
        if prev_frame_tokens and frame_tokens:
            sim = _cosine_similarity_mean(frame_tokens, prev_frame_tokens)
            motion_score = 1.0 - max(0.0, sim)
            is_scene_boundary = motion_score > self.scene_change_threshold
            if is_scene_boundary:
                self._scene_count += 1

        # Compress frame tokens using STC chunking.
        compressed_kv = self._stc_compress_frame(
            frame_tokens,
            is_scene_boundary=is_scene_boundary,
        )

        # Compute importance score.
        recency = 1.0  # New frames have maximum recency.
        importance = recency * (0.5 + 0.5 * motion_score) * task_relevance
        if is_scene_boundary:
            importance = min(1.0, importance * 1.5)  # Boost boundary frames.

        slot_id = f"frame_{frame_idx:06d}"
        slot = VideoKVSlot(
            slot_id=slot_id,
            frame_range=(frame_idx, frame_idx),
            kv_data=compressed_kv,
            n_tokens=len(compressed_kv) if isinstance(compressed_kv, list) else self.tokens_per_frame_compressed,
            importance=importance,
            is_scene_boundary=is_scene_boundary,
        )

        with self._lock:
            self._slots[slot_id] = slot
            self._frame_count += 1
            self._stats.frames_ingested += 1

            # Evict if over capacity.
            if len(self._slots) > self.max_kv_slots:
                self._evict_least_important()

        return slot

    def _stc_compress_frame(
        self,
        frame_tokens: list[Any],
        is_scene_boundary: bool,
    ) -> list[Any]:
        """Apply STC (Spatial-Temporal Compression) to frame tokens.

        For scene boundaries: retain tokens_per_frame_raw (no compression).
        For non-boundary: compress to tokens_per_frame_compressed via chunking.
        """
        if is_scene_boundary:
            # Scene boundary: keep all tokens for this frame.
            return list(frame_tokens)

        n = len(frame_tokens)
        target = self.tokens_per_frame_compressed
        if n <= target:
            return list(frame_tokens)

        # Stride-based sampling: evenly distributed across the frame.
        stride = max(1, n // target)
        return [frame_tokens[i] for i in range(0, n, stride)][:target]

    def get_attention_context(
        self,
        query_frame_idx: int,
        recent_window: int = 8,
    ) -> dict[str, list[Any]]:
        """Build the attention context for a query frame.

        Returns a dict with three tiers:
          - 'recent': Full KV for the most recent ``recent_window`` frames.
          - 'mid_term': STC-compressed KV for older frames in the same scene.
          - 'summary': Mean-pooled summary tokens for very old frames.
        """
        with self._lock:
            all_slots = sorted(self._slots.values(), key=lambda s: s.frame_range[0])

        recent_cutoff = query_frame_idx - recent_window
        scene_cutoff = query_frame_idx - recent_window * 4

        recent: list[Any] = []
        mid_term: list[Any] = []
        summary: list[Any] = []

        for slot in all_slots:
            frame_start = slot.frame_range[0]
            if frame_start >= recent_cutoff:
                recent.extend(slot.kv_data if isinstance(slot.kv_data, list) else [slot.kv_data])
            elif frame_start >= scene_cutoff:
                mid_term.extend(slot.kv_data if isinstance(slot.kv_data, list) else [slot.kv_data])
            else:
                # Summary: mean pool.
                if slot.kv_data:
                    summary.append(_mean_pool(slot.kv_data))

        return {"recent": recent, "mid_term": mid_term, "summary": summary}

    def decay_importance(self) -> None:
        """Apply time-based importance decay to all slots.

        Called periodically (e.g., after each generated token) to implement
        StreamingTOM's time-decaying importance scoring.
        """
        now = time.time()
        with self._lock:
            for slot in self._slots.values():
                age_s = now - slot.created_ts
                slot.importance = max(
                    0.01,  # Floor importance.
                    slot.importance * math.exp(-self.decay_rate * age_s),
                )

    def _evict_least_important(self) -> None:
        """Evict the slot with lowest importance score (StreamingTOM policy)."""
        if not self._slots:
            return

        # Never evict scene boundary frames unless necessary.
        candidates = [
            (s.importance, sid)
            for sid, s in self._slots.items()
            if not s.is_scene_boundary
        ]
        if not candidates:
            # All boundary frames: evict oldest.
            candidates = [(s.importance, sid) for sid, s in self._slots.items()]

        candidates.sort()
        _, evict_id = candidates[0]
        del self._slots[evict_id]
        self._stats.evictions += 1

    def total_visual_tokens(self) -> int:
        """Return total compressed visual tokens in the KV manager."""
        with self._lock:
            return sum(s.n_tokens for s in self._slots.values())

    @property
    def stats(self) -> "_VideoKVStats":
        return self._stats

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_frames_ingested": self._stats.frames_ingested,
                "active_kv_slots": len(self._slots),
                "scene_boundaries_detected": self._scene_count,
                "total_visual_tokens": self.total_visual_tokens(),
                "evictions": self._stats.evictions,
                "tokens_per_frame_compressed": self.tokens_per_frame_compressed,
            }


class _VideoKVStats:
    __slots__ = ("frames_ingested", "evictions")

    def __init__(self) -> None:
        self.frames_ingested = 0
        self.evictions = 0


def _cosine_similarity_mean(a: list[Any], b: list[Any]) -> float:
    """Compute mean cosine similarity between two lists of token vectors."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    total_sim = 0.0
    for i in range(n):
        va = a[i] if isinstance(a[i], list) else [float(a[i])]
        vb = b[i] if isinstance(b[i], list) else [float(b[i])]
        total_sim += _cos_sim(va, vb)
    return total_sim / n


def _cos_sim(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _mean_pool(tokens: list[Any]) -> Any:
    """Mean-pool a list of tokens into a single summary token."""
    if not tokens:
        return None
    if isinstance(tokens[0], (int, float)):
        return sum(tokens) / len(tokens)
    elif isinstance(tokens[0], list):
        dim = len(tokens[0])
        return [sum(t[j] if j < len(t) else 0.0 for t in tokens) / len(tokens) for j in range(dim)]
    return tokens[0]
