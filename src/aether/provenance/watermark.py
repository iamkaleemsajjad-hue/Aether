"""SynthID-style statistical output watermarking."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WatermarkResult:
    """Detection result for green-list token watermarking."""

    detected: bool
    z_score: float
    green_tokens: int
    total_tokens: int


class AetherOutputWatermark:
    """Deterministic green-list watermark for token IDs."""

    def __init__(self, vocab_size: int = 32000, green_fraction: float = 0.25, delta: float = 1.0, z_threshold: float = 4.0) -> None:
        self.vocab_size = vocab_size
        self.green_fraction = green_fraction
        self.delta = delta
        self.z_threshold = z_threshold

    def greenlist(self, context_ids: Sequence[int]) -> set[int]:
        context = tuple(context_ids[-16:])
        seed = int(hashlib.sha256(repr(context).encode("utf-8")).hexdigest()[:16], 16)
        size = max(1, int(self.vocab_size * self.green_fraction))
        return {(seed + i * 2654435761) % self.vocab_size for i in range(size)}

    def apply_watermark(self, logits: list[float], context_ids: Sequence[int]) -> list[float]:
        adjusted = list(logits)
        for token_id in self.greenlist(context_ids):
            if token_id < len(adjusted):
                adjusted[token_id] += self.delta
        return adjusted

    def detect_token_ids(self, token_ids: Sequence[int]) -> WatermarkResult:
        if not token_ids:
            return WatermarkResult(False, 0.0, 0, 0)
        green_hits = 0
        for index, token_id in enumerate(token_ids):
            if token_id in self.greenlist(token_ids[:index]):
                green_hits += 1
        expected = len(token_ids) * self.green_fraction
        variance = max(len(token_ids) * self.green_fraction * (1 - self.green_fraction), 1e-6)
        z_score = (green_hits - expected) / math.sqrt(variance)
        return WatermarkResult(z_score >= self.z_threshold, z_score, green_hits, len(token_ids))
