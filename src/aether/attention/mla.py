"""Multi-Head Latent Attention planning.

MLA reduces KV-cache footprint by storing compact latent vectors and absorbing
projection weights into compiled attention kernels where the architecture allows
it. This module provides a deterministic planner used by package manifests and
runtime selection tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MLACompressionPlan:
    """Compiled MLA latent-cache configuration."""

    enabled: bool
    latent_dim: int
    original_kv_dim: int
    compression_ratio: float
    weight_absorption: bool
    rope_split: bool
    kernel: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "mla_compression/1.0",
            "enabled": self.enabled,
            "latent_dim": self.latent_dim,
            "original_kv_dim": self.original_kv_dim,
            "compression_ratio": round(self.compression_ratio, 6),
            "estimated_kv_reduction": round(1.0 - min(1.0, 1.0 / max(self.compression_ratio, 1e-9)), 6),
            "weight_absorption": self.weight_absorption,
            "rope_split": self.rope_split,
            "kernel": self.kernel,
        }


class MLAPlanner:
    """Detect MLA-compatible architectures and emit latent-KV plans."""

    MLA_FAMILIES = {"deepseek_family", "kimi_family", "glm_family"}

    def plan(self, architecture: Any, target: str = "cuda_sm90") -> MLACompressionPlan:
        attention_type = str(getattr(architecture, "attention_type", "")).upper()
        family = getattr(architecture, "family", "")
        enabled = attention_type == "MLA" or family in self.MLA_FAMILIES
        num_heads = max(1, int(getattr(architecture, "num_attention_heads", 1)))
        hidden_size = max(1, int(getattr(architecture, "hidden_size", 1)))
        original_kv_dim = max(1, hidden_size // num_heads)
        if enabled:
            latent_dim = max(64, min(1024, original_kv_dim * max(1, int(getattr(architecture, "num_kv_heads", 1)))))
            compression_ratio = max(1.0, (original_kv_dim * num_heads * 2) / latent_dim)
        else:
            latent_dim = original_kv_dim
            compression_ratio = 1.0
        return MLACompressionPlan(
            enabled=enabled,
            latent_dim=latent_dim,
            original_kv_dim=original_kv_dim,
            compression_ratio=compression_ratio,
            weight_absorption=enabled,
            rope_split=enabled,
            kernel=self._kernel(target, enabled),
        )

    def _kernel(self, target: str, enabled: bool) -> str:
        if not enabled:
            return "dense_attention"
        if target in {"cuda_sm100", "cuda_sm120"}:
            return "aeg.mla_flash_attention_4"
        if target.startswith("cuda"):
            return "aeg.mla_flash_attention_3"
        if target.startswith("rocm"):
            return "aeg.mla_aiter_attention"
        return "aeg.mla_portable"
