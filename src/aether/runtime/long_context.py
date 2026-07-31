"""Long-Context Engine for Aether Runtime (Section 28 — v3.1 Elite Extensions).

Implements salience-aware KV eviction, ring attention planning, context parallelism
topology generation, and YaRN RoPE extension for 1M+ token native support.

Research:
- MInference (Microsoft, NeurIPS 2024) — sparse attention for 1M tokens
- StreamingLLM (2023) — anchor token retention + sliding window
- ScissorHands (2024) — heavy-hitter KV selection
- SnapKV (2025) — important-token KV selection
- Ring Attention (2023) — sequence parallelism ring topology
- Striped Attention (2023) — load-balanced ring
- YaRN (2023) — RoPE extension to 128K–1M tokens
- LongRoPE (2024) — 2M context RoPE scaling
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# KV salience scoring (StreamingLLM + ScissorHands + SnapKV fusion)
# ---------------------------------------------------------------------------

@dataclass
class KVBlock:
    """A block of key-value cache entries."""

    request_id: str
    layer_idx: int
    start_token: int
    end_token: int
    is_anchor: bool = False       # First ~4 tokens — always retained (StreamingLLM)
    attention_mass: float = 0.0   # Cumulative attention weight received

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_idx": self.layer_idx,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "is_anchor": self.is_anchor,
            "attention_mass": round(self.attention_mass, 6),
            "token_count": self.token_count,
        }


class SalienceKVEvictor:
    """
    Salience-aware KV cache evictor for long-context inference.

    Combines three eviction signals:
      1. Recency   (StreamingLLM, 2023) — recent tokens are more likely needed
      2. Attention (ScissorHands, 2024) — high-attention tokens are heavy hitters
      3. Anchor    (StreamingLLM, 2023) — first N tokens always retained (sink tokens)

    Score formula: 0.5 * attention_score + 0.3 * recency_score + 0.2 * anchor_score

    Eviction policy: blocks with lowest salience score are moved to CPU DRAM (L2)
    first, then to NVMe (L3) if L2 is full, then evicted (L4 = Hub CDN prefetch).

    Research: StreamingLLM (2023), ScissorHands (2024), SnapKV (2025).
    """

    TIER_LABELS = ["L1_GPU", "L2_CPU", "L3_NVME", "L4_EVICTED"]

    def __init__(
        self,
        window_size: int = 2048,
        anchor_tokens: int = 4,
        recency_weight: float = 0.3,
        attention_weight: float = 0.5,
        anchor_weight: float = 0.2,
    ) -> None:
        self.window_size = window_size
        self.anchor_tokens = anchor_tokens
        self.recency_weight = recency_weight
        self.attention_weight = attention_weight
        self.anchor_weight = anchor_weight

    def score_salience(
        self,
        block: KVBlock,
        current_seq_len: int,
        attention_weights: np.ndarray | None = None,
    ) -> float:
        """
        Compute salience score for a KV block.

        Args:
            block: KV block to score.
            current_seq_len: Current total sequence length (for recency computation).
            attention_weights: Optional (seq_len,) attention weight array for this layer.

        Returns:
            Salience score in [0, 1]. Higher = more important = keep in GPU.
        """
        # Anchor score: sink tokens (first N tokens) always maximally important
        anchor_score = 1.0 if block.is_anchor else 0.0

        # Recency score: linear decay from current position
        distance = current_seq_len - block.end_token
        recency_score = max(0.0, 1.0 - distance / max(self.window_size, 1))

        # Attention score: mean attention weight received by this block's tokens
        if attention_weights is not None and len(attention_weights) > block.start_token:
            block_weights = attention_weights[block.start_token:block.end_token]
            attention_score = float(block_weights.mean()) if len(block_weights) > 0 else 0.0
        else:
            attention_score = block.attention_mass  # Fall back to accumulated mass

        salience = (
            self.attention_weight * attention_score
            + self.recency_weight * recency_score
            + self.anchor_weight * anchor_score
        )
        return round(min(1.0, salience), 6)

    def eviction_order(
        self,
        blocks: list[KVBlock],
        current_seq_len: int,
        attention_weights: np.ndarray | None = None,
    ) -> list[tuple[KVBlock, float, str]]:
        """
        Return blocks sorted from least to most salient, with their tier assignment.

        Returns:
            List of (block, salience_score, tier_label) sorted ascending by salience.
        """
        scored = [
            (block, self.score_salience(block, current_seq_len, attention_weights))
            for block in blocks
        ]
        scored.sort(key=lambda x: x[1])  # lowest salience first (evict these)

        result = []
        for i, (block, score) in enumerate(scored):
            # Assign tier based on rank: top 25% stay in GPU, next 25% CPU, etc.
            rank_pct = i / max(len(scored) - 1, 1)
            if rank_pct < 0.25:
                tier = "L4_EVICTED"
            elif rank_pct < 0.50:
                tier = "L3_NVME"
            elif rank_pct < 0.75:
                tier = "L2_CPU"
            else:
                tier = "L1_GPU"
            result.append((block, score, tier))
        return result

    def config_dict(self) -> dict[str, Any]:
        return {
            "evictor": "salience_aware",
            "window_size": self.window_size,
            "anchor_tokens": self.anchor_tokens,
            "weights": {
                "recency": self.recency_weight,
                "attention": self.attention_weight,
                "anchor": self.anchor_weight,
            },
            "tiers": self.TIER_LABELS,
            "research": ["StreamingLLM:2023", "ScissorHands:2024", "SnapKV:2025"],
        }


# ---------------------------------------------------------------------------
# Ring Attention / Context Parallelism planner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextParallelismPlan:
    """Context parallelism topology for long-sequence ring attention."""

    num_gpus: int
    total_tokens: int
    tokens_per_gpu: int
    topology: str  # "ring" | "striped" | "ulysses_ring"
    ring_size: int
    pipeline_stages: int = 1
    all_reduce_ops_per_layer: int = 0

    @property
    def max_context_tokens(self) -> int:
        return self.tokens_per_gpu * self.num_gpus

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "context_parallelism/1.0",
            "num_gpus": self.num_gpus,
            "total_tokens": self.total_tokens,
            "tokens_per_gpu": self.tokens_per_gpu,
            "topology": self.topology,
            "ring_size": self.ring_size,
            "pipeline_stages": self.pipeline_stages,
            "all_reduce_ops_per_layer": self.all_reduce_ops_per_layer,
            "max_context_tokens": self.max_context_tokens,
            "research": ["RingAttention:2023", "StripedAttention:2023", "Ulysses:2023"],
        }


class RingAttentionPlanner:
    """
    Plans ring-topology context parallelism for sequences exceeding single-GPU memory.

    Supported topologies:
    - ring: Each GPU holds 1/N of tokens; KV blocks passed in ring
    - striped: Interleaved token distribution for better load balance (Striped Attention)
    - ulysses_ring: Hybrid Ulysses (head dim TP) + Ring (seq CP) for 128+ GPUs

    Research: Ring Attention (2023), Striped Attention (2023), DeepSpeed-Ulysses (2023).
    """

    # Approximate GPU memory available for KV cache (per GPU, in tokens)
    GPU_KV_CAPACITY: dict[str, int] = {
        "cuda_sm100": 1_000_000,   # B200 192GB HBM3e
        "cuda_sm90":    500_000,   # H100 80GB HBM3
        "cuda_sm89":    250_000,   # RTX 4090 24GB GDDR6X
        "rocm_gfx942":  800_000,   # MI300X 192GB HBM3
        "metal_m4":     200_000,   # Apple M4 48GB unified
        "cpu_avx512":   100_000,   # CPU DRAM
    }

    def plan(
        self,
        total_tokens: int,
        num_gpus: int,
        target: str = "cuda_sm90",
        prefer_striped: bool = True,
    ) -> ContextParallelismPlan:
        """
        Generate the optimal context parallelism plan.

        Args:
            total_tokens: Total sequence length to process.
            num_gpus: Number of GPUs in the ring.
            target: Hardware target identifier.
            prefer_striped: Use striped token distribution (better load balance).

        Returns:
            ContextParallelismPlan with topology and per-GPU token assignment.
        """
        tokens_per_gpu = math.ceil(total_tokens / num_gpus)
        single_gpu_capacity = self.GPU_KV_CAPACITY.get(target, 250_000)

        # Choose topology based on GPU count and token distribution
        if num_gpus >= 128:
            topology = "ulysses_ring"
            all_reduce_ops = 2  # Ulysses adds 2 all-reduce ops per layer
        elif prefer_striped and num_gpus > 1:
            topology = "striped"
            all_reduce_ops = 0
        else:
            topology = "ring"
            all_reduce_ops = 0

        # Validate that tokens fit per GPU
        if tokens_per_gpu > single_gpu_capacity:
            # Need more GPUs — compute minimum required
            min_gpus_needed = math.ceil(total_tokens / single_gpu_capacity)
            raise ValueError(
                f"Insufficient GPUs for {total_tokens:,} tokens on {target}. "
                f"Need at least {min_gpus_needed} GPUs (have {num_gpus})."
            )

        return ContextParallelismPlan(
            num_gpus=num_gpus,
            total_tokens=total_tokens,
            tokens_per_gpu=tokens_per_gpu,
            topology=topology,
            ring_size=num_gpus,
            pipeline_stages=1,
            all_reduce_ops_per_layer=all_reduce_ops,
        )

    def plans_for_context_sizes(
        self, target: str = "cuda_sm90"
    ) -> dict[str, dict[str, Any]]:
        """Generate standard multi-GPU plans for common context sizes."""
        plans = {}
        context_sizes = [
            (128_000,   1, "128k"),
            (500_000,   4, "500k_4gpu"),
            (1_000_000, 4, "1m_4gpu"),
            (4_000_000, 32, "4m_32gpu"),
        ]
        for tokens, gpus, label in context_sizes:
            try:
                plan = self.plan(tokens, gpus, target=target)
                plans[label] = plan.to_dict()
            except ValueError:
                pass
        return plans

    def write_plans(self, aeg_dir: str | Path, target: str = "cuda_sm90") -> list[Path]:
        """Write all standard parallelism plans to .aeg/parallelism/."""
        para_dir = Path(aeg_dir) / "parallelism"
        para_dir.mkdir(parents=True, exist_ok=True)
        written = []

        planner_plans = [
            (1,   131_072,   "1gpu"),
            (2,   262_144,   "2gpu"),
            (4,   1_000_000, "4gpu_1m"),
            (8,   2_000_000, "8gpu_2m"),
            (32,  4_000_000, "32gpu_cp"),
        ]

        for gpus, tokens, label in planner_plans:
            try:
                plan = self.plan(tokens, gpus, target=target)
                path = para_dir / f"{label}.json"
                path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
                written.append(path)
            except ValueError:
                pass
        return written


# ---------------------------------------------------------------------------
# YaRN RoPE extension configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YaRNConfig:
    """
    YaRN (Yet another RoPE extensioN) configuration for long-context scaling.

    Extends the original RoPE position embeddings to support context lengths
    far beyond the training context without significant quality degradation.

    Research: YaRN (2023) — supports 128K; LongRoPE (2024) — supports 2M tokens.
    """

    original_max_position: int = 4096      # Model's training context length
    target_max_position: int = 131072      # Desired extended context length
    rope_theta: float = 10000.0            # Original RoPE theta
    yarn_alpha: float = 1.0               # Linear interpolation factor
    yarn_beta: float = 32.0               # NTK-aware scaling correction
    method: str = "yarn"                   # "yarn" | "longrope" | "dynamic_ntk"

    @property
    def scale_factor(self) -> float:
        """Context extension ratio."""
        return self.target_max_position / self.original_max_position

    @property
    def extended_theta(self) -> float:
        """Extended RoPE theta for the new context length."""
        if self.method == "dynamic_ntk":
            # NTK-aware scaling: theta * scale^(d/(d-2))
            # d is head_dim — approximate with 128 (typical)
            d = 128
            return self.rope_theta * (self.scale_factor ** (d / (d - 2)))
        elif self.method == "longrope":
            # LongRoPE uses non-uniform frequency scaling
            return self.rope_theta * self.scale_factor
        else:
            # YaRN: frequency-interpolated with alpha/beta correction
            return self.rope_theta * self.scale_factor * self.yarn_beta

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "original_max_position": self.original_max_position,
            "target_max_position": self.target_max_position,
            "scale_factor": round(self.scale_factor, 4),
            "rope_theta_original": self.rope_theta,
            "rope_theta_extended": round(self.extended_theta, 2),
            "yarn_alpha": self.yarn_alpha,
            "yarn_beta": self.yarn_beta,
            "research": {
                "yarn": "YaRN 2023 — 128K context",
                "longrope": "LongRoPE 2024 — 2M context",
                "dynamic_ntk": "Dynamic NTK-aware scaling 2023",
            }[self.method],
        }


# ---------------------------------------------------------------------------
# Long-context AEG profile writer
# ---------------------------------------------------------------------------

class LongContextProfile:
    """
    Writes the `long_context_profile` section into the AEG manifest.

    Output in manifest.json:
    {
      "long_context_profile": {
        "max_context_tokens": 1000000,
        "rope_extension_method": "yarn",
        "sparse_attention_enabled": true,
        "ring_attention_enabled": true,
        "kv_eviction_policy": "salience",
        ...
      }
    }
    """

    def __init__(
        self,
        max_context_tokens: int = 1_000_000,
        rope_config: YaRNConfig | None = None,
        evictor: SalienceKVEvictor | None = None,
        planner: RingAttentionPlanner | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.rope_config = rope_config or YaRNConfig(
            original_max_position=4096,
            target_max_position=max_context_tokens,
        )
        self.evictor = evictor or SalienceKVEvictor()
        self.planner = planner or RingAttentionPlanner()

    def to_dict(self) -> dict[str, Any]:
        min_gpus = math.ceil(
            self.max_context_tokens
            / RingAttentionPlanner.GPU_KV_CAPACITY.get("cuda_sm90", 500_000)
        )
        return {
            "max_context_tokens": self.max_context_tokens,
            "rope_extension": self.rope_config.to_dict(),
            "sparse_attention_enabled": True,
            "sparse_attention_pass": "pass8_minference",
            "ring_attention_enabled": self.max_context_tokens > 500_000,
            "min_gpus_for_max_context": max(1, min_gpus),
            "kv_eviction_policy": "salience",
            "kv_eviction_config": self.evictor.config_dict(),
            "supported_context_lengths": [
                4096, 8192, 32768, 65536, 131072, 500000, 1000000
            ],
        }

    def write_to_manifest(self, aeg_dir: str | Path) -> Path:
        manifest_path = Path(aeg_dir) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
        manifest["long_context_profile"] = self.to_dict()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path
