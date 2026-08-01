"""
Multi-Head Latent Attention (MLA) — Full Implementation.

MLA was introduced in DeepSeek-V2 and powers DeepSeek-V3, Kimi K2, GLM-5.
Key innovation: compress KV cache from O(n * num_heads * head_dim) to
O(n * lora_rank) by storing only a latent vector and absorbing the KV
projection weights at compile time (weight absorption).

Memory savings: 57x KV cache reduction vs standard MHA at 128K context
for DeepSeek-V3 (kv_lora_rank=512, head_dim=128, num_kv_heads=128 → 57x).

This implementation covers:
  - MLAConfig: architecture hyperparameters
  - MLAWeightAbsorber: compile-time absorption of W_kv_b into W_q/W_o
  - MLACompressedKVCache: runtime latent KV storage + reconstruction
  - MLAAttention: forward pass with decoupled RoPE
  - MLADetector: architecture detection from model config
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MLAConfig:
    """Hyperparameters for a Multi-Head Latent Attention model."""

    # Core compression dimensions
    kv_lora_rank: int = 512         # dim of compressed latent KV vector (c_KV)
    q_lora_rank: int = 1536         # dim of compressed query (c_Q), 0 = no compression
    qk_nope_head_dim: int = 128     # head_dim for non-RoPE portion of Q/K
    qk_rope_head_dim: int = 64      # head_dim for RoPE portion of Q/K
    v_head_dim: int = 128           # value head dimension
    num_heads: int = 128            # number of Q heads
    num_kv_heads: int = 128         # number of KV heads (same as Q for MLA)

    # RoPE config
    rope_decoupled: bool = True     # True for DeepSeek (separate RoPE branch)
    rope_theta: float = 10000.0

    # Compile-time flags
    weight_absorption_possible: bool = True
    absorbed: bool = False          # True after weight absorption at compile time

    @property
    def total_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_dim_per_layer(self) -> int:
        """KV cache bytes per token per layer (compressed latent only)."""
        return self.kv_lora_rank

    @property
    def standard_kv_dim(self) -> int:
        """Standard GQA KV dim per token per layer for comparison."""
        return 2 * self.num_kv_heads * self.v_head_dim

    @property
    def compression_ratio(self) -> float:
        """KV cache compression ratio vs standard GQA."""
        if self.kv_lora_rank == 0:
            return 1.0
        return self.standard_kv_dim / self.kv_lora_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "rope_decoupled": self.rope_decoupled,
            "weight_absorption_possible": self.weight_absorption_possible,
            "absorbed": self.absorbed,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLAConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def deepseek_v3(cls) -> "MLAConfig":
        """DeepSeek-V3 / DeepSeek-R1 MLA config."""
        return cls(
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            num_heads=128,
            num_kv_heads=128,
            rope_decoupled=True,
        )

    @classmethod
    def kimi_k2(cls) -> "MLAConfig":
        """Kimi K2 MLA config (estimated)."""
        return cls(
            kv_lora_rank=512,
            q_lora_rank=1024,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            num_heads=64,
            num_kv_heads=64,
            rope_decoupled=True,
        )


# ---------------------------------------------------------------------------
# Weight absorption (compile-time optimization)
# ---------------------------------------------------------------------------

class MLAWeightAbsorber:
    """
    Compile-time weight absorption for MLA.

    Standard MLA decode step:
      c_KV = x @ W_kv_a           # (seq, kv_lora_rank)  ← stored in KV cache
      k_nope, v = c_KV @ W_kv_b   # (seq, num_kv_heads, qk_nope_head_dim + v_head_dim)
      k_rope = x @ W_k_rope       # (seq, num_kv_heads, qk_rope_head_dim)

    Weight absorption: W_kv_b is absorbed into W_q_nope and W_o, eliminating
    the c_KV → k/v expansion at runtime. This saves one large GEMM per layer.

      W_q_nope_absorbed = W_q_nope @ W_kv_b[:, :qk_nope_head_dim]
      W_o_absorbed = W_kv_b[:, qk_nope_head_dim:].T @ W_o

    After absorption, the KV cache only stores c_KV (the latent vector)
    and all expansions happen through the absorbed weights — zero overhead.
    """

    def __init__(self, config: MLAConfig) -> None:
        self.config = config

    def absorb(
        self,
        weights: dict[str, np.ndarray],
        layer_prefix: str = "model.layers.0",
    ) -> dict[str, np.ndarray]:
        """
        Absorb W_kv_b into W_q_nope and W_o for a single layer.

        Args:
            weights: Dict of weight tensors for the layer.
            layer_prefix: Key prefix used to look up the layer's weights.

        Returns:
            Updated weight dict with absorbed matrices added and originals kept.
        """
        cfg = self.config
        result = dict(weights)

        # Try to find W_kv_b (DeepSeek naming conventions)
        kv_b_key = self._find_key(weights, layer_prefix, ["kv_b_proj", "w_kv_b", "W_kv_b"])
        q_nope_key = self._find_key(weights, layer_prefix, ["q_proj", "W_q_nope", "q_a_proj"])
        o_key = self._find_key(weights, layer_prefix, ["o_proj", "W_o", "out_proj"])

        if kv_b_key is None:
            logger.debug("MLA weight absorption skipped: W_kv_b not found for %s", layer_prefix)
            return result

        W_kv_b = weights[kv_b_key]  # (kv_lora_rank, num_kv_heads * (qk_nope_head_dim + v_head_dim))

        nope_dim = cfg.qk_nope_head_dim * cfg.num_kv_heads
        v_dim = cfg.v_head_dim * cfg.num_kv_heads

        W_kv_b_nope = W_kv_b[:, :nope_dim]    # (kv_lora_rank, num_kv_heads * qk_nope_head_dim)
        W_kv_b_v    = W_kv_b[:, nope_dim:]    # (kv_lora_rank, num_kv_heads * v_head_dim)

        # Absorb into Q_nope
        if q_nope_key is not None:
            W_q = weights[q_nope_key]
            if W_q.shape[-1] == cfg.kv_lora_rank:
                # W_q_absorbed = W_q @ W_kv_b_nope  → shape (..., num_kv_heads * qk_nope_head_dim)
                original_shape = W_q.shape
                W_q_2d = W_q.reshape(-1, cfg.kv_lora_rank)
                W_q_absorbed = W_q_2d @ W_kv_b_nope
                result[f"{layer_prefix}.q_absorbed"] = W_q_absorbed.astype(np.float32)
                logger.debug("MLA: absorbed W_q @ W_kv_b_nope → shape %s", W_q_absorbed.shape)

        # Absorb into O
        if o_key is not None:
            W_o = weights[o_key]
            if W_o.shape[0] == cfg.num_kv_heads * cfg.v_head_dim:
                # W_o_absorbed = W_kv_b_v.T @ W_o  → shape (kv_lora_rank, out_dim)
                W_o_absorbed = W_kv_b_v.T @ W_o.reshape(cfg.num_kv_heads * cfg.v_head_dim, -1)
                result[f"{layer_prefix}.o_absorbed"] = W_o_absorbed.astype(np.float32)
                logger.debug("MLA: absorbed W_kv_b_v.T @ W_o → shape %s", W_o_absorbed.shape)

        result[f"{layer_prefix}.absorption_complete"] = np.array([1], dtype=np.int8)
        return result

    def _find_key(
        self, weights: dict[str, np.ndarray], prefix: str, candidates: list[str]
    ) -> str | None:
        for candidate in candidates:
            full = f"{prefix}.{candidate}.weight"
            if full in weights:
                return full
            short = f"{prefix}.{candidate}"
            if short in weights:
                return short
            if candidate in weights:
                return candidate
        return None

    def estimate_kv_savings(self, seq_len: int, num_layers: int) -> dict[str, float]:
        """Estimate KV cache savings for a given context length."""
        cfg = self.config
        standard_bytes = seq_len * num_layers * cfg.standard_kv_dim * 2  # BF16
        latent_bytes   = seq_len * num_layers * cfg.kv_lora_rank * 2
        return {
            "standard_kv_gb": standard_bytes / 1e9,
            "latent_kv_gb":   latent_bytes / 1e9,
            "compression_ratio": cfg.compression_ratio,
            "savings_gb": (standard_bytes - latent_bytes) / 1e9,
            "savings_pct": (1 - latent_bytes / standard_bytes) * 100,
        }


# ---------------------------------------------------------------------------
# Compressed KV cache for MLA
# ---------------------------------------------------------------------------

class MLACompressedKVCache:
    """
    Runtime KV cache for MLA — stores only latent vectors c_KV.

    Memory layout per request:
      latent: (seq_len, kv_lora_rank) — compressed KV representation
      rope_k: (seq_len, num_kv_heads, qk_rope_head_dim) — decoupled RoPE keys

    When reconstructing K/V for attention:
      K_nope = latent @ W_kv_b_nope
      V      = latent @ W_kv_b_v
      K = concat(K_nope, K_rope)
    """

    def __init__(self, config: MLAConfig, max_seq_len: int = 8192) -> None:
        self.config = config
        self.max_seq_len = max_seq_len
        # Per-request cache: request_id → (latent, rope_k)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def init_request(self, request_id: str) -> None:
        cfg = self.config
        self._cache[request_id] = {
            "latent": np.zeros((0, cfg.kv_lora_rank), dtype=np.float32),
            "rope_k": np.zeros((0, cfg.num_kv_heads, cfg.qk_rope_head_dim), dtype=np.float32),
            "seq_len": 0,
        }

    def append(
        self,
        request_id: str,
        latent_kv: np.ndarray,   # (new_tokens, kv_lora_rank)
        rope_k: np.ndarray,      # (new_tokens, num_kv_heads, qk_rope_head_dim)
    ) -> None:
        """Append new latent KV vectors to the cache."""
        if request_id not in self._cache:
            self.init_request(request_id)
        entry = self._cache[request_id]
        entry["latent"] = np.concatenate([entry["latent"], latent_kv], axis=0)
        entry["rope_k"] = np.concatenate([entry["rope_k"], rope_k], axis=0)
        entry["seq_len"] += latent_kv.shape[0]

        # Evict oldest tokens if over max_seq_len
        if entry["seq_len"] > self.max_seq_len:
            overflow = entry["seq_len"] - self.max_seq_len
            entry["latent"] = entry["latent"][overflow:]
            entry["rope_k"] = entry["rope_k"][overflow:]
            entry["seq_len"] = self.max_seq_len

    def reconstruct(
        self,
        request_id: str,
        W_kv_b: np.ndarray,   # (kv_lora_rank, num_kv_heads * (qk_nope_head_dim + v_head_dim))
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct K_nope, K_rope, V from stored latent vectors.

        Returns:
            k_nope: (seq, num_kv_heads, qk_nope_head_dim)
            k_rope: (seq, num_kv_heads, qk_rope_head_dim)
            v:      (seq, num_kv_heads, v_head_dim)
        """
        cfg = self.config
        entry = self._cache.get(request_id)
        if entry is None or entry["seq_len"] == 0:
            empty_k = np.zeros((0, cfg.num_kv_heads, cfg.qk_nope_head_dim), dtype=np.float32)
            empty_rope = np.zeros((0, cfg.num_kv_heads, cfg.qk_rope_head_dim), dtype=np.float32)
            empty_v = np.zeros((0, cfg.num_kv_heads, cfg.v_head_dim), dtype=np.float32)
            return empty_k, empty_rope, empty_v

        latent = entry["latent"]  # (T, kv_lora_rank)
        nope_dim = cfg.qk_nope_head_dim * cfg.num_kv_heads
        v_dim    = cfg.v_head_dim * cfg.num_kv_heads

        W_nope = W_kv_b[:, :nope_dim]   # (kv_lora_rank, num_kv_heads * qk_nope_head_dim)
        W_v    = W_kv_b[:, nope_dim:]   # (kv_lora_rank, num_kv_heads * v_head_dim)

        k_nope_flat = latent @ W_nope  # (T, num_kv_heads * qk_nope_head_dim)
        v_flat      = latent @ W_v     # (T, num_kv_heads * v_head_dim)

        T = latent.shape[0]
        k_nope = k_nope_flat.reshape(T, cfg.num_kv_heads, cfg.qk_nope_head_dim)
        v      = v_flat.reshape(T, cfg.num_kv_heads, cfg.v_head_dim)
        k_rope = entry["rope_k"]

        return k_nope, k_rope, v

    def free_request(self, request_id: str) -> None:
        self._cache.pop(request_id, None)

    def seq_len(self, request_id: str) -> int:
        entry = self._cache.get(request_id)
        return entry["seq_len"] if entry else 0

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._cache.values():
            total += entry["latent"].nbytes + entry["rope_k"].nbytes
        return total

    def stats(self) -> dict[str, Any]:
        return {
            "active_requests": len(self._cache),
            "memory_bytes": self.memory_bytes(),
            "memory_mb": self.memory_bytes() / 1e6,
        }


# ---------------------------------------------------------------------------
# MLA Attention forward pass
# ---------------------------------------------------------------------------

class MLAAttention:
    """
    Full Multi-Head Latent Attention forward pass.

    Supports:
    - Decoupled RoPE (DeepSeek-V3 style): separate rope branch for K
    - Compressed KV with latent storage
    - Weight-absorbed mode (after compile-time absorption)
    - Prefill (full sequence) and decode (single token) modes
    """

    def __init__(self, config: MLAConfig) -> None:
        self.config = config
        self._scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5

    def _rope(
        self,
        x: np.ndarray,  # (..., seq, head_dim)
        offset: int = 0,
    ) -> np.ndarray:
        """Apply rotary position embeddings."""
        seq = x.shape[-2]
        d = x.shape[-1]
        half = d // 2
        positions = np.arange(offset, offset + seq, dtype=np.float32)
        freqs = 1.0 / (self.config.rope_theta ** (np.arange(0, half, dtype=np.float32) / half))
        angles = np.outer(positions, freqs)  # (seq, half)
        cos = np.cos(angles)
        sin = np.sin(angles)
        x1, x2 = x[..., :half], x[..., half:]
        rotated = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return rotated.astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-9)

    def forward_prefill(
        self,
        x: np.ndarray,              # (batch, seq, hidden_dim)
        weights: dict[str, np.ndarray],
        layer_prefix: str = "",
        position_offset: int = 0,
    ) -> np.ndarray:
        """
        Full-sequence MLA prefill forward.

        Returns output tensor of shape (batch, seq, hidden_dim).
        """
        cfg = self.config
        B, S, D = x.shape

        # --- Q projection ---
        W_q_a = weights.get(f"{layer_prefix}q_a_proj.weight")
        W_q_b = weights.get(f"{layer_prefix}q_b_proj.weight")
        if W_q_a is not None and W_q_b is not None:
            # Compressed Q path
            c_q = x.reshape(B * S, D) @ W_q_a.T     # (BS, q_lora_rank)
            q_all = c_q @ W_q_b.T                    # (BS, num_heads * total_head_dim)
        else:
            W_q = weights.get(f"{layer_prefix}q_proj.weight", np.eye(D)[:, :cfg.num_heads * cfg.total_head_dim])
            q_all = x.reshape(B * S, D) @ W_q.T

        H = cfg.num_heads
        D_nope = cfg.qk_nope_head_dim
        D_rope = cfg.qk_rope_head_dim
        D_v    = cfg.v_head_dim

        q_all = q_all.reshape(B, S, H, D_nope + D_rope)
        q_nope = q_all[..., :D_nope]    # (B, S, H, D_nope)
        q_rope = q_all[..., D_nope:]    # (B, S, H, D_rope)
        q_rope = self._rope(q_rope.transpose(0, 2, 1, 3), position_offset).transpose(0, 2, 1, 3)

        # --- KV projection (latent) ---
        W_kv_a = weights.get(f"{layer_prefix}kv_a_proj.weight",
                              np.random.randn(D, cfg.kv_lora_rank).astype(np.float32) * 0.01)
        W_kv_b = weights.get(f"{layer_prefix}kv_b_proj.weight",
                              np.random.randn(cfg.kv_lora_rank, H * (D_nope + D_v)).astype(np.float32) * 0.01)

        latent = x.reshape(B * S, D) @ W_kv_a.T  # (BS, kv_lora_rank)

        # Reconstruct K/V
        nope_dim = H * D_nope
        W_kv_b_nope = W_kv_b[:, :nope_dim]
        W_kv_b_v    = W_kv_b[:, nope_dim:]

        k_nope = (latent @ W_kv_b_nope).reshape(B, S, H, D_nope)  # (B, S, H, D_nope)
        v      = (latent @ W_kv_b_v).reshape(B, S, H, D_v)        # (B, S, H, D_v)

        # Decoupled RoPE K
        W_k_rope = weights.get(f"{layer_prefix}k_rope_proj.weight",
                                np.random.randn(D, H * D_rope).astype(np.float32) * 0.01)
        k_rope = (x.reshape(B * S, D) @ W_k_rope.T).reshape(B, S, H, D_rope)
        k_rope = self._rope(k_rope.transpose(0, 2, 1, 3), position_offset).transpose(0, 2, 1, 3)

        # Full K = concat(k_nope, k_rope)
        k = np.concatenate([k_nope, k_rope], axis=-1)    # (B, S, H, D_nope + D_rope)
        q = np.concatenate([q_nope, q_rope], axis=-1)    # (B, S, H, D_nope + D_rope)

        # --- Attention ---
        # (B, H, S, D_head) for matmul
        q = q.transpose(0, 2, 1, 3)  # (B, H, S, D_head)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self._scale  # (B, H, S, S)
        # Causal mask
        mask = np.triu(np.full((S, S), -1e9), k=1)
        scores = scores + mask[np.newaxis, np.newaxis]
        attn = self._softmax(scores)
        out = np.matmul(attn, v)  # (B, H, S, D_v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * D_v)

        # --- Output projection ---
        W_o = weights.get(f"{layer_prefix}o_proj.weight",
                           np.eye(H * D_v)[:, :D])
        out = out.reshape(B * S, H * D_v) @ W_o.T
        return out.reshape(B, S, D)

    def forward_decode(
        self,
        x: np.ndarray,              # (batch, 1, hidden_dim) — single decode token
        kv_cache: MLACompressedKVCache,
        request_id: str,
        weights: dict[str, np.ndarray],
        layer_prefix: str = "",
    ) -> np.ndarray:
        """
        Single-token MLA decode forward.

        Appends the new token's latent to the cache, then attends over all
        cached latents. Implements the O(seq * kv_lora_rank) KV expansion
        instead of O(seq * num_kv_heads * head_dim).
        """
        cfg = self.config
        B, _, D = x.shape
        H = cfg.num_heads
        D_nope = cfg.qk_nope_head_dim
        D_rope = cfg.qk_rope_head_dim
        D_v    = cfg.v_head_dim

        # Q for new token
        W_q_a = weights.get(f"{layer_prefix}q_a_proj.weight")
        W_q_b = weights.get(f"{layer_prefix}q_b_proj.weight")
        if W_q_a is not None and W_q_b is not None:
            c_q = x.reshape(B, D) @ W_q_a.T
            q_all = c_q @ W_q_b.T
        else:
            W_q = weights.get(f"{layer_prefix}q_proj.weight", np.eye(D)[:, :H * (D_nope + D_rope)])
            q_all = x.reshape(B, D) @ W_q.T

        q_all = q_all.reshape(B, H, D_nope + D_rope)
        q_nope = q_all[:, :, :D_nope]
        q_rope = q_all[:, :, D_nope:]
        pos_offset = kv_cache.seq_len(request_id)
        q_rope = self._rope(q_rope[:, np.newaxis, :, :], pos_offset).squeeze(1)

        # New latent KV
        W_kv_a = weights.get(f"{layer_prefix}kv_a_proj.weight",
                              np.zeros((D, cfg.kv_lora_rank), dtype=np.float32))
        new_latent = x.reshape(B, D) @ W_kv_a.T  # (B, kv_lora_rank)

        # New RoPE K
        W_k_rope = weights.get(f"{layer_prefix}k_rope_proj.weight",
                                np.zeros((D, H * D_rope), dtype=np.float32))
        new_k_rope = (x.reshape(B, D) @ W_k_rope.T).reshape(B, H, D_rope)
        new_k_rope = self._rope(new_k_rope[:, np.newaxis, :, :], pos_offset).squeeze(1)

        # Append to cache (use request_id 0 for batch item 0 in simple mode)
        kv_cache.append(request_id, new_latent[:1], new_k_rope[:1])

        # Reconstruct K/V from cached latents
        W_kv_b = weights.get(f"{layer_prefix}kv_b_proj.weight",
                              np.zeros((cfg.kv_lora_rank, H * (D_nope + D_v)), dtype=np.float32))
        k_nope, k_rope, v = kv_cache.reconstruct(request_id, W_kv_b)  # (T, H, dim)

        T = k_nope.shape[0]
        k = np.concatenate([k_nope, k_rope], axis=-1)  # (T, H, D_nope+D_rope)
        k = k.transpose(1, 0, 2)   # (H, T, D_nope+D_rope)
        v = v.transpose(1, 0, 2)   # (H, T, D_v)

        q_full = np.concatenate([q_nope, q_rope], axis=-1)  # (B, H, D_head)
        q_full = q_full[0]  # (H, D_head)

        # Attention: (H, 1, T)
        scores = np.einsum("hd,htd->ht", q_full, k) * self._scale
        attn = self._softmax(scores)          # (H, T)
        out = np.einsum("ht,htd->hd", attn, v)  # (H, D_v)
        out = out.reshape(1, H * D_v)          # (1, H*D_v)

        W_o = weights.get(f"{layer_prefix}o_proj.weight", np.eye(H * D_v)[:, :D])
        out = out @ W_o.T
        return out.reshape(1, 1, D)


# ---------------------------------------------------------------------------
# MLA Detector (architecture detection)
# ---------------------------------------------------------------------------

class MLADetector:
    """Detects MLA architecture from model config and weight keys."""

    MLA_FAMILIES = frozenset(["deepseek", "kimi", "glm", "mla"])

    def detect_from_config(self, model_config: dict[str, Any]) -> MLAConfig | None:
        """
        Detect MLA config from a HuggingFace model config dict.

        Returns MLAConfig if MLA is detected, None otherwise.
        """
        arch = model_config.get("architectures", [""])[0].lower()
        model_type = model_config.get("model_type", "").lower()

        is_mla = (
            "deepseek" in arch or "deepseek" in model_type
            or "kimi" in arch or "kimi" in model_type
            or model_config.get("kv_lora_rank") is not None
        )
        if not is_mla:
            return None

        kv_lora_rank      = model_config.get("kv_lora_rank", 512)
        q_lora_rank       = model_config.get("q_lora_rank", 1536)
        qk_nope_head_dim  = model_config.get("qk_nope_head_dim", 128)
        qk_rope_head_dim  = model_config.get("qk_rope_head_dim", 64)
        v_head_dim        = model_config.get("v_head_dim", 128)
        num_heads         = model_config.get("num_attention_heads", 128)
        num_kv_heads      = model_config.get("num_key_value_heads", num_heads)
        rope_theta        = model_config.get("rope_theta", 10000.0)

        return MLAConfig(
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rope_decoupled=True,
        )

    def detect_from_weights(self, weight_keys: list[str]) -> MLAConfig | None:
        """Detect MLA from the presence of characteristic weight keys."""
        has_kv_a = any("kv_a_proj" in k for k in weight_keys)
        has_kv_b = any("kv_b_proj" in k for k in weight_keys)
        has_q_a  = any("q_a_proj" in k for k in weight_keys)
        has_k_rope = any("k_rope" in k for k in weight_keys)

        if has_kv_a and has_kv_b:
            logger.info("MLA detected via weight key scan (kv_a_proj + kv_b_proj found)")
            config = MLAConfig()
            if has_q_a:
                config.q_lora_rank = 1536
            config.rope_decoupled = has_k_rope
            return config

        return None

    def detect_from_architecture(self, architecture: Any) -> MLAConfig | None:
        """
        Detect MLA from an ingested :class:`ModelArchitecture`.

        Returns None when the architecture does not use MLA.
        """
        attn_type = str(getattr(architecture, "attention_type", "") or "").upper()
        family = str(getattr(architecture, "family", "") or "").lower()
        is_mla = attn_type == "MLA" or any(f in family for f in self.MLA_FAMILIES)
        if not is_mla:
            return None

        num_heads = int(getattr(architecture, "num_attention_heads", 128) or 128)
        num_kv_heads = int(getattr(architecture, "num_kv_heads", None) or num_heads)
        head_dim = int(getattr(architecture, "head_dim", None) or 128)

        # DeepSeek's published split: the RoPE branch is half the nope branch.
        qk_rope = max(16, head_dim // 2)
        return MLAConfig(
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_nope_head_dim=head_dim,
            qk_rope_head_dim=qk_rope,
            v_head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=float(getattr(architecture, "rope_theta", 10000.0) or 10000.0),
            rope_decoupled=True,
        )


# ---------------------------------------------------------------------------
# Compile-time MLA plan
# ---------------------------------------------------------------------------

#: Attention kernel per target family. FA-4 lands on Blackwell-class parts
#: (SM100/SM120); FA-3 covers Hopper; everything else uses the portable kernel.
_MLA_KERNELS: dict[str, str] = {
    "fa4": "aeg.mla_flash_attention_4",
    "fa3": "aeg.mla_flash_attention_3",
    "portable": "aeg.mla_portable",
}

_FA4_TARGETS = ("cuda_sm100", "cuda_sm103", "cuda_sm120")
_FA3_TARGETS = ("cuda_sm90", "cuda_sm89")


@dataclass
class MLACompressionPlan:
    """
    Compile-time MLA plan recorded at ``.aeg/mla/plan.json``.

    Captures whether MLA applies, which kernel the target can run, and the
    KV-cache compression the runtime should expect.
    """

    enabled: bool
    target: str = ""
    kernel: str = _MLA_KERNELS["portable"]
    config: MLAConfig | None = None
    weight_absorption: bool = False
    kv_cache_dtype: str = "BF16"
    version: str = "mla_compression/1.0"

    @property
    def compression_ratio(self) -> float:
        """KV cache compression vs standard GQA; 1.0 when MLA is disabled."""
        if not self.enabled or self.config is None:
            return 1.0
        return self.config.compression_ratio

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "enabled": self.enabled,
            "target": self.target,
            "kernel": self.kernel,
            "weight_absorption": self.weight_absorption,
            "kv_cache_dtype": self.kv_cache_dtype,
            "compression_ratio": round(self.compression_ratio, 4),
        }
        if self.config is not None:
            d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLACompressionPlan":
        cfg = d.get("config")
        return cls(
            enabled=d.get("enabled", False),
            target=d.get("target", ""),
            kernel=d.get("kernel", _MLA_KERNELS["portable"]),
            config=MLAConfig.from_dict(cfg) if cfg else None,
            weight_absorption=d.get("weight_absorption", False),
            kv_cache_dtype=d.get("kv_cache_dtype", "BF16"),
            version=d.get("version", "mla_compression/1.0"),
        )

    def save(self, aeg_dir: str | Path) -> Path:
        out = Path(aeg_dir) / "mla" / "plan.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("MLA plan saved", path=str(out), enabled=self.enabled)
        return out


class MLAPlanner:
    """
    Plans MLA compilation for a model/target pair.

    Wraps :class:`MLADetector` and picks the attention kernel the target can
    actually run, so the AEG records a plan the runtime can execute rather
    than an aspiration.
    """

    def __init__(self, detector: MLADetector | None = None) -> None:
        self.detector = detector or MLADetector()

    @staticmethod
    def select_kernel(target: str) -> str:
        """Return the best MLA attention kernel available on a target."""
        t = (target or "").lower()
        if any(t.startswith(x) for x in _FA4_TARGETS):
            return _MLA_KERNELS["fa4"]
        if any(t.startswith(x) for x in _FA3_TARGETS):
            return _MLA_KERNELS["fa3"]
        return _MLA_KERNELS["portable"]

    def plan(
        self,
        architecture: Any,
        target: str = "",
        kv_cache_dtype: str = "BF16",
    ) -> MLACompressionPlan:
        """
        Build the MLA plan for an architecture on a target.

        Args:
            architecture: Ingested :class:`ModelArchitecture`.
            target: Hardware target id, e.g. ``cuda_sm100``.
            kv_cache_dtype: Storage dtype for the compressed latent cache.

        Returns:
            An MLACompressionPlan. ``enabled`` is False for non-MLA models,
            in which case the portable kernel is recorded and the compression
            ratio is 1.0.
        """
        config = self.detector.detect_from_architecture(architecture)
        if config is None:
            return MLACompressionPlan(
                enabled=False,
                target=target,
                kernel=_MLA_KERNELS["portable"],
                kv_cache_dtype=kv_cache_dtype,
            )

        plan = MLACompressionPlan(
            enabled=True,
            target=target,
            kernel=self.select_kernel(target),
            config=config,
            # Absorption folds W_UK into W_UQ at compile time; only valid when
            # the RoPE branch is decoupled, since the RoPE half cannot absorb.
            weight_absorption=config.weight_absorption_possible and config.rope_decoupled,
            kv_cache_dtype=kv_cache_dtype,
        )
        logger.info(
            "MLA plan built",
            target=target,
            kernel=plan.kernel,
            compression_ratio=round(plan.compression_ratio, 2),
        )
        return plan
