"""
Attention kernels — FlashAttention-2/3 reference + optional flash_attn dispatch.

Provides:
  - ``VanillaAttention``: O(n²) memory, numerically stable, always available
  - ``FlashAttention2``: O(n) memory tiled attention (numpy reference + optional flash_attn)
  - ``GroupedQueryAttention``: GQA / MQA attention for LLaMA-3, Qwen3, Mistral etc.
  - ``SlidingWindowAttention``: Mistral-style local attention
  - ``PagedAttention``: vLLM-style block-sparse KV cache attention
  - ``AttentionDispatcher``: selects the best available kernel at runtime
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aether.kernels.base import Kernel
from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along axis."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-9)


def _causal_mask(seq_len: int, kv_len: int, dtype: np.dtype) -> np.ndarray:
    """Upper-triangular causal attention mask (−∞ for future tokens)."""
    mask = np.zeros((seq_len, kv_len), dtype=dtype)
    if seq_len > 1:
        future = np.triu(np.ones((seq_len, kv_len), dtype=bool), k=kv_len - seq_len + 1)
        mask[future] = -1e9
    return mask


# ---------------------------------------------------------------------------
# Vanilla attention (reference, O(n²) memory)
# ---------------------------------------------------------------------------

class VanillaAttention(Kernel):
    """
    Standard scaled dot-product attention.

    Inputs: (batch, heads, seq_len, head_dim)
    Output: (batch, heads, seq_len, head_dim)
    """

    name = "vanilla_attention"
    supported_formats: list[str] = ["vanilla"]

    def __init__(self, head_dim: int = 128, softmax_scale: float | None = None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)

    def forward(
        self,
        query: np.ndarray,      # (B, H, S, D)
        key: np.ndarray,        # (B, H, T, D)
        value: np.ndarray,      # (B, H, T, D)
        mask: np.ndarray | None = None,
        causal: bool = True,
    ) -> np.ndarray:
        B, H, S, D = query.shape
        T = key.shape[2]
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * self.softmax_scale  # (B, H, S, T)
        if causal and S > 1:
            cm = _causal_mask(S, T, scores.dtype)
            scores = scores + cm[np.newaxis, np.newaxis, :, :]
        if mask is not None:
            scores = scores + mask
        attn_weights = _softmax(scores, axis=-1)
        return np.matmul(attn_weights, value)  # (B, H, S, D)


# ---------------------------------------------------------------------------
# FlashAttention-2 numpy reference (O(n) memory via tiling)
# ---------------------------------------------------------------------------

class FlashAttention2(Kernel):
    """
    FlashAttention-2 tiled implementation in numpy.

    This is the reference algorithm from Dao et al. (2023) in pure numpy.
    On CPU it is slower than VanillaAttention due to Python loop overhead,
    but it demonstrates the correct memory-efficient tiling and is used for
    accuracy validation. When ``flash_attn`` is installed, the real CUDA
    kernel is dispatched automatically.

    Tile sizes default to 64 which is tuned for typical CPU cache lines.
    """

    name = "flash_attention_2"
    supported_formats: list[str] = ["flash_attention_2", "flash_attention_3"]

    def __init__(
        self,
        head_dim: int = 128,
        softmax_scale: float | None = None,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
        self.block_size = block_size
        self._flash_attn_available = self._check_flash_attn()

    def _check_flash_attn(self) -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    def forward(
        self,
        query: np.ndarray,      # (B, H, S, D)
        key: np.ndarray,        # (B, H, T, D)
        value: np.ndarray,      # (B, H, T, D)
        mask: np.ndarray | None = None,
        causal: bool = True,
    ) -> np.ndarray:
        """
        Dispatch to real flash_attn kernel if available, else use tiled numpy.
        """
        if self._flash_attn_available:
            return self._dispatch_flash_attn(query, key, value, causal)
        return self._tiled_forward(query, key, value, mask, causal)

    def _dispatch_flash_attn(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal: bool,
    ) -> np.ndarray:
        """Dispatch to flash_attn CUDA kernel (requires torch tensors)."""
        try:
            import torch
            from flash_attn import flash_attn_func

            q = torch.from_numpy(query.astype(np.float16)).cuda()
            k = torch.from_numpy(key.astype(np.float16)).cuda()
            v = torch.from_numpy(value.astype(np.float16)).cuda()
            # flash_attn_func expects (B, S, H, D)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = flash_attn_func(q, k, v, causal=causal, softmax_scale=self.softmax_scale)
            out = out.permute(0, 2, 1, 3).cpu().numpy().astype(np.float32)
            return out
        except Exception as exc:
            logger.debug("flash_attn dispatch failed (%s), falling back to tiled numpy", exc)
            return self._tiled_forward(query, key, value, None, causal)

    def _tiled_forward(
        self,
        query: np.ndarray,      # (B, H, S, D)
        key: np.ndarray,        # (B, H, T, D)
        value: np.ndarray,      # (B, H, T, D)
        mask: np.ndarray | None,
        causal: bool,
    ) -> np.ndarray:
        """
        O(n) memory FlashAttention-2 tiled algorithm (numpy reference).

        Per-head, per-batch online softmax with block tiling.
        """
        B, H, S, D = query.shape
        T = key.shape[2]
        Br = min(self.block_size, S)
        Bc = min(self.block_size, T)

        output = np.zeros_like(query)

        for b in range(B):
            for h in range(H):
                q_bh = query[b, h]  # (S, D)
                k_bh = key[b, h]    # (T, D)
                v_bh = value[b, h]  # (T, D)

                # Online softmax state per query block
                O = np.zeros((S, D), dtype=np.float32)
                m = np.full(S, -np.inf, dtype=np.float32)   # running max
                l = np.zeros(S, dtype=np.float32)            # running denominator

                # Iterate over key/value blocks
                for kv_start in range(0, T, Bc):
                    kv_end = min(kv_start + Bc, T)
                    k_block = k_bh[kv_start:kv_end]  # (Bc, D)
                    v_block = v_bh[kv_start:kv_end]

                    # Iterate over query blocks
                    for q_start in range(0, S, Br):
                        q_end = min(q_start + Br, S)
                        q_block = q_bh[q_start:q_end]   # (Br, D)
                        m_prev = m[q_start:q_end].copy()
                        l_prev = l[q_start:q_end].copy()
                        O_prev = O[q_start:q_end].copy()

                        # Attention scores for this tile
                        s = q_block @ k_block.T * self.softmax_scale  # (Br, Bc)

                        # Apply causal mask
                        if causal:
                            for qi in range(q_end - q_start):
                                for ki in range(kv_end - kv_start):
                                    abs_q = q_start + qi
                                    abs_k = kv_start + ki
                                    if abs_k > abs_q:
                                        s[qi, ki] = -1e9

                        if mask is not None:
                            s = s + mask[q_start:q_end, kv_start:kv_end]

                        # Online softmax update
                        m_new = np.maximum(m_prev, s.max(axis=-1))
                        exp_s = np.exp(s - m_new[:, np.newaxis])
                        l_new = np.exp(m_prev - m_new) * l_prev + exp_s.sum(axis=-1)

                        # Update output
                        O[q_start:q_end] = (
                            np.exp(m_prev - m_new)[:, np.newaxis] * O_prev
                            + exp_s @ v_block
                        )
                        m[q_start:q_end] = m_new
                        l[q_start:q_end] = l_new

                # Normalize
                output[b, h] = O / (l[:, np.newaxis] + 1e-9)

        return output


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA / MQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(Kernel):
    """
    Grouped Query Attention (GQA) — LLaMA-3, Qwen3, Mistral.

    KV heads are shared across groups of Q heads. ``num_kv_heads`` divides
    ``num_q_heads`` evenly.
    """

    name = "grouped_query_attention"
    supported_formats: list[str] = ["gqa", "mqa"]

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_q_heads % num_kv_heads != 0:
            msg = f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            raise ValueError(msg)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
        self._attn = VanillaAttention(head_dim=head_dim, softmax_scale=self.softmax_scale)

    def forward(
        self,
        query: np.ndarray,      # (B, num_q_heads, S, D)
        key: np.ndarray,        # (B, num_kv_heads, T, D)
        value: np.ndarray,      # (B, num_kv_heads, T, D)
        mask: np.ndarray | None = None,
        causal: bool = True,
    ) -> np.ndarray:
        B, Hq, S, D = query.shape
        _, Hkv, T, _ = key.shape
        # Expand KV to match Q heads
        key_exp = np.repeat(key, self.groups, axis=1)      # (B, Hq, T, D)
        value_exp = np.repeat(value, self.groups, axis=1)  # (B, Hq, T, D)
        return self._attn.forward(query, key_exp, value_exp, mask=mask, causal=causal)


# ---------------------------------------------------------------------------
# Sliding Window Attention
# ---------------------------------------------------------------------------

class SlidingWindowAttention(Kernel):
    """
    Mistral-style sliding window attention.

    Each query position only attends to the last ``window_size`` key positions,
    plus the full prefix (full-context cache for the first n layers).
    """

    name = "sliding_window_attention"
    supported_formats: list[str] = ["sliding_window"]

    def __init__(
        self,
        head_dim: int,
        window_size: int = 4096,
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)

    def forward(
        self,
        query: np.ndarray,      # (B, H, S, D)
        key: np.ndarray,        # (B, H, T, D)
        value: np.ndarray,      # (B, H, T, D)
        mask: np.ndarray | None = None,
        causal: bool = True,
    ) -> np.ndarray:
        B, H, S, D = query.shape
        T = key.shape[2]
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * self.softmax_scale

        # Build sliding window mask
        sw_mask = np.full((S, T), -1e9, dtype=np.float32)
        for i in range(S):
            abs_q = T - S + i
            win_start = max(0, abs_q - self.window_size + 1)
            win_end = abs_q + 1 if causal else T
            sw_mask[i, win_start:win_end] = 0.0

        scores = scores + sw_mask[np.newaxis, np.newaxis, :, :]
        if mask is not None:
            scores = scores + mask
        attn = _softmax(scores, axis=-1)
        return np.matmul(attn, value)


# ---------------------------------------------------------------------------
# Paged Attention (block-sparse KV cache)
# ---------------------------------------------------------------------------

class PagedAttention(Kernel):
    """
    Block-sparse paged attention for continuous batching (vLLM style).

    KV cache is stored in fixed-size blocks. The block table maps logical
    KV positions to physical block IDs, allowing fragmented KV storage.
    """

    name = "paged_attention"
    supported_formats: list[str] = ["paged"]

    def __init__(
        self,
        head_dim: int,
        block_size: int = 16,
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
        # KV block store: block_id -> (key_block, value_block) shape (H, block_size, D)
        self._kv_store: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._next_block_id = 0

    def allocate_block(self, num_heads: int) -> int:
        """Allocate a new KV cache block, returning its block ID."""
        block_id = self._next_block_id
        self._next_block_id += 1
        self._kv_store[block_id] = (
            np.zeros((num_heads, self.block_size, self.head_dim), dtype=np.float32),
            np.zeros((num_heads, self.block_size, self.head_dim), dtype=np.float32),
        )
        return block_id

    def write_kv(self, block_id: int, slot: int, key: np.ndarray, value: np.ndarray) -> None:
        """Write key/value into a specific slot of a block."""
        if block_id not in self._kv_store:
            msg = f"Block {block_id} not allocated"
            raise ValueError(msg)
        self._kv_store[block_id][0][:, slot, :] = key  # (H, D)
        self._kv_store[block_id][1][:, slot, :] = value

    def forward(
        self,
        query: np.ndarray,          # (B, H, 1, D) — decode step
        block_table: list[list[int]],  # [batch] -> list of block IDs
        seq_lens: list[int],           # number of valid tokens per sequence
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Paged attention forward for the decode step.

        Gathers KV from physical blocks specified in block_table.
        """
        B, H, S, D = query.shape
        outputs = np.zeros_like(query)
        scale = self.softmax_scale

        for b in range(B):
            blocks = block_table[b]
            seq_len = seq_lens[b]
            if seq_len == 0 or not blocks:
                continue
            # Gather key/value from blocks
            k_list, v_list = [], []
            gathered = 0
            for blk_id in blocks:
                if blk_id not in self._kv_store:
                    continue
                k_blk, v_blk = self._kv_store[blk_id]
                remaining = seq_len - gathered
                use = min(self.block_size, remaining)
                k_list.append(k_blk[:, :use, :])
                v_list.append(v_blk[:, :use, :])
                gathered += use
                if gathered >= seq_len:
                    break
            if not k_list:
                continue
            k_gathered = np.concatenate(k_list, axis=1)  # (H, T, D)
            v_gathered = np.concatenate(v_list, axis=1)

            q_b = query[b, :, 0, :]  # (H, D)
            scores = np.einsum("hd,htd->ht", q_b, k_gathered) * scale  # (H, T)
            attn = _softmax(scores, axis=-1)                            # (H, T)
            out = np.einsum("ht,htd->hd", attn, v_gathered)             # (H, D)
            outputs[b, :, 0, :] = out

        return outputs


# ---------------------------------------------------------------------------
# Attention dispatcher
# ---------------------------------------------------------------------------

class AttentionDispatcher:
    """
    Selects the best available attention kernel at runtime.

    Priority:
      1. flash_attn CUDA (if GPU + flash_attn installed)
      2. FlashAttention2 tiled numpy (FA-2 reference, always available)
      3. GroupedQueryAttention (when num_kv_heads < num_q_heads)
      4. VanillaAttention (fallback)
    """

    def __init__(
        self,
        num_q_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int = 128,
        window_size: int | None = None,
        softmax_scale: float | None = None,
    ) -> None:
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads or num_q_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
        self._kernel = self._select_kernel()
        logger.info("Attention dispatcher: selected %s", self._kernel.name)

    def _select_kernel(self) -> Kernel:
        if self.window_size is not None:
            return SlidingWindowAttention(
                head_dim=self.head_dim,
                window_size=self.window_size,
                softmax_scale=self.softmax_scale,
            )
        if self.num_kv_heads < self.num_q_heads:
            return GroupedQueryAttention(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                softmax_scale=self.softmax_scale,
            )
        # Prefer FlashAttention-2 (handles optional CUDA dispatch internally)
        return FlashAttention2(
            head_dim=self.head_dim,
            softmax_scale=self.softmax_scale,
        )

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray | None = None,
        causal: bool = True,
    ) -> np.ndarray:
        return self._kernel.forward(query, key, value, mask=mask, causal=causal)

    @property
    def kernel_name(self) -> str:
        return self._kernel.name


# ---------------------------------------------------------------------------
# Legacy alias for backward compatibility
# ---------------------------------------------------------------------------

class AttentionKernel(VanillaAttention):
    """Backward-compatible alias for VanillaAttention."""
    pass
