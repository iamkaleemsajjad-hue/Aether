"""
Native CPU paged-attention kernel.

Implements *real* paged attention: the attention kernel reads K/V directly
from :class:`~aether.runtime.paged_kv_cache.PagedKVPool` blocks **without**
first materialising the whole sequence into a contiguous K/V tensor.

Algorithm
---------
Online softmax accumulation (Flash-Attention style) across paged blocks:

    for each block b:
        k_block, v_block ← pool[b.page_id, :, layer_idx, 0/1]   # O(page_size) load
        scores  ← q · k_block^T * scale
        apply causal / ALiBi / window / sparse mask
        update (m_i, l_i, o_i) via online softmax correction

Never allocates a contiguous (seq_len × total_tokens) intermediate tensor.

Pool tensor layout (from PagedKVPool)
--------------------------------------
    _np_pool[page_id, slot, layer_idx, kv_idx, kv_head, head_dim]
        kv_idx 0 → K
        kv_idx 1 → V

References
----------
- Dao, T. et al. (2022) "FlashAttention" https://arxiv.org/abs/2205.14135
- Kwon, W. et al. (2023) "PagedAttention" https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aether.runtime.paged_kv_cache import PagedKVPool

__all__ = [
    "paged_attention_cpu",
    "PagedAttentionKernel",
]


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------

def paged_attention_cpu(
    query: np.ndarray,
    pool: "PagedKVPool",
    seq_id: str,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal_offset: int = 0,
    scale: float | None = None,
    position_type: str = "rope",
    alibi_slopes: np.ndarray | None = None,
    window_size: int | None = None,
    sparse_block_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Paged attention forward pass (CPU, pure NumPy, no contiguous KV copy).

    Parameters
    ----------
    query:
        Shape ``(seq_len, num_heads, head_dim)``.
    pool:
        :class:`~aether.runtime.paged_kv_cache.PagedKVPool` holding K/V.
    seq_id:
        Sequence identifier registered in *pool*.
    layer_idx:
        Transformer layer index (used to select the correct pool slice).
    num_heads:
        Total query heads.
    num_kv_heads:
        KV heads (``num_heads // num_kv_heads`` = GQA repeat factor).
    head_dim:
        Per-head feature dimension.
    causal_offset:
        Absolute position of the first query token (= previous KV length).
    scale:
        Softmax temperature; defaults to ``1/sqrt(head_dim)``.
    position_type:
        ``"rope"`` (bias applied externally before call), ``"alibi"``,
        or ``"none"``.
    alibi_slopes:
        Shape ``(num_heads,)`` — required when ``position_type="alibi"``.
    window_size:
        Sliding-window length; ``None`` = full causal.
    sparse_block_mask:
        Boolean ``(num_heads, n_blocks)`` — per-head block sparsity.
        A ``False`` entry means the head may skip that entire block.

    Returns
    -------
    np.ndarray
        Shape ``(seq_len, num_heads, head_dim)``.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_len = query.shape[0]
    gqa_repeats = num_heads // num_kv_heads

    # Retrieve the block table for this sequence and layer
    layer_map = pool._page_table.get(seq_id)
    if layer_map is None:
        raise KeyError(f"sequence {seq_id!r} not found in PagedKVPool")
    layer_refs = layer_map.get(layer_idx, [])

    # query: (seq, H, D) — ensure contiguous float32
    q = np.ascontiguousarray(query, dtype=np.float32)

    # Running online-softmax accumulators (seq, H)
    m_i = np.full((seq_len, num_heads), -np.inf, dtype=np.float32)  # running max
    l_i = np.zeros((seq_len, num_heads), dtype=np.float32)           # running sum
    o_i = np.zeros((seq_len, num_heads, head_dim), dtype=np.float32) # running output

    kv_token_offset = 0  # absolute position cursor

    for b_idx, ref in enumerate(layer_refs):
        page_id = ref.page_id
        tokens_in_block = ref.write_pos  # number of valid token slots in this page
        if tokens_in_block == 0:
            kv_token_offset += pool.page_size
            continue

        # ----------------------------------------------------------------
        # Optional block-level sparsity: skip entire block for all heads
        # ----------------------------------------------------------------
        if sparse_block_mask is not None:
            # shape (num_heads, n_blocks) — if every head skips this block
            if not sparse_block_mask[:, b_idx].any():
                kv_token_offset += tokens_in_block
                continue

        # ----------------------------------------------------------------
        # Load K/V from pool — shape (tokens_in_block, num_kv_heads, D)
        # _np_pool[page_id, slot, layer_idx, kv_idx, kv_head, head_dim]
        # ----------------------------------------------------------------
        k_block = pool._np_pool[page_id, :tokens_in_block, layer_idx, 0].astype(np.float32)
        v_block = pool._np_pool[page_id, :tokens_in_block, layer_idx, 1].astype(np.float32)

        # ----------------------------------------------------------------
        # GQA expansion
        # ----------------------------------------------------------------
        if gqa_repeats > 1:
            k_block = np.repeat(k_block, gqa_repeats, axis=1)  # (T_b, H, D)
            v_block = np.repeat(v_block, gqa_repeats, axis=1)

        k_block = np.ascontiguousarray(k_block, dtype=np.float32)  # (T_b, H, D)
        v_block = np.ascontiguousarray(v_block, dtype=np.float32)

        # Absolute positions of this block's tokens
        block_positions = np.arange(
            kv_token_offset, kv_token_offset + tokens_in_block, dtype=np.int64
        )

        # ----------------------------------------------------------------
        # Scores: (H, seq, T_b)  via einsum "shd,thd->hst"
        # ----------------------------------------------------------------
        scores = np.einsum("shd,thd->hst", q, k_block) * scale  # (H, seq, T_b)

        # ----------------------------------------------------------------
        # ALiBi positional bias
        # ----------------------------------------------------------------
        if position_type in {"alibi", "alibi_bias"} and alibi_slopes is not None:
            q_positions = np.arange(seq_len, dtype=np.int64) + causal_offset
            # distance: (seq, T_b) → broadcast over H
            distance = (
                block_positions[None, :] - q_positions[:, None]
            )  # (seq, T_b)
            alibi_bias = alibi_slopes[:, None, None] * distance[None, :, :]
            scores = scores + alibi_bias  # (H, seq, T_b)

        # ----------------------------------------------------------------
        # Causal mask
        # ----------------------------------------------------------------
        q_positions = np.arange(seq_len, dtype=np.int64) + causal_offset
        causal_ok = block_positions[None, :] <= q_positions[:, None]  # (seq, T_b)

        if window_size is not None and window_size > 0:
            window_ok = block_positions[None, :] >= (q_positions[:, None] - window_size + 1)
            causal_ok = causal_ok & window_ok

        # Broadcast mask over heads: (H, seq, T_b)
        scores = np.where(causal_ok[None, :, :], scores, np.float32(-np.inf))

        # ----------------------------------------------------------------
        # Per-head block sparsity mask
        # ----------------------------------------------------------------
        if sparse_block_mask is not None:
            head_ok = sparse_block_mask[:, b_idx]  # (H,)
            scores = np.where(head_ok[:, None, None], scores, np.float32(-np.inf))

        # ----------------------------------------------------------------
        # Online softmax update (Flash-Attention Algorithm 1)
        # ----------------------------------------------------------------
        # m_block: (H, seq) — max over T_b
        m_block = scores.max(axis=-1)             # (H, seq)
        m_block_T = m_block.transpose(1, 0)       # (seq, H)
        m_new = np.maximum(m_i, m_block_T)        # (seq, H)

        # Correction factors for existing accumulator
        corr_old = np.exp(np.clip(m_i - m_new, -80, 0))    # (seq, H)
        # Weights for this block's softmax
        p = np.exp(np.clip(scores - m_block[:, :, None], -80, 0))  # (H, seq, T_b)
        p_T = p.transpose(1, 0, 2)                 # (seq, H, T_b)

        # Sum of softmax weights for this block
        p_sum = p_T.sum(axis=-1)                   # (seq, H)

        # Weighted value contribution: (seq, H, T_b) @ (T_b, H, D) → (seq, H, D)
        pv = np.einsum("sht,thd->shd", p_T, v_block)  # (seq, H, D)

        # Update accumulators
        o_i = o_i * corr_old[:, :, None] + pv
        l_i = l_i * corr_old + p_sum
        m_i = m_new

        kv_token_offset += tokens_in_block

    # ----------------------------------------------------------------
    # Final normalisation
    # ----------------------------------------------------------------
    l_safe = np.maximum(l_i, 1e-10)
    output = o_i / l_safe[:, :, None]
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class PagedAttentionKernel:
    """
    Per-layer stateless wrapper with the same call signature as
    ``CPUExecutionEngine._attention()`` so the engine can swap kernels
    transparently.

    The caller must write K/V into the pool **before** calling this.
    """

    def __init__(
        self,
        pool: "PagedKVPool",
        seq_id: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        position_type: str = "rope",
        alibi_slopes: np.ndarray | None = None,
        window_size: int | None = None,
    ) -> None:
        self.pool = pool
        self.seq_id = seq_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.position_type = position_type
        self.alibi_slopes = alibi_slopes
        self.window_size = window_size

    def __call__(
        self,
        query: np.ndarray,
        layer_idx: int = 0,
        causal_offset: int = 0,
        scale: float | None = None,
        sparse_block_mask: np.ndarray | None = None,
        **_: Any,
    ) -> np.ndarray:
        return paged_attention_cpu(
            query=query,
            pool=self.pool,
            seq_id=self.seq_id,
            layer_idx=layer_idx,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            causal_offset=causal_offset,
            scale=scale,
            position_type=self.position_type,
            alibi_slopes=self.alibi_slopes,
            window_size=self.window_size,
            sparse_block_mask=sparse_block_mask,
        )
