"""
Reference CPU execution engine for compiled AEG models.

Runs a real transformer forward pass on numpy arrays, dispatching the hot
primitives to the natively compiled kernels in
:mod:`aether.kernels.native_cpu` when a host compiler is available.

This is a *reference* engine: correctness and portability come first, and it
exists so a compiled ``.aeg`` artifact can actually produce logits on any machine
without a GPU. It implements the standard decoder-only stack used by Llama, Qwen
and Mistral:

    embed -> [RMSNorm -> GQA attention -> residual
              -> RMSNorm -> SwiGLU FFN -> residual] x L
          -> RMSNorm -> LM head

Grouped-query attention, RoPE, and an incremental KV cache are supported, so
decode is O(1) in past length rather than re-running the full prefill each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.kernels.native_cpu import NativeCPUKernels, get_native_kernels
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["LayerWeights", "ModelWeights", "KVCache", "CPUExecutionEngine"]


@dataclass
class LayerWeights:
    """Weight matrices for one transformer layer.

    All projections are stored in ``(out_features, in_features)`` orientation,
    matching the HuggingFace checkpoint convention.
    """

    attention_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    ffn_norm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray

    def validate(self, layer_index: int, hidden_size: int) -> None:
        """Check that the projections compose into a valid layer."""
        if self.q_proj.shape[1] != hidden_size:
            msg = (
                f"layer {layer_index}: q_proj expects input {self.q_proj.shape[1]}, "
                f"but hidden_size is {hidden_size}"
            )
            raise ValueError(msg)
        if self.k_proj.shape != self.v_proj.shape:
            msg = (
                f"layer {layer_index}: k_proj {self.k_proj.shape} and "
                f"v_proj {self.v_proj.shape} must match"
            )
            raise ValueError(msg)
        if self.gate_proj.shape != self.up_proj.shape:
            msg = (
                f"layer {layer_index}: gate_proj {self.gate_proj.shape} and "
                f"up_proj {self.up_proj.shape} must match"
            )
            raise ValueError(msg)


@dataclass
class ModelWeights:
    """All tensors needed to run a decoder-only transformer."""

    embedding: np.ndarray
    layers: list[LayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    #: RoPE base frequency.
    rope_theta: float = 10000.0
    #: RMSNorm epsilon.
    norm_eps: float = 1e-5

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def validate(self) -> None:
        """Verify the weights are mutually consistent before execution."""
        if self.embedding.ndim != 2:
            msg = f"embedding must be 2-D (vocab, hidden), got shape {self.embedding.shape}"
            raise ValueError(msg)
        if self.lm_head.shape[1] != self.hidden_size:
            msg = (
                f"lm_head input dim {self.lm_head.shape[1]} does not match "
                f"hidden_size {self.hidden_size}"
            )
            raise ValueError(msg)
        if self.final_norm.size != self.hidden_size:
            msg = (
                f"final_norm has {self.final_norm.size} elements, "
                f"expected hidden_size {self.hidden_size}"
            )
            raise ValueError(msg)
        for index, layer in enumerate(self.layers):
            layer.validate(index, self.hidden_size)


@dataclass
class KVCache:
    """Incremental key/value cache for one model.

    Stores per-layer keys and values as ``(seq, kv_heads, head_dim)`` and grows by
    appending. Without this, each decode step would recompute the whole prefix.
    """

    num_layers: int
    keys: list[np.ndarray | None] = field(default_factory=list)
    values: list[np.ndarray | None] = field(default_factory=list)
    #: Logits for the final cached position.  This allows a session to append
    #: an empty suffix without recomputing the entire prefix.
    last_logits: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.keys:
            self.keys = [None] * self.num_layers
            self.values = [None] * self.num_layers

    @property
    def length(self) -> int:
        """Number of cached positions."""
        first = self.keys[0] if self.keys else None
        return 0 if first is None else int(first.shape[0])

    def append(self, layer: int, key: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append this step's key/value and return the full cached history."""
        if self.keys[layer] is None:
            self.keys[layer] = key
            self.values[layer] = value
        else:
            self.keys[layer] = np.concatenate([self.keys[layer], key], axis=0)
            self.values[layer] = np.concatenate([self.values[layer], value], axis=0)
        return self.keys[layer], self.values[layer]

    def reset(self) -> None:
        """Drop all cached state."""
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self.last_logits = None


class CPUExecutionEngine:
    """Executes a decoder-only transformer on CPU.

    Args:
        weights: The model tensors.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads; equals ``num_heads`` for MHA and
            is smaller for grouped-query attention.
        kernels: Kernel provider; defaults to the shared native instance.
    """

    def __init__(
        self,
        weights: ModelWeights,
        num_heads: int,
        num_kv_heads: int | None = None,
        kernels: NativeCPUKernels | None = None,
    ) -> None:
        weights.validate()
        self.weights = weights
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads != 0:
            msg = (
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention"
            )
            raise ValueError(msg)
        self.head_dim = weights.hidden_size // num_heads
        if self.head_dim % 2 != 0:
            msg = f"head_dim must be even for RoPE, got {self.head_dim}"
            raise ValueError(msg)
        self.kernels = kernels or get_native_kernels()
        self._cos, self._sin = self._build_rope_tables()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_rope_tables(self, max_positions: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Precompute RoPE cos/sin tables for positions ``[0, max_positions)``."""
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.weights.rope_theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / self.head_dim)
        )
        angles = np.arange(max_positions, dtype=np.float64)[:, None] * inv_freq[None, :]
        return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def _ensure_rope_capacity(self, required: int) -> None:
        """Grow the RoPE tables when a sequence runs past the precomputed range."""
        if required <= self._cos.shape[0]:
            return
        self._cos, self._sin = self._build_rope_tables(max_positions=max(required, 2 * self._cos.shape[0]))

    # ── Primitives ───────────────────────────────────────────────────────────

    def _linear(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Apply ``y = x @ W.T`` for a ``(out, in)`` weight matrix."""
        return self.kernels.sgemm(np.ascontiguousarray(x, dtype=np.float32), weight.T)

    def _attention(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        causal_offset: int,
    ) -> np.ndarray:
        """Scaled dot-product attention with causal masking and GQA broadcast.

        Args:
            query: ``(seq, num_heads, head_dim)``.
            keys: ``(total, num_kv_heads, head_dim)``.
            values: ``(total, num_kv_heads, head_dim)``.
            causal_offset: Number of cached positions preceding ``query``.

        Returns:
            Attention output of shape ``(seq, num_heads, head_dim)``.
        """
        seq_len = query.shape[0]
        total = keys.shape[0]
        scale = 1.0 / np.sqrt(self.head_dim)
        repeats = self.num_heads // self.num_kv_heads

        # Broadcast each KV head across the query heads that share it.
        keys_full = np.repeat(keys, repeats, axis=1)
        values_full = np.repeat(values, repeats, axis=1)

        # (heads, seq, dim) @ (heads, dim, total) -> (heads, seq, total)
        q = np.ascontiguousarray(query.transpose(1, 0, 2), dtype=np.float32)
        k = np.ascontiguousarray(keys_full.transpose(1, 2, 0), dtype=np.float32)
        scores = np.matmul(q, k) * scale

        # Causal mask: query position i may attend to key positions <= i + offset.
        positions = np.arange(total)[None, :]
        allowed = positions <= (np.arange(seq_len)[:, None] + causal_offset)
        scores = np.where(allowed[None, :, :], scores, np.float32(-np.inf))

        weights = self.kernels.softmax(scores.reshape(-1, total)).reshape(scores.shape)
        v = np.ascontiguousarray(values_full.transpose(1, 0, 2), dtype=np.float32)
        context = np.matmul(weights, v)
        return context.transpose(1, 0, 2)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self, token_ids: np.ndarray, cache: KVCache | None = None
    ) -> tuple[np.ndarray, KVCache]:
        """Run the transformer over ``token_ids``.

        Args:
            token_ids: 1-D array of token ids for this step. During prefill this
                is the whole prompt; during decode it is a single token.
            cache: Existing KV cache to extend, or None to start fresh.

        Returns:
            Tuple of ``(logits, cache)`` where logits has shape
            ``(seq, vocab_size)``.
        """
        ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            msg = "forward() requires at least one token"
            raise ValueError(msg)
        if int(ids.max()) >= self.weights.vocab_size or int(ids.min()) < 0:
            msg = (
                f"token id out of range: got [{ids.min()}, {ids.max()}], "
                f"vocab_size is {self.weights.vocab_size}"
            )
            raise ValueError(msg)

        cache = cache or KVCache(num_layers=self.weights.num_layers)
        past = cache.length
        seq_len = int(ids.size)
        self._ensure_rope_capacity(past + seq_len)

        hidden = self.weights.embedding[ids].astype(np.float32)

        for index, layer in enumerate(self.weights.layers):
            # ── Attention block ──
            normed = self.kernels.rmsnorm(hidden, layer.attention_norm, self.weights.norm_eps)
            q = self._linear(normed, layer.q_proj).reshape(seq_len, self.num_heads, self.head_dim)
            k = self._linear(normed, layer.k_proj).reshape(seq_len, self.num_kv_heads, self.head_dim)
            v = self._linear(normed, layer.v_proj).reshape(seq_len, self.num_kv_heads, self.head_dim)

            q = self.kernels.rope(q, self._cos, self._sin, position_offset=past)
            k = self.kernels.rope(k, self._cos, self._sin, position_offset=past)

            keys, values = cache.append(index, k, v)
            context = self._attention(q, keys, values, causal_offset=past)
            attention_out = self._linear(
                context.reshape(seq_len, self.num_heads * self.head_dim), layer.o_proj
            )
            hidden = hidden + attention_out

            # ── FFN block ──
            normed = self.kernels.rmsnorm(hidden, layer.ffn_norm, self.weights.norm_eps)
            gate = self._linear(normed, layer.gate_proj)
            up = self._linear(normed, layer.up_proj)
            hidden = hidden + self._linear(self.kernels.swiglu(gate, up), layer.down_proj)

        hidden = self.kernels.rmsnorm(hidden, self.weights.final_norm, self.weights.norm_eps)
        logits = self._linear(hidden, self.weights.lm_head)
        cache.last_logits = np.asarray(logits[-1], dtype=np.float32).copy()
        return logits, cache

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        grammar_session: Any | None = None,
    ) -> list[int]:
        """Autoregressively generate token ids.

        Args:
            prompt_ids: Prompt token ids.
            max_tokens: Maximum new tokens to emit.
            temperature: 0 selects greedily; higher values sample more randomly.
            top_k: Restrict sampling to the k highest-probability tokens (0 = off).
            eos_token_id: Stop early when this token is produced.
            seed: Seed for reproducible sampling.

        Returns:
            The generated token ids, excluding the prompt.
        """
        generated, _cache = self.generate_with_cache(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            seed=seed,
            grammar_session=grammar_session,
        )
        return generated

    def generate_with_cache(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        grammar_session: Any | None = None,
        cache: KVCache | None = None,
    ) -> tuple[list[int], KVCache]:
        """Generate tokens while accepting and returning an incremental KV cache.

        ``prompt_ids`` is interpreted as the next uncached suffix when
        ``cache`` is supplied.  The caller is responsible for proving that the
        suffix follows the sequence represented by that cache.  This explicit
        contract prevents accidental reuse across unrelated requests.
        """
        generated: list[int] = []
        final_cache: list[KVCache | None] = [cache]

        def remember(updated: KVCache) -> None:
            final_cache[0] = updated

        generated.extend(
            self.generate_iter(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                seed=seed,
                grammar_session=grammar_session,
                cache=cache,
                cache_callback=remember,
            )
        )
        if final_cache[0] is None:
            raise RuntimeError("generation completed without producing a KV cache")
        return generated, final_cache[0]

    def generate_iter(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        grammar_session: Any | None = None,
        cache: KVCache | None = None,
        cache_callback: Any | None = None,
    ) -> Any:
        """Yield generated token IDs as they are produced.

        This is the streaming counterpart to :meth:`generate_with_cache`.  It
        executes one decoder step before yielding each token, so callers can
        transport actual incremental output instead of slicing a completed
        response.  ``cache_callback`` receives the final cache after normal
        completion and is intentionally not called when generation raises.
        """
        ids = np.ascontiguousarray(prompt_ids, dtype=np.int64).reshape(-1)
        rng = np.random.default_rng(seed)
        if cache is None:
            if ids.size == 0:
                raise ValueError("generate_iter() requires prompt tokens for a new cache")
            logits, cache = self.forward(ids)
        elif ids.size:
            logits, cache = self.forward(ids, cache)
        elif cache.last_logits is not None:
            logits = cache.last_logits.reshape(1, -1)
        else:
            raise ValueError("provided KV cache has no logits for an empty prompt suffix")

        for _ in range(max(0, max_tokens)):
            next_logits = np.asarray(logits[-1], dtype=np.float32).copy()
            if grammar_session is not None:
                mask = grammar_session.get_token_mask()
                if len(mask) * 8 < next_logits.size:
                    raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
                allowed = np.fromiter(
                    ((mask[i // 8] & (1 << (i % 8))) != 0 for i in range(next_logits.size)),
                    dtype=bool,
                    count=next_logits.size,
                )
                if not np.any(allowed):
                    raise ValueError("Grammar FSM has no valid next token")
                next_logits[~allowed] = -np.inf
            next_token = self._sample(next_logits, temperature, top_k, rng)
            if grammar_session is not None and grammar_session.advance(next_token) < 0:
                raise ValueError("The CPU engine produced a token rejected by the grammar FSM")
            yield next_token
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits, cache = self.forward(np.array([next_token], dtype=np.int64), cache)

        if cache_callback is not None:
            cache_callback(cache)

    def _sample(
        self, logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator
    ) -> int:
        """Choose the next token from a logit vector."""
        if temperature <= 0.0:
            return self.kernels.argmax(logits)
        scaled = logits.astype(np.float32) / np.float32(temperature)
        if top_k > 0:
            k = min(top_k, scaled.size)
            cutoff = np.partition(scaled, -k)[-k]
            scaled = np.where(scaled < cutoff, np.float32(-np.inf), scaled)
        probabilities = self.kernels.softmax(scaled.reshape(1, -1)).reshape(-1)
        # Renormalise: softmax over a masked vector can drift by a few ulps.
        probabilities = probabilities / probabilities.sum()
        return int(rng.choice(probabilities.size, p=probabilities))

    def __repr__(self) -> str:
        backend = "native" if self.kernels.is_native else "numpy"
        return (
            f"CPUExecutionEngine(layers={self.weights.num_layers}, "
            f"hidden={self.weights.hidden_size}, heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, kernels={backend})"
        )
