"""Reference CPU execution for mixed attention/state-space decoders.

Jamba-style checkpoints are neither ordinary transformers nor pure Mamba
models: the block schedule is part of the model contract.  This module keeps
that schedule explicit and executes each block with the corresponding real
checkpoint tensors.  It deliberately reuses the audited transformer and
selective-scan primitives so the hybrid path has the same normalization,
RoPE, GQA, cache, and sampling semantics as the standalone engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from aether.runtime.cpu_engine import CPUExecutionEngine, KVCache, LayerWeights, ModelWeights
from aether.runtime.mamba_engine import MambaCache, MambaExecutionEngine, MambaLayerWeights, MambaModelWeights
from aether.runtime.stopping import stop_token_set


@dataclass
class HybridLayerWeights:
    kind: str
    transformer: LayerWeights | None = None
    mamba: MambaLayerWeights | None = None


@dataclass
class HybridModelWeights:
    embedding: np.ndarray
    layers: list[HybridLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    position_embedding: np.ndarray | None = None
    embedding_norm: np.ndarray | None = None
    embedding_norm_bias: np.ndarray | None = None
    final_norm_bias: np.ndarray | None = None
    position_type: str = "RoPE"
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    norm_type: str = "RMSNorm"
    ffn_type: str = "SwiGLU"

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])


@dataclass
class HybridCache:
    kv: KVCache
    ssm: MambaCache
    length: int = 0
    last_logits: np.ndarray | None = None
    last_hidden: np.ndarray | None = None

    def clone(self) -> "HybridCache":
        return HybridCache(
            kv=self.kv.clone(),
            ssm=self.ssm.clone(),
            length=self.length,
            last_logits=None if self.last_logits is None else self.last_logits.copy(),
            last_hidden=None if self.last_hidden is None else self.last_hidden.copy(),
        )


class HybridExecutionEngine:
    """Execute a capability-described attention/SSM layer schedule on CPU."""

    def __init__(
        self,
        weights: HybridModelWeights,
        *,
        layer_types: list[str],
        num_heads: int,
        num_kv_heads: int,
        state_size: int,
        inner_size: int,
        dt_rank: int,
        conv_kernel: int,
    ) -> None:
        if len(layer_types) != len(weights.layers):
            raise ValueError("hybrid layer schedule length does not match the weight list")
        normalized = [str(value).lower() for value in layer_types]
        if any(value not in {"attention", "ssm"} for value in normalized):
            raise ValueError("hybrid layer schedule entries must be 'attention' or 'ssm'")
        if not any(value == "attention" for value in normalized):
            raise ValueError("hybrid execution requires at least one attention layer")
        if not any(value == "ssm" for value in normalized):
            raise ValueError("hybrid execution requires at least one SSM layer")
        self.weights = weights
        self.layer_types = normalized
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)

        # The helpers contain no model state beyond immutable weights and RoPE
        # tables.  Every layer is represented with a valid shape so their
        # private primitives can be reused without executing the other block
        # kind.  The actual layer tensor is selected by this engine below.
        transformer_layers = [
            layer.transformer for layer in weights.layers
        ]
        if any(layer is None for layer in transformer_layers):
            raise ValueError("hybrid transformer layer table is incomplete")
        transformer_model = ModelWeights(
            embedding=weights.embedding,
            layers=[layer for layer in transformer_layers if layer is not None],
            final_norm=weights.final_norm,
            lm_head=weights.lm_head,
            position_embedding=weights.position_embedding,
            embedding_norm=weights.embedding_norm,
            embedding_norm_bias=weights.embedding_norm_bias,
            final_norm_bias=weights.final_norm_bias,
            position_type=weights.position_type,
            rope_theta=weights.rope_theta,
            norm_eps=weights.norm_eps,
            norm_type=weights.norm_type,
            ffn_type=weights.ffn_type,
        )
        self._transformer = CPUExecutionEngine(
            transformer_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

        mamba_layers = [layer.mamba for layer in weights.layers]
        if any(layer is None for layer in mamba_layers):
            raise ValueError("hybrid SSM layer table is incomplete")
        self._mamba = MambaExecutionEngine(
            MambaModelWeights(
                embedding=weights.embedding,
                layers=[layer for layer in mamba_layers if layer is not None],
                final_norm=weights.final_norm,
                lm_head=weights.lm_head,
                norm_eps=weights.norm_eps,
                norm_type=weights.norm_type,
            ),
            state_size=state_size,
            inner_size=inner_size,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
        )
        self.num_layers = len(weights.layers)
        self.sparse_attention_plan = None
        self.semantic_kv_plan = None
        self.cross_layer_kv_plan = None

    def _new_cache(self) -> HybridCache:
        return HybridCache(
            kv=KVCache(num_layers=self.num_layers),
            ssm=self._mamba._new_cache(),
        )

    def _attention_step(self, hidden: np.ndarray, index: int, cache: HybridCache, past: int) -> np.ndarray:
        layer = self.weights.layers[index].transformer
        if layer is None:
            raise ValueError(f"hybrid attention layer {index} has no transformer weights")
        normed = self._transformer._norm(hidden, layer.attention_norm, layer.attention_norm_bias)
        head_dim = self._transformer.head_dim
        q = self._transformer._linear(normed, layer.q_proj, (index, "q_proj"), layer.q_proj_bias)
        q = q.reshape(1, self.num_heads, head_dim)
        if layer.q_norm is not None:
            q = self._transformer.kernels.rmsnorm(q, layer.q_norm, self.weights.norm_eps)
        uses_rope = str(self.weights.position_type or "RoPE").lower() in {
            "rope", "rotary", "rotary_embedding"
        }
        if uses_rope:
            self._transformer._ensure_rope_capacity(past + 1)
            q = self._transformer.kernels.rope(q, self._transformer._cos, self._transformer._sin, position_offset=past)
        k = self._transformer._linear(normed, layer.k_proj, (index, "k_proj"), layer.k_proj_bias)
        v = self._transformer._linear(normed, layer.v_proj, (index, "v_proj"), layer.v_proj_bias)
        k = k.reshape(1, self.num_kv_heads, head_dim)
        v = v.reshape(1, self.num_kv_heads, head_dim)
        if layer.k_norm is not None:
            k = self._transformer.kernels.rmsnorm(k, layer.k_norm, self.weights.norm_eps)
        if uses_rope:
            k = self._transformer.kernels.rope(k, self._transformer._cos, self._transformer._sin, position_offset=past)
        keys, values, positions = cache.kv.append(
            index, k, v, positions=np.asarray([past], dtype=np.int64), update_length=False
        )
        context = self._transformer._attention(
            q, keys, values, causal_offset=past,
            key_positions=positions, query_positions=np.asarray([past], dtype=np.int64),
        )
        output = self._transformer._linear(
            context.reshape(1, self.num_heads * head_dim), layer.o_proj, (index, "o_proj"), layer.o_proj_bias
        )
        hidden = hidden + output
        ffn_norm = self._transformer._norm(hidden, layer.ffn_norm, layer.ffn_norm_bias)
        if layer.experts:
            raise ValueError("hybrid MoE attention blocks require a routed hybrid executor")
        if layer.gate_proj is None or layer.down_proj is None:
            raise ValueError(f"hybrid attention layer {index} has incomplete FFN weights")
        gate = self._transformer._linear(ffn_norm, layer.gate_proj, (index, "gate_proj"), layer.gate_proj_bias)
        up = (
            self._transformer._linear(ffn_norm, layer.up_proj, (index, "up_proj"), layer.up_proj_bias)
            if layer.up_proj is not None else None
        )
        return hidden + self._transformer._linear(
            self._transformer._ffn_activation(gate, up),
            layer.down_proj, (index, "down_proj"), layer.down_proj_bias,
        )

    def forward(self, token_ids: np.ndarray, cache: HybridCache | None = None) -> tuple[np.ndarray, HybridCache]:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.weights.vocab_size:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or self._new_cache()
        outputs: list[np.ndarray] = []
        for token in ids:
            past = cache.length
            hidden = self.weights.embedding[int(token)].astype(np.float32, copy=False)[None, :]
            if self.weights.embedding_norm is not None:
                hidden = self._transformer._norm(hidden, self.weights.embedding_norm, self.weights.embedding_norm_bias)
            if self.weights.position_embedding is not None:
                if past >= self.weights.position_embedding.shape[0]:
                    raise ValueError("sequence exceeds the compiled position embedding capacity")
                hidden = hidden + self.weights.position_embedding[past:past + 1]
            for index, layer_type in enumerate(self.layer_types):
                if layer_type == "ssm":
                    hidden = self._mamba._step(hidden, index, cache.ssm)
                else:
                    hidden = self._attention_step(hidden, index, cache, past)
            hidden = self._transformer._norm(hidden, self.weights.final_norm, self.weights.final_norm_bias)
            logits = self._transformer._linear(hidden, self.weights.lm_head)
            outputs.append(logits[0])
            cache.kv.advance(1)
            cache.ssm.length += 1
            cache.length += 1
            cache.last_hidden = hidden[0].copy()
            cache.last_logits = logits[0].copy()
        return np.stack(outputs, axis=0), cache

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float, top_k: int, top_p: float, rng: np.random.Generator) -> int:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if temperature <= 0.0:
            return int(np.argmax(logits))
        values = np.asarray(logits, dtype=np.float64) / float(temperature)
        if top_k > 0:
            k = min(int(top_k), values.size)
            values[values < np.partition(values, -k)[-k]] = -np.inf
        values -= np.max(values)
        probabilities = np.exp(values)
        probabilities /= np.maximum(probabilities.sum(), 1e-12)
        if top_p < 1.0:
            order = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[order])
            keep = cumulative <= top_p
            keep[max(0, int(np.searchsorted(cumulative, top_p)))] = True
            mask = np.zeros(values.size, dtype=bool)
            mask[order[keep]] = True
            probabilities[~mask] = 0.0
            probabilities /= np.maximum(probabilities.sum(), 1e-12)
        return int(rng.choice(probabilities.size, p=probabilities))

    def generate_iter(
        self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
        top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
        seed: int | None = None, cache: HybridCache | None = None,
        cache_callback: Any | None = None, **_: Any,
    ) -> Iterator[int]:
        # Normalized once per request: a checkpoint may declare several stop
        # ids (an instruct model's turn delimiter is often not its eos_token),
        # and every engine must agree on what stopping means.
        stops = stop_token_set(eos_token_id)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        prompt = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        if prompt.size:
            logits, cache = self.forward(prompt, cache)
            next_logits = logits[-1]
        elif cache is not None and cache.last_logits is not None:
            next_logits = cache.last_logits
        else:
            raise ValueError("generation requires prompt ids or a populated cache")
        rng = np.random.default_rng(seed)
        for _ in range(int(max_tokens)):
            token = self._sample(next_logits, temperature, top_k, top_p, rng)
            yield token
            if token in stops:
                break
            _, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
            next_logits = cache.last_logits
        if cache_callback is not None and cache is not None:
            cache_callback(cache)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

    def generate_with_cache(
        self, prompt_ids: np.ndarray, max_tokens: int = 16,
        cache: HybridCache | None = None, **kwargs: Any,
    ) -> tuple[list[int], HybridCache]:
        holder: list[HybridCache | None] = [cache]
        result = list(self.generate_iter(
            prompt_ids, max_tokens=max_tokens, cache=cache,
            cache_callback=lambda value: holder.__setitem__(0, value), **kwargs,
        ))
        if holder[0] is None:
            raise RuntimeError("generation completed without a hybrid cache")
        return result, holder[0]

