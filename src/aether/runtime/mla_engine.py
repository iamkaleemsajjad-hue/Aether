"""Reference CPU executor for the model-generic MLA AEG contract.

The implementation follows the DeepSeek-V2 MLA equations through the shared
``MLAAttention`` reference kernel.  It deliberately keeps a replayable token
cache for the first integration: this preserves exact semantics for prefill
and decode while a target-specific latent-cache kernel can later replace the
replay path without changing the AEG contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from aether.attention.mla import MLAConfig, MLAAttention


@dataclass
class MLAExpertWeights:
    """One routed MLA FFN expert in checkpoint matrix orientation."""

    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass
class MLALayerWeights:
    attention_norm: np.ndarray
    ffn_norm: np.ndarray
    o_proj: np.ndarray
    ffn_in: np.ndarray
    ffn_out: np.ndarray
    ffn_up: np.ndarray | None
    mla: dict[str, np.ndarray]
    router: np.ndarray | None = None
    experts: list[MLAExpertWeights] = field(default_factory=list)
    num_activated_experts: int = 1


@dataclass
class MLAModelWeights:
    embedding: np.ndarray
    layers: list[MLALayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    norm_eps: float = 1e-5
    norm_type: str = "RMSNorm"
    ffn_type: str = "SwiGLU"


@dataclass
class MLACache:
    token_ids: list[int] = field(default_factory=list)
    last_logits: np.ndarray | None = None

    @property
    def length(self) -> int:
        return len(self.token_ids)


class MLAExecutionEngine:
    """Execute an authenticated MLA decoder on the host CPU."""

    def __init__(self, weights: MLAModelWeights, config: MLAConfig) -> None:
        self.weights = weights
        self.config = config
        self.attention = MLAAttention(config)
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.total_head_dim
        self.sparse_attention_plan = None
        self.semantic_kv_plan = None
        self.cross_layer_kv_plan = None

    def _norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if self.weights.norm_type.lower() == "layernorm":
            mean = x.mean(axis=-1, keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(variance + self.weights.norm_eps) * weight
        return x * (1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.weights.norm_eps)) * weight

    def _ffn(self, x: np.ndarray, layer: MLALayerWeights) -> np.ndarray:
        if layer.experts:
            if layer.router is None:
                raise ValueError("MLA MoE layer is missing its router")
            router_logits = x @ np.asarray(layer.router, dtype=np.float32).T
            top_k = min(max(1, int(layer.num_activated_experts)), len(layer.experts))
            selected = np.argpartition(router_logits, -top_k, axis=-1)[:, -top_k:]
            selected_logits = np.take_along_axis(router_logits, selected, axis=-1)
            selected_logits -= selected_logits.max(axis=-1, keepdims=True)
            weights = np.exp(selected_logits)
            weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
            result = np.zeros_like(x, dtype=np.float32)
            for expert_index, expert in enumerate(layer.experts):
                positions = np.where(selected == expert_index)
                if positions[0].size == 0:
                    continue
                tokens = x[positions[0]]
                gate = tokens @ expert.gate_proj.T
                up = tokens @ expert.up_proj.T
                value = (1.0 / (1.0 + np.exp(-gate)) * gate) * up
                output = value @ expert.down_proj.T
                result[positions[0]] += output * weights[positions[0], positions[1], None]
            return result
        first = x @ layer.ffn_in.T
        kind = self.weights.ffn_type.lower()
        if layer.ffn_up is None:
            if kind == "relu":
                activated = np.maximum(first, 0.0)
            elif kind in {"gelu", "geglu"}:
                activated = 0.5 * first * (1.0 + np.tanh(
                    np.sqrt(2.0 / np.pi) * (first + 0.044715 * first ** 3)
                ))
            else:
                activated = 1.0 / (1.0 + np.exp(-first)) * first
        elif kind in {"gelu", "geglu"}:
            gelu = 0.5 * first * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (first + 0.044715 * first ** 3)
            ))
            activated = gelu * (x @ layer.ffn_up.T)
        else:
            activated = (1.0 / (1.0 + np.exp(-first)) * first) * (x @ layer.ffn_up.T)
        return activated @ layer.ffn_out.T

    def forward(
        self, token_ids: np.ndarray, cache: MLACache | None = None
    ) -> tuple[np.ndarray, MLACache]:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.weights.embedding.shape[0]:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or MLACache()
        all_ids = np.asarray(cache.token_ids + ids.tolist(), dtype=np.int64)
        hidden = self.weights.embedding[all_ids].astype(np.float32, copy=False)[None, :, :]
        for layer in self.weights.layers:
            normed = self._norm(hidden, layer.attention_norm)
            attn_weights = dict(layer.mla)
            attn_weights["o_proj.weight"] = layer.o_proj
            attention_out = self.attention.forward_prefill(
                normed, attn_weights, layer_prefix="", position_offset=0
            )
            hidden = hidden + attention_out
            hidden = hidden + self._ffn(self._norm(hidden, layer.ffn_norm)[0], layer)[None, :, :]
        normalized = self._norm(hidden, self.weights.final_norm)
        logits = normalized[0] @ self.weights.lm_head.T
        cache.token_ids = all_ids.tolist()
        cache.last_logits = logits[-1].copy()
        return logits[-ids.size :], cache

    def _sample(self, logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> int:
        values = logits.astype(np.float64, copy=True)
        if temperature <= 0:
            return int(np.argmax(values))
        values /= float(temperature)
        if top_k > 0:
            keep = min(int(top_k), values.size)
            threshold = np.partition(values, -keep)[-keep]
            values[values < threshold] = -np.inf
        values -= np.max(values)
        probs = np.exp(values)
        probs /= probs.sum()
        if 0 < top_p < 1:
            order = np.argsort(-probs)
            cumulative = np.cumsum(probs[order])
            remove = cumulative - probs[order] > top_p
            probs[order[remove]] = 0.0
            probs /= probs.sum()
        return int(np.random.default_rng().choice(values.size, p=probs))

    def generate_iter(
        self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
        top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
        cache: MLACache | None = None, cache_callback: Any | None = None, **_: Any,
    ) -> Iterator[int]:
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
        for _ in range(int(max_tokens)):
            token = self._sample(next_logits, temperature, top_k, top_p)
            yield token
            if eos_token_id is not None and token == int(eos_token_id):
                break
            _, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
            next_logits = cache.last_logits
        if cache_callback is not None and cache is not None:
            cache_callback(cache)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

    def generate_with_cache(self, prompt_ids: np.ndarray, max_tokens: int = 16, cache: MLACache | None = None, **kwargs: Any) -> tuple[list[int], MLACache]:
        holder: list[MLACache | None] = [cache]
        result = list(self.generate_iter(
            prompt_ids, max_tokens=max_tokens, cache=cache,
            cache_callback=lambda value: holder.__setitem__(0, value), **kwargs,
        ))
        if holder[0] is None:
            raise RuntimeError("generation completed without an MLA cache")
        return result, holder[0]
