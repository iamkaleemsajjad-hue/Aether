"""Reference CPU execution for the RWKV-4/5 time-mix contract.

This is the stable, recurrent WKV formulation used by RWKV checkpoints. The
compiler stores the source parameters under family-neutral ``ssm_*`` keys;
the executor therefore depends on tensor capability, not an RWKV model ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from aether.runtime.stopping import stop_token_set


@dataclass
class RWKVLayerWeights:
    norm: np.ndarray
    ffn_norm: np.ndarray
    time_decay: np.ndarray
    time_first: np.ndarray
    time_mix_k: np.ndarray
    time_mix_v: np.ndarray
    time_mix_r: np.ndarray
    ffn_time_mix_k: np.ndarray
    ffn_time_mix_r: np.ndarray
    key: np.ndarray
    value: np.ndarray
    receptance: np.ndarray
    output: np.ndarray
    ffn_key: np.ndarray
    ffn_value: np.ndarray
    ffn_receptance: np.ndarray


@dataclass
class RWKVModelWeights:
    embedding: np.ndarray
    layers: list[RWKVLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    norm_eps: float = 1e-5


@dataclass
class RWKVCache:
    aa: list[np.ndarray]
    bb: list[np.ndarray]
    pp: list[np.ndarray]
    previous: list[np.ndarray]
    length: int = 0
    last_logits: np.ndarray | None = None

    def clone(self) -> "RWKVCache":
        return RWKVCache(
            aa=[value.copy() for value in self.aa], bb=[value.copy() for value in self.bb],
            pp=[value.copy() for value in self.pp], previous=[value.copy() for value in self.previous],
            length=self.length, last_logits=None if self.last_logits is None else self.last_logits.copy(),
        )


class RWKVExecutionEngine:
    """Execute the numerically stable scalar-channel WKV recurrence."""

    def __init__(self, weights: RWKVModelWeights) -> None:
        self.weights = weights
        hidden = int(weights.embedding.shape[1])
        self.num_heads = 1
        self.num_kv_heads = 1
        self.head_dim = hidden
        self.sparse_attention_plan = None
        self.semantic_kv_plan = None
        self.cross_layer_kv_plan = None

    @staticmethod
    def _norm(value: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        mean = value.mean(axis=-1, keepdims=True)
        var = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
        return (value - mean) / np.sqrt(var + eps) * weight

    @staticmethod
    def _linear(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return value @ np.asarray(weight, dtype=np.float32).T

    def _new_cache(self) -> RWKVCache:
        hidden = int(self.weights.embedding.shape[1])
        return RWKVCache(
            aa=[np.zeros(hidden, dtype=np.float32) for _ in self.weights.layers],
            bb=[np.zeros(hidden, dtype=np.float32) for _ in self.weights.layers],
            pp=[np.full(hidden, -np.inf, dtype=np.float32) for _ in self.weights.layers],
            previous=[np.zeros(hidden, dtype=np.float32) for _ in self.weights.layers],
        )

    def _wkv(self, k: np.ndarray, v: np.ndarray, layer: RWKVLayerWeights, cache: RWKVCache, index: int) -> np.ndarray:
        aa, bb, pp = cache.aa[index], cache.bb[index], cache.pp[index]
        first = np.asarray(layer.time_first, dtype=np.float32).reshape(-1)
        decay = -np.exp(np.asarray(layer.time_decay, dtype=np.float32).reshape(-1))
        # Stable two-term log-sum-exp evaluation of the time-first output.
        p = np.maximum(pp + first, k)
        e1 = np.exp(pp + first - p)
        e2 = np.exp(k - p)
        result = (e1 * aa + e2 * v) / np.maximum(e1 * bb + e2, 1e-30)
        # Persist the decayed key/value accumulator for the next token.
        next_p = np.maximum(pp + decay, k)
        d1 = np.exp(pp + decay - next_p)
        d2 = np.exp(k - next_p)
        cache.aa[index] = d1 * aa + d2 * v
        cache.bb[index] = d1 * bb + d2
        cache.pp[index] = next_p
        return result

    def _step(self, hidden: np.ndarray, index: int, cache: RWKVCache) -> np.ndarray:
        layer = self.weights.layers[index]
        x = self._norm(hidden, layer.norm, self.weights.norm_eps)[0]
        previous = cache.previous[index]
        cache.previous[index] = x.copy()
        mix_k = np.asarray(layer.time_mix_k).reshape(-1)
        mix_v = np.asarray(layer.time_mix_v).reshape(-1)
        mix_r = np.asarray(layer.time_mix_r).reshape(-1)
        xk = x * mix_k + previous * (1.0 - mix_k)
        xv = x * mix_v + previous * (1.0 - mix_v)
        xr = x * mix_r + previous * (1.0 - mix_r)
        r = 1.0 / (1.0 + np.exp(-np.clip(self._linear(xr[None, :], layer.receptance)[0], -60.0, 60.0)))
        k = self._linear(xk[None, :], layer.key)[0]
        v = self._linear(xv[None, :], layer.value)[0]
        mixed = r * self._wkv(k, v, layer, cache, index)
        hidden = hidden + self._linear(mixed[None, :], layer.output)
        ffn_input = self._norm(hidden, layer.ffn_norm, self.weights.norm_eps)[0]
        ffn_k = ffn_input * np.asarray(layer.ffn_time_mix_k).reshape(-1) + previous * (1.0 - np.asarray(layer.ffn_time_mix_k).reshape(-1))
        ffn_r = ffn_input * np.asarray(layer.ffn_time_mix_r).reshape(-1) + previous * (1.0 - np.asarray(layer.ffn_time_mix_r).reshape(-1))
        ffn_gate = np.maximum(self._linear(ffn_k[None, :], layer.ffn_key)[0], 0.0) ** 2
        ffn_value = self._linear(ffn_gate[None, :], layer.ffn_value)[0]
        ffn_receptance = 1.0 / (1.0 + np.exp(-np.clip(self._linear(ffn_r[None, :], layer.ffn_receptance)[0], -60.0, 60.0)))
        return hidden + ffn_receptance[None, :] * ffn_value[None, :]

    def forward(self, token_ids: np.ndarray, cache: RWKVCache | None = None) -> tuple[np.ndarray, RWKVCache]:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.weights.embedding.shape[0]:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or self._new_cache()
        outputs: list[np.ndarray] = []
        for token in ids:
            hidden = self.weights.embedding[int(token)][None, :].astype(np.float32)
            for index in range(len(self.weights.layers)):
                hidden = self._step(hidden, index, cache)
            hidden = self._norm(hidden, self.weights.final_norm, self.weights.norm_eps)
            outputs.append((hidden @ self.weights.lm_head.T)[0])
            cache.length += 1
        logits = np.stack(outputs, axis=0)
        cache.last_logits = logits[-1].copy()
        return logits, cache

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      eos_token_id: int | None = None, cache: RWKVCache | None = None,
                      cache_callback: Any | None = None, **_: Any) -> Iterator[int]:
        # Normalized once per request: a checkpoint may declare several stop
        # ids (an instruct model's turn delimiter is often not its eos_token),
        # and every engine must agree on what stopping means.
        stops = stop_token_set(eos_token_id)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        logits, cache = self.forward(np.asarray(prompt_ids, dtype=np.int64), cache)
        for _ in range(int(max_tokens)):
            if temperature <= 0:
                token = int(np.argmax(logits[-1]))
            else:
                values = logits[-1].astype(np.float64) / float(temperature)
                values -= values.max(); probs = np.exp(values); probs /= probs.sum()
                token = int(np.random.default_rng().choice(values.size, p=probs))
            yield token
            if token in stops:
                break
            logits, cache = self.forward(np.asarray([token]), cache)
        if cache_callback is not None:
            cache_callback(cache)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))
