"""Reference CPU execution for Mamba-1 selective-scan AEG artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from aether.hybrid.state import MambaSSM, MambaState


@dataclass
class MambaLayerWeights:
    norm: np.ndarray
    in_proj: np.ndarray
    conv1d: np.ndarray
    x_proj: np.ndarray
    dt_proj: np.ndarray
    a_log: np.ndarray
    d: np.ndarray
    out_proj: np.ndarray
    conv_bias: np.ndarray | None = None
    dt_bias: np.ndarray | None = None


@dataclass
class MambaModelWeights:
    embedding: np.ndarray
    layers: list[MambaLayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    norm_eps: float = 1e-5
    norm_type: str = "RMSNorm"


@dataclass
class MambaCache:
    states: list[MambaState]
    conv_history: list[np.ndarray]
    length: int = 0
    last_logits: np.ndarray | None = None

    def clone(self) -> "MambaCache":
        return MambaCache(
            states=[state.copy() for state in self.states],
            conv_history=[history.copy() for history in self.conv_history],
            length=self.length,
            last_logits=None if self.last_logits is None else self.last_logits.copy(),
        )


class MambaExecutionEngine:
    """Execute the selective-scan recurrence with real checkpoint tensors."""

    def __init__(self, weights: MambaModelWeights, state_size: int, inner_size: int, dt_rank: int, conv_kernel: int) -> None:
        self.weights = weights
        self.state_size = int(state_size)
        self.inner_size = int(inner_size)
        self.dt_rank = int(dt_rank)
        self.conv_kernel = int(conv_kernel)
        self._scan = MambaSSM(
            d_model=int(weights.embedding.shape[1]), d_state=self.state_size, d_inner=self.inner_size
        )
        self.num_heads = 1
        self.num_kv_heads = 1
        self.head_dim = int(weights.embedding.shape[1])
        self.sparse_attention_plan = None
        self.semantic_kv_plan = None
        self.cross_layer_kv_plan = None

    def _norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if self.weights.norm_type.lower() == "layernorm":
            mean = x.mean(axis=-1, keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(variance + self.weights.norm_eps) * weight
        return x * (1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.weights.norm_eps)) * weight

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def _conv(self, x: np.ndarray, layer: MambaLayerWeights, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # HF Mamba stores depthwise Conv1d as (inner, 1, kernel) or
        # (inner, kernel).  Convert both to the channel-wise causal equation.
        kernel = np.asarray(layer.conv1d, dtype=np.float32)
        if kernel.ndim == 3:
            kernel = kernel[:, 0, :]
        if kernel.ndim != 2 or kernel.shape[0] != self.inner_size:
            raise ValueError("Mamba conv1d weight must have shape (d_inner, kernel)")
        window = np.concatenate([history, x[:, :, None]], axis=2)
        result = np.sum(kernel[None, :, :] * window, axis=2)
        if layer.conv_bias is not None:
            result = result + layer.conv_bias
        return self._silu(result), window[:, :, -max(kernel.shape[1] - 1, 0):]

    def _step(self, hidden: np.ndarray, index: int, cache: MambaCache) -> np.ndarray:
        layer = self.weights.layers[index]
        x = self._norm(hidden, layer.norm)
        projected = x @ layer.in_proj.T
        ssm_input, gate = np.split(projected, 2, axis=-1)
        conv_input, cache.conv_history[index] = self._conv(
            ssm_input, layer, cache.conv_history[index]
        )
        selective = conv_input @ layer.x_proj.T
        dt_raw = selective[:, : self.dt_rank]
        b = selective[:, self.dt_rank : self.dt_rank + self.state_size]
        c = selective[:, self.dt_rank + self.state_size :]
        dt = dt_raw @ layer.dt_proj.T
        if layer.dt_bias is not None:
            dt = dt + layer.dt_bias
        a = -np.exp(np.asarray(layer.a_log, dtype=np.float32))
        y, cache.states[index] = self._scan.step(
            conv_input, cache.states[index], a, b, c,
            np.asarray(layer.d, dtype=np.float32).reshape(-1), dt,
        )
        return hidden + y * self._silu(gate) @ layer.out_proj.T

    def _new_cache(self) -> MambaCache:
        return MambaCache(
            states=[MambaState(
                layer_idx=index,
                h=np.zeros((1, self.state_size, self.inner_size), dtype=np.float32),
                last_x=np.zeros((1, self.inner_size), dtype=np.float32),
            ) for index in range(len(self.weights.layers))],
            conv_history=[np.zeros((1, self.inner_size, max(self.conv_kernel - 1, 0)), dtype=np.float32) for _ in self.weights.layers],
        )

    def forward(self, token_ids: np.ndarray, cache: MambaCache | None = None) -> tuple[np.ndarray, MambaCache]:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.weights.embedding.shape[0]:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or self._new_cache()
        outputs: list[np.ndarray] = []
        for token in ids:
            hidden = self.weights.embedding[int(token)].astype(np.float32)[None, :]
            for index in range(len(self.weights.layers)):
                hidden = self._step(hidden, index, cache)
            hidden = self._norm(hidden, self.weights.final_norm)
            outputs.append((hidden @ self.weights.lm_head.T)[0])
            cache.length += 1
        logits = np.stack(outputs, axis=0)
        cache.last_logits = logits[-1].copy()
        return logits, cache

    def _sample(self, logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))
        values = logits.astype(np.float64) / float(temperature)
        if top_k > 0:
            threshold = np.partition(values, -min(top_k, values.size))[-min(top_k, values.size)]
            values[values < threshold] = -np.inf
        values -= np.max(values)
        probs = np.exp(values); probs /= probs.sum()
        if 0 < top_p < 1:
            order = np.argsort(-probs); cumulative = np.cumsum(probs[order])
            probs[order[cumulative - probs[order] > top_p]] = 0.0; probs /= probs.sum()
        return int(np.random.default_rng().choice(values.size, p=probs))

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: MambaCache | None = None, cache_callback: Any | None = None, **_: Any) -> Iterator[int]:
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
            token = self._sample(next_logits, temperature, top_k, top_p); yield token
            if eos_token_id is not None and token == int(eos_token_id): break
            _, cache = self.forward(np.asarray([token]), cache); next_logits = cache.last_logits
        if cache_callback is not None and cache is not None: cache_callback(cache)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

    def generate_with_cache(self, prompt_ids: np.ndarray, max_tokens: int = 16, cache: MambaCache | None = None, **kwargs: Any) -> tuple[list[int], MambaCache]:
        holder: list[MambaCache | None] = [cache]
        result = list(self.generate_iter(prompt_ids, max_tokens=max_tokens, cache=cache,
                                         cache_callback=lambda value: holder.__setitem__(0, value), **kwargs))
        if holder[0] is None: raise RuntimeError("generation completed without a Mamba cache")
        return result, holder[0]
