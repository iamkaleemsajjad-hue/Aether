"""Reference CPU executor for the Mamba-2 structured state-space contract.

The implementation is the token-wise form of the SSD recurrence from
"Transformers are SSMs" (Dao & Gu, 2024).  It deliberately uses the same
canonical tensors as the compiler graph, so a Mamba-2 checkpoint is neither
silently interpreted as a transformer nor dependent on a model-name branch.
The chunked SSD kernel can replace this recurrence on an accelerator without
changing the AEG artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from aether.runtime.stopping import stop_token_set


@dataclass
class Mamba2LayerWeights:
    norm: np.ndarray
    in_proj: np.ndarray
    conv1d: np.ndarray
    a_log: np.ndarray
    d: np.ndarray
    dt: np.ndarray
    out_proj: np.ndarray
    in_proj_bias: np.ndarray | None = None
    conv_bias: np.ndarray | None = None


@dataclass
class Mamba2ModelWeights:
    embedding: np.ndarray
    layers: list[Mamba2LayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    norm_eps: float = 1e-5
    norm_type: str = "RMSNorm"


@dataclass
class Mamba2Cache:
    states: list[np.ndarray]
    conv_history: list[np.ndarray]
    length: int = 0
    last_logits: np.ndarray | None = None

    def clone(self) -> "Mamba2Cache":
        return Mamba2Cache(
            states=[state.copy() for state in self.states],
            conv_history=[history.copy() for history in self.conv_history],
            length=self.length,
            last_logits=None if self.last_logits is None else self.last_logits.copy(),
        )


class Mamba2ExecutionEngine:
    """Execute Mamba-2/SSD tensors with a numerically explicit CPU reference."""

    def __init__(
        self,
        weights: Mamba2ModelWeights,
        *,
        state_size: int,
        inner_size: int,
        num_heads: int,
        num_groups: int,
        conv_kernel: int,
        head_dim: int | None = None,
    ) -> None:
        self.weights = weights
        self.state_size = int(state_size)
        self.inner_size = int(inner_size)
        self.num_heads = int(num_heads)
        self.num_groups = int(num_groups)
        self.conv_kernel = int(conv_kernel)
        self.head_dim = int(head_dim or (inner_size // max(num_heads, 1)))
        if self.num_heads <= 0 or self.num_groups <= 0 or self.num_heads % self.num_groups:
            raise ValueError("Mamba-2 requires positive n_heads/n_groups with n_heads divisible by n_groups")
        if self.num_heads * self.head_dim != self.inner_size:
            raise ValueError("Mamba-2 head geometry does not match the expanded inner dimension")
        self.num_kv_heads = self.num_heads
        self.sparse_attention_plan = None
        self.semantic_kv_plan = None
        self.cross_layer_kv_plan = None

    @staticmethod
    def _silu(value: np.ndarray) -> np.ndarray:
        return value / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))

    def _norm(self, value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if self.weights.norm_type.lower() == "layernorm":
            mean = value.mean(axis=-1, keepdims=True)
            var = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
            return (value - mean) / np.sqrt(var + self.weights.norm_eps) * weight
        return value * (1.0 / np.sqrt((value * value).mean(axis=-1, keepdims=True) + self.weights.norm_eps)) * weight

    def _new_cache(self) -> Mamba2Cache:
        channels = self.inner_size + 2 * self.num_groups * self.state_size
        return Mamba2Cache(
            states=[np.zeros((1, self.num_heads, self.head_dim, self.state_size), dtype=np.float32) for _ in self.weights.layers],
            conv_history=[np.zeros((1, channels, max(self.conv_kernel - 1, 0)), dtype=np.float32) for _ in self.weights.layers],
        )

    def _conv(self, value: np.ndarray, layer: Mamba2LayerWeights, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kernel = np.asarray(layer.conv1d, dtype=np.float32)
        if kernel.ndim == 3:
            kernel = kernel[:, 0, :]
        if kernel.ndim != 2 or kernel.shape[0] != value.shape[1]:
            raise ValueError("Mamba-2 conv1d must have one depthwise filter per xBC channel")
        window = np.concatenate((history, value[:, :, None]), axis=2)
        result = np.sum(kernel[None, :, :] * window, axis=2)
        if layer.conv_bias is not None:
            result = result + np.asarray(layer.conv_bias, dtype=np.float32).reshape(1, -1)
        return self._silu(result), window[:, :, -max(kernel.shape[1] - 1, 0):]

    def _step(self, hidden: np.ndarray, index: int, cache: Mamba2Cache) -> np.ndarray:
        layer = self.weights.layers[index]
        normalized = self._norm(hidden, layer.norm)
        projected = normalized @ layer.in_proj.T
        if layer.in_proj_bias is not None:
            projected = projected + layer.in_proj_bias.reshape(1, -1)
        groups = self.num_groups * self.state_size
        expected = 2 * self.inner_size + 2 * groups + self.num_heads
        if projected.shape[-1] != expected:
            raise ValueError(f"Mamba-2 in_proj has {projected.shape[-1]} outputs; expected {expected}")
        z, xbc, dt = (
            projected[:, : self.inner_size],
            projected[:, self.inner_size : 2 * self.inner_size + 2 * groups],
            projected[:, 2 * self.inner_size + 2 * groups :],
        )
        xbc, cache.conv_history[index] = self._conv(xbc, layer, cache.conv_history[index])
        x = xbc[:, : self.inner_size].reshape(-1, self.num_heads, self.head_dim)
        b = xbc[:, self.inner_size : self.inner_size + groups].reshape(-1, self.num_groups, self.state_size)
        c = xbc[:, self.inner_size + groups :].reshape(-1, self.num_groups, self.state_size)
        repeat = self.num_heads // self.num_groups
        b = np.repeat(b, repeat, axis=1)
        c = np.repeat(c, repeat, axis=1)
        delta = np.log1p(np.exp(np.clip(dt + layer.dt.reshape(1, -1), -60.0, 60.0)))
        a = -np.exp(np.asarray(layer.a_log, dtype=np.float32).reshape(1, -1))
        decay = np.exp(delta[:, :, None, None] * a[:, :, None, None])
        state = cache.states[index]
        state *= decay
        state += delta[:, :, None, None] * x[:, :, :, None] * b[:, :, None, :]
        y = np.sum(state * c[:, :, None, :], axis=-1)
        y += x * np.asarray(layer.d, dtype=np.float32).reshape(1, self.num_heads, 1)
        mixed = y.reshape(-1, self.inner_size) * self._silu(z)
        return hidden + mixed @ layer.out_proj.T

    def forward(self, token_ids: np.ndarray, cache: Mamba2Cache | None = None) -> tuple[np.ndarray, Mamba2Cache]:
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

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      eos_token_id: int | None = None, cache: Mamba2Cache | None = None,
                      cache_callback: Any | None = None, **_: Any) -> Iterator[int]:
        # Normalized once per request: a checkpoint may declare several stop
        # ids (an instruct model's turn delimiter is often not its eos_token),
        # and every engine must agree on what stopping means.
        stops = stop_token_set(eos_token_id)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        logits, cache = self.forward(np.asarray(prompt_ids, dtype=np.int64), cache)
        for _ in range(int(max_tokens)):
            token = int(np.argmax(logits[-1]) if temperature <= 0 else np.random.default_rng().choice(logits.shape[-1], p=self._probabilities(logits[-1], temperature)))
            yield token
            if token in stops:
                break
            logits, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
        if cache_callback is not None:
            cache_callback(cache)

    @staticmethod
    def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
        values = logits.astype(np.float64) / float(temperature)
        values -= values.max()
        probs = np.exp(values)
        return probs / probs.sum()

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

