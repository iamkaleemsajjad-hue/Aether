"""Portable NumPy execution for BERT-style encoder AEG artifacts.

The causal CPU engine cannot execute an encoder graph: encoder attention is
bidirectional, uses LayerNorm/GELU, and exposes pooled embeddings rather than
next-token logits.  This module is deliberately separate so a malformed
decoder artifact can never be interpreted as an encoder (or vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["EncoderLayerWeights", "EncoderModelWeights", "EncoderExecutionEngine"]


@dataclass
class EncoderLayerWeights:
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    attention_norm: np.ndarray
    intermediate_proj: np.ndarray
    output_proj: np.ndarray
    output_norm: np.ndarray
    q_bias: np.ndarray | None = None
    k_bias: np.ndarray | None = None
    v_bias: np.ndarray | None = None
    o_bias: np.ndarray | None = None
    intermediate_bias: np.ndarray | None = None
    output_bias: np.ndarray | None = None
    attention_norm_bias: np.ndarray | None = None
    output_norm_bias: np.ndarray | None = None


@dataclass
class EncoderModelWeights:
    embedding: np.ndarray
    position_embedding: np.ndarray
    token_type_embedding: np.ndarray
    embedding_norm: np.ndarray
    pooler: np.ndarray
    layers: list[EncoderLayerWeights]
    embedding_norm_bias: np.ndarray | None = None
    pooler_bias: np.ndarray | None = None
    norm_eps: float = 1e-12

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])

    def validate(self) -> None:
        if self.embedding.ndim != 2:
            raise ValueError("encoder embedding must be a 2-D matrix")
        if self.position_embedding.ndim != 2 or self.position_embedding.shape[1] != self.hidden_size:
            raise ValueError("position embedding shape does not match encoder hidden size")
        if self.token_type_embedding.ndim != 2 or self.token_type_embedding.shape[1] != self.hidden_size:
            raise ValueError("token-type embedding shape does not match encoder hidden size")
        if self.embedding_norm.size != self.hidden_size:
            raise ValueError("embedding LayerNorm size does not match encoder hidden size")
        if self.pooler.shape != (self.hidden_size, self.hidden_size):
            raise ValueError("pooler matrix shape does not match encoder hidden size")
        for index, layer in enumerate(self.layers):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                value = getattr(layer, name)
                if value.ndim != 2 or value.shape[1] != self.hidden_size:
                    raise ValueError(f"encoder layer {index} {name} has invalid input shape")
            if layer.intermediate_proj.shape[1] != self.hidden_size:
                raise ValueError(f"encoder layer {index} intermediate projection has invalid shape")
            if layer.output_proj.shape[1] != layer.intermediate_proj.shape[0]:
                raise ValueError(f"encoder layer {index} FFN projections do not compose")


class EncoderExecutionEngine:
    """Execute a real BERT-style encoder using persisted AEG tensors."""

    name = "aeg_encoder_cpu"

    def __init__(self, weights: EncoderModelWeights, num_heads: int) -> None:
        if num_heads <= 0 or weights.hidden_size % num_heads:
            raise ValueError("encoder hidden size must be divisible by num_heads")
        self.weights = weights
        self.num_heads = int(num_heads)
        self.head_dim = weights.hidden_size // self.num_heads
        weights.validate()

    @staticmethod
    def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        result = x @ np.asarray(weight, dtype=np.float32).T
        if bias is not None:
            result = result + np.asarray(bias, dtype=np.float32)
        return result

    def _layer_norm(
        self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None
    ) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        result = (x - mean) / np.sqrt(variance + self.weights.norm_eps)
        result = result * np.asarray(weight, dtype=np.float32)
        if bias is not None:
            result = result + np.asarray(bias, dtype=np.float32)
        return result.astype(np.float32, copy=False)

    def encode(
        self,
        input_ids: np.ndarray | list[list[int]] | list[int],
        attention_mask: np.ndarray | list[list[int]] | list[int] | None = None,
        token_type_ids: np.ndarray | list[list[int]] | list[int] | None = None,
    ) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, sequence] array")
        if ids.min() < 0 or ids.max() >= self.weights.vocab_size:
            raise ValueError("input_ids contain a token outside the AEG vocabulary")
        batch, sequence = ids.shape
        mask = np.ones((batch, sequence), dtype=bool) if attention_mask is None else np.asarray(attention_mask, dtype=bool)
        if mask.shape != (batch, sequence):
            raise ValueError("attention_mask shape must match input_ids")
        types = np.zeros((batch, sequence), dtype=np.int64) if token_type_ids is None else np.asarray(token_type_ids, dtype=np.int64)
        if types.shape != (batch, sequence):
            raise ValueError("token_type_ids shape must match input_ids")
        if types.min() < 0 or types.max() >= self.weights.token_type_embedding.shape[0]:
            raise ValueError("token_type_ids contain an invalid segment id")
        if sequence > self.weights.position_embedding.shape[0]:
            raise ValueError("sequence length exceeds the compiled position embedding table")

        positions = self.weights.position_embedding[np.arange(sequence)]
        hidden = (
            self.weights.embedding[ids]
            + positions[None, :, :]
            + self.weights.token_type_embedding[types]
        )
        hidden = self._layer_norm(hidden, self.weights.embedding_norm, self.weights.embedding_norm_bias)
        for layer in self.weights.layers:
            q = self._linear(hidden, layer.q_proj, layer.q_bias).reshape(batch, sequence, self.num_heads, self.head_dim)
            k = self._linear(hidden, layer.k_proj, layer.k_bias).reshape(batch, sequence, self.num_heads, self.head_dim)
            v = self._linear(hidden, layer.v_proj, layer.v_bias).reshape(batch, sequence, self.num_heads, self.head_dim)
            scores = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(self.head_dim)
            scores = np.where(mask[:, None, None, :], scores, -np.inf)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-12)
            attended = np.einsum("bhqk,bkhd->bqhd", probabilities, v).reshape(batch, sequence, -1)
            attention_output = self._linear(attended, layer.o_proj, layer.o_bias)
            hidden = self._layer_norm(hidden + attention_output, layer.attention_norm, layer.attention_norm_bias)
            intermediate = self._linear(hidden, layer.intermediate_proj, layer.intermediate_bias)
            # Exact GELU used by the BERT family (the tanh form is the
            # approximation used by the original PyTorch implementation).
            intermediate = 0.5 * intermediate * (
                1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (intermediate + 0.044715 * intermediate**3))
            )
            output = self._linear(intermediate, layer.output_proj, layer.output_bias)
            hidden = self._layer_norm(hidden + output, layer.output_norm, layer.output_norm_bias)
        return hidden * mask[:, :, None]

    def pooled(self, input_ids: Any, attention_mask: Any = None, token_type_ids: Any = None) -> np.ndarray:
        sequence = self.encode(input_ids, attention_mask, token_type_ids)
        pooled = self._linear(sequence[:, 0, :], self.weights.pooler, self.weights.pooler_bias)
        return np.tanh(pooled).astype(np.float32, copy=False)

    def embed(self, input_ids: Any, attention_mask: Any = None, token_type_ids: Any = None) -> list[list[float]]:
        return self.pooled(input_ids, attention_mask, token_type_ids).tolist()
