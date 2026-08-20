"""Reference T5-style encoder-decoder execution for portable AEG artifacts.

The implementation follows the public T5 computation contract: pre-norm
encoder/decoder blocks, relative position buckets, causal decoder self
attention, encoder-decoder cross attention, and autoregressive decoding. It is
deliberately NumPy-based so a compiled T5/FLAN-T5/mT5/ByT5/UL2 artifact can be
run without importing a model-specific Transformers class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class Seq2SeqLayer:
    norm1: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    o: np.ndarray
    norm2: np.ndarray
    ffn_in: np.ndarray
    ffn_out: np.ndarray
    relative_bias: np.ndarray | None = None
    ffn_in_0: np.ndarray | None = None
    ffn_in_1: np.ndarray | None = None


@dataclass
class Seq2SeqDecoderLayer:
    self_norm: np.ndarray
    self_q: np.ndarray
    self_k: np.ndarray
    self_v: np.ndarray
    self_o: np.ndarray
    cross_norm: np.ndarray
    cross_q: np.ndarray
    cross_k: np.ndarray
    cross_v: np.ndarray
    cross_o: np.ndarray
    ffn_norm: np.ndarray
    ffn_in: np.ndarray
    ffn_out: np.ndarray
    relative_bias: np.ndarray | None = None
    ffn_in_0: np.ndarray | None = None
    ffn_in_1: np.ndarray | None = None


class Seq2SeqExecutionEngine:
    """Execute a T5-compatible encoder-decoder model with real weights."""

    name = "aeg_seq2seq_cpu"

    def __init__(
        self,
        embedding: np.ndarray,
        encoder_layers: list[Seq2SeqLayer],
        decoder_layers: list[Seq2SeqDecoderLayer],
        encoder_final_norm: np.ndarray,
        final_norm: np.ndarray,
        lm_head: np.ndarray,
        *,
        num_heads: int,
        head_dim: int,
        norm_eps: float = 1e-6,
        ffn_type: str = "ReLU",
        tie_word_embeddings: bool = True,
        relative_attention_num_buckets: int = 32,
    ) -> None:
        self.embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_final_norm = np.asarray(encoder_final_norm, dtype=np.float32)
        self.final_norm = np.asarray(final_norm, dtype=np.float32)
        self.lm_head = np.ascontiguousarray(lm_head, dtype=np.float32)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.norm_eps = float(norm_eps)
        self.ffn_type = str(ffn_type).lower()
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.relative_attention_num_buckets = max(0, int(relative_attention_num_buckets))
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("seq2seq attention dimensions must be positive")
        if self.embedding.ndim != 2 or self.embedding.shape[1] != self.hidden_size:
            raise ValueError("seq2seq embedding must have shape [vocab, hidden]")
        if self.lm_head.shape != (self.vocab_size, self.hidden_size):
            raise ValueError("seq2seq lm_head shape does not match embedding vocabulary/hidden size")

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])

    def _norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.norm_eps) * weight

    @staticmethod
    def _linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T

    def _relative_bucket(self, relative_position: np.ndarray, bidirectional: bool) -> np.ndarray:
        """T5 logarithmic relative-position bucketization."""
        buckets = self.relative_attention_num_buckets
        if buckets <= 0:
            return np.zeros_like(relative_position, dtype=np.int64)
        relative = np.asarray(relative_position, dtype=np.int64)
        result = np.zeros_like(relative, dtype=np.int64)
        half = buckets // 2 if bidirectional else buckets
        if bidirectional:
            sign = (relative > 0).astype(np.int64)
            result = sign * half
            relative = np.abs(relative)
        else:
            relative = -np.minimum(relative, 0)
        max_exact = max(1, half // 2)
        is_small = relative < max_exact
        large = max_exact + (
            np.log(np.maximum(relative, max_exact) / max_exact)
            / np.log(max(2.0, 128.0 / max_exact))
            * max(1, half - max_exact)
        ).astype(np.int64)
        result += np.where(is_small, relative, np.minimum(large, half - 1))
        return result.astype(np.int64)

    def _attention(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
        *, query_positions: np.ndarray, key_positions: np.ndarray,
        relative_bias: np.ndarray | None, bidirectional: bool,
    ) -> np.ndarray:
        qh = q.reshape(q.shape[0], self.num_heads, self.head_dim).transpose(1, 0, 2)
        kh = k.reshape(k.shape[0], self.num_heads, self.head_dim).transpose(1, 0, 2)
        vh = v.reshape(v.shape[0], self.num_heads, self.head_dim).transpose(1, 0, 2)
        scores = np.einsum("hqd,hkd->hqk", qh, kh) / np.sqrt(self.head_dim)
        allowed = np.ones((query_positions.size, key_positions.size), dtype=bool)
        if not bidirectional:
            allowed = key_positions[None, :] <= query_positions[:, None]
        if relative_bias is not None and relative_bias.ndim == 2:
            buckets = self._relative_bucket(
                key_positions[None, :] - query_positions[:, None], bidirectional
            )
            valid = buckets < relative_bias.shape[0]
            bias = np.zeros((self.num_heads, query_positions.size, key_positions.size), dtype=np.float32)
            for head in range(min(self.num_heads, relative_bias.shape[1])):
                selected = np.zeros_like(buckets, dtype=np.float32)
                selected[valid] = relative_bias[buckets[valid], head]
                bias[head] = selected
            scores += bias
        scores = np.where(allowed[None, :, :], scores, -np.inf)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-12)
        context = np.einsum("hqk,hkd->hqd", probabilities, vh)
        return context.transpose(1, 0, 2).reshape(q.shape[0], -1)

    def _ffn(self, x: np.ndarray, layer: Seq2SeqLayer | Seq2SeqDecoderLayer) -> np.ndarray:
        first = getattr(layer, "ffn_in_0", None)
        second = getattr(layer, "ffn_in_1", None)
        if first is not None and second is not None:
            activated = self._gelu(self._linear(x, first)) * self._linear(x, second)
        else:
            activated = self._activation(self._linear(x, layer.ffn_in))
        return self._linear(activated, layer.ffn_out)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        if self.ffn_type in {"gelu", "gatedgelu", "geglu"}:
            return self._gelu(x)
        if self.ffn_type in {"relu", "relu2"}:
            value = np.maximum(x, 0.0)
            return value * value if self.ffn_type == "relu2" else value
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def encode(self, input_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("encoder input IDs are empty or outside the compiled vocabulary")
        hidden = self.embedding[ids]
        positions = np.arange(ids.size, dtype=np.int64)
        for layer in self.encoder_layers:
            normed = self._norm(hidden, layer.norm1)
            context = self._attention(
                self._linear(normed, layer.q), self._linear(normed, layer.k), self._linear(normed, layer.v),
                query_positions=positions, key_positions=positions,
                relative_bias=layer.relative_bias, bidirectional=True,
            )
            hidden = hidden + self._linear(context, layer.o)
            hidden = hidden + self._ffn(self._norm(hidden, layer.norm2), layer)
        return self._norm(hidden, self.encoder_final_norm)

    def decode(self, decoder_ids: np.ndarray, encoder_hidden: np.ndarray) -> np.ndarray:
        ids = np.asarray(decoder_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("decoder input IDs are empty or outside the compiled vocabulary")
        hidden = self.embedding[ids]
        positions = np.arange(ids.size, dtype=np.int64)
        encoder_positions = np.arange(encoder_hidden.shape[0], dtype=np.int64)
        for layer in self.decoder_layers:
            normed = self._norm(hidden, layer.self_norm)
            context = self._attention(
                self._linear(normed, layer.self_q), self._linear(normed, layer.self_k), self._linear(normed, layer.self_v),
                query_positions=positions, key_positions=positions,
                relative_bias=layer.relative_bias, bidirectional=False,
            )
            hidden = hidden + self._linear(context, layer.self_o)
            normed = self._norm(hidden, layer.cross_norm)
            cross = self._attention(
                self._linear(normed, layer.cross_q), self._linear(encoder_hidden, layer.cross_k), self._linear(encoder_hidden, layer.cross_v),
                query_positions=positions, key_positions=encoder_positions,
                relative_bias=None, bidirectional=True,
            )
            hidden = hidden + self._linear(cross, layer.cross_o)
            hidden = hidden + self._ffn(self._norm(hidden, layer.ffn_norm), layer)
        hidden = self._norm(hidden, self.final_norm)
        if self.tie_word_embeddings:
            hidden = hidden * (self.hidden_size ** -0.5)
        return self._linear(hidden, self.lm_head)

    def forward(self, token_ids: np.ndarray, **_: Any) -> tuple[np.ndarray, None]:
        encoder_hidden = self.encode(token_ids)
        decoder_ids = np.asarray([0], dtype=np.int64)
        return self.decode(decoder_ids, encoder_hidden), None

    def generate_iter(
        self, prompt_ids: np.ndarray, max_tokens: int = 16,
        temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
        eos_token_id: int | None = None, cache_callback: Any | None = None, **_: Any,
    ) -> Iterator[int]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        encoder_hidden = self.encode(prompt_ids)
        decoder_ids = [0]
        for _ in range(int(max_tokens)):
            logits = self.decode(np.asarray(decoder_ids, dtype=np.int64), encoder_hidden)[-1]
            if temperature <= 0:
                token = int(np.argmax(logits))
            else:
                scaled = logits / float(temperature)
                if top_k > 0:
                    keep = np.argpartition(scaled, -min(top_k, scaled.size))[-min(top_k, scaled.size):]
                    mask = np.ones(scaled.shape, dtype=bool)
                    mask[keep] = False
                    scaled[mask] = -np.inf
                probabilities = np.exp(scaled - np.max(scaled))
                probabilities /= np.maximum(probabilities.sum(), 1e-12)
                token = int(np.random.default_rng().choice(scaled.size, p=probabilities))
            decoder_ids.append(token)
            yield token
            if eos_token_id is not None and token == int(eos_token_id):
                break
        if cache_callback is not None:
            cache_callback(None)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

    def generate_with_cache(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> tuple[list[int], None]:
        return self.generate(prompt_ids, max_tokens=max_tokens, **kwargs), None

    def speculative_stats(self) -> dict[str, int]:
        return {"draft_tokens": 0, "accepted_tokens": 0, "cycles": 0}

