"""Portable PyTorch executors for encoder and encoder-decoder AEGs.

These implementations mirror the model-generic NumPy reference engines.  They
keep authenticated AEG tensors on the selected PyTorch device and intentionally
avoid importing a model family's Transformers class.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from aether.runtime.torch_engine import _resolve_device
from aether.runtime.stopping import stop_token_set


class TorchEncoderAEGEngine:
    """Device-resident BERT-style bidirectional encoder."""

    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for portable encoder execution") from exc
        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.devices = [_resolve_device(torch, value) for value in (devices or [device])]
        self.source_engine = source_engine
        self.weights = source_engine.weights
        self.num_heads = int(source_engine.num_heads)
        self.head_dim = int(source_engine.head_dim)
        self.layer_devices = [self.devices[index % len(self.devices)] for index in range(len(self.weights.layers))]
        self.embedding = self._tensor(self.weights.embedding)
        self.position_embedding = self._tensor(self.weights.position_embedding)
        self.token_type_embedding = self._tensor(self.weights.token_type_embedding)
        self.embedding_norm = self._tensor(self.weights.embedding_norm)
        self.embedding_norm_bias = self._optional(self.weights.embedding_norm_bias)
        self.pooler = self._tensor(self.weights.pooler)
        self.pooler_bias = self._optional(self.weights.pooler_bias)
        self.layers: list[dict[str, Any | None]] = []
        for index, layer in enumerate(self.weights.layers):
            layer_device = self.layer_devices[index]
            self.layers.append({
                name: self._tensor(getattr(layer, name), layer_device)
                for name in (
                    "q_proj", "k_proj", "v_proj", "o_proj", "attention_norm",
                    "intermediate_proj", "output_proj", "output_norm",
                )
            } | {
                f"{name}_bias": self._optional(getattr(layer, f"{name}_bias", None), layer_device)
                for name in (
                    "q_proj", "k_proj", "v_proj", "o_proj", "intermediate_proj",
                    "output_proj", "attention_norm", "output_norm",
                )
            })

    def _tensor(self, value: Any, device: Any | None = None) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device or self.device)

    def _optional(self, value: Any | None, device: Any | None = None) -> Any | None:
        return None if value is None else self._tensor(value, device)

    def _linear(self, value: Any, weight: Any, bias: Any | None = None) -> Any:
        return self.torch.nn.functional.linear(value, weight, bias)

    def _norm(self, value: Any, weight: Any, bias: Any | None = None) -> Any:
        mean = value.mean(dim=-1, keepdim=True)
        variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
        result = (value - mean) / self.torch.sqrt(variance + float(self.weights.norm_eps)) * weight
        return result if bias is None else result + bias

    def encode(
        self, input_ids: Any, attention_mask: Any | None = None,
        token_type_ids: Any | None = None,
    ) -> np.ndarray:
        torch = self.torch
        ids = torch.as_tensor(np.asarray(input_ids, dtype=np.int64), device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, sequence] array")
        if int(ids.min()) < 0 or int(ids.max()) >= self.embedding.shape[0]:
            raise ValueError("input_ids contain a token outside the AEG vocabulary")
        batch, sequence = ids.shape
        mask = torch.ones((batch, sequence), dtype=torch.bool, device=self.device)
        if attention_mask is not None:
            mask = torch.as_tensor(np.asarray(attention_mask, dtype=bool), device=self.device)
            if tuple(mask.shape) != (batch, sequence):
                raise ValueError("attention_mask shape must match input_ids")
        types = torch.zeros((batch, sequence), dtype=torch.long, device=self.device)
        if token_type_ids is not None:
            types = torch.as_tensor(np.asarray(token_type_ids, dtype=np.int64), device=self.device)
            if tuple(types.shape) != (batch, sequence):
                raise ValueError("token_type_ids shape must match input_ids")
            if int(types.min()) < 0 or int(types.max()) >= self.token_type_embedding.shape[0]:
                raise ValueError("token_type_ids contain an invalid segment id")
        if sequence > self.position_embedding.shape[0]:
            raise ValueError("sequence length exceeds the compiled position embedding table")
        positions = self.position_embedding[:sequence].unsqueeze(0)
        hidden = self.embedding.index_select(0, ids.reshape(-1)).reshape(batch, sequence, -1)
        hidden = hidden + positions + self.token_type_embedding.index_select(0, types.reshape(-1)).reshape(batch, sequence, -1)
        with torch.no_grad():
            hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
            for index, layer in enumerate(self.layers):
                hidden = hidden.to(self.layer_devices[index])
                layer_mask = mask.to(self.layer_devices[index])
                q = self._linear(hidden, layer["q_proj"], layer["q_proj_bias"]).reshape(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
                k = self._linear(hidden, layer["k_proj"], layer["k_proj_bias"]).reshape(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
                v = self._linear(hidden, layer["v_proj"], layer["v_proj_bias"]).reshape(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
                scores = scores.masked_fill(~layer_mask[:, None, None, :], -torch.inf)
                attended = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(batch, sequence, -1)
                attention = self._linear(attended, layer["o_proj"], layer["o_proj_bias"])
                hidden = self._norm(hidden + attention, layer["attention_norm"], layer["attention_norm_bias"])
                intermediate = self._linear(hidden, layer["intermediate_proj"], layer["intermediate_proj_bias"])
                intermediate = 0.5 * intermediate * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (intermediate + 0.044715 * intermediate.pow(3))))
                output = self._linear(intermediate, layer["output_proj"], layer["output_proj_bias"])
                hidden = self._norm(hidden + output, layer["output_norm"], layer["output_norm_bias"])
        return (hidden.to(self.device) * mask.to(self.device)[:, :, None]).float().cpu().numpy()

    def pooled(self, input_ids: Any, attention_mask: Any | None = None, token_type_ids: Any | None = None) -> np.ndarray:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        value = self.torch.as_tensor(hidden[:, 0, :], device=self.device)
        with self.torch.no_grad():
            pooled = self.torch.tanh(self._linear(value, self.pooler, self.pooler_bias))
        return pooled.float().cpu().numpy()

    def embed(self, input_ids: Any, attention_mask: Any | None = None, token_type_ids: Any | None = None) -> list[list[float]]:
        return self.pooled(input_ids, attention_mask, token_type_ids).tolist()


class TorchSeq2SeqAEGEngine:
    """Device-resident T5-style encoder-decoder executor."""

    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for portable seq2seq execution") from exc
        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.devices = [_resolve_device(torch, value) for value in (devices or [device])]
        self.source_engine = source_engine
        self.embedding = self._tensor(source_engine.embedding)
        self.encoder_final_norm = self._tensor(source_engine.encoder_final_norm)
        self.final_norm = self._tensor(source_engine.final_norm)
        self.lm_head = self._tensor(source_engine.lm_head)
        self.num_heads = int(source_engine.num_heads)
        self.head_dim = int(source_engine.head_dim)
        self.norm_eps = float(source_engine.norm_eps)
        self.ffn_type = str(source_engine.ffn_type).lower()
        self.tie_word_embeddings = bool(source_engine.tie_word_embeddings)
        self.relative_attention_num_buckets = int(source_engine.relative_attention_num_buckets)
        all_layers = list(source_engine.encoder_layers) + list(source_engine.decoder_layers)
        self.layer_devices = [self.devices[index % len(self.devices)] for index in range(len(all_layers))]
        self.encoder_layers = [self._convert_layer(layer, self.layer_devices[index]) for index, layer in enumerate(source_engine.encoder_layers)]
        decoder_offset = len(self.encoder_layers)
        self.decoder_layers = [self._convert_layer(layer, self.layer_devices[decoder_offset + index]) for index, layer in enumerate(source_engine.decoder_layers)]

    def _tensor(self, value: Any, device: Any | None = None) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device or self.device)

    def _optional(self, value: Any | None, device: Any | None = None) -> Any | None:
        return None if value is None else self._tensor(value, device)

    def _convert_layer(self, layer: Any, device: Any) -> dict[str, Any | None]:
        result = {name: self._tensor(getattr(layer, name), device) for name in layer.__dataclass_fields__ if getattr(layer, name) is not None}
        for name in ("relative_bias", "ffn_in_0", "ffn_in_1"):
            result[name] = self._optional(getattr(layer, name, None), device)
        return result

    def _norm(self, x: Any, weight: Any) -> Any:
        return x * self.torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps) * weight

    def _linear(self, x: Any, weight: Any) -> Any:
        return self.torch.nn.functional.linear(x, weight)

    def _relative_bucket(self, relative: Any, bidirectional: bool) -> Any:
        buckets = self.relative_attention_num_buckets
        if buckets <= 0:
            return self.torch.zeros_like(relative, dtype=self.torch.long)
        result = self.torch.zeros_like(relative, dtype=self.torch.long)
        half = buckets // 2 if bidirectional else buckets
        if bidirectional:
            result = (relative > 0).to(self.torch.long) * half
            relative = relative.abs()
        else:
            relative = -self.torch.minimum(relative, self.torch.zeros_like(relative))
        max_exact = max(1, half // 2)
        is_small = relative < max_exact
        large = max_exact + (
            self.torch.log(self.torch.maximum(relative, self.torch.tensor(max_exact, device=relative.device, dtype=relative.dtype)) / max_exact)
            / np.log(max(2.0, 128.0 / max_exact)) * max(1, half - max_exact)
        ).to(self.torch.long)
        return result + self.torch.where(is_small, relative, self.torch.minimum(large, torch_full_like(self.torch, half - 1, relative)))

    def _attention(self, q: Any, k: Any, v: Any, query_positions: Any, key_positions: Any, relative_bias: Any | None, bidirectional: bool) -> Any:
        torch = self.torch
        qh = q.reshape(q.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        kh = k.reshape(k.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        vh = v.reshape(v.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / np.sqrt(self.head_dim)
        query_positions = query_positions.to(q.device)
        key_positions = key_positions.to(q.device)
        allowed = torch.ones((query_positions.numel(), key_positions.numel()), dtype=torch.bool, device=q.device)
        if not bidirectional:
            allowed = key_positions[None, :] <= query_positions[:, None]
        if relative_bias is not None and relative_bias.ndim == 2:
            buckets = self._relative_bucket(key_positions[None, :] - query_positions[:, None], bidirectional)
            bias = torch.zeros((self.num_heads, query_positions.numel(), key_positions.numel()), device=q.device)
            valid = buckets < relative_bias.shape[0]
            for head in range(min(self.num_heads, int(relative_bias.shape[1]))):
                selected = torch.zeros_like(buckets, dtype=torch.float32)
                selected = torch.where(valid, relative_bias[buckets.clamp(min=0), head], selected)
                bias[head] = selected
            scores = scores + bias
        scores = scores.masked_fill(~allowed[None, :, :], -torch.inf)
        context = torch.matmul(torch.softmax(scores, dim=-1), vh)
        return context.transpose(0, 1).reshape(q.shape[0], -1)

    def _activation(self, x: Any) -> Any:
        if self.ffn_type in {"gelu", "gatedgelu", "geglu"}:
            return 0.5 * x * (1.0 + self.torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))
        if self.ffn_type in {"relu", "relu2"}:
            value = self.torch.relu(x)
            return value * value if self.ffn_type == "relu2" else value
        return x

    def _ffn(self, x: Any, layer: dict[str, Any | None]) -> Any:
        if layer.get("ffn_in_0") is not None and layer.get("ffn_in_1") is not None:
            value = self._activation(self._linear(x, layer["ffn_in_0"])) * self._linear(x, layer["ffn_in_1"])
        else:
            value = self._activation(self._linear(x, layer["ffn_in"]))
        return self._linear(value, layer["ffn_out"])

    def _encode(self, ids: Any) -> Any:
        hidden = self.embedding.index_select(0, ids)
        positions = self.torch.arange(ids.numel(), device=self.device)
        for index, layer in enumerate(self.encoder_layers):
            hidden = hidden.to(self.layer_devices[index])
            normed = self._norm(hidden, layer["norm1"])
            context = self._attention(self._linear(normed, layer["q"]), self._linear(normed, layer["k"]), self._linear(normed, layer["v"]), positions, positions, layer.get("relative_bias"), True)
            hidden = hidden + self._linear(context, layer["o"])
            hidden = hidden + self._ffn(self._norm(hidden, layer["norm2"]), layer)
        return self._norm(hidden.to(self.device), self.encoder_final_norm)

    def _decode(self, ids: Any, encoder_hidden: Any) -> Any:
        hidden = self.embedding.index_select(0, ids)
        positions = self.torch.arange(ids.numel(), device=self.device)
        encoder_positions = self.torch.arange(encoder_hidden.shape[0], device=self.device)
        decoder_offset = len(self.encoder_layers)
        for index, layer in enumerate(self.decoder_layers):
            hidden = hidden.to(self.layer_devices[decoder_offset + index])
            encoder_hidden = encoder_hidden.to(hidden.device)
            normed = self._norm(hidden, layer["self_norm"])
            context = self._attention(self._linear(normed, layer["self_q"]), self._linear(normed, layer["self_k"]), self._linear(normed, layer["self_v"]), positions, positions, layer.get("relative_bias"), False)
            hidden = hidden + self._linear(context, layer["self_o"])
            normed = self._norm(hidden, layer["cross_norm"])
            cross = self._attention(self._linear(normed, layer["cross_q"]), self._linear(encoder_hidden, layer["cross_k"]), self._linear(encoder_hidden, layer["cross_v"]), positions, encoder_positions, None, True)
            hidden = hidden + self._linear(cross, layer["cross_o"])
            hidden = hidden + self._ffn(self._norm(hidden, layer["ffn_norm"]), layer)
        hidden = self._norm(hidden.to(self.device), self.final_norm)
        if self.tie_word_embeddings:
            hidden = hidden * (self.embedding.shape[1] ** -0.5)
        return self._linear(hidden, self.lm_head)

    def forward(self, token_ids: Any, **_: Any) -> tuple[np.ndarray, None]:
        ids = self.torch.as_tensor(np.asarray(token_ids, dtype=np.int64).reshape(-1), device=self.device)
        if ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= self.embedding.shape[0]:
            raise ValueError("encoder input IDs are empty or outside the compiled vocabulary")
        with self.torch.no_grad():
            logits = self._decode(self.torch.zeros(1, dtype=self.torch.long, device=self.device), self._encode(ids))
        return logits.float().cpu().numpy(), None

    def generate(self, prompt_ids: Any, max_tokens: int = 16, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None, grammar_session: Any | None = None, **_: Any) -> list[int]:
        # Normalized once per request: a checkpoint may declare several stop
        # ids (an instruct model's turn delimiter is often not its eos_token),
        # and every engine must agree on what stopping means.
        stops = stop_token_set(eos_token_id)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        prompt = self.torch.as_tensor(np.asarray(prompt_ids, dtype=np.int64).reshape(-1), device=self.device)
        if prompt.numel() == 0:
            raise ValueError("generation requires prompt ids")
        with self.torch.no_grad():
            encoded = self._encode(prompt)
            decoder_ids = [0]
            generated: list[int] = []
            for _ in range(int(max_tokens)):
                logits = self._decode(self.torch.as_tensor(decoder_ids, dtype=self.torch.long, device=self.device), encoded)[-1]
                if grammar_session is not None:
                    mask = grammar_session.get_token_mask()
                    if len(mask) * 8 < int(logits.numel()):
                        raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
                    allowed = self.torch.tensor(
                        [
                            (mask[index // 8] & (1 << (index % 8))) != 0
                            for index in range(int(logits.numel()))
                        ], dtype=self.torch.bool, device=logits.device,
                    )
                    if not bool(self.torch.any(allowed).item()):
                        raise ValueError("Grammar FSM has no valid next token")
                    logits = logits.masked_fill(~allowed, -self.torch.finfo(logits.dtype).max)
                token = int(self.torch.argmax(logits).item()) if temperature <= 0 else int(self.torch.multinomial(self.torch.softmax(logits / float(temperature), dim=-1), 1).item())
                if grammar_session is not None and grammar_session.advance(token) < 0:
                    raise ValueError("the portable seq2seq executor produced a token rejected by the grammar FSM")
                generated.append(token)
                decoder_ids.append(token)
                if grammar_session is not None and getattr(grammar_session, "is_accepting", lambda: False)():
                    break
                if token in stops:
                    break
            return generated

    def generate_iter(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> Iterator[int]:
        yield from self.generate(prompt_ids, max_tokens=max_tokens, **kwargs)

    def generate_with_cache(self, prompt_ids: Any, max_tokens: int = 16, **kwargs: Any) -> tuple[list[int], None]:
        return self.generate(prompt_ids, max_tokens=max_tokens, **kwargs), None


def torch_full_like(torch: Any, value: int, reference: Any) -> Any:
    """Create a scalar-shaped tensor with the same dtype/device as reference."""
    return torch.full_like(reference, value)
