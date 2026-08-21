"""Portable tensor executor for standard decoder-only AEG graphs.

This module is intentionally independent of Transformers model classes.  It
materializes the tensors already authenticated by the AEG loader and evaluates
the same pre-norm transformer equations on a PyTorch device.  Consequently one
AEG can execute on CUDA, ROCm, or Apple MPS when the installed PyTorch build
supports that device.  Vendor NPUs and FPGA targets still require their own
verified backend contract.

The attention implementation follows the ordinary exact scaled dot-product
equation with grouped-query broadcasting.  It is a portability executor, not
a claim that every accelerator-specific FlashAttention kernel is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from aether.runtime.positional import alibi_slopes


@dataclass
class TorchKVCache:
    """Device-resident incremental KV state used by :class:`TorchAEGEngine`."""

    keys: list[Any | None]
    values: list[Any | None]
    length: int = 0
    last_logits: Any | None = None

    def clone(self) -> "TorchKVCache":
        return TorchKVCache(
            keys=[None if value is None else value.clone() for value in self.keys],
            values=[None if value is None else value.clone() for value in self.values],
            length=self.length,
            last_logits=None if self.last_logits is None else self.last_logits.clone(),
        )


class TorchAEGEngine:
    """Execute a standard dense decoder AEG on a PyTorch device."""

    def __init__(self, cpu_engine: Any, device: str) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by backend
            raise RuntimeError("PyTorch is required for portable AEG execution") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.source_engine = cpu_engine
        self.weights = cpu_engine.weights
        self.num_heads = int(cpu_engine.num_heads)
        self.num_kv_heads = int(cpu_engine.num_kv_heads)
        self.head_dim = int(cpu_engine.head_dim)
        self.num_layers = len(self.weights.layers)
        self._alibi_slopes = self._tensor(alibi_slopes(self.num_heads))
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if cpu_engine.sparse_attention_plan or cpu_engine.semantic_kv_plan or cpu_engine.cross_layer_kv_plan:
            raise ValueError(
                "portable PyTorch AEG execution does not yet implement persisted sparse/KV alias plans"
            )
        self.embedding = self._tensor(self.weights.embedding)
        self.final_norm = self._tensor(self.weights.final_norm)
        self.lm_head = self._tensor(self.weights.lm_head)
        self.embedding_norm = (
            self._tensor(self.weights.embedding_norm)
            if self.weights.embedding_norm is not None else None
        )
        self.embedding_norm_bias = (
            self._tensor(self.weights.embedding_norm_bias)
            if self.weights.embedding_norm_bias is not None else None
        )
        self.position_embedding = (
            self._tensor(self.weights.position_embedding)
            if self.weights.position_embedding is not None else None
        )
        self.layers = [self._convert_layer(layer) for layer in self.weights.layers]
        self._cos: Any | None = None
        self._sin: Any | None = None

    def _tensor(self, value: Any) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)

    def _optional_tensor(self, value: Any | None) -> Any | None:
        return None if value is None else self._tensor(value)

    def _convert_layer(self, layer: Any) -> dict[str, Any | None]:
        converted = {
            name: self._optional_tensor(getattr(layer, name))
            for name in (
                "attention_norm", "attention_norm_bias", "q_proj", "k_proj", "v_proj",
                "o_proj", "q_norm", "k_norm", "ffn_norm", "ffn_norm_bias", "gate_proj",
                "up_proj", "down_proj", "q_proj_bias", "k_proj_bias", "v_proj_bias",
                "o_proj_bias", "gate_proj_bias", "up_proj_bias", "down_proj_bias",
            )
        }
        converted["router"] = self._optional_tensor(getattr(layer, "router", None))
        converted["experts"] = [
            {
                name: self._optional_tensor(getattr(expert, name))
                for name in ("gate_proj", "up_proj", "down_proj", "gate_proj_bias", "up_proj_bias", "down_proj_bias")
            }
            for expert in (getattr(layer, "experts", None) or [])
        ]
        converted["num_activated_experts"] = int(getattr(layer, "num_activated_experts", 1) or 1)
        return converted

    def _ensure_rope(self, required: int, device: Any | None = None) -> None:
        device = device or self.device
        if self._cos is not None and int(self._cos.shape[0]) >= required and self._cos.device == device:
            return
        half = self.head_dim // 2
        positions = self.torch.arange(required, device=device, dtype=self.torch.float32)[:, None]
        exponent = self.torch.arange(half, device=device, dtype=self.torch.float32) * (2.0 / self.head_dim)
        inv_freq = self.weights.rope_theta ** (-exponent)
        angles = positions * inv_freq[None, :]
        self._cos = self.torch.cos(angles)
        self._sin = self.torch.sin(angles)

    def _norm(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        eps = float(self.weights.norm_eps)
        if str(self.weights.norm_type).lower() == "layernorm":
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            result = (x - mean) / self.torch.sqrt(var + eps) * weight
            return result if bias is None else result + bias
        return x * self.torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight

    def _linear(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        return self.torch.nn.functional.linear(x, weight, bias)

    def _rope(self, x: Any, positions: Any) -> Any:
        assert self._cos is not None and self._sin is not None
        cos = self._cos.index_select(0, positions).unsqueeze(1)
        sin = self._sin.index_select(0, positions).unsqueeze(1)
        # AEG's canonical RoPE layout is the half-split form used by the
        # reference CPU kernel: [x_0..x_h, x_h..x_2h], not an interleaved
        # even/odd layout.  Keeping this convention makes CPU and accelerator
        # execution numerically equivalent across model families.
        half = self.head_dim // 2
        first, second = x[..., :half], x[..., half:]
        return self.torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    def _activation(self, gate: Any, up: Any | None) -> Any:
        kind = str(self.weights.ffn_type or "SwiGLU").lower()
        if up is None:
            if kind == "relu":
                return self.torch.relu(gate)
            if kind == "relu2":
                return self.torch.relu(gate).pow(2)
            return self.torch.nn.functional.gelu(gate, approximate="tanh")
        if kind in {"gelu", "geglu"}:
            return self.torch.nn.functional.gelu(gate, approximate="tanh") * up
        return self.torch.nn.functional.silu(gate) * up

    def _moe_ffn(self, hidden: Any, layer: dict[str, Any]) -> Any:
        """Execute top-k routed SwiGLU experts on the selected device."""
        torch = self.torch
        router = layer["router"]
        experts = layer["experts"]
        if router is None or not experts:
            raise ValueError("portable MoE layer is missing its router or experts")
        router_logits = self._linear(hidden, router)
        top_k = min(int(layer["num_activated_experts"]), len(experts))
        if top_k <= 0:
            raise ValueError("portable MoE top-k must be positive")
        selected_logits, selected = torch.topk(router_logits, top_k, dim=-1)
        routing = torch.softmax(selected_logits, dim=-1)
        output = torch.zeros_like(hidden)
        for expert_index, expert in enumerate(experts):
            rows, slots = torch.where(selected == expert_index)
            if rows.numel() == 0:
                continue
            source = hidden.index_select(0, rows)
            gate = self._linear(source, expert["gate_proj"], expert["gate_proj_bias"])
            up = self._linear(source, expert["up_proj"], expert["up_proj_bias"])
            activated = self._activation(gate, up)
            value = self._linear(activated, expert["down_proj"], expert["down_proj_bias"])
            output.index_put_((rows,), value * routing[rows, slots].unsqueeze(-1), accumulate=True)
        return output

    def _attention(self, q: Any, k: Any, v: Any, query_positions: Any, key_positions: Any) -> Any:
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = self.torch.einsum("qhd,khd->hqk", q, k) / np.sqrt(self.head_dim)
        if str(self.weights.position_type or "RoPE").lower() in {"alibi", "alibi_bias"}:
            distance = key_positions[None, :] - query_positions[:, None]
            distance_tensor = self.torch.as_tensor(
                distance, dtype=scores.dtype, device=self.device
            )
            scores = scores + self._alibi_slopes[:, None, None] * distance_tensor[None, :, :]
        allowed = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~allowed.unsqueeze(0), -torch_inf(self.torch, scores.dtype))
        probs = self.torch.softmax(scores, dim=-1)
        return self.torch.einsum("hqk,khd->qhd", probs, v)

    def forward(self, token_ids: np.ndarray | Any, cache: TorchKVCache | None = None) -> tuple[np.ndarray, TorchKVCache]:
        torch = self.torch
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64).reshape(-1), device=self.device)
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.embedding.shape[0]:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or TorchKVCache([None] * self.num_layers, [None] * self.num_layers)
        past = int(cache.length)
        seq_len = int(ids.numel())
        positions = torch.arange(past, past + seq_len, device=self.device, dtype=torch.long)
        uses_rope = str(self.weights.position_type or "RoPE").lower() in {"rope", "rotary", "rotary_embedding"}
        if uses_rope:
            self._ensure_rope(past + seq_len)
        hidden = self.embedding.index_select(0, ids)
        if self.embedding_norm is not None:
            hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding.index_select(0, positions)

        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                normed = self._norm(hidden, layer["attention_norm"], layer["attention_norm_bias"])
                q = self._linear(normed, layer["q_proj"], layer["q_proj_bias"]).reshape(seq_len, self.num_heads, self.head_dim)
                if layer["q_norm"] is not None:
                    q = self._norm(q, layer["q_norm"])
                if uses_rope:
                    q = self._rope(q, positions)
                k = self._linear(normed, layer["k_proj"], layer["k_proj_bias"]).reshape(seq_len, self.num_kv_heads, self.head_dim)
                if layer["k_norm"] is not None:
                    k = self._norm(k, layer["k_norm"])
                v = self._linear(normed, layer["v_proj"], layer["v_proj_bias"]).reshape(seq_len, self.num_kv_heads, self.head_dim)
                if uses_rope:
                    k = self._rope(k, positions)
                if cache.keys[index] is not None:
                    k_all = torch.cat((cache.keys[index], k), dim=0)
                    v_all = torch.cat((cache.values[index], v), dim=0)
                else:
                    k_all, v_all = k, v
                context = self._attention(q, k_all, v_all, positions, torch.arange(past + seq_len, device=self.device))
                cache.keys[index] = k_all
                cache.values[index] = v_all
                attention_out = self._linear(
                    context.reshape(seq_len, self.num_heads * self.head_dim),
                    layer["o_proj"], layer["o_proj_bias"],
                )
                hidden = hidden + attention_out
                normed = self._norm(hidden, layer["ffn_norm"], layer["ffn_norm_bias"])
                if layer["experts"]:
                    hidden = hidden + self._moe_ffn(normed, layer)
                else:
                    gate = self._linear(normed, layer["gate_proj"], layer["gate_proj_bias"])
                    up = self._linear(normed, layer["up_proj"], layer["up_proj_bias"]) if layer["up_proj"] is not None else None
                    hidden = hidden + self._linear(self._activation(gate, up), layer["down_proj"], layer["down_proj_bias"])
            hidden = self._norm(hidden, self.final_norm)
            logits = self._linear(hidden, self.lm_head)
            cache.length = past + seq_len
            cache.last_logits = logits[-1].detach()
        return logits.detach().float().cpu().numpy(), cache

    def _sample(self, logits: Any, temperature: float, top_k: int, top_p: float) -> int:
        torch = self.torch
        values = logits.float()
        if temperature <= 0:
            return int(torch.argmax(values).item())
        values = values / float(temperature)
        if top_k > 0:
            top_k = min(int(top_k), int(values.numel()))
            threshold = torch.topk(values, top_k).values[-1]
            values = values.masked_fill(values < threshold, -torch_inf(torch, values.dtype))
        probs = torch.softmax(values, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs > float(top_p)
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            return int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())
        return int(torch.multinomial(probs, 1).item())

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: TorchKVCache | None = None, cache_callback: Any | None = None,
                      grammar_session: Any | None = None, **_: Any) -> Iterator[int]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        ids = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        if ids.size:
            logits, cache = self.forward(ids, cache)
            next_logits = self.torch.as_tensor(logits[-1], device=self.device)
        elif cache is not None and cache.last_logits is not None:
            next_logits = cache.last_logits
        else:
            raise ValueError("generation requires prompt ids or a populated cache")
        generated: list[int] = []
        for _ in range(int(max_tokens)):
            if grammar_session is not None:
                mask = grammar_session.get_token_mask()
                if len(mask) * 8 < int(next_logits.numel()):
                    raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
                allowed = self.torch.tensor(
                    [
                        (mask[index // 8] & (1 << (index % 8))) != 0
                        for index in range(int(next_logits.numel()))
                    ],
                    dtype=self.torch.bool,
                    device=next_logits.device,
                )
                if not bool(self.torch.any(allowed).item()):
                    raise ValueError("Grammar FSM has no valid next token")
                next_logits = next_logits.masked_fill(~allowed, -torch_inf(self.torch, next_logits.dtype))
            token = self._sample(next_logits, temperature, top_k, top_p)
            if grammar_session is not None and grammar_session.advance(token) < 0:
                raise ValueError("The portable PyTorch executor produced a token rejected by the grammar FSM")
            generated.append(token)
            yield token
            if grammar_session is not None and getattr(grammar_session, "is_accepting", lambda: False)():
                break
            if eos_token_id is not None and token == int(eos_token_id):
                break
            _, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
            next_logits = cache.last_logits
        if cache_callback is not None and cache is not None:
            cache_callback(cache)
    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

    def generate_with_cache(self, prompt_ids: np.ndarray, max_tokens: int = 16, cache: TorchKVCache | None = None,
                            **kwargs: Any) -> tuple[list[int], TorchKVCache]:
        result: list[int] = []
        final: list[TorchKVCache | None] = [cache]
        result.extend(list(self.generate_iter(prompt_ids, max_tokens=max_tokens, cache=cache,
                                              cache_callback=lambda value: final.__setitem__(0, value), **kwargs)))
        if final[0] is None:
            raise RuntimeError("generation completed without a KV cache")
        return result, final[0]

    def speculative_stats(self) -> dict[str, int]:
        return {"draft_tokens": 0, "accepted_tokens": 0, "cycles": 0}


@dataclass
class TorchHybridCache:
    """Device-resident cache for a mixed attention/selective-scan decoder."""

    keys: list[Any | None]
    values: list[Any | None]
    states: list[Any]
    conv_history: list[Any]
    length: int = 0
    last_logits: Any | None = None
    last_hidden: Any | None = None

    def clone(self) -> "TorchHybridCache":
        return TorchHybridCache(
            keys=[None if value is None else value.clone() for value in self.keys],
            values=[None if value is None else value.clone() for value in self.values],
            states=[value.clone() for value in self.states],
            conv_history=[value.clone() for value in self.conv_history],
            length=self.length,
            last_logits=None if self.last_logits is None else self.last_logits.clone(),
            last_hidden=None if self.last_hidden is None else self.last_hidden.clone(),
        )


class TorchHybridAEGEngine:
    """PyTorch portability executor for the Jamba hybrid contract.

    This is the device equivalent of :class:`HybridExecutionEngine`.  It uses
    the discrete selective-scan recurrence from Mamba (Gu & Dao 2023) and
    exact scaled dot-product/GQA attention; no source-framework model class or
    model-family name is consulted at runtime.
    """

    def __init__(self, hybrid_engine: Any, device: str, devices: list[str] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by backend
            raise RuntimeError("PyTorch is required for portable hybrid execution") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.devices = [torch.device(value) for value in (devices or [device])]
        self.source_engine = hybrid_engine
        self.weights = hybrid_engine.weights
        self.layer_types = [str(value).lower() for value in hybrid_engine.layer_types]
        self.num_heads = int(hybrid_engine.num_heads)
        self.num_kv_heads = int(hybrid_engine.num_kv_heads)
        self.head_dim = int(hybrid_engine._transformer.head_dim)
        self.num_layers = len(self.layer_types)
        self.layer_devices = [self.devices[index % len(self.devices)] for index in range(self.num_layers)]
        self.state_size = int(hybrid_engine._mamba.state_size)
        self.inner_size = int(hybrid_engine._mamba.inner_size)
        self.dt_rank = int(hybrid_engine._mamba.dt_rank)
        self.conv_kernel = int(hybrid_engine._mamba.conv_kernel)
        self.embedding = self._tensor(self.weights.embedding)
        self.final_norm = self._tensor(self.weights.final_norm)
        self.lm_head = self._tensor(self.weights.lm_head)
        self.embedding_norm = self._optional_tensor(self.weights.embedding_norm)
        self.embedding_norm_bias = self._optional_tensor(self.weights.embedding_norm_bias)
        self.final_norm_bias = self._optional_tensor(self.weights.final_norm_bias)
        self.position_embedding = self._optional_tensor(self.weights.position_embedding)
        self.layers: list[dict[str, Any | None]] = []
        for index, layer in enumerate(self.weights.layers):
            layer_device = self.layer_devices[index]
            transformer = layer.transformer
            mamba = layer.mamba
            if transformer is None or mamba is None:
                raise ValueError("hybrid artifact contains an incomplete layer representation")
            self.layers.append({
                "attention_norm": self._tensor(transformer.attention_norm, layer_device),
                "attention_norm_bias": self._optional_tensor(transformer.attention_norm_bias, layer_device),
                "q_proj": self._tensor(transformer.q_proj, layer_device),
                "k_proj": self._tensor(transformer.k_proj, layer_device),
                "v_proj": self._tensor(transformer.v_proj, layer_device),
                "o_proj": self._tensor(transformer.o_proj, layer_device),
                "q_proj_bias": self._optional_tensor(transformer.q_proj_bias, layer_device),
                "k_proj_bias": self._optional_tensor(transformer.k_proj_bias, layer_device),
                "v_proj_bias": self._optional_tensor(transformer.v_proj_bias, layer_device),
                "o_proj_bias": self._optional_tensor(transformer.o_proj_bias, layer_device),
                "ffn_norm": self._tensor(transformer.ffn_norm, layer_device),
                "ffn_norm_bias": self._optional_tensor(transformer.ffn_norm_bias, layer_device),
                "gate_proj": self._optional_tensor(transformer.gate_proj, layer_device),
                "up_proj": self._optional_tensor(transformer.up_proj, layer_device),
                "down_proj": self._optional_tensor(transformer.down_proj, layer_device),
                "gate_proj_bias": self._optional_tensor(transformer.gate_proj_bias, layer_device),
                "up_proj_bias": self._optional_tensor(transformer.up_proj_bias, layer_device),
                "down_proj_bias": self._optional_tensor(transformer.down_proj_bias, layer_device),
                "q_norm": self._optional_tensor(transformer.q_norm, layer_device),
                "k_norm": self._optional_tensor(transformer.k_norm, layer_device),
                "ssm_norm": self._tensor(mamba.norm, layer_device),
                "ssm_in_proj": self._tensor(mamba.in_proj, layer_device),
                "ssm_conv1d": self._tensor(mamba.conv1d, layer_device),
                "ssm_x_proj": self._tensor(mamba.x_proj, layer_device),
                "ssm_dt_proj": self._tensor(mamba.dt_proj, layer_device),
                "ssm_a_log": self._tensor(mamba.a_log, layer_device),
                "ssm_d": self._tensor(mamba.d, layer_device),
                "ssm_out_proj": self._tensor(mamba.out_proj, layer_device),
                "ssm_conv_bias": self._optional_tensor(mamba.conv_bias, layer_device),
                "ssm_dt_bias": self._optional_tensor(mamba.dt_bias, layer_device),
            })
        self._cos: Any | None = None
        self._sin: Any | None = None

    def _tensor(self, value: Any, device: Any | None = None) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device or self.device)

    def _optional_tensor(self, value: Any | None, device: Any | None = None) -> Any | None:
        return None if value is None else self._tensor(value, device)

    def _ensure_rope(self, required: int, device: Any | None = None) -> None:
        device = device or self.device
        if self._cos is not None and int(self._cos.shape[0]) >= required and self._cos.device == device:
            return
        half = self.head_dim // 2
        positions = self.torch.arange(required, device=device, dtype=self.torch.float32)[:, None]
        exponent = self.torch.arange(half, device=device, dtype=self.torch.float32) * (2.0 / self.head_dim)
        inv_freq = float(self.weights.rope_theta) ** (-exponent)
        angles = positions * inv_freq[None, :]
        self._cos, self._sin = self.torch.cos(angles), self.torch.sin(angles)

    def _norm(self, value: Any, weight: Any, bias: Any | None = None) -> Any:
        if str(self.weights.norm_type).lower() == "layernorm":
            mean = value.mean(dim=-1, keepdim=True)
            variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
            result = (value - mean) / self.torch.sqrt(variance + float(self.weights.norm_eps)) * weight
            return result if bias is None else result + bias
        return value * self.torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + float(self.weights.norm_eps)) * weight

    @staticmethod
    def _silu(value: Any, torch: Any) -> Any:
        return torch.nn.functional.silu(value)

    def _rope(self, value: Any, positions: Any) -> Any:
        assert self._cos is not None and self._sin is not None
        cos = self._cos.index_select(0, positions).unsqueeze(1)
        sin = self._sin.index_select(0, positions).unsqueeze(1)
        half = self.head_dim // 2
        first, second = value[..., :half], value[..., half:]
        return self.torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    def _linear(self, value: Any, weight: Any, bias: Any | None = None) -> Any:
        return self.torch.nn.functional.linear(value, weight, bias)

    def _ssm_step(self, hidden: Any, index: int, cache: TorchHybridCache) -> Any:
        layer = self.layers[index]
        normalized = self._norm(hidden, layer["ssm_norm"])
        projected = self._linear(normalized, layer["ssm_in_proj"])
        ssm_input, gate = projected.split(self.inner_size, dim=-1)
        kernel = layer["ssm_conv1d"]
        window = self.torch.cat((cache.conv_history[index], ssm_input.unsqueeze(-1)), dim=-1)
        conv = (window * kernel.reshape(1, self.inner_size, -1)).sum(dim=-1)
        if layer["ssm_conv_bias"] is not None:
            conv = conv + layer["ssm_conv_bias"].reshape(1, -1)
        cache.conv_history[index] = window[..., -max(self.conv_kernel - 1, 0):].detach()
        conv = self._silu(conv, self.torch)
        selective = self._linear(conv, layer["ssm_x_proj"])
        dt_raw = selective[..., :self.dt_rank]
        b = selective[..., self.dt_rank:self.dt_rank + self.state_size]
        c = selective[..., self.dt_rank + self.state_size:]
        dt = self._linear(dt_raw, layer["ssm_dt_proj"])
        if layer["ssm_dt_bias"] is not None:
            dt = dt + layer["ssm_dt_bias"].reshape(1, -1)
        dt = self.torch.nn.functional.softplus(dt)
        a = -self.torch.exp(layer["ssm_a_log"])
        state = cache.states[index]
        state = state * self.torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0))
        state = state + dt.unsqueeze(-1) * conv.unsqueeze(-1) * b.unsqueeze(1)
        cache.states[index] = state.detach()
        output = (state * c.unsqueeze(1)).sum(dim=-1) + conv * layer["ssm_d"].reshape(1, -1)
        return hidden + self._linear(output * self._silu(gate, self.torch), layer["ssm_out_proj"])

    def _attention_step(self, hidden: Any, index: int, cache: TorchHybridCache, past: int) -> Any:
        torch = self.torch
        layer = self.layers[index]
        normed = self._norm(hidden, layer["attention_norm"], layer["attention_norm_bias"])
        query = self._linear(normed, layer["q_proj"], layer["q_proj_bias"]).reshape(1, self.num_heads, self.head_dim)
        key = self._linear(normed, layer["k_proj"], layer["k_proj_bias"]).reshape(1, self.num_kv_heads, self.head_dim)
        value = self._linear(normed, layer["v_proj"], layer["v_proj_bias"]).reshape(1, self.num_kv_heads, self.head_dim)
        if layer["q_norm"] is not None:
            query = self._norm(query, layer["q_norm"])
        if layer["k_norm"] is not None:
            key = self._norm(key, layer["k_norm"])
        uses_rope = str(self.weights.position_type or "RoPE").lower() in {"rope", "rotary", "rotary_embedding"}
        positions = torch.tensor([past], device=hidden.device, dtype=torch.long)
        if uses_rope:
            self._ensure_rope(past + 1, hidden.device)
            query, key = self._rope(query, positions), self._rope(key, positions)
        if cache.keys[index] is None:
            keys, values = key, value
        else:
            keys = torch.cat((cache.keys[index], key), dim=0)
            values = torch.cat((cache.values[index], value), dim=0)
        cache.keys[index], cache.values[index] = keys.detach(), values.detach()
        repeats = self.num_heads // self.num_kv_heads
        keys_full, values_full = keys.repeat_interleave(repeats, dim=1), values.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query, keys_full) / np.sqrt(self.head_dim)
        key_positions = torch.arange(past + 1, device=self.device, dtype=torch.long)
        allowed = key_positions <= positions[:, None]
        scores = scores.masked_fill(~allowed.unsqueeze(0), -torch.finfo(scores.dtype).max)
        context = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), values_full)
        attention_out = self._linear(
            context.reshape(1, self.num_heads * self.head_dim), layer["o_proj"], layer["o_proj_bias"]
        )
        if bool(getattr(self.weights, "parallel_residual", False)):
            # GPT-J evaluates attention and feed-forward branches from one normed state.
            ffn_normed = normed
        else:
            hidden = hidden + attention_out
            ffn_normed = self._norm(hidden, layer["ffn_norm"], layer["ffn_norm_bias"])
        gate = self._linear(ffn_normed, layer["gate_proj"], layer["gate_proj_bias"])
        up = None if layer["up_proj"] is None else self._linear(ffn_normed, layer["up_proj"], layer["up_proj_bias"])
        kind = str(self.weights.ffn_type or "SwiGLU").lower()
        if up is None:
            activated = torch.nn.functional.gelu(gate, approximate="tanh") if kind == "gelu" else torch.relu(gate)
        elif kind in {"gelu", "geglu"}:
            activated = torch.nn.functional.gelu(gate, approximate="tanh") * up
        else:
            activated = torch.nn.functional.silu(gate) * up
        ffn_out = self._linear(activated, layer["down_proj"], layer["down_proj_bias"])
        return (
            hidden + attention_out + ffn_out
            if bool(getattr(self.weights, "parallel_residual", False))
            else hidden + ffn_out
        )

    def _new_cache(self) -> TorchHybridCache:
        torch = self.torch
        return TorchHybridCache(
            keys=[None] * self.num_layers,
            values=[None] * self.num_layers,
            states=[torch.zeros((1, self.inner_size, self.state_size), device=device) for device in self.layer_devices],
            conv_history=[torch.zeros((1, self.inner_size, max(self.conv_kernel - 1, 0)), device=device) for device in self.layer_devices],
        )

    def forward(self, token_ids: np.ndarray | Any, cache: TorchHybridCache | None = None) -> tuple[np.ndarray, TorchHybridCache]:
        torch = self.torch
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64).reshape(-1), device=self.device)
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.embedding.shape[0]:
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or self._new_cache()
        outputs: list[Any] = []
        with torch.no_grad():
            for token in ids:
                past = int(cache.length)
                hidden = self.embedding.index_select(0, token.reshape(1))
                if self.embedding_norm is not None:
                    hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
                if self.position_embedding is not None:
                    hidden = hidden + self.position_embedding[past:past + 1]
                for index, kind in enumerate(self.layer_types):
                    hidden = hidden.to(self.layer_devices[index])
                    hidden = self._ssm_step(hidden, index, cache) if kind == "ssm" else self._attention_step(hidden, index, cache, past)
                hidden = self._norm(hidden.to(self.device), self.final_norm, self.final_norm_bias)
                logits = self._linear(hidden, self.lm_head)
                outputs.append(logits[0])
                cache.length += 1
                cache.last_hidden = hidden[0].detach()
                cache.last_logits = logits[0].detach()
        return torch.stack(outputs).float().cpu().numpy(), cache

    def _sample(self, logits: Any, temperature: float, top_k: int, top_p: float) -> int:
        if temperature <= 0:
            return int(self.torch.argmax(logits).item())
        values = logits.float() / float(temperature)
        if top_k > 0:
            k = min(int(top_k), int(values.numel()))
            values = values.masked_fill(values < self.torch.topk(values, k).values[-1], -self.torch.inf)
        probs = self.torch.softmax(values, dim=-1)
        if 0.0 < top_p < 1.0:
            ordered, order = self.torch.sort(probs, descending=True)
            keep = self.torch.cumsum(ordered, dim=-1) - ordered <= float(top_p)
            ordered = ordered * keep
            probs = self.torch.zeros_like(probs).scatter(0, order, ordered)
            probs = probs / probs.sum()
        return int(self.torch.multinomial(probs, 1).item())

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: TorchHybridCache | None = None, cache_callback: Any | None = None,
                      grammar_session: Any | None = None, **_: Any) -> Iterator[int]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        prompt = np.asarray(prompt_ids, dtype=np.int64).reshape(-1)
        if prompt.size:
            logits, cache = self.forward(prompt, cache)
            next_logits = self.torch.as_tensor(logits[-1], device=self.device)
        elif cache is not None and cache.last_logits is not None:
            next_logits = cache.last_logits
        else:
            raise ValueError("generation requires prompt ids or a populated cache")
        for _ in range(int(max_tokens)):
            if grammar_session is not None:
                mask = grammar_session.get_token_mask()
                if len(mask) * 8 < int(next_logits.numel()):
                    raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
                allowed = self.torch.tensor(
                    [
                        (mask[index // 8] & (1 << (index % 8))) != 0
                        for index in range(int(next_logits.numel()))
                    ], dtype=self.torch.bool, device=next_logits.device,
                )
                if not bool(self.torch.any(allowed).item()):
                    raise ValueError("Grammar FSM has no valid next token")
                next_logits = next_logits.masked_fill(~allowed, -self.torch.finfo(next_logits.dtype).max)
            token = self._sample(next_logits, temperature, top_k, top_p)
            if grammar_session is not None and grammar_session.advance(token) < 0:
                raise ValueError("the portable hybrid executor produced a token rejected by the grammar FSM")
            yield token
            if grammar_session is not None and getattr(grammar_session, "is_accepting", lambda: False)():
                break
            if eos_token_id is not None and token == int(eos_token_id):
                break
            _, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
            next_logits = cache.last_logits
        if cache_callback is not None and cache is not None:
            cache_callback(cache)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))

    def generate_with_cache(self, prompt_ids: np.ndarray, max_tokens: int = 16,
                            cache: TorchHybridCache | None = None, **kwargs: Any) -> tuple[list[int], TorchHybridCache]:
        holder: list[TorchHybridCache | None] = [cache]
        result = list(self.generate_iter(prompt_ids, max_tokens=max_tokens, cache=cache,
                                         cache_callback=lambda value: holder.__setitem__(0, value), **kwargs))
        if holder[0] is None:
            raise RuntimeError("generation completed without a hybrid cache")
        return result, holder[0]


def torch_inf(torch: Any, dtype: Any) -> Any:
    """Return negative infinity without depending on a global torch import."""
    return torch.tensor(float("inf"), dtype=dtype, device="cpu").to(dtype=dtype).item()
