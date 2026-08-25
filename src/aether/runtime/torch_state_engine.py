"""PyTorch portable executors for recurrent AEG contracts.

The equations mirror the audited CPU reference engines.  These executors are
intentionally token-wise: they preserve the exact recurrent state semantics
while allowing the same authenticated AEG weights to run on CUDA, ROCm, or
MPS when the corresponding PyTorch device exists.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np


def _apply_torch_grammar_mask(logits: Any, grammar_session: Any, torch: Any) -> Any:
    """Mask a Torch logits vector using the authenticated byte-mask FSM."""
    mask = grammar_session.get_token_mask()
    vocab_size = int(logits.numel())
    if len(mask) * 8 < vocab_size:
        raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
    allowed = torch.tensor(
        [
            (mask[index // 8] & (1 << (index % 8))) != 0
            for index in range(vocab_size)
        ],
        dtype=torch.bool,
        device=logits.device,
    )
    if not bool(torch.any(allowed).item()):
        raise ValueError("Grammar FSM has no valid next token")
    return logits.masked_fill(~allowed, -torch.finfo(logits.dtype).max)


class _TorchStateBase:
    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for portable state execution") from exc
        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.devices = [_resolve_device(torch, value) for value in (devices or [device])]
        self.layer_devices = [self.devices[index % len(self.devices)] for index in range(len(source_engine.weights.layers))]
        self.source_engine = source_engine
        self.weights = source_engine.weights
        self.embedding = self._tensor(self.weights.embedding)
        self.final_norm = self._tensor(self.weights.final_norm)
        self.lm_head = self._tensor(self.weights.lm_head)
        self.num_layers = len(self.weights.layers)

    def _tensor(self, value: Any, device: Any | None = None) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device or self.device)

    @staticmethod
    def _ids(torch: Any, values: Any, device: Any, vocab: int) -> Any:
        ids = torch.as_tensor(np.asarray(values, dtype=np.int64).reshape(-1), device=device)
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= vocab:
            raise ValueError("token id is outside the compiled vocabulary")
        return ids

    def _norm(self, value: Any, weight: Any) -> Any:
        if str(self.weights.norm_type).lower() == "layernorm":
            mean = value.mean(dim=-1, keepdim=True)
            variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
            return (value - mean) / self.torch.sqrt(variance + float(self.weights.norm_eps)) * weight
        return value * self.torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + float(self.weights.norm_eps)) * weight

    def _linear(self, value: Any, weight: Any, bias: Any | None = None) -> Any:
        return self.torch.nn.functional.linear(value, weight, bias)

    @staticmethod
    def _silu(value: Any, torch: Any) -> Any:
        return torch.nn.functional.silu(value)

    def _sample(self, logits: Any, temperature: float, top_k: int, top_p: float) -> int:
        if temperature <= 0:
            return int(self.torch.argmax(logits).item())
        values = logits.float() / float(temperature)
        if top_k > 0:
            k = min(int(top_k), int(values.numel()))
            cutoff = self.torch.topk(values, k).values[-1]
            values = values.masked_fill(values < cutoff, -self.torch.inf)
        probs = self.torch.softmax(values, dim=-1)
        if 0.0 < top_p < 1.0:
            ordered, order = self.torch.sort(probs, descending=True)
            keep = self.torch.cumsum(ordered, dim=-1) - ordered <= float(top_p)
            filtered = ordered * keep
            probs = self.torch.zeros_like(probs).scatter(0, order, filtered)
            probs = probs / probs.sum()
        return int(self.torch.multinomial(probs, 1).item())

    def _generate_iter(self, prompt_ids: np.ndarray, max_tokens: int, temperature: float,
                       top_k: int, top_p: float, eos_token_id: int | None, cache: Any,
                       cache_callback: Any | None, grammar_session: Any | None = None) -> Iterator[int]:
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
                next_logits = _apply_torch_grammar_mask(next_logits, grammar_session, self.torch)
            token = self._sample(next_logits, temperature, top_k, top_p)
            if grammar_session is not None and grammar_session.advance(token) < 0:
                raise ValueError("the portable state executor produced a token rejected by the grammar FSM")
            yield token
            if grammar_session is not None and getattr(grammar_session, "is_accepting", lambda: False)():
                break
            if eos_token_id is not None and token == int(eos_token_id):
                break
            _, cache = self.forward(np.asarray([token], dtype=np.int64), cache)
            next_logits = cache.last_logits
        if cache_callback is not None and cache is not None:
            cache_callback(cache)


class TorchMamba2AEGEngine(_TorchStateBase):
    """Portable PyTorch implementation of the Mamba-2 SSD recurrence."""

    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        super().__init__(source_engine, device, devices)
        self.state_size = int(source_engine.state_size)
        self.inner_size = int(source_engine.inner_size)
        self.num_heads = int(source_engine.num_heads)
        self.num_groups = int(source_engine.num_groups)
        self.head_dim = int(source_engine.head_dim)
        self.conv_kernel = int(source_engine.conv_kernel)
        self.layers: list[dict[str, Any | None]] = []
        for index, layer in enumerate(self.weights.layers):
            layer_device = self.layer_devices[index]
            self.layers.append({
                "norm": self._tensor(layer.norm, layer_device), "in_proj": self._tensor(layer.in_proj, layer_device),
                "conv1d": self._tensor(layer.conv1d, layer_device), "a_log": self._tensor(layer.a_log, layer_device),
                "d": self._tensor(layer.d, layer_device), "dt": self._tensor(layer.dt, layer_device),
                "out_proj": self._tensor(layer.out_proj, layer_device),
                "in_bias": None if layer.in_proj_bias is None else self._tensor(layer.in_proj_bias, layer_device),
                "conv_bias": None if layer.conv_bias is None else self._tensor(layer.conv_bias, layer_device),
            })

    def _new_cache(self) -> Any:
        torch = self.torch
        channels = self.inner_size + 2 * self.num_groups * self.state_size
        return type("TorchMamba2Cache", (), {
            "states": [torch.zeros((1, self.num_heads, self.head_dim, self.state_size), device=device) for device in self.layer_devices],
            "conv_history": [torch.zeros((1, channels, max(self.conv_kernel - 1, 0)), device=device) for device in self.layer_devices],
            "length": 0, "last_logits": None,
        })()

    def _step(self, hidden: Any, index: int, cache: Any) -> Any:
        torch = self.torch
        layer = self.layers[index]
        normalized = self._norm(hidden, layer["norm"])
        projected = self._linear(normalized, layer["in_proj"], layer["in_bias"])
        groups = self.num_groups * self.state_size
        expected = 2 * self.inner_size + 2 * groups + self.num_heads
        if projected.shape[-1] != expected:
            raise ValueError(f"Mamba-2 in_proj has {projected.shape[-1]} outputs; expected {expected}")
        z = projected[:, :self.inner_size]
        xbc = projected[:, self.inner_size:2 * self.inner_size + 2 * groups]
        dt = projected[:, 2 * self.inner_size + 2 * groups:]
        kernel = layer["conv1d"].reshape(xbc.shape[1], -1)
        window = torch.cat((cache.conv_history[index], xbc.unsqueeze(-1)), dim=-1)
        xbc = (window * kernel.reshape(1, xbc.shape[1], -1)).sum(dim=-1)
        if layer["conv_bias"] is not None:
            xbc = xbc + layer["conv_bias"].reshape(1, -1)
        cache.conv_history[index] = window[..., -max(self.conv_kernel - 1, 0):].detach()
        xbc = self._silu(xbc, torch)
        x = xbc[:, :self.inner_size].reshape(-1, self.num_heads, self.head_dim)
        b = xbc[:, self.inner_size:self.inner_size + groups].reshape(-1, self.num_groups, self.state_size)
        c = xbc[:, self.inner_size + groups:].reshape(-1, self.num_groups, self.state_size)
        repeat = self.num_heads // self.num_groups
        b, c = b.repeat_interleave(repeat, dim=1), c.repeat_interleave(repeat, dim=1)
        delta = torch.nn.functional.softplus(dt + layer["dt"].reshape(1, -1))
        a = -torch.exp(layer["a_log"].reshape(1, -1))
        state = cache.states[index] * torch.exp(delta[:, :, None, None] * a[:, :, None, None])
        state = state + delta[:, :, None, None] * x[:, :, :, None] * b[:, :, None, :]
        cache.states[index] = state.detach()
        y = (state * c[:, :, None, :]).sum(dim=-1) + x * layer["d"].reshape(1, self.num_heads, 1)
        mixed = y.reshape(-1, self.inner_size) * self._silu(z, torch)
        return hidden + self._linear(mixed, layer["out_proj"])

    def forward(self, token_ids: Any, cache: Any | None = None) -> tuple[np.ndarray, Any]:
        torch = self.torch
        ids = self._ids(torch, token_ids, self.device, int(self.embedding.shape[0]))
        cache = cache or self._new_cache()
        outputs: list[Any] = []
        with torch.no_grad():
            for token in ids:
                hidden = self.embedding.index_select(0, token.reshape(1))
                for index in range(self.num_layers):
                    hidden = self._step(hidden.to(self.layer_devices[index]), index, cache)
                hidden = self._norm(hidden.to(self.device), self.final_norm)
                logits = self._linear(hidden, self.lm_head)
                outputs.append(logits[0])
                cache.length += 1
                cache.last_logits = logits[0].detach()
        return torch.stack(outputs).float().cpu().numpy(), cache

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: Any | None = None, cache_callback: Any | None = None,
                      grammar_session: Any | None = None, **_: Any) -> Iterator[int]:
        yield from self._generate_iter(prompt_ids, max_tokens, temperature, top_k, top_p, eos_token_id, cache, cache_callback, grammar_session)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))


class TorchMambaAEGEngine(_TorchStateBase):
    """Portable PyTorch implementation of Mamba-1 selective scan."""

    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        super().__init__(source_engine, device, devices)
        self.state_size = int(source_engine.state_size)
        self.inner_size = int(source_engine.inner_size)
        self.dt_rank = int(source_engine.dt_rank)
        self.conv_kernel = int(source_engine.conv_kernel)
        self.layers: list[dict[str, Any | None]] = []
        for index, layer in enumerate(self.weights.layers):
            layer_device = self.layer_devices[index]
            self.layers.append({
                "norm": self._tensor(layer.norm, layer_device), "in_proj": self._tensor(layer.in_proj, layer_device),
                "conv1d": self._tensor(layer.conv1d, layer_device), "x_proj": self._tensor(layer.x_proj, layer_device),
                "dt_proj": self._tensor(layer.dt_proj, layer_device), "a_log": self._tensor(layer.a_log, layer_device),
                "d": self._tensor(layer.d, layer_device), "out_proj": self._tensor(layer.out_proj, layer_device),
                "conv_bias": None if layer.conv_bias is None else self._tensor(layer.conv_bias, layer_device),
                "dt_bias": None if layer.dt_bias is None else self._tensor(layer.dt_bias, layer_device),
            })

    def _new_cache(self) -> Any:
        torch = self.torch
        return type("TorchMambaCache", (), {
            "states": [torch.zeros((1, self.state_size, self.inner_size), device=device) for device in self.layer_devices],
            "conv_history": [torch.zeros((1, self.inner_size, max(self.conv_kernel - 1, 0)), device=device) for device in self.layer_devices],
            "length": 0, "last_logits": None,
        })()

    def _step(self, hidden: Any, index: int, cache: Any) -> Any:
        torch = self.torch
        layer = self.layers[index]
        normalized = self._norm(hidden, layer["norm"])
        projected = self._linear(normalized, layer["in_proj"])
        ssm_input, gate = projected.split(self.inner_size, dim=-1)
        kernel = layer["conv1d"].reshape(self.inner_size, -1)
        window = torch.cat((cache.conv_history[index], ssm_input.unsqueeze(-1)), dim=-1)
        conv = (window * kernel.reshape(1, self.inner_size, -1)).sum(dim=-1)
        if layer["conv_bias"] is not None:
            conv = conv + layer["conv_bias"].reshape(1, -1)
        cache.conv_history[index] = window[..., -max(self.conv_kernel - 1, 0):].detach()
        conv = self._silu(conv, torch)
        selective = self._linear(conv, layer["x_proj"])
        dt_raw = selective[..., :self.dt_rank]
        b = selective[..., self.dt_rank:self.dt_rank + self.state_size]
        c = selective[..., self.dt_rank + self.state_size:]
        dt = self._linear(dt_raw, layer["dt_proj"])
        if layer["dt_bias"] is not None:
            dt = dt + layer["dt_bias"].reshape(1, -1)
        dt = torch.nn.functional.softplus(dt)
        # Mamba-1 stores A as (inner, state), while the recurrent state is
        # represented as (batch, state, inner), matching the CPU reference.
        a = -torch.exp(layer["a_log"]).transpose(0, 1)
        state = cache.states[index] * torch.exp(dt.unsqueeze(1) * a.unsqueeze(0))
        state = state + dt.unsqueeze(1) * conv.unsqueeze(1) * b.unsqueeze(-1)
        cache.states[index] = state.detach()
        output = (state * c.unsqueeze(-1)).sum(dim=1) + conv * layer["d"].reshape(1, -1)
        return hidden + self._linear(output * self._silu(gate, torch), layer["out_proj"])

    def forward(self, token_ids: Any, cache: Any | None = None) -> tuple[np.ndarray, Any]:
        torch = self.torch
        ids = self._ids(torch, token_ids, self.device, int(self.embedding.shape[0]))
        cache = cache or self._new_cache()
        outputs: list[Any] = []
        with torch.no_grad():
            for token in ids:
                hidden = self.embedding.index_select(0, token.reshape(1))
                for index in range(self.num_layers):
                    hidden = self._step(hidden.to(self.layer_devices[index]), index, cache)
                hidden = self._norm(hidden.to(self.device), self.final_norm)
                logits = self._linear(hidden, self.lm_head)
                outputs.append(logits[0])
                cache.length += 1
                cache.last_logits = logits[0].detach()
        return torch.stack(outputs).float().cpu().numpy(), cache

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: Any | None = None, cache_callback: Any | None = None,
                      grammar_session: Any | None = None, **_: Any) -> Iterator[int]:
        yield from self._generate_iter(prompt_ids, max_tokens, temperature, top_k, top_p, eos_token_id, cache, cache_callback, grammar_session)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))


class TorchMLAAEGEngine:
    """Portable PyTorch executor for the dense MLA AEG contract.

    The implementation intentionally mirrors ``MLAAttention.forward_prefill``
    and replays the authenticated prefix on decode.  This preserves exact
    artifact semantics while a target-specific latent-cache kernel can be
    added later without changing the AEG representation.
    """

    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for portable MLA execution") from exc
        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.devices = [_resolve_device(torch, value) for value in (devices or [device])]
        self.source_engine = source_engine
        self.weights = source_engine.weights
        self.config = source_engine.config
        self.layer_devices = [self.devices[index % len(self.devices)] for index in range(len(self.weights.layers))]
        self.embedding = self._tensor(self.weights.embedding)
        self.final_norm = self._tensor(self.weights.final_norm)
        self.lm_head = self._tensor(self.weights.lm_head)
        self.layers: list[dict[str, Any | None]] = []
        for index, layer in enumerate(self.weights.layers):
            layer_device = self.layer_devices[index]
            converted = {name: self._tensor(value, layer_device) for name, value in layer.mla.items()}
            converted.update({
                "attention_norm": self._tensor(layer.attention_norm, layer_device),
                "ffn_norm": self._tensor(layer.ffn_norm, layer_device),
                "o_proj": self._tensor(layer.o_proj, layer_device),
                "ffn_in": self._tensor(layer.ffn_in, layer_device),
                "ffn_out": self._tensor(layer.ffn_out, layer_device),
                "ffn_up": None if layer.ffn_up is None else self._tensor(layer.ffn_up, layer_device),
                "router": None if layer.router is None else self._tensor(layer.router, layer_device),
                "experts": [
                    {
                        "gate_proj": self._tensor(expert.gate_proj, layer_device),
                        "up_proj": self._tensor(expert.up_proj, layer_device),
                        "down_proj": self._tensor(expert.down_proj, layer_device),
                    }
                    for expert in layer.experts
                ],
                "num_activated_experts": int(layer.num_activated_experts),
            })
            self.layers.append(converted)

    def _tensor(self, value: Any, device: Any | None = None) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device or self.device)

    def _norm(self, value: Any, weight: Any) -> Any:
        if str(self.weights.norm_type).lower() == "layernorm":
            mean = value.mean(dim=-1, keepdim=True)
            variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
            return (value - mean) / self.torch.sqrt(variance + float(self.weights.norm_eps)) * weight
        return value * self.torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + float(self.weights.norm_eps)) * weight

    def _rms(self, value: Any, weight: Any | None) -> Any:
        if weight is None:
            return value
        return value * self.torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight

    def _linear(self, value: Any, weight: Any) -> Any:
        return self.torch.nn.functional.linear(value, weight)

    def _ffn(self, value: Any, layer: dict[str, Any]) -> Any:
        torch = self.torch
        experts = layer.get("experts", [])
        if experts:
            router = layer.get("router")
            if router is None:
                raise ValueError("MLA MoE layer is missing its router")
            router_logits = self._linear(value, router)
            top_k = min(max(1, int(layer.get("num_activated_experts", 1))), len(experts))
            selected_logits, selected = torch.topk(router_logits, top_k, dim=-1)
            weights = torch.softmax(selected_logits, dim=-1)
            result = torch.zeros_like(value)
            for expert_index, expert in enumerate(experts):
                positions = (selected == expert_index).nonzero(as_tuple=False)
                if positions.numel() == 0:
                    continue
                token_indices = positions[:, 0]
                choice_indices = positions[:, 1]
                tokens = value.index_select(0, token_indices)
                gate = self._linear(tokens, expert["gate_proj"])
                up = self._linear(tokens, expert["up_proj"])
                output = self._linear(torch.nn.functional.silu(gate) * up, expert["down_proj"])
                result.index_add_(0, token_indices, output * weights[token_indices, choice_indices].unsqueeze(-1))
            return result
        first = self._linear(value, layer["ffn_in"])
        if layer["ffn_up"] is None:
            return self._linear(torch.sigmoid(first) * first, layer["ffn_out"])
        up = self._linear(value, layer["ffn_up"])
        if str(self.weights.ffn_type).lower() in {"gelu", "geglu"}:
            activated = torch.nn.functional.gelu(first, approximate="tanh") * up
        else:
            activated = torch.nn.functional.silu(first) * up
        return self._linear(activated, layer["ffn_out"])

    def _rope(self, value: Any, offset: int) -> Any:
        torch = self.torch
        d = int(value.shape[-1])
        half = d // 2
        device = value.device
        positions = torch.arange(offset, offset + int(value.shape[-2]), device=device, dtype=torch.float32)[:, None]
        freqs = 1.0 / (float(self.config.rope_theta) ** (torch.arange(half, device=device, dtype=torch.float32) / half))
        angles = positions * freqs[None, :]
        cos, sin = torch.cos(angles), torch.sin(angles)
        first, second = value[..., :half], value[..., half:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    def _layer(self, value: Any, layer: dict[str, Any], sequence_length: int) -> Any:
        torch = self.torch
        batch, seq, hidden = value.shape
        normed = self._norm(value, layer["attention_norm"])
        q_a = self._linear(normed, layer["q_a_proj.weight"])
        q_a = self._rms(q_a, layer["q_a_norm.weight"])
        q_all = self._linear(q_a, layer["q_b_proj.weight"])
        heads = int(self.config.num_heads)
        nope = int(self.config.qk_nope_head_dim)
        rope = int(self.config.qk_rope_head_dim)
        vdim = int(self.config.v_head_dim)
        q_all = q_all.reshape(batch, seq, heads, nope + rope)
        q_nope, q_rope = q_all[..., :nope], q_all[..., nope:]
        q_rope = self._rope(q_rope.transpose(1, 2), 0).transpose(1, 2)

        latent = self._linear(normed, layer["kv_a_proj.weight"])
        latent = self._rms(latent, layer["kv_a_norm.weight"])
        # SafeTensors stores linear weights as (out_features, in_features).
        # The NumPy MLA reference transposes once to express the mathematical
        # rank-by-output matrix; torch.nn.functional.linear consumes the
        # stored orientation directly.
        kv_math = layer["kv_b_proj.weight"]
        nope_width = heads * nope
        v_width = heads * vdim
        k_nope = self._linear(latent, kv_math[:nope_width, :]).reshape(batch, seq, heads, nope)
        v = self._linear(latent, kv_math[nope_width:nope_width + v_width, :]).reshape(batch, seq, heads, vdim)
        k_rope = self._linear(normed, layer["k_rope_proj.weight"]).reshape(batch, seq, heads, rope)
        k_rope = self._rope(k_rope.transpose(1, 2), 0).transpose(1, 2)
        q = torch.cat((q_nope, q_rope), dim=-1).transpose(1, 2)
        k = torch.cat((k_nope, k_rope), dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(nope + rope)
        mask = torch.triu(torch.full((seq, seq), -torch.inf, device=value.device), diagonal=1)
        scores = scores + mask.reshape(1, 1, seq, seq)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, v).transpose(1, 2).reshape(batch, seq, heads * vdim)
        hidden_out = value + self._linear(out, layer["o_proj"])
        ffn_norm = self._norm(hidden_out, layer["ffn_norm"])
        return hidden_out + self._ffn(ffn_norm.reshape(-1, hidden), layer).reshape(batch, seq, hidden)

    def forward(self, token_ids: Any, cache: Any | None = None) -> tuple[np.ndarray, Any]:
        torch = self.torch
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64).reshape(-1), device=self.device)
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if int(ids.min()) < 0 or int(ids.max()) >= self.embedding.shape[0]:
            raise ValueError("token id is outside the compiled vocabulary")
        if cache is None:
            cache = type("TorchMLACache", (), {"token_ids": [], "length": 0, "last_logits": None})()
        all_ids = np.asarray(cache.token_ids + ids.detach().cpu().tolist(), dtype=np.int64)
        hidden = self.embedding.index_select(0, torch.as_tensor(all_ids, device=self.device)).unsqueeze(0)
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                hidden = self._layer(hidden.to(self.layer_devices[index]), layer, int(all_ids.size))
            hidden = self._norm(hidden.to(self.device), self.final_norm)
            logits = self._linear(hidden, self.lm_head)[0]
        cache.token_ids = all_ids.tolist()
        cache.length = len(cache.token_ids)
        cache.last_logits = logits[-1].detach()
        return logits[-int(ids.numel()):].float().cpu().numpy(), cache

    def _sample(self, logits: Any, temperature: float, top_k: int, top_p: float) -> int:
        if temperature <= 0:
            return int(self.torch.argmax(logits).item())
        values = logits.float() / float(temperature)
        if top_k > 0:
            k = min(int(top_k), int(values.numel()))
            values = values.masked_fill(values < self.torch.topk(values, k).values[-1], -self.torch.inf)
        probs = self.torch.softmax(values, dim=-1)
        return int(self.torch.multinomial(probs, 1).item())

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: Any | None = None, cache_callback: Any | None = None,
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
                next_logits = _apply_torch_grammar_mask(next_logits, grammar_session, self.torch)
            token = self._sample(next_logits, temperature, top_k, top_p)
            if grammar_session is not None and grammar_session.advance(token) < 0:
                raise ValueError("the portable MLA executor produced a token rejected by the grammar FSM")
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


class TorchRWKVAEGEngine(_TorchStateBase):
    """Portable PyTorch implementation of the stable RWKV WKV recurrence."""

    def __init__(self, source_engine: Any, device: str, devices: list[str] | None = None) -> None:
        super().__init__(source_engine, device, devices)
        self.layers: list[dict[str, Any]] = []
        for index, layer in enumerate(self.weights.layers):
            layer_device = self.layer_devices[index]
            self.layers.append({name: self._tensor(getattr(layer, name), layer_device) for name in (
                "norm", "ffn_norm", "time_decay", "time_first", "time_mix_k", "time_mix_v",
                "time_mix_r", "ffn_time_mix_k", "ffn_time_mix_r", "key", "value",
                "receptance", "output", "ffn_key", "ffn_value", "ffn_receptance",
            )})

    def _new_cache(self) -> Any:
        torch = self.torch
        hidden = int(self.embedding.shape[1])
        return type("TorchRWKVCache", (), {
            "aa": [torch.zeros((1, hidden), device=device) for device in self.layer_devices],
            "bb": [torch.zeros((1, hidden), device=device) for device in self.layer_devices],
            "pp": [torch.full((1, hidden), -torch.inf, device=device) for device in self.layer_devices],
            "previous": [torch.zeros((1, hidden), device=device) for device in self.layer_devices],
            "length": 0, "last_logits": None,
        })()

    def _rwkv_norm(self, value: Any, weight: Any) -> Any:
        mean = value.mean(dim=-1, keepdim=True)
        variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
        return (value - mean) / self.torch.sqrt(variance + float(self.weights.norm_eps)) * weight

    def _wkv(self, key: Any, value: Any, layer: dict[str, Any], cache: Any, index: int) -> Any:
        torch = self.torch
        aa, bb, pp = cache.aa[index], cache.bb[index], cache.pp[index]
        first = layer["time_first"].reshape(1, -1)
        decay = -torch.exp(layer["time_decay"].reshape(1, -1))
        p = torch.maximum(pp + first, key)
        e1, e2 = torch.exp(pp + first - p), torch.exp(key - p)
        result = (e1 * aa + e2 * value) / torch.clamp(e1 * bb + e2, min=1e-30)
        next_p = torch.maximum(pp + decay, key)
        d1, d2 = torch.exp(pp + decay - next_p), torch.exp(key - next_p)
        cache.aa[index], cache.bb[index], cache.pp[index] = (d1 * aa + d2 * value).detach(), (d1 * bb + d2).detach(), next_p.detach()
        return result

    def _step(self, hidden: Any, index: int, cache: Any) -> Any:
        torch = self.torch
        layer = self.layers[index]
        x = self._rwkv_norm(hidden, layer["norm"])
        previous = cache.previous[index]
        cache.previous[index] = x.detach()
        xk = x * layer["time_mix_k"] + previous * (1.0 - layer["time_mix_k"])
        xv = x * layer["time_mix_v"] + previous * (1.0 - layer["time_mix_v"])
        xr = x * layer["time_mix_r"] + previous * (1.0 - layer["time_mix_r"])
        receptance = torch.sigmoid(self._linear(xr, layer["receptance"]))
        key, value = self._linear(xk, layer["key"]), self._linear(xv, layer["value"])
        mixed = receptance * self._wkv(key, value, layer, cache, index)
        hidden = hidden + self._linear(mixed, layer["output"])
        ffn_input = self._rwkv_norm(hidden, layer["ffn_norm"])
        ffn_k = ffn_input * layer["ffn_time_mix_k"] + previous * (1.0 - layer["ffn_time_mix_k"])
        ffn_r = ffn_input * layer["ffn_time_mix_r"] + previous * (1.0 - layer["ffn_time_mix_r"])
        ffn_gate = torch.relu(self._linear(ffn_k, layer["ffn_key"])).pow(2)
        ffn_value = self._linear(ffn_gate, layer["ffn_value"])
        ffn_receptance = torch.sigmoid(self._linear(ffn_r, layer["ffn_receptance"]))
        return hidden + ffn_receptance * ffn_value

    def forward(self, token_ids: Any, cache: Any | None = None) -> tuple[np.ndarray, Any]:
        torch = self.torch
        ids = self._ids(torch, token_ids, self.device, int(self.embedding.shape[0]))
        cache = cache or self._new_cache()
        outputs: list[Any] = []
        with torch.no_grad():
            for token in ids:
                hidden = self.embedding.index_select(0, token.reshape(1))
                for index in range(self.num_layers):
                    hidden = self._step(hidden.to(self.layer_devices[index]), index, cache)
                hidden = self._rwkv_norm(hidden.to(self.device), self.final_norm)
                logits = self._linear(hidden, self.lm_head)
                outputs.append(logits[0])
                cache.length += 1
                cache.last_logits = logits[0].detach()
        return torch.stack(outputs).float().cpu().numpy(), cache

    def generate_iter(self, prompt_ids: np.ndarray, max_tokens: int = 16, temperature: float = 0.0,
                      top_k: int = 0, top_p: float = 1.0, eos_token_id: int | None = None,
                      cache: Any | None = None, cache_callback: Any | None = None,
                      grammar_session: Any | None = None, **_: Any) -> Iterator[int]:
        yield from self._generate_iter(prompt_ids, max_tokens, temperature, top_k, top_p, eos_token_id, cache, cache_callback, grammar_session)

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 16, **kwargs: Any) -> list[int]:
        return list(self.generate_iter(prompt_ids, max_tokens=max_tokens, **kwargs))
