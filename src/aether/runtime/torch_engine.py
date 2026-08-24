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
import os
from typing import Any, Iterator

import numpy as np

from aether.runtime.positional import alibi_slopes


def _multiplier(value: Any) -> float | None:
    """Coerce a scalar multiplier, treating absence and ``1.0`` as "none".

    A missing multiplier means the architecture declares none; an explicit
    ``1.0`` is a no-op that should not cost a kernel launch per token.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number == 1.0 else number


def _cap(value: Any) -> float | None:
    """Coerce a soft-cap limit; only a positive value is meaningful."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


#: Execution-numerics fields carried by :class:`~aether.runtime.cpu_engine.ModelWeights`.
#: Listed once so every accelerator executor that rebuilds a reduced weight
#: container (tensor-parallel sharding) propagates the complete contract instead
#: of silently reverting part of the model to Llama-style defaults.
EXECUTION_NUMERICS_FIELDS: tuple[str, ...] = (
    "attention_scale",
    "attention_scale_by_layer_index",
    "embedding_scale",
    "residual_scale",
    "logit_scale",
    "attn_logit_softcap",
    "final_logit_softcap",
    "norm_offset_one",
    "rope_partial_dim",
    "rope_interleaved",
    "rope_local_theta",
    "norm_placement",
    "qk_norm_scope",
    "no_rope_layers",
    "gelu_approximate",
)


def execution_numerics(source: Any) -> dict[str, Any]:
    """Extract the execution-numerics contract from a weight container."""
    return {name: getattr(source, name, None) for name in EXECUTION_NUMERICS_FIELDS}


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
        # The portable executor used to materialize every artifact as FP32.
        # That is needlessly expensive on CUDA and also defeats the fused
        # half-precision kernels used by PyTorch.  Keep an explicit override
        # for validation/reproducibility, while selecting the safe fast path
        # for the target device by default.
        requested_dtype = str(os.environ.get("AETHER_TORCH_DTYPE", "auto")).lower()
        if requested_dtype in {"fp16", "float16", "half"}:
            self.compute_dtype = torch.float16
        elif requested_dtype in {"bf16", "bfloat16"}:
            self.compute_dtype = torch.bfloat16
        elif requested_dtype in {"fp32", "float32"}:
            self.compute_dtype = torch.float32
        elif self.device.type == "cuda":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32
        self.source_engine = cpu_engine
        self.weights = cpu_engine.weights
        self.num_heads = int(cpu_engine.num_heads)
        self.num_kv_heads = int(cpu_engine.num_kv_heads)
        self.head_dim = int(cpu_engine.head_dim)
        self.num_layers = len(self.weights.layers)
        self._alibi_slopes = self._tensor(alibi_slopes(self.num_heads))
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")

        # ── Execution numerics ─────────────────────────────────────────────
        # Resolved once here so the decode loop performs no string handling,
        # attribute probing, or Python-level branching per layer.  These are
        # the source architecture's own constants; see
        # aether.core.types.ModelArchitecture.
        self._resolve_execution_numerics()

        self.embedding = self._tensor(self.weights.embedding)
        self.final_norm = self._norm_tensor(self.weights.final_norm)
        self.final_norm_bias = self._optional_tensor(getattr(self.weights, "final_norm_bias", None))
        self.lm_head = self._tensor(self.weights.lm_head)
        self.embedding_norm = (
            self._norm_tensor(self.weights.embedding_norm)
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
        self._local_cos: Any | None = None
        self._local_sin: Any | None = None
        # Cached key-position ranges, indexed by length.  ``torch.arange`` was
        # previously allocated once per layer per decoded token.
        self._positions_cache: Any | None = None
        # The AEG metadata may contain approximate sparse/KV plans selected by
        # the CPU target.  Those plans are not portable PyTorch kernels: using
        # them here would silently change the model's attention graph.  Keep
        # the exact dense implementation as the safe accelerator fallback.
        # This is intentionally a fallback, not an error: a compiled artifact
        # must remain runnable on a different supported target.
        self._ignored_optimized_plans = {
            "sparse_attention": bool(cpu_engine.sparse_attention_plan),
            "semantic_kv": bool(cpu_engine.semantic_kv_plan),
            "cross_layer_kv": bool(cpu_engine.cross_layer_kv_plan),
        }

    def _resolve_execution_numerics(self) -> None:
        """Resolve the architecture's scalar constants and per-layer schedule.

        Called once at construction.  Every value comes from the AEG's declared
        execution numerics, so the decode loop performs no string handling,
        attribute probing, or Python-level branching per layer — on a small model
        that bookkeeping costs more than the GEMMs themselves.

        Accelerator executors that rebuild a reduced weight container (the
        tensor-parallel sharder) call this as well, so the complete contract is
        never silently reverted to Llama-style defaults.
        """
        weights = self.weights
        declared_rotary = getattr(weights, "rope_partial_dim", None)
        rotary = int(declared_rotary) if declared_rotary else self.head_dim
        rotary = max(2, min(rotary, self.head_dim))
        if rotary % 2:
            rotary -= 1
        self.rotary_dim = rotary
        self.rope_interleaved = bool(getattr(weights, "rope_interleaved", False))
        self.rope_theta = float(weights.rope_theta)
        local_theta = getattr(weights, "rope_local_theta", None)
        self.rope_local_theta = (
            float(local_theta)
            if local_theta and float(local_theta) != self.rope_theta
            else None
        )
        declared_scale = getattr(weights, "attention_scale", None)
        self.base_attention_scale = (
            float(declared_scale)
            if declared_scale is not None and float(declared_scale) > 0
            else 1.0 / float(np.sqrt(self.head_dim))
        )
        self.scale_by_layer_index = bool(
            getattr(weights, "attention_scale_by_layer_index", False)
        )
        self.embedding_scale = _multiplier(getattr(weights, "embedding_scale", None))
        self.residual_scale = _multiplier(getattr(weights, "residual_scale", None))
        self.logit_scale = _multiplier(getattr(weights, "logit_scale", None))
        self.attn_logit_softcap = _cap(getattr(weights, "attn_logit_softcap", None))
        self.final_logit_softcap = _cap(getattr(weights, "final_logit_softcap", None))
        self.norm_offset_one = bool(getattr(weights, "norm_offset_one", False))
        self.norm_placement = str(getattr(weights, "norm_placement", "pre") or "pre").lower()
        self.qk_norm_is_full = (
            str(getattr(weights, "qk_norm_scope", "head") or "head").lower() == "full"
        )
        self.gelu_approximate = bool(getattr(weights, "gelu_approximate", True))
        # PyTorch 2.4+ exposes a fused RMSNorm that accumulates in FP32.  It
        # replaces a seven-op reduction per normalization — with four norms per
        # layer that is the single largest source of launch overhead in decode.
        self._fused_rms_norm = getattr(self.torch.nn.functional, "rms_norm", None)
        self.parallel_residual = bool(getattr(weights, "parallel_residual", False))
        self.is_layernorm = str(weights.norm_type).lower() == "layernorm"
        position_type = str(weights.position_type or "RoPE").lower()
        self.uses_rope = position_type in {"rope", "rotary", "rotary_embedding"}
        self.is_alibi = position_type in {"alibi", "alibi_bias"}
        self.norm_eps = float(weights.norm_eps)
        ffn_kind = str(weights.ffn_type or "SwiGLU").lower()
        self.ffn_is_gelu = ffn_kind in {"gelu", "geglu"}
        self.ffn_is_relu = ffn_kind == "relu"
        self.ffn_is_relu2 = ffn_kind == "relu2"
        no_rope = getattr(weights, "no_rope_layers", None)
        self._no_rope_layers = frozenset(int(value) for value in (no_rope or ()))

        # Per-layer static schedule.  Reading ``attention_layers`` and
        # ``attention_window`` inside the decode loop cost a string compare and
        # two attribute lookups per layer per token; a small model spends more
        # time on that than on its GEMMs.
        attention_layers = getattr(weights, "attention_layers", None)
        window = int(getattr(weights, "attention_window", 0) or 0)
        self.layer_plan: list[tuple[bool, int | None, float, bool]] = []
        for index in range(self.num_layers):
            kind = (
                str(attention_layers[index]).lower()
                if isinstance(attention_layers, list) and index < len(attention_layers)
                else "global"
            )
            local = kind in {"local", "sliding_window", "window"} and window > 0
            scale = self.base_attention_scale
            if self.scale_by_layer_index:
                scale = scale / float(index + 1)
            self.layer_plan.append(
                (local, window if local else None, scale, self.uses_rope and index not in self._no_rope_layers)
            )

    def _tensor(self, value: Any) -> Any:
        return self.torch.as_tensor(
            np.asarray(value, dtype=np.float32), device=self.device, dtype=self.compute_dtype
        )

    def _norm_tensor(self, value: Any) -> Any:
        """Materialize a normalization weight in its effective form.

        Gemma stores normalization weights as offsets from unity, so the scale
        actually applied is ``1 + w``.  Folding the offset in once at load time
        keeps it out of the per-token path.
        """
        array = np.asarray(value, dtype=np.float32)
        if self.norm_offset_one:
            array = array + np.float32(1.0)
        return self.torch.as_tensor(array, device=self.device, dtype=self.compute_dtype)

    def _optional_tensor(self, value: Any | None) -> Any | None:
        return None if value is None else self._tensor(value)

    def _optional_norm_tensor(self, value: Any | None) -> Any | None:
        return None if value is None else self._norm_tensor(value)

    def _convert_layer(self, layer: Any) -> dict[str, Any | None]:
        converted: dict[str, Any | None] = {
            name: self._optional_tensor(getattr(layer, name, None))
            for name in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "attention_norm_bias", "ffn_norm_bias",
                "q_proj_bias", "k_proj_bias", "v_proj_bias",
                "o_proj_bias", "gate_proj_bias", "up_proj_bias", "down_proj_bias",
                "post_attention_norm_bias", "post_ffn_norm_bias",
            )
        }
        # Normalization weights carry the architecture's unity-offset
        # convention, so they must be materialized through _norm_tensor.
        for name in (
            "attention_norm", "ffn_norm", "q_norm", "k_norm",
            "post_attention_norm", "post_ffn_norm",
        ):
            converted[name] = self._optional_norm_tensor(getattr(layer, name, None))
        converted["router"] = self._optional_tensor(getattr(layer, "router", None))
        converted["experts"] = [
            {
                name: self._optional_tensor(getattr(expert, name))
                for name in ("gate_proj", "up_proj", "down_proj", "gate_proj_bias", "up_proj_bias", "down_proj_bias")
            }
            for expert in (getattr(layer, "experts", None) or [])
        ]
        converted["num_activated_experts"] = int(getattr(layer, "num_activated_experts", 1) or 1)

        # Pack projections that share the same input into one device GEMM.
        # The concatenation is algebraically lossless and removes two launch
        # boundaries from the decode hot path.  Keep the logical AEG layout in
        # the source engine; only the device resident representation is packed.
        hidden = int(self.weights.embedding.shape[1])

        def canonical(value: Any, role: str) -> Any:
            """Return ``value`` oriented as ``(out_features, hidden)``.

            Some source checkpoints store Conv1D weights transposed relative to
            ``nn.Linear``.  Packing must happen in one orientation, otherwise
            the concatenation joins the wrong axis and the fused GEMM contracts
            over the output dimension.
            """
            if int(value.shape[1]) == hidden:
                return value
            if int(value.shape[0]) == hidden:
                return value.transpose(0, 1)
            raise ValueError(
                f"{role} projection does not contract the model hidden size: "
                f"{tuple(value.shape)} vs {hidden}"
            )

        q, k, v = converted["q_proj"], converted["k_proj"], converted["v_proj"]
        if q is not None and k is not None and v is not None:
            q = canonical(q, "attention")
            k = canonical(k, "attention")
            v = canonical(v, "attention")
            biases = (
                converted["q_proj_bias"],
                converted["k_proj_bias"],
                converted["v_proj_bias"],
            )
            if all(bias is None for bias in biases) or all(bias is not None for bias in biases):
                converted["qkv_weight"] = self.torch.cat((q, k, v), dim=0)
                converted["qkv_bias"] = None if biases[0] is None else self.torch.cat(biases, dim=0)
                converted["q_width"] = int(q.shape[0])
                converted["k_width"] = int(k.shape[0])
                converted["v_width"] = int(v.shape[0])
                converted["q_proj"] = converted["k_proj"] = converted["v_proj"] = None
                converted["q_proj_bias"] = converted["k_proj_bias"] = converted["v_proj_bias"] = None

        gate, up = converted["gate_proj"], converted["up_proj"]
        if gate is not None and up is not None:
            gate = canonical(gate, "FFN gate")
            up = canonical(up, "FFN up")
            converted["gate_proj"], converted["up_proj"] = gate, up
        if gate is not None and up is not None and int(gate.shape[1]) == int(up.shape[1]):
            gate_bias, up_bias = converted["gate_proj_bias"], converted["up_proj_bias"]
            if (gate_bias is None) == (up_bias is None):
                converted["gate_up_weight"] = self.torch.cat((gate, up), dim=0)
                converted["gate_up_bias"] = (
                    None if gate_bias is None else self.torch.cat((gate_bias, up_bias), dim=0)
                )
                converted["gate_width"] = int(gate.shape[0])
                converted["up_width"] = int(up.shape[0])
                converted["gate_proj"] = converted["up_proj"] = None
                converted["gate_proj_bias"] = converted["up_proj_bias"] = None
        return converted

    def _ensure_rope(self, required: int, device: Any | None = None) -> None:
        device = device or self.device
        if self._cos is not None and int(self._cos.shape[0]) >= required and self._cos.device == device:
            return
        capacity = max(int(required), 2 * int(self._cos.shape[0]) if self._cos is not None else 0, 16)
        self._cos, self._sin = self._rope_tables(self.rope_theta, capacity, device)
        if self.rope_local_theta is not None:
            self._local_cos, self._local_sin = self._rope_tables(
                self.rope_local_theta, capacity, device
            )

    def _rope_tables(self, theta: float, required: int, device: Any) -> tuple[Any, Any]:
        """Build cos/sin tables for one rotary base.

        The table spans ``rotary_dim / 2`` frequencies, which is the full head
        half-width unless the architecture rotates only a prefix of each head
        (GPT-NeoX, GPT-J, StableLM, Phi).  Angles are computed in FP32 and the
        result is stored in the compute dtype, so the per-token path performs no
        dtype conversion.
        """
        half = self.rotary_dim // 2
        positions = self.torch.arange(required, device=device, dtype=self.torch.float32)[:, None]
        exponent = self.torch.arange(half, device=device, dtype=self.torch.float32) * (
            2.0 / self.rotary_dim
        )
        inv_freq = theta ** (-exponent)
        angles = positions * inv_freq[None, :]
        return (
            self.torch.cos(angles).to(dtype=self.compute_dtype),
            self.torch.sin(angles).to(dtype=self.compute_dtype),
        )

    def _rope_slice(self, positions: Any, *, local: bool) -> tuple[Any, Any]:
        """Gather the cos/sin rows for ``positions``, shaped for broadcasting.

        Hoisted out of the layer loop: the rotation factors depend only on the
        positions and the rotary base, so a decode step needs them once rather
        than twice per layer.
        """
        cos_table, sin_table = (
            (self._local_cos, self._local_sin)
            if local and self._local_cos is not None
            else (self._cos, self._sin)
        )
        assert cos_table is not None and sin_table is not None
        return (
            cos_table.index_select(0, positions).unsqueeze(1),
            sin_table.index_select(0, positions).unsqueeze(1),
        )

    def _key_positions(self, total: int) -> Any:
        """Return ``arange(total)`` from a cached, monotonically grown buffer.

        The decode loop needs this once per layer per token.  Allocating it each
        time dominates the step for a small model, so grow one buffer instead.
        """
        cache = self._positions_cache
        if cache is None or int(cache.shape[0]) < total:
            capacity = max(total, 2 * int(cache.shape[0]) if cache is not None else 0, 64)
            cache = self.torch.arange(capacity, device=self.device, dtype=self.torch.long)
            self._positions_cache = cache
        return cache[:total]

    def _norm(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        if self.is_layernorm:
            return self.torch.nn.functional.layer_norm(
                x, (int(x.shape[-1]),), weight, bias, self.norm_eps
            )
        if self._fused_rms_norm is not None:
            # One fused kernel instead of the seven-op reduction below.  It
            # accumulates in FP32 internally, so it is both faster and at least
            # as accurate as the manual form.
            return self._fused_rms_norm(x, (int(x.shape[-1]),), weight, self.norm_eps)
        # Accumulate the RMS in FP32 even when activations use FP16.  This is
        # the standard stable RMSNorm evaluation and avoids quality loss from
        # a half-precision reduction over a large hidden dimension.
        rms = self.torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.norm_eps
        ).to(dtype=x.dtype)
        return x * rms * weight

    def _linear(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        if weight.ndim != 2:
            raise ValueError(f"linear weight must be rank-2, got shape {tuple(weight.shape)}")
        input_features = int(x.shape[-1])
        if int(weight.shape[1]) == input_features:
            canonical = weight
        elif int(weight.shape[0]) == input_features:
            # Some source checkpoints store Conv1D weights as
            # (in_features, out_features), unlike nn.Linear.
            canonical = weight.transpose(0, 1)
        else:
            raise ValueError(
                "linear weight/input mismatch: input has "
                f"{input_features} features but weight shape is {tuple(weight.shape)}"
            )
        if bias is not None and int(bias.numel()) != int(canonical.shape[0]):
            raise ValueError(
                f"linear bias has {int(bias.numel())} elements but output has "
                f"{int(canonical.shape[0])} features"
            )
        return self.torch.nn.functional.linear(x, canonical, bias)

    def _rope(self, x: Any, cos: Any, sin: Any) -> Any:
        """Rotate ``(seq, heads, head_dim)`` using the declared RoPE convention.

        Three source conventions are supported and are not interchangeable:
        half-split pairing ``(x_i, x_{i+d/2})`` used by GPT-NeoX, Llama, Qwen
        and Mistral; interleaved pairing ``(x_{2i}, x_{2i+1})`` published by
        GPT-J and Cohere; and partial rotation, which leaves the trailing
        ``head_dim - rotary_dim`` channels untouched.

        ``cos``/``sin`` are prepared once per step by :meth:`_rope_slice`;
        sliding-window layers may pass factors from a separate rotary base
        (Gemma-3).
        """
        rotary = self.rotary_dim
        half = rotary // 2
        full = int(x.shape[-1])
        if self.rope_interleaved:
            even = x[..., 0:rotary:2]
            odd = x[..., 1:rotary:2]
            rotated = self.torch.stack(
                (even * cos - odd * sin, odd * cos + even * sin), dim=-1
            ).flatten(-2)
        else:
            first, second = x[..., :half], x[..., half:rotary]
            rotated = self.torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
        if rotary == full:
            return rotated
        return self.torch.cat((rotated, x[..., rotary:]), dim=-1)

    def _activation(self, gate: Any, up: Any | None) -> Any:
        torch = self.torch
        if up is None:
            if self.ffn_is_relu:
                return torch.relu(gate)
            if self.ffn_is_relu2:
                return torch.relu(gate).pow(2)
            return self._gelu(gate)
        if self.ffn_is_gelu:
            return self._gelu(gate) * up
        return torch.nn.functional.silu(gate) * up

    def _gelu(self, value: Any) -> Any:
        """Evaluate GELU in the form the source architecture declares."""
        return self.torch.nn.functional.gelu(
            value, approximate="tanh" if self.gelu_approximate else "none"
        )

    def _softcap(self, value: Any, cap: float) -> Any:
        """Bound a tensor with ``cap * tanh(value / cap)`` (Gemma-2)."""
        limit = float(cap)
        return self.torch.tanh(value / limit) * limit

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

    def _attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        query_positions: Any,
        key_positions: Any,
        window_size: int | None = None,
        scale: float | None = None,
    ) -> Any:
        """Exact attention, dispatching to PyTorch's fused SDPA when possible.

        The boolean mask is intentionally expressed in source token
        positions, not cache row indices.  That keeps the same semantics for
        normal prefill, incremental decode, local GPT-Neo layers, and future
        cache implementations that compact rows.
        """
        torch = self.torch
        is_alibi = self.is_alibi
        query_length = int(q.shape[0])
        key_length = int(k.shape[0])
        is_prefill = query_length > 1 and key_length == query_length
        # TorchAEGEngine owns a monotonic, zero-based cache.  The caller passes
        # ``arange(total)`` as key positions, so this invariant is known from
        # tensor shapes and does not require reading a CUDA scalar.  The old
        # implementation called .item() on position tensors for every layer
        # and every generated token, synchronizing the host with the GPU and
        # destroying decode throughput.
        contiguous = True
        local = window_size is not None and int(window_size) > 0
        if local and key_length <= int(window_size):
            # Every cached key is inside the window, so the sliding constraint
            # is vacuous.  Dropping it lets SDPA keep its fused path instead of
            # falling back to the math backend for an all-true mask — material
            # for GPT-Neo and Gemma, where half or more of the layers are local.
            local = False
            window_size = None
        scale = self.base_attention_scale if scale is None else float(scale)
        softcap = self.attn_logit_softcap

        # SDPA accepts [batch, heads, query, dim].  enable_gqa avoids
        # materializing repeated KV heads on supported PyTorch versions.
        # SDPA has no soft-capping stage, so architectures that declare one
        # (Gemma-2) must take the exact path below.
        if not is_alibi and not softcap:
            q4 = q.transpose(0, 1).unsqueeze(0)
            k4 = k.transpose(0, 1).unsqueeze(0)
            v4 = v.transpose(0, 1).unsqueeze(0)
            attn_mask = None
            is_causal = bool(is_prefill and contiguous and not local)
            if not is_causal and not (query_length == 1 and not local):
                allowed = key_positions[None, :] <= query_positions[:, None]
                if local:
                    allowed &= key_positions[None, :] >= (
                        query_positions[:, None] - int(window_size) + 1
                    )
                attn_mask = allowed
            try:
                # Passing enable_gqa=True for ordinary MHA can prevent the
                # backend from selecting its fastest FlashAttention path on
                # some PyTorch/CUDA combinations.  Only request GQA when the
                # K/V head geometry actually requires it.
                sdpa_kwargs = {
                    "attn_mask": attn_mask,
                    "dropout_p": 0.0,
                    "is_causal": is_causal,
                    "scale": scale,
                }
                if self.num_kv_heads != self.num_heads:
                    sdpa_kwargs["enable_gqa"] = True
                context = torch.nn.functional.scaled_dot_product_attention(
                    q4, k4, v4, **sdpa_kwargs,
                )
                return context.squeeze(0).transpose(0, 1)
            except (TypeError, RuntimeError):
                # Older PyTorch builds do not expose enable_gqa or may lack a
                # suitable fused backend.  The exact fallback remains valid.
                pass

        repeats = self.num_heads // self.num_kv_heads
        if repeats == 1:
            k_full, v_full = k, v
        else:
            k_full = k.repeat_interleave(repeats, dim=1)
            v_full = v.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k_full) * scale
        if is_alibi:
            distance = key_positions[None, :] - query_positions[:, None]
            scores = scores + self._alibi_slopes[:, None, None] * distance.to(dtype=scores.dtype)[None, :, :]
        if softcap:
            # Gemma-2 bounds attention logits before masking; capping after the
            # mask would turn -inf into -cap and admit masked positions.
            scores = self._softcap(scores, softcap)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if local:
            allowed &= key_positions[None, :] >= query_positions[:, None] - int(window_size) + 1
        scores = scores.masked_fill(~allowed.unsqueeze(0), -torch.finfo(scores.dtype).max)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hqk,khd->qhd", probs, v_full)

    def _append_kv(self, old: Any | None, value: Any, past: int, total: int, reserve: int = 0) -> Any:
        """Append KV in amortized-linear storage instead of copying the prefix.

        ``reserve`` lets a caller that already knows the final sequence length
        allocate once, removing every reallocation and prefix copy from the
        decode loop.
        """
        torch = self.torch
        if old is None or int(old.shape[0]) < total:
            old_capacity = 0 if old is None else int(old.shape[0])
            capacity = max(total, reserve, max(16, old_capacity * 2))
            result = torch.empty(
                (capacity, *tuple(value.shape[1:])), dtype=value.dtype, device=value.device
            )
            if old is not None and past:
                result[:past].copy_(old[:past])
        else:
            result = old
        result[past:total].copy_(value)
        return result

    def _forward_device(
        self,
        token_ids: np.ndarray | Any,
        cache: TorchKVCache | None = None,
        *,
        validate_ids: bool = False,
        reserve: int = 0,
    ) -> tuple[Any, TorchKVCache]:
        """Forward pass that keeps logits on the accelerator for generation."""
        torch = self.torch
        # Keep decode tokens on the execution device.  The old path converted
        # every one-token decode step through NumPy, which forced a host-to-
        # device copy and made CUDA's scalar validation below synchronize the
        # stream.  Public ``forward`` and the initial prompt still opt into
        # validation; internally generated tokens are already produced by the
        # model and do not need a second round trip through the host.
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.reshape(-1)
            if ids.device != self.device or ids.dtype != torch.long:
                ids = ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(
                np.asarray(token_ids, dtype=np.int64).reshape(-1),
                device=self.device,
            )
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if validate_ids and (int(ids.min()) < 0 or int(ids.max()) >= self.embedding.shape[0]):
            raise ValueError("token id is outside the compiled vocabulary")
        cache = cache or TorchKVCache([None] * self.num_layers, [None] * self.num_layers)
        past = int(cache.length)
        seq_len = int(ids.numel())
        positions = torch.arange(past, past + seq_len, device=self.device, dtype=torch.long)
        if self.uses_rope:
            self._ensure_rope(past + seq_len)
        hidden = self.embedding.index_select(0, ids)
        if self.embedding_scale is not None:
            # Gemma scales embeddings by sqrt(hidden_size); Granite uses an
            # explicit embedding_multiplier.  Both are part of the model.
            hidden = hidden * hidden.new_tensor(self.embedding_scale)
        if self.embedding_norm is not None:
            hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding.index_select(0, positions)

        post_norm = self.norm_placement == "post"
        parallel = self.parallel_residual
        residual_scale = self.residual_scale
        # The rotation factors depend only on the positions and the rotary base,
        # so gather them once per step instead of twice per layer.
        rope_global = self._rope_slice(positions, local=False) if self.uses_rope else None
        rope_local = (
            self._rope_slice(positions, local=True)
            if self.uses_rope and self._local_cos is not None
            else rope_global
        )
        with torch.inference_mode():
            for index, layer in enumerate(self.layers):
                local_attention, attention_window, attention_scale, layer_uses_rope = (
                    self.layer_plan[index]
                )
                block_input = hidden
                if post_norm:
                    # OLMo-2 feeds the raw residual into each sublayer and
                    # normalizes the sublayer output instead.
                    normed = hidden
                else:
                    normed = self._norm(
                        hidden, layer["attention_norm"], layer["attention_norm_bias"]
                    )
                if layer.get("qkv_weight") is not None:
                    qkv = self._linear(normed, layer["qkv_weight"], layer["qkv_bias"])
                    q_width = int(layer["q_width"])
                    k_width = int(layer["k_width"])
                    q = qkv[..., :q_width]
                    k = qkv[..., q_width:q_width + k_width]
                    v = qkv[..., q_width + k_width:]
                else:
                    q = self._linear(normed, layer["q_proj"], layer["q_proj_bias"])
                    k = self._linear(normed, layer["k_proj"], layer["k_proj_bias"])
                    v = self._linear(normed, layer["v_proj"], layer["v_proj_bias"])
                if self.qk_norm_is_full:
                    # OLMo-2 normalizes the whole projection, before it is
                    # split into heads.
                    if layer["q_norm"] is not None:
                        q = self._norm(q, layer["q_norm"])
                    if layer["k_norm"] is not None:
                        k = self._norm(k, layer["k_norm"])
                q = q.reshape(seq_len, self.num_heads, self.head_dim)
                k = k.reshape(seq_len, self.num_kv_heads, self.head_dim)
                if not self.qk_norm_is_full:
                    if layer["q_norm"] is not None:
                        q = self._norm(q, layer["q_norm"])
                    if layer["k_norm"] is not None:
                        k = self._norm(k, layer["k_norm"])
                if layer_uses_rope:
                    cos, sin = rope_local if local_attention else rope_global
                    q = self._rope(q, cos, sin)
                    k = self._rope(k, cos, sin)
                v = v.reshape(seq_len, self.num_kv_heads, self.head_dim)
                total = past + seq_len
                k_all = self._append_kv(cache.keys[index], k, past, total, reserve)
                v_all = self._append_kv(cache.values[index], v, past, total, reserve)
                context = self._attention(
                    q, k_all[:total], v_all[:total], positions,
                    self._key_positions(total), attention_window, attention_scale,
                )
                cache.keys[index] = k_all
                cache.values[index] = v_all
                attention_out = self._linear(
                    context.reshape(seq_len, self.num_heads * self.head_dim),
                    layer["o_proj"], layer["o_proj_bias"],
                )
                # ``sandwich`` blocks normalize the sublayer output too
                # (Gemma-2/3, EXAONE-4); ``post`` blocks normalize it instead
                # of the input (OLMo-2).  Both precede the residual add.
                if layer["post_attention_norm"] is not None:
                    attention_out = self._norm(
                        attention_out,
                        layer["post_attention_norm"],
                        layer["post_attention_norm_bias"],
                    )
                elif post_norm:
                    attention_out = self._norm(
                        attention_out, layer["attention_norm"], layer["attention_norm_bias"]
                    )
                if residual_scale is not None:
                    attention_out = attention_out * attention_out.new_tensor(residual_scale)

                if parallel:
                    # GPT-J, GPT-NeoX, Falcon and Cohere evaluate the
                    # feed-forward branch from the *block input*, not from the
                    # post-attention state:
                    #   y = x + Attn(N1(x)) + FFN(N2(x)).
                    # Checkpoints with a single block norm bind ffn_norm to the
                    # same tensor, so one rule covers both spellings.
                    ffn_input = self._norm(
                        block_input, layer["ffn_norm"], layer["ffn_norm_bias"]
                    )
                elif post_norm:
                    hidden = block_input + attention_out
                    ffn_input = hidden
                else:
                    hidden = hidden + attention_out
                    ffn_input = self._norm(
                        hidden, layer["ffn_norm"], layer["ffn_norm_bias"]
                    )
                if layer["experts"]:
                    ffn_out = self._moe_ffn(ffn_input, layer)
                else:
                    if layer.get("gate_up_weight") is not None:
                        gate_up = self._linear(ffn_input, layer["gate_up_weight"], layer["gate_up_bias"])
                        gate_width = int(layer["gate_width"])
                        gate = gate_up[..., :gate_width]
                        up = gate_up[..., gate_width:]
                    else:
                        gate = self._linear(ffn_input, layer["gate_proj"], layer["gate_proj_bias"])
                        up = self._linear(ffn_input, layer["up_proj"], layer["up_proj_bias"]) if layer["up_proj"] is not None else None
                    ffn_out = self._linear(self._activation(gate, up), layer["down_proj"], layer["down_proj_bias"])
                if layer["post_ffn_norm"] is not None:
                    ffn_out = self._norm(
                        ffn_out, layer["post_ffn_norm"], layer["post_ffn_norm_bias"]
                    )
                elif post_norm:
                    ffn_out = self._norm(ffn_out, layer["ffn_norm"], layer["ffn_norm_bias"])
                if residual_scale is not None:
                    ffn_out = ffn_out * ffn_out.new_tensor(residual_scale)
                if parallel:
                    hidden = block_input + attention_out + ffn_out
                else:
                    hidden = hidden + ffn_out
            hidden = self._norm(hidden, self.final_norm, self.final_norm_bias)
            logits = self._linear(hidden, self.lm_head)
            if self.logit_scale is not None:
                logits = logits * logits.new_tensor(self.logit_scale)
            if self.final_logit_softcap:
                logits = self._softcap(logits, self.final_logit_softcap)
            cache.length = past + seq_len
            cache.last_logits = logits[-1].detach()
        return logits, cache

    def forward(self, token_ids: np.ndarray | Any, cache: TorchKVCache | None = None) -> tuple[np.ndarray, TorchKVCache]:
        logits, cache = self._forward_device(token_ids, cache, validate_ids=True)
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
            # The final sequence length is known here, so the KV cache can be
            # sized once instead of doubling during decode.
            reserve = int(ids.size) + int(max_tokens) + (0 if cache is None else int(cache.length))
            _, cache = self._forward_device(ids, cache, validate_ids=True, reserve=reserve)
            next_logits = cache.last_logits
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
            # ``token`` is sampled on-device.  Reusing a device tensor avoids
            # a synchronization plus a NumPy/device copy on every decode
            # iteration, which is material for small-batch generation.
            _, cache = self._forward_device(
                self.torch.tensor([token], dtype=self.torch.long, device=self.device),
                cache,
            )
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
