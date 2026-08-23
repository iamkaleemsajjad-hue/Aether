"""
Reference CPU execution engine for compiled AEG models.

Runs a real transformer forward pass on numpy arrays, dispatching the hot
primitives to the natively compiled kernels in
:mod:`aether.kernels.native_cpu` when a host compiler is available.

This is a *reference* engine: correctness and portability come first, and it
exists so a compiled ``.aeg`` artifact can actually produce logits on any machine
without a GPU. It implements the standard decoder-only stack used by Llama, Qwen
and Mistral:

    embed -> [RMSNorm -> GQA attention -> residual
              -> RMSNorm -> SwiGLU FFN -> residual] x L
          -> RMSNorm -> LM head

Grouped-query attention, RoPE, and an incremental KV cache are supported, so
decode is O(1) in past length rather than re-running the full prefill each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.kernels.native_cpu import NativeCPUKernels, get_native_kernels
from aether.runtime.positional import alibi_slopes
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ExpertWeights", "LayerWeights", "ModelWeights", "KVCache",
    "CPUExecutionEngine",
]


@dataclass
class ExpertWeights:
    """One routed expert's SwiGLU projections.

    Matrices use the same ``(out_features, in_features)`` orientation as the
    dense transformer projections.  This is the executable representation of
    the top-k MoE contract described by Switch/Mixtral-style models.
    """

    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray
    gate_proj_bias: np.ndarray | None = None
    up_proj_bias: np.ndarray | None = None
    down_proj_bias: np.ndarray | None = None

    def validate(self, layer_index: int, expert_index: int, hidden_size: int) -> None:
        if self.gate_proj.ndim != 2 or self.gate_proj.shape[1] != hidden_size:
            raise ValueError(
                f"layer {layer_index} expert {expert_index}: invalid gate projection"
            )
        if self.up_proj.shape != self.gate_proj.shape:
            raise ValueError(
                f"layer {layer_index} expert {expert_index}: gate/up shapes differ"
            )
        if self.down_proj.ndim != 2 or self.down_proj.shape[1] != self.gate_proj.shape[0]:
            raise ValueError(
                f"layer {layer_index} expert {expert_index}: invalid down projection"
            )


@dataclass
class LayerWeights:
    """Weight matrices for one transformer layer.

    All projections are stored in ``(out_features, in_features)`` orientation,
    matching the HuggingFace checkpoint convention.
    """

    attention_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    ffn_norm: np.ndarray
    gate_proj: np.ndarray | None
    up_proj: np.ndarray | None
    down_proj: np.ndarray | None
    attention_norm_bias: np.ndarray | None = None
    ffn_norm_bias: np.ndarray | None = None
    q_proj_bias: np.ndarray | None = None
    k_proj_bias: np.ndarray | None = None
    v_proj_bias: np.ndarray | None = None
    o_proj_bias: np.ndarray | None = None
    gate_proj_bias: np.ndarray | None = None
    up_proj_bias: np.ndarray | None = None
    down_proj_bias: np.ndarray | None = None
    q_norm: np.ndarray | None = None
    k_norm: np.ndarray | None = None
    router: np.ndarray | None = None
    experts: list[ExpertWeights] = field(default_factory=list)
    num_activated_experts: int = 1

    def validate(self, layer_index: int, hidden_size: int) -> None:
        """Check that the projections compose into a valid layer."""
        if self.q_proj.shape[1] != hidden_size:
            msg = (
                f"layer {layer_index}: q_proj expects input {self.q_proj.shape[1]}, "
                f"but hidden_size is {hidden_size}"
            )
            raise ValueError(msg)
        if self.k_proj.shape != self.v_proj.shape:
            msg = (
                f"layer {layer_index}: k_proj {self.k_proj.shape} and "
                f"v_proj {self.v_proj.shape} must match"
            )
            raise ValueError(msg)
        if self.experts:
            if self.router is None or self.router.ndim != 2 or self.router.shape[1] != hidden_size:
                raise ValueError(f"layer {layer_index}: MoE router must have shape (experts, hidden)")
            if not 0 < self.num_activated_experts <= len(self.experts):
                raise ValueError(f"layer {layer_index}: invalid MoE top-k")
            for expert_index, expert in enumerate(self.experts):
                expert.validate(layer_index, expert_index, hidden_size)
            return
        if self.gate_proj is None or self.down_proj is None:
            raise ValueError(f"layer {layer_index}: dense FFN projections are incomplete")
        if self.up_proj is not None and self.gate_proj.shape != self.up_proj.shape:
            msg = (
                f"layer {layer_index}: gate_proj {self.gate_proj.shape} and "
                f"up_proj {self.up_proj.shape} must match"
            )
            raise ValueError(msg)
        if self.q_norm is not None and self.q_norm.ndim != 1:
            raise ValueError(f"layer {layer_index}: q_norm must be a vector")
        if self.k_norm is not None and self.k_norm.ndim != 1:
            raise ValueError(f"layer {layer_index}: k_norm must be a vector")


@dataclass
class ModelWeights:
    """All tensors needed to run a decoder-only transformer."""

    embedding: np.ndarray
    layers: list[LayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray
    position_embedding: np.ndarray | None = None
    embedding_norm: np.ndarray | None = None
    embedding_norm_bias: np.ndarray | None = None
    final_norm_bias: np.ndarray | None = None
    position_type: str = "RoPE"
    #: RoPE base frequency.
    rope_theta: float = 10000.0
    #: RMSNorm epsilon.
    norm_eps: float = 1e-5
    #: Decoder normalization family.
    norm_type: str = "RMSNorm"
    #: FFN activation family.
    ffn_type: str = "SwiGLU"
    #: Whether attention and FFN branches share one pre-norm input.
    parallel_residual: bool = False
    attention_layers: list[str] | None = None
    attention_window: int | None = None

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def validate(self) -> None:
        """Verify the weights are mutually consistent before execution."""
        if self.embedding.ndim != 2:
            msg = f"embedding must be 2-D (vocab, hidden), got shape {self.embedding.shape}"
            raise ValueError(msg)
        if self.lm_head.shape[1] != self.hidden_size:
            msg = (
                f"lm_head input dim {self.lm_head.shape[1]} does not match "
                f"hidden_size {self.hidden_size}"
            )
            raise ValueError(msg)
        if self.final_norm.size != self.hidden_size:
            msg = (
                f"final_norm has {self.final_norm.size} elements, "
                f"expected hidden_size {self.hidden_size}"
            )
            raise ValueError(msg)
        if self.position_embedding is not None and (
            self.position_embedding.ndim != 2
            or self.position_embedding.shape[1] != self.hidden_size
        ):
            raise ValueError(
                "position_embedding must have shape (context_length, hidden_size)"
            )
        if self.embedding_norm is not None and self.embedding_norm.size != self.hidden_size:
            raise ValueError("embedding_norm must have hidden_size elements")
        if self.embedding_norm_bias is not None and self.embedding_norm_bias.size != self.hidden_size:
            raise ValueError("embedding_norm_bias must have hidden_size elements")
        if self.final_norm_bias is not None and self.final_norm_bias.size != self.hidden_size:
            raise ValueError(
                f"final_norm_bias has {self.final_norm_bias.size} elements, "
                f"expected hidden_size {self.hidden_size}"
            )
        for index, layer in enumerate(self.layers):
            layer.validate(index, self.hidden_size)


@dataclass
class KVCache:
    """Incremental key/value cache for one model.

    Stores per-layer keys and values as ``(seq, kv_heads, head_dim)`` and grows by
    appending. Without this, each decode step would recompute the whole prefix.
    """

    num_layers: int
    keys: list[np.ndarray | None] = field(default_factory=list)
    values: list[np.ndarray | None] = field(default_factory=list)
    positions: list[np.ndarray | None] = field(default_factory=list)
    #: Logical (uncompressed) sequence length.  Stored KV rows may be fewer
    #: when a verified semantic-KV plan is active, but RoPE and causal masks
    #: must continue to use the original token positions.
    logical_length: int = 0
    #: Logits for the final cached position.  This allows a session to append
    #: an empty suffix without recomputing the entire prefix.
    last_logits: np.ndarray | None = None
    #: Final normalized hidden state for an optional MTP drafter.
    last_hidden: np.ndarray | None = None
    #: Allocated capacity and logical row count are tracked separately so
    #: decode does not concatenate the complete KV history every token.
    stored_lengths: list[int] = field(default_factory=list)
    reserve_length: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            self.keys = [None] * self.num_layers
            self.values = [None] * self.num_layers
        if not self.positions:
            self.positions = [None] * self.num_layers
        if not self.stored_lengths:
            self.stored_lengths = [0] * self.num_layers
        elif len(self.stored_lengths) != self.num_layers:
            raise ValueError("stored_lengths must match num_layers")

    @property
    def length(self) -> int:
        """Logical number of cached positions, including compressed tokens."""
        return self.logical_length

    @property
    def stored_length(self) -> int:
        """Number of physically stored KV rows in the first layer."""
        return self.stored_lengths[0] if self.stored_lengths else 0

    def reserve(self, length: int) -> None:
        """Request a future capacity without changing the logical cache."""
        if length < 0:
            raise ValueError("cache reserve length must be non-negative")
        self.reserve_length = max(self.reserve_length, int(length))

    def layer_view(self, layer: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return valid KV rows, excluding geometric spare capacity."""
        key = self.keys[layer]
        value = self.values[layer]
        positions = self.positions[layer]
        if key is None or value is None or positions is None:
            return None, None, None
        length = self.stored_lengths[layer]
        return key[:length], value[:length], positions[:length]

    def append(
        self,
        layer: int,
        key: np.ndarray,
        value: np.ndarray,
        positions: np.ndarray | None = None,
        *,
        update_length: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Append this step's key/value and return values plus token positions.

        ``update_length=False`` is used by the transformer, which appends one
        layer at a time and advances the shared logical length only after all
        layers have completed.  This prevents a compressed cache from treating
        each layer as additional sequence tokens.
        """
        key = np.ascontiguousarray(key, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)
        if key.shape[0] != value.shape[0]:
            raise ValueError("KV key/value sequence lengths must match")
        if positions is None:
            start = self.logical_length
            positions = np.arange(start, start + key.shape[0], dtype=np.int64)
        else:
            positions = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
            if positions.size != key.shape[0]:
                raise ValueError("KV position count must match key/value sequence length")
        current = self.stored_lengths[layer]
        required = current + key.shape[0]
        if self.keys[layer] is None:
            capacity = max(required, self.reserve_length)
            self.keys[layer] = np.empty((capacity, *key.shape[1:]), dtype=np.float32)
            self.values[layer] = np.empty((capacity, *value.shape[1:]), dtype=np.float32)
            self.positions[layer] = np.empty(capacity, dtype=np.int64)
        elif required > self.keys[layer].shape[0]:
            capacity = max(required, max(1, self.keys[layer].shape[0] * 2))
            old_key, old_value, old_positions = self.layer_view(layer)
            if old_key is None or old_value is None or old_positions is None:
                raise ValueError("KV cache has incomplete layer state")
            new_key = np.empty((capacity, *key.shape[1:]), dtype=np.float32)
            new_value = np.empty((capacity, *value.shape[1:]), dtype=np.float32)
            new_positions = np.empty(capacity, dtype=np.int64)
            new_key[:current] = old_key
            new_value[:current] = old_value
            new_positions[:current] = old_positions
            self.keys[layer], self.values[layer], self.positions[layer] = (
                new_key, new_value, new_positions
            )
        if self.keys[layer] is None or self.values[layer] is None or self.positions[layer] is None:
            raise ValueError("KV cache allocation failed")
        self.keys[layer][current:required] = key
        self.values[layer][current:required] = value
        self.positions[layer][current:required] = positions
        self.stored_lengths[layer] = required
        if update_length:
            self.logical_length = max(
                self.logical_length,
                int(positions[-1]) + 1 if positions.size else self.logical_length,
            )
        stored_positions = self.positions[layer]
        if stored_positions is None:
            raise ValueError("KV cache append did not produce token positions")
        stored_key, stored_value, stored_positions = self.layer_view(layer)
        if stored_key is None or stored_value is None or stored_positions is None:
            raise ValueError("KV cache append did not produce stored positions")
        return stored_key, stored_value, stored_positions

    def replace_layer(
        self,
        layer: int,
        key: np.ndarray,
        value: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """Replace one layer with a validated compressed KV representation."""
        key = np.ascontiguousarray(key, dtype=np.float32)
        value = np.ascontiguousarray(value, dtype=np.float32)
        positions = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
        if key.shape[0] != value.shape[0] or key.shape[0] != positions.size:
            raise ValueError("compressed KV rows and positions must have matching lengths")
        if positions.size and np.any(np.diff(positions) <= 0):
            raise ValueError("compressed KV positions must remain strictly increasing")
        self.keys[layer] = key
        self.values[layer] = value
        self.positions[layer] = positions
        self.stored_lengths[layer] = int(positions.size)

    def advance(self, token_count: int) -> None:
        """Advance the shared logical sequence length after a forward pass."""
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        self.logical_length += int(token_count)

    def clone(self) -> "KVCache":
        """Copy cache state so multiple requests can safely diverge from a prefix."""
        return KVCache(
            num_layers=self.num_layers,
            keys=[None if value is None else value.copy() for value in self.keys],
            values=[None if value is None else value.copy() for value in self.values],
            positions=[None if value is None else value.copy() for value in self.positions],
            logical_length=self.logical_length,
            last_logits=None if self.last_logits is None else self.last_logits.copy(),
            last_hidden=None if self.last_hidden is None else self.last_hidden.copy(),
            stored_lengths=list(self.stored_lengths),
            reserve_length=self.reserve_length,
        )

    def reset(self) -> None:
        """Drop all cached state."""
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self.positions = [None] * self.num_layers
        self.stored_lengths = [0] * self.num_layers
        self.reserve_length = 0
        self.logical_length = 0
        self.last_logits = None
        self.last_hidden = None


class CPUExecutionEngine:
    """Executes a decoder-only transformer on CPU.

    Args:
        weights: The model tensors.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads; equals ``num_heads`` for MHA and
            is smaller for grouped-query attention.
        kernels: Kernel provider; defaults to the shared native instance.
    """

    def __init__(
        self,
        weights: ModelWeights,
        num_heads: int,
        num_kv_heads: int | None = None,
        kernels: NativeCPUKernels | None = None,
        lora_adapters: dict[str, dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]]] | None = None,
        active_lora_adapter: str | None = None,
        sparse_attention_plan: dict[str, Any] | None = None,
        semantic_kv_plan: dict[str, Any] | None = None,
        cross_layer_kv_plan: dict[str, Any] | None = None,
    ) -> None:
        weights.validate()
        self.weights = weights
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads != 0:
            msg = (
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention"
            )
            raise ValueError(msg)
        if weights.layers and hasattr(weights.layers[0], "q_proj") and getattr(weights.layers[0], "q_proj") is not None:
            self.head_dim = weights.layers[0].q_proj.shape[0] // num_heads
        else:
            self.head_dim = weights.hidden_size // num_heads
        if self.head_dim % 2 != 0:
            msg = f"head_dim must be even for RoPE, got {self.head_dim}"
            raise ValueError(msg)
        self.kernels = kernels or get_native_kernels()
        self.lora_adapters = lora_adapters or {}
        self.active_lora_adapter = active_lora_adapter
        self.sparse_attention_plan = sparse_attention_plan
        self.semantic_kv_plan = self._validate_semantic_kv_plan(semantic_kv_plan)
        self.cross_layer_kv_plan = self._validate_cross_layer_kv_plan(cross_layer_kv_plan, len(weights.layers))
        self._alibi_slopes = alibi_slopes(self.num_heads)
        self._speculative_stats = {
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "cycles": 0,
        }
        self._validate_lora_adapters()
        self._cos, self._sin = self._build_rope_tables()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_rope_tables(self, max_positions: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Precompute RoPE cos/sin tables for positions ``[0, max_positions)``."""
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.weights.rope_theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / self.head_dim)
        )
        angles = np.arange(max_positions, dtype=np.float64)[:, None] * inv_freq[None, :]
        return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def with_task_deltas(self, deltas: dict[str, np.ndarray]) -> "CPUExecutionEngine":
        """Return a request-local engine with validated task-vector deltas applied.

        The base engine is never mutated, so concurrent requests retain
        isolated model state.  This is intentionally an explicit copy-on-write
        operation: a caller can only obtain it after loading authenticated AEG
        task-vector payloads, and malformed names/shapes fail before decode.
        """
        if not deltas:
            return self

        def updated(name: str, value: np.ndarray | None) -> np.ndarray | None:
            if value is None:
                if name in deltas:
                    raise ValueError(f"task delta {name!r} targets a missing base tensor")
                return None
            delta = deltas.get(name)
            if delta is None:
                return np.array(value, dtype=np.float32, copy=True)
            candidate = np.asarray(delta, dtype=np.float32)
            if candidate.shape != value.shape:
                raise ValueError(
                    f"task delta {name!r} shape {candidate.shape} does not match base {value.shape}"
                )
            return np.ascontiguousarray(value.astype(np.float32) + candidate, dtype=np.float32)

        layers: list[LayerWeights] = []
        for index, layer in enumerate(self.weights.layers):
            fields = {
                field_name: updated(
                    f"layer_{index}_{field_name}",
                    getattr(layer, field_name),
                )
                for field_name in (
                    "attention_norm", "q_proj", "k_proj", "v_proj", "o_proj",
                "ffn_norm", "gate_proj", "up_proj", "down_proj",
                "attention_norm_bias", "ffn_norm_bias",
                "q_proj_bias", "k_proj_bias", "v_proj_bias", "o_proj_bias",
                "gate_proj_bias", "up_proj_bias", "down_proj_bias",
                )
            }
            if layer.experts:
                fields["router"] = updated(f"layer_{index}_moe_router", layer.router)
                fields["experts"] = [
                    ExpertWeights(
                        gate_proj=updated(
                            f"layer_{index}_expert_{expert_index}_gate_proj",
                            expert.gate_proj,
                        ),
                        up_proj=updated(
                            f"layer_{index}_expert_{expert_index}_up_proj",
                            expert.up_proj,
                        ),
                        down_proj=updated(
                            f"layer_{index}_expert_{expert_index}_down_proj",
                            expert.down_proj,
                        ),
                    )
                    for expert_index, expert in enumerate(layer.experts)
                ]
                fields["num_activated_experts"] = layer.num_activated_experts
            layers.append(LayerWeights(**fields))
        weights = ModelWeights(
            embedding=updated("embedding", self.weights.embedding),
            layers=layers,
            final_norm=updated("final_norm", self.weights.final_norm),
            lm_head=updated("lm_head", self.weights.lm_head),
            position_embedding=(
                updated("position_embedding", self.weights.position_embedding)
                if self.weights.position_embedding is not None
                else None
            ),
            embedding_norm=(
                updated("embedding_norm", self.weights.embedding_norm)
                if self.weights.embedding_norm is not None else None
            ),
            embedding_norm_bias=(
                updated("embedding_norm_bias", self.weights.embedding_norm_bias)
                if self.weights.embedding_norm_bias is not None else None
            ),
            final_norm_bias=(
                updated("final_norm_bias", self.weights.final_norm_bias)
                if self.weights.final_norm_bias is not None
                else None
            ),
            rope_theta=self.weights.rope_theta,
            norm_eps=self.weights.norm_eps,
            norm_type=self.weights.norm_type,
            ffn_type=self.weights.ffn_type,
            position_type=self.weights.position_type,
            parallel_residual=self.weights.parallel_residual,
            attention_layers=self.weights.attention_layers,
            attention_window=self.weights.attention_window,
        )
        return CPUExecutionEngine(
            weights,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kernels=self.kernels,
            lora_adapters=self.lora_adapters,
            active_lora_adapter=self.active_lora_adapter,
            sparse_attention_plan=self.sparse_attention_plan,
            semantic_kv_plan=self.semantic_kv_plan,
            cross_layer_kv_plan=self.cross_layer_kv_plan,
        )

    def with_lora_adapter(
        self,
        adapters: dict[str, dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]]],
        adapter_id: str | None,
    ) -> "CPUExecutionEngine":
        """Return an immutable request-local engine with one adapter selected."""
        if adapter_id is not None and adapter_id not in adapters:
            raise ValueError(f"unknown compiled LoRA adapter {adapter_id!r}")
        return CPUExecutionEngine(
            self.weights,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kernels=self.kernels,
            lora_adapters=adapters,
            active_lora_adapter=adapter_id,
            sparse_attention_plan=self.sparse_attention_plan,
            semantic_kv_plan=self.semantic_kv_plan,
            cross_layer_kv_plan=self.cross_layer_kv_plan,
        )

    @staticmethod
    def _validate_semantic_kv_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate the persisted Pass 14 plan before enabling compression."""
        if plan is None:
            return None
        if not isinstance(plan, dict) or plan.get("format") != "aether_kv_compression_v1":
            raise ValueError("semantic KV plan has an unsupported format")
        layers = plan.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("semantic KV plan must contain layer policies")
        strategy = str(plan.get("strategy", "chunk"))
        if strategy not in {"chunk", "hybrid"}:
            raise ValueError(
                f"semantic KV strategy {strategy!r} requires tokenizer boundary metadata; "
                "only chunk and hybrid plans are executable by the CPU cache"
            )
        for index, item in enumerate(layers):
            if not isinstance(item, dict):
                raise ValueError(f"semantic KV layer {index} must be an object")
            retention = float(item.get("retention_ratio", -1.0))
            chunk_size = int(item.get("chunk_size", 0))
            if not 0.0 < retention <= 1.0 or chunk_size <= 0:
                raise ValueError(f"invalid semantic KV policy for layer {index}")
        return plan

    def _semantic_kv_policy(self, layer: int) -> dict[str, Any] | None:
        if self.semantic_kv_plan is None:
            return None
        layers = self.semantic_kv_plan["layers"]
        if layer >= len(layers):
            raise ValueError("semantic KV plan has fewer policies than model layers")
        return layers[layer]

    @staticmethod
    def _validate_cross_layer_kv_plan(
        plan: dict[str, Any] | None,
        num_layers: int,
    ) -> dict[str, Any] | None:
        """Validate Pass 15 aliases before they can affect inference."""
        if plan is None:
            return None
        if not isinstance(plan, dict) or plan.get("format") != "aether_cross_layer_kv_v1":
            raise ValueError("cross-layer KV plan has an unsupported format")
        if int(plan.get("n_layers", -1)) != num_layers:
            raise ValueError("cross-layer KV plan layer count does not match the model")
        groups = plan.get("sharing_groups")
        if not isinstance(groups, list):
            raise ValueError("cross-layer KV plan must contain sharing_groups")
        targets: set[int] = set()
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("cross-layer KV sharing group must be an object")
            src = int(group.get("src_layer", -1))
            shared = group.get("shared_with")
            if src < 0 or src >= num_layers or not isinstance(shared, list):
                raise ValueError("invalid cross-layer KV source group")
            for target_value in shared:
                target = int(target_value)
                # The streaming CPU engine can only alias a source whose cache
                # has already been computed in this forward pass.
                if target <= src or target >= num_layers or target in targets:
                    raise ValueError("cross-layer KV targets must have one earlier source layer")
                targets.add(target)
        return plan

    def _cross_layer_kv_source(self, layer: int) -> int | None:
        if self.cross_layer_kv_plan is None:
            return None
        for group in self.cross_layer_kv_plan["sharing_groups"]:
            if layer in group["shared_with"]:
                return int(group["src_layer"])
        return None

    def _compress_semantic_kv(
        self,
        cache: KVCache,
        layer: int,
        key: np.ndarray,
        value: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the real Pass 14 chunk compressor while preserving positions."""
        policy = self._semantic_kv_policy(layer)
        if policy is None or float(policy["retention_ratio"]) >= 1.0 or key.shape[0] < 2:
            return key, value, positions
        from aether.compiler.stage2_optimizer.pass14_semantic_kv_compression import chunk_kv_compress

        _compressed_keys, _compressed_values, retained = chunk_kv_compress(
            # The pass operates on one vector per token.  For grouped-query
            # attention, cluster the mean key across KV heads, then retain the
            # same rows for every head so the cache remains shape-aligned.
            key.mean(axis=1).tolist(),
            value.mean(axis=1).tolist(),
            retention_ratio=float(policy["retention_ratio"]),
            chunk_size=int(policy["chunk_size"]),
        )
        # The newest token is always required for the next decode step's
        # causal self-attention.  The representative policy may otherwise
        # choose the first row of a merged chunk and drop this position.
        if retained and retained[-1] != len(positions) - 1:
            retained = sorted(set(retained) | {len(positions) - 1})
        if not retained or len(retained) >= len(positions):
            return key, value, positions
        retained_positions = positions[np.asarray(retained, dtype=np.int64)]
        return (
            key[np.asarray(retained, dtype=np.int64)],
            value[np.asarray(retained, dtype=np.int64)],
            retained_positions,
        )

    def _validate_lora_adapters(self) -> None:
        """Validate adapter targets against the actual compiled model shapes."""
        for adapter_id, targets in self.lora_adapters.items():
            for (layer_index, projection), (a, b, scale) in targets.items():
                if layer_index < 0 or layer_index >= len(self.weights.layers):
                    raise ValueError(f"LoRA adapter {adapter_id!r} references invalid layer {layer_index}")
                if not hasattr(self.weights.layers[layer_index], projection):
                    raise ValueError(f"LoRA adapter {adapter_id!r} references unknown projection {projection!r}")
                base = np.asarray(getattr(self.weights.layers[layer_index], projection))
                A = np.asarray(a)
                B = np.asarray(b)
                if A.ndim != 2 or B.ndim != 2 or A.shape[1] != base.shape[1] or B.shape[0] != base.shape[0] or A.shape[0] != B.shape[1]:
                    raise ValueError(
                        f"LoRA adapter {adapter_id!r} target layer {layer_index} {projection} "
                        f"does not match base shape {base.shape}: A={A.shape}, B={B.shape}"
                    )
                if not np.isfinite(float(scale)) or float(scale) < 0:
                    raise ValueError(f"LoRA adapter {adapter_id!r} has invalid scale")

    def _ensure_rope_capacity(self, required: int) -> None:
        """Grow the RoPE tables when a sequence runs past the precomputed range."""
        if required <= self._cos.shape[0]:
            return
        self._cos, self._sin = self._build_rope_tables(max_positions=max(required, 2 * self._cos.shape[0]))

    # ── Primitives ───────────────────────────────────────────────────────────

    def _linear(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        target: tuple[int, str] | None = None,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply a base linear projection and the selected real LoRA delta."""
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        matrix = np.asarray(weight, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"linear weight must be a rank-2 matrix, got shape {matrix.shape}"
            )
        input_features = int(x32.shape[-1])
        # AEG stores the usual PyTorch layout (out_features, in_features),
        # while some source checkpoints use Conv1D layout
        # (in_features, out_features). Infer the orientation from the actual
        # contraction dimension instead of identifying a model family.
        if matrix.shape[1] == input_features:
            # ``matrix.T`` is a zero-copy view.  ``sgemm`` accepts strided
            # NumPy operands and the native forced-GEMM path materializes its
            # own contiguous operand only when it actually needs one.  Do not
            # copy every large projection during single-token decode.
            kernel = matrix.T
        elif matrix.shape[0] == input_features:
            kernel = matrix
        else:
            raise ValueError(
                "linear weight/input mismatch: input has "
                f"{input_features} features but weight shape is {matrix.shape}"
            )
        output = self.kernels.sgemm(x32, kernel)
        if bias is not None:
            bias_array = np.asarray(bias, dtype=np.float32).reshape(-1)
            if bias_array.size != output.shape[-1]:
                raise ValueError(
                    f"linear bias has {bias_array.size} elements but output has "
                    f"{output.shape[-1]} features"
                )
            output = output + bias_array
        if self.active_lora_adapter is None or target is None:
            return output
        adapter = self.lora_adapters.get(self.active_lora_adapter)
        if adapter is None or target not in adapter:
            return output
        A, B, scale = adapter[target]
        delta = self.kernels.sgemm(
            self.kernels.sgemm(x32, np.asarray(A, dtype=np.float32).T),
            np.asarray(B, dtype=np.float32).T,
        )
        return np.ascontiguousarray(output + delta * np.float32(scale), dtype=np.float32)

    def _norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """Apply the source model's declared normalization."""
        if str(self.weights.norm_type).lower() == "layernorm":
            arr = np.asarray(x, dtype=np.float32)
            mean = np.mean(arr, axis=-1, keepdims=True)
            variance = np.mean((arr - mean) ** 2, axis=-1, keepdims=True)
            result = (arr - mean) / np.sqrt(variance + self.weights.norm_eps)
            result = result * np.asarray(weight, dtype=np.float32)
            if bias is not None:
                result = result + np.asarray(bias, dtype=np.float32)
            return np.ascontiguousarray(result, dtype=np.float32)
        return self.kernels.rmsnorm(x, weight, self.weights.norm_eps)

    def _ffn_activation(self, gate: np.ndarray, up: np.ndarray | None) -> np.ndarray:
        """Evaluate the declared FFN variant without substituting weights."""
        ffn_type = str(self.weights.ffn_type or "SwiGLU").lower()
        if up is None:
            # Classic GPT-style blocks have one intermediate projection.
            if ffn_type in {"gelu", "relu", "relu2"}:
                if ffn_type == "relu":
                    return np.maximum(gate, 0.0).astype(np.float32)
                if ffn_type == "relu2":
                    return np.square(np.maximum(gate, 0.0)).astype(np.float32)
                return (0.5 * gate * (1.0 + np.tanh(
                    np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3)
                ))).astype(np.float32)
            raise ValueError(f"FFN type {self.weights.ffn_type!r} requires an up projection")
        if ffn_type in {"geglu", "gelu"}:
            activated = 0.5 * gate * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3)
            ))
            return np.asarray(activated * up, dtype=np.float32)
        return self.kernels.swiglu(gate, up)

    def _moe_ffn(self, hidden: np.ndarray, layer: LayerWeights) -> np.ndarray:
        """Execute top-k routed SwiGLU experts for a token batch.

        The router computes logits ``x W_router^T`` and selects the declared
        top-k experts per token.  Softmax is applied over the selected logits,
        then each selected expert contributes its weighted FFN output.  This
        is the reference dispatch equation used by sparse MoE transformers;
        it intentionally favors transparent correctness over fused kernels.
        """
        if layer.router is None or not layer.experts:
            raise ValueError("MoE layer is missing its router or expert bank")
        source = np.asarray(hidden, dtype=np.float32)
        router_logits = self._linear(source, layer.router)
        top_k = min(int(layer.num_activated_experts), len(layer.experts))
        if top_k <= 0:
            raise ValueError("MoE top-k must be positive")
        # argpartition avoids a full expert sort while retaining exact top-k
        # membership. Sorting the selected columns gives deterministic ties.
        selected = np.argpartition(router_logits, -top_k, axis=-1)[:, -top_k:]
        selected_logits = np.take_along_axis(router_logits, selected, axis=-1)
        order = np.argsort(-selected_logits, axis=-1, kind="stable")
        selected = np.take_along_axis(selected, order, axis=-1)
        selected_logits = np.take_along_axis(selected_logits, order, axis=-1)
        selected_logits -= np.max(selected_logits, axis=-1, keepdims=True)
        routing = np.exp(selected_logits)
        routing /= np.maximum(routing.sum(axis=-1, keepdims=True), 1e-12)

        output = np.zeros_like(source, dtype=np.float32)
        for expert_index, expert in enumerate(layer.experts):
            rows, slots = np.where(selected == expert_index)
            if rows.size == 0:
                continue
            expert_input = source[rows]
            gate = self._linear(expert_input, expert.gate_proj, bias=expert.gate_proj_bias)
            up = self._linear(expert_input, expert.up_proj, bias=expert.up_proj_bias)
            activated = self._ffn_activation(gate, up)
            expert_output = self._linear(
                activated, expert.down_proj, bias=expert.down_proj_bias
            )
            output[rows] += expert_output * routing[rows, slots, None]
        return np.ascontiguousarray(output, dtype=np.float32)

    def _apply_ttt_slot(self, hidden: np.ndarray, slot: dict[str, Any] | None) -> np.ndarray:
        """Apply one request-local R5 fast-weight slot to normalized states."""
        if slot is None:
            return hidden
        width = self.weights.hidden_size
        try:
            mu = np.asarray(slot["mu"], dtype=np.float32).reshape(width)
            sigma = np.asarray(slot["sigma"], dtype=np.float32).reshape(width)
            adapted = (hidden - mu) * sigma
            a = np.asarray(slot["A"], dtype=np.float32).reshape(width, -1)
            b = np.asarray(slot["B"], dtype=np.float32).reshape(a.shape[1], width)
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid R5 TTT slot dimensions") from exc
        if np.any(a) or np.any(b):
            adapted = adapted + (adapted @ a) @ b
        return np.ascontiguousarray(adapted, dtype=np.float32)

    def _attention(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        causal_offset: int,
        key_positions: np.ndarray | None = None,
        query_positions: np.ndarray | None = None,
        window_size: int | None = None,
    ) -> np.ndarray:
        """Scaled dot-product attention with causal masking and GQA broadcast.

        Args:
            query: ``(seq, num_heads, head_dim)``.
            keys: ``(total, num_kv_heads, head_dim)``.
            values: ``(total, num_kv_heads, head_dim)``.
            causal_offset: Number of cached positions preceding ``query``.

        Returns:
            Attention output of shape ``(seq, num_heads, head_dim)``.
        """
        seq_len = query.shape[0]
        total = keys.shape[0]
        if key_positions is None:
            key_positions = np.arange(total, dtype=np.int64)
        if query_positions is None:
            query_positions = np.arange(seq_len, dtype=np.int64) + causal_offset
        key_positions = np.asarray(key_positions, dtype=np.int64).reshape(-1)
        query_positions = np.asarray(query_positions, dtype=np.int64).reshape(-1)
        if key_positions.size != total or query_positions.size != seq_len:
            raise ValueError("attention position metadata does not match KV/query shapes")
        scale = 1.0 / np.sqrt(self.head_dim)
        repeats = self.num_heads // self.num_kv_heads

        # Decode is memory-bandwidth bound.  For GQA, repeating K/V from
        # ``num_kv_heads`` to ``num_heads`` allocates two full temporary
        # tensors on every token.  Compute each query group against its shared
        # KV head directly instead.  This is algebraically identical to the
        # broadcast form but avoids O(T * H * D) copies in the hot loop.
        if seq_len == 1 and repeats > 1:
            q_grouped = np.ascontiguousarray(
                query.transpose(1, 0, 2).reshape(self.num_kv_heads, repeats, self.head_dim),
                dtype=np.float32,
            )
            keys_grouped = np.ascontiguousarray(keys, dtype=np.float32)
            scores_grouped = np.einsum(
                "hrd,thd->hrt", q_grouped, keys_grouped, optimize=True
            )
            scores = scores_grouped.reshape(self.num_heads, 1, total) * scale
        else:
            # (heads, seq, dim) @ (heads, dim, total) -> (heads, seq, total)
            keys_full = np.repeat(keys, repeats, axis=1)
            q = np.ascontiguousarray(query.transpose(1, 0, 2), dtype=np.float32)
            k = np.ascontiguousarray(keys_full.transpose(1, 2, 0), dtype=np.float32)
            scores = np.matmul(q, k) * scale

        if str(self.weights.position_type or "RoPE").lower() in {"alibi", "alibi_bias"}:
            # ALiBi's causal bias is slope * (key_position - query_position),
            # which is non-positive for permitted causal keys.
            distance = key_positions[None, :] - query_positions[:, None]
            scores = scores + self._alibi_slopes[:, None, None] * distance[None, :, :]

        # Causal mask uses original token positions, not compressed row indices.
        allowed = key_positions[None, :] <= query_positions[:, None]
        if window_size is not None and window_size > 0:
            allowed &= key_positions[None, :] >= (query_positions[:, None] - int(window_size) + 1)
        sparse_allowed = self._sparse_allowed_mask(
            seq_len=seq_len,
            total=total,
            causal_offset=causal_offset,
            heads=scores.shape[0],
            key_positions=key_positions,
            query_positions=query_positions,
        )
        if sparse_allowed is not None:
            allowed = allowed[None, :, :] & sparse_allowed
        else:
            allowed = allowed[None, :, :]
        scores = np.where(allowed, scores, np.float32(-np.inf))

        weights = self.kernels.softmax(scores.reshape(-1, total)).reshape(scores.shape)
        if seq_len == 1 and repeats > 1:
            weights_grouped = weights.reshape(self.num_kv_heads, repeats, 1, total)[:, :, 0, :]
            context_grouped = np.einsum(
                "hrt,thd->hrd", weights_grouped, np.ascontiguousarray(values, dtype=np.float32),
                optimize=True,
            )
            return context_grouped.reshape(1, self.num_heads, self.head_dim)
        values_full = np.repeat(values, repeats, axis=1)
        v = np.ascontiguousarray(values_full.transpose(1, 0, 2), dtype=np.float32)
        context = np.matmul(weights, v)
        return context.transpose(1, 0, 2)

    def _sparse_allowed_mask(
        self,
        *,
        seq_len: int,
        total: int,
        causal_offset: int,
        heads: int,
        key_positions: np.ndarray | None = None,
        query_positions: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Build the actual per-head mask described by the persisted Pass 8 plan."""
        plan = self.sparse_attention_plan
        if not isinstance(plan, dict) or not bool(plan.get("enabled")):
            return None
        # Pass 8 is a long-context optimization.  Its persisted patterns must
        # not alter ordinary short prompts, otherwise a compiled model silently
        # diverges from the dense reference model.  The compiler records the
        # activation threshold alongside the plan for this runtime decision.
        activation_threshold = int(plan.get("activation_context_length", 0) or 0)
        if activation_threshold > 0 and total < activation_threshold:
            return None
        patterns = plan.get("patterns")
        if not isinstance(patterns, list) or len(patterns) != heads:
            raise ValueError("sparse attention plan must contain one pattern per attention head")
        mask = np.zeros((heads, seq_len, total), dtype=bool)
        if key_positions is None:
            key_positions = np.arange(total, dtype=np.int64)
        if query_positions is None:
            query_positions = np.arange(seq_len, dtype=np.int64) + causal_offset
        for head, descriptor in enumerate(patterns):
            if not isinstance(descriptor, dict):
                raise ValueError("sparse attention pattern must be an object")
            pattern = descriptor.get("pattern", descriptor.get("pattern_type"))
            if pattern == "dense":
                mask[head, :, :] = True
            elif pattern == "a_shape":
                window = int(descriptor.get("local_window", descriptor.get("local_window_size", 128)))
                sinks = int(descriptor.get("sink_tokens", descriptor.get("num_sink_tokens", 16)))
                if window < 0 or sinks < 0:
                    raise ValueError("sparse attention window and sink count must be non-negative")
                for row, position in enumerate(query_positions):
                    low = max(0, int(position) - window)
                    mask[head, row, :] = (key_positions >= low) & (key_positions <= int(position))
                    mask[head, row, key_positions < min(sinks, total)] = True
            elif pattern == "vertical_slash":
                width = int(descriptor.get("slash_width", descriptor.get("local_window", 64)))
                stride = int(descriptor.get("stride", 0))
                if width <= 0 or stride < 0:
                    raise ValueError("invalid vertical-slash sparse attention parameters")
                half = max(0, width // 2)
                for row, position in enumerate(query_positions):
                    center = int(position) - stride
                    low = max(0, center - half)
                    high = min(int(position) + 1, center + half + 1)
                    mask[head, row, :] = (key_positions >= low) & (key_positions < high)
                    exact = np.where(key_positions == int(position))[0]
                    if exact.size:
                        mask[head, row, int(exact[-1])] = True
            elif pattern == "block_sparse":
                block = int(descriptor.get("block_size", descriptor.get("local_window", 64)))
                stride = int(descriptor.get("block_stride") or descriptor.get("stride") or 128)
                if block <= 0 or stride <= 0:
                    raise ValueError("invalid block-sparse attention parameters")
                for row, position in enumerate(query_positions):
                    block_start = (int(position) // stride) * stride
                    for start in range(0, block_start + 1, stride):
                        low = max(0, start)
                        high = min(int(position) + 1, start + block)
                        mask[head, row, :] |= (key_positions >= low) & (key_positions < high)
                    exact = np.where(key_positions == int(position))[0]
                    if exact.size:
                        mask[head, row, int(exact[-1])] = True
            else:
                raise ValueError(f"unknown sparse attention pattern {pattern!r}")
        return mask

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids: np.ndarray,
        cache: KVCache | None = None,
        ttt_slots: list[dict[str, Any]] | None = None,
        adapter_id: str | None = None,
    ) -> tuple[np.ndarray, KVCache]:
        """Run the transformer over ``token_ids``.

        Args:
            token_ids: 1-D array of token ids for this step. During prefill this
                is the whole prompt; during decode it is a single token.
            cache: Existing KV cache to extend, or None to start fresh.

        Returns:
            Tuple of ``(logits, cache)`` where logits has shape
            ``(seq, vocab_size)``.
        """
        if adapter_id is not None and adapter_id != self.active_lora_adapter:
            raise ValueError("adapter_id must match the request-local engine selection")
        ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            msg = "forward() requires at least one token"
            raise ValueError(msg)
        if int(ids.max()) >= self.weights.vocab_size or int(ids.min()) < 0:
            msg = (
                f"token id out of range: got [{ids.min()}, {ids.max()}], "
                f"vocab_size is {self.weights.vocab_size}"
            )
            raise ValueError(msg)

        cache = cache or KVCache(num_layers=self.weights.num_layers)
        past = cache.length
        seq_len = int(ids.size)
        uses_rope = str(self.weights.position_type or "RoPE").lower() in {
            "rope", "rotary", "rotary_embedding"
        }
        if uses_rope:
            self._ensure_rope_capacity(past + seq_len)

        hidden = self.weights.embedding[ids].astype(np.float32)
        if self.weights.embedding_norm is not None:
            hidden = self._norm(
                hidden,
                self.weights.embedding_norm,
                self.weights.embedding_norm_bias,
            )
        if self.weights.position_embedding is not None:
            end = past + seq_len
            if end > self.weights.position_embedding.shape[0]:
                raise ValueError(
                    f"sequence end {end} exceeds learned position embedding capacity "
                    f"{self.weights.position_embedding.shape[0]}"
                )
            hidden = hidden + self.weights.position_embedding[past:end]

        for index, layer in enumerate(self.weights.layers):
            attention_layers = getattr(self.weights, "attention_layers", None)
            attention_kind = (
                str(attention_layers[index]).lower()
                if isinstance(attention_layers, list) and index < len(attention_layers)
                else "global"
            )
            local_attention = attention_kind in {"local", "sliding_window", "window"}
            attention_window = (
                int(getattr(self.weights, "attention_window", 0) or 0)
                if local_attention else None
            )
            # ── Attention block ──
            # For decode (seq_len == 1) with no LoRA: use the fused
            # rmsnorm_linear kernel to eliminate the intermediate normed buffer.
            # With LoRA active we must keep the separate path so the delta can
            # be applied after projection.
            if (
                seq_len == 1
                and self.active_lora_adapter is None
                and ttt_slots is None
                and self.kernels.is_native
                # ``rmsnorm_linear`` implements x / RMS(x) * weight.  It is
                # not algebraically equivalent to LayerNorm, which also
                # subtracts the feature mean and may add a bias.  Using the
                # fused RMS kernel for GPT-2/GPT-Neo/Falcon/BLOOM/OPT-style
                # models silently changes every decode step after prefill and
                # can turn an otherwise valid model into repetition garbage.
                and str(self.weights.norm_type).lower() != "layernorm"
                and layer.attention_norm_bias is None
            ):
                q = self.kernels.rmsnorm_linear(
                    hidden, layer.attention_norm, layer.q_proj, self.weights.norm_eps,
                ).reshape(seq_len, self.num_heads, self.head_dim)
                normed = self._norm(hidden, layer.attention_norm, layer.attention_norm_bias)
            else:
                normed = self._norm(hidden, layer.attention_norm, layer.attention_norm_bias)
                if ttt_slots is not None:
                    normed = self._apply_ttt_slot(
                        normed, ttt_slots[index] if index < len(ttt_slots) else None
                    )
                q = self._linear(normed, layer.q_proj, (index, "q_proj"), layer.q_proj_bias).reshape(seq_len, self.num_heads, self.head_dim)
            if layer.q_norm is not None:
                q = self.kernels.rmsnorm(q, layer.q_norm, self.weights.norm_eps)
            if uses_rope:
                q = self.kernels.rope(q, self._cos, self._sin, position_offset=past)

            shared_source = self._cross_layer_kv_source(index)
            if shared_source is not None:
                source_keys, source_values, source_positions = cache.layer_view(shared_source)
                if source_keys is None or source_values is None or source_positions is None:
                    raise ValueError(
                        f"cross-layer KV source {shared_source} has no computed cache"
                    )
                # Keep the exact ndarray objects: this is physical pointer
                # sharing, not a copied approximation of the source cache.
                keys, values, key_positions = source_keys, source_values, source_positions
            else:
                k = self._linear(normed, layer.k_proj, (index, "k_proj"), layer.k_proj_bias).reshape(
                    seq_len, self.num_kv_heads, self.head_dim
                )
                if layer.k_norm is not None:
                    k = self.kernels.rmsnorm(k, layer.k_norm, self.weights.norm_eps)
                v = self._linear(normed, layer.v_proj, (index, "v_proj"), layer.v_proj_bias).reshape(
                    seq_len, self.num_kv_heads, self.head_dim
                )
                if uses_rope:
                    k = self.kernels.rope(k, self._cos, self._sin, position_offset=past)
                keys, values, key_positions = cache.append(
                    index,
                    k,
                    v,
                    positions=np.arange(past, past + seq_len, dtype=np.int64),
                    update_length=False,
                )
            # ── Attention computation ──
            # For decode (seq_len == 1) with no sparse attention plan: use the
            # native FlashAttention-2 kernel — O(seq*d) memory, no QKᵀ buffer.
            if (
                seq_len == 1
                and self.sparse_attention_plan is None
                and self.kernels.is_native
                and not local_attention
                and str(self.weights.position_type or "RoPE").lower() not in {"alibi", "alibi_bias"}
            ):
                context = self.kernels.flash_attn(
                    q.reshape(self.num_heads, self.head_dim),
                    keys,
                    values,
                    num_kv_heads=self.num_kv_heads,
                ).reshape(1, self.num_heads, self.head_dim)
            else:
                context = self._attention(
                    q,
                    keys,
                    values,
                    causal_offset=past,
                    key_positions=key_positions,
                    query_positions=np.arange(past, past + seq_len, dtype=np.int64),
                    window_size=attention_window,
                )
            compressed_keys, compressed_values, compressed_positions = self._compress_semantic_kv(
                cache, index, keys, values, key_positions
            )
            if self._semantic_kv_policy(index) is not None and shared_source is None:
                cache.replace_layer(
                    index,
                    compressed_keys,
                    compressed_values,
                    compressed_positions,
                )
            elif shared_source is not None:
                cache.keys[index] = cache.keys[shared_source]
                cache.values[index] = cache.values[shared_source]
                cache.positions[index] = cache.positions[shared_source]
                cache.stored_lengths[index] = cache.stored_lengths[shared_source]
            attention_out = self._linear(
                context.reshape(seq_len, self.num_heads * self.head_dim), layer.o_proj, (index, "o_proj"), layer.o_proj_bias
            )
            if self.weights.parallel_residual:
                # GPT-J computes both branches from one normalized state.
                ffn_normed = normed
            else:
                hidden = hidden + attention_out
                ffn_normed = self._norm(hidden, layer.ffn_norm, layer.ffn_norm_bias)

            # ── FFN block ──
            if layer.experts:
                ffn_out = self._moe_ffn(ffn_normed, layer)
            else:
                if layer.gate_proj is None or layer.down_proj is None:
                    raise ValueError(f"layer {index} has incomplete dense FFN weights")
                gate = self._linear(ffn_normed, layer.gate_proj, (index, "gate_proj"), layer.gate_proj_bias)
                up = (
                    self._linear(ffn_normed, layer.up_proj, (index, "up_proj"), layer.up_proj_bias)
                    if layer.up_proj is not None
                    else None
                )
                ffn_out = self._linear(
                    self._ffn_activation(gate, up), layer.down_proj, (index, "down_proj"), layer.down_proj_bias
                )
            hidden = hidden + attention_out + ffn_out if self.weights.parallel_residual else hidden + ffn_out

        hidden = self._norm(hidden, self.weights.final_norm, self.weights.final_norm_bias)
        cache.last_hidden = np.asarray(hidden[-1], dtype=np.float32).copy()
        logits = self._linear(hidden, self.weights.lm_head)
        cache.advance(seq_len)
        cache.last_logits = np.asarray(logits[-1], dtype=np.float32).copy()
        return logits, cache

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        grammar_session: Any | None = None,
        ttt_slots: list[dict[str, Any]] | None = None,
        adapter_id: str | None = None,
        peagle_engine: Any | None = None,
    ) -> list[int]:
        """Autoregressively generate token ids.

        Args:
            prompt_ids: Prompt token ids.
            max_tokens: Maximum new tokens to emit.
            temperature: 0 selects greedily; higher values sample more randomly.
            top_k: Restrict sampling to the k highest-probability tokens (0 = off).
            eos_token_id: Stop early when this token is produced.
            seed: Seed for reproducible sampling.

        Returns:
            The generated token ids, excluding the prompt.
        """
        generated, _cache = self.generate_with_cache(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            seed=seed,
            grammar_session=grammar_session,
            ttt_slots=ttt_slots,
            adapter_id=adapter_id,
            peagle_engine=peagle_engine,
        )
        return generated

    def generate_with_cache(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        grammar_session: Any | None = None,
        cache: KVCache | None = None,
        ttt_slots: list[dict[str, Any]] | None = None,
        adapter_id: str | None = None,
        peagle_engine: Any | None = None,
    ) -> tuple[list[int], KVCache]:
        """Generate tokens while accepting and returning an incremental KV cache.

        ``prompt_ids`` is interpreted as the next uncached suffix when
        ``cache`` is supplied.  The caller is responsible for proving that the
        suffix follows the sequence represented by that cache.  This explicit
        contract prevents accidental reuse across unrelated requests.
        """
        generated: list[int] = []
        final_cache: list[KVCache | None] = [cache]

        def remember(updated: KVCache) -> None:
            final_cache[0] = updated

        generated.extend(
            self.generate_iter(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                seed=seed,
                grammar_session=grammar_session,
                cache=cache,
                cache_callback=remember,
                ttt_slots=ttt_slots,
                adapter_id=adapter_id,
                peagle_engine=peagle_engine,
            )
        )
        if final_cache[0] is None:
            raise RuntimeError("generation completed without producing a KV cache")
        return generated, final_cache[0]

    def generate_iter(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        seed: int | None = None,
        grammar_session: Any | None = None,
        cache: KVCache | None = None,
        cache_callback: Any | None = None,
        ttt_slots: list[dict[str, Any]] | None = None,
        adapter_id: str | None = None,
        peagle_engine: Any | None = None,
    ) -> Any:
        """Yield generated token IDs as they are produced.

        This is the streaming counterpart to :meth:`generate_with_cache`.  It
        executes one decoder step before yielding each token, so callers can
        transport actual incremental output instead of slicing a completed
        response.  ``cache_callback`` receives the final cache after normal
        completion and is intentionally not called when generation raises.
        """
        ids = np.ascontiguousarray(prompt_ids, dtype=np.int64).reshape(-1)
        rng = np.random.default_rng(seed)
        if cache is None:
            if ids.size == 0:
                raise ValueError("generate_iter() requires prompt tokens for a new cache")
            cache = KVCache(num_layers=self.weights.num_layers)
            cache.reserve(int(ids.size) + max(0, int(max_tokens)))
            logits, cache = self.forward(ids, ttt_slots=ttt_slots, adapter_id=adapter_id)
        elif ids.size:
            logits, cache = self.forward(ids, cache, ttt_slots=ttt_slots, adapter_id=adapter_id)
        elif cache.last_logits is not None:
            logits = cache.last_logits.reshape(1, -1)
        else:
            raise ValueError("provided KV cache has no logits for an empty prompt suffix")

        context_tokens = ids.tolist()
        generated_count = 0
        while generated_count < max(0, max_tokens):
            # Compiled MTP heads are connected to the real target engine only
            # for exact greedy decoding.  Each proposed token is verified by
            # the target argmax before it is emitted and the KV cache is
            # advanced with the verified token.  Sampling and grammar modes
            # retain the ordinary one-token path because their acceptance
            # semantics are different.
            if (
                peagle_engine is not None
                and temperature <= 0.0
                and grammar_session is None
                and cache.last_hidden is not None
                and not getattr(peagle_engine, "should_disable_speculation", lambda: False)()
            ):
                remaining = max(0, max_tokens - generated_count)
                try:
                    proposals = peagle_engine.draft_tokens(
                        cache.last_hidden,
                        context_tokens,
                        limit=min(remaining, getattr(peagle_engine, "draft_K", remaining)),
                    )
                except Exception:
                    # An unusable or stale draft artifact must never fabricate
                    # output or fail an otherwise valid target model.  Fall
                    # back to ordinary target decoding for this cycle.
                    proposals = []
                if proposals:
                    accepted = 0
                    for proposal in proposals:
                        target_logits = np.asarray(logits[-1], dtype=np.float32)
                        target_token = self.kernels.argmax(target_logits)
                        next_token = int(proposal) if int(proposal) == target_token else target_token
                        if int(proposal) == target_token:
                            accepted += 1
                        yield next_token
                        generated_count += 1
                        context_tokens.append(next_token)
                        self._speculative_stats["draft_tokens"] += 1
                        self._speculative_stats["accepted_tokens"] += int(int(proposal) == target_token)
                        if eos_token_id is not None and next_token == eos_token_id:
                            generated_count = max_tokens
                            break
                        logits, cache = self.forward(
                            np.array([next_token], dtype=np.int64),
                            cache,
                            ttt_slots=ttt_slots,
                            adapter_id=adapter_id,
                        )
                        if generated_count >= max_tokens:
                            break
                    self._speculative_stats["cycles"] += 1
                    continue

            next_logits = np.asarray(logits[-1], dtype=np.float32).copy()
            if grammar_session is not None:
                mask = grammar_session.get_token_mask()
                if len(mask) * 8 < next_logits.size:
                    raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
                allowed = np.fromiter(
                    ((mask[i // 8] & (1 << (i % 8))) != 0 for i in range(next_logits.size)),
                    dtype=bool,
                    count=next_logits.size,
                )
                if not np.any(allowed):
                    raise ValueError("Grammar FSM has no valid next token")
                next_logits[~allowed] = -np.inf
            next_token = self._sample(next_logits, temperature, top_k, top_p, rng)
            if grammar_session is not None and grammar_session.advance(next_token) < 0:
                raise ValueError("The CPU engine produced a token rejected by the grammar FSM")
            yield next_token
            generated_count += 1
            context_tokens.append(next_token)
            if grammar_session is not None and getattr(grammar_session, "is_accepting", lambda: False)():
                break
            if eos_token_id is not None and next_token == eos_token_id:
                break
            logits, cache = self.forward(
                np.array([next_token], dtype=np.int64), cache, ttt_slots=ttt_slots, adapter_id=adapter_id
            )

        if cache_callback is not None:
            cache_callback(cache)

    def speculative_stats(self) -> dict[str, int | float]:
        """Return measured exact-greedy MTP verification counters."""
        draft = int(self._speculative_stats["draft_tokens"])
        accepted = int(self._speculative_stats["accepted_tokens"])
        return {
            "draft_tokens": draft,
            "accepted_tokens": accepted,
            "cycles": int(self._speculative_stats["cycles"]),
            "acceptance_rate": accepted / max(1, draft),
        }

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        rng: np.random.Generator,
    ) -> int:
        """Choose the next token from a logit vector."""
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if temperature <= 0.0:
            return self.kernels.argmax(logits)
        scaled = logits.astype(np.float32) / np.float32(temperature)
        if top_k > 0:
            k = min(top_k, scaled.size)
            cutoff = np.partition(scaled, -k)[-k]
            scaled = np.where(scaled < cutoff, np.float32(-np.inf), scaled)
        if top_p < 1.0:
            order = np.argsort(scaled)[::-1]
            ordered = scaled[order]
            probabilities = self.kernels.softmax(ordered.reshape(1, -1)).reshape(-1)
            cumulative = np.cumsum(probabilities)
            keep = cumulative <= np.float32(top_p)
            # Always retain the first token crossing the threshold.
            crossing = int(np.searchsorted(cumulative, top_p, side="right"))
            if crossing < keep.size:
                keep[crossing] = True
            scaled = np.where(np.isin(np.arange(scaled.size), order[keep]), scaled, np.float32(-np.inf))
        probabilities = self.kernels.softmax(scaled.reshape(1, -1)).reshape(-1)
        # Renormalise: softmax over a masked vector can drift by a few ulps.
        probabilities = probabilities / probabilities.sum()
        return int(rng.choice(probabilities.size, p=probabilities))

    def __repr__(self) -> str:
        backend = "native" if self.kernels.is_native else "numpy"
        return (
            f"CPUExecutionEngine(layers={self.weights.num_layers}, "
            f"hidden={self.weights.hidden_size}, heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, kernels={backend})"
        )
