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

from aether.utils.logging import get_logger

logger = get_logger(__name__)

from aether.runtime.batch import (
    DEFAULT_PAD_TOKEN_ID,
    BatchLayout,
    PackedBatch,
    pack_left_padded,
)
from aether.runtime.positional import alibi_slopes
from aether.runtime.stopping import stop_token_set
from aether.runtime.rope_scaling import (
    parse_rope_scaling,
    scaled_inverse_frequencies,
)


def _resolve_device(torch: Any, spec: Any) -> Any:
    """Return a fully qualified device, with an explicit index where one applies.

    ``torch.device("cuda")`` carries no index, but every tensor placed on it
    reports ``cuda:0``, and the two compare **unequal**.  Any cache keyed on
    "is this tensor already on my device?" therefore misses forever, silently
    rebuilding or re-copying on every step.  Resolving the index once here makes
    those comparisons meaningful.
    """
    device = torch.device(spec)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


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
    # The rotary frequency transform is part of the model, not a runtime option:
    # executing a scaled checkpoint with unscaled frequencies is a silent,
    # model-wide numerical error.  It must survive any rebuilt weight container.
    "rope_scaling",
    "original_context_length",
    "norm_placement",
    "qk_norm_scope",
    "no_rope_layers",
    "gelu_approximate",
    "moe_renormalize_topk",
    # Not a numeric constant, but it bounds every position-indexed table, so it
    # must survive a rebuilt weight container exactly as the rest does.
    "context_length",
)


def execution_numerics(source: Any) -> dict[str, Any]:
    """Extract the execution-numerics contract from a weight container."""
    return {name: getattr(source, name, None) for name in EXECUTION_NUMERICS_FIELDS}


#: Host-resident weight attributes large enough to matter, in the order a reader
#: would expect them.  Normalization weights, biases and routers are deliberately
#: absent: they are metadata-scale, several code paths read them after load, and
#: freeing them would buy kilobytes while adding failure modes.
_BULK_MODEL_ARRAYS: tuple[str, ...] = (
    "embedding", "lm_head", "position_embedding",
)
_BULK_LAYER_ARRAYS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
_BULK_EXPERT_ARRAYS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def host_weights_reclaimable() -> bool:
    """Whether host weight copies may be freed once the device owns its own.

    The canonical reading of ``AETHER_KEEP_HOST_WEIGHTS``, defined here rather than
    in the backend so the streaming release inside a load and the sweep after it
    cannot disagree about whether the operator asked to keep the host set.

    Set it when driving a host-weight-dependent path (task-vector merging, compiled
    LoRA selection, R5 fast weights) against an accelerator-resident engine.
    """
    keep = os.environ.get("AETHER_KEEP_HOST_WEIGHTS", "").strip().lower()
    return keep not in {"1", "true", "yes", "on"}


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


@dataclass
class BatchedKVCache:
    """Device-resident KV state for a left-padded batch of independent sequences.

    Each layer's tensors are ``(batch, capacity, kv_heads, head_dim)``: batch-major
    and contiguous.  Two properties follow from that choice and are the reason for
    it.  Appending a decode step is one slice assignment across every row, because
    left padding puts every row's frontier at the same index.  And
    ``transpose(1, 2)`` yields the ``(batch, heads, seq, dim)`` view that fused
    attention wants, as a view rather than a copy.

    Isolation is structural rather than enforced.  The batch axis is the outermost
    axis of every cache tensor and every activation, and a row is only ever written
    through its own index, so row ``i`` has no code path that reaches row ``j``'s
    state.  See ``docs/adr-batched-inference.md`` for the two operations where that
    is not automatic (expert routing and sampling) and how each is handled.
    """

    keys: list[Any | None]
    values: list[Any | None]

    pad_counts: Any
    """``(batch, 1)`` int64 leading pad count per row — the row's position offset.

    Row ``b``'s token at padded index ``i`` sits at position ``max(0, i - pad)``.
    Carrying the offset here is what lets a batched result equal the same sequence
    decoded alone; using the padded index as the position would rotate every short
    row as though it began mid-sequence.
    """

    batch_size: int = 1

    live: Any | None = None
    """``(batch, capacity)`` bool, or ``None`` when the batch carries no padding.

    ``None`` means "every slot is real", which is both true and the trigger for
    keeping fused attention: a uniform batch needs no mask, so prefill stays
    ``is_causal=True`` and decode passes no mask at all.
    """

    length: int = 0
    """Uniform write frontier.  One value is correct only because rows are
    right-aligned; with right padding this would have to be per row."""

    layout: BatchLayout | None = None
    last_logits: Any | None = None
    """``(batch, vocab)`` logits at each row's final real position."""

    finished: Any | None = None
    """``(batch,)`` bool: has this row emitted its stop token."""

    def clone(self) -> "BatchedKVCache":
        return BatchedKVCache(
            keys=[None if value is None else value.clone() for value in self.keys],
            values=[None if value is None else value.clone() for value in self.values],
            pad_counts=self.pad_counts.clone(),
            batch_size=self.batch_size,
            live=None if self.live is None else self.live.clone(),
            length=self.length,
            layout=self.layout,
            last_logits=None if self.last_logits is None else self.last_logits.clone(),
            finished=None if self.finished is None else self.finished.clone(),
        )

    def live_view(self, total: int) -> Any | None:
        """The validity mask over the first ``total`` slots, or ``None`` if unpadded."""
        return None if self.live is None else self.live[:, :total]


class TorchAEGEngine:
    """Execute a standard dense decoder AEG on a PyTorch device."""

    #: Extra rotary positions materialized beyond the current requirement, so a
    #: growing sequence rebuilds the tables rarely without over-reserving.
    _ROPE_HEADROOM = 512

    #: Ceiling used when the artifact declares no context length.
    _DEFAULT_MAX_POSITIONS = 1 << 20

    # ── Inherited runtime state ───────────────────────────────────────────────
    #
    # Declared on the class, not only assigned in ``__init__``, because
    # ``TorchTensorParallelAEGEngine`` deliberately does *not* call
    # ``super().__init__`` — it would upload the unsharded weights to one device
    # before splitting them.  Every attribute the parent assigns only in its
    # constructor is therefore absent on that subclass, and an inherited method
    # that reads one raises ``AttributeError`` at load time.  That is not a
    # hypothetical: ``release_host_weights`` reads ``host_weights_released`` and
    # crashed a tensor-parallel load, and each attribute added to the parent since
    # then silently widened the gap.
    #
    # Class-level defaults close the class of bug rather than one instance of it.
    # A subclass that skips the constructor inherits values that are correct for
    # "nothing has happened yet", and instance assignment still shadows them for
    # the normal path, so behaviour there is unchanged.

    #: Set once ``release_host_weights`` has dropped the bulk host arrays.
    host_weights_released = False

    #: Host bytes already freed during the upload itself.  Zero is the honest
    #: default for any path that did not stream (a sharded executor uploads its own
    #: slices), and ``release_host_weights`` adds it to what it reclaims.
    host_bytes_streamed: int = 0

    #: Reusable pinned staging slots for sampled-token readback, or ``None`` when
    #: this device cannot stage asynchronously.  ``False`` means "not yet probed";
    #: the distinction matters because ``None`` is a valid, final answer.
    _readback_slots: "list[tuple[Any, Any]] | None | bool" = False

    #: Which staging slot the next readback will use.
    _readback_slot: int = 0

    #: Whether this model's rotary frequencies move with the length being
    #: evaluated.  ``None`` means "not yet derived from the scaling spec".
    _rope_length_sensitive: "bool | None" = None

    #: Rotary schemes whose frequencies are a function of the sequence length.
    #: Every other scheme in :mod:`aether.runtime.rope_scaling` derives them from
    #: the checkpoint's configuration alone, so their tables are final once built.
    _LENGTH_SENSITIVE_ROPE = frozenset({"dynamic", "longrope"})

    #: Persisted optimizer plans this accelerator path does not implement.
    _ignored_optimized_plans: "dict[str, Any] | frozenset[str] | tuple[str, ...]" = ()

    #: Decode projection strategy. ``None`` means "not resolved yet"; the
    #: accessors below fall back to the reference kernel, which is always correct.
    _projection: "Any | None" = None
    _reference_projection: "Any | None" = None
    _projection_choice: "Any | None" = None
    _projection_choices: "dict[Any, Any] | None" = None
    _projection_table: "dict[Any, Any] | None" = None
    _projection_rows: "tuple[int, str] | None" = None
    _strategies: "Any | None" = None

    def __init__(self, cpu_engine: Any, device: str) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by backend
            raise RuntimeError("PyTorch is required for portable AEG execution") from exc
        self.torch = torch
        # Resolve the device index up front: an index-less ``cuda`` never
        # compares equal to a tensor's ``cuda:0``, which would defeat every
        # "already on my device?" check in the decode path.
        self.device = _resolve_device(torch, device)
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
            self.compute_dtype = self._probe_compute_dtype(torch, self.device)
        self.source_engine = cpu_engine
        self.weights = cpu_engine.weights
        #: Set once ``release_host_weights`` has dropped the bulk host arrays, so
        #: a caller that needs them can detect it instead of meeting a ``None``.
        self.host_weights_released = False
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
        self.lm_head = self._tied_tensor(self.weights.lm_head, self.weights.embedding, self.embedding)
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
        #: Host bytes freed *during* the upload rather than after it, so
        #: ``release_host_weights`` can still report the load's true total.
        self.host_bytes_streamed = 0
        self.layers = self._convert_layers()
        # ── Decode kernel strategy ─────────────────────────────────────────
        # The projection formulation is measured per (device class, phase, row
        # count, dtype) and cached; until a measurement exists this is exactly
        # ``F.linear``, so an uncalibrated device is unchanged.  Constructed here
        # but never *run* here: calibration happens on the first pass of each
        # shape class, so a load that never decodes pays nothing.
        self._reference_projection = self._reference_projection_factory(torch)
        self._projection = self._reference_projection
        self._projection_choice: Any = None
        self._projection_choices: dict[Any, Any] = {}
        self._projection_table: dict[Any, Any] = {}
        self._projection_rows: "tuple[int, str] | None" = None
        self._strategies = self._build_strategy_calibrator()
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
        # ── Rotary frequency transform ─────────────────────────────────────
        # Parsed once here, never per token.  ``None`` means the checkpoint
        # declares no scaling and the standard frequencies apply, which is the
        # majority of models and costs nothing.
        declared_context = int(getattr(weights, "context_length", 0) or 0) or None
        self.rope_scaling_spec = parse_rope_scaling(
            getattr(weights, "rope_scaling", None),
            context_length=declared_context,
            original_context_length=getattr(weights, "original_context_length", None),
        )
        #: Multiplier YaRN and LongRoPE apply to the rotary tables.  Folded into
        #: cos/sin exactly as the reference does, rather than into the softmax
        #: scale: it multiplies the *rotated* query and key, so the two are not
        #: interchangeable.
        self.rope_attention_scaling = 1.0
        #: ``inv_freq`` the cached tables were built from.  Two schemes are
        #: length-dependent — ``dynamic`` rescales the base and ``longrope``
        #: switches factor tables — so a request that crosses their boundary needs
        #: new tables even when the cached ones are tall enough.  Comparing the
        #: frequencies themselves catches every scheme without special cases, and
        #: costs a few dozen float operations per growth check.
        self._rope_inv_freq: np.ndarray | None = None
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
        self.moe_renormalize_topk = bool(getattr(weights, "moe_renormalize_topk", True))
        # A learned position table is a hard ceiling; RoPE models are bounded by
        # the declared context length, falling back to a generous cap.
        position_table = getattr(weights, "position_embedding", None)
        if position_table is not None:
            self.max_positions = int(np.asarray(position_table).shape[0])
        else:
            declared = int(getattr(weights, "context_length", 0) or 0)
            self.max_positions = declared if declared > 0 else self._DEFAULT_MAX_POSITIONS
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

    def _tied_tensor(self, value: Any, tied_to: Any, resident: Any) -> Any:
        """Materialize ``value``, reusing ``resident`` when it is the same matrix.

        A tied language model has one matrix serving as both the input embedding and
        the output projection. Uploading it twice costs a second full copy of the
        largest tensor in a small model — 297 MiB of a 1.1 GiB Qwen3-0.6B residency,
        26% of the total — for storage that is read identically by both sites and
        written by neither.

        Identity is decided on the host array, not on the manifest: the loader aliases
        the two only after proving their stored bytes match, so ``is`` here means the
        same parameter, and ``lm_head`` genuinely being a distinct matrix still gets
        its own upload.
        """
        if value is tied_to and resident is not None:
            return resident
        return self._tensor(value)

    def _host_tensor(self, value: Any) -> Any:
        """Wrap a host array as a CPU tensor without copying it.

        ``torch.from_numpy`` shares the buffer, so this costs nothing and lets a
        device-side ``copy_`` perform the host-to-device transfer *and* the dtype
        narrowing in one step, with no intermediate of either dtype.
        """
        array = np.asarray(value)
        if array.dtype.kind != "f":
            array = array.astype(np.float32)
        return self.torch.from_numpy(np.ascontiguousarray(array))

    def _packed_tensor(self, parts: "list[Any]") -> Any:
        """Upload host matrices straight into the row slices of one device buffer.

        The fused ``qkv`` and ``gate_up`` projections are algebraically a row
        concatenation, and the obvious spelling is to materialize each part on the
        device and ``torch.cat`` them. That is the worst possible order for the
        caching allocator, on two counts the allocator cannot recover from:

        * It holds every part *and* the concatenation live at once, so the transient
          device high-water mark during load is twice the fused projection's size.
        * It asks for several small blocks and then one large block. Blocks in
          different segments can never be merged, so the parts' memory cannot satisfy
          the fused request: it comes from fresh segments, and the freed parts stay
          reserved-but-unusable for the process's lifetime. That gap is exactly what
          ``peak_reserved`` reports.

        Allocating the destination first and filling its slices inverts both: one
        allocation, no transient device copy, and a single large segment the allocator
        can subdivide and reuse afterwards.
        """
        rows = sum(int(part.shape[0]) for part in parts)
        trailing = tuple(int(dim) for dim in parts[0].shape[1:])
        out = self.torch.empty(
            (rows, *trailing), device=self.device, dtype=self.compute_dtype
        )
        offset = 0
        for part in parts:
            height = int(part.shape[0])
            out[offset:offset + height].copy_(self._host_tensor(part))
            offset += height
        return out

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
        # Which projections fuse is settled *before* anything is uploaded, so a
        # fused member never exists as a standalone device tensor.  The fused
        # layout is the only one the decode path reads, and materializing the
        # parts first would double this layer's transient device cost for nothing.
        hidden = int(self.weights.embedding.shape[1])
        qkv_group = self._fusible(layer, ("q_proj", "k_proj", "v_proj"), hidden)
        ffn_group = self._fusible(layer, ("gate_proj", "up_proj"), hidden)
        fused: set[str] = set()
        if qkv_group is not None:
            fused |= {"q_proj", "k_proj", "v_proj",
                      "q_proj_bias", "k_proj_bias", "v_proj_bias"}
        if ffn_group is not None:
            fused |= {"gate_proj", "up_proj", "gate_proj_bias", "up_proj_bias"}
        converted: dict[str, Any | None] = {
            name: (
                None if name in fused
                else self._optional_tensor(getattr(layer, name, None))
            )
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

        # ── Settle every projection's orientation once, at load time ───────
        # Some checkpoints (GPT-2 and GPT-Neo's Conv1D) store linear weights as
        # ``(in, out)``.  Deriving that per call, as the public ``_linear`` does,
        # costs a handful of Python operations for every projection in every
        # layer for every token — on a small model that is a real share of the
        # step.  Orienting them here lets the decode loop call the GEMM directly.
        attention_width = self.num_heads * self.head_dim
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            if converted[name] is not None:
                converted[name] = self._orient(converted[name], hidden, name)
        if converted["o_proj"] is not None:
            converted["o_proj"] = self._orient(
                converted["o_proj"], attention_width, "o_proj"
            )
        # The FFN input width follows from whichever gate/up projection exists;
        # a GELU block (GPT-2, GPT-Neo) has only the one.  When the pair was fused
        # it comes from the host arrays, which the fused upload read but never
        # materialized on the device.
        ffn_width = next(
            (int(converted[name].shape[0]) for name in ("gate_proj", "up_proj")
             if converted[name] is not None),
            None,
        )
        if ffn_width is None and ffn_group is not None:
            ffn_width = int(ffn_group[0][0].shape[0])
        if converted["down_proj"] is not None and ffn_width is not None:
            converted["down_proj"] = self._orient(
                converted["down_proj"], ffn_width, "down_proj"
            )
        for expert in converted["experts"]:
            for name in ("gate_proj", "up_proj"):
                if expert[name] is not None:
                    expert[name] = self._orient(expert[name], hidden, f"expert {name}")
            expert_width = next(
                (int(expert[name].shape[0]) for name in ("gate_proj", "up_proj")
                 if expert[name] is not None),
                None,
            )
            if expert["down_proj"] is not None and expert_width is not None:
                expert["down_proj"] = self._orient(
                    expert["down_proj"], expert_width, "expert down_proj"
                )

        # ── Pack projections that share an input into one device GEMM ───────
        # The concatenation is algebraically lossless and removes two launch
        # boundaries from the decode hot path.  Keep the logical AEG layout in
        # the source engine; only the device-resident representation is packed.
        #
        # The parts are uploaded straight into the fused buffer's row slices
        # (:meth:`_packed_tensor`) rather than materialized on the device and then
        # concatenated, so the fused layout costs one device allocation instead of
        # two and leaves the allocator nothing it cannot reuse.
        if qkv_group is not None:
            parts, biases = qkv_group
            converted["qkv_weight"] = self._packed_tensor(list(parts))
            converted["qkv_bias"] = (
                None if biases[0] is None else self._packed_tensor(list(biases))
            )
            converted["q_width"] = int(parts[0].shape[0])
            converted["k_width"] = int(parts[1].shape[0])
            converted["v_width"] = int(parts[2].shape[0])
        if ffn_group is not None:
            parts, biases = ffn_group
            converted["gate_up_weight"] = self._packed_tensor(list(parts))
            converted["gate_up_bias"] = (
                None if biases[0] is None else self._packed_tensor(list(biases))
            )
            converted["gate_width"] = int(parts[0].shape[0])
            converted["up_width"] = int(parts[1].shape[0])
        return converted

    def _fusible(
        self, layer: Any, names: "tuple[str, ...]", in_features: int
    ) -> "tuple[tuple[Any, ...], tuple[Any, ...]] | None":
        """Host arrays for a fusible projection group, oriented, or ``None``.

        A group fuses only when every member is present and the biases are either
        all present or all absent — a partially-biased fusion would silently change
        the arithmetic. Deciding this from the *host* arrays, before anything is
        uploaded, is what lets the members go straight into the fused device buffer
        instead of existing as separate device tensors first.
        """
        parts = [getattr(layer, name, None) for name in names]
        if any(part is None for part in parts):
            return None
        biases = [getattr(layer, f"{name}_bias", None) for name in names]
        present = [bias is not None for bias in biases]
        if any(present) and not all(present):
            return None
        try:
            oriented = [
                self._orient_host(np.asarray(part), in_features, name)
                for part, name in zip(parts, names)
            ]
        except (ValueError, IndexError):
            return None  # an unexpected layout: fall back to the unfused path
        if len({int(part.shape[1]) for part in oriented}) != 1:
            return None
        return tuple(oriented), tuple(
            None if bias is None else np.asarray(bias).reshape(-1) for bias in biases
        )

    @staticmethod
    def _orient_host(value: Any, in_features: int, role: str) -> Any:
        """``_orient`` for a host array: returns a ``(out, in)`` view, no copy.

        The transpose is a view, and the contiguous copy the transfer needs is made
        by :meth:`_host_tensor` during the upload — so a Conv1D checkpoint pays one
        host copy per projection rather than a device round trip.
        """
        if value.ndim != 2:
            raise ValueError(f"{role} projection is not a matrix: shape {tuple(value.shape)}")
        if int(value.shape[1]) == in_features:
            return value
        if int(value.shape[0]) == in_features:
            return value.T
        raise ValueError(
            f"{role} projection does not contract {in_features} input features: "
            f"got shape {tuple(value.shape)}"
        )

    @staticmethod
    def _orient(value: Any, in_features: int, role: str) -> Any:
        """Return ``value`` laid out as ``(out_features, in_features)``.

        Some source checkpoints (GPT-2's Conv1D) store linear weights
        transposed relative to ``nn.Linear``.  Settling the orientation once, at
        load time, is what allows both the fused packing below and the decode
        loop to assume a single layout.
        """
        if int(value.shape[1]) == in_features:
            return value
        if int(value.shape[0]) == in_features:
            return value.transpose(0, 1).contiguous()
        raise ValueError(
            f"{role} projection does not contract {in_features} input features: "
            f"got shape {tuple(value.shape)}"
        )

    def _rope_is_length_sensitive(self) -> bool:
        """Whether the rotary frequencies must be re-derived per length.

        Derived once from the scaling spec, because the answer is a property of the
        compiled checkpoint and cannot change while the engine lives.  ``dynamic``
        rescales the base with the current length and ``longrope`` switches tables
        at a threshold; identity, ``linear``/``su``, ``yarn`` and ``llama3`` are
        functions of the configuration alone.
        """
        cached = self._rope_length_sensitive
        if cached is None:
            spec = self.rope_scaling_spec
            cached = bool(
                spec is not None
                and not spec.is_identity
                and str(spec.rope_type) in self._LENGTH_SENSITIVE_ROPE
            )
            self._rope_length_sensitive = cached
        return cached

    def _ensure_rope(self, required: int, device: Any | None = None) -> None:
        """Grow the rotary tables to cover ``required`` positions.

        Growth is bounded by a fixed headroom rather than by doubling.  The
        tables are indexed by absolute position, so their height tracks sequence
        length, and a multiplicative policy would keep reserving positions the
        model can never reach — for a 40960-position context that is gigabytes
        of sin/cos on the accelerator.  Adding a slab amortizes the rebuild just
        as well without the runaway.
        """
        device = device or self.device
        if (
            self._cos is not None
            and int(self._cos.shape[0]) >= required
            and self._cos.device == device
            and self._rope_inv_freq is not None
            and not self._rope_is_length_sensitive()
        ):
            # The tables already cover this length, and this model's frequencies do
            # not move with it, so recomputing them could only establish that they
            # are unchanged.  Establishing that costs a double-precision power over
            # the head half-width plus a full array comparison, on the host, at the
            # head of every forward pass — including the prompt pass, where it is
            # the first thing time-to-first-token pays for.  The two schemes that
            # *are* length-dependent skip this and take the path below, unchanged.
            return
        inverse, attention = scaled_inverse_frequencies(
            self.rope_theta,
            self.rotary_dim,
            self.rope_scaling_spec,
            sequence_length=int(required),
        )
        current = 0 if self._cos is None else int(self._cos.shape[0])
        unchanged = (
            self._rope_inv_freq is not None
            and self._rope_inv_freq.shape == inverse.shape
            and np.array_equal(self._rope_inv_freq, inverse)
        )
        if (
            self._cos is not None
            and current >= required
            and self._cos.device == device
            and unchanged
        ):
            return
        self._rope_inv_freq = inverse
        self.rope_attention_scaling = float(attention)
        capacity = min(
            max(int(required) + self._ROPE_HEADROOM, self._ROPE_HEADROOM),
            self.max_positions,
        )
        if capacity < int(required):
            raise ValueError(
                f"sequence length {required} exceeds the compiled context length "
                f"{self.max_positions}"
            )
        # The frequencies were derived for ``required`` — the length this request will
        # actually evaluate — and ``capacity`` is only how many rows of the table are
        # materialized so the next few steps need no rebuild.  Handing the derived
        # pair down, rather than letting the table builder re-derive at ``capacity``,
        # is what keeps the two in agreement; see the note in :meth:`_rope_tables`.
        self._cos, self._sin = self._rope_tables(
            self.rope_theta, capacity, device, frequencies=(inverse, attention)
        )
        if self.rope_local_theta is not None:
            # A second base needs its own derivation, but at the same length.
            self._local_cos, self._local_sin = self._rope_tables(
                self.rope_local_theta, capacity, device, length=int(required)
            )

    def _rope_tables(
        self,
        theta: float,
        required: int,
        device: Any,
        *,
        frequencies: "tuple[np.ndarray, float] | None" = None,
        length: int | None = None,
    ) -> tuple[Any, Any]:
        """Build cos/sin tables for one rotary base, pre-expanded to full width.

        The angles span ``rotary_dim / 2`` frequencies — the full head half-width
        unless the architecture rotates only a prefix of each head (GPT-NeoX,
        GPT-J, StableLM, Phi, GLM-4).  Each frequency is then duplicated to the
        rotated width *in the layout the model's convention needs*, so the
        per-token path applies the rotation as

            x·cos + rotate(x)·sin

        with no table reshaping and no dtype conversion at all.  Storing the
        expansion once, at table-build time, removes several tensor operations
        per layer per token, which is the dominant cost of small-model decode.

        ``required`` is the table *height* — how many absolute positions to
        materialize — and it is deliberately not the length the frequencies are
        derived for.  Two rotary schemes choose their frequencies by length
        (``dynamic`` rescales the base, ``longrope`` switches between a short and a
        long factor table at the trained context), and the height carries headroom so
        the next few steps need no rebuild.  Deriving from the height would therefore
        answer a question nobody asked: for a Phi-3.5 prompt of 3700 positions the
        height is 4212, which is past the trained 4096 boundary, so the table would be
        built with the *long* factors for a request the reference evaluates with the
        short ones — every slow dimension rotated at a fraction of its correct rate.

        So the length is stated separately.  ``frequencies`` passes a pair the caller
        already derived — the normal path, because the caller has to record what the
        tables were built from — and ``length`` derives a second base at the same
        length.  When neither is given the height stands in, which is correct only for
        a length-insensitive scheme and is kept for callers that have no length to
        offer.
        """
        if frequencies is not None:
            inverse, attention = frequencies
        else:
            inverse, attention = scaled_inverse_frequencies(
                theta,
                self.rotary_dim,
                self.rope_scaling_spec,
                sequence_length=int(length if length is not None else required),
            )
        positions = self.torch.arange(required, device=device, dtype=self.torch.float32)[:, None]
        inv_freq = self.torch.as_tensor(
            inverse, device=device, dtype=self.torch.float32
        )
        angles = positions * inv_freq[None, :]
        cos, sin = self.torch.cos(angles), self.torch.sin(angles)
        if attention != 1.0:
            # YaRN's temperature correction, applied to the tables exactly as the
            # reference applies it.  It scales the rotated Q and K, so folding it
            # into the softmax scale instead would change the result by its square.
            cos = cos * attention
            sin = sin * attention
        if self.rope_interleaved:
            # Adjacent-channel pairing (GPT-J, Cohere, GLM-4): frequency i covers
            # channels 2i and 2i+1.
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        else:
            # Half-split pairing (Llama, Qwen, Mistral, GPT-NeoX): frequency i
            # covers channels i and i + rotary/2.
            cos = self.torch.cat((cos, cos), dim=-1)
            sin = self.torch.cat((sin, sin), dim=-1)
        return cos.to(dtype=self.compute_dtype), sin.to(dtype=self.compute_dtype)

    def _rope_slice(self, positions: Any, *, local: bool) -> tuple[Any, Any]:
        """Gather the cos/sin rows for ``positions``, shaped for broadcasting.

        Hoisted out of the layer loop: the rotation factors depend only on the
        positions and the rotary base, so a decode step needs them once rather
        than twice per layer.

        ``positions`` is rank-1 ``(seq,)`` for a single sequence and rank-2
        ``(batch, seq)`` for a batch, where each row carries its *own* positions
        because left padding shifts a short row's padded indices but must not
        shift its angles.  The result broadcasts against the matching activation
        layout — ``(seq, 1, dim)`` or ``(batch, seq, 1, dim)``.
        """
        cos_table, sin_table = (
            (self._local_cos, self._local_sin)
            if local and self._local_cos is not None
            else (self._cos, self._sin)
        )
        assert cos_table is not None and sin_table is not None
        if positions.dim() == 2:
            flat = positions.reshape(-1)
            batch, seq = int(positions.shape[0]), int(positions.shape[1])
            cos = cos_table.index_select(0, flat)
            sin = sin_table.index_select(0, flat)
            width = int(cos.shape[-1])
            return (
                cos.reshape(batch, seq, 1, width),
                sin.reshape(batch, seq, 1, width),
            )
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

    #: Half-precision verdict per device type, so the probe below runs once per
    #: process rather than once per loaded model.
    _HALF_PRECISION_SUPPORT: "dict[str, Any]" = {}

    @classmethod
    def _probe_compute_dtype(cls, torch: Any, device: Any) -> Any:
        """Choose residency precision for a non-CUDA device by *asking* the device.

        A CPU stays at FP32 — half precision there is emulated and slower.  Every other
        accelerator is a question rather than an assumption: Metal, XPU, and backends
        Aether has not seen all support half precision to differing degrees, and
        hard-coding FP32 for "not CUDA" leaves an Apple or Intel GPU running at a
        fraction of its rate for no reason other than the branch it failed.

        So a tiny matmul is run in FP16 and checked against the FP32 result.  Half
        precision is adopted only if the backend both executes it and agrees — the same
        measure-then-select discipline the decode kernel strategy uses, and the reason
        this cannot regress a device where FP16 is unimplemented or wrong.

        The tolerance is loose because FP16 accumulation genuinely differs from FP32;
        what is being detected is a backend that silently produces garbage or refuses,
        not the last bit of a rounding difference.
        """
        kind = getattr(device, "type", "cpu")
        cached = cls._HALF_PRECISION_SUPPORT.get(kind)
        if cached is not None:
            return cached
        if kind == "cpu":
            cls._HALF_PRECISION_SUPPORT[kind] = torch.float32
            return torch.float32
        verdict = torch.float32
        try:
            left = torch.ones((8, 8), dtype=torch.float32, device=device)
            right = torch.full((8, 8), 0.5, dtype=torch.float32, device=device)
            reference = left @ right
            produced = (left.half() @ right.half()).float()
            if torch.allclose(produced, reference, rtol=5e-2, atol=5e-2):
                verdict = torch.float16
        except Exception as exc:  # noqa: BLE001 - an unsupported dtype is an answer
            logger.debug("half precision unavailable on %s (%s); using fp32", kind, exc)
        cls._HALF_PRECISION_SUPPORT[kind] = verdict
        logger.debug("residency precision for %s: %s", kind, verdict)
        return verdict

    def _matmul(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        """Apply a projection whose orientation was settled at load time.

        ``_linear`` re-derives the layout and re-validates shapes on every call.
        That is the right behaviour at the public boundary, but inside the decode
        loop it is several Python operations per projection per layer, repeated
        for every token — enough to matter on a small model where each kernel is
        only microseconds of real work.

        ``self._projection_table`` holds the *measured* formulation for each
        projection shape this pass will run; see
        :mod:`aether.runtime.kernel_strategy`.  The lookup is by the weight's own
        ``(out, in)``, because that — not the row count alone — is what decides
        which formulation wins: :class:`~aether.runtime.kernel_strategy.ShapeClass`
        keys on the weight magnitude as well as on ``M``, and a decoder layer
        contains several magnitudes.  Selecting on one of them and applying the
        winner to the rest asks the backend for a layout that was measured on a
        different GEMM, which is exactly the case a measured mechanism exists to
        avoid.  Everything is ``F.linear`` until a calibration says otherwise, so
        an uncalibrated device behaves exactly as it did before the mechanism
        existed.

        ``None`` means a subclass never ran the parent constructor — the
        tensor-parallel executor is the case — and the reference kernel is used
        directly.  Reading the attribute rather than requiring it is what keeps a
        subclass from having to know that this optimisation exists at all.
        """
        table = self._projection_table
        if table:
            projection = table.get(weight.shape)
            if projection is not None:
                return projection(x, weight, bias)
        projection = self._projection
        if projection is None:
            return self.torch.nn.functional.linear(x, weight, bias)
        return projection(x, weight, bias)

    @staticmethod
    def _reference_projection_factory(torch: Any) -> Any:
        """The always-correct projection, bound once so the hot path has no branch."""
        linear = torch.nn.functional.linear

        def apply(x: Any, weight: Any, bias: Any | None = None) -> Any:
            return linear(x, weight, bias)

        return apply

    def _build_strategy_calibrator(self) -> Any:
        """Construct the projection-strategy calibrator, or ``None`` if unavailable.

        Never raises and never measures: an environment without the calibration module,
        without a writable store, or with calibration switched off simply keeps the
        reference kernel. Sharing the placement ledger means the strategy winners live
        beside the placement calibration, under one key per device and backend build.
        """
        try:
            from aether.runtime.kernel_strategy import (
                calibration_enabled,
                projection_strategies,
            )

            if not calibration_enabled():
                return None
            store = signature = backend_build = None
            try:
                from aether.placement.census import DeviceCapability, _backend_build
                from aether.placement.ledger import CalibrationLedger

                store = CalibrationLedger()
                backend_build = _backend_build()
                signature = DeviceCapability(
                    device_id=str(self.device), kind=self.device.type,
                    name=self._device_name(), total_bytes=0, free_bytes=0,
                    external_bytes=0, bandwidth_bps=0.0, flops=0.0,
                ).signature
            except Exception as exc:  # noqa: BLE001 - in-memory calibration is valid
                logger.debug("decode strategy calibration will not persist: %s", exc)
                store = None
            return projection_strategies(
                self.torch, self.device, store=store,
                signature=signature or "", backend_build=backend_build or "",
            )
        except Exception as exc:  # noqa: BLE001 - the reference kernel always works
            logger.debug("decode strategy calibration unavailable: %s", exc)
            return None

    def _device_name(self) -> str:
        """A stable name for the device class, for the calibration key."""
        try:
            if self.device.type == "cuda":
                return str(self.torch.cuda.get_device_name(self.device))
        except Exception:  # noqa: BLE001 - naming is advisory
            pass
        return self.device.type

    _PROBE_NAMES = (
        "qkv_weight", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_up_weight", "gate_proj", "up_proj", "down_proj",
    )
    """Projection slots a decoder block can hold, in the order a block runs them."""

    def _projection_probe_shapes(self) -> "list[tuple[Any, Any | None]]":
        """One class representative per *distinct projection shape* in a block.

        A shape class is ``(phase, M, K·N, dtype, device)`` — the weight's magnitude
        is part of it, because which formulation wins depends on how the backend
        tiles that particular ``(K, N)`` and not only on how thin ``M`` is.  A
        decoder block holds several magnitudes: Phi-3.5-mini's are ``9216×3072``
        (fused QKV), ``3072×3072`` (output), ``16384×3072`` (fused gate/up) and
        ``3072×8192`` (down).  Measuring one of them and applying its winner to the
        others hands the backend a formulation chosen for a different GEMM — the
        mistake a measured mechanism exists to prevent, and one that stays invisible
        until the model is large enough for the projections to dominate the step.

        The count is bounded by the architecture rather than by the model: a block
        has at most this many distinct projection shapes however many layers or
        parameters the model has, and every layer in the stack repeats the same set.
        Shapes are returned largest first, so if the calibration budget runs out the
        projection that moves the step time most is the one that was measured.
        """
        if not self.layers:
            return []
        block = self.layers[0]
        candidates: "list[Any]" = [block.get(name) for name in self._PROBE_NAMES]
        # An MoE block's experts carry their own shapes, and the activated experts
        # are as much of the step as a dense FFN is.
        for expert in (block.get("experts") or [])[:1]:
            candidates.extend(
                expert.get(name) for name in ("gate_proj", "up_proj", "down_proj")
            )
        # The head is a projection every step runs exactly once, and on a model with
        # a large vocabulary it is one of the largest reads in the step.
        candidates.append(getattr(self, "lm_head", None))
        seen: "dict[Any, Any]" = {}
        for weight in candidates:
            if weight is None or getattr(weight, "ndim", 0) != 2:
                continue
            seen.setdefault(tuple(weight.shape), weight)
        ordered = sorted(seen.values(), key=lambda weight: -int(weight.numel()))
        return [(weight, None) for weight in ordered]

    def _resolve_projection(self, rows: int, phase: str) -> None:
        """Select a projection formulation per shape class, measuring at most once.

        Called once per forward pass rather than once per projection: every layer in
        a pass shares the same row count, and every layer repeats the same set of
        weight shapes, so one resolution covers the whole stack.  Within a pass the
        choice is made *per shape*, because that is the granularity the calibrator's
        own key already has — see :meth:`_projection_probe_shapes`.  The calibrator
        memoises per shape class, so after the first pass of each class this is a
        handful of dictionary lookups and performs no measurement on the hot path.
        """
        calibrator = self._strategies
        if calibrator is None:
            return
        torch = self.torch
        table: "dict[Any, Any]" = {}
        choices: "dict[Any, Any]" = {}
        for weight, bias in self._projection_probe_shapes():

            def probe(weight: Any = weight, bias: Any = bias) -> "tuple[Any, Any, Any | None]":
                # The real weight, not a copy: it measures the layout that will
                # actually be used and costs no extra device memory at a moment when
                # the model is already resident.
                activation = torch.zeros(
                    (int(rows), int(weight.shape[1])),
                    dtype=weight.dtype, device=weight.device,
                )
                return activation, weight, bias

            try:
                choice = calibrator.choose(
                    phase=phase, rows=int(rows),
                    in_features=int(weight.shape[1]), out_features=int(weight.shape[0]),
                    dtype=weight.dtype, probe=probe,
                )
            except Exception as exc:  # noqa: BLE001 - never let selection break a pass
                logger.debug("projection strategy selection failed: %s", exc)
                continue
            shape = tuple(weight.shape)
            choices[shape] = choice
            table[shape] = (
                calibrator.strategy(choice)
                if choice.name != "linear"
                else self._reference_projection
            )
        if not table:
            return
        self._projection_table = table
        self._projection_choices = choices
        # The largest shape's choice stands in for any projection a block does not
        # hold — an adapter, or a shape that appears only in a later layer.  It is
        # the reference kernel unless a measurement replaced it, exactly as before.
        largest = max(choices, key=lambda shape: shape[0] * shape[1])
        self._projection_choice = choices[largest]
        self._projection = table[largest]

    def projection_report(self) -> dict[str, Any]:
        """Which projection formulation is in force, and how it was chosen."""
        if self._strategies is None:
            return {"enabled": False, "reason": "calibration unavailable"}
        report = self._strategies.report()
        if self._projection_choice is not None:
            report["active"] = self._projection_choice.to_dict()
        if self._projection_choices:
            report["per_shape"] = {
                str(shape): choice.to_dict()
                for shape, choice in self._projection_choices.items()
            }
        return report

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

        Evaluated as the single expression ``x·cos + rotate(x)·sin``, which is the
        published form and needs five tensor operations regardless of pairing
        convention.  Writing out the two halves explicitly instead costs seven,
        and rotary is applied twice per layer, so on a small model that
        difference is a measurable share of the whole decode step.

        Three source conventions are supported and are not interchangeable:
        half-split pairing ``(x_i, x_{i+d/2})`` used by GPT-NeoX, Llama, Qwen and
        Mistral; interleaved pairing ``(x_{2i}, x_{2i+1})`` published by GPT-J,
        Cohere and GLM-4; and partial rotation, which leaves the trailing
        ``head_dim - rotary_dim`` channels untouched.

        ``cos``/``sin`` come pre-expanded to the rotated width from
        :meth:`_rope_tables`; sliding-window layers may pass factors built from a
        separate rotary base (Gemma-3).
        """
        rotary = self.rotary_dim
        if rotary == int(x.shape[-1]):
            return x * cos + self._rotate(x) * sin
        prefix = x[..., :rotary]
        return self.torch.cat(
            (prefix * cos + self._rotate(prefix) * sin, x[..., rotary:]), dim=-1
        )

    @staticmethod
    def _has_qk_norm(layer: dict[str, Any]) -> bool:
        """Whether this layer normalizes Q or K between projection and rotary."""
        return layer["q_norm"] is not None or layer["k_norm"] is not None

    def _rotate(self, x: Any) -> Any:
        """Return the 90°-rotated companion of ``x`` for its pairing convention.

        Half-split yields ``[-x_h .. -x_2h, x_0 .. x_h]``; interleaved yields
        ``[-x_1, x_0, -x_3, x_2, ...]``.  Both are the standard formulations.
        """
        if self.rope_interleaved:
            even = x[..., 0::2]
            odd = x[..., 1::2]
            return self.torch.stack((-odd, even), dim=-1).flatten(-2)
        half = int(x.shape[-1]) // 2
        return self.torch.cat((-x[..., half:], x[..., :half]), dim=-1)

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
        """Execute top-k routed SwiGLU experts on the selected device.

        The router probabilities come from a softmax over **all** experts; only
        architectures that declare ``norm_topk_prob`` renormalize the selected
        k.  See :meth:`CPUExecutionEngine._moe_ffn` for why the two forms are
        not interchangeable.
        """
        torch = self.torch
        router = layer["router"]
        experts = layer["experts"]
        if router is None or not experts:
            raise ValueError("portable MoE layer is missing its router or experts")
        # Expert routing is strictly per token: a token's expert choice and output
        # depend on nothing but that token.  Collapsing a batch's leading axes into
        # one token axis is therefore lossless here, and it keeps the scatter/gather
        # dispatch below rank-2 for every batch size.
        #
        # This is the one place a flatten is safe.  The same flatten applied to
        # attention would splice independent sequences into one — the exact defect
        # the batched layout exists to prevent.
        lead = tuple(hidden.shape[:-1])
        flat = hidden.reshape(-1, int(hidden.shape[-1])) if len(lead) > 1 else hidden
        router_logits = self._linear(flat, router)
        top_k = min(int(layer["num_activated_experts"]), len(experts))
        if top_k <= 0:
            raise ValueError("portable MoE top-k must be positive")
        probabilities = torch.softmax(router_logits.float(), dim=-1)
        routing, selected = torch.topk(probabilities, top_k, dim=-1)
        if self.moe_renormalize_topk:
            routing = routing / routing.sum(dim=-1, keepdim=True)
        routing = routing.to(dtype=flat.dtype)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(experts):
            rows, slots = torch.where(selected == expert_index)
            if rows.numel() == 0:
                continue
            source = flat.index_select(0, rows)
            gate = self._linear(source, expert["gate_proj"], expert["gate_proj_bias"])
            up = self._linear(source, expert["up_proj"], expert["up_proj_bias"])
            activated = self._activation(gate, up)
            value = self._linear(activated, expert["down_proj"], expert["down_proj_bias"])
            output.index_put_((rows,), value * routing[rows, slots].unsqueeze(-1), accumulate=True)
        return output.reshape(*lead, int(output.shape[-1])) if len(lead) > 1 else output

    def _attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        query_positions: Any,
        key_positions: Any,
        window_size: int | None = None,
        scale: float | None = None,
        *,
        live: Any | None = None,
    ) -> Any:
        """Exact attention, dispatching to PyTorch's fused SDPA when possible.

        Rank-generic: ``q``/``k``/``v`` are ``(seq, heads, dim)`` for a single
        sequence or ``(batch, seq, heads, dim)`` for a batch, and the batch axis —
        being outermost — never mixes rows in any operation below.

        The boolean mask is intentionally expressed in source token positions, not
        cache row indices.  That keeps the same semantics for normal prefill,
        incremental decode, local GPT-Neo layers, and future cache
        implementations that compact rows.  In a padded batch the positions are
        per row, since a row's position at a padded slot is that slot minus the
        row's own pad count.

        ``live`` marks which key slots hold a real token, per row.  It is ``None``
        whenever nothing is masked — every single-sequence call, and every batch
        whose rows are equal length — and that is what preserves the fused path:
        a ``None`` mask lets prefill stay ``is_causal=True`` and lets a one-token
        decode pass no mask at all.
        """
        torch = self.torch
        is_alibi = self.is_alibi
        batched = q.dim() == 4
        seq_axis = 1 if batched else 0
        query_length = int(q.shape[seq_axis])
        key_length = int(k.shape[seq_axis])
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
        if local and query_length == 1 and live is None:
            # A one-token step against a sliding window admits a *contiguous
            # suffix* of the cache, so the constraint can be satisfied by slicing
            # the keys instead of masking them.  Two things follow, and both are
            # what makes local-attention families scale with batch:
            #
            #   * the mask disappears, so SDPA keeps its fused kernel instead of
            #     falling back to the math backend, which materializes the whole
            #     (batch, heads, 1, key) score tensor and expands grouped KV
            #     heads to full width;
            #   * attention reads ``window`` keys per step instead of ``key``,
            #     which at a 1024-token context with a 256-token window is four
            #     times less traffic on every local layer.
            #
            # Restricted to an unpadded batch on purpose: with ragged rows each
            # row's window starts at a different slot, so one slice cannot express
            # it and the mask remains the correct answer.
            span = int(window_size)
            if key_length > span:
                start = key_length - span
                k = k[:, start:] if batched else k[start:]
                v = v[:, start:] if batched else v[start:]
                key_positions = key_positions[..., start:]
                key_length = span
            local = False
            window_size = None
        scale = self.base_attention_scale if scale is None else float(scale)
        softcap = self.attn_logit_softcap

        # Broadcast the positions to (..., query, key) once; the causal/window mask
        # and the ALiBi distance both consume this shape.
        query_pos = query_positions.unsqueeze(-1)
        key_pos = key_positions.unsqueeze(1) if batched else key_positions.unsqueeze(0)

        def allowed_mask() -> Any:
            allowed = key_pos <= query_pos
            if local:
                allowed = allowed & (
                    key_pos >= query_pos - int(window_size) + 1
                )
            if live is not None:
                allowed = allowed & (live.unsqueeze(1) if batched else live)
            return allowed

        # SDPA accepts [batch, heads, query, dim].  enable_gqa avoids
        # materializing repeated KV heads on supported PyTorch versions.
        # SDPA has no soft-capping stage, so architectures that declare one
        # (Gemma-2) must take the exact path below.
        if not is_alibi and not softcap:
            if batched:
                q4 = q.transpose(1, 2)
                k4 = k.transpose(1, 2)
                v4 = v.transpose(1, 2)
            else:
                q4 = q.transpose(0, 1).unsqueeze(0)
                k4 = k.transpose(0, 1).unsqueeze(0)
                v4 = v.transpose(0, 1).unsqueeze(0)
            attn_mask = None
            is_causal = bool(is_prefill and contiguous and not local and live is None)
            if not is_causal and not (
                query_length == 1 and not local and live is None
            ):
                attn_mask = allowed_mask()
                if batched:
                    attn_mask = attn_mask.unsqueeze(1)
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
                if batched:
                    return context.transpose(1, 2)
                return context.squeeze(0).transpose(0, 1)
            except (TypeError, RuntimeError):
                # Older PyTorch builds do not expose enable_gqa or may lack a
                # suitable fused backend.  The exact fallback remains valid.
                pass

        repeats = self.num_heads // self.num_kv_heads
        if repeats == 1:
            k_full, v_full = k, v
        else:
            # The head axis is second-from-last in both ranks.
            k_full = k.repeat_interleave(repeats, dim=-2)
            v_full = v.repeat_interleave(repeats, dim=-2)
        if batched:
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k_full) * scale
        else:
            scores = torch.einsum("qhd,khd->hqk", q, k_full) * scale
        if is_alibi:
            distance = (key_pos - query_pos).to(dtype=scores.dtype)
            if batched:
                scores = scores + self._alibi_slopes[None, :, None, None] * distance.unsqueeze(1)
            else:
                scores = scores + self._alibi_slopes[:, None, None] * distance.unsqueeze(0)
        if softcap:
            # Gemma-2 bounds attention logits before masking; capping after the
            # mask would turn -inf into -cap and admit masked positions.
            scores = self._softcap(scores, softcap)
        allowed = allowed_mask()
        mask = allowed.unsqueeze(1) if batched else allowed.unsqueeze(0)
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
        probs = torch.softmax(scores, dim=-1)
        if batched:
            return torch.einsum("bhqk,bkhd->bqhd", probs, v_full)
        return torch.einsum("hqk,khd->qhd", probs, v_full)

    def _append_kv(
        self, old: Any | None, value: Any, past: int, total: int, reserve: int = 0
    ) -> Any:
        """Append KV in amortized-linear storage instead of copying the prefix.

        ``reserve`` lets a caller that already knows the final sequence length
        allocate once, removing every reallocation and prefix copy from the
        decode loop.

        Handles both ``(seq, heads, dim)`` and ``(batch, seq, heads, dim)``.  In
        the batched case the write is a single slice across every row, which is
        exactly what right-aligning the sequences buys: one frontier for the whole
        batch instead of a per-row scatter.
        """
        torch = self.torch
        batched = value.dim() == 4
        axis = 1 if batched else 0
        if old is None or int(old.shape[axis]) < total:
            old_capacity = 0 if old is None else int(old.shape[axis])
            capacity = max(total, reserve, max(16, old_capacity * 2))
            shape = (
                (int(value.shape[0]), capacity, *tuple(value.shape[2:]))
                if batched
                else (capacity, *tuple(value.shape[1:]))
            )
            result = torch.empty(shape, dtype=value.dtype, device=value.device)
            if old is not None and past:
                if batched:
                    result[:, :past].copy_(old[:, :past])
                else:
                    result[:past].copy_(old[:past])
        else:
            result = old
        if batched:
            result[:, past:total].copy_(value)
        else:
            result[past:total].copy_(value)
        return result

    def _forward_device(
        self,
        token_ids: np.ndarray | Any,
        cache: TorchKVCache | BatchedKVCache | None = None,
        *,
        validate_ids: bool = False,
        reserve: int = 0,
        batched: bool = False,
        logits: str = "all",
    ) -> tuple[Any, TorchKVCache | BatchedKVCache]:
        """Forward pass that keeps logits on the accelerator for generation.

        One implementation serves both shapes.  With ``batched=False`` the ids are
        rank-1 ``(seq,)`` and every activation is ``(seq, heads, dim)`` — the
        single-sequence path, behaviourally unchanged.  With ``batched=True`` the
        ids are rank-2 ``(batch, seq)`` and the batch axis rides outermost through
        the same layer loop.

        The architecture-variant handling below — MoE, ALiBi, parallel residual,
        post and sandwich norm, Q/K norm scope, partial and interleaved rotary,
        sliding window, logit softcap — is deliberately *not* duplicated per rank.
        Two copies would drift, and a drifted variant is a silent numerical error
        rather than a crash.  Only the rank-dependent sites branch.

        ``logits`` selects how much of the vocabulary projection to evaluate:

        ``"all"``
            Every position, ``(seq, vocab)`` or ``(batch, seq, vocab)``.  What the
            public :meth:`forward` and :meth:`forward_batch` contracts promise.
        ``"last"``
            Only each row's final position.  The vocabulary projection is applied
            per position independently, so in exact arithmetic
            ``(W x)[-1] == W x[-1]``: this declines to compute discarded rows rather
            than approximating them.  Generation reads nothing but the final row, so
            every generation path uses it.  See :meth:`_project_logits` for the
            floating-point caveat.
        """
        torch = self.torch
        # Keep decode tokens on the execution device.  The old path converted
        # every one-token decode step through NumPy, which forced a host-to-
        # device copy and made CUDA's scalar validation below synchronize the
        # stream.  Public ``forward`` and the initial prompt still opt into
        # validation; internally generated tokens are already produced by the
        # model and do not need a second round trip through the host.
        vocabulary = int(self.embedding.shape[0])
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids if batched else token_ids.reshape(-1)
            if ids.device != self.device or ids.dtype != torch.long:
                ids = ids.to(device=self.device, dtype=torch.long)
            if validate_ids and ids.numel():
                # One readback for both bounds instead of two.  Each ``.item()`` is a
                # blocking copy, and this check runs at the head of the prompt pass —
                # the two it used to issue were the first thing time-to-first-token
                # paid for, before any arithmetic had been queued.
                low, high = torch.stack((ids.min(), ids.max())).tolist()
                if low < 0 or high >= vocabulary:
                    raise ValueError("token id is outside the compiled vocabulary")
        else:
            array = np.asarray(token_ids, dtype=np.int64)
            if not batched:
                array = array.reshape(-1)
            if validate_ids and array.size:
                # The ids arrive from a tokenizer as a host array, so their bounds are
                # a host question.  Asking it before the upload keeps the answer free:
                # asking it afterwards means copying two scalars back off the device,
                # which stalls the stream to learn something the host already knew.
                if int(array.min()) < 0 or int(array.max()) >= vocabulary:
                    raise ValueError("token id is outside the compiled vocabulary")
            ids = torch.as_tensor(array, device=self.device)
        if batched and ids.dim() != 2:
            raise ValueError(
                "a batched forward pass requires rank-2 (batch, seq) ids, got rank "
                f"{ids.dim()}"
            )
        if ids.numel() == 0:
            raise ValueError("forward() requires at least one token")
        if cache is None:
            cache = (
                self._new_batched_cache(batch_size=int(ids.shape[0]), reserve=reserve)
                if batched
                else TorchKVCache([None] * self.num_layers, [None] * self.num_layers)
            )
        past = int(cache.length)
        lead = tuple(ids.shape)
        seq_len = int(ids.shape[-1])
        total = past + seq_len
        # A learned position table has exactly as many rows as the model has
        # positions, and nothing extends it: the rotary tables below are *derived*
        # and can be rebuilt taller, but a trained table cannot, so a step past its
        # last row has no position to read.  The reference and hybrid executors both
        # refuse here; this one handed the index to ``index_select``, which on an
        # accelerator is an asynchronous device-side assert -- it poisons the CUDA
        # context, so the process dies at the next unrelated call (cleanup, an
        # allocator query) with a stack that points anywhere but here.  One host
        # integer comparison per pass buys the same refusal the reference gives.
        if self.position_embedding is not None and total > self.max_positions:
            raise ValueError(
                f"sequence end {total} exceeds learned position embedding capacity "
                f"{self.max_positions}"
            )
        span = torch.arange(past, total, device=self.device, dtype=torch.long)
        if batched:
            # A row's position is its padded index minus its *own* pad count, so a
            # short row is not rotated as though it began mid-sequence.  Clamped
            # because a pad slot has no position; the mask excludes it from
            # attention, so the clamped value is never observable.
            positions = (span.unsqueeze(0) - cache.pad_counts).clamp_(min=0)
            key_positions = (
                self._key_positions(total).unsqueeze(0) - cache.pad_counts
            ).clamp_(min=0)
            live_view = cache.live_view(total)
        else:
            positions = span
            key_positions = self._key_positions(total)
            live_view = None
        if self.uses_rope:
            self._ensure_rope(total)
        if batched:
            hidden = self.embedding.index_select(0, ids.reshape(-1)).reshape(
                *lead, int(self.embedding.shape[1])
            )
        else:
            hidden = self.embedding.index_select(0, ids)
        if self.embedding_scale is not None:
            # Gemma scales embeddings by sqrt(hidden_size); Granite uses an
            # explicit embedding_multiplier.  Both are part of the model.
            hidden = hidden * self.embedding_scale
        if self.embedding_norm is not None:
            hidden = self._norm(hidden, self.embedding_norm, self.embedding_norm_bias)
        if self.position_embedding is not None:
            # Learned absolute positions read the same per-row positions as rotary
            # does.  Indexing this table by the padded slot instead would shift
            # every short row's embedding — the failure is invisible in shapes and
            # is why positions are computed once, above, for both consumers.
            table = self.position_embedding
            if batched:
                hidden = hidden + table.index_select(0, positions.reshape(-1)).reshape(
                    *lead, int(table.shape[1])
                )
            else:
                hidden = hidden + table.index_select(0, positions)

        post_norm = self.norm_placement == "post"
        parallel = self.parallel_residual
        residual_scale = self.residual_scale
        # One strategy resolution per pass, not per projection: every layer shares this
        # pass's row count, which is the dimension the choice turns on. Memoised inside
        # the calibrator, so this is a dictionary lookup after the first pass of each
        # shape class and never a measurement on the hot path.
        #
        # The gate carries the phase as well as the row count, because the phase is
        # part of the calibrator's key and the two are not in one-to-one
        # correspondence: a batch-8 decode step and an 8-token prefill both have eight
        # rows.  Keying on rows alone let whichever came first supply the other's
        # formulation — the same category error as choosing on one weight shape and
        # applying it to all of them.
        rows = int(ids.numel())
        phase = "prefill" if seq_len > 1 else "decode"
        if (rows, phase) != self._projection_rows:
            self._projection_rows = (rows, phase)
            self._resolve_projection(rows, phase)
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
                    qkv = self._matmul(normed, layer["qkv_weight"], layer["qkv_bias"])
                    q_width = int(layer["q_width"])
                    k_width = int(layer["k_width"])
                    q = qkv[..., :q_width]
                    k = qkv[..., q_width:q_width + k_width]
                    v = qkv[..., q_width + k_width:]
                    # Q and K are adjacent in the fused output and share
                    # ``head_dim``, so when no Q/K normalization intervenes the
                    # pair can be viewed as one head-major tensor and rotated in
                    # a single pass — halving the rotary work per layer.
                    qk_fused = (
                        qkv[..., : q_width + k_width]
                        if not self._has_qk_norm(layer)
                        else None
                    )
                else:
                    q = self._matmul(normed, layer["q_proj"], layer["q_proj_bias"])
                    k = self._matmul(normed, layer["k_proj"], layer["k_proj_bias"])
                    v = self._matmul(normed, layer["v_proj"], layer["v_proj_bias"])
                    qk_fused = None
                if layer_uses_rope and qk_fused is not None:
                    cos, sin = rope_local if local_attention else rope_global
                    rotated = self._rope(
                        qk_fused.reshape(
                            *lead, self.num_heads + self.num_kv_heads, self.head_dim
                        ),
                        cos,
                        sin,
                    )
                    # Index the head axis explicitly: it is second-from-last in both
                    # ranks, whereas ``[:, :n]`` would slice the sequence in a batch.
                    q = rotated[..., : self.num_heads, :]
                    k = rotated[..., self.num_heads :, :]
                else:
                    if self.qk_norm_is_full:
                        # OLMo-2 and OLMoE normalize the whole projection, before
                        # it is split into heads.
                        if layer["q_norm"] is not None:
                            q = self._norm(q, layer["q_norm"])
                        if layer["k_norm"] is not None:
                            k = self._norm(k, layer["k_norm"])
                    q = q.reshape(*lead, self.num_heads, self.head_dim)
                    k = k.reshape(*lead, self.num_kv_heads, self.head_dim)
                    if not self.qk_norm_is_full:
                        if layer["q_norm"] is not None:
                            q = self._norm(q, layer["q_norm"])
                        if layer["k_norm"] is not None:
                            k = self._norm(k, layer["k_norm"])
                    if layer_uses_rope:
                        cos, sin = rope_local if local_attention else rope_global
                        # Normalized Q and K are separate tensors but still share
                        # the rotation, so one concatenation still beats rotating
                        # each of them independently.
                        rotated = self._rope(torch.cat((q, k), dim=-2), cos, sin)
                        q = rotated[..., : self.num_heads, :]
                        k = rotated[..., self.num_heads :, :]
                v = v.reshape(*lead, self.num_kv_heads, self.head_dim)
                k_all = self._append_kv(cache.keys[index], k, past, total, reserve)
                v_all = self._append_kv(cache.values[index], v, past, total, reserve)
                context = self._attention(
                    q,
                    k_all[:, :total] if batched else k_all[:total],
                    v_all[:, :total] if batched else v_all[:total],
                    positions,
                    key_positions,
                    attention_window,
                    attention_scale,
                    live=live_view,
                )
                cache.keys[index] = k_all
                cache.values[index] = v_all
                attention_out = self._matmul(
                    context.reshape(*lead, self.num_heads * self.head_dim),
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
                    attention_out = attention_out * residual_scale

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
                        gate_up = self._matmul(ffn_input, layer["gate_up_weight"], layer["gate_up_bias"])
                        gate_width = int(layer["gate_width"])
                        gate = gate_up[..., :gate_width]
                        up = gate_up[..., gate_width:]
                    else:
                        gate = self._matmul(ffn_input, layer["gate_proj"], layer["gate_proj_bias"])
                        up = self._matmul(ffn_input, layer["up_proj"], layer["up_proj_bias"]) if layer["up_proj"] is not None else None
                    ffn_out = self._matmul(self._activation(gate, up), layer["down_proj"], layer["down_proj_bias"])
                if layer["post_ffn_norm"] is not None:
                    ffn_out = self._norm(
                        ffn_out, layer["post_ffn_norm"], layer["post_ffn_norm_bias"]
                    )
                elif post_norm:
                    ffn_out = self._norm(ffn_out, layer["ffn_norm"], layer["ffn_norm_bias"])
                if residual_scale is not None:
                    ffn_out = ffn_out * residual_scale
                if parallel:
                    hidden = block_input + attention_out + ffn_out
                else:
                    hidden = hidden + ffn_out
            logits_out = self._project_logits(hidden, batched=batched, mode=logits)
            cache.length = total
            cache.last_logits = (
                logits_out[:, -1] if batched else logits_out[-1]
            ).detach()
        return logits_out, cache

    def _project_logits(self, hidden: Any, *, batched: bool, mode: str) -> Any:
        """Normalize and project hidden states to vocabulary logits.

        ``mode="last"`` slices to each row's final position *before* the projection
        rather than after.  Both the final normalization and the vocabulary
        projection act on one position at a time — the norm reduces over the hidden
        axis, and the projection is a per-position matrix product — so in exact
        arithmetic

            (W · X)[..., -1, :]  ==  W · X[..., -1, :]

        Slicing first therefore declines to compute rows the caller discards; it
        does not approximate them.

        In floating point the two are close but not bit-identical: BLAS blocks a
        one-row GEMV differently from an S-row GEMM, so the sum over the contracted
        hidden dimension accumulates in a different order.  Measured disagreement is
        at rounding scale — order 1e-8 absolute and 1e-6 relative on FP32 logits —
        which is the same class of difference that already exists between this
        executor's batched and unbatched paths.  It is far below the gap between
        competing greedy candidates in practice, and
        ``tests/unit/test_batched_inference.py`` asserts both the numerical bound and
        that greedy decoding is unchanged.

        The saving is substantial on any model with a large vocabulary.  For
        Qwen3-0.6B the ``lm_head`` is 156M of ~596M matmul parameters — 26% of the
        prefill matmul work — and at a 1024-token prompt 1023/1024 of that was being
        thrown away.  The discarded logits also had to be *materialized*:
        ``(4, 1024, 151936)`` in FP16 is 1.19 GiB of allocation and write bandwidth.
        Measured on a real Qwen3-0.6B AEG (FP32, CPU), restricting the projection cut
        a 1024-token prefill by 39% (1.64x) and a 512-token prefill by 23% (1.30x);
        see ``scripts/profile_prefill.py``.

        Decode is unaffected either way, since a decode step already has one
        position per row.
        """
        if mode not in {"all", "last"}:
            raise ValueError(f"logits mode must be 'all' or 'last', got {mode!r}")
        if mode == "last":
            hidden = hidden[:, -1:] if batched else hidden[-1:]
        hidden = self._norm(hidden, self.final_norm, self.final_norm_bias)
        logits = self._linear(hidden, self.lm_head)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale
        if self.final_logit_softcap:
            logits = self._softcap(logits, self.final_logit_softcap)
        return logits

    def forward(self, token_ids: np.ndarray | Any, cache: TorchKVCache | None = None) -> tuple[np.ndarray, TorchKVCache]:
        logits, cache = self._forward_device(token_ids, cache, validate_ids=True)
        return logits.detach().float().cpu().numpy(), cache

    def calibration_pass(self, batch: int = 1, tokens: int = 1) -> None:
        """Run one prefill of the given shape exactly as generation would, then discard.

        The placement planner calibrates its memory margin from a measured peak, and
        the measurement is only worth having if it measures the pass that will actually
        run. The obvious probe — call the public :meth:`forward` — does not: ``forward``
        is the *inspection* entry point, so it projects logits at every position and
        then copies them to the host as FP32. At the planner's default 2048-token
        ceiling on Qwen3-0.6B that is a ``(2048, 151936)`` FP16 tensor (594 MiB), an
        FP32 widening of the same (1.19 GiB), and a host copy of the FP32 — none of
        which any generation step ever allocates, because :meth:`generate_iter` asks
        for ``logits="last"`` and keeps them on the device.

        Probing through ``forward`` therefore did two things wrong at once. It
        inflated the load-time high-water mark by ~1.8 GiB of device memory that is
        never released afterwards, and it taught the ledger a residual for a pass that
        does not exist — which makes every later plan reserve transient headroom for
        phantom logits and hand the KV cache less than it could have.

        This probe takes the generation path: the prefill shape the plan has to
        survive, ``logits="last"``, KV reserved for the whole workload as
        :meth:`generate_iter` reserves it, and no host round trip. What it measures is
        what a request costs.

        It then takes one decode step on the cache the prefill built, for two reasons.
        A request is a prefill *and* a decode loop, so the peak the margin has to cover
        includes the decode step's transients — small, but they are allocated after the
        prefill's have been freed, which is exactly the pattern a fragmentation margin
        exists to survive. And the decode step is what resolves the decode kernel shape
        classes, so measuring it here is also what keeps that measurement off the first
        request's time-to-first-token. The KV cache is reserved one position longer so
        the step fits the allocation the prefill already made rather than doubling it.
        """
        tokens = max(1, int(tokens))
        batch = max(1, int(batch))
        # This probe is a prefill *and* one decode step on the cache it built, so it
        # evaluates ``tokens + 1`` positions -- one more than the caller's ceiling.
        # For a learned-absolute model that ceiling is a trained table height, and
        # there is no row past it: asked for GPT-Neo's full 2048-position context,
        # the decode step below would read position 2048 of a 2048-row table.  The
        # widest *representable* probe is derived from the model rather than trusted
        # from the caller, so the step the margin is calibrated against is a step the
        # model can actually take.
        tokens = min(tokens, max(1, self.max_positions - 1))
        ids = np.zeros(tokens, dtype=np.int64)
        # Reserve as generation does, so the KV cache is allocated at its final size
        # once instead of doubling — the same allocation shape a real request makes.
        # The +1 is the decode step below: without it the append would find the cache
        # exactly full and double it, which would report a peak no request ever hits.
        reserve = tokens + 1
        cache = None
        logits = None
        try:
            if batch > 1 and self.supports_batch(batch):
                logits, cache = self._forward_device(
                    np.broadcast_to(ids, (batch, tokens)).copy(),
                    None,
                    reserve=reserve,
                    batched=True,
                    logits="last",
                )
                logits, cache = self._forward_device(
                    np.zeros((batch, 1), dtype=np.int64),
                    cache,
                    reserve=reserve,
                    batched=True,
                    logits="last",
                )
            else:
                logits, cache = self._forward_device(
                    ids, None, reserve=reserve, logits="last"
                )
                logits, cache = self._forward_device(
                    np.zeros(1, dtype=np.int64), cache, reserve=reserve, logits="last"
                )
        finally:
            del logits, cache

    def warmup_kernels(self) -> bool:
        """Resolve the decode kernel shape classes before any request exists.

        Which GEMM formulation is fastest for a projection is *measured*, not assumed
        (see :mod:`aether.runtime.kernel_strategy`), and a measurement has to happen
        somewhere. Left to the hot path it happens inside the first request, where it
        is charged to that request's time-to-first-token: four candidates × sixteen
        iterations for each distinct projection shape in a block, on a device that is
        also paying allocator growth and first-touch for the same shapes. The
        calibrator's own docstring calls this "a bound on a load path rather than a
        per-request cost" — this method is what makes that true.

        The pass is deliberately the smallest one that resolves the classes that
        matter: a one-token prefill and a one-token decode step. Decode is where a
        request spends nearly all of its time and its row count is always one, so one
        step fixes the whole class. Prefill row counts vary per request and cannot be
        enumerated, so no attempt is made to cover them; a prefill still resolves its
        own class on first use, and it is the pass that amortizes a probe best.

        Cheap enough to be unconditional: two forwards at one token, a two-position KV
        cache, and the probes' own thin activations. Everything is released on the way
        out. It also warms what the first request would otherwise warm anyway — the
        allocator's decode-sized blocks and the backend's kernel selection cache.

        Non-fatal by construction. Returns whether the pass ran; a failure leaves the
        engine exactly as it was, with the classes to be resolved on first use.

        ``AETHER_KERNEL_WARMUP=0`` skips it, because an operator reproducing a
        first-token measurement needs to be able to put the cost back where it was.
        """
        setting = os.environ.get("AETHER_KERNEL_WARMUP", "1").strip().lower()
        if setting in {"0", "false", "no", "off"}:
            return False
        if not self.layers:
            return False
        cache = None
        logits = None
        try:
            logits, cache = self._forward_device(
                np.zeros(1, dtype=np.int64), None, reserve=2, logits="last"
            )
            logits, cache = self._forward_device(
                np.zeros(1, dtype=np.int64), cache, reserve=2, logits="last"
            )
        except Exception as exc:  # noqa: BLE001 - warming must never fail a load
            logger.debug("decode kernel warm-up unavailable: %s", exc)
            return False
        finally:
            del logits, cache
        return True

    def _sample(self, logits: Any, temperature: float, top_k: int, top_p: float) -> int:
        return int(self._sample_device(logits, temperature, top_k, top_p).item())

    def _sample_device(self, logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
        """Select the next token, leaving the result on the execution device.

        Returning a device tensor is what allows the caller to queue the next
        forward pass before reading this token back to the host.  Reading it
        immediately would drain the whole queue first.

        Rank-generic: a rank-1 ``(vocab,)`` input yields a 0-dim token, and a
        rank-2 ``(batch, vocab)`` input yields ``(batch,)`` — one token per row,
        every reduction taken over the last axis so rows never interact.
        """
        torch = self.torch
        batched = logits.dim() == 2
        values = logits.float()
        if temperature <= 0:
            return torch.argmax(values, dim=-1)
        values = values / float(temperature)
        if top_k > 0:
            top_k = min(int(top_k), int(values.shape[-1]))
            # Keep the threshold as a trailing singleton so it broadcasts against
            # both ranks; each row is cut at its own k-th best logit.
            threshold = torch.topk(values, top_k, dim=-1).values[..., -1:]
            values = values.masked_fill(values < threshold, -torch_inf(torch, values.dtype))
        probs = torch.softmax(values, dim=-1)
        if 0.0 < top_p < 1.0:
            return self._sample_top_p(probs, float(top_p))
        drawn = torch.multinomial(probs, 1)
        return drawn.reshape(-1) if batched else drawn.reshape(())

    # ── sampled-token handoff ─────────────────────────────────────────────────
    #
    # A decode step ends with one number the host needs: the token, to yield it and
    # to test it for a stop.  Where that number is *read* is a latency decision that
    # is independent of where the next step is *launched*, and conflating the two is
    # what makes a token wait for work it does not depend on.
    #
    # ``token.item()`` is a blocking copy, so it completes only after everything
    # already queued ahead of it.  Issued after the lookahead step has been queued —
    # which is what keeps the device busy and is worth keeping — it waits for that
    # entire step.  The token was ready before it started.
    #
    # Staging the copy to pinned host memory the moment the token is sampled, and
    # waiting on an event recorded right after it, separates the two: the stream is
    # in order, so the event completes when the copy does, no matter how much work
    # was queued behind it.  The lookahead still overlaps the host's launch cost, and
    # the token no longer waits a step to be read.

    def _readback_staging(self) -> "list[tuple[Any, Any]] | None":
        """Pinned slots and completion events for asynchronous token readback.

        ``None`` when this device cannot stage asynchronously, which is not a
        failure: the caller then falls back to reading the token late, exactly as
        before.  Two slots, because one token is in flight while the next is sampled.
        """
        slots = self._readback_slots
        if slots is not False:
            return slots  # type: ignore[return-value]
        torch = self.torch
        built: "list[tuple[Any, Any]] | None" = None
        try:
            namespace = getattr(torch, self.device.type, None)
            event_type = getattr(namespace, "Event", None)
            if event_type is not None:
                built = [
                    (
                        torch.empty(1, dtype=torch.long, pin_memory=True),
                        event_type(),
                    )
                    for _ in range(2)
                ]
        except Exception as exc:  # noqa: BLE001 - staging is an optimization
            logger.debug("token readback staging unavailable on %s: %s", self.device, exc)
            built = None
        self._readback_slots = built
        return built

    def _begin_readback(self, token: Any) -> "tuple[int | None, Any, Any, Any]":
        """Start moving one sampled token to the host without draining the queue.

        Returns ``(value, buffer, event, tensor)``, of which exactly one route to the
        token is live: ``value`` when it is already on the host, ``buffer`` with
        ``event`` when a staged copy is in flight, and ``tensor`` when this device
        offers no staging and the token must be read late — the behaviour that
        predates this method, kept so no device regresses.
        """
        if self.device.type == "cpu":
            # Nothing is asynchronous here, so reading now costs nothing and there
            # is no queue for a late read to skip ahead of.
            return int(token.item()), None, None, None
        slots = self._readback_staging()
        if slots is None:
            return None, None, None, token
        buffer, event = slots[self._readback_slot]
        self._readback_slot = (self._readback_slot + 1) % len(slots)
        buffer.copy_(token.reshape(1), non_blocking=True)
        event.record()
        return None, buffer, event, None

    @staticmethod
    def _finish_readback(handoff: "tuple[int | None, Any, Any, Any]") -> int:
        """Obtain the token, waiting for its own copy and nothing else."""
        value, buffer, event, tensor = handoff
        if value is not None:
            return value
        if event is not None:
            event.synchronize()
            return int(buffer[0])
        return int(tensor.item())

    def _sample_top_p(self, probs: Any, top_p: float) -> Any:
        """Nucleus sampling, entirely on the execution device.

        The nucleus is the shortest prefix of the descending-probability order
        whose cumulative mass reaches ``p``.  Finding it needs the full ordering:
        a cheaper top-k pre-truncation would only be exact if its mass already
        reached ``p``, and testing that requires reading a value back to the
        host.  One extra sort kernel is much cheaper than the pipeline drain a
        host read would cause on every token, so this keeps the sort.

        Every reduction is over the last axis, so a batched call computes an
        independent nucleus per row.
        """
        torch = self.torch
        batched = probs.dim() == 2
        ordered, order = torch.sort(probs, descending=True, dim=-1)
        # ``cumulative - ordered`` is the mass strictly before each entry, so
        # this keeps the first entry whose predecessors have not yet reached p,
        # always retaining at least the most probable token.
        cumulative = ordered.cumsum(dim=-1)
        selected = ordered * (cumulative - ordered <= top_p).to(dtype=ordered.dtype)
        selected = selected / selected.sum(dim=-1, keepdim=True)
        drawn = torch.multinomial(selected, 1)
        picked = order.gather(-1, drawn)
        return picked.reshape(-1) if batched else picked.reshape(())

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
            _, cache = self._forward_device(
                ids, cache, validate_ids=True, reserve=reserve, logits="last"
            )
            next_logits = cache.last_logits
        elif cache is not None and cache.last_logits is not None:
            next_logits = cache.last_logits
        else:
            raise ValueError("generation requires prompt ids or a populated cache")
        if grammar_session is None:
            yield from self._generate_pipelined(
                cache, next_logits, max_tokens, temperature, top_k, top_p, eos_token_id
            )
        else:
            yield from self._generate_constrained(
                cache, next_logits, max_tokens, temperature, top_k, top_p,
                eos_token_id, grammar_session,
            )
        if cache_callback is not None and cache is not None:
            cache_callback(cache)

    def _generate_pipelined(
        self,
        cache: TorchKVCache,
        next_logits: Any,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int | None,
    ) -> Iterator[int]:
        """Decode with the host running one step ahead of the device.

        Autoregressive decode has to move each sampled token back to the host,
        both to yield it and to test it against the stop token.  Doing that
        immediately after sampling serializes the pipeline: the host blocks
        until every kernel queued for this step has retired, so the several
        hundred launches that make up the *next* step cannot be issued while the
        device is still busy with this one.  For a small model — where a step is
        a few hundred microseconds of arithmetic but a comparable amount of
        launch work — that alternation roughly doubles the time per token.

        So the next step's forward pass and sampling are queued *first*, and only
        then is the current token collected.  The host's launch cost overlaps
        device execution instead of following it.

        What the queue-first ordering must *not* also do is make the token wait for
        that lookahead.  ``token.item()`` would: it is a blocking copy issued behind
        a full step of queued kernels, so the first token — which the prefill had
        already determined — arrived one whole decode step late, and every token
        after it arrived a step late too.  The token's copy is therefore staged the
        moment it is sampled, before the lookahead is queued, and only that copy is
        waited on.  See :meth:`_begin_readback`.  Identical kernels, identical
        overlap, identical tokens; each one just stops waiting for the next.

        The lookahead step is speculative: when generation stops early its KV
        rows are discarded by rewinding the cache length, so the returned cache
        describes exactly the tokens that were yielded.
        """
        if self.device.type == "cpu":
            # A synchronous device has no queue for the lookahead to overlap with, so
            # running it before yielding would only postpone the token by a whole
            # step.  Same kernels, same count, same tokens — just not in an order
            # that pays for an overlap this device cannot provide.
            yield from self._generate_serially(
                cache, next_logits, max_tokens, temperature, top_k, top_p, eos_token_id
            )
            return
        token_device = self._sample_device(next_logits, temperature, top_k, top_p)
        handoff = self._begin_readback(token_device)
        stops = stop_token_set(eos_token_id)
        for _ in range(int(max_tokens)):
            checkpoint = int(cache.length)
            _, cache = self._forward_device(
                token_device.reshape(1), cache, logits="last"
            )
            following = self._sample_device(cache.last_logits, temperature, top_k, top_p)
            following_handoff = self._begin_readback(following)
            token = self._finish_readback(handoff)
            yield token
            if token in stops:
                cache.length = checkpoint
                break
            token_device, handoff = following, following_handoff

    def _generate_serially(
        self,
        cache: TorchKVCache,
        next_logits: Any,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int | None,
    ) -> Iterator[int]:
        """Decode on a device that executes as it is called.

        The lookahead in :meth:`_generate_pipelined` buys nothing here — there is no
        asynchronous queue for the host's launch cost to hide behind — and it costs
        the token a full step of latency plus, when generation stops, one forward
        pass whose rows are discarded.  So this yields each token as soon as it is
        sampled.  The arithmetic, the sampling and the stop rule are the same, so the
        token sequence is the same.
        """
        stops = stop_token_set(eos_token_id)
        logits = next_logits
        for _ in range(int(max_tokens)):
            token_device = self._sample_device(logits, temperature, top_k, top_p)
            token = int(token_device.item())
            yield token
            if token in stops:
                break
            _, cache = self._forward_device(
                token_device.reshape(1), cache, logits="last"
            )
            logits = cache.last_logits

    def _generate_constrained(
        self,
        cache: TorchKVCache,
        next_logits: Any,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int | None,
        grammar_session: Any,
    ) -> Iterator[int]:
        """Decode under a grammar FSM, which cannot run ahead.

        The FSM has to observe each accepted token on the host before it can
        compute the mask for the next step, so this path keeps the strict
        sample-then-advance ordering and forgoes the lookahead above.
        """
        torch = self.torch
        stops = stop_token_set(eos_token_id)
        for _ in range(int(max_tokens)):
            mask = grammar_session.get_token_mask()
            if len(mask) * 8 < int(next_logits.numel()):
                raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
            allowed = torch.tensor(
                [
                    (mask[index // 8] & (1 << (index % 8))) != 0
                    for index in range(int(next_logits.numel()))
                ],
                dtype=torch.bool,
                device=next_logits.device,
            )
            if not bool(torch.any(allowed).item()):
                raise ValueError("Grammar FSM has no valid next token")
            next_logits = next_logits.masked_fill(
                ~allowed, -torch_inf(torch, next_logits.dtype)
            )
            token = self._sample(next_logits, temperature, top_k, top_p)
            if grammar_session.advance(token) < 0:
                raise ValueError(
                    "The portable PyTorch executor produced a token rejected by the grammar FSM"
                )
            yield token
            if getattr(grammar_session, "is_accepting", lambda: False)():
                break
            if token in stops:
                break
            _, cache = self._forward_device(
                torch.tensor([token], dtype=torch.long, device=self.device),
                cache,
                logits="last",
            )
            next_logits = cache.last_logits

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

    # ── Batched execution ───────────────────────────────────────────────────
    #
    # These are the explicitly-batched entry points.  They are separate from
    # ``forward``/``generate`` rather than replacing them because the
    # single-sequence signatures are public and return rank-1 results; a caller
    # passing a ``(1, seq)`` array to ``forward`` has always meant one sequence,
    # and still does.

    def _convert_layers(self) -> list[dict[str, Any]]:
        """Upload every layer, freeing each host copy as soon as the device has it.

        Converting all layers and *then* releasing the host set means the load's
        high-water mark holds both copies in full: for a 3.8B model at FP32 on the
        host and FP16 on the device that is ~15 GiB plus ~7.6 GiB at the same
        instant, and the peak is what decides whether a load fits. Releasing layer
        ``i`` before layer ``i+1`` is uploaded replaces that with one shrinking copy
        beside one growing copy, so the peak is the size of the larger set plus a
        single layer rather than the sum of both.

        It is the same operation :meth:`release_host_weights` performs and rests on
        the same three properties, plus one specific to doing it early: nothing
        between a layer's conversion and the end of ``__init__`` reads that layer's
        host arrays again. Placement planning has already run by then -- it profiles
        the *CPU* engine, before this one exists -- and the decode path reads device
        tensors exclusively.

        Skipped, not attempted-and-recovered, in the three cases where it would be
        wrong: when the operator asked to keep the host set, when the device tensors
        *alias* the host arrays (CPU at FP32, where there is no second copy and
        dropping the reference would leave the executor pointing at storage nothing
        owns), and when two entries in the layer list are the same object, where
        freeing one would empty another that has not been uploaded yet.
        """
        layers = list(self.weights.layers)
        distinct = len({id(layer) for layer in layers}) == len(layers)
        stream = (
            distinct
            and host_weights_reclaimable()
            and not self.device_tensors_alias_host()
        )
        converted: list[Any] = []
        for layer in layers:
            converted.append(self._convert_layer(layer))
            if stream:
                self.host_bytes_streamed += self._drop_host_layer(layer)
        return converted

    def _drop_host_layer(self, layer: Any) -> int:
        """Free one layer's bulk host matrices; return the bytes reclaimed."""
        freed = 0
        seen: set[int] = set()

        def drop(owner: Any, name: str) -> None:
            nonlocal freed
            value = getattr(owner, name, None)
            if not isinstance(value, np.ndarray):
                return
            if id(value) not in seen:
                seen.add(id(value))
                freed += int(value.nbytes)
            try:
                setattr(owner, name, None)
            except (AttributeError, TypeError):  # frozen or slotted container
                pass

        for name in _BULK_LAYER_ARRAYS:
            drop(layer, name)
        for expert in getattr(layer, "experts", None) or []:
            for name in _BULK_EXPERT_ARRAYS:
                drop(expert, name)
        return freed

    def device_tensors_alias_host(self) -> bool:
        """Whether the device weights share storage with the host arrays.

        ``torch.as_tensor`` on a host array whose dtype already matches returns a
        *view*: on a CPU device at FP32 the executor's weights and the loader's
        arrays are the same memory. Distinguishing that from a genuine second copy
        is what makes :meth:`release_host_weights` safe to call unconditionally —
        where the two alias, there is nothing duplicated to free, and dropping the
        reference would leave the executor pointing at storage nothing owns.

        A *sharded* executor holds a list of per-device slices rather than one
        tensor. A slice is always a copy, so it can never alias, and asking it for a
        data pointer would raise instead of answering. The shape of the attribute is
        therefore inspected rather than assumed.
        """
        source = getattr(self.weights, "embedding", None)
        if source is None or not isinstance(source, np.ndarray):
            return False
        resident = getattr(self, "embedding", None)
        pointer = getattr(resident, "data_ptr", None)
        if not callable(pointer):
            # A sequence of shards, or no resident embedding at all: nothing aliases.
            return False
        try:
            host_pointer = source.__array_interface__["data"][0]
        except (AttributeError, KeyError, TypeError):
            return False
        return int(pointer()) == int(host_pointer)

    def release_host_weights(self) -> int:
        """Free host-resident weight matrices once the device owns its own copy.

        Returns the number of bytes released.

        The AEG loader materializes every weight as a host FP32 array, and the
        executor then materializes a device copy — on CUDA usually at FP16. Both
        stay live for the executor's whole lifetime, so a model costs its parameter
        count *twice*: once on the accelerator where it executes, and once in host
        RAM where nothing reads it again. Measured on a real Qwen3-0.6B AEG, the host
        set is 2.80 GiB over 751.6M elements at 4 bytes each
        (``scripts/profile_host_memory.py``), which matches the host-RSS gap the
        benchmark reports against Transformers.

        That cost is linear in parameter count, so it is not a small-model curiosity:
        the same ratio at 70B is ~280 GiB of host RAM, and at that point it is the
        difference between a model loading and not loading. Releasing it is therefore
        a scaling fix rather than a tidy-up.

        Safety comes from three properties rather than from a flag:

        * It is a no-op when the device tensors *alias* the host arrays
          (:meth:`device_tensors_alias_host`), which is the CPU-at-FP32 case, so a
          CPU-hosted engine and a promoted batched executor are untouched.
        * Only bulk matrices are dropped. Normalization weights, biases, routers and
          every scalar of the execution-numerics contract stay resident, because
          those are read after load and are metadata-scale.
        * The decode path reads device tensors exclusively. Nothing in
          ``_forward_device`` touches ``self.weights``.

        ``host_weights_released`` records that this happened so a caller that does
        need the host set can detect it rather than meeting a ``None``.

        Most of the per-layer work has usually already happened during the upload
        (:meth:`_convert_layers`), which is what keeps the load's *peak* down. This
        sweep is still the thing that frees the model-level matrices and the
        authority on whether the host set is gone, so it remains the entry point a
        caller uses.
        """
        if self.host_weights_released:
            return 0
        if self.device_tensors_alias_host():
            # Nothing is duplicated: the "host" arrays *are* the device tensors.
            return 0
        # Layers may already have been freed during the upload itself; see
        # ``_convert_layers``.  Those bytes are counted here so the number reported
        # is the host set this load reclaimed, not just the remainder.
        freed = int(getattr(self, "host_bytes_streamed", 0) or 0)
        seen: set[int] = set()

        def drop(owner: Any, name: str) -> None:
            nonlocal freed
            value = getattr(owner, name, None)
            if not isinstance(value, np.ndarray):
                return
            if id(value) not in seen:
                seen.add(id(value))
                freed += int(value.nbytes)
            try:
                setattr(owner, name, None)
            except (AttributeError, TypeError):  # frozen or slotted container
                pass

        for name in _BULK_MODEL_ARRAYS:
            drop(self.weights, name)
        for layer in getattr(self.weights, "layers", None) or []:
            for name in _BULK_LAYER_ARRAYS:
                drop(layer, name)
            for expert in getattr(layer, "experts", None) or []:
                for name in _BULK_EXPERT_ARRAYS:
                    drop(expert, name)
        self.host_weights_released = True
        return freed

    def memory_census(self) -> dict[str, Any]:
        """Account for every byte this engine holds, on both sides of the bus.

        Reported rather than inferred, because the two obvious proxies both lie. Host
        RSS counts mapped artifact pages and never shrinks predictably after a free,
        and PyTorch's ``memory_allocated`` counts only what the caching allocator
        currently hands out -- not the segments it is still holding, which is what a
        capacity check and a peak report actually see. Byte-accounting the tensors
        directly, deduplicated by storage pointer so an aliased ``lm_head`` is counted
        once, is the only reading that attributes a load's footprint to its causes.

        Deduplication by ``data_ptr`` is the point of the exercise as much as the
        total: two entries sharing one pointer is the observable difference between a
        tied head that was aliased and one that was copied.
        """
        torch = self.torch
        host_seen: set[int] = set()
        host_bytes = 0

        def host_add(owner: Any, name: str) -> None:
            nonlocal host_bytes
            value = getattr(owner, name, None)
            if isinstance(value, np.ndarray) and id(value) not in host_seen:
                host_seen.add(id(value))
                host_bytes += int(value.nbytes)

        weights = getattr(self, "weights", None)
        if weights is not None:
            for name in _BULK_MODEL_ARRAYS:
                host_add(weights, name)
            for layer in getattr(weights, "layers", None) or []:
                for name in _BULK_LAYER_ARRAYS:
                    host_add(layer, name)
                for expert in getattr(layer, "experts", None) or []:
                    for name in _BULK_EXPERT_ARRAYS:
                        host_add(expert, name)

        device_seen: set[int] = set()
        device_bytes = 0

        def device_add(value: Any) -> None:
            nonlocal device_bytes
            if not isinstance(value, torch.Tensor):
                return
            try:
                storage = value.untyped_storage()
                pointer, size = storage.data_ptr(), int(storage.nbytes())
            except Exception:  # noqa: BLE001 - a meta or fake tensor has no storage
                return
            if pointer in device_seen:
                return
            device_seen.add(pointer)
            device_bytes += size

        def device_walk(value: Any, depth: int = 0) -> None:
            if isinstance(value, torch.Tensor):
                device_add(value)
            elif depth < 4 and isinstance(value, dict):
                for item in value.values():
                    device_walk(item, depth + 1)
            elif depth < 4 and isinstance(value, (list, tuple)):
                for item in value:
                    device_walk(item, depth + 1)

        for value in vars(self).values():
            device_walk(value)

        census: dict[str, Any] = {
            "device": self.device,
            "compute_dtype": str(self.compute_dtype),
            "host_weight_bytes": host_bytes,
            "host_weights_released": bool(self.host_weights_released),
            "device_weight_bytes": device_bytes,
            "device_storages": len(device_seen),
            "aliased": self.device_tensors_alias_host(),
        }
        kind = str(self.device or "").split(":")[0]
        if kind == "cuda":
            try:
                if torch.cuda.is_available():
                    index = torch.device(self.device).index or 0
                    census.update(
                        torch_allocated_bytes=int(torch.cuda.memory_allocated(index)),
                        torch_reserved_bytes=int(torch.cuda.memory_reserved(index)),
                        torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
                        torch_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(index)),
                    )
            except Exception as exc:  # noqa: BLE001 - a census must never fail a load
                census["torch_error"] = str(exc)
        try:
            import psutil

            census["host_rss_bytes"] = int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - psutil is an optional benchmark dependency
            pass
        return census

    def supports_batch(self, batch_size: int = 1) -> bool:
        """Whether this executor can run ``batch_size`` sequences in one pass.

        There is no architectural bound.  The batch axis is a property of a call,
        not of the artifact — the AEG IR already declares it dynamic and the
        portable path emits no shape-specialized kernel — so the only real limit
        is device memory for the KV cache.
        """
        return int(batch_size) >= 1

    @property
    def max_batch_size(self) -> int | None:
        """``None``: no compiled-in bound, device memory is the only limit."""
        return None

    def _new_batched_cache(
        self,
        *,
        batch_size: int | None = None,
        reserve: int = 0,
        packed: PackedBatch | None = None,
    ) -> BatchedKVCache:
        """Allocate batched KV state, either from a packed batch or bare.

        Built from a :class:`PackedBatch` the cache inherits that batch's pad
        counts and validity mask.  Built bare (a decode-only continuation) it
        assumes no padding, which is the correct reading of "no layout given".
        """
        torch = self.torch
        if packed is not None:
            batch = packed.batch_size
            pad_counts = torch.as_tensor(
                packed.layout.pad_counts, device=self.device, dtype=torch.long
            ).reshape(batch, 1)
            layout: BatchLayout | None = packed.layout
            live = None
            if not layout.is_uniform:
                # Only a padded batch materializes a mask.  A uniform batch leaves
                # it None so prefill keeps ``is_causal=True`` and decode passes no
                # mask, matching the single-sequence path kernel for kernel.
                capacity = max(int(reserve), layout.padded_length)
                live = torch.zeros(
                    (batch, capacity), dtype=torch.bool, device=self.device
                )
                live[:, : layout.padded_length] = torch.as_tensor(
                    packed.live, device=self.device
                )
        elif batch_size is not None:
            batch = int(batch_size)
            pad_counts = torch.zeros((batch, 1), dtype=torch.long, device=self.device)
            layout = None
            live = None
        else:
            raise ValueError("a batched cache needs either a batch size or a packed batch")
        return BatchedKVCache(
            keys=[None] * self.num_layers,
            values=[None] * self.num_layers,
            pad_counts=pad_counts,
            batch_size=batch,
            live=live,
            layout=layout,
            finished=torch.zeros(batch, dtype=torch.bool, device=self.device),
        )

    def _extend_live(self, cache: BatchedKVCache, total: int) -> None:
        """Mark the slots about to be written as holding real tokens.

        A no-op for an unpadded batch: ``live is None`` already means "every slot
        is real", and keeping it None is what preserves the fused attention path.
        """
        if cache.live is None:
            return
        torch = self.torch
        capacity = int(cache.live.shape[1])
        if capacity < total:
            grown = torch.zeros(
                (cache.batch_size, max(total, capacity * 2)),
                dtype=torch.bool,
                device=cache.live.device,
            )
            grown[:, :capacity] = cache.live
            cache.live = grown
        cache.live[:, cache.length : total] = True

    def forward_batch(
        self,
        sequences: Any,
        *,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        reserve: int = 0,
    ) -> tuple[Any, BatchedKVCache]:
        """Prefill a batch of independent sequences in one pass.

        ``sequences`` is a list of rank-1 id arrays (or a rank-2 array of equal-
        length rows).  Returns ``(logits, cache)`` with logits shaped
        ``(batch, padded_seq, vocab)``.  Because rows are right-aligned,
        ``logits[:, -1]`` is every row's final *real* position whatever its length
        — which is the whole reason for left padding.
        """
        packed = pack_left_padded(sequences, pad_token_id=pad_token_id)
        reserve = max(int(reserve), packed.padded_length)
        cache = self._new_batched_cache(packed=packed, reserve=reserve)
        return self._forward_device(
            packed.token_ids, cache, validate_ids=True, reserve=reserve, batched=True
        )

    def generate_batch(
        self,
        prompts: Any,
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        **_: Any,
    ) -> list[list[int]]:
        """Decode several independent sequences concurrently.

        One prefill pass over the padded batch, then one forward pass per decode
        step carrying all rows.  Returns one token list per input prompt, in input
        order.

        Per-row stopping: a row that emits ``eos_token_id`` stops *recording*, and
        the batch keeps its width until every row has stopped.  Continuing to
        compute a finished row cannot perturb the others — rows share no state —
        so this costs some arithmetic on skewed batches and buys a decode loop with
        no mid-flight reshaping.

        Note on determinism: with ``temperature > 0`` all rows are drawn in one
        ``multinomial`` call, so a batched sampled run is not token-identical to N
        separately-seeded single-sequence runs.  That is true of every batched
        runtime.  Greedy decoding is comparable, and is what the equivalence tests
        assert.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        torch = self.torch
        packed = pack_left_padded(prompts, pad_token_id=pad_token_id)
        batch = packed.batch_size
        # The final length is known, so the KV cache is sized once here rather
        # than doubling during decode.
        reserve = packed.padded_length + int(max_tokens)
        cache = self._new_batched_cache(packed=packed, reserve=reserve)
        _, cache = self._forward_device(
            packed.token_ids, cache, validate_ids=True, reserve=reserve,
            batched=True, logits="last",
        )
        outputs: list[list[int]] = [[] for _ in range(batch)]
        finished = [False] * batch
        stops = stop_token_set(eos_token_id)
        tokens = self._sample_device(cache.last_logits, temperature, top_k, top_p)
        for _ in range(int(max_tokens)):
            # Queue the next step before reading this one back, for the reason
            # documented on ``_generate_pipelined``: the host's launch cost then
            # overlaps device execution instead of following it.
            self._extend_live(cache, cache.length + 1)
            _, cache = self._forward_device(
                tokens.reshape(batch, 1), cache, reserve=reserve,
                batched=True, logits="last",
            )
            following = self._sample_device(cache.last_logits, temperature, top_k, top_p)
            # One host synchronization per step — the same count the
            # single-sequence path pays — but it now covers the whole batch.
            row_tokens = tokens.tolist()
            for index in range(batch):
                if finished[index]:
                    continue
                token = int(row_tokens[index])
                outputs[index].append(token)
                if token in stops:
                    finished[index] = True
            if all(finished):
                break
            tokens = following
        cache.finished = torch.as_tensor(finished, dtype=torch.bool, device=self.device)
        return outputs

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

    _ROPE_HEADROOM = 512
    """Rows reserved past the current length, so decode does not rebuild per token."""

    def __init__(self, hybrid_engine: Any, device: str, devices: list[str] | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by backend
            raise RuntimeError("PyTorch is required for portable hybrid execution") from exc
        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.devices = [_resolve_device(torch, value) for value in (devices or [device])]
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
        self.lm_head = self._tied_tensor(self.weights.lm_head, self.weights.embedding, self.embedding)
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
        self._rope_inv_freq: "np.ndarray | None" = None
        self.rope_attention_scaling = 1.0

    def _tensor(self, value: Any, device: Any | None = None) -> Any:
        return self.torch.as_tensor(np.asarray(value, dtype=np.float32), device=device or self.device)

    def _optional_tensor(self, value: Any | None, device: Any | None = None) -> Any | None:
        return None if value is None else self._tensor(value, device)

    def _tied_tensor(self, value: Any, tied_to: Any, resident: Any) -> Any:
        """Materialize ``value``, reusing ``resident`` when it is the same matrix.

        The hybrid executor is a standalone implementation of the same contract, so
        it carries its own copy of this rule for the reason
        :meth:`TorchAEGEngine._tied_tensor` gives: in a tied model the output
        projection *is* the input embedding, and uploading that matrix twice
        doubles the largest allocation in a small model to store bytes both sites
        only read.  Identity is decided on the host array, which the loader aliases
        only after proving the stored bytes match, so an ``lm_head`` that really is
        a distinct matrix still gets its own upload.
        """
        if value is tied_to and resident is not None:
            return resident
        return self._tensor(value)

    def _rope_scaling_spec(self) -> Any:
        """The parsed rotary transform, or ``None`` for an unscaled checkpoint.

        Parsed once and cached: the mapping is immutable for the life of the engine,
        and the parse is a handful of dictionary reads that would otherwise land on
        every attention layer of every step.
        """
        cached = getattr(self, "_rope_spec_cache", False)
        if cached is not False:
            return cached
        from aether.runtime.rope_scaling import parse_rope_scaling

        declared = int(getattr(self.weights, "context_length", 0) or 0) or None
        spec = parse_rope_scaling(
            getattr(self.weights, "rope_scaling", None),
            context_length=declared,
            original_context_length=getattr(
                self.weights, "original_context_length", None
            ),
        )
        self._rope_spec_cache = spec
        return spec

    def _rope_is_length_sensitive(self) -> bool:
        """Whether the frequencies change with the length being evaluated."""
        spec = self._rope_scaling_spec()
        return spec is not None and spec.rope_type in {"dynamic", "longrope"}

    def _ensure_rope(self, required: int, device: Any | None = None) -> None:
        """Build rotary tables for ``required`` positions on ``device``.

        The frequencies come from :mod:`aether.runtime.rope_scaling` rather than from
        ``theta ** -exponent`` computed here.  A hybrid checkpoint declares its
        rotary transform in the same place a dense one does, and deriving the plain
        frequencies instead means a context-extended model rotates at the unscaled
        rate for every attention layer in the schedule — the transform is applied to
        the *slow* dimensions, so the failure grows with the position and is silent.

        ``required`` is both the table height and the length being evaluated here,
        because this executor advances one position at a time: the caller passes
        ``past + 1``, which is the highest position in the step plus one — the same
        quantity the reference derives from ``max(position_ids) + 1``.  For the two
        length-sensitive schemes a taller table is therefore not evidence that its
        frequencies are the right ones, so a scheme that can switch tables re-derives
        and compares rather than accepting the height.
        """
        from aether.runtime.rope_scaling import scaled_inverse_frequencies

        torch = self.torch
        device = device or self.device
        tall_enough = (
            self._cos is not None
            and int(self._cos.shape[0]) >= required
            and self._cos.device == device
        )
        if tall_enough and not self._rope_is_length_sensitive():
            return
        inverse, attention = scaled_inverse_frequencies(
            float(self.weights.rope_theta),
            self.head_dim,
            self._rope_scaling_spec(),
            sequence_length=int(required),
        )
        if tall_enough:
            current = getattr(self, "_rope_inv_freq", None)
            if current is not None and np.array_equal(current, inverse):
                return
        self._rope_inv_freq = inverse
        self.rope_attention_scaling = float(attention)
        # Headroom, so a table is not rebuilt once per token.  The old code sized the
        # table at exactly ``required``, and ``required`` grows by one every step, so
        # every decode step rebuilt the whole thing.  A slab amortizes that without a
        # doubling policy's runaway; it is height only, never the derivation length.
        height = max(
            int(required) + self._ROPE_HEADROOM,
            int(self._cos.shape[0]) if tall_enough else 0,
        )
        positions = torch.arange(height, device=device, dtype=torch.float64)[:, None]
        frequencies = torch.as_tensor(inverse, device=device, dtype=torch.float64)
        angles = positions * frequencies[None, :]
        cos, sin = torch.cos(angles), torch.sin(angles)
        if attention != 1.0:
            # YaRN and LongRoPE correct the attention temperature by scaling the
            # rotation itself; folding it into the tables keeps the hot path free of
            # a second multiply.
            cos, sin = cos * attention, sin * attention
        self._cos = cos.to(torch.float32)
        self._sin = sin.to(torch.float32)

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
                    # The reference hybrid executor refuses this by name; a torch
                    # slice past the last row returns an *empty* tensor instead, so
                    # without the check the failure surfaces as a broadcast error
                    # about shapes that names neither the table nor the position.
                    if past >= int(self.position_embedding.shape[0]):
                        raise ValueError(
                            f"sequence end {past + 1} exceeds learned position "
                            f"embedding capacity {int(self.position_embedding.shape[0])}"
                        )
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
        # Normalized once per request: a checkpoint may declare several stop
        # ids (an instruct model's turn delimiter is often not its eos_token),
        # and every engine must agree on what stopping means.
        stops = stop_token_set(eos_token_id)
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
            if token in stops:
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
