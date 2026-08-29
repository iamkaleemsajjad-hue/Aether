"""Exact model facts, derived from the graph rather than guessed from a name.

This is Aether's structural advantage over a serving runtime that only sees a
checkpoint at startup: the compiler already holds every tensor shape, the operator
list and the KV geometry, so the terms the planner needs most are *exact* rather
than estimated.  Weight bytes have zero error here, which is what lets the
feasibility lane spend its whole uncertainty budget on the one term that deserves
it — the transient peak.

Three quantities are computed and each is used by a different roof:

``weight_bytes``      the bandwidth roof, because decode streams every weight per token
``ops_per_token``     the dispatch roof, because host cost is per graph operation
``flops_per_token``   the compute roof

and one more feeds the residual: ``kv_bytes_per_token``, which turns a KV budget
into a token capacity.

The activation estimate is a liveness bound — the maximum bytes live across any cut
of the layer graph — not a sum over all tensors.  A sum over-predicts by the depth
of the graph and would waste most of a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ModelProfile",
    "profile_from_architecture",
    "profile_from_engine",
    "profile_from_manifest",
]

_FP32 = 4
"""Softmax and reduction intermediates are materialised in fp32 by every backend
Aether targets, so attention score buffers are sized at four bytes regardless of
the model's residency precision."""


@dataclass(frozen=True)
class ModelProfile:
    """Everything the planner needs to know about one compiled model."""

    model_id: str
    family_kind: str = "decoder"
    """``decoder`` | ``encoder`` | ``encoder_decoder`` | ``ssm_hybrid`` | ``multimodal``."""

    layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0

    weight_dtype_bytes: float = 2.0
    kv_dtype_bytes: float = 2.0

    embedding_bytes: int = 0
    lm_head_bytes: int = 0
    per_layer_bytes: int = 0
    persistent_bytes: int = 0
    """Rope tables, logit buffers, and other allocations that live for the run."""

    ops_per_layer: int = 0
    ops_fixed: int = 0
    """Operations outside the layer stack: embedding lookup, final norm, LM head."""

    gated_ffn: bool = True
    flash_attention: bool = True
    is_moe: bool = False
    num_experts: int = 0
    experts_per_token: int = 0

    supports_tensor_parallel: bool = True
    supports_pipeline_parallel: bool = True
    restriction: str = ""
    """Why a parallelism form is unavailable, when one is. Empty when unrestricted."""

    exact_weights: bool = False
    """True when weight bytes were summed from materialised tensors rather than
    computed from declared dimensions."""

    extra: dict[str, Any] = field(default_factory=dict)

    # ── totals ────────────────────────────────────────────────────────────────

    @property
    def weight_bytes(self) -> int:
        """Total resident weight bytes for the whole model."""
        return self.embedding_bytes + self.lm_head_bytes + self.per_layer_bytes * self.layers

    @property
    def params(self) -> int:
        """Parameter count implied by the byte totals and residency precision."""
        if self.weight_dtype_bytes <= 0:
            return 0
        return int(self.weight_bytes / self.weight_dtype_bytes)

    @property
    def ops_per_token(self) -> int:
        """Host-dispatched graph operations for one decode step."""
        return self.ops_per_layer * self.layers + self.ops_fixed

    @property
    def kv_bytes_per_token(self) -> int:
        """KV bytes one token adds, across the whole model.

        Both K and V, every layer, every KV head. This is the divisor that turns a
        byte budget into a token capacity, so an error here is an error in every
        capacity number the planner reports.
        """
        return int(
            2 * self.layers * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes
        )

    @property
    def flops_per_token_decode(self) -> int:
        """Two FLOPs per parameter per token — one multiply, one accumulate."""
        return 2 * self.params

    @property
    def tp_ceiling_for_kv(self) -> int:
        """Largest TP degree that still shards the KV cache.

        Beyond the KV-head count the cache has to replicate, which destroys the
        capacity benefit that motivated the split. A GQA model with four KV heads
        has a hard ceiling of four however many devices are present.
        """
        return max(1, self.num_kv_heads)

    def attention_flops(self, batch: int, context: int) -> int:
        """FLOPs for the attention score and context matmuls at one step."""
        return int(4 * batch * self.num_heads * self.head_dim * context)

    # ── activation liveness ───────────────────────────────────────────────────

    def activation_bytes(
        self,
        batch: int,
        step_tokens: int,
        context: int,
        *,
        tp_degree: int = 1,
        all_logits: bool = False,
    ) -> int:
        """Peak live activation bytes for one layer step, at this workload.

        A *maximum over cuts*, not a sum: within one layer the attention buffers and
        the FFN intermediates are not simultaneously live, so the peak is whichever
        of them is larger, plus the residual stream that spans the whole layer.
        Summing every intermediate would over-predict by roughly the depth of the
        block and hand most of the device back for no reason.

        ``tp_degree`` divides the head and intermediate dimensions, because a
        tensor-parallel rank materialises only its own shard of both.
        """
        batch = max(1, batch)
        step_tokens = max(1, step_tokens)
        context = max(step_tokens, context)
        shards = max(1, tp_degree)
        dtype = self.weight_dtype_bytes

        local_heads = max(1, self.num_heads // shards)
        local_inter = max(1, self.intermediate_size // shards)

        # Residual stream: input and output of the block are both live.
        residual = int(2 * batch * step_tokens * self.hidden_size * dtype)

        # Attention. With a flash-style kernel the score matrix is never
        # materialised, which is the difference between O(S) and O(S²) memory and
        # therefore the difference between a long context fitting and not.
        projections = int(3 * batch * step_tokens * local_heads * self.head_dim * dtype)
        if self.flash_attention:
            attention = projections + int(batch * step_tokens * local_heads * self.head_dim * dtype)
        else:
            scores = int(batch * local_heads * step_tokens * context * _FP32)
            attention = projections + scores

        # Gated FFN holds the gate and up projections at once before the product.
        branches = 2 if self.gated_ffn else 1
        ffn = int(branches * batch * step_tokens * local_inter * dtype)
        if self.is_moe and self.experts_per_token > 0:
            # Routed experts process disjoint token subsets, so the live width is
            # the routed fraction rather than the full batch, plus the router logits.
            ffn = int(ffn * min(1.0, self.experts_per_token / max(1, self.num_experts)) + ffn / branches)
            ffn += int(batch * step_tokens * self.num_experts * _FP32)

        logit_tokens = step_tokens if all_logits else 1
        logits = int(batch * logit_tokens * self.vocab_size * _FP32)

        return residual + max(attention, ffn, logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family_kind": self.family_kind,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "params_billion": round(self.params / 1e9, 3),
            "weight_gib": round(self.weight_bytes / 1024 ** 3, 3),
            "weight_dtype_bytes": self.weight_dtype_bytes,
            "kv_dtype_bytes": self.kv_dtype_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "ops_per_token": self.ops_per_token,
            "is_moe": self.is_moe,
            "tp_ceiling_for_kv": self.tp_ceiling_for_kv,
            "supports_tensor_parallel": self.supports_tensor_parallel,
            "supports_pipeline_parallel": self.supports_pipeline_parallel,
            "restriction": self.restriction,
            "exact_weights": self.exact_weights,
        }


# ── operation counting ────────────────────────────────────────────────────────

def _ops_per_layer(
    *,
    gated_ffn: bool,
    qk_norm: bool,
    parallel_residual: bool,
    norm_placement: str,
    is_moe: bool,
    experts_per_token: int,
) -> int:
    """Count the host-dispatched operations in one decoder block.

    The dispatch roof is ``ops × t_dispatch``, so this count is load-bearing: it is
    the term that explains why splitting a small model across two devices makes it
    slower. Every adjustment below corresponds to a real structural difference that
    adds or removes kernel launches.
    """
    # norm, q, k, v, rope_q, rope_k, cache_append, attn, o_proj, residual,
    # post_norm, gate, act, down, residual
    ops = 15
    if gated_ffn:
        ops += 2          # up projection and the gate·up product
    if qk_norm:
        ops += 2
    if parallel_residual:
        ops -= 1          # one shared norm and one fused residual add
    if norm_placement in ("sandwich", "sandwich_glm"):
        ops += 2
    elif norm_placement == "post":
        ops += 0
    if is_moe:
        # router + top-k, then a gate/act/down triple per activated expert in place
        # of the single dense FFN, plus the weighted scatter-add.
        ops += 2 + max(0, experts_per_token - 1) * (5 if gated_ffn else 3) + 1
    return ops


def _family_kind(architecture: Any) -> str:
    if getattr(architecture, "is_encoder_decoder", False):
        return "encoder_decoder"
    if getattr(architecture, "is_encoder", False):
        return "encoder"
    if getattr(architecture, "ssm_variant", None):
        return "ssm_hybrid"
    if getattr(architecture, "is_multimodal", False):
        return "multimodal"
    return "decoder"


def profile_from_architecture(
    architecture: Any,
    *,
    model_id: str = "",
    weight_dtype_bytes: float = 2.0,
    kv_dtype_bytes: float | None = None,
    flash_attention: bool = True,
) -> ModelProfile:
    """Build a profile from a declared architecture, analytically.

    Weight bytes are computed from the tensor shapes the architecture implies —
    never from ``params_billion``, which is frequently zero on the config-driven
    detection path and would silently produce a weightless model.

    Args:
        architecture: A :class:`~aether.core.types.ModelArchitecture`.
        weight_dtype_bytes: Bytes per stored parameter at accelerator residency.
        kv_dtype_bytes: Bytes per stored KV element. Defaults to the weight dtype.
        flash_attention: Whether the executor materialises the score matrix.
    """
    hidden = int(getattr(architecture, "hidden_size", 0) or 0)
    layers = int(getattr(architecture, "layers", 0) or 0)
    heads = int(getattr(architecture, "num_attention_heads", 0) or 0) or 1
    kv_heads = int(getattr(architecture, "num_kv_heads", 0) or 0) or heads
    head_dim = int(getattr(architecture, "head_dim", 0) or 0) or max(1, hidden // heads)
    vocab = int(getattr(architecture, "vocab_size", 0) or 0)
    intermediate = int(getattr(architecture, "intermediate_size", 0) or 0) or hidden * 4
    ffn_type = str(getattr(architecture, "ffn_type", "SwiGLU") or "SwiGLU")
    gated = "glu" in ffn_type.lower()
    norm_placement = str(getattr(architecture, "norm_placement", "pre") or "pre")
    is_moe = bool(getattr(architecture, "is_moe", False))
    num_experts = int(getattr(architecture, "num_experts", 0) or 0)
    experts_per_token = int(getattr(architecture, "num_activated_experts", 0) or 0)
    context = int(getattr(architecture, "context_length", 4096) or 4096)

    attention_params = hidden * head_dim * (2 * heads + 2 * kv_heads)
    dense_ffn_params = (3 if gated else 2) * hidden * intermediate
    if is_moe and num_experts > 0:
        # Every expert is resident even though only a few run per token, which is
        # exactly why MoE is a memory problem before it is a compute problem.
        ffn_params = num_experts * dense_ffn_params + hidden * num_experts
    else:
        ffn_params = dense_ffn_params
    norm_count = 4 if norm_placement.startswith("sandwich") else 2
    per_layer = attention_params + ffn_params + norm_count * hidden

    tied = bool(getattr(architecture, "tie_word_embeddings", True))
    embedding_bytes = int(vocab * hidden * weight_dtype_bytes)
    lm_head_bytes = 0 if tied else embedding_bytes

    # Rotary tables are sized by the artifact's declared context and stored in
    # fp32; they are persistent, not transient, and large enough to matter at
    # long context.
    rope_bytes = 0
    if str(getattr(architecture, "position_type", "RoPE")).upper() == "ROPE":
        rope_bytes = 2 * context * head_dim * _FP32

    kind = _family_kind(architecture)
    ssm = bool(getattr(architecture, "ssm_variant", None))
    return ModelProfile(
        model_id=model_id or str(getattr(architecture, "family", "model")),
        family_kind=kind,
        layers=layers,
        hidden_size=hidden,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate,
        vocab_size=vocab,
        weight_dtype_bytes=weight_dtype_bytes,
        kv_dtype_bytes=weight_dtype_bytes if kv_dtype_bytes is None else kv_dtype_bytes,
        embedding_bytes=embedding_bytes,
        lm_head_bytes=lm_head_bytes,
        per_layer_bytes=int(per_layer * weight_dtype_bytes),
        persistent_bytes=rope_bytes,
        ops_per_layer=_ops_per_layer(
            gated_ffn=gated,
            qk_norm=bool(getattr(architecture, "qk_norm", False)),
            parallel_residual=bool(getattr(architecture, "parallel_residual", False)),
            norm_placement=norm_placement,
            is_moe=is_moe,
            experts_per_token=experts_per_token,
        ),
        ops_fixed=4,
        gated_ffn=gated,
        flash_attention=flash_attention,
        is_moe=is_moe,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        supports_tensor_parallel=not ssm and kind in ("decoder", "encoder"),
        supports_pipeline_parallel=True,
        restriction=(
            "state-space layers carry recurrent state per layer, so a tensor split "
            "of the scan needs a state-exchange contract Aether does not implement; "
            "layer-wise placement only"
            if ssm else
            "" if kind in ("decoder", "encoder") else
            f"{kind} graphs have no verified sharded execution contract"
        ),
        exact_weights=False,
    )


def profile_from_engine(
    engine: Any,
    *,
    model_id: str = "",
    weight_dtype_bytes: float = 2.0,
    kv_dtype_bytes: float | None = None,
    flash_attention: bool = True,
    architecture: Any | None = None,
) -> ModelProfile:
    """Build a profile from a materialised engine, with exact weight bytes.

    Sums the real tensors the loader bound, so ``weight_bytes`` carries no error at
    all.  Everything the tensors cannot reveal — the operation count, the family
    restrictions — is taken from ``architecture`` when supplied, and otherwise
    reconstructed from the engine's own scalar metadata.
    """
    import numpy as np

    weights = engine.weights
    layers = list(getattr(weights, "layers", []) or [])
    hidden = int(getattr(engine, "hidden_size", 0) or 0)
    heads = int(getattr(engine, "num_heads", 0) or 0) or 1
    kv_heads = int(getattr(engine, "num_kv_heads", 0) or 0) or heads
    head_dim = int(getattr(engine, "head_dim", 0) or 0) or max(1, hidden // heads)

    def nbytes(tensor: Any) -> int:
        if tensor is None:
            return 0
        array = np.asarray(tensor)
        return int(array.size * weight_dtype_bytes)

    embedding_bytes = nbytes(getattr(weights, "embedding", None))
    lm_head_bytes = nbytes(getattr(weights, "lm_head", None))
    if hidden == 0 and getattr(weights, "embedding", None) is not None:
        hidden = int(np.asarray(weights.embedding).shape[-1])

    projection_names = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "qkv_proj", "router",
    )
    layer_total = 0
    gated = False
    is_moe = False
    experts_per_token = 0
    num_experts = 0
    for layer in layers:
        for name in projection_names:
            layer_total += nbytes(getattr(layer, name, None))
        for name in ("input_norm", "post_norm", "pre_ffn_norm", "post_ffn_norm"):
            layer_total += nbytes(getattr(layer, name, None))
        if getattr(layer, "up_proj", None) is not None:
            gated = True
        experts = list(getattr(layer, "experts", None) or [])
        if experts:
            is_moe = True
            num_experts = max(num_experts, len(experts))
            experts_per_token = max(
                experts_per_token, int(getattr(layer, "num_activated_experts", 1) or 1)
            )
            for expert in experts:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    layer_total += nbytes(getattr(expert, name, None))

    layer_count = len(layers) or int(getattr(architecture, "layers", 0) or 0)
    per_layer_bytes = layer_total // layer_count if layer_count else 0

    base = (
        profile_from_architecture(
            architecture,
            model_id=model_id,
            weight_dtype_bytes=weight_dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes,
            flash_attention=flash_attention,
        )
        if architecture is not None else None
    )

    intermediate = int(getattr(architecture, "intermediate_size", 0) or 0)
    if not intermediate and layers:
        gate = getattr(layers[0], "gate_proj", None)
        if gate is not None:
            intermediate = int(np.asarray(gate).shape[0])
    vocab = int(getattr(architecture, "vocab_size", 0) or 0)
    if not vocab and getattr(weights, "embedding", None) is not None:
        vocab = int(np.asarray(weights.embedding).shape[0])

    return ModelProfile(
        model_id=model_id or (base.model_id if base else "model"),
        family_kind=base.family_kind if base else "decoder",
        layers=layer_count,
        hidden_size=hidden,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate or hidden * 4,
        vocab_size=vocab,
        weight_dtype_bytes=weight_dtype_bytes,
        kv_dtype_bytes=weight_dtype_bytes if kv_dtype_bytes is None else kv_dtype_bytes,
        embedding_bytes=embedding_bytes,
        lm_head_bytes=lm_head_bytes,
        per_layer_bytes=per_layer_bytes,
        persistent_bytes=base.persistent_bytes if base else 0,
        ops_per_layer=(
            base.ops_per_layer if base
            else _ops_per_layer(
                gated_ffn=gated, qk_norm=False, parallel_residual=False,
                norm_placement="pre", is_moe=is_moe, experts_per_token=experts_per_token,
            )
        ),
        ops_fixed=4,
        gated_ffn=gated if base is None else base.gated_ffn,
        flash_attention=flash_attention,
        is_moe=is_moe or (base.is_moe if base else False),
        num_experts=num_experts or (base.num_experts if base else 0),
        experts_per_token=experts_per_token or (base.experts_per_token if base else 0),
        supports_tensor_parallel=base.supports_tensor_parallel if base else True,
        supports_pipeline_parallel=base.supports_pipeline_parallel if base else True,
        restriction=base.restriction if base else "",
        exact_weights=True,
    )


_ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "family", "params_billion", "layers", "hidden_size", "num_attention_heads",
    "num_kv_heads", "head_dim", "context_length", "vocab_size", "intermediate_size",
    "norm_eps", "rope_theta", "qk_norm", "parallel_residual", "is_moe", "is_encoder",
    "is_encoder_decoder", "is_multimodal", "num_experts", "num_activated_experts",
    "attention_type", "ssm_variant", "ffn_type", "norm_type", "position_type",
    "norm_placement", "tie_word_embeddings",
)


def profile_from_manifest(
    manifest: dict[str, Any],
    *,
    model_id: str = "",
    weight_dtype_bytes: float | None = None,
    kv_dtype_bytes: float | None = None,
) -> ModelProfile:
    """Build a profile from an AEG manifest, without loading any weights.

    This is the compile-time half of the split the design calls for: the artifact
    already records every dimension the planner needs, so a placement can be planned
    before a single tensor is materialised — which is what makes it possible to refuse
    an impossible workload *before* paying for the load.

    When ``weight_dtype_bytes`` is not given it is derived from the manifest's own
    recorded footprint, so a quantised artifact is planned at its real residency
    rather than at an assumed FP16.
    """
    from aether.core.types import ModelArchitecture

    declared = dict(manifest.get("architecture") or {})
    fields = {
        key: declared[key] for key in _ARCHITECTURE_FIELDS if key in declared
    }
    fields.setdefault("family", "generic_decoder_family")
    fields.setdefault("params_billion", 0.0)
    try:
        architecture = ModelArchitecture(**fields)
    except TypeError as exc:
        raise ValueError(f"AEG manifest architecture is not usable: {exc}") from exc

    identifier = model_id or str(manifest.get("model_id", "") or "aeg")
    dtype = weight_dtype_bytes if weight_dtype_bytes is not None else 2.0
    # The KV cache dtype is independent of weight residency: a 4-bit model normally
    # still caches in half precision, so inheriting the weight dtype would understate
    # the cache by 4x and hand back capacity the device does not have.
    kv_dtype = kv_dtype_bytes
    if kv_dtype is None:
        declared_kv = str(
            (manifest.get("runtime") or {}).get("kv_cache_dtype", "") or ""
        ).upper()
        kv_dtype = {"FP8": 1.0, "INT8": 1.0, "Q8_0": 1.0, "FP32": 4.0}.get(declared_kv, 2.0)
    profile = profile_from_architecture(
        architecture, model_id=identifier, weight_dtype_bytes=dtype,
        kv_dtype_bytes=kv_dtype,
    )
    if weight_dtype_bytes is not None or profile.params <= 0:
        return profile

    # The manifest records what the compiled artifact actually occupies. Dividing by
    # the analytic parameter count recovers the residency precision, which is how a
    # 4-bit artifact gets planned as a 4-bit artifact.
    recorded = manifest.get("memory_requirements") or {}
    compiled_gb = float(recorded.get("compiled_min_gb", 0.0) or 0.0)
    if compiled_gb <= 0:
        return profile
    observed = compiled_gb * (1024 ** 3) / profile.params
    if not 0.1 <= observed <= 4.5:
        logger.debug(
            "manifest footprint implies %.2f bytes/param, outside the plausible "
            "range; keeping the declared %.1f", observed, dtype,
        )
        return profile
    return profile_from_architecture(
        architecture, model_id=identifier,
        weight_dtype_bytes=round(observed, 4), kv_dtype_bytes=kv_dtype,
    )
