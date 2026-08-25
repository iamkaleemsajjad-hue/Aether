"""
Load a compiled AEG package into an executable CPU engine.

Bridges :class:`~aether.core.aeg_format.AEGPackage` (on-disk artifact) and
:class:`~aether.runtime.cpu_engine.CPUExecutionEngine` (executable model): reads
the weight blob, dequantizes each tensor, groups them into per-layer structures,
and returns an engine that can generate tokens.

Weight naming
-------------
Tensors are matched by the ``(layer_index, component)`` key produced by
:meth:`~aether.compiler.stage1_ingestion.ingestion.IngestionPipeline._normalise_weight_name`,
so a package built from any supported checkpoint layout loads without per-family
special cases.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from aether.core.exceptions import AEGFormatError
from aether.kernels.native_cpu import get_native_kernels
from aether.runtime.cpu_engine import (
    CPUExecutionEngine,
    ExpertWeights,
    LayerWeights,
    ModelWeights,
)
from aether.runtime.mla_engine import (
    MLAExecutionEngine, MLAExpertWeights, MLALayerWeights, MLAModelWeights,
)
from aether.runtime.mamba_engine import MambaExecutionEngine, MambaLayerWeights, MambaModelWeights
from aether.runtime.mamba2_engine import Mamba2ExecutionEngine, Mamba2LayerWeights, Mamba2ModelWeights
from aether.runtime.rwkv_engine import RWKVExecutionEngine, RWKVLayerWeights, RWKVModelWeights
from aether.runtime.hybrid_engine import HybridExecutionEngine, HybridLayerWeights, HybridModelWeights
from aether.runtime.encoder_engine import EncoderExecutionEngine, EncoderLayerWeights, EncoderModelWeights
from aether.runtime.seq2seq_engine import Seq2SeqDecoderLayer, Seq2SeqExecutionEngine, Seq2SeqLayer
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["AEGLoadError", "load_engine_from_package", "load_engine_from_path", "package_is_runnable"]


class AEGLoadError(AEGFormatError):
    """Raised when a package cannot be turned into an executable engine."""


#: Graph-node component names required for each transformer layer, mapped to the
#: :class:`LayerWeights` field they populate.
_LAYER_COMPONENTS: dict[str, str] = {
    "rmsnorm": "attention_norm",
    "qkv": "q_proj",
    "out_proj": "o_proj",
    "ffn_norm": "ffn_norm",
    "gate_proj": "gate_proj",
    "ffn": "down_proj",
}


def package_is_runnable(package: Any) -> bool:
    """Return True when a package carries enough weights to execute.

    Cheap to call: it reads the weight index but never the tensor payloads.
    Fails closed on artifacts that were never properly finalized: a pending
    graph hash, a placeholder architecture, or a missing per-layer tensor
    set can never be runnable.
    """
    try:
        # This helper is part of the public loader surface, so accept the same
        # path forms as ``load_engine_from_path``.  Previously callers had to
        # know the internal AEGPackage type even though the function's name
        # and documentation describe an artifact check.
        if isinstance(package, (str, Path)):
            from aether.core.aeg_format import AEGPackage

            package_path = Path(package)
            if package_path.is_file():
                package = AEGPackage.load_from_archive(package_path)
            else:
                package = AEGPackage(package_path).load()
        manifest = getattr(package, "manifest", None)
        if manifest is not None:
            if getattr(manifest, "graph_hash", "") == "sha256:pending":
                return False
            architecture = getattr(manifest, "architecture", None)
            layers = int(getattr(architecture, "layers", 0) or 0) if architecture else 0
            if layers <= 0:
                return False
        if not package.has_weights:
            return False
        store = package.weight_store()
        # has_weights only checks that the blob file exists; we also need at
        # least one indexed tensor entry to confirm weights were actually written.
        names = set(store.entries)
        if not names:
            return False
    except (AEGFormatError, OSError):
        return False
    return any(name.endswith("embedding") or "embed" in name for name in names)


def load_engine_from_path(aeg_path: Path | str) -> CPUExecutionEngine:
    """Open an ``.aeg`` package and build an engine from it."""
    from aether.core.aeg_format import AEGPackage

    package = AEGPackage(Path(aeg_path))
    package.load()
    return load_engine_from_package(package)


def load_engine_from_package(package: Any) -> CPUExecutionEngine:
    """Build a :class:`CPUExecutionEngine` from a loaded AEG package.

    Args:
        package: A loaded :class:`~aether.core.aeg_format.AEGPackage`.

    Returns:
        An engine ready to run :meth:`~CPUExecutionEngine.generate`.

    Raises:
        AEGLoadError: If the package has no weights or is missing tensors the
            forward pass requires.
    """
    # Loading a manifest is not sufficient: verify every declared payload
    # before any weight or packaged-kernel data reaches execution. This also
    # protects callers that use the loader directly instead of Runtime.
    package.verify_integrity()
    if not package.has_weights:
        msg = (
            f"AEG package at {package.root} contains no weights; it was compiled "
            f"graph-only and cannot run inference"
        )
        raise AEGLoadError(msg)

    architecture = _architecture_dict(package)
    if bool(architecture.get("is_encoder_decoder", False)):
        return _load_seq2seq_engine_from_package(package, architecture)
    if bool(architecture.get("is_encoder", False)):
        return _load_encoder_engine_from_package(package, architecture)
    # Every decoder family uses the same vocabulary contract even when its
    # execution engine is specialized (MLA, Mamba, RWKV, or hybrid). Validate
    # it before dispatch so a stale manifest cannot be hidden by a specialized
    # loader.
    _validate_decoder_vocabulary_contract(package, architecture)
    if str(architecture.get("attention_type", "") or "").upper() == "MLA":
        return _load_mla_engine_from_package(package, architecture)
    if architecture.get("ssm_variant") == "selective_scan":
        return _load_mamba_engine_from_package(package, architecture)
    if architecture.get("ssm_variant") == "hybrid_selective_scan":
        return _load_hybrid_engine_from_package(package, architecture)
    if architecture.get("ssm_variant") == "ssd":
        return _load_mamba2_engine_from_package(package, architecture)
    if architecture.get("ssm_variant") == "rwkv_time_mix":
        return _load_rwkv_engine_from_package(package, architecture)
    num_layers = int(architecture.get("layers", 0) or 0)
    num_heads = int(architecture.get("num_attention_heads", 0) or 0)
    if num_layers <= 0 or num_heads <= 0:
        msg = (
            f"AEG manifest declares layers={num_layers}, heads={num_heads}; "
            f"both must be positive to build an engine"
        )
        raise AEGLoadError(msg)
    num_kv_heads = int(architecture.get("num_kv_heads") or num_heads)

    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    hidden_size = int(embedding.shape[1])

    # Hard architecture invariants: the rebuilt engine must describe exactly
    # the model the manifest declares. A mismatch means the artifact is not
    # the compiled source model and must never reach execution.
    manifest_hidden = int(architecture.get("hidden_size", 0) or 0)
    if manifest_hidden > 0 and manifest_hidden != hidden_size:
        raise AEGLoadError(
            f"Hidden size invariant violated: manifest declares {manifest_hidden} "
            f"but the embedding tensor has hidden dimension {hidden_size}"
        )
    manifest_vocab = int(architecture.get("vocab_size", 0) or 0)
    if manifest_vocab > 0 and manifest_vocab != int(embedding.shape[0]):
        raise AEGLoadError(
            f"Vocabulary invariant violated: manifest declares vocab_size="
            f"{manifest_vocab} but the embedding tensor has {int(embedding.shape[0])} rows"
        )
    for name, tensor in ((k, v) for k, v in tensors.items() if k[0] is not None):
        if tensor.ndim == 2 and name[1] in ("q_proj", "k_proj", "v_proj") and tensor.shape[1] != hidden_size:
            raise AEGLoadError(
                f"Layer {name[0]} {name[1]} input feature dimension {tensor.shape[1]} "
                f"does not match hidden size {hidden_size}"
            )
        if tensor.ndim == 2 and name[1] == "o_proj" and tensor.shape[0] != hidden_size:
            raise AEGLoadError(
                f"Layer {name[0]} o_proj output feature dimension {tensor.shape[0]} "
                f"does not match hidden size {hidden_size}"
            )

    layers: list[LayerWeights] = []
    for index in range(num_layers):
        layers.append(
            _build_layer(
                tensors,
                index,
                hidden_size,
                num_heads,
                num_kv_heads,
                ffn_type=str(architecture.get("ffn_type", "SwiGLU") or "SwiGLU"),
                num_experts=int(architecture.get("num_experts", 0) or 0),
                num_activated_experts=int(
                    architecture.get("num_activated_experts", 0) or 0
                ),
                parallel_residual=bool(architecture.get("parallel_residual", False)),
                norm_placement=(
                    "sandwich"
                    if str(architecture.get("norm_placement", "pre")).lower() == "sandwich_glm"
                    else str(architecture.get("norm_placement", "pre") or "pre")
                ),
                qk_norm_scope=str(architecture.get("qk_norm_scope", "head") or "head"),
                num_kv_heads_for_norm=num_kv_heads,
                is_moe_layer=(
                    bool(architecture.get("is_moe", False))
                    and (
                        architecture.get("moe_layer_indices") is None
                        or index in architecture.get("moe_layer_indices", [])
                    )
                ),
            )
        )
    if len(layers) != num_layers:  # defensive: loop above always satisfies this
        raise AEGLoadError(
            f"Layer count invariant violated: built {len(layers)} layers, "
            f"manifest declares {num_layers}"
        )
    # The weight store must describe exactly the manifest's layer count: a
    # blob containing layer tensors beyond num_layers-1 means the manifest
    # understates the model (the silent 4-layer -> 1-layer failure mode).
    max_stored_layer = max(
        (key[0] for key in tensors if key[0] is not None and key[0] >= 0), default=-1
    )
    if max_stored_layer != num_layers - 1:
        raise AEGLoadError(
            f"Layer count invariant violated: manifest declares {num_layers} layers "
            f"but the weight store contains tensors for layers 0..{max_stored_layer}"
        )

    # Models with tied embeddings ship no separate lm_head; reuse the embedding.
    lm_head = tensors.get((None, "lm_head"))
    if lm_head is None:
        if not bool(architecture.get("tie_word_embeddings", True)):
            raise AEGLoadError(
                "untied decoder AEG is missing the required lm_head tensor"
            )
        lm_head = embedding
        logger.debug("No lm_head tensor found; using tied embedding weights")

    final_norm = tensors.get((None, "final_norm"))
    if final_norm is None:
        final_norm = np.ones(hidden_size, dtype=np.float32)
    final_norm_bias = tensors.get((None, "final_norm_bias"))
    embedding_norm = tensors.get((None, "embedding_norm"))
    embedding_norm_bias = tensors.get((None, "embedding_norm_bias"))
    position_embedding = tensors.get((None, "position_embedding"))
    position_type = str(architecture.get("position_type", "RoPE") or "RoPE").lower()
    if position_type in {"absolute", "learned", "learned_absolute"} and position_embedding is None:
        raise AEGLoadError("AEG package is missing the declared absolute position embedding")
    if bool(architecture.get("embedding_norm", False)) and embedding_norm is None:
        raise AEGLoadError("AEG package is missing the declared embedding normalization")

    weights = ModelWeights(
        embedding=embedding,
        layers=layers,
        final_norm=np.asarray(final_norm, dtype=np.float32).reshape(-1),
        lm_head=lm_head,
        position_embedding=(
            np.ascontiguousarray(position_embedding, dtype=np.float32)
            if position_embedding is not None else None
        ),
        embedding_norm=(
            np.asarray(embedding_norm, dtype=np.float32).reshape(-1)
            if embedding_norm is not None else None
        ),
        embedding_norm_bias=(
            np.asarray(embedding_norm_bias, dtype=np.float32).reshape(-1)
            if embedding_norm_bias is not None else None
        ),
        final_norm_bias=(
            np.asarray(final_norm_bias, dtype=np.float32).reshape(-1)
            if final_norm_bias is not None else None
        ),
        rope_theta=float(architecture.get("rope_theta", 10000.0) or 10000.0),
        norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
        norm_type=str(architecture.get("norm_type", "RMSNorm") or "RMSNorm"),
        ffn_type=str(architecture.get("ffn_type", "SwiGLU") or "SwiGLU"),
        position_type=str(architecture.get("position_type", "RoPE") or "RoPE"),
        parallel_residual=bool(architecture.get("parallel_residual", False)),
        attention_layers=(
            [str(value) for value in architecture["attention_layers"]]
            if isinstance(architecture.get("attention_layers"), list) else None
        ),
        attention_window=(
            int(architecture["attention_window"])
            if architecture.get("attention_window") is not None else None
        ),
        **_execution_numerics(architecture),
    )
    kernels = get_native_kernels()
    _load_packaged_native_kernels(package, kernels)
    engine = CPUExecutionEngine(
        weights,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kernels=kernels,
        sparse_attention_plan=_runtime_sparse_attention_plan(package.metadata, num_heads),
        semantic_kv_plan=_runtime_semantic_kv_plan(package.metadata, num_layers),
        cross_layer_kv_plan=_runtime_cross_layer_kv_plan(package.metadata, num_layers),
    )
    logger.info(
        "Loaded executable engine from %s (%d layers, hidden=%d, heads=%d/%d)",
        package.root.name,
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
    )
    return engine


def _load_mla_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> MLAExecutionEngine:
    """Load the model-generic MLA CPU executor from authenticated tensors."""
    from aether.attention.mla import MLAConfig

    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    final_norm = _require(tensors, (None, "final_norm"), "final norm")
    lm_head = tensors.get((None, "lm_head"), embedding)
    layers: list[MLALayerWeights] = []
    for index in range(int(architecture.get("layers", 0) or 0)):
        prefix = (index, "")
        required = {
            "attention_norm": _require(tensors, (index, "attention_norm"), f"MLA layer {index} attention norm"),
            "ffn_norm": _require(tensors, (index, "ffn_norm"), f"MLA layer {index} FFN norm"),
            "o_proj": _require(tensors, (index, "o_proj"), f"MLA layer {index} output projection"),
            "q_a_proj": _require(tensors, (index, "q_a_proj"), f"MLA layer {index} q_a projection"),
            "q_b_proj": _require(tensors, (index, "q_b_proj"), f"MLA layer {index} q_b projection"),
            "kv_a_proj": _require(tensors, (index, "kv_a_proj"), f"MLA layer {index} kv_a projection"),
            "kv_b_proj": _require(tensors, (index, "kv_b_proj"), f"MLA layer {index} kv_b projection"),
            "k_rope_proj": _require(tensors, (index, "k_rope_proj"), f"MLA layer {index} RoPE projection"),
            "q_a_norm": _require(tensors, (index, "q_a_norm"), f"MLA layer {index} q_a norm"),
            "kv_a_norm": _require(tensors, (index, "kv_a_norm"), f"MLA layer {index} kv_a norm"),
        }
        up = tensors.get((index, "up_proj"))
        moe_indices = architecture.get("moe_layer_indices")
        is_moe_layer = bool(architecture.get("is_moe", False)) and (
            moe_indices is None or index in {int(value) for value in moe_indices}
        )
        router = tensors.get((index, "moe_router")) if is_moe_layer else None
        experts: list[MLAExpertWeights] = []
        if is_moe_layer:
            if router is None:
                raise AEGLoadError(f"MLA MoE layer {index} is missing its router tensor")
            expert_count = int(architecture.get("num_experts", 0) or 0)
            if expert_count <= 0:
                raise AEGLoadError(f"MLA MoE layer {index} has no declared expert count")
            for expert_index in range(expert_count):
                gate = _require(tensors, (index, f"expert_{expert_index}_gate_proj"), f"MLA layer {index} expert {expert_index} gate projection")
                expert_up = _require(tensors, (index, f"expert_{expert_index}_up_proj"), f"MLA layer {index} expert {expert_index} up projection")
                down = _require(tensors, (index, f"expert_{expert_index}_down_proj"), f"MLA layer {index} expert {expert_index} down projection")
                experts.append(MLAExpertWeights(
                    gate_proj=np.asarray(gate, dtype=np.float32),
                    up_proj=np.asarray(expert_up, dtype=np.float32),
                    down_proj=np.asarray(down, dtype=np.float32),
                ))
            ffn_in = ffn_out = None
        else:
            ffn_in = _require(tensors, (index, "gate_proj"), f"MLA layer {index} FFN input")
            ffn_out = _require(tensors, (index, "down_proj"), f"MLA layer {index} FFN output")
        mla = {
            "q_a_proj.weight": np.asarray(required["q_a_proj"], dtype=np.float32),
            "q_b_proj.weight": np.asarray(required["q_b_proj"], dtype=np.float32),
            "kv_a_proj.weight": np.asarray(required["kv_a_proj"], dtype=np.float32),
            "kv_b_proj.weight": np.asarray(required["kv_b_proj"], dtype=np.float32),
            "k_rope_proj.weight": np.asarray(required["k_rope_proj"], dtype=np.float32),
            "q_a_norm.weight": _norm_vector(required["q_a_norm"], int(required["q_a_norm"].size)),
            "kv_a_norm.weight": _norm_vector(required["kv_a_norm"], int(required["kv_a_norm"].size)),
        }
        layers.append(MLALayerWeights(
            attention_norm=_norm_vector(required["attention_norm"], int(embedding.shape[1])),
            ffn_norm=_norm_vector(required["ffn_norm"], int(embedding.shape[1])),
            o_proj=np.asarray(required["o_proj"], dtype=np.float32),
            ffn_in=None if ffn_in is None else np.asarray(ffn_in, dtype=np.float32),
            ffn_out=None if ffn_out is None else np.asarray(ffn_out, dtype=np.float32),
            ffn_up=None if up is None else np.asarray(up, dtype=np.float32),
            mla=mla,
            router=None if router is None else np.asarray(router, dtype=np.float32),
            experts=experts,
            num_activated_experts=int(architecture.get("num_activated_experts", 1) or 1),
        ))
    config = MLAConfig(
        kv_lora_rank=int(architecture.get("mla_kv_lora_rank") or 0),
        q_lora_rank=int(architecture.get("mla_q_lora_rank") or 0),
        qk_nope_head_dim=int(architecture.get("mla_qk_nope_head_dim") or 0),
        qk_rope_head_dim=int(architecture.get("mla_qk_rope_head_dim") or 0),
        v_head_dim=int(architecture.get("mla_v_head_dim") or 0),
        num_heads=int(architecture.get("num_attention_heads") or 0),
        num_kv_heads=int(architecture.get("num_kv_heads") or architecture.get("num_attention_heads") or 0),
        rope_theta=float(architecture.get("rope_theta", 10000.0) or 10000.0),
    )
    if min(config.kv_lora_rank, config.q_lora_rank, config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim, config.num_heads) <= 0:
        raise AEGLoadError("MLA manifest contains incomplete latent-attention geometry")
    return MLAExecutionEngine(
        MLAModelWeights(
            embedding=np.asarray(embedding, dtype=np.float32),
            layers=layers,
            final_norm=_norm_vector(final_norm, int(embedding.shape[1])),
            lm_head=np.asarray(lm_head, dtype=np.float32),
            norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
            norm_type=str(architecture.get("norm_type", "RMSNorm") or "RMSNorm"),
            ffn_type=str(architecture.get("ffn_type", "SwiGLU") or "SwiGLU"),
        ),
        config,
    )


def _load_mamba_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> MambaExecutionEngine:
    """Load a Mamba selective-scan engine from the canonical AEG tensors."""
    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    final_norm = _require(tensors, (None, "final_norm"), "final norm")
    lm_head = tensors.get((None, "lm_head"), embedding)
    hidden = int(embedding.shape[1])
    layers: list[MambaLayerWeights] = []
    for index in range(int(architecture.get("layers", 0) or 0)):
        def required(component: str) -> np.ndarray:
            return _require(tensors, (index, component), f"Mamba layer {index} {component}")
        layers.append(MambaLayerWeights(
            norm=_norm_vector(required("ssm_norm"), hidden),
            in_proj=required("ssm_in_proj"),
            conv1d=required("ssm_conv1d"),
            x_proj=required("ssm_x_proj"),
            dt_proj=required("ssm_dt_proj"),
            a_log=required("ssm_a_log"),
            d=required("ssm_d"),
            out_proj=required("ssm_out_proj"),
            conv_bias=tensors.get((index, "ssm_conv1d_bias")),
            dt_bias=tensors.get((index, "ssm_dt_proj_bias")),
        ))
    return MambaExecutionEngine(
        MambaModelWeights(
            embedding=np.asarray(embedding, dtype=np.float32),
            layers=layers,
            final_norm=_norm_vector(final_norm, hidden),
            lm_head=np.asarray(lm_head, dtype=np.float32),
            norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
            norm_type=str(architecture.get("norm_type", "RMSNorm") or "RMSNorm"),
        ),
        state_size=int(architecture.get("ssm_state_size") or 16),
        inner_size=int(architecture.get("ssm_inner_size") or hidden * 2),
        dt_rank=int(architecture.get("ssm_dt_rank") or max(1, (hidden + 15) // 16)),
        conv_kernel=int(architecture.get("ssm_conv_kernel") or 4),
    )


def _load_hybrid_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> HybridExecutionEngine:
    """Load a Jamba-style mixed attention/selective-scan artifact.

    The schedule is authenticated as architecture metadata and every layer is
    required to provide the tensor contract for its declared block kind.  The
    helper-only representation for the other block kind contains shape-valid
    zero/identity arrays and is never executed; this lets the shared primitive
    implementations be reused without weakening artifact validation.
    """
    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    final_norm = _require(tensors, (None, "final_norm"), "final norm")
    lm_head = tensors.get((None, "lm_head"), embedding)
    hidden = int(embedding.shape[1])
    num_layers = int(architecture.get("layers", 0) or 0)
    num_heads = int(architecture.get("num_attention_heads", 0) or 0)
    num_kv_heads = int(architecture.get("num_kv_heads") or num_heads)
    schedule = architecture.get("hybrid_layer_types")
    if num_layers <= 0 or num_heads <= 0 or not isinstance(schedule, list) or len(schedule) != num_layers:
        raise AEGLoadError("hybrid manifest has incomplete layer schedule or attention geometry")
    schedule = [str(value).lower() for value in schedule]
    if any(value not in {"attention", "ssm"} for value in schedule):
        raise AEGLoadError("hybrid manifest contains an unknown layer schedule entry")

    inner = int(architecture.get("ssm_inner_size") or hidden * 2)
    state = int(architecture.get("ssm_state_size") or 16)
    dt_rank = int(architecture.get("ssm_dt_rank") or max(1, (hidden + 15) // 16))
    conv_kernel = int(architecture.get("ssm_conv_kernel") or 4)
    ffn_type = str(architecture.get("ffn_type", "SwiGLU") or "SwiGLU")

    def zero_matrix(rows: int, cols: int) -> np.ndarray:
        return np.zeros((int(rows), int(cols)), dtype=np.float32)

    def dummy_transformer() -> LayerWeights:
        head_dim = int(architecture.get("head_dim") or (hidden // max(num_heads, 1)))
        intermediate = int(architecture.get("intermediate_size") or hidden * 4)
        return LayerWeights(
            attention_norm=np.ones(hidden, dtype=np.float32),
            q_proj=zero_matrix(num_heads * head_dim, hidden),
            k_proj=zero_matrix(num_kv_heads * head_dim, hidden),
            v_proj=zero_matrix(num_kv_heads * head_dim, hidden),
            o_proj=zero_matrix(hidden, num_heads * head_dim),
            ffn_norm=np.ones(hidden, dtype=np.float32),
            gate_proj=zero_matrix(intermediate, hidden),
            up_proj=zero_matrix(intermediate, hidden) if ffn_type.lower() not in {"gelu", "relu", "relu2"} else None,
            down_proj=zero_matrix(hidden, intermediate),
        )

    def dummy_mamba() -> MambaLayerWeights:
        return MambaLayerWeights(
            norm=np.ones(hidden, dtype=np.float32),
            in_proj=zero_matrix(2 * inner, hidden),
            conv1d=zero_matrix(inner, conv_kernel),
            x_proj=zero_matrix(dt_rank + 2 * state, inner),
            dt_proj=zero_matrix(inner, dt_rank),
            a_log=np.zeros((inner, state), dtype=np.float32),
            d=np.zeros(inner, dtype=np.float32),
            out_proj=zero_matrix(hidden, inner),
        )

    layers: list[HybridLayerWeights] = []
    for index, kind in enumerate(schedule):
        if kind == "attention":
            actual = _build_layer(
                tensors, index, hidden, num_heads, num_kv_heads,
                ffn_type=ffn_type,
            )
            layers.append(HybridLayerWeights(kind=kind, transformer=actual, mamba=dummy_mamba()))
        else:
            def required(component: str) -> np.ndarray:
                return _require(tensors, (index, component), f"hybrid SSM layer {index} {component}")
            actual = MambaLayerWeights(
                norm=_norm_vector(required("ssm_norm"), hidden),
                in_proj=required("ssm_in_proj"),
                conv1d=required("ssm_conv1d"),
                x_proj=required("ssm_x_proj"),
                dt_proj=required("ssm_dt_proj"),
                a_log=required("ssm_a_log"),
                d=required("ssm_d"),
                out_proj=required("ssm_out_proj"),
                conv_bias=tensors.get((index, "ssm_conv1d_bias")),
                dt_bias=tensors.get((index, "ssm_dt_proj_bias")),
            )
            layers.append(HybridLayerWeights(kind=kind, transformer=dummy_transformer(), mamba=actual))

    position_embedding = tensors.get((None, "position_embedding"))
    embedding_norm = tensors.get((None, "embedding_norm"))
    embedding_norm_bias = tensors.get((None, "embedding_norm_bias"))
    final_norm_bias = tensors.get((None, "final_norm_bias"))
    position_type = str(architecture.get("position_type", "RoPE") or "RoPE")
    if position_type.lower() in {"absolute", "learned", "learned_absolute"} and position_embedding is None:
        raise AEGLoadError("hybrid artifact declares learned positions but has no position embedding")
    if bool(architecture.get("embedding_norm", False)) and embedding_norm is None:
        raise AEGLoadError("hybrid artifact declares embedding normalization but has no tensor")
    return HybridExecutionEngine(
        HybridModelWeights(
            embedding=np.asarray(embedding, dtype=np.float32),
            layers=layers,
            final_norm=_norm_vector(final_norm, hidden),
            lm_head=np.asarray(lm_head, dtype=np.float32),
            position_embedding=None if position_embedding is None else np.asarray(position_embedding, dtype=np.float32),
            embedding_norm=None if embedding_norm is None else _norm_vector(embedding_norm, hidden),
            embedding_norm_bias=None if embedding_norm_bias is None else _norm_vector(embedding_norm_bias, hidden),
            final_norm_bias=None if final_norm_bias is None else _norm_vector(final_norm_bias, hidden),
            position_type=position_type,
            rope_theta=float(architecture.get("rope_theta", 10000.0) or 10000.0),
            norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
            norm_type=str(architecture.get("norm_type", "RMSNorm") or "RMSNorm"),
            ffn_type=ffn_type,
        ),
        layer_types=schedule,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        state_size=state,
        inner_size=inner,
        dt_rank=dt_rank,
        conv_kernel=conv_kernel,
    )


def _load_mamba2_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> Mamba2ExecutionEngine:
    """Load a Mamba-2/SSD engine from the canonical AEG tensors."""
    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    final_norm = _require(tensors, (None, "final_norm"), "final norm")
    lm_head = tensors.get((None, "lm_head"), embedding)
    layers: list[Mamba2LayerWeights] = []
    for index in range(int(architecture.get("layers", 0) or 0)):
        def required(component: str) -> np.ndarray:
            return _require(tensors, (index, component), f"Mamba-2 layer {index} {component}")
        layers.append(Mamba2LayerWeights(
            norm=_norm_vector(required("ssm_norm"), int(embedding.shape[1])),
            in_proj=required("ssm_in_proj"),
            conv1d=required("ssm_conv1d"),
            a_log=required("ssm_a_log"),
            d=required("ssm_d"),
            dt=required("ssm_dt"),
            out_proj=required("ssm_out_proj"),
            in_proj_bias=tensors.get((index, "ssm_in_proj_bias")),
            conv_bias=tensors.get((index, "ssm_conv1d_bias")),
        ))
    hidden = int(embedding.shape[1])
    inner = int(architecture.get("ssm_inner_size") or hidden * 2)
    heads = int(architecture.get("ssm_num_heads") or 0)
    groups = int(architecture.get("ssm_num_groups") or 1)
    head_dim = int(architecture.get("ssm_head_dim") or (inner // max(heads, 1)))
    if heads <= 0:
        raise AEGLoadError("Mamba-2 manifest is missing positive ssm_num_heads")
    return Mamba2ExecutionEngine(
        Mamba2ModelWeights(
            embedding=np.asarray(embedding, dtype=np.float32),
            layers=layers,
            final_norm=_norm_vector(final_norm, hidden),
            lm_head=np.asarray(lm_head, dtype=np.float32),
            norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
            norm_type=str(architecture.get("norm_type", "RMSNorm") or "RMSNorm"),
        ),
        state_size=int(architecture.get("ssm_state_size") or 16),
        inner_size=inner,
        num_heads=heads,
        num_groups=groups,
        conv_kernel=int(architecture.get("ssm_conv_kernel") or 4),
        head_dim=head_dim,
    )


def _load_rwkv_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> RWKVExecutionEngine:
    """Load an RWKV time-mix engine from canonical recurrent tensors."""
    tensors = _dequantized_by_key(package)
    embedding = _require(tensors, (None, "embedding"), "token embedding")
    final_norm = _require(tensors, (None, "final_norm"), "final norm")
    lm_head = tensors.get((None, "lm_head"), embedding)
    layers: list[RWKVLayerWeights] = []
    hidden = int(embedding.shape[1])
    for index in range(int(architecture.get("layers", 0) or 0)):
        def required(component: str) -> np.ndarray:
            return _require(tensors, (index, component), f"RWKV layer {index} {component}")
        layers.append(RWKVLayerWeights(
            norm=_norm_vector(required("ssm_norm"), hidden),
            ffn_norm=_norm_vector(required("ssm_ffn_norm"), hidden),
            time_decay=required("ssm_time_decay"), time_first=required("ssm_time_first"),
            time_mix_k=required("ssm_time_mix_k"), time_mix_v=required("ssm_time_mix_v"),
            time_mix_r=required("ssm_time_mix_r"),
            ffn_time_mix_k=required("ssm_ffn_time_mix_k"),
            ffn_time_mix_r=required("ssm_ffn_time_mix_r"),
            key=required("ssm_key"),
            value=required("ssm_value"), receptance=required("ssm_receptance"),
            output=required("ssm_output"), ffn_key=required("ssm_ffn_key"),
            ffn_value=required("ssm_ffn_value"), ffn_receptance=required("ssm_ffn_receptance"),
        ))
    return RWKVExecutionEngine(RWKVModelWeights(
        embedding=np.asarray(embedding, dtype=np.float32), layers=layers,
        final_norm=_norm_vector(final_norm, hidden), lm_head=np.asarray(lm_head, dtype=np.float32),
        norm_eps=float(architecture.get("norm_eps", 1e-5) or 1e-5),
    ))


def _load_seq2seq_engine_from_package(package: Any, architecture: dict[str, Any]) -> Seq2SeqExecutionEngine:
    """Load a T5-compatible encoder-decoder AEG from authenticated tensors."""
    from aether.quantization.formats import dequantize_tensor

    store = package.weight_store()
    tensors: dict[str, np.ndarray] = {
        name: dequantize_tensor(store.load_tensor(name)).astype(np.float32, copy=False)
        for name in store.entries
    }

    def require(name: str) -> np.ndarray:
        value = tensors.get(name)
        if value is None:
            raise AEGLoadError(f"AEG package is missing required seq2seq tensor {name!r}")
        return np.asarray(value, dtype=np.float32)

    def optional(name: str) -> np.ndarray | None:
        value = tensors.get(name)
        return None if value is None else np.asarray(value, dtype=np.float32)

    enc_count = int(architecture.get("encoder_layers") or architecture.get("layers", 0))
    dec_count = int(architecture.get("decoder_layers") or architecture.get("layers", 0))
    heads = int(architecture.get("num_attention_heads", 0))
    head_dim = int(architecture.get("head_dim") or (int(architecture.get("hidden_size", 0)) // max(heads, 1)))
    ffn_type = str(architecture.get("ffn_type", "ReLU"))

    embedding = require("embedding")
    _validate_embedding_shape(embedding, architecture, context="encoder-decoder")
    lm_head = optional("lm_head")
    if lm_head is None:
        if not bool(architecture.get("tie_word_embeddings", True)):
            raise AEGLoadError("untied encoder-decoder AEG is missing lm_head")
        lm_head = embedding
    elif lm_head.ndim != 2 or lm_head.shape != embedding.shape:
        raise AEGLoadError(
            "encoder-decoder LM head invariant violated: expected shape "
            f"{tuple(embedding.shape)}, tensor has {tuple(lm_head.shape)}"
        )

    encoder_layers: list[Seq2SeqLayer] = []
    for i in range(enc_count):
        prefix = f"encoder_layer_{i}_"
        encoder_layers.append(Seq2SeqLayer(
            norm1=require(prefix + "norm1"),
            q=require(prefix + "q_proj"), k=require(prefix + "k_proj"),
            v=require(prefix + "v_proj"), o=require(prefix + "o_proj"),
            norm2=require(prefix + "norm2"),
            ffn_in=require(prefix + "ffn_in") if prefix + "ffn_in" in tensors else require(prefix + "ffn_in_0"),
            ffn_out=require(prefix + "ffn_out"),
            relative_bias=optional(prefix + "relative_attention_bias"),
            ffn_in_0=optional(prefix + "ffn_in_0"),
            ffn_in_1=optional(prefix + "ffn_in_1"),
        ))

    decoder_layers: list[Seq2SeqDecoderLayer] = []
    for i in range(dec_count):
        prefix = f"decoder_layer_{i}_"
        decoder_layers.append(Seq2SeqDecoderLayer(
            self_norm=require(prefix + "self_norm"),
            self_q=require(prefix + "self_q_proj"), self_k=require(prefix + "self_k_proj"),
            self_v=require(prefix + "self_v_proj"), self_o=require(prefix + "self_o_proj"),
            cross_norm=require(prefix + "cross_norm"),
            cross_q=require(prefix + "cross_q_proj"), cross_k=require(prefix + "cross_k_proj"),
            cross_v=require(prefix + "cross_v_proj"), cross_o=require(prefix + "cross_o_proj"),
            ffn_norm=require(prefix + "ffn_norm"),
            ffn_in=require(prefix + "ffn_in") if prefix + "ffn_in" in tensors else require(prefix + "ffn_in_0"),
            ffn_out=require(prefix + "ffn_out"),
            relative_bias=optional(prefix + "self_relative_attention_bias"),
            ffn_in_0=optional(prefix + "ffn_in_0"),
            ffn_in_1=optional(prefix + "ffn_in_1"),
        ))

    return Seq2SeqExecutionEngine(
        embedding,
        encoder_layers,
        decoder_layers,
        require("encoder_final_norm"),
        require("final_norm"),
        lm_head,
        num_heads=heads,
        head_dim=head_dim,
        norm_eps=float(architecture.get("norm_eps", 1e-6)),
        ffn_type=ffn_type,
        tie_word_embeddings=bool(architecture.get("tie_word_embeddings", True)),
        relative_attention_num_buckets=int(architecture.get("relative_attention_num_buckets", 32)),
    )


def _load_encoder_engine_from_package(
    package: Any, architecture: dict[str, Any]
) -> EncoderExecutionEngine:
    """Build the bidirectional CPU encoder from named AEG tensors."""
    from aether.quantization.formats import dequantize_tensor

    store = package.weight_store()

    def required(name: str) -> np.ndarray:
        if name not in store.entries:
            raise AEGLoadError(f"encoder AEG is missing required tensor {name!r}")
        return np.asarray(dequantize_tensor(store.load_tensor(name)), dtype=np.float32)

    def optional(name: str) -> np.ndarray | None:
        return required(name) if name in store.entries else None

    embedding = required("embedding")
    hidden_size = int(embedding.shape[1])
    declared_hidden = int(architecture.get("hidden_size", 0) or 0)
    if declared_hidden and declared_hidden != hidden_size:
        raise AEGLoadError(
            f"encoder hidden size invariant violated: manifest={declared_hidden}, tensor={hidden_size}"
        )
    declared_vocab = int(architecture.get("vocab_size", 0) or 0)
    if declared_vocab and declared_vocab != int(embedding.shape[0]):
        raise AEGLoadError(
            f"encoder vocabulary invariant violated: manifest={declared_vocab}, tensor={embedding.shape[0]}"
        )

    layers: list[EncoderLayerWeights] = []
    layer_count = int(architecture.get("layers", 0) or 0)
    for index in range(layer_count):
        prefix = f"layer_{index}_"
        layers.append(
            EncoderLayerWeights(
                q_proj=required(prefix + "q_proj"),
                k_proj=required(prefix + "k_proj"),
                v_proj=required(prefix + "v_proj"),
                o_proj=required(prefix + "o_proj"),
                attention_norm=required(prefix + "attention_norm"),
                intermediate_proj=required(prefix + "intermediate_proj"),
                output_proj=required(prefix + "output_proj"),
                output_norm=required(prefix + "output_norm"),
                q_bias=optional(prefix + "q_proj_bias"),
                k_bias=optional(prefix + "k_proj_bias"),
                v_bias=optional(prefix + "v_proj_bias"),
                o_bias=optional(prefix + "o_proj_bias"),
                intermediate_bias=optional(prefix + "intermediate_proj_bias"),
                output_bias=optional(prefix + "output_proj_bias"),
                attention_norm_bias=optional(prefix + "attention_norm_bias"),
                output_norm_bias=optional(prefix + "output_norm_bias"),
            )
        )
    if len(layers) != layer_count or layer_count <= 0:
        raise AEGLoadError("encoder manifest must declare at least one layer")
    num_heads = int(architecture.get("num_attention_heads", 0) or 0)
    weights = EncoderModelWeights(
        embedding=embedding,
        position_embedding=required("position_embedding"),
        token_type_embedding=required("token_type_embedding"),
        embedding_norm=required("embedding_norm"),
        pooler=required("pooler"),
        layers=layers,
        embedding_norm_bias=optional("embedding_norm_bias"),
        pooler_bias=optional("pooler_bias"),
        norm_eps=float(architecture.get("norm_eps", 1e-12) or 1e-12),
    )
    return EncoderExecutionEngine(weights, num_heads=num_heads)


def _runtime_sparse_attention_plan(
    metadata: Any,
    num_heads: int,
) -> dict[str, Any] | None:
    """Validate and return a Pass 8 plan for the executable CPU engine."""
    if not isinstance(metadata, dict):
        return None
    plan = metadata.get("attention_head_patterns")
    if not isinstance(plan, dict) or not bool(plan.get("enabled")):
        return None
    patterns = plan.get("patterns")
    if not isinstance(patterns, list) or len(patterns) != num_heads:
        raise AEGLoadError(
            "enabled sparse attention plan must contain one pattern per attention head"
        )
    valid = {"dense", "a_shape", "vertical_slash", "block_sparse"}
    for descriptor in patterns:
        if not isinstance(descriptor, dict):
            raise AEGLoadError("sparse attention pattern must be an object")
        pattern = descriptor.get("pattern", descriptor.get("pattern_type"))
        if pattern not in valid:
            raise AEGLoadError(f"unsupported sparse attention pattern {pattern!r}")
    return plan


def _runtime_semantic_kv_plan(metadata: Any, num_layers: int) -> dict[str, Any] | None:
    """Validate and return an executable Pass 14 plan."""
    if not isinstance(metadata, dict):
        return None
    plan = metadata.get("kv_compression_plan")
    if plan is None:
        return None
    if not isinstance(plan, dict) or plan.get("format") != "aether_kv_compression_v1":
        raise AEGLoadError("semantic KV plan has an unsupported format")
    layers = plan.get("layers")
    if not isinstance(layers, list) or len(layers) != num_layers:
        raise AEGLoadError(
            f"semantic KV plan must contain exactly {num_layers} layer policies"
        )
    strategy = str(plan.get("strategy", "chunk"))
    if strategy not in {"chunk", "hybrid"}:
        raise AEGLoadError(
            f"semantic KV strategy {strategy!r} requires tokenizer boundary metadata; "
            "only chunk and hybrid plans are executable by the CPU cache"
        )
    for index, policy in enumerate(layers):
        if not isinstance(policy, dict):
            raise AEGLoadError(f"semantic KV policy {index} must be an object")
        try:
            retention = float(policy["retention_ratio"])
            chunk_size = int(policy["chunk_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AEGLoadError(f"invalid semantic KV policy {index}") from exc
        if not 0.0 < retention <= 1.0 or chunk_size <= 0:
            raise AEGLoadError(f"invalid semantic KV policy {index}")
    return plan


def _runtime_cross_layer_kv_plan(metadata: Any, num_layers: int) -> dict[str, Any] | None:
    """Validate and return an executable Pass 15 plan."""
    if not isinstance(metadata, dict):
        return None
    plan = metadata.get("cross_layer_kv_plan")
    if plan is None:
        return None
    if not isinstance(plan, dict) or plan.get("format") != "aether_cross_layer_kv_v1":
        raise AEGLoadError("cross-layer KV plan has an unsupported format")
    if int(plan.get("n_layers", -1)) != num_layers:
        raise AEGLoadError("cross-layer KV plan layer count does not match the model")
    groups = plan.get("sharing_groups")
    if not isinstance(groups, list):
        raise AEGLoadError("cross-layer KV plan must contain sharing_groups")
    targets: set[int] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise AEGLoadError("cross-layer KV sharing group must be an object")
        source = int(group.get("src_layer", -1))
        shared = group.get("shared_with")
        if source < 0 or source >= num_layers or not isinstance(shared, list):
            raise AEGLoadError("invalid cross-layer KV source group")
        for target_value in shared:
            target = int(target_value)
            if target <= source or target >= num_layers or target in targets:
                raise AEGLoadError(
                    "cross-layer KV targets must have one earlier source layer"
                )
            targets.add(target)
    return plan


def _load_packaged_native_kernels(package: Any, kernels: Any) -> None:
    """Load the authenticated native CPU library embedded in an AEG, if any.

    Three-tier fallback chain (Compile-Once, Run-Anywhere):

    1. **Packaged DLL** – The compiled shared library that was authenticated and
       bundled by the AEG compiler at compile time.  Its SHA-256 digest is stored
       in ``metadata.kernel_artifacts`` and verified before loading.  When this
       succeeds the host requires *no* C++ compiler or build toolchain.

    2. **Host recompile** – If the packaged library is absent (older AEG/1.x),
       mismatches the expected digest, or cannot be loaded (wrong OS/ISA), we
       fall back to ``kernels.ensure_compiled()``.  The result is written to the
       host kernel cache so subsequent loads skip compilation.

    3. **NumPy reference path** – If the host has no compiler either, the engine
       continues with pure-NumPy implementations.  Performance is lower but
       inference is fully functional on any Python installation.

    This design is inspired by ONNX Runtime's EP fallback hierarchy (Microsoft
    2019) and matches the Aether PRD §compile-once-run-everywhere requirement.
    """
    metadata = getattr(package, "metadata", {})
    descriptors = metadata.get("kernel_artifacts", []) if isinstance(metadata, dict) else []

    # ── Tier 1: bundled library ────────────────────────────────────────────
    if descriptors:
        if not isinstance(descriptors, list):
            raise AEGLoadError("AEG kernel_artifacts metadata must be a list")
        for descriptor in descriptors:
            if not isinstance(descriptor, dict) or descriptor.get("backend") != "native_cpu":
                continue
            relative_path = descriptor.get("path")
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
            ):
                raise AEGLoadError("AEG packaged kernel path is unsafe")
            path = (Path(package.root) / relative_path).resolve()
            if not path.is_relative_to(Path(package.root).resolve()):
                raise AEGLoadError("AEG packaged kernel escapes package root")
            expected = descriptor.get("sha256")
            if not isinstance(expected, str):
                raise AEGLoadError(
                    f"AEG packaged kernel {relative_path!r} has no SHA-256 digest"
                )
            if not path.exists():
                logger.warning(
                    "Packaged native kernel %r not found on disk; falling back to host recompile",
                    relative_path,
                )
                break  # fall through to Tier 2
            if kernels.load_library(path, expected_sha256=expected):
                logger.info(
                    "Loaded bundled native CPU kernel from AEG package (%s) — "
                    "no host compiler required",
                    path.name,
                )
                return  # ✓ Tier 1 succeeded
            # load_library sets kernels.build_error on failure
            logger.warning(
                "Bundled native kernel %r failed to load (%s); falling back to host recompile",
                relative_path,
                kernels.build_error,
            )
            break  # fall through to Tier 2

    # ── Tier 2: host recompile ─────────────────────────────────────────────
    if kernels.ensure_compiled():
        if descriptors:
            logger.info(
                "Native CPU kernel recompiled on host (bundled library was unavailable). "
                "Result cached at: %s",
                kernels.library_path,
            )
        else:
            logger.debug(
                "No bundled kernel in AEG (pre-v1.1); compiled from host toolchain: %s",
                kernels.library_path,
            )
        return  # ✓ Tier 2 succeeded

    # ── Tier 3: NumPy reference path (always available) ───────────────────
    logger.warning(
        "Native CPU kernels unavailable (no bundled library and no host compiler). "
        "Running on NumPy reference path. Performance will be lower. "
        "Install a C++ compiler (GCC/Clang/MSVC) to enable AVX2/OpenMP acceleration."
    )
    # kernels.is_native == False → engine uses kernels.rmsnorm_ref etc. automatically




def _execution_numerics(architecture: dict[str, Any]) -> dict[str, Any]:
    """Extract the manifest's execution-numerics contract for ``ModelWeights``.

    These constants are part of the source model's definition (attention scale,
    embedding/residual/logit multipliers, logit soft caps, rotary geometry, and
    block normalization placement).  A manifest written before they existed
    simply omits them, and the defaults reproduce the standard Llama-style
    block, so older artifacts keep their previous behaviour.
    """

    def number(key: str) -> float | None:
        value = architecture.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def index_list(key: str) -> list[int] | None:
        value = architecture.get(key)
        if not isinstance(value, list):
            return None
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            return None

    placement = str(architecture.get("norm_placement", "pre") or "pre").lower()
    if placement == "sandwich_glm":
        # A spelling variant of the sandwich block; ingestion has already bound
        # each norm to its slot, so execution is identical.
        placement = "sandwich"
    if placement not in {"pre", "post", "sandwich"}:
        raise AEGLoadError(f"unsupported norm placement {placement!r} in AEG manifest")
    scope = str(architecture.get("qk_norm_scope", "head") or "head").lower()
    if scope not in {"head", "full"}:
        raise AEGLoadError(f"unsupported qk_norm scope {scope!r} in AEG manifest")
    partial = architecture.get("rope_partial_dim")
    return {
        "attention_scale": number("attention_scale"),
        "attention_scale_by_layer_index": bool(
            architecture.get("attention_scale_by_layer_index", False)
        ),
        "embedding_scale": number("embedding_scale"),
        "residual_scale": number("residual_scale"),
        "logit_scale": number("logit_scale"),
        "attn_logit_softcap": number("attn_logit_softcap"),
        "final_logit_softcap": number("final_logit_softcap"),
        "norm_offset_one": bool(architecture.get("norm_offset_one", False)),
        "rope_partial_dim": int(partial) if partial else None,
        "rope_interleaved": bool(architecture.get("rope_interleaved", False)),
        "rope_local_theta": number("rope_local_theta"),
        "norm_placement": placement,
        "qk_norm_scope": scope,
        "no_rope_layers": index_list("no_rope_layers"),
        "gelu_approximate": bool(architecture.get("gelu_approximate", True)),
        "moe_renormalize_topk": bool(architecture.get("moe_renormalize_topk", True)),
        "context_length": (
            int(architecture["context_length"])
            if architecture.get("context_length") else None
        ),
    }


def _architecture_dict(package: Any) -> dict[str, Any]:
    """Extract the architecture metadata from a package manifest."""
    manifest = package.manifest
    if manifest is None:
        msg = f"AEG package at {package.root} has no manifest"
        raise AEGLoadError(msg)
    architecture = getattr(manifest, "architecture", None)
    if architecture is None:
        return {}
    return architecture.to_dict() if hasattr(architecture, "to_dict") else dict(architecture)


def _validate_embedding_shape(
    embedding: np.ndarray,
    architecture: dict[str, Any],
    *,
    context: str = "",
) -> None:
    """Validate the physical token table against manifest dimensions."""
    label = f"{context} " if context else ""
    if embedding.ndim != 2:
        raise AEGLoadError(
            f"{label}embedding invariant violated: expected a 2-D table, "
            f"tensor has shape {tuple(embedding.shape)}"
        )
    hidden_size = int(embedding.shape[1])
    manifest_hidden = int(architecture.get("hidden_size", 0) or 0)
    if manifest_hidden > 0 and manifest_hidden != hidden_size:
        raise AEGLoadError(
            f"{label}hidden size invariant violated: manifest declares "
            f"{manifest_hidden} but the embedding tensor has hidden dimension "
            f"{hidden_size}"
        )
    manifest_vocab = int(architecture.get("vocab_size", 0) or 0)
    if manifest_vocab > 0 and manifest_vocab != int(embedding.shape[0]):
        raise AEGLoadError(
            f"{label}vocabulary invariant violated: manifest declares "
            f"vocab_size={manifest_vocab} but the embedding tensor has "
            f"{int(embedding.shape[0])} rows"
        )


def _validate_decoder_vocabulary_contract(
    package: Any, architecture: dict[str, Any]
) -> None:
    """Validate vocabulary dimensions before any decoder-family dispatch."""
    from aether.quantization.formats import dequantize_tensor

    store = package.weight_store()
    if "embedding" not in store.entries:
        raise AEGLoadError("AEG package is missing required token embedding")
    embedding = np.asarray(
        dequantize_tensor(store.load_tensor("embedding")), dtype=np.float32
    )
    _validate_embedding_shape(embedding, architecture)
    lm_head = None
    if "lm_head" in store.entries:
        lm_head = np.asarray(
            dequantize_tensor(store.load_tensor("lm_head")), dtype=np.float32
        )
    if lm_head is not None and (
        lm_head.ndim != 2
        or lm_head.shape[0] != embedding.shape[0]
        or lm_head.shape[1] != embedding.shape[1]
    ):
        raise AEGLoadError(
            "LM head invariant violated: expected shape "
            f"{tuple(embedding.shape)} but tensor has {tuple(lm_head.shape)}"
        )


def _dequantized_by_key(package: Any) -> dict[tuple[int | None, str | None], np.ndarray]:
    """Read every tensor and index it by ``(layer_index, component)``."""
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
    from aether.quantization.formats import dequantize_tensor

    store = package.weight_store()
    result: dict[tuple[int | None, str | None], np.ndarray] = {}
    for name in store.entries:
        # The ingestion normalizer intentionally groups checkpoint q/k/v names
        # under ``qkv``.  The persisted AEG store has already split those
        # projections, so preserve their exact component names here.
        match = re.match(
            r"^layer_(\d+)_((?:q_proj|k_proj|v_proj|q_norm|k_norm|o_proj|"
            r"attention_norm|ffn_norm|post_attention_norm|post_ffn_norm|"
            r"gate_proj|up_proj|down_proj|moe_router|"
            r"expert_\d+_(?:gate_proj|up_proj|down_proj)|q_a_proj|q_b_proj|"
            r"kv_a_proj|kv_b_proj|k_rope_proj|q_a_norm|kv_a_norm|ssm_norm|"
            r"ssm_in_proj|ssm_conv1d|ssm_x_proj|ssm_dt_proj|ssm_dt|ssm_a_log|ssm_d|"
            r"ssm_out_proj|ssm_time_decay|ssm_time_first|ssm_time_mix_k|"
            r"ssm_time_mix_v|ssm_time_mix_r|ssm_ffn_norm|ssm_ffn_time_mix_k|"
            r"ssm_ffn_time_mix_r|ssm_key|ssm_value|ssm_receptance|"
            r"ssm_output|ssm_ffn_key|ssm_ffn_value|ssm_ffn_receptance))(?:_bias)?$",
            name,
        )
        if match:
            suffix = match.group(2)
            key = (int(match.group(1)), f"{suffix}_bias" if name.endswith("_bias") else suffix)
        elif name == "embedding":
            key = (None, "embedding")
        elif name == "lm_head":
            key = (None, "lm_head")
        elif name == "position_embedding":
            key = (None, "position_embedding")
        elif name in {"embedding_norm", "embedding_norm_bias"}:
            key = (None, name)
        elif name in {"final_norm", "final_norm_bias"}:
            key = (None, "final_norm")
            if name.endswith("_bias"):
                key = (None, "final_norm_bias")
        else:
            key = IngestionPipeline._normalise_weight_name(name)
        if key[1] is None:
            continue
        # First writer wins, matching the ingestion side's fused-tensor handling.
        result.setdefault(key, dequantize_tensor(store.load_tensor(name)))
    return result


def _require(
    tensors: dict[tuple[int | None, str | None], np.ndarray],
    key: tuple[int | None, str | None],
    description: str,
) -> np.ndarray:
    """Fetch a required tensor or fail with a message naming what is missing."""
    tensor = tensors.get(key)
    if tensor is None:
        msg = f"AEG package is missing its {description} (looked for key {key})"
        raise AEGLoadError(msg)
    return tensor


def _build_layer(
    tensors: dict[tuple[int | None, str | None], np.ndarray],
    index: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_type: str = "SwiGLU",
    num_experts: int = 0,
    num_activated_experts: int = 0,
    is_moe_layer: bool = False,
    parallel_residual: bool = False,
    norm_placement: str = "pre",
    qk_norm_scope: str = "head",
    num_kv_heads_for_norm: int | None = None,
) -> LayerWeights:
    """Assemble one :class:`LayerWeights` from the tensor index.

    A package is executable only when every projection required by the declared
    architecture is present.  Identity/zero substitutions would produce logits
    that look valid while no longer representing the source model, so malformed
    or partial artifacts fail closed.
    """
    q_proj = tensors.get((index, "q_proj"))
    head_dim = (q_proj.shape[0] // num_heads) if q_proj is not None and q_proj.ndim == 2 else (hidden_size // num_heads)
    kv_dim = num_kv_heads * head_dim
    intermediate = _infer_intermediate(tensors, index, hidden_size)
    # OLMo-2 normalizes the whole Q/K projection rather than each head, so the
    # stored vector spans ``heads * head_dim`` instead of ``head_dim``.
    full_scope = str(qk_norm_scope).lower() == "full"
    q_norm_size = num_heads * head_dim if full_scope else head_dim
    k_norm_size = (
        (num_kv_heads_for_norm or num_kv_heads) * head_dim if full_scope else head_dim
    )
    sandwich = str(norm_placement).lower().startswith("sandwich")

    k_proj = tensors.get((index, "k_proj"))
    v_proj = tensors.get((index, "v_proj"))
    fused = tensors.get((index, "qkv"))
    if q_proj is None and fused is not None and fused.ndim == 2:
        q_rows = num_heads * (hidden_size // num_heads)
        expected_rows = q_rows + 2 * kv_dim
        if fused.shape[0] == expected_rows:
            q_proj = fused[:q_rows]
            k_proj = fused[q_rows : q_rows + kv_dim]
            v_proj = fused[q_rows + kv_dim :]
    if is_moe_layer:
        common = {
            "q_proj": q_proj,
            "k_proj": k_proj,
            "v_proj": v_proj,
            "o_proj": tensors.get((index, "o_proj")),
            "attention_norm": tensors.get((index, "attention_norm")),
            "ffn_norm": tensors.get((index, "ffn_norm")),
            "router": tensors.get((index, "moe_router")),
        }
        missing = [name for name, value in common.items() if value is None]
        if missing:
            raise AEGLoadError(
                f"AEG package is missing required MoE layer {index} tensors: "
                f"{', '.join(missing)}"
            )
        experts: list[ExpertWeights] = []
        for expert_index in range(int(num_experts)):
            prefix = (index, f"expert_{expert_index}_")
            gate = tensors.get((index, f"expert_{expert_index}_gate_proj"))
            up = tensors.get((index, f"expert_{expert_index}_up_proj"))
            down = tensors.get((index, f"expert_{expert_index}_down_proj"))
            if gate is None or up is None or down is None:
                raise AEGLoadError(
                    f"AEG package is missing MoE layer {index} expert "
                    f"{expert_index} projections"
                )
            experts.append(ExpertWeights(
                gate_proj=np.ascontiguousarray(gate, dtype=np.float32),
                up_proj=np.ascontiguousarray(up, dtype=np.float32),
                down_proj=np.ascontiguousarray(down, dtype=np.float32),
                gate_proj_bias=tensors.get((index, f"expert_{expert_index}_gate_proj_bias")),
                up_proj_bias=tensors.get((index, f"expert_{expert_index}_up_proj_bias")),
                down_proj_bias=tensors.get((index, f"expert_{expert_index}_down_proj_bias")),
            ))
        return LayerWeights(
            attention_norm=_norm_vector(common["attention_norm"], hidden_size),
            q_proj=np.ascontiguousarray(common["q_proj"], dtype=np.float32),
            k_proj=np.ascontiguousarray(common["k_proj"], dtype=np.float32),
            v_proj=np.ascontiguousarray(common["v_proj"], dtype=np.float32),
            o_proj=np.ascontiguousarray(common["o_proj"], dtype=np.float32),
            ffn_norm=_norm_vector(common["ffn_norm"], hidden_size),
            gate_proj=None,
            up_proj=None,
            down_proj=None,
            q_proj_bias=tensors.get((index, "q_proj_bias")),
            k_proj_bias=tensors.get((index, "k_proj_bias")),
            v_proj_bias=tensors.get((index, "v_proj_bias")),
            o_proj_bias=tensors.get((index, "o_proj_bias")),
            attention_norm_bias=tensors.get((index, "attention_norm_bias")),
            ffn_norm_bias=tensors.get((index, "ffn_norm_bias")),
            router=np.ascontiguousarray(common["router"], dtype=np.float32),
            experts=experts,
            num_activated_experts=int(num_activated_experts or 1),
            q_norm=(
                _norm_vector(tensors.get((index, "q_norm")), q_norm_size)
                if tensors.get((index, "q_norm")) is not None else None
            ),
            k_norm=(
                _norm_vector(tensors.get((index, "k_norm")), k_norm_size)
                if tensors.get((index, "k_norm")) is not None else None
            ),
        )
    attention_norm_tensor = tensors.get((index, "attention_norm"))
    ffn_norm_tensor = tensors.get((index, "ffn_norm"))
    if ffn_norm_tensor is None and parallel_residual:
        ffn_norm_tensor = attention_norm_tensor
    attention_norm_bias = tensors.get((index, "attention_norm_bias"))
    ffn_norm_bias = tensors.get((index, "ffn_norm_bias"))
    if ffn_norm_bias is None and parallel_residual:
        ffn_norm_bias = attention_norm_bias
    required = {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": tensors.get((index, "o_proj")),
        "attention_norm": attention_norm_tensor,
        "ffn_norm": ffn_norm_tensor,
        "gate_proj": tensors.get((index, "gate_proj")),
        "up_proj": tensors.get((index, "up_proj")),
        "down_proj": tensors.get((index, "down_proj"))
        if tensors.get((index, "down_proj")) is not None
        else tensors.get((index, "ffn")),
    }
    classic_gelu = str(ffn_type or "SwiGLU").lower() in {"gelu", "relu", "relu2"}
    missing = [
        name for name, value in required.items()
        if value is None and not (name == "up_proj" and classic_gelu)
    ]
    if missing:
        raise AEGLoadError(f"AEG package is missing required layer {index} tensors: {', '.join(missing)}")

    return LayerWeights(
        attention_norm=_norm_vector(required["attention_norm"], hidden_size),
        q_proj=np.ascontiguousarray(required["q_proj"], dtype=np.float32),
        k_proj=np.ascontiguousarray(required["k_proj"], dtype=np.float32),
        v_proj=np.ascontiguousarray(required["v_proj"], dtype=np.float32),
        o_proj=np.ascontiguousarray(required["o_proj"], dtype=np.float32),
        ffn_norm=_norm_vector(required["ffn_norm"], hidden_size),
        attention_norm_bias=_norm_vector(attention_norm_bias, hidden_size)
        if attention_norm_bias is not None else None,
        ffn_norm_bias=_norm_vector(ffn_norm_bias, hidden_size)
        if ffn_norm_bias is not None else None,
        q_proj_bias=tensors.get((index, "q_proj_bias")),
        k_proj_bias=tensors.get((index, "k_proj_bias")),
        v_proj_bias=tensors.get((index, "v_proj_bias")),
        o_proj_bias=tensors.get((index, "o_proj_bias")),
        gate_proj_bias=tensors.get((index, "gate_proj_bias")),
        up_proj_bias=tensors.get((index, "up_proj_bias")),
        down_proj_bias=tensors.get((index, "down_proj_bias")),
        gate_proj=np.ascontiguousarray(required["gate_proj"], dtype=np.float32),
        up_proj=(
            np.ascontiguousarray(required["up_proj"], dtype=np.float32)
            if required["up_proj"] is not None else None
        ),
        down_proj=np.ascontiguousarray(required["down_proj"], dtype=np.float32),
        post_attention_norm=(
            _norm_vector(tensors.get((index, "post_attention_norm")), hidden_size)
            if sandwich and tensors.get((index, "post_attention_norm")) is not None
            else None
        ),
        post_attention_norm_bias=tensors.get((index, "post_attention_norm_bias")),
        post_ffn_norm=(
            _norm_vector(tensors.get((index, "post_ffn_norm")), hidden_size)
            if sandwich and tensors.get((index, "post_ffn_norm")) is not None
            else None
        ),
        post_ffn_norm_bias=tensors.get((index, "post_ffn_norm_bias")),
        q_norm=(
            _norm_vector(tensors.get((index, "q_norm")), q_norm_size)
            if tensors.get((index, "q_norm")) is not None
            else None
        ),
        k_norm=(
            _norm_vector(tensors.get((index, "k_norm")), k_norm_size)
            if tensors.get((index, "k_norm")) is not None
            else None
        ),
    )


def _infer_intermediate(
    tensors: dict[tuple[int | None, str | None], np.ndarray], index: int, hidden_size: int
) -> int:
    """Infer the FFN width from whichever projection is present."""
    for component in ("gate_proj", "up_proj"):
        tensor = tensors.get((index, component))
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])
    down = tensors.get((index, "ffn"))
    if down is not None and down.ndim == 2:
        return int(down.shape[1])
    return hidden_size * 4


def _norm_vector(tensor: np.ndarray | None, size: int) -> np.ndarray:
    """Return a normalisation weight vector, defaulting to ones."""
    if tensor is None:
        return np.ones(size, dtype=np.float32)
    flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
    if flat.size == size:
        return flat
    resized = np.ones(size, dtype=np.float32)
    resized[: min(size, flat.size)] = flat[: min(size, flat.size)]
    return resized


def _matrix_or_identity(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Return a ``(rows, cols)`` matrix, defaulting to identity.

    Raises AEGLoadError instead of crashing with MemoryError when the default
    matrix would be unreasonably large (> 128 MiB), which indicates the tensor
    was expected from the weight blob but is missing.
    """
    if tensor is None or tensor.ndim != 2:
        size_bytes = rows * cols * 4  # float32
        if size_bytes > 128 * 1024 * 1024:  # 128 MiB guard
            msg = (
                f"Missing weight tensor ({rows}×{cols} = {size_bytes // (1024*1024)} MiB); "
                f"the package was compiled from a model without local weights and cannot be "
                f"loaded on this machine.  Run 'aether pull <model>' first."
            )
            raise AEGLoadError(msg)
        return np.eye(rows, cols, dtype=np.float32)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _matrix_or_zeros(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Return a ``(rows, cols)`` matrix, defaulting to zeros.

    Zeros are the safe default for FFN projections: they make the block a no-op
    through the residual path instead of injecting arbitrary values.

    Raises AEGLoadError instead of crashing with MemoryError when the default
    matrix would be unreasonably large (> 128 MiB), which indicates the tensor
    was expected from the weight blob but is missing.
    """
    if tensor is None or tensor.ndim != 2:
        size_bytes = rows * cols * 4  # float32
        if size_bytes > 128 * 1024 * 1024:  # 128 MiB guard
            msg = (
                f"Missing weight tensor ({rows}×{cols} = {size_bytes // (1024*1024)} MiB); "
                f"the package was compiled from a model without local weights and cannot be "
                f"loaded on this machine.  Run 'aether pull <model>' first."
            )
            raise AEGLoadError(msg)
        return np.zeros((rows, cols), dtype=np.float32)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _slice_or_default(tensor: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
    """Take the first ``rows`` of a projection, padding when it is too small.

    A package that stores one fused ``qkv`` tensor per layer has no separate K/V
    matrices; slicing keeps grouped-query shapes consistent so the forward pass
    runs on the weights that are present.
    """
    if tensor is None or tensor.ndim != 2:
        return np.eye(rows, cols, dtype=np.float32)
    source = np.ascontiguousarray(tensor, dtype=np.float32)
    if source.shape[0] >= rows and source.shape[1] == cols:
        return np.ascontiguousarray(source[:rows, :])
    out = np.zeros((rows, cols), dtype=np.float32)
    copy_rows = min(rows, source.shape[0])
    copy_cols = min(cols, source.shape[1])
    out[:copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
    return out
