"""
Aether Runtime — SSM / Mamba / Jamba / RWKV Loader.

Implements graph extraction for State Space Models (SSMs) and their hybrid variants:
  - Mamba (SSM-only, Gu & Dao 2023)
  - Mamba-2 (SSD architecture, Dao & Gu 2024)
  - Jamba (SSM + attention hybrid, AI21 Labs 2024)
  - Falcon Mamba (SSM variant, TII 2024)
  - RWKV-6 (RNN-like LM with linear attention)
  - Griffin (linear recurrence + attention, DeepMind 2024)
  - Samba (SSM + sliding window attention)

SSM operations use dedicated AEG IR opcodes distinct from transformer operations:
  - aeg.ssm.selective_scan — core Mamba selective scan
  - aeg.ssm.ssd — Mamba-2 structured state space dual
  - aeg.rwkv.time_mix — RWKV time mixing
  - aeg.rwkv.channel_mix — RWKV channel mixing
  - aeg.linear_recurrence — Griffin/Hawk linear recurrence

Research basis:
  - Mamba: Gu & Dao (2023) - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  - Mamba-2: Dao & Gu (2024) - "Transformers are SSMs: Generalized Models and Efficient Algorithms"
  - Jamba: Lieber et al. (2024) - "Jamba: A Hybrid Transformer-Mamba Language Model"
  - RWKV-6: Peng et al. (2023)
  - Griffin: De et al. (2024) - "Griffin: Mixing Gated Linear Recurrences with Local Attention"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SSM architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class SSMArchitecture:
    """Describes the architecture of an SSM-based model."""

    model_type: str  # "mamba", "mamba2", "jamba", "rwkv", "griffin", "falcon_mamba"
    ssm_variant: str  # "selective_scan", "ssd", "rwkv_time_mix", "linear_recurrence"
    num_layers: int
    hidden_size: int
    intermediate_size: int
    state_size: int  # SSM state dimension (d_state in Mamba)
    dt_rank: int | str  # delta t rank ("auto" or int)
    conv_kernel_size: int = 4
    expand_factor: float = 2.0  # Expansion factor for inner dimension
    vocab_size: int = 50280
    family: str = "ssm"
    is_hybrid: bool = False  # True if SSM+attention layers are interleaved
    attention_layers: list[int] = field(default_factory=list)  # Which layers use attention
    num_attention_heads: int = 0
    head_dim: int | None = None
    tie_embeddings: bool = True
    rms_norm: bool = True
    # RWKV-specific
    num_key_value_heads: int | None = None
    # Mamba-specific
    ngroups: int = 1  # For Mamba-2 SSD
    chunk_size: int = 256  # For Mamba-2 SSD chunked computation
    # Config metadata
    original_config: dict[str, Any] = field(default_factory=dict)

    @property
    def inner_dim(self) -> int:
        """Inner (expanded) dimension."""
        return int(self.hidden_size * self.expand_factor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "ssm_variant": self.ssm_variant,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "state_size": self.state_size,
            "is_hybrid": self.is_hybrid,
            "attention_layers": self.attention_layers,
            "family": self.family,
        }


# ---------------------------------------------------------------------------
# SSM architecture detector
# ---------------------------------------------------------------------------

_SSM_TYPE_MAP: dict[str, dict[str, Any]] = {
    "mamba": {
        "ssm_variant": "selective_scan",
        "is_hybrid": False,
        "default_state_size": 16,
        "default_conv_kernel": 4,
        "default_expand": 2.0,
    },
    "mamba2": {
        "ssm_variant": "ssd",
        "is_hybrid": False,
        "default_state_size": 128,
        "default_conv_kernel": 4,
        "ngroups": 8,
        "chunk_size": 256,
    },
    "falcon_mamba": {
        "ssm_variant": "selective_scan",
        "is_hybrid": False,
        "default_state_size": 16,
        "default_conv_kernel": 4,
    },
    "jamba": {
        "ssm_variant": "selective_scan",
        "is_hybrid": True,
        "default_state_size": 16,
        "default_conv_kernel": 4,
        # Jamba has 1 attention layer per 8 SSM layers
        "attention_period": 8,
    },
    "rwkv": {
        "ssm_variant": "rwkv_time_mix",
        "is_hybrid": False,
        "default_state_size": 64,
    },
    "griffin": {
        "ssm_variant": "linear_recurrence",
        "is_hybrid": True,
        "default_state_size": 64,
        # Griffin alternates: 2 recurrence + 1 local attention
        "attention_period": 3,
    },
    "samba": {
        "ssm_variant": "selective_scan",
        "is_hybrid": True,
        "default_state_size": 16,
        "attention_period": 4,
    },
}

_SSM_ALIASES: dict[str, str] = {
    "mamba-2": "mamba2",
    "mamba_2": "mamba2",
    "falcon-mamba": "falcon_mamba",
    "rwkv6": "rwkv",
    "rwkv_6": "rwkv",
    "hawk": "griffin",
}


def detect_ssm_architecture(model_path: str | Path) -> SSMArchitecture | None:
    """
    Detect SSM architecture from a model directory.

    Returns None if not a recognized SSM architecture.
    """
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except Exception:  # noqa: BLE001
        return None

    model_type = config.get("model_type", "").lower()
    model_type = _SSM_ALIASES.get(model_type, model_type)

    if model_type not in _SSM_TYPE_MAP:
        return None

    spec = _SSM_TYPE_MAP[model_type]
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
    hidden_size = config.get("hidden_size", config.get("d_model", 1024))
    intermediate_size = config.get("intermediate_size", int(hidden_size * 2.0 * spec.get("default_expand", 2.0)))
    state_size = config.get("d_state", config.get("state_size", spec["default_state_size"]))
    conv_kernel = config.get("d_conv", spec.get("default_conv_kernel", 4))
    dt_rank = config.get("dt_rank", "auto")
    vocab_size = config.get("vocab_size", 50280)

    # Determine which layers use attention (for hybrids)
    attention_layers: list[int] = []
    if spec["is_hybrid"]:
        period = spec.get("attention_period", 8)
        # Explicit attention layer list from config (Jamba style)
        if "attn_layer_offset" in config:
            offset = config["attn_layer_offset"]
            attn_period = config.get("attn_layer_period", period)
            attention_layers = list(range(offset, num_layers, attn_period))
        else:
            attention_layers = list(range(period - 1, num_layers, period))

    # Attention head info for hybrid models
    num_heads = config.get("num_attention_heads", 0)
    kv_heads = config.get("num_key_value_heads", num_heads)

    arch = SSMArchitecture(
        model_type=model_type,
        ssm_variant=spec["ssm_variant"],
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        state_size=state_size,
        dt_rank=dt_rank,
        conv_kernel_size=conv_kernel,
        expand_factor=spec.get("default_expand", 2.0),
        vocab_size=vocab_size,
        is_hybrid=spec["is_hybrid"],
        attention_layers=attention_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=kv_heads,
        ngroups=spec.get("ngroups", 1),
        chunk_size=spec.get("chunk_size", 256),
        original_config=config,
    )
    return arch


# ---------------------------------------------------------------------------
# SSM graph builder
# ---------------------------------------------------------------------------

@dataclass
class SSMGraphNode:
    """A node in an SSM computation graph."""

    node_id: str
    node_type: str
    op: str
    attrs: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


class SSMGraphBuilder:
    """
    Builds a computation graph for an SSM architecture.

    The graph is organized as a sequence of layers, each being either:
    - A selective scan / SSM block (Mamba-style)
    - A self-attention block (for hybrid attention layers)
    - An RWKV time/channel mixing block
    - A linear recurrence block (Griffin)
    """

    def build(self, arch: SSMArchitecture) -> list[SSMGraphNode]:
        nodes: list[SSMGraphNode] = []

        # Embedding
        nodes.append(SSMGraphNode(
            node_id="embedding",
            node_type="embedding",
            op="aeg.embedding_lookup",
            attrs={
                "vocab_size": arch.vocab_size,
                "hidden_size": arch.hidden_size,
            },
        ))

        # Build layers
        for layer_idx in range(arch.num_layers):
            is_attention = layer_idx in arch.attention_layers

            if is_attention and arch.is_hybrid:
                nodes.extend(self._build_attention_layer(arch, layer_idx))
            else:
                nodes.extend(self._build_ssm_layer(arch, layer_idx))

        # LM head
        nodes.append(SSMGraphNode(
            node_id="lm_head",
            node_type="output",
            op="aeg.lm_head",
            attrs={
                "vocab_size": arch.vocab_size,
                "hidden_size": arch.hidden_size,
                "tied": arch.tie_embeddings,
            },
        ))

        return nodes

    def _build_ssm_layer(self, arch: SSMArchitecture, layer_idx: int) -> list[SSMGraphNode]:
        """Build nodes for a single SSM layer."""
        prefix = f"layer_{layer_idx}_ssm"
        nodes = []

        # Normalization
        nodes.append(SSMGraphNode(
            node_id=f"{prefix}_norm",
            node_type="normalization",
            op="aeg.rms_norm" if arch.rms_norm else "aeg.layer_norm",
            attrs={"hidden_size": arch.hidden_size},
            inputs=[f"layer_{layer_idx - 1}_output" if layer_idx > 0 else "embedding"],
        ))

        # SSM-specific operations
        if arch.ssm_variant == "selective_scan":
            # Mamba selective scan
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_in_proj",
                node_type="linear",
                op="aeg.linear",
                attrs={
                    "in_features": arch.hidden_size,
                    "out_features": arch.inner_dim * 2,  # x + z (gating)
                },
                inputs=[f"{prefix}_norm"],
            ))
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_conv1d",
                node_type="convolution",
                op="aeg.conv1d",
                attrs={"kernel_size": arch.conv_kernel_size, "channels": arch.inner_dim},
                inputs=[f"{prefix}_in_proj"],
            ))
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_ssm",
                node_type="ssm",
                op="aeg.ssm.selective_scan",
                attrs={
                    "d_model": arch.inner_dim,
                    "d_state": arch.state_size,
                    "dt_rank": arch.dt_rank,
                    "layer_idx": layer_idx,
                },
                inputs=[f"{prefix}_conv1d"],
            ))
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_out_proj",
                node_type="linear",
                op="aeg.linear",
                attrs={
                    "in_features": arch.inner_dim,
                    "out_features": arch.hidden_size,
                },
                inputs=[f"{prefix}_ssm"],
            ))

        elif arch.ssm_variant == "ssd":
            # Mamba-2 SSD
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_ssd",
                node_type="ssm",
                op="aeg.ssm.ssd",
                attrs={
                    "d_model": arch.hidden_size,
                    "d_state": arch.state_size,
                    "ngroups": arch.ngroups,
                    "chunk_size": arch.chunk_size,
                    "layer_idx": layer_idx,
                },
                inputs=[f"{prefix}_norm"],
            ))

        elif arch.ssm_variant == "rwkv_time_mix":
            # RWKV time mixing
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_time_mix",
                node_type="ssm",
                op="aeg.rwkv.time_mix",
                attrs={
                    "hidden_size": arch.hidden_size,
                    "layer_idx": layer_idx,
                    "time_mix_k": 0.5,
                    "time_mix_v": 0.5,
                    "time_mix_r": 0.5,
                },
                inputs=[f"{prefix}_norm"],
            ))
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_channel_mix",
                node_type="ssm",
                op="aeg.rwkv.channel_mix",
                attrs={"hidden_size": arch.hidden_size, "layer_idx": layer_idx},
                inputs=[f"{prefix}_time_mix"],
            ))

        elif arch.ssm_variant == "linear_recurrence":
            # Griffin linear recurrence
            nodes.append(SSMGraphNode(
                node_id=f"{prefix}_recurrence",
                node_type="ssm",
                op="aeg.linear_recurrence",
                attrs={
                    "hidden_size": arch.hidden_size,
                    "state_size": arch.state_size,
                    "layer_idx": layer_idx,
                },
                inputs=[f"{prefix}_norm"],
            ))

        # Layer output (residual connection)
        nodes.append(SSMGraphNode(
            node_id=f"layer_{layer_idx}_output",
            node_type="residual",
            op="aeg.add",
            inputs=[f"{prefix}_out_proj" if arch.ssm_variant == "selective_scan" else f"{prefix}_ssm",
                    f"layer_{layer_idx - 1}_output" if layer_idx > 0 else "embedding"],
        ))

        return nodes

    def _build_attention_layer(self, arch: SSMArchitecture, layer_idx: int) -> list[SSMGraphNode]:
        """Build nodes for an attention layer in a hybrid SSM model."""
        prefix = f"layer_{layer_idx}_attn"
        nodes = []

        nodes.append(SSMGraphNode(
            node_id=f"{prefix}_norm",
            node_type="normalization",
            op="aeg.rms_norm",
            attrs={"hidden_size": arch.hidden_size},
            inputs=[f"layer_{layer_idx - 1}_output" if layer_idx > 0 else "embedding"],
        ))
        nodes.append(SSMGraphNode(
            node_id=f"{prefix}_qkv",
            node_type="linear",
            op="aeg.linear",
            attrs={
                "in_features": arch.hidden_size,
                "out_features": arch.hidden_size * 3,
            },
            inputs=[f"{prefix}_norm"],
        ))
        nodes.append(SSMGraphNode(
            node_id=f"{prefix}_attention",
            node_type="attention",
            op="aeg.attention",
            attrs={
                "num_heads": arch.num_attention_heads,
                "head_dim": arch.head_dim or (arch.hidden_size // max(arch.num_attention_heads, 1)),
                "layer_idx": layer_idx,
            },
            inputs=[f"{prefix}_qkv"],
        ))
        nodes.append(SSMGraphNode(
            node_id=f"layer_{layer_idx}_output",
            node_type="residual",
            op="aeg.add",
            inputs=[f"{prefix}_attention",
                    f"layer_{layer_idx - 1}_output" if layer_idx > 0 else "embedding"],
        ))
        return nodes


# ---------------------------------------------------------------------------
# SSM loader entry point
# ---------------------------------------------------------------------------

class SSMLoader:
    """
    Complete SSM / Mamba / RWKV / Griffin model loader.

    Detects SSM architecture and builds the computation graph for compiler
    ingestion. Works for pure SSM models and hybrid SSM+attention models.
    """

    def __init__(self) -> None:
        self._builder = SSMGraphBuilder()

    def load(
        self,
        model_path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> tuple[SSMArchitecture, list[SSMGraphNode]] | None:
        """
        Load an SSM model from a directory.

        Returns:
            (architecture, graph_nodes) or None if not an SSM model.
        """
        arch = detect_ssm_architecture(model_path)
        if arch is None:
            return None

        logger.info(
            f"Loading SSM: {arch.model_type} | "
            f"variant={arch.ssm_variant} | "
            f"layers={arch.num_layers} | "
            f"d_model={arch.hidden_size} | "
            f"d_state={arch.state_size} | "
            f"hybrid={arch.is_hybrid} | "
            f"attention_layers={len(arch.attention_layers)}"
        )

        nodes = self._builder.build(arch)
        return arch, nodes

    def is_ssm(self, model_path: str | Path) -> bool:
        """Check if a model directory contains an SSM model."""
        return detect_ssm_architecture(model_path) is not None

    @staticmethod
    def list_supported_types() -> list[str]:
        """Return supported SSM model types."""
        return sorted(_SSM_TYPE_MAP.keys())
