"""Small authenticated AEGs for every normalization-epsilon loader branch.

These isolate loader configuration, not trained-model quality. All tensors are
real FP32 arrays; end-to-end source-model parity is tested separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from aether.core.aeg_format import AEGPackage
from aether.core.aeg_ir import AEGIRModule
from aether.core.types import ModelArchitecture
from aether.quantization.formats import quantize_tensor

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

FAMILIES = ("dense", "moe", "mla", "mamba", "mamba2", "hybrid", "rwkv", "encoder", "seq2seq")


def write_package(path: Path, family: str, epsilon: float | None) -> AEGPackage:
    """Persist actual weight/index/manifest/IR bytes, then reopen and authenticate."""
    hidden, inner, state, vocab = 8, 16, 2, 11
    rng = np.random.default_rng(20261023)

    def matrix(*shape: int) -> NDArray[np.float32]:
        return rng.normal(0, 0.1, shape).astype(np.float32)

    def norm(size: int) -> NDArray[np.float32]:
        return np.linspace(0.7, 1.3, size, dtype=np.float32)

    architecture: dict[str, Any] = {
        "family": family, "layers": 1,
        "hidden_size": hidden, "num_attention_heads": 2, "num_kv_heads": 2,
        "head_dim": 4, "intermediate_size": inner, "vocab_size": vocab,
        "norm_eps": epsilon, "context_length": 32, "tie_word_embeddings": False,
        "norm_type": "RMSNorm", "position_type": "none",
    }
    tensors = {"embedding": matrix(vocab, hidden), "final_norm": norm(hidden),
               "lm_head": matrix(vocab, hidden)}

    def dense(index: int) -> None:
        prefix = f"layer_{index}_"
        tensors.update({prefix + name: matrix(hidden, hidden)
                        for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
        tensors.update({prefix + name: norm(hidden) for name in ("attention_norm", "ffn_norm")})
        tensors.update({prefix + name: matrix(inner, hidden) for name in ("gate_proj", "up_proj")})
        tensors[prefix + "down_proj"] = matrix(hidden, inner)

    def mamba(index: int) -> None:
        prefix = f"layer_{index}_ssm_"
        tensors.update({
            prefix + "norm": norm(hidden), prefix + "in_proj": matrix(2 * inner, hidden),
            prefix + "conv1d": matrix(inner, 3), prefix + "x_proj": matrix(1 + 2 * state, inner),
            prefix + "dt_proj": matrix(inner, 1), prefix + "a_log": matrix(inner, state),
            prefix + "d": norm(inner), prefix + "out_proj": matrix(hidden, inner),
        })

    if family in {"dense", "moe"}:
        dense(0)
        if family == "moe":
            architecture.update(is_moe=True, num_experts=2, num_activated_experts=1)
            tensors["layer_0_moe_router"] = matrix(2, hidden)
            for index in range(2):
                for component, shape in (("gate_proj", (inner, hidden)),
                                         ("up_proj", (inner, hidden)),
                                         ("down_proj", (hidden, inner))):
                    tensors[f"layer_0_expert_{index}_{component}"] = matrix(*shape)
    elif family == "mla":
        architecture.update(attention_type="MLA", mla_kv_lora_rank=2, mla_q_lora_rank=2,
                            mla_qk_nope_head_dim=2, mla_qk_rope_head_dim=2, mla_v_head_dim=2)
        for name in ("attention_norm", "ffn_norm"):
            tensors["layer_0_" + name] = norm(hidden)
        for name, shape in {
            "o_proj": (hidden, 4), "q_a_proj": (2, hidden), "q_b_proj": (8, 2),
            "kv_a_proj": (2, hidden), "kv_b_proj": (8, 2), "k_rope_proj": (2, hidden),
            "gate_proj": (inner, hidden), "up_proj": (inner, hidden), "down_proj": (hidden, inner),
        }.items():
            tensors["layer_0_" + name] = matrix(*shape)
        tensors.update(layer_0_q_a_norm=norm(2), layer_0_kv_a_norm=norm(2))
    elif family in {"mamba", "hybrid", "mamba2"}:
        architecture.update(ssm_variant="selective_scan", ssm_state_size=state,
                            ssm_inner_size=inner, ssm_dt_rank=1, ssm_conv_kernel=3)
        if family == "mamba2":
            architecture.update(
                ssm_variant="ssd", ssm_num_heads=2, ssm_num_groups=1, ssm_head_dim=8,
            )
            tensors.update({
                "layer_0_ssm_norm": norm(hidden), "layer_0_ssm_output_norm": norm(inner),
                "layer_0_ssm_in_proj": matrix(2 * inner + 2 * state + 2, hidden),
                "layer_0_ssm_conv1d": matrix(inner + 2 * state, 3),
                "layer_0_ssm_a_log": matrix(2), "layer_0_ssm_d": norm(2),
                "layer_0_ssm_dt": matrix(2), "layer_0_ssm_out_proj": matrix(hidden, inner),
            })
        else:
            mamba(0)
            if family == "hybrid":
                architecture.update(ssm_variant="hybrid_selective_scan", layers=2,
                                    hybrid_layer_types=["ssm", "attention"])
                dense(1)
    elif family == "rwkv":
        architecture.update(ssm_variant="rwkv_time_mix", norm_type="LayerNorm")
        for name in ("norm", "ffn_norm", "time_mix_k", "time_mix_v", "time_mix_r",
                     "ffn_time_mix_k", "ffn_time_mix_r"):
            tensors["layer_0_ssm_" + name] = norm(hidden)
        for name in ("time_decay", "time_first"):
            tensors["layer_0_ssm_" + name] = matrix(hidden)
        for name in ("key", "value", "receptance", "output", "ffn_receptance"):
            tensors["layer_0_ssm_" + name] = matrix(hidden, hidden)
        tensors.update(layer_0_ssm_ffn_key=matrix(inner, hidden),
                       layer_0_ssm_ffn_value=matrix(hidden, inner))
    elif family == "encoder":
        architecture.update(is_encoder=True, norm_type="LayerNorm")
        tensors.update(
            position_embedding=matrix(32, hidden), token_type_embedding=matrix(2, hidden),
            embedding_norm=norm(hidden), pooler=matrix(hidden, hidden),
        )
        dense(0)
        tensors.update(layer_0_intermediate_proj=matrix(inner, hidden),
                       layer_0_output_proj=matrix(hidden, inner), layer_0_output_norm=norm(hidden))
    elif family == "seq2seq":
        architecture.update(is_encoder_decoder=True, encoder_layers=1, decoder_layers=1,
                            ffn_type="ReLU")
        tensors["encoder_final_norm"] = norm(hidden)
        for name in ("norm1", "norm2"):
            tensors["encoder_layer_0_" + name] = norm(hidden)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors["encoder_layer_0_" + name] = matrix(hidden, hidden)
        for name in ("self_norm", "cross_norm", "ffn_norm"):
            tensors["decoder_layer_0_" + name] = norm(hidden)
        for block in ("self", "cross"):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tensors[f"decoder_layer_0_{block}_{projection}"] = matrix(hidden, hidden)
        for prefix in ("encoder_layer_0_", "decoder_layer_0_"):
            tensors[prefix + "ffn_in"] = matrix(inner, hidden)
            tensors[prefix + "ffn_out"] = matrix(hidden, inner)
    else:
        raise ValueError(f"Unknown fixture family {family!r}")

    architecture["params_billion"] = sum(value.size for value in tensors.values()) / 1e9
    package = AEGPackage.create(path, model_id="epsilon-" + family, aether_version="0.1.0")
    assert package.manifest is not None
    package.manifest.architecture = ModelArchitecture.from_dict(architecture)
    package.set_ir(AEGIRModule(version="AEG-IR/1.0"))
    package.weights = {name: quantize_tensor(value, "FP32") for name, value in tensors.items()}
    package.save()
    reloaded = AEGPackage(path).load()
    reloaded.verify_integrity()
    return reloaded
