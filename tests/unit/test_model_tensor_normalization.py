"""Cross-family checkpoint naming tests for the model-generic ingestion ABI."""

from __future__ import annotations

import pytest

from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline


@pytest.mark.parametrize(
    ("checkpoint_name", "expected"),
    [
        ("model.layers.0.self_attn.q_proj.weight", (0, "qkv")),
        ("model.layers.0.self_attn.k_proj.weight", (0, "qkv")),
        ("model.layers.0.self_attn.o_proj.weight", (0, "out_proj")),
        ("model.layers.0.mlp.gate_proj.weight", (0, "gate_proj")),
        ("model.layers.0.mlp.up_proj.weight", (0, "gate_proj")),
        ("model.layers.0.mlp.down_proj.weight", (0, "ffn")),
        ("transformer.h.0.attn.c_attn.weight", (0, "qkv")),
        ("transformer.h.0.attn.c_proj.weight", (0, "out_proj")),
        ("transformer.h.0.mlp.c_fc.weight", (0, "gate_proj")),
        ("transformer.h.0.mlp.c_proj.weight", (0, "ffn")),
        ("transformer.h.0.attention.query_key_value.weight", (0, "qkv")),
        ("transformer.h.0.mlp.dense_h_to_4h.weight", (0, "gate_proj")),
        ("transformer.h.0.mlp.dense_4h_to_h.weight", (0, "ffn")),
        ("transformer.h.0.self_attention.dense.weight", (0, "out_proj")),
        ("model.layers.0.attention.wqkv.weight", (0, "qkv")),
        ("model.layers.0.attention.wo.weight", (0, "out_proj")),
        ("model.layers.0.attention_norm.weight", (0, "rmsnorm")),
        ("model.layers.0.feed_forward.w1.weight", (0, "gate_proj")),
        ("model.layers.0.feed_forward.w2.weight", (0, "ffn")),
        ("model.layers.0.feed_forward.w3.weight", (0, "gate_proj")),
        ("transformer.h.0.mlp.fc_in.weight", (0, "gate_proj")),
        ("transformer.h.0.mlp.fc_out.weight", (0, "ffn")),
        ("model.layers.0.block_sparse_moe.gate.weight", (0, "moe_router")),
        ("model.layers.0.block_sparse_moe.experts.3.w1.weight", (0, "expert_3_gate_proj")),
        ("model.layers.0.block_sparse_moe.experts.3.w2.weight", (0, "expert_3_down_proj")),
        ("model.layers.0.block_sparse_moe.experts.3.w3.weight", (0, "expert_3_up_proj")),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", (0, "kv_a_proj")),
        ("backbone.layers.0.mixer.A_log", (0, "ssm_a_log")),
        ("backbone.layers.0.mixer.out_proj.weight", (0, "ssm_out_proj")),
    ],
)
def test_checkpoint_spelling_normalizes_to_capability_contract(
    checkpoint_name: str, expected: tuple[int | None, str | None]
) -> None:
    assert IngestionPipeline._normalise_weight_name(checkpoint_name) == expected


def test_unrelated_tensor_is_not_silently_assigned_to_a_layer() -> None:
    assert IngestionPipeline._normalise_weight_name("vision.encoder.random.weight") == (None, None)
