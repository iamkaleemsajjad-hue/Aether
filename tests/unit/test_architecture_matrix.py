"""Offline architecture-detection contract tests for supported HF configs."""

from __future__ import annotations

import pytest

from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector


@pytest.mark.parametrize(
    ("model_type", "architecture", "family", "is_encoder"),
    [
        ("bert", "BertModel", "bert_family", True),
        ("roberta", "RobertaModel", "roberta_family", True),
        ("deberta-v2", "DebertaV2Model", "deberta_family", True),
        ("electra", "ElectraModel", "electra_family", True),
        ("albert", "AlbertModel", "albert_family", True),
        ("distilbert", "DistilBertModel", "bert_family", True),
        ("mpnet", "MPNetModel", "bert_family", True),
        ("gpt2", "GPT2LMHeadModel", "gpt_family", False),
        ("gpt_neox", "GPTNeoXForCausalLM", "gpt_family", False),
        ("llama", "LlamaForCausalLM", "llama_family", False),
    ],
)
def test_huggingface_architecture_matrix(
    model_type: str,
    architecture: str,
    family: str,
    is_encoder: bool,
) -> None:
    config = {
        "model_type": model_type,
        "architectures": [architecture],
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        # GPT-2 compatibility fields are included deliberately; the detector
        # must prefer the modern fields when both are present.
        "n_layer": 2,
        "n_embd": 32,
        "n_head": 4,
        "n_positions": 64,
    }
    detected = ArchitectureDetector()._from_config(config)
    assert detected.family == family
    assert detected.is_encoder is is_encoder
    assert detected.layers == 2
    assert detected.hidden_size == 32
    assert detected.context_length == 64


def test_gpt2_legacy_dimensions_are_normalized() -> None:
    detected = ArchitectureDetector()._from_config(
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12,
            "n_positions": 1024,
            "n_inner": 3072,
            "vocab_size": 50257,
        }
    )
    assert detected.family == "gpt_family"
    assert detected.layers == 12
    assert detected.hidden_size == 768
    assert detected.num_attention_heads == 12
    assert detected.num_kv_heads == 12
    assert detected.context_length == 1024
    assert detected.intermediate_size == 3072

