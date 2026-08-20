"""Capability-driven model-family coverage.

These tests intentionally exercise architecture metadata, not model-name
fallback geometry.  Every family is represented by the architecture class or
model type that a real Hugging Face checkpoint declares.  The test proves that
Aether classifies the family without assuming Qwen dimensions; executable
weight/runtime coverage remains in the local SafeTensors integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector


STANDARD_DECODER_CASES = [
    ("olmo", "OlmoForCausalLM"), ("olmoe", "OlmoeForCausalLM"),
    ("command", "CommandRForCausalLM"), ("granite", "GraniteForCausalLM"),
    ("yi", "YiForCausalLM"), ("internlm", "InternLMForCausalLM"),
    ("minicpm", "MiniCPMForCausalLM"), ("smollm", "SmolLMForCausalLM"),
    ("pythia", "PythiaForCausalLM"), ("gptj", "GPTJForCausalLM"),
    ("bloom", "BloomForCausalLM"), ("mpt", "MptForCausalLM"),
    ("redpajama", "RedPajamaForCausalLM"), ("openelm", "OpenELMForCausalLM"),
    ("stablelm", "StableLmForCausalLM"), ("starcoder", "Starcoder2ForCausalLM"),
    ("opt", "OPTForCausalLM"), ("gpt_oss", "GptOssForCausalLM"),
    ("glm", "GlmForCausalLM"), ("kimi", "KimiForCausalLM"),
    ("hunyuan", "HunYuanForCausalLM"), ("minimax", "MiniMaxForCausalLM"),
    ("exaone", "ExaoneForCausalLM"), ("sola", "SolarForCausalLM"),
    ("jais", "JaisForCausalLM"), ("seallm", "SeaLLMForCausalLM"),
    ("aya", "AyaForCausalLM"), ("nous", "NousForCausalLM"),
    ("openchat", "OpenChatForCausalLM"), ("zephyr", "ZephyrForCausalLM"),
    ("dolphin", "DolphinForCausalLM"), ("tulu", "TuluForCausalLM"),
    ("tinyllama", "TinyLlamaForCausalLM"), ("mobilellm", "MobileLLMForCausalLM"),
    ("bitnet", "BitNetForCausalLM"), ("liquid", "LiquidForCausalLM"),
    ("nemotron", "NemotronForCausalLM"), ("megatron", "MegatronForCausalLM"),
    ("apertus", "ApertusForCausalLM"), ("sarvam", "SarvamForCausalLM"),
    ("step", "Step1ForCausalLM"), ("arctic", "ArcticForCausalLM"),
    ("stepfun", "StepFunForCausalLM"), ("grok", "GrokForCausalLM"),
    ("nvidia", "NVIDIAForCausalLM"),
    ("hyperclova", "HyperCLOVAForCausalLM"), ("codestral", "CodestralForCausalLM"),
    ("codegeex", "CodeGeeXForCausalLM"), ("dbrx", "DbrxForCausalLM"),
]


KNOWN_DECODER_CASES = [
    ("qwen", "Qwen3ForCausalLM"), ("deepseek", "DeepseekForCausalLM"),
    ("llama", "LlamaForCausalLM"), ("gemma", "GemmaForCausalLM"),
    ("mistral", "MistralForCausalLM"), ("mixtral", "MixtralForCausalLM"),
    ("phi", "Phi3ForCausalLM"), ("falcon", "FalconForCausalLM"),
    ("gpt2", "GPT2LMHeadModel"), ("gpt_neox", "GPTNeoXForCausalLM"),
    ("mamba", "MambaForCausalLM"), ("jamba", "JambaForCausalLM"),
    ("rwkv", "RwkvForCausalLM"),
]


ENCODER_CASES = [
    ("bert", "BertModel"), ("roberta", "RobertaModel"),
    ("deberta", "DebertaV2Model"), ("albert", "AlbertModel"),
    ("electra", "ElectraModel"), ("xlnet", "XLNetModel"),
]


ENCODER_DECODER_CASES = [
    ("t5", "T5ForConditionalGeneration"),
    ("flan_t5", "T5ForConditionalGeneration"),
    ("mt5", "MT5ForConditionalGeneration"),
    ("byt5", "ByT5ForConditionalGeneration"),
    ("ul2", "UL2ForConditionalGeneration"),
]


def _config(architecture: str, model_type: str) -> dict[str, object]:
    return {
        "architectures": [architecture],
        "model_type": model_type,
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 32,
    }


@pytest.mark.parametrize("model_type,architecture", STANDARD_DECODER_CASES)
def test_requested_standard_decoder_families_are_detected(
    tmp_path: Path, model_type: str, architecture: str
) -> None:
    path = tmp_path / model_type.replace(" ", "_")
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(_config(architecture, model_type)), encoding="utf-8"
    )
    detected = ArchitectureDetector().detect(str(path))
    assert not detected.is_encoder
    assert detected.family != "encoder_decoder_family"
    assert detected.family != "qwen_family"
    assert detected.hidden_size == 16
    assert detected.layers == 1


@pytest.mark.parametrize("model_type,architecture", KNOWN_DECODER_CASES)
def test_existing_decoder_families_remain_distinct(
    tmp_path: Path, model_type: str, architecture: str
) -> None:
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(_config(architecture, model_type)), encoding="utf-8"
    )
    detected = ArchitectureDetector().detect(str(path))
    assert detected.family != "generic_decoder_family"


@pytest.mark.parametrize("model_type,architecture", ENCODER_CASES)
def test_encoder_families_are_not_classified_as_decoders(
    tmp_path: Path, model_type: str, architecture: str
) -> None:
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(_config(architecture, model_type)), encoding="utf-8"
    )
    detected = ArchitectureDetector().detect(str(path))
    assert detected.is_encoder


@pytest.mark.parametrize("model_type,architecture", ENCODER_DECODER_CASES)
def test_encoder_decoder_families_are_explicitly_classified(
    tmp_path: Path, model_type: str, architecture: str
) -> None:
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(_config(architecture, model_type)), encoding="utf-8"
    )
    detected = ArchitectureDetector().detect(str(path))
    assert detected.family == "encoder_decoder_family"
    assert detected.is_encoder_decoder


def test_unknown_declared_causal_lm_uses_capability_driven_generic_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "new-family"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(_config("AcmeTransformerForCausalLM", "acme_transformer")),
        encoding="utf-8",
    )
    detected = ArchitectureDetector().detect(str(path))
    assert detected.family == "generic_decoder_family"
    assert not detected.is_encoder
    assert not detected.is_encoder_decoder
