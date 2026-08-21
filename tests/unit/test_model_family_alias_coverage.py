"""Regression coverage for the public model-family compatibility matrix.

The detector must classify names and architecture class spellings without
requiring Transformers or PyTorch.  Geometry and tensor layout still come
from a checkpoint config; this matrix only protects family dispatch.
"""

from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector


DECODER_FAMILIES = (
    "Qwen", "DeepSeek", "Llama", "Gemma", "Mistral", "Mixtral", "Phi",
    "OLMo", "OLMoE", "Falcon", "Command R", "Command A", "Granite",
    "Granite Code", "Granite 3", "Granite 4", "Yi", "InternLM", "MiniCPM",
    "SmolLM", "Pythia", "GPT-Neo", "GPT-J", "GPT-NeoX", "BLOOM", "BLOOMZ",
    "MPT", "RedPajama", "OpenELM", "StableLM", "GPT-2", "StarCoder",
    "StarCoder2", "Code Llama", "CodeGemma", "CodeQwen", "DeepSeek-Coder",
    "Codestral", "CodeGeeX", "WizardCoder", "WizardLM", "Vicuna", "XGen",
    "OPT", "GPT-OSS", "GLM", "GLM-4", "GLM-5", "Kimi", "Kimi-K2",
    "Hunyuan", "MiniMax", "EXAONE", "HyperCLOVA X", "Solar", "DBRX", "Grok",
    "Apertus", "Sarvam", "StepFun", "Step-1", "Nemotron", "NVIDIA Nemotron",
    "Megatron", "Arctic", "Snowflake Arctic", "Jais", "SeaLLM", "Aya",
    "Aya Expanse", "Nous Hermes", "OpenChat", "Zephyr", "Dolphin", "Tulu",
    "TinyLlama", "MobileLLM", "MobileLLM2", "MobileLLM3", "BitNet", "Liquid",
    "RecurrentGemma",
)


def test_named_decoder_families_have_dispatch_classification():
    detector = ArchitectureDetector()
    for name in DECODER_FAMILIES:
        family = detector._match_family(name.lower())
        assert family is not None, name


def test_non_decoder_families_have_their_own_dispatch_classification():
    detector = ArchitectureDetector()
    expected = {
        "BERT": "bert_family",
        "T5": "encoder_decoder_family",
        "FLAN-T5": "encoder_decoder_family",
        "mT5": "encoder_decoder_family",
        "ByT5": "encoder_decoder_family",
        "UL2": "encoder_decoder_family",
        "RoBERTa": "roberta_family",
        "DeBERTa": "deberta_family",
        "XLNet": "bert_family",
        "ELECTRA": "electra_family",
        "InternVL": "vision_family",
        "RWKV": "hybrid_ssm_family",
        "Mamba": "hybrid_ssm_family",
        "Jamba": "hybrid_ssm_family",
    }
    for name, expected_family in expected.items():
        assert detector._match_family(name.lower()) == expected_family, name


def test_variant_architecture_class_names_use_config_independent_fallback():
    detector = ArchitectureDetector()
    for name in ("OLMo", "Granite", "StarCoder2", "CodeQwen", "GLM-4", "Kimi-K2"):
        class_name = name.replace("-", "").replace(" ", "") + "ForCausalLM"
        assert detector._detect_family_from_arch_type(class_name) is not None, class_name
