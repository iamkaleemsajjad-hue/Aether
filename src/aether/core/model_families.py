"""Authoritative registry of the model families Aether can execute.

This module is the single source of truth for the question "how many models does
Aether support?".  Every number published in ``README.md``,
``SUPPORTED_MODELS.md`` and ``docs/`` is derived from the table below, and
``tests/unit/test_model_family_registry.py`` fails the build when a document and
this registry disagree.

Three different quantities are easy to conflate, so they are named separately:

``architecture families``
    Distinct *computation contracts* — a block structure plus the scalar
    numerics that go with it.  This is the number that matters for correctness,
    because one family is one code path through the compiler and the runtime.

``detection keys``
    Names and Hugging Face ``architectures``/``model_type`` spellings the
    detector maps onto a family.  Many keys resolve to one family: a Vicuna
    checkpoint *is* a Llama block, so it adds a detection key, not a family.

``checkpoints``
    Individual published weights.  Aether does not enumerate these and makes no
    claim about their count; a family covers every checkpoint that shares its
    contract.

Support is graded, not boolean.  A family that merely runs is not the same as a
family whose logits are compared against a reference implementation, and saying
so is the difference between a supported-model list and a marketing number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

__all__ = [
    "FamilyKind",
    "SupportLevel",
    "ModelFamily",
    "MODEL_FAMILIES",
    "family_counts",
    "families_by_level",
    "families_by_kind",
    "executable_families",
    "detection_key_count",
    "support_summary",
    "SUPPORT_SENTENCE",
]


class FamilyKind(str, Enum):
    """The graph shape a family compiles to."""

    DECODER = "decoder"
    """Causal decoder-only language model."""

    ENCODER = "encoder"
    """Bidirectional encoder (no causal mask, no KV cache)."""

    ENCODER_DECODER = "encoder_decoder"
    """Encoder + cross-attending decoder (seq2seq)."""

    SSM_HYBRID = "ssm_hybrid"
    """State-space or hybrid recurrent/attention model with carried state."""

    MULTIMODAL = "multimodal"
    """Vision or audio encoder feeding a text decoder."""


class SupportLevel(str, Enum):
    """How strongly a family's output is guaranteed.

    The ordering is deliberate: only ``PARITY_VERIFIED`` licenses the claim that
    Aether reproduces the source model.
    """

    PARITY_VERIFIED = "parity_verified"
    """Per-logit comparison against the reference implementation passes on the
    CPU, PyTorch and tensor-parallel engines, for prefill and decode."""

    RUNS = "runs"
    """Compiles and executes with round-trip tests, but no automatic per-logit
    comparison against a reference implementation."""

    KNOWN_INCORRECT = "known_incorrect"
    """Ingests and executes, but its output provably diverges from the
    reference.  Present so the divergence is documented rather than implied."""

    UNSUPPORTED = "unsupported"
    """No execution path.  Compilation fails, by design, rather than producing
    an artifact that would generate wrong output silently."""


@dataclass(frozen=True)
class ModelFamily:
    """One architecture family: one computation contract, one code path."""

    key: str
    """Stable identifier used in tests and CLI output."""

    name: str
    """Human-readable family name as published in the support matrix."""

    kind: FamilyKind
    level: SupportLevel

    representative_models: tuple[str, ...] = ()
    """Published checkpoints that exercise this contract."""

    numerics: str = ""
    """The constants that distinguish this family from the baseline decoder
    block.  Getting any of them wrong changes every logit, which is why each is
    pinned by ``tests/unit/test_execution_numerics.py``."""

    detection_keys: tuple[str, ...] = ()
    """Name prefixes and Hugging Face class / ``model_type`` spellings that
    resolve to this family.  Aliases are *coverage*, not additional families."""

    aether_family: str = ""
    """The ``SUPPORTED_ARCHITECTURES`` entry this family compiles through."""

    note: str = ""
    """Why a non-verified level applies.  Required for every level other than
    ``PARITY_VERIFIED`` so no degraded status is left unexplained."""

    @property
    def is_executable(self) -> bool:
        """Whether an execution path exists at all (correct or not)."""
        return self.level is not SupportLevel.UNSUPPORTED

    @property
    def is_trusted(self) -> bool:
        """Whether output may be relied on as matching the source model."""
        return self.level is SupportLevel.PARITY_VERIFIED


def _decoder(
    key: str,
    name: str,
    models: tuple[str, ...],
    numerics: str,
    keys: tuple[str, ...],
    aether_family: str = "generic_decoder_family",
    level: SupportLevel = SupportLevel.PARITY_VERIFIED,
    note: str = "",
) -> ModelFamily:
    return ModelFamily(
        key=key,
        name=name,
        kind=FamilyKind.DECODER,
        level=level,
        representative_models=models,
        numerics=numerics,
        detection_keys=keys,
        aether_family=aether_family,
        note=note,
    )


# ── Parity-verified decoder families ──────────────────────────────────────────
# Each row is one block contract.  ``numerics`` names what makes it distinct
# from the Llama baseline; ``detection_keys`` are the spellings that reach it.

_PARITY_DECODERS: tuple[ModelFamily, ...] = (
    _decoder(
        "llama", "Llama 3.x",
        ("Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.3-70B"),
        "GQA · SwiGLU · RMSNorm · RoPE — the baseline block",
        ("llama", "llama2", "llama3", "code_llama", "codelama", "tinyllama",
         "vicuna", "zephyr", "dolphin", "tulu", "openchat", "nous_hermes",
         "openhermes", "wizardlm", "solar", "yi", "internlm", "minicpm",
         "LlamaForCausalLM"),
        aether_family="llama_family",
    ),
    _decoder(
        "qwen2", "Qwen 2 / 2.5",
        ("Qwen2-7B", "Qwen2-72B", "Qwen2.5-7B", "CodeQwen"),
        "GQA · SwiGLU; a declared sliding window applies only where the "
        "per-layer schedule enables it",
        ("qwen", "qwen2", "codeqwen", "qwen_coder", "Qwen2ForCausalLM"),
        aether_family="qwen_family",
    ),
    _decoder(
        "qwen3", "Qwen 3",
        ("Qwen3-0.6B", "Qwen3-1.5B", "Qwen3-8B", "Qwen3-32B", "Qwen3-72B"),
        "per-head Q/K normalization · checkpoint-declared head_dim decoupled "
        "from hidden_size / heads",
        ("qwen3", "Qwen3ForCausalLM"),
        aether_family="qwen_family",
    ),
    _decoder(
        "qwen3_moe", "Qwen 3 MoE",
        ("Qwen3-MoE",),
        "routed experts without top-k renormalization (norm_topk_prob: false)",
        ("qwen3_moe", "qwen2_moe", "Qwen2MoeForCausalLM"),
        aether_family="moe_family",
    ),
    _decoder(
        "mistral", "Mistral",
        ("Mistral-7B-v0.1", "Mistral-7B-v0.3", "Ministral-8B"),
        "GQA · SwiGLU",
        ("mistral", "codestral", "MistralForCausalLM"),
        aether_family="mistral_family",
    ),
    _decoder(
        "mixtral", "Mixtral",
        ("Mixtral-8x7B", "Mixtral-8x22B"),
        "top-2 of 8 routed experts with top-k renormalization",
        ("mixtral", "MixtralForCausalLM"),
        aether_family="moe_family",
    ),
    _decoder(
        "gemma2", "Gemma 2",
        ("Gemma-2-2B", "Gemma-2-9B", "Gemma-2-27B"),
        "×√H embedding scale · (1+w) norms · sandwich norm · "
        "query_pre_attn_scalar attention scale · attention and final logit "
        "soft-caps · GeGLU",
        ("gemma", "gemma2", "codegemma", "Gemma2ForCausalLM"),
        aether_family="gemma_family",
    ),
    _decoder(
        "gemma3", "Gemma 3 (text)",
        ("Gemma-3-1B", "Gemma-3-4B", "Gemma-3-12B", "Gemma-3-27B"),
        "all of Gemma 2 plus a separate rotary base for sliding-window layers",
        ("gemma3", "Gemma3ForCausalLM"),
        aether_family="gemma_family",
    ),
    _decoder(
        "gpt2", "GPT-2",
        ("GPT-2 117M", "GPT-2 1.5B", "DialoGPT"),
        "MHA · Conv1D weight layout · GELU-tanh · learned absolute positions · "
        "LayerNorm",
        ("gpt2", "GPT2LMHeadModel", "GPT2Model"),
        aether_family="gpt_family",
    ),
    _decoder(
        "gpt_neo", "GPT-Neo",
        ("GPT-Neo-125M", "GPT-Neo-1.3B", "GPT-Neo-2.7B"),
        "unscaled attention (no 1/√d) · local/global layer schedule expanded "
        "from the grouped attention_types form",
        ("gpt_neo", "GPTNeoForCausalLM"),
        aether_family="gpt_family",
    ),
    _decoder(
        "gpt_neox", "GPT-NeoX",
        ("GPT-NeoX-20B", "Pythia-70M", "Pythia-12B", "RedPajama-INCITE"),
        "25% partial rotary · per-head-interleaved fused QKV · parallel "
        "residual · exact-erf GELU",
        ("gpt_neox", "pythia", "redpajama", "GPTNeoXForCausalLM"),
        aether_family="gpt_family",
    ),
    _decoder(
        "gptj", "GPT-J",
        ("GPT-J-6B",),
        "interleaved (rotate-every-two) rotary · partial rotary · parallel "
        "residual",
        ("gptj", "GPTJForCausalLM"),
    ),
    _decoder(
        "phi3", "Phi-3 / Phi-4",
        ("Phi-3-mini", "Phi-3-small", "Phi-3-medium", "Phi-4"),
        "fused QKV in one contiguous projection · SwiGLU · LongRoPE factor "
        "tables keyed on the pretrained context length",
        ("phi", "phi3", "phi4", "Phi3ForCausalLM", "PhiForCausalLM"),
        aether_family="phi_family",
    ),
    _decoder(
        "falcon", "Falcon",
        ("Falcon-7B", "Falcon-40B"),
        "per-KV-group interleaved fused QKV · parallel residual · "
        "single or dual block norms · legacy multi_query with one KV head",
        ("falcon", "refinedweb", "FalconForCausalLM"),
        aether_family="falcon_family",
    ),
    _decoder(
        "bloom", "BLOOM",
        ("BLOOM-560M", "BLOOM-176B", "BLOOMZ"),
        "ALiBi positions · per-head-interleaved fused QKV · embedding "
        "LayerNorm · tanh GELU with no declared activation",
        ("bloom", "BloomForCausalLM"),
    ),
    _decoder(
        "mpt", "MPT",
        ("MPT-7B", "MPT-30B"),
        "ALiBi · d_model / n_heads / n_layers config spellings · nested "
        "attn_config · low-precision LayerNorm read as LayerNorm",
        ("mpt", "MptForCausalLM"),
    ),
    _decoder(
        "starcoder2", "StarCoder2",
        ("StarCoder2-3B", "StarCoder2-7B", "StarCoder2-15B"),
        "GQA · GELU-tanh · layer_types sliding-window schedule",
        ("starcoder", "starcoder2", "gpt_bigcode", "starchat",
         "Starcoder2ForCausalLM"),
    ),
    _decoder(
        "cohere", "Cohere / Command-R",
        ("Command-R", "Command-R+", "Command-A", "Aya Expanse"),
        "interleaved rotary · logit_scale · parallel residual · one block norm",
        ("cohere", "cohere2", "command", "command_r", "command_a", "aya",
         "aya_expanse", "CohereForCausalLM"),
    ),
    _decoder(
        "olmo2", "OLMo 2",
        ("OLMo-2-7B", "OLMo-2-13B"),
        "post-norm block (norms on sublayer outputs) · full-projection Q/K norm",
        ("olmo", "olmo2", "Olmo2ForCausalLM"),
    ),
    _decoder(
        "olmoe", "OLMoE",
        ("OLMoE-1B-7B",),
        "pre-norm · full-projection Q/K norm · experts without top-k "
        "renormalization",
        ("olmoe", "OlmoeForCausalLM"),
    ),
    _decoder(
        "stablelm", "StableLM",
        ("StableLM-2-1.6B", "StableLM-3B"),
        "25% partial rotary",
        ("stablelm", "StableLmForCausalLM"),
    ),
    _decoder(
        "granite", "Granite",
        ("Granite-3.0-8B", "Granite-3.1-8B", "Granite Code"),
        "explicit embedding / residual / attention / logit multipliers "
        "(logits_scaling expressed as one logit multiplier)",
        ("granite", "granite3", "granite4", "granite_code",
         "GraniteForCausalLM"),
    ),
    _decoder(
        "exaone4", "EXAONE 4",
        ("EXAONE-4-32B",),
        "post-norm · per-head Q/K norm · NoPE global layers (rotary applied "
        "only on sliding-window layers)",
        ("exaone", "exaone4", "Exaone4ForCausalLM"),
    ),
    _decoder(
        "smollm3", "SmolLM 3",
        ("SmolLM3-3B",),
        "interleaved NoPE layers among RoPE layers via the no_rope_layers "
        "enable flags",
        ("smollm", "smollm2", "smollm3", "SmolLM3ForCausalLM"),
    ),
    _decoder(
        "glm4", "GLM-4",
        ("GLM-4-9B", "GLM-4-32B"),
        "interleaved rotary · 50% partial rotary · GLM-spelled sandwich norm",
        ("glm", "glm4", "glm_4", "chatglm", "Glm4ForCausalLM"),
    ),
    _decoder(
        "nemotron", "Nemotron",
        ("Nemotron-4-15B", "Nemotron-Mini-4B"),
        "LayerNorm1P ((1+w) LayerNorm) · squared-ReLU FFN · 50% partial rotary",
        ("nemotron", "nvidia_nemotron", "megatron", "NemotronForCausalLM"),
    ),
)

# ── Encoder and encoder-decoder families ─────────────────────────────────────
# Dedicated engines with compile → load → execute round-trip tests and a
# portable/CPU cross-check, but no automatic per-logit comparison against a
# reference implementation yet.

_ENCODER_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        key="bert", name="BERT", kind=FamilyKind.ENCODER,
        level=SupportLevel.RUNS,
        representative_models=("BERT-base-uncased", "BERT-large-cased",
                               "DistilBERT", "MPNet"),
        numerics="bidirectional self-attention · learned absolute positions · "
                 "LayerNorm · GELU FFN · no KV cache",
        detection_keys=("bert", "distilbert", "mpnet", "xlnet", "BertModel",
                        "BertForMaskedLM", "BertForSequenceClassification"),
        aether_family="bert_family",
        note="round-trip tested; not per-logit gated against Transformers",
    ),
    ModelFamily(
        key="roberta", name="RoBERTa", kind=FamilyKind.ENCODER,
        level=SupportLevel.RUNS,
        representative_models=("RoBERTa-base", "RoBERTa-large"),
        numerics="BERT block with a position offset of 2 and byte-level BPE",
        detection_keys=("roberta", "RobertaModel", "RobertaForMaskedLM"),
        aether_family="roberta_family",
        note="round-trip tested; not per-logit gated against Transformers",
    ),
    ModelFamily(
        key="deberta", name="DeBERTa", kind=FamilyKind.ENCODER,
        level=SupportLevel.RUNS,
        representative_models=("DeBERTa-v3-base", "DeBERTa-v3-large"),
        numerics="disentangled content/position attention with relative "
                 "position buckets",
        detection_keys=("deberta", "deberta_v2", "DebertaModel",
                        "DebertaV2Model"),
        aether_family="deberta_family",
        note="round-trip tested; not per-logit gated against Transformers",
    ),
    ModelFamily(
        key="electra", name="ELECTRA", kind=FamilyKind.ENCODER,
        level=SupportLevel.RUNS,
        representative_models=("ELECTRA-base", "ELECTRA-large"),
        numerics="BERT block with a separate embedding size and projection",
        detection_keys=("electra", "ElectraModel", "ElectraForMaskedLM"),
        aether_family="electra_family",
        note="round-trip tested; not per-logit gated against Transformers",
    ),
    ModelFamily(
        key="albert", name="ALBERT", kind=FamilyKind.ENCODER,
        level=SupportLevel.RUNS,
        representative_models=("ALBERT-base-v2", "ALBERT-xxlarge-v2"),
        numerics="cross-layer weight sharing · factorized embedding projection",
        detection_keys=("albert", "AlbertModel"),
        aether_family="albert_family",
        note="round-trip tested; not per-logit gated against Transformers",
    ),
    ModelFamily(
        key="t5", name="T5 / mT5 / FLAN-T5 / BART", kind=FamilyKind.ENCODER_DECODER,
        level=SupportLevel.RUNS,
        representative_models=("T5-small", "T5-11B", "FLAN-T5-base",
                               "mT5-base", "ByT5", "BART-large"),
        numerics="cross-attention · relative-attention position buckets · "
                 "gated or plain ReLU FFN · RMSNorm without bias",
        detection_keys=("t5", "mt5", "byt5", "ul2", "flan_t5", "bart", "mbart",
                        "pegasus", "marian", "T5ForConditionalGeneration"),
        aether_family="encoder_decoder_family",
        note="round-trip tested; not per-logit gated against Transformers",
    ),
)

# ── State-space and hybrid families ──────────────────────────────────────────
# These have dedicated engines that execute, and whose output has been measured
# against the reference and found to diverge.  They are listed so the divergence
# is a documented fact rather than an omission.

_SSM_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        key="mamba", name="Mamba", kind=FamilyKind.SSM_HYBRID,
        level=SupportLevel.KNOWN_INCORRECT,
        representative_models=("Mamba-130M", "Mamba-2.8B"),
        numerics="selective scan with input-dependent Δ, B, C · depthwise "
                 "causal conv · no attention",
        detection_keys=("mamba", "MambaForCausalLM"),
        aether_family="hybrid_ssm_family",
        note="selective-scan output diverges from the reference "
             "(cosine ≈ 0.97); do not rely on its output",
    ),
    ModelFamily(
        key="mamba2", name="Mamba-2 / SSD", kind=FamilyKind.SSM_HYBRID,
        level=SupportLevel.KNOWN_INCORRECT,
        representative_models=("Mamba2-2.7B", "Codestral-Mamba"),
        numerics="state-space duality with n_heads × headdim channel geometry "
                 "and n_groups shared B/C",
        detection_keys=("mamba2", "Mamba2ForCausalLM"),
        aether_family="hybrid_ssm_family",
        note="output badly diverges from the reference (cosine ≈ 0.21)",
    ),
    ModelFamily(
        key="rwkv", name="RWKV-7", kind=FamilyKind.SSM_HYBRID,
        level=SupportLevel.KNOWN_INCORRECT,
        representative_models=("RWKV-7-World-1.5B", "RWKV-7-World-7B"),
        numerics="time-mix / channel-mix recurrence with per-channel decay",
        detection_keys=("rwkv", "RwkvForCausalLM"),
        aether_family="hybrid_ssm_family",
        note="does not currently bind its time-mix/channel-mix weights during "
             "compile",
    ),
    ModelFamily(
        key="jamba", name="Jamba / hybrid SSM", kind=FamilyKind.SSM_HYBRID,
        level=SupportLevel.KNOWN_INCORRECT,
        representative_models=("Jamba-v0.1", "Zamba2", "Hymba", "Falcon-H1",
                               "Bamba", "PLaMo"),
        numerics="per-layer attention/SSM schedule from layers_block_type or "
                 "attn_layer_period · MoE FFN on a subset of layers",
        detection_keys=("jamba", "zamba", "zamba2", "hymba", "bamba",
                        "falcon_h1", "plamo", "JambaForCausalLM"),
        aether_family="hybrid_ssm_family",
        note="reference path requires CUDA Mamba kernels, so the hybrid is "
             "unverified on CPU",
    ),
)

# ── Families detected but not executable ─────────────────────────────────────
# Compilation fails on purpose.  An artifact that runs the wrong graph is worse
# than a compile error, so these are refused rather than approximated.

_UNSUPPORTED_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        key="deepseek_mla", name="DeepSeek V3 / R1 (MLA)", kind=FamilyKind.DECODER,
        level=SupportLevel.UNSUPPORTED,
        representative_models=("DeepSeek-V3", "DeepSeek-R1-671B"),
        numerics="multi-head latent attention with k_rope fused into "
                 "kv_a_proj · shared experts alongside routed · sigmoid "
                 "group-limited routing",
        detection_keys=("deepseek", "deepseek_v2", "deepseek_v3",
                        "DeepseekForCausalLM"),
        aether_family="deepseek_family",
        note="the MLA engine does not yet reconstruct the fused projection "
             "contract; compile fails on kv_a_proj",
    ),
    ModelFamily(
        key="minimax", name="MiniMax", kind=FamilyKind.DECODER,
        level=SupportLevel.UNSUPPORTED,
        representative_models=("MiniMax-Text-01",),
        numerics="lightning attention alternating linear and softmax layers",
        detection_keys=("minimax", "MiniMaxForCausalLM"),
        note="a distinct architecture class with no engine",
    ),
    ModelFamily(
        key="vlm", name="Vision-language", kind=FamilyKind.MULTIMODAL,
        level=SupportLevel.UNSUPPORTED,
        representative_models=("LLaVA-1.5", "InternVL2", "PaliGemma",
                               "Qwen2-VL", "Pixtral"),
        numerics="ViT or native-resolution vision encoder · projector · "
                 "interleaved image/text token stream",
        detection_keys=("llava", "internvl", "paligemma", "pixtral", "vit",
                        "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
                        "ViTForImageClassification"),
        aether_family="vision_family",
        note="detected and ingested, but there is no verified execution "
             "contract for the vision tower",
    ),
    ModelFamily(
        key="whisper", name="Whisper / audio", kind=FamilyKind.MULTIMODAL,
        level=SupportLevel.UNSUPPORTED,
        representative_models=("Whisper-tiny", "Whisper-large-v3"),
        numerics="Conv1D audio front-end · encoder-decoder cross-attention · "
                 "log-Mel feature extraction",
        detection_keys=("whisper", "WhisperForConditionalGeneration"),
        aether_family="whisper_family",
        note="detected and ingested, but there is no verified execution "
             "contract for the audio front-end",
    ),
)

MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    _PARITY_DECODERS + _ENCODER_FAMILIES + _SSM_FAMILIES + _UNSUPPORTED_FAMILIES
)
"""Every architecture family Aether classifies, with its graded support level.

The length of this tuple is the answer to "how many model families does Aether
support?" once filtered by :class:`SupportLevel` — see :func:`family_counts`.
"""


def families_by_level(level: SupportLevel) -> tuple[ModelFamily, ...]:
    """Return every family at exactly ``level``."""
    return tuple(family for family in MODEL_FAMILIES if family.level is level)


def families_by_kind(kind: FamilyKind) -> tuple[ModelFamily, ...]:
    """Return every family whose compiled graph shape is ``kind``."""
    return tuple(family for family in MODEL_FAMILIES if family.kind is kind)


def executable_families() -> tuple[ModelFamily, ...]:
    """Return every family with an execution path, correct or not."""
    return tuple(family for family in MODEL_FAMILIES if family.is_executable)


def detection_key_count() -> int:
    """Number of distinct names / class spellings that resolve to a family.

    This counts *coverage*, not families.  It is the honest form of the claim
    that used to be written as "100+ model families": many names, far fewer
    distinct computation contracts.
    """
    keys: set[str] = set()
    for family in MODEL_FAMILIES:
        keys.update(key.lower() for key in family.detection_keys)
    return len(keys)


def family_counts() -> dict[str, int]:
    """Return the exact family counts published in the documentation.

    Keys:
      ``parity_verified``    families gated per-logit against the reference
      ``runs``               families that execute but are not per-logit gated
      ``known_incorrect``    families that execute with measured divergence
      ``unsupported``        families detected but refused at compile time
      ``executable``         parity_verified + runs + known_incorrect
      ``total``              every family in the registry
      ``decoder`` / ``encoder`` / ``encoder_decoder`` / ``ssm_hybrid`` /
      ``multimodal``         families by compiled graph shape
      ``parity_verified_decoder``  the headline number: verified decoders
      ``detection_keys``     name / class spellings resolving to a family
    """
    counts = {
        level.value: len(families_by_level(level)) for level in SupportLevel
    }
    counts.update(
        {kind.value: len(families_by_kind(kind)) for kind in FamilyKind}
    )
    counts["executable"] = len(executable_families())
    counts["total"] = len(MODEL_FAMILIES)
    counts["parity_verified_decoder"] = sum(
        1
        for family in MODEL_FAMILIES
        if family.kind is FamilyKind.DECODER and family.is_trusted
    )
    counts["detection_keys"] = detection_key_count()
    return counts


SUPPORT_SENTENCE: str = (
    "{parity_verified} architecture families are parity-verified against the "
    "reference implementation, {runs} more execute without a per-logit gate, "
    "and {known_incorrect} execute with measured divergence — "
    "{executable} executable families in total, reached through "
    "{detection_keys} detected name and architecture-class spellings."
)
"""Format string for the one-line support claim.

Rendered by :func:`support_summary`, so every document that states the number
states the same number.
"""


def support_summary() -> str:
    """Render the canonical one-line support claim from the registry."""
    return SUPPORT_SENTENCE.format(**family_counts())


def iter_matrix_rows(
    levels: Iterable[SupportLevel] | None = None,
) -> Iterable[ModelFamily]:
    """Yield registry rows in publication order, optionally filtered by level."""
    wanted = set(levels) if levels is not None else set(SupportLevel)
    for family in MODEL_FAMILIES:
        if family.level in wanted:
            yield family
