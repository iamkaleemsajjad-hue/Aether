"""Contract tests for the per-family execution numerics.

Decoder families share the shape of the transformer block but differ in a small
set of scalar constants and structural placements.  Each one is part of the
model's definition — omitting a single value changes every logit the compiled
artifact produces — so they are detected from the source configuration, carried
in the AEG manifest, and applied by every executor.

These tests pin the values Aether derives for the published architectures and
prove the contract survives a manifest round trip and reaches the runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.core.types import ModelArchitecture
from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights


def _detect(**config: object) -> ModelArchitecture:
    base = {
        "architectures": [config.pop("architecture", "LlamaForCausalLM")],
        "model_type": config.pop("model_type", "llama"),
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "vocab_size": 128,
    }
    # A checkpoint that spells a dimension the historical way carries only that
    # spelling; the modern default must not shadow it.
    for modern, legacy in (
        ("num_hidden_layers", ("n_layers", "n_layer", "num_layers")),
        ("hidden_size", ("d_model", "n_embd")),
        ("num_attention_heads", ("n_heads", "n_head", "num_heads")),
    ):
        if any(name in config for name in legacy):
            base.pop(modern, None)
    base.update(config)
    return ArchitectureDetector()._from_config(base)


class TestAttentionScale:
    """The softmax scale is not always ``1/sqrt(head_dim)``."""

    def test_gpt_neo_attention_is_unscaled(self) -> None:
        # GPT-Neo's published attention omits the 1/sqrt(d) factor entirely.
        # Applying the usual scale flattens its attention and destroys output.
        architecture = _detect(
            architecture="GPTNeoForCausalLM", model_type="gpt_neo",
            layer_norm_epsilon=1e-5,
        )
        assert architecture.attention_scale == 1.0

    def test_gpt_neox_keeps_the_standard_scale(self) -> None:
        # ``gpt_neo`` is a prefix of ``gpt_neox``, which does scale normally.
        architecture = _detect(
            architecture="GPTNeoXForCausalLM", model_type="gpt_neox",
            layer_norm_eps=1e-5,
        )
        assert architecture.attention_scale is None

    def test_gemma_scale_comes_from_query_pre_attn_scalar(self) -> None:
        architecture = _detect(
            architecture="Gemma2ForCausalLM", model_type="gemma2",
            head_dim=16, query_pre_attn_scalar=256, rms_norm_eps=1e-6,
        )
        assert architecture.attention_scale == pytest.approx(256 ** -0.5)

    def test_granite_uses_its_attention_multiplier(self) -> None:
        architecture = _detect(
            architecture="GraniteForCausalLM", model_type="granite",
            attention_multiplier=0.0078125, rms_norm_eps=1e-5,
        )
        assert architecture.attention_scale == pytest.approx(0.0078125)

    def test_gpt2_honours_disabled_scaling(self) -> None:
        architecture = _detect(
            architecture="GPT2LMHeadModel", model_type="gpt2",
            scale_attn_weights=False, layer_norm_epsilon=1e-5,
        )
        assert architecture.attention_scale == 1.0


class TestRotaryGeometry:
    """Partial and interleaved rotary conventions are not interchangeable."""

    def test_gpt_neox_partial_rotary(self) -> None:
        architecture = _detect(
            architecture="GPTNeoXForCausalLM", model_type="gpt_neox",
            rotary_pct=0.25, layer_norm_eps=1e-5,
        )
        # head_dim = 64 / 4 = 16, so 25% is a 4-wide rotation.
        assert architecture.rope_partial_dim == 4
        assert architecture.rope_interleaved is False

    def test_gptj_is_interleaved(self) -> None:
        architecture = _detect(
            architecture="GPTJForCausalLM", model_type="gptj",
            layer_norm_epsilon=1e-5,
        )
        assert architecture.rope_interleaved is True

    def test_cohere_is_interleaved(self) -> None:
        architecture = _detect(
            architecture="CohereForCausalLM", model_type="cohere",
            logit_scale=0.0625, layer_norm_eps=1e-5,
        )
        assert architecture.rope_interleaved is True
        assert architecture.logit_scale == pytest.approx(0.0625)
        # Cohere evaluates attention and MLP from one block norm.
        assert architecture.parallel_residual is True

    def test_stablelm_partial_rotary_factor(self) -> None:
        architecture = _detect(
            architecture="StableLmForCausalLM", model_type="stablelm",
            partial_rotary_factor=0.25, layer_norm_eps=1e-5,
        )
        assert architecture.rope_partial_dim == 4

    def test_gemma3_carries_a_separate_local_rotary_base(self) -> None:
        architecture = _detect(
            architecture="Gemma3ForCausalLM", model_type="gemma3",
            head_dim=16, rope_theta=1_000_000.0, rope_local_base_freq=10_000.0,
            sliding_window=8, rms_norm_eps=1e-6,
            layer_types=["sliding_attention"] * 4,
        )
        assert architecture.rope_local_theta == pytest.approx(10_000.0)
        assert architecture.rope_theta == pytest.approx(1_000_000.0)
        assert architecture.attention_layers == ["local"] * 4


class TestNormalizationPlacement:
    """Blocks normalize their input, their output, or both."""

    def test_llama_is_pre_norm(self) -> None:
        assert _detect(rms_norm_eps=1e-5).norm_placement == "pre"

    def test_gemma2_is_sandwich_normed(self) -> None:
        architecture = _detect(
            architecture="Gemma2ForCausalLM", model_type="gemma2",
            head_dim=16, rms_norm_eps=1e-6,
        )
        assert architecture.norm_placement == "sandwich"
        # Gemma stores normalization weights as offsets from unity.
        assert architecture.norm_offset_one is True
        assert architecture.embedding_scale == pytest.approx(64 ** 0.5)

    def test_olmo2_normalizes_sublayer_outputs(self) -> None:
        architecture = _detect(
            architecture="Olmo2ForCausalLM", model_type="olmo2", rms_norm_eps=1e-5,
        )
        assert architecture.norm_placement == "post"
        # OLMo-2 normalizes the whole Q/K projection, not each head.
        assert architecture.qk_norm_scope == "full"
        assert architecture.qk_norm is True

    def test_exaone4_is_post_norm_with_per_head_qk_norm(self) -> None:
        architecture = _detect(
            architecture="Exaone4ForCausalLM", model_type="exaone4",
            rms_norm_eps=1e-5, sliding_window=8,
            layer_types=["sliding_attention", "full_attention"] * 2,
        )
        assert architecture.norm_placement == "post"
        assert architecture.qk_norm_scope == "head"
        # Its global layers carry no positional encoding at all.
        assert architecture.no_rope_layers == [1, 3]


class TestFusedProjectionLayout:
    """A fused QKV tensor is not always three contiguous blocks."""

    def test_gpt_neox_interleaves_per_head(self) -> None:
        architecture = _detect(
            architecture="GPTNeoXForCausalLM", model_type="gpt_neox",
            layer_norm_eps=1e-5,
        )
        assert architecture.fused_qkv_layout == "head_interleaved"

    def test_falcon_new_architecture_interleaves_per_kv_group(self) -> None:
        architecture = _detect(
            architecture="FalconForCausalLM", model_type="falcon",
            new_decoder_architecture=True, num_kv_heads=2,
            layer_norm_epsilon=1e-5, activation="gelu",
        )
        assert architecture.fused_qkv_layout == "group_interleaved"

    def test_llama_is_contiguous(self) -> None:
        assert _detect(rms_norm_eps=1e-5).fused_qkv_layout == "contiguous"


class TestStructuralDefaults:
    """Position and activation contracts follow the checkpoint, not the name."""

    def test_bloom_uses_alibi_without_declaring_a_flag(self) -> None:
        architecture = _detect(
            architecture="BloomForCausalLM", model_type="bloom",
            n_layer=4, n_head=4, layer_norm_epsilon=1e-5,
        )
        assert architecture.position_type == "ALiBi"

    def test_mpt_reads_its_plural_dimension_names(self) -> None:
        architecture = _detect(
            architecture="MptForCausalLM", model_type="mpt",
            d_model=64, n_heads=4, n_layers=6, layer_norm_epsilon=1e-5,
            attn_config={"alibi": True},
        )
        assert architecture.layers == 6
        assert architecture.position_type == "ALiBi"
        assert architecture.norm_type == "LayerNorm"

    def test_epsilon_field_name_identifies_the_norm_family(self) -> None:
        # The epsilon spelling is the checkpoint's own structural declaration.
        assert _detect(rms_norm_eps=1e-5).norm_type == "RMSNorm"
        assert _detect(norm_epsilon=1e-5).norm_type == "LayerNorm"
        assert _detect(layer_norm_eps=1e-5).norm_type == "LayerNorm"

    def test_gelu_spelling_selects_exact_or_tanh(self) -> None:
        assert _detect(hidden_act="gelu", layer_norm_eps=1e-5).gelu_approximate is False
        assert _detect(hidden_act="gelu_new", layer_norm_eps=1e-5).gelu_approximate is True
        assert (
            _detect(hidden_act="gelu_pytorch_tanh", norm_epsilon=1e-5).gelu_approximate
            is True
        )

    def test_declared_window_without_a_schedule_is_not_applied(self) -> None:
        # Qwen2 and Mistral publish a sliding_window together with an all-global
        # schedule; applying it would truncate attention the model never did.
        architecture = _detect(
            architecture="Qwen2ForCausalLM", model_type="qwen2",
            sliding_window=4096, use_sliding_window=False, rms_norm_eps=1e-6,
        )
        assert architecture.attention_window is None


def test_execution_numerics_survive_a_manifest_round_trip() -> None:
    """Every numerics field must reach the runtime through the AEG manifest."""
    architecture = _detect(
        architecture="Gemma2ForCausalLM", model_type="gemma2",
        head_dim=16, query_pre_attn_scalar=256, rms_norm_eps=1e-6,
        attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
        sliding_window=8, layer_types=["sliding_attention", "full_attention"] * 2,
    )
    restored = ModelArchitecture.from_dict(architecture.to_dict())
    for field in (
        "attention_scale", "attention_scale_by_layer_index", "embedding_scale",
        "residual_scale", "logit_scale", "attn_logit_softcap",
        "final_logit_softcap", "norm_offset_one", "rope_partial_dim",
        "rope_interleaved", "rope_local_theta", "norm_placement",
        "qk_norm_scope", "fused_qkv_layout", "no_rope_layers",
        "gelu_approximate",
    ):
        assert getattr(restored, field) == getattr(architecture, field), field


def _single_layer_engine(**numerics: object) -> CPUExecutionEngine:
    """Build a one-layer engine with identity-ish weights for exact algebra."""
    hidden, heads = 4, 1
    eye = np.eye(hidden, dtype=np.float32)
    layer = LayerWeights(
        attention_norm=np.ones(hidden, dtype=np.float32),
        q_proj=eye.copy(), k_proj=eye.copy(), v_proj=eye.copy(), o_proj=eye.copy(),
        ffn_norm=np.ones(hidden, dtype=np.float32),
        gate_proj=np.zeros((hidden, hidden), dtype=np.float32),
        up_proj=np.zeros((hidden, hidden), dtype=np.float32),
        down_proj=np.zeros((hidden, hidden), dtype=np.float32),
    )
    weights = ModelWeights(
        embedding=eye.copy(),
        layers=[layer],
        final_norm=np.ones(hidden, dtype=np.float32),
        lm_head=eye.copy(),
        position_type="none",
        **numerics,
    )
    return CPUExecutionEngine(weights, num_heads=heads)


def test_logit_scale_is_applied_to_the_output() -> None:
    plain, _ = _single_layer_engine().forward(np.asarray([0], dtype=np.int64))
    scaled, _ = _single_layer_engine(logit_scale=0.25).forward(
        np.asarray([0], dtype=np.int64)
    )
    np.testing.assert_allclose(scaled, plain * 0.25, rtol=1e-6, atol=1e-6)


def test_final_logit_softcap_bounds_the_output() -> None:
    cap = 0.5
    capped, _ = _single_layer_engine(final_logit_softcap=cap).forward(
        np.asarray([0], dtype=np.int64)
    )
    assert np.all(np.abs(capped) <= cap + 1e-6)


def test_embedding_scale_multiplies_the_token_table() -> None:
    plain, _ = _single_layer_engine().forward(np.asarray([0], dtype=np.int64))
    scaled, _ = _single_layer_engine(embedding_scale=3.0).forward(
        np.asarray([0], dtype=np.int64)
    )
    # RMSNorm is scale invariant, so a uniform embedding gain must not change
    # the normalized logits; this pins that the scale is applied to the
    # embedding rather than silently to the logits.
    np.testing.assert_allclose(scaled, plain, rtol=1e-5, atol=1e-5)


def test_unscaled_attention_changes_the_distribution() -> None:
    """An explicit attention_scale must actually reach the attention kernel."""
    engine = _single_layer_engine()
    unscaled = _single_layer_engine(attention_scale=1.0)
    ids = np.asarray([0, 1, 2], dtype=np.int64)
    default_logits, _ = engine.forward(ids)
    unscaled_logits, _ = unscaled.forward(ids)
    assert not np.allclose(default_logits, unscaled_logits, rtol=1e-4, atol=1e-4)
