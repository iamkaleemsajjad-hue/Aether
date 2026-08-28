"""The rotary transform must actually reach the executors.

``test_rope_scaling.py`` proves the mathematics matches Hugging Face's reference.
This file proves the mathematics is *applied*: a correct module that nothing calls
is precisely the failure this change exists to fix, and it would leave output
degraded while every unit test passed.

Also covers stopping. An instruction-tuned checkpoint routinely ends a turn on a
delimiter that is not the tokenizer's canonical ``eos_token`` — Phi-3.5 stops on
``<|end|>`` while its ``eos_token`` is ``<|endoftext|>`` — and comparing against a
single id means never stopping: generation runs to ``max_tokens`` and the tail
degenerates. That reads as a model-quality problem and is not one.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.runtime.rope_scaling import base_inverse_frequencies

#: Bounded by the reference's FP32 rounding, not by ours (this path is FP64).
REFERENCE_TOLERANCE = 1e-6

#: Phi-3.5-mini's rotary width: head_dim 96, so 48 rotary pairs.
HEAD_DIM = 96
HALF = HEAD_DIM // 2

LONGROPE = {
    "rope_type": "longrope",
    "short_factor": [1.0 + 0.02 * index for index in range(HALF)],
    "long_factor": [1.0 + 0.15 * index for index in range(HALF)],
}


def _reference(scaling, seq_len, *, original):
    """LongRoPE frequencies and temperature from Transformers itself."""
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "longrope" not in ROPE_INIT_FUNCTIONS:
        pytest.skip(f"transformers {transformers.__version__} has no longrope")

    class Config:
        pass

    config = Config()
    config.rope_theta = 10000.0
    config.max_position_embeddings = 131072
    config.head_dim = HEAD_DIM
    config.hidden_size = HEAD_DIM * 4
    config.num_attention_heads = 4
    config.partial_rotary_factor = 1.0
    config.rope_scaling = scaling
    config.original_max_position_embeddings = original
    inverse, attention = ROPE_INIT_FUNCTIONS["longrope"](
        config, device="cpu", seq_len=seq_len
    )
    return inverse.double().numpy(), float(attention)


def _engine(scaling, original):
    """A tiny engine carrying Phi-3.5's rotary geometry."""
    from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights

    heads, vocab, intermediate = 2, 17, 12
    hidden = HEAD_DIM * heads
    rng = np.random.default_rng(3)

    def matrix(out: int, inn: int) -> np.ndarray:
        return rng.normal(0.0, 0.05, (out, inn)).astype(np.float32)

    layer = LayerWeights(
        attention_norm=np.ones(hidden, dtype=np.float32),
        q_proj=matrix(hidden, hidden), k_proj=matrix(hidden, hidden),
        v_proj=matrix(hidden, hidden), o_proj=matrix(hidden, hidden),
        ffn_norm=np.ones(hidden, dtype=np.float32),
        gate_proj=matrix(intermediate, hidden), up_proj=matrix(intermediate, hidden),
        down_proj=matrix(hidden, intermediate),
    )
    weights = ModelWeights(
        embedding=matrix(vocab, hidden), layers=[layer],
        final_norm=np.ones(hidden, dtype=np.float32), lm_head=matrix(vocab, hidden),
        rope_theta=10000.0, norm_eps=1e-5,
        rope_scaling=scaling, original_context_length=original,
        context_length=131072 if scaling else None,
    )
    return CPUExecutionEngine(weights, num_heads=heads, num_kv_heads=heads)


def _portable(scaling, original):
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    return TorchAEGEngine(_engine(scaling, original), "cpu")


# ── The transform reaches the tensor executor ───────────────────────────────


def test_tensor_executor_applies_the_declared_transform() -> None:
    engine = _portable(LONGROPE, 4096)
    assert engine.rope_scaling_spec is not None
    assert engine.rope_scaling_spec.rope_type == "longrope"

    engine._ensure_rope(64)
    expected, expected_attention = _reference(LONGROPE, 64, original=4096)
    np.testing.assert_allclose(
        engine._rope_inv_freq, expected, rtol=REFERENCE_TOLERANCE, atol=0
    )
    assert abs(engine.rope_attention_scaling - expected_attention) < REFERENCE_TOLERANCE


def test_tables_are_rebuilt_when_the_length_crosses_the_trained_window() -> None:
    """Growth alone is not enough to reuse the tables.

    LongRoPE switches factor tables at the trained length and ``dynamic`` rescales
    its base with length, so a cache keyed only on height would serve stale
    frequencies to a request that crossed the boundary.
    """
    engine = _portable(LONGROPE, 4096)
    engine._ensure_rope(64)
    short = engine._rope_inv_freq.copy()

    engine._ensure_rope(8192)
    long = engine._rope_inv_freq
    assert not np.array_equal(short, long), "the long factor table was never applied"

    expected, _ = _reference(LONGROPE, 8192, original=4096)
    np.testing.assert_allclose(long, expected, rtol=REFERENCE_TOLERANCE, atol=0)


def test_unscaled_model_keeps_the_standard_frequencies() -> None:
    """The regression guard: a model without rope_scaling must not move at all."""
    engine = _portable(None, None)
    assert engine.rope_scaling_spec is None
    engine._ensure_rope(64)
    np.testing.assert_allclose(
        engine._rope_inv_freq,
        base_inverse_frequencies(10000.0, HEAD_DIM),
        rtol=0,
        atol=1e-15,
    )
    assert engine.rope_attention_scaling == 1.0


def test_both_executors_agree_under_scaling() -> None:
    """The NumPy reference executor and the tensor executor must not diverge.

    They have separate table builders, so this is where a transform wired into one
    and forgotten in the other would surface.
    """
    reference = _engine(LONGROPE, 4096)
    portable = _portable(LONGROPE, 4096)
    # Rebuild the same weights so both see identical tensors.
    from aether.runtime.torch_engine import TorchAEGEngine

    portable = TorchAEGEngine(reference, "cpu")
    ids = np.asarray([1, 3, 5, 2, 7], dtype=np.int64)
    np.testing.assert_allclose(
        np.asarray(portable.forward(ids)[0]),
        np.asarray(reference.forward(ids)[0]),
        rtol=2e-4,
        atol=2e-4,
    )


def test_scaling_actually_changes_the_output() -> None:
    """Guards against wiring that parses the config and then ignores it."""
    ids = np.asarray([1, 3, 5, 2, 7], dtype=np.int64)
    scaled = np.asarray(_portable(LONGROPE, 4096).forward(ids)[0])
    plain = np.asarray(_portable(None, None).forward(ids)[0])
    assert not np.allclose(scaled, plain, atol=1e-4)


# ── The transform survives the artifact and a rebuilt weight container ──────


def test_the_contract_carries_scaling_through_a_rebuilt_container() -> None:
    """Sharding rebuilds a reduced weight container; the transform must survive.

    Dropping it there would execute the model unscaled on multi-GPU only — a defect
    no single-device test could see.
    """
    from aether.runtime.torch_engine import EXECUTION_NUMERICS_FIELDS, execution_numerics

    assert "rope_scaling" in EXECUTION_NUMERICS_FIELDS
    assert "original_context_length" in EXECUTION_NUMERICS_FIELDS

    carried = execution_numerics(_engine(LONGROPE, 4096).weights)
    assert carried["rope_scaling"]["rope_type"] == "longrope"
    assert carried["original_context_length"] == 4096


def test_architecture_metadata_round_trips_and_older_artifacts_still_load() -> None:
    from aether.core.types import ModelArchitecture

    architecture = ModelArchitecture(
        family="phi3", params_billion=3.8, layers=32, hidden_size=3072,
        num_attention_heads=32, head_dim=HEAD_DIM, context_length=131072,
        original_context_length=4096, rope_scaling=dict(LONGROPE),
    )
    payload = architecture.to_dict()
    restored = ModelArchitecture.from_dict(payload)
    assert restored.rope_scaling["rope_type"] == "longrope"
    assert restored.original_context_length == 4096

    legacy = {
        key: value for key, value in payload.items()
        if key not in {"rope_scaling", "original_context_length"}
    }
    older = ModelArchitecture.from_dict(legacy)
    assert older.rope_scaling is None
    assert older.original_context_length is None


def test_compiler_refuses_a_scheme_it_cannot_reproduce() -> None:
    """A compile error beats an artifact that runs with the wrong geometry."""
    from aether.compiler.stage1_ingestion.architecture_detector import (
        _rope_scaling_config,
    )

    assert _rope_scaling_config({}) is None
    assert _rope_scaling_config({"rope_scaling": None}) is None
    assert (
        _rope_scaling_config({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}})[
            "rope_type"
        ]
        == "llama3"
    )
    with pytest.raises(ValueError, match="cannot reproduce"):
        _rope_scaling_config({"rope_scaling": {"rope_type": "invented", "factor": 2.0}})


def test_loader_refuses_an_unreproducible_scheme_from_a_manifest() -> None:
    from aether.runtime.aeg_loader import AEGLoadError, _rope_scaling_mapping

    assert _rope_scaling_mapping({}) is None
    assert _rope_scaling_mapping({"rope_scaling": {}}) is None
    assert _rope_scaling_mapping(
        {"rope_scaling": {"type": "yarn", "factor": 4.0}}
    )["type"] == "yarn"
    with pytest.raises(AEGLoadError, match="cannot reproduce"):
        _rope_scaling_mapping({"rope_scaling": {"rope_type": "invented"}})


# ── Stopping on every delimiter the checkpoint declares ─────────────────────


def test_stop_token_set_normalizes_every_shape() -> None:
    from aether.runtime.torch_engine import stop_token_set

    assert stop_token_set(None) == frozenset()
    assert stop_token_set(5) == frozenset({5})
    assert stop_token_set([32007, 32001, 32000]) == frozenset({32000, 32001, 32007})
    assert stop_token_set((1, 2)) == frozenset({1, 2})
    # A bool is an int in Python; treating True as token 1 would be a real bug.
    assert stop_token_set(True) == frozenset()
    assert stop_token_set("32007") == frozenset()


def test_generation_stops_on_a_secondary_delimiter() -> None:
    """The end-to-end guard: a non-canonical stop id must actually halt decoding."""
    engine = _portable(None, None)
    prompt = np.asarray([1, 3, 5], dtype=np.int64)

    unbounded = engine.generate(prompt, max_tokens=8, temperature=0.0)
    assert len(unbounded) == 8, "the baseline must run to the cap for this to mean anything"

    # Stop on a token the run actually produces, passed as a *list* beside an
    # unrelated id — the shape a real generation_config.eos_token_id has.  The
    # expected output is the prefix up to and including its first occurrence, since
    # a stop token is yielded and then halts decoding.
    delimiter = unbounded[-1]
    cut = unbounded.index(delimiter) + 1
    stopped = engine.generate(
        prompt, max_tokens=8, temperature=0.0, eos_token_id=[999999, delimiter]
    )
    assert stopped == unbounded[:cut]
    assert stopped[-1] == delimiter

    batched = engine.generate_batch(
        [prompt], max_tokens=8, temperature=0.0, eos_token_id=[999999, delimiter]
    )
    assert batched == [unbounded[:cut]]


def test_packaged_tokenizer_merges_generation_config_stop_ids(tmp_path) -> None:
    """The real source of an instruct model's stop ids is generation_config.json."""
    pytest.importorskip("tokenizers")
    import json

    from aether.backends.native_cpu_backend import PackagedTokenizer

    # A minimal WordLevel tokenizer is enough: only the id plumbing is under test.
    vocab = {"a": 0, "b": 1, "<|endoftext|>": 2, "<|end|>": 3}
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "a"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>"}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [3, 2]}), encoding="utf-8"
    )

    tokenizer = PackagedTokenizer(tmp_path / "tokenizer.json")
    # Both the turn delimiter and the canonical eos must be present.
    assert set(tokenizer.eos_token_ids) == {2, 3}
    # The canonical single id keeps working for callers that want one.
    assert tokenizer.eos_token_id == 2


def test_packaged_tokenizer_without_generation_config_still_stops(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    import json

    from aether.backends.native_cpu_backend import PackagedTokenizer

    vocab = {"a": 0, "</s>": 1}
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "a"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "</s>"}), encoding="utf-8"
    )

    tokenizer = PackagedTokenizer(tmp_path / "tokenizer.json")
    assert tokenizer.eos_token_ids == (1,)
