"""Rotary frequency scaling, validated against the reference implementation.

Every long-context model ships a rotary *frequency transform* in its config, and
the transform is part of the model. Executing a scaled checkpoint with unscaled
frequencies rotates every query and key by the wrong angle: attention degrades
progressively along the sequence and generation collapses into repetition. It is
not a crash and no shape check catches it, which is exactly why these tests exist.

The oracle is Hugging Face's own ``ROPE_INIT_FUNCTIONS``. Differential testing
against the reference — rather than against expectations written here — is what
found three real errors during development: an inverted YaRN ramp, a ``dynamic``
scheme measuring against the trained rather than the extended length, and LongRoPE
reading its trained length from the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.runtime.rope_scaling import (
    SUPPORTED_ROPE_SCALING,
    base_inverse_frequencies,
    parse_rope_scaling,
    scaled_inverse_frequencies,
)

#: The reference computes in FP32, so agreement is bounded by its rounding, not by
#: ours (this module works in FP64). Anything above this is a formula error, which
#: would show up orders of magnitude larger.
REFERENCE_TOLERANCE = 1e-6


def _reference(kind, scaling, seq_len, *, theta, head_dim, max_positions, original=None):
    """Frequencies and attention factor from Transformers' own implementation."""
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if kind not in ROPE_INIT_FUNCTIONS:
        pytest.skip(f"installed transformers {transformers.__version__} has no {kind!r}")

    class Config:
        pass

    config = Config()
    config.rope_theta = theta
    config.max_position_embeddings = max_positions
    config.head_dim = head_dim
    config.hidden_size = head_dim * 4
    config.num_attention_heads = 4
    config.partial_rotary_factor = 1.0
    config.rope_scaling = scaling
    if original is not None:
        config.original_max_position_embeddings = original
    inverse, attention = ROPE_INIT_FUNCTIONS[kind](config, device="cpu", seq_len=seq_len)
    return inverse.double().numpy(), float(attention)


def _ours(scaling, seq_len, *, theta, head_dim, max_positions, original=None):
    spec = parse_rope_scaling(
        scaling, context_length=max_positions, original_context_length=original
    )
    return scaled_inverse_frequencies(theta, head_dim, spec, sequence_length=seq_len)


def _agrees(kind, scaling, seq_len, *, theta=10000.0, head_dim=96,
            max_positions=131072, original=None):
    expected, expected_attention = _reference(
        kind, scaling, seq_len, theta=theta, head_dim=head_dim,
        max_positions=max_positions, original=original,
    )
    actual, actual_attention = _ours(
        scaling, seq_len, theta=theta, head_dim=head_dim,
        max_positions=max_positions, original=original,
    )
    np.testing.assert_allclose(
        actual, expected, rtol=REFERENCE_TOLERANCE, atol=0,
        err_msg=f"{kind} frequencies diverge from the reference",
    )
    assert abs(actual_attention - expected_attention) < REFERENCE_TOLERANCE, (
        f"{kind} attention factor {actual_attention} != reference {expected_attention}"
    )


# ── No scaling declared: the standard frequencies, untouched ────────────────


def test_absent_scaling_is_the_identity() -> None:
    """A model without rope_scaling must be bit-identical to the plain formula."""
    assert parse_rope_scaling(None) is None
    assert parse_rope_scaling({}) is None
    assert parse_rope_scaling({"rope_type": "default"}) is None
    assert parse_rope_scaling({"type": "none"}) is None

    inverse, attention = scaled_inverse_frequencies(
        10000.0, 96, None, sequence_length=1024
    )
    np.testing.assert_array_equal(inverse, base_inverse_frequencies(10000.0, 96))
    assert attention == 1.0


def test_a_declared_scheme_with_no_effect_is_also_the_identity() -> None:
    spec = parse_rope_scaling({"rope_type": "linear", "factor": 1.0})
    assert spec is not None and spec.is_identity
    inverse, attention = scaled_inverse_frequencies(
        10000.0, 96, spec, sequence_length=1024
    )
    np.testing.assert_array_equal(inverse, base_inverse_frequencies(10000.0, 96))
    assert attention == 1.0


# ── Each published scheme, against the reference ────────────────────────────


def test_linear_matches_reference() -> None:
    """Position interpolation (Chen et al. 2023, arXiv:2306.15595)."""
    _agrees("linear", {"rope_type": "linear", "factor": 4.0}, 1024)


def test_dynamic_matches_reference_beyond_and_within_the_window() -> None:
    """NTK-aware interpolation, which is length-dependent.

    The reference clamps the current length up to the extended context, so a short
    request must reproduce the base the model was published with rather than an
    accidentally weaker scaling.
    """
    scaling = {"rope_type": "dynamic", "factor": 4.0}
    _agrees("dynamic", scaling, 262144)
    _agrees("dynamic", scaling, 1024)


def test_yarn_matches_reference() -> None:
    """YaRN (Peng et al. 2023, arXiv:2309.00071), ramp and temperature.

    The ramp direction is the subtle part: it rises with dimension index, i.e. with
    wavelength, and a long-wavelength dimension is the one needing *interpolation*.
    Inverting it leaves every shape valid and only the rotation wrong.
    """
    _agrees(
        "yarn",
        {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32, "beta_slow": 1,
        },
        16384,
    )


def test_yarn_honours_deepseek_mscale_temperature() -> None:
    """DeepSeek-V2/V3 express YaRN's temperature as a ratio of two mscale terms."""
    _agrees(
        "yarn",
        {
            "rope_type": "yarn", "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "mscale": 1.0, "mscale_all_dim": 1.0,
        },
        16384,
    )


def test_longrope_matches_reference_in_both_regimes() -> None:
    """LongRoPE (Ding et al. 2024, arXiv:2402.13753) — Phi-3/3.5's scheme.

    The factors are searched, not closed-form, so the only correct implementation
    applies the table the checkpoint ships. Which table depends on the length, and
    the trained length is a *top-level* config field for this family.
    """
    half = 96 // 2
    scaling = {
        "rope_type": "longrope",
        "short_factor": [1.0 + 0.02 * index for index in range(half)],
        "long_factor": [1.0 + 0.15 * index for index in range(half)],
        "factor": 32.0,
    }
    _agrees("longrope", scaling, 1024, original=4096)     # short table
    _agrees("longrope", scaling, 65536, original=4096)    # long table
    _agrees("longrope", scaling, 4096, original=4096)     # at the boundary


def test_llama3_matches_reference() -> None:
    """Llama 3.1's piecewise-by-wavelength transform."""
    _agrees(
        "llama3",
        {
            "rope_type": "llama3", "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        },
        16384,
        theta=500000.0,
        head_dim=128,
    )


# ── Refusal rather than silent degradation ──────────────────────────────────


def test_an_unknown_scheme_is_refused_not_ignored() -> None:
    """Falling back to unscaled frequencies would be the worst outcome.

    It produces fluent, plausible, subtly wrong text — the failure mode hardest for
    a user to attribute. An explicit error is strictly better.
    """
    spec = parse_rope_scaling({"rope_type": "some_future_scheme", "factor": 2.0})
    assert spec is not None
    with pytest.raises(ValueError, match="unsupported rope_scaling"):
        scaled_inverse_frequencies(10000.0, 96, spec, sequence_length=1024)


def test_longrope_rejects_a_mismatched_factor_table() -> None:
    """A table of the wrong width means the config does not match the head dim."""
    spec = parse_rope_scaling(
        {"rope_type": "longrope", "short_factor": [1.0, 1.1], "long_factor": [1.0, 1.1]},
        context_length=131072,
        original_context_length=4096,
    )
    with pytest.raises(ValueError, match="does not match its head dimension"):
        scaled_inverse_frequencies(10000.0, 96, spec, sequence_length=512)


def test_both_config_spellings_are_accepted() -> None:
    """Checkpoints in the wild use ``type`` and ``rope_type`` interchangeably."""
    old = parse_rope_scaling({"type": "linear", "factor": 4.0})
    new = parse_rope_scaling({"rope_type": "linear", "factor": 4.0})
    assert old is not None and new is not None
    assert old.rope_type == new.rope_type == "linear"


def test_supported_set_covers_every_published_scheme() -> None:
    for scheme in ("linear", "su", "dynamic", "yarn", "longrope", "llama3"):
        assert scheme in SUPPORTED_ROPE_SCALING
