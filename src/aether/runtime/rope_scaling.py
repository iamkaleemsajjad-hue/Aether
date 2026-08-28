"""Rotary position embedding frequency scaling, for every published scheme.

Rotary embeddings (Su et al. 2021, *RoFormer*, arXiv:2104.09864) rotate each
query/key pair by an angle proportional to its absolute position::

    inv_freq[i] = theta ** (-2i / d)      i = 0 .. d/2
    angle(p, i)  = p * inv_freq[i]

A model trained at one context length cannot be evaluated at a longer one with
those frequencies unchanged: the low-frequency dimensions run off the end of the
range they ever saw. Every long-context model therefore ships a *frequency
transform* in its config under ``rope_scaling``, and the transform is part of the
model, not a runtime option. Evaluating a scaled model with unscaled frequencies
produces a subtly wrong rotation at every position, which is not a crash and not a
shape error — it degrades attention, and the damage compounds along the sequence
until generation collapses into repetition.

This module implements the transforms as one pure function so that every executor
shares a single definition. It imports no tensor framework.

The schemes, and where each comes from:

``linear`` (also spelled ``su``)
    Position interpolation (Chen et al. 2023, arXiv:2306.15595). Divide every
    frequency by the extension factor, equivalently compress positions.

``dynamic``
    NTK-aware interpolation, applied per sequence length: raise the rotary base so
    that the highest frequency is preserved while low frequencies interpolate.

``yarn``
    YaRN (Peng et al. 2023, arXiv:2309.00071). Interpolate only the dimensions
    whose wavelength exceeds the trained context, extrapolate the rest, and ramp
    smoothly between them — plus a temperature correction on attention logits.

``longrope``
    LongRoPE (Ding et al. 2024, arXiv:2402.13753). A *per-dimension* factor found
    by search rather than a closed form, with separate short and long schedules
    selected by the sequence length.

``llama3``
    Llama 3.1's piecewise-by-wavelength transform: leave high frequencies alone,
    interpolate low ones, and blend across a band defined by two factors.

All five reduce to the identity when no scaling is declared, so a model without
``rope_scaling`` is unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Schemes this module evaluates.  A config naming anything else is a model whose
#: rotary geometry we cannot reproduce, and the caller is expected to refuse rather
#: than silently fall back to unscaled frequencies.
SUPPORTED_ROPE_SCALING: frozenset[str] = frozenset(
    {"linear", "su", "dynamic", "yarn", "longrope", "llama3", "default", "none"}
)


@dataclass(frozen=True)
class RopeScaling:
    """A model's declared rotary frequency transform.

    Immutable and framework-free.  ``factor`` is the context extension ratio;
    the remaining fields are only meaningful to some schemes, which is why they
    carry scheme-appropriate defaults rather than being required.
    """

    rope_type: str = "default"
    factor: float = 1.0
    original_max_position_embeddings: int | None = None
    #: The *extended* context length the checkpoint advertises. Distinct from
    #: ``original_max_position_embeddings``, which is the length it was trained at.
    #: ``dynamic`` measures against the extended value, and ``longrope`` derives its
    #: attention temperature from the ratio of the two, so conflating them changes
    #: the rotation.
    max_position_embeddings: int | None = None
    #: LongRoPE: one multiplier per rotary pair, selected by sequence length.
    short_factor: tuple[float, ...] = field(default_factory=tuple)
    long_factor: tuple[float, ...] = field(default_factory=tuple)
    #: YaRN ramp bounds, in cycles.  The published defaults.
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    #: YaRN/LongRoPE attention temperature multiplier, when a model overrides it.
    attention_factor: float | None = None
    #: DeepSeek-V2/V3 split YaRN's temperature into two mscale terms.
    mscale: float | None = None
    mscale_all_dim: float | None = None
    #: YaRN floors/ceils the correction range unless a config opts out.
    truncate: bool = True
    #: Llama 3.1 band edges, as divisors of the original context length.
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0

    @property
    def is_identity(self) -> bool:
        """Whether this transform leaves the standard frequencies untouched."""
        if self.rope_type in {"default", "none", ""}:
            return True
        # A declared scheme with factor 1 and no per-dimension table is a no-op.
        return (
            self.factor == 1.0
            and not self.short_factor
            and not self.long_factor
            and self.rope_type != "yarn"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the AEG execution-numerics contract."""
        payload: dict[str, Any] = {"rope_type": self.rope_type, "factor": self.factor}
        if self.original_max_position_embeddings is not None:
            payload["original_max_position_embeddings"] = (
                self.original_max_position_embeddings
            )
        if self.short_factor:
            payload["short_factor"] = list(self.short_factor)
        if self.long_factor:
            payload["long_factor"] = list(self.long_factor)
        if self.rope_type == "yarn":
            payload["beta_fast"] = self.beta_fast
            payload["beta_slow"] = self.beta_slow
        if self.rope_type == "llama3":
            payload["low_freq_factor"] = self.low_freq_factor
            payload["high_freq_factor"] = self.high_freq_factor
        if self.attention_factor is not None:
            payload["attention_factor"] = self.attention_factor
        return payload


def parse_rope_scaling(
    config: Any,
    *,
    context_length: int | None = None,
    original_context_length: int | None = None,
) -> RopeScaling | None:
    """Read a ``rope_scaling`` mapping into a :class:`RopeScaling`.

    Returns ``None`` when the model declares no scaling, so a caller can keep the
    standard frequencies without a special case.

    Both the historical spelling (``type``) and the current one (``rope_type``) are
    accepted, because checkpoints in the wild use each.

    ``context_length`` is the checkpoint's advertised (extended) length and
    ``original_context_length`` the length it was pretrained at.  The latter is a
    *top-level* config field for the Phi family rather than part of the scaling
    mapping, and LongRoPE needs it twice over: it selects the short or long factor
    table, and its ratio to the extended length sets the attention temperature.
    Defaulting it to the extended length — which is what happens when a caller omits
    it — silently disables both.
    """
    if not isinstance(config, dict) or not config:
        return None
    raw_type = config.get("rope_type", config.get("type", "default"))
    rope_type = str(raw_type or "default").strip().lower()
    if rope_type in {"default", "none", ""}:
        return None

    def number(name: str, fallback: float) -> float:
        value = config.get(name, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def table(name: str) -> tuple[float, ...]:
        value = config.get(name)
        if not isinstance(value, (list, tuple)):
            return ()
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return ()

    def integer(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    original_int = integer(
        config.get("original_max_position_embeddings")
    ) or integer(original_context_length)
    extended = integer(context_length)
    if original_int is None and extended:
        # YaRN falls back to the extended length when a config omits the trained
        # one; that is the reference behaviour, not a guess.
        original_int = extended

    attention_factor = config.get("attention_factor")
    try:
        attention = float(attention_factor) if attention_factor is not None else None
    except (TypeError, ValueError):
        attention = None

    def optional_number(name: str) -> float | None:
        if config.get(name) is None:
            return None
        try:
            return float(config[name])
        except (TypeError, ValueError):
            return None

    return RopeScaling(
        rope_type=rope_type,
        factor=number("factor", 1.0),
        original_max_position_embeddings=original_int,
        max_position_embeddings=extended,
        short_factor=table("short_factor"),
        long_factor=table("long_factor"),
        beta_fast=number("beta_fast", 32.0),
        beta_slow=number("beta_slow", 1.0),
        attention_factor=attention,
        mscale=optional_number("mscale"),
        mscale_all_dim=optional_number("mscale_all_dim"),
        truncate=bool(config.get("truncate", True)),
        low_freq_factor=number("low_freq_factor", 1.0),
        high_freq_factor=number("high_freq_factor", 4.0),
    )


def base_inverse_frequencies(theta: float, rotary_dim: int) -> np.ndarray:
    """The unscaled RoPE frequencies ``theta ** (-2i/d)`` in FP64.

    Computed in double precision because the exponent spans several orders of
    magnitude and the scaled variants divide by search-derived factors; rounding
    here would show up as a position-dependent phase error.
    """
    half = rotary_dim // 2
    exponent = np.arange(half, dtype=np.float64) * (2.0 / float(rotary_dim))
    return np.power(float(theta), -exponent)


def scaled_inverse_frequencies(
    theta: float,
    rotary_dim: int,
    scaling: RopeScaling | None,
    *,
    sequence_length: int,
) -> tuple[np.ndarray, float]:
    """Return ``(inv_freq, attention_multiplier)`` for this model and length.

    ``attention_multiplier`` is the temperature correction YaRN and LongRoPE apply
    to attention logits; it is ``1.0`` for the schemes that do not use one.

    ``sequence_length`` matters for the two length-dependent schemes: ``dynamic``
    rescales the base with the current length, and ``longrope`` switches between
    its short and long tables. Passing the full length that will be evaluated —
    prompt plus generation budget — keeps a single set of tables valid for the
    whole request, which is what lets the executor build them once.
    """
    inverse = base_inverse_frequencies(theta, rotary_dim)
    if scaling is None or scaling.is_identity:
        return inverse, 1.0

    kind = scaling.rope_type
    factor = float(scaling.factor)
    original = int(scaling.original_max_position_embeddings or 0)

    if kind in {"linear", "su"}:
        # Position interpolation: compress positions by the extension factor.
        if factor > 0:
            inverse = inverse / factor
        return inverse, 1.0

    if kind == "dynamic":
        # NTK-aware interpolation: raise the rotary base so the fastest dimension
        # keeps its wavelength while the slow ones interpolate.  The reference
        # measures against the *extended* context length and clamps the current
        # length up to it, so a short request reproduces the base the model was
        # published with rather than an accidentally stronger scaling.
        reference = int(scaling.max_position_embeddings or original or 0)
        if reference > 0 and factor > 0:
            effective = max(int(sequence_length), reference)
            adjusted = factor * (effective / reference) - (factor - 1.0)
            scaled_theta = float(theta) * adjusted ** (
                rotary_dim / (rotary_dim - 2.0)
            )
            inverse = base_inverse_frequencies(scaled_theta, rotary_dim)
        return inverse, 1.0

    if kind == "yarn":
        return _yarn(inverse, theta, rotary_dim, scaling, original)

    if kind == "longrope":
        return _longrope(inverse, scaling, original, sequence_length)

    if kind == "llama3":
        return _llama3(inverse, scaling, original), 1.0

    raise ValueError(
        f"unsupported rope_scaling type {kind!r}; the model's rotary geometry "
        "cannot be reproduced, and executing it with unscaled frequencies would "
        "silently degrade quality"
    )


def _yarn(
    inverse: np.ndarray,
    theta: float,
    rotary_dim: int,
    scaling: RopeScaling,
    original: int,
) -> tuple[np.ndarray, float]:
    """YaRN: ramp between interpolated and extrapolated frequencies.

    A dimension whose wavelength still fits inside the trained context needs no
    correction (extrapolate); one whose wavelength exceeds it must be interpolated.
    ``beta_fast``/``beta_slow`` set the cycle counts bounding the transition, and
    the ramp is linear in dimension index between them (Peng et al. 2023, §3.2).
    """
    factor = float(scaling.factor)
    if original <= 0 or factor <= 0:
        return inverse, 1.0
    extrapolation = inverse
    interpolation = inverse / factor

    def dimension_for_cycles(cycles: float) -> float:
        # Invert wavelength(i) = 2π·theta^(2i/d) for the index at `cycles` cycles.
        return (
            rotary_dim
            * math.log(original / (cycles * 2.0 * math.pi))
            / (2.0 * math.log(float(theta)))
        )

    low = dimension_for_cycles(scaling.beta_fast)
    high = dimension_for_cycles(scaling.beta_slow)
    if scaling.truncate:
        low, high = math.floor(low), math.ceil(high)
    low = max(low, 0.0)
    high = min(high, rotary_dim - 1.0)
    half = rotary_dim // 2
    if high == low:
        high += 0.001  # the reference's singularity guard
    index = np.arange(half, dtype=np.float64)
    ramp = np.clip((index - low) / (high - low), 0.0, 1.0)
    # ``ramp`` rises with dimension index, i.e. with wavelength.  A long-wavelength
    # dimension is the one that must be *interpolated*, so ramp selects
    # interpolation and its complement keeps extrapolation.  Inverting this is
    # silent: every shape still matches and only the rotation is wrong.
    blended = interpolation * ramp + extrapolation * (1.0 - ramp)
    return blended, _yarn_attention(scaling, factor)


def _yarn_attention(scaling: RopeScaling, factor: float) -> float:
    """YaRN's temperature correction on attention logits.

    The published form is ``0.1·ln(factor) + 1``.  DeepSeek-V2/V3 instead express it
    as a ratio of two ``mscale`` terms, which is why both are supported: using the
    plain form for a model that declares mscale would apply the wrong temperature.
    """
    if scaling.attention_factor is not None:
        return float(scaling.attention_factor)
    if factor <= 1.0:
        return 1.0

    def mscale(scale: float, multiplier: float) -> float:
        return 1.0 if scale <= 1.0 else 0.1 * multiplier * math.log(scale) + 1.0

    if scaling.mscale is not None and scaling.mscale_all_dim is not None:
        numerator = mscale(factor, scaling.mscale)
        denominator = mscale(factor, scaling.mscale_all_dim)
        return float(numerator / denominator) if denominator else 1.0
    return float(0.1 * math.log(factor) + 1.0)


def _longrope(
    inverse: np.ndarray,
    scaling: RopeScaling,
    original: int,
    sequence_length: int,
) -> tuple[np.ndarray, float]:
    """LongRoPE: a searched per-dimension factor, with a short and a long table.

    The factors are not a closed form — they come from an evolutionary search over
    per-dimension rescalings (Ding et al. 2024) — so the only correct implementation
    is to apply the table the checkpoint ships. Which table depends on the length
    being evaluated, matching the reference behaviour.
    """
    long_table = np.asarray(scaling.long_factor, dtype=np.float64)
    short_table = np.asarray(scaling.short_factor, dtype=np.float64)
    # The trained length is the switch point: at or below it the short schedule is
    # what the model saw, beyond it the long one.  A request that crosses the
    # boundary mid-generation would need the long table for the whole pass, which is
    # why the caller passes prompt + generation budget rather than the prompt alone.
    use_long = bool(original > 0 and sequence_length > original and long_table.size)
    table = long_table if use_long else short_table
    if table.size == 0:
        return inverse, 1.0
    if table.size != inverse.size:
        raise ValueError(
            f"longrope factor table has {table.size} entries but the rotary width "
            f"needs {inverse.size}; the checkpoint's rope_scaling does not match its "
            "head dimension"
        )
    scaled = inverse / table
    if scaling.attention_factor is not None:
        return scaled, float(scaling.attention_factor)
    # Phi-3 and its relatives derive the temperature from the *ratio* of the
    # extended context to the trained one, not from the declared ``factor``. The
    # two usually agree, but where a checkpoint sets both they can differ, and the
    # ratio is what the reference uses.
    extended = int(scaling.max_position_embeddings or 0)
    factor = (
        extended / original
        if original > 0 and extended > 0
        else float(scaling.factor)
    )
    if original <= 0 or factor <= 1.0:
        return scaled, 1.0
    attention = math.sqrt(1.0 + math.log(factor) / math.log(original))
    return scaled, float(attention)


def _llama3(inverse: np.ndarray, scaling: RopeScaling, original: int) -> np.ndarray:
    """Llama 3.1: piecewise by wavelength, with a smooth band between.

    High-frequency dimensions (wavelength well inside the trained context) are left
    alone; low-frequency ones are fully interpolated; between the two edges the two
    treatments are blended linearly in ``original / wavelength``.
    """
    factor = float(scaling.factor)
    if original <= 0 or factor <= 0:
        return inverse
    low_factor = scaling.low_freq_factor
    high_factor = scaling.high_freq_factor
    if high_factor == low_factor:
        return inverse / factor
    low_wavelength = original / low_factor
    high_wavelength = original / high_factor
    wavelength = 2.0 * math.pi / inverse

    interpolated = inverse / factor
    smooth = (original / wavelength - low_factor) / (high_factor - low_factor)
    blended = (1.0 - smooth) * interpolated + smooth * inverse
    result = np.where(wavelength < high_wavelength, inverse, blended)
    return np.where(wavelength > low_wavelength, interpolated, result)
