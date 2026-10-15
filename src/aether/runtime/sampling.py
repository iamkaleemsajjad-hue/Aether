"""Shared request sampling and grammar masks for ordinary and speculative decode."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if isinstance(self.top_p, bool) or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")


def token_probabilities(logits: Any, config: SamplingConfig | None = None,
                        *, grammar_session: Any | None = None) -> np.ndarray:
    """Return the actual temperature/top-k/nucleus/grammar target distribution.

    The nucleus includes the first token crossing top_p, as the CPU and Torch
    serving paths do. Logit ties at the top-k threshold are retained. Greedy
    temperature zero produces a delta distribution on the first maximum.
    """
    config = config or SamplingConfig()
    values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if not values.size or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("sampling logits must be nonempty and contain no NaN or positive infinity")
    if grammar_session is not None:
        mask = grammar_session.get_token_mask()
        if len(mask) * 8 < values.size:
            raise ValueError("Grammar FSM vocabulary is smaller than model vocabulary")
        allowed = np.fromiter(((mask[i // 8] & (1 << (i % 8))) != 0 for i in range(values.size)),
                              dtype=bool, count=values.size)
        values[~allowed] = -np.inf
    if not np.isfinite(values).any():
        raise ValueError("sampling constraints permit no finite next-token logits")
    if config.temperature == 0:
        result = np.zeros(values.size, dtype=np.float64)
        result[int(np.argmax(values))] = 1.0
        return result
    if config.top_k:
        k = min(config.top_k, values.size)
        values[values < np.partition(values, -k)[-k]] = -np.inf
    values -= values.max()
    # Subtract first: a finite logit divided by a tiny temperature can
    # overflow to +inf, making the subsequent subtraction produce NaNs.
    with np.errstate(over="ignore"):
        values /= float(config.temperature)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    if config.top_p < 1:
        order = np.argsort(values)[::-1]
        cumulative = np.cumsum(probabilities[order])
        keep = cumulative <= config.top_p
        crossing = int(np.searchsorted(cumulative, config.top_p, side="right"))
        if crossing < keep.size:
            keep[crossing] = True
        allowed = np.zeros(values.size, dtype=bool)
        allowed[order[keep]] = True
        probabilities[~allowed] = 0
        probabilities /= probabilities.sum()
    return probabilities


def sample_token(logits: Any, config: SamplingConfig | None = None, *,
                 rng: np.random.Generator | None = None, grammar_session: Any | None = None) -> int:
    """Draw one token, advancing a supplied request-local grammar exactly once."""
    config = config or SamplingConfig()
    probabilities = token_probabilities(logits, config, grammar_session=grammar_session)
    token = int(np.argmax(probabilities)) if config.temperature == 0 else int(
        (rng or np.random.default_rng()).choice(probabilities.size, p=probabilities))
    if grammar_session is not None and grammar_session.advance(token) < 0:
        raise ValueError("sampled token was rejected by the grammar FSM")
    return token
