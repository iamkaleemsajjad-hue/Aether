"""Portable, JSON-safe Mamba-2 source execution parameters.

The source contract is Transformers v4.57.3 Mamba2Mixer.torch_forward:
https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mamba2/modeling_mamba2.py
Convolution uses hidden_act; the post-convolution gate remains SiLU. Time-step
limits clamp softplus(dt+bias), before both SSD decay and input injection.
The underlying recurrence is Dao & Gu (2024), equations 2/3 and section 7.1:
https://arxiv.org/html/2405.21060v1
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class Mamba2Parameters:
    """Validated constants shared by compilation and actual CPU/Torch execution."""

    conv_activation: str = "silu"
    time_step_limit: tuple[float | None, float | None] | None = None

    def __post_init__(self) -> None:
        if self.conv_activation not in ("silu", "swish", "relu"):
            raise ValueError(
                f"Unsupported Mamba-2 hidden_act activation {self.conv_activation!r}; "
                "the CPU and Torch executors support 'silu', 'swish' and 'relu'."
            )
        limits = self.time_step_limit
        if limits is None:
            return
        if not isinstance(limits, (tuple, list)) or len(limits) != 2:
            raise ValueError("Mamba-2 time_step_limit must contain exactly two bounds")
        lower, upper = limits
        if (lower is not None and (isinstance(lower, bool) or not isinstance(lower, Real))) or (
            upper is not None and (isinstance(upper, bool) or not isinstance(upper, Real))
        ):
            raise ValueError("Mamba-2 time_step_limit bounds must be numbers")
        try:
            minimum = None if lower is None else float(lower)
            maximum = None if upper is None else float(upper)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "Mamba-2 time_step_limit bounds exceed supported FP32 arithmetic"
            ) from exc
        if minimum == -math.inf:
            minimum = None
        if maximum == math.inf:
            maximum = None
        largest = float.fromhex("0x1.fffffep127")
        if (minimum is not None and (not math.isfinite(minimum) or abs(minimum) > largest)) or (
            maximum is not None and (not math.isfinite(maximum) or abs(maximum) > largest)
        ):
            raise ValueError(
                "Mamba-2 time_step_limit requires finite FP32 bounds or unbounded limits"
            )
        # Match torch.clamp even for min > max: every value becomes max.
        normalized = None if minimum in (0, None) and maximum is None else (minimum, maximum)
        object.__setattr__(self, "time_step_limit", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Encode unbounded limits as JSON null, never Infinity/NaN."""
        return {
            "conv_activation": self.conv_activation,
            "time_step_limit": None if self.time_step_limit is None else list(self.time_step_limit),
        }

    @classmethod
    def from_source_config(cls, config: Mapping[str, Any]) -> Mamba2Parameters:
        """Reject malformed source bounds before they can disappear in packaging."""
        limits = config.get("time_step_limit")
        if "time_step_limit" in config:
            if not isinstance(limits, (list, tuple)) or len(limits) != 2 or any(
                value is not None and (isinstance(value, bool) or not isinstance(value, Real))
                for value in limits
            ):
                raise ValueError(
                    "Mamba-2 source time_step_limit must contain exactly two "
                    "numeric/optional bounds"
                )
            if limits[0] is None and limits[1] is None:
                raise ValueError("Mamba-2 source time_step_limit needs at least one bound")
            limits = (limits[0], limits[1])
        return cls(config.get("hidden_act", "silu"), limits)

    @classmethod
    def from_dict(cls, value: Any) -> Mamba2Parameters:
        """Decode authenticated AEG parameters; an absent envelope uses defaults."""
        if not isinstance(value, Mapping):
            raise ValueError("Mamba-2 runtime parameters must be a JSON object")
        if set(value) - {"conv_activation", "time_step_limit"}:
            raise ValueError("Mamba-2 runtime parameters contain unsupported fields")
        limits = value.get("time_step_limit")
        parameters = cls(value.get("conv_activation", "silu"), limits)
        if limits is not None and any(
            isinstance(bound, Real) and not math.isfinite(float(bound)) for bound in limits
        ):
            raise ValueError("AEG Mamba-2 time_step_limit must encode unbounded limits as null")
        return parameters
