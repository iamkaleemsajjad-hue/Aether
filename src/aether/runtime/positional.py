"""Model-generic positional-bias utilities used by portable runtimes."""

from __future__ import annotations

import math

import numpy as np


def alibi_slopes(num_heads: int) -> np.ndarray:
    """Return deterministic ALiBi slopes for any positive head count."""

    heads = int(num_heads)
    if heads <= 0:
        raise ValueError("ALiBi requires a positive head count")

    def power_of_two_slopes(count: int) -> list[float]:
        start = 2.0 ** (-2.0 ** -(math.log2(count) - 3.0))
        return [start * start**index for index in range(count)]

    if heads & (heads - 1) == 0:
        values = power_of_two_slopes(heads)
    else:
        lower_power = 2 ** int(math.floor(math.log2(heads)))
        values = power_of_two_slopes(lower_power)
        values.extend(power_of_two_slopes(lower_power * 2)[0::2][: heads - lower_power])
    return np.asarray(values[:heads], dtype=np.float32)

