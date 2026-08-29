"""Capped water-filling, and the greedy fill that is correct for pipeline latency.

Three placement problems appear in the planner, and only two of them are min-max.
Using the wrong one is not a small inefficiency: balancing a pipeline whose stage
times *add* makes single-token latency worse, not better.

    TP shard fractions          min max_i t_i      water-fill ∝ θ, capped
    PP layers, throughput       min max_i t_i      water-fill ∝ θ, capped
    PP layers, latency          min Σ_i t_i        greedy, fastest device first

The water-filling algorithm is the standard fixed point for min-max under one
linear resource with per-element caps: distribute proportionally to throughput, pin
anything that exceeds its cap, redistribute the remainder, repeat.  It terminates in
at most ``P`` rounds because each round pins at least one element, and the result is
the unique optimum — at the fixed point every unpinned element has equal ``t_i`` and
every pinned one is at its bound, so no exchange can lower the maximum.
"""

from __future__ import annotations

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "WaterfillInfeasible",
    "water_fill",
    "greedy_fill",
    "equalise_slack",
    "stage_time_units",
]


class WaterfillInfeasible(ValueError):
    """Raised when the caps cannot hold the total — no partition exists."""


def water_fill(
    total: float,
    throughputs: "list[float] | tuple[float, ...]",
    caps: "list[float] | tuple[float, ...] | None" = None,
) -> list[float]:
    """Return fractions minimising ``max_i f_i·total/θ_i`` subject to the caps.

    Args:
        total: The quantity being divided — bytes of weight, or layer count.
        throughputs: Per-device throughput on the binding roof. Bandwidth for
            decode, FLOPs for prefill; the caller decides which, because the
            correct choice is a property of the phase and not of this function.
        caps: Per-device upper bounds in the same units as ``total``. ``None``
            means unbounded.

    Returns:
        Fractions summing to 1.0, in the input order.

    Raises:
        ValueError: If the inputs are malformed.
        WaterfillInfeasible: If ``sum(caps) < total``.
    """
    count = len(throughputs)
    if count == 0:
        raise ValueError("at least one device is required")
    if any(value <= 0 for value in throughputs):
        raise ValueError("throughputs must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    if count == 1:
        if caps is not None and caps[0] < total:
            raise WaterfillInfeasible(
                f"one device with a cap of {caps[0]:.0f} cannot hold {total:.0f}"
            )
        return [1.0]

    limits = [float("inf")] * count if caps is None else [float(value) for value in caps]
    if sum(limits) < total:
        raise WaterfillInfeasible(
            f"aggregate capacity {sum(limits):.0f} is below the required {total:.0f}"
        )
    if total == 0:
        return [1.0 / count] * count

    amounts = [0.0] * count
    pinned = [False] * count
    remaining = float(total)

    for _ in range(count + 1):
        open_indices = [index for index in range(count) if not pinned[index]]
        if not open_indices:
            break
        open_throughput = sum(throughputs[index] for index in open_indices)
        overflowed = False
        for index in open_indices:
            share = remaining * throughputs[index] / open_throughput
            if share > limits[index]:
                # This device cannot absorb its proportional share; pin it at its
                # cap and let the rest of the mass redistribute among the others.
                amounts[index] = limits[index]
                pinned[index] = True
                remaining -= limits[index]
                overflowed = True
                break
            amounts[index] = share
        if not overflowed:
            break
    else:  # pragma: no cover - the loop bound above is P + 1 by construction
        raise WaterfillInfeasible("water-filling failed to converge")

    assigned = sum(amounts)
    if assigned <= 0:
        raise WaterfillInfeasible("water-filling produced an empty assignment")
    # Renormalise against the assigned total rather than `total` so floating-point
    # drift in the pinning arithmetic cannot make the fractions miss 1.0.
    return [value / assigned for value in amounts]


def greedy_fill(
    total: float,
    throughputs: "list[float] | tuple[float, ...]",
    caps: "list[float] | tuple[float, ...] | None" = None,
) -> list[float]:
    """Return fractions minimising ``Σ_i f_i·total/θ_i`` subject to the caps.

    This is the correct rule for **pipeline latency**, where a single token
    traverses every stage in sequence so the stage times add.  Balancing them —
    the instinct that water-filling encodes — is actively wrong here: the optimum
    puts as much work as possible on the fastest device and only the spill on the
    slower ones.

    Raises:
        WaterfillInfeasible: If ``sum(caps) < total``.
    """
    count = len(throughputs)
    if count == 0:
        raise ValueError("at least one device is required")
    if any(value <= 0 for value in throughputs):
        raise ValueError("throughputs must be positive")
    limits = [float("inf")] * count if caps is None else [float(value) for value in caps]
    if sum(limits) < total:
        raise WaterfillInfeasible(
            f"aggregate capacity {sum(limits):.0f} is below the required {total:.0f}"
        )
    if total <= 0:
        return [1.0 / count] * count

    amounts = [0.0] * count
    remaining = float(total)
    for index in sorted(range(count), key=lambda i: -throughputs[i]):
        take = min(remaining, limits[index])
        amounts[index] = take
        remaining -= take
        if remaining <= 0:
            break
    assigned = sum(amounts)
    if assigned <= 0:
        raise WaterfillInfeasible("greedy fill produced an empty assignment")
    return [value / assigned for value in amounts]


def stage_time_units(
    total: float, fractions: "list[float]", throughputs: "list[float] | tuple[float, ...]"
) -> float:
    """Max over devices of ``f_i·total/θ_i`` — the barrier time for a split."""
    return max(
        fraction * total / throughput
        for fraction, throughput in zip(fractions, throughputs, strict=False)
    )


def equalise_slack(total: float, caps: "list[float] | tuple[float, ...]") -> list[float]:
    """Return fractions that leave every device the *same* spare bytes.

    Water-filling by throughput minimises stage time, and that is the right
    objective when the plan is compute- or bandwidth-bound.  It is the wrong one when
    the plan is *capacity*-bound, because the leftover bytes are the KV budget: a
    time-optimal split on a 16 GiB + 24 GiB pair leaves the small device with a
    fraction of the cache the large one has, and ``tokens_max`` is the minimum over
    devices, so the small device throws away the large one's headroom.

    Equalising slack solves that in closed form.  With slack ``s`` on every device,
    ``Σ(cap_i − s) = total``, so ``s = (Σcap − total)/P`` and ``f_i = (cap_i − s)/total``.
    Devices whose cap is below the mean slack are pinned at zero and the rest is
    redistributed, so the result stays a valid partition.

    Raises:
        WaterfillInfeasible: If ``sum(caps) < total``.
    """
    count = len(caps)
    if count == 0:
        raise ValueError("at least one device is required")
    limits = [float(value) for value in caps]
    if sum(limits) < total:
        raise WaterfillInfeasible(
            f"aggregate capacity {sum(limits):.0f} is below the required {total:.0f}"
        )
    if total <= 0 or count == 1:
        return [1.0 / count] * count

    amounts = [0.0] * count
    active = list(range(count))
    remaining = float(total)
    for _ in range(count + 1):
        if not active:
            break
        slack = (sum(limits[i] for i in active) - remaining) / len(active)
        negative = [i for i in active if limits[i] - slack <= 0]
        if not negative:
            for i in active:
                amounts[i] = limits[i] - slack
            break
        # A device too small to hold even an equal-slack share takes nothing; the
        # remainder redistributes over the others.
        for i in negative:
            amounts[i] = 0.0
            active.remove(i)
    assigned = sum(amounts)
    if assigned <= 0:
        raise WaterfillInfeasible("slack equalisation produced an empty assignment")
    return [value / assigned for value in amounts]
