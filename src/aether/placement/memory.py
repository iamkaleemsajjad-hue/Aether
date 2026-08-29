"""Safe capacity, the memory residual, and ``tokens_max``.

The formulation, in one line: **KV cache is not a requirement, it is what is left
over.**  Subtract everything that is not cache from what the device can safely give,
and the remainder — expressed in tokens — is the answer to "does this fit", "how much
room is left", "what happens if the batch doubles" and "which device binds".

    C_safe(d) = min( free(d) − R_ext(d), total(d)·κ ) − R_fixed(d)
    K(d)      = C_safe(d) − S(d) − ( T̂(d) + z·σ_T(d) )
    tokens_max = min_d ⌊ K(d) / kv_per_token(d) ⌋

κ is the only percentage in the planner, and its job is to absorb driver-side growth
rather than to test fit.  Everything else is bytes: measured free memory, measured
external occupancy, calibrated fixed overhead, and a one-sided margin whose width
comes from recorded prediction error.

Error handling here is deliberately asymmetric.  Over-predicting the transient peak
costs KV capacity; under-predicting it kills the process.  So the bound is one-sided
and the margin is added, never subtracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether.placement.census import DeviceCapability
from aether.placement.ledger import LedgerEntry
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "KAPPA_DEFAULT",
    "Z_DEFAULT",
    "MemoryBudget",
    "safe_capacity",
    "evaluate_budget",
]

KAPPA_DEFAULT = 0.95
"""Hard ceiling on total device memory. Absorbs driver growth, not model fit."""

Z_DEFAULT = 3.0
"""One-sided quantile for the transient-peak bound. 3.0 for a serving process that
must not die; 1.5 is defensible for a batch job that can simply retry."""


@dataclass(frozen=True)
class MemoryBudget:
    """The residual arithmetic for one device under one plan.

    Every field is a byte count the planner actually computed, which is why the
    decision record can print the whole derivation instead of a verdict.
    """

    device_id: str
    total_bytes: int
    free_bytes: int
    external_bytes: int
    safe_capacity_bytes: int
    weight_bytes: int
    persistent_bytes: int
    static_bytes: int
    transient_bytes: int
    transient_margin_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_token: int
    tokens_max: int
    calibrated: bool
    fragmentation: float

    @property
    def feasible(self) -> bool:
        """Whether this device can hold the plan at all, before any token of cache."""
        return self.kv_budget_bytes > 0

    @property
    def predicted_peak_bytes(self) -> int:
        """Static plus transient, without the safety margin.

        This is the number the ledger compares against an observed peak — the
        margin is the planner's uncertainty, not part of its prediction, and folding
        it in would make the residual statistics measure the margin instead of the
        estimator.
        """
        return self.static_bytes + self.transient_bytes

    def shortfall_bytes(self, required_tokens: int) -> int:
        """Bytes still needed to hold ``required_tokens`` of cache. Zero when it fits."""
        needed = required_tokens * self.kv_bytes_per_token
        return max(0, needed - self.kv_budget_bytes)

    def to_dict(self) -> dict[str, Any]:
        gib = 1024 ** 3
        return {
            "device_id": self.device_id,
            "total_gib": round(self.total_bytes / gib, 2),
            "free_gib": round(self.free_bytes / gib, 2),
            "external_gib": round(self.external_bytes / gib, 2),
            "safe_capacity_gib": round(self.safe_capacity_bytes / gib, 2),
            "weight_gib": round(self.weight_bytes / gib, 3),
            "static_gib": round(self.static_bytes / gib, 3),
            "transient_gib": round(self.transient_bytes / gib, 3),
            "margin_gib": round(self.transient_margin_bytes / gib, 3),
            "kv_budget_gib": round(self.kv_budget_bytes / gib, 3),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "tokens_max": self.tokens_max,
            "feasible": self.feasible,
            "calibrated": self.calibrated,
            "fragmentation": round(self.fragmentation, 3),
        }


def safe_capacity(
    device: DeviceCapability,
    entry: LedgerEntry,
    *,
    kappa: float = KAPPA_DEFAULT,
) -> int:
    """Bytes the planner may commit on this device.

    Two independent limits, and the smaller wins. ``free − external`` is what is
    physically available right now with other tenants respected; ``total·κ`` is a
    ceiling that keeps the driver from being squeezed as it grows. Then the
    calibrated fixed overhead — CUDA context, library workspace, collective buffers
    — comes off the top, because it is spent before the first model byte lands.
    """
    if not 0.0 < kappa <= 1.0:
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")
    available = max(0, device.free_bytes - device.external_bytes)
    ceiling = int(device.total_bytes * kappa)
    return max(0, min(available, ceiling) - max(0, entry.r_fixed_bytes))


def evaluate_budget(
    device: DeviceCapability,
    entry: LedgerEntry,
    *,
    weight_bytes: int,
    persistent_bytes: int,
    activation_bytes: int,
    kv_bytes_per_token: int,
    kappa: float = KAPPA_DEFAULT,
    z: float = Z_DEFAULT,
) -> MemoryBudget:
    """Compute the residual for one device under one plan's footprint.

    Args:
        weight_bytes: This device's *sharded* weight bytes — exact, from the profile.
        persistent_bytes: Rope tables, logit buffers, comm staging on this device.
        activation_bytes: Predicted transient peak, before fragmentation.
        kv_bytes_per_token: This device's share of the cache, per token.
        kappa: Device-memory ceiling fraction.
        z: One-sided quantile for the transient margin.

    Returns:
        A :class:`MemoryBudget`. A negative KV budget is returned rather than
        raised, because the caller needs the number to explain *why* a plan failed.
    """
    capacity = safe_capacity(device, entry, kappa=kappa)
    static = int(weight_bytes + persistent_bytes)
    # Fragmentation is multiplicative on the transient pool, not additive: the
    # allocator's inability to merge blocks across segments inflates the reserved
    # bytes needed to satisfy a given peak allocation.
    transient = int(activation_bytes * max(1.0, entry.fragmentation))
    # The margin is taken over the *transient* term only. Weight bytes are read
    # from the AEG tensor table and carry no error, so inflating them by an
    # uncertainty fraction would reserve gigabytes against a quantity that is known
    # exactly — which is how a 13 B model on two 16 GiB cards becomes "infeasible".
    # Once calibrated, `margin_bytes` uses the measured residual sigma and ignores
    # this argument entirely.
    margin = entry.margin_bytes(transient, z)
    kv_budget = capacity - static - transient - margin
    per_token = max(1, int(kv_bytes_per_token))
    tokens = max(0, kv_budget) // per_token
    return MemoryBudget(
        device_id=device.device_id,
        total_bytes=device.total_bytes,
        free_bytes=device.free_bytes,
        external_bytes=device.external_bytes,
        safe_capacity_bytes=capacity,
        weight_bytes=int(weight_bytes),
        persistent_bytes=int(persistent_bytes),
        static_bytes=static,
        transient_bytes=transient,
        transient_margin_bytes=margin,
        kv_budget_bytes=kv_budget,
        kv_bytes_per_token=per_token,
        tokens_max=int(tokens),
        calibrated=entry.is_calibrated,
        fragmentation=entry.fragmentation,
    )
