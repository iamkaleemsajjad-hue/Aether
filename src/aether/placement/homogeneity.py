"""Law I's tolerance, derived from the cost model rather than chosen.

The original design admitted one constant without a derivation: a tensor-parallel
group could only contain devices whose throughput ratio stayed inside ``1.3``.  That
number was a placeholder, and a placeholder in a hard constraint is worse than one in
a score — a constraint *removes* plans from the search space, so nothing downstream can
recover from it being wrong.

What Law I is, and what it is not
---------------------------------
Law I is a **structural** constraint.  A tensor-parallel group synchronises twice per
layer, so its stage time is the max over its members, and the law's only job is to keep
a device out of a synchronous group when the group cannot be balanced around it.

It is emphatically *not* the question "is a wider plan worth it".  That is §08's
marginal-value inequality, and it belongs to the ranking lane, where the σ-gated
tie-break already enforces it.  Putting a value test inside a hard constraint breaks
the fallback ladder: a model that does not fit on one device needs its tensor-parallel
plan *generated* even when the split is a poor trade, because there is nothing else to
fall back to.  The design states the separation directly — the feasibility lane may only
remove plans, the performance lane may only order them — and a value test in the
generator violates it.

So this module computes the structural bound, and reports the value bound alongside it
without letting it prune.

The derivation
--------------
Water-filling can equalise the *divisible* work across heterogeneous devices, which is
why heterogeneity is not harmful in itself.  What it cannot equalise is the
**granularity** of the split: a real tensor-parallel partition assigns whole attention
heads, so a device's share is quantised to multiples of ``1/H`` where ``H`` is the head
count.

Two consequences, and they are the whole law.

**A candidate needs at least one head.**  Its ideal share is ``θ_j/Σθ``, and with
``ρ = θ_A/θ_j`` for the group's aggregate ``θ_A`` that is ``1/(ρ+1)``.  Requiring at
least one head out of ``H`` gives ``ρ ≤ H − 1``.

**Rounding to a head boundary leaves a barrier imbalance.**  In the worst case the
slowest member carries one whole head beyond its ideal share, costing
``(1/H)·W/θ_j``, against an ideal stage time of ``W/Σθ``.  The relative penalty is::

    penalty = (1/H)·W/θ_j ÷ W/Σθ = Σθ/(H·θ_j) = (ρ + 1)/H

and the group is only balanceable if that irreducible penalty stays inside the
planner's own prediction error::

    (ρ + 1)/H ≤ σ_rel   ⟺   ρ ≤ H·σ_rel − 1

That is precisely the criterion the design asked for — "the ratio at which a TP group's
measured time exceeds the water-filled prediction by more than σ" — written
analytically instead of waited for.

**A floor, because θ is measured.**  Two devices whose throughputs differ by less than
the measurement error are indistinguishable, and refusing to group them would be
refusing to group a device with itself.  So the bound never falls below ``1 + σ_rel``.

    tolerance = max( 1 + σ_rel , H·σ_rel − 1 )

Worth noting where that lands: with no calibration at all, ``σ_rel`` is 0.30 for a pair
of devices and the floor is **1.30** — the constant the original design guessed.  It was
the right order of magnitude; what it lacked was a reason and a way to improve.  This
bound tightens as σ shrinks, so a well-calibrated host enforces near-strict homogeneity
while a fresh one stays permissive about *generating* plans and conservative about
*choosing* them, which is the correct direction for both.

Why permissive-when-uncertain is right here
-------------------------------------------
Uncertainty makes the feasibility lane conservative — it refuses.  It makes Law I
permissive — it admits.  That asymmetry is deliberate: admitting a plan costs
microseconds of enumeration and commits to nothing, while removing one is irreversible
and can leave the ladder with no rung. The same uncertainty simultaneously widens the
σ gate, which pushes the *choice* toward the narrower plan. Both effects point the same
way.

The measured crossover
----------------------
Once real heterogeneous groups have run, measurement replaces derivation.  Each
observation brackets the crossover from below (a ratio that met its water-filled
prediction) or above (one that did not), and a bracket in the ledger overrides the
analytic bound.  Measurement beats derivation; derivation beats a constant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aether.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aether.placement.census import DeviceCapability, DeviceCensus
    from aether.placement.ledger import LedgerEntry
    from aether.placement.model_profile import ModelProfile
    from aether.placement.workload import WorkloadEnvelope

logger = get_logger(__name__)

__all__ = [
    "TOLERANCE_PRIOR",
    "HomogeneityBound",
    "HomogeneityLaw",
    "shard_granularity",
]

TOLERANCE_PRIOR = 1.3
"""Fallback ratio for a caller with no model, fabric or workload context.

Retained only so :func:`~aether.placement.plans.capability_groups` stays callable on
its own — from a test or an inspection tool.  The planner never uses it: it supplies a
:class:`HomogeneityLaw`, which derives the bound instead.  Its value is not a
coincidence, it is where the derived bound sits with no calibration at all."""


def shard_granularity(profile: "ModelProfile") -> int:
    """Number of independently assignable units in a tensor-parallel split.

    Attention heads are the coarse unit of a Megatron-style partition — QKV splits by
    head, and the FFN's far finer column split is never the binding granularity — so
    the smallest shard a device can be given is one head out of ``num_heads``.
    """
    return max(1, int(getattr(profile, "num_heads", 0) or 0))


@dataclass(frozen=True)
class HomogeneityBound:
    """One Law I decision, with the arithmetic that produced it."""

    group: tuple[str, ...]
    candidate: str
    ratio: float
    """The observed aggregate-to-candidate throughput ratio being tested."""

    limit: float
    """Largest ratio admissible here. ``ratio <= limit`` means the group admits it."""

    source: str
    """Which term set the limit: ``measured`` | ``granularity`` | ``noise`` |
    ``override``."""

    sigma: float = 0.0
    """Relative prediction error for a group this wide — the σ in the derivation."""

    heads: int = 1
    """Shard granularity: independently assignable units in the split."""

    granularity_limit: float = 0.0
    """``H·σ − 1``: the ratio at which rounding to a head boundary breaks σ."""

    noise_floor: float = 0.0
    """``1 + σ``: below this, two devices are indistinguishable by measurement."""

    samples: int = 0
    """Heterogeneous group observations behind a measured limit."""

    # Reported, never enforced: §08's marginal-value test, which belongs to ranking.
    value_limit: float = 0.0
    """Ratio at which one more rank stops paying for itself. Diagnostics only.

    May be negative, and a negative value is the *most* informative case: it means one
    more rank costs more than it saves at any ratio, so the narrow plan wins on cost
    rather than on structure."""

    value_computed: bool = False
    """Whether :attr:`value_limit` was computed. False means no marginal cost existed."""

    group_time_s: float = 0.0
    marginal_cost_s: float = 0.0
    comm_s: float = 0.0
    dispatch_s: float = 0.0

    @property
    def admitted(self) -> bool:
        return self.ratio <= self.limit

    @property
    def worth_it(self) -> bool:
        """Whether the candidate also clears §08's marginal-value test.

        A group can be structurally admissible and still not worth widening — that is
        the normal case for a small model, and it is the ranking lane's call, not this
        one's. Surfacing it separately is what lets the record explain a narrow plan
        without Law I having to lie about why.
        """
        if not self.value_computed:
            return True
        return self.ratio <= self.value_limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": list(self.group),
            "candidate": self.candidate,
            "ratio": round(self.ratio, 4) if self.ratio != float("inf") else None,
            "limit": round(self.limit, 4),
            "source": self.source,
            "admitted": self.admitted,
            "sigma": round(self.sigma, 4),
            "heads": self.heads,
            "granularity_limit": round(self.granularity_limit, 4),
            "noise_floor": round(self.noise_floor, 4),
            "samples": self.samples,
            "value_limit": round(self.value_limit, 4),
            "value_computed": self.value_computed,
            "worth_it": self.worth_it,
            "group_time_ms": round(self.group_time_s * 1e3, 4),
            "marginal_cost_ms": round(self.marginal_cost_s * 1e3, 4),
            "comm_ms": round(self.comm_s * 1e3, 4),
            "dispatch_ms": round(self.dispatch_s * 1e3, 4),
        }

    def explain(self) -> str:
        """One line naming the bound, the term that set it, and the value test."""
        if self.source == "override":
            head = (
                f"TP homogeneity limit {self.limit:.2f}x, pinned by the caller - the "
                f"derived bound was {max(self.noise_floor, self.granularity_limit):.2f}x"
            )
        elif self.source == "measured":
            head = (
                f"TP homogeneity limit {self.limit:.2f}x, measured crossover over "
                f"{self.samples} heterogeneous group run(s)"
            )
        elif self.source == "granularity":
            head = (
                f"TP homogeneity limit {self.limit:.2f}x = {self.heads} heads x sigma "
                f"{self.sigma:.2f} - 1, the ratio at which whole-head rounding "
                f"exceeds the prediction error"
            )
        else:
            head = (
                f"TP homogeneity limit {self.limit:.2f}x = 1 + sigma {self.sigma:.2f}, "
                f"the floor below which throughputs are indistinguishable"
            )
        if not self.value_computed:
            return head
        if self.value_limit < 1.0:
            # 1.0 is the metric's floor — a group of identical devices — so a bound
            # below it is unreachable, which is the same as "never".
            head += (
                f"; but one more rank never pays here - it saves "
                f"{self.group_time_s * 1e3:.2f} ms of divisible time and costs "
                f"{self.marginal_cost_s * 1e3:.2f} ms (comm {self.comm_s * 1e3:.2f} + "
                f"dispatch {self.dispatch_s * 1e3:.2f}), so a wider plan loses on cost "
                f"rather than on structure"
            )
        else:
            head += (
                f"; one more rank pays for itself below {self.value_limit:.2f}x "
                f"({self.group_time_s * 1e3:.2f} ms divisible vs "
                f"{self.marginal_cost_s * 1e3:.2f} ms marginal cost)"
            )
        return head


@dataclass
class HomogeneityLaw:
    """Law I as a predicate, with the tolerance derived per candidate group.

    Call the instance with ``(group, candidate, theta)`` to ask whether the group may
    admit the device.  Every decision is retained in :attr:`decisions`, so the planner
    can print the bound that shaped the plan space instead of asserting a constant.

    The bound is structural only: it asks whether the group can be *balanced* around
    the candidate, never whether widening is worthwhile.  The marginal-value arithmetic
    is computed alongside and reported, because it is what explains a narrow choice —
    but it does not prune, or a model that fits on no single device would lose the only
    plan it has.

    Args:
        profile: The model, for the head granularity and the op count.
        census: The machine, for the fabric between candidate members.
        workload: The envelope, for the prefill/decode blend and the batch.
        entry_for: Calibration lookup for one device — normally
            :meth:`~aether.placement.planner.ExecutionPlanner.entry_for`.
        latency_sigma: Relative prediction error for a group of ``n`` devices. The
            planner passes its own, so Law I and the σ-gated tie-break are statements
            about one error bar rather than two that can drift apart.
        override: A fixed ratio that replaces the derivation entirely. For an operator
            reproducing an old decision or bisecting a suspected planner bug — not a
            default, and reported as ``override`` in the record so a pinned bound can
            never be mistaken for a derived one.
    """

    profile: "ModelProfile"
    census: "DeviceCensus"
    workload: "WorkloadEnvelope"
    entry_for: "Callable[[DeviceCapability], LedgerEntry]"
    latency_sigma: "Callable[[int], float]"
    override: float | None = None
    decisions: list[HomogeneityBound] = field(default_factory=list)
    _cache: dict[tuple[tuple[str, ...], str], HomogeneityBound] = field(
        default_factory=dict, repr=False
    )

    # ── the predicate ─────────────────────────────────────────────────────────

    def __call__(
        self,
        group: "tuple[DeviceCapability, ...]",
        candidate: "DeviceCapability",
        theta: "dict[str, float]",
    ) -> bool:
        """Whether ``group`` may take ``candidate`` into one tensor-parallel stage."""
        return self.bound(group, candidate, theta).admitted

    def bound(
        self,
        group: "tuple[DeviceCapability, ...]",
        candidate: "DeviceCapability",
        theta: "dict[str, float]",
    ) -> HomogeneityBound:
        """The full Law I arithmetic for one candidate, cached per group."""
        key = (tuple(device.device_id for device in group), candidate.device_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        bound = self._derive(group, candidate, theta)
        self._cache[key] = bound
        self.decisions.append(bound)
        return bound

    # ── derivation ────────────────────────────────────────────────────────────

    def _derive(
        self,
        group: "tuple[DeviceCapability, ...]",
        candidate: "DeviceCapability",
        theta: "dict[str, float]",
    ) -> HomogeneityBound:
        ids = tuple(device.device_id for device in group)
        aggregate = sum(max(theta.get(device.device_id, 0.0), 0.0) for device in group)
        own = max(theta.get(candidate.device_id, 0.0), 0.0)
        heads = shard_granularity(self.profile)
        sigma = max(0.0, self.latency_sigma(len(group) + 1))
        noise_floor = 1.0 + sigma
        granularity_limit = heads * sigma - 1.0

        if own <= 0 or aggregate <= 0:
            # A device with no measurable throughput cannot be balanced against one
            # that has any, so it is never admitted. This is the rule that keeps a
            # CPU out of a TP group without naming the CPU.
            return HomogeneityBound(
                group=ids, candidate=candidate.device_id, ratio=float("inf"),
                limit=noise_floor, source="noise", sigma=sigma, heads=heads,
                granularity_limit=granularity_limit, noise_floor=noise_floor,
            )

        ratio = aggregate / own
        limit = max(noise_floor, granularity_limit)
        source = "granularity" if granularity_limit >= noise_floor else "noise"

        samples = 0
        measured = self._measured_crossover(group)
        if measured is not None:
            crossover, samples, proven = measured
            if crossover > 0:
                # A measured crossover replaces the analytic bound outright: it is the
                # ratio at which a real group missed its water-filled prediction.
                limit = max(1.0, crossover)
                source = "measured"
            elif proven > limit:
                # No upper bracket yet, but a ratio this wide has been *observed to
                # work*. Refusing it now would contradict our own evidence.
                limit = proven
                source = "measured"

        if self.override is not None:
            # An explicit pin outranks both, and is labelled so, because a bound the
            # operator chose must never be read back as one the planner derived.
            limit = max(1.0, float(self.override))
            source = "override"

        # §08's marginal-value test, computed for the record and deliberately not
        # applied. Whether widening is worth it is the ranking lane's decision; a
        # value test here would delete the only plan a too-large model has.
        divisible = self._divisible_time(group + (candidate,))
        marginal, comm, dispatch = self._marginal_cost(group, candidate)
        value_limit = (divisible / marginal - 1.0) if marginal > 0 else 0.0

        return HomogeneityBound(
            group=ids, candidate=candidate.device_id, ratio=ratio, limit=limit,
            source=source, sigma=sigma, heads=heads,
            granularity_limit=granularity_limit, noise_floor=noise_floor,
            samples=samples, value_limit=value_limit, value_computed=marginal > 0,
            group_time_s=divisible, marginal_cost_s=marginal,
            comm_s=comm, dispatch_s=dispatch,
        )

    def _divisible_time(self, group: "tuple[DeviceCapability, ...]") -> float:
        """Stage time for the part of the work that adding throughput actually shrinks.

        Only the compute and bandwidth terms are divisible; host dispatch is not, and
        counting it here would make a dispatch-bound model look like it had room to
        absorb another device.  The two phases are blended by the workload's own token
        mix, so the figure is phase-correct without a second formula.
        """
        weight_bytes = float(self.profile.weight_bytes)
        params = max(1.0, float(self.profile.params))
        batch = max(1, self.workload.batch_target)
        bandwidth = sum(max(d.effective_bandwidth_bps, 0.0) for d in group)
        flops = sum(max(d.effective_flops, 0.0) for d in group)

        decode = weight_bytes / bandwidth if bandwidth > 0 else 0.0
        # Prefill per token: 2·B·params FLOPs per token, so the sequence length
        # cancels — the same cancellation that makes the comm ratio in §06 independent
        # of batch and context.
        prefill = (2.0 * batch * params / flops) if flops > 0 else 0.0

        fraction = self.workload.effective_prefill_fraction
        return (1.0 - fraction) * decode + fraction * prefill

    def _marginal_cost(
        self,
        group: "tuple[DeviceCapability, ...]",
        candidate: "DeviceCapability",
    ) -> tuple[float, float, float]:
        """``(total, comm, dispatch)`` cost of taking the group one rank wider.

        Reported rather than enforced — see :attr:`HomogeneityBound.value_limit`.
        """
        from aether.placement.cost import tp_comm_seconds

        members = tuple(device.device_id for device in group)
        widened = members + (candidate.device_id,)
        lead = self.entry_for(group[0])
        layers = max(1, self.profile.layers)
        batch = max(1, self.workload.batch_target)
        context = max(1, self.workload.context_target)
        fraction = self.workload.effective_prefill_fraction

        def comm(device_ids: "tuple[str, ...]") -> float:
            if len(device_ids) < 2:
                return 0.0
            decode = tp_comm_seconds(
                self.profile, self.census, device_ids, layers=layers,
                batch=batch, step_tokens=1,
                dispatch_seconds=lead.dispatch_seconds,
            )
            prefill = tp_comm_seconds(
                self.profile, self.census, device_ids, layers=layers,
                batch=batch, step_tokens=context,
                dispatch_seconds=lead.dispatch_seconds,
            )
            return (1.0 - fraction) * decode + fraction * prefill / context

        comm_delta = max(0.0, comm(widened) - comm(members))

        # One more rank re-issues every sharded operation on the same serial host
        # thread, and crossing from one device to two also adds the two all-reduce
        # launches per layer. Prefill amortises the host cost over the sequence.
        extra_ops = self.profile.ops_per_layer * layers
        if len(members) == 1:
            extra_ops += 2 * layers
        dispatch_decode = extra_ops * max(0.0, lead.dispatch_seconds)
        dispatch_delta = (1.0 - fraction) * dispatch_decode + fraction * dispatch_decode / context

        return comm_delta + dispatch_delta, comm_delta, dispatch_delta

    def _measured_crossover(
        self, group: "tuple[DeviceCapability, ...]"
    ) -> tuple[float, int, float] | None:
        """``(crossover, samples, proven_good)`` from the ledger, or ``None``.

        Keyed on the group's *lead* device, because that is the rank whose calibration
        the group's predicted time was built from.
        """
        entry = self.entry_for(group[0])
        crossover = float(getattr(entry, "tp_crossover_ratio", 0.0) or 0.0)
        proven = float(getattr(entry, "tp_ratio_ok_max", 0.0) or 0.0)
        samples = int(getattr(entry, "tp_samples", 0) or 0)
        if samples <= 0:
            return None
        return crossover, samples, proven

    # ── reporting ─────────────────────────────────────────────────────────────

    def binding_bound(self) -> HomogeneityBound | None:
        """The decision that most constrained the plan space, for the record.

        A *rejection* is more informative than an admission — it names hardware the
        planner declined to combine — so a rejection wins, and among rejections the
        one closest to its limit, since that is the marginal call.
        """
        if not self.decisions:
            return None
        rejected = [bound for bound in self.decisions if not bound.admitted]
        pool = rejected or list(self.decisions)
        return min(pool, key=lambda bound: abs(bound.ratio - bound.limit))

    def to_dict(self) -> dict[str, Any]:
        binding = self.binding_bound()
        return {
            "evaluated": len(self.decisions),
            "rejected": sum(1 for bound in self.decisions if not bound.admitted),
            "binding": binding.to_dict() if binding is not None else None,
        }
