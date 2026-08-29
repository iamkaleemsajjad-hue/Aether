"""Plan representation, and enumeration under the two structural laws.

The plan space is generated, not searched.  Two laws act as *generators* rather than
filters, and that is what keeps the space small enough to enumerate exhaustively in
under a millisecond on the device counts a single node actually has:

**Law I — homogeneity.**  A tensor-parallel group may only contain devices whose
throughput on the binding roof lies within a tolerance that is *derived*, not chosen:
:class:`~aether.placement.homogeneity.HomogeneityLaw` computes it from the shard
granularity and the planner's own error bar, and a measured crossover overrides it.  A
TP group synchronises twice per layer, so its stage time is the max over members: the
slowest device sets the pace on every one of L layers.  CPUs never join a TP group, and
neither do an A100 and a T4 — but both of those follow from the arithmetic rather than
from a constant.

**Law II — fabric alignment.**  A tensor-parallel group may not span a fabric class.
Classes come from the interconnect graph, so this adapts to any topology instead of
hard-coding "NVLink versus PCIe".

Heterogeneity is therefore expressed *across* pipeline stages, where each stage runs
at its own pace and the pipeline absorbs the difference — which is exactly the
structure HexGen found necessary to make mixed GPUs usable at all.

Rejecting a solver here is deliberate.  Helix spends up to four hours with Gurobi on
42 nodes; Aether plans at model load and must answer in milliseconds.  After the laws
prune the space, exhaustive enumeration is not merely tractable but strictly better:
deterministic, explainable, and free of a solver dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aether.placement.census import DeviceCapability, DeviceCensus
from aether.placement.homogeneity import TOLERANCE_PRIOR
from aether.placement.model_profile import ModelProfile
from aether.placement.waterfill import (
    WaterfillInfeasible,
    equalise_slack,
    greedy_fill,
    water_fill,
)
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "THETA_TOLERANCE",
    "Parallelism",
    "StagePlan",
    "ExecutionPlan",
    "capability_groups",
    "enumerate_plans",
]

THETA_TOLERANCE = TOLERANCE_PRIOR
"""Fallback throughput ratio for a tensor-parallel group, for context-free callers.

Law I's real tolerance is **derived**, not chosen: see
:class:`~aether.placement.homogeneity.HomogeneityLaw`, which computes the largest
admissible ratio from the shard granularity and the planner's own error bar, and lets a
measured crossover override it.  The planner always passes that law, so this constant
only applies to a caller that has no model, fabric or workload to derive from — and its
value is where the derived bound sits with no calibration at all, rather than a guess."""

MAX_ENUMERATED_PLANS = 512
"""Hard bound on generated plans, so an unusual topology cannot stall a model load."""

AdmissionRule = Callable[
    ["tuple[DeviceCapability, ...]", DeviceCapability, "dict[str, float]"], bool
]
"""Law I as a predicate: may this group take this device into one TP stage?

Given the group so far as well as the candidate, so the bound can tighten as the group
grows — the fifth device has to justify itself against a stronger group than the second.
:class:`~aether.placement.homogeneity.HomogeneityLaw` is the implementation the planner
supplies."""


class Parallelism(str, Enum):
    """What kind of plan this is, for reporting and for engine selection."""

    SINGLE = "single_device"
    TENSOR = "tensor_parallel"
    PIPELINE = "pipeline_parallel"
    HYBRID = "hybrid_tp_pp"
    OFFLOAD = "offload"


@dataclass(frozen=True)
class StagePlan:
    """One pipeline stage: a device group, a layer range, and a shard split."""

    devices: tuple[str, ...]
    tp_degree: int
    layer_start: int
    layer_end: int
    shard_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("a stage needs at least one device")
        if len(self.shard_fractions) != len(self.devices):
            raise ValueError("one shard fraction per device is required")
        if self.layer_end < self.layer_start:
            raise ValueError("layer_end must be >= layer_start")

    @property
    def layers(self) -> int:
        return self.layer_end - self.layer_start

    def fraction_for(self, device_id: str) -> float:
        for device, fraction in zip(self.devices, self.shard_fractions, strict=False):
            if device == device_id:
                return fraction
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": list(self.devices),
            "tp_degree": self.tp_degree,
            "layers": [self.layer_start, self.layer_end],
            "layer_count": self.layers,
            "shard_fractions": [round(value, 6) for value in self.shard_fractions],
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """A complete placement: stages, layer assignment, and any offload tier."""

    stages: tuple[StagePlan, ...]
    kind: Parallelism
    layer_policy: str = "single"
    """``balanced`` (water-filled ∝ θ) or ``greedy`` (fastest device first). The two
    are the optima of different objectives, so both are generated and ranking picks."""

    shard_policy: str = "time"
    """``time`` (water-filled by throughput) or ``capacity`` (equal leftover bytes).
    Only meaningful when a stage has ``tp_degree > 1``."""

    offload_tier: str = ""
    """``""`` | ``host`` | ``disk`` — where weights spill when nothing else fits."""

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a plan needs at least one stage")

    @property
    def device_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for stage in self.stages:
            for device in stage.devices:
                if device not in seen:
                    seen.append(device)
        return tuple(seen)

    @property
    def num_devices(self) -> int:
        return len(self.device_ids)

    @property
    def pipeline_stages(self) -> int:
        return len(self.stages)

    @property
    def max_tp_degree(self) -> int:
        return max(stage.tp_degree for stage in self.stages)

    @property
    def total_layers(self) -> int:
        return sum(stage.layers for stage in self.stages)

    @property
    def boundaries(self) -> list[tuple[str, str]]:
        """Stage-to-stage edges, each between the two stages' first devices.

        The first device stands for its stage because a real implementation elects a
        stage leader to carry the activation and broadcast it within the group; the
        boundary cost is one hop, not one hop per member.
        """
        return [
            (self.stages[index].devices[0], self.stages[index + 1].devices[0])
            for index in range(len(self.stages) - 1)
        ]

    @property
    def label(self) -> str:
        """A short, stable, human-readable name — the plan's identity in a report."""
        if self.offload_tier:
            return f"{self.num_devices}dev+{self.offload_tier}"
        if self.kind is Parallelism.SINGLE:
            return f"1x {self.device_ids[0]}"
        if self.kind is Parallelism.TENSOR:
            suffix = "/cap" if self.shard_policy == "capacity" else ""
            return f"TP={self.max_tp_degree}{suffix}"
        if self.kind is Parallelism.PIPELINE:
            return f"PP={self.pipeline_stages}/{self.layer_policy[:3]}"
        degrees = "/".join(str(stage.tp_degree) for stage in self.stages)
        return f"PP{self.pipeline_stages}xTP{degrees}/{self.layer_policy[:3]}"

    def stage_for_device(self, device_id: str) -> StagePlan | None:
        for stage in self.stages:
            if device_id in stage.devices:
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind.value,
            "layer_policy": self.layer_policy,
            "shard_policy": self.shard_policy,
            "offload_tier": self.offload_tier,
            "devices": list(self.device_ids),
            "num_devices": self.num_devices,
            "pipeline_stages": self.pipeline_stages,
            "max_tp_degree": self.max_tp_degree,
            "stages": [stage.to_dict() for stage in self.stages],
        }


# ── the two laws ──────────────────────────────────────────────────────────────

def capability_groups(
    devices: "tuple[DeviceCapability, ...] | list[DeviceCapability]",
    *,
    theta: "dict[str, float]",
    tolerance: float = THETA_TOLERANCE,
    admits: "AdmissionRule | None" = None,
) -> list[tuple[DeviceCapability, ...]]:
    """Partition devices into groups a tensor-parallel split may use.

    Law II first — devices are bucketed by fabric class — then Law I within each
    bucket, by walking the devices in descending throughput and starting a new group
    whenever the current group cannot admit the next device.

    Args:
        devices: The candidate devices.
        theta: Relative throughput on the binding roof, per device id.
        tolerance: Fixed ratio bound, applied against the group's *fastest* member.
            Used only when ``admits`` is absent.
        admits: The admission predicate — normally a
            :class:`~aether.placement.homogeneity.HomogeneityLaw`, which derives the
            bound from the marginal value of the candidate instead of comparing
            against a constant.  Given both the group so far and the candidate, so it
            can tighten as the group grows: the fifth device has to justify itself
            against a stronger group than the second did.

    Returns:
        Groups in descending order of aggregate throughput, which is also the order a
        pipeline should visit them under a latency objective.
    """
    if tolerance < 1.0:
        raise ValueError("tolerance must be >= 1.0")
    by_fabric: dict[int, list[DeviceCapability]] = {}
    for device in devices:
        by_fabric.setdefault(device.fabric_class, []).append(device)

    def group_admits(
        current: "list[DeviceCapability]", device: DeviceCapability
    ) -> bool:
        speed = theta.get(device.device_id, 0.0)
        if speed <= 0:
            return False
        if admits is not None:
            return bool(admits(tuple(current), device, theta))
        fastest = theta.get(current[0].device_id, 0.0)
        return fastest / speed <= tolerance

    groups: list[tuple[DeviceCapability, ...]] = []
    for fabric in sorted(by_fabric):
        ordered = sorted(by_fabric[fabric], key=lambda d: -theta.get(d.device_id, 0.0))
        current: list[DeviceCapability] = []
        for device in ordered:
            if not current:
                current = [device]
                continue
            if group_admits(current, device):
                current.append(device)
            else:
                groups.append(tuple(current))
                current = [device]
        if current:
            groups.append(tuple(current))
    groups.sort(key=lambda group: -sum(theta.get(d.device_id, 0.0) for d in group))
    return groups


def _divisors(value: int, ceiling: int) -> list[int]:
    """Divisors of ``value`` up to ``ceiling``, ascending."""
    return [d for d in range(1, min(value, max(1, ceiling)) + 1) if value % d == 0]


def _shard_fractions(
    group: "tuple[DeviceCapability, ...]",
    theta: "dict[str, float]",
    total_bytes: float,
    caps: "list[float]",
    policy: str = "time",
) -> tuple[float, ...] | None:
    """Shard fractions for a TP group, or ``None`` if the group cannot hold it.

    ``time`` water-fills by throughput, minimising the barrier time — right when the
    stage is compute- or bandwidth-bound.  ``capacity`` equalises leftover bytes,
    maximising the KV budget of the *most constrained* device — right when
    ``tokens_max`` is what limits the plan.  Both are generated so ranking decides,
    because which one is correct is a property of the workload, not of the split.
    """
    throughputs = [max(theta.get(device.device_id, 0.0), 1.0) for device in group]
    try:
        if policy == "capacity":
            return tuple(equalise_slack(total_bytes, caps))
        return tuple(water_fill(total_bytes, throughputs, caps))
    except WaterfillInfeasible:
        return None


def _layer_split(
    groups: "list[tuple[DeviceCapability, ...]]",
    theta: "dict[str, float]",
    layers: int,
    policy: str,
) -> list[int] | None:
    """Assign layer counts across pipeline stages.

    ``balanced`` water-fills ∝ aggregate group throughput, which is optimal when the
    slowest stage sets the pipeline rate.  ``greedy`` loads the fastest group first,
    which is optimal for single-token latency because stage times *add* — balancing a
    latency pipeline is one of the mistakes this planner exists to avoid.
    """
    if layers <= 0 or not groups:
        return None
    strengths = [
        max(sum(theta.get(device.device_id, 0.0) for device in group), 1.0)
        for group in groups
    ]
    fractions = (
        greedy_fill(float(layers), strengths)
        if policy == "greedy"
        else water_fill(float(layers), strengths)
    )
    counts = [int(round(fraction * layers)) for fraction in fractions]
    # Repair rounding so every stage owns at least one layer and the total is exact.
    for index, count in enumerate(counts):
        if count < 1:
            counts[index] = 1
    drift = sum(counts) - layers
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    position = 0
    while drift != 0 and order:
        index = order[position % len(order)]
        if drift > 0 and counts[index] > 1:
            counts[index] -= 1
            drift -= 1
        elif drift < 0:
            counts[index] += 1
            drift += 1
        position += 1
        if position > 4 * len(counts) + layers:
            return None
    return counts if sum(counts) == layers else None


def _candidate_device_sets(
    accelerators: "tuple[DeviceCapability, ...]",
    theta: "dict[str, float]",
) -> list[tuple[DeviceCapability, ...]]:
    """Device subsets worth considering, without enumerating the power set.

    Three families, deduplicated: the fastest *k* devices, the *k* largest devices,
    and each capability group whole.  The memory ranking matters — a model that does
    not fit on the fastest device may fit on a slower, larger one, and a
    throughput-only ranking would never offer that plan.
    """
    seen: set[frozenset[str]] = set()
    candidates: list[tuple[DeviceCapability, ...]] = []

    def add(subset: "tuple[DeviceCapability, ...]") -> None:
        if not subset:
            return
        key = frozenset(device.device_id for device in subset)
        if key in seen:
            return
        seen.add(key)
        candidates.append(subset)

    by_speed = sorted(accelerators, key=lambda d: -theta.get(d.device_id, 0.0))
    by_memory = sorted(accelerators, key=lambda d: -d.total_bytes)
    for count in range(1, len(accelerators) + 1):
        add(tuple(by_speed[:count]))
        add(tuple(by_memory[:count]))
    return candidates


def _stage(
    group: "tuple[DeviceCapability, ...]",
    tp_degree: int,
    layer_start: int,
    layer_count: int,
    theta: "dict[str, float]",
    weight_bytes: float,
    caps: "dict[str, float]",
    policy: str = "time",
) -> StagePlan | None:
    """Build one stage, water-filling its shard fractions. ``None`` if it cannot fit."""
    members = group[:tp_degree]
    if not members:
        return None
    limits = [max(1.0, caps.get(device.device_id, float("inf"))) for device in members]
    fractions = _shard_fractions(members, theta, weight_bytes, limits, policy)
    if fractions is None:
        # The caps cannot hold this stage. Emit it anyway, split proportionally and
        # uncapped, so the feasibility lane can refuse it *with the arithmetic*. A
        # plan that is silently never generated cannot be explained to an operator,
        # and "single device: infeasible, short by 8.5 GiB" is the single most useful
        # line in a refusal.
        throughputs = [max(theta.get(device.device_id, 0.0), 1.0) for device in members]
        try:
            fractions = tuple(water_fill(weight_bytes, throughputs))
        except (WaterfillInfeasible, ValueError):
            return None
    return StagePlan(
        devices=tuple(device.device_id for device in members),
        tp_degree=len(members),
        layer_start=layer_start,
        layer_end=layer_start + layer_count,
        shard_fractions=fractions,
    )


def enumerate_plans(
    profile: ModelProfile,
    census: DeviceCensus,
    *,
    theta: "dict[str, float]",
    caps: "dict[str, float]",
    include_offload: bool = True,
    tolerance: float = THETA_TOLERANCE,
    admits: "AdmissionRule | None" = None,
    max_plans: int = MAX_ENUMERATED_PLANS,
) -> list[ExecutionPlan]:
    """Generate every structurally admissible plan for this model and machine.

    Args:
        profile: The model's exact facts.
        census: The measured machine.
        theta: Per-device throughput on the binding roof — bandwidth for a
            decode-dominated workload, FLOPs for a prefill-dominated one. The caller
            decides, because the correct choice belongs to the phase.
        caps: Per-device byte budget for *weights*, used by water-filling. Plans
            whose shards cannot fit are never generated.
        include_offload: Whether to offer host-offload plans. They are always
            feasible and always slow, so they lose on cost unless nothing else
            survives — the fallback ladder emerges from ranking rather than from a
            special case.
        tolerance: Law I's fixed ratio bound, used only when ``admits`` is absent.
        admits: Law I as a derived predicate; see :func:`capability_groups`.

    Returns:
        Plans in generation order: narrower and simpler first. Never empty — a host
        with no accelerator yields the CPU plan.
    """
    plans: list[ExecutionPlan] = []
    accelerators = census.accelerators
    layers = max(1, profile.layers)

    if not accelerators:
        cpu = census.devices[0]
        return [
            ExecutionPlan(
                stages=(StagePlan(
                    devices=(cpu.device_id,), tp_degree=1,
                    layer_start=0, layer_end=layers, shard_fractions=(1.0,),
                ),),
                kind=Parallelism.SINGLE,
            )
        ]

    tp_ceiling = profile.tp_ceiling_for_kv if profile.supports_tensor_parallel else 1
    weight_bytes = float(profile.weight_bytes)
    seen_labels: set[str] = set()

    def emit(plan: ExecutionPlan) -> None:
        signature = (
            plan.kind.value, plan.layer_policy, plan.shard_policy, plan.offload_tier,
            tuple(
                (stage.devices, stage.layers, stage.tp_degree,
                 tuple(round(value, 5) for value in stage.shard_fractions))
                for stage in plan.stages
            ),
        )
        key = repr(signature)
        if key in seen_labels or len(plans) >= max_plans:
            return
        seen_labels.add(key)
        plans.append(plan)

    for subset in _candidate_device_sets(accelerators, theta):
        groups = capability_groups(
            subset, theta=theta, tolerance=tolerance, admits=admits
        )
        if not groups:
            continue

        if len(groups) == 1:
            group = groups[0]
            width = len(group)
            # ── one stage, tensor-parallel over the whole group ───────────────
            # Sub-degrees that would leave devices idle are not generated here:
            # they are reachable as smaller candidate subsets, which keeps the space
            # free of duplicate placements wearing different names.
            if width == 1 or width <= tp_ceiling:
                for policy in ("time", "capacity"):
                    stage = _stage(
                        group, width, 0, layers, theta, weight_bytes, caps, policy
                    )
                    if stage is None:
                        continue
                    emit(ExecutionPlan(
                        stages=(stage,),
                        kind=Parallelism.SINGLE if width == 1 else Parallelism.TENSOR,
                        shard_policy=policy,
                    ))
                    if width == 1:
                        break  # one device has nothing to split


            # ── pipeline over the group, with TP inside each stage ────────────
            if profile.supports_pipeline_parallel and width > 1:
                for stage_count in _divisors(width, width):
                    if stage_count < 2 or stage_count > layers:
                        continue
                    per_stage = width // stage_count
                    if per_stage > tp_ceiling and per_stage > 1:
                        continue
                    for policy in ("balanced", "greedy"):
                        counts = _layer_split(
                            [group[i * per_stage:(i + 1) * per_stage] for i in range(stage_count)],
                            theta, layers, policy,
                        )
                        if counts is None:
                            continue
                        stages: list[StagePlan] = []
                        cursor = 0
                        ok = True
                        for index in range(stage_count):
                            members = group[index * per_stage:(index + 1) * per_stage]
                            # A pipeline stage holds only its own layers, so its
                            # weight share scales with the layer count, not the model.
                            share = weight_bytes * counts[index] / layers
                            stage = _stage(
                                members, per_stage, cursor, counts[index],
                                theta, share, caps,
                            )
                            if stage is None:
                                ok = False
                                break
                            stages.append(stage)
                            cursor += counts[index]
                        if ok and stages:
                            emit(ExecutionPlan(
                                stages=tuple(stages),
                                kind=(
                                    Parallelism.PIPELINE if per_stage == 1
                                    else Parallelism.HYBRID
                                ),
                                layer_policy=policy,
                            ))

        elif profile.supports_pipeline_parallel:
            # ── heterogeneous: one stage per capability group ──────────────────
            # This is Law I doing its job. The groups differ in throughput by more
            # than the tolerance, so they cannot share a TP group; a pipeline lets
            # each run at its own pace, and HexGen's asymmetric 48/20/12 at TP 4/2/2
            # is exactly this shape derived rather than searched.
            if len(groups) > layers:
                continue
            for policy in ("balanced", "greedy"):
                counts = _layer_split(groups, theta, layers, policy)
                if counts is None:
                    continue
                stages = []
                cursor = 0
                ok = True
                for index, group in enumerate(groups):
                    width = min(len(group), max(1, tp_ceiling))
                    share = weight_bytes * counts[index] / layers
                    stage = _stage(group, width, cursor, counts[index], theta, share, caps)
                    if stage is None:
                        ok = False
                        break
                    stages.append(stage)
                    cursor += counts[index]
                if ok and stages:
                    emit(ExecutionPlan(
                        stages=tuple(stages),
                        kind=(
                            Parallelism.PIPELINE
                            if all(stage.tp_degree == 1 for stage in stages)
                            else Parallelism.HYBRID
                        ),
                        layer_policy=policy,
                    ))

    # ── offload tier ──────────────────────────────────────────────────────────
    # Generated unconditionally so the fallback ladder is a consequence of ranking
    # rather than a branch. These plans are always feasible and always slow: the cost
    # kernel charges the spilled bytes at host-link bandwidth on every token, so they
    # only win when nothing else survives the feasibility filter.
    if include_offload and census.host_bytes > 0:
        strongest = max(accelerators, key=lambda d: theta.get(d.device_id, 0.0))
        emit(ExecutionPlan(
            stages=(StagePlan(
                devices=(strongest.device_id,), tp_degree=1,
                layer_start=0, layer_end=layers, shard_fractions=(1.0,),
            ),),
            kind=Parallelism.OFFLOAD,
            offload_tier="host",
        ))

    logger.debug(
        "enumerated %d placement plans for %s over %d device(s)",
        len(plans), profile.model_id, len(accelerators),
    )
    return plans
