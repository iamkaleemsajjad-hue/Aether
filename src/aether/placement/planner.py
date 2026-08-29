"""The planner: filter by feasibility, rank by intent, explain the choice.

Two passes over one generated plan space, with opposite treatments of error.

The **feasibility** pass may only ever *remove* plans, and it errs high on memory:
each device's KV budget is the residual after a one-sided upper bound on the
transient peak, so a plan survives only if it holds the workload floor.

The **performance** pass may only ever *order* the survivors, and it errs toward the
mean: the three-roof kernel plus α–β communication gives an expected step time.

Selection then applies the rule that makes "use the minimum hardware necessary"
rigorous rather than a preference: **a wider plan is chosen only when its predicted
advantage exceeds the planner's own prediction error.** Two plans within noise of
each other resolve to the narrower one, deterministically, every time.

Nothing here is reactive. There is no OOM-and-retry path, because a planner that
learns placement from allocation failures converges slowly, wastes a model load per
attempt, and returns different answers on identical hardware depending on arrival
order. The feedback loop exists to tighten the *next* prediction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from aether.placement.census import DeviceCapability, DeviceCensus, take_census
from aether.placement.cost import (
    RoofBreakdown,
    StageCost,
    decode_cost,
    link_model_for,
    pp_comm_seconds,
    prefill_cost,
    tp_comm_seconds,
    tp_prefill_comm_ratio,
)
from aether.placement.ledger import CalibrationLedger, LedgerEntry
from aether.placement.memory import (
    KAPPA_DEFAULT,
    Z_DEFAULT,
    MemoryBudget,
    evaluate_budget,
    safe_capacity,
)
from aether.placement.model_profile import ModelProfile
from aether.placement.plans import (
    THETA_TOLERANCE,
    ExecutionPlan,
    Parallelism,
    StagePlan,
    enumerate_plans,
)
from aether.placement.workload import Intent, WorkloadEnvelope
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "PlacementInfeasible",
    "PlanEvaluation",
    "PlannerDecision",
    "ExecutionPlanner",
]

_UNCALIBRATED_LATENCY_SIGMA = 0.25
"""Relative latency uncertainty before any measurement exists."""

_PER_DEVICE_LATENCY_SIGMA = 0.05
"""Extra relative uncertainty per additional device: each one adds an interaction
the cost model does not measure directly."""

_HOST_LINK_BPS = 6e9
"""Conservative host-to-device streaming bandwidth for offload plans."""


class PlacementInfeasible(RuntimeError):
    """No plan can run this workload on this machine.

    Carries the residual arithmetic rather than a bare message, so the caller can
    report *what* was needed, *where* it fell short, and which concrete changes would
    make it feasible. A refusal that cannot be acted on is only marginally better than
    an OOM.
    """

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


@dataclass
class PlanEvaluation:
    """One plan, fully judged: admissibility, capacity, and predicted time."""

    plan: ExecutionPlan
    budgets: tuple[MemoryBudget, ...]
    stage_costs: tuple[StageCost, ...]
    feasible: bool
    reason: str
    tokens_max: int
    binding_device: str
    decode_seconds: float
    prefill_seconds: float
    blended_token_seconds: float
    pipelined_token_seconds: float
    latency_sigma: float
    comm_ratio: float
    offload_bytes: int = 0
    flags: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return self.plan.label

    @property
    def binding_roof(self) -> str:
        """Which ceiling limits the slowest stage - the planner's diagnosis."""
        if not self.stage_costs:
            return "unknown"
        slowest = max(self.stage_costs, key=lambda stage: stage.seconds)
        return slowest.roofs.binding

    @property
    def predicted_peak_bytes(self) -> int:
        return sum(budget.predicted_peak_bytes for budget in self.budgets)

    def headroom(self, workload: WorkloadEnvelope) -> float:
        """Capacity relative to the target workload. Below 1.0 means it will not hold."""
        required = workload.target_kv_tokens
        if required <= 0:
            return float("inf")
        return self.tokens_max / required

    def batch_ceiling(self, workload: WorkloadEnvelope) -> int:
        """Largest batch this plan admits at the target context, before a replan."""
        per_request = workload.context_target + workload.generate_target
        if per_request <= 0:
            return 0
        return self.tokens_max // per_request

    def objective(self, workload: WorkloadEnvelope) -> float:
        """Lower is better, for every intent.

        Latency and balanced use a *blended* per-token time: prefill amortised over
        its own tokens, decode per token, weighted by the workload's actual token mix.
        A pure decode metric would misrank a prompt-heavy workload, and a pure prefill
        metric would misrank a chat one.
        """
        intent = workload.intent
        if intent is Intent.CAPACITY:
            return -float(self.tokens_max)
        if intent is Intent.THROUGHPUT:
            batch = max(1, self.batch_ceiling(workload) or workload.batch_target)
            # Steady-state rate, not single-request latency: pipeline stages overlap
            # across concurrent requests, so a pipeline's throughput is set by its
            # slowest stage rather than by the sum of them. Using the latency figure
            # here would make the planner reject pipelines for exactly the workload
            # they are good at.
            return self.pipelined_token_seconds / batch
        return self.blended_token_seconds

    def sigma_objective(self, workload: WorkloadEnvelope) -> float:
        """Absolute uncertainty on :meth:`objective`, in the objective's own units."""
        value = abs(self.objective(workload))
        if workload.intent is Intent.CAPACITY:
            # Capacity error is driven by the memory residual, not by latency.
            return value * 0.05
        return value * self.latency_sigma

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "feasible": self.feasible,
            "reason": self.reason,
            "tokens_max": self.tokens_max,
            "binding_device": self.binding_device,
            "binding_roof": self.binding_roof,
            "decode_ms": round(self.decode_seconds * 1e3, 3),
            "prefill_ms": round(self.prefill_seconds * 1e3, 3),
            "blended_token_ms": round(self.blended_token_seconds * 1e3, 3),
            "pipelined_token_ms": round(self.pipelined_token_seconds * 1e3, 3),
            "latency_sigma": round(self.latency_sigma, 4),
            "tp_prefill_comm_ratio": round(self.comm_ratio, 4),
            "offload_gib": round(self.offload_bytes / 1024 ** 3, 3),
            "flags": list(self.flags),
            "budgets": [budget.to_dict() for budget in self.budgets],
            "stages": [stage.to_dict() for stage in self.stage_costs],
        }


@dataclass
class PlannerDecision:
    """The planner's output: a choice, the evidence, and the reasoning."""

    selected: PlanEvaluation
    candidates: tuple[PlanEvaluation, ...]
    profile: ModelProfile
    census: DeviceCensus
    workload: WorkloadEnvelope
    reason: str
    flags: tuple[str, ...] = ()
    ladder: tuple[str, ...] = ()
    planning_seconds: float = 0.0
    calibrated: bool = False

    @property
    def plan(self) -> ExecutionPlan:
        return self.selected.plan

    @property
    def feasible_candidates(self) -> tuple[PlanEvaluation, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.feasible)

    @property
    def runner_up(self) -> PlanEvaluation | None:
        feasible = [c for c in self.feasible_candidates if c is not self.selected]
        if not feasible:
            return None
        return min(feasible, key=lambda c: c.objective(self.workload))

    def render(self) -> str:
        """The human-readable decision record."""
        from aether.placement.record import render_record

        return render_record(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "reason": self.reason,
            "flags": list(self.flags),
            "ladder": list(self.ladder),
            "calibrated": self.calibrated,
            "planning_ms": round(self.planning_seconds * 1e3, 3),
            "model": self.profile.to_dict(),
            "workload": self.workload.to_dict(),
            "hardware": self.census.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class ExecutionPlanner:
    """Decides where a model runs, and can explain why.

    Args:
        profile: Exact model facts. Build with
            :func:`~aether.placement.model_profile.profile_from_engine` when an engine
            is loaded, or ``profile_from_architecture`` at compile time.
        census: The machine. Measured on construction when omitted.
        ledger: Calibration store. A default, on-disk one is used when omitted.
        kappa: Device-memory ceiling fraction — the only percentage in the design.
        z: One-sided quantile for the transient-peak margin.
        tolerance: Law I's throughput ratio bound for a tensor-parallel group.
    """

    def __init__(
        self,
        profile: ModelProfile,
        census: DeviceCensus | None = None,
        ledger: CalibrationLedger | None = None,
        *,
        kappa: float = KAPPA_DEFAULT,
        z: float = Z_DEFAULT,
        tolerance: float = THETA_TOLERANCE,
        probe_bandwidth: bool = True,
    ) -> None:
        self.profile = profile
        self.ledger = ledger if ledger is not None else CalibrationLedger()
        self.census = census if census is not None else take_census(
            ledger=self.ledger, probe_bandwidth=probe_bandwidth
        )
        self.kappa = kappa
        self.z = z
        self.tolerance = tolerance
        self._entries: dict[str, LedgerEntry] = {}

    # ── inputs ────────────────────────────────────────────────────────────────

    def entry_for(self, device: DeviceCapability) -> LedgerEntry:
        """Calibration for one device, cached for the life of this planner."""
        if device.device_id not in self._entries:
            self._entries[device.device_id] = self.ledger.get(
                device.signature, self.census.backend_build
            )
        return self._entries[device.device_id]

    def theta(self, workload: WorkloadEnvelope) -> dict[str, float]:
        """Relative capability per device, on the roof the workload actually hits.

        Bandwidth and FLOPs cannot be blended in their own units, so each is
        normalised against the fleet's best before mixing.  Water-filling needs only
        *relative* throughput, which makes the normalised blend dimensionally sound
        and phase-correct: a decode-dominated workload weights bandwidth, a
        prefill-dominated one weights arithmetic.
        """
        fraction = workload.effective_prefill_fraction
        bandwidths = {d.device_id: max(d.effective_bandwidth_bps, 1.0) for d in self.census.devices}
        flops = {d.device_id: max(d.effective_flops, 1.0) for d in self.census.devices}
        best_bandwidth = max(bandwidths.values())
        best_flops = max(flops.values())
        return {
            device_id: (
                (1.0 - fraction) * bandwidths[device_id] / best_bandwidth
                + fraction * flops[device_id] / best_flops
            )
            for device_id in bandwidths
        }

    def weight_caps(self) -> dict[str, float]:
        """Per-device byte budget for weights, used by water-filling.

        Deliberately generous — the whole safe capacity — because enumeration only
        needs *plausible* fractions.  Whether a plan really fits is the feasibility
        lane's job, and mixing the two would make the generator quietly responsible
        for a decision it cannot explain.
        """
        return {
            device.device_id: float(safe_capacity(device, self.entry_for(device), kappa=self.kappa))
            for device in self.census.devices
        }

    # ── per-device footprint ──────────────────────────────────────────────────

    def _stage_weight_bytes(self, plan: ExecutionPlan, index: int) -> int:
        """Weight bytes a stage holds, embeddings and LM head placed exactly.

        The embedding table lives on the first stage and the LM head on the last, so
        a pipeline's stages are *not* equal even when their layer counts are — and on
        a large-vocabulary model that difference is gigabytes.
        """
        stage = plan.stages[index]
        total = self.profile.per_layer_bytes * stage.layers
        if index == 0:
            total += self.profile.embedding_bytes
        if index == len(plan.stages) - 1:
            total += self.profile.lm_head_bytes
        return int(total)

    def _kv_bytes_per_token(self, stage: StagePlan) -> int:
        """This device's KV bytes per token, after the layer and head splits.

        Tensor parallelism divides the cache by KV heads, capped at the head count:
        past that the cache replicates and the capacity benefit disappears, which is
        why the plan generator refuses TP degrees above it.
        """
        heads = max(1, self.profile.num_kv_heads)
        shards = max(1, min(stage.tp_degree, heads))
        local_heads = heads / shards
        return int(
            2 * stage.layers * local_heads * self.profile.head_dim * self.profile.kv_dtype_bytes
        )

    def _persistent_bytes(self, plan: ExecutionPlan, stage: StagePlan, workload: WorkloadEnvelope) -> int:
        """Replicated per-device allocations that live for the whole run."""
        total = self.profile.persistent_bytes
        if stage.tp_degree > 1 or plan.pipeline_stages > 1:
            # Collective staging: one send and one receive buffer at the widest
            # payload the plan will move, which is a prefill activation slab.
            payload = int(
                workload.batch_target * workload.context_target
                * self.profile.hidden_size * self.profile.weight_dtype_bytes
            )
            total += 2 * payload
        return int(total)

    def _host_ops(self, plan: ExecutionPlan, index: int) -> int:
        """Host-dispatched operations for one stage, per decode step.

        Tensor parallelism multiplies this: in a single process the host issues each
        sharded operation once per device and then one collective per all-reduce
        point, all on one serial host thread.  That is the term that makes TP *slower*
        on a dispatch-bound model, and leaving it out is why a two-roof planner
        recommends sharding a 0.6 B model across two T4s.
        """
        stage = plan.stages[index]
        ops = self.profile.ops_per_layer * stage.layers * stage.tp_degree
        if stage.tp_degree > 1:
            ops += 2 * stage.layers
        if index == 0:
            ops += 2
        if index == len(plan.stages) - 1:
            ops += 2
        return int(ops)

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, plan: ExecutionPlan, workload: WorkloadEnvelope) -> PlanEvaluation:
        """Judge one plan: admissibility, capacity, and predicted time."""
        budgets: list[MemoryBudget] = []
        stage_costs: list[StageCost] = []
        flags: list[str] = []
        offload_bytes = 0
        decode_stage_seconds: list[float] = []
        prefill_stage_seconds: list[float] = []

        # Peak memory must cover the widest step the plan will run, which is a
        # prefill at the target context — not a decode step, which is far smaller.
        peak_batch = workload.batch_target
        peak_tokens = workload.context_target

        for index, stage in enumerate(plan.stages):
            stage_weight = self._stage_weight_bytes(plan, index)
            persistent = self._persistent_bytes(plan, stage, workload)
            kv_per_token = self._kv_bytes_per_token(stage)
            per_device_roofs: list[RoofBreakdown] = []
            prefill_roofs: list[RoofBreakdown] = []
            host_ops = self._host_ops(plan, index)

            for device_id in stage.devices:
                device = self.census.by_id(device_id)
                entry = self.entry_for(device)
                resident = int(stage_weight * stage.fraction_for(device_id))
                activations = self.profile.activation_bytes(
                    peak_batch, peak_tokens, peak_tokens, tp_degree=stage.tp_degree
                )

                spilled = 0
                if plan.offload_tier:
                    # Offload trades resident weights for exactly the floor's worth of
                    # cache: keep enough KV to run, stream the rest of the weights.
                    # The safety margin has to come out of the room too, or the bottom
                    # rung of the fallback ladder is itself infeasible — which would
                    # leave the planner with nothing to fall back to.
                    capacity = safe_capacity(device, entry, kappa=self.kappa)
                    transient = int(activations * max(1.0, entry.fragmentation))
                    reserve = workload.floor_kv_tokens * kv_per_token
                    room = (
                        capacity - persistent - transient
                        - entry.margin_bytes(transient, self.z) - reserve
                    )
                    keep = max(0, min(resident, room))
                    spilled = max(0, resident - keep)
                    offload_bytes += spilled
                    resident = keep

                budget = evaluate_budget(
                    device, entry,
                    weight_bytes=resident,
                    persistent_bytes=persistent,
                    activation_bytes=activations,
                    kv_bytes_per_token=kv_per_token,
                    kappa=self.kappa, z=self.z,
                )
                budgets.append(budget)

                # Decode reads the resident weights plus the live cache; an offloaded
                # remainder is charged at host-link bandwidth, which is what makes an
                # offload plan honestly slow rather than quietly attractive.
                live_kv = workload.target_kv_tokens * kv_per_token
                decode_roof = decode_cost(
                    self.profile, device, entry,
                    weight_bytes=resident, kv_bytes=live_kv,
                    ops=host_ops, batch=workload.batch_target,
                    context=workload.context_target + workload.generate_target,
                    tp_degree=stage.tp_degree,
                )
                if spilled:
                    decode_roof = RoofBreakdown(
                        compute_s=decode_roof.compute_s,
                        bandwidth_s=decode_roof.bandwidth_s + spilled / _HOST_LINK_BPS,
                        dispatch_s=decode_roof.dispatch_s,
                    )
                per_device_roofs.append(decode_roof)
                prefill_roofs.append(prefill_cost(
                    self.profile, device, entry,
                    weight_bytes=resident, ops=host_ops,
                    batch=workload.batch_target, sequence=workload.context_target,
                    tp_degree=stage.tp_degree, layers=stage.layers,
                ))

            # Compute and bandwidth run in parallel across the devices in a stage, so
            # the stage pays the max. Host dispatch does not: it is one serial thread,
            # so the stage pays the total, which _host_ops already accumulated.
            stage_decode = RoofBreakdown(
                compute_s=max(roof.compute_s for roof in per_device_roofs),
                bandwidth_s=max(roof.bandwidth_s for roof in per_device_roofs),
                dispatch_s=per_device_roofs[0].dispatch_s,
            )
            stage_prefill = RoofBreakdown(
                compute_s=max(roof.compute_s for roof in prefill_roofs),
                bandwidth_s=max(roof.bandwidth_s for roof in prefill_roofs),
                dispatch_s=prefill_roofs[0].dispatch_s,
            )
            lead_entry = self.entry_for(self.census.by_id(stage.devices[0]))
            decode_comm = tp_comm_seconds(
                self.profile, self.census, stage.devices,
                layers=stage.layers, batch=workload.batch_target, step_tokens=1,
                dispatch_seconds=lead_entry.dispatch_seconds,
            )
            # Prefill moves the whole sequence per collective, not one token — which
            # is why a slow fabric kills TP in prefill long before it kills decode.
            prefill_comm = tp_comm_seconds(
                self.profile, self.census, stage.devices,
                layers=stage.layers, batch=workload.batch_target,
                step_tokens=workload.context_target,
                dispatch_seconds=lead_entry.dispatch_seconds,
            )
            stage_costs.append(StageCost(
                device_ids=stage.devices, roofs=stage_decode,
                comm_s=decode_comm, layers=stage.layers,
            ))
            decode_stage_seconds.append(stage_decode.seconds + decode_comm)
            prefill_stage_seconds.append(stage_prefill.seconds + prefill_comm)

        # Pipeline stages are sequential for a single token, so their times add. That
        # is also why greedy layer assignment beats balanced under a latency intent.
        boundary_decode = pp_comm_seconds(
            self.profile, self.census, plan.boundaries,
            batch=workload.batch_target, step_tokens=1,
        )
        boundary_prefill = pp_comm_seconds(
            self.profile, self.census, plan.boundaries,
            batch=workload.batch_target, step_tokens=workload.context_target,
        )
        decode_seconds = sum(decode_stage_seconds) + boundary_decode
        prefill_seconds = sum(prefill_stage_seconds) + boundary_prefill
        # Steady-state pipeline rate: with requests in flight the stages run
        # concurrently, so the rate is the slowest stage plus one boundary hop. This
        # is the number a throughput objective must use; the sums above are the
        # single-request latency, which is what a latency objective must use.
        rate_decode = max(decode_stage_seconds) + (
            boundary_decode / max(1, len(plan.boundaries)) if plan.boundaries else 0.0
        )
        rate_prefill = max(prefill_stage_seconds) + (
            boundary_prefill / max(1, len(plan.boundaries)) if plan.boundaries else 0.0
        )

        # ── capacity and admissibility ─────────────────────────────────────────
        tokens_max = min(budget.tokens_max for budget in budgets)
        binding = min(budgets, key=lambda b: b.tokens_max).device_id
        infeasible = [budget for budget in budgets if not budget.feasible]
        floor = workload.floor_kv_tokens

        if infeasible:
            worst = min(infeasible, key=lambda b: b.kv_budget_bytes)
            reason = (
                f"{worst.device_id} cannot hold this plan: "
                f"{worst.static_bytes / 1024 ** 3:.2f} GiB static + "
                f"{worst.transient_bytes / 1024 ** 3:.2f} GiB transient + "
                f"{worst.transient_margin_bytes / 1024 ** 3:.2f} GiB margin exceeds "
                f"{worst.safe_capacity_bytes / 1024 ** 3:.2f} GiB safe capacity by "
                f"{-worst.kv_budget_bytes / 1024 ** 3:.2f} GiB"
            )
            feasible = False
        elif tokens_max < floor:
            shortfall = min(budgets, key=lambda b: b.tokens_max)
            reason = (
                f"holds {tokens_max} KV tokens but the workload floor needs {floor} "
                f"({shortfall.device_id} binds; short by "
                f"{shortfall.shortfall_bytes(floor) / 1024 ** 3:.2f} GiB)"
            )
            feasible = False
        else:
            feasible = True
            reason = f"holds {tokens_max} KV tokens against a floor of {floor}"

        # ── prediction uncertainty ─────────────────────────────────────────────
        # Measured where evidence exists, structural where it does not. Each extra
        # device adds an interaction the cost model does not observe directly, so the
        # σ-gated tie-break naturally demands more evidence from wider plans.
        entries = [self.entry_for(self.census.by_id(d)) for d in plan.device_ids]
        measured = [e.latency_sigma for e in entries if e.latency_sigma > 0]
        base = (
            sum(measured) / len(measured) if measured else _UNCALIBRATED_LATENCY_SIGMA
        )
        latency_sigma = base + _PER_DEVICE_LATENCY_SIGMA * max(0, plan.num_devices - 1)

        # ── blended per-token time ─────────────────────────────────────────────
        fraction = workload.effective_prefill_fraction
        per_prefill_token = prefill_seconds / max(1, workload.context_target)
        blended = (1.0 - fraction) * decode_seconds + fraction * per_prefill_token
        pipelined = (
            (1.0 - fraction) * rate_decode
            + fraction * rate_prefill / max(1, workload.context_target)
        )

        # ── diagnostics worth surfacing ────────────────────────────────────────
        lead = self.census.by_id(plan.device_ids[0])
        widest = max(plan.stages, key=lambda s: s.tp_degree)
        link = link_model_for(self.census, widest.devices)
        comm_ratio = tp_prefill_comm_ratio(
            self.profile, lead, getattr(link, "bandwidth_bps", 0.0), widest.tp_degree
        )
        slowest = max(stage_costs, key=lambda s: s.seconds) if stage_costs else None
        if slowest is not None and slowest.roofs.binding == "dispatch":
            ratio = slowest.roofs.headroom_ratio
            flags.append(
                f"dispatch-bound: host op cost exceeds the next roof by {ratio:.1f}x. "
                "CUDA-graph capture or operator fusion is the fix, not more devices."
            )
        if plan.max_tp_degree > 1 and comm_ratio > 0.5:
            flags.append(
                f"tensor-parallel prefill communication is {comm_ratio:.2f}x its compute "
                "on this fabric; a pipeline split moves ~3x fewer bytes on the weak link."
            )
        if plan.offload_tier and offload_bytes > 0:
            flags.append(
                f"{offload_bytes / 1024 ** 3:.2f} GiB of weights stream from "
                f"{plan.offload_tier} memory every token; latency is an order of "
                "magnitude worse than a resident plan."
            )
        if plan.max_tp_degree > self.profile.tp_ceiling_for_kv:
            flags.append(
                f"TP degree {plan.max_tp_degree} exceeds the {self.profile.num_kv_heads} "
                "KV heads, so the cache replicates instead of sharding."
            )
        if not all(budget.calibrated for budget in budgets):
            flags.append(
                "uncalibrated device: the transient margin uses a conservative prior "
                "and will tighten after the first successful run."
            )
        if feasible and tokens_max < workload.target_kv_tokens:
            # Feasible against the floor but short of the target: the plan will run,
            # and it will start rejecting requests before reaching the stated
            # workload. Saying so is the difference between a capacity number and a
            # capacity guarantee.
            flags.append(
                f"holds {tokens_max:,} KV tokens but the target workload needs "
                f"{workload.target_kv_tokens:,}; requests above batch "
                f"{tokens_max // max(1, workload.context_target + workload.generate_target)} "
                f"at this context will be queued or refused."
            )

        return PlanEvaluation(
            plan=plan,
            budgets=tuple(budgets),
            stage_costs=tuple(stage_costs),
            feasible=feasible,
            reason=reason,
            tokens_max=tokens_max,
            binding_device=binding,
            decode_seconds=decode_seconds,
            prefill_seconds=prefill_seconds,
            blended_token_seconds=blended,
            pipelined_token_seconds=pipelined,
            latency_sigma=latency_sigma,
            comm_ratio=comm_ratio,
            offload_bytes=offload_bytes,
            flags=tuple(flags),
        )

    # ── selection ─────────────────────────────────────────────────────────────

    def plan(self, workload: WorkloadEnvelope | None = None) -> PlannerDecision:
        """Choose a placement, or refuse with the arithmetic that explains why.

        Raises:
            PlacementInfeasible: When no generated plan holds the workload floor. The
                exception carries the per-device residuals and concrete remedies.
        """
        started = time.perf_counter()
        workload = workload or WorkloadEnvelope()
        theta = self.theta(workload)
        caps = self.weight_caps()
        candidates = enumerate_plans(
            self.profile, self.census, theta=theta, caps=caps, tolerance=self.tolerance
        )
        evaluations = [self.evaluate(plan, workload) for plan in candidates]
        feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
        ladder = self._ladder(evaluations)

        if not feasible:
            raise PlacementInfeasible(
                self._refusal_message(evaluations, workload),
                detail={
                    "model": self.profile.to_dict(),
                    "workload": workload.to_dict(),
                    "hardware": self.census.to_dict(),
                    "remedies": self._remedies(evaluations, workload),
                    "candidates": [evaluation.to_dict() for evaluation in evaluations],
                },
            )

        ranked = sorted(feasible, key=lambda evaluation: evaluation.objective(workload))
        pool = ranked
        if workload.intent is Intent.BALANCED:
            # Balanced means "fast enough, and it must actually hold the target".
            with_headroom = [
                evaluation for evaluation in ranked
                if evaluation.headroom(workload) >= 1.0
            ]
            if with_headroom:
                pool = with_headroom
        best = pool[0]

        # The σ gate. A wider plan is accepted only when its advantage clears the
        # planner's own error bar; everything inside that bar is a tie, and ties go to
        # the narrower plan. This is what makes "minimum hardware" a consequence of
        # the cost model rather than a preference.
        tolerance_band = max(best.sigma_objective(workload), 1e-12)
        contenders = [
            evaluation for evaluation in pool
            if evaluation.objective(workload) - best.objective(workload)
            <= max(tolerance_band, evaluation.sigma_objective(workload))
        ]
        selected = min(
            contenders,
            key=lambda evaluation: (
                evaluation.plan.num_devices,
                evaluation.plan.pipeline_stages,
                1 if evaluation.plan.offload_tier else 0,
                # Among plans that are equally fast and equally wide, take the one
                # with more room. Capacity is free insurance: it costs nothing here
                # and it is what postpones the next replan.
                -evaluation.tokens_max,
                evaluation.objective(workload),
            ),
        )

        decision = PlannerDecision(
            selected=selected,
            candidates=tuple(evaluations),
            profile=self.profile,
            census=self.census,
            workload=workload,
            reason=self._reason(selected, best, pool, workload),
            flags=selected.flags,
            ladder=ladder,
            planning_seconds=time.perf_counter() - started,
            calibrated=all(budget.calibrated for budget in selected.budgets),
        )
        logger.info(
            "placement: %s selected for %s (%d feasible of %d plans, %.2f ms)",
            selected.label, self.profile.model_id,
            len(feasible), len(evaluations), decision.planning_seconds * 1e3,
        )
        return decision

    def _reason(
        self,
        selected: PlanEvaluation,
        best: PlanEvaluation,
        pool: "list[PlanEvaluation]",
        workload: WorkloadEnvelope,
    ) -> str:
        """One paragraph explaining the choice, in the model's own terms."""
        parts: list[str] = []
        infeasible_single = [
            evaluation for evaluation in pool if evaluation.plan.num_devices == 1
        ]
        if selected.plan.num_devices == 1:
            parts.append(
                f"A single device holds the workload: {selected.tokens_max} KV tokens "
                f"against a floor of {workload.floor_kv_tokens}"
            )
        else:
            parts.append(
                f"{selected.plan.num_devices} devices are used because "
                + (
                    "no single-device plan is feasible"
                    if not infeasible_single
                    else "the predicted gain clears the planner's error bar"
                )
            )
        if selected is not best:
            delta = (best.objective(workload) - selected.objective(workload))
            parts.append(
                f"{best.label} scored better by {abs(delta) * 1e3:.2f} ms/token but that "
                f"is inside the +/-{best.sigma_objective(workload) * 1e3:.2f} ms error bar, "
                f"so the narrower plan wins the tie"
            )
        else:
            # The informative runner-up is a *structurally different* plan, not
            # another spelling of the same placement — an offload plan that spilled
            # nothing is the selected plan wearing a different label.
            runners = [
                candidate for candidate in pool
                if candidate is not selected
                and (
                    candidate.plan.num_devices != selected.plan.num_devices
                    or candidate.plan.kind is not selected.plan.kind
                )
            ]
            if runners:
                second = min(runners, key=lambda e: e.objective(workload))
                delta = second.objective(workload) - selected.objective(workload)
                bar = selected.sigma_objective(workload)
                verdict = "outside" if abs(delta) > bar else "inside"
                parts.append(
                    f"it beats {second.label} by {abs(delta) * 1e3:.2f} ms/token, "
                    f"{verdict} the +/-{bar * 1e3:.2f} ms error bar"
                )
        parts.append(
            f"the binding roof is {selected.binding_roof} on {selected.binding_device}"
        )
        return "; ".join(parts) + "."

    @staticmethod
    def _ladder(evaluations: "list[PlanEvaluation]") -> tuple[str, ...]:
        """What each rung of the fallback ladder produced, in order.

        Printed whether or not a fallback was needed, because the ladder is only
        trustworthy if it is visible when it *did not* fire.
        """
        rungs = [
            ("single device", lambda e: e.plan.num_devices == 1 and not e.plan.offload_tier),
            ("tensor parallel", lambda e: e.plan.kind is Parallelism.TENSOR),
            ("pipeline parallel", lambda e: e.plan.kind in (Parallelism.PIPELINE, Parallelism.HYBRID)),
            ("host offload", lambda e: bool(e.plan.offload_tier)),
        ]
        lines: list[str] = []
        for name, predicate in rungs:
            matching = [e for e in evaluations if predicate(e)]
            if not matching:
                lines.append(f"{name}: not generated")
                continue
            feasible = [e for e in matching if e.feasible]
            if feasible:
                best = min(feasible, key=lambda e: e.blended_token_seconds)
                lines.append(
                    f"{name}: {len(feasible)}/{len(matching)} feasible, "
                    f"best {best.label} at {best.blended_token_seconds * 1e3:.2f} ms/token"
                )
            else:
                worst = min(matching, key=lambda e: e.tokens_max)
                lines.append(f"{name}: infeasible - {worst.reason}")
        return tuple(lines)

    def _refusal_message(
        self, evaluations: "list[PlanEvaluation]", workload: WorkloadEnvelope
    ) -> str:
        best = max(evaluations, key=lambda e: e.tokens_max) if evaluations else None
        aggregate = sum(device.total_bytes for device in self.census.accelerators)
        message = (
            f"no feasible placement for {self.profile.model_id} "
            f"({self.profile.weight_bytes / 1024 ** 3:.2f} GiB of weights, "
            f"{self.profile.layers} layers) at batch {workload.batch_floor} x "
            f"{workload.context_floor + workload.generate_floor} tokens on "
            f"{len(self.census.accelerators)} accelerator(s) totalling "
            f"{aggregate / 1024 ** 3:.1f} GiB"
        )
        if best is not None:
            message += f". Closest plan: {best.label} - {best.reason}"
        return message

    def _remedies(
        self, evaluations: "list[PlanEvaluation]", workload: WorkloadEnvelope
    ) -> list[str]:
        """Concrete, ordered changes that would make this workload feasible.

        A refusal that cannot be acted on is barely better than an OOM, so the
        planner computes what would actually have to change instead of suggesting the
        caller try something.
        """
        remedies: list[str] = []
        best = max(evaluations, key=lambda e: e.tokens_max) if evaluations else None
        if best is None:
            return ["no plans were generated: check that a device is visible"]

        needed = workload.floor_kv_tokens
        if best.tokens_max > 0 and best.tokens_max < needed:
            ratio = needed / best.tokens_max
            remedies.append(
                f"reduce the workload floor by {ratio:.1f}x - for example batch "
                f"{max(1, int(workload.batch_floor / ratio))} at the same context, or "
                f"context {max(1, int(workload.context_floor / ratio))} at the same batch"
            )
        if self.profile.kv_dtype_bytes > 1:
            remedies.append(
                f"halve the KV cache with an fp8 cache dtype "
                f"(currently {self.profile.kv_dtype_bytes:.0f} bytes/element), which "
                f"roughly doubles tokens_max"
            )
        if self.profile.weight_dtype_bytes > 1:
            saved = self.profile.weight_bytes / 2 / 1024 ** 3
            remedies.append(
                f"quantise the weights: 8-bit residency frees about {saved:.1f} GiB "
                f"and usually also moves the model off the bandwidth roof"
            )
        shortfall = min(
            (b for e in evaluations for b in e.budgets),
            key=lambda b: b.kv_budget_bytes,
        )
        remedies.append(
            f"add {max(0, -shortfall.kv_budget_bytes + needed * shortfall.kv_bytes_per_token) / 1024 ** 3:.1f} GiB "
            f"of accelerator memory, or a device in the same fabric class as "
            f"{shortfall.device_id}"
        )
        if self.census.host_bytes > 0:
            remedies.append(
                "allow host offload explicitly, accepting an order-of-magnitude "
                "latency increase"
            )
        return remedies

    # ── feedback ──────────────────────────────────────────────────────────────

    def observe(
        self,
        decision: PlannerDecision,
        *,
        observed_peak_bytes: "dict[str, int]",
        observed_decode_seconds: float = 0.0,
        observed_r_fixed_bytes: "dict[str, int] | None" = None,
        observed_fragmentation: "dict[str, float] | None" = None,
    ) -> None:
        """Fold a real execution back into the ledger.

        This is the whole feedback path, and it is calibration rather than control:
        it never changes the plan that is running, only the accuracy of the next
        prediction. The success metric is that the residual σ shrinks over repeated
        runs on one host, which is directly observable.
        """
        for budget in decision.selected.budgets:
            observed = observed_peak_bytes.get(budget.device_id)
            if observed is None:
                continue
            device = self.census.by_id(budget.device_id)
            self.ledger.observe_execution(
                device.signature,
                self.census.backend_build,
                predicted_transient_bytes=budget.predicted_peak_bytes,
                observed_peak_bytes=int(observed),
                predicted_seconds=decision.selected.decode_seconds,
                observed_seconds=observed_decode_seconds,
                r_fixed_bytes=(observed_r_fixed_bytes or {}).get(budget.device_id),
                fragmentation=(observed_fragmentation or {}).get(budget.device_id),
            )
        self._entries.clear()

    def __repr__(self) -> str:
        return (
            f"ExecutionPlanner({self.profile.model_id!r}, "
            f"{len(self.census.devices)} devices, kappa={self.kappa}, z={self.z})"
        )