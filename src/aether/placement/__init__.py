"""Universal hardware-aware execution planning.

This package answers one question — *given this model, this machine and this
workload, where should execution happen?* — and it answers it in two
structurally separate passes, because admissibility and desirability need
opposite treatments of uncertainty:

``feasibility``
    Can the workload run at all without exhausting memory?  Computed as a
    **residual**: safe capacity minus everything that is not KV cache is the KV
    budget, and the budget expressed in tokens is the answer.  Errs *high* on
    memory, using a one-sided confidence bound whose width comes from recorded
    prediction error rather than a hand-set reserve.

``performance``
    Among the admissible plans, which is fastest for the stated intent?
    Computed from a **three-roof** cost model — compute, memory bandwidth and
    host dispatch — plus an α–β communication term.  Errs toward the mean.

A single weighted score cannot do both jobs, which is why there is no score.

Module map
----------
:mod:`~aether.placement.census`         measured device capability and fabric classes
:mod:`~aether.placement.ledger`         persisted calibration, keyed by device + backend
:mod:`~aether.placement.model_profile`  exact model facts, from the AEG or a live engine
:mod:`~aether.placement.workload`       the workload envelope and the planning intent
:mod:`~aether.placement.memory`         safe capacity, the residual, ``tokens_max``
:mod:`~aether.placement.cost`           the three-roof kernel and communication terms
:mod:`~aether.placement.waterfill`      capped water-filling and greedy fill
:mod:`~aether.placement.plans`          plan representation and law-pruned enumeration
:mod:`~aether.placement.planner`        the planner: filter, rank, select, explain
:mod:`~aether.placement.record`         the human-readable decision record

Design notes live in ``docs/architecture-execution-planner.html``.
"""

from __future__ import annotations

from aether.placement.census import (
    DeviceCapability,
    DeviceCensus,
    FabricLink,
    take_census,
)
from aether.placement.cost import RoofBreakdown, StageCost, decode_cost, prefill_cost
from aether.placement.ledger import CalibrationLedger, LedgerEntry
from aether.placement.memory import MemoryBudget, safe_capacity
from aether.placement.model_profile import ModelProfile
from aether.placement.planner import (
    ExecutionPlanner,
    PlacementInfeasible,
    PlanEvaluation,
    PlannerDecision,
)
from aether.placement.plans import ExecutionPlan, Parallelism, StagePlan, enumerate_plans
from aether.placement.record import render_record
from aether.placement.waterfill import greedy_fill, water_fill
from aether.placement.workload import Intent, WorkloadEnvelope

__all__ = [
    "CalibrationLedger",
    "DeviceCapability",
    "DeviceCensus",
    "ExecutionPlan",
    "ExecutionPlanner",
    "FabricLink",
    "Intent",
    "LedgerEntry",
    "MemoryBudget",
    "ModelProfile",
    "Parallelism",
    "PlacementInfeasible",
    "PlanEvaluation",
    "PlannerDecision",
    "RoofBreakdown",
    "StageCost",
    "StagePlan",
    "WorkloadEnvelope",
    "decode_cost",
    "enumerate_plans",
    "greedy_fill",
    "prefill_cost",
    "render_record",
    "safe_capacity",
    "take_census",
    "water_fill",
]
