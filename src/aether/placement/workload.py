"""The workload envelope and the planning intent.

A workload is a *range*, not a point, and the distinction is load-bearing:
feasibility is tested against the floor, performance is scored at the target, and
the ceiling sets admission control.  Planning against a single point is what makes
a runtime OOM the first time someone raises the batch size.

The intent is an explicit input rather than a bias baked into a scoring function.
The same model on the same hardware has different correct placements under a
latency target and a throughput target, and a planner that cannot be told which one
matters will be wrong for half its callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["Intent", "WorkloadEnvelope"]


class Intent(str, Enum):
    """What the caller is optimising for."""

    LATENCY = "latency"
    """Minimise time per output token. Favours tensor parallelism on a fast fabric."""

    THROUGHPUT = "throughput"
    """Maximise tokens per second in aggregate. Favours large KV budgets, pipelines
    and — when nothing else is feasible — host offload."""

    CAPACITY = "capacity"
    """Maximise the context and batch the plan can hold. Favours any split that
    divides the KV cache."""

    BALANCED = "balanced"
    """Minimise latency subject to meeting the target workload's capacity, then
    prefer the fewest devices. The default."""


@dataclass(frozen=True)
class WorkloadEnvelope:
    """The range of requests a plan must serve, and what to optimise within it.

    ``*_floor`` is what must work or the plan is infeasible.  ``*_target`` is what
    the plan is scored on.  ``*_ceiling`` is the largest request the plan will admit
    without a replan, and it is reported so an operator sees the boundary before
    hitting it.
    """

    batch_floor: int = 1
    batch_target: int = 1
    batch_ceiling: int = 0
    """Zero means "derive the ceiling from the plan's capacity"."""

    context_floor: int = 512
    context_target: int = 2048
    context_ceiling: int = 0

    generate_floor: int = 64
    generate_target: int = 256
    generate_ceiling: int = 0

    intent: Intent = Intent.BALANCED
    prefill_fraction: float = 0.0
    """Share of tokens processed in prefill rather than decode. Zero means derive it
    from the target context and generation lengths, which is right for a normal
    single-turn request; set it explicitly for a workload that is dominated by long
    prompts and short answers, or the reverse."""

    def __post_init__(self) -> None:
        if self.batch_floor < 1 or self.context_floor < 1 or self.generate_floor < 0:
            raise ValueError("workload floors must be positive")
        if self.batch_target < self.batch_floor:
            raise ValueError("batch_target must be >= batch_floor")
        if self.context_target < self.context_floor:
            raise ValueError("context_target must be >= context_floor")
        if not 0.0 <= self.prefill_fraction <= 1.0:
            raise ValueError("prefill_fraction must be in [0, 1]")

    # ── token requirements ────────────────────────────────────────────────────

    @property
    def floor_kv_tokens(self) -> int:
        """KV tokens the plan must hold, or it is infeasible."""
        return self.batch_floor * (self.context_floor + self.generate_floor)

    @property
    def target_kv_tokens(self) -> int:
        """KV tokens the plan is scored against."""
        return self.batch_target * (self.context_target + self.generate_target)

    @property
    def ceiling_kv_tokens(self) -> int:
        """KV tokens at the declared ceiling, or zero when it is plan-derived."""
        batch = self.batch_ceiling or 0
        context = self.context_ceiling or self.context_target
        generate = self.generate_ceiling or self.generate_target
        return batch * (context + generate) if batch else 0

    @property
    def effective_prefill_fraction(self) -> float:
        """Prefill's share of processed tokens, derived when not stated.

        A request of ``ctx`` prompt tokens and ``gen`` output tokens processes
        ``ctx`` tokens in one prefill pass and ``gen`` tokens one at a time, so
        prefill's token share is ``ctx / (ctx + gen)``. That share governs which
        roof the cost model should weight, not which phase takes longer.
        """
        if self.prefill_fraction > 0:
            return self.prefill_fraction
        total = self.context_target + self.generate_target
        return self.context_target / total if total else 0.0

    def with_intent(self, intent: Intent) -> "WorkloadEnvelope":
        return WorkloadEnvelope(
            batch_floor=self.batch_floor, batch_target=self.batch_target,
            batch_ceiling=self.batch_ceiling,
            context_floor=self.context_floor, context_target=self.context_target,
            context_ceiling=self.context_ceiling,
            generate_floor=self.generate_floor, generate_target=self.generate_target,
            generate_ceiling=self.generate_ceiling,
            intent=intent, prefill_fraction=self.prefill_fraction,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "floor": {
                "batch": self.batch_floor, "context": self.context_floor,
                "generate": self.generate_floor, "kv_tokens": self.floor_kv_tokens,
            },
            "target": {
                "batch": self.batch_target, "context": self.context_target,
                "generate": self.generate_target, "kv_tokens": self.target_kv_tokens,
            },
            "prefill_fraction": round(self.effective_prefill_fraction, 4),
        }
