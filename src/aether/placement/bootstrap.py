"""The cold-start bootstrap: turn the first execution into the calibration event.

The feasibility lane's margin is ``z·σ_T``, and σ_T comes from recorded prediction
error.  On a fresh install there is no record, so the planner falls back to a
documented prior — and a prior is exactly what the design exists to avoid.  The
answer the design named is a single cheap measurement: allocate the weights, run one
forward pass at the workload ceiling, read peak allocated and
``cuda_used − torch_reserved``, discard.  One profile run for the chosen plan, not one
per candidate.

This module is that measurement.  Three properties make it safe to run on a load path:

**It is one pass, at the ceiling.**  The widest step the plan will ever take is a
prefill at the target context, so that is what is measured.  Anything smaller would
seed σ from a workload the plan does not have to survive.

**An OOM is evidence, not a crash.**  If the bootstrap pass exhausts the device, that
is the single most informative reading available: it proves the prediction was low, and
it must make the *next* plan more conservative rather than kill the process being
protected.  So the failure is caught, folded into the ledger as a residual at least as
large as the remaining capacity, and reported.

**It never changes the running plan.**  Calibration is not control.  The bootstrap
tightens the next prediction; it does not re-place a model that is already loaded.  A
caller that wants the tighter answer asks for a replan explicitly, and
:meth:`~aether.placement.planner.ExecutionPlanner.calibrate` returns whether one would
now differ.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aether.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aether.placement.planner import PlannerDecision

logger = get_logger(__name__)

__all__ = ["MemoryReading", "BootstrapResult", "read_memory", "reset_peak_stats", "bootstrap"]


@dataclass(frozen=True)
class MemoryReading:
    """What one device actually held, read from the allocator and the driver."""

    device_id: str
    peak_allocated_bytes: int = 0
    """Framework peak — the number the planner's transient prediction is about."""

    reserved_bytes: int = 0
    driver_used_bytes: int = 0
    total_bytes: int = 0

    @property
    def non_framework_bytes(self) -> int:
        """``cuda_used − torch_reserved``: the term that cannot be modelled.

        This is vLLM's "non-torch memory" — CUDA context, cuBLAS and cuDNN workspaces,
        collective buffers.  Measuring it is the only honest option, which is why it is
        read here rather than estimated anywhere.
        """
        return max(0, self.driver_used_bytes - self.reserved_bytes)

    @property
    def fragmentation(self) -> float:
        """Reserved over allocated — the allocator's inability to merge blocks.

        Returns ``0.0`` when nothing was allocated, so the caller can tell "no
        fragmentation measured" from "fragmentation of 1.0".
        """
        if self.peak_allocated_bytes <= 0:
            return 0.0
        return max(1.0, self.reserved_bytes / self.peak_allocated_bytes)

    def to_dict(self) -> dict[str, Any]:
        gib = 1024 ** 3
        return {
            "device_id": self.device_id,
            "peak_allocated_gib": round(self.peak_allocated_bytes / gib, 3),
            "reserved_gib": round(self.reserved_bytes / gib, 3),
            "non_framework_gib": round(self.non_framework_bytes / gib, 3),
            "fragmentation": round(self.fragmentation, 3),
        }


@dataclass
class BootstrapResult:
    """Outcome of the calibration pass, including the case where it failed."""

    ran: bool = False
    readings: tuple[MemoryReading, ...] = ()
    seconds: float = 0.0
    oom: bool = False
    error: str = ""
    skipped: str = ""
    """Why no measurement was taken, when none was. Empty when one was."""

    updated: tuple[str, ...] = ()
    """Ledger keys this pass wrote."""

    notes: list[str] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        """Whether the ledger now holds a real measurement for these devices."""
        return self.ran and bool(self.readings) and not self.oom

    def summary(self) -> str:
        if self.skipped:
            return f"bootstrap skipped: {self.skipped}"
        if self.oom:
            return (
                "bootstrap pass exhausted the device; the shortfall was recorded, so "
                "the next plan is more conservative rather than optimistic"
            )
        if not self.ran:
            return f"bootstrap did not run: {self.error or 'unknown reason'}"
        peaks = ", ".join(
            f"{reading.device_id} {reading.peak_allocated_bytes / 1024 ** 3:.2f} GiB"
            for reading in self.readings
        )
        return f"bootstrap measured peak {peaks} in {self.seconds * 1e3:.0f} ms"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "calibrated": self.calibrated,
            "oom": self.oom,
            "seconds": round(self.seconds, 4),
            "skipped": self.skipped,
            "error": self.error,
            "readings": [reading.to_dict() for reading in self.readings],
            "updated": list(self.updated),
            "notes": list(self.notes),
        }


# ── memory probes ─────────────────────────────────────────────────────────────

def reset_peak_stats(device_ids: "tuple[str, ...] | list[str]") -> None:
    """Clear the allocator's peak counters so the next read describes one pass.

    Without this the peak reflects whatever the process did earlier — model loading,
    another model, a previous request — and the residual would measure history rather
    than this workload.
    """
    try:
        import torch
    except ImportError:
        return
    for device_id in device_ids:
        try:
            if device_id.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device_id)
            elif device_id.startswith("xpu") and hasattr(
                getattr(torch, "xpu", None), "reset_peak_memory_stats"
            ):
                torch.xpu.reset_peak_memory_stats(device_id)
            elif device_id == "mps" and hasattr(torch.mps, "reset_peak_memory_stats"):
                torch.mps.reset_peak_memory_stats()
        except Exception as exc:  # noqa: BLE001 - a missing counter is not fatal
            logger.debug("could not reset peak stats on %s: %s", device_id, exc)


def read_memory(device_id: str) -> MemoryReading | None:
    """Read peak allocated, reserved and driver-used bytes for one device.

    Returns ``None`` when the device has no readable counters — a CPU, or a backend
    without allocator statistics — because a fabricated zero would be folded into σ as
    if it were a measurement.

    Each backend is asked for the counters *it* publishes rather than being assumed to
    have CUDA's set.  A backend that publishes none defers the bootstrap, which is the
    honest outcome: an unmeasured device keeps its documented prior instead of being
    credited with a fabricated measurement.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if device_id.startswith("cuda"):
            index = int(device_id.split(":")[1]) if ":" in device_id else 0
            free, total = torch.cuda.mem_get_info(index)
            return MemoryReading(
                device_id=device_id,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
                reserved_bytes=int(torch.cuda.memory_reserved(index)),
                driver_used_bytes=int(total) - int(free),
                total_bytes=int(total),
            )
        if device_id.startswith("xpu"):
            xpu = getattr(torch, "xpu", None)
            peak = getattr(xpu, "max_memory_allocated", None)
            reserved = getattr(xpu, "memory_reserved", None)
            if peak is None:
                return None
            allocated = int(peak(device_id))
            held = int(reserved(device_id)) if reserved is not None else allocated
            # Intel's runtime publishes no driver-wide total, so the non-framework term
            # is unknown rather than zero; reporting `driver_used == reserved` makes
            # `non_framework_bytes` zero, which the caller reads as "not measured".
            return MemoryReading(
                device_id=device_id,
                peak_allocated_bytes=allocated,
                reserved_bytes=held,
                driver_used_bytes=held,
            )
        if device_id == "mps":
            current = getattr(torch.mps, "current_allocated_memory", None)
            driver = getattr(torch.mps, "driver_allocated_memory", None)
            if current is None:
                return None
            allocated = int(current())
            return MemoryReading(
                device_id=device_id,
                peak_allocated_bytes=allocated,
                reserved_bytes=int(driver()) if driver is not None else allocated,
                driver_used_bytes=int(driver()) if driver is not None else allocated,
            )
    except Exception as exc:  # noqa: BLE001 - never let a probe break a load
        logger.debug("could not read memory on %s: %s", device_id, exc)
    return None


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether an exception is an allocation failure rather than a real bug.

    Matched on the type name and message rather than by importing
    ``torch.cuda.OutOfMemoryError``, so the check works on a torch-free install and
    across the versions that moved the class.
    """
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "OutOfMemoryException"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text


# ── the pass ──────────────────────────────────────────────────────────────────

def bootstrap(
    decision: "PlannerDecision",
    forward: "Callable[[int, int], Any]",
    ledger: Any,
    *,
    tokens: int = 0,
    batch: int = 0,
) -> BootstrapResult:
    """Run one forward pass at the workload ceiling and fold the result into the ledger.

    Args:
        decision: The plan that was selected. Its budgets carry the predictions this
            pass is measuring against.
        forward: ``forward(batch, tokens)`` — runs exactly one pass and returns
            anything. The caller owns it because only the caller knows how to drive its
            engine; the bootstrap owns the *measurement*, which is the part that has to
            be right.
        ledger: The calibration store to write to.
        tokens: Sequence length for the pass. Defaults to the workload's target
            context, which is the widest step the plan has to survive.
        batch: Batch for the pass. Defaults to the workload's target batch.

    Returns:
        A :class:`BootstrapResult`. Never raises for an allocation failure — that is
        recorded as evidence — and never raises for a missing probe.
    """
    workload = decision.workload
    tokens = tokens or workload.context_target
    batch = batch or workload.batch_target
    device_ids = [
        budget.device_id for budget in decision.selected.budgets
        if not budget.device_id.startswith("cpu")
    ]
    if not device_ids:
        return BootstrapResult(skipped="no accelerator in the selected plan")

    reset_peak_stats(device_ids)
    # Probe once before running anything: a backend with no allocator counters must be
    # detected *before* the pass, so a device that cannot be measured costs nothing
    # rather than a wasted prefill at the ceiling.
    if all(read_memory(device_id) is None for device_id in device_ids):
        return BootstrapResult(skipped="no readable allocator counters on this backend")

    started = time.perf_counter()
    oom = False
    error = ""
    try:
        forward(batch, tokens)
    except Exception as exc:  # noqa: BLE001 - the failure is the measurement
        if _is_out_of_memory(exc):
            oom = True
            error = str(exc)[:300]
            logger.warning(
                "bootstrap forward pass at batch %d x %d tokens ran out of memory; "
                "recording the shortfall so the next plan is tighter", batch, tokens,
            )
        else:
            logger.warning("bootstrap forward pass failed (%s); keeping priors", exc)
            return BootstrapResult(ran=False, error=str(exc)[:300])
    elapsed = time.perf_counter() - started

    readings = tuple(
        reading for reading in (read_memory(device_id) for device_id in device_ids)
        if reading is not None
    )
    if not readings:
        return BootstrapResult(skipped="allocator counters disappeared mid-pass")

    result = BootstrapResult(
        ran=True, readings=readings, seconds=elapsed, oom=oom, error=error
    )
    updated: list[str] = []
    by_device = {budget.device_id: budget for budget in decision.selected.budgets}

    for reading in readings:
        budget = by_device.get(reading.device_id)
        if budget is None:
            continue
        device = decision.census.by_id(reading.device_id)
        predicted = int(budget.predicted_peak_bytes)
        observed = int(reading.peak_allocated_bytes)
        if oom:
            # The pass did not complete, so the peak counter is a *lower* bound on
            # what was needed. Charging the whole safe capacity is the honest reading:
            # we know the requirement exceeded what was available.
            observed = max(observed, int(budget.safe_capacity_bytes))
            result.notes.append(
                f"{reading.device_id}: recorded a peak of at least "
                f"{observed / 1024 ** 3:.2f} GiB against a prediction of "
                f"{predicted / 1024 ** 3:.2f} GiB"
            )
        entry = ledger.observe_execution(
            device.signature,
            decision.census.backend_build,
            predicted_transient_bytes=predicted,
            observed_peak_bytes=observed,
            r_fixed_bytes=reading.non_framework_bytes or None,
            fragmentation=reading.fragmentation or None,
        )
        updated.append(entry.key)
        logger.info(
            "bootstrap calibrated %s: predicted %.2f GiB, observed %.2f GiB, "
            "sigma now %.0f MiB over %d sample(s)",
            reading.device_id, predicted / 1024 ** 3, observed / 1024 ** 3,
            entry.residual_sigma / 1024 ** 2, entry.residual_samples,
        )

    result.updated = tuple(updated)
    return result
