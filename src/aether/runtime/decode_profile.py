"""Where a decode step's time actually goes, on whatever hardware is present.

The batch-scaling question — *why does the per-step cost jump when the batch grows
from one to two?* — cannot be answered by reading code, because the answer lives in
kernel selection and that is a property of the device.  So this module measures it,
attributing each step to the phases a decoder actually has:

    projections   the ``x @ Wᵀ`` calls: qkv, o, gate/up, down, lm_head
    attention     the fused or exact attention kernel
    kv            appending this step's keys and values to the cache
    norm          RMS/layer normalisation
    rope          rotary position application
    logits        the final projection to vocabulary width
    sampling      argmax or the temperature/top-k/top-p pipeline
    sync          the host waiting for the device
    other         everything not attributed above, which is dispatch overhead

Instrumentation is installed by a context manager and removed on exit, so the decode
path carries no profiling cost when nobody is profiling.  On CUDA the phase timers are
bracketed by a device synchronise, because a wall-clock reading taken while kernels are
still queued measures the *launch* of the work rather than the work.

Nothing here is vendor-specific: the phases are named after the model's own structure
and the timers use whatever synchronisation primitive the active backend exposes.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["PhaseTotals", "DecodeProfile", "profile_engine", "profile_batch_scaling"]

#: Engine methods wrapped, and the phase each is charged to.
_PHASES: "dict[str, str]" = {
    "_matmul": "projections",
    "_attention": "attention",
    "_append_kv": "kv",
    "_norm": "norm",
    "_rope": "rope",
    "_project_logits": "logits",
    "_sample_device": "sampling",
}


@dataclass
class PhaseTotals:
    """Accumulated time and call count per phase."""

    seconds: "dict[str, float]" = field(default_factory=lambda: defaultdict(float))
    calls: "dict[str, int]" = field(default_factory=lambda: defaultdict(int))
    steps: int = 0
    wall_seconds: float = 0.0

    def add(self, phase: str, elapsed: float) -> None:
        self.seconds[phase] += elapsed
        self.calls[phase] += 1

    @property
    def attributed(self) -> float:
        return sum(self.seconds.values())

    @property
    def unattributed(self) -> float:
        """Wall time not inside any instrumented phase.

        This is the dispatch and Python overhead — the term a roofline model has no
        name for and the one that dominates small-model decode. Reporting it as a
        residual rather than trying to time it directly is what keeps it honest.
        """
        return max(0.0, self.wall_seconds - self.attributed)

    def per_step(self) -> "dict[str, float]":
        steps = max(1, self.steps)
        rows = {phase: total / steps for phase, total in self.seconds.items()}
        rows["other"] = self.unattributed / steps
        return rows

    def to_dict(self) -> dict[str, Any]:
        steps = max(1, self.steps)
        per_step = self.per_step()
        total = self.wall_seconds / steps
        return {
            "steps": self.steps,
            "ms_per_step": round(total * 1e3, 4),
            "phases": {
                phase: {
                    "ms_per_step": round(value * 1e3, 4),
                    "share": round(value / total, 4) if total > 0 else 0.0,
                    "calls_per_step": round(self.calls.get(phase, 0) / steps, 2),
                }
                for phase, value in sorted(
                    per_step.items(), key=lambda item: -item[1]
                )
            },
        }

    def render(self, title: str = "") -> str:
        steps = max(1, self.steps)
        total = self.wall_seconds / steps
        lines = [
            f"{title}  {total * 1e3:.3f} ms/step over {self.steps} step(s)",
            f"  {'phase':<14}{'ms/step':>10}{'share':>8}{'calls':>9}",
        ]
        for phase, value in sorted(self.per_step().items(), key=lambda i: -i[1]):
            share = value / total * 100 if total > 0 else 0.0
            calls = self.calls.get(phase, 0) / steps
            lines.append(f"  {phase:<14}{value * 1e3:>10.3f}{share:>7.1f}%{calls:>9.1f}")
        return "\n".join(lines)


class DecodeProfile:
    """Instrument one engine instance for the duration of a ``with`` block.

    The wrapping is per instance rather than per class, so profiling one engine never
    perturbs another in the same process — which matters because the benchmark harness
    keeps two backends resident at once.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.totals = PhaseTotals()
        self._saved: "dict[str, Any]" = {}
        self._sync = _synchronizer(engine)

    def __enter__(self) -> "DecodeProfile":
        for name, phase in _PHASES.items():
            original = getattr(self.engine, name, None)
            if original is None or not callable(original):
                continue
            self._saved[name] = original
            setattr(self.engine, name, self._wrap(original, phase))
        return self

    def __exit__(self, *exc: Any) -> None:
        for name in self._saved:
            # Delete the instance override rather than reassigning it, so the engine is
            # left byte-identical to an engine that was never profiled — a reassigned
            # bound method would keep an instance attribute shadowing the class one.
            self.engine.__dict__.pop(name, None)
        self._saved.clear()

    def _wrap(self, original: "Callable[..., Any]", phase: str) -> "Callable[..., Any]":
        totals = self.totals
        sync = self._sync

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = original(*args, **kwargs)
            sync()
            totals.add(phase, time.perf_counter() - start)
            return result

        return wrapped

    @contextmanager
    def step(self) -> "Iterator[None]":
        """Bracket one decode step so wall time and phase time share a denominator."""
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals.wall_seconds += time.perf_counter() - start
            self.totals.steps += 1


def _synchronizer(engine: Any) -> "Callable[[], None]":
    """A barrier for the engine's device, or a no-op where none is needed.

    Under a profiler the barrier is required for the numbers to mean anything; it is
    also why a profiled step is slower than an unprofiled one, and why the *shares*
    rather than the absolute times are the figure to read.
    """
    torch = getattr(engine, "torch", None)
    device = getattr(engine, "device", None)
    kind = getattr(device, "type", "cpu")
    if torch is None or kind == "cpu":
        return lambda: None
    if kind == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if kind == "mps":
        return torch.mps.synchronize
    if kind == "xpu":
        return lambda: torch.xpu.synchronize()
    return lambda: None


# ── drivers ───────────────────────────────────────────────────────────────────

def profile_engine(
    engine: Any,
    *,
    batch: int = 1,
    context: int = 32,
    steps: int = 16,
    warmup: int = 4,
) -> PhaseTotals:
    """Profile ``steps`` decode steps at one batch size and context length.

    Works for both the single-sequence and the batched entry points, choosing by
    ``batch`` — which is the comparison the batch-scaling question needs.
    """
    import numpy as np

    torch = engine.torch
    reserve = context + steps + warmup + 2
    ids = np.zeros((batch, context), dtype=np.int64)

    if batch > 1:
        _, cache = engine._forward_device(
            torch.as_tensor(ids), None, reserve=reserve, batched=True, logits="last"
        )
        token = torch.zeros((batch, 1), dtype=torch.long, device=engine.device)

        def advance() -> None:
            nonlocal cache
            engine._extend_live(cache, cache.length + 1)
            _, cache = engine._forward_device(
                token, cache, reserve=reserve, batched=True, logits="last"
            )
    else:
        _, cache = engine._forward_device(
            ids[0], None, reserve=reserve, logits="last"
        )
        token = torch.zeros(1, dtype=torch.long, device=engine.device)

        def advance() -> None:
            nonlocal cache
            _, cache = engine._forward_device(token, cache, logits="last")

    for _ in range(max(0, warmup)):
        advance()
    with DecodeProfile(engine) as profile:
        for _ in range(max(1, steps)):
            with profile.step():
                advance()
    return profile.totals


def profile_batch_scaling(
    engine_factory: "Callable[[], Any]",
    *,
    batches: "tuple[int, ...]" = (1, 2, 4, 8),
    context: int = 32,
    steps: int = 16,
) -> dict[str, Any]:
    """Profile several batch sizes and report where the extra time went.

    A fresh engine per batch size, because a calibrated strategy for one row count must
    not be credited to another. The report names the phase whose *share* grew most from
    the narrowest batch, which is the phase a fix has to address.
    """
    results: "dict[int, PhaseTotals]" = {}
    for batch in batches:
        engine = engine_factory()
        results[batch] = profile_engine(
            engine, batch=batch, context=context, steps=steps
        )

    baseline = results[batches[0]]
    base_step = baseline.wall_seconds / max(1, baseline.steps)
    scaling = []
    for batch, totals in results.items():
        step = totals.wall_seconds / max(1, totals.steps)
        scaling.append({
            "batch": batch,
            "ms_per_step": round(step * 1e3, 4),
            "step_cost_vs_base": round(step / base_step, 4) if base_step else 0.0,
            "tokens_per_s": round(batch / step, 2) if step > 0 else 0.0,
            "scaling_efficiency": (
                round((batch / step) / (batches[0] / base_step) / (batch / batches[0]), 4)
                if step > 0 and base_step > 0 else 0.0
            ),
            "phases": totals.to_dict()["phases"],
        })

    widest = results[batches[-1]]
    growth = {
        phase: widest.per_step().get(phase, 0.0) - baseline.per_step().get(phase, 0.0)
        for phase in set(widest.per_step()) | set(baseline.per_step())
    }
    culprit = max(growth.items(), key=lambda item: item[1]) if growth else ("none", 0.0)
    return {
        "context": context,
        "steps": steps,
        "cells": scaling,
        "largest_absolute_growth": {
            "phase": culprit[0], "ms_per_step": round(culprit[1] * 1e3, 4),
        },
    }
