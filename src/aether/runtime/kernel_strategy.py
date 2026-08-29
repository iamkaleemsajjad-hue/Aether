"""Measured selection of decode kernel strategies, per hardware and per shape.

Autoregressive decode multiplies a *very* thin activation against a large weight
matrix: ``M`` is the batch (often 1–8), while ``K`` and ``N`` are the model's hidden
and intermediate widths.  Vendor GEMM libraries are tuned for large ``M`` and pad it
up to a tile, so the wall-clock cost of a decode projection is frequently unrelated
to its arithmetic — the same call can be faster or slower depending on operand
orientation, on whether ``M`` is padded to the tile explicitly, and on which library
path the shape happens to select.  The literature calls this the *flat GEMM* regime
and reports that no single formulation wins across shapes, dtypes and architectures.

That is the whole justification for this module: **the right formulation is a property
of the device, the dtype and the shape class, and it is therefore measured rather than
chosen.**  Nothing here assumes which candidate is fastest, on any hardware.

The contract
------------
1. Detect the backend and device.
2. Look for a stored calibration for this ``(device, backend, shape class)``.
3. If one exists, use it — no measurement, no benchmarking on the hot path.
4. Otherwise run a *bounded* calibration: time each candidate, discard any whose
   output is not numerically equivalent to the reference, and store the winner.
5. If the calibration budget or the environment makes measurement unsafe, keep the
   reference implementation and record that calibration was deferred.

The reference implementation is always eligible and always correct, so a device with
no calibration behaves exactly as it did before this module existed.  A candidate can
only ever be selected by beating the reference on a real measurement *and* matching it
numerically, which is what makes the mechanism safe to enable everywhere.

Why the shape class and not the exact shape
-------------------------------------------
Calibrating every ``(M, K, N)`` a model contains would run hundreds of probes at load
time.  The flat-GEMM effect is dominated by ``M`` — the padded dimension — with a
weaker dependence on the ``K×N`` magnitude, so the class is ``(phase, M bucket, K·N
magnitude bucket, dtype, device)``.  One probe therefore covers every projection of a
similar size in a phase, and a model's layers share it.

References:
  * Flat/thin GEMM in LLM decode: small ``M``, large ``K``/``N``, tile padding —
    surveyed in the low-latency GEMM literature for both CUDA and ROCm.
  * "Virtual padding" of the token dimension to a fixed tile, from production serving
    systems that mix prefill and decode shapes in one kernel set.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CALIBRATION_VERSION",
    "ShapeClass",
    "StrategyCandidate",
    "StrategyChoice",
    "StrategyCalibrator",
    "projection_strategies",
    "calibration_enabled",
]

CALIBRATION_VERSION = "aether_decode_strategy/1"
"""Bumped when a candidate's semantics change, so stale winners are not reused."""

_REFERENCE = "linear"
"""The always-correct baseline: ``torch.nn.functional.linear``.

Selected whenever calibration has not run, has been deferred, or found nothing
faster. Naming it explicitly is what makes "uncalibrated" and "calibrated to the
reference" distinguishable in the record."""

_EQUIVALENCE_RTOL = 2e-2
_EQUIVALENCE_ATOL = 2e-2
"""Tolerance for accepting a candidate as numerically equivalent.

Deliberately loose in *relative* terms and tight in absolute: a candidate is allowed
to reorder accumulation (which is what a different tile shape does) but not to change
the result. The bound is checked against the reference on the same inputs, in the
model's own dtype, so a half-precision candidate is judged at half-precision
tolerance rather than being failed for being half precision."""

_MAX_CANDIDATE_SECONDS = 0.75
"""Wall-clock ceiling for calibrating one shape class.

A model has a handful of classes and each is calibrated at most once per device, so
this is a bound on a load path rather than a per-request cost. It is set generously
enough that the *first* class in a cold process — which pays allocator growth and
first-touch on top of its own timing — still completes, because the classes that
matter most in decode are the ones a mean budget would starve. Exceeding it defers
the class to the reference kernel instead of extending the load or, worse, ranking a
partial field."""

_REPETITIONS = 12
_WARMUP = 4


def calibration_enabled() -> bool:
    """Whether decode-strategy calibration may run at all.

    ``AETHER_DECODE_CALIBRATION=0`` disables it, which pins every projection to the
    reference implementation. Provided because an operator reproducing a measurement
    needs to be able to take the mechanism out of the picture.
    """
    value = os.environ.get("AETHER_DECODE_CALIBRATION", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


# ── shape classes ─────────────────────────────────────────────────────────────

def _bucket(value: int) -> int:
    """Round down to a power of two — the granularity a GEMM tile actually sees."""
    if value <= 1:
        return 1
    return 1 << int(math.floor(math.log2(value)))


@dataclass(frozen=True)
class ShapeClass:
    """The class of projection a single strategy choice covers.

    ``phase`` separates prefill from decode because they sit in different regimes:
    prefill has a large ``M`` and behaves like a classical GEMM, decode has a thin one
    and does not. Bucketing ``M`` by powers of two matches how a tile pads it, and
    bucketing the weight magnitude keeps one probe covering a model's whole layer
    stack instead of one probe per projection.
    """

    phase: str
    rows: int
    """Bucketed ``M`` — the token count in this pass (batch × sequence)."""

    weight_magnitude: int
    """Bucketed ``K·N``, the weight element count."""

    dtype: str
    device_kind: str

    @classmethod
    def of(
        cls, *, phase: str, rows: int, in_features: int, out_features: int,
        dtype: str, device_kind: str,
    ) -> "ShapeClass":
        return cls(
            phase=phase,
            rows=_bucket(max(1, int(rows))),
            weight_magnitude=_bucket(max(1, int(in_features) * int(out_features))),
            dtype=str(dtype),
            device_kind=str(device_kind),
        )

    @property
    def key(self) -> str:
        return (
            f"{CALIBRATION_VERSION}|{self.device_kind}|{self.phase}"
            f"|m{self.rows}|w{self.weight_magnitude}|{self.dtype}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "rows_bucket": self.rows,
            "weight_magnitude_bucket": self.weight_magnitude,
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "key": self.key,
        }


# ── candidates ────────────────────────────────────────────────────────────────
#
# Every candidate computes exactly ``x @ Wᵀ + b`` and differs only in how it asks the
# backend to do it. They are written against the public torch API rather than any
# vendor library, so the same set is eligible on CUDA, ROCm, MPS, XPU and CPU, and the
# measurement decides which of them that particular backend implements well.


def _linear(torch: Any, x: Any, weight: Any, bias: Any | None) -> Any:
    """The reference. One fused call, whatever the backend maps it to."""
    return torch.nn.functional.linear(x, weight, bias)


def _transposed(torch: Any, x: Any, weight: Any, bias: Any | None) -> Any:
    """Compute ``(W · xᵀ)ᵀ`` so the thin dimension lands in ``N`` rather than ``M``.

    Mathematically identical; operationally different, because a library that pads the
    ``M`` dimension to a tile is asked to pad a different one. Whether that helps is
    exactly what cannot be known without measuring.
    """
    flat = x.reshape(-1, int(x.shape[-1]))
    out = torch.matmul(weight, flat.transpose(0, 1)).transpose(0, 1)
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], int(weight.shape[0]))


def _padded_rows(torch: Any, x: Any, weight: Any, bias: Any | None) -> Any:
    """Pad ``M`` up to a tile multiple, run one GEMM, then slice the result back.

    The padding is not overhead in the arithmetic sense — a tiled kernel was going to
    process those rows anyway — and handing the library its designed shape can select a
    better path than a ragged one. This is the "virtual padding" trick from production
    serving systems, expressed portably.
    """
    flat = x.reshape(-1, int(x.shape[-1]))
    rows = int(flat.shape[0])
    tile = _PAD_TILE
    remainder = rows % tile
    if remainder == 0:
        return _linear(torch, x, weight, bias)
    pad = tile - remainder
    padded = torch.nn.functional.pad(flat, (0, 0, 0, pad))
    out = torch.nn.functional.linear(padded, weight, bias)[:rows]
    return out.reshape(*x.shape[:-1], int(weight.shape[0]))


def _addmm(torch: Any, x: Any, weight: Any, bias: Any | None) -> Any:
    """Go through ``addmm``/``mm`` explicitly rather than through ``linear``.

    ``linear`` dispatches on rank and bias presence and may take a different library
    path than the two-dimensional primitives; on some backends the explicit form is
    the faster one and on others it is slower, so it is a candidate and not a rule.
    """
    flat = x.reshape(-1, int(x.shape[-1]))
    weight_t = weight.transpose(0, 1)
    out = (
        torch.mm(flat, weight_t)
        if bias is None
        else torch.addmm(bias, flat, weight_t)
    )
    return out.reshape(*x.shape[:-1], int(weight.shape[0]))


_PAD_TILE = 8
"""Row multiple used by the padding candidate.

Eight is the smallest tile every current tensor-core generation is a multiple of, so
padding to it never *over*-pads a decode batch by more than seven rows. The value only
affects a candidate that must beat the reference on measurement to be used at all."""

_CANDIDATES: "dict[str, Callable[[Any, Any, Any, Any | None], Any]]" = {
    _REFERENCE: _linear,
    "transposed": _transposed,
    "padded_rows": _padded_rows,
    "addmm": _addmm,
}


@dataclass(frozen=True)
class StrategyCandidate:
    """One measured candidate: its name, its time, and whether it was correct."""

    name: str
    seconds: float = 0.0
    equivalent: bool = True
    error: str = ""

    @property
    def eligible(self) -> bool:
        return self.equivalent and not self.error and self.seconds > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "us": round(self.seconds * 1e6, 3),
            "equivalent": self.equivalent,
            "error": self.error,
        }


@dataclass(frozen=True)
class StrategyChoice:
    """The outcome of calibrating one shape class."""

    shape: ShapeClass
    name: str = _REFERENCE
    source: str = "default"
    """``measured`` | ``cached`` | ``default`` | ``deferred`` | ``disabled``."""

    speedup: float = 1.0
    """Reference time over chosen time. ``1.0`` when the reference was kept."""

    candidates: tuple[StrategyCandidate, ...] = ()
    detail: str = ""

    @property
    def calibrated(self) -> bool:
        return self.source in ("measured", "cached")

    def explain(self) -> str:
        if self.source == "disabled":
            return "decode strategy calibration is disabled; using the reference kernel"
        if self.source == "deferred":
            return (
                f"decode strategy calibration deferred for {self.shape.key} "
                f"({self.detail or 'budget or backend'}); using the reference kernel"
            )
        if self.source == "default":
            return f"no calibration for {self.shape.key}; using the reference kernel"
        if self.name == _REFERENCE:
            return (
                f"{self.shape.key}: the reference kernel measured fastest of "
                f"{len(self.candidates)} candidates"
            )
        return (
            f"{self.shape.key}: {self.name} measured {self.speedup:.2f}x the reference "
            f"and matched it numerically"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.to_dict(),
            "name": self.name,
            "source": self.source,
            "speedup": round(self.speedup, 4),
            "calibrated": self.calibrated,
            "detail": self.detail,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


# ── the calibrator ────────────────────────────────────────────────────────────


class StrategyCalibrator:
    """Chooses a projection strategy per shape class, by measurement, once.

    One instance is held per engine. Choices are memoised in memory for the life of
    the process and persisted through ``store`` — normally the placement
    :class:`~aether.placement.ledger.CalibrationLedger`, which is already keyed by
    device signature and backend build — so the second load of a model on a machine
    performs no measurement at all.

    Args:
        torch: The framework module.
        device: The device strategies are calibrated for.
        store: Anything exposing the ledger's ``get``/``record_notes`` surface. When
            ``None`` the calibration lives only in memory, which is the right
            behaviour for a test or a one-shot script.
        signature: Device signature for the persisted key.
        backend_build: Runtime identity for the persisted key.
    """

    def __init__(
        self,
        torch: Any,
        device: Any,
        *,
        store: Any | None = None,
        signature: str = "",
        backend_build: str = "",
        budget_seconds: float = _MAX_CANDIDATE_SECONDS,
    ) -> None:
        self.torch = torch
        self.device = device
        self.store = store
        self.signature = signature
        self.backend_build = backend_build
        self.budget_seconds = float(budget_seconds)
        self._choices: dict[str, StrategyChoice] = {}
        self._persisted: dict[str, str] | None = None

    # ── device identity ───────────────────────────────────────────────────────

    @property
    def device_kind(self) -> str:
        """The class of hardware, not the individual card.

        A strategy that wins on one Ampere card wins on another, so the calibration is
        keyed by *architecture* where the backend exposes one and by device type
        otherwise.  That is what lets a fleet share one calibration instead of measuring
        per host.

        The vendor is part of the key rather than assumed from the device type.  PyTorch
        exposes AMD GPUs through ``torch.cuda``, so keying on ``device.type`` alone would
        label an MI300X ``cuda-sm…`` and could give two different vendors' architectures
        the same key.  ROCm reports a GCN architecture name, so that is what is used
        there; CUDA reports a compute capability; every other backend contributes its own
        type.  No branch here asserts what a device *is capable of* — only what it calls
        itself.
        """
        kind = getattr(self.device, "type", str(self.device))
        try:
            if kind == "cuda":
                if getattr(self.torch.version, "hip", None):
                    properties = self.torch.cuda.get_device_properties(self.device)
                    arch = str(
                        getattr(properties, "gcnArchName", "") or ""
                    ).split(":")[0].strip()
                    return f"rocm-{arch}" if arch else "rocm"
                major, minor = self.torch.cuda.get_device_capability(self.device)
                return f"cuda-sm{major}{minor}"
            if kind in ("xpu", "mps", "cpu"):
                return kind
        except Exception:  # noqa: BLE001 - identity is advisory, never fatal
            pass
        return str(kind)

    # ── selection ─────────────────────────────────────────────────────────────

    def strategy(self, choice: StrategyChoice) -> "Callable[[Any, Any, Any | None], Any]":
        """Bind a choice to a callable with the engine's ``(x, weight, bias)`` shape."""
        implementation = _CANDIDATES.get(choice.name, _linear)
        torch = self.torch

        def apply(x: Any, weight: Any, bias: Any | None = None) -> Any:
            return implementation(torch, x, weight, bias)

        return apply

    def choose(
        self,
        *,
        phase: str,
        rows: int,
        in_features: int,
        out_features: int,
        dtype: Any,
        probe: "Callable[[], tuple[Any, Any, Any | None]] | None" = None,
    ) -> StrategyChoice:
        """Return the strategy for this shape class, calibrating it at most once.

        ``probe`` supplies representative ``(x, weight, bias)`` tensors for the
        measurement. It is a callable so that nothing is allocated when the answer is
        already known — which is the common case after the first load.
        """
        shape = ShapeClass.of(
            phase=phase, rows=rows, in_features=in_features, out_features=out_features,
            dtype=str(dtype).replace("torch.", ""), device_kind=self.device_kind,
        )
        cached = self._choices.get(shape.key)
        if cached is not None:
            return cached
        choice = self._resolve(shape, probe)
        self._choices[shape.key] = choice
        return choice

    def _resolve(
        self,
        shape: ShapeClass,
        probe: "Callable[[], tuple[Any, Any, Any | None]] | None",
    ) -> StrategyChoice:
        if not calibration_enabled():
            return StrategyChoice(shape=shape, source="disabled")
        stored = self._load_persisted().get(shape.key)
        if stored in _CANDIDATES:
            logger.debug("decode strategy for %s from calibration: %s", shape.key, stored)
            return StrategyChoice(shape=shape, name=stored, source="cached")
        if probe is None:
            return StrategyChoice(
                shape=shape, source="deferred", detail="no probe tensors available"
            )
        try:
            choice = self._measure(shape, probe)
        except Exception as exc:  # noqa: BLE001 - never let calibration break a load
            logger.debug("decode strategy calibration failed for %s: %s", shape.key, exc)
            return StrategyChoice(shape=shape, source="deferred", detail=str(exc)[:120])
        if choice.source == "measured":
            self._persist(shape.key, choice.name)
            logger.info("decode strategy: %s", choice.explain())
        return choice

    # ── measurement ───────────────────────────────────────────────────────────

    def _synchronize(self) -> None:
        """Drain the device queue so a wall-clock reading describes real work.

        Without this a launch-only measurement would rank the candidates by how
        cheaply they *enqueue*, which is the opposite of the question being asked.

        The barrier is looked up by device type rather than enumerated per vendor, so a
        backend Aether has not seen before is still measured correctly instead of having
        its launch cost mistaken for its execution cost.
        """
        torch = self.torch
        kind = getattr(self.device, "type", "cpu")
        if kind == "cpu":
            return
        try:
            if kind == "cuda":
                torch.cuda.synchronize(self.device)
                return
            namespace = getattr(torch, kind, None)
            synchronize = getattr(namespace, "synchronize", None)
            if synchronize is not None:
                synchronize()
        except Exception:  # noqa: BLE001 - a backend without a barrier needs none
            pass

    def _measure(
        self,
        shape: ShapeClass,
        probe: "Callable[[], tuple[Any, Any, Any | None]]",
    ) -> StrategyChoice:
        torch = self.torch
        x, weight, bias = probe()

        with torch.inference_mode():
            # Warm the backend *before* the clock starts. A first matmul in a fresh
            # process pays thread-pool spin-up, allocator growth and kernel
            # autotuning, none of which belong to any candidate — and charging them to
            # whichever candidate happened to run first would both mis-rank the field
            # and burn the whole budget on cold-start cost.
            for _ in range(_WARMUP):
                _linear(torch, x, weight, bias)
            self._synchronize()
            reference = _linear(torch, x, weight, bias).detach().clone()
            started = time.perf_counter()

            results: list[StrategyCandidate] = []
            for name, implementation in _CANDIDATES.items():
                if time.perf_counter() - started > self.budget_seconds:
                    # Out of budget with candidates left: the honest outcome is a
                    # deferral, not a winner picked from a partial field.
                    return StrategyChoice(
                        shape=shape, source="deferred",
                        detail=f"budget of {self.budget_seconds * 1e3:.0f} ms exhausted",
                        candidates=tuple(results),
                    )
                results.append(
                    self._time_candidate(name, implementation, x, weight, bias, reference)
                )

        eligible = [candidate for candidate in results if candidate.eligible]
        if not eligible:
            return StrategyChoice(
                shape=shape, source="deferred", detail="no candidate was measurable",
                candidates=tuple(results),
            )
        best = min(eligible, key=lambda candidate: candidate.seconds)
        baseline = next(
            (c.seconds for c in eligible if c.name == _REFERENCE), best.seconds
        )
        speedup = baseline / best.seconds if best.seconds > 0 else 1.0
        # A candidate has to be *convincingly* faster, not faster within noise: a
        # margin below the measurement's own repeatability would make the choice
        # depend on which order the probes happened to run in.
        if best.name != _REFERENCE and speedup < 1.0 + _MIN_MARGIN:
            best = next(c for c in eligible if c.name == _REFERENCE)
            speedup = 1.0
        return StrategyChoice(
            shape=shape, name=best.name, source="measured", speedup=speedup,
            candidates=tuple(results),
        )

    def _time_candidate(
        self,
        name: str,
        implementation: "Callable[[Any, Any, Any, Any | None], Any]",
        x: Any,
        weight: Any,
        bias: Any | None,
        reference: Any,
    ) -> StrategyCandidate:
        """Time one candidate and check it against the reference.

        Correctness is checked before speed is trusted: a formulation that is fast
        because it computes something else is not a candidate, and on a backend with
        an incomplete operator set that is a real possibility rather than a
        hypothetical one.
        """
        torch = self.torch
        try:
            produced = implementation(torch, x, weight, bias)
            if tuple(produced.shape) != tuple(reference.shape):
                return StrategyCandidate(name, equivalent=False, error="shape mismatch")
            equivalent = bool(
                torch.allclose(
                    produced.float(), reference.float(),
                    rtol=_EQUIVALENCE_RTOL, atol=_EQUIVALENCE_ATOL,
                )
            )
            if not equivalent:
                return StrategyCandidate(name, equivalent=False, error="numeric mismatch")
            for _ in range(_WARMUP):
                implementation(torch, x, weight, bias)
            self._synchronize()
            start = time.perf_counter()
            for _ in range(_REPETITIONS):
                implementation(torch, x, weight, bias)
            self._synchronize()
            elapsed = (time.perf_counter() - start) / _REPETITIONS
            return StrategyCandidate(name, seconds=max(elapsed, 1e-12))
        except Exception as exc:  # noqa: BLE001 - an unsupported op is a failed candidate
            return StrategyCandidate(name, equivalent=False, error=str(exc)[:120])

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_persisted(self) -> dict[str, str]:
        if self._persisted is not None:
            return self._persisted
        self._persisted = {}
        if self.store is None or not self.signature:
            return self._persisted
        try:
            entry = self.store.get(self.signature, self.backend_build)
            stored = (entry.notes or {}).get("decode_strategies") or {}
            self._persisted = {str(k): str(v) for k, v in stored.items()}
        except Exception as exc:  # noqa: BLE001 - a missing store is not an error
            logger.debug("could not read decode strategy calibration: %s", exc)
        return self._persisted

    def _persist(self, key: str, name: str) -> None:
        self._load_persisted()[key] = name
        if self.store is None or not self.signature:
            return
        recorder = getattr(self.store, "record_notes", None)
        if recorder is None:
            return
        try:
            recorder(
                self.signature, self.backend_build,
                {"decode_strategies": {key: name}},
            )
        except Exception as exc:  # noqa: BLE001 - persistence is best effort
            logger.debug("could not persist decode strategy calibration: %s", exc)

    # ── reporting ─────────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        """Every choice this process has made, for a decision record or a test."""
        return {
            "device_kind": self.device_kind,
            "enabled": calibration_enabled(),
            "choices": [choice.to_dict() for choice in self._choices.values()],
        }


_MIN_MARGIN = 0.05
"""Fraction a candidate must beat the reference by before it is selected.

Five percent is comfortably outside the spread of a twelve-repetition timing on a
quiet device and well inside the effect the flat-GEMM literature reports, so the gate
rejects noise without rejecting a real win."""


def projection_strategies(
    torch: Any,
    device: Any,
    *,
    store: Any | None = None,
    signature: str = "",
    backend_build: str = "",
) -> StrategyCalibrator:
    """Construct a calibrator. Kept as a function so callers need not import the class."""
    return StrategyCalibrator(
        torch, device, store=store, signature=signature, backend_build=backend_build,
    )



