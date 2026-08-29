"""Persisted calibration — the terms that cannot be derived, only measured.

Four quantities in the planner have no analytic form, and pretending otherwise is
how a memory model becomes a fudge factor:

``r_fixed_bytes``
    CUDA context, cuBLAS/cuDNN workspace, collective buffers — memory the driver
    and the libraries hold outside the framework's accounting.  Measured as
    ``cuda_used − torch_reserved``.

``fragmentation``
    The multiplicative gap between peak *allocated* bytes and the reserved bytes
    needed to satisfy them, caused by the caching allocator's inability to merge
    blocks across segments.  Near 1.05–1.10 with expandable segments; up to 2×
    without, because a small-then-large allocation order can leave every freed
    block too small for the next request.

``dispatch_seconds``
    Host-side cost per graph operation.  A property of the *runtime build*, not the
    device, which is why entries are keyed by backend as well as by device.

``peak residual σ``
    The standard deviation of (observed peak − predicted peak).  This is the width
    of the feasibility lane's one-sided margin, and it is the reason the margin
    shrinks as evidence accumulates instead of staying at a hand-set reserve.

``Law I crossover``
    The throughput ratio at which a heterogeneous tensor-parallel group stops meeting
    its water-filled prediction.  Bracketed from both sides across runs, so the
    homogeneity tolerance becomes a measurement of this machine rather than a constant.

Statistics use Welford's online algorithm so the ledger stores three numbers per
entry rather than a growing sample list, and remains numerically stable over
thousands of runs.

Reference: B. P. Welford, "Note on a method for calculating corrected sums of
squares and products", Technometrics 4(3), 1962.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["LEDGER_VERSION", "DISPATCH_STALE_RATIO", "LedgerEntry", "CalibrationLedger"]

LEDGER_VERSION = "aether_placement_ledger/1"

_CUDA_CONTEXT_PRIOR_BYTES = 640 << 20
"""Conservative CUDA context + library workspace prior, until measured."""

_FRAGMENTATION_PRIOR_EXPANDABLE = 1.08
_FRAGMENTATION_PRIOR_DEFAULT = 1.25
_DISPATCH_PRIOR_SECONDS = 25e-6
"""Per-op host cost prior. Deliberately mid-range: too low and the planner shards
dispatch-bound models, too high and it refuses to shard bandwidth-bound ones."""

_UNCALIBRATED_SIGMA_FRACTION = 0.15
"""With no residual history, sigma is taken as this fraction of the predicted peak."""

_CALIBRATED_MARGIN_FLOOR = 0.02
"""Smallest margin a calibrated entry may produce, as a fraction of the prediction.

Several identical readings prove that nothing has varied *yet*, not that nothing can.
Without a floor, a handful of repeatable runs would drive the safety margin to zero
and the first unusual allocation would kill the process."""

_UNCALIBRATED_LATENCY_FRACTION = 0.25
"""Relative latency uncertainty before any timing evidence exists.

Shared with the planner's σ-gated tie-break so that Law I's derivation, the plan
ranking and the TP-crossover test all speak about the same error bar."""

_LATENCY_SIGMA_FLOOR = 0.02
"""Smallest relative latency σ a measured history may claim.

The same floor, for the same reason, as :data:`_CALIBRATED_MARGIN_FLOOR`: identical
readings prove nothing has varied yet, not that nothing can.  Without it a perfectly
repeatable device would judge every TP group against a zero-width band and record a
crossover on the first microsecond of noise."""

DISPATCH_STALE_RATIO = 2.0
"""Divergence at which a stored dispatch cost is treated as mis-keyed.

``t_dispatch`` is a property of the runtime build, and the key cannot capture every
change to it — a fused decode path or a captured CUDA graph moves the number without
moving a version string.  A fresh probe that disagrees by this factor is therefore
taken as proof that the stored value belongs to a different runtime, and it is
replaced rather than trusted.  Two-fold is chosen because it is well outside probe
noise (a few percent) and well inside the order-of-magnitude change that graph capture
produces, so the check fires on real staleness and not on jitter."""


def _expandable_segments_enabled() -> bool:
    """Whether PyTorch's expandable-segment allocator is on.

    It changes the fragmentation prior by more than a factor of two, because
    expandable segments share one virtual range and therefore coalesce freed blocks
    that separate ``cudaMalloc`` segments never could.
    """
    config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:True" in config.replace(" ", "")


@dataclass
class LedgerEntry:
    """Calibration for one (device signature, backend build) pair."""

    key: str
    r_fixed_bytes: int = 0
    fragmentation: float = 0.0
    dispatch_seconds: float = 0.0
    achieved_flops: float = 0.0
    measured_bandwidth_bps: float = 0.0

    # Welford accumulators over (observed_peak − predicted_peak), in bytes.
    residual_samples: int = 0
    residual_mean: float = 0.0
    residual_m2: float = 0.0

    # Welford accumulators over the *relative* latency error,
    # (observed − predicted) / predicted, which is dimensionless and therefore
    # comparable across models on the same device.
    latency_samples: int = 0
    latency_mean: float = 0.0
    latency_m2: float = 0.0

    # Law I's measured crossover, bracketed from both sides. A heterogeneous
    # tensor-parallel group either meets its water-filled prediction or it does not,
    # and the ratio at which that flips is the tolerance the design asked for.
    tp_ratio_ok_max: float = 0.0
    """Widest θ ratio whose measured group time matched the prediction within σ."""

    tp_ratio_bad_min: float = 0.0
    """Narrowest θ ratio whose measured group time exceeded the prediction by > σ."""

    tp_samples: int = 0

    dispatch_measured: bool = False
    """True when ``dispatch_seconds`` came from a probe rather than the prior.

    The dispatch roof is a property of the runtime build, so a stale or mis-keyed
    value systematically distorts every sharding verdict. Recording whether the number
    was measured is what lets the planner say so instead of presenting a prior as a
    fact."""

    runs: int = 0
    last_seen: float = field(default_factory=time.time)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def residual_sigma(self) -> float:
        """Sample standard deviation of the peak-memory prediction residual.

        Returns ``0.0`` below two samples; the caller must then fall back to an
        uncalibrated prior rather than treat zero variance as certainty.
        """
        if self.residual_samples < 2:
            return 0.0
        return math.sqrt(max(0.0, self.residual_m2 / (self.residual_samples - 1)))

    @property
    def latency_sigma(self) -> float:
        """Relative standard deviation of the latency prediction error.

        Dimensionless, so one number covers every model on this device. It is the
        width used by the σ-gated tie-break and by Law I's derivation: a wider plan
        must beat a narrower one by more than this, or the planner is chasing its own
        noise.

        Returns the raw statistic, which is ``0.0`` both when there is no evidence and
        when every reading agreed exactly.  Those are different states — see
        :attr:`has_latency_evidence` — and a caller that conflates them will treat a
        perfectly predictable device as an unmeasured one.
        """
        if self.latency_samples < 2:
            return 0.0
        return math.sqrt(max(0.0, self.latency_m2 / (self.latency_samples - 1)))

    @property
    def has_latency_evidence(self) -> bool:
        """Whether enough timings exist to prefer the measurement over the prior.

        Separate from ``latency_sigma > 0`` on purpose: a device that predicted itself
        exactly twelve times has a σ of zero *and* the strongest possible evidence, and
        reading that as "unmeasured" would keep the planner permanently uncertain about
        its most reliable hardware.
        """
        return self.latency_samples >= 2

    @property
    def is_calibrated(self) -> bool:
        """Whether this entry has enough evidence to narrow the safety margin."""
        return self.residual_samples >= 5 and self.r_fixed_bytes > 0

    @property
    def tp_crossover_ratio(self) -> float:
        """Measured Law I tolerance: the θ ratio at which a TP group breaks.

        The two accumulators bracket the crossover from below and above, so the
        estimate is the geometric mean of the bracket — geometric because the quantity
        is a ratio, and the midpoint of a ratio interval is its geometric centre.

        Returns ``0.0`` when there is no upper bracket, which means "no crossover has
        been observed yet"; the caller must then keep its analytic bound rather than
        read zero as a refusal to shard.
        """
        if self.tp_ratio_bad_min <= 0:
            return 0.0
        if self.tp_ratio_ok_max <= 0 or self.tp_ratio_ok_max >= self.tp_ratio_bad_min:
            # No usable lower bracket: the known-bad ratio is the only evidence, and a
            # bound *at* it is the conservative reading.
            return self.tp_ratio_bad_min
        return math.sqrt(self.tp_ratio_ok_max * self.tp_ratio_bad_min)

    def observe_tp_group(
        self, theta_ratio: float, predicted_seconds: float, observed_seconds: float
    ) -> None:
        """Record whether a heterogeneous TP group met its water-filled prediction.

        A group whose measured time exceeds the prediction by more than the entry's own
        relative σ has broken the water-filling assumption, and its θ ratio bounds the
        crossover from above.  One that met the prediction bounds it from below.  This
        is the measurement the design named as the replacement for a guessed tolerance,
        and it uses the σ already being tracked rather than a new threshold.
        """
        if theta_ratio < 1.0 or predicted_seconds <= 0 or observed_seconds <= 0:
            return
        # The comparison band is the entry's own relative σ, so the crossover is defined
        # against the planner's real error bar rather than a new threshold. Before any
        # timing history exists there is nothing to compare against, so the documented
        # uncalibrated fraction stands in; once history exists it is floored for the same
        # reason the memory margin is — readings that agreed exactly prove nothing has
        # varied yet, not that nothing can.
        if self.has_latency_evidence:
            sigma = max(self.latency_sigma, _LATENCY_SIGMA_FLOOR)
        else:
            sigma = _UNCALIBRATED_LATENCY_FRACTION
        overshoot = (observed_seconds - predicted_seconds) / predicted_seconds
        self.tp_samples += 1
        if overshoot > sigma:
            self.tp_ratio_bad_min = (
                theta_ratio if self.tp_ratio_bad_min <= 0
                else min(self.tp_ratio_bad_min, theta_ratio)
            )
        else:
            self.tp_ratio_ok_max = max(self.tp_ratio_ok_max, theta_ratio)

    def observe_residual(self, predicted_bytes: int, observed_bytes: int) -> None:
        """Fold one prediction error into the running statistics (Welford)."""
        residual = float(observed_bytes - predicted_bytes)
        self.residual_samples += 1
        delta = residual - self.residual_mean
        self.residual_mean += delta / self.residual_samples
        self.residual_m2 += delta * (residual - self.residual_mean)

    def observe_latency(self, predicted_seconds: float, observed_seconds: float) -> None:
        """Fold one relative latency error into the running statistics."""
        if predicted_seconds <= 0:
            return
        relative = (observed_seconds - predicted_seconds) / predicted_seconds
        self.latency_samples += 1
        delta = relative - self.latency_mean
        self.latency_mean += delta / self.latency_samples
        self.latency_m2 += delta * (relative - self.latency_mean)

    def margin_bytes(self, predicted_bytes: int, z: float) -> int:
        """The one-sided upper bound to add to a predicted transient peak.

        Two properties matter. The mean residual is carried through, so a
        systematically low estimator is corrected rather than merely padded; and
        only *positive* mean residual is added, because a systematically high
        estimator should give its slack back to the KV cache instead of keeping it.
        """
        if self.residual_samples < 2:
            return int(predicted_bytes * _UNCALIBRATED_SIGMA_FRACTION * max(z, 1.0))
        measured = max(0.0, self.residual_mean) + z * self.residual_sigma
        # A floor, because a handful of identical readings is not proof of zero
        # variance — it is proof that nothing has varied *yet*.
        return int(max(measured, predicted_bytes * _CALIBRATED_MARGIN_FLOOR))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "r_fixed_bytes": self.r_fixed_bytes,
            "fragmentation": self.fragmentation,
            "dispatch_seconds": self.dispatch_seconds,
            "achieved_flops": self.achieved_flops,
            "measured_bandwidth_bps": self.measured_bandwidth_bps,
            "residual_samples": self.residual_samples,
            "residual_mean": self.residual_mean,
            "residual_m2": self.residual_m2,
            "latency_samples": self.latency_samples,
            "latency_mean": self.latency_mean,
            "latency_m2": self.latency_m2,
            "tp_ratio_ok_max": self.tp_ratio_ok_max,
            "tp_ratio_bad_min": self.tp_ratio_bad_min,
            "tp_samples": self.tp_samples,
            "dispatch_measured": self.dispatch_measured,
            "runs": self.runs,
            "last_seen": self.last_seen,
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LedgerEntry":
        return cls(
            key=str(data.get("key", "")),
            r_fixed_bytes=int(data.get("r_fixed_bytes", 0) or 0),
            fragmentation=float(data.get("fragmentation", 0.0) or 0.0),
            dispatch_seconds=float(data.get("dispatch_seconds", 0.0) or 0.0),
            achieved_flops=float(data.get("achieved_flops", 0.0) or 0.0),
            measured_bandwidth_bps=float(data.get("measured_bandwidth_bps", 0.0) or 0.0),
            residual_samples=int(data.get("residual_samples", 0) or 0),
            residual_mean=float(data.get("residual_mean", 0.0) or 0.0),
            residual_m2=float(data.get("residual_m2", 0.0) or 0.0),
            latency_samples=int(data.get("latency_samples", 0) or 0),
            latency_mean=float(data.get("latency_mean", 0.0) or 0.0),
            latency_m2=float(data.get("latency_m2", 0.0) or 0.0),
            tp_ratio_ok_max=float(data.get("tp_ratio_ok_max", 0.0) or 0.0),
            tp_ratio_bad_min=float(data.get("tp_ratio_bad_min", 0.0) or 0.0),
            tp_samples=int(data.get("tp_samples", 0) or 0),
            dispatch_measured=bool(data.get("dispatch_measured", False)),
            runs=int(data.get("runs", 0) or 0),
            last_seen=float(data.get("last_seen", 0.0) or 0.0),
            notes=dict(data.get("notes", {}) or {}),
        )


class CalibrationLedger:
    """The planner's memory of what this machine actually does.

    Entries are keyed by ``"<device signature>|<backend build>"``.  Bandwidth is
    keyed by device signature alone, because memory bandwidth is a property of the
    silicon and does not change when the framework is upgraded — whereas dispatch
    cost does, which is exactly why the two are keyed differently.

    Reads return a fully-populated entry whether or not one exists on disk: an
    absent entry yields documented priors and ``is_calibrated == False``, so a
    caller can never mistake a default for a measurement.
    """

    def __init__(self, path: str | Path | None = None, *, autosave: bool = True) -> None:
        self.path = Path(path) if path is not None else self._default_path()
        self.autosave = autosave
        self._entries: dict[str, LedgerEntry] = {}
        self._bandwidth: dict[str, float] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._load()

    @staticmethod
    def _default_path() -> Path:
        override = os.environ.get("AETHER_PLACEMENT_LEDGER", "").strip()
        if override:
            return Path(override).expanduser()
        try:
            from aether.utils.file_io import aether_cache_dir

            return Path(aether_cache_dir()) / "placement" / "calibration.json"
        except Exception:  # noqa: BLE001
            return Path.home() / ".aether" / "placement" / "calibration.json"

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("placement ledger at %s is unreadable (%s); starting empty", self.path, exc)
            return
        if str(payload.get("version", "")) != LEDGER_VERSION:
            logger.info("placement ledger version mismatch; starting empty")
            return
        self._bandwidth = {
            str(key): float(value)
            for key, value in (payload.get("bandwidth") or {}).items()
        }
        for key, data in (payload.get("entries") or {}).items():
            entry = LedgerEntry.from_dict(data)
            entry.key = str(key)
            self._entries[str(key)] = entry
        logger.debug("placement ledger loaded: %d entries", len(self._entries))

    def save(self) -> None:
        """Persist atomically, so a crash mid-write cannot corrupt the ledger."""
        with self._lock:
            if not self._dirty:
                return
            payload = {
                "version": LEDGER_VERSION,
                "updated_at": time.time(),
                "bandwidth": self._bandwidth,
                "entries": {key: entry.to_dict() for key, entry in self._entries.items()},
            }
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Not a context manager on the outer call by design: the file must
                # outlive the `with` below so os.replace can move it into place.
                handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
                    "w", encoding="utf-8", dir=str(self.path.parent),
                    prefix=".calibration-", suffix=".tmp", delete=False,
                )
                with handle:
                    json.dump(payload, handle, indent=2)
                os.replace(handle.name, self.path)
                self._dirty = False
            except OSError as exc:
                logger.warning("could not persist placement ledger: %s", exc)

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(self, signature: str, backend_build: str = "") -> LedgerEntry:
        """Return the entry for a device, populated with priors where unmeasured.

        Never returns ``None``: the caller always gets usable numbers, and
        :attr:`LedgerEntry.is_calibrated` says whether to trust them narrowly.
        """
        key = f"{signature}|{backend_build}" if backend_build else signature
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = LedgerEntry(key=key)
            else:
                entry = LedgerEntry.from_dict(entry.to_dict())
            if entry.measured_bandwidth_bps <= 0:
                entry.measured_bandwidth_bps = self._bandwidth.get(signature, 0.0)
        if entry.r_fixed_bytes <= 0:
            entry.r_fixed_bytes = (
                _CUDA_CONTEXT_PRIOR_BYTES if signature.startswith(("cuda", "rocm")) else 0
            )
        if entry.fragmentation <= 0:
            entry.fragmentation = (
                _FRAGMENTATION_PRIOR_EXPANDABLE if _expandable_segments_enabled()
                else _FRAGMENTATION_PRIOR_DEFAULT
            )
        if entry.dispatch_seconds <= 0:
            entry.dispatch_seconds = _DISPATCH_PRIOR_SECONDS
        return entry

    def has_entry(self, signature: str, backend_build: str = "") -> bool:
        key = f"{signature}|{backend_build}" if backend_build else signature
        with self._lock:
            return key in self._entries

    def entries(self) -> dict[str, LedgerEntry]:
        with self._lock:
            return {key: LedgerEntry.from_dict(value.to_dict()) for key, value in self._entries.items()}

    # ── writes ────────────────────────────────────────────────────────────────

    def _mutable(self, key: str) -> LedgerEntry:
        entry = self._entries.get(key)
        if entry is None:
            entry = LedgerEntry(key=key)
            self._entries[key] = entry
        return entry

    def record_bandwidth(self, signature: str, bandwidth_bps: float) -> None:
        """Cache a measured sustained bandwidth against the device signature."""
        if bandwidth_bps <= 0:
            return
        with self._lock:
            self._bandwidth[signature] = float(bandwidth_bps)
            self._dirty = True
        if self.autosave:
            self.save()

    def record_dispatch(
        self,
        signature: str,
        backend_build: str,
        seconds_per_op: float,
        *,
        measured: bool = True,
    ) -> None:
        """Cache the host cost per graph operation for this backend.

        Args:
            signature: Device signature.
            backend_build: Runtime identity — the key that makes this cost meaningful.
            seconds_per_op: The cost.
            measured: Whether the value came from a probe. A measured value replaces a
                stored one; a *prior* never overwrites a measurement, because a
                documented default must not silently displace evidence.
        """
        if seconds_per_op <= 0:
            return
        with self._lock:
            entry = self._mutable(f"{signature}|{backend_build}")
            if entry.dispatch_measured and not measured:
                return
            entry.dispatch_seconds = float(seconds_per_op)
            entry.dispatch_measured = bool(measured)
            entry.last_seen = time.time()
            self._dirty = True
        if self.autosave:
            self.save()

    def reconcile_dispatch(
        self, signature: str, backend_build: str, probed_seconds: float
    ) -> tuple[float, bool]:
        """Check a fresh dispatch probe against the stored value and keep the truth.

        This is the guard for the design's named failure mode.  ``t_dispatch`` belongs
        to the runtime build, and the key — Python, framework, CUDA, Aether version,
        execution mode — cannot capture every change to it.  If a stored value is ever
        wrong the planner does not merely mispredict: it systematically refuses to
        shard models it should, because an inflated dispatch roof makes every wider
        plan look worse.  So the value is *verified* rather than trusted, and a probe
        that disagrees by :data:`DISPATCH_STALE_RATIO` replaces it.

        Returns:
            ``(seconds_per_op, replaced)`` — the value now in force, and whether the
            stored one was discarded as stale.
        """
        if probed_seconds <= 0:
            return 0.0, False
        key = f"{signature}|{backend_build}"
        with self._lock:
            existing = self._entries.get(key)
            stored = existing.dispatch_seconds if existing is not None else 0.0
            was_measured = bool(existing.dispatch_measured) if existing is not None else False
        replaced = False
        if stored > 0 and was_measured:
            ratio = max(stored / probed_seconds, probed_seconds / stored)
            if ratio >= DISPATCH_STALE_RATIO:
                replaced = True
                logger.warning(
                    "dispatch cost for %s under %s was %.1f us/op but probes at "
                    "%.1f us/op (%.1fx); the stored value belongs to a different "
                    "runtime and is being replaced. A stale dispatch roof makes the "
                    "planner refuse to shard models it should.",
                    signature, backend_build, stored * 1e6, probed_seconds * 1e6, ratio,
                )
            else:
                return stored, False
        self.record_dispatch(signature, backend_build, probed_seconds, measured=True)
        return probed_seconds, replaced

    def record_notes(
        self, signature: str, backend_build: str, notes: "dict[str, Any]"
    ) -> None:
        """Merge free-form calibration into an entry, one namespace at a time.

        Used by calibrations that belong to the same (device, backend) key but are not
        part of the placement schema — the decode kernel-strategy winners in
        :mod:`aether.runtime.kernel_strategy` are the first. Nested dictionaries are
        merged rather than replaced, so two namespaces, or two shape classes inside
        one namespace, cannot overwrite each other's measurements.
        """
        if not notes:
            return
        with self._lock:
            entry = self._mutable(f"{signature}|{backend_build}")
            for namespace, value in notes.items():
                existing = entry.notes.get(namespace)
                if isinstance(existing, dict) and isinstance(value, dict):
                    merged = dict(existing)
                    merged.update(value)
                    entry.notes[namespace] = merged
                else:
                    entry.notes[namespace] = value
            entry.last_seen = time.time()
            self._dirty = True
        if self.autosave:
            self.save()

    def observe_tp_group(
        self,
        signature: str,
        backend_build: str,
        *,
        theta_ratio: float,
        predicted_seconds: float,
        observed_seconds: float,
    ) -> None:
        """Fold one heterogeneous tensor-parallel group's realised time into Law I.

        Over runs this brackets the crossover ratio from both sides, which is what
        turns Law I's tolerance from a derivation into a measurement.
        """
        if theta_ratio < 1.0 or predicted_seconds <= 0 or observed_seconds <= 0:
            return
        with self._lock:
            entry = self._mutable(f"{signature}|{backend_build}")
            entry.observe_tp_group(theta_ratio, predicted_seconds, observed_seconds)
            entry.last_seen = time.time()
            self._dirty = True
            crossover = entry.tp_crossover_ratio
            samples = entry.tp_samples
        if self.autosave:
            self.save()
        logger.debug(
            "Law I crossover for %s: %.2fx from %d group observation(s)",
            signature, crossover, samples,
        )

    def observe_execution(
        self,
        signature: str,
        backend_build: str,
        *,
        predicted_transient_bytes: int,
        observed_peak_bytes: int,
        predicted_seconds: float = 0.0,
        observed_seconds: float = 0.0,
        r_fixed_bytes: int | None = None,
        fragmentation: float | None = None,
        achieved_flops: float | None = None,
        dispatch_seconds: float | None = None,
    ) -> LedgerEntry:
        """Fold one real execution into the ledger.

        This is the whole feedback path. Its success metric is that the residual σ
        shrinks over repeated runs on one host — directly observable, and therefore
        a real test of whether the calibration loop does anything.
        """
        with self._lock:
            entry = self._mutable(f"{signature}|{backend_build}")
            entry.observe_residual(predicted_transient_bytes, observed_peak_bytes)
            if predicted_seconds > 0 and observed_seconds > 0:
                entry.observe_latency(predicted_seconds, observed_seconds)
            if r_fixed_bytes is not None and r_fixed_bytes >= 0:
                # Exponential smoothing: the context cost is stable, but a single
                # reading can be perturbed by whatever else touched the device.
                previous = entry.r_fixed_bytes or r_fixed_bytes
                entry.r_fixed_bytes = int(0.7 * previous + 0.3 * r_fixed_bytes)
            if fragmentation is not None and fragmentation > 0:
                previous_fragmentation = entry.fragmentation or fragmentation
                entry.fragmentation = 0.7 * previous_fragmentation + 0.3 * fragmentation
            if achieved_flops is not None and achieved_flops > 0:
                entry.achieved_flops = achieved_flops
            if dispatch_seconds is not None and dispatch_seconds > 0:
                entry.dispatch_seconds = dispatch_seconds
                entry.dispatch_measured = True
            entry.runs += 1
            entry.last_seen = time.time()
            self._dirty = True
            snapshot = LedgerEntry.from_dict(entry.to_dict())
        if self.autosave:
            self.save()
        logger.debug(
            "ledger updated %s: runs=%d sigma=%.1f MiB",
            snapshot.key, snapshot.runs, snapshot.residual_sigma / 1024 ** 2,
        )
        return snapshot

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bandwidth.clear()
            self._dirty = True
        if self.autosave:
            self.save()
