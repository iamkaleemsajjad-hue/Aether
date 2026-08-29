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

__all__ = ["LEDGER_VERSION", "LedgerEntry", "CalibrationLedger"]

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
"""With no residual history, σ is taken as this fraction of the predicted peak."""


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
        width used by the σ-gated tie-break: a wider plan must beat a narrower one by
        more than this, or the planner is chasing its own noise.
        """
        if self.latency_samples < 2:
            return 0.0
        return math.sqrt(max(0.0, self.latency_m2 / (self.latency_samples - 1)))

    @property
    def is_calibrated(self) -> bool:
        """Whether this entry has enough evidence to narrow the safety margin."""
        return self.residual_samples >= 5 and self.r_fixed_bytes > 0

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

    def record_dispatch(self, signature: str, backend_build: str, seconds_per_op: float) -> None:
        """Cache the measured host cost per graph operation for this backend."""
        if seconds_per_op <= 0:
            return
        with self._lock:
            entry = self._mutable(f"{signature}|{backend_build}")
            entry.dispatch_seconds = float(seconds_per_op)
            entry.last_seen = time.time()
            self._dirty = True
        if self.autosave:
            self.save()

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
