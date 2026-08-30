"""The status vocabulary, and what each value is allowed to mean.

A benchmark's credibility rests on the difference between "slow" and "not
measured". These constants are the only way this suite expresses the second, and
:func:`is_measured` is the only predicate any aggregation may use to decide
whether a number exists — so a missing result can never be arithmetic'd into a
zero.
"""

from __future__ import annotations

from typing import Any

#: The engine ran the configuration and the numbers below it are real.
MEASURED = "MEASURED"
#: The engine's package is not importable in this environment.
NOT_INSTALLED = "NOT_INSTALLED"
#: The engine is installed but cannot execute this model or this configuration
#: (architecture unsupported, batch width unsupported, representation missing).
NOT_SUPPORTED = "NOT_SUPPORTED"
#: The engine is fundamentally inapplicable to the detected hardware — a CUDA-only
#: compiler on a host with no NVIDIA device, for instance. Not a failure.
NOT_APPLICABLE = "NOT_APPLICABLE"
#: The engine attempted the configuration and raised.
FAILED = "FAILED"
#: The engine ran out of memory on this configuration.
OOM = "OOM"
#: Deliberately not run — excluded by a flag, or by a budget guard.
SKIPPED = "SKIPPED"

ALL_STATUSES: tuple[str, ...] = (
    MEASURED, NOT_INSTALLED, NOT_SUPPORTED, NOT_APPLICABLE, FAILED, OOM, SKIPPED,
)

#: Per-batch support vocabulary required by the benchmark charter, kept distinct
#: from the run-level statuses above so a batch table can be read on its own.
BATCH_SUPPORTED = "SUPPORTED"
BATCH_OOM = "OOM"
BATCH_UNSUPPORTED = "UNSUPPORTED"


def is_measured(record: Any) -> bool:
    """Whether ``record`` carries a real measurement.

    Accepts a status string or any mapping with a ``status`` key, because cells
    arrive from several layers of the pipeline and every one of them must ask
    this question the same way.
    """
    if isinstance(record, str):
        return record == MEASURED
    if isinstance(record, dict):
        return record.get("status") == MEASURED
    return False


def from_exception(exc: BaseException) -> tuple[str, str]:
    """Classify an exception into ``(status, message)``.

    Out-of-memory is separated from a generic failure because the two say
    different things about an engine: one is a capacity limit at this batch
    width, the other is a defect or an incompatibility.
    """
    from benchmark.backends import UnsupportedConfiguration

    text = f"{type(exc).__name__}: {exc}"
    lowered = str(exc).lower()
    if isinstance(exc, UnsupportedConfiguration):
        return NOT_SUPPORTED, text
    # "oom" has to be a whole word. As a bare substring it also matches "boom",
    # "zoom" and "room", which would file ordinary defects as capacity limits and
    # so excuse an engine that is actually broken.
    words = set("".join(c if c.isalnum() else " " for c in lowered).split())
    if isinstance(exc, MemoryError) or "out of memory" in lowered or "oom" in words:
        return OOM, text
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return NOT_INSTALLED, text
    return FAILED, text


#: Maps the legacy per-cell status strings produced by :mod:`benchmark.runner`
#: onto this vocabulary, so the existing measurement primitives can be reused
#: verbatim instead of being reimplemented for the sake of a label.
RUNNER_STATUS_MAP: dict[str, str] = {
    "ok": MEASURED,
    "unsupported": NOT_SUPPORTED,
    "oom": OOM,
    "error": FAILED,
    "load-failed": FAILED,
}


def from_runner(record: dict[str, Any]) -> str:
    """Translate a :mod:`benchmark.runner` record's status into this vocabulary."""
    return RUNNER_STATUS_MAP.get(str(record.get("status")), FAILED)
