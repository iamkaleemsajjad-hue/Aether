"""Statistics for repeated measurements, and the timing primitive both backends use.

Two rules this module exists to enforce:

* a sample of one is not a measurement, so every timed quantity is summarized
  with dispersion alongside its centre;
* CUDA work is asynchronous, so a wall-clock reading is only meaningful after
  the device has been synchronized.  ``timed`` does that on both edges.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Any, Iterator


def synchronize() -> None:
    """Block until all queued accelerator work has retired."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def timed(store: list[float]) -> Iterator[None]:
    """Append the elapsed seconds of the enclosed block to ``store``.

    Synchronizes before starting the clock so unrelated queued work is not
    attributed to this block, and again before stopping it so asynchronous
    kernels launched inside the block are actually accounted for.
    """
    synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize()
        store.append(time.perf_counter() - start)


def summarize(samples: list[float]) -> dict[str, Any]:
    """Return the standard descriptive statistics for a list of samples."""
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    n = len(ordered)

    def percentile(fraction: float) -> float:
        # Nearest-rank percentile: unambiguous and does not invent values
        # between samples, which matters at the small n a GPU budget allows.
        index = min(n - 1, max(0, int(round(fraction * (n - 1)))))
        return ordered[index]

    return {
        "n": n,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "stdev": statistics.stdev(ordered) if n > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95) if n >= 3 else None,
        # A nearest-rank p99 over a handful of samples is just the maximum, so it
        # is only reported once there are enough samples for the two to differ.
        "p99": percentile(0.99) if n >= 10 else None,
        "coefficient_of_variation": (
            statistics.stdev(ordered) / statistics.fmean(ordered)
            if n > 1 and statistics.fmean(ordered) > 0 else 0.0
        ),
    }


def throughput_samples(latencies: list[float], tokens: int) -> list[float]:
    """Convert per-iteration latencies into per-iteration tokens/second.

    Averaging latency and then dividing is not the same as averaging
    throughput; keeping the per-iteration ratio lets the report state the
    dispersion of the quantity it actually claims.
    """
    return [tokens / latency for latency in latencies if latency > 0]


def speedup(aether: dict[str, Any], reference: dict[str, Any], key: str = "median") -> dict[str, Any]:
    """Express an Aether measurement relative to the Transformers baseline.

    Reported as both a ratio and a percentage, with the two underlying values
    kept alongside so the reader never has to trust the ratio alone.
    """
    a = aether.get(key)
    b = reference.get(key)
    if not a or not b:
        return {"ratio": None, "percent": None, "aether": a, "transformers": b}
    return {
        "ratio": a / b,
        "percent": (a / b - 1.0) * 100.0,
        "aether": a,
        "transformers": b,
    }
