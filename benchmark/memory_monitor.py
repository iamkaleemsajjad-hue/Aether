"""Host memory observation for a benchmark phase.

Reports process resident set size from the OS rather than any Python-level
accounting, because the interpreter's view excludes the allocator arenas,
CUDA context, and library buffers that dominate a model load.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001 - absence is reported, not raised
        return None


def _available_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


class HostMemoryMonitor:
    """Sample process RSS and CPU utilization on a background thread.

    Sampling is done off the measured thread so it does not serialize with the
    work being timed, and the interval is recorded in the result so the reader
    can judge how much of a transient peak the sampler could have missed.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = float(interval_s)
        self._samples: list[tuple[float, int, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None

    def __enter__(self) -> "HostMemoryMonitor":
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            # Prime the CPU-percent baseline; the first call always returns 0.0.
            self._process.cpu_percent(None)
        except Exception:  # noqa: BLE001
            self._process = None
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._loop, name="bench-host-mem", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._process is not None:
                try:
                    rss = int(self._process.memory_info().rss)
                    cpu = float(self._process.cpu_percent(None))
                    self._samples.append((time.perf_counter(), rss, cpu))
                except Exception:  # noqa: BLE001 - process may be shutting down
                    pass
            self._stop.wait(self.interval_s)

    def report(self) -> dict[str, Any]:
        """Summarize the sampled window."""
        if not self._samples:
            return {
                "sampled": False,
                "sample_interval_s": self.interval_s,
                "rss_bytes": _rss_bytes(),
            }
        rss_values = [rss for _, rss, _ in self._samples]
        # Drop the priming sample: psutil's first cpu_percent after the baseline
        # call covers a near-zero window and reads as 0.0.
        cpu_values = [cpu for _, _, cpu in self._samples[1:]] or [
            cpu for _, _, cpu in self._samples
        ]
        cores = os.cpu_count() or 1
        return {
            "sampled": True,
            "samples": len(self._samples),
            "sample_interval_s": self.interval_s,
            "rss_peak_bytes": max(rss_values),
            "rss_mean_bytes": sum(rss_values) / len(rss_values),
            "rss_final_bytes": rss_values[-1],
            "cpu_percent_mean": sum(cpu_values) / len(cpu_values),
            "cpu_percent_peak": max(cpu_values),
            # psutil reports percent-of-one-core, so this normalizes to the box.
            "cpu_cores_mean": sum(cpu_values) / len(cpu_values) / 100.0,
            "logical_cores": cores,
            "threads": _thread_count(),
        }


def _thread_count() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).num_threads())
    except Exception:  # noqa: BLE001
        return None


def snapshot() -> dict[str, Any]:
    """A single point-in-time host memory reading."""
    return {
        "rss_bytes": _rss_bytes(),
        "system_available_bytes": _available_bytes(),
        "threads": _thread_count(),
    }
