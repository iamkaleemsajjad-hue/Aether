"""Accelerator memory accounting and device telemetry.

Two distinct things live here and must not be confused:

* **allocator accounting** (``torch.cuda.memory_*``) is exact, free, and is what
  the memory comparison is based on;
* **NVML telemetry** (utilization, power, temperature, clocks) is sampled by a
  background thread and is therefore approximate — a sampler can miss a
  transient, and polling costs a little host time.

Because of the second point, telemetry is collected in dedicated monitored
iterations and never during the iterations whose latency is reported as the
official result.
"""

from __future__ import annotations

import threading
from typing import Any


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def reset_peak_stats() -> None:
    """Zero the allocator's peak counters so the next phase measures itself."""
    if not cuda_available():
        return
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    torch.cuda.synchronize()


def empty_cache() -> None:
    """Release cached blocks so a 'before' reading reflects real occupancy."""
    if not cuda_available():
        return
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def memory_snapshot() -> dict[str, Any]:
    """Per-device allocator state plus the driver's own free/total view."""
    if not cuda_available():
        return {"available": False}
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({
            "index": index,
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            "driver_free_bytes": int(free),
            "driver_total_bytes": int(total),
        })
    return {"available": True, "devices": devices}


class GPUTelemetryMonitor:
    """Sample NVML counters on a background thread.

    The sampling period is a deliberate trade-off and is recorded in the result.
    A shorter period resolves brief peaks better but takes more host time away
    from the work being measured, which is exactly the resource a launch-bound
    decode loop is contending for.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = float(interval_s)
        self._handles: list[Any] = []
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None

    def _open(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            return True
        except Exception:  # noqa: BLE001 - NVML is optional
            self._nvml = None
            return False

    def __enter__(self) -> "GPUTelemetryMonitor":
        self._samples = []
        self._stop.clear()
        if self._open():
            self._thread = threading.Thread(target=self._loop, name="bench-gpu", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def _read_once(self) -> list[dict[str, Any]]:
        nvml = self._nvml
        reading = []
        for index, handle in enumerate(self._handles):
            entry: dict[str, Any] = {"index": index}
            for key, call in (
                ("gpu_util_percent", lambda h: nvml.nvmlDeviceGetUtilizationRates(h).gpu),
                ("mem_util_percent", lambda h: nvml.nvmlDeviceGetUtilizationRates(h).memory),
                ("temperature_c", lambda h: nvml.nvmlDeviceGetTemperature(h, 0)),
                ("power_watts", lambda h: nvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
                ("sm_clock_mhz", lambda h: nvml.nvmlDeviceGetClockInfo(h, 1)),
                ("mem_clock_mhz", lambda h: nvml.nvmlDeviceGetClockInfo(h, 2)),
            ):
                try:
                    entry[key] = call(handle)
                except Exception:  # noqa: BLE001 - unsupported counters are common
                    entry[key] = None
            reading.append(entry)
        return reading

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.extend(self._read_once())
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval_s)

    def report(self) -> dict[str, Any]:
        if not self._samples:
            return {"sampled": False, "sample_interval_s": self.interval_s,
                    "reason": "NVML unavailable" if self._nvml is None else "no samples"}
        by_device: dict[int, list[dict[str, Any]]] = {}
        for sample in self._samples:
            by_device.setdefault(int(sample["index"]), []).append(sample)
        devices = []
        for index, entries in sorted(by_device.items()):
            summary: dict[str, Any] = {"index": index, "samples": len(entries)}
            for key in ("gpu_util_percent", "mem_util_percent", "temperature_c",
                        "power_watts", "sm_clock_mhz", "mem_clock_mhz"):
                values = [e[key] for e in entries if e.get(key) is not None]
                if values:
                    summary[f"{key}_mean"] = sum(values) / len(values)
                    summary[f"{key}_max"] = max(values)
                    summary[f"{key}_min"] = min(values)
            devices.append(summary)
        return {"sampled": True, "sample_interval_s": self.interval_s, "devices": devices}


def temperature_snapshot() -> dict[str, Any]:
    """Instantaneous per-device temperature, for thermal bookkeeping."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            readings = {}
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                readings[index] = pynvml.nvmlDeviceGetTemperature(handle, 0)
            return {"available": True, "temperature_c": readings}
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return {"available": False, "temperature_c": {}}
