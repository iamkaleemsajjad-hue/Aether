"""
Profiling utilities for operation-level timing and memory tracking.

Provides a lightweight profiler that records per-operation latency and memory
usage. The profiler is designed to be low overhead when disabled and to
integrate cleanly with the executor and runtime metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OpProfile:
    """Profile record for a single operation execution."""

    op_name: str
    duration_ms: float = 0.0
    memory_bytes: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "duration_ms": self.duration_ms,
            "memory_bytes": self.memory_bytes,
            "attributes": self.attributes,
            "timestamp": self.timestamp,
        }


@dataclass
class KernelProfile:
    """Profile record for a kernel execution."""

    kernel_name: str
    target: str
    duration_ms: float = 0.0
    grid: tuple[int, ...] = field(default_factory=tuple)
    block: tuple[int, ...] = field(default_factory=tuple)
    shared_memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "target": self.target,
            "duration_ms": self.duration_ms,
            "grid": list(self.grid),
            "block": list(self.block),
            "shared_memory_bytes": self.shared_memory_bytes,
        }


class Profiler:
    """Lightweight operation profiler.

    When enabled, records timing and memory estimates for each operation. When
    disabled, all recording calls are no-ops.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._records: list[OpProfile] = []
        self._kernel_records: list[KernelProfile] = []
        self._last: OpProfile | None = None

    @property
    def last(self) -> OpProfile | None:
        """Return the most recent profile record."""
        return self._last

    def record(self, profile: OpProfile) -> None:
        """Record an operation profile."""
        if not self.enabled:
            return
        self._records.append(profile)
        self._last = profile

    def record_kernel(self, profile: KernelProfile) -> None:
        """Record a kernel profile."""
        if not self.enabled:
            return
        self._kernel_records.append(profile)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all recorded profiles."""
        if not self._records:
            return {"enabled": self.enabled, "count": 0, "total_duration_ms": 0.0}
        by_op: dict[str, list[float]] = {}
        by_mem: dict[str, list[int]] = {}
        for record in self._records:
            by_op.setdefault(record.op_name, []).append(record.duration_ms)
            by_mem.setdefault(record.op_name, []).append(record.memory_bytes)
        return {
            "enabled": self.enabled,
            "count": len(self._records),
            "total_duration_ms": sum(r.duration_ms for r in self._records),
            "total_memory_bytes": sum(r.memory_bytes for r in self._records),
            "op_summary": {
                name: {
                    "calls": len(durations),
                    "total_ms": sum(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "total_memory_bytes": sum(by_mem[name]),
                }
                for name, durations in by_op.items()
            },
        }

    def reset(self) -> None:
        """Clear all recorded profiles."""
        self._records.clear()
        self._kernel_records.clear()
        self._last = None

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"Profiler(enabled={self.enabled}, records={len(self._records)})"


class Timer:
    """Context manager for timing blocks of code."""

    def __init__(self, name: str = "block", profiler: Profiler | None = None) -> None:
        self.name = name
        self.profiler = profiler
        self.start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.profiler is not None and self.profiler.enabled:
            self.profiler.record(
                OpProfile(
                    op_name=self.name,
                    duration_ms=self.elapsed_ms,
                )
            )

    def __repr__(self) -> str:
        return f"Timer({self.name}, elapsed_ms={self.elapsed_ms:.2f})"
