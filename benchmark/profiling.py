"""Kernel-level diagnostics — deliberately separate from the performance numbers.

Profiling perturbs what it measures: the PyTorch profiler adds per-operator
bookkeeping, and that cost falls hardest on exactly the workload this benchmark
studies, a launch-bound decode loop.  So nothing here feeds the official
throughput results.  Its job is to answer *why* a difference exists, using two
independent instruments:

* ``count_dispatches`` records every ATen call through a dispatch mode.  It is
  deterministic, hardware-independent, and separates real kernels from
  metadata-only view operations, which launch nothing on a GPU.
* ``profile_step`` uses ``torch.profiler`` to attribute wall-clock and device
  time to individual kernels, including whether a fused attention kernel or a
  CUDA graph was used.
"""

from __future__ import annotations

import collections
from typing import Any, Callable

#: Operations that only rewrite tensor metadata.  They cost a Python dispatch but
#: launch no GPU kernel, so counting them against either runtime would overstate
#: its device work.  They are reported separately instead.
VIEW_OPS = frozenset({
    "aten.slice.Tensor", "aten.view.default", "aten.reshape.default",
    "aten._unsafe_view.default", "aten.transpose.int", "aten.permute.default",
    "aten.unsqueeze.default", "aten.squeeze.default", "aten.squeeze.dim",
    "aten.expand.default", "aten.detach.default", "aten.alias.default",
    "aten.select.int", "aten.split.Tensor", "aten.split_with_sizes.default",
    "aten.flatten.using_ints", "aten.contiguous.default", "aten.t.default",
    "aten.chunk.default", "aten.narrow.default", "aten.as_strided.default",
})

#: Coarse buckets so the two runtimes can be compared by *category* of work
#: rather than by kernel name, which differs between them.
CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gemm", ("mm", "addmm", "bmm", "linear", "matmul", "baddbmm", "gemv")),
    ("attention", ("scaled_dot_product", "flash", "efficient_attention", "softmax")),
    ("normalization", ("layer_norm", "rms_norm", "native_layer_norm", "group_norm")),
    ("activation", ("silu", "gelu", "relu", "sigmoid", "tanh")),
    ("elementwise", ("mul", "add", "sub", "div", "neg", "pow", "rsqrt", "mean", "cat", "stack")),
    ("memory", ("copy_", "memcpy", "empty", "zeros", "clone", "index_select", "index_put")),
    ("sync", ("_local_scalar_dense", "item", "synchronize")),
)


def categorize(name: str) -> str:
    lowered = name.lower()
    for category, markers in CATEGORIES:
        if any(marker in lowered for marker in markers):
            return category
    return "other"


class DispatchCounter:
    """Count every ATen call made inside the context."""

    def __init__(self) -> None:
        self.by_op: collections.Counter[str] = collections.Counter()
        self._mode: Any = None

    def __enter__(self) -> "DispatchCounter":
        from torch.utils._python_dispatch import TorchDispatchMode

        counter = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
                counter.by_op[str(func)] += 1
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exc)

    def report(self, steps: int, layers: int | None = None) -> dict[str, Any]:
        kernels = {op: n for op, n in self.by_op.items() if op not in VIEW_OPS}
        views = {op: n for op, n in self.by_op.items() if op in VIEW_OPS}
        kernel_total = sum(kernels.values())
        by_category: collections.Counter[str] = collections.Counter()
        for op, count in kernels.items():
            by_category[categorize(op)] += count
        return {
            "steps": steps,
            "layers": layers,
            "kernel_calls_total": kernel_total,
            "kernel_calls_per_step": kernel_total / max(steps, 1),
            "kernel_calls_per_layer_per_step": (
                kernel_total / max(steps, 1) / layers if layers else None
            ),
            "view_calls_per_step": sum(views.values()) / max(steps, 1),
            "by_category_per_step": {
                name: count / max(steps, 1) for name, count in by_category.most_common()
            },
            "top_kernels_per_step": [
                {"op": op, "per_step": count / max(steps, 1), "category": categorize(op)}
                for op, count in sorted(kernels.items(), key=lambda item: -item[1])[:15]
            ],
        }


def count_dispatches(step: Callable[[], Any], *, steps: int, layers: int | None) -> dict[str, Any]:
    """Warm up, then count the ATen calls of ``steps`` invocations."""
    step()
    counter = DispatchCounter()
    with counter:
        for _ in range(steps):
            step()
    return counter.report(steps, layers)


def profile_step(step: Callable[[], Any], *, steps: int = 3) -> dict[str, Any]:
    """Attribute time to individual kernels with the PyTorch profiler.

    Reported alongside the summed device time so a reader can see how much of the
    wall clock the device was actually busy for — the gap is host-side cost, which
    is the quantity that distinguishes these two runtimes on small models.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    on_cuda = torch.cuda.is_available()
    if on_cuda:
        activities.append(ProfilerActivity.CUDA)

    step()  # warm up outside the profiler
    if on_cuda:
        torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(steps):
            step()
        if on_cuda:
            torch.cuda.synchronize()

    events = prof.key_averages()

    def device_time(event: Any) -> float:
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0)
        return float(value or 0.0)

    total_device_us = sum(device_time(event) for event in events)
    total_cpu_us = sum(float(event.self_cpu_time_total or 0.0) for event in events)
    ranked = sorted(events, key=device_time, reverse=True) if on_cuda else sorted(
        events, key=lambda e: -(e.self_cpu_time_total or 0.0)
    )
    return {
        "steps": steps,
        "device_profiled": on_cuda,
        "summed_device_time_ms": total_device_us / 1000.0,
        "summed_self_cpu_time_ms": total_cpu_us / 1000.0,
        "device_time_per_step_ms": total_device_us / 1000.0 / max(steps, 1),
        "kernels": [
            {
                "name": event.key[:120],
                "count": int(event.count),
                "self_device_time_ms": device_time(event) / 1000.0,
                "self_cpu_time_ms": float(event.self_cpu_time_total or 0.0) / 1000.0,
                "share_of_device_time": (
                    device_time(event) / total_device_us if total_device_us else None
                ),
                "category": categorize(event.key),
            }
            for event in ranked[:25]
        ],
        "fused_attention_kernels": sorted({
            event.key[:120] for event in events
            if any(marker in event.key.lower() for marker in ("flash", "fmha", "efficient_attention"))
        }),
        "cuda_graph_kernels": sorted({
            event.key[:120] for event in events if "graph" in event.key.lower()
        }),
        "synchronizations": sum(
            int(event.count) for event in events
            if any(marker in event.key for marker in
                   ("_local_scalar_dense", "cudaStreamSynchronize", "cudaDeviceSynchronize"))
        ),
        "memcpy_calls": sum(
            int(event.count) for event in events if "memcpy" in event.key.lower()
        ),
    }
