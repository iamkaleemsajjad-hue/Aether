"""Profile a decode step on the current device and attribute the time.

Small-model decode should be memory-bound: a 350M model in FP16 reads ~700 MB of
weights per token, which a V100 streams in well under a millisecond.  When
measured throughput is far below that, the time is going somewhere other than
arithmetic — launch overhead, host/device synchronization, or a slow kernel
choice.  This separates those cases.

    python scripts/profile_decode.py <aeg_path> [device] [tokens]

Report:
  * total wall clock and per-token latency;
  * summed GPU kernel time vs wall clock — the gap is host-side stall;
  * the ops with the largest self CUDA time and the largest self CPU time;
  * a count of host/device synchronizations, which serialize the pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    aeg = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    import torch
    from torch.profiler import ProfilerActivity, profile

    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(load_engine_from_path(aeg), device)
    prompt = np.array([9707, 0, 28338, 6133, 323, 10339], dtype=np.int64)
    is_cuda = torch.device(device).type == "cuda"

    print(f"device={device} dtype={engine.compute_dtype} layers={engine.num_layers}")
    if is_cuda:
        print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
        print(f"sdpa backends: flash={torch.backends.cuda.flash_sdp_enabled()} "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
              f"math={torch.backends.cuda.math_sdp_enabled()}")

    # Warm up: kernel autotuning, lazy tables, allocator growth.
    engine.generate(prompt, max_tokens=4, temperature=0.0)
    if is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    out = engine.generate(prompt, max_tokens=tokens, temperature=0.0)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"\nwall clock: {elapsed:.3f}s for {len(out)} tokens "
        f"= {len(out) / elapsed:.2f} tok/s ({elapsed / len(out) * 1000:.2f} ms/token)"
    )

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if is_cuda else [])
    with profile(activities=activities, record_shapes=False) as prof:
        engine.generate(prompt, max_tokens=tokens, temperature=0.0)
        if is_cuda:
            torch.cuda.synchronize()

    events = prof.key_averages()
    if is_cuda:
        gpu_us = sum(getattr(e, "self_device_time_total", 0) or 0 for e in events)
        if gpu_us == 0:  # older PyTorch attribute name
            gpu_us = sum(getattr(e, "self_cuda_time_total", 0) or 0 for e in events)
        print(f"summed GPU kernel time: {gpu_us / 1e6:.3f}s "
              f"({gpu_us / 1e3 / max(len(out), 1):.2f} ms/token)")
        print(f"host-side stall (wall - GPU): {max(elapsed - gpu_us / 1e6, 0.0):.3f}s "
              "— if this dominates, the decode is bound by the host, not the GPU")

    print("\ntop ops by self CPU time:")
    for event in sorted(events, key=lambda e: -(e.self_cpu_time_total or 0))[:14]:
        print(f"  {(event.self_cpu_time_total or 0) / 1e3:9.2f} ms  "
              f"{event.count:6d}x  {event.key}")

    if is_cuda:
        def device_time(event: object) -> float:
            value = getattr(event, "self_device_time_total", None)
            if value is None:
                value = getattr(event, "self_cuda_time_total", 0)
            return value or 0.0

        print("\ntop ops by self GPU time:")
        for event in sorted(events, key=lambda e: -device_time(e))[:14]:
            print(f"  {device_time(event) / 1e3:9.2f} ms  {event.count:6d}x  {event.key}")

    syncs = [
        event for event in events
        if any(marker in event.key for marker in ("_local_scalar_dense", "item", "Synchronize", "cudaMemcpy"))
    ]
    if syncs:
        print("\nhost/device synchronizations (each one serializes the pipeline):")
        for event in syncs:
            print(f"  {event.count:6d}x  {event.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
