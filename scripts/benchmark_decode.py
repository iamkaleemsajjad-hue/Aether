"""Decode-throughput benchmark for the Torch AEG executor.

Diagnostic only.  Usage:
    python scripts/_decode_bench.py <aeg_path> [device] [tokens]
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    aeg = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    profile = os.environ.get("AETHER_BENCH_PROFILE", "") == "1"

    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = TorchAEGEngine(load_engine_from_path(aeg), device)
    prompt = np.array([9707, 0, 28338, 6133, 323, 10339], dtype=np.int64)

    # Warm up kernels, autotuning, and any lazy table construction.
    engine.generate(prompt, max_tokens=4, temperature=0.0)

    def run() -> list[int]:
        return engine.generate(prompt, max_tokens=tokens, temperature=0.0)

    start = time.perf_counter()
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        out = run()
        profiler.disable()
    else:
        out = run()
    elapsed = time.perf_counter() - start

    print(f"device={device} tokens={len(out)} elapsed={elapsed:.3f}s "
          f"throughput={len(out)/elapsed:.2f} tok/s "
          f"per_token={elapsed/max(len(out),1)*1000:.2f} ms")
    if profile:
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(28)
        print(buffer.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
