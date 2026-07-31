#!/usr/bin/env python3
"""
profile_memory.py — Memory profiling for Aether model compilation and inference.

Tracks peak RSS memory usage at each stage of the compile pipeline and
during inference, helping identify memory bottlenecks before deploying
to memory-constrained hardware.

Usage:
    # Profile compilation of a model
    python scripts/profile_memory.py --compile Qwen/Qwen3-0.6B

    # Profile inference on a compiled package
    python scripts/profile_memory.py --infer ./my-model.aeg

    # Profile both
    python scripts/profile_memory.py --compile Qwen/Qwen3-0.6B --infer-after

    # Save trace to JSON
    python scripts/profile_memory.py --compile Qwen/Qwen3-0.6B --output mem_trace.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
DIM    = "\033[2m"

try:
    import psutil
    _PROC = psutil.Process()
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class MemorySnapshot:
    label: str
    rss_mb: float
    vms_mb: float
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class MemoryTrace:
    snapshots: list[MemorySnapshot] = field(default_factory=list)

    def capture(self, label: str) -> MemorySnapshot:
        if HAS_PSUTIL:
            info = _PROC.memory_info()
            snap = MemorySnapshot(
                label=label,
                rss_mb=info.rss / (1024 ** 2),
                vms_mb=info.vms / (1024 ** 2),
            )
        else:
            import resource
            snap = MemorySnapshot(
                label=label,
                rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                vms_mb=0.0,
            )
        self.snapshots.append(snap)
        return snap

    @property
    def peak_rss_mb(self) -> float:
        return max((s.rss_mb for s in self.snapshots), default=0.0)

    def delta_mb(self, from_label: str, to_label: str) -> float:
        frm = next((s for s in self.snapshots if s.label == from_label), None)
        to  = next((s for s in self.snapshots if s.label == to_label), None)
        if frm and to:
            return to.rss_mb - frm.rss_mb
        return 0.0


class MemorySampler:
    """Background thread that samples memory every interval_s seconds."""

    def __init__(self, trace: MemoryTrace, interval_s: float = 0.5) -> None:
        self.trace = trace
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(self.interval):
            self.trace.capture(f"sample_{i}")
            i += 1


def _print_trace(trace: MemoryTrace) -> None:
    if not trace.snapshots:
        print(f"  {YELLOW}No memory snapshots collected{RESET}")
        return

    baseline = trace.snapshots[0].rss_mb
    print(f"\n{BOLD}Memory Trace{RESET}")
    print(f"  {'Label':<40} {'RSS (MB)':>10} {'Delta':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for snap in trace.snapshots:
        if snap.label.startswith("sample_"):
            continue  # omit background samples from table
        delta = snap.rss_mb - baseline
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"  {snap.label:<40} {snap.rss_mb:>10.1f} {delta_str:>10}")

    print(f"\n  Peak RSS : {trace.peak_rss_mb:.1f} MB")
    print(f"  Net Δ    : {trace.peak_rss_mb - baseline:+.1f} MB")


def profile_compile(model: str, trace: MemoryTrace, args: argparse.Namespace):
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig

    print(f"\n{BOLD}{CYAN}Profiling compilation: {model}{RESET}")
    config = CompilerConfig(optimization_level=1, targets=["cpu_avx512"], overwrite=True)
    compiler = Compiler(config=config)

    trace.capture("start")
    print(f"  {DIM}·{RESET} Stage 1: Ingestion...")
    t0 = time.perf_counter()

    output_path = None
    if args.output_aeg:
        output_path = Path(args.output_aeg)

    package = compiler.compile(model, output_path=output_path, targets=["cpu_avx512"])
    trace.capture("after_compile")

    elapsed = time.perf_counter() - t0
    print(f"  {GREEN}✓{RESET} Compilation complete in {elapsed:.2f}s")
    print(f"  {GREEN}✓{RESET} Package: {package.root}")
    return package


def profile_infer(package_path: Path, trace: MemoryTrace, args: argparse.Namespace) -> None:
    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import load_engine_from_package

    import numpy as np

    print(f"\n{BOLD}{CYAN}Profiling inference: {package_path}{RESET}")

    trace.capture("before_load")
    pkg = AEGPackage(package_path)
    pkg.load()
    trace.capture("after_load")

    engine = load_engine_from_package(pkg)
    trace.capture("after_engine_build")

    prompt = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    logits, cache = engine.forward(prompt)
    trace.capture("after_prefill")

    for _ in range(10):
        token = int(np.argmax(logits[-1]))
        logits, cache = engine.forward(np.array([token], dtype=np.int64), cache)
    trace.capture("after_10_decode_steps")

    print(f"  {GREEN}✓{RESET} Inference complete, KV cache length: {cache.length}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory profiling for Aether Runtime")
    parser.add_argument("--compile", metavar="MODEL",
                        help="Model to compile and profile")
    parser.add_argument("--infer", metavar="AEG_PATH",
                        help="Path to .aeg package to profile inference on")
    parser.add_argument("--infer-after", action="store_true",
                        help="Run inference on the compiled package after --compile")
    parser.add_argument("--output-aeg", metavar="PATH",
                        help="Save compiled .aeg to this path")
    parser.add_argument("--output", metavar="JSON",
                        help="Save memory trace to JSON file")
    parser.add_argument("--sample-interval", type=float, default=0.5,
                        help="Background sampling interval in seconds (default: 0.5)")
    args = parser.parse_args()

    if not args.compile and not args.infer:
        parser.print_help()
        return 1

    if not HAS_PSUTIL:
        print(f"{YELLOW}⚠ psutil not installed — memory readings may be inaccurate{RESET}")
        print(f"  Install with: pip install psutil\n")

    trace = MemoryTrace()
    sampler = MemorySampler(trace, interval_s=args.sample_interval)
    sampler.start()

    compiled_package = None
    try:
        if args.compile:
            compiled_package = profile_compile(args.compile, trace, args)

        if args.infer:
            profile_infer(Path(args.infer), trace, args)
        elif args.infer_after and compiled_package is not None:
            profile_infer(compiled_package.root, trace, args)

    finally:
        sampler.stop()

    _print_trace(trace)

    if args.output:
        data = {
            "snapshots": [
                {
                    "label": s.label,
                    "rss_mb": s.rss_mb,
                    "vms_mb": s.vms_mb,
                    "t": s.timestamp,
                }
                for s in trace.snapshots
            ],
            "peak_rss_mb": trace.peak_rss_mb,
        }
        Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"\nTrace saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
