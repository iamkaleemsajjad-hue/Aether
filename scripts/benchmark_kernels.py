#!/usr/bin/env python3
"""
benchmark_kernels.py — Micro-benchmark the Aether native CPU kernels.

Measures wall-clock throughput and latency for every kernel primitive
(SGEMM, RMSNorm, SwiGLU, Softmax, RoPE, Argmax, dequantize) across
matrix sizes representative of real LLM workloads (7B, 70B class).

The script prints a Markdown table and optionally saves JSON results for
CI tracking.

Usage:
    python scripts/benchmark_kernels.py
    python scripts/benchmark_kernels.py --size 70b
    python scripts/benchmark_kernels.py --iterations 200 --output bench_results.json
    python scripts/benchmark_kernels.py --kernel sgemm --warmup 10 --iterations 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

import numpy as np

# ── Model size presets (hidden, intermediate, vocab, heads, head_dim) ─────────

SIZE_PRESETS: dict[str, dict[str, int]] = {
    "tiny": dict(hidden=64,   intermediate=128,  vocab=256,   heads=4,  head_dim=16,  kv_heads=2),
    "1b":   dict(hidden=2048, intermediate=5504, vocab=32000, heads=16, head_dim=128, kv_heads=8),
    "7b":   dict(hidden=4096, intermediate=11008, vocab=32000, heads=32, head_dim=128, kv_heads=8),
    "70b":  dict(hidden=8192, intermediate=28672, vocab=32000, heads=64, head_dim=128, kv_heads=8),
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class KernelResult:
    kernel: str
    size_label: str
    is_native: bool
    warmup_iters: int
    bench_iters: int
    mean_ms: float
    min_ms: float
    max_ms: float
    throughput_label: str = ""  # e.g. "XX GFLOP/s" or "XX GB/s"

    @property
    def std_ms(self) -> float:
        return 0.0  # filled in by benchmark runner


@dataclass
class BenchmarkReport:
    timestamp: str
    platform: str
    python_version: str
    native_kernels: bool
    results: list[KernelResult] = field(default_factory=list)


# ── Timing harness ────────────────────────────────────────────────────────────

def _time_fn(fn: Callable, warmup: int, iters: int) -> tuple[float, float, float]:
    """Return (mean_ms, min_ms, max_ms) for ``fn()``."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.min(times)), float(np.max(times))


# ── Individual kernel benchmarks ──────────────────────────────────────────────

def bench_sgemm(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    m, k, n = 1, cfg["hidden"], cfg["intermediate"]
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    mean, lo, hi = _time_fn(lambda: kernels.sgemm(a, b), warmup, iters)
    gflops = 2.0 * m * k * n / (mean * 1e-3) / 1e9
    return KernelResult("sgemm", label, kernels.is_native, warmup, iters, mean, lo, hi,
                        throughput_label=f"{gflops:.1f} GFLOP/s")


def bench_rmsnorm(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    seq = 128
    h = cfg["hidden"]
    x = np.random.randn(seq, h).astype(np.float32)
    w = np.ones(h, dtype=np.float32)
    mean, lo, hi = _time_fn(lambda: kernels.rmsnorm(x, w), warmup, iters)
    return KernelResult("rmsnorm", label, kernels.is_native, warmup, iters, mean, lo, hi)


def bench_swiglu(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    seq = 128
    inter = cfg["intermediate"]
    gate = np.random.randn(seq, inter).astype(np.float32)
    up   = np.random.randn(seq, inter).astype(np.float32)
    mean, lo, hi = _time_fn(lambda: kernels.swiglu(gate, up), warmup, iters)
    return KernelResult("swiglu", label, kernels.is_native, warmup, iters, mean, lo, hi)


def bench_softmax(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    # Attention score matrix: (heads * seq, kv_len)
    heads, seq, kv_len = cfg["heads"], 32, 512
    x = np.random.randn(heads * seq, kv_len).astype(np.float32)
    mean, lo, hi = _time_fn(lambda: kernels.softmax(x), warmup, iters)
    gb_s = x.nbytes * 2 / (mean * 1e-3) / 1e9
    return KernelResult("softmax", label, kernels.is_native, warmup, iters, mean, lo, hi,
                        throughput_label=f"{gb_s:.1f} GB/s")


def bench_rope(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    seq, heads, hd = 32, cfg["heads"], cfg["head_dim"]
    half = hd // 2
    x = np.random.randn(seq, heads, hd).astype(np.float32)
    cos_t = np.cos(np.arange(seq * half).reshape(seq, half)).astype(np.float32)
    sin_t = np.sin(np.arange(seq * half).reshape(seq, half)).astype(np.float32)
    mean, lo, hi = _time_fn(lambda: kernels.rope(x, cos_t, sin_t), warmup, iters)
    return KernelResult("rope", label, kernels.is_native, warmup, iters, mean, lo, hi)


def bench_argmax(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    v = cfg["vocab"]
    logits = np.random.randn(v).astype(np.float32)
    mean, lo, hi = _time_fn(lambda: kernels.argmax(logits), warmup, iters)
    return KernelResult("argmax", label, kernels.is_native, warmup, iters, mean, lo, hi)


def bench_dequantize(kernels, cfg: dict, warmup: int, iters: int, label: str) -> KernelResult:
    from aether.quantization.formats import quantize_tensor
    h = cfg["hidden"]
    inter = cfg["intermediate"]
    w = np.random.randn(inter, h).astype(np.float32) * 0.02
    qt = quantize_tensor(w, "Q4_K_M", block_size=32)
    mean, lo, hi = _time_fn(lambda: kernels.dequantize(qt), warmup, iters)
    gb_s = w.nbytes / (mean * 1e-3) / 1e9
    return KernelResult("dequantize_q4", label, kernels.is_native, warmup, iters, mean, lo, hi,
                        throughput_label=f"{gb_s:.1f} GB/s (output)")


KERNEL_FNS: dict[str, Callable] = {
    "sgemm":      bench_sgemm,
    "rmsnorm":    bench_rmsnorm,
    "swiglu":     bench_swiglu,
    "softmax":    bench_softmax,
    "rope":       bench_rope,
    "argmax":     bench_argmax,
    "dequantize": bench_dequantize,
}


# ── Report rendering ──────────────────────────────────────────────────────────

def _print_table(results: list[KernelResult], native: bool) -> None:
    backend = "native C++" if native else "numpy reference"
    print(f"\n## Aether CPU Kernel Benchmarks  —  backend: {backend}\n")
    header = f"{'Kernel':<16} {'Size':<6} {'Mean (ms)':>10} {'Min (ms)':>9} {'Max (ms)':>9} {'Throughput':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.kernel:<16} {r.size_label:<6} "
            f"{r.mean_ms:>10.3f} {r.min_ms:>9.3f} {r.max_ms:>9.3f} "
            f"{r.throughput_label:<18}"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import platform
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Benchmark Aether native CPU kernels")
    parser.add_argument("--size", choices=list(SIZE_PRESETS), default="7b",
                        help="Model size preset (default: 7b)")
    parser.add_argument("--kernel", choices=list(KERNEL_FNS) + ["all"], default="all",
                        help="Which kernel to benchmark (default: all)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", help="Save JSON results to this file")
    args = parser.parse_args()

    from aether.kernels.native_cpu import get_native_kernels
    kernels = get_native_kernels()
    compiled = kernels.ensure_compiled()
    print(f"Native kernels: {'yes (' + kernels.toolchain.name + ')' if compiled else 'no — using numpy'}")

    cfg = SIZE_PRESETS[args.size]
    label = args.size

    kernel_keys = list(KERNEL_FNS) if args.kernel == "all" else [args.kernel]
    results: list[KernelResult] = []
    for key in kernel_keys:
        print(f"  Benchmarking {key} ({args.warmup} warmup, {args.iterations} iters)...", end=" ", flush=True)
        try:
            r = KERNEL_FNS[key](kernels, cfg, args.warmup, args.iterations, label)
            results.append(r)
            print(f"{r.mean_ms:.3f} ms")
        except Exception as exc:
            print(f"FAILED: {exc}")

    _print_table(results, compiled)

    if args.output:
        report = BenchmarkReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            platform=f"{platform.system()} {platform.machine()}",
            python_version=sys.version,
            native_kernels=compiled,
            results=results,
        )
        Path(args.output).write_text(json.dumps(asdict(report), indent=2))
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
