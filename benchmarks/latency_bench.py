"""
Aether Runtime — Latency & Throughput Micro-benchmarks.

Provides focused benchmarks for:
  - TTFT (Time To First Token) across prompt lengths
  - TBT (Time Between Tokens) decode latency
  - Throughput sweep (requests per second at varying batch sizes)
  - Memory scaling with context length
  - KV cache efficiency measurement

Research basis:
  - MLPerf Inference v4.0 (2025)
  - DeepSpeed-MoE benchmark methodology (2024)
  - vLLM continuous batching benchmarks (2024)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner, _percentile


# ---------------------------------------------------------------------------
# Latency sweep benchmark
# ---------------------------------------------------------------------------

@dataclass
class LatencySweepResult:
    """Results from sweeping across prompt lengths."""

    prompt_lengths: list[int]
    ttft_p50_by_length: dict[int, float] = field(default_factory=dict)
    ttft_p99_by_length: dict[int, float] = field(default_factory=dict)
    tbt_p50_by_length: dict[int, float] = field(default_factory=dict)
    tbt_p99_by_length: dict[int, float] = field(default_factory=dict)
    throughput_by_length: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_lengths": self.prompt_lengths,
            "ttft_p50_by_length": self.ttft_p50_by_length,
            "ttft_p99_by_length": self.ttft_p99_by_length,
            "tbt_p50_by_length": self.tbt_p50_by_length,
            "tbt_p99_by_length": self.tbt_p99_by_length,
            "throughput_by_length": self.throughput_by_length,
        }

    def summary(self) -> str:
        lines = ["Latency Sweep Results:", "─" * 60]
        lines.append(f"{'Prompt Len':>12} {'TTFT P50':>12} {'TTFT P99':>12} {'TBT P50':>10} {'Throughput':>12}")
        for plen in self.prompt_lengths:
            ttft50 = self.ttft_p50_by_length.get(plen, 0.0)
            ttft99 = self.ttft_p99_by_length.get(plen, 0.0)
            tbt50 = self.tbt_p50_by_length.get(plen, 0.0)
            thr = self.throughput_by_length.get(plen, 0.0)
            lines.append(f"{plen:>12} {ttft50:>11.1f}ms {ttft99:>11.1f}ms {tbt50:>9.1f}ms {thr:>11.1f}")
        return "\n".join(lines)


def run_latency_sweep(
    model_path: str,
    prompt_lengths: list[int] | None = None,
    requests_per_length: int = 20,
    max_output_tokens: int = 64,
    hardware_target: str = "cpu_avx512",
) -> LatencySweepResult:
    """
    Sweep across different prompt lengths to characterize TTFT scaling.

    This measures how TTFT grows with input length — critical for understanding
    prefill bottlenecks. Linear growth indicates compute-bound prefill;
    super-linear growth indicates memory bandwidth bottlenecks.
    """
    if prompt_lengths is None:
        prompt_lengths = [32, 64, 128, 256, 512, 1024]

    result = LatencySweepResult(prompt_lengths=prompt_lengths)

    # Build a vocabulary for synthetic prompts of exact lengths
    words = [
        "the", "model", "processes", "input", "through", "attention", "layers",
        "generating", "output", "tokens", "one", "at", "a", "time", "using",
        "transformer", "architecture", "with", "causal", "masking",
    ]

    for plen in prompt_lengths:
        print(f"[LATENCY_BENCH] Testing prompt length: {plen} tokens")

        # Build a synthetic prompt of approximately plen tokens
        prompt_words = []
        while len(prompt_words) < plen:
            prompt_words.extend(words)
        prompt = " ".join(prompt_words[:plen])

        config = BenchmarkConfig(
            model_path=model_path,
            num_requests=requests_per_length,
            warmup_requests=3,
            max_input_tokens=plen + 16,
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            hardware_target=hardware_target,
        )
        runner = BenchmarkRunner(config)
        try:
            bench_result = runner.run()
            result.ttft_p50_by_length[plen] = bench_result.ttft_p50_ms
            result.ttft_p99_by_length[plen] = bench_result.ttft_p99_ms
            result.tbt_p50_by_length[plen] = bench_result.tbt_p50_ms
            result.tbt_p99_by_length[plen] = bench_result.tbt_p99_ms
            result.throughput_by_length[plen] = bench_result.throughput_tps
        except Exception as exc:  # noqa: BLE001
            print(f"[LATENCY_BENCH] Error at length {plen}: {exc}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Throughput sweep benchmark
# ---------------------------------------------------------------------------

@dataclass
class ThroughputSweepResult:
    """Results from sweeping across batch sizes / concurrency."""

    concurrency_levels: list[int]
    throughput_by_concurrency: dict[int, float] = field(default_factory=dict)
    latency_p99_by_concurrency: dict[int, float] = field(default_factory=dict)
    gpu_utilization_by_concurrency: dict[int, float] = field(default_factory=dict)
    optimal_concurrency: int = 1
    peak_throughput_tps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency_levels": self.concurrency_levels,
            "throughput_by_concurrency": self.throughput_by_concurrency,
            "latency_p99_by_concurrency": self.latency_p99_by_concurrency,
            "gpu_utilization_by_concurrency": self.gpu_utilization_by_concurrency,
            "optimal_concurrency": self.optimal_concurrency,
            "peak_throughput_tps": self.peak_throughput_tps,
        }


def run_throughput_sweep(
    model_path: str,
    concurrency_levels: list[int] | None = None,
    requests_per_level: int = 50,
    max_output_tokens: int = 128,
    hardware_target: str = "cpu_avx512",
) -> ThroughputSweepResult:
    """
    Sweep across concurrency levels to find the throughput-optimal batch size.

    This reproduces the classic throughput vs. latency tradeoff curve used in
    vLLM, TGI, and SGLang production benchmarks.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]

    result = ThroughputSweepResult(concurrency_levels=concurrency_levels)
    best_tps = 0.0
    best_concurrency = 1

    for concurrency in concurrency_levels:
        print(f"[THROUGHPUT_BENCH] Testing concurrency: {concurrency}")
        config = BenchmarkConfig(
            model_path=model_path,
            num_requests=requests_per_level,
            warmup_requests=5,
            max_output_tokens=max_output_tokens,
            concurrency=concurrency,
            temperature=0.0,
            hardware_target=hardware_target,
        )
        runner = BenchmarkRunner(config)
        try:
            bench_result = runner.run()
            tps = bench_result.throughput_tps
            result.throughput_by_concurrency[concurrency] = tps
            result.latency_p99_by_concurrency[concurrency] = bench_result.e2e_p99_ms
            if tps > best_tps:
                best_tps = tps
                best_concurrency = concurrency
        except Exception as exc:  # noqa: BLE001
            print(f"[THROUGHPUT_BENCH] Error at concurrency {concurrency}: {exc}", file=sys.stderr)

    result.optimal_concurrency = best_concurrency
    result.peak_throughput_tps = best_tps
    return result


# ---------------------------------------------------------------------------
# Memory scaling benchmark
# ---------------------------------------------------------------------------

@dataclass
class MemoryScalingResult:
    """Results from memory scaling across context lengths."""

    context_lengths: list[int]
    peak_memory_mb_by_context: dict[int, float] = field(default_factory=dict)
    kv_cache_mb_by_context: dict[int, float] = field(default_factory=dict)
    max_supported_context: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_lengths": self.context_lengths,
            "peak_memory_mb_by_context": self.peak_memory_mb_by_context,
            "kv_cache_mb_by_context": self.kv_cache_mb_by_context,
            "max_supported_context": self.max_supported_context,
        }


def run_memory_scaling(
    model_path: str,
    context_lengths: list[int] | None = None,
    hardware_target: str = "cpu_avx512",
) -> MemoryScalingResult:
    """
    Measure memory consumption across context lengths.

    Critical for understanding KV cache scaling and maximum context limits.
    """
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192]

    result = MemoryScalingResult(context_lengths=context_lengths)
    max_supported = 0

    for ctx_len in context_lengths:
        print(f"[MEMORY_BENCH] Testing context length: {ctx_len}")
        config = BenchmarkConfig(
            model_path=model_path,
            num_requests=5,
            warmup_requests=1,
            max_input_tokens=ctx_len,
            max_output_tokens=32,
            temperature=0.0,
            hardware_target=hardware_target,
        )
        runner = BenchmarkRunner(config)
        try:
            bench_result = runner.run()
            if bench_result.error_rate < 0.5:
                result.peak_memory_mb_by_context[ctx_len] = bench_result.peak_memory_mb
                # Estimate KV cache from total - model overhead
                model_overhead = list(result.peak_memory_mb_by_context.values())[0] if result.peak_memory_mb_by_context else 0
                result.kv_cache_mb_by_context[ctx_len] = max(0, bench_result.peak_memory_mb - model_overhead * 0.8)
                max_supported = ctx_len
        except Exception as exc:  # noqa: BLE001
            print(f"[MEMORY_BENCH] Error at context {ctx_len}: {exc}", file=sys.stderr)
            break  # Stop at first failure — longer contexts will also fail

    result.max_supported_context = max_supported
    return result


# ---------------------------------------------------------------------------
# Complete benchmark suite
# ---------------------------------------------------------------------------

def run_full_suite(
    model_path: str,
    output_dir: str = "benchmark_results",
    hardware_target: str = "cpu_avx512",
    quick: bool = False,
) -> dict[str, Any]:
    """
    Run the complete Aether benchmark suite.

    Returns aggregated results from all sub-benchmarks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # 1. Standard benchmark
    print("\n" + "=" * 60)
    print("Running standard performance benchmark...")
    print("=" * 60)
    config = BenchmarkConfig(
        model_path=model_path,
        num_requests=20 if quick else 100,
        warmup_requests=3 if quick else 10,
        max_output_tokens=64 if quick else 256,
        output_path=str(output_path / "standard_bench.json"),
        hardware_target=hardware_target,
    )
    runner = BenchmarkRunner(config)
    standard_result = runner.run()
    results["standard"] = standard_result.to_dict()

    # 2. Latency sweep
    if not quick:
        print("\n" + "=" * 60)
        print("Running latency sweep benchmark...")
        print("=" * 60)
        latency_result = run_latency_sweep(
            model_path=model_path,
            prompt_lengths=[32, 128, 512],
            requests_per_length=10,
            hardware_target=hardware_target,
        )
        latency_output = output_path / "latency_sweep.json"
        latency_output.write_text(json.dumps(latency_result.to_dict(), indent=2))
        print(latency_result.summary())
        results["latency_sweep"] = latency_result.to_dict()

    # 3. Memory scaling
    print("\n" + "=" * 60)
    print("Running memory scaling benchmark...")
    print("=" * 60)
    mem_result = run_memory_scaling(
        model_path=model_path,
        context_lengths=[512, 1024, 2048] if quick else [512, 1024, 2048, 4096],
        hardware_target=hardware_target,
    )
    mem_output = output_path / "memory_scaling.json"
    mem_output.write_text(json.dumps(mem_result.to_dict(), indent=2))
    results["memory_scaling"] = mem_result.to_dict()

    # Save combined results
    combined_output = output_path / "full_suite_results.json"
    combined_output.write_text(json.dumps(results, indent=2))
    print(f"\n[BENCH] Full suite results saved to {combined_output}")

    return results
