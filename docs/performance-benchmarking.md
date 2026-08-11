"""
Aether Runtime — Performance Benchmarking Guide

A practical guide to measuring and validating Aether Runtime inference performance
using the built-in benchmark infrastructure.

Research basis:
  - MLPerf Inference v4.0 (2025) benchmark methodology
  - vLLM benchmark suite (2024)
  - SGLang benchmarking (2024)
"""

# Performance Benchmarking Guide

## Overview

Aether Runtime provides a production-grade benchmarking system that measures:

- **TTFT** (Time To First Token): Latency until the first generated token
- **TBT** (Time Between Tokens): Per-token decode latency
- **P50/P95/P99** end-to-end latency percentiles
- **Throughput**: Total tokens per second
- **Memory**: Peak RSS memory usage
- **Energy**: Estimated watt-hours (via TDP heuristic)
- **KV Cache hit rate**
- **Speculative decoding acceptance rate**

## Quick Start

```bash
# Basic benchmark against a compiled AEG model
python benchmarks/benchmark_runner.py /path/to/model.aeg

# With more requests and output
python benchmarks/benchmark_runner.py /path/to/model.aeg \
    --requests 200 \
    --warmup 20 \
    --max-output-tokens 256 \
    --output results.json

# Compare against a saved baseline
python benchmarks/benchmark_runner.py /path/to/model.aeg \
    --compare-baseline results_baseline.json \
    --output results_current.json
```

## Benchmark Scenarios

### Single-stream (Default)
One request at a time. Measures pure latency characteristics.

```bash
python benchmarks/benchmark_runner.py model.aeg --concurrency 1
```

### Throughput Scenario
Maximum concurrent requests. Measures aggregate tokens/sec.

```bash
python benchmarks/benchmark_runner.py model.aeg --concurrency 8 --requests 500
```

### Quick Validation
Fast sanity check during development.

```bash
python benchmarks/benchmark_runner.py model.aeg --requests 10 --warmup 2
```

## Using the Python API

```python
from benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    model_path="path/to/model.aeg",
    num_requests=100,
    warmup_requests=10,
    max_output_tokens=256,
    temperature=0.0,       # Greedy decoding for deterministic results
    hardware_target="cpu_avx512",
    output_path="results.json",
)

runner = BenchmarkRunner(config)
result = runner.run()

print(f"TTFT P50: {result.ttft_p50_ms:.1f} ms")
print(f"TTFT P95: {result.ttft_p95_ms:.1f} ms")
print(f"Throughput: {result.throughput_tps:.1f} tokens/sec")
print(f"Peak Memory: {result.peak_memory_mb:.0f} MB")
```

## Interpreting Results

### Latency Targets (per PRD v4.0)

| Target | TTFT P50 | TBT P50 | Throughput |
|--------|----------|---------|------------|
| CPU (cpu_avx512) | <2000 ms | <100 ms | >50 tok/s |
| GPU sm80 (A100) | <100 ms | <10 ms | >3000 tok/s |
| GPU sm90 (H100) | <50 ms | <5 ms | >8000 tok/s |
| GPU sm100 (B200) | <30 ms | <3 ms | >15000 tok/s |

### Reading the Report

```
────────────────────────────────────────────────────────────
 Aether Benchmark Results
────────────────────────────────────────────────────────────
 Model:          /path/to/model.aeg
 Target:         cpu_avx512
 Requests:       100
────────────────────────────────────────────────────────────
 TTFT P50:       850.2 ms     ← Median time to first token
 TTFT P95:       1240.7 ms    ← 95th pct TTFT
 TTFT P99:       1890.3 ms    ← 99th pct TTFT (SLA boundary)
 TBT  P50:       45.3 ms      ← Median inter-token latency
 TBT  P95:       72.1 ms
 E2E  P99:       8920.4 ms    ← 99th pct total request latency
────────────────────────────────────────────────────────────
 Throughput:     1247.5 tokens/sec
 Req/sec:        4.87
────────────────────────────────────────────────────────────
 KV Hit Rate:    72.4%         ← Prefix cache effectiveness
 Error Rate:     0.0%
 Peak Mem MB:    2048
 Energy Wh:      0.0245
────────────────────────────────────────────────────────────
```

## Performance Regression CI

Add a performance gate to your CI pipeline:

```python
# scripts/ci_perf_gate.py
from benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    model_path="model.aeg",
    num_requests=50,
    warmup_requests=5,
    compare_baseline="baseline.json",
)
runner = BenchmarkRunner(config)
result = runner.run()

# Fail if throughput regressed >10%
if result.baseline_comparison.get("throughput", "").startswith("-") and \
   float(result.baseline_comparison["throughput"].replace("%", "")) < -10:
    raise SystemExit("Performance regression: throughput dropped >10%")
```

## Latency Benchmark

For detailed per-request latency distribution:

```bash
python benchmarks/latency_bench.py --model model.aeg --num-runs 50
```

## Environment Notes

- Always run benchmarks with no other heavy processes active
- Use `--temperature 0.0` for deterministic, reproducible results
- Run 3+ benchmark sets and average to reduce variance
- Record Python version, numpy version, and CPU model with results
