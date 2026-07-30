# Aether Benchmarks

This directory contains the Aether benchmark suite and comparison tools.

## Run a single benchmark

```bash
python -m benchmarks.bench_suite --models Qwen/Qwen3-0.6B
```

## Run multiple models

```bash
python -m benchmarks.bench_suite --models "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.5B"
```

## Smoke test

```bash
python -m benchmarks.bench_suite --smoke
```

## Save results

```bash
python -m benchmarks.bench_suite --output results.json
```

## Benchmark CSV

`models.csv` lists models and prompts for batch benchmarking. Use this to define
standard benchmark configurations.

## Metrics

Each benchmark run reports:

- `duration_s`: total generation time
- `throughput_tps`: tokens per second
- `ttft_ms`: time to first token
- `prompt_tokens`: input token count
- `completion_tokens`: generated token count
