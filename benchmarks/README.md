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

## Verified local checkpoint

The repository includes a measured real-checkpoint result at
`benchmarks/results/qwen3_0.6b_cpu_avx2_2026-08-19.json`. It compiles the
provided Qwen3 0.6B SafeTensors checkpoint for `cpu_avx2`, runs three greedy
generation probes, and compares the next-token logits against the local Hugging
Face reference. The quality probe matched top-1 and all top-50 candidates.

Approximate pruning and sparse-attention plans are not enabled by default until
they have a task/perplexity quality gate; explicit precision modes can still be
used for experiments.

The post-optimization measurement is recorded at
`benchmarks/results/qwen3_0.6b_cpu_avx2_2026-08-20.json`. It reports both
first-run and steady-state throughput because JIT/kernel loading and weight
cache effects materially change the first request. The benchmark now performs
prefill once and decodes from the returned KV cache; comparing it to an older
run that prefills twice is not a valid speedup claim.
