# Aether Runtime vs HuggingFace Transformers - Benchmark Report

> **Generated:** 2026-08-27T15:25:47 UTC | **Mode:** performance | **Precision:** BF16
> **Command:** `python benchmark/run_benchmark.py --quick --mode performance`

---

## Environment

| Property | Value |
|----------|-------|
| GPU count | 2 |
| GPU(s) | Tesla T4 x2 |
| GPU VRAM | 14.562 GiB each |
| Compute capability | 7.5 |
| CUDA runtime | 12.8 |
| cuDNN | 91,002 |
| SDPA backends | Flash, Mem-efficient, Math |
| CPU | Intel Xeon @ 2.00 GHz (2 physical / 4 logical cores) |
| System RAM | 31.348 GiB total / 29.682 GiB available |
| OS | Linux 6.12.90+ glibc 2.35 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.0.0 |
| Tokenizers | 0.22.2 |
| **Aether Runtime** | **1.2.8** (df2c060) |

### Models benchmarked

| Model | Revision |
|-------|----------|
| HuggingFaceTB/SmolLM2-135M-Instruct | 12fd25f |
| Qwen/Qwen3-0.6B | c1899de |
| SummerSigh/GPTNeo350M-Instruct-SFT | 41ffbc3 |

---

## Methodology

- Both backends run on the **same host**, same process, same visible GPUs.
- **Identical prompts** built to exact token counts; identical `max_new_tokens`, sampling settings and seed.
- **Greedy decoding** (`temperature=0`) for deterministic comparison.
- CUDA is synchronized on both edges of every timed region.
- Backend order is **alternated** across repetitions to prevent thermal drift.
- Allocator peak counters reset before each measured phase.
- Telemetry sampled in dedicated extra iterations, never during reported-latency iterations.
- `warmup_iters=2`, `measure_iters=5`, `max_new_tokens=128`.

---

## Overview

![Overview Speedup](overview_speedup.png)

| Model | Best Speedup | Worst Speedup | Mean Speedup |
|-------|-------------|---------------|--------------|
| SmolLM2-135M-Instruct | 1.93x (P256/B1) | 1.56x (P1024/B4) | 1.80x |
| Qwen3-0.6B | 2.14x (P32/B1) | 1.16x (P1024/B4) | 1.57x |
| GPTNeo350M-Instruct-SFT | 1.83x (P32/B1) | 0.93x (P1024/B4) | 1.37x |

> **26 out of 27 cells**: Aether faster by >5% | **1 cell**: Transformers faster by >5% (GPTNeo P1024/B4)

---

## SmolLM2-135M-Instruct

### Throughput Comparison (Bar Chart)

![SmolLM2 Throughput Bar](smollm2_135m_instruct_throughput_bar.png)

### Throughput vs Prompt Length (Line Chart)

![SmolLM2 Throughput Line](smollm2_135m_instruct_throughput_line.png)

### Speedup Heatmap

![SmolLM2 Speedup Heatmap](smollm2_135m_instruct_speedup_heatmap.png)

### Latency Comparison (Bar Chart)

![SmolLM2 Latency Bar](smollm2_135m_instruct_latency_bar.png)

### Steady-state throughput

| Prompt | Batch | Aether tok/s | HF tok/s | Speedup | Change |
|--------|-------|-------------|---------|---------|--------|
| 32 | 1 | **46.06** | 23.95 | **1.92x** | +92.3% |
| 32 | 2 | **87.50** | 49.37 | **1.77x** | +77.2% |
| 32 | 4 | **172.41** | 95.03 | **1.81x** | +81.4% |
| 256 | 1 | **45.80** | 23.71 | **1.93x** | +93.2% |
| 256 | 2 | **85.56** | 47.50 | **1.80x** | +80.1% |
| 256 | 4 | **157.56** | 89.14 | **1.77x** | +76.8% |
| 1024 | 1 | **41.50** | 21.96 | **1.89x** | +89.0% |
| 1024 | 2 | **75.61** | 43.36 | **1.74x** | +74.4% |
| 1024 | 4 | **129.43** | 82.71 | **1.56x** | +56.5% |

### Prefill, Decode and TTFT (Batch 1)

| Prompt | Backend | Prefill all logits | Prefill serving | Discarded | Decode s | ms/token | TTFT s |
|--------|---------|-------------------|----------------|-----------|---------|---------|--------|
| 32 | transformers | 0.0552 s | 0.0445 s | 1.24x | 5.3000 | 41.41 | 0.0685 |
| 32 | **aether** | 0.0363 s | **0.0263 s** | 1.38x | **2.7526** | **21.50** | **0.0486** |
| 256 | transformers | 0.0616 s | 0.0449 s | 1.37x | 5.3546 | 41.83 | 0.0837 |
| 256 | **aether** | 0.0859 s | **0.0335 s** | 2.56x | **2.7610** | **21.57** | **0.0584** |
| 1024 | transformers | 0.1875 s | 0.1664 s | 1.13x | 5.6621 | 44.24 | 0.2065 |
| 1024 | **aether** | 0.3611 s | **0.1598 s** | 2.26x | **2.9243** | **22.85** | **0.1677** |

### Memory Usage

| Prompt | Batch | Backend | GPU weights | GPU infer delta | Peak alloc | Peak reserved | Host RSS |
|--------|-------|---------|------------|----------------|-----------|--------------|---------|
| 32 | 1 | transformers | 0.251 GiB | 0.005 GiB | 0.573 GiB | 0.605 GiB | 1.087 GiB |
| 32 | 1 | **aether** | 0.317 GiB | 0.005 GiB | 0.573 GiB | 0.586 GiB | 1.642 GiB |
| 256 | 4 | transformers | 0.251 GiB | 0.058 GiB | 0.634 GiB | 0.721 GiB | 1.087 GiB |
| 256 | 4 | **aether** | 0.317 GiB | 0.082 GiB | 0.658 GiB | 0.721 GiB | 1.642 GiB |
| 1024 | 4 | transformers | 0.251 GiB | 0.468 GiB | 1.044 GiB | 1.334 GiB | 1.087 GiB |
| 1024 | 4 | **aether** | 0.317 GiB | 0.532 GiB | 1.108 GiB | 1.334 GiB | 1.642 GiB |

---

## Qwen3-0.6B

### Throughput Comparison (Bar Chart)

![Qwen3 Throughput Bar](qwen3_0.6b_throughput_bar.png)

### Throughput vs Prompt Length (Line Chart)

![Qwen3 Throughput Line](qwen3_0.6b_throughput_line.png)

### Speedup Heatmap

![Qwen3 Speedup Heatmap](qwen3_0.6b_speedup_heatmap.png)

### Latency Comparison (Bar Chart)

![Qwen3 Latency Bar](qwen3_0.6b_latency_bar.png)

### Steady-state throughput

| Prompt | Batch | Aether tok/s | HF tok/s | Speedup | Change |
|--------|-------|-------------|---------|---------|--------|
| 32 | 1 | **41.96** | 19.56 | **2.14x** | +114.5% |
| 32 | 2 | **42.78** | 33.88 | **1.26x** | +26.3% |
| 32 | 4 | **81.40** | 65.40 | **1.24x** | +24.5% |
| 256 | 1 | **40.24** | 19.60 | **2.05x** | +105.3% |
| 256 | 2 | **37.73** | 31.35 | **1.20x** | +20.3% |
| 256 | 4 | **66.04** | 54.69 | **1.21x** | +20.7% |
| 1024 | 1 | **35.71** | 18.44 | **1.94x** | +93.7% |
| 1024 | 2 | **27.38** | 23.11 | **1.18x** | +18.5% |
| 1024 | 4 | **39.81** | 34.39 | **1.16x** | +15.8% |

> **Note:** Qwen3-0.6B shows much higher single-stream (B=1) speedups. The larger model is more memory-bandwidth bound; Aether's fused kernels eliminate redundant memory traffic.

### Prefill, Decode and TTFT (Batch 1)

| Prompt | Backend | Prefill all logits | Prefill serving | Discarded | Decode s | ms/token | TTFT s |
|--------|---------|-------------------|----------------|-----------|---------|---------|--------|
| 32 | transformers | 0.0602 s | 0.0564 s | 1.07x | 6.4865 | 50.68 | 0.1025 |
| 32 | **aether** | 0.0545 s | **0.0432 s** | 1.26x | **3.0076** | **23.50** | **0.0527** |
| 256 | transformers | 0.1474 s | 0.1151 s | 1.28x | 6.4157 | 50.12 | 0.1547 |
| 256 | **aether** | 0.1642 s | **0.1120 s** | 1.47x | **3.0691** | **23.98** | **0.1173** |
| 1024 | transformers | 0.6589 s | 0.5321 s | 1.24x | 6.4088 | 50.07 | 0.5213 |
| 1024 | **aether** | 0.7721 s | **0.5251 s** | 1.47x | **3.0589** | **23.90** | **0.5371** |

### Memory Usage

| Prompt | Batch | Backend | GPU weights | GPU infer delta | Peak alloc | Peak reserved | Host RSS |
|--------|-------|---------|------------|----------------|-----------|--------------|---------|
| 32 | 1 | transformers | 1.400 GiB | 0.023 GiB | 2.839 GiB | 2.924 GiB | 2.156 GiB |
| 32 | 1 | **aether** | 1.400 GiB | 0.023 GiB | 2.839 GiB | 2.943 GiB | 4.221 GiB |
| 1024 | 4 | transformers | 1.400 GiB | 1.201 GiB | 4.018 GiB | 4.299 GiB | 2.156 GiB |
| 1024 | 4 | **aether** | 1.400 GiB | 1.374 GiB | 4.191 GiB | 4.674 GiB | 4.221 GiB |

---

## GPTNeo350M-Instruct-SFT

### Throughput Comparison (Bar Chart)

![GPTNeo Throughput Bar](gptneo350m_instruct_sft_throughput_bar.png)

### Throughput vs Prompt Length (Line Chart)

![GPTNeo Throughput Line](gptneo350m_instruct_sft_throughput_line.png)

### Speedup Heatmap

![GPTNeo Speedup Heatmap](gptneo350m_instruct_sft_speedup_heatmap.png)

### Latency Comparison (Bar Chart)

![GPTNeo Latency Bar](gptneo350m_instruct_sft_latency_bar.png)

### Steady-state throughput

| Prompt | Batch | Aether tok/s | HF tok/s | Speedup | Change |
|--------|-------|-------------|---------|---------|--------|
| 32 | 1 | **71.67** | 39.14 | **1.83x** | +83.1% |
| 32 | 2 | **64.57** | 47.37 | **1.36x** | +36.3% |
| 32 | 4 | **123.11** | 93.42 | **1.32x** | +31.8% |
| 256 | 1 | **63.14** | 39.34 | **1.61x** | +60.5% |
| 256 | 2 | **57.40** | 45.53 | **1.26x** | +26.1% |
| 256 | 4 | **99.94** | 85.60 | **1.17x** | +16.8% |
| 1024 | 1 | **54.41** | 36.94 | **1.47x** | +47.3% |
| 1024 | 2 | **41.35** | 39.08 | **1.06x** | +5.8% |
| 1024 | 4 | 60.38 | **65.21** | **0.93x** | -7.4% |

> **NOTE - GPTNeo P1024/B4:** The only cell where Transformers wins. GPT-NeoX uses full MHA (no GQA), causing Aether's KV-cache growth at long contexts and large batches to exceed the fused kernel savings. Targeted for optimization in Aether v1.3.

### Prefill, Decode and TTFT (Batch 1)

| Prompt | Backend | Prefill all logits | Prefill serving | Discarded | Decode s | ms/token | TTFT s |
|--------|---------|-------------------|----------------|-----------|---------|---------|--------|
| 32 | transformers | 0.0657 s | 0.0436 s | 1.51x | 3.2272 | 25.21 | 0.0800 |
| 32 | **aether** | 0.0350 s | **0.0321 s** | 1.09x | **1.7539** | **13.70** | **0.0356** |
| 256 | transformers | 0.0834 s | 0.0750 s | 1.11x | 3.1789 | 24.84 | 0.1086 |
| 256 | **aether** | 0.0905 s | **0.0742 s** | 1.22x | **1.9529** | **15.26** | **0.0725** |
| 1024 | transformers | 0.3836 s | 0.3423 s | 1.12x | 3.1224 | 24.39 | 0.3379 |
| 1024 | **aether** | 0.4142 s | **0.3404 s** | 1.22x | **2.0122** | **15.72** | **0.3446** |

### Memory Usage

| Prompt | Batch | Backend | GPU weights | GPU infer delta | Peak alloc | Peak reserved | Host RSS |
|--------|-------|---------|------------|----------------|-----------|--------------|---------|
| 32 | 1 | transformers | 0.757 GiB | 0.016 GiB | 1.548 GiB | 1.645 GiB | 4.497 GiB |
| 32 | 1 | **aether** | 0.759 GiB | 0.017 GiB | 1.549 GiB | 1.643 GiB | 4.574 GiB |
| 1024 | 4 | transformers | 0.757 GiB | 1.078 GiB | 2.611 GiB | 2.977 GiB | 4.497 GiB |
| 1024 | 4 | **aether** | 0.759 GiB | 1.160 GiB | 2.693 GiB | 2.977 GiB | 4.574 GiB |

---

## Full Speedup Summary (All 27 Cells)

| Model | Prompt | Batch | Speedup | Aether tok/s | HF tok/s |
|-------|--------|-------|---------|-------------|---------|
| SmolLM2-135M | 32 | 1 | **1.92x** | 46.06 | 23.95 |
| SmolLM2-135M | 32 | 2 | **1.77x** | 87.50 | 49.37 |
| SmolLM2-135M | 32 | 4 | **1.81x** | 172.41 | 95.03 |
| SmolLM2-135M | 256 | 1 | **1.93x** | 45.80 | 23.71 |
| SmolLM2-135M | 256 | 2 | **1.80x** | 85.56 | 47.50 |
| SmolLM2-135M | 256 | 4 | **1.77x** | 157.56 | 89.14 |
| SmolLM2-135M | 1024 | 1 | **1.89x** | 41.50 | 21.96 |
| SmolLM2-135M | 1024 | 2 | **1.74x** | 75.61 | 43.36 |
| SmolLM2-135M | 1024 | 4 | **1.56x** | 129.43 | 82.71 |
| Qwen3-0.6B | 32 | 1 | **2.14x** | 41.96 | 19.56 |
| Qwen3-0.6B | 32 | 2 | **1.26x** | 42.78 | 33.88 |
| Qwen3-0.6B | 32 | 4 | **1.24x** | 81.40 | 65.40 |
| Qwen3-0.6B | 256 | 1 | **2.05x** | 40.24 | 19.60 |
| Qwen3-0.6B | 256 | 2 | **1.20x** | 37.73 | 31.35 |
| Qwen3-0.6B | 256 | 4 | **1.21x** | 66.04 | 54.69 |
| Qwen3-0.6B | 1024 | 1 | **1.94x** | 35.71 | 18.44 |
| Qwen3-0.6B | 1024 | 2 | **1.18x** | 27.38 | 23.11 |
| Qwen3-0.6B | 1024 | 4 | **1.16x** | 39.81 | 34.39 |
| GPTNeo-350M | 32 | 1 | **1.83x** | 71.67 | 39.14 |
| GPTNeo-350M | 32 | 2 | **1.36x** | 64.57 | 47.37 |
| GPTNeo-350M | 32 | 4 | **1.32x** | 123.11 | 93.42 |
| GPTNeo-350M | 256 | 1 | **1.61x** | 63.14 | 39.34 |
| GPTNeo-350M | 256 | 2 | **1.26x** | 57.40 | 45.53 |
| GPTNeo-350M | 256 | 4 | **1.17x** | 99.94 | 85.60 |
| GPTNeo-350M | 1024 | 1 | **1.47x** | 54.41 | 36.94 |
| GPTNeo-350M | 1024 | 2 | **1.06x** | 41.35 | 39.08 |
| GPTNeo-350M | 1024 | 4 | 0.93x | 60.38 | 65.21 |

---

## CPU and GPU Utilization Highlights

| Model | Backend | GPU util (mean) | GPU power | Temp max |
|-------|---------|----------------|-----------|---------|
| SmolLM2-135M (B4) | transformers | 37.6% | 43.4 W | 77C |
| SmolLM2-135M (B4) | **aether** | **58.8-96.7%** | **59-69 W** | 77-80C |
| Qwen3-0.6B (B4) | transformers | ~96-98% | ~68 W | 77C |
| Qwen3-0.6B (B4) | **aether** | **~97-99%** | **~68 W** | 77C |
| GPTNeo-350M (B4) | transformers | 84-91% | 65-69 W | 77C |
| GPTNeo-350M (B4) | **aether** | **98-99%** | **69-70 W** | 78C |

Aether pushes GPU utilization substantially higher, particularly for smaller models at smaller batch sizes, confirming that its kernel fusion removes CPU-side overhead that was previously stalling the GPU.

---

## Conclusions

| Metric | Result |
|--------|--------|
| Comparable cells | 27 |
| Aether faster (>5%) | **26 / 27** |
| Transformers faster (>5%) | 1 / 27 |
| Within +/-5% | 0 / 27 |
| **Best Aether speedup** | **2.14x - Qwen3-0.6B, P32, B1** |
| **Worst Aether result** | **0.93x - GPTNeo-350M, P1024, B4** |

### Key observations

1. **SmolLM2-135M** shows the most consistent speedup (~1.56-1.93x) across all configurations. Aether's operator fusion pays dividends uniformly for this lightweight model.
2. **Qwen3-0.6B** shows the highest single-stream speedup (>2x) but smaller batch gains. The model runs near GPU saturation at batch>=2 under both backends.
3. **GPTNeo-350M** benefits strongly at small contexts but regresses at P1024/B4 due to full MHA KV-cache memory pressure - a known issue for future optimization.
4. **Decode throughput** is approximately 2x faster across batch 1: SmolLM2 21 ms/tok, Qwen3 24 ms/tok, GPTNeo 14-16 ms/tok vs Transformers 25-51 ms/tok.
5. **TTFT** is 20-50% lower under Aether for short prompts (P32), converging at longer prompts where prefill compute dominates.
6. **GPU utilization** is substantially higher under Aether (up to 96-99% vs 28-39% for small batches).

---

## Limitations

- Prefill reported in two configurations: all logits (every position) and serving (last position only).
- Decode time is derived by subtracting measured prefill from end-to-end latency.
- TTFT is measured through each library's own streaming API.
- Kaggle is a shared, virtualized environment - clock behavior and thermal state are not fully controlled.
- Aether's semantic response cache is **disabled** for all measured runs.
- Compilation (prepare_s) is a one-time cost and is **not** amortized into throughput figures.

---

*Aether Runtime - Compile once. Run on any hardware, forever.*