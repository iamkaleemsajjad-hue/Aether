# Aether Runtime

The Aether Runtime is the execution engine that loads AEG artifacts, detects hardware, selects the best backend, and serves inference requests.

## Startup Flow

When you run `aether run` or `aether serve`:

1. Detect hardware fingerprint (GPU, VRAM, driver, compute capability).
2. Load the AEG manifest.
3. Select the best backend for the detected hardware.
4. Load the model into the selected backend.
5. Allocate KV cache across tiers.
6. Start the disaggregated prefill/decode scheduler.
7. Enable tree-speculative decoding if configured.
8. Serve requests.

## Hardware Detection

The runtime detects the current platform and maps it to a hardware target:

- NVIDIA → `cuda_sm70/80/89/90/100`
- Apple Silicon → `metal_m1` or `metal_m3`
- AMD → `rocm_rdna3` or `rocm_cdna3`
- Intel NPU / CPU → `openvino_npu` or `cpu_avx512`
- ARM CPU → `cpu_neon`

You can inspect the detected hardware with:

```bash
aether hw
```

## Backend Selection

The runtime selects the best available backend for the target:

- NVIDIA → vLLM, TensorRT-LLM, PyTorch
- Apple Silicon → MLX, PyTorch
- AMD → PyTorch, llama.cpp
- Intel/CPU → ONNX Runtime, llama.cpp, PyTorch

You can force a backend with the `AETHER_BACKEND` environment variable or `RuntimeConfig.backend_name`.

## Disaggregated Prefill/Decode

Aether separates compute-bound prefill from memory-bandwidth-bound decode:

- **Prefill scheduler**: processes all input tokens in parallel, chunked to maintain TTFT SLO.
- **Decode scheduler**: generates one token at a time with continuous batching.
- **KV transfer**: shared memory or RDMA between prefill and decode pools.

Enable with:

```python
RuntimeConfig(disaggregate_prefill_decode=True)
```

## Tree-Speculative Decoding

A draft model proposes a branching tree of candidate tokens. The target model verifies the entire tree in one forward pass using tree-masked attention.

Configuration:

```python
RuntimeConfig(
    speculative_decoding=True,
    speculative_tree_depth=4,
)
```

The draft model is auto-selected for common model families (e.g., Qwen3-72B uses Qwen3-1.5B as draft). If the acceptance rate drops below 70%, Aether falls back to standard decoding.

## Global KV Cache Manager

The KV cache is tiered across four storage levels:

| Tier | Storage | Use |
|------|---------|-----|
| L1 | GPU HBM | Active request blocks |
| L2 | CPU DRAM | Prefix cache, recently evicted blocks |
| L3 | NVMe SSD | Long system prompts, RAG documents |
| L4 | Aether Hub | Globally shared system prompts |

Prefix cache hits are tracked by a RadixTree over KV block hashes. Common system prompts and RAG documents are stored once and reused across sessions.

## Dynamic Precision Adjustment

Under memory pressure, the runtime downgrades the lowest-sensitivity layers to a lower precision in place, then restores them when pressure eases. This is automatic, transparent, and reversible.

```python
RuntimeConfig(dynamic_precision=True)
```

## Continuous Batching

The decode scheduler uses iteration-level admission (similar to vLLM/Orca) to maximize throughput while meeting latency SLOs.

## Configuration

```python
from aether import RuntimeConfig

config = RuntimeConfig(
    optimize_for="latency",
    speculative_decoding=True,
    prefill_chunk_size=2048,
    max_batch_size=256,
    kv_cache_dtype="fp8",
    kv_cache_cpu_gb=32,
    kv_cache_nvme_gb=200,
    dynamic_precision=True,
)
rt = Runtime(config)
```

## Metrics

The runtime exposes Prometheus-compatible metrics at `/v1/metrics`:

- `runtime_up`
- `loaded_models`
- `kv_cache_blocks`
- `kv_cache_hit_rate`
- `throughput_tps`
- `ttft_ms`
