# Research-backed execution contracts

This page records the equations that are implemented in the runtime. A PRD
statement is not treated as experimental evidence: a speedup is reported only
when a benchmark measures it on the selected hardware.

## Model-wide tensor partitioning

For a matrix `W` with `m` rows and `n` columns and `p` devices, device `i`
owns the contiguous interval

```text
s_i = floor(i m / p),  e_i = floor((i + 1) m / p),  W_i = W[s_i:e_i, :]
```

The intervals form a disjoint cover of `[0, m)`, and their sizes differ by at
most one. A row-parallel projection partitions the input dimension with the
same equation. Column-parallel outputs are concatenated; row-parallel outputs
are summed with an all-reduce. This is the inference form of tensor model
parallelism described by Shoeybi et al. and Narayanan et al. The implementation
also handles dimensions that are not divisible by GPU count; it never drops
remainder rows and never loads a full parameter replica on every GPU.

## Memory accounting

For `P` parameters represented at `b` bits, the weight lower bound is

```text
M_weights = P * b / 8
M_rank >= M_weights / p + M_KV + M_workspace
```

The `/p` term applies only to tensors actually sharded by the plan. Replicated
embeddings, normalization vectors, metadata, KV cache, and communication
workspace remain explicit additions. This prevents the planner from claiming
that every byte scales as `1/p`; it follows the non-redundant-state principle
from ZeRO (Rajbhandari et al.).

For a heterogeneous mesh, let `r_i` be the measured sustained GEMM rate of
device `i`. A contiguous weight interval is assigned with

```text
b_i = floor(N * (sum(j < i) r_j) / sum(j) r_j)
```

and `W_i = W[b_i:b_(i+1)]`. Aether obtains `r_i` from a short startup probe
only when CPU and accelerator devices are mixed; homogeneous GPU meshes use
equal partitions. If probing fails, it falls back to the lossless equal
partition rather than inventing vendor-specific FLOP conversions. This is a
placement heuristic, not a guarantee of linear speedup: transfers and
collective latency remain in the critical path.

## Attention and decode

Scaled dot-product attention uses

```text
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
```

The CPU path uses a numerically stable max-shifted softmax. GPU backends may
replace it with an IO-aware tiled kernel, but the AEG graph keeps the same
equation. FlashAttention (Dao et al.) motivates that replacement because the
dominant optimization is reducing high-bandwidth-memory traffic, not changing
the result.

## Quantization and claims

Weight-only quantization reduces the decode bandwidth term, but quality and
speed are model-, calibration-, and kernel-dependent. Aether therefore records
precision and calibration metadata in the artifact and does not claim a fixed
tokens/second multiplier. AWQ (Lin et al.) is the research basis for using
activation statistics to protect salient channels; it is not a license to
enable lossy quantization without a quality gate.

## Execution scope

The framework-free backend is validated for executable AEG artifacts and
supports the native CPU graph families covered by the loader. The portable
PyTorch path currently routes dense decoder AEGs through tensor parallelism;
specialized MLA, SSM, encoder, and sequence-to-sequence engines have separate
contracts. A model-family name in a registry is not treated as proof that every
checkpoint variant is executable. Unsupported graph operators, missing
weights, unavailable vendor runtimes, and failed quality gates fail closed.

## Primary references

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
