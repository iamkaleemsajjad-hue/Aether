# Aether Optimizer Passes

Aether's Stage 2 optimizer runs six graph-level compiler passes. Each pass is a self-contained transformation that can be enabled or disabled independently.

## Pass 1: Operator Fusion

Merges sequences of operations into single GPU "megakernels" to reduce kernel launches and eliminate intermediate DRAM round-trips.

Typical fusion patterns:

- `RMSNorm → QKV projection → RoPE` → `fused_qkv_rope_norm`
- `Attention → output projection → residual add` → `fused_attn_out_residual`
- `FFN gate/up/down projections + residual` → `fused_ffn_residual`

## Pass 2: Sensitivity Analysis

Computes a per-layer sensitivity score:

```
sensitivity[L] = Δperplexity / bits_saved(L)
```

Higher sensitivity means the layer is more important to preserve precision. This is the mathematical basis for mixed-precision quantization.

## Pass 3: Precision Assignment

Using the sensitivity map, each layer receives an optimal precision under the quality budget:

| Sensitivity | Precision |
|-------------|-----------|
| > 0.9 | BF16 |
| 0.7–0.9 | FP8 or Q6_K |
| 0.4–0.7 | Q4_K_M |
| < 0.4 | Q3_K or IQ3_XS |

Modes:

- `sensitivity` — default, uses the sensitivity map
- `uniform` — same precision for all layers
- `manual` — user-provided precision map

## Pass 4: KV Cache Structuring

Annotates the AEG-IR with explicit KV cache nodes and policies:

- Paged blocks aligned to hardware page size
- RadixTree prefix hints for shared system prompts
- Memory tier thresholds: GPU HBM → CPU DRAM → NVMe SSD → Aether Hub
- Cross-session sharing policies

## Pass 5: MoE Expert Routing

For MoE models, Aether:

1. Profiles expert activation on a calibration set.
2. Classifies experts as hot (>5%), warm (0.1–5%), or cold (<0.1%).
3. Pins hot experts to GPU HBM, stages warm experts to CPU + prefetch, and lazy-loads cold experts from NVMe.
4. Replaces rigid top-K routing with adaptive threshold-based routing (DynaMoE).
5. Emits intra-expert sparsity kernels to skip dead activation channels.

## Pass 6: Automatic Parallelism Discovery

Searches over the strategy space of tensor, pipeline, expert, and context parallelism. Produces separate prefill and decode sharding plans for 1/2/4/8 GPU configurations. At runtime, the correct plan loads automatically with zero user configuration.

Search space:

- Tensor parallelism degree: 1, 2, 4, 8
- Pipeline stages: 1, 2, 4
- Expert parallelism degree: 1, 2, 4 (MoE only)
- Context parallelism degree: 1, 2, 4 (long context only)

## Configuration

Passes are controlled by `CompilerConfig`:

```python
from aether import CompilerConfig

config = CompilerConfig(
    enable_fusion=True,
    enable_sensitivity=True,
    enable_precision_assignment=True,
    enable_kv_cache_structuring=True,
    enable_moe_routing=True,
    enable_parallelism_discovery=True,
)
```

Or via environment variables:

```bash
export AETHER_OPTIMIZATION_LEVEL=2
```

## Research Foundation

- Operator fusion: ClusterFusion (NeurIPS 2025), FlashAttention-3, TensorRT-LLM
- Sensitivity: AutoMixQ, AMQ, GPTQ, AWQ
- KV cache: PagedAttention, SGLang RadixAttention, DistServe, Mooncake
- MoE: MoE-Infinity, CommitMoE, FinDEP, DynaMoE
- Parallelism: Alpa, Megatron-LM, Seesaw, Ring Attention

See [research.md](research.md) for the full paper mapping.


## Pass 7: Reasoning Graph Compiler

The reasoning graph pass emits `graph/reasoning_graph.aeg-ir` with explicit budget, early-exit, verification, and speculative-CoT metadata. Runtime implementations can execute this as a first-class workflow instead of treating reasoning as unstructured text generation.

## Pass 8: Sparse Attention

The sparse attention pass emits `graph/attention_head_patterns.json` with MInference-style vertical-slash, block-sparse, and A-shape head patterns. The plan activates only above the configured long-context threshold and retains dense attention as a fallback.

## Pass 9: Pruning and Sparsity

The pruning pass emits `weights/sparsity_masks.json` using a Wanda/SparseGPT-inspired importance policy. Sensitive layers receive lower sparsity, while eligible linear, QKV, FFN, and expert nodes receive 2:4 or unstructured mask metadata.
