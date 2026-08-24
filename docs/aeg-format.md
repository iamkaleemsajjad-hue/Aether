# AEG Format Specification v1.0

The **Aether Execution Graph (AEG)** is Aether's portable, versioned, content-addressed compiled model format. This document specifies the on-disk layout and stability guarantees.

## Overview

An AEG artifact is a directory (or compressed archive) containing:

- A versioned AEG-IR computation graph
- Mixed-precision quantized weights
- Per-target backend/kernel plans
- Pre-computed parallelism plans for 1/2/4/8 GPUs
- A top-level manifest with hashes and provenance

## Package Layout

```
my-model.aeg/
├── FORMAT_VERSION                    # e.g., "AEG/1.0"
├── manifest.json                     # Top-level metadata and hashes
├── graph/
│   ├── computation_graph.aeg-ir    # AEG-IR (text or binary)
│   ├── metadata.json                 # Model family, params, context length
│   └── graph.sha256                  # Hash of computation_graph.aeg-ir
├── weights/
│   └── quantized/
│       ├── precision_map.json        # Per-layer precision assignments
│       └── model.aeg-quant           # Mixed-precision weight blob
├── kernels/
│   ├── kernel_sets.json              # Target -> backend plan mapping
│   └── <target_id>/                  # Per-target kernel metadata (optional)
└── parallelism/
    ├── 1gpu.json                     # Single-GPU plan
    ├── 2gpu.json                     # 2-GPU plan
    ├── 4gpu.json                     # 4-GPU plan
    └── 8gpu.json                     # 8-GPU plan
```

## Versioning

AEG format versions follow semantic compatibility rules:

- **AEG/1.x** — Stable forever. All 1.x `.aeg` files are readable by all future Aether versions.
- **AEG/2.x** — When introduced, backward compatibility mode will be maintained for 3 years.

This mirrors LLVM IR's stability guarantee and makes AEG a reliable distribution format.

## Manifest Schema

```json
{
  "format_version": "AEG/1.0",
  "model_id": "Qwen/Qwen3-72B-Instruct",
  "aether_version": "1.2.5",
  "compiled_at": "2026-07-27T00:00:00Z",
  "graph_hash": "sha256:abc123...",
  "architecture": {
    "family": "qwen_family",
    "params_billion": 72.0,
    "layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_kv_heads": 8,
    "context_length": 131072,
    "vocab_size": 152064,
    "is_moe": false
  },
  "optimization": {
    "fusion_passes_applied": ["qkv_rope_norm", "ffn_swiglu"],
    "fused_ops_count": 156,
    "sensitivity_calibration_dataset": "wikitext-2",
    "quality_budget_ppl_increase": 0.02,
    "actual_ppl_increase": 0.018
  },
  "kernels": {
    "targets": ["cuda_sm90", "metal_m3", "cpu_avx512"],
    "backend_plans": {
      "cuda_sm90": "vllm",
      "metal_m3": "mlx",
      "cpu_avx512": "pytorch"
    }
  },
  "memory_requirements": {
    "bf16_gb": 144.0,
    "compiled_min_gb": 38.5,
    "recommended_gb": 48.0
  },
  "manifest_hash": "sha256:def456..."
}
```

## AEG-IR Format

AEG-IR is a textual/binary intermediate representation inspired by MLIR but specialized for transformer-family models. It preserves high-level semantics like GQA, SwiGLU, and RoPE through the optimizer passes.

Example:

```aeg-ir
func @transformer_layer(%x: tensor<*xbf16>, %pos: i64) -> tensor<*xbf16> {
  %norm = aeg.rmsnorm(%x, %weight[0]) {eps = 1e-6}
  %q, %k, %v = aeg.qkv_proj(%norm, %wq[0], %wk[0], %wv[0])
  %q_rope = aeg.rope(%q, %pos) {theta = 1000000.0}
  %k_rope = aeg.rope(%k, %pos) {theta = 1000000.0}
  %attn = aeg.gqa(%q_rope, %k_rope, %v) {
    num_heads = 64, num_kv_heads = 8, head_dim = 128,
    kv_cache = @global_kv_cache[layer=0]
  }
  %o_proj = aeg.linear(%attn, %wo[0])
  %residual = aeg.add(%x, %o_proj)
  %ffn = aeg.swiglu_ffn(%residual, %wg[0], %wu[0], %wd[0]) {sensitivity = LOW}
  return aeg.add(%residual, %ffn)
}
```

## Precision Map

`precision_map.json` assigns a precision format to each layer or weight group:

```json
{
  "embedding": "BF16",
  "layer_0": "FP8",
  "layer_1": "Q4_K_M",
  "layer_2": "Q3_K",
  "lm_head": "BF16"
}
```

Allowed formats: `BF16`, `FP16`, `FP8`, `Q8_0`, `Q6_K`, `Q4_K_M`, `Q4_0`, `Q3_K`, `IQ3_XS`, `Q2_K`, `INT4`, `INT8`.

## Content Addressing

Every AEG file is content-addressed using SHA-256. The `graph_hash` is the hash of `computation_graph.aeg-ir`. The `manifest_hash` is the hash of the manifest excluding the `manifest_hash` field. This enables:

- Integrity verification
- Deterministic caching
- Aether Hub deduplication
- Reproducible builds

## Stability Guarantees

1. The AEG/1.0 package layout is frozen.
2. AEG-IR v1.0 operations are backward-compatible within the 1.x line.
3. New optional fields may be added; unknown fields are ignored by older readers.
4. Required fields never change semantics within the same major version.

## Distribution

AEG packages may be distributed as directories or compressed tar archives (`.tar.gz`). The `.aeg` extension is reserved for both forms. Future AEG versions may introduce additional compression options.


## AEG v3.1 Extension Directories

The current package writer emits backward-compatible v3.1 extension directories while keeping the manifest format at AEG/1.x compatibility:

- `graph/reasoning_graph.aeg-ir`: compiled reasoning workflow metadata.
- `graph/rag_pipeline.aeg-ir`: RAG workflow graph contract.
- `graph/attention_head_patterns.json`: sparse attention head pattern plan.
- `weights/precision_map.json`: root precision map mirror for v3.1 readers.
- `weights/sparsity_masks.json`: pruning and sparse-kernel eligibility plan when available.
- `parallelism/prefill_decode_split.json`: disaggregated prefill/decode pool contract.
- `inference/compute_profiles.json`: greedy, beam, best-of-N, and MCTS cost profiles.
- `safety/`: prompt guard, output filter, and audit-log configuration.
- `provenance/`: model provenance manifest and fingerprint metadata.
- `watermark/`: SynthID-style statistical watermark configuration.
- `adapters/`: LoRA adapter manifest and slot configuration.
- `cuda_graphs/`: piecewise CUDA graph capture manifest.

Each emitted extension file is hashed into `manifest.artifacts` for integrity tracking.


## AEG v3.1 Platform Artifacts

Compiled packages now include deterministic manifests for the PRD v3.1 platform surface:

- `agentic/workflow_cache.json` stores meta-tool sequences, cascade routes, and context-cache policy.
- `observability/eval_gates.json`, `observability/drift_monitor.json`, and `observability/metrics_schema.json` define production gates and telemetry.
- `rollout/ab_config.json` stores deterministic A/B bucket assignment and rollback triggers.
- `fleet/deployment_plan.json` and `fleet/hot_reload.json` describe heterogeneous placement and candidate promotion/rollback.
- `distillation/plan.json` defines teacher/student modes, datasets, loss weights, and eval gates.
- `cuda_graphs/capture_manifest.json` records decode and prefill graph buckets plus persistent kernels.
- `mla/plan.json`, `speculation/eagle3.json`, and `multimodal/graph.json` cover MLA, EAGLE-3, and unified VLM/RAG workflows.
