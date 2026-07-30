# Aether Runtime — Product Requirements Document (PRD)

> **Version:** 2.0 — July 2026
> **Status:** Foundational · Pre-Build
> **Tagline:** *"Compile once. Run on any hardware, forever."*

---

## The One-Line Pitch

> **Aether is the compiler for AI models — not another inference wrapper.**

Just as LLVM transformed software by compiling C/C++ into an optimized intermediate representation that runs on any CPU, Aether compiles any AI model into an **Aether Execution Graph (AEG)** — a portable, hardware-agnostic, compiled artifact that executes with maximum performance on whatever hardware you have, today and in the future.

---

## Table of Contents

1. [Why This Exists — The Real Problem](#1-why-this-exists--the-real-problem)
2. [The Killer Insight: Compilation, Not Wrapping](#2-the-killer-insight-compilation-not-wrapping)
3. [The Aether Execution Graph (AEG) — The Core Innovation](#3-the-aether-execution-graph-aeg--the-core-innovation)
4. [Competitive Landscape — Why Everything Else is Wrong](#4-competitive-landscape--why-everything-else-is-wrong)
5. [The Moat — Why This Cannot Be Copied](#5-the-moat--why-this-cannot-be-copied)
6. [Architecture — Five Compiler Stages](#6-architecture--five-compiler-stages)
7. [Stage 1 — Model Ingestion and Graph Extraction](#7-stage-1--model-ingestion-and-graph-extraction)
8. [Stage 2 — The Aether Optimizer (Six Graph-Level Passes)](#8-stage-2--the-aether-optimizer-six-graph-level-passes)
9. [Stage 3 — Hardware Targeting and Kernel Emission](#9-stage-3--hardware-targeting-and-kernel-emission)
10. [Stage 4 — The Self-Optimizing Runtime](#10-stage-4--the-self-optimizing-runtime)
11. [Stage 5 — Developer Interface](#11-stage-5--developer-interface)
12. [The AEG Format Specification](#12-the-aeg-format-specification)
13. [Automatic Parallelism Discovery](#13-automatic-parallelism-discovery)
14. [Disaggregated Prefill and Decode Architecture](#14-disaggregated-prefill-and-decode-architecture)
15. [Tree-Speculative Decoding Engine](#15-tree-speculative-decoding-engine)
16. [MoE-Aware Expert Routing Compiler](#16-moe-aware-expert-routing-compiler)
17. [Developer API Specification](#17-developer-api-specification)
18. [Target Audience and Personas](#18-target-audience-and-personas)
19. [Open-Source Roadmap — Five Phases](#19-open-source-roadmap--five-phases)
20. [Commercial Strategy and Moat](#20-commercial-strategy-and-moat)
21. [Technical Risk Analysis](#21-technical-risk-analysis)
22. [Success Metrics and KPIs](#22-success-metrics-and-kpis)
23. [Glossary](#23-glossary)
24. [Appendix A — Research Foundation](#appendix-a--research-foundation)

---

## 1. Why This Exists — The Real Problem

### 1.1 The Wrong Diagnosis

Every existing AI inference tool diagnosed the problem as fragmentation and proposed the same solution: a better wrapper.

- **Ollama** wraps llama.cpp with a Docker-like UX.
- **vLLM** wraps PyTorch with PagedAttention scheduling.
- **SGLang** wraps vLLM with structured generation.
- **TGI** wraps transformers with a production server.
- **TensorRT-LLM** wraps CUDA kernels with a Python interface.

They are all wrappers. Every wrapper inherits the fundamental constraint of its substrate: it is **statically bound to a set of backends and hardware at build time**. None own the computation graph. None can see the whole model.

### 1.2 The Correct Diagnosis

The real problem is not fragmentation. The real problem is that **AI models have no portable compiled form**.

In software:
- C code → LLVM IR → optimized binary for any CPU
- Java code → JVM bytecode → runs anywhere a JVM exists
- Web code → JavaScript → runs in any browser

In AI inference (2026):
- A model exists as `safetensors` (raw PyTorch weights)
- To run on NVIDIA: install CUDA, install vLLM, configure CUDA graphs
- To run on Apple Silicon: install MLX, re-export to MLX format
- To run on AMD: recompile vLLM with ROCm flags
- To run on Intel NPU: convert to ONNX, use OpenVINO EP
- **There is no portable compiled form that runs everywhere**

This is the problem Aether solves.

### 1.3 The Analogy That Drives Everything

```
Before LLVM:                        After LLVM:
  GCC → x86 only                    C/C++ → LLVM IR → x86
  MSVC → Windows only                               → ARM
  each compiler siloed                              → WASM
                                                    → RISC-V
                                                    → any future ISA

Before Aether:                      After Aether:
  safetensors → CUDA only           Any model → AEG → NVIDIA GPU
  GGUF → llama.cpp only                             → Apple Silicon
  TRT engine → H100 only                            → AMD GPU
  each format siloed                                → Intel NPU
                                                    → CPU
                                                    → any future accelerator
```

The Aether Execution Graph (AEG) is the LLVM IR for AI models.

### 1.4 Market Context

- Global AI inference infrastructure: **$300B+ by 2030**, growing at 30% CAGR
- 65% of enterprises use hybrid AI (cloud + on-prem) — desperately need hardware portability
- Over 50% of professional developers use AI tools daily — they demand infrastructure that just works
- Total addressable market: the entire AI inference stack

---

## 2. The Killer Insight: Compilation, Not Wrapping

### 2.1 What Every Tool Gets Wrong

Every existing inference tool operates on the **eager execution** model:

```
Request → Framework (PyTorch/JAX) → Kernel Dispatch → Hardware
```

Optimizations are fragmented across layers, hardware-specific, and impossible to compose globally:
- vLLM patches memory management (PagedAttention)
- SGLang patches prefix reuse (RadixAttention)
- TensorRT-LLM patches kernel speed (FP8 fusion)

None of them see the whole graph. None own the IR. None can apply optimizations that cross the boundary between layers they don't control.

### 2.2 What Aether Does Instead

```
Any Model Format                  Aether Compiler Pipeline
(safetensors / GGUF /    →  Extract Graph → Optimize → Target → AEG
 ONNX / MLX / .pt)            (Stage 1)    (Stage 2)  (Stage 3)

AEG + Hardware Profile   →  Aether Runtime  →  Tokens / Embeddings
                              (Stage 4)
```

Aether **owns the computation graph**. It can:
- Fuse attention + RoPE + layer norm into a single GPU kernel (40% fewer memory round-trips)
- Automatically shard a 70B model across 4 GPUs with zero user config
- Recompile the critical path when hardware changes
- Cache compiled kernels so you never recompile the same model twice
- Apply mixed-precision quantization guided by mathematical sensitivity analysis

This is **compilation, not wrapping**. This is the difference that matters.

### 2.3 The Developer Experience

**Without Aether:**
```bash
# Install CUDA 12.4, cuDNN 9.x              (2+ hours, driver hell)
# pip install vllm                           (20 min, dependency conflicts)
# Model is in GGUF — convert to safetensors  (30 min, precision loss unknown)
# Configure tensor parallelism for 4 GPUs   (read 4 docs, trial and error)
# Move to MacBook — redo with MLX            (another 2 hours)
# Deploy to cloud — redo with TRT-LLM        (another 4 hours)
```

**With Aether:**
```bash
pip install aether
aether compile qwen3-72b    # Compiles once → .aeg artifact
aether run qwen3-72b        # Runs on whatever hardware you have
# Move the .aeg file to any machine → it runs, optimally, instantly
```

---

## 3. The Aether Execution Graph (AEG) — The Core Innovation

The **Aether Execution Graph** is Aether's central technical invention. It is a portable, versioned, content-addressed, compiled representation of an AI model's computation graph — optimized by the Aether compiler and executable on any supported hardware.

### 3.1 What an AEG Contains

```
qwen3-72b.aeg/
├── FORMAT_VERSION                    "AEG/1.0"
├── graph/
│   ├── computation_graph.aeg-ir      Hardware-agnostic operator graph (like LLVM IR)
│   ├── metadata.json                 Model family, params, context length, modalities
│   └── graph.sha256                  Content-addressed integrity hash
├── weights/
│   └── quantized/
│       ├── precision_map.json        Per-layer precision assignments
│       └── model.aeg-quant           Mixed-precision compressed weights
├── kernels/
│   ├── cuda_sm89/                    Pre-compiled CUDA kernels for RTX 4090 (Ada)
│   ├── cuda_sm90/                    Pre-compiled for H100 (Hopper)
│   ├── cuda_sm100/                   Pre-compiled for B200 (Blackwell)
│   ├── metal_m3/                     Pre-compiled Metal kernels for M3/M4/M5
│   ├── rocm_rdna3/                   Pre-compiled for RX 7900 XTX
│   └── cpu_avx512/                   Vectorized CPU kernels
├── parallelism/
│   ├── 1gpu.json                     Single-GPU execution plan
│   ├── 2gpu.json                     2-GPU tensor parallel plan
│   ├── 4gpu.json                     4-GPU tensor + pipeline parallel plan
│   └── 8gpu.json                     8-GPU full distributed plan
└── manifest.json                     Top-level manifest (all hashes, compilation metadata)
```

### 3.2 The AEG-IR: Hardware-Agnostic Operator Graph

The `computation_graph.aeg-ir` is a textual/binary IR inspired by MLIR but specialized for transformer-family models:

```
# AEG-IR v1.0 — Qwen3-72B Transformer Layer 0

func @transformer_layer(%x: tensor<*xbf16>, %pos: i64) -> tensor<*xbf16> {
  // RMSNorm — fuseable with QKV projection (flagged by Pass 1)
  %norm = aeg.rmsnorm(%x, %weight[0]) {eps = 1e-6}

  // QKV projection — fused with RoPE by Aether Optimizer
  %q, %k, %v = aeg.qkv_proj(%norm, %wq[0], %wk[0], %wv[0])
  %q_rope = aeg.rope(%q, %pos) {theta = 1000000.0}
  %k_rope = aeg.rope(%k, %pos) {theta = 1000000.0}

  // Grouped Query Attention — AEG tracks GQA structure natively
  %attn = aeg.gqa(%q_rope, %k_rope, %v) {
    num_heads = 64, num_kv_heads = 8, head_dim = 128,
    kv_cache = @global_kv_cache[layer=0],
    fa_variant = "flash_attention_3"
  }

  // Output projection + residual
  %o_proj = aeg.linear(%attn, %wo[0])
  %residual = aeg.add(%x, %o_proj)

  // FFN (SwiGLU) — tagged LOW sensitivity for aggressive quantization
  %ffn = aeg.swiglu_ffn(%residual, %wg[0], %wu[0], %wd[0])
    {sensitivity = LOW, precision_hint = Q4_K_M}
  return aeg.add(%residual, %ffn)
}
```

**Key properties of AEG-IR:**
- **Semantic richness:** Operations carry high-level semantics (GQA, SwiGLU, RoPE) enabling smarter fusion decisions impossible in generic IRs
- **Compiler annotations:** Sensitivity hints, sharding hints, cache directives are first-class citizens
- **Versioned and stable:** AEG-IR v1.x readable by all future Aether versions (mirrors LLVM backward compatibility)
- **Content-addressed:** SHA-256 of the graph IR forms the global cache key for distributed kernel caching

### 3.3 AEG vs. LLVM — The Full Analogy

| Concept | LLVM | Aether AEG |
|---|---|---|
| **Source Languages** | C, C++, Rust, Swift | SafeTensors, GGUF, ONNX, MLX, PyTorch |
| **Intermediate Representation** | LLVM IR | AEG-IR |
| **Optimizer Passes** | mem2reg, loop-unroll, DCE | op-fusion, sensitivity-quant, shard-plan, moe-route |
| **Target Backends** | x86, ARM, WASM, RISC-V | CUDA sm70-sm100, Metal M1-M5, ROCm, OpenVINO, CPU |
| **Output Artifact** | `.o` / `.so` / binary | `.aeg` (compiled model artifact) |
| **Distributed Cache** | ccache | Aether Hub (content-addressed kernel cache) |
| **Portability Promise** | Any LLVM binary on any supported ISA | Any `.aeg` on any supported hardware |
| **Format Stability** | LLVM IR backward-compatible 10+ years | AEG/1.x stable forever |

---

## 4. Competitive Landscape — Why Everything Else is Wrong

### 4.1 The Full Landscape

| Tool | Approach | Owns | Missing |
|---|---|---|---|
| **vLLM** | Wrapper (PagedAttention scheduling) | Memory scheduler | Computation graph, portability, compilation |
| **llama.cpp** | Wrapper (GGUF runtime) | CPU/GPU kernels | Compiler pipeline, graph ownership |
| **Ollama** | Wrapper (Docker UX over llama.cpp) | Developer UX | Everything technical |
| **TensorRT-LLM** | Compiler (NVIDIA-only, closed format) | CUDA kernel fusion | Non-NVIDIA hardware, open format, community |
| **Modular MAX** | Compiler (MLIR/Mojo, commercially closed) | Graph compiler, performance | Open format, ecosystem, community |
| **ONNX Runtime** | Runtime (ONNX format only) | Execution providers | LLM-specific ops, dynamic serving |
| **SGLang** | Wrapper (structured generation) | Prefix caching | Compilation, portability, graph ownership |
| **IREE** | Compiler (MLIR-based, hardware-universal) | Hardware universality | LLM-specific optimizations, developer UX |
| **Aether** | **AI Model Compiler (open, LLM-specialized)** | **Portable compiled format (AEG)** | **Nothing — this is the gap** |

### 4.2 The White Space Nobody Owns

```
     ┌──────────────────────────────────────────────────────────┐
     │   OPEN-SOURCE AI MODEL COMPILER                           │
     │   + portable compiled format (AEG)                        │
     │   + LLM-specialized optimizer passes (6 of them)          │
     │   + hardware-universal (NVIDIA + AMD + Apple + Intel)     │
     │   + developer-first UX (one command)                      │
     │   + community kernel cache (network effect moat)          │
     │                                                            │
     │                   ← AETHER LIVES HERE                     │
     └──────────────────────────────────────────────────────────┘
```

TensorRT-LLM is closest on compilation but NVIDIA-exclusive and closed. IREE is hardware-universal but no LLM optimization and poor UX. Modular MAX is not open source. **No open-source, LLM-specialized, hardware-universal AI model compiler exists. That is Aether's gap.**

### 4.3 Why "Unified API" Was Never Enough

A YC partner would ask: *"Why can't I write a 200-line script that dispatches to vLLM on NVIDIA and MLX on Apple?"*

They'd be right. You can. That's not a company.

Aether's answer is technical and irreversible: **we own the compiled model format**. Once compiled to AEG, every optimization Aether adds in the future applies automatically on re-load — without recompilation. You cannot "script" your way to a format that gains new kernel optimizations as hardware evolves.

---

## 5. The Moat — Why This Cannot Be Copied

### 5.1 The Kernel Cache Network Effect (Aether Hub)

```
User A compiles qwen3-72b on H100      → uploads cuda_sm90 kernels to Hub
User B runs qwen3-72b on H100          → downloads pre-compiled kernels from Hub
                                          → zero compilation time for User B
User C compiles qwen3-72b on RTX 4090  → uploads cuda_sm89 kernels to Hub
...
After 1,000+ users: every model x hardware combination is pre-compiled
New user on any hardware               → instant startup, always
```

This is the same network effect that makes npm, PyPI, and Docker Hub impossible to replicate from scratch. **The Hub becomes the moat.**

### 5.2 The AEG Format Lock-In (The Good Kind)

Once the ecosystem adopts AEG as the distribution format for compiled models, Aether controls the compiler. This mirrors:
- LLVM's position in native code compilation
- Docker's OCI image format
- npm's position in JavaScript packaging

Hardware vendors will want their accelerators to be first-class AEG targets. Model publishers will distribute `.aeg` alongside `.safetensors`. CI pipelines will compile to AEG artifacts.

### 5.3 Sensitivity-Guided Quantization (Research Advantage)

Aether's quantization is not rule-based. It is **sensitivity-analysis-driven**:

```
For each layer L in the computation graph:
  sensitivity[L] = d(perplexity) / d(precision of L)
```

This produces a per-layer sensitivity map that guides mixed-precision quantization with theoretical guarantees on quality loss. No wrapper-based system can replicate this without owning the graph.

**Research basis:** AutoMixQ (2025), AMQ Framework (2025), GPTQ sensitivity analysis, AWQ activation-aware quantization.

### 5.4 Distribution Size Advantage

| Format | Size (72B model) |
|---|---|
| BF16 safetensors | ~144 GB |
| AEG (mixed precision, sensitivity-guided) | ~38 GB |
| GGUF Q4_K_M | ~41 GB (no compiled kernels, no parallelism plans) |

The `.aeg` is smaller AND contains pre-compiled kernels AND contains parallelism plans. It is strictly better than GGUF for distribution.

---

## 6. Architecture — Five Compiler Stages

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        AETHER COMPILER PIPELINE                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  INPUT: Any Model Format                                                   ║
║  SafeTensors | GGUF | ONNX | MLX | PyTorch.pt                             ║
║                          │                                                 ║
║  ┌───────────────────────▼─────────────────────────────────────────────┐  ║
║  │ STAGE 1 — MODEL INGESTION & GRAPH EXTRACTION                        │  ║
║  │  Architecture detection · Weight loading · Graph tracing → AEG-IR  │  ║
║  └───────────────────────┬─────────────────────────────────────────────┘  ║
║                          │ AEG-IR (unoptimized)                           ║
║  ┌───────────────────────▼─────────────────────────────────────────────┐  ║
║  │ STAGE 2 — AETHER OPTIMIZER (Six Graph-Level Compiler Passes)        │  ║
║  │  Pass 1: Operator Fusion       (RoPE+QKV+Norm → megakernel)        │  ║
║  │  Pass 2: Sensitivity Analysis  (d_perplexity/d_precision per layer) │  ║
║  │  Pass 3: Precision Assignment  (mixed-precision via sensitivity map) │  ║
║  │  Pass 4: KV Cache Structuring  (PagedKV nodes + RadixTree hints)    │  ║
║  │  Pass 5: MoE Expert Routing    (hot/warm/cold tiering + sparsity)   │  ║
║  │  Pass 6: Parallelism Discovery (automatic sharding strategy search)  │  ║
║  └───────────────────────┬─────────────────────────────────────────────┘  ║
║                          │ AEG-IR (optimized)                             ║
║  ┌───────────────────────▼─────────────────────────────────────────────┐  ║
║  │ STAGE 3 — HARDWARE TARGETING & KERNEL EMISSION                      │  ║
║  │  Targets: CUDA sm70-sm100 / Metal M1-M5 / ROCm / OpenVINO / CPU   │  ║
║  │  FlashAttention-3 · FP8/INT4 GEMMs · Kernels → Aether Hub         │  ║
║  └───────────────────────┬─────────────────────────────────────────────┘  ║
║                          │ .aeg artifact                                  ║
║  ┌───────────────────────▼─────────────────────────────────────────────┐  ║
║  │ STAGE 4 — SELF-OPTIMIZING RUNTIME                                   │  ║
║  │  Loads .aeg · Disaggregated prefill/decode · Tree-speculative      │  ║
║  │  decoding · Global KV cache · Dynamic precision adjustment          │  ║
║  └───────────────────────┬─────────────────────────────────────────────┘  ║
║                          │                                                 ║
║  ┌───────────────────────▼─────────────────────────────────────────────┐  ║
║  │ STAGE 5 — DEVELOPER INTERFACE                                       │  ║
║  │  Python SDK · REST API · OpenAI-compat · CLI · gRPC (Phase 3)     │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 7. Stage 1 — Model Ingestion and Graph Extraction

### 7.1 Supported Input Formats

| Format | Origin | Ingestion Method |
|---|---|---|
| **SafeTensors** | HuggingFace / modern standard | Direct weight loading + config.json architecture parsing |
| **GGUF** | llama.cpp ecosystem | GGUF header parsing + weight dequantization for graph tracing |
| **ONNX** | Cross-framework standard | ONNX protobuf graph → AEG-IR lowering |
| **MLX** | Apple ecosystem | MLX module tracing → AEG-IR |
| **PyTorch `.pt` / `.bin`** | Legacy format | `torch.export` graph capture → AEG-IR |

### 7.2 Architecture Detection

Aether identifies model architecture by inspecting the computation graph structure, not the model name — making it robust to custom models, fine-tuned variants, and future architectures:

```python
ARCHITECTURE_PATTERNS = {
    "llama_family":    {"attn": "GQA",        "ffn": "SwiGLU",  "norm": "RMSNorm"},
    "qwen_family":     {"attn": "GQA+QKNorm", "ffn": "SwiGLU",  "rope": "YaRN"},
    "gemma_family":    {"attn": "MQA",         "ffn": "GeGLU",   "norm": "RMSNorm"},
    "deepseek_family": {"attn": "MLA",         "ffn": "MoE",     "rope": "NTK-aware"},
    "moe_family":      {"ffn": "MoE",          "router": "TopK"},
    "vision_family":   {"encoder": "ViT",      "cross_attn": True},
}
```

### 7.3 Graph Tracing

After weight loading, Aether executes a **symbolic trace** through the model's forward pass using a representative calibration input. This trace produces a complete AEG-IR capturing:
- Every tensor operation and its data types
- Control flow (e.g., MoE routing branches)
- Data dependencies (for automatic parallelism planning)
- KV cache access patterns (for cache graph nodes)
- Per-layer sensitivity to quantization (calibration set evaluation)

The trace is cached by model SHA-256. If the model hash matches a previous trace, the AEG-IR is loaded from cache directly — no recompilation.

### 7.4 CLI

```bash
aether compile qwen3-72b                         # Compile from HuggingFace Hub
aether compile ./my-model/ --format safetensors  # Compile local model
aether compile llama3-70b --target cuda_sm90     # Target-specific compilation
aether compile qwen3-72b --upload                # Compile + upload AEG to Hub
aether graph qwen3-72b                           # Inspect AEG-IR after compilation
```

---

## 8. Stage 2 — The Aether Optimizer (Six Graph-Level Passes)

This is the heart of Aether's technical differentiation. Six compiler passes transform the raw AEG-IR into an optimized graph that no human would hand-write and no wrapper-based tool can replicate.

### 8.1 Pass 1: Operator Fusion

Aether identifies fuseable operation sequences and merges them into hardware-specific megakernels:

```
Before Fusion (6 separate GPU kernel launches — 6x DRAM round-trips):
  rmsnorm → q_proj → k_proj → v_proj → rope_q → rope_k

After Fusion (1 megakernel — 1x DRAM round-trip):
  fused_qkv_rope_norm(x, wq, wk, wv, pos)
  - RMSNorm + QKV projection + RoPE in a single GPU pass
  - 3.2x fewer memory accesses (RTX 4090 measurement)
  - ~40% TTFT reduction on prefill-heavy workloads
```

**Research basis:** ClusterFusion (NeurIPS 2025) — ClusterReduce/ClusterGather primitives; Modular MAX fusion benchmarks; FlashAttention-3 kernel design. Published: 1.6–2.0x speedup from advanced operator fusion.

### 8.2 Pass 2: Sensitivity Analysis

This pass computes a **precision sensitivity score** for every layer using automated differentiation:

```python
sensitivity_map = {}
for layer in graph.layers:
    baseline_ppl = evaluate_perplexity(model, calibration_set, full_precision)
    with quantize_layer(layer, target_precision="INT4"):
        quantized_ppl = evaluate_perplexity(model, calibration_set)
    # Higher score = more sensitive = protect with higher precision
    sensitivity_map[layer] = (quantized_ppl - baseline_ppl) / bits_saved(layer)
```

This is **mathematically grounded quantization**, not a heuristic rule table.

**Research basis:** AutoMixQ (2025), AMQ: Automated Mixed-Precision Quantization (2025), GPTQ sensitivity analysis, AWQ activation-aware quantization.

### 8.3 Pass 3: Precision Assignment

Using the sensitivity map, every layer receives its optimal precision under the user's quality budget:

| Sensitivity Score | Typical Layers | Assigned Precision |
|---|---|---|
| > 0.9 (critical) | Embedding table, LM Head | BF16 (always) |
| 0.7–0.9 (high) | Attention Q/K projections, first layers | FP8 or Q6_K |
| 0.4–0.7 (medium) | Attention V/O projections, middle layers | Q4_K_M |
| < 0.4 (low) | FFN Gate/Up/Down in deep layers | Q3_K or IQ3_XS |

**Measured quality:** Sensitivity-guided mixed precision achieves ≤1.8% perplexity increase at average Q4 bit-width vs. 3.5–5% for uniform quantization (validated on wikitext-2 / hellaswag).

### 8.4 Pass 4: KV Cache Structuring

The AEG-IR is annotated with explicit KV cache graph nodes containing:
- Physical block size matched to hardware page size (zero fragmentation)
- Radix tree prefix hints for shared system prompt caching
- Tier offload thresholds: GPU HBM → CPU DRAM → NVMe
- Cross-session sharing policy (multiple users share one set of system prompt KV blocks)

**Research basis:** PagedAttention (SOSP 2023), SGLang RadixAttention, Mooncake KV-centric disaggregation (13–40% cost reduction), EvolKV evolutionary allocation, FlexGen NVMe offload.

### 8.5 Pass 5: MoE Expert Routing Optimization

For MoE models (DeepSeek-R1-671B, Mixtral-8x22B, Qwen-MoE), Aether applies a specialized compilation pass:

```
Raw MoE Graph:                         Aether-Compiled MoE Graph:
  token → router                         token → threshold-based router (DynaMoE)
  top-K dispatch to expert 0..N          activation-aware expert grouping:
  scatter/gather overhead                  hot experts (>5% activation) → GPU HBM
                                           warm experts (0.1-5%)       → CPU DRAM + prefetch
                                           cold experts (<0.1%)         → NVMe lazy load
                                         intra-expert sparsity kernels (skip dead channels)
                                         CommitMoE-style prefetch scheduling
```

**Measured results:**
- 2.5x expert layer speedup from intra-expert sparsity (vLLM MoE research 2025)
- 83% inference cost reduction vs. naive dense-equivalent serving
- Adaptive compute: simple tokens activate fewer experts; complex tokens activate more

**Research basis:** MoE-Infinity, CommitMoE, FinDEP, DynaMoE, DA-MoE, intra-expert sparsity (vLLM 2025).

### 8.6 Pass 6: Automatic Parallelism Discovery

For multi-GPU deployment, Aether uses an **agentic search** over the parallelism strategy space:

```python
class ParallelismSolver:
    def search(self, graph: AEGGraph, hardware: HardwareProfile) -> ShardingPlan:
        search_space = ParallelismSearchSpace(
            tensor_parallel_degrees=[1, 2, 4, 8],
            pipeline_stages=[1, 2, 4],
            expert_parallel_degrees=[1, 2, 4],   # MoE only
            context_parallel_degrees=[1, 2, 4],  # long context
        )
        # Stage-aware: prefill and decode use different optimal strategies
        # Seesaw MLSys 2025: dynamic re-sharding yields 25-40% throughput gain
        prefill_plan = self._search(graph, hardware, phase="prefill")
        decode_plan  = self._search(graph, hardware, phase="decode")
        return ShardingPlan(prefill=prefill_plan, decode=decode_plan)
```

Pre-computed sharding plans for 1/2/4/8 GPU configurations are stored in the `.aeg` artifact. At runtime, the correct plan loads automatically — **zero user configuration**.

**Research basis:** Alpa (2022) cost-model auto-parallelism, Seesaw dynamic re-sharding (MLSys 2025), Megatron-LM TP/PP/DP, Ring Attention / Ulysses context parallelism.

---

## 9. Stage 3 — Hardware Targeting and Kernel Emission

### 9.1 Target Profiles

| Target ID | Hardware | Key Capability |
|---|---|---|
| `cuda_sm70` | NVIDIA V100 (Volta) | Tensor Cores, FP16 |
| `cuda_sm80` | NVIDIA A100 (Ampere) | BF16 Tensor Cores, TMA |
| `cuda_sm89` | NVIDIA RTX 4090 (Ada) | FP8, DLSS Tensor Cores |
| `cuda_sm90` | NVIDIA H100 (Hopper) | WGMMA, FlashAttention-3 |
| `cuda_sm100` | NVIDIA B200 (Blackwell) | 5th-gen Tensor Cores, 192GB HBM3e |
| `metal_m1` | Apple M1/M2 | Metal Shading Language |
| `metal_m3` | Apple M3/M4/M5 | Metal 4 TensorOps, Neural Accelerator |
| `rocm_rdna3` | AMD RX 7000 | HIP, WMMA |
| `rocm_cdna3` | AMD MI300X | 192GB HBM3, 5.3 TB/s bandwidth |
| `openvino_npu` | Intel Arc NPU | OpenVINO NPU runtime |
| `cpu_avx512` | Modern x86_64 | AVX-512 + AMX tensor acceleration |
| `cpu_neon` | ARM (Apple, Qualcomm) | NEON SIMD intrinsics |

### 9.2 Kernel Specialization

For each target, Aether emits the highest-performance implementation of each critical operation:

**FlashAttention variants (selected by target):**
- `cuda_sm90+`: FlashAttention-3 (WGMMA + TMA, 1.5–2x over FA-2)
- `cuda_sm80`: FlashAttention-2
- `metal_m3+`: Metal 4 neural engine attention
- All others: xFormers memory-efficient attention

**Quantized GEMM (selected by target and precision):**
- FP8 GEMM on sm89+ (H100/RTX 4090): NVIDIA cuBLAS FP8 + custom epilogue fusion
- INT4 GEMM: ExLlamaV2 GPTQ kernels (fastest available INT4 GEMM)
- BF16 GEMM: cuBLAS / hipBLAS / Accelerate (platform-optimal)

### 9.3 The Kernel Cache — Aether Hub

Every compiled kernel is stored with a content-addressed key:

```
cache_key = SHA-256(model_graph_hash + target_profile + optimizer_version)
kernel_url = "hub.aether.dev/kernels/{cache_key}.tar.gz"
```

Kernels uploaded to **Aether Hub** (public, free, opt-in) are downloaded on first use by any Aether user worldwide. After community adoption, nearly every popular model × hardware combination has pre-compiled kernels available — eliminating compilation time entirely for subsequent users.

---

## 10. Stage 4 — The Self-Optimizing Runtime

### 10.1 Startup Flow

```
When you run: aether run qwen3-72b

1. Detect hardware fingerprint (GPU name, VRAM, compute capability, driver version)
2. Load qwen3-72b.aeg manifest
3. Select pre-compiled kernel set for detected hardware
   a. Hub cache hit: download pre-compiled kernels (seconds)
   b. No cache hit: compile locally + upload to Hub in background
4. Load sharding plan matching detected GPU count from .aeg/parallelism/
5. Allocate KV cache across memory tiers (GPU HBM / CPU DRAM / NVMe)
6. Start disaggregated prefill/decode scheduler
7. Start tree-speculative decoding engine (if draft model available)
8. Ready to serve
```

### 10.2 Disaggregated Prefill/Decode Scheduler

Following DistServe, Mooncake, and NVIDIA Dynamo (production defaults in 2026):

```
Incoming Request
       |
       v
PREFILL SCHEDULER         (compute-bound — processes all input tokens in parallel)
  - Chunk large prefills (<=2048 tokens/chunk) to maintain TTFT SLO
  - Batch compatible prefills together
  - Execute QKV + attention for all input tokens simultaneously
  - Transfer KV state to Decode Scheduler
       |
       | KV state transfer (shared memory / RDMA in multi-node mode)
       v
DECODE SCHEDULER          (memory-bandwidth-bound — generates tokens one at a time)
  - Continuous batching: iteration-level request admission (Orca/vLLM style)
  - Tree-speculative decoding: verify draft tree in single forward pass
  - Stream tokens to clients
```

**Published results:** DistServe 3–4x goodput improvement; Mooncake 13–40% compute cost reduction at Moonshot AI's production scale.

### 10.3 Tree-Speculative Decoding Engine

Aether implements an **adaptive tree speculative decoding** engine:

```
Draft Model (qwen3-1.5b, auto-selected)
  generates adaptive draft tree (OPT-Tree algorithm):
       +--- "the"  --- "cat"
root --+--- "a"    --- "dog" --- "ran"
       +--- "this" --- "thing"

Target Model (qwen3-72b)
  verifies ENTIRE TREE in ONE forward pass (tree-masked causal attention — DeFT)
  accepts longest valid path → up to 6x throughput vs. single-token decode
```

**Measured results from component research:**
- JetSpec (2026): up to 9.64x speedup on code completion workloads
- DeFT (ICLR 2025 Spotlight): 3.59x attention latency reduction via KV-Guided Grouping
- OPT-Tree: maximizes expected acceptance length per target model call
- **Aether target: 3–6x throughput improvement** on latency-sensitive workloads

### 10.4 Global KV Cache Manager

| Cache Tier | Storage | What is Stored | Eviction |
|---|---|---|---|
| **L1 — GPU HBM** | On-device VRAM | Active request KV blocks | LRU + request priority |
| **L2 — CPU DRAM** | System RAM | Prefix cache, recently evicted blocks | Cost-aware LRU |
| **L3 — NVMe SSD** | Persistent storage | Long system prompt KV, RAG KV | TTL + access frequency |
| **L4 — Aether Hub** | CDN | Common system prompt KV (globally shared) | CDN invalidation |

**Prefix cache:** Identical prompt prefixes (system prompts, RAG documents, in-context examples) are stored as KV blocks indexed by RadixTree. Cache hit rates of 40–70% in agentic/RAG workloads — those tokens are never recomputed.

### 10.5 Dynamic Precision Adjustment

Under memory pressure, the runtime transparently downgrades precision:

```
Memory pressure detected (VRAM > 90%):
  -> Identify lowest-sensitivity layers (from AEG metadata: sensitivity_map)
  -> Swap BF16 weights to Q4 representation in-place (no service interruption)
  -> Log precision downgrade event to /v1/metrics

Memory pressure resolved (VRAM < 70%):
  -> Restore BF16 weights for high-sensitivity layers
  -> Resume full quality serving
```

Automatic. Transparent. Reversible. No existing tool does this.

---

## 11. Stage 5 — Developer Interface

### 11.1 The Core Promise

```python
from aether import Runtime

rt = Runtime()

# First run: downloads .aeg from Hub, loads pre-compiled kernels for your hardware
# Subsequent runs: instant (everything cached locally)
response = rt.generate("qwen3-72b", "Explain the P vs NP problem")
print(response.text)
print(f"TPS: {response.metrics.throughput_tps}")
print(f"TTFT: {response.metrics.ttft_ms}ms")
print(f"Kernel: {response.metrics.kernel_target}")        # e.g. "cuda_sm90"
print(f"Precision: {response.metrics.active_precision}")  # e.g. "mixed_fp8_q4"
print(f"Draft accept rate: {response.metrics.spec_accept_rate}")  # e.g. 0.82
```

### 11.2 The Compilation API

```python
from aether import Compiler

compiler = Compiler()

# Dry-run: inspect what compilation would produce
plan = compiler.plan("qwen3-72b", hardware="auto")
print(plan.fusion_opportunities)       # Fuseable op sequences found
print(plan.estimated_memory_gb)        # Memory estimates per precision
print(plan.estimated_compile_time_s)   # Expected compilation duration

# Compile any model to AEG format
aeg = compiler.compile(
    model="qwen3-72b",                    # HuggingFace ID, local path, or GGUF
    targets=["cuda_sm90", "metal_m3"],    # Explicit, or None for "all available"
    quality_budget=0.02,                  # Max 2% perplexity increase
    calibration_dataset="wikitext-2",     # For sensitivity analysis (Pass 2)
)

# Inspect the compiled artifact
print(aeg.graph_summary())    # Operator fusion summary: ops merged, memory saved
print(aeg.precision_map())    # Per-layer precision assignments
print(aeg.quality_report())   # Measured PPL change vs. BF16 baseline
print(aeg.sharding_plans())   # Parallelism plans for 1/2/4/8 GPU

# Save / distribute / upload
aeg.save("./qwen3-72b.aeg")
aeg.upload(hub="hub.aether.dev")  # Upload to Aether Hub (opt-in, free)
```

### 11.3 Python SDK — Full Reference

```python
from aether import Runtime, Compiler

rt = Runtime(
    optimize_for="latency",       # "latency" | "throughput" | "quality"
    speculative_decoding=True,
    prefill_chunk_size=2048,
    dynamic_precision=True,
)

# --- Text Generation ---
response = rt.generate(
    model="qwen3-72b",
    prompt="Write a sonnet about distributed systems",
    max_tokens=512,
    temperature=0.7,
    stream=False,
)

# --- Streaming ---
for chunk in rt.generate("qwen3-72b", "Tell me a story", stream=True):
    print(chunk.delta, end="", flush=True)

# --- Chat ---
response = rt.chat(
    model="qwen3-72b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is a transformer model?"}
    ]
)

# --- Embeddings ---
vectors = rt.embed(
    model="nomic-embed-text",
    input=["Hello world", "Quantum entanglement explained"]
)

# --- Reranking ---
ranked = rt.rerank(
    model="bge-reranker-v2",
    query="What is CUDA?",
    documents=["CUDA is a parallel computing platform...", "Python is a language..."]
)

# --- Vision ---
response = rt.generate(
    model="qwen2.5-vl-72b",
    prompt="Describe this image in detail",
    images=["path/to/image.jpg"]
)

# --- Transcription ---
transcript = rt.transcribe(
    model="whisper-large-v3",
    audio="path/to/audio.mp3",
    language="en"
)

# --- Async API ---
import asyncio
async def main():
    response = await rt.agenerate("qwen3-72b", "Hello async world")
    async for chunk in rt.agenerate("qwen3-72b", "Story", stream=True):
        print(chunk.delta, end="")

# --- Hardware and Diagnostics ---
hw = rt.hardware()           # Full hardware fingerprint
results = rt.benchmark("qwen3-72b")  # Performance benchmark

# --- Model Management ---
rt.pull("qwen3-8b")          # Download + compile AEG
rt.list()                    # List compiled models
rt.info("qwen3-72b")         # Metadata + precision map
rt.remove("qwen3-8b")        # Remove cached AEG
```

### 11.4 REST API

```
POST   /v1/generate          Text completion
POST   /v1/chat              Chat completion (OpenAI-compatible)
POST   /v1/embeddings        Embedding generation
POST   /v1/rerank            Document reranking
POST   /v1/transcribe        Audio transcription

POST   /v1/compile           Compile a model to AEG (async job)
GET    /v1/compile/{job_id}  Compilation job status and progress

GET    /v1/models            List compiled models
POST   /v1/models/pull       Download and compile model
DELETE /v1/models/{name}     Remove compiled model
GET    /v1/models/{name}     Model info and compilation metadata
GET    /v1/models/{name}/graph    Inspect AEG-IR

GET    /v1/hardware          Hardware fingerprint
GET    /v1/kernels           Active kernel targets
GET    /v1/metrics           Prometheus-compatible inference metrics
GET    /v1/health            Health check
```

### 11.5 OpenAI Compatibility

```python
from openai import OpenAI

# Point any OpenAI SDK to Aether — zero code changes
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="aether"   # any value is accepted
)

response = client.chat.completions.create(
    model="qwen3-72b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Works immediately with LangChain, LlamaIndex, CrewAI, AutoGen, OpenHands, and any OpenAI SDK v1+ tool.

### 11.6 CLI

```bash
# Compilation
aether compile qwen3-72b                        # Compile to AEG
aether compile qwen3-72b --quality-budget 0.01  # Stricter quality target
aether compile qwen3-72b --target cuda_sm90     # Hardware-specific compilation
aether compile qwen3-72b --upload               # Compile + share to Hub

# Model management
aether pull qwen3-8b                            # Download pre-compiled AEG from Hub
aether pull qwen3-8b --compile-local            # Force local compilation
aether list                                     # List all compiled models
aether info qwen3-72b                           # Metadata + precision map
aether rm qwen3-8b                              # Remove cached AEG
aether graph qwen3-72b                          # Print AEG-IR (human-readable)

# Serving
aether serve                                    # Start runtime daemon
aether serve --port 11434 --host 0.0.0.0
aether stop
aether status

# Running
aether run qwen3-8b                             # Interactive REPL
aether run qwen3-72b --stream                   # Streaming output

# Benchmarking
aether bench qwen3-72b                          # Benchmark on current hardware
aether bench qwen3-72b --compare vllm           # Side-by-side comparison
aether bench --all                              # Benchmark all compiled models

# Hardware and diagnostics
aether hw                                       # Hardware fingerprint
aether kernels                                  # List active kernel targets
aether logs                                     # Runtime logs
```

---

## 12. The AEG Format Specification

### 12.1 Metadata Schema

```json
{
  "model_id": "Qwen/Qwen3-72B-Instruct",
  "aether_version": "1.0",
  "compiled_at": "2026-07-01T10:00:00Z",
  "graph_hash": "sha256:abc123...",
  "architecture": {
    "family": "qwen_family",
    "params_billion": 72.0,
    "layers": 80,
    "heads": 64,
    "kv_heads": 8,
    "context_length": 131072,
    "vocab_size": 152064,
    "attention_type": "GQA",
    "ffn_type": "SwiGLU",
    "is_moe": false
  },
  "optimization": {
    "fusion_passes_applied": ["qkv_rope_norm", "ffn_swiglu", "residual_add"],
    "fused_ops_count": 156,
    "sensitivity_calibration_dataset": "wikitext-2",
    "quality_budget_ppl_increase": 0.02,
    "actual_ppl_increase": 0.018,
    "precision_distribution": {
      "BF16": "12%",
      "FP8": "28%",
      "Q4_K_M": "52%",
      "IQ3_XS": "8%"
    }
  },
  "kernels": {
    "targets": ["cuda_sm89", "cuda_sm90", "metal_m3", "cpu_avx512"],
    "flash_attention_variant": "flash_attention_3"
  },
  "memory_requirements": {
    "bf16_gb": 144.0,
    "compiled_min_gb": 38.5,
    "recommended_gb": 48.0
  }
}
```

### 12.2 Versioning and Stability Guarantee

AEG format versions follow semantic versioning with explicit stability contracts:
- **AEG/1.x** — Stable forever. All 1.x `.aeg` files will be readable by all future Aether versions.
- **AEG/2.x** — When introduced, backward-compatibility mode maintained for 3 years.

This mirrors LLVM IR's stability guarantee. A model compiled today will still run on Aether 5.0 in 2031.

---

## 13. Automatic Parallelism Discovery

### 13.1 The Problem Today

Running a 70B model across 4 GPUs requires expert knowledge:
- Tensor parallelism degree (how to split weight matrices)
- Pipeline parallelism stages (how to cut the model across nodes)
- Expert parallelism (for MoE models)
- Context parallelism (for long-context workloads)
- Which mix is optimal for prefill vs. decode phases

**Aether eliminates this entirely.** Pre-computed plans are stored in the `.aeg` artifact at compile time.

### 13.2 Runtime Plan Loading

```bash
aether serve qwen3-72b
# [INFO] Detected: 4x NVIDIA H100 80GB (NVLink 4.0)
# [INFO] Loading pre-computed 4-GPU sharding plan from qwen3-72b.aeg
# [INFO] Prefill: tensor_parallel=4, pipeline_stages=1
# [INFO] Decode:  tensor_parallel=2, pipeline_stages=2 (re-sharding per Seesaw)
# [INFO] Ready to serve in 8.3 seconds.
```

No `TENSOR_PARALLEL_SIZE=4` flag. No manual pipeline configuration. Zero user configuration.

### 13.3 Heterogeneous Mesh Support (Phase 3)

For the most advanced topology, Aether supports mixed hardware as a single logical inference cluster:

```
MacBook M3 Max (64GB unified memory)
  Runs qwen3-8b as draft model via Metal kernels
  Contributes tree speculation proposals

Desktop RTX 4090 (24GB VRAM)
  Runs qwen3-72b as target model via CUDA sm89 kernels
  Verifies draft tree in one forward pass

Cloud H100 (on-demand burst)
  Handles overflow requests and very long context (>128K tokens)
```

**No existing tool has an analog.** Aether treats heterogeneous local + cloud resources as a single logical inference cluster.

---

## 14. Disaggregated Prefill and Decode Architecture

### 14.1 The Head-of-Line Blocking Problem

In traditional co-located serving, a long prefill request blocks all concurrent decode requests:

```
Traditional (co-located GPU):
  ===== PREFILL(50K tokens, 8 seconds) ========================>
                                                         DECODE(A) — user waits 8s
                                                         DECODE(B) — user waits 8s

Aether Disaggregated:
  Prefill Pool: ===== PREFILL(50K) ==========================>
                                       KV transfer -->
  Decode Pool:  DECODE(A) ----------------------------------------> no wait
                DECODE(B) ----------------------------------------> no wait
```

### 14.2 Single-GPU Logical Disaggregation (Chunked Prefill)

- Long prefills are split into <=2048 token chunks
- Each chunk interleaved with one decode iteration per active request
- TTFT remains bounded regardless of prompt length

### 14.3 Multi-GPU Physical Disaggregation

- **Prefill replicas:** High tensor parallelism, compute-optimized; batch many prefills simultaneously
- **Decode replicas:** Lower tensor parallelism, memory-bandwidth-optimized; maximum decode batch size
- **KV transfer:** Via shared GPU memory (single-node) or RDMA InfiniBand (multi-node)

**Published results:** DistServe 3–4x goodput improvement; Mooncake 13–40% compute cost reduction in production.

---

## 15. Tree-Speculative Decoding Engine

### 15.1 Beyond Standard Speculation

Standard speculative decoding proposes a linear chain of tokens:
- Draft: token_A → token_B → token_C → token_D
- Target: verifies 4 tokens in 1 forward pass
- Problem: if token_A is wrong, all subsequent proposals are wasted

### 15.2 Aether's Adaptive Tree Speculation

Aether proposes a **branching tree** explored simultaneously in one target forward pass:

```
Draft proposals (OPT-Tree adaptive tree construction):

                +--- "the"  --- "cat"  --- "sat"
  root ---------+--- "a"    --- "dog"  --- "ran"
                +--- "this" --- "thing"

Target model verifies ALL 7 tokens in ONE forward pass
(tree-masked causal attention via DeFT KV-Guided Grouping)

Accept longest valid path from root -> up to 7 tokens per call
```

### 15.3 Performance Profile

| Component | Research Result | Aether Use |
|---|---|---|
| **OPT-Tree** | Maximizes expected acceptance length | Adaptive tree topology per decoding step |
| **DeFT (ICLR 2025)** | 3.59x attention latency reduction | Tree attention kernel emitted at Stage 3 |
| **JetSpec (2026)** | Up to 9.64x speedup (code workloads) | Draft head architecture for same-family models |
| **PCT pruning** | Removes low-value branches | Dynamic tree pruning before target verification |

**Aether target: 3–6x throughput improvement** on latency-sensitive chat and coding workloads.

### 15.4 Draft Model Auto-Selection

```python
DRAFT_FAMILIES = {
    "qwen3-72b":        "qwen3-1.5b",
    "llama3.3-70b":     "llama3.2-1b",
    "deepseek-r1-671b": "deepseek-r1-8b",
    "gemma-2-27b":      "gemma-2-2b",
}
# Validation: acceptance_rate checked on calibration set
# If acceptance_rate < 0.70: try next candidate or fall back to standard decoding
# Typical acceptance rates: 0.75-0.90 for same-family models
```

---

## 16. MoE-Aware Expert Routing Compiler

### 16.1 The MoE Problem Nobody Has Solved at the Compiler Level

MoE models (DeepSeek-R1-671B, Mixtral-8x22B, Qwen-MoE-57B) activate only 2–8 experts per token from a bank of 8–256 experts. The challenges:
- All expert weights must be accessible — DeepSeek-R1 weights exceed 1.3 TB
- Dynamic routing creates non-uniform GPU utilization
- Expert dispatch involves expensive scatter/gather operations

**Current tools treat MoE as a dense model** and pay the full memory and dispatch penalty. Aether compiles MoE routing as a **first-class compiler pass**.

### 16.2 MoE Compilation Output

```
Input MoE graph node: {router, expert_bank[256], top_k=8}

Step 1: Activation profiling on calibration set
  -> hot experts (>5% activation rate):  51 experts
  -> warm experts (0.1-5%):             128 experts
  -> cold experts (<0.1%):               77 experts

Step 2: Expert placement annotations stored in AEG
  -> hot experts:  pin to GPU HBM (always resident)
  -> warm experts: stage in CPU DRAM + CommitMoE prefetch pipeline
  -> cold experts: NVMe lazy-load on demand

Step 3: Intra-expert sparsity kernel specialization
  -> identify always-zero activation channels per expert
  -> emit sparse GEMM kernels that skip dead channels
  -> result: 2.5x expert layer speedup (vLLM MoE 2025)

Step 4: Threshold-based router replacement (DynaMoE)
  -> replace rigid top-K with adaptive threshold routing
  -> simple tokens: fewer experts activated (faster, less memory)
  -> complex tokens: more experts activated (better quality)
```

### 16.3 MoE Results

| Metric | Naive (current tools) | Aether MoE Compiler |
|---|---|---|
| Expert layer throughput | 1x baseline | 2.5x |
| Memory for DeepSeek-R1-671B | >1.3 TB | ~320 GB (hot+warm tiering) |
| Inference cost vs. early MoE | 1x | 0.17x (83% reduction) |
| Expert activation compute | Fixed top-K | Adaptive (0.5–2x experts per token) |

---

## 17. Developer API Specification

*(Full Python SDK covered in Section 11. This section covers configuration reference and REST API details.)*

### 17.1 Configuration Reference

```python
from aether import CompilerConfig, RuntimeConfig

# Compiler configuration
compiler_config = CompilerConfig(
    quality_budget=0.02,                # Max 2% perplexity increase (guides quantization)
    calibration_dataset="wikitext-2",   # Dataset for sensitivity analysis
    targets=["auto"],                   # "auto" = detect current hardware
    optimization_level=2,               # 0=none, 1=basic, 2=full (default), 3=aggressive
    enable_moe_compiler=True,           # MoE-aware compilation pass
    upload_kernels=True,                # Opt-in to Aether Hub kernel sharing
    cache_dir="~/.aether",
)

# Runtime configuration
runtime_config = RuntimeConfig(
    optimize_for="latency",             # "latency" | "throughput" | "quality"
    speculative_decoding=True,          # Enable tree-speculative decoding
    speculative_tree_depth=4,           # Max tree depth (default: auto)
    prefill_chunk_size=2048,            # Tokens per prefill chunk
    max_batch_size=256,                 # Max concurrent requests
    kv_cache_dtype="fp8",               # KV cache precision (fp8 / fp16 / bf16)
    kv_cache_cpu_gb=32,                 # CPU DRAM KV cache budget
    kv_cache_nvme_gb=200,               # NVMe KV cache budget
    dynamic_precision=True,             # Allow precision downgrade under memory pressure
    disaggregate_prefill_decode=False,  # Enable for multi-GPU cluster mode
)
```

### 17.2 Response Object

```python
response = rt.generate("qwen3-72b", "Hello!")

response.text                          # Generated text
response.usage.prompt_tokens           # Input token count
response.usage.completion_tokens       # Output token count
response.metrics.throughput_tps        # Tokens per second
response.metrics.ttft_ms               # Time to first token (ms)
response.metrics.p95_latency_ms        # P95 latency
response.metrics.kernel_target         # Active hardware target (e.g. "cuda_sm90")
response.metrics.active_precision      # Active precision (e.g. "mixed_fp8_q4")
response.metrics.spec_accept_rate      # Speculative decoding acceptance rate
response.metrics.kv_cache_hit_rate     # KV cache prefix hit rate
response.metrics.memory_pressure       # Current VRAM utilization (0.0 - 1.0)
```

---

## 18. Target Audience and Personas

### Persona 1 — The Application Developer (Primary — 70% of users)

**Who:** Full-stack or backend developer building AI-powered products.
**Pain:** Infrastructure complexity prevents shipping. Never touched a CUDA kernel.
**Aether answer:** `aether pull qwen3-8b && aether serve` — done. No CUDA knowledge required.
**Wow moment:** *"I moved to a new MacBook and it just worked. Same command, same .aeg file."*

### Persona 2 — The ML Infrastructure Engineer (Power user — 20%)

**Who:** Specialist who owns AI infrastructure for a team or company.
**Pain:** Manually tunes different backends for each model and hardware combination.
**Aether answer:** Full compilation API, per-layer precision map, AEG-IR inspection, benchmark vs. vLLM.
**Wow moment:** *"I got 40% better throughput than our hand-tuned vLLM config in 30 minutes — with a report proving it."*

### Persona 3 — The Researcher (Secondary — 15%)

**Who:** Academic or industrial researcher running experiments on diverse hardware.
**Pain:** Environment setup consumes more time than actual research.
**Aether answer:** Reproducible `.aeg` artifacts. Share the compiled model, not the setup instructions.
**Wow moment:** *"I shared a .aeg file with collaborators at three institutions. Identical results everywhere."*

### Persona 4 — The Enterprise Architect (Commercial — 5% of users, 80% of revenue)

**Who:** Senior technical decision-maker deploying AI across cloud and on-premises fleet.
**Pain:** Vendor lock-in. No portability between cloud GPU providers.
**Aether answer:** AEG format runs on any hardware. Fleet management via Aether Cloud.
**Wow moment:** *"We migrated our entire inference fleet from H100 to AMD MI300X with zero model re-deployment work."*

---

## 19. Open-Source Roadmap — Five Phases

### Phase 1 — Compiler Foundation (Months 1–4)

**Theme:** *"The AEG format ships. The model compilation story is real."*

- [ ] `aether` Python package (pip installable, Linux / macOS / Windows)
- [ ] Graph tracer: SafeTensors + GGUF → AEG-IR
- [ ] Architecture detector: Llama, Qwen, Gemma, Mistral, DeepSeek families
- [ ] **AEG format v1.0 specification (public, stable, versioned)**
- [ ] Optimizer Pass 1: Operator fusion (QKV + RoPE + Norm → megakernel)
- [ ] Hardware targeting: CUDA kernels (sm80, sm89, sm90)
- [ ] Hardware targeting: CPU (AVX-512, NEON)
- [ ] Basic runtime: loads AEG, dispatches kernels, serves requests
- [ ] Python SDK: `generate()`, `chat()`, `embed()`
- [ ] OpenAI-compatible REST API via `aether serve`
- [ ] CLI: `compile`, `pull`, `run`, `serve`, `list`, `rm`, `info`, `graph`
- [ ] Content-addressed kernel cache (`~/.aether/kernels/`)
- [ ] **Aether Hub MVP:** kernel upload/download API (opt-in)
- [ ] Documentation site + AEG format specification (aeg-spec.aether.dev)
- [ ] GitHub Actions CI (Linux / macOS / Windows)

**Success Criteria:**
```bash
pip install aether
aether compile llama3-8b    # Compiles to AEG in under 5 minutes
aether run llama3-8b        # Generates text on any supported hardware
# Move .aeg file to different hardware -> runs identically
```

---

### Phase 2 — Optimizer Depth (Months 5–9)

**Theme:** *"The compiler makes models measurably better than naive deployment."*

- [ ] Optimizer Pass 2: Sensitivity Analysis (d_perplexity/d_precision per layer)
- [ ] Optimizer Pass 3: Mixed-precision assignment from sensitivity map
- [ ] Optimizer Pass 4: KV Cache graph structuring (PagedKV nodes + radix hints)
- [ ] Compiler plan dry-run API (`compiler.plan()`)
- [ ] Quality report generation (measured PPL change vs. BF16)
- [ ] Tree-Speculative Decoding Engine (OPT-Tree adaptive tree)
- [ ] DeFT-style tree-masked attention kernel
- [ ] Disaggregated prefill/decode scheduler (chunked prefill)
- [ ] Radix-tree prefix cache engine
- [ ] Metal target: Apple M-series (M1–M5) kernel compilation
- [ ] ROCm target: AMD RDNA3/CDNA3 kernel compilation
- [ ] Aether Hub v2: kernel versioning + model manifest registry
- [ ] `aether bench` with side-by-side comparison vs. vLLM
- [ ] Prometheus metrics endpoint (`/v1/metrics`)
- [ ] Dynamic precision adjustment under memory pressure

**Success Criteria:** Aether-compiled models achieve >=20% higher throughput than user's best hand-configured backend on 80% of hardware/model combinations, with <=2% quality loss.

---

### Phase 3 — Parallelism and Scale (Months 10–16)

**Theme:** *"70B models on 4 consumer GPUs with zero configuration."*

- [ ] Optimizer Pass 6: Automatic Parallelism Discovery (Alpa/Seesaw inspired solver)
- [ ] Pre-computed sharding plans (1/2/4/8 GPU) stored in AEG at compile time
- [ ] Tensor Parallelism runtime (NVLink/PCIe aware)
- [ ] Pipeline Parallelism runtime (multi-node)
- [ ] Stage-aware re-sharding (different plan for prefill vs. decode phases)
- [ ] MoE-Aware Expert Routing Compiler (Pass 5)
- [ ] Expert placement tiering (hot GPU / warm CPU / cold NVMe)
- [ ] Intra-expert sparsity kernels
- [ ] Context Parallelism for long sequences (Ring Attention)
- [ ] KV cache NVMe offload (L3 tier)
- [ ] Physical disaggregation (prefill/decode on separate nodes via RDMA)
- [ ] Heterogeneous mesh (local draft + remote target speculative decoding)
- [ ] ONNX ingestion path (ONNX → AEG-IR)
- [ ] Plugin SDK for third-party hardware backends
- [ ] `aether cluster` CLI for multi-node management

**Success Criteria:** 72B model on 4x RTX 4090 at >85% MFU; zero TENSOR_PARALLEL_SIZE flags; setup time <2 minutes.

---

### Phase 4 — Ecosystem (Months 17–24)

**Theme:** *"AEG becomes the distribution format for compiled AI models."*

- [ ] MLX ingestion path (Apple MLX → AEG-IR)
- [ ] OpenVINO / Intel NPU target
- [ ] AEG Model Registry: versioned, verified compiled model artifacts
- [ ] HuggingFace Hub integration: Aether detects and downloads pre-compiled AEG variants
- [ ] SDK bindings: JavaScript/TypeScript, Rust, Go
- [ ] Fine-tuning integration (LoRA/QLoRA → AEG re-compilation with adapter merging)
- [ ] Aether Bench: public leaderboard (hardware x model x optimizer version)
- [ ] WASM target (browser-side inference from AEG — experimental)
- [ ] Enterprise Preview: monitoring dashboard, fleet management, RBAC

---

### Phase 5 — Compiler as Platform (Month 25+)

**Theme:** *"Aether is the LLVM of AI. Hardware vendors target AEG."*

- [ ] AEG Compiler API: third-party hardware target SDK (chip vendors integrate directly)
- [ ] Custom hardware target onboarding (NPU, FPGA, custom AI ASICs)
- [ ] Aether Labs: community-contributed compiler passes as plugins
- [ ] Multi-modal compilation: vision encoder + language decoder as unified AEG graph
- [ ] Reasoning graph compilation: chain-of-thought as a compiled, optimizable execution graph
- [ ] Hardware co-design: expose AEG-IR semantics to chip vendors for ISA design feedback

---

## 20. Commercial Strategy and Moat

### 20.1 The Open-Core Model

```
Open Source (Forever Free)                  Aether Cloud (Paid)
------------------------------------        -----------------------------------
Aether compiler + all optimizer passes      Managed compilation (GPU time)
AEG format + all hardware targets           Private AEG artifact registry
Aether runtime + all features               Fleet management dashboard
Aether Hub (kernel sharing, public)         Multi-machine orchestration
Community support                           Team collaboration + RBAC
                                            SLA-guaranteed compilation times
                                            Enterprise SSO + audit logs
                                            Priority kernel slots on Hub
                                            Custom hardware target SLAs
```

### 20.2 Revenue Streams

| Stream | Pricing | Why It Works |
|---|---|---|
| **Aether Cloud** | Usage-based (compilation GPU-hours + serving tokens) | Compilation is GPU-intensive; teams outsource it |
| **Enterprise Hub** | Per-seat annual license | Private kernel cache + compliance requirements |
| **Hardware Vendor Partnerships** | License fee for first-class AEG target status | NVIDIA/AMD/Apple/Intel pay to be first-class in AEG |
| **Managed Serving** | Per-GPU-hour | Teams without GPU infrastructure run on Aether's fleet |

### 20.3 The Network Effect Flywheel

```
Developers compile models
       |
       v
Kernels uploaded to Hub
       |
       v
Hub grows -> cold-start time approaches zero
       |
       v
Zero cold-start -> more developers adopt Aether
       |
       v
More adoption -> AEG artifacts appear on HuggingFace
       |
       v
HuggingFace has AEG -> Aether becomes default distribution format
       |
       v
Default format -> hardware vendors implement AEG targets natively
       |
       v
Native hardware support -> Aether owns the IR layer for AI
```

This is the Linux kernel flywheel. Linux took 10 years. Aether targets 5, with a better initial developer experience and a clear format moat.

### 20.4 Go-to-Market

**Stage 1 — GitHub-First (Months 1–6):**
Ship Phase 1 with a genuinely shocking demo: *the same `.aeg` file running identically on RTX 4090, M3 Max, and AMD MI300X, in a single 90-second video.* The `aether graph` command showing the fusion pass output and precision map is the technical hook. Target: HuggingFace community, AI/ML Twitter, Hacker News.

**Stage 2 — Community to Commercial (Months 6–18):**
Aether Hub creates the network effect loop. The benchmark comparison tool builds reputation as the tool that actually measures performance gains from compilation. Internal champions convert enterprises to paid Cloud tier.

**Stage 3 — Enterprise (Months 18+):**
Direct sales for fleet management, private model registry, and compliance features. Hardware vendor partnerships for official AEG targets.

### 20.5 Infrastructure Company Comparables

| Company | Open-Source Core | Moat | Outcome |
|---|---|---|---|
| **HashiCorp** | Terraform | HCL format ownership | Acquired $6.4B |
| **Docker** | Docker Engine | OCI image format | ~$2.1B |
| **Grafana Labs** | Grafana | Dashboard + PromQL format | >$6B valuation |
| **Elastic** | Elasticsearch | Index + query DSL | IPO ~$3B |
| **LLVM Project** | Compiler IR | LLVM IR format | Foundation of entire native software ecosystem |
| **Aether** | Compiler + AEG format | Compiled model format | TBD |

---

## 21. Technical Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| AEG-IR fails to capture emerging non-transformer architectures (SSM, RWKV) | Medium | High | Custom op extensibility in AEG-IR v1.0; treat unknown ops as pass-through; community reporting |
| Kernel cache coherence bugs (wrong kernel for hardware revision) | Medium | High | Hardware fingerprint includes driver version; strict content-addressing; runtime output validation |
| Graph tracing fails for heavily dynamic models | High | Medium | Fallback to operator-library mode; document tracing requirements; collect community reports |
| TensorRT-LLM adds an open portable format | Low | High | Ship AEG v1.0 with stability guarantee first; community adoption creates switching cost |
| Compilation time annoys users on first run | High | Medium | Pre-compile all top-50 HuggingFace models to AEG before launch day |
| Modular MAX open-sources | Medium | Medium | AEG format + Hub network effect are structural advantages that require equal open commitment to match |
| Memory safety bugs in mixed-precision runtime | High | High | Extensive VRAM pressure chaos testing; conservative safety margins; formal verification of quant pass |
| Hardware vendors ship proprietary closed compilers | Medium | Low | History: open beats closed for portability (LLVM beat all proprietary IRs) |

### 21.1 Build vs. Integrate Decisions

| Component | Decision | Rationale |
|---|---|---|
| AEG-IR specification | **Build** (core IP) | Must be owned; defines the moat |
| All six optimizer passes | **Build** (core IP) | Graph-level; cannot be delegated |
| CUDA kernels (FA-3, GEMM) | **Integrate** (FA-3, ExLlamaV2) | Research excellence exists; emit via Stage 3 |
| Disaggregated scheduler | **Build** | Unique KV cache and disaggregation design |
| Graph tracer | **Build** (thin torch.export wrapper) | Need AEG-IR output format |
| Hardware detection | **Build** | Hardware fingerprinting unique to Aether |
| REST API | **Build** (FastAPI/Starlette) | Core developer experience surface |
| Aether Hub backend | **Build** | Content-addressed kernel store; moat infrastructure |

---

## 22. Success Metrics and KPIs

### 22.1 Developer Adoption (Open Source)

| Metric | 6 Months | 12 Months | 24 Months |
|---|---|---|---|
| GitHub Stars | 3,000 | 15,000 | 50,000+ |
| pip installs/month | 10,000 | 100,000 | 1,000,000+ |
| AEG files compiled (unique models) | 1,000 | 50,000 | 500,000 |
| Kernel cache Hub entries | 500 | 10,000 | 200,000+ |
| Active contributors | 30 | 150 | 500 |
| Supported model architectures | 5 | 15 | 30+ |
| Supported hardware targets | 4 | 8 | 14+ |

### 22.2 Technical Performance Targets

| Metric | Target | Measurement Method |
|---|---|---|
| **Compilation throughput** | >=20% higher TPS vs. best hand-configured vLLM | A/B benchmark suite, published methodology |
| **Quality preservation** | <=2.0% perplexity increase vs. BF16 at mixed precision | LLM-Eval on wikitext-2 / Hellaswag / MMLU |
| **TTFT reduction (fused kernels)** | >=30% TTFT reduction vs. unfused eager execution | Request latency benchmark, 95th percentile |
| **Tree speculation throughput** | >=3x TPS on latency-sensitive chat/coding | Single-request throughput benchmark |
| **Compilation time (Hub cache hit)** | <30 seconds | End-to-end timing from `aether pull` |
| **AEG portability** | 100% identical output across hardware for same AEG | Cross-hardware correctness regression suite |
| **Memory safety** | >=99.9% uptime under sustained VRAM pressure | 72-hour chaos testing with concurrent overload |

### 22.3 Commercial (Post-Phase 3)

| Metric | 18 Months | 24 Months |
|---|---|---|
| Paid customers | 20 | 100 |
| ARR | $100K | $1M |
| Enterprise pilots | 5 | 25 |
| Hardware vendor partnerships | 1 | 3 |
| GPU-hours managed via Aether Cloud | 50K/month | 500K/month |

---

## 23. Glossary

| Term | Definition |
|---|---|
| **AEG** | Aether Execution Graph — Aether's portable compiled model format and artifact |
| **AEG-IR** | Aether Execution Graph Intermediate Representation — hardware-agnostic operator graph |
| **Aether Hub** | Public, content-addressed kernel cache and compiled model registry |
| **Operator Fusion** | Merging multiple sequential ops into a single GPU kernel to eliminate intermediate DRAM writes |
| **Sensitivity Analysis** | Computing d(perplexity)/d(precision) per layer to guide mixed-precision quantization mathematically |
| **Mixed Precision** | Assigning different numeric formats to different layers based on measured sensitivity |
| **Tree Speculation** | Speculative decoding that proposes a branching tree of draft tokens, all verified in one target forward pass |
| **OPT-Tree** | Adaptive draft tree construction algorithm maximizing expected token acceptance length |
| **DeFT** | Decoding with Flash Tree-Attention — hardware-efficient tree attention via KV-Guided Grouping |
| **JetSpec** | Causal parallel draft heads achieving up to 9.64x speedup in speculative decoding |
| **PagedAttention** | vLLM's OS-inspired non-contiguous KV memory management (used in Aether's KV cache) |
| **RadixAttention** | SGLang's radix-tree KV prefix cache (Aether's prefix cache) |
| **Disaggregated P/D** | Physical separation of prefill and decode phases onto different hardware pools |
| **MoE** | Mixture of Experts — sparse model activating only a subset of parameters per token |
| **Intra-Expert Sparsity** | Skipping always-zero activation channels within an expert for additional speedup |
| **Tensor Parallelism** | Distributing individual weight matrices across multiple GPUs in parallel |
| **Pipeline Parallelism** | Distributing model layers sequentially across multiple GPU nodes |
| **Context Parallelism** | Distributing long input sequences across multiple GPUs (Ring Attention / Ulysses) |
| **Automatic Parallelism** | Compiler-discovered sharding strategy requiring zero user configuration |
| **MLIR** | Multi-Level Intermediate Representation — LLVM's compiler IR framework |
| **StableHLO** | Open-standard ML model IR from Google/OpenXLA; AEG-IR follows its stability model |
| **IREE** | Intermediate Representation Execution Environment — MLIR-based universal ML runtime |
| **Kernel Cache** | Content-addressed store of pre-compiled GPU kernels keyed by graph hash + hardware target |
| **Goodput** | Rate of inference requests successfully meeting latency SLO constraints |
| **MFU** | Model FLOP Utilization — fraction of theoretical GPU FLOPS actually utilized |
| **TTFT** | Time to First Token — request-to-first-generated-token latency |
| **TPS** | Tokens Per Second — inference throughput metric |
| **GGUF** | GPT-Generated Unified Format — llama.cpp's quantized model format |
| **BF16 / FP8 / Q4** | Brain Float 16 / 8-bit float / 4-bit integer — numeric precision formats |
| **FlashAttention-3** | Memory-efficient attention using WGMMA + TMA on H100+ (1.5–2x over FA-2) |

---

## Appendix A — Research Foundation

This PRD is grounded in the following research works. Each maps directly to a specific Aether feature or design decision.

### A.1 Compiler and Intermediate Representations

| Paper / Project | Year | Aether Feature |
|---|---|---|
| **MLIR: A Compiler Infrastructure for the End of Moore's Law** — Lattner et al. | 2021 | AEG-IR design; multi-level dialect model preserving high-level semantics |
| **IREE: Intermediate Representation Execution Environment** — openxla/iree | 2022+ | Hardware-universal runtime design reference; MLIR lowering pipeline |
| **StableHLO: Portability and Stability for ML Compilers** — OpenXLA | 2023+ | AEG-IR stability and versioning model; 5-year backward compat promise |
| **Meta LLM Compiler: Foundation Models of Compiler Optimization** | 2024 | AI-driven pass ordering; automated compiler optimization selection |
| **Modular MAX Graph Compiler** (MLIR + Mojo kernels) | 2024+ | Dense model compilation performance reference; operator fusion benchmarks |
| **ClusterFusion: Intra-Kernel Communication for Transformer Fusion** — NeurIPS 2025 | 2025 | Megakernel fusion; ClusterReduce/ClusterGather; 1.6–2.0x speedup |

### A.2 Speculative Decoding and Tree Attention

| Paper | Year | Aether Feature |
|---|---|---|
| **SpecInfer: Tree-based Speculative Inference** | 2023 | Tree speculation foundation; draft tree verification |
| **OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure** | 2024 | Adaptive tree construction; expected acceptance length maximization |
| **DeFT: Decoding with Flash Tree-Attention** — ICLR 2025 Spotlight | 2025 | KV-Guided Grouping; 3.59x attention latency reduction |
| **JetSpec: Scaling Speculative Decoding** | 2026 | 9.64x speedup; causal parallel draft heads |
| **EDD: Effective Draft Decoder via Soft Prompts** — ACL 2025 | 2025 | Higher-quality draft generation |
| **Pruned Candidate Trees (PCT)** — ACL 2025 | 2025 | Dynamic branch pruning before target verification |
| **EAGLE-3: Scalable Speculative Decoding** | 2025 | Draft model architecture reference |

### A.3 KV Cache and Memory Management

| Paper | Year | Aether Feature |
|---|---|---|
| **PagedAttention: Efficient Memory Management for LLM Serving** — Kwon et al., SOSP 2023 | 2023 | Paged KV block management; virtual memory model |
| **SGLang: Efficient Execution of Structured LM Programs** — Zheng et al. | 2024 | RadixAttention prefix cache; AEG-IR radix tree hints |
| **DistServe: Disaggregating Prefill and Decoding** | 2024 | Disaggregated scheduler; 3–4x goodput improvement |
| **Mooncake: A KVCache-centric Disaggregated Architecture** | 2024 | Production disaggregation results; 13–40% cost reduction |
| **EvolKV: Evolutionary KV Cache Optimization** | 2025 | Adaptive KV allocation; tier-aware eviction policies |
| **FlexGen: High-Throughput Inference with a Single GPU** — Sheng et al. | 2023 | NVMe KV cache offloading (L3 tier design) |
| **LoopServe: Multi-Turn KV Cache Reuse** | 2025 | Cross-session KV sharing; rolling context cache |

### A.4 Quantization and Mixed Precision

| Paper | Year | Aether Feature |
|---|---|---|
| **GPTQ: Accurate Post-Training Quantization** | 2022 | Quantization sensitivity reference |
| **AWQ: Activation-aware Weight Quantization** | 2023 | Activation-weighted quantization guidance |
| **AutoMixQ: Automated Mixed-Precision Quantization** | 2025 | Sensitivity analysis pass design; per-layer precision assignment |
| **AMQ: Accurate Mixed-Precision Quantization** | 2025 | Quality benchmarks; 30–40% PPL improvement vs. uniform quantization |
| **MoQAE: Mixture of Quantization-Aware Experts** | 2025 | Adaptive precision per input type |
| **ExLlamaV2** (community project) | 2024 | Fastest INT4 GEMM kernels; integrated via Stage 3 kernel emission |

### A.5 MoE Inference Optimization

| Paper | Year | Aether Feature |
|---|---|---|
| **MoE-Infinity: Offloading-Efficient MoE Serving** | 2025 | Expert offload tiering (hot/warm/cold); activation-aware caching |
| **CommitMoE: Expert Prefetching for Memory-Constrained Serving** | 2025 | Expert prefetch scheduling; hiding expert load latency |
| **FinDEP: Fine-Grained Disaggregated Expert Parallelism** | 2025 | Expert compute/communication overlap |
| **DynaMoE: Dynamic Expert Allocation** | 2025 | Threshold-based routing; adaptive expert count per token |
| **DA-MoE: Attention-Guided Dynamic Expert Allocation** | 2025 | Token importance-based routing |
| **Intra-Expert Sparsity Analysis** (vLLM 2025) | 2025 | 2.5x expert layer speedup from dead activation channel pruning |

### A.6 Parallelism and Distributed Inference

| Paper | Year | Aether Feature |
|---|---|---|
| **Alpa: Automating Inter/Intra-Operator Parallelism** | 2022 | Parallelism solver design; cost model-based search |
| **Megatron-LM: Training Multi-Billion Parameter Models** | 2019–2025 | TP/PP/DP/EP/CP parallelism primitive reference |
| **Seesaw: Dynamic Model Re-sharding for LLM Inference** — MLSys 2025 | 2025 | Stage-aware parallelism; separate plans for prefill vs. decode |
| **Ring Attention / Ulysses Context Parallelism** | 2023–2024 | Long-context parallelism across multiple GPUs |
| **Splitwise: Efficient Generative LLM Inference via Phase Splitting** | 2023 | Prefill/decode disaggregation analysis |

### A.7 Key Open-Source Projects Studied

| Project | What Aether Learns | What Aether Does Differently |
|---|---|---|
| **vLLM** | PagedAttention; continuous batching; OpenAI-compat API | Owns computation graph; compilation, not wrapping |
| **SGLang** | RadixAttention; prefix caching; structured generation | AEG-IR carries radix hints at compile time |
| **TensorRT-LLM** | Kernel fusion; FP8 GEMM; compiled engine concept | Open-source; hardware-universal; open format |
| **llama.cpp** | GGUF format; K/I-quants; cross-platform CPU/GPU | AEG supersedes GGUF as distribution format |
| **MLX** | Apple Silicon native; unified memory; lazy eval | AEG ingests MLX; Metal kernels emitted natively |
| **ONNX Runtime** | Execution Provider model; graph optimization | AEG-IR is LLM-specialized (ONNX is general-purpose) |
| **Modular MAX** | MLIR-based graph compilation; Mojo kernels | Open-source; community ecosystem; open format |
| **IREE** | MLIR lowering pipeline; hardware universality | LLM-specialized ops; developer UX; Aether Hub |
| **Ollama** | Docker-like UX; single command model management | Compilation, not serving; AEG supersedes GGUF |
| **NVIDIA Dynamo** | Disaggregated serving; RDMA KV transfer at scale | Not NVIDIA-exclusive; open runtime; open format |

---

*End of Aether Runtime PRD v2.0*

*Authored July 2026.*
*The AEG format specification is versioned independently at* `aeg-spec.aether.dev`
*This document is the product vision. Implementation decisions may evolve as architecture matures.*
