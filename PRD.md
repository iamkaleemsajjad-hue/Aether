# Aether Runtime — Product Requirements Document (PRD)

> **Version:** 3.0 — July 2026
> **Status:** Foundational · Active Development
> **Tagline:** *"Compile once. Run on any hardware, forever."*
> **Current Codebase:** 113 Python files · 16,810 lines · 595 KB — scaffold complete, Phase 1 implementation in progress

---

## The One-Line Pitch

> **Aether is the LLVM for AI models — a compiler that turns any model into a portable, optimized, hardware-universal artifact.**

Just as LLVM compiles C/C++ into a portable IR that runs optimally on any CPU, Aether compiles any AI model into an **Aether Execution Graph (AEG)** — a compiled artifact that runs with maximum hardware-native performance on any GPU, Apple Silicon, AMD, Intel NPU, or CPU, today and on every future accelerator.

---

## Table of Contents

1. [Why This Exists](#1-why-this-exists)
2. [The Killer Insight — Compilation Not Wrapping](#2-the-killer-insight)
3. [The AEG Format — Core Innovation](#3-the-aeg-format)
4. [Competitive Landscape](#4-competitive-landscape)
5. [The Moat](#5-the-moat)
6. [Architecture — Five Compiler Stages](#6-architecture)
7. [Stage 1 — Model Ingestion](#7-stage-1-model-ingestion)
8. [Stage 2 — Six Optimizer Passes](#8-stage-2-optimizer)
9. [Stage 3 — Hardware Kernel Emission](#9-stage-3-hardware-targeting)
10. [Stage 4 — Self-Optimizing Runtime](#10-stage-4-runtime)
11. [Stage 5 — Developer Interface](#11-stage-5-developer-interface)
12. [NEW: Pass 7 — Reasoning Graph Compiler](#12-new-reasoning-graph-compiler)
13. [NEW: MLA Native Support](#13-new-mla-native-support)
14. [NEW: FP4 Blackwell Targeting](#14-new-fp4-blackwell-targeting)
15. [NEW: Agentic Workflow Optimizer](#15-new-agentic-workflow-optimizer)
16. [NEW: Multi-Modal Unified Graph](#16-new-multimodal-unified-graph)
17. [NEW: EAGLE-3 Speculative Decoding](#17-new-eagle3-speculative-decoding)
18. [NEW: Quantization-Aware Compilation Pipeline](#18-new-quantization-aware-pipeline)
19. [NEW: Aether Observability Stack](#19-new-observability-stack)
20. [NEW: AEG Safety & Guardrail Layer](#20-new-safety-guardrail-layer)
21. [Developer API Specification](#21-developer-api)
22. [Target Personas](#22-target-personas)
23. [Open-Source Roadmap — Six Phases](#23-roadmap)
24. [Commercial Strategy](#24-commercial-strategy)
25. [Technical Risk Analysis](#25-technical-risks)
26. [Success Metrics](#26-success-metrics)
27. [Glossary](#27-glossary)
28. [Appendix A — Research Foundation (60+ Papers)](#appendix-a)

---

## 1. Why This Exists

### 1.1 The Wrong Diagnosis

Every existing AI inference tool diagnosed the problem as fragmentation and built a better wrapper:

- **Ollama** — wraps llama.cpp with Docker-like UX
- **vLLM** — wraps PyTorch with PagedAttention scheduling
- **SGLang** — wraps vLLM with structured generation
- **TGI** — wraps transformers with a production server
- **TensorRT-LLM** — wraps CUDA kernels with a Python API

They are all wrappers. Every wrapper inherits the fundamental constraint of its substrate: **statically bound to one set of backends and hardware at build time**. None own the computation graph. None can apply global, cross-layer optimizations.

### 1.2 The Correct Diagnosis — No Portable Compiled Form

| Software Ecosystem | Portable Compiled Form | Runs Everywhere? |
|---|---|---|
| C/C++ | LLVM IR → binary | Yes — any CPU with LLVM backend |
| Java | JVM bytecode | Yes — any JVM |
| AI Models (2026) | Raw weights (safetensors/GGUF) | **NO** — must re-setup per hardware |

AI models have no portable compiled form. A 72B model in safetensors format requires completely different setup on NVIDIA (vLLM + CUDA), Apple (MLX), AMD (ROCm + vLLM), and Intel (OpenVINO). **Aether fixes this.**

### 1.3 The Analogy

```
Before LLVM:   GCC → x86 only | MSVC → Windows only | each compiler siloed
After LLVM:    C++ → LLVM IR → x86 | ARM | WASM | RISC-V | any future ISA

Before Aether: safetensors → CUDA only | GGUF → llama.cpp only | each format siloed
After Aether:  Any model → AEG → NVIDIA (sm70–sm100) | Apple Silicon | AMD | Intel | CPU
```

### 1.4 Market Scale (2026)

- AI inference infrastructure TAM: **$300B+ by 2030**, 30% CAGR
- 65% of enterprises use hybrid AI (cloud + on-prem) — desperate for hardware portability
- DeepSeek-R1, Qwen3, Llama 3.3 are open weights — **inference tooling is the product**
- NVIDIA Blackwell (B200) deployed at scale — new FP4 precision requires compiler support
- MoE models (DeepSeek-V3-671B) dominate frontier — require specialized compiler passes

---

## 2. The Killer Insight

### 2.1 What Every Tool Gets Wrong

All existing tools use the eager execution model:
```
Request → Framework (PyTorch) → Kernel Dispatch → Hardware
```

Each tool adds its own patch at a different layer — none see the whole model. None own the graph.

### 2.2 What Aether Does

```
Any Model → [Extract Graph] → [Optimize] → [Target] → .aeg artifact
                Stage 1          Stage 2    Stage 3

.aeg + Hardware → [Aether Runtime] → Tokens / Embeddings
                      Stage 4
```

Aether **owns the computation graph**. It can:
- Fuse RMSNorm + QKV + RoPE into a single GPU megakernel (40% fewer DRAM round-trips)
- Apply MLA native compression passes (90%+ KV cache reduction for DeepSeek-family models)
- Target FP4 on Blackwell (4x throughput vs H100 on same model)
- Compile reasoning chains as executable graphs with budget control
- Auto-shard 70B models across 4 GPUs with zero user configuration
- Cache compiled kernels globally — every user who compiles contributes to the network

### 2.3 Developer Experience

**Without Aether:**
```bash
# NVIDIA: install CUDA + cuDNN + vLLM (hours), handle tensor parallel config manually
# Apple: install MLX, re-export model to MLX format (hours)
# AMD: rebuild vLLM with ROCm flags (hours)
# Move model to different hardware: repeat from scratch
# Total: days of infrastructure before first token
```

**With Aether:**
```bash
pip install aether
aether compile qwen3-72b          # 4 minutes → qwen3-72b.aeg
aether run qwen3-72b              # runs on whatever hardware you have
# Move qwen3-72b.aeg to any machine → runs identically, optimally, instantly
```

---

## 3. The AEG Format

The **Aether Execution Graph (AEG)** is Aether's central invention. A portable, versioned, content-addressed, compiled representation of an AI model — ready to execute on any supported hardware.

### 3.1 AEG Package Layout

```
qwen3-72b.aeg/
├── FORMAT_VERSION               "AEG/1.0"
├── graph/
│   ├── computation_graph.aeg-ir   Hardware-agnostic operator graph (like LLVM IR)
│   ├── reasoning_graph.aeg-ir     [NEW v3.0] Compiled reasoning/CoT execution graph
│   ├── metadata.json              Architecture, modalities, capabilities
│   └── graph.sha256               Content-addressed hash (global cache key)
├── weights/
│   ├── precision_map.json         Per-layer precision (sensitivity-guided)
│   ├── model.aeg-quant            Mixed-precision compressed weights
│   └── mla_compressed/            [NEW v3.0] MLA latent vectors (if MLA architecture)
│       └── latent_kv.bin
├── kernels/
│   ├── cuda_sm70/                 NVIDIA V100 (FP16 Tensor Cores)
│   ├── cuda_sm80/                 NVIDIA A100 (BF16 TMA)
│   ├── cuda_sm89/                 NVIDIA RTX 4090 (FP8 Ada)
│   ├── cuda_sm90/                 NVIDIA H100 (WGMMA + FA-3)
│   ├── cuda_sm100/                NVIDIA B200 (FP4 Blackwell — NEW v3.0)
│   ├── cuda_sm120/                NVIDIA Rubin (future-proofed — NEW v3.0)
│   ├── metal_m1/                  Apple M1/M2 (Metal Shading Language)
│   ├── metal_m3/                  Apple M3/M4/M5 (Metal 4 TensorOps)
│   ├── rocm_rdna3/                AMD RX 7000 (HIP + WMMA)
│   ├── rocm_cdna3/                AMD MI300X (192GB HBM3)
│   ├── openvino_npu/              Intel Arc NPU
│   ├── cpu_avx512/                x86 AVX-512 + AMX
│   └── cpu_neon/                  ARM NEON (Apple, Qualcomm Snapdragon)
├── parallelism/
│   ├── 1gpu.json                  Single-GPU execution plan
│   ├── 2gpu.json                  2-GPU TP plan
│   ├── 4gpu.json                  4-GPU TP + PP plan
│   ├── 8gpu.json                  8-GPU full distributed plan
│   └── prefill_decode_split.json  [NEW v3.0] Disaggregated P/D pool configs
├── safety/                        [NEW v3.0] Guardrail layer
│   ├── prompt_guard.json          Prompt injection detection config
│   └── output_filter.json         Content policy filter parameters
└── manifest.json                  Top-level manifest + all hashes
```

### 3.2 The AEG-IR — Hardware-Agnostic Operator Graph

```
# AEG-IR v1.0 — Qwen3-72B Layer 0 (with v3.0 extensions)

func @transformer_layer(%x: tensor<*xbf16>, %pos: i64, %reasoning_budget: i32) -> tensor<*xbf16> {
  // RMSNorm — fuseable with QKV (Pass 1)
  %norm = aeg.rmsnorm(%x, %weight[0]) {eps = 1e-6}

  // QKV + RoPE — fused megakernel by Pass 1
  %q, %k, %v = aeg.qkv_proj(%norm, %wq[0], %wk[0], %wv[0])
  %q_rope = aeg.rope(%q, %pos) {theta = 1000000.0, rope_type = "yarn"}
  %k_rope = aeg.rope(%k, %pos) {theta = 1000000.0, rope_type = "yarn"}

  // GQA — native AEG semantic (not lowered to matmuls)
  %attn = aeg.gqa(%q_rope, %k_rope, %v) {
    num_heads = 64, num_kv_heads = 8, head_dim = 128,
    kv_cache = @global_kv_cache[layer=0],
    fa_variant = "flash_attention_3",           // selected by Stage 3 per target
    kv_precision = "fp8",                       // KV cache quantization [NEW v3.0]
    mla_compression = false,                    // true for DeepSeek-family
    reasoning_budget_gate = @reasoning_budget,  // [NEW v3.0] CoT budget control
  }

  // FFN (SwiGLU) — sensitivity=LOW → Q4_K_M (from Pass 3)
  %ffn = aeg.swiglu_ffn(%residual, %wg[0], %wu[0], %wd[0])
    {sensitivity = LOW, precision_hint = Q4_K_M}

  return aeg.add(%residual, %ffn)
}

// [NEW v3.0] Reasoning graph node — compiled CoT execution
func @reasoning_step(%context: tensor<*xbf16>, %budget: i32) -> (tensor<*xbf16>, i32) {
  %thought = aeg.reasoning_forward(%context, @transformer_layer)
    {max_tokens = %budget, early_exit_threshold = 0.95}
  %remaining = aeg.budget_decrement(%budget, %thought.length)
  return (%thought.hidden, %remaining)
}
```

### 3.3 AEG vs. LLVM Comparison

| Concept | LLVM | Aether AEG |
|---|---|---|
| **Source Languages** | C, C++, Rust, Swift, Julia | SafeTensors, GGUF, ONNX, MLX, PyTorch |
| **Intermediate Representation** | LLVM IR | AEG-IR |
| **Optimizer Passes** | mem2reg, loop-unroll, SROA, DCE | op-fusion, sensitivity-quant, shard-plan, moe-route, reasoning-graph, mla-compress |
| **Target Backends** | x86, ARM, WASM, RISC-V | CUDA sm70–sm120, Metal M1–M5, ROCm, OpenVINO, CPU |
| **Output Artifact** | `.o` / `.so` / binary | `.aeg` compiled model package |
| **Distributed Cache** | ccache | Aether Hub (content-addressed kernel cache) |
| **Portability Promise** | Any LLVM binary on any supported ISA | Any `.aeg` on any supported hardware |
| **Format Stability** | LLVM IR backward-compat 10+ years | AEG/1.x stable forever |
| **New (v3.0)** | — | Reasoning graph, MLA latent vectors, FP4, safety layer |

---

## 4. Competitive Landscape

### 4.1 Full 2026 Landscape

| Tool | Type | Owns | Missing |
|---|---|---|---|
| **vLLM** | Wrapper | PagedAttention, production API | Graph ownership, compilation, portability |
| **llama.cpp** | Wrapper | GGUF runtime, CPU/GPU kernels | Compiler, portability beyond GGUF |
| **Ollama** | Wrapper | Developer UX | Everything technical |
| **SGLang** | Wrapper + Scheduler | RadixAttention, structured gen | Compilation, format ownership |
| **TensorRT-LLM** | Compiler (NVIDIA-only) | CUDA kernel fusion, FP8/FP4 | Non-NVIDIA, open format, community |
| **Modular MAX** | Compiler (closed) | Graph compiler, Mojo kernels | Open source, open format, community |
| **ONNX Runtime** | Runtime (ONNX only) | Execution providers | LLM-specific ops, speculative decoding |
| **IREE** | Compiler (MLIR) | Hardware universality | LLM-specific optimization, developer UX |
| **Triton** | Kernel language | Custom CUDA/ROCm kernels | Model-level compilation, portability |
| **lm-evaluation-harness** | Evaluation only | Benchmark tooling | Runtime, compilation |
| **Aether** | **AI Model Compiler (open)** | **Portable AEG format + compiler** | **Nothing — this is the gap** |

### 4.2 What Nobody Has (2026)

```
     ┌─────────────────────────────────────────────────────────────┐
     │   OPEN-SOURCE AI MODEL COMPILER                              │
     │   + portable compiled format (AEG) ← only Aether            │
     │   + LLM-specialized optimizer (7 passes) ← only Aether      │
     │   + hardware-universal (NVIDIA + AMD + Apple + Intel)        │
     │   + FP4 Blackwell support ← only TRT-LLM (closed)           │
     │   + MLA native compilation pass ← nothing open has this      │
     │   + reasoning graph compiler ← COMPLETELY NEW                │
     │   + agentic workflow optimizer ← COMPLETELY NEW              │
     │   + community kernel cache (network effect moat)             │
     │                                                               │
     │                     ← AETHER LIVES HERE                      │
     └─────────────────────────────────────────────────────────────┘
```

---

## 5. The Moat

### 5.1 The Kernel Cache Network Effect (Aether Hub)

```
User A compiles qwen3-72b on H100 → uploads cuda_sm90 kernels to Hub
User B compiles qwen3-72b on B200 → uploads cuda_sm100 + FP4 kernels
User C runs qwen3-72b on H100   → downloads pre-compiled kernels → zero wait
...
After 10,000 users: every model × hardware pair is pre-compiled
Every new user gets instant startup on any hardware
```

This is the npm/Docker Hub flywheel. **The Hub becomes the moat.**

### 5.2 AEG Format Ownership

Once the ecosystem adopts AEG as the distribution format for compiled models:
- HuggingFace hosts `.aeg` alongside `.safetensors`
- Model publishers compile to AEG at release time
- Hardware vendors implement first-class AEG targets
- CI/CD pipelines compile to AEG artifacts

Mirrors Docker OCI, npm package.json, LLVM IR. Format adoption is irreversible.

### 5.3 Sensitivity-Guided Quantization Moat

Aether's quantization is mathematically grounded (not rule-based):
```
sensitivity[layer] = d(perplexity) / d(precision_bits_of_layer)
```
This requires owning the computation graph. **No wrapper can replicate this.**

### 5.4 Reasoning Graph Moat (NEW v3.0)

Aether compiles chain-of-thought reasoning as an executable graph with:
- Budget-controlled token allocation per reasoning step
- Early exit when confidence threshold is reached
- Speculative CoT: draft reasoning chains verified by target model
- 21–66% latency reduction on complex reasoning tasks

No other tool compiles reasoning as a first-class graph operation. **This is unique.**

### 5.5 MLA Native Compilation Moat (NEW v3.0)

DeepSeek-V3, Kimi K2, GLM-5 all use Multi-Head Latent Attention (MLA). Aether compiles MLA natively:
- Stores compressed latent KV vectors (not full KV states) in the AEG artifact
- Weight absorption trick compiled into kernel at Stage 3 — no runtime decompression overhead
- 90%+ KV cache reduction vs. standard GQA compilation
- Makes trillion-parameter models servable on 4× H100 instead of 8×

---

## 6. Architecture — Five Compiler Stages + Two New Systems

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        AETHER COMPILER PIPELINE v3.0                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT: Any Model Format                                                     ║
║  SafeTensors | GGUF | ONNX | MLX | PyTorch.pt | HuggingFace Hub ID          ║
║                            |                                                 ║
║  ┌─────────────────────────▼────────────────────────────────────────────┐   ║
║  │ STAGE 1 — MODEL INGESTION & GRAPH EXTRACTION                         │   ║
║  │ Architecture detection · Weight loading · Graph tracing → AEG-IR    │   ║
║  │ [NEW] MLA structure detection · Shared expert analysis              │   ║
║  └─────────────────────────┬────────────────────────────────────────────┘   ║
║                            │ AEG-IR (unoptimized)                           ║
║  ┌─────────────────────────▼────────────────────────────────────────────┐   ║
║  │ STAGE 2 — AETHER OPTIMIZER (Seven Graph-Level Passes)                │   ║
║  │  Pass 1: Operator Fusion      (RoPE+QKV+Norm → megakernel)          │   ║
║  │  Pass 2: Sensitivity Analysis (d_ppl/d_precision per layer)         │   ║
║  │  Pass 3: Precision Assignment (mixed BF16/FP8/FP4/INT4 per layer)   │   ║
║  │  Pass 4: KV Cache Structuring (PagedKV + RadixTree + MLA latents)   │   ║
║  │  Pass 5: MoE Expert Routing   (hot/warm/cold + sparsity + shared)   │   ║
║  │  Pass 6: Parallelism Discovery(auto TP/PP/EP/CP sharding)           │   ║
║  │  Pass 7: Reasoning Graph      [NEW] CoT graph + budget compiler      │   ║
║  └─────────────────────────┬────────────────────────────────────────────┘   ║
║                            │ AEG-IR (optimized)                             ║
║  ┌─────────────────────────▼────────────────────────────────────────────┐   ║
║  │ STAGE 3 — HARDWARE TARGETING & KERNEL EMISSION                       │   ║
║  │  CUDA sm70-sm100 + [NEW] sm100 FP4 (Blackwell) + sm120 (Rubin)     │   ║
║  │  Metal M1-M5 · ROCm · OpenVINO · CPU · [NEW] Qualcomm QNN          │   ║
║  │  FA-3 / [NEW] FA-4 (Blackwell) · FP8/FP4 GEMMs · MLA kernels      │   ║
║  └─────────────────────────┬────────────────────────────────────────────┘   ║
║                            │ .aeg artifact                                  ║
║  ┌─────────────────────────▼────────────────────────────────────────────┐   ║
║  │ STAGE 4 — SELF-OPTIMIZING RUNTIME                                    │   ║
║  │  Loads .aeg · EAGLE-3 speculative decoding · Disaggregated P/D      │   ║
║  │  Tiered KV cache · Dynamic precision · [NEW] Agentic scheduler      │   ║
║  │  [NEW] Reasoning budget manager · [NEW] Multi-modal graph dispatch  │   ║
║  │  [NEW] Safety guardrail layer · [NEW] OpenTelemetry observability   │   ║
║  └─────────────────────────┬────────────────────────────────────────────┘   ║
║                            │                                                 ║
║  ┌─────────────────────────▼────────────────────────────────────────────┐   ║
║  │ STAGE 5 — DEVELOPER INTERFACE                                        │   ║
║  │  Python SDK · REST API (OpenAI-compat) · CLI · gRPC · [NEW] JS SDK  │   ║
║  │  [NEW] Eval gates · [NEW] A/B rollout API · [NEW] Fleet management  │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  CROSS-CUTTING SYSTEMS (v3.0)                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │ Aether Hub — content-addressed kernel cache + AEG model registry   │    ║
║  │ Aether Observability — OpenTelemetry tracing + metrics + eval gates│    ║
║  │ Aether Safety — prompt guard + output filter + audit logging       │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 7. Stage 1 — Model Ingestion

### 7.1 Supported Input Formats (v3.0 Complete List)

| Format | Source | Ingestion Method | Status |
|---|---|---|---|
| **SafeTensors** | HuggingFace | Weight loading + config.json arch parsing | Phase 1 |
| **GGUF** | llama.cpp | Binary header parsing + K/I-quant dequant | Phase 1 |
| **ONNX** | Cross-framework | Protobuf graph → AEG-IR lowering | Phase 1 |
| **MLX** | Apple ecosystem | MLX module tracing → AEG-IR | Phase 2 |
| **PyTorch `.pt`** | Legacy | `torch.export` graph capture → AEG-IR | Phase 1 |
| **HuggingFace Hub ID** | Cloud | Auto-download + compile pipeline | Phase 1 |
| **AWQ quant** | Quantized models | AWQ weight loading with scale recovery | Phase 2 |
| **GPTQ quant** | Quantized models | GPTQ weight loading + Marlin kernel emit | Phase 2 |
| **TensorRT engine** | NVIDIA-compiled | Engine plan → AEG wrapper | Phase 3 |

### 7.2 Architecture Detection (v3.0 Expanded)

```python
ARCHITECTURE_PATTERNS = {
    # Standard transformer families
    "llama_family":    {"attn": "GQA",         "ffn": "SwiGLU",   "norm": "RMSNorm"},
    "qwen_family":     {"attn": "GQA+QKNorm",  "ffn": "SwiGLU",   "rope": "YaRN"},
    "gemma_family":    {"attn": "MQA",          "ffn": "GeGLU",    "norm": "RMSNorm"},
    "phi_family":      {"attn": "MHA",          "ffn": "SwiGLU",   "norm": "LayerNorm"},
    "mistral_family":  {"attn": "GQA+SW-Attn",  "ffn": "SwiGLU",   "window": True},

    # MLA families (NEW v3.0)
    "deepseek_family": {"attn": "MLA",          "ffn": "MoE",      "rope": "NTK-aware",
                        "shared_experts": True, "mla_rank": 512},
    "kimi_family":     {"attn": "MLA",          "ffn": "MoE",      "rope": "YaRN"},
    "glm_family":      {"attn": "MLA",          "ffn": "SwiGLU"},

    # MoE families
    "moe_generic":     {"ffn": "MoE",           "router": "TopK"},
    "mixtral_family":  {"attn": "GQA",          "ffn": "MoE",      "num_experts": 8},

    # Vision-Language (NEW v3.0)
    "qwen_vl_family":  {"encoder": "ViT-Dynamic","cross_attn": "early_fusion",
                        "visual_tokens": "dynamic", "image_rope": True},
    "llava_family":    {"encoder": "CLIP-ViT",  "connector": "MLP"},
    "internvl_family": {"encoder": "InternViT", "cross_attn": "late_fusion"},

    # Reasoning models (NEW v3.0)
    "reasoning_family": {"cot_mode": True,      "budget_tokens": True,
                         "early_exit": True,     "reflection": True},

    # SSM / non-transformer (NEW v3.0)
    "mamba_family":    {"type": "SSM",           "selective_scan": True},
    "rwkv_family":     {"type": "linear_attn",   "token_shift": True},
}
```

### 7.3 MLA Structure Detection (NEW v3.0)

For DeepSeek, Kimi K2, GLM-5, and any model using Multi-Head Latent Attention:

```python
class MLADetector:
    """Detects MLA architecture and extracts compression parameters."""

    def detect(self, graph: AEGGraph) -> MLAConfig | None:
        """Return MLA config if model uses Multi-Head Latent Attention."""
        # Look for low-rank projection pattern: Q/K/V projected through small bottleneck
        for layer in graph.layers:
            if self._has_low_rank_kv_projection(layer):
                return MLAConfig(
                    kv_lora_rank=self._detect_lora_rank(layer),      # e.g., 512
                    kv_head_dim=self._detect_head_dim(layer),         # e.g., 128
                    q_lora_rank=self._detect_q_rank(layer),           # e.g., 1536
                    rope_decoupled=self._detect_decoupled_rope(layer), # True for DeepSeek
                    weight_absorption_possible=True,
                )
        return None
```

### 7.4 Shared Expert Detection (NEW v3.0)

For DeepSeek-style MoE with always-active shared experts:

```python
def detect_shared_experts(graph: AEGGraph) -> SharedExpertConfig | None:
    """Detect DeepSeek-style shared experts that activate for every token."""
    for layer in graph.moe_layers:
        if hasattr(layer, "shared_expert_count") and layer.shared_expert_count > 0:
            return SharedExpertConfig(
                count=layer.shared_expert_count,         # typically 1-2
                always_active=True,                       # never offloaded to CPU
                memory_budget_fraction=0.05,              # 5% of GPU HBM reserved
            )
    return None
```

---

## 8. Stage 2 — Seven Optimizer Passes

### Pass 1: Operator Fusion (Updated v3.0)

Identifies fuseable sequences and merges into hardware megakernels:

```
Before Fusion (6 kernel launches — 6x DRAM round-trips):
  rmsnorm → q_proj → k_proj → v_proj → rope_q → rope_k

After Fusion (1 megakernel — 1x DRAM round-trip):
  fused_qkv_rope_norm(x, wq, wk, wv, pos)

[NEW v3.0] MLA-aware fusion:
  mla_compress_kv(x, w_kv_a, w_kv_b) → fused_latent_kv_rope
  Stores only latent KV vector — 10x smaller KV cache node
```

**Updated research basis:**
- ClusterFusion (NeurIPS 2025): ClusterReduce/ClusterGather — 1.6–2.0x speedup
- FlashAttention-4 (Blackwell 2026): asymmetric pipeline design, 1613 TFLOPs/s on B200
- Agentic MLIR (2026): LLM-planned transform IR for automated pass ordering

### Pass 2: Sensitivity Analysis (Updated v3.0)

Computes per-layer sensitivity using calibration set differentiation:

```python
# v3.0: Extended sensitivity scoring
sensitivity_map = {}
for layer in graph.layers:
    baseline_ppl = evaluate_perplexity(model, calibration_set, full_precision)
    for precision in [INT4, FP8, FP4, Q4_K_M, IQ3_XS]:
        with quantize_layer(layer, precision):
            quant_ppl = evaluate_perplexity(model, calibration_set)
        sensitivity_map[layer][precision] = (quant_ppl - baseline_ppl) / bits_saved(layer, precision)

# [NEW v3.0] Task-specific calibration sets
CALIBRATION_SETS = {
    "reasoning": "aime-2024 + gsm8k + math-500",    # for o1/R1-style models
    "coding":    "humaneval + mbpp + swebench-lite",  # for Qwen-Coder, CodeLlama
    "general":   "wikitext-2 + hellaswag + mmlu",    # default
    "multilingual": "flores-200 + xcopa",             # for multilingual models
}
```

**New research:** AutoMixQ (2025), AMQ Framework (2025), GPTQ + Marlin kernels for INT4, NVFP4/MXFP4 for Blackwell.

### Pass 3: Precision Assignment (Updated — FP4 Support)

Per-layer precision now includes FP4 for Blackwell targets:

| Sensitivity | Typical Layers | Default | Blackwell B200 |
|---|---|---|---|
| > 0.9 (critical) | Embedding, LM Head | BF16 | BF16 |
| 0.7–0.9 (high) | Attn Q/K, first layers | FP8 | FP8 |
| 0.4–0.7 (medium) | Attn V/O, middle | Q4_K_M | FP4 (NVFP4) |
| < 0.4 (low) | FFN deep layers | IQ3_XS | FP4 (NVFP4) |
| MLA latent vectors | KV compression | BF16 | BF16 (loss-sensitive) |

**FP4 target:** 4x throughput vs H100 on B200; 9 PFLOPS peak dense FP4 compute; 95–99% accuracy retention with dual-level scaling.

### Pass 4: KV Cache Structuring (Updated — MLA + Cross-Session)

```
Standard KV Cache (GQA):
  Per layer: store K[seq, kv_heads, head_dim] + V[seq, kv_heads, head_dim]
  Memory: ~2.0 GB per request at 128K context (Qwen3-72B, GQA-8)

[NEW v3.0] MLA Compressed KV Cache (DeepSeek-family):
  Per layer: store compressed_kv[seq, lora_rank] only (rank=512)
  Memory: ~0.2 GB per request at 128K context
  Reduction: 90%+ KV cache savings (57x in some configurations)
  Reconstruction: weight absorption compiled into kernel — zero runtime overhead

[NEW v3.0] Cross-Session KV Sharing:
  System prompts shared across all sessions → one KV block per prompt hash
  RAG documents shared across concurrent users → deduplicated KV blocks
  Expected hit rate: 40–70% in RAG/agentic workloads
  Tier: L4 Aether Hub CDN for globally common system prompts
```

**Research:** Mooncake Conductor KV-cache-aware routing, PrefillOnly (memory-efficient prefill), LMCache for MoE KV management.

### Pass 5: MoE Expert Routing (Updated — Shared Experts + Fine-Grained)

```
[NEW v3.0] DeepSeek-V3 Fine-Grained Routing (256 experts, top-8):
  shared experts:
    - count: 1 (always activated for every token)
    - always pinned to GPU HBM — never offloaded
    - compiled as static kernel, not dispatched through router

  routed experts classification (256 total, top-8 activated):
    hot (>2% activation):    ~25 experts → GPU HBM resident
    warm (0.02-2%):         ~100 experts → CPU DRAM + CommitMoE prefetch
    cold (<0.02%):          ~131 experts → NVMe lazy-load

[NEW v3.0] Auxiliary-Loss-Free Router Compilation:
  DeepSeek's bias-term gating (not auxiliary loss)
  Compiles to: router(x) = softmax(x @ W_gate + bias_per_expert)
  No auxiliary loss term in inference graph → cleaner kernel emission

Intra-Expert Sparsity (existing, now extended):
  Identify zero-channel masks per expert on calibration data
  Emit sparse GEMM kernels skipping dead channels → 2.5x expert speedup
```

**Research:** DeepSeek-V3 technical report (fine-grained 256-expert MoE), FineMoE (semantic prefetch), MoE-Infinity, CommitMoE, ISCA 2026 prefill-aware expert placement.

### Pass 6: Parallelism Discovery (Updated — Disaggregated Plans)

```python
class ParallelismSolver:
    def search(self, graph: AEGGraph, hw: HardwareProfile) -> ShardingPlan:
        # [NEW v3.0] separate plans for prefill pool vs decode pool
        prefill_plan = self._search(graph, hw, phase="prefill",
            priorities=["TP_high", "batch_max"])     # compute-bound: max TP
        decode_plan  = self._search(graph, hw, phase="decode",
            priorities=["TP_low", "batch_small"])    # memory-bound: min TP, max batch

        # [NEW v3.0] MLA-aware: latent KV is smaller → fit more decode batches
        if graph.has_mla:
            decode_plan.batch_size_multiplier = 10   # 10x more concurrent requests

        # [NEW v3.0] Vision encoder uses DP, not TP (avoids all-reduce overhead)
        if graph.has_vision_encoder:
            return ShardingPlan(
                vision_encoder_strategy="data_parallel",
                language_backbone_strategy=decode_plan,
            )
        return ShardingPlan(prefill=prefill_plan, decode=decode_plan)
```

**Research:** Seesaw MLSys 2025 (dynamic re-sharding 25–40% gain), vLLM/SGLang hybrid ViT-DP + LLM-TP for VLM inference (2026), DistServe 3–4x goodput.

### Pass 7: Reasoning Graph Compiler (NEW v3.0)

This is Aether's most differentiated new pass, designed for o1/R1/Claude-style reasoning models:

```
Traditional Inference Graph:
  input → [single forward pass] → output token

[NEW] Aether Reasoning Graph:
  input → [initialize reasoning budget B]
    → reasoning_step_0 → check_confidence → [continue / early_exit]
    → reasoning_step_1 → check_confidence → [continue / early_exit]
    → ... up to N steps
    → synthesis_step → output

Compiled representation (stored in .aeg/graph/reasoning_graph.aeg-ir):
  func @reasoning_loop(%ctx, %budget: i32) -> tensor<*xbf16> {
    %state = %ctx
    %remaining = %budget
    while(%remaining > 0) {
      %thought = aeg.reasoning_forward(%state, @transformer_layer)
        {early_exit_threshold = 0.95}
      %confidence = aeg.reasoning_confidence(%thought)
      %state = aeg.update_context(%state, %thought)
      %remaining = aeg.budget_decrement(%remaining, %thought.length)
      if aeg.should_exit(%confidence, %remaining) { break }
    }
    return aeg.synthesis(%state, @transformer_layer)
  }
```

**Speculative CoT (integrated with Stage 4 speculation engine):**
```
Draft Model generates reasoning chain (cheap)
Target Model verifies/corrects reasoning chain (expensive, but once)
Result: 21–66% latency reduction on complex reasoning (MATH-500, AIME-2024)
```

**Research:** Tree-of-Thoughts (ToT), Graph-of-Thoughts (GoT), Speculative CoT (SCoT) 21–66% reduction, Diagram of Thought, GAN-CoT iterative refinement, RLVR (Reinforcement Learning with Verifiable Rewards).

---

## 9. Stage 3 — Hardware Targeting and Kernel Emission

### 9.1 Updated Target Profiles (v3.0)

| Target ID | Hardware | New in v3.0 | Key Capability |
|---|---|---|---|
| `cuda_sm70` | NVIDIA V100 | — | Tensor Cores, FP16 |
| `cuda_sm80` | NVIDIA A100 | — | BF16 TMA |
| `cuda_sm89` | NVIDIA RTX 4090 | — | FP8 Ada Tensor Cores |
| `cuda_sm90` | NVIDIA H100/H200 | FA-3 production | WGMMA + TMA + FP8 |
| `cuda_sm100` | NVIDIA B200 | **FP4 Blackwell NEW** | FP4 + 8TB/s + FA-4 |
| `cuda_sm120` | NVIDIA Rubin (2027) | **Future-proofed NEW** | Next-gen (placeholder) |
| `metal_m1` | Apple M1/M2 | — | Metal Shading Language |
| `metal_m3` | Apple M3/M4/M5 | Metal 4 TensorOps | Neural Engine, BF16 |
| `rocm_rdna3` | AMD RX 7000 | — | HIP + WMMA |
| `rocm_cdna3` | AMD MI300X | — | 192GB HBM3, 5.3TB/s |
| `openvino_npu` | Intel Arc NPU | — | OpenVINO NPU runtime |
| `qualcomm_qnn` | Qualcomm AI 100 | **NEW v3.0** | QNN SDK, Hexagon DSP |
| `cpu_avx512` | Modern x86_64 | AMX tensor tiles | AVX-512 + AMX |
| `cpu_neon` | ARM (Apple, Snapdragon) | — | NEON SIMD |

### 9.2 FlashAttention-3 and FlashAttention-4 (NEW v3.0)

| Target | Attention Implementation | Details |
|---|---|---|
| `cuda_sm90` (H100/H200) | **FlashAttention-3** | WGMMA + TMA; 840 TFLOPs BF16; 1.2+ PFLOPs FP8; asynchronous producer/consumer warp split |
| `cuda_sm100` (B200) | **FlashAttention-4 NEW** | ~1613 TFLOPs/s; redesigned pipeline for B200's asymmetric scaling; larger tile sizes |
| `cuda_sm80` (A100) | FlashAttention-2 | TMA-based; production standard |
| `metal_m3` | Metal 4 TensorOps | Apple Neural Engine attention |
| Others | xFormers memory-efficient | Fallback for all other targets |

**FA-3 details:** Asynchronous execution + warp specialization. Producer warps use TMA to load K/V tiles asynchronously while consumer warps perform GEMM. 33% faster than FP16 with FP8 incoherent processing.

### 9.3 FP4 Blackwell Kernel Emission (NEW v3.0)

```python
class BlackwellKernelEmitter:
    """Emits FP4 kernels for NVIDIA Blackwell (sm_100) architecture."""

    NVFP4_FORMAT = "E2M1"                # 4-bit float: 2 exponent, 1 mantissa
    MXFP4_FORMAT = "microscaling_fp4"    # Microscaling FP4 (MXFP4 standard)

    def emit_fp4_gemm(self, layer: AEGNode, precision_map: dict) -> KernelDescriptor:
        """Emit FP4 GEMM kernel for layers with sensitivity < 0.4."""
        if precision_map.get(layer.name) == "FP4":
            return KernelDescriptor(
                kernel_type="fp4_gemm_blackwell",
                format=self.NVFP4_FORMAT,
                scaling="per_tensor_fp32",    # dual-level scaling for accuracy
                peak_compute="9_pflops",
                memory_footprint_ratio=0.25,  # vs BF16
                accuracy_retention="95-99%",
            )

    def emit_fp4_attention(self, attn_node: AEGNode) -> KernelDescriptor:
        """Emit FP4 attention + FlashAttention-4 for Blackwell."""
        return KernelDescriptor(
            kernel_type="flash_attention_4_fp4",
            tflops_peak=1613,
            pipeline="asymmetric_blackwell",
            kv_cache_format="fp4_kv",         # KV cache in FP4 for max context
        )
```

**Impact:** On B200, FP4 across medium/low sensitivity layers yields 4x inference throughput vs H100 baseline. Models previously requiring 8× H100 fit on 2× B200.

### 9.4 MLA Kernel Emission (NEW v3.0)

```python
class MLAKernelEmitter:
    """Emits MLA kernels that absorb projection weights at compile time."""

    def emit_mla_attention(self, mla_config: MLAConfig, target: str) -> KernelDescriptor:
        """Emit MLA attention with weight absorption compiled in."""
        # Weight absorption: merge W_kv_b into W_o at compile time
        # Result: no separate decompression step during inference
        absorbed_weights = self._absorb_weights(
            w_kv_b=mla_config.kv_up_proj,
            w_o=mla_config.output_proj,
        )
        return KernelDescriptor(
            kernel_type="mla_absorbed_attention",
            kv_lora_rank=mla_config.kv_lora_rank,        # e.g., 512
            stored_kv_size_ratio=0.05,                    # 5% vs standard GQA
            rope_type="decoupled",                        # separate positional RoPE
            absorbed_weights=absorbed_weights,            # baked into kernel
            kv_cache_compression_ratio=0.1,              # 90%+ reduction
        )
```

---

## 10. Stage 4 — Self-Optimizing Runtime

### 10.1 Startup Flow (v3.0)

```
aether run qwen3-72b

1. Detect hardware fingerprint (GPU model, VRAM, SM version, driver, NVLink topology)
2. Load manifest from qwen3-72b.aeg
3. Select pre-compiled kernel set for detected hardware
   - Hub cache hit: download in seconds, zero recompilation
   - No cache hit: compile locally + upload to Hub in background
4. Load sharding plan matching GPU count from .aeg/parallelism/
5. Initialize KV cache across tiers (GPU HBM / CPU DRAM / NVMe / Hub CDN)
6. [NEW v3.0] Initialize MLA compressed KV store (if MLA architecture)
7. [NEW v3.0] Initialize reasoning budget manager (if reasoning model)
8. Start disaggregated prefill/decode scheduler
9. Start EAGLE-3 speculative decoding engine
10. [NEW v3.0] Initialize safety guardrail layer
11. [NEW v3.0] Start OpenTelemetry trace exporter
12. Ready to serve
```

### 10.2 EAGLE-3 Speculative Decoding Engine (Updated — Replacing OPT-Tree)

v3.0 upgrades the speculative engine from OPT-Tree to **EAGLE-3** (multi-layer feature fusion), which is now the production standard in vLLM, SGLang, and TensorRT-LLM:

```
EAGLE-3 Architecture:
  Standard speculative decoding: uses final hidden state only for draft head
  EAGLE-3: multi-layer feature fusion → aggregates ALL transformer layers

  Layer 0 hidden: [h_0]
  Layer 16 hidden: [h_16]     All layers fused via
  Layer 32 hidden: [h_32] →   lightweight aggregator → draft head
  Layer 48 hidden: [h_48]
  Layer 64 hidden: [h_64]

  Result: draft model has global context → significantly higher acceptance rate
  EAGLE-3.1 extension: addresses "attention drift" in long sequences
  Group Tree Optimization (GTO): aligns draft training with decoding-time tree policy
```

**Implementation in AEG Runtime:**
```python
class EAGLE3Engine:
    """EAGLE-3 speculative decoding with multi-layer feature fusion."""

    def __init__(self, target_model_id: str, draft_heads: int = 4) -> None:
        self.target = target_model_id
        self.draft_model = self._load_eagle3_head(draft_heads)
        # EAGLE-3.1: attention drift correction
        self.drift_correction = True

    def build_draft_tree(self, hidden_states: list[Tensor], max_depth: int = 6,
                         max_width: int = 5) -> DraftTreeNode:
        """Build OPT-Tree using ALL hidden states (EAGLE-3 multi-layer fusion)."""
        # Aggregate hidden states from all layers
        fused_features = self._fuse_hidden_states(hidden_states)
        return self._opt_tree_construct(fused_features, max_depth, max_width)

    def verify_tree_deft(self, draft_tree: DraftTreeNode,
                         target_logits: Tensor) -> list[int]:
        """Verify using DeFT KV-Guided Grouping (3.59x attn latency reduction)."""
        flat_tokens, group_indices = self._flatten_tree_deft(draft_tree)
        verified = self._tree_masked_attention(flat_tokens, group_indices, target_logits)
        return self._accept_longest_path(verified, draft_tree)
```

**Measured performance (2026 production benchmarks):**
- Standard chat (Qwen3-72B + Qwen3-1.5B draft): 3–4x throughput improvement
- Code completion (JetSpec causal parallel): up to 9.64x speedup
- Reasoning (Speculative CoT): 21–66% latency reduction
- High-batch feasibility: gamma-tolerance analysis shows >=75% acceptance rate at batch ≤ 32

### 10.3 KV Cache Manager (v3.0 — MLA + Cross-Session)

| Tier | Storage | What | Eviction | New in v3.0 |
|---|---|---|---|---|
| **L1 — GPU HBM** | On-device VRAM | Active KV + MLA latents | LRU + priority | MLA latent storage |
| **L2 — CPU DRAM** | System RAM | Prefix cache, warm KV | Cost-aware LRU | Cross-session sharing |
| **L3 — NVMe SSD** | Persistent | Long prompts, RAG KV | TTL + frequency | Persistent across restarts |
| **L4 — Aether Hub** | CDN global | Common system prompt KV | CDN TTL | Cross-user sharing |

**KV Cache Quantization (NEW v3.0):**
```python
KV_CACHE_PRECISION_STRATEGY = {
    "recent_tokens":    "FP8",    # last 2048 tokens — high precision
    "middle_tokens":    "INT4",   # tokens 2048–32768 — moderate compression
    "distant_tokens":   "INT2",   # tokens >32768 — aggressive compression
    # Result: 70% KV cache memory reduction at <1% quality loss on long context
}
```

### 10.4 Disaggregated Prefill/Decode Scheduler (v3.0)

```
PREFILL SCHEDULER:
  Chunked prefill: <=2048 tokens/chunk (Sarathi-Serve design)
  Interleaved with decode iterations: no head-of-line blocking
  [NEW] Prefill-aware expert placement: hot MoE experts pre-loaded before prefill chunk
  [NEW] Mooncake Conductor: KV-cache-aware routing to minimize data movement

DECODE SCHEDULER:
  Continuous batching: iteration-level admission (Orca design)
  EAGLE-3 tree verification: entire draft tree in one forward pass
  [NEW] Reasoning budget tracking: decrement budget per thinking token
  [NEW] Dynamic model routing: simple queries → fast model; complex → full model

CROSS-CUTTING:
  PrefillOnly mode: prefill-heavy workloads retain only final-layer KV (NEW v3.0)
  MuxWise: spatial-temporal multiplexing to meet strict SLOs (NEW v3.0)
```

### 10.5 Dynamic Precision Adjustment (v3.0)

```python
class DynamicPrecisionManager:
    """Auto-adjusts precision under memory pressure using sensitivity map."""

    # v3.0: now includes FP4 as a new compression target
    DOWNGRADE_SEQUENCE = ["BF16 → FP8 → FP4 → Q4_K_M → IQ3_XS"]

    def under_pressure(self, vram_utilization: float) -> None:
        if vram_utilization > 0.90:
            # Downgrade lowest-sensitivity layers first (from AEG precision_map)
            for layer in self.sorted_by_sensitivity_asc():
                new_precision = self.next_lower_precision(layer.current_precision)
                self.apply_precision_change(layer, new_precision, log=True)
                if self.vram_utilization() < 0.80:
                    break

    def [NEW_v30]_fp4_compress_kv(self) -> None:
        """[NEW] Compress KV cache to FP4 under extreme memory pressure."""
        for tier in [L1_GPU, L2_CPU]:
            tier.compress_kv(precision="INT2", except_recent_tokens=2048)
```

---

## 11. Stage 5 — Developer Interface

### 11.1 Complete Python SDK (v3.0)

```python
from aether import Runtime, Compiler, CompilerConfig, RuntimeConfig

# ── Compiler ─────────────────────────────────────────────────────────────────
compiler = Compiler(config=CompilerConfig(
    quality_budget=0.02,
    calibration_dataset="general",      # or "reasoning", "coding", "multilingual"
    targets=["auto"],
    optimization_level=2,
    enable_moe_compiler=True,
    enable_mla_compiler=True,           # [NEW v3.0] MLA native compilation
    enable_reasoning_graph=True,        # [NEW v3.0] CoT graph compilation
    enable_fp4=True,                    # [NEW v3.0] FP4 Blackwell targeting
    upload_kernels=True,
))

aeg = compiler.compile("deepseek-r1-671b")
print(aeg.graph_summary())        # fusion stats
print(aeg.precision_map())        # per-layer precision (now includes FP4)
print(aeg.mla_config())           # [NEW] MLA compression parameters
print(aeg.reasoning_graph())      # [NEW] compiled CoT graph
print(aeg.quality_report())
aeg.save("./deepseek-r1-671b.aeg")
aeg.upload()

# ── Runtime ──────────────────────────────────────────────────────────────────
rt = Runtime(config=RuntimeConfig(
    optimize_for="latency",
    speculative_decoding="eagle3",      # [NEW v3.0] EAGLE-3 (not OPT-Tree default)
    reasoning_budget=16384,             # [NEW v3.0] max thinking tokens for CoT
    enable_safety_layer=True,           # [NEW v3.0] prompt guard + output filter
    telemetry_endpoint="otel://...",    # [NEW v3.0] OpenTelemetry endpoint
    model_routing={                     # [NEW v3.0] cascade routing
        "simple": "qwen3-8b",
        "complex": "qwen3-72b",
        "reasoning": "deepseek-r1-671b",
    },
))

# Text generation
resp = rt.generate("qwen3-72b", "Explain quantum entanglement", max_tokens=512)
print(resp.text)
print(f"TPS: {resp.metrics.throughput_tps}")
print(f"TTFT: {resp.metrics.ttft_ms}ms")
print(f"EAGLE-3 accept rate: {resp.metrics.spec_accept_rate}")   # e.g. 0.87
print(f"KV hit rate: {resp.metrics.kv_cache_hit_rate}")          # e.g. 0.62
print(f"Precision: {resp.metrics.active_precision}")             # e.g. "mixed_fp8_fp4"
print(f"MLA compression: {resp.metrics.mla_kv_compression}")    # [NEW] e.g. "10x"

# Reasoning with budget
resp = rt.generate(
    "deepseek-r1-671b",
    "Solve AIME 2024 Problem 15...",
    reasoning_budget=8192,              # [NEW] token budget for thinking
    reasoning_mode=True,               # [NEW] enable reasoning graph
    stream_thinking=False,             # [NEW] hide thinking tokens from output
)

# [NEW v3.0] Cascade routing
resp = rt.generate_cascade(
    query="What is 2+2?",              # routed to qwen3-8b (simple)
)
resp2 = rt.generate_cascade(
    query="Prove the Riemann hypothesis",  # routed to deepseek-r1-671b (reasoning)
)

# [NEW v3.0] Agentic context manager (long multi-turn with KV reuse)
async with rt.agentic_session("qwen3-72b", system="You are a coding assistant") as session:
    resp = await session.generate("Write a binary search tree in Python")
    resp2 = await session.generate("Now add a delete method")
    # KV cache from turn 1 is reused in turn 2 — zero re-prefill cost

# [NEW v3.0] Eval gate — validate before serving
eval_results = rt.eval_gate(
    model="qwen3-72b",
    benchmarks=["hellaswag", "mmlu", "gsm8k"],
    baseline_model="qwen3-72b-bf16",
    max_regression=0.02,               # fail if >2% regression on any benchmark
)
if not eval_results.passed:
    raise ValueError(f"Eval gate failed: {eval_results.regressions}")

# [NEW v3.0] A/B rollout
rt.ab_rollout(
    model_a="qwen3-72b-v1",
    model_b="qwen3-72b-v2",
    traffic_split=0.1,                 # 10% to v2 initially
    auto_rollout=True,                 # increase traffic if v2 wins
    rollback_on_regression=True,
)
```

### 11.2 REST API v3.0

```
# Inference (v1 compatible, new fields added)
POST   /v1/generate           Text completion + [NEW] reasoning_budget, reasoning_mode
POST   /v1/chat               Chat (OpenAI-compatible, OpenAI Responses API-compatible)
POST   /v1/embeddings         Embedding generation
POST   /v1/rerank             Document reranking
POST   /v1/transcribe         Audio transcription
POST   /v1/generate/cascade   [NEW v3.0] Cascade model routing

# Compilation
POST   /v1/compile            Compile model to AEG (async job)
GET    /v1/compile/{job_id}   Job status and progress

# Model management
GET    /v1/models             List compiled models
POST   /v1/models/pull        Download + compile from Hub or HF
DELETE /v1/models/{name}      Remove
GET    /v1/models/{name}      Info + AEG metadata
GET    /v1/models/{name}/graph     AEG-IR inspection
GET    /v1/models/{name}/mla       [NEW] MLA compression stats
GET    /v1/models/{name}/reasoning [NEW] Reasoning graph inspection

# Operations (NEW v3.0)
POST   /v1/eval               Run eval gate benchmark suite
GET    /v1/eval/{job_id}      Eval results
POST   /v1/ab/start           Start A/B rollout
GET    /v1/ab/{experiment_id} A/B experiment stats
POST   /v1/ab/rollback        Rollback to model_a

# Observability
GET    /v1/metrics            Prometheus-compatible metrics
GET    /v1/traces             [NEW] OpenTelemetry trace export
GET    /v1/health             Health check + GPU status
GET    /v1/hardware           Hardware fingerprint
GET    /v1/kernels            Active kernel targets
```

---

## 12. NEW: Reasoning Graph Compiler (Pass 7 Deep Dive)

This is the most architecturally novel component of Aether v3.0 — the first compiler pass that treats chain-of-thought reasoning as a first-class compiled artifact.

### 12.1 The Problem

Reasoning models (DeepSeek-R1, Claude 3.7, o3, Qwen3) generate thousands of thinking tokens before producing an answer. Current inference engines treat these as plain text generation — no compiler visibility, no optimization, no budget enforcement.

This means:
- No early exit when reasoning confidence is high (wasted compute)
- No speculative reasoning (expensive target model does all thinking)
- No compiled reasoning strategy per model (re-learned at runtime)
- No global budget control (model may think for 32K tokens on a simple question)

### 12.2 The Aether Solution — Compiled Reasoning Graphs

Stage 1 detects reasoning architecture. Stage 2 Pass 7 compiles CoT graph into .aeg at path .aeg/graph/reasoning_graph.aeg-ir. The runtime loads the reasoning graph from AEG. Speculative CoT uses a lightweight draft model to generate thoughts which the target model then verifies (21-66% latency reduction on complex reasoning tasks).

### 12.3 Reasoning Budget Controller

`python
class ReasoningBudgetController:
    COMPLEXITY_BUDGET_MAP = {
        "simple":  512,
        "medium":  2048,
        "hard":    8192,
        "max":     32768,
    }

    def compute_budget(self, prompt: str, explicit_budget: int | None) -> int:
        if explicit_budget:
            return explicit_budget
        complexity = self.estimate_complexity(prompt)
        return self.COMPLEXITY_BUDGET_MAP[complexity]
`

---

## 13. NEW: MLA Native Support

Multi-Head Latent Attention (MLA) is used by DeepSeek-V3 (671B), Kimi K2, GLM-5. It provides 90%+ KV cache reduction. Aether v3.0 supports MLA natively.

Standard GQA: K_cache [seq, kv_heads=8, head_dim=128] + V_cache = 256 MB per request at 8K context across 128 layers.
MLA Compressed: C_KV [seq, lora_rank=512] only = 12.8 MB per request at 8K context (20x reduction).

Weight Absorption: W_kv_b is merged into W_o at compile time. The up-projection from latent to full K/V is NEVER materialized in VRAM. Zero runtime decompression overhead.

For DeepSeek-R1-671B on H100 with 4x tensor parallelism:
- Without Aether MLA: 8x H100 required
- With Aether MLA Native Compilation: 4x H100 sufficient
- KV cache for 100 concurrent requests at 32K context: 1.6 GB vs 80+ GB standard

---

## 14. NEW: FP4 Blackwell Targeting

### 14.1 Blackwell vs Hopper

| Spec | H100 | B200 | Gain |
|---|---|---|---|
| Native FP4 | No | Yes (NVFP4+MXFP4) | — |
| FP4 compute | — | 9.0 PFLOPS dense | — |
| Memory | 80 GB HBM3 | 192 GB HBM3e | 2.4x |
| Bandwidth | 3.35 TB/s | 8.0 TB/s | 2.4x |
| Inference throughput | Baseline | ~4x vs H100 | 4x |

FP4 Model Size: Qwen3-72B BF16 = 144 GB (requires 2x H100). Qwen3-72B with Aether FP4 = ~36 GB (fits 1x B200 with room for larger context).

MXFP4 (Microscaling): OCP open standard. 16 elements share one FP8 scaling factor. Format E2M1 (4-bit: 2 exponent + 1 mantissa + 1 sign).

---

## 15. NEW: Agentic Workflow Optimizer

### 15.1 The Problem

Agent tasks making 10 LLM calls with a 2000-token system prompt:
- Without Aether: 20,000 tokens prefilled (wasted)
- With Aether: 2,000 tokens (KV cache reused across all turns)
- Savings: 90% prefill reduction for long-context agentic workflows

### 15.2 Cascade Model Routing

`python
ROUTING_STRATEGY = {
    "simple_factual":  "qwen3-8b",
    "code_generation": "qwen3-72b",
    "math_reasoning":  "deepseek-r1-671b",
    "multimodal":      "qwen3-vl-72b",
}
`

---

## 16. NEW: Multi-Modal Unified Graph

VLMs treated as unified computation graphs (not ViT + LLM glued together). New AEG ops: aeg.vision_encode, aeg.audio_encode, aeg.early_fuse, aeg.dynamic_resolution_resize.

Hybrid Parallelism: ViT-DP (data parallel, no all-reduce) + LLM-TP (tensor parallel). Rationale: TP causes expensive all-reduce for small ViT (<10B params). DP avoids this.

Visual Token Compression: 75% token reduction with <2% quality loss using dynamic token merging (Fast-VLM approach). Reduces quadratic attention cost for high-res images.

---

## 17. NEW: EAGLE-3 Speculative Engine

| Approach | Acceptance Rate | Speedup |
|---|---|---|
| Standard | 0.6–0.7 | 2–3x |
| EAGLE-2 | 0.7–0.8 | 3–4x |
| EAGLE-3 | 0.8–0.9 | 3–5x |
| JetSpec | Very high (code) | 9.64x |
| Speculative CoT | Task-specific | 21-66% |

EAGLE-3: Multi-layer feature fusion aggregates ALL transformer layers for draft head. EAGLE-3.1: addresses attention drift in long sequences. DeFT: tree flattened to groups sharing KV — 3.59x attention latency reduction. GTO: aligns draft training with decoding-time tree policy.

---

## 18. NEW: Quantization-Aware Compilation Pipeline

Complete format support: BF16, FP8 (E4M3/E5M2), FP4 (NVFP4 E2M1) NEW, MXFP4 NEW, INT8, INT4 (Q4_K_M), INT4 (AWQ), INT4 (GPTQ+Marlin), IQ3_XS, IQ2_XXS, FP4 KV Cache NEW, INT8 KV Cache NEW, INT2 KV Cache NEW.

Eval Gate: Mandatory quality benchmark before production. Tests on hellaswag, mmlu, gsm8k, math-500, humaneval per task type. Fails if >2% regression vs baseline.

---

## 19. NEW: Aether Observability Stack

OpenTelemetry-native tracing for every request phase. Key metrics: tokens/sec, TTFT P50/P95/P99, EAGLE-3 accept rate, KV hit rate, MLA compression ratio, reasoning budget used, GPU VRAM utilization. Quality drift monitoring: alerts if live win-rate drops >5% vs baseline.

---

## 20. NEW: AEG Safety and Guardrail Layer

Compiled into .aeg/safety/: prompt_guard.json, output_filter.json, audit_log.json. Runtime safety: prompt injection detection, content policy enforcement, immutable audit trail for compliance. Configurable thresholds. All decisions logged with SHA-256 request hash.

---

## 21. Developer API (Complete v3.0 CLI Reference)

`ash
aether compile <model>
aether compile --target cuda_sm100
aether compile --fp4
aether compile --mla-native
aether compile --reasoning-graph
aether compile --calibration reasoning
aether compile --eval-gate
aether compile --dry-run
aether run <model.aeg>
aether serve <model.aeg>
aether graph <model.aeg>
aether graph --reasoning
aether graph --mla
aether precision-map <model.aeg>
aether hardware
aether bench <model.aeg> --compare vllm
aether eval <model.aeg> --suite reasoning
aether hub login
aether hub push <model.aeg>
aether hub pull <model_id>
aether hub search <query>
aether ab start model_a.aeg model_b.aeg
aether safety check <model.aeg>
aether trace export --format otlp
aether mla-stats <model.aeg>
aether reasoning analyze <model.aeg>
`

---

## 22. Target Personas

| Persona | What They Want |
|---|---|
| MLOps Engineer | compile once, run everywhere |
| Startup CTO | pip install and serve in 5 minutes |
| Research Lab | AEG artifacts on NVIDIA + AMD + Apple |
| Enterprise AI Platform | Eval gates, A/B rollout, safety, observability |
| Edge Developer | Portable AEG targeting Intel NPU / Qualcomm |
| OSS Contributor | Target plugin API + Hub kernel upload |
| Fine-tuning Engineer | LoRA adapter + base model = new AEG |
| Agentic AI Developer | Agentic sessions, KV reuse, cascade routing |

---

## 23. Open-Source Roadmap — Six Phases

Phase 1 (Months 1-6): Core compiler working end-to-end. SafeTensors loader, GGUF parser, perplexity calibration, llama.cpp dispatch, eval gate.

Phase 2 (Months 7-12): GPU native. FA-2/3, vLLM backend, FP8 production, EAGLE-3 full, disaggregated scheduler real, Hub HTTP API.

Phase 3 (Months 13-18): Multi-hardware. Metal M3/M4, ROCm, FP4 Blackwell sm100, MLA native, OpenVINO NPU, Qualcomm QNN, Hub CDN.

Phase 4 (Months 19-24): Reasoning and Agentic. Pass 7 reasoning graph, EAGLE-3 speculative CoT, budget controller, agentic KV sessions, cascade router, meta-tool compiler, multi-modal VLMs.

Phase 5 (Months 25-30): Observability and Safety. OpenTelemetry production, eval CI/CD, A/B rollout, drift monitoring, safety guardrails, fleet management.

Phase 6 (Months 31-36): Ecosystem. HuggingFace AEG hosting, GitHub Actions, TypeScript/Rust/Go SDKs, VS Code plugin, LoRA compilation, Hub premium.

---

## 24. Commercial Strategy

Open Source (Apache-2.0): Compiler, AEG format, Runtime, CLI, SDKs, Hub 50GB free, NVIDIA+AMD+Apple+CPU targets.

Aether Cloud (Paid): Cloud compilation, unlimited Hub+CDN, eval CI/CD automation, A/B+drift monitoring, priority kernel slots. .10/compile-GPU-hour, /mo flat tier.

Aether Enterprise: Private Hub, compiled safety policies, fleet management, 24h support SLA, custom hardware target development. From ,000/yr.

---

## 25. Technical Risk Analysis

| Risk | Mitigation |
|---|---|
| TRT-LLM goes open source | Portability moat + Hub acceleration |
| vLLM adds compilation | AEG format + Hub lock-in + multimodal + reasoning |
| FP4 accuracy worse than expected | Eval gate mandatory, fallback to FP8 |
| MLA weight absorption correctness | Mathematical verification + unit tests per paper |
| EAGLE-3 acceptance varies | Adaptive fallback to standard decoding |
| Hub CDN cold-start | Seed with top-50 models pre-compiled |

---

## 26. Success Metrics

Phase 1: compile qwen3-8b < 6 min, first token < 2s cold start, TTFT H100 < 200ms at 1024 tokens, throughput within 10% of vLLM.
Phase 2: EAGLE-3 accept rate >75%, Hub hot-start <5s vs 45s cold, FP8 vs BF16 <0.5% PPL regression.
Phase 3: Same AEG on H100+MI300X+M4+CPU, Blackwell FP4 3.5-4x vs H100 BF16, MLA <0.2% PPL regression.
Phase 4: Reasoning budget saves >30% tokens on AIME-2024, agentic KV reuse >80% prefill reduction.

---

## 27. Glossary

AEG = Aether Execution Graph. AEG-IR = hardware-agnostic operator graph. MLA = Multi-Head Latent Attention (90%+ KV savings). NVFP4 = NVIDIA FP4 E2M1 format. MXFP4 = OCP Microscaling FP4. EAGLE-3 = multi-layer fusion speculative decoding 3-5x. JetSpec = causal parallel tree 9.64x coding. Speculative CoT = draft reasoning verified by target 21-66% savings. DeFT = Flattened Tree 3.59x attn. GTO = Group Tree Optimization. FA-3 = FlashAttention-3 840 TFLOPs BF16. FA-4 = FlashAttention-4 1613 TFLOPs Blackwell. Disaggregated P/D = separate prefill and decode pools. Chunked Prefill = Sarathi-Serve technique. RadixAttention = SGLang KV radix tree. Fine-grained MoE = 256+ experts. Auxiliary-Loss-Free = DeepSeek bias-term gating. Eval Gate = mandatory quality benchmark. Cascade Routing = complexity-based model selection. Meta-Tool = compiled frequent tool call sequence.

---

## Appendix A — Research Foundation (60+ Papers)

### A.1 Compiler and IR

PagedAttention (SOSP 2023), MLIR (2021), LLVM (2004), ClusterFusion NeurIPS 2025 (1.6-2.0x fusion speedup), Agentic MLIR 2026 (LLM-planned transforms), torch.compile (2023).

### A.2 Speculative Decoding

Speculative Decoding (Chen et al. 2023), EAGLE-2 (2024), EAGLE-3 (2025) multi-layer fusion, EAGLE-3.1 (2026) attention drift fix, JetSpec (2026) 9.64x causal parallel, OPT-Tree (2024) adaptive tree, DeFT ACL 2025 3.59x, GTO (2025), Speculative CoT 21-66% (2025), EDD ACL 2025, Pruned Candidate Trees ACL 2025.

### A.3 KV Cache and Memory

PagedAttention SOSP 2023, SGLang RadixAttention 2024, DistServe 2024 3-4x goodput, Mooncake 2024 Conductor scheduler, PrefillOnly 2025 final-layer KV, EvolKV 2025 adaptive allocation, FlexGen 2023 NVMe offload, LoopServe 2025 cross-session reuse, LMCache 2025 MoE KV, MuxWise 2026 SLO-aware scheduling.

### A.4 Quantization and Precision

GPTQ 2022, AWQ 2023, Marlin kernels 2024, AutoMixQ 2025, AMQ 2025, FlashAttention-3 FP8 2024, NVFP4 NVIDIA 2025, MXFP4 OCP 2025, FP4 KV Cache 2026.

### A.5 Attention Mechanisms

FlashAttention 2022, FlashAttention-2 2023, FlashAttention-3 Shah et al. 2024 (840 TFLOPs BF16), FlashAttention-4 2026 (1613 TFLOPs Blackwell), MLA DeepSeek-V2 2024, MHA2MLA 2025, GQA 2023, FlashDecoding 2024.

### A.6 MoE and Expert Routing

DeepSeekMoE 2024 (shared experts), DeepSeek-V3 2024 (256-expert fine-grained), Auxiliary-Loss-Free 2024 (bias-term gating), FineMoE 2025 (semantic prefetch), MoE-Infinity 2025, ISCA 2026 prefill-aware placement.

### A.7 Parallelism and Scheduling

Megatron-LM 2021 (TP), Alpa 2022 (auto-parallelism), Seesaw MLSys 2025 (25-40% dynamic resharding), Sarathi-Serve 2023 (chunked prefill), Orca 2022 (continuous batching).

### A.8 Reasoning and Agentic

Chain-of-Thought Wei et al. 2022, Tree-of-Thoughts 2023, Graph-of-Thoughts 2023, Speculative CoT 2025, RLVR 2024, AWO Agent Workflow Optimization 2025, Helium Workflow-Aware Serving 2026.

### A.9 Multi-Modal

LLaVA 2023, InternVL2 2024, Fast-VLM 2025 (dynamic token compression), Qwen3-VL 2025 (early-fusion), AttentionPack 2025, ViT-DP + LLM-TP hybrid parallelism 2026.

### A.10 Hardware

NVIDIA Hopper GH100 2022 (FA-3 WGMMA TMA), NVIDIA Blackwell B200 2024 (FP4 FA-4 8TB/s), AMD MI300X 2023 (192GB HBM3 ROCm), Apple M4 2024 (Metal 4 Neural Engine), Qualcomm AI 100 2024 (QNN Hexagon DSP).

---

# PART II — ELITE-CLASS EXTENSIONS (v3.1)

> Added July 2026 after reading 100+ research papers across long-context, pruning/sparsity, inference-time compute scaling, LoRA runtime fusion, SSM/hybrid architectures, model provenance, RAG-native compilation, and distillation pipelines.

---

## 28. Long-Context Engine (1M+ Token Native Support)

### 28.1 The Problem at Million-Token Scale

Processing 1M-token contexts requires four simultaneous strategies:

- Strategy 1: Sparse Attention (MInference — 10x prefill speedup)
- Strategy 2: Ring Attention (distribute sequence across GPUs)
- Strategy 3: KV Compression (quantized tiered KV + MLA)
- Strategy 4: Salience-Aware KV Eviction (keep only important tokens hot)

### 28.2 MInference — Dynamic Sparse Attention (Pass 8, NEW)

MInference (Microsoft Research, NeurIPS 2024 → SGLang/vLLM 2025) achieves up to 10x prefill speedup for 1M-token inputs on a single A100. Aether v3.1 compiles MInference patterns directly into the AEG artifact as Pass 8.

Three sparse attention patterns identified per-head offline at compile time:
- A-shape: Local + global token attention (local sink + beginning tokens)
- Vertical-Slash: Diagonally shifted sparse attention bands
- Block-Sparse: Fixed-size attention blocks with gaps

`python
# AEG-IR new op emitted by Pass 8
aeg.minference_attention(q, k, v,
    pattern_type="A_shape|vertical_slash|block_sparse",
    sparsity_ratio=0.9,
    head_pattern_map=@head_patterns)
# Same output quality, 10x less compute for 1M-token prefill
`

MMInference (VLMs, 2025): Grid sparse pattern for spatial/temporal locality in video tokens — 8.3x speedup for 1M-token multimodal (video+text) inputs.

Stored in AEG:
- .aeg/graph/attention_head_patterns.json — per-head MInference pattern assignments
- .aeg/kernels/{target}/minference_*.so — pre-compiled sparse attention kernels

### 28.3 Ring Attention + Context Parallelism

Context Parallelism (CP): Sequence sharded across GPUs in a ring topology. Each GPU holds 1/N of the sequence (e.g., 4-GPU CP = 250K tokens/GPU for 1M input). GPUs exchange KV blocks in ring pattern — no single GPU bottlenecks.

Striped Attention: interleaved token distribution for better load balancing. Ulysses+Ring hybrid optimal for 128+ GPU clusters.

New AEG parallelism plans:
- .aeg/parallelism/4gpu_cp.json
- .aeg/parallelism/8gpu_cp.json
- .aeg/parallelism/32gpu_cp.json (4M+ tokens)

### 28.4 Salience-Aware KV Eviction

`python
class SalienceKVEvictor:
    """Research: StreamingLLM (2023), ScissorHands (2024), SnapKV (2025)"""
    def score_salience(self, kv_block, attention_weights) -> float:
        recency_score   = self._recency_weight(kv_block.position, self.window_size)
        attention_score = attention_weights[kv_block.token_range].mean()
        anchor_score    = 1.0 if kv_block.is_anchor else 0.0
        return 0.5*attention_score + 0.3*recency_score + 0.2*anchor_score
    # Lowest-salience KV blocks evicted to CPU DRAM tier first
`

### 28.5 AEG Long-Context Profile (in manifest)

`json
"long_context_profile": {
  "max_context_tokens": 1000000,
  "rope_extension_method": "yarn",
  "sparse_attention_enabled": true,
  "sparse_head_patterns": {...},
  "ring_attention_enabled": true,
  "min_gpus_for_1m_tokens": 4,
  "kv_eviction_policy": "salience"
}
`

---

## 29. Model Pruning and Sparsity Compiler (Pass 9, NEW)

### 29.1 Why Pruning Belongs in the Compiler

Aether v3.1 integrates pruning as a compiler pass — sparsity structure compiled into AEG kernels, hardware-native 2:4 Sparse Tensor Cores targeted automatically.

Pruning methods integrated:

UNSTRUCTURED: SparseGPT (Hessian-based, no retraining), Wanda (|W| x ||X||_2 scoring, fastest), M-Wanda (multilingual), ROSE (reordering for SparseGPT)

SEMI-STRUCTURED (2:4): Exactly 2 zeros per 4 consecutive weights. Native Sparse Tensor Core support on A100/H100/B200. Up to 2x GEMM throughput.

STRUCTURED: Head pruning (LLM Surgeon), Channel pruning (IntraExpert Sparsity), Layer dropping (ShortGPT analysis — redundant layers).

### 29.2 Pruning Pipeline

`python
class PruningPass:
    """Pass 9: Sparsity analysis and mask emission into AEG-IR."""
    STRATEGIES = {
        "speed":     {"method": "wanda_24", "target": "2:4_semi_structured"},
        "quality":   {"method": "sparsegpt", "target": "unstructured_50"},
        "edge":      {"method": "structured_head_channel", "target": "50pct_smaller"},
        "blackwell": {"method": "wanda_24", "target": "2:4_plus_fp4"},
    }
    # Wanda importance: |W| * ||X||_2 per weight element
    # Masks baked into AEG-IR, sparse GEMM kernels emitted in Stage 3
`

### 29.3 Sparsity + Quantization Stacking

- Stack 1 (Speed): 2:4 Wanda pruning + FP8 = ~2.5x throughput vs dense BF16
- Stack 2 (Maximum/B200): 2:4 Wanda + FP4 = ~3.5x throughput vs dense BF16
- Stack 3 (Edge): Structured 50% + INT4 = Qwen3-72B at effective 18B memory cost

Sparsity metadata stored in .aeg/manifest.json.

---

## 30. Inference-Time Compute Scaling Engine (NEW)

### 30.1 The Third Scaling Law

In 2026, test-time compute scaling is projected to account for 75% of AI compute demand by 2030. Smaller models + more thinking tokens outperform larger models with no thinking time (proven by DeepSeek-R1, 2025).

### 30.2 Aether Inference Compute Controller

`python
class InferenceComputeController:
    """Research: Inference-Time Scaling (Google 2025), compute-optimal BoN,
       ThreadWeaver parallel reasoning (2026), InferenceTimePessimism (2026)"""

    STRATEGIES = {
        "best_of_n": {"n_samples": 8, "selection": "reward_model", "parallel": True},
        "beam_search": {"beam_width": 4, "length_penalty": 1.0},
        "mcts": {"simulations": 32, "ucb_constant": 1.4, "max_depth": 10},
        "adaptive": {
            "complexity_classifier": "qwen3-1.7b",
            "budget_map": {
                "simple":    {"strategy": "greedy",    "max_tokens": 512},
                "medium":    {"strategy": "best_of_4", "max_tokens": 2048},
                "hard":      {"strategy": "beam_4",    "max_tokens": 8192},
                "very_hard": {"strategy": "mcts",      "max_tokens": 32768},
            }
        }
    }
`

### 30.3 Process Reward Model (PRM)

`python
class ProcessRewardModel:
    """Scores intermediate reasoning steps.
    Research: Let's Verify Step by Step (2023), Math-Shepherd (2024), OmegaPRM (2025)"""
    def score(self, prompt: str, response: str) -> float:
        steps = self._parse_reasoning_steps(response)
        step_scores = [self.step_scorer(prompt, step) for step in steps]
        return min(step_scores)  # conservative: minimum step quality
`

Stored in AEG:
- .aeg/inference/compute_profiles.json — greedy/BoN/beam/MCTS configs + relative costs
- .aeg/inference/prm_head.bin — compiled process reward model head


---

## 31. LoRA Runtime Fusion Engine (NEW)

### 31.1 The Multi-Adapter Problem

Enterprises deploy one base model with dozens of domain-specific LoRA adapters (legal, medical, coding, finance). Aether v3.1 introduces LoRA compilation as a first-class compiler feature.

### 31.2 Three LoRA Compilation Modes

Mode 1 — COMPILE (static merge): aether compile base_model --lora adapter.safetensors --merge. Single .aeg with weights baked in. W_merged = W_base + (alpha/r)BA. Zero inference overhead.

Mode 2 — MULTI-SLOT (compiled multi-adapter): aether compile base_model --lora-slots 8. .aeg with 8 pre-allocated adapter slots. Swap adapters per-request with zero base model reload. Best for platforms serving multiple tasks.

Mode 3 — DELTA-COMPRESS: aether compile base_model --lora legal.safetensors medical.safetensors coding.safetensors. Uses Pico method (output-side calibration). 4-8x smaller than uncompressed LoRA weights.

### 31.3 LoRA Hot-Swap Engine (BGMV Kernel)

`python
class LoRAHotSwapEngine:
    """Research: S-LoRA (2023), Punica BGMV kernel (2024),
       Multi-LoRA vLLM (2025), Pico adapter calibration (2025)"""

    def serve(self, request: InferenceRequest) -> str:
        adapter = self.adapter_pool.get(request.adapter_id)
        # BGMV: Batched Gather Matrix-Vector — different batch items
        # can use different A,B matrices in same GPU kernel dispatch
        return self._bgmv_forward(request.prompt, self.base_weights, adapter)
`

BGMV key insight: Standard GEMM requires same A,B for all batch items. BGMV allows different A,B matrices per batch item — enables true per-request adapter switching in a single fused kernel.

### 31.4 LoRA in AEG Format

`
.aeg/adapters/
├── manifest.json                 # list of embedded adapters + routing config
├── legal_v2/
│   ├── delta_A.bin              # LoRA A matrices (Pico compressed)
│   ├── delta_B.bin              # LoRA B matrices
│   └── config.json              # rank=64, alpha=128, target_modules
├── medical_v1/
└── coding_v3/
`

---

## 32. SSM and Hybrid Architecture Support (NEW)

### 32.1 Beyond Transformers

Hybrid architectures (Jamba, Bamba, Mamba-3 2026) achieve up to 3x higher inference throughput than pure transformers for long sequences due to constant-time recurrent state vs quadratic attention.

### 32.2 New AEG-IR SSM Opcodes

`
aeg.ssm_scan(x, A, B, C, D)           # Mamba selective scan
aeg.ssm_state_update(state, x, dt)    # Mamba recurrent state update
aeg.ssm_state_snapshot(state)         # State snapshot for spec decoding rollback
aeg.rwkv_time_mix(x, state, w, u)     # RWKV WKV attention mechanism
aeg.hybrid_dispatch(x, layer_type)    # Route to attn or SSM per layer type
aeg.mimo_ssm(x, A, B, C, D, order=2) # Mamba-3 MIMO complex-valued formulation
`

### 32.3 Dual Memory Pool for Hybrid Models

`python
class HybridMemoryPool:
    """Research: SGLang Hybrid Serving (Alibaba Cloud 2026),
       State Snapshotting for Speculative Decoding on SSMs (2026)"""
    def __init__(self):
        self.kv_pool   = PagedKVCache()    # KV for transformer layers
        self.ssm_pool  = SSMStatePool()    # Recurrent state for SSM layers
        self.snapshots = StateSnapshotStore()  # for spec decoding rollback

    def snapshot(self, request_id: str) -> Snapshot:
        return self.snapshots.save(request_id,
            kv_state=self.kv_pool.get(request_id),
            ssm_state=self.ssm_pool.get(request_id))

    def rollback(self, request_id: str, snapshot_id: str) -> None:
        """Restore SSM state when speculative tokens are rejected."""
        snapshot = self.snapshots.load(snapshot_id)
        self.kv_pool.restore(request_id, snapshot.kv_state)
        self.ssm_pool.restore(request_id, snapshot.ssm_state)
`

### 32.4 Supported Hybrid Patterns

| Architecture | Attn Ratio | SSM Ratio | SSM Type |
|---|---|---|---|
| Jamba | 12.5% | 87.5% | Mamba |
| Bamba | 25% | 75% | Mamba-2 |
| Mamba-3 | 0% | 100% | MIMO complex-valued |
| Zamba2 | 10% | 90% | Shared attention + Mamba |
| RWKV-7 | 0% | 100% | Linear attention RNN |

---

## 33. Aether Distillation Pipeline (NEW)

### 33.1 Compiled Distillation

Aether v3.1 compiles both teacher and student models into AEG, then orchestrates knowledge transfer at the compiled graph level. Result: 5-30x cost reduction, 4x faster inference, 95-97% quality retention.

### 33.2 Four Distillation Modes

`python
DISTILLATION_MODES = {
    "response_based": {
        "targets": ["logits", "token_probabilities"],
        "loss": "forward_kl + cross_entropy",
        "use_when": "black-box teacher access only"
    },
    "feature_based": {
        "targets": ["hidden_states", "attention_maps"],
        "loss": "MSE_hidden + CKA_attention",
        "use_when": "white-box teacher — best quality transfer"
    },
    "reasoning_chain": {
        "targets": ["thinking_tokens", "step_quality_scores"],
        "loss": "reasoning_chain_alignment",
        "use_when": "teacher is R1/o3-style reasoning model"
    },
    "self_distillation": {
        "teacher": "same_model_larger_context",
        "loss": "ICL_self_teacher",
        "use_when": "no external teacher, anti-forgetting fine-tune (SDFT)"
    }
}
`

Research: MiniLLM (2024), DistiLLM (2024), DeepSeek-R1 Distillation (2025), Feature-Based Distillation (IEEE 2025), Self-Distillation SDFT (2026 — autonomous optimization).

---

## 34. RAG-Native Compilation (NEW)

### 34.1 Compile the Entire RAG Pipeline

Aether v3.1 treats the entire RAG pipeline as a compiled execution graph — LLM + retriever + reranker + embedding model are all compiled into a single AEG workflow.

### 34.2 AEG RAG Pipeline Graph

`
.aeg/graph/rag_pipeline.aeg-ir:

Stage 1: Query encoding
  aeg.embedding_encode(query, @embedding_model_aeg) -> query_vector

Stage 2: Parallel retrieval (all sources simultaneously)
  aeg.async_retrieve([
    aeg.vector_search(query_vector, @faiss_index, top_k=50),
    aeg.bm25_search(query_text, @bm25_index, top_k=50),
    aeg.graph_retrieve(query_entities, @kg_index, hops=2),
  ]) -> candidates[150]

Stage 3: Cross-encoder reranking
  aeg.cross_encoder_rerank(query_text, candidates, @reranker_aeg, top_k=5)

Stage 4: Context assembly + generation
  aeg.context_pack(ranked_docs, max_tokens=4096)
  aeg.generate(query_text, context, @llm_aeg) -> response
`

Compile-time optimizations: embedding + reranker compiled to same target as LLM; async_retrieve runs all sources in parallel; common RAG system prompts pinned to KV L2 cache; frequently-retrieved documents pre-cached (85% TTFT reduction for hot docs).

---

## 35. AEG Model Provenance and Watermarking (NEW)

### 35.1 Regulatory Context (EU AI Act, Aug 2026)

Every AEG compiled artifact must carry full provenance for EU AI Act compliance. Aether v3.1 bakes provenance into every .aeg at compile time.

### 35.2 AEG Provenance Manifest

`json
".aeg/provenance/manifest.json": {
  "model_hash": "sha256:a3f4b2c1...",
  "compiler_version": "aether/3.1.0",
  "source_model": {"id": "qwen3-72b", "license": "Apache-2.0"},
  "transformations": [
    {"pass": "operator_fusion", "version": "1.2"},
    {"pass": "sensitivity_quantization", "calibration": "general", "budget": 0.02},
    {"pass": "wanda_pruning", "sparsity": 0.5}
  ],
  "c2pa_binding": "c2pa://...",
  "eu_ai_act": {
    "risk_category": "limited_risk",
    "transparency_obligations_met": true
  },
  "hardware_certification": {
    "certified_targets": ["cuda_sm90", "cuda_sm100", "metal_m3"],
    "eval_gate_passed": true,
    "eval_results": {"hellaswag": 0.892, "mmlu": 0.847, "gsm8k": 0.913}
  }
}
`

### 35.3 SynthID-Style Output Watermarking

`python
class AetherOutputWatermark:
    """Research: SynthID-Text (Google DeepMind 2024), green-list token watermarking.
    EU AI Act Art. 50 — disclosure obligation for AI-generated content."""

    def apply_watermark(self, logits: Tensor, context_ids: list[int]) -> Tensor:
        context_hash = hash(tuple(context_ids[-16:]))
        greenlist = self.greenlist_fn(context_hash)  # pseudo-random per context
        logits[greenlist] += self.delta              # delta=1.0; invisible to readers
        return logits

    def detect_watermark(self, text: str) -> WatermarkResult:
        z_score = self._compute_z_score(self.tokenizer.encode(text))
        return WatermarkResult(
            detected=z_score > 4.0,    # p < 0.00003 false positive rate
            z_score=z_score)
`

### 35.4 Model IP Fingerprinting

`python
class AEGModelFingerprint:
    """Research: MetaFinger (2024), ADV-TRA (2025), ZK-proof ownership (2026)"""

    def embed(self, model_aeg: str, owner_id: str) -> str:
        """Embed ownership fingerprint at COMPILE TIME — survives pruning+quantization."""
        triggers = self._generate_trigger_set(owner_id, n=100)
        return self._embed_fingerprint(model_aeg, triggers)

    def verify(self, suspect_model: str, owner_id: str) -> FingerprintResult:
        match_rate = self._test_trigger_responses(suspect_model, self._load_triggers(owner_id))
        return FingerprintResult(is_derived=match_rate > 0.85, match_rate=match_rate)
`

---

## 36. Adaptive Learning and Zero-Downtime Hot-Reload (NEW)

### 36.1 The Tri-Layer Adaptation Framework (2026 Production Standard)

| Layer | Mechanism | Cadence | Aether Feature |
|---|---|---|---|
| Facts | RAG + dynamic vector index | Milliseconds | AEG RAG Pipeline |
| Style/Domain | LoRA adapter hot-swap | Daily/weekly | LoRA Fusion Engine |
| Reasoning | DPO/GRPO fine-tune + compile | Monthly | Distillation Pipeline + A/B rollout |

### 36.2 Zero-Downtime Model Updates

`python
class AetherHotReload:
    """Load new AEG alongside old one, zero dropped requests."""

    def hot_reload(self, new_aeg: str, traffic_pct: float = 0.0) -> str:
        new_instance = Runtime.load_background(new_aeg)  # load while old serves
        return self.ab_router.register(old=self.current, new=new_instance, split=traffic_pct)

    def auto_rollout(self, experiment_id: str,
                     step_pct: float = 0.1, step_interval_sec: int = 3600):
        """Auto-increase new traffic if quality holds; rollback on regression."""
        while self.ab_router.get_split(experiment_id) < 1.0:
            metrics = self.monitor.compare(experiment_id)
            if metrics.new_wins and not metrics.regression_detected:
                self.ab_router.increase(experiment_id, step_pct)
            elif metrics.regression_detected:
                self.rollback(experiment_id); return
            time.sleep(step_interval_sec)
`

---

## 37. CUDA Graph Capture and Persistent Kernels (NEW)

### 37.1 Eliminating CPU-GPU Overhead

`
Without CUDA Graphs:
  CPU dispatches each kernel individually: attn -> FFN -> residual
  GPU: execute -> idle -> execute -> idle
  Overhead: 50-200us CPU overhead per decode step

With Aether CUDA Graph Capture (stored in .aeg):
  CPU: submit entire decode step as single CUDA Graph replay
  GPU: execute entire decode step without interruption
  Overhead: <5us per decode step
  Result: 15-30% throughput improvement at small batch sizes (research: vLLM 2026)
`

### 37.2 AEG Pre-Captured Decode Graphs

`
.aeg/cuda_graphs/
├── sm90_decode_b1.graph    # Batch=1 decode CUDA graph
├── sm90_decode_b2.graph    # Batch=2
├── sm90_decode_b4.graph
├── sm90_decode_b8.graph
├── sm90_decode_b16.graph
├── sm90_decode_b32.graph
└── sm90_prefill_chunked.graph
`

Piecewise CUDA Graph: captures parts excluding dynamic-shape ops; selects correct graph at runtime by rounding batch size up to next captured size. Research: vLLM CUDA Graphs dispatcher (2026).

---

## 38. Fleet Management and Multi-Node Orchestration (NEW)

### 38.1 Enterprise Fleet

`python
class AetherFleetManager:
    """Research: Helium workflow-aware serving (2026), Kubernetes AI operators."""

    def deploy(self, model_aeg: str, fleet: FleetConfig) -> DeploymentHandle:
        """Auto-detect hardware per node, deliver optimal kernel from Hub CDN."""
        node_assignments = self._assign_targets(model_aeg, fleet.nodes)
        deployments = []
        for node, target in node_assignments.items():
            kernel_url = self.hub.get_kernel_url(model_aeg, target)
            deployments.append(self._deploy_node(node, model_aeg, kernel_url))
        return DeploymentHandle(deployments, LoadBalancer(deployments))
`

### 38.2 Multi-Region Topology Example

`
Region 1 (US-East):  2x B200,    cuda_sm100 kernel from Hub CDN
Region 2 (EU-West):  4x H100,    cuda_sm90  kernel from Hub CDN
Region 3 (AP-South): 2x MI300X,  rocm_cdna3 kernel from Hub CDN
Edge nodes:          Apple M4,   metal_m3   kernel from Hub CDN

Same qwen3-72b.aeg artifact across ALL nodes.
Hub CDN delivers the right compiled kernel for each hardware type.
Zero recompilation on production nodes — ever.
`

---

## 39. Complete AEG Format v3.1

`
model.aeg/
├── FORMAT_VERSION                      "AEG/1.1"
├── graph/
│   ├── computation_graph.aeg-ir        Core transformer graph
│   ├── reasoning_graph.aeg-ir          [3.0] Compiled CoT graph
│   ├── rag_pipeline.aeg-ir             [3.1 NEW] RAG workflow graph
│   └── attention_head_patterns.json    [3.1 NEW] MInference per-head patterns
├── weights/
│   ├── precision_map.json              Per-layer precision
│   ├── model.aeg-quant                 Mixed-precision weights
│   ├── sparsity_masks.bin              [3.1 NEW] Wanda/SparseGPT masks
│   └── mla_compressed/                [3.0] MLA latent vectors
├── adapters/                           [3.1 NEW] LoRA multi-slot
│   ├── manifest.json
│   └── {name}/delta_A.bin, delta_B.bin, config.json
├── kernels/
│   ├── cuda_sm70/ ... cuda_sm120/
│   ├── metal_m1/ ... metal_m3/
│   ├── rocm_rdna3/, rocm_cdna3/
│   ├── openvino_npu/, qualcomm_qnn/
│   └── cpu_avx512/, cpu_neon/
├── cuda_graphs/                        [3.1 NEW] Pre-captured decode graphs
│   └── {target}_decode_b{N}.graph
├── parallelism/
│   ├── 1gpu.json ... 8gpu.json
│   ├── prefill_decode_split.json       [3.0]
│   └── 32gpu_cp.json                   [3.1 NEW] Context parallelism 1M+ tokens
├── inference/
│   ├── compute_profiles.json           [3.1 NEW] BoN/beam/MCTS profiles
│   └── prm_head.bin                    [3.1 NEW] Process reward model head
├── safety/                             [3.0] Guardrail configs
├── provenance/                         [3.1 NEW] Full provenance
│   ├── manifest.json                   C2PA, EU AI Act, transformations
│   └── fingerprint.json                IP fingerprint triggers
├── watermark/                          [3.1 NEW] SynthID-style config
│   └── config.json
└── manifest.json                       Top-level with all hashes
`

---

## 40. Complete Nine-Pass Optimizer (v3.1)

| Pass | Name | Research Basis | Key Speedup |
|---|---|---|---|
| 1 | Operator Fusion | ClusterFusion NeurIPS 2025 | 1.6-2.0x, 40% fewer DRAM round-trips |
| 2 | Sensitivity Analysis | AutoMixQ 2025, AMQ 2025 | Foundation for precision assignment |
| 3 | Precision Assignment | NVFP4 Blackwell, MXFP4 OCP | Up to 4x on B200 vs H100 BF16 |
| 4 | KV Cache Structuring | Mooncake 2024, DistServe 2024 | 90%+ KV reduction (MLA), 85% TTFT reduction (RAG) |
| 5 | MoE Expert Routing | DeepSeek-V3 2024, FineMoE 2025 | 2.5x expert speedup |
| 6 | Parallelism Discovery | Seesaw MLSys 2025, Alpa 2022 | 25-40% gain from dynamic resharding |
| 7 | Reasoning Graph | Speculative CoT 2025, GoT 2023 | 21-66% latency reduction |
| 8 | Sparse Attention | MInference NeurIPS 2024 | 10x prefill speedup at 1M tokens |
| 9 | Pruning/Sparsity | Wanda 2023, SparseGPT 2022 | 2x GEMM via 2:4 Sparse TC |

---

## 41. Additional Research Foundation (New Papers, 100+ Total)

### A.11 Long-Context and Sparse Attention

MInference (NeurIPS 2024) — 10x prefill at 1M tokens, 3 sparse patterns, plug-and-play. MMInference (ICML 2025) — 8.3x for VLM video. Ring Attention (2023) — sequence parallelism ring topology. Striped Attention (2023) — load-balanced ring. DeepSpeed-Ulysses (2023) — TP+SP hybrid. SnapKV (2024) — important-token KV selection. ScissorHands (2024) — KV eviction. StreamingLLM (2023) — anchor token KV retention. YaRN (2023) — RoPE extension to 128K. LongRoPE (2024) — 2M context RoPE scaling. LLaMA-3-1M (2024) — 1M native context.

### A.12 Pruning and Sparsity

SparseGPT (2022) — Hessian one-shot pruning. Wanda (2023) — weight x activation importance, fastest. ROSE (2025) — reordering for SparseGPT. M-Wanda (2025) — multilingual extension. Elsa (2025) — ADMM extreme 90%+ sparsity. LLM Surgeon (2024) — structured head pruning. ShortGPT (2024) — layer redundancy. NVIDIA 2:4 Sparsity (Ampere 2020) — Sparse TC native. Accelerating Unstructured Sparse Inference (2026).

### A.13 Inference-Time Compute Scaling

Let's Verify Step by Step (2023) — PRM foundation. Math-Shepherd (2024) — automated PRM. OmegaPRM (2025) — MCTS-based PRM data collection. DeepSeek-R1 (2025) — RLVR + extended CoT proof. Inference-Time Scaling (Google 2025) — compute-optimal BoN, 4x over naive baseline. InferenceTimePessimism (2026) — monotonic scaling guarantee. ThreadWeaver (2026) — hardware-parallel reasoning. Recurrent Depth Models (2025) — latent space thinking.

### A.14 LoRA and Adapter Serving

LoRA (2022). QLoRA (2023) — 4-bit quantized LoRA. S-LoRA (2023) — serving thousands of adapters. Punica (2024) — BGMV kernel for multi-adapter batching. Pico (2025) — output-side calibration compression. vLLM Multi-LoRA (2025) — per-request adapter routing.

### A.15 SSM and Hybrid Architectures

Mamba (2023) — selective SSM. Mamba-2 (2024) — SSD structured SSM. Mamba-3 (March 2026) — MIMO complex-valued states. Jamba (2024) — Transformer+Mamba hybrid. Bamba (2024) — IBM 9B hybrid. Zamba2 (2025) — shared attention hybrid. RWKV-7 (2025) — linear attention RNN. SGLang Hybrid Serving (Alibaba 2026) — dual memory pool. State Snapshotting for Speculative Decoding on SSMs (2026).

### A.16 Distillation and Compression

Knowledge Distillation (Hinton 2015). MiniLLM (2024) — RL LLM distillation. DistiLLM (2024) — diverse distillation. DeepSeek-R1 Distillation (2025) — reasoning chain transfer. Feature-Based LLM Distillation (IEEE 2025) — hidden state matching. SDFT (2026) — self-distillation via ICL teacher. 5-30x cost reduction, 95-97% quality retention benchmarks.

### A.17 Model Provenance and Safety

SynthID-Text (Google DeepMind 2024) — green-list statistical watermarking. C2PA (2024) — Coalition for Content Provenance and Authenticity standard. EU AI Act (binding Aug 2026) — Art. 50 AI disclosure obligation. MetaFinger (2024) — model fingerprinting. ADV-TRA (2025) — adversarial trajectory fingerprinting. Zero-Knowledge Proof Model Ownership (2026) — privacy-preserving IP verification.

### A.18 Systems and Infrastructure

CUDA Graphs for LLM (2024-2026). vLLM CUDA Graphs Dispatcher (2026) — piecewise capture, dynamic shape dispatch. KTransformers (2025) — heterogeneous CPU/GPU inference. PPipe (USENIX 2025) — pipeline parallelism on mixed GPU clusters. Kubernetes AI Operators (2025) — production fleet management. LanceDB (2024) — embedded vector DB for edge RAG. Dynamic embedding streams (2026) — real-time RAG index updates.

