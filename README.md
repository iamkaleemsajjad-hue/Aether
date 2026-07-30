# Aether Runtime

> **Compile once. Run on any hardware, forever.**

Aether is the **compiler for AI models** — not another inference wrapper. It takes any model format (SafeTensors, GGUF, ONNX, MLX, PyTorch) and compiles it into an **Aether Execution Graph (AEG)**, a portable, hardware-agnostic, optimized artifact. The AEG file contains everything needed to run the model on NVIDIA, AMD, Apple Silicon, Intel, or CPU — today and in the future.

Aether is inspired by LLVM: just as C/C++ compiles to LLVM IR and then to any ISA, any AI model compiles to AEG-IR and then to any inference backend. The runtime selects the best available backend for your hardware (vLLM, llama.cpp, TensorRT-LLM, MLX, ONNX Runtime, or PyTorch) and applies Aether's own scheduling, caching, speculative decoding, and precision management on top.

---

## Table of Contents

1. [Why Aether?](#why-aether)
2. [What is an AEG?](#what-is-an-aeg)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Python SDK](#python-sdk)
6. [CLI](#cli)
7. [REST API](#rest-api)
8. [OpenAI Compatibility](#openai-compatibility)
9. [Architecture](#architecture)
10. [Compiler Pipeline](#compiler-pipeline)
11. [Optimizer Passes](#optimizer-passes)
12. [Runtime Intelligence](#runtime-intelligence)
13. [Supported Backends](#supported-backends)
14. [Supported Hardware](#supported-hardware)
15. [AEG Format](#aeg-format)
16. [Contributing](#contributing)
17. [License](#license)
18. [Research Foundation](#research-foundation)
19. [Roadmap](#roadmap)
20. [Commercial](#commercial)
21. [Support](#support)

---

## Why Aether?

Every existing AI inference tool is a wrapper around a backend that was chosen at build time:

- **Ollama** wraps llama.cpp with a Docker-like UX.
- **vLLM** wraps PyTorch with PagedAttention scheduling.
- **SGLang** wraps vLLM with structured generation.
- **TensorRT-LLM** wraps CUDA kernels with a Python interface.
- **MLX** is Apple-only.
- **ONNX Runtime** is great for cross-platform but not LLM-optimized.

Wrappers inherit the constraints of their substrate. They are statically bound to hardware, cannot see the whole model, and cannot apply optimizations that cross backend boundaries. Aether does something different: it **owns the compiled model format**.

### The problem Aether solves

A model today exists as raw weights (`safetensors`, `GGUF`, `ONNX`, `MLX`, `pytorch.bin`). To deploy it you must:

1. Install the right CUDA / ROCm / Metal / OpenVINO stack.
2. Install a backend that supports that stack and that model.
3. Configure tensor parallelism, quantization, and KV cache by hand.
4. Repeat from scratch when you change hardware or cloud provider.

Aether replaces this with a single compile step:

```bash
pip install aether-runtime
aether compile Qwen/Qwen3-8B
aether run Qwen/Qwen3-8B
```

The `.aeg` file is now a portable artifact. Move it to a different machine and it runs optimally there too — no reinstall, no reconfigure, no recompile (if the Aether Hub has the kernel cache).

---

## What is an AEG?

The **Aether Execution Graph** is Aether's central invention. A `.aeg` file contains:

- `graph/computation_graph.aeg-ir` — a hardware-agnostic operator graph (AEG-IR) that preserves high-level transformer semantics.
- `weights/quantized/` — mixed-precision weights with a per-layer `precision_map.json`.
- `kernels/` — pre-compiled or backend-selected kernels for each target profile.
- `parallelism/` — pre-computed 1/2/4/8 GPU sharding plans.
- `manifest.json` — top-level metadata, hashes, and compilation provenance.

AEG is versioned and stable. An AEG/1.x file compiled today will run on all future Aether versions.

---

## Quick Start

```bash
# Install Aether
pip install aether-runtime

# Compile a small model from HuggingFace
aether compile Qwen/Qwen3-0.6B

# Run it interactively
aether run Qwen/Qwen3-0.6B

# Or start a server
aether serve Qwen/Qwen3-0.6B --port 11434
```

### Python SDK

```python
from aether import Runtime

rt = Runtime()
response = rt.generate("Qwen/Qwen3-0.6B", "Explain quantum computing in one sentence.")
print(response.text)
print(f"TPS: {response.metrics.throughput_tps}")
print(f"TTFT: {response.metrics.ttft_ms}ms")
```

---

## Installation

### Base install (CPU, PyTorch fallback)

```bash
pip install aether-runtime
```

### With your preferred backend

```bash
# NVIDIA / high-throughput serving
pip install aether-runtime[vllm]

# Apple Silicon
pip install aether-runtime[mlx]

# ONNX Runtime
pip install aether-runtime[onnxruntime]

# All backends (development)
pip install aether-runtime[dev]
```

### Optional dependencies

- `vllm` — NVIDIA serving backend.
- `llamacpp` — llama.cpp backend for CPU/GGUF.
- `trtllm` — TensorRT-LLM backend.
- `mlx` — Apple Silicon backend.
- `onnxruntime` — ONNX Runtime backend.
- `triton` — Triton kernel templates (Linux).
- `dev` — lint, test, docs, benchmark tooling.

---

## Python SDK

### Runtime

```python
from aether import Runtime, RuntimeConfig

config = RuntimeConfig(
    optimize_for="latency",
    speculative_decoding=True,
    prefill_chunk_size=2048,
    dynamic_precision=True,
)
rt = Runtime(config)

# Text generation
response = rt.generate(
    model="Qwen/Qwen3-8B",
    prompt="Write a haiku about compilers.",
    max_tokens=64,
    temperature=0.7,
)
print(response.text)

# Streaming
for chunk in rt.generate("Qwen/Qwen3-8B", "Count to 10", stream=True):
    print(chunk.delta, end="", flush=True)

# Chat
response = rt.chat(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is an AEG?"},
    ],
)
print(response.text)

# Embeddings
vectors = rt.embed(
    model="nomic-ai/nomic-embed-text-v1.5",
    input=["Hello world", "Machine learning"],
)

# Reranking
ranked = rt.rerank(
    model="BAAI/bge-reranker-v2-m3",
    query="What is CUDA?",
    documents=["CUDA is a parallel computing platform.", "Python is a programming language."],
)

# Transcription
transcript = rt.transcribe(
    model="openai/whisper-large-v3",
    audio="path/to/audio.mp3",
    language="en",
)
```

### Compiler

```python
from aether import Compiler, CompilerConfig

config = CompilerConfig(
    quality_budget=0.02,
    calibration_dataset="wikitext-2",
    targets=["auto"],
)
compiler = Compiler(config)

# Dry-run: inspect the plan before compiling
plan = compiler.plan("Qwen/Qwen3-8B")
print(plan.fusion_opportunities)
print(plan.estimated_memory_gb)

# Compile
aeg = compiler.compile("Qwen/Qwen3-8B")
print(aeg.graph_summary())
print(aeg.precision_map())
print(aeg.quality_report())
print(aeg.sharding_plans())

# Save / distribute
aeg.save("./qwen3-8b.aeg")
aeg.upload(hub="hub.aether.dev")
```

---

## CLI

```bash
# Compilation
aether compile Qwen/Qwen3-8B
aether compile Qwen/Qwen3-8B --quality-budget 0.01
aether compile Qwen/Qwen3-8B --target cuda_sm90
aether compile Qwen/Qwen3-8B --upload

# Model management
aether pull Qwen/Qwen3-0.6B
aether list
aether info Qwen/Qwen3-8B
aether graph Qwen/Qwen3-8B
aether rm Qwen/Qwen3-0.6B

# Serving
aether serve Qwen/Qwen3-8B --port 11434
aether status
aether stop

# Running
aether run Qwen/Qwen3-0.6B
aether run Qwen/Qwen3-0.6B --stream

# Benchmarking
aether bench Qwen/Qwen3-8B
aether bench Qwen/Qwen3-8B --compare vllm
aether bench --all

# Hardware and diagnostics
aether hw
aether kernels
aether logs
```

---

## REST API

```bash
aether serve Qwen/Qwen3-0.6B --port 11434
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/generate` | Text completion |
| POST | `/v1/chat` | Chat completion (OpenAI-compatible) |
| POST | `/v1/embeddings` | Embedding generation |
| POST | `/v1/rerank` | Document reranking |
| POST | `/v1/transcribe` | Audio transcription |
| POST | `/v1/compile` | Compile a model (async job) |
| GET | `/v1/compile/{job_id}` | Compilation job status |
| GET | `/v1/models` | List compiled models |
| POST | `/v1/models/pull` | Download and compile model |
| GET | `/v1/models/{name}` | Model info and metadata |
| DELETE | `/v1/models/{name}` | Remove compiled model |
| GET | `/v1/models/{name}/graph` | Inspect AEG-IR |
| GET | `/v1/hardware` | Hardware fingerprint |
| GET | `/v1/kernels` | Active kernel targets |
| GET | `/v1/metrics` | Prometheus metrics |
| GET | `/v1/health` | Health check |

---

## OpenAI Compatibility

Point any OpenAI SDK v1+ client at Aether:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="aether")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Works with LangChain, LlamaIndex, CrewAI, AutoGen, OpenHands, and any OpenAI-compatible tool.

---

## Architecture

Aether is a five-stage compiler plus a backend-orchestrated runtime.

```
Any Model Format (safetensors / GGUF / ONNX / MLX / .pt)
                    │
                    ▼
Stage 1: Model Ingestion & Graph Extraction → AEG-IR
                    │
                    ▼
Stage 2: Aether Optimizer (6 graph-level passes) → optimized AEG-IR
                    │
                    ▼
Stage 3: Hardware Targeting & Backend Selection → AEG artifact
                    │
                    ▼
Stage 4: Self-Optimizing Runtime → tokens / embeddings
                    │
                    ▼
Stage 5: Developer Interface (Python SDK / REST / CLI / OpenAI-compat)
```

### Backend plugin model

Aether does not write custom kernels for every accelerator. Instead, it integrates best-in-class backends behind a stable `Backend` interface:

- **vLLM** — NVIDIA high-throughput serving.
- **llama.cpp** — cross-platform CPU/GGUF.
- **TensorRT-LLM** — NVIDIA compiled engines.
- **MLX** — Apple Silicon native.
- **ONNX Runtime** — Intel / cross-platform execution providers.
- **PyTorch** — universal fallback and graph tracing.

Aether's value is in choosing the right backend, scheduling requests, managing the KV cache, applying speculative decoding, and compiling the model into a portable AEG. New backends can be added as plugins without changing the public API.

---

## Compiler Pipeline

### Stage 1: Model Ingestion

Aether supports:

- SafeTensors (direct loading + config.json parsing)
- GGUF (header parsing + dequantization for tracing)
- ONNX (protobuf graph → AEG-IR lowering)
- MLX (module tracing → AEG-IR)
- PyTorch `.pt` / `.bin` (`torch.export` graph capture → AEG-IR)

Architecture detection inspects the graph structure, not the model name, so it works with custom models and future variants.

### Stage 2: Nine Optimizer Passes

1. **Operator Fusion** — fuse `RMSNorm → QKV → RoPE → GQA` into megakernels.
2. **Sensitivity Analysis** — compute `d(perplexity)/d(precision)` per layer.
3. **Precision Assignment** — mixed-precision quantization based on sensitivity.
4. **KV Cache Structuring** — paged blocks, radix-tree prefix hints, tiering.
5. **MoE Expert Routing** — hot/warm/cold expert tiering, threshold-based routing.
6. **Automatic Parallelism Discovery** — search tensor/pipeline/expert/context parallelism.

### Stage 3: Hardware Targeting

Aether selects the best backend and precision plan for each target profile:

- `cuda_sm70` — V100
- `cuda_sm80` — A100
- `cuda_sm89` — RTX 4090
- `cuda_sm90` — H100
- `cuda_sm100` — B200
- `metal_m1` — Apple M1/M2
- `metal_m3` — Apple M3/M4/M5
- `rocm_rdna3` — AMD RX 7000
- `rocm_cdna3` — AMD MI300X
- `openvino_npu` — Intel Arc NPU
- `cpu_avx512` — x86 AVX-512
- `cpu_neon` — ARM NEON

### Stage 4: Runtime

The runtime loads the AEG, fingerprints the hardware, picks the backend, and runs a disaggregated prefill/decode scheduler with tree-speculative decoding and a global tiered KV cache.

### Stage 5: Developer Interface

Python SDK, REST API, OpenAI-compatible endpoints, and the `aether` CLI.

---

## Optimizer Passes

### Pass 1: Operator Fusion

Before fusion:
```
rmsnorm → q_proj → k_proj → v_proj → rope_q → rope_k
```
After fusion:
```
fused_qkv_rope_norm(x, wq, wk, wv, pos)
```

### Pass 2: Sensitivity Analysis

For each layer, compute the perplexity change when quantized. This is the mathematical basis for mixed precision.

```python
sensitivity[L] = (ppl_quantized - ppl_baseline) / bits_saved(L)
```

### Pass 3: Precision Assignment

| Sensitivity | Typical Layers | Precision |
|-------------|----------------|-----------|
| > 0.9       | Embeddings, LM head | BF16 |
| 0.7–0.9     | Q/K projections | FP8 or Q6_K |
| 0.4–0.7     | V/O projections | Q4_K_M |
| < 0.4       | FFN deep layers | Q3_K / IQ3_XS |

### Pass 4: KV Cache Structuring

- Paged blocks aligned to hardware page size.
- Radix-tree prefix hints for shared system prompts.
- Tiered storage: GPU HBM → CPU DRAM → NVMe SSD → Aether Hub CDN.

### Pass 5: MoE Expert Routing

- Activation profiling on a calibration set.
- Hot experts (>5% activation) → GPU HBM.
- Warm experts (0.1–5%) → CPU DRAM + prefetch.
- Cold experts (<0.1%) → NVMe lazy load.
- Threshold-based routing replaces rigid top-K.
- Intra-expert sparsity kernels skip dead channels.

### Pass 6: Automatic Parallelism Discovery

Searches the space of tensor/pipeline/expert/context parallelism and produces separate prefill and decode plans, stored in the AEG.

---

## Runtime Intelligence

### Disaggregated Prefill/Decode

Separates compute-bound prefill from memory-bandwidth-bound decode. Long prefills are chunked to maintain TTFT SLOs. KV state is transferred via shared memory or RDMA.

### Tree-Speculative Decoding

A draft model proposes a branching tree of candidate tokens. The target model verifies the entire tree in one forward pass using tree-masked attention. Target: 3–6x throughput on latency-sensitive workloads.

### Global KV Cache Manager

| Tier | Storage | Use |
|------|---------|-----|
| L1 | GPU HBM | Active requests |
| L2 | CPU DRAM | Prefix cache, recently evicted |
| L3 | NVMe SSD | Long system prompts, RAG KV |
| L4 | Aether Hub | Globally shared system prompts |

### Dynamic Precision Adjustment

Under memory pressure, the runtime downgrades the lowest-sensitivity layers to a lower precision in place, then restores them when pressure eases.

---

## Supported Backends

| Backend | When Used | Notes |
|---------|-----------|-------|
| vLLM | NVIDIA high-throughput serving | PagedAttention, continuous batching |
| llama.cpp | CPU/GGUF | Cross-platform, quantized models |
| TensorRT-LLM | NVIDIA production | Compiled engines, FP8 |
| MLX | Apple Silicon | Unified memory, native performance |
| ONNX Runtime | Intel / cross-platform | OpenVINO EP, NPU support |
| PyTorch | Fallback / tracing | Universal, always available |

---

## Supported Hardware

| Vendor | Hardware | Target ID |
|--------|----------|-----------|
| NVIDIA | V100, A100, RTX 4090, H100, B200 | cuda_sm70–sm100 |
| Apple | M1/M2/M3/M4/M5 | metal_m1, metal_m3 |
| AMD | RX 7000, MI300X | rocm_rdna3, rocm_cdna3 |
| Intel | Arc NPU, x86 | openvino_npu, cpu_avx512 |
| ARM | Qualcomm, Apple CPU | cpu_neon |

---

## AEG Format

The AEG format is documented in `docs/aeg-format.md`. It is versioned and stable:

- **AEG/1.x** — readable forever by all future Aether versions.
- **AEG/2.x** — when introduced, backward compatibility maintained for 3 years.

The format is content-addressed and includes:

- Computation graph (AEG-IR)
- Quantized weights and precision map
- Backend/kernel plans per target
- Parallelism plans for 1/2/4/8 GPUs
- Manifest with hashes and provenance

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding standards, and the PR process. Key areas:

- New ingestion loaders (model formats)
- New backend plugins
- Compiler passes
- Optimizations and benchmarks
- Documentation and examples

---

## License

Aether Runtime is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

---

## Research Foundation

Aether's design is grounded in recent research. See [research/research_foundation.md](research/research_foundation.md) for the full paper mapping.

Key works referenced include:

- MLIR, IREE, StableHLO for compiler/IR design.
- PagedAttention, SGLang RadixAttention, DistServe, Mooncake for KV cache and serving.
- GPTQ, AWQ, AutoMixQ, AMQ for quantization.
- SpecInfer, OPT-Tree, DeFT, JetSpec for speculative decoding.
- MoE-Infinity, CommitMoE, FinDEP, DynaMoE for MoE optimization.
- Alpa, Megatron-LM, Seesaw, Ring Attention for parallelism.

---

## Roadmap

Aether is developed in five phases:

1. **Phase 1 — Compiler Foundation** (months 1–4): AEG format, ingestion, optimizer Pass 1, basic runtime, SDK, CLI, Hub MVP.
2. **Phase 2 — Optimizer Depth** (months 5–9): Passes 2–4, mixed precision, KV cache, speculative decoding, disaggregated scheduler.
3. **Phase 3 — Parallelism and Scale** (months 10–16): Pass 6, automatic parallelism, MoE compiler, multi-node disaggregation.
4. **Phase 4 — Ecosystem** (months 17–24): MLX ingestion, OpenVINO target, model registry, more SDK bindings, WASM experimental.
5. **Phase 5 — Compiler as Platform** (month 25+): Hardware vendor SDK, custom targets, community compiler passes, multimodal compilation.

See the full roadmap in `docs/roadmap.md`.

---

## Commercial

Aether follows an open-core model:

- **Open source**: compiler, AEG format, runtime, Hub, all features.
- **Aether Cloud**: managed compilation, private AEG registry, fleet management, enterprise SSO, RBAC, audit logs.

Contact `enterprise@aether.dev` for commercial inquiries.

---

## Support

- GitHub Issues: [github.com/aether-dev/aether-runtime/issues](https://github.com/aether-dev/aether-runtime/issues)
- GitHub Discussions: [github.com/aether-dev/aether-runtime/discussions](https://github.com/aether-dev/aether-runtime/discussions)
- Documentation: [docs.aether.dev](https://docs.aether.dev)
- Email: `dev@aether.dev`

---

<p align="center">
  <strong>Aether Runtime — the compiler for AI models.</strong>
</p>


### PRD v3.1 Runtime Layers

Aether now includes functional reference implementations and artifact contracts for the v3.1 platform layer:

- Agentic workflow optimizer with meta-tool mining, context-cache policy, and cascade routing.
- EAGLE-3 planner with multi-layer fusion, flattened tree metadata, and drift-correction flags.
- MLA native planner with latent-KV compression ratios and target-specific kernel selection.
- Observability contracts for eval gates, drift monitoring, OpenTelemetry-style metrics, and A/B rollout.
- Fleet management, hot reload routing, CUDA Graph capture manifests, multimodal graph planning, and distillation manifests.

