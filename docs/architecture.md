# Aether Architecture

Aether is a five-stage compiler plus a backend-orchestrated runtime. The design is inspired by LLVM: just as C/C++ compiles to LLVM IR and then to any ISA, any AI model compiles to AEG-IR and then to any inference backend.

## Five Compiler Stages

```
Any Model Format (safetensors / GGUF / ONNX / MLX / .pt / video / VLM / MoE / MLA)
                    │
                    ▼
Stage 1: Model Ingestion & Graph Extraction → AEG-IR
         (SafeTensors, GGUF, ONNX, MLX, PyTorch, VLM, Video, MLA, MoE, SSM)
                    │
                    ▼
Stage 2: Aether Optimizer (22 graph-level passes) → optimized AEG-IR
                    │
                    ▼
Stage 3: Hardware Targeting & Backend Selection → AEG artifact
                    │
                    ▼
Stage 4: Self-Optimizing Runtime (R1–R12 layers) → tokens / embeddings
                    │
                    ▼
Stage 5: Developer Interface (Python SDK / REST / CLI / gRPC / OpenAI-compat)
```

## Stage 1: Model Ingestion

Supported formats and specialized loaders:

| Format | Loader | Notes |
|--------|--------|-------|
| SafeTensors | `safetensors_loader.py` | HuggingFace standard, multi-shard |
| GGUF | `gguf_loader.py` | llama.cpp ecosystem, embedded tokenizer |
| ONNX | `onnx_loader.py` | Cross-framework, opset 11–17 |
| MLX | `mlx_loader.py` | Apple ecosystem |
| PyTorch `.pt`/`.bin` | `pytorch_loader.py` | state dict + TorchScript JIT |
| VLM/Multimodal | `vlm_loader.py` | LLaVA, Qwen-VL, InternVL2, PaliGemma, Phi-3V, Pixtral |
| Video Models | `video_loader.py` | Video-LLaMA2, VideoChat2, LLaVA-Video, InternVideo2 |
| MLA (DeepSeek) | `mla_loader.py` | DeepSeek-V2/V3/R1, KV compression metadata |
| MoE | `moe_loader.py` | Mixtral, Qwen MoE, Jamba, DBRX, OLMoE, expert tier classification |
| SSM/Mamba | `ssm_loader.py` | Mamba-1/2, Jamba, RWKV, RetNet, hybrid architectures |

Architecture detection inspects `config.json` model_type, tensor key patterns, and structural features — making Aether robust to custom models, fine-tuned variants, and future architectures.

## Stage 2: The Aether Optimizer (22 Passes)

### v3.1 Core Passes (1–9)

| Pass | Name | Status |
|---|---|---|
| 1 | Operator Fusion | Fuses RMSNorm+QKV+RoPE+Attention into megakernels |
| 2 | Sensitivity Analysis | Computes `d(perplexity)/d(precision)` per layer |
| 3 | Precision Assignment | Mixed-precision map from sensitivity + quality budget |
| 4 | KV Cache Structuring | Paged blocks, radix-tree hints, multi-tier offload |
| 5 | MoE Expert Routing | Hot/warm/cold expert tiering, threshold routing |
| 6 | Parallelism Discovery | Tensor/pipeline/expert/context parallel search |
| 7 | Reasoning Graph Compilation | Budget-controlled CoT/tool-call graph |
| 8 | Sparse Attention (MInference) | Head-level pattern planning for long-context |
| 9 | Pruning & Sparsity | Magnitude/gradient mask generation |

### v4.0 Advanced Passes (10–17)

| Pass | Name | Status |
|---|---|---|
| 10 | Native MTP Head Compilation | Real 2-D speculation blobs, greedy verification |
| 11 | Grammar Constraint Compiler | Tokenizer-aware FSA, vocabulary-width metadata |
| 12 | Model Merging | SLERP/task-vector merge with integrity hashes |
| 13 | TTT Fast-Weight Injection | Slot tensors, R5 prompt-driven adaptation |
| 14 | Semantic KV Compression | Semantic-boundary K/V row compression |
| 15 | Cross-Layer KV Sharing | Exact-alias K/V sharing between layers |
| 16 | Green Energy Compilation | Carbon/DVFS metadata, R7 energy recording |
| 17 | TEE Enclave Emission | Requires real TEE backend (fail-closed otherwise) |

### v5.0 Frontier Passes (18–22)

| Pass | Name | Status |
|---|---|---|
| 18 | Diffusion Drafter Compilation | MDLM drafter — requires trained weights |
| 19 | Sub-2-Bit/Ternary Quantization | BitNet two-bit packed CPU AEG, per-block scales |
| 20 | Video Token Compression | Frame KV manager, temporal compression plan |
| 21 | Advanced PEFT Compilation | Shape-bearing adapter blobs, CPU BGMV apply |
| 22 | RLVR Verifier Head Injection | SymPy/subprocess verifier, zero reward for unverified |

## Stage 3: Hardware Targeting

Aether maintains a registry of 28 hardware target profiles. Each profile declares preferred backends:

| Target Class | Profiles | Preferred Backends |
|---|---|---|
| NVIDIA (Ampere/Hopper/Blackwell) | sm70–sm130 | vLLM, TensorRT-LLM, PyTorch CUDA |
| AMD (RDNA3/CDNA3/CDNA5) | rocm_rdna3, cdna3, cdna5 | ROCm/HIP, PyTorch |
| Apple Silicon (M1–M3) | metal_m1, metal_m3 | MLX, PyTorch MPS |
| CPU (AVX512/NEON) | cpu_avx512, cpu_neon | Native DLL, ONNX Runtime, llama.cpp |
| Qualcomm QNN | qualcomm_qnn | QNN SDK |
| Intel NPU | openvino_npu | ONNX Runtime, OpenVINO |
| RISC-V | riscv_* | ONNX Runtime, ONNX MLIR |
| FPGA | fpga_xilinx_vu9p | ONNX Runtime |

CPU compilation embeds a real native shared library (`.dll`/`.so`) in `generated_kernels/` using CFFI and NumPy FFI. GPU targets emit backend invocation plans rather than raw PTX/HSACO to leverage existing high-performance inference engines.

## Stage 4: Runtime Layers (R1–R12)

The self-optimizing runtime loads AEG artifacts and applies 12 execution enhancement layers:

### v3.1 Runtime Enhancements

| Layer | Name | Description |
|---|---|---|
| — | KV Manager | Paged allocation, radix-tree prefix sharing, GPU→CPU→NVMe tiering |
| — | EAGLE-3 | Speculative decoding planner (draft model required) |
| — | Disaggregated Prefill | Separate prefill/decode scheduling configuration |
| — | Dynamic Precision | Runtime precision switching under memory pressure |

### v4.0 Runtime Layers (R1–R8)

| Layer | Name | Description |
|---|---|---|
| R1 | P-EAGLE/Saguaro | MTP blob loading, projection, greedy target verification, draft counters |
| R2 | Multi-Agent KV | Session-owned cache, exact prefix sharing across agents |
| R3 | Grammar FSM | Tokenizer-aware constrained decoding, vocabulary-fingerprinted FSAs |
| R4 | SLO Scheduler | Priority/deadline-aware admission, per-tier routing |
| R5 | TTT Engine | Prompt-driven fast-weight adaptation, LoRA/LayerNorm slot apply |
| R6 | MCP Integration | JSON-RPC stdio/HTTP/WebSocket tool dispatch, schema validation |
| R7 | Green Power Manager | Energy/carbon recording, DVFS metadata, provenance tracking |
| R8 | Confidential TEE | Software simulation (hardware-backed requires TEE CPU) |

### v5.0 Runtime Layers (R9–R12)

| Layer | Name | Description |
|---|---|---|
| R9 | Diffusion Speculative Engine | MDLM drafter dispatch (drafter weights required) |
| R10 | KV Network Transfer | Local CPU tier movement with block/token/route statistics |
| R11 | Semantic Request Cache | Exact-match + offline-embedding cache with hit metrics |
| R12 | CXL Rack-Scale KV Pool | File-backed mmap fallback (physical CXL requires CXL hardware) |

## Stage 5: Developer Interfaces

### Python SDK
```python
from aether import Runtime, Compiler, RuntimeConfig, CompilerConfig

rt = Runtime(RuntimeConfig(speculative_decoding=True))
resp = rt.generate("model.aeg", "What is quantum entanglement?")
```

### REST API (OpenAI-Compatible)
```bash
POST /v1/generate      # Text completion
POST /v1/chat          # Chat completion (OpenAI format)
POST /v1/embeddings    # Embeddings
POST /v1/eval          # Benchmark evaluation
GET  /v1/health        # Health check
GET  /v1/models        # List loaded models
```

### CLI
```bash
aether compile model_dir/ --target auto --quality-budget 0.02
aether serve model.aeg --port 8080 --host 0.0.0.0
aether run model.aeg --prompt "Hello" --stream
aether bench model.aeg --compare baseline.aeg
aether eval model.aeg --dataset hellaswag=data/hellaswag.jsonl
aether info model.aeg
aether graph model.aeg
```

### gRPC
```proto
service AetherService {
  rpc Generate(GenerateRequest) returns (GenerateResponse);
  rpc GenerateStream(GenerateRequest) returns (stream StreamChunk);
  rpc Health(HealthRequest) returns (HealthResponse);
}
```

## AEG Format

The Aether Execution Graph (AEG) is a self-describing ZIP archive:

```
model.aeg/
├── manifest.json           # Version, integrity hashes, metadata
├── graph/graph.json        # AEG-IR computation graph
├── weights/                # Quantized weight blobs (content-addressed)
├── kernels/                # Target-specific kernel descriptors
├── safety/                 # Guardrail policies, prompt guard config
├── tokenizer/              # Vocabulary, merges, model files
├── adapters/               # LoRA/PEFT adapter manifests
├── provenance/             # Compilation lineage, hardware fingerprint
└── generated_kernels/      # Native CPU .dll/.so libraries
```

### AEG Versions
- **AEG/1.1** — v3.1 baseline (current production format)
- **AEG/2.0** — v4.0 extensions (MTP, grammar, TTT, MCP metadata)
- **AEG/3.0** — v5.0 extensions (video frames, ternary weights, RLVR verifier heads)

## Backend Plugin Model

Aether's backends are hot-swappable plugins. The `Backend` interface abstracts:

- `generate(request)` — Text generation
- `chat(messages)` — Chat completion  
- `embed(texts)` — Embedding generation
- `rerank(query, docs)` — Passage reranking
- `transcribe(audio)` — Audio transcription

New backends are registered via `BackendRegistry.register()` without changing the public API. The runtime selects the highest-capability available backend at load time, with automatic fallback chains.

## Platform Modules

| Module | Capability |
|---|---|
| `aether.agentic.workflow` | Tool trace compilation, cascade routing, KV reuse policy |
| `aether.runtime.eagle` | EAGLE-3 fusion layers, flattened tree verification |
| `aether.attention.mla` | MLA latent-KV compression plans (DeepSeek family) |
| `aether.observability.gates` | EvalGate, QualityGate, quality drift detection, A/B rollout |
| `aether.fleet.manager` | Heterogeneous node placement, hot-reload routing with rollback |
| `aether.cuda.graphs` | Piecewise CUDA Graph capture buckets for prefill/decode |
| `aether.distillation.pipeline` | Logit/feature/reasoning/self-distillation plans |
| `aether.inference.multimodal` | Unified multimodal graph with ViT-DP and LLM-TP hints |
| `aether.hub.client` | Content-addressed model registry, push/pull/search |
| `aether.safety.production_safety` | Jailbreak detection, C2PA watermarking, tenant isolation |
| `aether.parallelism.distributed` | Ring all-reduce, tensor/pipeline/expert parallelism |
