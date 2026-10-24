# Aether Architecture

Aether has four compiler stages and a backend-orchestrated runtime. Supported
operators execute through compatible declared backend contracts. CPU and Torch
engines own handwritten forward passes built from weights/config; they do not
interpret optimized graph nodes. See the [traced architecture decision](mission-architecture-note.md)
and [measured pass dispositions](mission-default-policy.md#all-22-dispositions-and-remaining-work).

## Compiler, runtime and interfaces

```
Supported model/source (normal numerical admission currently requires HF SafeTensors)
                    │
                    ▼
Original FP32 HF reference worker → bounded fixed-probe evidence, then worker exits
                    │
                    ▼
Stage 1: Model Ingestion & Graph Extraction → AEG-IR
         (SafeTensors, GGUF, ONNX, MLX, PyTorch, VLM, Video, MLA, MoE, SSM)
                    │
                    ▼
Stage 2: Aether Optimizer (22 registered passes) → weights / IR / selected payloads
                    │
                    ▼
Stage 3: Hardware Targeting & Backend Selection → compatible backend contracts
                    │
                    ▼
Stage 4: Packaging → private AEG → actual CPU + Torch CPU quality gate → publication
                    │
                    ▼
Runtime: authenticated admission → engine execution → tokens / embeddings
Interfaces: Python SDK / REST / CLI / gRPC / OpenAI-compat
```

## Stage 1: Model Ingestion

Loader catalog (presence does not establish family execution or parity):

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
| 1 | Operator Fusion | Experimental IR groups; no engine schedule consumer |
| 2 | Sensitivity Analysis | Experimental reconstruction/entropy proxy, not measured PPL |
| 3 | Precision Assignment | Default BF16 storage; explicit FP32 preserved; independent numerical gate required |
| 4 | KV Cache Structuring | Experimental graph hints; engine caches are independent |
| 5 | MoE Expert Routing | Experimental tiers; source routing executes independently |
| 6 | Parallelism Discovery | Experimental inspection plans; runtime uses a separate planner |
| 7 | Reasoning Graph Compilation | Experimental annotations; serving benefit unproved |
| 8 | Sparse Attention | Experimental dense-score masking; no sparse FLOP savings verified |
| 9 | Pruning & Sparsity | Experimental lossy zeroed dense weights; sparse-kernel benefit unproved |

### v4.0 Advanced Passes (10–17)

| Pass | Name | Status |
|---|---|---|
| 10 | Native MTP Head Compilation | Supplied-head experimental feature; full requested-head benefit unproved |
| 11 | Grammar Constraint Compiler | Explicit functional FSA; constrained A/B remains required |
| 12 | Model Merging | Changed weights consumed; independent transformed-reference evaluation required |
| 13 | TTT Fast-Weight Injection | Experimental slots; independent adaptation quality unproved |
| 14 | Semantic KV Compression | Experimental CPU retention; generation regressions recorded |
| 15 | Cross-Layer KV Sharing | Experimental CPU aliases; quality regressions recorded |
| 16 | Green Energy Compilation | Accounting; no measured device energy reduction |
| 17 | TEE Enclave Emission | Requires real TEE backend (fail-closed otherwise) |

### v5.0 Frontier Passes (18–22)

| Pass | Name | Status |
|---|---|---|
| 18 | Diffusion Drafter Compilation | MDLM drafter — requires trained weights |
| 19 | Sub-2-Bit/Ternary Quantization | Experimental lossy storage with dense dequantization; no QAT/kernel speed claim |
| 20 | Video Token Compression | Unsupported end-to-end CPU VLM execution |
| 21 | Advanced PEFT Compilation | Experimental supplied tensors; scaling/reference checks remain open |
| 22 | RLVR Verifier Head Injection | Experimental training integration; gradient callbacks required |

Only P3 is enabled by default. An `applied` pass status is not a numerical
certificate or a measured performance benefit. [Numerical admission](mission-numerical-admission.md)
checks the resulting base CPU/Torch CPU execution before ordinary publication;
unsupported references and insufficient memory are explicit refusals.

## Stage 3: Hardware Targeting

Target profiles describe preferred backends. Actual availability and dated
execution evidence are separate; use the generated README table and
`hardware_validation_matrix.json` for current status:

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

The following runtime module catalog includes optional and experimental paths.
Module availability does not prove automatic use, correctness or speed. Ordinary
AEG loads validate [bound numerical evidence](mission-numerical-admission.md)
before backend selection; diagnostic mode waives absence only.

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

The Aether Execution Graph (AEG) is a directory package with versioned manifests
and authenticated payloads; distribution archives are unpacked before execution:

```
model.aeg/
├── manifest.json           # Version, integrity hashes, metadata
├── graph/computation_graph.aeg-ir  # AEG-IR computation graph
├── weights/                # Quantized weight blobs (content-addressed)
├── kernels/                # Target-specific kernel descriptors
├── safety/                 # Guardrail policies, prompt guard config
├── tokenizer/              # Vocabulary, merges, model files
├── adapters/               # LoRA/PEFT adapter manifests
├── provenance/             # Compilation lineage, hardware fingerprint
├── generated_kernels/      # Optional native CPU .dll/.so libraries
└── quality/numerical_gate.json  # Measured admission, absent in diagnostic builds
```

### AEG Versions
- **AEG/1.1** — v3.1 baseline reader format
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
| `aether.safety.production_safety` | Jailbreak detection, HMAC output disclosure markers, tenant isolation |
| `aether.parallelism.distributed` | TCP ring collectives (CPU multi-process), tensor/pipeline/expert parallelism |
