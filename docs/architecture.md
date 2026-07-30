# Aether Architecture

Aether is a five-stage compiler plus a backend-orchestrated runtime. The design is inspired by LLVM: just as C/C++ compiles to LLVM IR and then to any ISA, any AI model compiles to AEG-IR and then to any inference backend.

## Five Compiler Stages

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

## Stage 1: Model Ingestion

Supported formats:

| Format | Method |
|--------|--------|
| SafeTensors | Direct weight loading + config.json parsing |
| GGUF | Header parsing + dequantization for tracing |
| ONNX | ONNX protobuf graph → AEG-IR lowering |
| MLX | Module tracing → AEG-IR |
| PyTorch `.pt` / `.bin` | `torch.export` graph capture → AEG-IR |

Architecture detection inspects the graph structure, not just the model name. This makes Aether robust to custom models, fine-tuned variants, and future architectures.

## Stage 2: The Aether Optimizer

Six compiler passes transform the raw AEG-IR into an optimized graph:

1. **Operator Fusion** — fuse RMSNorm + QKV + RoPE + attention into megakernels.
2. **Sensitivity Analysis** — compute `d(perplexity)/d(precision)` per layer.
3. **Precision Assignment** — assign mixed precision using the sensitivity map.
4. **KV Cache Structuring** — paged blocks, radix-tree hints, tiering.
5. **MoE Expert Routing** — hot/warm/cold expert tiering and threshold-based routing.
6. **Automatic Parallelism Discovery** — search tensor/pipeline/expert/context parallelism.

## Stage 3: Hardware Targeting

Aether maintains a registry of hardware targets and selects the best available backend for each target:

| Target | Preferred Backend |
|--------|-------------------|
| NVIDIA | vLLM, TensorRT-LLM, PyTorch |
| Apple Silicon | MLX, PyTorch |
| AMD | PyTorch, llama.cpp |
| Intel / CPU | ONNX Runtime, llama.cpp, PyTorch |

Kernel emission is expressed as backend-specific invocation plans, not hand-written CUDA/Metal/ROCm kernels. Triton or Python kernels are used only where they provide clear value.

## Stage 4: Runtime

The runtime loads the AEG artifact, fingerprints the hardware, selects the best backend, and runs a disaggregated prefill/decode scheduler with:

- Tree-speculative decoding
- Global tiered KV cache (GPU HBM → CPU DRAM → NVMe SSD → Aether Hub)
- Dynamic precision adjustment under memory pressure
- Continuous batching

## Stage 5: Developer Interface

- **Python SDK**: `Runtime`, `Compiler`, `RuntimeConfig`, `CompilerConfig`
- **CLI**: `compile`, `run`, `serve`, `bench`, `info`, `graph`, `list`, `rm`, `hw`, `kernels`, `logs`
- **REST API**: OpenAI-compatible `/v1/chat`, `/v1/generate`, `/v1/embeddings`, `/v1/rerank`, `/v1/transcribe`
- **Aether Hub**: opt-in public content-addressed kernel cache and model registry

## Backend Plugin Model

Aether's backends are replaceable plugins. The `Backend` interface abstracts:

- Model loading
- Text generation
- Chat completion
- Embeddings
- Reranking
- Transcription

New backends can be added without changing the public API. This lets Aether leverage the best existing inference engines while adding Aether's own scheduling, caching, and compilation layers.
