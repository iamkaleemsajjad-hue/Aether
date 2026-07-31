# Aether Runtime

> **Compile once. Run on any hardware, forever.**

Aether Runtime is an open-source, hardware-portable LLM inference engine that compiles any model into a single binary **AEG** (Aether Executable Graph) artifact and runs it natively across CPU, CUDA, ROCm, Metal, and OpenVINO targets — all without re-quantizing or re-compiling per device.

---

## Architecture

```
Input Model                 AEG Compiler                  Aether Runtime
─────────────  ──────────────────────────────────  ────────────────────────
safetensors ─┐                                     ┌─ CPU (AVX-512 / NEON)
GGUF        ─┤  Stage 1: Ingestion                 ├─ CUDA (sm70 → sm100)
ONNX        ─┤  Stage 2: Optimization (9 passes)   ├─ ROCm (RDNA3/CDNA3)
MLX         ─┤  Stage 3: Targeting                 ├─ Metal (M1 → M3)
PyTorch     ─┘  ─────────────────────────────────▶ └─ OpenVINO
                         AEG Package
                  manifest + graph + weights
```

---

## Phases Implemented

### ✅ Phase 1 — Core Compiler E2E (100%)

| Feature | Status | File |
|---------|--------|------|
| SafeTensors ingestion + weight attachment | ✅ | `stage1_ingestion/ingestion.py` |
| **GGUF ingestion** — pure-Python binary parser, K-quant dequant | ✅ | `stage1_ingestion/gguf_loader.py` |
| **ONNX ingestion** — ONNX→AEG-IR op lowering + weight extraction | ✅ | `stage1_ingestion/onnx_loader.py` |
| **PyTorch ingestion** — state dict + sharded HF checkpoints | ✅ | `stage1_ingestion/pytorch_loader.py` |
| **MLX ingestion** — safetensors / npz / native mlx.core | ✅ | `stage1_ingestion/mlx_loader.py` |
| Architecture detection (LLaMA, Qwen, Mistral, Falcon, Gemma, GPT-2…) | ✅ | `stage1_ingestion/architecture_detector.py` |
| AEG graph IR (`AEGGraph`, `AEGNode`, `AEGInstruction`) | ✅ | `core/aeg_ir.py` |
| AEG package format (`AEGPackage`, `AEGManifest`, `load_aeg_package`) | ✅ | `core/aeg_format.py` |
| Optimizer pass 1 — Operator fusion | ✅ | `stage2_optimizer/pass1_operator_fusion.py` |
| Optimizer pass 2 — Sensitivity analysis | ✅ | `stage2_optimizer/pass2_sensitivity_analysis.py` |
| Optimizer pass 3 — Precision assignment | ✅ | `stage2_optimizer/pass3_precision_assignment.py` |
| Optimizer passes 4–9 — KV structuring, MoE, parallelism, reasoning, sparse, pruning | ✅ | `stage2_optimizer/optimizer.py` |
| Calibration datasets (WikiText-2, Hellaswag, custom JSONL) | ✅ | `calibration/datasets.py` |
| Perplexity evaluator (corpus entropy + precision penalty model) | ✅ | `calibration/perplexity.py` |
| Weight quantization codecs (Q4_K_M, Q8_0, FP8_E4M3, INT4, INT8, BF16) | ✅ | `quantization/` |
| AEG weight persistence (quantized weights in `.aed` directory) | ✅ | `runtime/aeg_loader.py` |
| CPU execution engine (numpy forward pass, all LLM op types) | ✅ | `runtime/cpu_engine.py` |
| E2E compile → run pipeline (safetensors → AEG → CPU inference) | ✅ | `tests/unit/test_e2e_compile_run_cpu.py` |
| Hardware detection & fingerprinting | ✅ | `runtime/hardware.py` |

### ✅ Phase 2 — GPU-Native + Advanced Runtime (100%)

| Feature | Status | File |
|---------|--------|------|
| **EAGLE-3** tree-speculative decoding engine | ✅ | `runtime/eagle.py` |
| EAGLE-3 multi-layer feature extrapolation (offline mode) | ✅ | `runtime/eagle.py` |
| EAGLE-3 speculative sampling verify + rejection correction | ✅ | `runtime/eagle.py` |
| **FlashAttention-2** tiled numpy reference + flash_attn dispatch | ✅ | `kernels/attention.py` |
| Grouped Query Attention (GQA/MQA — LLaMA-3, Qwen3) | ✅ | `kernels/attention.py` |
| Sliding Window Attention (Mistral style) | ✅ | `kernels/attention.py` |
| Paged Attention (vLLM-style block-sparse KV cache) | ✅ | `kernels/attention.py` |
| Attention dispatcher (auto-selects best kernel) | ✅ | `kernels/attention.py` |
| **Dynamic precision manager** (BF16→FP8→Q4 on pressure) | ✅ | `runtime/precision_manager.py` |
| Precision ladder with quality budget enforcement | ✅ | `runtime/precision_manager.py` |
| **Model registry** with LRU eviction + reference counting | ✅ | `runtime/model_registry.py` |
| Hot-reload (atomic model replacement) | ✅ | `runtime/model_registry.py` |
| KV cache manager — tiered L1/L2/L3/L4 + prefix hashing | ✅ | `runtime/kv_cache.py` |
| Disaggregated prefill/decode scheduler | ✅ | `runtime/scheduler.py` |
| TreeSpeculativeEngine (high-level wrapper) | ✅ | `runtime/speculative.py` |
| **llama.cpp backend** — in-process + subprocess/REST modes | ✅ | `backends/llamacpp_backend.py` |
| **vLLM backend** — in-process LLM engine | ✅ | `backends/vllm_backend.py` |
| PyTorch backend — HF AutoModel + AEG handle | ✅ | `backends/torch_backend.py` |
| MLX backend (Apple Silicon) | ✅ | `backends/mlx_backend.py` |
| ONNX Runtime backend | ✅ | `backends/onnx_backend.py` |
| TensorRT-LLM backend | ✅ | `backends/trtllm_backend.py` |
| OpenAI-compatible REST server (FastAPI) | ✅ | `server/routes.py` |
| Server middleware (CORS, auth, rate-limit) | ✅ | `server/middleware.py` |
| **Hub client** — real HTTP with retry/backoff + local fallback | ✅ | `hub/client.py` |
| Hub ZIP archive upload/download | ✅ | `hub/client.py` |
| Runtime `generate()` / `chat()` / `embed()` / `rerank()` | ✅ | `runtime/runtime.py` |
| Inference metrics (TPS, TTFT, P95, spec accept rate, KV hit rate) | ✅ | `runtime/runtime.py` |

---

## GGUF Dequantization Support

The GGUF loader supports **zero-dependency** dequantization (no `gguf` pip package required):

| GGML Type | Status | Details |
|-----------|--------|---------|
| `F32`     | ✅ | Direct cast |
| `F16`     | ✅ | numpy float16→float32 |
| `BF16`    | ✅ | uint16 bit-shift |
| `Q8_0`    | ✅ | 32 int8 + f16 scale per block |
| `Q4_0`    | ✅ | 32 nibbles + f16 scale, shifted by −8 |
| `Q4_K_M`  | ✅ | 256-element super-block, NF4 lookup |
| `Q5_K`    | ✅ | 256-element super-block |
| `Q6_K`    | ✅ | 256-element super-block, 6-bit quantized |
| `Q2_K`    | ✅ | 256-element super-block, 2-bit quantized |
| `Q3_K`    | ✅ | 256-element super-block, 3-bit quantized |

---

## Optimizer Passes

| Pass | Name | Status |
|------|------|--------|
| 1 | Operator Fusion (QKV, FFN-SwiGLU) | ✅ Full |
| 2 | Sensitivity Analysis (calibration-driven) | ✅ Full |
| 3 | Mixed-Precision Assignment | ✅ Full |
| 4 | KV Cache Structuring | ✅ Full |
| 5 | MoE Expert Routing Optimization | ✅ Full |
| 6 | Tensor Parallelism Discovery | ✅ Full |
| 7 | Reasoning Graph Extraction | ✅ Full |
| 8 | Sparse Attention Pattern Detection | ✅ Full |
| 9 | Pruning & Sparsity | ✅ Full |

---

## Quick Start

```python
from aether import Compiler, Runtime

# Compile a model once
compiler = Compiler()
aeg = compiler.compile("Qwen/Qwen3-0.6B")
aeg.save("./models/qwen3-0.6b")

# Load and run on any hardware
rt = Runtime()
response = rt.generate(
    "Qwen/Qwen3-0.6B",
    "Explain the AEG format in one sentence.",
    max_tokens=100,
)
print(response.text)
print(f"TPS: {response.metrics.throughput_tps:.1f}")
```

### Serve via OpenAI-compatible API

```bash
aether serve --model Qwen/Qwen3-0.6B --port 8080
```

```bash
curl http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello!", "max_tokens": 50}'
```

---

## Supported Hardware Targets

| Target | Backend | Status |
|--------|---------|--------|
| `cpu_avx512` | PyTorch / llama.cpp | ✅ |
| `cpu_neon` | PyTorch / llama.cpp | ✅ |
| `cuda_sm80` (A100) | vLLM / PyTorch / TRT-LLM | ✅ |
| `cuda_sm90` (H100) | vLLM / PyTorch / TRT-LLM | ✅ |
| `cuda_sm100` (B200) | vLLM | ✅ |
| `rocm_rdna3` | PyTorch / ROCm | ✅ |
| `metal_m1`/`metal_m3` | MLX / PyTorch | ✅ |
| `openvino` | OpenVINO | ✅ |

---

## Project Structure

```
src/aether/
├── compiler/
│   ├── stage1_ingestion/     # Format loaders: safetensors, GGUF, ONNX, MLX, PyTorch
│   ├── stage2_optimizer/     # 9 optimization passes
│   ├── stage3_targeting/     # Hardware targeting and kernel selection
│   └── calibration/          # Calibration datasets + perplexity evaluator
├── core/                     # AEG graph IR, format spec, types, constants
├── quantization/             # Q4_K_M, Q8_0, FP8, INT4/INT8 codecs
├── runtime/
│   ├── cpu_engine.py         # Full numpy CPU forward pass
│   ├── eagle.py              # EAGLE-3 tree-speculative decoding
│   ├── precision_manager.py  # Dynamic BF16→FP8→Q4 on pressure
│   ├── model_registry.py     # LRU model registry with ref-counting
│   ├── kv_cache.py           # Tiered KV cache (L1/L2/L3/L4)
│   ├── scheduler.py          # Disaggregated prefill/decode
│   └── speculative.py        # TreeSpeculativeEngine wrapper
├── kernels/
│   ├── attention.py          # FA-2, GQA, SlidingWindow, Paged, Dispatcher
│   ├── gemm.py               # GEMM kernels
│   ├── ffn.py                # FFN (SwiGLU, GEGLU)
│   ├── norm.py               # RMSNorm, LayerNorm
│   └── rope.py               # Rotary position embedding
├── backends/                 # PyTorch, vLLM, llama.cpp, MLX, ONNX, TRT-LLM
├── server/                   # FastAPI OpenAI-compatible server
├── hub/                      # Aether Hub HTTP client
└── utils/                    # Logging, file I/O, profiling
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Phase 1 tests only
python -m pytest tests/unit/test_e2e_compile_run_cpu.py tests/unit/test_gguf_loader.py tests/unit/test_format_loaders.py -v

# Phase 2 tests only
python -m pytest tests/unit/test_phase2_runtime.py tests/unit/test_hub_client.py -v
```

---

## Installation

```bash
# Core (CPU inference, compiler)
pip install aether-runtime

# With CUDA support
pip install aether-runtime[cuda]

# With llama.cpp (GGUF / CPU K-quant)
pip install aether-runtime[llamacpp]

# With vLLM (NVIDIA high-throughput)
pip install aether-runtime[vllm]

# With MLX (Apple Silicon)
pip install aether-runtime[mlx]

# Everything
pip install aether-runtime[all]
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
