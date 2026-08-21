# Aether Runtime

**Compile once. Run on any hardware, forever.**

Aether is an open-source AI model compiler and inference runtime. It ingests any open-source model (HuggingFace, GGUF, SafeTensors, ONNX) and produces a portable **Aether Execution Graph (AEG)** artifact that runs on any detected hardware — CPU, GPU, NPU, FPGA — with zero framework dependency and zero re-compilation.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

---

## Core Principles

| Principle | Implementation |
|-----------|---------------|
| **Compile once, run anywhere** | AEG artifacts are hardware-portable; they contain multi-target sharding plans and run without re-compilation |
| **PyTorch-free core** | The runtime, compiler, and CPU engine require only NumPy + tokenizers. PyTorch is optional (`pip install "aether-runtime[pytorch]"`) |
| **Universal hardware detection** | Detects NVIDIA (CUDA), AMD (ROCm), Apple (Metal/MPS), Intel (OpenVINO), Qualcomm (QNN), RISC-V, FPGA, and pure CPU — no driver installation required |
| **Multi-GPU with VRAM-weighted distribution** | Automatically shards model weights across all available GPUs proportional to each GPU's VRAM capacity |
| **Framework-free native kernels** | C++ kernels compiled at runtime: INT4-GEMV, FlashAttention-2, fused RMSNorm+SwiGLU+Linear, GeGLU, RoPE, OpenMP parallel SGEMM |

---

## 5-Stage Compiler Pipeline

```
Model (HuggingFace / GGUF / SafeTensors / ONNX)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Ingestion & Architecture Detection                     │
│  • Reads config.json / GGUF header / SafeTensors metadata       │
│  • Detects 60+ model families without relying on model names    │
│  • Outputs: AEG-IR computation graph + ModelArchitecture        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Optimizer (22 Passes)                                  │
│  Pass 1: Operator Fusion (RMSNorm→QKV→RoPE, SwiGLU fusion)     │
│  Pass 2: Sensitivity Analysis (per-layer perplexity gradient)   │
│  Pass 3: Precision Assignment (mixed-precision per sensitivity) │
│  Pass 4: KV Cache Structuring (paged blocks, radix-tree hints)  │
│  Pass 5: MoE Expert Routing (hot/warm/cold tier classification) │
│  Pass 6: Parallelism Discovery (TP/PP/EP/CP strategy search)   │
│  Pass 7–22: Graph lowering, sparse attention, pruning, etc.     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Quantization                                           │
│  • Q4_K_M (INT4 block-scaled) — default for ≤70B models        │
│  • Q8_0 (INT8 symmetric)                                        │
│  • BF16 / FP16 / FP8 (E4M3 / E5M2)                            │
│  • MXFP4 / MXFP6 (microscaling, PRD v4.0+)                     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: AEG Packaging                                          │
│  • Self-contained .aeg/ directory with manifest, weights,       │
│    tokenizer, precision map, sharding plans                     │
│  • Integrity-verified (SHA-256 per artifact)                    │
│  • Version-stamped (AEG/1.1 – AEG/3.0)                        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: Target Code Generation                                 │
│  • CUDA (sm70–sm130), ROCm (RDNA3, CDNA3/4/5)                 │
│  • Apple Metal (M1–M5), OpenVINO (NPU/GPU)                     │
│  • Qualcomm QNN, RISC-V NPU, FPGA                              │
│  • Native CPU (AVX-512, AVX2, NEON, ternary BitNet)            │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
  AEG artifact (.aeg/) — runs anywhere, forever
```

---

## Quick Start

```bash
pip install aether-runtime

# Compile a model to AEG
aether compile meta-llama/Llama-3.1-8B --target cuda_sm90 --precision q4_k_m

# Inspect the compiled artifact
aether inspect llama-3.1-8b.aeg/

# Run inference (no GPU required for CPU target)
aether serve llama-3.1-8b.aeg/ --port 8080

# Benchmark performance
aether bench llama-3.1-8b.aeg/

# Run evaluation gate
aether eval llama-3.1-8b.aeg/ --suite reasoning --max-regression 0.02
```

### Python API

```python
from aether.compiler import AetherCompiler
from aether.compiler.config import CompilerConfig

# Compile — no PyTorch required
compiler = AetherCompiler()
config = CompilerConfig(
    target="cuda_sm90",
    precision="q4_k_m",
    max_context_length=131072,
)
artifact = compiler.compile("meta-llama/Llama-3.1-8B", config)
print(artifact.summary())   # model_id, target, precision, size_gb, tok/s estimate

# Run compiled AEG
from aether.backends import get_backend

backend = get_backend("aether_cpu")   # or "vllm", "mlx", "onnxruntime"
backend.load_model("llama-3.1-8b", aeg_path="llama-3.1-8b.aeg/")
result = backend.generate(GenerationRequest(
    model_id="llama-3.1-8b",
    prompt="What is quantum entanglement?",
    max_tokens=512,
))
print(result.text)
print(f"Throughput: {result.metrics['throughput_tps']:.1f} tok/s")
```

---

## Multi-GPU Execution (VRAM-Weighted Tensor Parallel)

When multiple GPUs are present, Aether automatically distributes the model across all of them using **VRAM-weighted capacity partitioning**. Each GPU holds a fraction of every projection layer proportional to its available VRAM — not a simple equal split.

```python
from aether.parallelism.planner import ParallelismPlanner
from aether.parallelism.sharding import DeviceCapacity

# VRAM-weighted heterogeneous sharding plan
planner = ParallelismPlanner(architecture)
plan = planner.plan_for_devices([
    DeviceCapacity(device_id="cuda:0", compute_units=1.0, memory_bytes=24 * 1024**3),  # 24 GB
    DeviceCapacity(device_id="cuda:1", compute_units=0.5, memory_bytes=12 * 1024**3),  # 12 GB
])
# cuda:0 holds ~67% of weights, cuda:1 holds ~33%
print(plan.weight_fractions)   # {"cuda:0": 0.667, "cuda:1": 0.333}
```

The compiler embeds sharding plans for 1–8 GPUs in every AEG artifact (Pass 6: Parallelism Discovery). At runtime, the distributed engine reads the matching plan and launches a custom ring-allreduce collective — no NCCL, no PyTorch distributed required.

---

## Hardware Detection

```python
from aether.backends.hardware_detector import detect_hardware

profile = detect_hardware()
print(profile.summary())
# ┌─────────────────────────────────────────────────────┐
# │ Aether Hardware Profile                             │
# │  GPUs:  2x NVIDIA RTX 4090 (24 GB each) → cuda_sm89│
# │  CPU:   AMD Ryzen 9 7950X  (AVX-512)   → cpu_avx512│
# │  RAM:   128 GB                                      │
# │  Best target: cuda_sm89                             │
# └─────────────────────────────────────────────────────┘
```

Detected targets span: NVIDIA CUDA (sm70–sm130), AMD ROCm (RDNA3, CDNA3/4/5), Apple Metal (M1–M5), Intel OpenVINO (NPU/GPU), Qualcomm QNN, RISC-V NPUs (SiFive X160, XuanTie C930, MIPS S8200), Xilinx FPGA, and CPU (AVX-512, AVX2, NEON, ternary).

---

## Native CPU Kernel Stack

The Aether CPU engine compiles a C++ shared library at first run (cached thereafter) providing the following kernels — all OpenMP-parallel and auto-vectorized to AVX-512/NEON:

| Kernel | Description | Research Basis |
|--------|-------------|----------------|
| `aether_int4_gemv` | **INT4-packed GEMV** — 2x bandwidth vs INT8, ~2x tok/s | GGML Q4_0 (2023), GPTQ (2022) |
| `aether_qgemv_affine` | INT8 affine-quantized GEMV | Frantar et al. 2022 |
| `aether_flash_attn` | FlashAttention-2 online softmax (O(seq·d) memory) | Dao, NeurIPS 2023 |
| `aether_rmsnorm_linear` | Fused RMSNorm + QKV projection (1 buffer) | ClusterFusion NeurIPS 2025 |
| `aether_rmsnorm_swiglu_linear` | Fused RMSNorm + full SwiGLU FFN (0 intermediate buffers) | Shazeer 2020, ClusterFusion 2025 |
| `aether_geglu` | GeGLU for Gemma/Gemma-2 FFN | Hendrycks 2016, Google Gemma 2024 |
| `aether_swiglu` | SwiGLU activation (Llama/Qwen/Mistral) | Shazeer 2020 |
| `aether_rope` | Rotary position embedding in-place | Su et al. 2021 |
| `aether_sgemv` | FP32 GEMV (M=1 decode fast path, ~3x vs SGEMM) | BLIS (Van Zee 2015) |
| `aether_sgemm` | Cache-blocked FP32 SGEMM (prefill) | BLIS tile layout |
| `aether_softmax` | Numerically stable row-wise softmax | Standard |
| `aether_rmsnorm` | Double-accumulation RMSNorm | Zhang & Sennrich 2019 |
| `aether_argmax` | Greedy token selection (OpenMP reduction) | Standard |

No compiler toolchain? Every kernel has a NumPy fallback — the module always imports and runs.

---

## Supported Model Families

See [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md) for the full matrix. Detection covers 100+ model families. Currently validated for end-to-end compile + AEG execution:

| Family | Models | Architecture |
|--------|--------|-------------|
| Llama | Llama-3.1-8B, 3.2-1B, 3.3-70B | GQA + SwiGLU + RMSNorm |
| Qwen | Qwen3-0.6B → 72B, Qwen2.5-VL | GQA + QKNorm + YaRN RoPE |
| Mistral | Mistral-7B, Mixtral-8x7B/8x22B | GQA + SwiGLU + MoE |
| Gemma | Gemma-2-2B/9B/27B | MQA + GeGLU |
| DeepSeek | DeepSeek-V3, DeepSeek-R1-671B | MLA + MoE (256 experts) |
| Phi | Phi-3, Phi-4 | GQA + GELU |
| BERT / RoBERTa | BERT-base/large, RoBERTa, DeBERTa | Bidirectional encoder |
| Mamba / RWKV | Mamba-3, RWKV-7, Jamba | Selective scan SSM |
| GPT-2 / GPT-NeoX | GPT-2, GPT-Neo, Pythia | Causal MHA |
| Generic decoder | OLMo, Granite, Command R/A, GLM, Kimi, 60+ more | Config-driven |
| Encoder-decoder | T5, mT5, FLAN-T5, BART | Cross-attention seq2seq |
| Vision-Language | LLaVA, InternVL, PaliGemma, Qwen2-VL | ViT encoder + decoder |
| Whisper | Whisper-{tiny…large} | Conv + cross-attention |

---

## Supported Hardware Targets

| Target ID | Hardware |
|-----------|----------|
| `cuda_sm89` | NVIDIA RTX 4090 (Ada Lovelace) |
| `cuda_sm90` | NVIDIA H100 (Hopper) |
| `cuda_sm100` | NVIDIA B200 (Blackwell) |
| `cuda_sm130` | NVIDIA Rubin Ultra (sm_130) |
| `rocm_cdna3` | AMD MI300X |
| `rocm_cdna5_mi455x` | AMD MI455X (CDNA5) |
| `metal_m3` | Apple M3/M4/M5 |
| `openvino_npu` | Intel Arc NPU |
| `qualcomm_qnn` | Qualcomm Snapdragon NPU |
| `cpu_avx512` | x86-64 with AVX-512 |
| `cpu_neon` | ARM NEON (mobile, Raspberry Pi) |
| `cpu_avx512_ternary` | BitNet b1.58 ternary on x86 |
| `riscv_sifive_x160` | SiFive Intelligence X160 |
| `fpga_xilinx_vu9p` | Xilinx VU9P (decode-only) |

Full list: 30+ targets in [`src/aether/core/constants.py`](src/aether/core/constants.py).

---

## Phase 5 — Observability

```python
from aether.observability.otel import AetherTracer, OTLPExporter
from aether.observability.ci_pipeline import CIEvalPipeline
from aether.observability.gates import DriftMonitor, ABRolloutController

# OpenTelemetry distributed tracing
tracer = AetherTracer(service_name="aether-prod", sample_rate=0.01)
exporter = OTLPExporter(endpoint='http://otel-collector:4318/v1/traces')

# CI eval gate (blocks regressions > 2% on HellaSwag/MMLU/GSM8K)
pipeline = CIEvalPipeline(aeg_path='model.aeg', max_regression=0.02)
report = pipeline.run_and_save('eval_report.json', benchmarks=['hellaswag', 'mmlu', 'gsm8k'])

# A/B rollout with auto drift detection
ctrl = ABRolloutController('exp-001', candidate_percent=0.01)
monitor = DriftMonitor(baseline_win_rate=0.80, alert_drop=0.05, min_samples=20)
```

**Prometheus metrics:** `aether_request_total`, `aether_ttft_ms{quantile=p50|p95|p99}`, `aether_tokens_per_second`, `aether_kv_hit_rate`, `aether_eagle_accept_rate`

---

## Phase 6 — Ecosystem

```python
from aether.ecosystem.sdks import TypeScriptSDKGenerator, GoSDKGenerator, RustSDKGenerator

TypeScriptSDKGenerator().write('./sdk/typescript/')   # aether-sdk.ts
GoSDKGenerator().write('./sdk/go/')                   # aether_client.go
RustSDKGenerator().write('./sdk/rust/src/')           # aether_client.rs
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `aether compile <model>` | Compile model to AEG package |
| `aether inspect <path.aeg>` | Show AEG package summary |
| `aether bench <path.aeg>` | Run benchmark suite |
| `aether serve <path.aeg>` | Start inference server |
| `aether eval <path.aeg>` | Run eval gate CI check |
| `aether hardware` | Show hardware profile |
| `aether hub push <path.aeg>` | Push to Aether Hub CDN |
| `aether hub pull <model-id>` | Pull from Aether Hub CDN |
| `aether sdk generate` | Generate TypeScript/Go/Rust SDKs |
| `aether sign <path.aeg>` | Sign package (C2PA binding) |
| `aether verify <path.aeg>` | Verify signature + fingerprint |

---

## Installation

```bash
# Core runtime (no PyTorch, no CUDA required)
pip install aether-runtime

# With PyTorch for .pt/.pth model ingestion
pip install "aether-runtime[pytorch]"

# With HuggingFace Transformers for AutoTokenizer, AutoConfig
pip install "aether-runtime[transformers-frontend]"

# Full install (all optional backends)
pip install "aether-runtime[full]"
```

---

## Testing

```bash
python -m pytest tests/ -v                                     # All tests
python -m pytest tests/unit/test_native_cpu_kernels.py -v     # CPU kernels
python -m pytest tests/unit/test_phase5_observability.py -v   # Observability
python -m pytest tests/unit/test_phase6_ecosystem.py -v       # Ecosystem SDKs
python -m pytest tests/unit/test_v31_elite_extensions.py -v   # v3.1 extensions
```

> **Run the suite serially.** `test_e2e_compile_run_cpu.py` and `test_v31_features.py` share the `~/.aether` cache; parallel pytest workers race on it.
>
> Tests requiring HuggingFace weights skip cleanly when offline.

---

## Research Citations

| Feature | Research |
|---------|----------|
| INT4 GEMV | Gerganov GGML Q4_0 (2023), Frantar GPTQ (2022) |
| FlashAttention-2 | Dao et al., NeurIPS 2023 |
| Operator Fusion | ClusterFusion, NeurIPS 2025 |
| SwiGLU / GeGLU | Shazeer 2020 (GLU Variants); Hendrycks & Gimpel 2016 |
| BLIS SGEMM tiles | Van Zee & van de Geijn, TOMS 2015 |
| VRAM-weighted TP | Megatron-LM (Shoeybi et al. 2019); DeepSpeed (Rasley et al. 2020) |
| MoE Expert Routing | Zipf prior: Zoph et al. 2022; Fedus et al. 2022 |
| Sparse Attention | MInference (Microsoft, NeurIPS 2024) |
| KV Eviction | StreamingLLM (2023), ScissorHands (2024), SnapKV (2025) |
| Ring Attention | Ring Attention (2023), Striped Attention (2023) |
| YaRN RoPE | YaRN (2023), LongRoPE (2024) |
| Speculative Decoding | EAGLE-2 (2024), Medusa (2024) |
| CUDA Graphs | vLLM CUDA Graphs Dispatcher (2026) |
| Process Reward Model | Let's Verify Step by Step (2023), OmegaPRM (2025) |
| IP Fingerprinting | MetaFinger (2024), ADV-TRA (2025) |
| EU AI Act Compliance | Article 50 — AI content transparency obligations |
| Fleet Scheduling | Helium (2026), MuxWise SLO-aware scheduling (2026) |
| Disaggregated Serving | DistServe (2024), Mooncake (2024) |

---

## License

Apache 2.0

*Aether Runtime — Compile once. Run on any hardware, forever.*
