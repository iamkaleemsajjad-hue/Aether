# Getting Started with Aether Runtime

Welcome! This guide walks you through installing Aether, compiling your first AI model, and running it — step by step.

> **Aether's core promise:** Install once, compile a model once, run it on any hardware — forever. No PyTorch required to run a compiled model.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Verify the Installation](#3-verify-the-installation)
4. [Compile Your First Model](#4-compile-your-first-model)
5. [Run Inference](#5-run-inference)
6. [Start the API Server](#6-start-the-api-server)
7. [Inspect a Compiled Model](#7-inspect-a-compiled-model)
8. [Benchmark Performance](#8-benchmark-performance)
9. [Multi-GPU Setup](#9-multi-gpu-setup)
10. [Using the Python API](#10-using-the-python-api)
11. [Common Workflows](#11-common-workflows)
12. [CLI Quick Reference](#12-cli-quick-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### Python

You need **Python 3.10 or newer**.

```bash
# Check your Python version
python --version
# Should print: Python 3.10.x or higher
```

Download Python from [python.org](https://www.python.org/downloads/) if needed.

### C++ Compiler (Strongly Recommended)

Aether compiles native CPU kernels at first run for maximum performance. This requires a C++ compiler.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install -y build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:** Download [WinLibs GCC](https://winlibs.com/) (recommended) or install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022).

> Without a compiler, Aether still works using a NumPy fallback — results are identical, just slower.

---

## 2. Installation

### Option A — CPU Only (Fastest Install, No GPU Required)

The default install includes everything needed to compile and run models on CPU:

```bash
pip install aether-runtime
```

### Option B — Compile from HuggingFace Model Names

To compile models directly from HuggingFace (e.g. `meta-llama/Llama-3.1-8B`):

```bash
pip install "aether-runtime[transformers-frontend]"
```

### Option C — GGUF Files (llama.cpp format)

To compile `.gguf` model files:

```bash
pip install "aether-runtime[formats]"
```

### Option D — With API Server

To run the OpenAI-compatible HTTP server:

```bash
pip install "aether-runtime[server]"
```

### Option E — NVIDIA GPU Backend

```bash
pip install "aether-runtime[vllm]"
```

### Option F — Apple Silicon (M1/M2/M3/M4)

```bash
pip install "aether-runtime[mlx]"
```

### Option G — Everything (Full Install)

```bash
pip install "aether-runtime[full]"
```

> See [REQUIREMENTS.md](REQUIREMENTS.md) for a complete breakdown of every optional extra.

---

## 3. Verify the Installation

Run these commands to confirm everything is working:

```bash
# Check the CLI is installed
aether --version

# Detect what hardware Aether can see on your system
aether hardware
```

Expected output from `aether hardware`:
```
┌──────────────────────────────────────────────────────────┐
│  Aether Hardware Profile                                  │
│  GPUs:  1x NVIDIA RTX 4090 (24 GB) → cuda_sm89          │
│  CPU:   Intel Core i9-13900K  (AVX-512) → cpu_avx512    │
│  RAM:   64 GB                                            │
│  Best target: cuda_sm89                                  │
└──────────────────────────────────────────────────────────┘
```

Verify the native CPU kernels compile correctly:

```bash
python -c "
from aether.kernels.native_cpu import get_native_kernels
k = get_native_kernels()
k.ensure_compiled()
print('Native kernels:', repr(k))
print('Available:', k.available_kernels())
"
```

Expected output:
```
Native kernels: NativeCPUKernels(native/g++, 16 kernels)
Available: ['aether_argmax', 'aether_flash_attn', 'aether_geglu', ...]
```

---

## 4. Compile Your First Model

### From a HuggingFace Model Name

> Requires `pip install "aether-runtime[transformers-frontend]"`

```bash
# Compile Qwen3-0.6B for your CPU (smallest model, great for testing)
aether compile Qwen/Qwen3-0.6B --target cpu_avx512 --precision q4_k_m
```

```bash
# Compile Llama-3.1-8B for NVIDIA GPU (requires HuggingFace login for gated models)
aether compile meta-llama/Llama-3.1-8B --target cuda_sm89 --precision q4_k_m
```

```bash
# Compile Gemma-2-2B for Apple Silicon
aether compile google/gemma-2-2b --target metal_m3 --precision q4_k_m
```

### From a GGUF File

> Requires `pip install "aether-runtime[formats]"`

```bash
# Download a GGUF file first (example: llama.cpp format)
aether compile ./llama-3.1-8B.Q4_K_M.gguf --target cpu_avx512
```

### From a Local Directory (SafeTensors or PyTorch)

```bash
# From a local HuggingFace model directory
aether compile ./my-model-directory/ --target cpu_avx512 --precision q8_0
```

### Compilation Options

| Flag | Description | Example |
|------|-------------|---------|
| `--target` | Hardware target to optimize for | `cuda_sm89`, `cpu_avx512`, `metal_m3` |
| `--precision` | Quantization format | `q4_k_m`, `q8_0`, `bf16`, `fp16` |
| `--output` | Output path for the `.aeg/` artifact | `--output ./my-model.aeg` |
| `--max-context` | Maximum context length in tokens | `--max-context 131072` |
| `--num-gpus` | Number of GPUs for tensor-parallel sharding | `--num-gpus 2` |

**All available hardware targets:**
```bash
aether targets list
```

---

## 5. Run Inference

### From the CLI

```bash
# Interactive prompt (single generation)
aether run qwen3-0.6b.aeg/ --prompt "What is quantum computing?"

# Pipe a prompt via stdin
echo "Explain transformers in one paragraph." | aether run qwen3-0.6b.aeg/

# Set generation parameters
aether run llama-3.1-8b.aeg/ \
  --prompt "Write a Python function to sort a list" \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9
```

### From Python

```python
from aether.backends import get_backend
from aether.backends.base import GenerationRequest

# Load the compiled AEG artifact
backend = get_backend("aether_cpu")   # Use CPU engine (no GPU needed)
backend.load_model("my-model", aeg_path="qwen3-0.6b.aeg/")

# Generate text
result = backend.generate(GenerationRequest(
    model_id="my-model",
    prompt="Explain quantum entanglement simply.",
    max_tokens=256,
    temperature=0.7,
))

print(result.text)
print(f"Tokens generated: {result.metrics['completion_tokens']}")
print(f"Throughput: {result.metrics['throughput_tps']:.1f} tok/s")
print(f"Time to first token: {result.metrics['ttft_ms']:.0f} ms")
```

---

## 6. Start the API Server

The server exposes an **OpenAI-compatible REST API**, so any tool that works with OpenAI's API (LangChain, LlamaIndex, Continue, etc.) works with Aether automatically.

> Requires `pip install "aether-runtime[server]"`

```bash
# Start the server on port 8080
aether serve llama-3.1-8b.aeg/ --port 8080
```

```bash
# Multiple models, custom host
aether serve qwen3-0.6b.aeg/ llama-3.1-8b.aeg/ --host 0.0.0.0 --port 8080
```

### Test the Server

```bash
# Using curl (OpenAI-compatible endpoint)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello! What can you do?"}],
    "max_tokens": 200
  }'
```

```bash
# List available models
curl http://localhost:8080/v1/models
```

### Using the Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",      # Aether doesn't require an API key locally
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the theory of relativity."},
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
```

### Prometheus Metrics

The server exposes metrics at `/metrics` (Prometheus format):

```bash
curl http://localhost:8080/metrics
```

Key metrics:
- `aether_request_total` — total requests served
- `aether_ttft_ms{quantile="0.95"}` — P95 time to first token
- `aether_tokens_per_second` — current decode throughput
- `aether_kv_hit_rate` — KV cache prefix hit rate

---

## 7. Inspect a Compiled Model

```bash
# Summary: model family, precision, size, targets, token count
aether inspect qwen3-0.6b.aeg/
```

Expected output:
```
┌─────────────────────────────────────────────────────────┐
│  AEG Package: qwen3-0.6b.aeg/                           │
│  Model:       Qwen3-0.6B                                │
│  Family:      qwen3 (GQA + SwiGLU + RMSNorm)           │
│  Precision:   Q4_K_M                                    │
│  Parameters:  0.6B                                      │
│  Disk size:   362 MB                                    │
│  Context:     32 768 tokens                             │
│  Targets:     cpu_avx512, cpu_avx2, cuda_sm70+          │
│  AEG version: 2.0                                       │
│  Signed:      No                                        │
│  SHA-256:     a3f9d1...                                 │
└─────────────────────────────────────────────────────────┘
```

```bash
# Detailed: list all files in the AEG package
aether inspect qwen3-0.6b.aeg/ --verbose

# Show the sharding plan for a 2-GPU setup
aether inspect qwen3-0.6b.aeg/ --sharding-plan 2
```

---

## 8. Benchmark Performance

```bash
# Quick benchmark (50 requests, greedy decode)
aether bench qwen3-0.6b.aeg/

# Full benchmark with custom config
aether bench llama-3.1-8b.aeg/ \
  --num-requests 200 \
  --warmup 20 \
  --max-input-tokens 512 \
  --max-output-tokens 256 \
  --concurrency 4

# Save results to JSON
aether bench llama-3.1-8b.aeg/ --output results.json

# Compare against a previous run
aether bench llama-3.1-8b.aeg/ --compare baseline.json
```

Benchmark output includes:

| Metric | Description |
|--------|-------------|
| **TTFT** (P50/P95/P99) | Time To First Token in milliseconds |
| **TBT** | Time Between Tokens (decode step latency) |
| **Throughput** | Tokens per second (tok/s) |
| **Memory** | Peak RAM / VRAM used |
| **KV hit rate** | Percentage of prefix cache hits |

---

## 9. Multi-GPU Setup

When multiple GPUs are detected, Aether automatically distributes the model weights across all of them, **proportional to each GPU's VRAM** (not a simple equal split).

```bash
# Compile for 2 GPUs — sharding plan embedded into the artifact
aether compile meta-llama/Llama-3.1-70B \
  --target cuda_sm89 \
  --precision q4_k_m \
  --num-gpus 2

# Run — Aether picks the right sharding plan automatically
aether serve llama-3.1-70b.aeg/ --port 8080
```

### Python: Custom Device Weights

```python
from aether.parallelism.planner import ParallelismPlanner
from aether.parallelism.sharding import DeviceCapacity

# Example: GPU 0 has 24 GB, GPU 1 has 12 GB
# Aether gives GPU 0 twice as many layers
plan = planner.plan_for_devices([
    DeviceCapacity(device_id="cuda:0", compute_units=1.0, memory_bytes=24 * 1024**3),
    DeviceCapacity(device_id="cuda:1", compute_units=0.5, memory_bytes=12 * 1024**3),
])
print(plan.weight_fractions)
# → {"cuda:0": 0.667, "cuda:1": 0.333}
```

---

## 10. Using the Python API

### Compile a Model

```python
from aether.compiler import AetherCompiler
from aether.compiler.config import CompilerConfig

compiler = AetherCompiler()

config = CompilerConfig(
    target="cpu_avx512",           # Hardware target
    precision="q4_k_m",            # Quantization format
    max_context_length=32768,       # Max tokens in context
)

# Compile from a HuggingFace model name
artifact = compiler.compile("Qwen/Qwen3-0.6B", config)
print(artifact.summary())
```

### Detect Hardware

```python
from aether.backends.hardware_detector import detect_hardware

profile = detect_hardware()
print(profile.summary())
print("Best target:", profile.best_target)
print("GPU count:", len(profile.gpus))
```

### Detect Model Architecture

```python
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector

detector = ArchitectureDetector()
arch = detector.detect("./my-local-model/")   # reads config.json

print("Family:", arch.family)          # e.g. "llama3"
print("Attention:", arch.attention)    # e.g. "gqa"
print("FFN:", arch.ffn_type)           # e.g. "swiglu"
print("Hidden dim:", arch.hidden_size)
```

### Use Native CPU Kernels Directly

```python
import numpy as np
from aether.kernels.native_cpu import get_native_kernels

k = get_native_kernels()
k.ensure_compiled()   # Compile C++ kernels once (cached for future runs)

# Flash Attention-2 (no O(seq^2) memory)
q = np.random.randn(32, 128).astype(np.float32)     # (n_heads, head_dim)
k_cache = np.random.randn(512, 8, 128).astype(np.float32)  # (seq, kv_heads, head_dim)
v_cache = np.random.randn(512, 8, 128).astype(np.float32)
out = k.flash_attn(q, k_cache, v_cache, num_kv_heads=8)
print("Attention output:", out.shape)   # (32, 128)

# INT4 matrix-vector product (2x faster than FP32 decode)
codes = np.random.randint(0, 256, 4096, dtype=np.uint8)
scales = np.ones(256, dtype=np.float32) * 0.01
zp = np.zeros(256, dtype=np.int8)
x = np.random.randn(256).astype(np.float32)
y = k.int4_gemv(codes, scales, zp, x, rows=32, cols=256, block_size=32)
print("INT4 GEMV output:", y.shape)     # (32,)
```

---

## 11. Common Workflows

### Workflow 1: Download → Compile → Serve

```bash
# Step 1: Download and compile a model
aether compile Qwen/Qwen3-0.6B \
  --target cpu_avx512 \
  --precision q4_k_m \
  --output qwen3-0.6b.aeg

# Step 2: Inspect the compiled artifact
aether inspect qwen3-0.6b.aeg/

# Step 3: Serve it
aether serve qwen3-0.6b.aeg/ --port 8080
```

### Workflow 2: Evaluate a Model Before Deploying

```bash
# Run eval gate — blocks deployment if quality drops > 2%
aether eval qwen3-0.6b.aeg/ \
  --suite reasoning \
  --max-regression 0.02 \
  --output eval_report.json

cat eval_report.json
```

### Workflow 3: Push a Compiled Model to Hub

```bash
# Sign the artifact for integrity verification
aether sign qwen3-0.6b.aeg/ --key ./my-signing-key.pem

# Push to Aether Hub CDN
aether hub push qwen3-0.6b.aeg/ --name "my-org/qwen3-0.6b-q4"
```

### Workflow 4: Pull and Run from Hub

```bash
# Pull a pre-compiled artifact (no compilation needed)
aether hub pull my-org/qwen3-0.6b-q4

# Verify its integrity
aether verify my-org--qwen3-0.6b-q4.aeg/

# Run it
aether serve my-org--qwen3-0.6b-q4.aeg/ --port 8080
```

### Workflow 5: Generate Client SDKs

```bash
# Generate an SDK for your language of choice
aether sdk generate --lang typescript --output ./sdk/
aether sdk generate --lang go --output ./sdk/
aether sdk generate --lang rust --output ./sdk/
```

---

## 12. CLI Quick Reference

```
aether [COMMAND] [OPTIONS]
```

| Command | Description |
|---------|-------------|
| `aether compile <model>` | Compile a model to AEG artifact |
| `aether run <path.aeg>` | Run one-shot inference from CLI |
| `aether serve <path.aeg>` | Start OpenAI-compatible HTTP server |
| `aether inspect <path.aeg>` | Show AEG artifact summary |
| `aether bench <path.aeg>` | Run performance benchmark |
| `aether eval <path.aeg>` | Run evaluation gate (CI quality check) |
| `aether hardware` | Show detected hardware profile |
| `aether targets list` | List all supported hardware targets |
| `aether hub push <path.aeg>` | Push artifact to Aether Hub CDN |
| `aether hub pull <model-id>` | Pull artifact from Aether Hub CDN |
| `aether sign <path.aeg>` | Sign artifact with C2PA key |
| `aether verify <path.aeg>` | Verify artifact signature and integrity |
| `aether sdk generate` | Generate TypeScript / Go / Rust SDK |
| `aether --help` | Show help for any command |

Use `aether COMMAND --help` to see all options for any command:

```bash
aether compile --help
aether serve --help
aether bench --help
```

---

## 13. Troubleshooting

### "No compiler found" / Kernels not compiling

Aether falls back to NumPy automatically, but if you want native performance:

```bash
# Linux
sudo apt install build-essential

# macOS
xcode-select --install

# Windows — verify g++ is on PATH
g++ --version
```

Check what compiler Aether detects:
```python
from aether.kernels.native_cpu import detect_toolchain
print(detect_toolchain())
```

---

### "Model not found" / Architecture not detected

If Aether cannot detect the architecture from the `config.json`:

```bash
# Check what architecture Aether sees
python -c "
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
arch = ArchitectureDetector().detect('./your-model-dir/')
print(arch)
"
```

If the family is not supported, check [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md) and open an issue.

---

### Slow inference / NumPy fallback mode

If you see `NativeCPUKernels(numpy-reference, ...)` the C++ kernels didn't compile. Install a compiler (see above) and delete the kernel cache:

```bash
# Clear the kernel cache to force recompilation
python -c "import tempfile, pathlib, shutil; shutil.rmtree(pathlib.Path(tempfile.gettempdir()) / 'aether_kernels', ignore_errors=True); print('Cache cleared')"
```

Then run your model again — it will recompile automatically.

---

### HuggingFace model download requires login (gated models)

Some models (Llama-3, Gemma) require accepting a license on HuggingFace:

```bash
# Log in once
pip install huggingface-hub
huggingface-cli login
# Paste your HuggingFace token when prompted
```

Then compile as normal.

---

### Windows: DLL load failed / OpenMP error

On Windows, the compiled kernel DLL needs the OpenMP runtime (`libgomp.dll`). If you see this error:

1. Make sure you installed **WinLibs GCC** (it includes `libgomp.dll`)
2. Add the MinGW `bin` folder to your PATH:
   ```
   setx PATH "%PATH%;C:\mingw64\bin"
   ```
3. Restart your terminal and try again.

---

### Running tests

```bash
pip install "aether-runtime[test]"

# Run all tests
python -m pytest tests/ -v

# Run only CPU kernel tests (fast, no GPU needed)
python -m pytest tests/unit/test_native_cpu_kernels.py -v

# Run skipping slow / network tests
python -m pytest tests/ -v -m "not slow and not network and not gpu"
```

---

## Next Steps

- 📖 **[README.md](README.md)** — Full feature overview and architecture
- 📋 **[SUPPORTED_MODELS.md](SUPPORTED_MODELS.md)** — All supported model families
- 📦 **[REQUIREMENTS.md](REQUIREMENTS.md)** — Complete dependency reference
- 🔬 **[benchmarks/](benchmarks/)** — Performance benchmark scripts
- 🧪 **[tests/](tests/)** — Test suite
- 💡 **[examples/](examples/)** — Code examples
