# Getting Started with Aether

This guide walks you through installing Aether, compiling your first model, and running it on your hardware.

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 16 GB+ |
| Disk | 5 GB | 50 GB |
| OS | Linux, macOS, Windows | Linux |
| GPU | None (CPU fallback) | NVIDIA Ampere / Apple M2+ |

## Installation

### One-click install (Linux/macOS)

```bash
git clone https://github.com/iamkaleemsajjad-hue/Aether
cd aether-runtime
./scripts/install.sh        # auto-detects CUDA/ROCm/MPS
# or with options:
./scripts/install.sh --dev  # include dev/test dependencies
```

### One-click install (Windows)

```powershell
git clone https://github.com/iamkaleemsajjad-hue/Aether
cd aether-runtime
.\scripts\install.ps1        # auto-detects CUDA
.\scripts\install.ps1 -Dev   # include dev/test dependencies
```

### Manual install

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# Install base package
pip install -e .

# Install with CUDA extras (NVIDIA)
pip install -e .[cuda]

# Install with Apple Silicon extras
pip install -e .[apple]

# Install all dev tools
pip install -e .[dev]
```

### Install from PyPI

```bash
pip install aether-runtime

# With extras:
pip install aether-runtime[cuda]    # NVIDIA
pip install aether-runtime[mlx]     # Apple Silicon
pip install aether-runtime[dev]     # Development
```

## Verify Installation

```bash
aether version
```

Run the environment check to verify all components:

```bash
python scripts/check_env.py
```

Expected output:
```
[OK] Python 3.11.x
[OK] PyTorch 2.5.x
[OK] aether-runtime 1.2.6.alpha
[OK] SafeTensors
[OK] GGUF reader
[OK] ONNX
[OK] FastAPI
[OK] gRPC
Hardware: CPU (AVX512)
```

## Compile Your First Model

### From a local HuggingFace checkpoint

```bash
# Download and compile Qwen3-0.6B
aether compile Qwen/Qwen3-0.6B

# Show what would happen without compiling (dry run)
aether compile Qwen/Qwen3-0.6B --dry-run

# Compile with explicit target
aether compile Qwen/Qwen3-0.6B --target cpu

# Compile with stricter quality budget (less aggressive quantization)
aether compile Qwen/Qwen3-0.6B --quality-budget 0.01
```

Aether will:
1. Download the model weights from HuggingFace.
2. Detect the architecture (family, layers, attention heads, etc.).
3. Trace the graph and produce AEG-IR.
4. Run all 22 optimizer passes.
5. Select backend plans for your hardware.
6. Save a `.aeg` artifact to `~/.cache/aether/models/`.

### From a local path

```bash
aether compile ./my-model-dir/
aether compile ./my-model.gguf
aether compile ./my-model.onnx
```

### Inspect the compiled artifact

```bash
aether info Qwen/Qwen3-0.6B
aether graph Qwen/Qwen3-0.6B     # print the AEG-IR graph
```

## Run a Model

### CLI

```bash
aether run Qwen/Qwen3-0.6B --prompt "What is the AEG format?" --max-tokens 128

# Stream tokens as they are generated
aether run Qwen/Qwen3-0.6B --prompt "Tell me about AI" --stream

# Chat mode
aether run Qwen/Qwen3-0.6B --chat
```

### Python SDK

```python
from aether import Runtime, RuntimeConfig

# Create runtime with defaults (auto-selects best backend)
rt = Runtime()

# Simple text generation
response = rt.generate(
    model_id="Qwen/Qwen3-0.6B",
    prompt="Explain the AEG format in one paragraph.",
    max_tokens=128,
    temperature=0.7,
)
print(response.text)
print(f"Throughput: {response.metrics.throughput_tps:.1f} tok/s")
print(f"TTFT: {response.metrics.ttft_ms:.1f} ms")

# Chat completion
response = rt.chat(
    model_id="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is speculative decoding?"},
    ],
    max_tokens=256,
)
print(response.text)

# Streaming
for chunk in rt.generate(
    model_id="Qwen/Qwen3-0.6B",
    prompt="Count from 1 to 10.",
    stream=True,
):
    print(chunk.text, end="", flush=True)
print()

# Embeddings
embeddings = rt.embed("Qwen/Qwen3-0.6B", ["Hello world", "Goodbye world"])
print(f"Embedding dim: {len(embeddings[0])}")
```

### Compile Programmatically

```python
from aether import Compiler, CompilerConfig

config = CompilerConfig(
    quality_budget=0.02,          # Max 2% perplexity degradation
    calibration_dataset="wikitext-2",
    targets=["auto"],             # Auto-detect hardware
    optimization_level=2,
    enable_fusion=True,
    enable_sensitivity=True,
    enable_precision_assignment=True,
    enable_kv_cache_structuring=True,
    enable_moe_routing=True,
    enable_parallelism_discovery=True,
)

compiler = Compiler(config)

# Show compilation plan
plan = compiler.plan("Qwen/Qwen3-0.6B")
print(plan)

# Compile
aeg = compiler.compile("Qwen/Qwen3-0.6B")
print(f"Compiled: {aeg.path}")
print(f"Size: {aeg.size_mb:.1f} MB")
```

## Serve a Model

### Start the OpenAI-compatible server

```bash
aether serve Qwen/Qwen3-0.6B --port 11434
```

### Use any OpenAI SDK client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="aether")

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128,
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### REST API (direct HTTP)

```bash
# Text generation
curl http://localhost:11434/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello!", "max_tokens": 50}'

# Chat
curl http://localhost:11434/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}]}'

# Health check
curl http://localhost:11434/v1/health
```

## Benchmark a Model

```bash
# Quick benchmark
aether bench Qwen/Qwen3-0.6B

# Compare two models
aether bench Qwen/Qwen3-0.6B --compare meta-llama/Llama-3.2-1B

# Run standard evaluation suite
aether eval Qwen/Qwen3-0.6B --dataset hellaswag
```

## Supported Model Formats

| Format | Command |
|---|---|
| SafeTensors (HuggingFace) | `aether compile Qwen/Qwen3-0.6B` |
| GGUF (llama.cpp) | `aether compile ./model.gguf` |
| ONNX | `aether compile ./model.onnx` |
| PyTorch `.pt` / `.bin` | `aether compile ./model.pt` |
| MLX (Apple) | `aether compile ./model.mlx` |
| VLM (LLaVA, Qwen-VL…) | `aether compile ./llava-model-dir/` |
| Video (Video-LLaMA2…) | `aether compile ./video-model-dir/` |
| MoE (Mixtral, Qwen-MoE…) | `aether compile ./mixtral-dir/` |
| MLA (DeepSeek V2/V3/R1) | `aether compile ./deepseek-dir/` |
| SSM (Mamba, Jamba…) | `aether compile ./mamba-dir/` |

## Next Steps

- [Architecture overview](architecture.md) — how the 5-stage compiler works
- [AEG format](aeg-format.md) — the Aether Execution Graph format
- [Optimizer passes](optimizer-passes.md) — all 22 optimization passes
- [API reference](api-reference.md) — complete SDK / REST / CLI reference
- [Performance benchmarking](performance-benchmarking.md) — how to benchmark
