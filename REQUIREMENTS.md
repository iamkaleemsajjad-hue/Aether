# Requirements — Aether Runtime

All requirements needed to install and run Aether Runtime. The core runtime runs with **no PyTorch, no CUDA, no GPU** — the optional extras unlock additional backends.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Operating System** | Windows 10, macOS 12, Ubuntu 20.04 | Windows 11, macOS 14, Ubuntu 22.04 |
| **Python** | 3.10 | 3.11 or 3.12 |
| **RAM** | 8 GB | 32 GB+ |
| **Disk** | 5 GB free | 50 GB+ (for large model weights) |
| **CPU** | x86-64 or ARM64 | AVX-512 or Apple M-series |
| **Internet** | Required for first model download | — |

> **GPU is not required.** The Aether CPU engine runs entirely on CPU using native C++ kernels. GPU accelerates large models but is never mandatory.

---

## Core Dependencies

> These are **always installed** automatically with `pip install aether-runtime`. You do not need to install them manually.

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24.0 | Tensor operations, kernel fallback math |
| `tokenizers` | ≥ 0.19.0 | Framework-free AEG tokenization (HuggingFace Tokenizers) |
| `safetensors` | ≥ 0.4.0 | Safe, fast model weight loading |
| `huggingface-hub` | ≥ 0.27.0 | Model downloading and caching |
| `pyyaml` | ≥ 6.0.2 | YAML config parsing |
| `pydantic` | ≥ 2.7.0 | Data validation and AEG-IR schema |
| `rich` | ≥ 13.0.0 | CLI output formatting, progress bars |
| `click` | ≥ 8.1.0 | CLI command framework |
| `toml` | ≥ 0.10.2 | TOML config file parsing |
| `packaging` | ≥ 24.0 | Version parsing and comparison |
| `typing-extensions` | ≥ 4.9.0 | Python 3.10 backport typing utilities |
| `structlog` | ≥ 24.1.0 | Structured JSON logging and telemetry |
| `platformdirs` | ≥ 4.2.0 | Cross-platform cache/config directories |
| `psutil` | ≥ 5.9.0 | CPU/RAM monitoring, hardware profiling |
| `httpx` | ≥ 0.28.0 | Async HTTP client (Hub CDN, model registry) |
| `aiofiles` | ≥ 23.2.0 | Async file I/O for large model downloads |
| `anyio` | ≥ 4.4.0 | Async backend compatibility (asyncio/trio) |
| `tenacity` | ≥ 8.4.0 | Retry logic for network/compilation failures |
| `jsonschema` | ≥ 4.22.0 | AEG manifest validation |
| `tabulate` | ≥ 0.9.0 | Table formatting in CLI output |
| `tqdm` | ≥ 4.66.0 | Download and compilation progress bars |

---

## C++ Compiler (For Native CPU Kernels)

> **Optional but strongly recommended.** Without a C++ compiler the CPU engine falls back to NumPy — correct results, but significantly slower.

Aether compiles its native CPU kernels (INT4-GEMV, FlashAttention-2, fused SwiGLU, RoPE, etc.) at first run using the host C++ compiler. The compiled library is cached and reused on subsequent runs.

### Supported Compilers

| Compiler | Platform | Install |
|----------|----------|---------|
| **GCC / g++** (≥ 9) | Linux, Windows (MinGW) | `sudo apt install build-essential` / WinLibs |
| **Clang / clang++** (≥ 11) | Linux, macOS, Windows | `sudo apt install clang` / Xcode CLI Tools |
| **MSVC** (Visual Studio 2022) | Windows | Visual Studio Build Tools |
| **Apple Clang** | macOS | `xcode-select --install` |

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install -y build-essential
```

**macOS:**
```bash
xcode-select --install
```

**Windows:** Install [WinLibs GCC](https://winlibs.com/) or the [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022).

> To check if a compiler is detected: `python -c "from aether.kernels.native_cpu import detect_toolchain; print(detect_toolchain())"`

### OpenMP (Parallel Kernels)

OpenMP enables multi-core parallelism inside every kernel (RMSNorm, GEMV, FlashAttn, etc.). It comes bundled with GCC and Clang on Linux/macOS. On Windows with MinGW, install **WinLibs** which includes `libgomp`.

---

## Optional Extras

Install any combination of the extras below depending on what you want to do.

---

### `[pytorch]` — PyTorch Model Ingestion

> Needed to load `.pt` / `.pth` / TorchScript model files for compilation.

```bash
pip install "aether-runtime[pytorch]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.5.0 | Load PyTorch checkpoints for compilation |

> The Aether runtime itself **does not require PyTorch**. This extra is only for ingesting PyTorch-format models.

---

### `[transformers-frontend]` — HuggingFace AutoModel Ingestion

> Needed to compile models directly from HuggingFace model names using `AutoConfig` / `AutoTokenizer`.

```bash
pip install "aether-runtime[transformers-frontend]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.5.0 | Required by `transformers` library |
| `transformers` | ≥ 4.48.0 | AutoConfig, AutoTokenizer, model configs |
| `tokenizers` | ≥ 0.19.0 | Fast tokenizer implementation |
| `sentencepiece` | ≥ 0.2.0 | SentencePiece tokenizer (Llama, T5, etc.) |

---

### `[formats]` — GGUF and ONNX Model Files

> Needed to compile models from `.gguf` or `.onnx` files.

```bash
pip install "aether-runtime[formats]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `gguf` | ≥ 0.1.0 | Read GGUF headers and quantized weights |
| `onnx` | ≥ 1.16.0 | Parse ONNX computation graphs |
| `protobuf` | ≥ 5.28.3 | Required by ONNX |

---

### `[server]` — OpenAI-Compatible REST API Server

> Needed to run `aether serve` and expose an OpenAI-compatible HTTP endpoint.

```bash
pip install "aether-runtime[server]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥ 0.115.6 | REST API framework |
| `uvicorn[standard]` | ≥ 0.30.0 | ASGI server |
| `starlette` | ≥ 0.41.3 | HTTP middleware |
| `prometheus-client` | ≥ 0.20.0 | `/metrics` endpoint for Grafana |
| `python-multipart` | ≥ 0.0.20 | Multipart file upload support |
| `grpcio` | ≥ 1.66.0 | gRPC distributed inference |
| `protobuf` | ≥ 5.28.3 | gRPC serialization |

---

### `[distributed]` — Multi-GPU / Multi-Node Execution

> Needed for distributed tensor-parallel inference across multiple GPUs or nodes.

```bash
pip install "aether-runtime[distributed]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `grpcio` | ≥ 1.66.0 | gRPC inference service. Collectives do **not** use it: the CPU ring uses stdlib sockets and the multi-GPU path uses peer-to-peer device copies (`aether.parallelism.p2p_ring`) |
| `protobuf` | ≥ 5.28.3 | gRPC message serialization |

---

### Inference Backend Extras

> Install only the backend that matches your hardware. You can install multiple.

#### `[vllm]` — vLLM Backend (NVIDIA / AMD GPU)

```bash
pip install "aether-runtime[vllm]"
```

| Package | Version | Hardware |
|---------|---------|---------|
| `vllm` | ≥ 0.5.0 | NVIDIA CUDA, AMD ROCm |

#### `[llamacpp]` — llama.cpp Backend (CPU, Metal, CUDA)

```bash
pip install "aether-runtime[llamacpp]"
```

| Package | Version | Hardware |
|---------|---------|---------|
| `llama-cpp-python` | ≥ 0.2.80 | CPU (AVX2/AVX-512), Apple Metal, CUDA |

#### `[trtllm]` — TensorRT-LLM Backend (NVIDIA GPU)

```bash
pip install "aether-runtime[trtllm]"
```

| Package | Version | Hardware |
|---------|---------|---------|
| `tensorrt-llm` | ≥ 0.11.0 | NVIDIA CUDA (sm80+, Ampere and newer) |

#### `[mlx]` — Apple MLX Backend (Apple Silicon only)

```bash
pip install "aether-runtime[mlx]"
```

| Package | Version | Hardware |
|---------|---------|---------|
| `mlx` | ≥ 0.16.0 | Apple M1 / M2 / M3 / M4 / M5 (macOS only) |

#### `[onnxruntime]` — ONNX Runtime Backend (CPU, OpenVINO, QNN)

```bash
pip install "aether-runtime[onnxruntime]"
```

| Package | Version | Hardware |
|---------|---------|---------|
| `onnxruntime` | ≥ 1.18.0 | CPU, Intel OpenVINO NPU, Qualcomm QNN |
| `onnxruntime-gpu` | ≥ 1.18.0 | NVIDIA GPU (Linux only) |

#### `[triton]` — Triton GPU Kernels (Linux + NVIDIA only)

```bash
pip install "aether-runtime[triton]"
```

| Package | Version | Hardware |
|---------|---------|---------|
| `triton` | ≥ 2.3.0 | NVIDIA CUDA (Linux only) |

---

### `[eval]` — Evaluation Benchmarks

> Needed to run `aether eval` with datasets like HellaSwag, MMLU, GSM8K.

```bash
pip install "aether-runtime[eval]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | ≥ 2.19.0 | HuggingFace datasets (eval benchmarks) |

---

### `[benchmark]` — Performance Benchmarking

> Needed to run `aether bench` with full statistical analysis and CSV export.

```bash
pip install "aether-runtime[benchmark]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | ≥ 2.19.0 | Benchmark prompt datasets |
| `pandas` | ≥ 2.2.0 | Results tabulation and CSV export |
| `scipy` | ≥ 1.13.0 | Statistical analysis (P50/P95/P99 latency) |

---

### `[dev]` — Full Developer Install (no PyTorch)

> Recommended for contributors and developers. Includes lint, test, docs, benchmark, server, eval, and distributed — but **no PyTorch**.

```bash
pip install "aether-runtime[dev]"
```

Installs: `[lint]` + `[test]` + `[docs]` + `[benchmark]` + `[distributed]` + `[eval]` + `[formats]` + `[server]`

---

### `[full]` — Everything

> Installs all optional extras including PyTorch and Transformers frontend.

```bash
pip install "aether-runtime[full]"
```

Installs: `[dev]` + `[pytorch]` + `[transformers-frontend]` + `[formats]`

---

### `[test]` — Test Suite

```bash
pip install "aether-runtime[test]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 8.2.0 | Test runner |
| `pytest-asyncio` | ≥ 0.23.0 | Async test support |
| `pytest-cov` | ≥ 5.0.0 | Code coverage reporting |
| `pytest-xdist` | ≥ 3.6.0 | Parallel test execution |
| `responses` | ≥ 0.25.0 | HTTP mock for network tests |
| `requests` | ≥ 2.32.0 | HTTP client for integration tests |
| `openai` | ≥ 1.35.0 | OpenAI-compatible API test client |

---

### `[lint]` — Linting and Type Checking

```bash
pip install "aether-runtime[lint]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `ruff` | ≥ 0.5.0 | Fast Python linter and formatter |
| `mypy` | ≥ 1.10.0 | Static type checker |
| `types-pyyaml` | latest | Type stubs for PyYAML |
| `types-toml` | latest | Type stubs for TOML |
| `types-protobuf` | latest | Type stubs for protobuf |
| `types-tabulate` | latest | Type stubs for tabulate |
| `types-tqdm` | latest | Type stubs for tqdm |

---

### `[docs]` — Documentation Build

```bash
pip install "aether-runtime[docs]"
```

| Package | Version | Purpose |
|---------|---------|---------|
| `sphinx` | ≥ 7.3.0 | Documentation generator |
| `myst-parser` | ≥ 3.0.0 | Markdown support in Sphinx |
| `sphinx-rtd-theme` | ≥ 2.0.0 | Read the Docs HTML theme |
| `sphinx-autodoc-typehints` | ≥ 2.2.0 | Automatic type hint docs |
| `sphinx-click` | ≥ 6.0.0 | CLI command documentation |
| `nbsphinx` | ≥ 0.9.0 | Jupyter notebook integration |

---

## Hardware-Specific Requirements

### NVIDIA GPU (CUDA)

| Requirement | Version |
|-------------|---------|
| NVIDIA Driver | ≥ 525.60 |
| CUDA Toolkit | ≥ 11.8 (12.x recommended) |
| GPU Architecture | Volta (sm70) or newer |

```bash
# Check CUDA availability
nvidia-smi
python -c "from aether.backends.hardware_detector import detect_hardware; print(detect_hardware().summary())"
```

### AMD GPU (ROCm)

| Requirement | Version |
|-------------|---------|
| ROCm | ≥ 5.7 |
| GPU Architecture | RDNA2 (RX 6000) or newer |

### Apple Silicon (Metal)

| Requirement | Version |
|-------------|---------|
| macOS | ≥ 12.0 (Monterey) |
| Apple Silicon | M1 or newer |

### Intel GPU / NPU (OpenVINO)

| Requirement | Version |
|-------------|---------|
| OpenVINO Runtime | ≥ 2024.0 |
| Intel Arc GPU / NPU | Arc A-series or newer |

---

## Quick Reference: Install by Use Case

| Use Case | Command |
|----------|---------|
| **CPU-only inference (no GPU, no PyTorch)** | `pip install aether-runtime` |
| **Compile HuggingFace models** | `pip install "aether-runtime[transformers-frontend]"` |
| **Compile GGUF files** | `pip install "aether-runtime[formats]"` |
| **Run inference server (OpenAI API)** | `pip install "aether-runtime[server]"` |
| **NVIDIA GPU inference** | `pip install "aether-runtime[vllm]"` |
| **Apple Silicon inference** | `pip install "aether-runtime[mlx]"` |
| **Multi-GPU distributed** | `pip install "aether-runtime[distributed]"` |
| **Full developer setup** | `pip install "aether-runtime[full]"` |
| **Run eval benchmarks** | `pip install "aether-runtime[eval]"` |
| **Run test suite** | `pip install "aether-runtime[test]"` |
