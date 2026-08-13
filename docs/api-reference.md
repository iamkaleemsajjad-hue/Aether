# Aether API Reference

Complete reference for the Aether Runtime Python SDK, REST API, CLI, and gRPC interface.

---

## Python SDK

### `Runtime`

The main inference API. Manages model loading, hardware backend selection, and generation.

```python
from aether import Runtime, RuntimeConfig

rt = Runtime()                          # Auto-detect hardware
rt = Runtime(RuntimeConfig(...))        # With explicit config
```

#### `Runtime.generate(model_id, prompt, **kwargs) → GenerationResponse`

Generate text from a prompt.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | required | HuggingFace ID, local path, or `.aeg` file |
| `prompt` | `str` | required | Input text |
| `max_tokens` | `int` | `512` | Maximum tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0 = greedy) |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `top_k` | `int` | `0` | Top-k sampling (0 = disabled) |
| `stream` | `bool` | `False` | Yield `StreamChunk` objects incrementally |
| `stop` | `list[str]` | `[]` | Stop sequences |
| `adapter_id` | `str \| None` | `None` | LoRA adapter ID from AEG |
| `grammar` | `str \| None` | `None` | GBNF grammar string for constrained generation |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

**Returns:** `GenerationResponse` with fields:
- `.text` (str): Generated text
- `.usage` (dict): `{prompt_tokens, completion_tokens, total_tokens}`
- `.metrics` (GenerationMetrics): `.ttft_ms`, `.throughput_tps`, `.total_ms`
- `.finish_reason` (str): `"stop"`, `"length"`, or `"error"`

**Streaming example:**
```python
for chunk in rt.generate("model.aeg", "Tell me a story", stream=True):
    print(chunk.text, end="", flush=True)
```

#### `Runtime.chat(model_id, messages, **kwargs) → GenerationResponse`

Chat completion with role-formatted messages.

| Parameter | Type | Description |
|---|---|---|
| `model_id` | `str` | Model identifier |
| `messages` | `list[dict]` | `[{"role": "user"/"system"/"assistant", "content": str}]` |
| `**kwargs` | | Same as `generate()` |

#### `Runtime.embed(model_id, input) → list[list[float]]`

Generate embeddings for one or more texts.

```python
vecs = rt.embed("nomic-ai/nomic-embed-text-v1.5", ["Hello", "World"])
# → [[0.12, -0.34, ...], [0.56, 0.78, ...]]
```

#### `Runtime.rerank(model_id, query, documents) → list[dict]`

Rerank documents by relevance to query.

```python
results = rt.rerank("BAAI/bge-reranker-v2-m3", "AI safety", ["doc1", "doc2"])
# → [{"index": 0, "document": "doc1", "score": 0.92}, ...]
```

#### `Runtime.transcribe(model_id, audio, language=None) → str`

Transcribe an audio file using a Whisper-family model.

```python
transcript = rt.transcribe("openai/whisper-large-v3", "audio.wav", language="en")
```

#### `Runtime.benchmark(model_id, prompt, max_tokens) → dict`

Run a simple latency/throughput benchmark.

```python
stats = rt.benchmark("model.aeg", "Hello", 128)
# → {"ttft_ms": 42.1, "throughput_tps": 3241.5, "total_ms": 395.2, "tokens": 128}
```

#### `Runtime.merge(model_a, model_b, **kwargs) → AEGPackage`

Merge two AEG models using SLERP or task-vector interpolation (Pass 12).

```python
merged = rt.merge("instruct.aeg", "math-tuned.aeg", alpha=0.5, method="slerp")
```

#### `Runtime.pull(model_id) → AEGPackage`

Download and compile a model from HuggingFace or the Aether Hub to local cache.

#### `Runtime.list() → list[dict]`

List all compiled models in the local AEG cache.

#### `Runtime.info(model_id) → dict`

Return model metadata, precision map, and AEG format version.

#### `Runtime.remove(model_id) → None`

Remove a model from the local cache.

---

### `Compiler`

Compiles AI models into AEG artifacts through the 5-stage pipeline.

```python
from aether import Compiler, CompilerConfig

compiler = Compiler(CompilerConfig(quality_budget=0.02))
```

#### `Compiler.plan(model_id, hardware=None) → CompilationPlan`

Dry-run — returns what the compiler would do without writing any files.

```python
plan = compiler.plan("Qwen/Qwen3-0.6B")
print(plan.target_hardware)
print(plan.estimated_size_mb)
print(plan.passes)
```

#### `Compiler.compile(model_id, targets=None, quality_budget=None, calibration_dataset=None, output_path=None) → AEGPackage`

Full compilation pipeline. Returns a handle to the resulting `.aeg` artifact.

```python
aeg = compiler.compile(
    "Qwen/Qwen3-0.6B",
    targets=["cpu_avx512"],
    quality_budget=0.02,
    calibration_dataset="wikitext-2",
    output_path="./qwen3.aeg",
)
```

#### `Compiler.quality_report(aeg) → QualityReport`

Evaluate quality degradation of a compiled model vs the original.

---

### `CompilerConfig`

```python
from aether import CompilerConfig

config = CompilerConfig(
    quality_budget=0.02,             # Max acceptable perplexity delta (0.02 = 2%)
    calibration_dataset="wikitext-2",# Dataset for sensitivity analysis
    calibration_samples=512,         # Number of calibration samples
    targets=["auto"],                # Hardware targets: "auto", "cpu", "cuda", ...
    optimization_level=2,            # 0=none, 1=basic, 2=full, 3=aggressive
    # Pass enable flags
    enable_fusion=True,
    enable_sensitivity=True,
    enable_precision_assignment=True,
    enable_kv_cache_structuring=True,
    enable_moe_routing=True,
    enable_parallelism_discovery=True,
    enable_reasoning_graph=False,    # Pass 7: requires reasoning models
    enable_sparse_attention=False,   # Pass 8: requires long-context calibration
    enable_pruning=False,            # Pass 9: requires weight tensors
    # Upload compiled kernels to Hub
    upload_kernels=False,
    hub_url="https://hub.aether.dev",
)
```

---

### `RuntimeConfig`

```python
from aether import RuntimeConfig

config = RuntimeConfig(
    optimize_for="latency",          # "latency" | "throughput" | "memory"
    speculative_decoding=True,       # Enable EAGLE-3 (requires draft model)
    prefill_chunk_size=2048,         # Chunked prefill size
    max_batch_size=256,              # Max concurrent batch size
    kv_cache_dtype="fp8",            # KV cache precision
    dynamic_precision=True,          # R1 dynamic precision under memory pressure
    # SLO enforcement
    slo_ttft_ms=500.0,               # R4 TTFT budget
    slo_throughput_tps=100.0,        # R4 throughput floor
    # Grammar / constrained decoding
    enable_grammar=True,             # R3 FSM-based constrained decoding
    # MCP tool dispatch
    enable_mcp=True,                 # R6 Model Context Protocol
    mcp_servers=[],                  # List of MCP server configs
    # Green compute
    enable_green=False,              # R7 energy/carbon recording
    target_carbon_intensity=None,    # gCO2/kWh threshold
    # Observability
    otlp_endpoint=None,             # OTLP collector URL (None = file export)
    metrics_port=9090,               # Prometheus metrics port
)
```

---

### `AEGPackage`

Handle to a compiled `.aeg` artifact.

```python
from aether.aeg import AEGPackage

pkg = AEGPackage.load("model.aeg")
print(pkg.manifest)          # Full manifest dict
print(pkg.format_version)    # "AEG/1.1", "AEG/2.0", or "AEG/3.0"
print(pkg.architecture)      # ModelArchitecture dataclass
print(pkg.precision_map)     # {layer_name: dtype}
print(pkg.aeg_path)          # Path to .aeg file
print(pkg.size_mb)           # Artifact size in MB

# Integrity verification
pkg.verify()                 # Raises on hash mismatch
```

---

## REST API

**Base URL:** `http://localhost:11434/v1`

All endpoints accept and return `application/json`. Errors use standard HTTP status codes with `{"error": "message"}` body.

### Generation

#### `POST /v1/generate`

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "What is quantum entanglement?",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["\n\n"]
}
```

Response:
```json
{
  "text": "Quantum entanglement is...",
  "usage": {"prompt_tokens": 8, "completion_tokens": 64, "total_tokens": 72},
  "finish_reason": "stop",
  "metrics": {"ttft_ms": 42.1, "throughput_tps": 3241.5, "total_ms": 395.2}
}
```

#### `POST /v1/chat`

OpenAI-compatible chat completion:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AEG?"}
  ],
  "max_tokens": 128,
  "stream": false
}
```

#### `POST /v1/embeddings`

```json
{"model": "nomic-ai/nomic-embed-text-v1.5", "input": ["Hello", "World"]}
```

#### `POST /v1/eval`

Run benchmark evaluation:

```json
{"model": "model.aeg", "dataset": "hellaswag", "num_samples": 100}
```

### Model Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List compiled models |
| `POST` | `/v1/models/pull` | Pull and compile a model |
| `GET` | `/v1/models/{name}` | Model metadata |
| `DELETE` | `/v1/models/{name}` | Remove model |
| `GET` | `/v1/models/{name}/graph` | AEG-IR graph |

### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Health check + backend status |
| `GET` | `/v1/hardware` | Hardware fingerprint |
| `GET` | `/v1/kernels` | Active kernel targets |
| `GET` | `/v1/metrics` | Prometheus-format metrics |
| `POST` | `/v1/compile` | Compile model (async, returns job ID) |
| `GET` | `/v1/compile/{job_id}` | Compilation job status |

---

## OpenAI Compatibility

Aether's `/v1/chat` endpoint is compatible with the OpenAI Python SDK v1+:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="aether")

# Chat
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128,
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Embeddings
result = client.embeddings.create(
    model="nomic-ai/nomic-embed-text-v1.5",
    input=["Hello", "World"]
)
print(result.data[0].embedding[:5])
```

---

## CLI Reference

```bash
# Compilation
aether compile <model> [--target TARGET] [--quality-budget FLOAT] [--dry-run]
aether compile <model> [--calibration-dataset STR] [--output PATH]
aether compile <model> [--upload] [--hub-url URL]

# Model management
aether pull <model> [--compile-local]
aether list
aether info <model>
aether graph <model>
aether rm <model>

# Inference
aether run <model> [--prompt TEXT] [--max-tokens N] [--temperature F] [--stream]
aether run <model> [--chat]

# Serving
aether serve <model> [--port N] [--host STR] [--workers N]

# Benchmarking & evaluation
aether bench <model> [--compare OTHER_MODEL] [--requests N] [--concurrency N]
aether eval <model> --dataset DATASET [--num-samples N]

# Hardware & kernels
aether hw                              # Show hardware fingerprint
aether kernels                         # List kernel targets
aether kernel generate <target> <op>  # Generate a native kernel

# Hub
aether hub push <model>
aether hub pull <model>
aether hub search <query>
aether hub info <model_id>

# Observability
aether trace <model> [--prompt TEXT]
aether mla-stats <model>
aether kv-share status
aether slo-status
aether logs [--follow]

# Safety
aether safety check --prompt TEXT
aether safety policy show

# Distributed
aether multi-agent status
aether kv transfer-stats

# Utilities
aether version
aether --help
```

---

## gRPC API

The Aether gRPC server runs on port `50051` by default.

### Service Definition

```protobuf
service AetherService {
  rpc Health (HealthRequest) returns (HealthResponse);
  rpc Generate (GenerateRequest) returns (GenerateResponse);
  rpc GenerateStream (GenerateRequest) returns (stream StreamChunk);
  rpc Chat (ChatRequest) returns (GenerateResponse);
  rpc Embed (EmbedRequest) returns (EmbedResponse);
}
```

### Python gRPC Client

```python
import grpc
from aether.grpc import aether_pb2, aether_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = aether_pb2_grpc.AetherServiceStub(channel)

# Health check
resp = stub.Health(aether_pb2.HealthRequest())
print(resp.status)      # "ok"
print(resp.backend)     # "cpu_native"

# Generation
resp = stub.Generate(aether_pb2.GenerateRequest(
    model_id="model.aeg",
    prompt="Hello!",
    max_tokens=64,
    temperature=0.7,
))
print(resp.text)

# Streaming
for chunk in stub.GenerateStream(aether_pb2.GenerateRequest(
    model_id="model.aeg",
    prompt="Tell me a story",
    max_tokens=256,
)):
    print(chunk.text, end="", flush=True)
```

---

## Python SDK — Advanced Usage

### Specialised Loaders (Stage 1)

```python
from aether.compiler.stage1_ingestion import (
    MLALoader, MoELoader, VideoModelLoader,
    is_mla_model, is_moe_model,
)

# Check model type from config
import json
config = json.loads(open("model/config.json").read())
print(is_mla_model(config))    # True for DeepSeek V2/V3/R1
print(is_moe_model(config))    # True for Mixtral, Qwen-MoE, etc.

# Load with specialised loader
result = MLALoader("deepseek-v2/").load()
print(result["architecture"].kv_compression_ratio)   # ~5.3x
print(result["kv_compression_ratio"])
```

### Distributed Execution

```python
from aether.parallelism.distributed import (
    DistributedConfig, DistributedFleetManager,
    CollectiveBackend, ParallelismStrategy,
)

config = DistributedConfig(
    strategy=ParallelismStrategy.TENSOR_PIPELINE,
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    data_parallel_size=1,
    master_addr="localhost",
    master_port=29500,
)

fleet = DistributedFleetManager(config)
fleet.launch(num_workers=8)
```

### Evaluation

```python
from aether.observability.evaluators import (
    create_evaluator, run_standard_suite,
    JsonlBenchmarkEvaluator, DatasetBenchmarkEvaluator,
)

# Run a standard benchmark
from aether import Runtime
rt = Runtime()
model_fn = lambda prompt: rt.generate("model.aeg", prompt, max_tokens=64).text

# Single benchmark
ev = create_evaluator("hellaswag", model_fn, num_samples=100)
result = ev.run(verbose=True)
print(f"HellaSwag: {result.score:.3f}")

# Full suite
results = run_standard_suite(model_fn, benchmarks=["hellaswag", "mmlu", "gsm8k"])

# Custom JSONL dataset
ev = JsonlBenchmarkEvaluator(model_fn, data_path="my_benchmark.jsonl")
result = ev.run()

# Standard format files
ev = DatasetBenchmarkEvaluator(model_fn, benchmark="hellaswag", data_path="hellaswag_val.jsonl")
result = ev.run()
```

### Hub Client

```python
from aether.hub.client import HubClient

client = HubClient(hub_url="https://hub.aether.dev", api_key="your-api-key")

# Upload
client.push("model.aeg", model_id="myorg/my-model", description="My compiled model")

# Download
client.pull("myorg/my-model", output_path="downloaded.aeg")

# Search
results = client.search("qwen llm quantized", limit=10)
for r in results:
    print(r["model_id"], r["size_mb"])

# Model info
info = client.info("myorg/my-model")
print(info["format_version"])
print(info["architecture"])
```

### Safety

```python
from aether.safety.production_safety import ProductionSafetyEngine, SafetyConfig

safety = ProductionSafetyEngine(SafetyConfig(
    enable_prompt_guard=True,
    enable_output_filter=True,
    enable_watermarking=True,
    jailbreak_sensitivity=0.7,
))

# Check a prompt
result = safety.check_prompt("Tell me how to make explosives")
print(result.blocked)         # True
print(result.category)        # "harmful_instructions"
print(result.confidence)      # 0.98

# Check an output
result = safety.check_output("The password is: 12345", context_prompt="...")
print(result.blocked)
```
