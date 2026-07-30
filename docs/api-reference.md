# Aether API Reference

## Python SDK

### `Runtime`

The main inference API.

```python
from aether import Runtime, RuntimeConfig

rt = Runtime(config)
```

#### `Runtime.generate(model_id, prompt, **kwargs)`

Generate text from a prompt.

Args:

- `model_id` (str): Model identifier or local AEG path.
- `prompt` (str): Text prompt.
- `max_tokens` (int): Maximum tokens to generate.
- `temperature` (float): Sampling temperature.
- `top_p` (float): Top-p sampling parameter.
- `top_k` (int): Top-k sampling parameter.
- `stream` (bool): Whether to stream output.
- `stop` (list[str]): Stop sequences.

Returns: `GenerationResponse` with `text`, `usage`, `metrics`, and `finish_reason`.

#### `Runtime.chat(model_id, messages, **kwargs)`

Chat completion with a list of messages.

Args:

- `model_id` (str): Model identifier.
- `messages` (list[dict]): List of `{role, content}` messages.

Returns: `GenerationResponse`.

#### `Runtime.embed(model_id, input)`

Generate embeddings for a list of texts.

Returns: `list[list[float]]`.

#### `Runtime.rerank(model_id, query, documents)`

Rerank documents for a query.

Returns: `list[dict]` with `index`, `document`, and `score`.

#### `Runtime.transcribe(model_id, audio, language=None)`

Transcribe an audio file.

Returns: `str` transcript.

#### `Runtime.benchmark(model_id, prompt, max_tokens)`

Run a simple benchmark.

Returns: `dict` with throughput, TTFT, and token counts.

#### `Runtime.pull(model_id)`

Download and compile a model to the local AEG cache.

#### `Runtime.list()`

List compiled models in cache.

#### `Runtime.info(model_id)`

Return model metadata and precision map.

#### `Runtime.remove(model_id)`

Remove a model from cache.

### `Compiler`

```python
from aether import Compiler, CompilerConfig

compiler = Compiler(config)
```

#### `Compiler.plan(model_id, hardware=None)`

Dry-run compilation plan. Returns `CompilationPlan`.

#### `Compiler.compile(model_id, targets=None, quality_budget=None, calibration_dataset=None, output_path=None)`

Compile a model into an AEG artifact. Returns `AEGPackage`.

#### `Compiler.quality_report(aeg)`

Generate a quality report from a compiled AEG. Returns `QualityReport`.

### `CompilerConfig`

```python
from aether import CompilerConfig

config = CompilerConfig(
    quality_budget=0.02,
    calibration_dataset="wikitext-2",
    targets=["auto"],
    optimization_level=2,
    enable_fusion=True,
    enable_sensitivity=True,
    enable_precision_assignment=True,
    enable_kv_cache_structuring=True,
    enable_moe_routing=True,
    enable_parallelism_discovery=True,
    upload_kernels=False,
)
```

### `RuntimeConfig`

```python
from aether import RuntimeConfig

config = RuntimeConfig(
    optimize_for="latency",
    speculative_decoding=True,
    prefill_chunk_size=2048,
    max_batch_size=256,
    kv_cache_dtype="fp8",
    dynamic_precision=True,
)
```

## REST API

Base URL: `http://localhost:11434/v1`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Text completion |
| POST | `/chat` | Chat completion |
| POST | `/embeddings` | Embeddings |
| POST | `/rerank` | Reranking |
| POST | `/transcribe` | Audio transcription |
| POST | `/compile` | Compile model (async) |
| GET | `/compile/{job_id}` | Compilation status |
| GET | `/models` | List models |
| POST | `/models/pull` | Pull model |
| GET | `/models/{name}` | Model info |
| DELETE | `/models/{name}` | Remove model |
| GET | `/models/{name}/graph` | AEG-IR graph |
| GET | `/hardware` | Hardware fingerprint |
| GET | `/kernels` | Active targets |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health` | Health check |

## OpenAI Compatibility

Aether's `/v1/chat` and `/v1/generate` endpoints are compatible with OpenAI SDK v1+:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="aether")
client.chat.completions.create(model="Qwen/Qwen3-8B", messages=[...])
```

## CLI Reference

```bash
aether compile <model> [--target ...] [--quality-budget] [--upload]
aether pull <model> [--compile-local]
aether run <model> [--prompt] [--max-tokens] [--stream]
aether serve <model> [--port] [--host]
aether bench <model> [--compare]
aether info <model>
aether graph <model>
aether list
aether rm <model>
aether hw
aether kernels
aether logs [--follow]
aether version
```
