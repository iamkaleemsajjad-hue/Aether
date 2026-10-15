# Getting started with Aether Runtime

For separate execution workers, persistent sessions, measured precision and
explicit draft models, see the [runtime integration guide](docs/runtime-feature-integration.md).
CLI generation/serving accepts validated RuntimeConfig JSON through global
`aether --runtime-config runtime.json ...`.

Aether compiles a model into an Aether Execution Graph (AEG): a directory
containing the graph, integrity-checked weights, tokenizer, precision map, and
execution contracts. The runtime loads that artifact and selects a compatible
execution path. Python 3.10 or newer is required.

## Install

```bash
python -m pip install aether-runtime
```

For a source checkout:

```bash
git clone https://github.com/iamkaleemsajjad-hue/Aether.git
cd Aether
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install -e "."
```

The base installation compiles supported SafeTensors checkpoints and executes
CPU AEGs without PyTorch or Transformers. A C++ compiler enables native CPU
kernels; without one, the NumPy reference path remains functional. The loader
can also use an authenticated native library packaged in the artifact when
that library is compatible with the host.

Install the extras required by your workflow:

| Extra | Purpose |
|---|---|
| `pytorch` | Optional portable CUDA/ROCm/MPS AEG executor and PyTorch ingestion |
| `transformers-frontend` | Hugging Face Transformers frontend |
| `server` | REST and gRPC serving dependencies |
| `formats` | GGUF and ONNX ingestion dependencies |
| `test` | Offline core, API, and CLI tests; includes pytest-timeout and FastAPI |
| `full` | Development dependencies plus PyTorch and Transformers |

For example, `python -m pip install "aether-runtime[server]"` enables serving.
Accelerator execution also requires a device-compatible installation of the
selected optional backend and its vendor runtime.

## Detect hardware

```bash
aether version
aether doctor --json
aether hardware detect --json
aether kernels
```

Hardware detection reports the host's capabilities. Execution also depends on
the artifact and installed backend. `cpu_avx2`, `cpu_avx512`, and `cpu_neon`
are CPU targets; use `cpu_neon` on ARM. Accelerator targets are reported by
`hardware detect`; use the identifier reported for your device.

## Compile and inspect

This public small model is a practical starting point:

```bash
aether compile HuggingFaceTB/SmolLM2-135M-Instruct --target cpu_avx2 --output smollm2.aeg
aether inspect smollm2.aeg
aether graph smollm2.aeg
```

You can also compile a local checkpoint:

```bash
aether compile ./my-model-directory --target cpu_avx2 --output local-model.aeg
```

Use `--dry-run` to inspect the compilation plan and `--overwrite` when you
intend to replace an existing output. Set the global cache option before the
command: `aether --cache-dir ./aether-cache compile ...`.

Compilation uses automatic precision assignment by default. To explicitly
request uniform layer precision:

```bash
aether compile ./my-model-directory --target cpu_avx2 --precision q4_k_m --output quantized.aeg
```

`--precision` accepts the formats shown by `aether compile --help`, including
`q4_k_m`, `q8_0`, `bf16`, `fp16`, and `fp32`, case-insensitively. Embeddings
and the output head retain BF16. Lossy precision is an explicit choice and
requires model-quality evaluation; a quality-budget setting alone is not
evidence that a task or perplexity gate has passed.

Repeat `--target` to include multiple variants:

```bash
aether compile ./my-model-directory --target cpu_avx2 --target cuda_sm80 --output portable.aeg
```

Choose your actual accelerator target. The manifest distinguishes executable,
portable, and planning-only variants. A planning-only target requires an
implementation before it can execute.

## Generate

```bash
aether run smollm2.aeg --prompt "Explain quantum entanglement." --max-tokens 128
aether run smollm2.aeg --prompt "Explain quantum entanglement." --max-tokens 128 --stream
```

The CLI applies a packaged chat template to instruction-tuned models.
`--raw` requests an untemplated continuation; `--chat-template` can supply a
template string or file when a checkpoint does not package its template.

Automatic selection prefers a usable GPU authorized by the artifact's executable
or portable contracts, then an authorized CPU path. A CPU-only contract runs on
CPU on GPU-equipped hosts. A portable PyTorch contract can authorize accelerator
execution. The reported `kernel_target` identifies the selected execution target.

Use `AETHER_BACKEND` and `AETHER_EXECUTION_DEVICES` for explicit constraints.
For example, `AETHER_BACKEND=aether_cpu` and `AETHER_EXECUTION_DEVICES=cpu`
request CPU execution. In PowerShell, set these with
`$env:AETHER_BACKEND = "aether_cpu"` and
`$env:AETHER_EXECUTION_DEVICES = "cpu"`. An explicit unavailable accelerator
request fails instead of being silently redirected.

## Python API

```python
from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig

artifact = Compiler(CompilerConfig(targets=["cpu_avx2"])).compile(
    "./my-model-directory", output_path="model.aeg"
)
print(artifact.root)

runtime = Runtime(RuntimeConfig(hf_offline=True))
response = runtime.generate(
    str(artifact.root), "Explain quantum entanglement.",
    max_tokens=128, apply_chat_template=True,
)
print(response.text)
print(response.metrics.to_dict())

for delta in runtime.generate_stream(
    str(artifact.root), "Tell me a story.", max_tokens=128,
    apply_chat_template=True,
):
    print(delta, end="", flush=True)
```

For direct backend access:

```python
from aether.backends import GenerationRequest, get_backend

backend = get_backend("aether_cpu")
backend.load_model("local-model", aeg_path="model.aeg")
result = backend.generate(GenerationRequest(
    model_id="local-model", prompt="Hello", max_tokens=64,
    extra={"apply_chat_template": True},
))
print(result.text)
```

`get_backend` raises `BackendNotAvailableError` for an unavailable backend.
Use `BackendRegistry()` when you need custom plugins or isolated instances.

## Benchmark

`bench` executes one greedy generation and reports backend, target, token
counts, elapsed time, throughput, and the backend's time-to-first-token:

```bash
aether bench smollm2.aeg --max-tokens 128 --output baseline.json
aether bench smollm2.aeg --max-tokens 128 --compare baseline.json --json
```

`--compare` accepts a previous `bench` JSON result or an installed backend
name. Compare identical model, prompt, output length, hardware, and precision
for meaningful measurements. A comparison backend must support the artifact.

For repeated measurements and warmup:

```bash
aether benchmark smollm2.aeg --runs 5 --warmup 1 --max-tokens 128 --output report.json
```

Both CLI benchmark commands use backend token counts and first-token timing.
The repeated benchmark reports TBT as unavailable when token timing is absent;
it does not substitute total latency divided by word count. Neither command
promises the same performance across hardware or model sizes. See
[performance benchmarking](docs/performance-benchmarking.md).

## Serve

```bash
python -m pip install "aether-runtime[server]"
aether serve smollm2.aeg --host 127.0.0.1 --port 8080
```

The model is loaded before the server accepts requests. For example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="aether")
response = client.chat.completions.create(
    model="smollm2.aeg",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

## Test a checkout

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The test extra contains the dependencies used by offline core and API tests.
Tests for optional frameworks, formats, or real hardware skip when their
dependencies are absent. Install `.[full]` to exercise PyTorch/Transformers
paths too. Network-marked tests require `AETHER_RUN_NETWORK_TESTS=1`.

For the feedback regression checks:

```bash
python -m pytest tests/unit/test_runtime_artifact_selection.py tests/unit/test_feedback_cli_contracts.py
```

For model and execution scope, see [supported models](SUPPORTED_MODELS.md),
[runtime selection](docs/runtime.md), and
[research-backed execution contracts](docs/research-backed-runtime.md).
