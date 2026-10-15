# Aether API reference

The public package type stub re-exports the typed implementation classes, so
constructor options and methods follow the actual runtime API. Maintainers can
regenerate it with `python scripts/generate_stubs.py`; CI checks it with
`python scripts/generate_stubs.py --check`.

Execution features now have concrete Runtime interfaces:
`adjust_precision(model_id, calibration_tokens, restore=False)`,
`persist_session(model_id, session_id, tier="disk")`,
`restore_session(model_id, session_id, key)`,
`configure_cache_hub(client, namespace)`, and `close()`/context-manager lifecycle.
`RuntimeConfig` includes explicit draft, phase-worker placement, bounded queues,
calibration limits and cache-store path. See the
[tested runtime integration contracts](runtime-feature-integration.md) for
examples and supported operator/cache combinations. The CLI accepts these
validated fields with global `--runtime-config runtime.json`.

This reference covers compilation, artifacts, execution, and serving. Backend
capabilities determine which model graphs and tasks can execute. See
[getting started](../GETTING_STARTED.md) and [runtime selection](runtime.md).

## Compiler

```python
from aether import Compiler, CompilerConfig

compiler = Compiler(CompilerConfig(
    targets=["cpu_avx2"],
    precision=None,  # Explicit uniform layer format, or automatic assignment
    quality_budget=0.02,
    calibration_dataset="wikitext-2",
    max_calibration_samples=512,
))
plan = compiler.plan("./my-model-directory")
print(plan.to_dict())
artifact = compiler.compile("./my-model-directory", output_path="model.aeg")
print(artifact.root)
print(compiler.quality_report(artifact).to_dict())
```

| Method | Return | Purpose |
|---|---|---|
| `plan(model, *, hardware=None)` | `CompilationPlan` | Inspect and estimate compilation; check the plan's errors |
| `compile(model, *, targets=None, quality_budget=None, calibration_dataset=None, output_path=None)` | `AEGPackage` | Compile and package graph, weights, and execution contracts |
| `quality_report(artifact)` | `QualityReport` | Recorded compilation and validation evidence |

`CompilerConfig.precision` accepts the formats advertised by
`aether compile --help`, case-insensitively. Explicit precision selects uniform
layer assignment. Embeddings and the output head retain BF16. A quality budget
does not establish that a measured quality gate passed; inspect validation
evidence. All fields are defined in [CompilerConfig](../src/aether/compiler/config.py).

## Artifact

```python
from aether.core.aeg_format import load_aeg_package

package = load_aeg_package("model.aeg")
print(package.root)
print(package.format_version)
print(package.manifest.to_dict())
print(package.get_precision_map())
print(package.compute_size())  # Bytes
package.verify_integrity()
```

`manifest.kernels` records `targets`, `variant_status`, `backend_plans`, and
`portable_backends`. Executable and portable variants have different contracts;
`plan_only` does not establish executable kernels.
`supports_declared_runtime_target(target)` checks emitted variants.
`supports_runtime_target(target)` also considers declared portable contracts.
See [AEGPackage and AEGManifest](../src/aether/core/aeg_format.py).

## Runtime

```python
from aether import Runtime, RuntimeConfig

with Runtime(RuntimeConfig(hf_offline=True)) as runtime:
    response = runtime.generate(
        "model.aeg", "Explain quantum entanglement.",
        max_tokens=128, temperature=0.0, apply_chat_template=True,
    )
    print(response.text)
    print(response.usage)
    print(response.metrics.to_dict())

    for delta in runtime.generate_stream(
        "model.aeg", "Tell me a story.", max_tokens=128,
        apply_chat_template=True,
    ):
        print(delta, end="", flush=True)
```

`generate` returns a `GenerationResponse` containing `text`, `usage`,
`finish_reason`, and `InferenceMetrics`. Metrics include `ttft_ms`,
`throughput_tps`, `backend_name`, and `kernel_target`. Use `generate_stream`
for incremental text; `generate(stream=True)` still returns a complete response.

Generation accepts `max_tokens`, `temperature`, `top_p`, `top_k`, and `stop`.
Omitted defaults come from runtime configuration. Additional parameters are
backend-specific. `messages` can replace a prompt; `chat` accepts role-formatted
messages.

| Method | Return | Purpose |
|---|---|---|
| `generate(model_id, prompt=None, *, messages=None, **kwargs)` | `GenerationResponse` | Complete generation |
| `generate_stream(model_id, prompt=None, *, messages=None, **kwargs)` | `Iterator[str]` | Incremental text |
| `chat(model_id, messages, **kwargs)` | `GenerationResponse` | Role-formatted chat |
| `generate_batch(model_id, prompts, **kwargs)` | `list[GenerationResponse]` | Backend batch execution |
| `benchmark(model_id, prompt="Hello, my name is", max_tokens=128)` | `dict` | Greedy generation with measured duration and backend counts |
| `pull(model_id)` | `None` | Compile into the local cache if absent |
| `list()` | `list[str]` | Cached model identifiers |
| `info(model_id)` | `dict` | Artifact metadata |
| `remove(model_id)` | `None` | Remove a cached artifact |
| `hardware()` | `dict` | Host fingerprint and detector evidence |
| `embed(model_id, input)` | `list[list[float]]` | Embeddings when the backend supports them |
| `rerank(model_id, query, documents)` | `list[dict]` | Reranking when the backend supports it |
| `transcribe(model_id, audio, language=None)` | `str` | Transcription when the backend supports it |

`benchmark` returns `model_id`, `backend`, `kernel_target`, `prompt_tokens`,
`completion_tokens`, `duration_s`, `throughput_tps`, and `ttft_ms`.
Artifact loading is excluded from that measured generation interval.

`RuntimeConfig.backend_name` and `execution_devices` constrain selection.
Automatic selection prefers a compatible, ready GPU when the AEG explicitly
declares GPU portability, including CPU-emitted graphs with a `pytorch`
portable contract. AEGs restricted to CPU execution remain on CPU. Missing
GPU backend dependencies permit automatic CPU fallback; an explicit GPU request
fails when incompatible or unavailable. An explicit CPU request forces CPU.
CLI commands honor
`AETHER_BACKEND`, `AETHER_EXECUTION_DEVICES`, and `AETHER_HF_OFFLINE`.
Python callers can use `Runtime(RuntimeConfig.from_env())` to opt into those
environment controls.

```python
config = RuntimeConfig(
    scheduler="slo_aware",
    slo_profiles={"interactive": {"max_ttft_ms": 500.0, "max_tbt_ms": 50.0}},
    mcp_servers={},
    mcp_timeout_ms=5000,
    green_power_management=False,
    telemetry_endpoint=None,
    server_host="127.0.0.1",
    server_port=11434,
)
```

Advanced controls require artifact features and backend hooks. Complete
definitions are in [RuntimeConfig](../src/aether/runtime/config.py) and
[Runtime](../src/aether/runtime/runtime.py).

## Backend access

```python
from aether.backends import GenerationRequest, get_backend

backend = get_backend("aether_cpu")
backend.load_model("local-model", aeg_path="model.aeg")
try:
    result = backend.generate(GenerationRequest(
        model_id="local-model", prompt="Hello", max_tokens=64,
        extra={"apply_chat_template": True},
    ))
    print(result.text)
finally:
    backend.unload_model("local-model")
```

`get_backend` raises `BackendNotAvailableError` when unavailable.
`BackendRegistry` permits custom registration and isolated instances. Discovery
is lazy: CPU lookup does not initialize optional frameworks. Explicit backend
inventory can initialize installed plugins to check availability.

## REST and OpenAI clients

Install the `server` extra, then run:

```bash
aether serve model.aeg --host 127.0.0.1 --port 11434
```

The live schema at `/openapi.json` and interactive documentation at `/docs`
describe the routes of the installed version. OpenAI clients use
`/v1/chat/completions`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="aether")
response = client.chat.completions.create(
    model="model.aeg",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

Set `stream=True` for streamed OpenAI completion events. Use the configured
server API key when authentication is enabled.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health`, `/v1/health` | Service health |
| `POST` | `/v1/generate`, `/v1/chat` | Aether generation and chat |
| `POST` | `/v1/chat/completions` | OpenAI chat completions |
| `POST` | `/v1/embeddings`, `/v1/rerank`, `/v1/transcribe` | Backend-specific tasks |
| `GET` | `/v1/models` | Cached model list |
| `POST` | `/v1/models/pull` | Pull and compile |
| `GET`, `DELETE` | `/v1/models/{name}` | Inspect or remove an artifact |
| `GET` | `/v1/models/{name}/graph` | Artifact graph |
| `GET` | `/v1/hardware`, `/v1/kernels`, `/v1/metrics` | Host, kernels, and telemetry |
| `POST`, `GET` | `/v1/compile`, `/v1/compile/{job_id}` | Compile and inspect status |

Additional structured-generation, evaluation, training, KV, and coordination
routes are defined in [server routes](../src/aether/server/routes.py). Unsupported
backend tasks report errors; route presence alone is not execution evidence.

## CLI

```bash
aether compile MODEL --target cpu_avx2 --precision q4_k_m --output model.aeg
aether inspect model.aeg
aether graph model.aeg
aether run model.aeg --prompt "Hello" --max-tokens 64 --stream
aether bench model.aeg --max-tokens 64 --output baseline.json --json
aether bench model.aeg --compare baseline.json --json
aether benchmark model.aeg --runs 5 --warmup 1 --max-tokens 64 --json
aether serve model.aeg --port 11434 --grpc-port 50051
aether pull MODEL --compile-local
aether list
aether info model.aeg
aether rm MODEL
aether hardware detect --json
aether hardware detect --json --save host-hardware.json
aether hardware capabilities --json
aether hardware validate cpu_generic --execute --json
aether hardware validate cuda_sm89 --backend pytorch --device-index 0 --execute --json
aether doctor --json
aether kernels
aether kernel generate cuda_sm90 rmsnorm
aether trace model.aeg --prompt "Hello"
aether safety model.aeg
aether kv-share MODEL --output shared-kv.aeg
aether multi-agent --agents 2
aether kv transfer-stats
aether hub --url http://127.0.0.1:8080 login
aether hub --url http://127.0.0.1:8080 push NAMESPACE/NAME model.aeg
aether hub --url http://127.0.0.1:8080 pull NAMESPACE/NAME downloaded.aeg
aether hub --url http://127.0.0.1:8080 search QUERY
aether version
```

Use `aether --help` and each command's `--help` for the complete installed
inventory. Compilation targets and experimental options require their execution
contracts. Native CUDA kernel generation requires its toolchain; Hub operations
require a configured service and authorization.
The [70-path command validation matrix](cli-command-validation.md) distinguishes
help coverage, actual local execution, contract tests, and required dependencies.

Both CLI benchmark commands use backend token counts and first-token timing.
Repeated reports label estimates for legacy callbacks and leave TBT unavailable
when per-token timing is absent.

`--json` emits unwrapped JSON to stdout; diagnostics use stderr. A nonzero exit
can accompany a structured failure report, so consumers should check both.
`pull --compile-local` forces local recompilation. `inspect` accepts an AEG path
or cached model ID and supports both `--mtp` and `--json`.
`hardware validate TARGET` checks the environment; add `--execute` to run a real
backend workload. Detection alone never establishes execution or production
validation. See [hardware evidence](hardware-feedback-validation.md).

`ttt-config --ttl` and explicit `--layers` selection, `kv-compress
--cross-layer-groups`, and merge `--method slerp` are reserved controls that
currently fail explicitly. GRPO needs backend training callbacks; the CLI does
not synthesize gradients when those callbacks are absent. `train verify` requires
`--ground-truth` for math or `--test-code` for code; its reward checks a response,
not a trained model. A TEE report can establish structural validity or a local
platform probe; CLI verification does not perform vendor cryptographic
attestation and returns `verified: false` with a nonzero exit.

`status`, `slo-status`, `trace` without a prompt, and KV status commands describe
the current CLI process. They do not query another running service. `multi-agent`
initializes a local KV coordinator; its `test` form does not establish model
generation or cross-machine transfer. `green-route` labels its static region
profile; that selection is not a measured live power or carbon reading.
`runtime reweight` applies persisted task vectors to its local runtime session;
it does not persist a modified AEG for the next CLI invocation.

For Python callers, `Runtime.set_model_task_weights(model_id, mapping)` accepts
arbitrary valid task labels, including `weights`, `self` and `model_id`.
`set_task_weights` remains the convenience interface, including mapping-only
global defaults. Per-model settings override those defaults; explicit request
`task_weights` override both, and explicit `None` selects base weights. The SDK
provides `AetherClient.set_task_weights_mapping(mapping)` for the same labels.
Coefficients must be finite nonnegative numbers, excluding booleans and strings;
normalization handles large finite values without overflowing. REST validates
these inputs before numeric coercion and returns 422 for invalid values.

## Compatible AEG merging

```bash
aether merge first.aeg second.aeg --method linear --weights 0.25,0.75 --output merged.aeg
aether merge-info merged.aeg
aether run merged.aeg --prompt "Hello" --max-tokens 16
```

```python
from aether.compiler.aeg_merging import merge_aeg_packages

merged = merge_aeg_packages(
    ["first.aeg", "second.aeg"], "merged.aeg",
    method="linear", coefficients=[0.25, 0.75],
)
merged.verify_integrity()
```

Every input must verify integrity and have matching architecture, computation
graph, operator metadata, tokenizer files, weight names, and tensor shapes.
The destination must be new and cannot overlap an input. Mixed AEG/checkpoint
inputs are rejected. The operation stages and verifies a complete package before
publishing it, processes one tensor at a time, and preserves base storage formats;
quantization therefore rounds the merged FP32 values to those formats.

Implemented AEG methods are linear interpolation, task arithmetic (Python API),
DARE, and TIES. Linear coefficients sum to one. TIES uses a per-tensor magnitude
threshold targeting the largest 20%, retaining all ties at the cutoff, so the
actual retained fraction can exceed 20%. It uses weighted magnitude sign
election and a weighted disjoint mean; its
coefficients must be nonnegative. DARE uses seeded 90% drop-and-rescale task
vectors. These explicit variants implement [task arithmetic](https://arxiv.org/abs/2212.04089),
[TIES](https://arxiv.org/abs/2306.01708), and [DARE](https://arxiv.org/abs/2311.03099)
operations without claiming a universal quality gain.

The base graph, tokenizer, execution contracts and compiled auxiliary features
are preserved. Old signatures, weight attestations and evaluation certification
are invalidated. Merging writes `quality_validated: false`; fresh evaluation is
required and the merge record does not automatically become a quality certificate.
Source-checkpoint compilation uses Pass 12 and the ordinary ingestion contract
for architecture-aware names, QKV packing, framework layouts and shard loading.
Projection matrices and biases retain their shapes until alignment to the base
graph or AEG weight index. Matching parameter subsets are allowed; every
requested source must contribute at least one matching tensor. Unmatched,
unreadable or shape-incompatible sources fail instead of producing a partial
success. Actual local checkpoint merging was checked against independently
interpolated persisted weights, including the separate GQA projections.

`Runtime.merge(model_id, task_vectors, method="task_arithmetic", density=1.0)`
accepts partial sources when each contributes an exact-shape matching tensor.
Its separate density control retains a fraction `density` for DARE, selects
exactly `ceil(density * tensor_size)` coordinates for TIES with stable tie
handling, and applies drop/rescale followed by that trimming for `dare_ties`.
Other methods require `density=1`; soup requires uniform coefficients.
FREE/free_merging and evolutionary methods retain the existing L2 proxy search,
which supplies no model-quality measurement. Changed tensors are requantized
with the base format and block size; untouched tensors keep their encoded bytes.
The result is staged and verified before publication, with stale measured PPL,
evaluation, signatures and attestations cleared. Persisted raw task deltas are
added to the already merged model when session reweighting is requested.

## Hub service and client

Install `.[server]` to run the HTTP service:

```python
import uvicorn
from aether.hub.server import AetherHubServer, create_hub_app

hub = AetherHubServer("./hub-storage")
uvicorn.run(create_hub_app(hub), host="127.0.0.1", port=8080)
```

New storage uses `AETHER_HUB_ADMIN_TOKEN` when explicitly supplied, or generates
a random administrator credential persisted locally in
`hub-storage/metadata/user_admin_001.json`. The service never logs that token.
Retrieve it only through administrator-controlled local storage, then create
scoped user accounts. Existing accounts are preserved when reopening storage.
Credential files use owner-only permissions on POSIX; Windows permissions inherit
from the storage directory, whose ACL the administrator must configure.

```bash
aether --cache-dir ./cache hub --url http://127.0.0.1:8080 login
aether --cache-dir ./cache hub --url http://127.0.0.1:8080 push alice/model model.aeg
aether --cache-dir ./cache hub --url http://127.0.0.1:8080 search alice/model
aether --cache-dir ./cache hub --url http://127.0.0.1:8080 pull alice/model downloaded.aeg
```

`AETHER_HUB_URL` can replace `--url`. Login saves the token in the selected cache
only after the service explicitly validates it. Push requires an acknowledgement
matching the uploaded model ID, SHA-256 hash and byte count. Namespace permissions
control writes; private/team/public visibility and the authenticated account
control reads. Upload validates bounded ZIP bodies, safe paths, AEG integrity and
hashes. Package archives reject symlinks, junctions and special files.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Service reachability; no authentication or inference proof |
| `POST` | `/auth/validate` | Validate a token |
| `GET` | `/v1/models`, `/v1/models/search?q=QUERY` | Authorized listings and search |
| `POST` | `/v1/models/{namespace/name}/upload` | Persist a verified AEG archive |
| `GET` | `/v1/models/{namespace/name}` | Model manifest |
| `GET` | `/v1/models/{namespace/name}/download` | Artifact bytes with content hash |
| `DELETE` | `/v1/models/{namespace/name}` | Authorized removal |

Legacy `/v1/hub/...` routes remain aliases. CLI Hub operations use strict remote
mode: offline/authentication failures return errors. Python `HubClient` also
offers explicit `allow_local_cache=True` development fallback; it reports
`source: local_cache` and `persistence: client_instance_memory`, which is neither
remote publication nor disk persistence. Successful remote responses report
`source: server` and `persistence: remote_service`. Local HTTP upload, download and
server-restart recovery were exercised; no external Hub publication was used for
this validation.

## MCP tools

```bash
aether mcp add model.aeg --server local --transport http --endpoint http://127.0.0.1:9000/mcp
aether mcp list model.aeg
aether mcp test model.aeg --tool local/echo --call '{"text":"hello"}'
```

`mcp add` persists the server registry and artifact integrity metadata. It does
not prove the server is installed or reachable. `mcp test` connects, negotiates
the MCP protocol, discovers the advertised tool and performs the actual call.
Duplicate bare tool names require `SERVER/TOOL` qualification. Tool-reported
errors, missing executables, failed handshakes and timeouts exit nonzero.
Connections are closed after the CLI operation.

The Python MCP client accepts a stdio command as an argument list or a quoted
command string. HTTP supports JSON and SSE responses with negotiated session and
protocol headers. `websocket` is an alias for `ws`; WebSocket execution requires
its optional dependency and a functioning server. Tests use real local stdio and
HTTP transports; alias tests alone do not establish WebSocket server execution.

## gRPC

`aether serve model.aeg --grpc-port 50051` enables gRPC alongside REST with
the server extra installed. The typed service is `AetherRuntime`, with
`Generate`, `GenerateStream`, and `Health` methods. See
[aether.proto](../src/aether/server/proto/aether.proto).

```python
import grpc
from aether.server.proto import aether_pb2, aether_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = aether_pb2_grpc.AetherRuntimeStub(channel)
print(stub.Health(aether_pb2.HealthRequest()).status)
response = stub.Generate(aether_pb2.GenerateRequest(
    model_id="model.aeg", prompt="Hello", max_tokens=64,
))
print(response.text)
channel.close()
```

If bearer authentication is configured, pass
`metadata=(("authorization", "Bearer YOUR_TOKEN"),)` to each RPC. TLS and
mutual TLS configuration are supported by
[start_grpc_server](../src/aether/server/grpc_service.py).
Python server owners should call `shutdown_grpc_server(server)` from
`aether.server` before closing the caller-owned Runtime. The helper stops RPC
admission, waits for shutdown and joins the server's owned executor.

## Further APIs

Specialized loaders, distributed execution, evaluation, and Hub clients are
defined in [ingestion](../src/aether/compiler/stage1_ingestion/),
[parallelism](../src/aether/parallelism/),
[evaluation](../src/aether/observability/evaluators.py), and
[HubClient](../src/aether/hub/client.py). See their signatures and capability
requirements before enabling optional workflows.
