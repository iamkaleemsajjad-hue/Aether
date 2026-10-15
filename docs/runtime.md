# Aether Runtime

The Aether Runtime is the execution engine that loads AEG artifacts, detects hardware, selects the best backend, and serves inference requests.

## Startup Flow

When you run `aether run` or `aether serve`:

1. Detect hardware fingerprint (GPU, VRAM, driver, compute capability).
2. Load the AEG manifest.
3. Intersect executable artifact variants with host devices and installed backends.
4. Load the model into the selected backend.
5. Initialize the backend's KV state and runtime cache accounting.
6. Initialize applicable scheduling and control components.
7. Enable speculative execution only with its required payload/backend contract.
8. Serve requests.

## Hardware Detection

The runtime derives targets from queried device and usable instruction evidence:

- NVIDIA → queried compute capability, such as `cuda_sm89`.
- Apple Silicon → identified Metal device/chip, such as `metal_m1` or `metal_m3`.
- AMD → detected HIP architecture, such as `rocm_rdna3` or `rocm_cdna3`.
- Intel GPU/NPU → `openvino_gpu` or `openvino_npu` device evidence. Supported
  dense portable AEG graphs can use the ONNX Runtime OpenVINO provider contract;
  the separate planning-only vendor adapter is not an executed generation path.
- CPU → `cpu_avx512`, `cpu_avx2`, `cpu_neon`, or `cpu_generic`, according to process-usable features.

Unknown architectures retain their detected identity; they are not assigned a
known profile merely from a vendor name. A detected target does not itself
authorize artifact execution.

You can inspect the detected hardware with:

```bash
aether hw
aether hardware detect --json
aether hardware validate cpu_generic --execute --json
aether hardware validate cuda_sm89 --backend pytorch --device-index 0 --execute --json
```

Detection does not run inference. `implemented` describes an inference adapter,
`available` requires hardware and a usable vendor runtime, and
`backend_available` records execution dependency readiness separately. Unknown
stages are `null`; `execution_tested` remains false until an explicit workload
succeeds. CPU and PyTorch AEG `--execute` validation compiles and reloads an
offline tiny AEG, runs prefill and cached decode on the requested device, and
compares logits with an independent NumPy reference. DirectML uses a raw ONNX
workload as described below. These functional tests never imply production validation.

Read `precision_scope` and `precision_semantics` with the precision flags:
device-arithmetic evidence is separate from queried vendor runtime formats.
Metal/OpenVINO runtime format support does not prove native arithmetic or
successful Aether execution. Generic AVX-512 does not
confirm FP16 or BF16 extensions. Aether can still execute supported FP16/BF16 or
packed quantized weights through FP32 conversion and accumulation. Unknown
memory is reported as unknown, with measurement source fields; a CUDA Driver API
version is recorded separately from the NVIDIA driver release.

A Windows C++ compiler is optional for CPU execution. MSVC, clang-cl, GCC/MinGW,
and Clang are discovered and checked; failed compilation produces a visible
warning and the real NumPy reference path remains available. Native libraries
carry platform and ISA identities and are recompiled or bypassed when incompatible.

DirectML supports raw ONNX models through the optional ONNX Runtime adapter.
DLL presence or a registered provider does not prove a usable GPU. In a separate
Windows environment, install the local source with `pip install -e ".[directml]"`
and run `aether hardware validate directml --execute --json`. This checks an
actual FP32 ONNX workload and requires DirectML node attribution. An AEG with
the portable `onnx` contract can also export supported dense decoder operators
to real prefill and cached-decode graphs. Unsupported operator plans fail
explicitly. See [runtime feature integration](runtime-feature-integration.md)
for actual Intel DirectML generation/streaming evidence and export boundaries.

## Backend Selection

Automatic selection prefers a ready accelerator authorized by the artifact's
executable variants or portable graph contract. It otherwise uses an authorized
CPU path. A CPU artifact without an accelerator contract stays on CPU; a
portable PyTorch contract can authorize CUDA, ROCm or MPS execution even if its
emitted native variant is CPU. `plan_only` entries alone do not authorize execution.
Explicit backend and device requests are strict constraints.

Hardware inventory is independent of installed inference packages. Windows
DXGI, Linux PCI/DRM and macOS native Metal/profiler probes retain visible GPUs
even when the compute SDK is missing. Stable PCI, UUID, LUID or Metal registry
identities join matching provider records; ambiguous counts remain qualified.
Dedicated video memory and shared system memory limits are separate values.
Non-Apple Metal devices use the generic `metal` contract, with no Apple ISA or
precision assumptions. PyTorch MPS only exposes its default adapter.

- CPU execution path / CPU-only AEG → framework-free `aether_cpu` (native library or NumPy reference).
- Portable CUDA/ROCm/MPS AEG → optional PyTorch graph executor.
- Other executable variants → a backend compatible with that target.
- Local `.onnx` → a compatible registered accelerator provider, then CPU if
  automatic accelerator initialization fails. Explicit provider requests disable
  CPU fallback. A successful session alone does not establish execution evidence.
- Source checkpoints → an installed source frontend with compatible device
  support; CPU fallback works when optional accelerator dependencies are absent.
- Local GGUF → llama.cpp with verified local GGML device enumeration. GPU
  selection passes GPU layers and a concrete ordinal; CPU requests use zero
  GPU layers. Builds without a usable device enumeration remain CPU-capable.

MLX requires `mlx-lm` as well as a usable Metal runtime. Its stream uses the
frontend's actual event counts and sampler. Synchronous vLLM returns a buffered
completion; incremental serving requires its separate asynchronous API. Runtime
options are translated before invoking vendor constructors rather than passed
through as unsupported vendor keywords.

The llama.cpp subprocess adapter launches a dedicated loopback server on an
unused port and verifies a unique model alias before routing requests. An
occupied explicitly requested port is rejected. A health response from an
unrelated server is insufficient to establish that the requested model loaded.

The CLI reads `AETHER_BACKEND`. Python callers can use
`RuntimeConfig.from_env()` or set `RuntimeConfig.backend_name` explicitly;
plain `Runtime()` does not import that backend environment setting.
Device constraints come from `AETHER_EXECUTION_DEVICES` or
`RuntimeConfig.execution_devices`.
Explicit accelerator requests fail if their execution contract or backend is
unavailable. `kernel_target` reports the selected target for each loaded model;
`hardware()` continues to describe the host.

See [research-backed execution contracts](research-backed-runtime.md) for the
distinction between portable graphs and target-specific execution.

## Disaggregated Prefill/Decode

Aether executes separate, owned prefill and decode workers when enabled. Workers
transfer typed, lossless model state through bounded in-process queues, with
measured phase/queue/handoff latency, cancellation and explicit shutdown.
CPU decoder families and supported nonsharded Torch/ONNX adapters have concrete
cache contracts. Placement uses explicit adapter factories; it does not enable
RDMA or establish a speedup on a single laptop:

```python
RuntimeConfig(disaggregate_prefill_decode=True, prefill_devices=["cpu"], decode_devices=["cpu"])
```

## Tree-Speculative Decoding

An explicitly loaded draft model proposes tokens and the target engine verifies
their probabilities. Exact greedy and corrected stochastic rejection paths
preserve the target sampler, roll back rejected state, and respect output bounds.
The tree-planning components are separate from this executing draft/target path.

Configuration:

```python
RuntimeConfig(
    draft_model_id="./compatible-draft.aeg",
    speculative_gamma=4,
)
```

The runtime does not select a draft checkpoint from the target's family name.
Tree speculation requires an explicitly supplied compatible draft/backend;
packaged MTP/draft-head paths have their own payload and executor contracts.
Acceptance monitoring can disable speculation, but configuration alone does
not establish that a draft model is loaded or that a speedup occurs.

## Global KV Cache Manager

Backend engines own active inference tensors. The global manager can attach
real typed snapshots, commit bytes to bounded local disk storage, reload after
restart, and upload/download private authenticated Hub snapshots. Descriptor-only
accounting remains available and is reported separately from tensor transfers:

| Tier | Storage | Use |
|------|---------|-----|
| L1 | GPU HBM | Active request blocks |
| L2 | CPU DRAM | Prefix cache, recently evicted blocks |
| L3 | Configured local disk | Committed, checksummed snapshots |
| L4 | Explicit authenticated Hub | Private, namespaced snapshots |

`Runtime.persist_session()` and `restore_session()` connect this storage to
actual execution caches. Local disk is not assumed to be physical NVMe or
GPU-direct storage. Hub I/O requires explicit client and namespace configuration.

## Dynamic Precision Adjustment

`Runtime.adjust_precision()` changes real execution tensors transactionally.
It measures supplied calibration inputs and accepts only candidates within
configured output-error and latency limits. Original tensors have exact disk
restoration data. Automatic pressure-based adjustment requires real calibration
token sequences; a fixed precision label or estimated PPL penalty is not proof
of execution or quality. Lossy changes require explicit permission:

```python
RuntimeConfig(dynamic_precision=True, precision_calibration_tokens=[[1, 2, 3]], precision_allow_lossy=False)
```

## Continuous Batching

See [runtime feature integration](runtime-feature-integration.md) for supported
cache families, configuration examples, measured evidence and combinations that
require a separate adapter. Global CLI `--runtime-config runtime.json` accepts
the same validated RuntimeConfig fields for generation and serving commands.

Scheduling components support iteration-level admission. Actual batching and
latency behavior depend on the selected backend and serving integration;
their presence does not guarantee throughput or an SLO on every device.

## Configuration

```python
from aether import Runtime, RuntimeConfig

config = RuntimeConfig(
    backend_name="auto",
    hf_offline=True,
    speculative_decoding=False,
)
rt = Runtime(config)
```

## Metrics

`/v1/metrics` currently returns a JSON object with these counters:

- `runtime_up`
- `loaded_models`
- `kv_cache_blocks`
- `kv_cache_hit_rate`

Per-request `InferenceMetrics` carries measured throughput and first-token
latency when the selected backend supplies them. The endpoint above is not a
Prometheus text exposition or a complete request-performance time series.


## v3.1 Runtime Controls

Runtime-facing PRD features are represented as small, testable control planes:

- EAGLE-3 plans choose fusion layers and tree shape, then disable speculation when acceptance drops below the floor.
- CUDA Graph selectors round dynamic batch and prefill sizes up to captured buckets, otherwise falling back to piecewise eager execution.
- Hot-reload routers assign candidate traffic by stable request hash and can promote or rollback instantly.
- Observability gates block rollout when required benchmark results are missing or exceed the configured regression budget.
