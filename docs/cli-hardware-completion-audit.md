# Hardware, dispatch and CLI integration audit

The historical audit below predates the executing phase-worker, measured
precision, lossless disk/Hub cache and dense AEG-to-ONNX integrations. Those
subsequent changes and their current boundaries are documented in
[runtime feature integration](runtime-feature-integration.md). Historical
full-suite counts below describe their original frozen source checkpoint.

This report covers the local-source continuation of the
[original user feedback fixes](user-feedback-validation.md) and
[hardware reporting investigation](hardware-feedback-validation.md).
It separates executable behavior from discovery, planning and test doubles.
Existing benchmark results from other machines are not measurements of this change.

## Project contract

Aether compiles supported model graphs into an Aether Execution Graph (AEG):
canonical operators, persisted weights, tokenizer resources, target variants and
portable execution contracts. The runtime selects a compatible installed
executor for that artifact or a supported source frontend. The portable graph
is the common boundary; a model name, GPU label or target plan is not an
executable implementation. This keeps dispatch independent of a particular
model family or laptop.

## Findings and corrections

| Finding | Classification and correction |
|---|---|
| CPU AEG benchmark selected host CUDA | Confirmed. Selection intersects artifact contracts with backend/device readiness. CPU-only contracts stay on CPU. An explicit portable PyTorch contract can authorize an accelerator. |
| GPU present but hidden when an SDK is absent | Confirmed. Bounded Windows DXGI, Linux PCI/DRM and macOS inventory preserve visible graphics adapters independently of optional inference frameworks. |
| GPU selected but frontend remains on CPU | Confirmed in several adapters. Device constraints and formats now reach the actual loader; unsupported combinations fail explicitly. GGUF offload requires a verified local GGML device and supplies its layers/ordinal. |
| DLL/provider presence presented as successful execution | Confirmed evidence error. Hardware, runtime, dependencies, implementation, execution and production validation have separate fields and scopes. |
| Generic AVX-512 implied FP16/BF16 | Confirmed. Process-usable instruction flags and distinct precision extensions determine native arithmetic support. Storage conversion/FP32 accumulation remains a separate supported path. |
| Windows compiler absent | Expected optional condition, with discovery bugs corrected. Supported MSVC, clang-cl, GCC/MinGW and Clang discovery is tested. Failure remains visible; NumPy execution is reported as reference execution. |
| Driver and memory placeholders | Confirmed reporting gaps. Actual API values retain source and scope; unknown measurements remain unknown. Dedicated video memory is separate from shared memory limits. |
| README precision option and backend import | Confirmed CLI/API drift. Precision option, public backend lookup, examples and error behavior are aligned. |
| Test extra cannot collect tests | Confirmed dependency gap. Test dependencies include pytest-timeout and FastAPI; optional hardware/framework tests skip explicitly. |
| Cache removal leaves loaded aliases | Confirmed. Unload precedes deletion, all aliases are invalidated, and removal is bounded to the model cache. Long generated cache names are shortened deterministically to avoid Windows native-payload path failures. |
| MCP accepts a transport but cannot call it correctly | Confirmed. Stdio framing, initialized notification, IDs, qualified tool names, HTTP sessions/SSE, timeout cleanup and dead-connection handling are integrated and tested against local servers. |
| Hub requests report local success after remote failure | Confirmed. Remote mode is explicit and strict; authentication/upload acknowledgements are checked. The local HTTP server persists artifacts and enforces namespace/authentication/integrity rules. |
| Merge claims success without a runnable weight result | Confirmed. AEG merging persists real tensors with graph/tokenizer/shape checks, verifies staged output, and invalidates old evaluation/provenance claims. Unsupported controls fail clearly. |
| llama-server health belongs to another model | Confirmed. The adapter uses an unused loopback port and verifies a unique launched alias. Occupied explicit ports are rejected before launch. |
| A later Torch model changes another model's device | Confirmed. Mesh, input device and runtime family are recorded per loaded model. Cached explicit requests are checked against actual model placement, including hidden CPU offload. |
| Runtime merging ignores density or retains old quality evidence | Confirmed. Density controls actual DARE retention and TIES trimming; DARE-to-TIES applies both. Output is staged and verified, quantization block sizes are preserved, and measured PPL/evaluation/ownership evidence is invalidated. |

Automatic dispatch prefers an authorized, ready accelerator and otherwise uses
an authorized CPU path. Explicit device/backend constraints are strict. A
visible GPU without compatible dependencies or model execution support remains
visible in inventory while CPU execution can still be selected. This is expected,
not evidence that GPU generation succeeded or that every GPU should accept every
artifact. Resource exhaustion and unsupported operators may still prevent loading.

## Evidence fields

| Field | Meaning |
|---|---|
| `hardware_present` | A physical/provider device was observed; unknown is `null`. |
| `runtime_available` | The relevant vendor runtime was successfully queried, separately from the adapter. |
| `backend_dependencies` | Individually observed software dependencies. |
| `implemented` | An inference adapter exists for its documented format/scope; planning code alone is insufficient. |
| `available` | Hardware/runtime readiness for the reported target; never a successful workload claim. |
| `backend_available` | Execution dependency/adapter readiness for the documented scope. |
| `execution_tested` | An explicit real workload completed successfully in the stated scope. Discovery starts false. |
| `production_validated` | Requires separate measured multi-model/workload evidence; no new production certification is issued here. |
| `memory_bytes` | Measured memory in `memory_kind`/`memory_source`; zero without evidence is not confirmed zero capacity. |
| `driver_version` | Queried release/version with source; CUDA API version is kept separate from driver release. |
| `supports_*` | Interpret with `precision_scope`/`precision_semantics`: queried device arithmetic or vendor runtime formats. Runtime-format support does not establish native arithmetic or execution. False also covers unconfirmed and does not prohibit supported storage conversion. |
| `gpu_count_known` | Whether physical identities establish an exact count. Ambiguous partitions/provider overlaps report a lower bound and maximum where available. |

Discovery does not import every optional framework or run inference. Profiling
is enabled for explicit validation; normal ONNX loading avoids its cost. Stable
PCI/UUID/LUID/Metal registry identities link records, without assuming that
different APIs use the same device ordinal. All-zero identifiers are unknown.
Graphics inventory is scoped to what the operating system exposes; it does not
guarantee discovery of every driverless, TCC-only or non-graphics accelerator.

## Actual local execution

The host is Windows with an Intel Family 6 Model 186 CPU, 12 logical processors,
about 7.69 GiB system RAM and Intel UHD Graphics. The final inventory queried
driver **32.0.101.7085**, **128 MiB dedicated video memory**, and a separately
reported shared system-memory limit. No NVIDIA, AMD HIP or Apple Metal device
was available for physical execution tests.

- CPU: real AEG compilation, reload, prefill and cached decode passed. Native
  versus independent NumPy reference maximum absolute errors were
  **2.0862e-7** and **1.1921e-7**, respectively. The available GCC/MinGW path ran;
  MSVC/clang-cl behavior has contract tests, not local native execution evidence.
- DirectML: a real input-dependent FP32 ONNX MatMul ran with CPU fallback
  disabled. Profiling attributed the operation to `DmlExecutionProvider`;
  maximum absolute error against NumPy was **0.0**.
- DirectML Runtime generation: a dependent Cast/ReduceSum/MatMul graph produced
  two greedy tokens and streamed them through the ONNX adapter. All **15**
  profiled nodes ran through DirectML with fallback disabled. This is a small integration
  workload, not a language-quality or throughput benchmark.
- Model coverage: CPU tests compare supported graph families, including
  Llama/Gemma, GPT-2, T5, Mixtral, MLA and Mamba-style execution. Raw local
  Llama/GPT-2 source inference was separately compared with Transformers.
  No universal model-family or parameter-scale guarantee follows from these tests.

Evidence files are generated under `build/` and are not repository source:
[inventory](../build/hardware-coverage/hardware-final.json),
[CPU validation](../build/hardware-coverage/cpu-final-execution.json),
[DirectML validation](../build/hardware-coverage/directml-execution.json), and
[DirectML generation](../build/hardware-audit/directml-generation/execution.json).

## Verification

Both final complete suites passed with **zero failures, zero collection errors
and process exit 0**:

| Environment | Actual full-suite result | Evidence |
|---|---|---|
| Windows Python 3.10.11, NumPy 1.26.4, CPU Torch/Transformers/ONNX installed | **4,379 passed, 22 skipped**, 6 warnings; 1,614.48 s | [Log](../build/hardware-completion/frozen310/suite.log), [JUnit](../build/hardware-completion/frozen310/suite.xml) |
| Clean Windows Python 3.12.15, NumPy 2.5.4, editable `.[test]`, no optional Torch/Transformers/ONNX runtime | **3,848 passed, 317 skipped**; 858.46 s | [Log](../build/hardware-completion/frozen312/suite.log), [JUnit](../build/hardware-completion/frozen312/suite.xml) |

The optional frameworks, network access, missing calibration data and
unavailable hardware explain the skips. Suite duration is test-run wall time,
not an inference-performance measurement. The six Python 3.10 warnings concern
existing dependency/runtime deprecations; the clean Python 3.12 run recorded no
warning summary. The generated
[verification summary](../build/hardware-completion/verification-summary.json)
contains the machine-readable counts.

Focused final integration checks also passed:

| Check | Actual result | Evidence |
|---|---|---|
| CLI/Hub/MCP/hardening, actual local services | 191 passed | [JUnit](../build/cli-audit/final-contracts.xml) |
| Checkpoint alignment/merging and Pass 12 | 88 passed | [JUnit](../build/cli-audit/source-merge-verified.xml) |
| Runtime merge math, staging, evidence and real CPU execution | 40 passed, 1 optional Transformers skip on Python 3.12 | [JUnit](../build/hardware-coverage/runtime-merge-final312-elevated.xml) |
| Torch per-model state, vendor and dispatch contracts | 127 passed | [JUnit](../build/hardware-audit/torch-state-freeze.xml) |
| Actual local Llama/GPT-2 generation and streaming, compiled Llama state/parity | 2 passed on Python 3.10 | [JUnit](../build/hardware-audit/torch-state-source-real.xml) |

The full runs used isolated workspace caches and offline model controls.
A snapshot of **429** source/test/configuration files verifies that source
remained unchanged during completion verification. Source-wide critical Ruff
checks, full Ruff checks on the refactored AEG helper/source frontend modules,
`git diff --check`, and `pip check` in the clean core and isolated DirectML
environments passed. Dependency warnings remain in the existing Python 3.10
installation. Intermediate failures remain in their logs and are not counted
as final passes.

The isolated DirectML provider bundle passed **28 tests, zero skips/failures**
in **5.51 s**, including real CPU ONNX, DirectML MatMul and Runtime generation/
streaming. On this host, the default pytest fault-handler plugin interacted
with a native Windows WMI exception during ONNX Runtime import; three attempts
aborted before completing validation. Disabling that diagnostic plugin with
`-p no:faulthandler` allowed the unchanged checks and real workloads to complete.
The standalone GPU generation/streaming script also exited 0. This is a local
test-runner limitation; its precise native cause is not established.
[Passing log](../build/hardware-coverage/directml-without-fault-plugin.log),
[JUnit](../build/hardware-coverage/directml-without-fault-plugin.xml),
[default-run failure](../build/hardware-coverage/directml-isolated-tests.log),
and [standalone GPU log](../build/hardware-coverage/directml-stream-final.log)
preserve both outcomes. CPython's
[Windows WMI issue](https://github.com/python/cpython/issues/125315) is relevant
background, not proof that this host encountered that exact bug.

The [CLI command matrix](cli-command-validation.md) distinguishes help coverage,
functional local workflows, optional dependency failures and reserved features.
Invoking every help page does not prove every distributed/vendor integration.

Reproduce the core suite from the checkout:

```powershell
python -m pip install -e ".[test]"
python -m pip check
python -m pytest -q --no-cov
python -m aether hardware --json
python -m aether hardware detect --json --save host-hardware.json
python -m aether hardware capabilities --json
python -m aether hardware validate cpu_generic --execute --json
```

DirectML belongs in a separate Windows environment because its ONNX Runtime
package replaces the CPU-only distribution:

```powershell
python -m pip install -e ".[test,directml]"
python -m aether hardware validate directml --device-index 0 --execute --json
python -m pytest -q --no-cov tests/unit/test_onnx_provider_contract.py
```

For the fault-handler issue observed on this test host, the successful command was:

```powershell
python -m pytest -q --no-cov -p no:faulthandler tests/unit/test_onnx_provider_contract.py
```

A machine with a compatible NVIDIA device and CUDA-enabled PyTorch can run:

```powershell
python -m aether hardware validate cuda_sm89 --backend pytorch --device-index 0 --execute --json
```

That command is a reproduction instruction, not a successful RTX test result
from this host. Validation fails when the requested device is unavailable.

## Changed implementation areas

- Discovery/evidence: `backends/{capabilities,hardware_detector,hardware_inventory,
  cuda_detection,auxiliary_detection,hardware_validation,hardware_backends}.py`,
  `runtime/hardware.py`, and `hardware_validation_matrix.json`.
- Dispatch/loaders: `runtime/{runtime,aeg_loader,hybrid_engine}.py`,
  `compiler/stage3_targeting/{backend_selector,hardware_profile}.py`,
  `core/{constants,types,aeg_format}.py`, `backends/{base,registry,model_contract,
  native_cpu_backend,native_cuda_backend,torch_backend,onnx_backend,
  llamacpp_backend,mlx_backend,vllm_backend,trtllm_backend}.py`.
- Native/format portability: `kernels/{native_cpu,native_cuda,quantized_linear}.py`,
  SafeTensors/GGUF/MLX ingestion and `utils/file_io.py`.
- CLI/integrations: `cli.py`, `hub/{client,server}.py`,
  `runtime/r6_mcp_integration.py`, `compiler/aeg_merging.py`,
  `compiler/stage2_optimizer/pass12_model_merging.py`, `core/weight_store.py`,
  dependency/CI configuration and corresponding regression/integration tests.
- Documentation: README, getting-started guides, runtime/API/research references,
  original feedback reports, this report and the CLI matrix.

Paths above are relative to `src/aether/` unless otherwise stated. The working
tree also contains prior user benchmark work; it is preserved and should not be
mistaken for changes introduced by this audit.

## Research and implementation decisions

[TVM](https://www.usenix.org/system/files/osdi18-chen.pdf) and
[MLIR](https://arxiv.org/pdf/2002.11054) motivate preserving computation
semantics separately from target scheduling/lowering. The engineering
application here is explicit portable/executable/planning contracts, verified
against actual loaders. Neither paper certifies Aether's implementation.

[FlexGen](https://arxiv.org/abs/2303.06865) demonstrates the importance of
memory placement and workload constraints. GPU preference therefore remains
conditional on compatible execution; detection alone is insufficient and no
blanket speed claim is made.

[Task Arithmetic](https://arxiv.org/pdf/2212.04089),
[TIES](https://arxiv.org/pdf/2306.01708), and
[DARE](https://arxiv.org/abs/2311.03099) inform tensor merge semantics. AEG TIES
uses a per-tensor magnitude threshold targeting the largest 20%, retaining all
cutoff ties, followed by magnitude-based sign election and a
weighted disjoint mean. DARE uses seeded drop/rescale task vectors. These
documented variants require new model-quality evaluation. Auxiliary adapters
from the base artifact are preserved and require reevaluation; old signatures,
attestations and evaluation scores do not certify changed weights. SLERP remains
reserved rather than silently behaving like linear interpolation.

The distinct `Runtime.merge` API supports partial task vectors. Its TIES retains
exactly `ceil(density * tensor_size)` coordinates with stable tie handling;
DARE retains/rescales at probability `density`; `dare_ties` applies both.
Density defaults to one and unsupported density/method combinations are
rejected. Runtime session reweighting adds persisted raw deltas to the already
merged model. The current Runtime task-vector path materializes full base
arrays, so large merges still require sufficient host RAM; the full-AEG helper
processes one tensor at a time. No unlimited parameter-scale guarantee is made.

Implementation contracts were checked against primary specifications:
[ONNX Runtime DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html),
[DXGI driver queries](https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgiadapter-checkinterfacesupport),
[MCP transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[GGML backend ABI](https://github.com/ggml-org/ggml/blob/master/include/ggml-backend.h),
[MLX-LM](https://github.com/ml-explore/mlx-lm), and
[vLLM offline API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/).

## Remaining boundaries

At this historical checkpoint, DirectML validation covered raw FP32 ONNX.
The subsequent [runtime integration](runtime-feature-integration.md) also
validates supported dense FP32 AEG export, generation and streaming on DirectML.
Native CUDA's
incomplete full-inference interface remains unavailable; portable CUDA AEG uses
PyTorch. OpenVINO task generation, QNN, ANE, RISC-V NPU, TensorRT-LLM and FPGA
Aether execution remain incomplete where documented. Their planning/discovery
features are retained, with unsupported execution reported explicitly.

Physical RTX 3050/4070, HIP, Metal, multi-GPU and NPU execution could not be
certified here. Simulated driver/API tests cover contracts and failure handling
only. External Hub publication, remote MCP services and production distributed
workloads were not exercised; local HTTP/stdio workflows were. No new universal
performance, language-quality or production-validation claim is made.

At this checkpoint, runtime disaggregation and automatic dynamic-precision
fields were reserved integration controls, and the global KV manager tracked
descriptors without persisting tensor bytes. Executing phase workers, measured
precision changes and lossless disk/Hub persistence were subsequently added;
their supported combinations and current evidence are documented in the
[completion verification report](runtime-completion-verification.md).
Automatic draft-checkpoint selection remains disabled; drafts are explicit.
