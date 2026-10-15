# Runtime completion verification

This report describes the local source continuation after the hardware and CLI
feedback investigation. The earlier reports retain their historical test
checkpoints. Current execution APIs and examples are in
[runtime feature integration](runtime-feature-integration.md).

## Root causes and implemented changes

| Problem | Correction |
|---|---|
| Several feature flags selected planners or changed cache descriptors without performing the advertised operation | Added executing phase workers, measured tensor precision changes and typed cache transport; descriptor counters remain explicitly separate |
| Cache state lacked a portable, lossless storage boundary | Added named typed snapshots, model/schema/geometry/precision checks, transactional disk storage and authenticated private Hub byte routes |
| Prefill and decode scheduling shared mutable engine state | Workers own separate engine state, share immutable weights where supported, use bounded queues and transfer actual cache tensors |
| Speculative generation could repeat full prefixes or exceed its token budget | Target/draft verification uses incremental cache state, rolls back rejected proposals, bounds output, and applies probability-corrected acceptance |
| Switching a speculative session to ordinary decoding could pass a composite cache to an incompatible engine | Extract the matching target cache; validate complete tokenizer and causal execution contracts before draft inference |
| A live speculative session could resume in memory but failed disk/Hub persistence | Persist the exact authoritative target cache/prefix through existing codecs and rebuild the compatible draft cache on resume; reject mismatched live target/prefix state |
| DirectML could execute raw ONNX but lacked compiled AEG execution | Added semantic dense-decoder prefill/decode ONNX export and strict provider/device selection; unsupported operators fail explicitly |
| Precision changes could retain original device tensors through conversion/restoration | Release replaced tensor references immediately, hash backups with bounded memory, restore alias groups transactionally, and discard backups when unloading |
| Interrupting precision calibration could leave a partial tensor conversion | Roll back staged changes on cancellation and propagate the interruption |
| Precision reports used a placeholder `mixed` value | Report the known arithmetic dtype and separate calibrated storage changes; unavailable contracts remain unknown or explicitly unsupported |
| JSON strings such as `"false"` could enable options through truthiness | Validate runtime field types and nested device/calibration inputs before constructing execution features |
| CLI completion, interrupted streaming and comparison could leave runtimes/workers alive | Close stream iterators before owned runtimes; release the primary benchmark model before loading its comparison; keep server lifetime explicit |
| A request arriving during backend unload could recreate workers after feature cleanup | Runtime admission closes before unloading, active operations finish cooperatively, and owned paused streams are closed |
| Simultaneous requests could mutate the same session cache and invalidate its token prefix | Serialize operations on the same session while preserving concurrent requests using different sessions |
| Canceled reused streams could leave cache state inconsistent with its saved token prefix | Invalidate unfinished mutable session state; successful completion still commits the exact consumed prefix |
| Response-cache hits skipped requested session state | Stateful requests bypass the response cache and execute the requested cache operation |
| Public API stub advertised obsolete constructor arguments and nonexistent exports | Generate explicit re-exports from the real public API and enforce freshness in CI |
| Profiling multiple ONNX sessions with one prefix could overwrite evidence within a second | Add a unique session identity to profiling filenames while preserving the requested prefix |
| Emulated CXL close released mappings while leaving a daemon alive | Stop and join the owned defrag worker before closing actual mmap/file resources; use-after-close fails explicitly |
| Terminal runtimes could retain serialized cache tensors in RAM | Discard owned resident snapshots during shutdown while preserving committed disk/Hub references for restart |
| Automatic target fallback masked a failed quality gate or corrupted artifact as an unavailable backend | Enforce persisted quality policy before hardware selection and preserve permanent integrity errors through every loader; retry only backend initialization failures |
| Token-only session and shared-prefix identities could reuse cache state from different execution weights | Bind artifact, precision, adapter and task coefficients to live/persisted state; partition raw shared state by executor representation and device; recompute unknown or mismatched live state |
| Native CPU streaming ignored task weights and TTT that ordinary generation applied | Share request preparation and cleanup across blocking/streaming generation; do not publish request-local TTT state as reusable base cache |
| Task clones dropped model/layer execution settings | Preserve complete dataclass records while replacing requested tensors; unchanged tensor objects remain shared |
| Decoded low-bit projection matrices retained metadata that selected a second raw-code decode | Keep storage provenance separate from active encoded-tensor dispatch; loaded CPU projections use the existing native FP32 execution path |
| Mapping-only task weights were unused, and reserved parameter names blocked valid labels | Apply copied global/model/request precedence and provide a canonical mapping API across Runtime, CLI, REST and SDK; validate all coordinates before normalization |
| Response caching and direct batching could silently ignore active request controls | Bypass incomplete response-cache keys and reject unsupported batched combinations before executing; retain real batching for ordinary supported requests |
| Dispatch calibration returned an unqualified minimum after noisy/cold probes | Warm up, adapt timed blocks, check repeated dispersion windows and return unknown when bounded measurements do not settle; retain the existing stale-value guard |
| Immediate Windows publication occasionally failed after cold compilation with WinError 5 | Retry only bounded Windows access/sharing failures around the final atomic rename; preserve destination checks and propagate persistent failures |

The preceding feedback fixes also preserve artifact-compatible target selection,
GPU inventory independent of optional SDKs, OS-usable CPU instruction evidence,
distinct FP16/BF16 extensions, compiler failure visibility, and honest hardware
execution/production evidence. A CPU-only artifact does not gain a CUDA contract
because the host has a GPU. A compatible ready accelerator is preferred only
when the artifact authorizes that execution path.

Loading verifies declared package hashes and rejects recorded or malformed
evaluation failures across backend fallback. A missing evaluation report is
allowed but uncertified. Checksums and a matching cache fingerprint establish
byte/identity consistency; publisher authentication and behavioral quality need
a trusted signing chain and workload evaluation. Merging clears stale evidence
and marks `quality_validated:false`, which neither blocks an otherwise valid
artifact nor supplies a new quality certificate.

The shutdown repro executed real CPU inference during a deliberately paused
unload boundary. Before the fix, `close()` returned with two live phase workers
and one pool despite an empty model map. A separate actual CPU repro submitted
two requests against one cached session: one failed, and the saved prefix had
four tokens while the mutated cache length was five. These are execution
failures, not hardware-detection simulations.

A further real CPU reproduction reused base-model K/V during an adapted request,
changing logits by up to 0.45367467 and changing greedy output. Native CPU
streaming also produced base output where blocking task-weighted generation
produced different tokens. Regression tests now compare actual engine state and
prefill/decode results, rather than accepting a flag or configuration assignment
as evidence that those controls executed.

## Files and structure

The integration is divided by execution contract:

- `runtime/cache_handoff.py`, `torch_cache_handoff.py`, `kv_payload.py` and
  `kv_storage.py`: typed state extraction, validation and real persistence.
- `runtime/session_contract.py`, compiled handles and `r2_multi_agent_kv.py`:
  execution-weight provenance, local target identity and bounded compatible
  shared-prefix partitions. `cpu_engine.py`: field-preserving execution clones.
  `runtime/aeg_loader.py`: decoded tensor versus storage-format separation.
- `runtime/disaggregated_execution.py`: independent bounded phase workers,
  handoff, cancellation and generation adapters.
- `runtime/sampling.py`, `spec_decode.py`, `spec_generation.py` and
  `speculative_engine.py`: shared sampling and target-verified speculation.
- `runtime/execution_precision.py` and `execution_features.py`: measured tensor
  changes, restoration, execution ownership and feature composition.
- `runtime/lifecycle.py`: concurrent admission, terminal shutdown, owned streams
  and bounded per-session lock ownership. `r12_cxl_kv_pool.py`: mapped-pool lifetime.
- `runtime/onnx_export.py` and `backends/onnx_backend.py`: semantic dense AEG
  lowering and actual selected-provider inference.
- `backends/artifact_validation.py`: shared persisted evaluation and permanent
  integrity-error classification across CPU, Torch, ONNX and automatic dispatch.
- `runtime/runtime.py`, `config.py`, `kv_cache.py`, backend handles, CPU/Torch
  dispatch, grammar sessions, Hub client/server and `cli.py`: public integration.
- `server/grpc_service.py` and server exports: explicit RPC/executor shutdown.
- `server/routes.py`, `sdk.py` and Runtime task APIs: strict coefficient validation
  and mapping-based task labels. `placement/census.py`: qualified dispatch timing.
- `utils/artifact_publication.py`, Runtime merge and AEG merge publication:
  immediate rename with bounded Windows-only recovery after access/sharing errors.
- `scripts/generate_stubs.py`, `aether/__init__.pyi` and CI: public typing
  consistency.
- Related regression suites cover real CPU engine families, actual DirectML,
  cache/server restart, sampling distribution, precision rollback, target
  selection, configuration, command behavior and resource lifetime.

Speculative persistence reuses the ordinary lossless target storage boundary.
Restart does not retain process-local target/draft identities or serialize a
second acceleration cache. The next tokenizer-validated draft prefills the saved
prefix while the target reuses its exact stored state; this adds draft prefill
cost to the first resumed speculative request.

README, getting-started, runtime and API documentation now link the supported
execution contracts. Existing unrelated work in the checkout was preserved.

## Research used

[DistServe](https://arxiv.org/abs/2401.09670) and
[Splitwise](https://arxiv.org/abs/2311.18677) inform phase separation and explicit
placement. This implementation provides in-process worker separation; it does
not claim those papers' cluster topology or throughput results.
[PagedAttention](https://arxiv.org/abs/2309.06180),
[CacheGen](https://arxiv.org/abs/2310.07240) and the
[LMCache technical report](https://lmcache.ai/tech_report.pdf) inform the cache
boundary. Current snapshots preserve exact bytes rather than adopting lossy
compression. [Speculative decoding](https://arxiv.org/abs/2211.17192) supplies
the target-distribution acceptance rule.
Execution-aware prefix identities follow the reuse boundaries in the official
[vLLM prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
and [SGLang](https://arxiv.org/abs/2312.07104); their performance claims are not
transferred to this implementation. Dispatch timing uses warmup, adaptive blocks,
median/dispersion checks and a bounded stopping policy informed by the official
[PyTorch benchmark guidance](https://docs.pytorch.org/docs/2.9/benchmark_utils.html).
The host issue-cost probe does not substitute asynchronous device completion
time or manufacture a stable result under changing load.
Windows publication follows the documented
[directory-rename](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information)
and [file-sharing](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)
contracts. A held member file can block directory publication. The process
responsible for the original intermittent WinError 5 remains unknown; bounded
recovery does not identify it or bypass an ACL. Persistent failures still raise
the concrete error. No partial-copy fallback or permission change is used.
Destination checks retain ordinary `Path.rename` semantics on POSIX, including
the pre-existing race with concurrent creation of an empty destination directory;
they do not claim an atomic `RENAME_NOREPLACE` guarantee there.
[HAWQ](https://arxiv.org/abs/1905.03696) motivates measuring sensitivity;
the implemented policy measures real logits and latency rather than claiming
Hessian-based optimization. DirectML follows the official
[ONNX Runtime provider constraints](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).

## Verification environment and commands

Tests run from the local editable repository on Windows. Python 3.10 includes
CPU Torch and CPU ONNX Runtime; the clean Python 3.12 environment contains the
project test extra without those optional frameworks. The separate Python
3.12 DirectML environment contains ONNX Runtime DirectML and ONNX.

The physical accelerator is Intel UHD Graphics, with driver 32.0.101.7085.
Strict DirectML tests disable CPU fallback and inspect actual provider node
profiles. The machine has no NVIDIA/AMD discrete GPU, Apple accelerator, vendor
NPU or FPGA available for physical execution tests.

Full-suite verification uses separate temporary/cache directories, one BLAS
thread, offline model lookup, actual native compilation when required and JUnit
results. Windows verification uses `-p no:faulthandler` after native WMI/ORT
aborts were observed with the default fault-handler plugin. This flag does not
simulate hardware or change provider selection, and it does not guarantee that
vendor runtime crashes are prevented.

```powershell
python scripts/generate_stubs.py --check
python -m ruff check src tests scripts/generate_stubs.py --select E9,F63,F7,F82
python build/runtime-completion/source_snapshot.py
& build/runtime-completion/run-verification.ps1 -PythonPath 'C:\Program Files\Python310\python.exe' -Label complete310
& build/runtime-completion/run-verification.ps1 -PythonPath '.\build\feedback-validation\python312-env\Scripts\python.exe' -Label complete312
python build/runtime-completion/source_snapshot.py --check
```

The helpers under `build/runtime-completion` are retained local evidence tools,
not installed project commands. A fresh checkout can run the same suite directly
with `python -m pytest --no-cov -p no:faulthandler --timeout=180 -v
--junitxml=results.xml` after configuring a writable temporary/cache directory.
The source manifest hashes implementation, tests, benchmark code, packaging,
API stub generation, hardware catalog and CI; it does not hash the local helpers.

The current hardware/CLI recheck runs these handlers from the clean editable
environment (`python -m aether` also makes the selected interpreter explicit):

```powershell
python -m aether hardware --json
python -m aether hardware detect --json
python -m aether hardware detect --json --save host-hardware.json
python -m aether hardware capabilities --json
python -m aether backend list --json
python -m aether hardware validate cpu_generic --json
python -m aether hardware validate cpu_generic --execute --json
python -m aether hardware validate cuda_sm89 --execute --json
```

The last command is an expected nonzero missing-device check on this host.

Focused DirectML verification uses the DirectML interpreter, `PYTHONPATH=src`,
and `tests/unit/test_aeg_onnx_export.py` plus
`tests/unit/test_onnx_provider_contract.py` and
`tests/unit/test_onnx_session_contract.py`. Detailed commands and process/JUnit
results are retained under `build/runtime-completion`.

The 2026-10-15 untraced DirectML run passed all 45 tests in 22.64 seconds with native
process exit 0. The frozen source manifest checked 524 files with no changes
before and after execution. Actual exported prefill/decode graphs and public Runtime generation,
streaming and phase workers executed on Intel UHD with CPU fallback disabled;
node profiles verify the selected provider. The new session tests also exercise
bound-cache reuse and recomputation of real state with unknown provenance.
Earlier attempts terminated with
Windows native exception `0xC000070A` at the first DirectML case. Standalone
matrix and dense-graph controls, captured/uncaptured individual cases and a
separate traced full run succeeded. The earlier native exception's cause remains
unresolved; these passes do not establish production stability. Its logs are
preserved as `directml-final-frozen-native-failure.*`, separately from successful
historical `directml-final-frozen.{log,xml,exit}` and current
`directml-complete.{log,xml,exit}`. The maintained hardware catalog retains its
explicit historical validation dates; this report records the later source run.

An earlier full audit reported 15 failures on Python 3.10 and two on Python
3.12. Twelve were caused by the verification wrapper forcing decode calibration
off while testing enabled calibration; an isolated default-policy rerun passed
all 38 decode-strategy tests. Two assertions described capabilities superseded
by the actual ONNX export integration. The remaining failures exposed permanent
artifact errors being masked by automatic fallback; those defects were fixed
and the quality/dispatch regression bundle passed all 63 tests on Python 3.12.
Interrupted/quota-inclusive command wall clocks are not performance benchmarks.

A subsequent pre-final Python 3.10 audit recorded 4,636 passed, 24 skipped and
one failed dispatch-repeatability check. Consecutive probes differed by 2.11x,
exceeding the unchanged 2x stale-value guard. The probe previously returned a
minimum even without convergence. The qualified timing implementation then
passed the 141-case placement/probe bundle, including the real CPU repeatability
case. Unsettled or unavailable measurements still return unknown rather than
being turned into successful hardware evidence.

## Complete-suite checkpoint

The implementation, tests and configuration manifest contains 524 files.
It matched before and after the fresh DirectML run and after the complete
Python 3.12 and Python 3.10 suites. Documentation and retained local evidence
helpers are outside that manifest. Results below are actual runner counts and native process exits;
test durations are not inference benchmarks.

| Scope | Actual result | Native exit | Evidence under `build/runtime-completion` |
|---|---|---|---|
| Full Python 3.10 suite with CPU Torch/ORT and native compiler | 4,865 passed, 25 skipped; 1,232.65 s | 0 | `complete310.{log,xml,exit}` |
| Full clean Python 3.12 suite, `[test]` dependencies without Torch/ORT | 4,278 passed, 376 skipped; 599.92 s | 0 | `complete312.{log,xml,exit}` |
| Physical Intel UHD DirectML, strict provider and real node profiles | 45 passed; 22.64 s | 0 | `directml-complete.{log,xml,exit}` |

Python 3.12 skips retain optional Torch/Transformers, ONNX, gRPC and telemetry
dependencies, absent physical hardware, opted-out network tests and unavailable
synthetic checkpoint cases. The full log records each reason. The separate
framework-enabled Python 3.10 suite exercised the installed CPU Torch/ORT paths,
with zero JUnit failures or errors. Its previously failing real CPU dispatch
repeatability case passed in the full suite.

The 25 Python 3.10 skips have these recorded reasons:

| Reason | Count |
|---|---:|
| External network tests deliberately disabled | 14 |
| DirectML wheel absent from the CPU ORT environment; separate physical run above | 5 |
| No physical NVIDIA CUDA device/matching runtime | 2 |
| Wikitext-2 and HellaSwag calibration datasets not cached locally | 2 |
| Synthetic fixture has no source weights and cannot produce a runnable AEG | 2 |

The two complete suites and physical DirectML suite all ended with native exit
0. Their scopes overlap and their test counts must not be summed into one total.

Focused execution/lifetime results overlap and must not be added together:

| Scope | Environment | Actual result |
|---|---|---|
| Lossless speculative target persistence, restart, changed compatible draft and mismatch guards | Python 3.10, actual CPU and Torch CPU dense/hybrid models | 7 passed; exit 0 |
| Existing ordinary and failed session persistence | Python 3.10 | 2 passed; exit 0 |
| Six-family cache handoff, bounded phase workers and speculative generation including cross-thread stream closure | Python 3.10 / clean Python 3.12 | 78 passed / 70 passed, 8 Torch-only skipped |
| Resident snapshot cleanup plus actual disk/Hub storage and restart | Python 3.10 / clean Python 3.12 | 41 passed / 38 passed, 3 Torch-only skipped |
| CLI/runtime ownership and RPC executor shutdown | Python 3.10 / clean Python 3.12 | 182 passed / 155 passed, 27 optional gRPC skipped |
| Actual localhost TLS/mTLS gRPC | Python 3.10 | 8 passed |
| Native mapped CXL-emulation lifetime | Python 3.10 | 8 passed; physical CXL not tested |
| Windows publication, public merge integrity/execution and field-preserving task clones | Python 3.10 / clean Python 3.12, frozen 524-file checkpoint | 76 passed in each environment; both native exits 0 |
| Real stored-codec execution across 12 formats and two decoder layouts | Python 3.10 | 25 passed; base/zero-task/zero-LoRA prefill and cached decode match stored-codec references |
| Quantized loading, CPU request clones and Native request parity | Clean Python 3.12 | 100 passed; native exit 0 |
| Related loader, compiler and CPU execution regressions | Python 3.10 | 149 passed, 2 expected missing-source skips; native exit 0 |

Optional-dependency skips remain visible. Controlled CUDA/HIP/MPS/vendor API
fixtures verify contracts and failures; they are not physical execution evidence.

The new numerical/publication regressions are in
[test_cpu_request_cloning.py](../tests/unit/test_cpu_request_cloning.py),
[test_aeg_quantized_execution_contract.py](../tests/unit/test_aeg_quantized_execution_contract.py),
[test_artifact_publication.py](../tests/unit/test_artifact_publication.py) and
[test_merge_publication.py](../tests/integration/test_merge_publication.py). Actual Win32
handles exercise temporary and persistent directory-rename failures. Cold
subprocess compilation is followed immediately by merge, integrity verification
and real CPU generation. Native backend/ONNX session tests and request policy
tests separately exercise execution-weight provenance, streaming/TTT cleanup,
strict REST validation, shared prefixes and batching guards.

The 2026-10-15 hardware/CLI recheck used the clean Python 3.12 environment and
local `src/aether` source. All 70 help paths passed and matched the documented
inventory. Fingerprint, detection/save, capabilities, backend inventory and CPU
validation commands returned parseable JSON. CPU validation with `--execute`
compiled an offline AEG and ran native prefill/decode, with maximum absolute
errors of 2.09e-7 and 1.49e-7 against its independent NumPy reference. Environment
checks without `--execute` retained `execution_tested:false`. Absent CUDA
returned exit 1, a concrete missing-driver error and no execution success.
The reporting regression bundle passed 69 tests; one physical CUDA case was
deselected from that bounded bundle. Detailed commands, native build identity
and outputs are retained in `hardware-20261015/summary.json` and its `final`
subdirectory under the local evidence directory. Initial sandbox-denied attempts
are retained separately from the successful escalated checks.

## Remaining boundaries

These changes implement shared operator/cache contracts, not model-name or
laptop exceptions. Tests execute six CPU architecture families and supported
dense ONNX graphs, but cannot certify every checkpoint or future architecture.
Calibrated quality/latency bounds apply to the supplied sequences. No universal
speedup, allocator memory reduction or production validation is inferred.

The native CUDA full-inference path, planning-only vendor NPU/FPGA adapters and the modern
TensorRT-LLM compiled-AEG integration remain unavailable where the required
executable contract is missing. Existing compilation/planning features are
preserved. CUDA/HIP/MPS framework adapters, multiple physical accelerators,
cross-host RDMA and physical CXL were not exercised on this machine.

Separate phase pools combined with a speculative draft or active adapter/TTT
need a worker construction contract and currently fail explicitly. Dense ONNX
export does not lower MoE, SSM, MLA, encoder or unsupported dynamic RoPE graphs.
Sharded precision/cache handoff requires collective adapters. These limits are
documented rather than hidden behind successful detection or simulated tests.

The [CLI command inventory](cli-command-validation.md) records the scope of all
70 registered paths. Help and prerequisite-error tests do not establish an
implemented training or attestation workflow. CLI GRPO still needs an integrated
gradient backend, and TEE verification lacks vendor cryptographic attestation;
neither is reported as successful execution. Planning, configuration and
current-process status commands retain their explicitly documented scope.
