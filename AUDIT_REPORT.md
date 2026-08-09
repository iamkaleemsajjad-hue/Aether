# Aether Runtime â€” Final Adversarial Audit

Audit date: 2026-08-09

Repository: C:\Users\pc\Desktop\Aether Runtime

Environment: Windows AMD64, Python 3.10.11, 12 logical CPUs, approximately 7.69 GiB RAM, PyTorch 2.9.1 CPU build, no CUDA, ROCm, Metal, TEE, CXL, NIXL, or accelerator hardware.

## 0.1 Remediation update â€” evidence collected after the baseline audit

This section supersedes only the execution results explicitly listed here. The
requirements matrix and feature classifications below remain intentionally
conservative where a feature is still metadata-only, hardware-dependent, or
not exercised end-to-end.

### Confirmed fixes

- `scripts/ci_smoke_test.py --verbose`: **15/15 PASS** on the CPU toolchain.
- A real local Llama-style checkpoint containing SafeTensors weights and a real
  Transformers tokenizer now compiles, saves, reloads in a fresh loader path,
  produces logits, and generates through the public `Runtime` API. This is
  covered by `tests/integration/test_local_safetensors_aeg_roundtrip.py`.
- Graph hashing and AEG-IR serialization now canonicalize dataclasses, enums,
  and tensor metadata instead of hashing object identities or stringifying
  runtime objects. The Pass 9 pruning metadata regression is covered by the
  hardening tests.
- Optional R3/R5/R6/R7/R8 layers now initialize from persisted artifact files
  after process restart when their real configuration exists. They are not
  inferred from compiler-only Python state.
- Grammar-constrained decoding no longer silently falls through to ordinary
  generation. The PyTorch backend now applies a real per-step
  `transformers.LogitsProcessor`, and the compiled CPU engine applies the FSM
  bitmask before sampling and advances the request-local FSM session. The
  current built-in compiler marks its character-code approximation as not
  tokenizer-aware, so Runtime rejects it instead of presenting it as a valid
  production constraint. `test_generate_applies_grammar_fsm_token_mask` and
  the local AEG integration tests pass.
- Unknown model identifiers fail closed unless a local configuration or explicit
  bounded Hub discovery is available; the compiler no longer invents a default
  architecture for an unknown model.
- Compiled AEG generation without a tokenizer-backed adapter fails explicitly
  instead of returning fabricated text. Video generation likewise refuses a
  text-only fallback when no executable video encoder is present.
- AEG tar and Hub ZIP extraction now reject absolute paths, traversal entries,
  and links; tampered/traversal cases are covered by hardening tests.
- Hugging Face `trust_remote_code` is disabled by default and is exposed as an
  explicit runtime opt-in rather than being silently enabled by the backend.
- Windows cache resolution now uses a writable per-user location and falls
  back to a temporary per-user cache only when the host denies both standard
  locations.
- The public CLI now exposes executable `eval`, `safety`, `trace`,
  `reasoning`, `mla-stats`, `kv-share`, `multi-agent`, `slo-status`, `hub`,
  `train grpo`, `kernel generate`, and `kv transfer-stats` commands. These
  commands call real runtime/compiler/client code and fail on backend errors.
- The measured-evaluator path now accepts only a `BenchmarkResult` (or a
  validated measured mapping), rejects inconsistent counts/scores, and includes
  a JSONL exact-match/accuracy evaluator. A real local AEG integration test
  invokes the compiled model through that evaluator and the regression gate
  rejects the result; no benchmark score is inferred from non-empty text.
- Local `torch.save`/`pytorch_model.bin` ingestion now has a real compile,
  AEG integrity, reload, and generation integration test. Corrupt or empty
  PyTorch shards fail closed instead of being silently skipped.
- The repository now collects **1,719 tests**. The complete local suite
  finishes with **1,704 passed, 15 skipped, 0 failed** in approximately 252
  seconds. The full run reports approximately **68% combined statement/
  branch coverage** under the repository's pytest coverage configuration; this
  is not equivalent to feature completeness.
  Skips
  are explicitly limited to network/Hugging Face access or synthetic-model
  availability.
- AEG/2.0 writer methods now persist their corresponding manifest claims, and
  validation rejects enabled flags without concrete enabled payloads, grammar
  entries, MCP registrations, TEE backend, task-vector files, TTT weights, or
  graph plans. Disabled descriptors remain explicitly disabled.
- The compiler now selects AEG/2.0 when applied v4 passes are present and
  AEG/3.0 when applied v5 passes are present. The manifest and FORMAT_VERSION
  sentinel are checked for agreement on reload. The local v5-enabled artifact
  test proves AEG/3.0 save, integrity verification, and reload.
- Pass 12 model merging now reads native AEG weight stores, rejects unreadable
  or empty sources, writes a verified merged AEG with its tokenizer, and has a
  local reload/generation test; sources without overlapping tensors still fail
  explicitly.
- R11 semantic request caching now intercepts repeated Runtime.generate calls
  and reports a real cache hit on the local CPU AEG path using the offline
  embedding fallback; the cache configuration uses a consistent max-token key.
- Agentic sessions now pass a session-owned cache handle through the compiled
  CPU AEG backend. The backend reuses only an exact token prefix, reports the
  measured reused-token count, and releases the cache on session close. A new
  local SafeTensors integration test proves the second turn reuses KV state;
  GPU and cross-model KV sharing remain unverified.
- Architecture ingestion now recognizes common lower-case Hugging Face
  `model_type` aliases when a config omits the `architectures` list. Unit tests
  cover Llama, Qwen, Gemma, Mixtral, DeepSeek, and Mamba aliases; this improves
  config routing but does not prove graph/weight/runtime support for each family.
- GGUF ingestion now reads architecture dimensions from the GGUF header,
  binds standard llama.cpp tensor names (including fused Q/K/V and FFN gate/up
  pairs), exports embedded Unigram tokenizer metadata, and has a local
  compile/save/reload/generate integration test. GGUF files without embedded
  tokenizer vocabulary still fail explicitly.
- `aether serve <model.aeg>` now validates and preloads the supplied artifact
  before binding the HTTP port. A subprocess integration test starts the real
  CLI, waits for `/health`, sends `/v1/generate` over TCP, verifies model output,
  and shuts the process down cleanly.

### Current evidence boundary

The CPU path is now proven for a real local SafeTensors model and one local
GGUF artifact, but this does not prove arbitrary Hugging Face/GGUF
compatibility, model quality, GPU execution,
AEG/2.0 or AEG/3.0 completeness, CXL, TEE hardware, distributed
execution, or the v4/v5 performance claims. Those remain classified below as
partial, unverified, or not implemented rather than upgraded by file presence.

## 1. Executive verdict

### NOT COMPLETE

The repository contains substantial real implementation, especially:

- v3.1 compiler architecture
- CPU-native kernels
- AEG/1.1 serialization
- optimizer pipeline wiring
- strict failure behavior for missing weights
- component-level runtime tests
- API-key authentication
- AEG integrity verification

The central end-to-end promise is now proven for one real local CPU model, but
not for the full PRD scope:

    Real local SafeTensors model
    -> ingest
    -> compile
    -> AEG
    -> reload
    -> inference
    -> REST serving

The official CPU smoke test now passes 15/15, and the local SafeTensors round
trip passes. A real remote Hugging Face model remains unverified here because
network access is unavailable; the compiler correctly fails rather than
fabricating weights.

The current code correctly refuses to fabricate weights or model output. The
primary local CPU workflow is now operationally demonstrated; the remaining
gap is breadth (model families, target backends, and v4/v5 execution).

The v4/v5 additions are mostly metadata planners, configuration emitters,
isolated components, or emulation layers. AEG version selection now exists,
but many v5 payloads and gRPC remain incomplete.

## 2. PRD lineage and audit method

Both PRDs were read completely.

- PRD.md defines the original v3.0 specification and its Part II v3.1 baseline.
- PRD_v2.md explicitly states that v3.1 is the implemented baseline, v4.0 is net-new functionality, and v5.0 is another net-new extension.
- v4.0 adds passes 10â€“17, runtime layers R1â€“R8, AEG/2.0, new APIs, hardware targets, MCP, TEE, green compilation, grammar, TTT, and multi-agent functionality.
- v5.0 adds passes 18â€“22, runtime layers R9â€“R12, AEG/3.0, video, diffusion drafting, ternary quantization, RLVR/GRPO, semantic caching, KV networking, CXL, and additional hardware.

This audit does not count a class, directory, configuration option, or comment as a working feature. A requirement is considered functional only when the real execution path consumes it, real inputs work, failure behavior is correct, and meaningful tests exercise the implementation.

Evidence included:

- complete PRD reading
- repository and source inspection
- full test collection and execution
- official environment and smoke scripts
- direct compiler, AEG, runtime, SDK, REST, CLI, TEE, Hub, and observability probes
- clean-environment installation attempt
- suspicious implementation search
- native CPU kernel execution

## 3. Execution summary

### Environment checks

The environment checker succeeds with UTF-8 output enabled:

- required Python packages: available
- native CPU compiler/toolchain: available
- CUDA: unavailable
- ROCm: unavailable
- Apple MPS/Metal: unavailable
- MLX: unavailable
- vLLM: unavailable
- Triton client: unavailable

The same checker fails under the default Windows CP1252 console because it prints Unicode check-mark characters.

### Test suite

    Collected: 1,719 tests
    Full repository run: 1,704 passed, 15 skipped, 0 failed
    Focused hardening/evaluation/runtime verification: 32 passed
    Local evaluator/AEG verification: 3 focused checks passed
    Official CPU smoke: 15/15 passed
    Full-suite duration: approximately 252 seconds

The full local suite completed without failures. The remaining execution
boundary is:

- real-model integration tests requiring Qwen weights are skipped because the configured proxy cannot reach Hugging Face;
- compiler tests requiring real remote weights are skipped rather than fabricating parameters;
- compiled-AEG generation refuses to fabricate output when tokenizer/model adapters are missing;
- remote-model tests remain unavailable without Hugging Face network access;
- GPU/TEE/CXL/NIXL/Metal/ROCm target tests remain unavailable on this host.

Native CPU and CPU end-to-end tests:

    100 passed
    2 skipped

These validate synthetic/local CPU execution and native kernel compilation. The
local SafeTensors, PyTorch, and GGUF integration tests additionally validate
real tokenizer-backed artifacts, but not model quality benchmarks.

### Clean installation

- Building a wheel in the existing environment succeeds.
- A fresh temporary virtual environment can install the built wheel with
  `--no-deps` and import `aether` successfully (`0.1.0`).
- Normal isolated installation fails offline while resolving build dependencies.
- The fresh no-dependency CLI probe did not complete within 30 seconds, and a
  complete clean install plus compile/run/serve/test workflow was not achieved.

## 4. Requirements matrix

| ID | PRD requirement | Version | Component | Required behavior | Code location | Implemented? | Actually functional? | Tested? | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | SafeTensors ingestion | v3.1 | Stage 1 | Load real tensors and bind them to graph | src/aether/compiler/stage1_ingestion/safetensors_loader.py | Yes | Functional for tested local Llama path | Unit + integration | Real local SafeTensors compile/reload/generate passed |  FUNCTIONAL BUT INCOMPLETE |
| M2 | GGUF ingestion | v3.1 | Stage 1 | Parse GGUF metadata and weights | src/aether/compiler/stage1_ingestion/gguf_loader.py, ingestion.py | Yes | Functional for tested local GGUF with embedded tokenizer; broader GGUF variants incomplete | Unit + integration | Tiny GGUF compiles, saves, reloads, verifies, and generates on CPU |  FUNCTIONAL BUT INCOMPLETE |
| M3 | ONNX ingestion | v3.1 | Stage 1 | Load graph and initializers | src/aether/compiler/stage1_ingestion/onnx_loader.py | Yes | Partial | Unit tests | ONNX backend cannot perform autoregressive generation | PARTIAL |
| M4 | MLX ingestion | v3.1 | Stage 1 | Load and execute MLX models on Apple | mlx_loader.py, mlx_backend.py | Yes | Unverified | No Apple hardware | MLX unavailable | âšª NOT TESTABLE ON CURRENT HARDWARE |
| M5 | PyTorch/Hugging Face ingestion | v3.1 | Stage 1 | Materialize weights, configuration, tokenizer, graph | ingestion.py, pytorch_loader.py, torch_backend.py | Yes | Functional for tested local SafeTensors and torch.save checkpoints; remote HF unverified | Local integration; remote blocked | Both local formats compile, reload, verify, and generate; corrupt shards fail closed |  FUNCTIONAL BUT INCOMPLETE |
| M6 | VLM/video/MLA/MoE/SSM/reasoning/MTP detection | v3.1â€“v5 | Stage 1 | Correct architecture detection and graph extraction | architecture_detector.py | Partial | Not proven | Static/unit tests | Lower-case model_type aliases now tested; no real family-specific graph/model runs | PARTIAL |
| O1â€“O9 | v3.1 optimizer passes | v3.1 | Stage 2 | Modify graph/IR and produce runtime-consumable artifacts | stage2_optimizer/optimizer.py | Yes | Partial | Unit tests | Several passes produce plans or metadata | PARTIAL |
| O10â€“O17 | v4 optimizer passes | v4.0 | Stage 2 | Produce real MTP, grammar, merge, TTT, KV, green, TEE artifacts | pass10 through pass17 | Yes | Mostly metadata/configuration | Mock/synthetic tests | No real artifact consumption | PARTIAL |
| O18â€“O22 | v5 optimizer passes | v5.0 | Stage 2 | Produce diffusion, ternary, video, PEFT, and RLVR artifacts | pass18 through pass22 | Yes | Partial; unsupported paths fail closed | Unit/integration subset | Direct transforms and verifier paths exist; no complete model/runtime integration | PARTIAL |
| H1 | Hardware target registry | v3.1â€“v5 | Stage 3 | Select executable backend per target | targets/registry.py | Yes | No | Registry tests | Profiles and backend candidates only | PARTIAL |
| H2 | Target kernel generation | v3.1â€“v5 | Stage 3 | Emit executable PTX, HSACO, MSL, QNN, FPGA, and RISC-V kernels | stage3_targeting/kernel_emitter.py | No | No | Plan tests | Emits KernelPlan, not binaries | NOT IMPLEMENTED |
| A1 | AEG/1.1 | v3.1 | Artifact | Save/load graph, weights, metadata, kernels, provenance | core/aeg_format.py | Yes | Functional on CPU path | Unit + smoke + local integration | 15/15 smoke and local reload pass |  FUNCTIONAL BUT INCOMPLETE |
| A2 | AEG/2.0 | v4.0 | Artifact | Persist v4 runtime/compiler features | compiler/aeg_format_v2.py | Yes | Partial | Format + hardening tests | Defaults are explicit disabled descriptors; enabled manifest claims require real payloads | PARTIAL |
| A3 | AEG/3.0 | v5.0 | Artifact | Persist v5 artifacts and metadata | core/aeg_format.py, compiler/compiler.py | Yes | Partial | Local v5 artifact reload | Versioned save/reload works; required v5 payloads and executable kernels are incomplete | PARTIAL |
| A4 | AEG integrity | v3.1+ | Security | Verify manifest, graph, weights, and declared artifacts | core/aeg_format.py | Yes | Yes for declared files | Direct tamper test passed | Tampered safety artifact rejected | âœ… COMPLETE |
| R1â€“R8 | v4 runtime layers | v4.0 | Runtime | Execute P-EAGLE, multi-agent KV, grammar, SLO, TTT, MCP, green, and TEE | runtime/r*.py | Yes | Mostly isolated | Component tests | Most are not used by normal generation | PARTIAL |
| R9â€“R12 | v5 runtime layers | v5.0 | Runtime | Execute diffusion, network KV, semantic cache, and CXL | runtime/r9 through r12 | Yes | Mostly isolated/emulated; R11 local initialization repaired | Component tests + local cache probe | No real CXL/network/diffusion backend | PARTIAL |
| API1 | Python Runtime and Compiler APIs | v3.1+ | SDK | Match PRD signatures and perform real operations | runtime.py, compiler.py | Yes | Aliases and async session contracts now align; backend-dependent operations remain incomplete | API + integration tests | Full suite plus local SafeTensors/GGUF generation; training/TEE/video/KV backends unavailable | PARTIAL |
| API2 | Baseline REST endpoints | v3.1 | Server | Implement all baseline /v1 endpoints | server/routes.py, cli.py | Partial | Partial | TestClient + TCP subprocess + OpenAPI probe | Core generation and expanded route registration work; real `aether serve <model.aeg>` now preloads and serves over TCP; full endpoint semantics not proven | PARTIAL |
| API3 | v4/v5 REST endpoints | v4â€“v5 | Server | Grammar, TTT, MCP, green, TEE, video, cache, GRPO, and CXL routes | server/routes.py | Yes | Partial; unavailable backends fail closed | OpenAPI + server tests | 67 routes registered; GRPO verify/status and video stats now execute/record real outcomes | PARTIAL |
| API4 | gRPC | v3.1+ | API | Protobuf, server, client, streaming inference | proto/aether.proto, server/grpc_service.py | Yes | Partial | Real local AEG integration | Authenticated Generate/Health and chunked stream pass; generated typed stubs and token-level streaming remain absent | PARTIAL |
| E1 | Evaluation gates | v3.1 | Quality | Run measured benchmarks and block regressions | observability/ci_pipeline.py, runtime.py | Yes, when configured | Functional for validated evaluator results and JSONL exact-match/accuracy; official datasets are not bundled | Unit + local AEG integration | Real compiled model invoked; deliberately poor result blocked; unavailable path fails closed | PARTIAL |
| P1 | Performance claims | v3.1â€“v5 | Benchmarking | Reproduce latency, throughput, memory, energy, and quality claims | runtime.py, scripts | No | No | No valid benchmark | No real model/baseline comparison | NOT IMPLEMENTED |
| S1 | Safety and provenance | v3.1+ | Security | Provenance, filtering, prompt injection, integrity, audit logs | safety, provenance, AEG | Partial | Functional when explicitly enabled | Hardening + policy tests | Runtime now enforces prompt/output policy when enabled; default remains opt-in and isolation remains incomplete | PARTIAL |
| OBS1 | OpenTelemetry | v3.1 | Observability | Export real traces to an OTLP collector | observability/otel.py | Custom implementation | Local JSON only | Unit tests | No OpenTelemetry SDK/exporter | PARTIAL |
| HUB1 | Aether Hub | v3.1+ | Hub | Login, search, push, pull, integrity, permissions | hub/client.py | Client only | Local archive fallback functional; no live Hub | Local tests | Offline upload/download now preserves and extracts the real uploaded ZIP; remote permissions/deduplication remain unverified | PARTIAL |
| D1 | Distributed execution | v3.1+ | Fleet/parallelism | Multi-process/multi-node inference and recovery | fleet, parallelism | Planning layer | No | Unit tests | Collectives are CPU reference operations | NOT IMPLEMENTED |
| I1 | Installation/distribution | v3.1+ | Packaging | New developer can install and execute | pyproject.toml | Yes | Not proven clean | Wheel build | Fresh isolated install failed offline | PARTIAL |

## 5. v3.1 baseline audit

### Model ingestion

The source contains branches for:

- SafeTensors
- GGUF
- ONNX
- MLX
- PyTorch checkpoints
- Hugging Face Hub IDs

The architecture detector contains entries and configuration parsing for standard decoder models, MoE, DeepSeek/MLA-style metadata, hybrid SSM, VLM-related families, and reasoning metadata.

The strict current behavior is:

    Local SafeTensors model
    -> weight/tokenizer ingestion
    -> compiler and optimizer
    -> AEG save/reload
    -> Runtime generation
    -> PASS

This is preferable to the older behavior of manufacturing synthetic weights, but it means the actual ingestion path is not complete.

One real local Llama-style model was successfully taken through compile, AEG
save/reload, logits, and public Runtime generation using both SafeTensors and
`torch.save` checkpoint formats. A tiny local GGUF artifact also passes this
path. Qwen, DeepSeek, Gemma, Mistral, Mixtral, VLM, video, MLA, LoRA, and
general remote Hugging Face compatibility remain unverified in this environment.

### v3.1 optimizer passes

Passes 1â€“9 are registered in the real optimizer pipeline and execute against graphs. Their implementation quality varies:

- operator fusion modifies graph structure;
- sensitivity analysis uses heuristic/synthetic calibration data;
- precision assignment produces maps;
- KV structuring produces graph nodes;
- MoE routing and parallelism produce plans;
- reasoning and sparse attention produce metadata;
- pruning can produce masks when real tensor data exists.

They are not all connected to executable target kernels or validated against real model quality.

### v3.1 runtime

CPU KV management, scheduling structures, precision management, graph loading, native kernels, and model registration are real components. However, the normal generation path primarily delegates to a backend and does not prove that every v3.1 runtime feature affects decoding.

### v3.1 public API

The basic Runtime and Compiler classes import and instantiate. The expanded
REST routes are registered and the core local generation route is executable,
but most non-core endpoint semantics and the v4/v5 backend integrations remain
partial or unavailable.

## 6. Optimizer passes 1â€“22

| Pass | Status | Implementation, execution, artifact, and test result |
|---:|---|---|
| 1 â€” Operator Fusion | FUNCTIONAL BUT INCOMPLETE | Real graph fusion logic and pipeline connection. No emitted or executed target megakernel. |
| 2 â€” Sensitivity Analysis | PARTIAL | Produces per-layer values, but calibration includes synthetic WikiText/HellaSwag-style samples. No real perplexity measurement. |
| 3 â€” Precision Assignment | FUNCTIONAL BUT INCOMPLETE | Produces maps consumed by quantization. No real quality validation. |
| 4 â€” KV Cache Structuring | FUNCTIONAL BUT INCOMPLETE | Adds KV structures and runtime KV code exists. No real compiled-model proof. |
| 5 â€” MoE Expert Routing | PARTIAL | Routing/planning logic exists. No real Mixtral/DeepSeek/MoE model run. |
| 6 â€” Parallelism Discovery | PARTIAL | Produces sharding plans. Distributed code is CPU reference communication. |
| 7 â€” Reasoning Graph Compiler | PARTIAL | Produces reasoning metadata. No proven generation or quality effect. |
| 8 â€” Sparse Attention | PARTIAL | Produces sparse patterns. No connected sparse target kernel. |
| 9 â€” Pruning/Sparsity | PARTIAL | Can compute masks for real tensors, but no complete sparse runtime path. |
| 10 â€” Native MTP Head Compilation | PARTIAL | Detects MTP declarations, but now refuses to emit zero-filled blobs when real head tensors are unavailable. No real MTP model/runtime proof. |
| 11 â€” Grammar Constraint Compiler | PARTIAL | FSM compilation produces an artifact and decode loops apply masks for trusted precompiled FSAs. The built-in compiler is explicitly marked non-tokenizer-aware, so its artifacts are rejected for production constrained generation until a tokenizer-aware compiler is integrated. |
| 12 â€” Model Merging | FUNCTIONAL BUT INCOMPLETE | Runtime.merge now dequantizes real AEG weights, applies the selected strategy, copies the tokenizer, writes a new AEG, verifies integrity, and a local end-to-end test reloads and generates from it. Multi-model quality validation and all source formats remain incomplete. |
| 13 â€” TTT Fast-Weight Injection | PARTIAL | Configuration and engine exist. No real model adaptation/reload/generation proof. |
| 14 â€” Semantic KV Compression | PARTIAL | Produces plans. No proof of actual KV tensor compression in inference. |
| 15 â€” Cross-Layer KV Sharing | PARTIAL | Analysis and plans exist. No verified pointer sharing in a running backend. |
| 16 â€” Green Energy Compilation | PARTIAL | Produces estimates and hints. No live energy measurement or DVFS control. |
| 17 â€” TEE Enclave Emission | STUB / PLACEHOLDER | Hashes/configuration/HMAC wrappers exist; no executable enclave kernels. |
| 18 â€” Diffusion Drafter Compilation | PARTIAL | Produces schedule/configuration metadata. No drafter weights or real diffusion decoding. |
| 19 â€” Sub-2-Bit/Ternary Quantization | PARTIAL | Direct ternary/BTC/NanoQuant tensor transforms exist, but compilation now fails closed without a measured baseline/candidate evaluator instead of accepting hardcoded perplexity estimates. No integrated ternary model runtime. |
| 20 â€” Video/Streaming Token Compression | PARTIAL | Planner and frame KV manager exist. No real VLM/video ingestion or generation. |
| 21 â€” Advanced PEFT Compilation | PARTIAL | Adapter manifests/opcodes exist; empty or missing adapters now fail/skip explicitly. Runtime adapter execution remains unproven. |
| 22 â€” RLVR Verifier Head Injection | PARTIAL | SymPy and subprocess verification paths exist when supplied ground truth/tests; unverified text now receives zero reward rather than heuristic credit. Runtime.grpo_train_step now fails explicitly because inference has no gradient/optimizer path; no trained verifier head or integrated GRPO compiler flow. |

## 7. Runtime layers R1â€“R12

| Layer | Status | Findings |
|---|---|---|
| Existing EAGLE-3 | PARTIAL | Planner/engine exists, but normal Runtime.generate does not demonstrably execute EAGLE-3 decoding. |
| Existing KV manager | FUNCTIONAL BUT INCOMPLETE | CPU allocation/eviction tests pass; no real compiled-model proof. |
| Disaggregated prefill/decode | PARTIAL | Configuration and metadata exist; no multi-process/network deployment. |
| Dynamic precision | PARTIAL | Manager exists; live backend switching is not proven. |
| R1 P-EAGLE/Saguaro | PARTIAL | Engine has real weighted MTP projection code, but now fails closed when draft weights are absent; it is still not wired into normal generation and no hardware speculative benchmark is proven. |
| R2 Multi-Agent KV | PARTIAL | Public async context manager and coordinator are functional; compiled CPU agentic sessions now reuse exact token-prefix KV, while cross-agent tensor sharing, GPU IPC/RDMA, and cross-model reuse remain incomplete. |
| R3 Grammar FSM | PARTIAL | Precompiled trusted FSAs are consumed by constrained PyTorch/CPU decode paths. The current built-in artifacts are rejected as non-tokenizer-aware; runtime grammar compilation and broad backend coverage remain incomplete. |
| R4 SLO Scheduler | PARTIAL | Scheduler exists but normal generation does not route through it. |
| R5 TTT Engine | PARTIAL | Adapt/reset methods exist; no real model weight adaptation path. |
| R6 MCP | FUNCTIONAL BUT INCOMPLETE | Real JSON-RPC stdio/HTTP/WebSocket client exists and fails closed; not automatically integrated into ordinary generation. |
| R7 Green Power Manager | PARTIAL | Produces estimates/status; no live hardware energy/carbon integration. |
| R8 Confidential TEE | STUB / PLACEHOLDER | Software simulation can initialize with hardware_backed=false; this is not confidential computing. |
| R9 Diffusion Speculative Engine | PARTIAL | Component initializes; no real drafter model is loaded by normal generation. |
| R10 KV Network Transfer | PARTIAL | Structures exist; no NIXL/RDMA/UCCL/NVLink execution. |
| R11 Semantic Request Cache | FUNCTIONAL BUT INCOMPLETE | Exact cache interception and cache-hit metrics now pass against a real local AEG using the offline embedding fallback; production embedding/model persistence and distributed cache behavior remain incomplete. |
| R12 CXL Rack-Scale KV Pool | STUB / PLACEHOLDER | File-backed mmap and in-memory fallback exist; no physical CXL or rack-scale pool. |

## 8. Hardware target matrix

The registry exposes 28 target profiles. Profiles and backend candidate strings are not proof of executable hardware support.

| Target | Evidence | Result |
|---|---|---|
| cuda_sm70 | Profile and generic PyTorch candidate; no executable CUDA kernel generation | PARTIAL / UNVERIFIED |
| cuda_sm80 | Profile and vLLM/PyTorch/TRT-LLM candidates; TRT backend incomplete | PARTIAL / UNVERIFIED |
| cuda_sm89 | Same | PARTIAL / UNVERIFIED |
| cuda_sm90 | Same | PARTIAL / UNVERIFIED |
| cuda_sm100 | Same | PARTIAL / UNVERIFIED |
| cuda_sm100_tee | Profile plus software TEE fallback; no confidential kernel path | STUB / PLACEHOLDER |
| cuda_sm120 | Profile only; no Rubin-specific compiler/kernel implementation | STUB / PLACEHOLDER |
| cuda_sm130 | Static placeholder/profile; no executable target | STUB / PLACEHOLDER |
| cuda_sm100_gb300 | Generic Blackwell profile; no GB300-specific execution | STUB / PLACEHOLDER |
| rocm_rdna3 | ROCm profile/source emitters exist; no hipcc or hardware | âšª NOT TESTABLE ON CURRENT HARDWARE |
| rocm_cdna3 | HIP source generation exists but is not integrated into AEG output | PARTIAL / UNVERIFIED |
| rocm_cdna5_mi455x | Profile only; no CDNA5 execution | STUB / PLACEHOLDER |
| amd_mi350x | Profile/backend candidate only | STUB / PLACEHOLDER |
| metal_m1 | Metal source emitters exist; no Apple device or xcrun | âšª NOT TESTABLE ON CURRENT HARDWARE |
| metal_m3 | Same; M4/M5 claims are not separately tested | âšª NOT TESTABLE ON CURRENT HARDWARE |
| openvino_npu | ONNX Runtime candidate; no NPU execution | âšª NOT TESTABLE ON CURRENT HARDWARE |
| qualcomm_qnn | Candidate string only; no QNN backend | STUB / PLACEHOLDER |
| qualcomm_cloud_ai100 | Candidate string only; no Cloud AI 100 backend | STUB / PLACEHOLDER |
| riscv_mips_s8200 | ONNX Runtime candidate only | STUB / PLACEHOLDER |
| riscv_sifive_x160 | ONNX Runtime candidate only | STUB / PLACEHOLDER |
| riscv_xuantie_c930 | ONNX/PyTorch candidates only | STUB / PLACEHOLDER |
| riscv_cervell | ONNX Runtime candidate only | STUB / PLACEHOLDER |
| fpga_xilinx_vu9p | ONNX Runtime candidate only; no bitstream | STUB / PLACEHOLDER |
| fpga_ternary | BitNet candidate string only | STUB / PLACEHOLDER |
| cpu_avx512 | Native CPU compilation/DLL tests work | FUNCTIONAL BUT INCOMPLETE |
| cpu_avx512_ternary | Profile only; no integrated ternary CPU runtime | PARTIAL |
| cpu_neon | Profile/backend candidate only; no ARM hardware | âšª NOT TESTABLE ON CURRENT HARDWARE |
| cpu_neon_ternary | Profile only | âšª NOT TESTABLE ON CURRENT HARDWARE |
| CPU x86/AVX2 | llama.cpp mentions AVX2, but no complete equivalent target path was verified | PARTIAL |

The critical Stage 3 limitation is that KernelEmitter.emit() returns a KernelPlan. It does not emit PTX, cubin, HSACO, metallib, QNN binary, FPGA bitstream, or RISC-V binary. The normal compiler Stage 3 path creates profiles and backend plans but does not compile these target artifacts.

## 9. AEG audit

### AEG/1.1

The normal compiler uses AEG/1.1. It supports:

- manifest
- graph
- precision map
- weight store
- kernel metadata
- provenance
- safety metadata
- reasoning metadata
- runtime metadata
- declared artifact hashes

AEG integrity is a strong area. A direct test tampering with safety/prompt_guard.json was rejected with AEGIntegrityError.

The official CPU smoke test now loads and executes the generated artifact:

    15/15 PASS, including compile -> quantize -> save -> load -> CPU inference

The local SafeTensors integration also proves tokenizer-backed public Runtime
generation after artifact reload. AEG/1.1 is therefore functional for the
tested CPU path, but arbitrary architectures and all declared target kernels
remain outside this evidence.

### AEG/2.0

AEGPackageV2 can create and validate an AEG/2.0 directory structure containing:

- speculation
- grammar
- merging
- TTT
- green energy
- TEE
- multi-agent
- MCP
- semantic cache
- training
- parallelism

Its creation path writes explicit disabled feature descriptors, not compiled
features. Writer methods persist enabled manifest claims only when concrete
payloads are written, and validation rejects enabled claims without those
payloads. A newly created package can therefore validate structurally while
containing no real model weights, kernels, or enabled runtime behavior; that is
an honest empty package, not v4 implementation.

### AEG/3.0

The core package writer now emits AEG/3.0 when v5 optimizer passes actually
apply, creates the v5 extension directories, hashes payloads, and reloads the
manifest with sentinel validation. This is a real versioning improvement, not
a claim that every PRD v5 payload is implemented: ternary/video/MDLM/PEFT/RLVR
artifacts remain partial and target kernels are not executable.

### AEG round trip

Minimal synthetic AEG save/load/integrity:

    PASS

Real local SafeTensors compiled model workflow:

    PASS â€” compile -> save -> close/reload -> logits -> tokenizer-backed
    Runtime.generate -> REST TestClient generation

This proves the tested AEG/1.1 CPU path and a v5-enabled AEG/3.0 versioned
reload path. The runtime still correctly refuses graph-only AEGs lacking a
tokenizer-backed generation adapter rather than returning fabricated output.
The same integration now saves a tar archive, closes the original package,
loads the archive, verifies its hashes, and executes a forward pass from the
retained extracted package.
The full arbitrary-model and cross-target AEG promise remains incomplete.

## 10. CLI audit

Diagnostic commands that executed:

- aether version
- aether hw
- aether kernels
- aether list
- aether graph
- aether info
- aether grammar
- aether mcp

Problems:

- aether merge now accepts multiple model arguments, but real merge artifacts still require source tensors.
- aether run now rejects a missing path-like AEG with a model-not-found error.
- ttt-config, kv-compress, green-profile, and tee do not robustly validate missing artifacts.
- eval, safety, trace, reasoning, mla-stats, multi-agent, slo-status, kv-share,
  Hub, GRPO, kernel, and KV transfer command surfaces are now registered and
  smoke-tested; backend-specific limitations remain.
- `aether compile` exposes the tested v4/v5 opt-in flags, but those flags do
  not make unavailable hardware or missing model modalities executable.

The kernels command successfully lists 28 profiles, but this proves registry exposure, not kernel execution.

## 11. Python SDK audit

Basic imports and object construction work. The current Runtime exposes methods including:

- generate
- chat
- embed
- rerank
- transcribe
- benchmark
- compile_async
- eval_gate
- generate_constrained
- generate_video
- generate_with_tools
- get_attestation_report
- grpo_train_step
- semantic_cache_stats
- kv_transfer_stats
- quantization_report
- merge
- set_task_weights
- multi_agent_session

Important PRD incompatibilities:

| PRD API | Current implementation |
|---|---|
| get_attestation_report(model.aeg) | Accepts the model and returns a mapping with PRD attribute access; hardware attestation remains unavailable on this host |
| set_task_weights(model.aeg, legal=..., medical=...) | Accepted and normalized; model-specific weights are stored but not yet applied by model inference |
| multi_agent_session(models=[...], coordination="relay") | Async context manager backed by the R2 coordinator; agent generation is Runtime-backed, but real model KV capture remains incomplete |
| grpo_train_step(..., verifier_domain="math") | PRD aliases are accepted, then fails closed because no gradient backend is configured |
| generate_with_tools(..., mcp_tools=[...]) | PRD alias is accepted and invokes configured MCP tools; MCP server availability remains required |

The SDK aliases and async context-manager contracts now match the PRD surface,
but several operations still fail closed because the required training, video,
TEE, MCP-server, or real cross-agent KV backend is unavailable. Accepted API
shape is not counted as functional backend execution.

## 12. REST API audit

The FastAPI application registers **67 `/v1` routes**. The previously listed
baseline and v4/v5 paths are now present, including structured generation,
model graph/MLA/reasoning inspection, evaluation, A/B rollout, merge, agent
sessions, SLO, TTT, MCP, green, TEE, video jobs, semantic cache controls, GRPO
start/status/verification, KV transfer/CXL, and sub-2-bit reporting.

Representative routes include:

    /v1/generate
    /v1/chat
    /v1/embeddings
    /v1/rerank
    /v1/transcribe
    /v1/compile
    /v1/compile/{job_id}
    /v1/models
    /v1/models/pull
    /v1/models/{name}
    /v1/hardware
    /v1/kernels
    /v1/metrics
    /v1/health
    /v1/tools/call
    /v1/grammar/compile
    /v1/grammar/list
    /v1/models/{name}/merge
    /v1/models/{name}/ttt
    /v1/targets
    /v1/targets/{target_id}
    /v1/green/status
    /v1/tee/session
    /v1/tee/session/{id}
    /v1/video/generate
    /v1/video/{job_id}/stats
    /v1/train/grpo/verify
    /v1/models/{name:path}/sub2bit

Authentication works when AETHER_API_KEYS is configured. With no configured
keys, the server intentionally accepts requests. Route registration is now
complete for the audited PRD list, but real semantics remain partial: video
and GRPO report explicit 501/failed jobs when unsupported, cache bypass invokes
real Runtime generation, and sub-2-bit reports are measurement-backed rather
than a compile success claim.

## 13. gRPC audit

The repository now contains `proto/aether.proto` and a real generic gRPC
service/client using protobuf `Struct` serialization. Authenticated Health,
Generate, and GenerateStream RPCs were exercised against a real local
SafeTensors AEG; generation returned actual model output. The stream currently
chunks a completed Runtime response because Runtime has no token-yielding
backend contract, and generated typed Python stubs are not checked in.
Therefore this is functional but incomplete, not a production-complete gRPC
surface.

## 14. Model compatibility audit

| Model/category | Result |
|---|---|
| Qwen | Attempted; failed because weights could not be materialized through the configured proxy |
| Llama | No real weights run |
| DeepSeek/MLA | Static architecture detection only |
| Gemma | No real weights run |
| Mistral | No real weights run |
| Mixtral/MoE | MoE logic unit-tested; no real Mixtral compile/run |
| Qwen-VL/VLM | No real VLM artifact |
| Video model | No real video model or graph extraction |
| Reasoning model | Metadata/heuristic support only |
| Long-context model | Static context handling only |
| LoRA model | Adapter unit tests only |
| GGUF | Tiny local GGUF compiles, reloads, verifies, and generates on CPU; broader variants remain untested |
| SafeTensors | Tiny local Llama-style checkpoint compiles, reloads, verifies, and generates on CPU |
| Ternary/sub-2-bit | Quantizer logic only |

Aether cannot honestly claim any Hugging Face model support based on this evidence.

## 15. Evaluation and quality gates

The gate framework contains:

- required benchmark names
- regression calculation
- missing-benchmark failure
- rollout-blocking decision logic

A direct deliberately poor MMLU replay score correctly failed the gate.

The quality system is still incomplete for the full PRD, but the execution
boundary is now real when configured:

- `BenchmarkRunner` accepts a configured evaluator that returns measured scores,
  counts, and latency, and validates those fields before gating;
- `JsonlBenchmarkEvaluator` executes a supplied model callback against explicit
  local JSONL records using exact-match/accuracy rules;
- a real local compiled AEG was evaluated and a deliberately non-matching
  response failed the gate;
- no official HellaSwag, MMLU, GSM8K, Math-500, or HumanEval dataset adapters
  are bundled;
- `score_override` remains a deterministic CI replay mechanism and is not a
  substitute for benchmark execution;
- compilation-wide rejection is wired when the caller supplies a measured
  evaluator (and is persisted into the AEG); the default compiler invocation
  remains opt-in because no official benchmark datasets are bundled.

Status: PARTIAL.

## 16. Performance audit

No PRD performance claim was validated.

Not reproduced:

- tokens/sec
- TTFT
- TBT
- P50/P95/P99 real inference latency
- GPU utilization
- VRAM
- energy
- CO2
- real-model compilation time
- real AEG size
- KV reduction
- speculative acceptance
- comparisons against llama.cpp, vLLM, SGLang, TensorRT-LLM, or MLX

All PRD performance figures must be labeled:

    CLAIM NOT VALIDATED

## 17. Security audit

Positive findings:

- AEG manifest and declared artifact hashes are verified.
- Tampered declared artifacts are rejected.
- API-key authentication works when configured.
- MCP calls fail closed when a server is unavailable.
- ONNX generation refuses to fabricate output.
- PyTorch generation refuses synthetic fallback.
- `RuntimeConfig(enable_safety_layer=True)` now enforces prompt injection/toxicity
  checks before inference and output filtering/audit logging after inference.
- TEE reports hardware_backed=false in simulation mode.

Risks:

1. Remote model code is now disabled by default and requires the explicit
   `RuntimeConfig.allow_remote_code=True` opt-in, but executing reviewed custom
   model code is still not sandboxed.
2. Native kernel compilation executes toolchains and subprocesses without a
   complete isolation boundary.
3. TEE software simulation can return successful initialization despite providing no hardware confidentiality.
4. Hub offline fallback is a local archive cache, not a live Hub; it must not be represented as remote publication.
5. Multi-tenant isolation is not demonstrated.
6. Authorization is a basic token check, not production-grade tenant authorization.
7. No malicious model/AEG fuzzing or sandbox validation was demonstrated.

Status: PARTIAL.

## 18. Observability audit

Implemented:

- request spans
- request metrics
- latency percentiles
- throughput metrics
- KV/speculation fields
- Prometheus text rendering
- OTLP-shaped JSON output

The implementation is custom rather than a real OpenTelemetry SDK exporter. It writes JSON files and was not shown exporting to Jaeger, Tempo, or an OTLP collector.

Status: PARTIAL.

## 19. Hub and cache audit

HubClient contains real HTTP request code and authentication checks.

When the Hub is unavailable:

- upload stores a local/in-memory manifest and the uploaded ZIP bytes;
- search searches local manifests;
- download extracts the retained uploaded package and refuses metadata-only
  payloads;
- no Hub server implementation exists in the repository;
- no real remote permissions, deduplication, or content-addressed artifact workflow was demonstrated;
- Hub CLI commands now exist; offline mode remains a local cache fallback rather than a live Hub proof.

Status: PARTIAL.

## 20. Distributed execution audit

The repository contains fleet placement, deployment manifests, sharding plans, CPU reference collectives, hot reload routing, and scheduling structures.

It does not demonstrate:

- multi-process inference;
- multi-node communication;
- NCCL/RCCL/NIXL/UCCL;
- GPU allocation;
- node failure recovery;
- session migration;
- tenant-isolated KV;
- prefill/decode separation across machines.

Status: NOT IMPLEMENTED for real distributed execution.

## 21. End-to-end demo results

The required flow was attempted as far as the environment allowed.

### Environment and hardware

Environment checks pass only with UTF-8 output. Hardware detection reports CPU-only operation.

### Real model compile

Attempted with Qwen/Qwen3-0.6B and other test paths.

Result:

    FAIL

Cause:

    Unable to materialize Hugging Face model; no weights were loaded.

The current code fails closed rather than creating fake weights.

### Synthetic CPU compile/run

Native CPU kernel and synthetic AEG tests pass, but these are not real model compatibility tests.

### Official smoke test

Result:

    PASS â€” 15/15 tests

### AEG save/load

Minimal synthetic AEG save/load/integrity:

    PASS

Real model AEG reload/inference:

    PASS â€” covered by tests/integration/test_local_safetensors_aeg_roundtrip.py

### REST

The FastAPI application starts, OpenAPI exposes 67 `/v1` routes, and a real local
AEG has been exercised through TestClient generation. Unsupported hardware and
missing model capabilities fail as errors; full endpoint semantics remain
partially tested.

### Serve

The real `aether serve <model.aeg>` subprocess now binds a TCP port, answers
`/health`, serves `/v1/generate` from the preloaded local AEG, and shuts down
cleanly in `test_cli_serve_exposes_real_tcp_api_for_local_aeg`. Production
deployment lifecycle, process supervision, TLS, and multi-worker behavior
remain unvalidated.

## 22. Fake, stub, placeholder, and decorative findings

Important findings from source inspection:

- TensorRT-LLM has no executable engine loader in this repository and now fails
  explicitly rather than returning placeholder text.
- AEG/2.0 creation writes explicit disabled descriptors for optional v4
  features; writer methods persist feature flags only with concrete payloads,
  and validation rejects malformed payloads and enabled claims without the
  required artifacts. This is honest format behavior, not implementation of
  the unavailable hardware/compiler features themselves.
- Hub client explicitly supports local archive fallback; this is not a live
  Hub deployment.
- CXL has in-memory fallback storage.
- TEE has software simulation fallback.
- distributed collectives are CPU reference implementations.
- calibration datasets contain synthetic samples.
- target profiles and backend candidates are present without target binaries.
- Pass 12 now rejects unreadable/empty sources; the runtime merge API has a
  real local AEG round-trip test, while multi-model quality validation remains
  unproven.
- R9 diffusion drafting now fails closed when no MDLM drafter head is loaded;
  it no longer emits deterministic random logits as a success path.
- R1 P-EAGLE no longer fills missing MTP slots with greedy/hash/random
  surrogates; absent draft weights are an explicit runtime error.
- Pass 22 RLVR no longer rewards fluent, numeric, or syntactically valid text
  without supplied ground truth or executable tests; its real verifier paths
  remain limited to configured SymPy/subprocess checks.
- Pass 10 no longer marks architecture-only MTP declarations as compiled:
  missing head weights cause an explicit skipped result instead of zero-filled
  speculation blobs.
- Pass 13 no longer creates TTT slots from architecture layer counts alone;
  concrete executable graph layers are required before a TTT artifact is
  emitted.
- Pass 21 now rejects an empty/missing adapter; real adapter runtime integration
  remains incomplete.
- many tests use MagicMock or synthetic graphs and do not exercise real model execution.

Legitimate abstract methods such as base backend NotImplementedError were not counted as defects by themselves. Concrete TensorRT-LLM placeholder behavior was counted as incomplete.

## 23. Critical bugs and unresolved blockers

1. AEG/2.0 and AEG/3.0 version selection is now integrated, and v2 default
   descriptors are explicitly disabled and self-validating. The format
   payloads are still not all executable because the underlying v4/v5
   backends are incomplete.
2. TensorRT-LLM backend has no executable engine loader in this repository.
3. Several v4/v5 optimizer artifacts are plans/configuration, not executable kernels or runtime tensors.
4. gRPC is now a generic Struct-based transport, but typed stubs, TLS, and
   true token streaming are absent.
5. Runtime layers are only partially connected to normal generation; hardware/network layers remain unavailable here.
6. Hub offline download is local fallback behavior and does not prove a live Hub deployment.
7. The public `Runtime.eval_gate` now fails closed when no real benchmark
   evaluator is configured; HellaSwag/MMLU/GSM8K/Math-500/HumanEval
   evaluators are still not wired into the deployment path.
8. Clean isolated installation was not successful in the audit environment.
9. Windows environment checking still depends on UTF-8 output in the current script.
10. Remote code is now disabled by default; explicit opt-in execution remains an isolation risk. Archive
    extraction now rejects traversal, absolute, and link entries and is covered
    by hardening tests.
11. Distributed collectives are CPU reference operations, not multi-node execution.

## 24. Remaining missing functionality

Priority 0:

1. Complete AEG/2.0 and AEG/3.0 payload schemas, migration, and executable
   artifact validation beyond the now-integrated version selection.
2. Replace remaining unavailable backend paths with real engine integrations.
3. Complete gRPC with generated typed stubs, true token streaming, TLS, and
   production authentication/authorization.
4. Connect v4/v5 runtime layers to executable inference paths.
5. Add real evaluation adapters and enforce quality gates before artifact acceptance.

Priority 1:

11. Add real target kernel compilation for CUDA, ROCm, Metal, OpenVINO, QNN, and CPU targets.
12. Implement VLM/video ingestion and generation.
13. Implement real MTP/speculative decoding.
14. Implement integrated ternary/sub-2-bit runtime execution.
15. Add production embedding-model packaging/persistence and distributed
    semantic-cache coordination.
16. Implement real KV transfer and distributed execution.
17. Replace Hub local simulation with a real server or clearly label it client-only.
18. Harden model loading trust boundaries and generated artifact execution.
19. Add clean-install CI on Windows and Linux.
20. Add real model compatibility and quality regression suites.

## 25. Unrealistic or ambiguous requirements

### Compile once, run anywhere

This requires either:

    Portable IR plus target-specific recompilation

or:

    One AEG containing independently compiled kernels for every target

The current repository primarily contains target profiles and plans, not compiled kernels.

### Any Hugging Face model

This requires an explicit architecture compatibility matrix, tokenizer handling, weight layout support, graph extraction, runtime adapters, and real model tests. It cannot be claimed without those constraints and tests.

### Portable TEE across vendors

NVIDIA CC, Intel TDX, and AMD SEV-SNP require different hardware, drivers, attestation, and trust models. Software simulation cannot count as equivalent.

### Future hardware targets

sm130, GB300, MI455X, CXL rack pools, ternary FPGAs, and new RISC-V NPUs cannot be classified as functional based on configuration profiles alone.

## 26. Exact fixes required before a full-implementation claim

- Extend the proven local SafeTensors compile/reload/run path to every PRD-
  claimed model family and validate it in fresh processes.
- Package and validate tokenizer/model adapters for every supported format.
- Complete AEG/2.0 and AEG/3.0 operational payloads and migration.
- Complete semantics for the expanded CLI/REST surfaces and finish gRPC typed
  stubs, TLS, authorization, and token streaming.
- Replace placeholder backend behavior.
- Connect all claimed runtime layers to real inference paths.
- Add real evaluators and enforce regression blocking.
- Build and execute target-specific kernels.
- Add model-family compatibility tests.
- Add clean installation and distribution tests.
- Harden remote code loading, generated-kernel execution, Hub, MCP, TEE, and tenant boundaries.

## 27. Final scorecard

| Category | Completion | Functional | Tested | Production ready |
|---|---:|---:|---:|---:|
| Model ingestion | 65% | 45% | 55% | 25% |
| AEG format | 70% | 58% | 70% | 30% |
| Optimizer | 85% | 50% | 75% | 30% |
| Hardware backends | 35% | 10% | 10% | 5% |
| Runtime | 70% | 50% | 65% | 25% |
| CLI | 70% | 55% | 55% | 30% |
| Python SDK | 60% | 40% | 45% | 20% |
| REST API | 60% | 40% | 50% | 20% |
| gRPC | 35% | 30% | 35% | 10% |
| Evaluation | 25% | 20% | 40% | 5% |
| Performance | 10% | 5% | 10% | 0% |
| Observability | 60% | 50% | 65% | 25% |
| Safety | 45% | 35% | 40% | 15% |
| Hub | 35% | 15% | 35% | 5% |
| Distributed execution | 20% | 10% | 25% | 0% |
| Documentation | 55% | 50% | 25% | 20% |
| Installation/distribution | 45% | 30% | 20% | 15% |

### Aether true completion score

Using higher weights for ingestion, AEG, optimizer, hardware, runtime, and installation:

    PRD/code coverage:       59%
    Functional coverage:     39%
    Tested coverage:         49%
    Production readiness:    19%

These are requirement-weighted audit estimates, not line-coverage percentages.

## 28. Final answers

### If I give this repository to a new developer today, can they install Aether, take a real Hugging Face model, compile it into a real AEG artifact, run it, serve it through the API, and receive correct model output without manually fixing source code?

PARTIALLY.

A new developer can use the proven local CPU path with a real local
tokenizer-backed SafeTensors checkpoint: compile, reload, run, and call the
public REST surface. They cannot yet rely on the full PRD promise for arbitrary
Hugging Face models, remote model download in this environment, v4/v5 artifact
semantics, GPU/hardware targets, gRPC, distributed execution, or quality gates.

### If I claim on GitHub that Aether is fully implemented according to both PRDs, is that technically honest?

NO.

That statement becomes honest only after extending the proven local
compile/save/load/run path to the claimed model families, implementing
AEG/2.0 and AEG/3.0 operationally, completing or removing v4/v5
metadata-only features, completing REST/CLI endpoint semantics and gRPC,
replacing placeholder backends, proving real model compatibility, running
real evaluation gates, validating supported hardware, completing clean
installation, and hardening production security.

## 29. Questions requiring a decision

These do not change the current verdict, but they affect future validation:

1. Do you have access to real Hugging Face model files or a network-enabled environment so the local-model compile/save/reload/run path can be re-tested?
2. Do you want hardware-specific targets classified as implemented-but-unverified only after physical validation, or should profile-only targets remain incomplete until executable kernels are demonstrated?
3. Should v4/v5 requirements remain mandatory as written, or should the project explicitly narrow its supported scope to the functioning CPU/v3.1 subset?

## 30. Recommendation

### FIX FIRST / MAJOR REWORK

Do not ship the repository as fully implemented according to both PRDs.

The v3.1 baseline contains meaningful implementation, particularly on CPU and
in compiler structure. The local SafeTensors compile-once/reload/run workflow
is now reliable and tested, but the v4 and v5 requirements, broad model
compatibility, hardware portability, evaluation gates, and production
distribution are not functionally complete.
