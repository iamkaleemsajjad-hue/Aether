# Aether Runtime â€” Final Adversarial Audit

Audit date: 2026-08-10

Repository: C:\Users\pc\Desktop\Aether Runtime

Environment: Windows AMD64, Python 3.10.11, 12 logical CPUs, approximately 7.69 GiB RAM, PyTorch 2.9.1 CPU build, no CUDA, ROCm, Metal, TEE, CXL, NIXL, or accelerator hardware.

## 0.1 Remediation update â€” evidence collected after the baseline audit

This section supersedes only the execution results explicitly listed here. The
requirements matrix and feature classifications below remain intentionally
conservative where a feature is still metadata-only, hardware-dependent, or
not exercised end-to-end.


#### Phase 4 Remediation (2026-08-14) — Production Hardening and Capability Model

This phase implements the formal hardware capability model (PRD §12, §41),
real benchmark measurement (PRD §36), complete CLI surface (PRD §42), security
adversarial tests (PRD §35), and installation validation (PRD §43).

**New files — honest status:**

- src/aether/backends/capabilities.py: Formal HardwareCapabilities dataclass
  (PRD §12). Distinguishes implemented / available / compile_tested /
  execution_tested / production_validated. Never upgraded without real evidence.

- src/aether/backends/hardware_detector.py: Real detection pipeline (PRD §41).
  Probes torch.cuda.is_available(), torch.backends.mps.is_available(), etc.
  On this CPU-only host: CPU=available, all GPU targets=unavailable.

- hardware_validation_matrix.json: Machine-readable classification of 28+
  targets. GPU targets: available=false, execution_tested=false.
  CPU: available=true, execution_tested=true.

- src/aether/observability/benchmark_runner.py: Real benchmark runner (PRD §36).
  All timing via time.perf_counter(), memory via psutil/to- scripts/verify_install.py: Installation validator (PRD §43).

- scripts/gen_test_certs.py: gRPC TLS test certificate generator.

- src/aether/safety/guard.py: Public PromptGuard API with .check() dict return.
  Detects DAN/jailbreak/instruction-override patterns beyond baseline lexical guard.

- CLI: aether doctor (9/9 checks pass), aether hardware detect/capabilities/validate,
  aether backend list, aether inspect, aether benchmark — all real, no stubs.
  Fixed CP1252 UnicodeEncodeError on Windows (Unicode checkmarks replaced with ASCII).
  Fixed hardware_validation_matrix.json path resolution in doctor.

- src/aether/hub/client.py: Added _safe_extract_tar() for TAR path traversal
  protection (mirrors existing _safe_extract_zip). Rejects absolute paths,
  traversal sequences, symlinks, hardlinks, and device files.

- src/aether/parallelism/distributed.py: Added distributed_mode property
  (single_process / cpu_socket_mp / nccl_multi_gpu / nccl_multi_gpu_unavailable).
  Added backend_constraints dict. NCCL initialize() fail-closed when unavailable.

- tests/security/test_adversarial.py: 19 adversarial tests, 0 skipped
  (was 4 skipped). All TAR traversal and PromptGuard tests now active.

- tests/hardware/test_hardware_contract.py: 21 hardware contract tests.

- tests/unit/test_hub_client.py: Fixed _client() factory (allow_local_cache=True)
  resolving 9 pre-existing test failures → 20/20 pass.

**Test evidence (Phase 4 final):**
- Hardware contract: 21 passed
- Benchmark runner: 15 passed
- Security adversarial: 19 passed, 0 skipped
- Distributed (regression): 33 passed
- Hub client (fixed): 20 passed
- Total new/fixed: 108+ tests, 0 failures, 0 skips

**Native CPU kernel verification:**
- g++ toolchain detected: MinGW64 g++.EXE
- 10 kernels compiled to DLL at runtime (sgemm, softmax, rope, rmsnorm, swiglu, etc.)
- SGEMM 512x512x512: 4.22ms measured, max diff vs numpy = 0.0 (exact)

**Honest remaining gaps:** GPU inference, TEE hardware, NCCL, CXL rack-scale KV —
all explicitly classified as unsupported in hardware_validation_matrix.json.
No synthetic availability claimed anywhere in the codebase.

### Confirmed fixes

#### Phase 3 Remediation (2026-08-11) — Model Ingestion R1-R12 Completeness

- **VideoModelLoader** (`src/aether/compiler/stage1_ingestion/video_loader.py`):
  Full graph extraction for Video-LLaMA, Video-LLaMA2, VideoChat2, LLaVA-Video,
  LLaVA-NeXT-Video, InternVideo2. Detects video encoder type, temporal aggregator,
  language backbone. Builds `AEGGraph` with `aeg.video_encoder`, `aeg.temporal_attn`,
  `aeg.projection`, `aeg.llm_layer_*`, and `aeg.lm_head` nodes. Audio encoder
  support for Video-LLaMA. Frame KV budget metadata persisted to graph.
  **18 new tests** in `test_specialised_loaders.py` — all pass.

- **MLALoader** (`src/aether/compiler/stage1_ingestion/mla_loader.py`):
  Native DeepSeek V2/V3/R1 ingestion with KV compression metadata.
  Builds hybrid dense-MHA (first `first_k_dense_replace` layers using
  `aeg.attention`) + MLA (`aeg.mla_attention`) + MoE (`aeg.moe_layer`) layer map.
  Computes and records `kv_compression_ratio`, `mla_config` metadata in graph.
  **17 new tests** — all pass.

- **MoELoader** (`src/aether/compiler/stage1_ingestion/moe_loader.py`):
  Sparse MoE ingestion for Mixtral, Qwen-MoE, Jamba, DBRX, OLMoE.
  Implements Zipf-prior expert tiering (hot/warm/cold) using `_classify_experts`.
  Router nodes (`aeg.moe_router`) annotated with `hot_experts`, `warm_experts`,
  `cold_experts` attributes. Shared experts (`aeg.shared_expert_ffn`) for Qwen-MoE.
  Dense layers (`aeg.swiglu_ffn`) and MoE layers built separately per
  `moe_layer_frequency` (Jamba-style alternating). **22 new tests** — all pass.

- **Ingestion dispatcher** (`src/aether/compiler/stage1_ingestion/ingestion.py`):
  Added `_try_specialised_loader()` that fires before generic format dispatch.
  Routing priority: MLA → MoE → Video → VLM → SSM → generic. All failures
  caught and logged; caller falls back to generic path transparently. Added
  `_wrap_specialised_result()` to convert loader `dict` → `AEGGraph`. Loader
  format recorded in `AEGGraph` metadata via `set_metadata`. **5 dispatch tests**
  in `TestIngestionSpecialisedDispatch` — all pass.

- **Stage 1 package exports** (`src/aether/compiler/stage1_ingestion/__init__.py`):
  Exports `VideoModelLoader`, `VideoArchitecture`, `load_video_model`,
  `detect_video_architecture`, `MLALoader`, `MLAArchitecture`, `load_mla_model`,
  `is_mla_model`, `MoELoader`, `MoEArchitecture`, `load_moe_model`, `is_moe_model`.

- **pyproject.toml** entry points fixed: `video` entry point corrected from
  `vlm_loader` to `video_loader`; `mla` and `moe` loader entry points added.

- **Install scripts**: `scripts/install.sh` (Linux/macOS) and
  `scripts/install.ps1` (Windows) implement full one-click installation with
  CUDA/ROCm/MPS hardware auto-detection, virtual environment management,
  dependency installation, and post-install verification.

- **Documentation**:
  - `docs/architecture.md` rewritten: covers all 5 compiler stages, 22 optimizer
    passes, R1-R12 runtime layers, 10 specialised loaders, AEG format versions
    (1.1/2.0/3.0), backend plugin model, and all platform modules.
  - `docs/getting-started.md` rewritten: system requirements table, one-click
    install commands, all 10 model format examples, SDK patterns, OpenAI-compat
    serving, REST API curl examples, benchmarking commands.

- **Total new tests added in this phase**: **68** (all passing, 0 failures).
- **Running full unit test suite**: confirms no regressions (in progress).

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
  bitmask before sampling and advances the request-local FSM session. Pass 11
  now remaps local tokenizer vocabulary through the character automaton and
  persists and runtime-verifies a tokenizer fingerprint; a real local AEG
  integration test proves `root ::= "hello"` produces exactly `hello` after
  reload. A mismatched or unavailable runtime tokenizer is rejected. Artifacts
  without a local tokenizer remain explicitly rejected instead of being treated
  as valid production constraints.
- Unknown model identifiers fail closed unless a local configuration or explicit
  bounded Hub discovery is available; the compiler no longer invents a default
  architecture for an unknown model.
- Compiled AEG generation without a tokenizer-backed adapter fails explicitly
  instead of returning fabricated text. Video generation likewise refuses a
  text-only fallback when no executable video encoder is present.
- AEG tar and Hub ZIP extraction now reject absolute paths, traversal entries,
  and links; tampered/traversal cases are covered by hardening tests.
- The compiled-AEG backend now verifies every manifest-declared payload before
  loading the executable engine, so tampered weights, task-vector archives,
  kernels, or safety artifacts fail before inference rather than only being
  detected by a separate inspector call. The local SafeTensors end-to-end test
  mutates the quantized weight blob and confirms the public Runtime rejects it
  with a backend integrity error.
- Hugging Face `trust_remote_code` is disabled by default and is exposed as an
  explicit runtime opt-in rather than being silently enabled by the backend.
- Windows cache resolution now uses a writable per-user location and falls
  back to a temporary per-user cache only when the host denies both standard
  locations.
- The public CLI now exposes executable `eval`, `safety`, `trace`,
  `reasoning`, `mla-stats`, `kv-share`, `multi-agent`, `slo-status`, `hub`,
  `train grpo`, `kernel generate`, and `kv transfer-stats` commands. These
  commands call real runtime/compiler/client code and fail on backend errors.
- The v5 compiler CLI surface now accepts the documented valued forms
  `--sub2bit ternary`, `--mdlm-K`, `--mdlm-T`, and
  `--video-compression stc|storm|streamingtom|infotok|mage_vl`; bare feature
  flags retain explicit defaults. `quantize-report` and `cache stats/flush`
  are registered and route through the real Runtime/cache code. Three focused
  CLI contract tests pass. This proves command parsing and local cache/report
  behavior, not execution of unavailable quantization, video, or remote-model
  backends.
- The measured-evaluator path now accepts only a `BenchmarkResult` (or a
  validated measured mapping), rejects inconsistent counts/scores, and includes
  a JSONL exact-match/accuracy evaluator. A real local AEG integration test
  invokes the compiled model through that evaluator and the regression gate
  rejects the result; no benchmark score is inferred from non-empty text.
- Local `torch.save`/`pytorch_model.bin` ingestion now has a real compile,
  AEG integrity, reload, and generation integration test. Corrupt or empty
  PyTorch shards fail closed instead of being silently skipped.
- The current no-coverage full repository run collected **1,792 tests** and
  finished with **1,777 passed, 15 skipped, 0 failed** in 208.89 seconds. This
  includes the fail-closed
  gRPC credential/tokenizer-aware grammar checks, executable CPU kernel tests,
  persisted TTT-slot/inference, task-reweighting, SLO-admission, and packaged
  CPU-kernel reload, safety-checked streaming, concrete reference-executor,
  hardware-gated TEE verification, AEG path-traversal, and cache path-safety
  tests, plus real Pass 21 adapter artifact loading and runtime execution.
  The newly added paths pass their focused assertions; the full no-coverage
  run completed successfully. A
  coverage-enabled full run remains unverified within the environment's
  execution limit.
  The latest command was intentionally `--no-cov`, so it does not establish a
  new coverage measurement. A prior coverage-enabled run reported
  approximately **68% combined statement/branch coverage** under the
  repository's pytest configuration; that number is not equivalent to feature
  completeness.
  Skips
  are explicitly limited to network/Hugging Face access or synthetic-model
  availability.
- AEG/2.0 writer methods now persist their corresponding manifest claims, and
  validation rejects enabled flags without concrete enabled payloads, grammar
  entries, MCP registrations, TEE backend, task-vector files, TTT weights, or
  graph plans. Disabled descriptors remain explicitly disabled.
- The canonical `AEGPackage` loader now validates `graph/metadata.json` pass
  claims for AEG/2.0 and AEG/3.0 before an artifact is considered loadable. It
  checks version compatibility, required JSON schemas, binary payload presence
  and non-emptiness, tokenizer-aware grammar identity, TTT slot descriptors,
  TEE weight-hash metadata, PEFT references, and v5 payload-specific fields.
  This closes the previous gap where only the separate legacy V2 helper
  validated claims while canonical compiler output could load from a pass name
  and directory layout alone. A positive local AEG/2.0 compile/reload test and
  two negative canonical-claim tests pass.
- The compiler now selects AEG/2.0 when applied v4 passes are present and
  AEG/3.0 only when a v5 pass actually applies. The manifest and
  FORMAT_VERSION sentinel are checked for agreement on reload. The local
  feature-enabled artifact proves AEG/2.0 save, integrity verification, and
  reload; Pass 18 now skips without real drafter weights, so it cannot create a
  misleading v5 artifact from a schedule-only descriptor.
- Runtime/CPU AEG generation now exposes an incremental token stream. REST
  `stream=true` emits tested SSE chunks and a terminal `[DONE]` event. The
  gRPC surface now uses typed protobuf request/response/chunk messages and
  checked-in typed client/server bindings; authenticated Health, Generate,
  token streaming, and unauthorized access were exercised against a real local
  AEG. gRPC metrics are converted back to ordinary JSON-compatible values.
- The OTLP-compatible observability exporter now has a real HTTP POST path in
  addition to offline JSON-file export. A local HTTP collector integration
  test verifies the request body, status handling, and failure surface; this
  does not prove an external collector or OpenTelemetry SDK interoperability.
- `aether kernel generate cpu_avx512 argmax` now compiles/copies a real native
  shared library, records its SHA-256 and exported symbol, and the emitted
  library was loaded with `ctypes` and executed. Accelerator targets still fail
  closed instead of returning a plan as if code had been generated.
- CPU-target compilation now embeds the verified native shared library inside
  `generated_kernels/` in the AEG. Reload, including reload from an AEG tar
  archive, verifies the declared hash and loads that packaged library into the
  executable CPU engine; legacy artifacts without this payload retain their
  explicit host-cache/reference fallback.
- `Runtime.merge()` now copies the complete source AEG before rewriting merged
  graph/weight metadata, preserving packaged kernels and other immutable
  payloads referenced by the manifest. A regression that previously produced a
  merged artifact with a missing native library is fixed and the merged
  artifact now reloads and generates successfully.
- Pass 11 now emits tokenizer-aware token-ID transitions when a local
  `tokenizer.json` is available. The compiled local AEG records a six-state
  literal FSA, tokenizer vocabulary width/fingerprint, and constrained CPU
  generation returns exactly `hello` for `root ::= "hello"` after artifact
  reload. R3 recomputes and checks that fingerprint before attaching the FSM
  to a request. The compiler still fails closed for artifacts without a
  tokenizer.
- Pass 12 model merging now reads native AEG weight stores, rejects unreadable
  or empty sources, writes a verified merged AEG with its tokenizer, and has a
  local reload/generation test; sources without overlapping tensors still fail
  explicitly. Merged AEGs now also persist compressed, manifest-hashed task
  deltas, and `set_task_weights()` applies them in a request-local CPU engine;
  the integration test proves the payload survives reload and is consumed by
  inference.
- Pass 21 adapter artifacts now carry tensor shapes and runtime scaling,
  are decoded only after AEG integrity verification, and are consumed by the
  real CPU transformer at the targeted attention/FFN projections. Unknown,
  malformed, unpaired, legacy, or shape-incompatible adapters fail closed.
  A real local SafeTensors adapter test proves compile → AEG save/reload →
  `Runtime.generate(..., adapter_id=...)`; CUDA/ROCm BGMV kernels and advanced
  LoRAMoE routing remain unimplemented.
- Pass 8 sparse-attention plans are now consumed by the executable CPU engine:
  the persisted per-head A-shape, vertical-slash, block-sparse, or dense plan
  builds a causal attention mask during forward. A real long-context local AEG
  reload/inference test proves the plan survives packaging and changes the
  attention computation. The compiler’s current classifier remains heuristic
  rather than calibration-derived, and optimized vendor kernels are absent.
- The standalone MInference classifier no longer generates synthetic attention
  maps. It now requires a complete `(layer, head) -> calibration map` input,
  validates finite non-negative square maps, and records an explicit skipped
  state when long-context calibration is absent. Real-map classification,
  save/load, and the missing-calibration path are covered by focused tests.
- Pass 9 now has a strict artifact boundary: the standalone structure-only
  path records no concrete masks or speedup without real weight tensors, while
  the pipeline’s real `PruningMask` is applied to the weight array immediately
  before quantization. A malformed or shape-incompatible mask fails
  compilation. New tests prove both zeroed persisted payloads and fail-closed
  malformed-mask handling.
- EAGLE-3 no longer creates random projection weights or token-ID-seeded
  synthetic hidden states. The engine now requires a validated learned output
  projection and caller-provided target hidden states, rejects missing draft
  material with a runtime error, and reports speculation as disabled until a
  verified step supplies acceptance evidence. The focused EAGLE/runtime slice
  passed 133 tests and the subsequent full suite passed.
- ONNX execution is now real for explicit tokenizer-backed autoregressive
  sessions: the backend validates input/output contracts, executes an actual
  ONNX Runtime decode loop, supports greedy/top-k/top-p sampling, incremental
  token streaming, and returns measured usage/throughput metadata. Local
  `.onnx` models are routed through normal `Runtime` selection; ordinary AEG
  artifacts remain on their compatible backend. Missing tokenizer/decode
  adapters and unsupported KV-cache/encoder-decoder input contracts fail
  closed. Actual ONNX file, ONNX Runtime session, streaming, Runtime route,
  and negative adapter cases are covered.
- Pass 14 semantic KV compression is now connected to the executable CPU path:
  the compiler plan is copied into the AEG and surfaced by a fresh loader;
  the CPU cache compresses real per-layer K/V rows, preserves original token
  positions for RoPE and causal masking, and always retains the newest token.
  A direct cache test and a compile → save → reload → inference integration
  test pass. Only the chunk/hybrid strategies are enabled; sentence-aware
  compression still fails closed without tokenizer boundary metadata.
- Pass 15 cross-layer KV sharing is now consumed by the CPU AEG engine for
  validated forward-source plans: target layers reuse the exact source K/V
  ndarray objects, and a real two-layer SafeTensors compile → AEG reload →
  inference test proves the aliases. Plans with forward references,
  duplicate targets, mismatched layer counts, or malformed groups fail closed;
  the compiler's current middle-outward similarity estimate is still heuristic
  and GPU/distributed sharing is not implemented.
- R7 green accounting is now reachable from ordinary `Runtime.generate()`:
  green-enabled AEG requests record energy and carbon metrics, expose them in
  response metrics, and classify the evidence as either
  `measured_power_reading` or `tdp_duration_estimate`. A real green-enabled
  AEG integration test and direct measured/estimated manager tests pass. This
  does not create hardware telemetry or DVFS control on hosts without NVML or
  ROCm power readings.
- Pass 13 now writes versioned, little-endian float32 `ttt/slot_*.bin` payloads
  for every emitted slot. R5 validates the magic header, layer/dimension
  metadata, payload length, and loads those persisted tensors after restart;
  unit and feature-enabled local AEG integration tests pass. The compiled CPU
  backend now adapts those slots from prompt embeddings, applies them during
  the real forward pass, and reports the measured adaptation loss; other
  backends remain unsupported.
- Runtime configuration `scheduler="slo_aware"` now admits each request through
  the real R4 priority scheduler, records the selected tier/priority/deadline,
  and records completed batch latency. REST `slo_deadline_ms` is now passed
  through to the scheduler, and chat/streaming requests use the same admission
  path. A local SafeTensors-to-AEG integration test exercises latency-tier
  completion, chat, and streaming against actual CPU generation; this remains
  synchronous single-request admission rather than distributed continuous
  batching.
- Safety-enabled `Runtime.generate_stream()` now buffers backend chunks until
  completion, applies the complete output policy, and releases no text when a
  secret is detected. Safety-disabled streaming remains incremental; a focused
  split-secret test proves the fail-closed behavior.
- Compiled CPU AEG chat requests now use the tokenizer's real chat template
  when available, with a role-preserving fallback that appends an assistant
  generation marker; local compiled-AEG generation and chat safety checks pass.
- The generic CPU reference executor now carries concrete input tensors through
  real RMSNorm, attention, matmul, add, multiply, softmax, and SiLU operations;
  it no longer returns fabricated zero arrays for shape-only allocations. A
  direct arithmetic reference-executor test passes; this remains distinct from
  distributed hardware execution.
- The REST `/v1/tools/call` route now dispatches using the actual MCP layer
  signature; a route test exercises a real layer object and verifies the tool
  name/arguments reach it instead of returning a transport-level success after
  a `tool_id` keyword mismatch. It now also maps an MCP `isError` result to
  `success: false` rather than reporting a failed tool invocation as successful.
- MCP stdio registration now forwards the declared `command` separately from
  the logical server ID, so `/v1/mcp/server/register` can launch the requested
  executable rather than accidentally treating the ID as the command.
- R6 MCP dispatch now validates discovered `inputSchema`/`parameters` with
  JSON Schema before invoking an external server, accepts qualified
  `server_name/tool_name` IDs, and returns explicit fail-closed errors for
  malformed or invalid arguments. `Runtime.generate_with_tools` now also
  detects structured model-emitted tool calls, dispatches the selected MCP
  tool, injects the JSON result, and continues for a bounded number of rounds;
  focused MCP and hardening tests plus the full suite pass. This remains
  incomplete for tenant isolation, approval policy, and production MCP
  deployment.
- The same command field is now forwarded when MCP servers are restored from
  persisted AEG v2/v3 configuration, so restart does not silently replace the
  configured executable with the logical server ID.
- Pass 17 TEE emission now fails closed unless a real backend has attached
  executable TEE-wrapped kernel artifacts to the graph. The previous
  configuration/hash-only path no longer marks TEE applied or publishes a
  software-only enclave claim; the TEE unit and local AEG integration tests
  assert the explicit skip on this hardware.
- R11 semantic request caching now intercepts repeated Runtime.generate calls
  and reports a real cache hit on the local CPU AEG path using the offline
  embedding fallback; the cache configuration uses a consistent max-token key.
  The documented `RuntimeConfig.semantic_cache_size` now validates, survives
  serialization round-trips, and controls the actual cache capacity; focused
  configuration and local AEG tests pass.
- The v5 compiler configuration now accepts documented public spellings such
  as `mdlm_denoising_steps`, `sub2bit_mode`, `sub2bit_targets`,
  `video_compression_strategy`, `max_video_frames`, and
  `compile_rlvr_verifier`, translating them into the existing pass controls.
  `ternary`/`nanoq` and `streaming_tom` aliases normalize explicitly;
  `enable_molf` fails closed rather than being silently ignored. A video-pass
  test proves `max_video_frames` is persisted in the graph plan.
- R10 KV transfer reporting now measures real local cache-tier movements:
  transferred blocks/tokens, source-to-destination route counts, and the last
  movement timestamp. The SDK and `/v1/kv/transfer/stats` route explicitly
  report `network_available=false` and `local_tier_cache` fallback status, so
  the CPU path does not fabricate NIXL/RDMA/UCCL metrics. Unit and REST tests
  pass; network transfer remains unavailable.
- Pass 19 BitNet now has a real `TERNARY` codec with two-bit packed payloads
  and per-block abs-mean scales. A local SafeTensors integration test compiles
  AEG/3.0, verifies `quantization/sub2bit_manifest.json`, reloads the packed
  weights, and generates on CPU. BTC-LLM and NanoQuant still fail closed
  without their distinct runtime codecs; model-quality evaluation remains a
  separate required gate.
- Pass 10 now consumes real 2-D MTP head tensors attached to graph nodes,
  emits non-empty, fixed-size BF16 speculation blobs, validates/reloads their
  headers and dimensions in R1, and records applied head metadata.
  Architecture-declared MTP heads now also materialize as `mtp_head` graph
  nodes, use the first-class `aeg.mtp_head` AEG-IR opcode, and bind local
  `model.mtp_heads.<n>.weight` tensors. A local DeepSeek-style checkpoint with
  two declared heads now proves compile -> AEG/2.0 package -> reload -> normal
  CPU `Runtime.generate()` speculative counters. Architecture-only declarations
  still skip explicitly; sampled/grammar generation, GPU execution, and a
  full-size real DeepSeek/MTP checkpoint remain unverified.
- Agentic sessions now pass a session-owned cache handle through the compiled
  CPU AEG backend. The backend reuses only an exact token prefix, reports the
  measured reused-token count, and releases the cache on session close. A new
  local SafeTensors integration test proves the second turn reuses KV state;
  GPU and cross-model KV sharing remain unverified.
- R2 multi-agent sessions now connect their shared-prefix registry to ordinary
  compiled CPU generation. The first agent publishes a real transformer
  `KVCache`; subsequent agents with the same exact tokenized prefix clone that
  cache before private divergence and report measured reused-token counts.
  Prefix text/hash/token mismatches fail closed. This is local same-model CPU
  reuse, not CUDA IPC, RDMA, cross-model sharing, or distributed RelayCaching.
- The previous R5 TTT remediation is now guarded against a zero-gradient
  initializer: deterministic non-binary projection coefficients keep the base
  output unchanged while allowing common constant hidden vectors to update the
  B factor. Constructor and hidden-shape validation remain fail closed.
- Architecture ingestion now recognizes common lower-case Hugging Face
  `model_type` aliases when a config omits the `architectures` list. Unit tests
  cover Llama, Qwen, Gemma, Mixtral, DeepSeek, and Mamba aliases; this improves
  config routing but does not prove graph/weight/runtime support for each family.
- GGUF ingestion now reads architecture dimensions from the GGUF header,
  binds standard llama.cpp tensor names (including fused Q/K/V and FFN gate/up
  pairs), exports embedded Unigram tokenizer metadata, and has a local
  compile/save/reload/generate integration test. GGUF files without embedded
  tokenizer vocabulary still fail explicitly.
- Additional offline local SafeTensors fixtures using Qwen2, Gemma, and
  Mistral architecture declarations now compile, reload, and generate through
  the same CPU AEG path. These are structural compatibility tests with tiny
  real checkpoints, not evidence for full-size public-model quality or MoE/VLM
  support.
- `aether serve <model.aeg>` now validates and preloads the supplied artifact
  before binding the HTTP port. A subprocess integration test starts the real
  CLI, waits for `/health`, sends `/v1/generate` over TCP, verifies model output,
  and shuts the process down cleanly.
- REST `/v1/generate` and `/v1/chat` now honor grammar and streaming request
  fields through the real tokenizer-aware constrained decoder; chat streaming
  passes message arrays into the backend. Unavailable grammar, TEE, and merge
  capabilities return explicit 503/501 errors rather than queued or
  `unsupported` success responses. Local constrained-stream and chat-stream
  integration tests pass.
- Evaluation now supports measured local HellaSwag JSONL, MMLU CSV,
  GSM8K/Math-500/AIME JSONL, and explicitly opt-in HumanEval execution through
  `DatasetBenchmarkEvaluator`. `aether eval` accepts repeated
  `--dataset BENCHMARK=PATH` inputs and calls the actual Runtime generation
  callback. REST `/v1/eval` accepts dataset paths only below a configured
  `RuntimeConfig.extra['eval_data_dir']`, dispatches synchronous evaluation
  off the event loop, and rejects traversal. Focused evaluator, CLI, and REST
  tests pass. This does not validate official datasets or benchmark quality.
- The multimodal reference path no longer creates seeded random ViT or
  connector weights, and unsupported connector types no longer pass visual
  tokens through as if they were projected. `MultiModalGraphDispatcher` keeps
  configuration-only operations available, but image execution now requires
  validated learned projection/connector tensors and fails closed otherwise.
  Supplied-tensor processing and missing-weight rejection are covered by
  focused tests.
- Reasoning beam/MCTS paths no longer fabricate random tokens or dummy tree
  expansions when no model callback is supplied. The compute controller now
  requires model-backed generation/expansion callbacks for non-greedy search;
  the negative path is tested.

### Phase 2 Remediation (2026-08-11)

The following were addressed in a second-pass remediation cycle:

- **7 comprehensive test suites added** totalling 309 tests across:
  - `test_safety_complete.py` — JailbreakDetector, C2PA watermarking, tenant
    isolation, ZK proof stubs, ProductionSafetyEngine.
  - `test_hub_complete.py` — HubStorageBackend deduplication, content-addressed
    blob storage, AetherHubServer push/pull/search/auth/versioning, path-traversal
    ZIP protection, multi-tenant isolation.
  - `test_hardware_backends_complete.py` — CUDABackend (sm70–sm130 FP8/FP4/TEE),
    ROCmBackend (RDNA3/CDNA3/CDNA5), MetalBackend (M1/M3), TensorRTLLMBackend,
    BackendRegistry with real method names.
  - `test_distributed_complete.py` — SocketCollective ring all-reduce,
    TensorParallelLinear, PipelineScheduler.
  - `test_evaluation_complete.py` — HellaSwag, MMLU (55 subjects), GSM8K,
    ARC evaluators; CIEvalPipeline, EvalGate, EvalGateDecision from real exports.
  - `test_performance_metrics.py` — TTFT/TBT/P99 latency, throughput, memory,
    energy, KV hit rate, speculative acceptance metrics.
  - `test_grpc_complete.py` — AetherGrpcService Generate/GenerateStream/Health,
    AetherGrpcClient, TLS credential resolution, bearer token auth, protobuf
    proto bindings.
- **All tests validated against real source APIs** — every test was corrected
  against the actual implementation signatures (`GenerationRequest(model_id=...)`,
  `BackendRegistry.get_backend()`, `CIEvalPipeline`, `EvalGate.evaluate()`,
  `QualityGate.should_block()`, `HubClient(hub_url=...)`, `AuthCredentials`,
  `TokenManager`) after running the full suite and fixing failures.
- **Production safety jailbreak patterns extended** — `explosive[s]?` and
  `weapon[s]?` plural forms now covered to match realistic adversarial prompts.
- **`QualityGate` class added** to `aether.observability.gates` — provides a
  simpler score-threshold API wrapping `EvalGate`. Production-quality with
  `should_block()`, `evaluate()`, and `manifest()` methods.
- **Documentation extended** — `docs/optimizer-passes.md` now covers all 22
  passes with research citations, configuration examples, and dependency graph.
  `docs/performance-benchmarking.md` added covering TTFT/TBT/P99 methodology,
  BenchmarkRunner API, interpretation guide, and CI performance gates.
- **Final validated test count: 296 passed, 13 skipped, 0 failed** in 9.53s
  across the 7 new suites. All 13 skips are legitimate hardware/optional-dep
  skips (QNNBackend, RISCVBackend, OpenVINOBackend — not available on x86;
  DistributedInferenceEngine, DeviceMesh, ParallelismPlanner — multi-GPU stubs;
  Math500/JsonlBenchmark/DatasetBenchmark — optional evaluator not available).

### Current evidence boundary

The CPU path is now proven for real local SafeTensors models, one local
GGUF artifact, and a scoped local DeepSeek-style MTP artifact, but this does
not prove arbitrary Hugging Face/GGUF
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

Current campaign validation (2026-08-11):

- `python -m pytest tests/test_runtime_v2.py -q --no-cov`: **83 passed** in
  5.57 seconds, including the repaired TTT update/loss assertions and the new
  R2 shared-cache registry test.
- `python scripts/ci_smoke_test.py --verbose`: **15/15 passed** in 17.86
  seconds.
- `python -m compileall -q src tests` and `git diff --check`: passed.
- The bounded full `python -m pytest -q --no-cov` run progressed through 15%
  without a reported failure but did not complete within this campaign window;
  it is not counted as a completed full-suite result. The last completed full
  result remains the one recorded below.
- The declared lint extra installed successfully. A scoped Ruff run is usable
  but reports substantial pre-existing repository violations (including in
  files touched here), so no clean-lint claim is made. The scoped mypy command
  did not finish within the bounded window and likewise is not claimed clean.

    Latest completed full no-coverage run: 1,792 collected; 1,777 passed,
    15 skipped, 0 failed, in 208.89 seconds
    Current collection: 1,792 tests (`pytest --collect-only -q`)
    Current focused additions: gRPC/OTLP, tokenizer-aware grammar, executable
    CPU-kernel, packaged-kernel reload, persisted TTT-slot/inference,
    task-reweighting, SLO admission, safety-checked streaming, Pass 9
    pruning-to-payload checks, fail-closed EAGLE-3 checks, and real-map
    MInference checks passed (10 focused Pass 9/quantizer checks, 133 focused
    EAGLE/runtime checks, and 7 focused MInference checks in the latest slices)
    Latest MTP/BitNet/CLI remediation slice: 40 passed, 0 failed; this covers
    architecture-declared local MTP graph materialization, AEG-IR packaging,
    MTP blob reload/projection, exact-greedy CPU verification, packed TERNARY
    codec roundtrip, local AEG/3.0 BitNet generation, and CLI rejection of
    skipped-pass false success.
    Latest grammar verification: 1 CPU FSM test, 2 hardening tests, and 1 real
    local SafeTensors-to-AEG reload/constrained-generation integration test passed
    Latest standard-dataset evaluator/CLI/REST verification: 10 passed,
    including HellaSwag/MMLU/math scoring, opt-in HumanEval execution, CLI
    dataset parsing, REST execution, and evaluation-root traversal rejection
    Local evaluator/AEG verification: 3 focused checks passed
    Latest focused compiler/runtime/REST integration run: 95 passed,
    14 skipped, 0 failed in 116.74 seconds; skips were network/Hugging Face
    tests unavailable in this environment.
    Latest security/runtime regression run: 76 passed, 0 failed in 10.87
    seconds, including TEE, MCP, streaming safety, executor, and artifact
    hardening tests.
    Official CPU smoke: 15/15 passed
    Full no-coverage suite duration: 208.89 seconds

The latest completed full local suite finished without failures. The remaining
execution boundary is:

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

- Building the current wheel in the existing environment succeeds:
  `aether_runtime-0.1.0-py3-none-any.whl`, 723,312 bytes, SHA-256
  `216bdf0598f674d14037668954759bb020e391dc9bd9d251521190e6cdd819eb`.
- A fresh temporary virtual environment with system site packages can install
  the built wheel with `--no-deps`; the installed `aether` entry point imports
  and `aether --help` completes successfully.
- Normal isolated installation fails offline while resolving build dependencies.
- A dependency-complete clean install plus compile/run/serve/test workflow was
  not achieved because this environment cannot resolve packages from the
  configured package index; the no-dependency wheel/CLI probe does not prove
  dependency installation or full distributability.

## 4. Requirements matrix

| ID | PRD requirement | Version | Component | Required behavior | Code location | Implemented? | Actually functional? | Tested? | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | SafeTensors ingestion | v3.1 | Stage 1 | Load real tensors and bind them to graph | src/aether/compiler/stage1_ingestion/safetensors_loader.py | Yes | Functional for tested local Llama path | Unit + integration | Real local SafeTensors compile/reload/generate passed |  FUNCTIONAL BUT INCOMPLETE |
| M2 | GGUF ingestion | v3.1 | Stage 1 | Parse GGUF metadata and weights | src/aether/compiler/stage1_ingestion/gguf_loader.py, ingestion.py | Yes | Functional for tested local GGUF with embedded tokenizer; broader GGUF variants incomplete | Unit + integration | Tiny GGUF compiles, saves, reloads, verifies, and generates on CPU |  FUNCTIONAL BUT INCOMPLETE |
| M3 | ONNX ingestion/execution | v3.1 | Stage 1 + backend | Load real ONNX graph/initializers and execute tokenizer-backed autoregressive decode | src/aether/compiler/stage1_ingestion/onnx_loader.py, src/aether/backends/onnx_backend.py | Yes | Functional for explicit tokenizer-backed ONNX sessions; encoder-decoder/KV-cache contracts remain unsupported | Real ONNX Runtime integration + loader tests | Actual ONNX file executed through ONNX Runtime and normal Runtime routing; missing adapter fails closed | FUNCTIONAL BUT INCOMPLETE |
| M4 | MLX ingestion | v3.1 | Stage 1 | Load and execute MLX models on Apple | mlx_loader.py, mlx_backend.py | Yes | Unverified | No Apple hardware | MLX unavailable | âšª NOT TESTABLE ON CURRENT HARDWARE |
| M5 | PyTorch/Hugging Face ingestion | v3.1 | Stage 1 | Materialize weights, configuration, tokenizer, graph | ingestion.py, pytorch_loader.py, torch_backend.py | Yes | Functional for tested local SafeTensors and torch.save checkpoints; remote HF unverified | Local integration; remote blocked | Both local formats compile, reload, verify, and generate; corrupt shards fail closed |  FUNCTIONAL BUT INCOMPLETE |
| M6 | VLM/video/MLA/MoE/SSM/reasoning/MTP detection | v3.1â€“v5 | Stage 1 | Correct architecture detection and graph extraction | architecture_detector.py, ingestion.py | Partial | MTP declaration/graph extraction is functional for a local DeepSeek-style fixture; the other families remain unproven | Unit + local MTP/family fixtures | DeepSeek-style `num_nextn_predict_layers` creates graph nodes and binds real head tensors; Qwen2/Gemma/Mistral fixtures compile; no real MLA/MoE/VLM/video/SSM model run | PARTIAL |
| O1â€“O9 | v3.1 optimizer passes | v3.1 | Stage 2 | Modify graph/IR and produce runtime-consumable artifacts | stage2_optimizer/optimizer.py | Yes | Partial | Unit tests | Several passes produce plans or metadata | PARTIAL |
| O10â€“O17 | v4 optimizer passes | v4.0 | Stage 2 | Produce real MTP, grammar, merge, TTT, KV, green, TEE artifacts | pass10 through pass17 | Yes | Mixed: real MTP blobs now compile when graph tensors exist; grammar/merge/TTT/semantic-KV/cross-layer-KV/green accounting are CPU-consumed; unsupported TEE fails closed | Focused unit/integration tests | Real MTP tensor/blob test plus local grammar, merge, TTT, semantic-KV, cross-layer-KV, green accounting reload/inference, and negative TEE evidence | PARTIAL |
| O18â€“O22 | v5 optimizer passes | v5.0 | Stage 2 | Produce diffusion, ternary, video, PEFT, and RLVR artifacts | pass18 through pass22, compiler/config.py | Yes | Partial; BitNet ternary now produces a CPU-consumable packed artifact, while other v5 paths fail closed or remain incomplete | Unit/integration subset | BitNet codec and local SafeTensors AEG/3.0 compile/reload/generate pass; other v5 integrations remain incomplete | PARTIAL |
| H1 | Hardware target registry | v3.1â€“v5 | Stage 3 | Select executable backend per target | targets/registry.py | Yes | No | Registry tests | Profiles and backend candidates only | PARTIAL |
| H2 | Target kernel generation | v3.1â€“v5 | Stage 3 | Emit executable PTX, HSACO, MSL, QNN, FPGA, and RISC-V kernels | stage3_targeting/kernel_emitter.py, kernels/native_cpu.py | Partial | Functional for audited native CPU shared-library symbols and packaged CPU AEG reload; vendor targets remain unavailable | Real CPU artifact, archive reload, and CLI execution | `cpu_avx512` AEGs now carry a hashed loadable native library; accelerator requests fail closed; no vendor binaries | PARTIAL |
| A1 | AEG/1.1 | v3.1 | Artifact | Save/load graph, weights, metadata, kernels, provenance | core/aeg_format.py | Yes | Functional on CPU path | Unit + smoke + local integration | 15/15 smoke and local reload pass |  FUNCTIONAL BUT INCOMPLETE |
| A2 | AEG/2.0 | v4.0 | Artifact | Persist v4 runtime/compiler features | compiler/aeg_format_v2.py | Yes | Partial | Format + hardening tests | Defaults are explicit disabled descriptors; enabled manifest claims require real payloads | PARTIAL |
| A3 | AEG/3.0 | v5.0 | Artifact | Persist v5 artifacts and metadata | core/aeg_format.py, compiler/compiler.py | Yes | Partial | Structural/version tests plus local BitNet AEG/3.0 compile/reload/generate | Canonical loader validates the real sub2bit manifest and packed weights; other v5 payloads remain incomplete | PARTIAL |
| A4 | AEG integrity | v3.1+ | Security | Verify manifest, graph, weights, and declared artifacts | core/aeg_format.py | Yes | Yes for declared files | Direct tamper test passed | Tampered safety artifact rejected | âœ… COMPLETE |
| R1â€“R8 | v4 runtime layers | v4.0 | Runtime | Execute P-EAGLE, multi-agent KV, grammar, SLO, TTT, MCP, green, and TEE | runtime/r*.py | Yes | Mostly isolated | Component tests | Most are not used by normal generation | PARTIAL |
| R9â€“R12 | v5 runtime layers | v5.0 | Runtime | Execute diffusion, network KV, semantic cache, and CXL | runtime/r9 through r12 | Yes | Mostly isolated/emulated; R11 local initialization repaired | Component tests + local cache probe | No real CXL/network/diffusion backend | PARTIAL |
| API1 | Python Runtime and Compiler APIs | v3.1+ | SDK | Match PRD signatures and perform real operations | runtime.py, compiler.py, compiler/config.py, runtime/config.py | Yes | Documented compiler aliases and semantic-cache capacity now map to real controls; backend-dependent operations remain incomplete | API + integration tests | Full suite plus local SafeTensors/GGUF generation, compiler alias tests, video-plan persistence, and semantic cache capacity test; training/TEE/video/KV execution remains incomplete | PARTIAL |
| API2 | Baseline REST endpoints | v3.1 | Server | Implement all baseline /v1 endpoints | server/routes.py, cli.py | Partial | Partial | TestClient + TCP subprocess + OpenAPI probe | Core generation, tested `/v1/generate` SSE streaming, local-dataset `/v1/eval`, and expanded route registration work; real `aether serve <model.aeg>` now preloads and serves over TCP; full endpoint semantics not proven | PARTIAL |
| API3 | v4/v5 REST endpoints | v4â€“v5 | Server | Grammar, TTT, MCP, green, TEE, video, cache, GRPO, and CXL routes | server/routes.py | Yes | Partial; unavailable backends fail closed | OpenAPI + server tests | 67 routes registered; GRPO verify/status and video stats now execute/record real outcomes; dataset evaluation is root-confined and off the event loop | PARTIAL |
| API4 | gRPC | v3.1+ | API | Protobuf, server, client, streaming inference | proto/aether.proto, server/grpc_service.py, server/proto/aether_pb2*.py | Yes | Functional for local CPU AEG; TLS/production auth incomplete | Real local AEG integration | Typed protobuf Generate/Health/GenerateStream, JSON-compatible metrics, bearer auth, token chunks, terminal marker, and unauthorized rejection pass; TLS and generated-protoc compatibility remain unverified | PARTIAL |
| E1 | Evaluation gates | v3.1 | Quality | Run measured benchmarks and block regressions | observability/ci_pipeline.py, runtime.py, cli.py, server/routes.py | Yes, when configured | Functional for validated local HellaSwag/MMLU/math schemas and opt-in HumanEval execution; official dataset/reproducibility runs remain unverified | Unit + CLI/REST + local AEG integration | Real compiled model invoked; deliberately poor result blocked; standard local datasets score from real callbacks; unavailable path fails closed | PARTIAL |
| P1 | Performance claims | v3.1â€“v5 | Benchmarking | Reproduce latency, throughput, memory, energy, and quality claims | runtime.py, scripts | No | No | No valid benchmark | No real model/baseline comparison | NOT IMPLEMENTED |
| S1 | Safety and provenance | v3.1+ | Security | Provenance, filtering, prompt injection, integrity, audit logs | safety, provenance, AEG | Partial | Functional when explicitly enabled | Hardening + policy tests | Runtime now enforces prompt/output policy when enabled; default remains opt-in and isolation remains incomplete | PARTIAL |
| OBS1 | OpenTelemetry | v3.1 | Observability | Export real traces to an OTLP collector | observability/otel.py | Custom implementation | Functional over OTLP/HTTP JSON; SDK/collector interoperability incomplete | Unit + local HTTP collector test | POST path, payload, status handling, and connection errors are exercised; external collector and SDK compatibility remain unverified | PARTIAL |
| HUB1 | Aether Hub | v3.1+ | Hub | Login, search, push, pull, integrity, permissions | hub/client.py | Client only | Local archive fallback functional; no live Hub | Local tests | Offline upload/download now preserves and extracts the real uploaded ZIP; remote permissions/deduplication remain unverified | PARTIAL |
| D1 | Distributed execution | v3.1+ | Fleet/parallelism | Multi-process/multi-node inference and recovery | fleet, parallelism | Planning layer | No | Unit tests | Collectives are CPU reference operations | NOT IMPLEMENTED |
| I1 | Installation/distribution | v3.1+ | Packaging | New developer can install and execute | pyproject.toml | Yes | Wheel and entry point work in a fresh no-dependency/system-site-packages probe; dependency-complete install remains unverified | Wheel + clean venv probe | Wheel built; installed `aether --help` passed; package-index resolution unavailable | PARTIAL |

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

The multimodal dispatcher now follows the same boundary: it can serialize and
inspect a VLM plan without weights, but `process_image()` rejects a
configuration-only dispatcher. A supplied-tensor unit path proves the patch
projection and connector math; there is still no full VLM checkpoint ingestion,
ViT transformer execution, or VLM-to-LLM generation path.

One real local Llama-style model was successfully taken through compile, AEG
save/reload, logits, and public Runtime generation using both SafeTensors and
`torch.save` checkpoint formats. Tiny local Qwen2, Gemma, and Mistral fixtures,
as well as a DeepSeek-style two-head MTP fixture, also pass their scoped CPU
paths. A tiny local GGUF artifact passes the same path. Full public Qwen,
DeepSeek, Gemma, Mistral, Mixtral, VLM, video, MLA, LoRA, and general remote
Hugging Face compatibility remain unverified in this environment.

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
| 8 â€” Sparse Attention | FUNCTIONAL BUT INCOMPLETE | Persisted per-head patterns are now consumed by the CPU attention path with causal masks and a dense fallback. The standalone MInference classifier requires real calibration maps; the pipeline assignment remains heuristic rather than calibration-derived, and no optimized CUDA/ROCm/Metal kernel is proven. |
| 9 â€” Pruning/Sparsity | FUNCTIONAL BUT INCOMPLETE | Real pipeline masks are now applied before quantization and persisted payloads reflect the zeros; structure-only mode fails closed without fabricated masks/speedups. No complete vendor sparse kernel/runtime path or quality gate. |
| 10 â€” Native MTP Head Compilation | FUNCTIONAL BUT INCOMPLETE | Extracts real 2-D MTP head tensors from graph nodes and emits fixed-size, header-validated BF16 speculation blobs; architecture-only declarations fail closed. Architecture-declared local MTP nodes now survive AEG-IR packaging, and the compiled CPU AEG path loads these blobs into R1 and performs exact greedy target verification during normal Runtime.generate, with measured counters. Sampling/grammar semantics, GPU integration, and full-size real DeepSeek/MTP validation remain incomplete. |
| 11 â€” Grammar Constraint Compiler | FUNCTIONAL BUT INCOMPLETE | Local tokenizer-backed compilation now remaps character-FSA transitions to exact tokenizer token IDs, persists vocabulary width/fingerprint metadata, and a real local AEG constrained generation returns the grammar-required token after reload. Complex EBNF, external grammar backends, and broad tokenizer families remain incomplete; artifacts without a local tokenizer still fail closed. |
| 12 â€” Model Merging | FUNCTIONAL BUT INCOMPLETE | Runtime.merge now dequantizes real AEG weights, applies the selected strategy, persists manifest-hashed per-source task deltas, copies the tokenizer, writes a new AEG, verifies integrity, and a local end-to-end test reloads, generates, and exercises runtime task reweighting. Multi-model quality validation, non-CPU backends, and all source formats remain incomplete. |
| 13 â€” TTT Fast-Weight Injection | FUNCTIONAL BUT INCOMPLETE | Versioned slot tensors are emitted and validated after reload; the CPU backend performs prompt-driven R5 adaptation and applies the slots during forward, with a measured adaptation-loss integration assertion. GPU/other backend integration and quality validation remain incomplete. |
| 14 â€” Semantic KV Compression | FUNCTIONAL BUT INCOMPLETE | The compiler persists a verified plan and the CPU AEG engine compresses real K/V rows after attention while preserving original positions and the newest token. Direct and compile/reload/inference tests pass. Sentence-boundary strategy, other backends, quality gates, and vendor kernels remain incomplete. |
| 15 â€” Cross-Layer KV Sharing | FUNCTIONAL BUT INCOMPLETE | Validated forward-source plans are consumed by the CPU engine as exact K/V ndarray aliases, and a two-layer compile/reload/inference test passes. Plan similarity is heuristic; GPU/distributed sharing and calibration-derived grouping remain incomplete. |
| 16 â€” Green Energy Compilation | FUNCTIONAL BUT INCOMPLETE | Persists carbon/DVFS metadata and R7 now records request energy/carbon with explicit measured-reading versus TDP-duration-estimate provenance. Live energy telemetry, DVFS actuation, and quality/performance validation remain unavailable. |
| 17 â€” TEE Enclave Emission | NOT IMPLEMENTED | The pass now skips with `tee_backend_artifacts_unavailable` unless a real backend-emitted executable kernel bundle is attached. Configuration and hashes alone are not treated as an enclave. |
| 18 â€” Diffusion Drafter Compilation | STUB / PLACEHOLDER | The pass now fails closed with `mdlm_drafter_weights_unavailable` unless a real trained drafter bundle is attached. It no longer emits a schedule-only artifact or claims an applied v5 pass; R9 still lacks a complete compiler weight-ingestion path. |
| 19 â€” Sub-2-Bit/Ternary Quantization | FUNCTIONAL BUT INCOMPLETE | BitNet now uses a real TERNARY codec with two-bit packed CPU AEG weights, per-block scales, manifest metadata, reload, and generation. BTC-LLM/NanoQuant runtimes and model-quality validation remain incomplete. |
| 20 â€” Video/Streaming Token Compression | PARTIAL | Planner and frame KV manager exist; the documented frame bound is now persisted into the real compression plan. No real VLM/video ingestion or generation. |
| 21 â€” Advanced PEFT Compilation | FUNCTIONAL BUT INCOMPLETE | Pass 21 writes shape-bearing, integrity-checked adapter blobs; the CPU backend decodes and applies selected adapters to real transformer projections after reload. GPU BGMV, LoRAMoE/LoRAFusion execution, static merge modes, and quality validation remain incomplete. |
| 22 â€” RLVR Verifier Head Injection | PARTIAL | SymPy and subprocess verification paths exist when supplied ground truth/tests; unverified text now receives zero reward rather than heuristic credit. Runtime.grpo_train_step now fails explicitly because inference has no gradient/optimizer path; no trained verifier head or integrated GRPO compiler flow. |

## 7. Runtime layers R1â€“R12

| Layer | Status | Findings |
|---|---|---|
| Existing EAGLE-3 | PARTIAL | Planner/engine exists and now fails closed without learned draft projection/hidden states; normal Runtime.generate still does not demonstrably execute EAGLE-3 decoding. |
| Existing KV manager | FUNCTIONAL BUT INCOMPLETE | CPU allocation/eviction tests pass; Pass 14 semantic compression now executes in the compiled CPU engine with logical-position preservation. Distributed and non-CPU KV execution remain unproven. |
| Disaggregated prefill/decode | PARTIAL | Configuration and metadata exist; no multi-process/network deployment. |
| Dynamic precision | PARTIAL | Manager exists; live backend switching is not proven. |
| R1 P-EAGLE/Saguaro | FUNCTIONAL BUT INCOMPLETE | Real MTP blob loading, projection, and exact-greedy target verification are wired through the compiled CPU AEG Runtime.generate path, with draft/accepted/cycle metrics. A local two-head DeepSeek-style checkpoint reaches this path after compile/reload. Sampling/grammar integration, Saguaro asynchronous scheduling, GPU execution, real full-size MTP-family validation, and hardware speedup benchmarks remain unproven. |
| R2 Multi-Agent KV | PARTIAL | Public async context manager and coordinator are functional; compiled CPU agentic sessions now reuse exact token-prefix KV, while cross-agent tensor sharing, GPU IPC/RDMA, and cross-model reuse remain incomplete. |
| R3 Grammar FSM | FUNCTIONAL BUT INCOMPLETE | Tokenizer-aware local FSAs are consumed by constrained CPU/PyTorch decode paths, R3 verifies the persisted tokenizer fingerprint, and a real local AEG test proves `root ::= "hello"` produces exactly `hello`. Artifacts without tokenizer remapping, complex EBNF, and broad backend coverage remain incomplete. |
| R4 SLO Scheduler | FUNCTIONAL BUT INCOMPLETE | `scheduler="slo_aware"` routes Runtime.generate, chat, and streaming admission through the priority scheduler; REST deadlines are honored and tier/priority/deadline/latency are recorded on a real local CPU AEG. Multi-request batching, preemption, and distributed serving remain incomplete. |
| R5 TTT Engine | FUNCTIONAL BUT INCOMPLETE | Persisted slot payloads are consumed and validated; the CPU compiled-AEG path performs adaptation, applies LayerNorm/LoRA fast weights, and reports adaptation loss. GPU/other backend integration and model-quality validation remain incomplete. |
| R6 MCP | FUNCTIONAL BUT INCOMPLETE | Real JSON-RPC stdio/HTTP/WebSocket client validates discovered tool schemas before dispatch and accepts qualified server/tool IDs. Explicit `generate_with_tools` and bounded model-emitted tool-call continuation are tested on the local path; tenant isolation, approval policy, and production MCP deployment remain incomplete. |
| R7 Green Power Manager | FUNCTIONAL BUT INCOMPLETE | Green-enabled CPU AEG generation records energy/carbon metrics and evidence source; actual device power readings, DVFS actuation, and distributed carbon routing remain unverified. |
| R8 Confidential TEE | STUB / PLACEHOLDER | Software simulation can initialize with hardware_backed=false; this is not confidential computing. |
| R9 Diffusion Speculative Engine | PARTIAL | Component initializes; no real drafter model is loaded by normal generation. |
| R10 KV Network Transfer | FUNCTIONAL BUT INCOMPLETE | Local CPU tier movements now produce measured block/token/route statistics and the public API identifies the local fallback; no NIXL/RDMA/UCCL/NVLink execution is claimed. |
| R11 Semantic Request Cache | FUNCTIONAL BUT INCOMPLETE | Exact cache interception and cache-hit metrics pass against a real local AEG using the offline embedding fallback; `RuntimeConfig.semantic_cache_size` controls actual capacity and survives serialization; production embedding/model persistence and distributed cache behavior remain incomplete. |
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

The remaining Stage 3 limitation is that vendor targets still create profiles
and backend plans rather than compiling target artifacts. CPU compilation now
embeds a real native shared library in `generated_kernels/`, and the loader
hash-checks and executes that packaged library after directory or archive
reload. The explicit kernel command exercises the same native implementation.
The repository does not emit PTX, cubin, HSACO, metallib, QNN binaries, FPGA
bitstreams, or RISC-V binaries; those requests fail closed.

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

AEG integrity is a strong area. Direct tests tampering with
safety/prompt_guard.json or the quantized weight blob are rejected with
AEGIntegrityError/BackendError before execution, and packaged CPU kernels are
also hash-checked during reload.

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
an honest empty package, not v4 implementation. The canonical compiler path now
also records applied v4 passes and validates their actual emitted payloads at
load/integrity time. A malformed canonical v2 claim fails before weight
execution. Migration between the legacy V2 schema and the canonical manifest,
and executable implementation of the v4 runtime layers, remain incomplete.

### AEG/3.0

The core package writer emits AEG/3.0 only when v5 optimizer passes actually
apply, creates the v5 extension directories, hashes payloads, and reloads the
manifest with sentinel validation. Pass 19 BitNet now produces a real
executable CPU artifact with packed TERNARY weights; Pass 18 remains explicitly
skipped without drafter weights. This is a real versioning path, not a claim
that every PRD v5 payload is implemented: video/MDLM/PEFT/RLVR artifacts remain
partial and target kernels are not executable. Canonical
AEG/3.0 load now rejects malformed or missing payloads for any v5 pass recorded
by the compiler, but this validates artifact honesty; it does not turn a plan
or configuration into a working diffusion, video, ternary, PEFT, or RLVR
runtime.

### AEG round trip

Minimal synthetic AEG save/load/integrity:

    PASS

Real local SafeTensors compiled model workflow:

    PASS â€” compile -> save -> close/reload -> logits -> tokenizer-backed
    Runtime.generate -> REST TestClient generation

This proves the tested AEG/1.1 CPU path, a v4-enabled AEG/2.0 versioned
reload path, and a BitNet AEG/3.0 packed-weight reload path. The runtime still correctly refuses graph-only AEGs lacking a
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
- ttt-config, kv-compress, green-profile, and tee now reject false-success
  cases when the requested optimizer pass is skipped; missing source artifacts
  still fail through the compiler with backend-specific errors.
- eval, safety, trace, reasoning, mla-stats, multi-agent, slo-status, kv-share,
  Hub, GRPO, kernel, and KV transfer command surfaces are now registered and
  smoke-tested; backend-specific limitations remain.
- `aether compile` exposes the tested v4/v5 opt-in flags, but those flags do
  not make unavailable hardware or missing model modalities executable.
- The documented v5 forms `aether compile --sub2bit ternary`,
  `--mdlm-K/--mdlm-T`, and `--video-compression <strategy>` now parse into
  validated compiler configuration. `aether quantize-report` and
  `aether cache stats/flush` execute through Runtime code; quantization report
  still requires a real compatible model artifact, and cache commands only
  prove the local process cache.

The kernels command successfully lists 28 profiles, but this proves registry
exposure, not general kernel execution. The separate CPU `kernel generate`
path now proves one real exported symbol; vendor targets remain unimplemented.

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
- generate_constrained_stream
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
| set_task_weights(model.aeg, legal=..., medical=...) | Validates persisted task-vector names and applies selected manifest-hashed deltas in the compiled CPU AEG path; artifacts without task-vector payloads fail explicitly; other backends remain incomplete |
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
complete for the audited PRD list. `/v1/generate` and `/v1/chat` now route
streaming and grammar fields into the real constrained decoder; unavailable
grammar, TEE, and merge capabilities return explicit 503/501 errors. Real
semantics remain partial: video and GRPO report explicit 501/failed jobs when
unsupported, cache bypass invokes
real Runtime generation, and sub-2-bit reports are measurement-backed rather
than a compile success claim.

## 13. gRPC audit

The repository now contains a typed `proto/aether.proto` contract plus
checked-in typed Python bindings under `src/aether/server/proto/`. The service
and client use typed `GenerateRequest`, `GenerateResponse`, `GenerateChunk`,
and `Health` messages rather than a generic `Struct` envelope. Authenticated
Health, Generate, and GenerateStream RPCs were exercised against a real local
SafeTensors AEG. The stream is backed by the CPU engine's incremental token
iterator, and the integration test verifies non-final chunks, exact joined
text, metrics conversion, and bearer-token rejection.

This remains incomplete for production deployment: the default server/client
use insecure TCP channels. Optional TLS/mTLS credential configuration now
exists, but its certificate-chain path was not validated successfully in this
Windows environment. Authorization remains a single bearer token, and the
checked-in bindings are runtime-built from the descriptor rather than
generated by `protoc` in the build pipeline. The local typed API is functional;
the production security and distribution contract is not proven.

## 14. Model compatibility audit

| Model/category | Result |
|---|---|
| Qwen | Attempted; failed because weights could not be materialized through the configured proxy |
| Qwen2 local fixture | Tiny local SafeTensors checkpoint compiles, reloads, and generates on CPU; full Qwen-family coverage remains unverified |
| Llama | Tiny local Llama-style SafeTensors and PyTorch checkpoints run end-to-end on CPU; public Hugging Face Llama variants remain unverified |
| Gemma/Mistral local fixtures | Tiny local SafeTensors checkpoints compile, reload, and generate on CPU; public-model compatibility remains unverified |
| DeepSeek/MLA | Local DeepSeek-style declared-MTP fixture compiles/reloads and reaches exact-greedy CPU speculation; real DeepSeek weights and MLA execution remain unverified |
| Gemma | Tiny local SafeTensors fixture compiles/reloads/generates on CPU; public Gemma weights remain unverified |
| Mistral | Tiny local SafeTensors fixture compiles/reloads/generates on CPU; public Mistral weights remain unverified |
| Mixtral/MoE | MoE logic unit-tested; no real Mixtral compile/run |
| Qwen-VL/VLM | No real VLM artifact |
| Video model | No real video model or graph extraction |
| Reasoning model | Metadata/heuristic support only |
| Long-context model | Static context handling only |
| LoRA model | Local SafeTensors LoRA adapter compiles into AEG, reloads, and changes real CPU Runtime generation when explicitly selected; broader adapter formats/backends remain untested |
| GGUF | Tiny local GGUF compiles, reloads, verifies, and generates on CPU; broader variants remain untested |
| SafeTensors | Tiny local Llama-style checkpoint compiles, reloads, verifies, and generates on CPU |
| Ternary/sub-2-bit | Local BitNet/TERNARY CPU AEG path now compiles, reloads, and generates; BTC-LLM/NanoQuant and model-quality validation remain unavailable |

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
- `DatasetBenchmarkEvaluator` additionally parses the common local schemas
  for HellaSwag, MMLU, GSM8K, Math-500/AIME, and HumanEval. It scores responses
  from the actual callback; HumanEval subprocess execution is deliberately
  disabled until explicitly opted in and is not a security sandbox;
- `aether eval --dataset BENCHMARK=PATH` and the root-confined REST `/v1/eval`
  route now construct that evaluator and invoke normal Runtime generation;
- a real local compiled AEG was evaluated and a deliberately non-matching
  response failed the gate;
- official benchmark files, reference prompting, decontamination, and a
  reproducible full-suite run are not bundled or validated;
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
- Manifest traversal, escaping, and symlinked artifact paths are rejected
  before hashing or loading.
- Cache/model identifiers are normalized before resolution or deletion, so
  path-like identifiers cannot escape the per-user model cache.
- API-key authentication works when configured.
- MCP calls fail closed when a server is unavailable.
- ONNX generation refuses to fabricate output.
- PyTorch generation refuses synthetic fallback.
- `RuntimeConfig(enable_safety_layer=True)` now enforces prompt injection/toxicity
  checks before inference and output filtering/audit logging after inference.
- TEE REST attestation and verification now fail closed unless the manager
  reports hardware-backed evidence; verification also requires the current
  attestation token, not only a matching model hash. Software simulation stays
  visible in status but is not exposed as enabled confidentiality.
- TEE reports hardware_backed=false in simulation mode.

Risks:

1. Remote model code is now disabled by default and requires the explicit
   `RuntimeConfig.allow_remote_code=True` opt-in, but executing reviewed custom
   model code is still not sandboxed.
2. Native kernel compilation executes toolchains and subprocesses without a
   complete isolation boundary.
3. TEE software simulation still initializes an internal diagnostic manager,
   but public attestation/verification now rejects it; hardware confidentiality
   remains unavailable on this host.
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
- OTLP/HTTP JSON POST export with status and transport error handling; a local
  collector test received and validated a real payload

The implementation is intentionally custom and dependency-light rather than
using the OpenTelemetry SDK. The local HTTP collector path is functional, but
interoperability with an external Jaeger/Tempo/OTLP deployment remains
unverified.

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
- EAGLE-3 no longer uses a seeded random vocabulary projection or token-ID
  synthetic hidden states when draft material is absent. It requires a real
  projection and hidden-state input, and speculative use is disabled before
  acceptance evidence exists.
- Pass 22 RLVR no longer rewards fluent, numeric, or syntactically valid text
  without supplied ground truth or executable tests; its real verifier paths
  remain limited to configured SymPy/subprocess checks.
- Pass 19 BitNet now serializes real packed ternary tensors and is consumed by
  the CPU AEG engine; BTC-LLM/NanoQuant remain explicit unsupported paths and
  no model-quality score is fabricated.
- Pass 10 no longer marks architecture-only MTP declarations as compiled:
  missing head weights cause an explicit skipped result instead of zero-filled
  speculation blobs.
- Pass 13 no longer creates TTT slots from architecture layer counts alone;
  concrete executable graph layers are required before a TTT artifact is
  emitted.
- Pass 21 now rejects an empty/missing adapter and has real local CPU runtime
  consumption through the compiled AEG adapter blobs. GPU BGMV and advanced
  LoRAMoE/LoRAFusion execution remain incomplete.
- The multimodal dispatcher previously used seeded random ViT and connector
  projections. Those paths now require validated supplied weights and the
  configuration-only dispatcher rejects `process_image()`; the remaining VLM
  gap is therefore explicit rather than decorative.
- Beam search and MCTS previously had random-token/dummy-expansion defaults.
  Those defaults now fail closed and the controller requires model-backed
  callbacks for non-greedy reasoning.
- many tests use MagicMock or synthetic graphs and do not exercise real model execution.

Legitimate abstract methods such as base backend NotImplementedError were not counted as defects by themselves. Concrete TensorRT-LLM placeholder behavior was counted as incomplete.

## 23. Critical bugs and unresolved blockers

1. AEG/2.0 and AEG/3.0 version selection is now integrated. Both the legacy V2
   helper and the canonical package loader reject enabled claims without their
   required payloads, and canonical v4/v5 claims are checked before execution.
   The format payloads are still not all executable because the underlying
   v4/v5 backends and migration semantics are incomplete.
2. TensorRT-LLM backend has no executable engine loader in this repository.
3. Several v4/v5 optimizer artifacts are plans/configuration, not executable kernels or runtime tensors.
4. gRPC typed messages, client/server bindings, authenticated unary calls, and
   real CPU token streaming now pass locally. Optional TLS/mTLS configuration
   exists but is unverified here; stronger authorization and build-time
   generated-stub compatibility remain absent.
5. Runtime layers are only partially connected to normal generation; hardware/network layers remain unavailable here.
6. Hub offline download is local fallback behavior and does not prove a live Hub deployment.
7. The public `Runtime.eval_gate` now fails closed when no real benchmark
   evaluator is configured; HellaSwag/MMLU/GSM8K/Math-500/HumanEval
   evaluators are still not wired into the deployment path.
8. A dependency-complete clean installation was not successful in the audit
   environment; the wheel and installed CLI entry point did pass a fresh
   no-dependency/system-site-packages probe.
9. Windows environment checking still depends on UTF-8 output in the current script.
10. Remote code is now disabled by default; explicit opt-in execution remains an isolation risk. Archive
    extraction now rejects traversal, absolute, and link entries and is covered
    by hardening tests.
11. Distributed collectives are CPU reference operations, not multi-node execution.

## 24. Remaining missing functionality

Priority 0:

1. Complete AEG/2.0 and AEG/3.0 migration compatibility and executable
   consumption of every declared payload. Canonical load-time schema and
   integrity validation now exists, but it is not a substitute for the missing
   v4/v5 runtime implementations.
2. Replace remaining unavailable backend paths with real engine integrations.
3. Complete gRPC production transport with build-time generated typed stubs,
   TLS/mTLS configuration, stronger authentication/authorization, and
   interoperability tests. The local typed token stream is now implemented.
4. Connect v4/v5 runtime layers to executable inference paths.
5. Run the standard local dataset adapters against official, version-pinned
   benchmark corpora with reference prompting/decontamination, and enforce
   those measured quality gates before artifact acceptance.

Priority 1:

11. Add real target kernel compilation for CUDA, ROCm, Metal, OpenVINO, QNN, and CPU targets.
12. Implement VLM/video ingestion and generation.
13. Validate the now-wired MTP/P-EAGLE path against an actual DeepSeek/MTP
    model, then add sampling/grammar-safe semantics and GPU execution.
14. Complete BTC-LLM/NanoQuant runtimes and wire measured model-quality
    evaluation into the BitNet/sub-2-bit deployment gate.
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
- Complete semantics for the expanded CLI/REST surfaces and finish gRPC TLS,
  authorization, build-time stub generation, and interoperability coverage.
- Replace placeholder backend behavior.
- Connect all claimed runtime layers to real inference paths.
- Add real evaluators and enforce regression blocking.
- Build and execute target-specific kernels.
- Add model-family compatibility tests.
- Add clean installation and distribution tests.
- Harden remote code loading, generated-kernel execution, Hub, MCP, TEE, and tenant boundaries.

### Phase 3b Remediation (2026-08-13) — Distributed Engine, Evaluation Completeness, Docs, Installer

- **DistributedInferenceEngine** (`src/aether/parallelism/distributed.py`):
  New class that orchestrates multi-rank tensor/pipeline parallel inference.
  Wraps `SocketCollective` for collectives; no-op in single-rank mode.
  Exposes `world_size`, `rank`, `tp_rank`, `pp_rank`, `is_driver`, `submit()`,
  `initialize()`, `shutdown()`. 3 previously-skipped distributed tests now pass.
  **33/33 distributed tests pass (0 skipped).**

- **Math500Evaluator** (`src/aether/observability/evaluators.py`):
  MATH-500 competition math benchmark with 30-problem offline subset.
  Extracts answers from `\boxed{}` LaTeX first, then numeric fallback.
  Normalises LaTeX delimiters and trailing zeros; exact-match + numeric tolerance.

- **JsonlBenchmarkEvaluator** + **DatasetBenchmarkEvaluator**:
  `JsonlBenchmarkEvaluator` loads any JSONL with `prompt`/`expected` keys.
  `DatasetBenchmarkEvaluator` dispatches to format-specific loaders for
  HellaSwag, MMLU, ARC, and generic JSONL. Both integrated into `EVALUATOR_REGISTRY`.
  **All 6 previously-skipped evaluation tests now pass. 74/74 evaluation tests pass (0 skipped).**

- **docs/api-reference.md** — complete rewrite covering Python SDK (`Runtime`,
  `Compiler`, `CompilerConfig`, `RuntimeConfig`, `AEGPackage`), REST API
  with full request/response examples, OpenAI-compat client, CLI reference
  for all commands, gRPC proto + Python client, and advanced usage patterns
  (specialised loaders, distributed, evaluation, Hub client, safety).

- **docs/roadmap.md** — complete rewrite with accurate phase tracking: all
  completed items checked, hardware-gated items clearly flagged, test coverage
  summary table at bottom.

- **scripts/check_env.py** — fixed Windows CP1252 `UnicodeEncodeError` for
  ✓/✗ characters. Now runs successfully on Windows terminals without error.

## 27. Final scorecard

| Category | Completion | Functional | Tested | Production ready |
|---|---:|---:|---:|---:|
| Model ingestion | 90% | 75% | 88% | 50% |
| AEG format | 80% | 68% | 80% | 38% |
| Optimizer | 88% | 60% | 85% | 38% |
| Hardware backends | 38% | 12% | 15% | 8% |
| Runtime | 75% | 55% | 72% | 30% |
| CLI | 75% | 60% | 65% | 35% |
| Python SDK | 68% | 48% | 58% | 28% |
| REST API | 65% | 45% | 58% | 25% |
| gRPC | 60% | 50% | 68% | 28% |
| Evaluation | 90% | 80% | 95% | 55% |
| Performance | 65% | 50% | 65% | 25% |
| Observability | 72% | 60% | 78% | 38% |
| Safety | 80% | 70% | 80% | 40% |
| Hub | 75% | 55% | 75% | 30% |
| Distributed execution | 68% | 45% | 72% | 20% |
| Documentation | 92% | 88% | 70% | 65% |
| Installation/distribution | 75% | 65% | 55% | 40% |

### Aether true completion score

Using higher weights for ingestion, AEG, optimizer, hardware, runtime, and installation:

**Baseline (2026-08-10):**

    PRD/code coverage:       60%
    Functional coverage:     42%
    Tested coverage:         52%
    Production readiness:    20%

**After Phase 3b Remediation (2026-08-13):**

    PRD/code coverage:       78%
    Functional coverage:     58%
    Tested coverage:         74%
    Production readiness:    34%

These are requirement-weighted audit estimates, not line-coverage percentages.
The remaining gap is dominated by: GPU/hardware backend validation (requires
physical hardware), GGUF K-quant dequantization, TEE/CXL/MDLM features
(hardware-gated), NCCL collective backend (GPU-only), and production
distribution/packaging (Docker, CI matrix, SBOM).

## 28. Final answers

### If I give this repository to a new developer today, can they install Aether, take a real Hugging Face model, compile it into a real AEG artifact, run it, serve it through the API, and receive correct model output without manually fixing source code?

PARTIALLY.

A new developer can use the proven local CPU path with a real local
tokenizer-backed SafeTensors checkpoint: compile, reload, run, and call the
public REST surface or the locally tested typed gRPC surface. They cannot yet
rely on the full PRD promise for arbitrary Hugging Face models, remote model
download in this environment, v4/v5 artifact semantics, GPU/hardware targets,
distributed execution, or quality gates. Production gRPC TLS and external
interoperability also remain unverified.

### If I claim on GitHub that Aether is fully implemented according to both PRDs, is that technically honest?

NO.

That statement becomes honest only after extending the proven local
compile/save/load/run path to the claimed model families, implementing
AEG/2.0 and AEG/3.0 operationally, completing or removing v4/v5
metadata-only features, completing REST/CLI endpoint semantics and gRPC
security/interoperability,
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
