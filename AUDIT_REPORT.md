# Aether Runtime — Adversarial Final Audit

Audit date: 2026-08-07  
Repository: `C:\Users\pc\Desktop\Aether Runtime`  
Environment: Windows, AMD64, Python 3.10.11, 12 logical CPUs, 7.69 GiB RAM, Torch 2.9.1 CPU build, no CUDA device, no ROCm, no Apple Metal, no TEE/CXL/NIXL hardware.

## 1. Executive verdict

### ⚫ PRD-COMPLETE ON PAPER, NOT FUNCTIONALLY COMPLETE

The repository contains a broad architecture and many named modules for the v3.1 baseline, v4.0 additions, and v5.0 additions. That is not equivalent to an implementation of those requirements.

The decisive failures are:

- A real Hugging Face identifier compiles as graph-only input and receives deterministic synthetic/random weights when no local checkpoint is present.
- The runtime can return hardcoded artifact-prose (`"Aether loaded the ..."`) instead of executing the model.
- The ordinary compiler emits AEG/1.0 and kernel/backend plans; it does not emit the target-specific executable kernels required by the PRDs.
- v4/v5 optimizer passes mostly produce planning metadata/opcodes; the v2 pass test suite has 8 failures, all caused by incorrect report status expectations.
- v4/v5 runtime layers are not initialized on the normal runtime load/generate path.
- The FastAPI request models are exposed as query parameters; valid JSON bodies to generation and compilation return HTTP 422.
- `aether run`, `aether graph`, and `aether kernels` crash during normal CLI initialization because the configured `structlog` processor expects a stdlib logger but receives `PrintLogger`.
- The complete test suite did not finish; the compiler unit suite hangs while trying to reach Hugging Face for an unknown model, and the real CPU E2E suite timed out.
- A clean package build with `pip install . --no-deps --no-build-isolation` fails because the declared Hatch build backend is not available in a fresh environment.

## 2. PRD lineage and audit method

Both PRDs were read in full. `PRD.md` defines the v3.0 specification and its Part II v3.1 baseline. `PRD_v2.md` explicitly says v3.1 is the implemented baseline, v4.0 is net-new functionality, and Part II v5.0 is another net-new extension. This audit therefore does not count v3.1 requirements as missing merely because they are not in v4 files, and does not credit v4/v5 requirements merely because a class, directory, or configuration field exists.

Evidence was collected through:

1. Full PRD reading and a requirement inventory.
2. Source-path and call-path inspection.
3. Real compiler execution using `Qwen/Qwen3-0.6B` as an online model identifier.
4. AEG save, process-independent reload, integrity check, CPU engine load, and token generation.
5. Direct SDK calls and signature inspection.
6. FastAPI app construction, route inspection, and TestClient requests.
7. CLI help and executable command probes.
8. Target registry inspection and hardware detection.
9. Targeted pytest runs, test coverage output, full-suite attempts, and clean-package installation.
10. Searches for placeholders, simulations, synthetic weights, hardcoded output, and unimplemented paths.

Major evidence chain:

`Qwen/Qwen3-0.6B` → `Compiler.compile()` → `.audit_tmp/qwen3-0.6b.aeg` → reload with `load_aeg_package()` → CPU engine generated token IDs successfully, but compile logs stated the model was not local and graph weights were synthesized. Runtime generation returned the hardcoded phrase `Aether loaded the ...`. Therefore the artifact is executable as a synthetic graph fixture, not a faithful compiled Qwen model.

## 3. Complete requirements matrix

The table is grouped by PRD requirement family so every requirement class is represented without pretending that repeated prose instances are separate implementations.

| ID | PRD Requirement | Version | Component | Required Behavior | Code Location | Implemented? | Actually Functional? | Tested? | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| M-01 | SafeTensors ingestion | v3.1 | Stage 1 | Parse metadata and tensors, build graph and weights | `stage1_ingestion/safetensors_loader.py` | Yes | No proof of real tensor path; loader has empty/fallback behavior | Synthetic loader tests only | No real checkpoint run | STUB / PLACEHOLDER |
| M-02 | GGUF ingestion | v3.1 | Stage 1 | Read GGUF tensors, tokenizer, quantization, graph | `stage1_ingestion/gguf_loader.py` | Partial | Not exercised with a real GGUF model | No real GGUF E2E | Loader exists, no artifact proof | PARTIAL |
| M-03 | ONNX ingestion | v3.1 | Stage 1 | Parse ONNX protobuf and lower executable graph | `stage1_ingestion/onnx_loader.py` | Partial | Graph parsing exists; backend returns placeholder text | Synthetic/parser coverage only | `onnx_backend.py` returns placeholder response | ⚫ IMPLEMENTED BUT BROKEN |
| M-04 | MLX ingestion | v3.1 | Stage 1 | Read MLX model and weights | `stage1_ingestion/mlx_loader.py` | Partial | Not testable on Windows/no MLX; no E2E | Parser tests only | No Apple path | ⚪ NOT TESTABLE ON CURRENT HARDWARE |
| M-05 | PyTorch/Hugging Face ingestion | v3.1 | Stage 1 | Load config, tokenizer, weights and graph from real model | `stage1_ingestion/ingestion.py`, `pytorch_loader.py` | Partial | HF identifier compiled graph-only; weights synthesized when absent | Graph-only compile executed | Logs explicitly say “graph built without weights” | ⚫ IMPLEMENTED BUT BROKEN |
| M-06 | Architecture detection | v3.1 | Stage 1 | Detect architecture, sizes, layers, dtype, modalities | `architecture_detector.py` | Yes | Works for known config/IDs; network lookup can hang | Some unit coverage | Unknown model test hung on HF request | FUNCTIONAL BUT INCOMPLETE |
| M-07 | VLM/video/MLA/MoE/SSM/reasoning/MTP/ternary detection | v3.1/v4/v5 | Stage 1 | Detect and extract each model family correctly | ingestion modules and `graph_tracer.py` | Partial | Metadata/config paths exist; real representatives not run | Synthetic tests | No real VLM/video/MLA/ternary run | PARTIAL |
| C-01 | Five-stage compiler pipeline | v3.1 | Compiler | Ingest → optimize → target → package → quality gate | `compiler/compiler.py` | Yes | Runs for graph-only model; quality gate is not enforced as PRD gate | Compile executed | AEG created in 58.6s | FUNCTIONAL BUT INCOMPLETE |
| C-02 | v3.1 passes 1–9 | v3.1 | Optimizer | Modify AEG-IR with real fusion, calibration, precision, KV, MoE, parallelism, reasoning, sparse attention, pruning | `stage2_optimizer/pass1...pass9.py` | Partial | Some real graph metadata transformations; no validated quality-preserving artifact path | Unit/synthetic coverage | No real model benchmark | PARTIAL |
| C-03 | v4 passes 10–17 | v4.0 | Optimizer | Emit MTP, grammar, merging, TTT, semantic KV, cross-layer KV, green, TEE artifacts | `stage2_optimizer/pass10...pass17.py` | Partial | Mostly planning/report/opcode metadata; config cloning drops v4/v5 fields | `test_passes_v2.py`: 8 failures | Reports say “Planning” and statuses mismatch tests | ⚫ IMPLEMENTED BUT BROKEN |
| C-04 | v5 passes 18–22 | v5.0 | Optimizer | Emit diffusion drafter, ternary, video compression, PEFT, RLVR artifacts | `stage2_optimizer/pass18...pass22.py` | Partial | Smoke paths accept synthetic input and estimate benefits; no real weights/training/video | `test_passes_v2.py`: failures for 18, 19, 22 | No production backend consumption proven | STUB / PLACEHOLDER |
| C-05 | All 22 passes registered in real distribution | v5.0 | Packaging | Discover and execute every pass through installed compiler | `pyproject.toml`, `stage2_optimizer/optimizer.py` | Partial | In-process pipeline imports 22; plugin entry points list only 1–9 | Static and compile log | v4/v5 absent from entry-point group | PARTIAL |
| T-01 | Hardware targeting and kernel emission | v3.1–v5 | Stage 3 | Generate executable target kernels and backend plans | `stage3_targeting/kernel_emitter.py` | Partial | Emits `KernelPlan`, explicitly not native code; compiler stage only creates profiles | Static + compile | Source says “kernel plan” rather than native artifact | STUB / PLACEHOLDER |
| A-01 | AEG/1.x including v3.1 layout | v3.1 | AEG | Versioned manifest, IR, weights, kernels, metadata, safety, provenance, runtime artifacts | `core/aeg_format.py`, `aeg_format_v31.py` | Partial | Core package works but emits AEG/1.0, not required v3.1 AEG/1.1 | Compile/reload/integrity executed | Manifest was `AEG/1.0` | ⚫ IMPLEMENTED BUT BROKEN |
| A-02 | AEG/2.0 | v4.0 | AEG | New directories and real v4 artifacts consumed by runtime | `compiler/aeg_format_v2.py` | Partial | Class creates directories and explicit config stubs; ordinary compiler does not use it | 84 AEG/format tests pass | Tests explicitly validate stubs | STUB / PLACEHOLDER |
| A-03 | AEG/3.0 | v5.0 | AEG | v5 artifacts/opcodes/weights/video/training/KV transfer | `aeg_format_v2.py`, core package | No | No AEG/3.0 package writer/reader in normal path | No | Search found AEG/1.0, 1.1, 2.0 only | NOT IMPLEMENTED |
| A-04 | AEG integrity and provenance | v3.1–v5 | AEG/security | Verify all artifact hashes, weights, provenance, safety and signatures | `core/aeg_format.py`, `provenance/*` | Partial | `verify_integrity()` verifies manifest/graph, not every declared payload/weight hash; defaults include unknown/disabled provenance | Direct integrity call passed | Returned `None`; no tamper matrix run | PARTIAL |
| R-01 | EAGLE-3, KV, disaggregated prefill/decode, dynamic precision | v3.1 | Runtime | Initialize and use all baseline layers in generation | `runtime/runtime.py`, `eagle.py`, `kv_cache.py` | Partial | Basic CPU path works; advanced components not shown on real generation | Baseline tests partial | Runtime output bypasses model semantics | PARTIAL |
| R-02 | R1 P-EAGLE + Saguaro | v4.0 | Runtime | Parallel speculative draft/verify with acceptance metrics | `r1_parallel_speculative.py` | Partial | Module/tests exist; normal `Runtime.generate()` does not initialize/use it | v2 runtime tests | No generated speculative output path | PARTIAL |
| R-03 | R2 Multi-Agent KV coordination | v4.0 | Runtime | Shared isolated KV sessions and coordination modes | `r2_multi_agent_kv.py`, `runtime.py` | Partial | `multi_agent_session()` returns session metadata synchronously, not the PRD async context manager; no generation integration | Unit/smoke | Returned four IDs only | PARTIAL |
| R-04 | R3 Grammar FSM | v4.0 | Runtime | Constrain decode loop with grammar | `r3_grammar_fsm.py`, `generate_constrained()` | Partial | Compiler/API pieces exist; ordinary generate route does not apply grammar | Unit tests, REST body fails | No end-to-end constrained output | PARTIAL |
| R-05 | R4 SLO scheduler | v4.0 | Runtime | Deadline-aware scheduling and status endpoints | `r4_slo_scheduler.py` | Partial | Isolated scheduler tests pass; no route/API integration | v2 runtime tests | Missing `/v1/slo/*` | PARTIAL |
| R-06 | R5 TTT engine | v4.0 | Runtime | Safe fast-weight adaptation/reset per session | `r5_ttt_engine.py`, runtime | Partial | Configuration and metadata exist; normal generation/base-weight isolation unproven | Isolated tests | No full update→generate→reset E2E | PARTIAL |
| R-07 | R6 MCP | v4.0 | Runtime | Register, authenticate, dispatch MCP JSON-RPC tools safely | `r6_mcp_integration.py` | Partial | JSON-RPC client/registry exists; no authenticated server lifecycle or generation integration | Unit tests | Existing `/v1/tools/call` only | PARTIAL |
| R-08 | R7 Green Power Manager | v4.0 | Runtime | Carbon routing, power/DVFS, babbling suppression, metrics | `r7_green_power_manager.py` | Partial | Software calculations exist; no hardware power/carbon telemetry in generation | Isolated tests | Compile report showed 0 DVFS hints/0 savings | PARTIAL |
| R-09 | R8 TEE | v4.0 | Runtime/security | Hardware-backed enclave, sealing, attestation, verification | `r8_tee_manager.py` | Partial | Explicit software simulation without confidentiality; test token length fails | 77/78 runtime tests | Logs state no guarantees; 1 test failed | ⚫ IMPLEMENTED BUT BROKEN |
| R-10 | R9 diffusion speculative engine | v5.0 | Runtime | Real diffusion drafter integrated into decode | `r9_diffusion_spec_engine.py` | Partial | Engine class initializes in optional stats path; not normal generation | No real E2E | `R9` not used by generate | PARTIAL |
| R-11 | R10 KV network transfer | v5.0 | Runtime | NIXL/NIXL-like transfer, stats, failure recovery | runtime/R10 modules | No | `kv_transfer_stats()` returns disabled metadata in this environment; no network engine | No | No gRPC/NIXL implementation found | NOT IMPLEMENTED |
| R-12 | R11 semantic request cache | v5.0 | Runtime | Cache requests, hit/miss, bypass/flush, savings | `r11_semantic_kv_cache.py` | Partial | Initialization failed due environment lock; generate does not consult cache | Isolated tests | Runtime stats returned disabled | ⚫ IMPLEMENTED BUT BROKEN |
| R-13 | R12 CXL rack-scale KV pool | v5.0 | Runtime | CXL memory pool, defrag, isolation, recovery | `r12_cxl_kv_pool.py` | Partial | Software class exists; no CXL device or normal generation integration | No hardware test | Status disabled with zero configured pool | ⚪ NOT TESTABLE ON CURRENT HARDWARE |
| API-01 | CLI v3.1 commands | v3.1 | CLI | compile/run/serve/graph/hardware/bench/eval/hub/safety/trace/etc. | `cli.py` | Partial | `--help`, version, hardware work; run/graph/kernels crash and eval/hub are absent | Executed | Inventory and failures below | ⚫ IMPLEMENTED BUT BROKEN |
| API-02 | CLI v4 commands | v4.0 | CLI | grammar, merge, TTT, KV, green, TEE, multi-agent, SLO, MCP | `cli.py` | Partial | Some renamed flat commands exist; required nested/API contract is absent or disconnected | Help and probes | No multi-agent/SLO/kv-share commands | PARTIAL |
| API-03 | CLI v5 commands | v5.0 | CLI | sub2bit, MDLM, video, GRPO, kernel generate, KV transfer | `cli.py` | No | No corresponding command group/flags in command inventory | `aether --help` | Required commands absent | NOT IMPLEMENTED |
| API-04 | Python SDK baseline | v3.1 | SDK | Public Compiler/Runtime APIs perform real model operations | `__init__.py`, runtime/compiler | Partial | Imports/instantiation work; real model operation does not | Signatures + direct calls | Synthetic artifact only | PARTIAL |
| API-05 | Python SDK v4/v5 methods | v4/v5 | SDK | Methods and signatures specified by PRDs | `runtime/runtime.py` | Partial | Some methods exist; `generate_with_tools`, `set_task_weights`, `get_attestation_report`, `quantization_report` absent; multi-agent signature differs | Signature inspection | Missing methods documented below | PARTIAL |
| API-06 | REST v3.1 | v3.1 | REST | OpenAI-compatible inference, compile, eval, traces, graph, reasoning, A/B | `server/routes.py` | Partial | GET health/hardware work; JSON POST body is rejected as query parameter; multiple routes absent | TestClient | `/v1/generate` JSON → 422 | ⚫ IMPLEMENTED BUT BROKEN |
| API-07 | REST v4/v5 | v4/v5 | REST | All new merge/agent/SLO/TTT/MCP/green/TEE/video/cache/GRPO/KV/kernel endpoints | `server/routes.py` | Partial | Only a subset exists; many required paths absent | Route inventory | Missing endpoints listed below | PARTIAL |
| API-08 | gRPC | v3.1–v5 | API | Protobuf, server/client, streaming, auth, real inference | repository-wide | No | No `.proto`, gRPC service, or server/client implementation found | Search | Only protobuf dependency and ONNX protobuf usage | NOT IMPLEMENTED |
| E-01 | Eval gates | v3.1–v5 | Evaluation | HellaSwag, MMLU, GSM8K, Math-500, HumanEval, regression blocking | `observability/gates.py`, `Runtime.eval_gate()` | Partial | Gate measures response length/nonempty success over hardcoded prompts; it does not run named suites or block bad AEG | Direct source inspection | No quality regression rejection demonstrated | STUB / PLACEHOLDER |
| E-02 | Performance claims | v3.1–v5 | Benchmarking | Reproducible TPS/TTFT/TBT/Pxx/VRAM/energy/CO2/acceptance/baselines | benchmark/observability modules | No | No comparable baseline run; synthetic runtime reports implausible TPS | No | Claims not reproduced | NOT IMPLEMENTED |
| S-01 | Safety/provenance/security | v3.1–v5 | Safety | Prompt injection, output filter, logs, provenance, integrity, auth, tenant isolation | `safety/*`, `provenance/*`, middleware | Partial | Metadata/default policies exist; server auth is optional and not installed by `create_app`; no hostile artifact/kernel audit | No end-to-end security test | CORS allows `*`; no mandatory auth | PARTIAL |
| O-01 | OpenTelemetry/observability | v3.1–v5 | Observability | Real spans/export and all operational/quality/energy metrics | `observability/otel.py`, server metrics | Partial | In-memory OTLP-shaped exporter works; spans simulate elapsed time and are not wired to runtime | Unit tests only | No exported trace from live inference | PARTIAL |
| H-01 | Aether Hub | v3.1–v5 | Hub/cache | Authenticated upload/download/search, content addressing, integrity, permissions, versioning | `hub/client.py`, `hub/auth.py` | Partial | Client exists but transparently falls back to local cache; local download writes a manifest JSON, not an AEG artifact | No live Hub test | Source explicitly calls local simulation | STUB / PLACEHOLDER |
| D-01 | Distributed/fleet/multi-tenant | v3.1–v5 | Fleet | Multi-request, tenant/KV isolation, multi-GPU/node, migration/recovery | `fleet/*`, manifests | Partial | Plans/metadata exist; no distributed process or network execution | No | Single-process runtime only | NOT IMPLEMENTED |
| P-01 | Installation/distribution | v3.1–v5 | Packaging | Fresh pip install, CLI entry point, optional/native deps, OS support | `pyproject.toml` | Partial | Source imports; clean no-deps build fails at Hatch backend | Clean install attempted | `Cannot import 'hatchling.build'` | ⚫ IMPLEMENTED BUT BROKEN |
| P-02 | Documentation alignment | v3.1–v5 | Docs | Commands/API/install docs match executable interface | `README.md` | Partial | README documents commands not in CLI (`inspect`, `eval`, `hub`, `sdk`, `sign`, `verify`) | Compared README/help | Drift is direct | PARTIAL |

## 4. Baseline v3.1 audit

The v3.1 baseline is not wholly absent. The repository has meaningful working pieces: AEG graph structures, a NumPy/native CPU execution engine for synthetic packages, several graph transformations, target profiles, cache/KV data structures, provenance/safety data models, and unit tests. Those pieces do not establish the v3.1 promise.

Important baseline findings:

- The current core constant is `AEG/1.0`; the v3.1 format module defines `AEG/1.1`, but the normal `Compiler` uses `AEGPackage` from `core/aeg_format.py`, producing `AEG/1.0`.
- Stage 3 calls `TargetRegistry.create_profiles()` only. `KernelEmitter` emits a `KernelPlan`, and its module documentation explicitly says it does not emit hand-written native code.
- `torch_backend.py` has a real Transformers path when a real local/downloadable model is loaded, but its graph-only AEG path is a deterministic text generator. The normal compile tested here had no real weights.
- The README and PRD claim `aether inspect`, `aether eval`, Hub commands, SDK generation, signing and verification, while the executable command inventory does not contain them.
- Baseline model compatibility was not established for Qwen, Llama, DeepSeek, Gemma, Mistral, Mixtral, VLM, video, MLA, reasoning, LoRA, GGUF, or SafeTensors real checkpoints. No real checkpoint was available locally and network-dependent paths are not bounded or reliable.

## 5. Optimizer passes 1–22

| Pass | Implementation | Input/output observed | Pipeline / AEG effect | Tests | Result |
|---|---|---|---|---|---|
| 1 Operator Fusion | `pass1_operator_fusion.py` | AEG graph; fused graph nodes/attributes | In `OptimizerPipeline`; compile log executed it | Baseline tests | PARTIAL — graph transformation exists, no kernel proof |
| 2 Sensitivity Analysis | `pass2_sensitivity_analysis.py` | Graph, architecture, calibration dataset; sensitivity map | Pipeline-connected; annotates graph | Synthetic/unit | PARTIAL — calibration uses simplified dataset/scoring, no quality validation |
| 3 Precision Assignment | `pass3_precision_assignment.py` | Sensitivity map/config; precision map | Pipeline-connected; package precision map | Synthetic/unit | PARTIAL — no real perplexity gate |
| 4 KV Cache Structuring | `pass4_kv_cache_structuring.py` | Graph/architecture; KV metadata | Pipeline-connected | Unit | PARTIAL |
| 5 MoE Expert Routing | `pass5_moe_routing.py` | MoE graph; routing metadata | Pipeline-connected when architecture says MoE | Unit/synthetic | PARTIAL — no Mixtral/DeepSeek-MoE E2E |
| 6 Parallelism Discovery | `pass6_parallelism_discovery.py` | Graph/targets; sharding plan | Pipeline-connected | Unit | PARTIAL — plan only, no multi-device run |
| 7 Reasoning Graph Compiler | `pass7_reasoning_graph.py` | Reasoning architecture/graph; reasoning graph metadata | Pipeline-connected | Unit | PARTIAL — artifact defaults may say disabled/empty |
| 8 Sparse Attention | `pass8_sparse_attention.py`/`pass8_minference.py` | Long-context graph; sparse pattern metadata | Pipeline-connected | Unit | PARTIAL — no 1M-token execution |
| 9 Pruning/Sparsity | `pass9_pruning_sparsity.py` | Graph/weights; masks/pruned graph | Pipeline-connected | Unit | PARTIAL — no quality gate/real weights |
| 10 Native MTP | `pass10_mtp_head.py` | MTP config/graph; MTP opcodes/config | Imported by pipeline; compile log detects no MTP in Qwen test | v2 smoke | PARTIAL |
| 11 Grammar | `pass11_grammar_constraint.py` | schema/regex/EBNF; grammar opcodes/FSM metadata | Config flag is vulnerable to `CompilerConfig.clone()` field loss; runtime not on normal generate | v2/unit | PARTIAL |
| 12 Merging | `pass12_model_merging.py` | task vectors; merged weights/manifest | Pipeline and CLI exist, but CLI assigns non-dataclass config fields and compiles first model | v2/unit | STUB / PLACEHOLDER |
| 13 TTT injection | `pass13_ttt_fast_weight.py` | base graph/config; adapter/fast-weight metadata | Pipeline-connected only when enabled; runtime integration incomplete | v2/unit | PARTIAL |
| 14 Semantic KV | `pass14_semantic_kv_compression.py` | graph/config; retention/chunk plan | Executed in compile; report says “Planning chunk KV compression”; does not prove cache implementation | v2: fail | ⚫ IMPLEMENTED BUT BROKEN |
| 15 Cross-layer KV | `pass15_cross_layer_kv.py` | graph/config; share groups/opcodes | Executed; report says “Planning”; test status mismatch | v2: fail | ⚫ IMPLEMENTED BUT BROKEN |
| 16 Green energy | `pass16_green_energy.py` | target/carbon/TDP config; energy profile/DVFS hints | Executed; compile report had 0 DVFS hints and 0% estimated savings | v2: fail | PARTIAL |
| 17 TEE wrapping | `pass17_tee_wrapping.py` | graph/weights/TEE config; wrapped kernel/weight metadata | Executed as metadata; no enclave emission or hardware | v2: 2 fails | STUB / PLACEHOLDER |
| 18 Diffusion drafter | `pass18_mdlm_drafter.py` | model/config; MDLM plan/opcodes | Executed as plan; report estimated 2.34x | v2: fail | STUB / PLACEHOLDER |
| 19 Ternary/sub-2-bit | `pass19_sub2bit_quant.py` | tensor/list in smoke; quantized representation/report | Smoke reports compression; real weight storage/backend unproven | v2: fail | PARTIAL |
| 20 Video compression | `pass20_video_compression.py` | VLM/video graph; token compression plan | Skips without vision; no real video graph | v2 pass | PARTIAL |
| 21 Advanced PEFT | `pass21_advanced_peft.py` | adapter paths/config; PEFT plan | Skips without adapter; no real LoRA compilation/run | v2 pass | PARTIAL |
| 22 RLVR verifier | `pass22_rlvr_verifier.py` | verifier config/graph; verifier head/opcodes | Metadata/estimated group factor; no real GRPO training or runtime head use | v2: fail | STUB / PLACEHOLDER |

The v4/v5 smoke failures are not cosmetic: they show the suite and implementation disagree on the contract (`status="applied"` while tests accept only `ok/skipped/failed`). More importantly, passing smoke tests would still only establish synthetic pass behavior, not end-to-end model quality or backend execution.

## 6. Runtime R1–R12 audit

| Layer | Initialization | Execution/integration | Fallback/error behavior | Status |
|---|---|---|---|---|
| Existing EAGLE-3/KV/prefill-decode/dynamic precision | Runtime initializes baseline managers | Basic CPU generation path works for synthetic AEG | Specialized backends unavailable errors; graph-only path silently returns deterministic text | PARTIAL |
| R1 P-EAGLE/Saguaro | Classes/tests present | Not initialized by ordinary `Runtime._load_model`/`generate` | No safe feature-level diagnostic in response | PARTIAL |
| R2 Multi-Agent KV | `multi_agent_session()` returns IDs/config | No agent generation/coordination; wrong API shape for PRD context manager | Returns success metadata | STUB / PLACEHOLDER |
| R3 Grammar FSM | Can be instantiated/compiled in isolated paths | Not wired into ordinary decode; REST body itself fails | No end-to-end constrained-output failure | PARTIAL |
| R4 SLO scheduler | Isolated scheduler initializes | No `/v1/slo/*`; no proof requests affect generation | Metadata only | PARTIAL |
| R5 TTT | Isolated engine/config exists | No complete update/reset/generate lifecycle | No base-weight isolation proof | PARTIAL |
| R6 MCP | JSON-RPC client and registry exist | `/v1/tools/call` is the only relevant route; no secure compiled tool invocation in generation | Local registry paths can operate without authenticated server | PARTIAL |
| R7 Green | Software manager exists | Runtime responses do not expose PRD green metrics; no real power telemetry | Simulation/default values | PARTIAL |
| R8 TEE | Initializes software simulation when no CC hardware exists | Not confidential; no hardware attestation | Explicitly warns that simulation provides no guarantees; one test fails | ⚫ IMPLEMENTED BUT BROKEN |
| R9 Diffusion | Created by `_init_v5_layers()` in optional stats flow | `generate()` does not use it | Stats can report disabled/unused state | STUB / PLACEHOLDER |
| R10 KV transfer | Stats method exists | No gRPC/NIXL/network transfer engine | Returns `enabled: false` | NOT IMPLEMENTED |
| R11 Semantic cache | Optional initializer exists; failed in this run due matplotlib lock | Not consulted by generate | Stats returned disabled | ⚫ IMPLEMENTED BUT BROKEN |
| R12 CXL pool | Software class exists | No CXL device and default size is zero | Status returns disabled | ⚪ NOT TESTABLE ON CURRENT HARDWARE |

Critical reachability result: `Runtime._init_v4_layers()` and `_init_v5_layers()` exist but are not called by normal model loading in the tested execution path. Some v5 statistics methods initialize optional layers, but inference does not consume those layers.

## 7. Hardware target audit

The registry contains 28 target IDs. This proves naming/profile coverage, not backend execution.

| Family / requested targets | Registry/profile | Real executable kernel proven | Physical test | Finding |
|---|---:|---:|---:|---|
| NVIDIA sm70, sm80, sm89, sm90, sm100, sm120 | Yes | No | No CUDA | Profiles/backend candidates only |
| NVIDIA sm130 | Yes | No; source marks placeholder | No | STUB / PLACEHOLDER |
| NVIDIA sm100 TEE | Yes | No | No CC hardware | Software simulation only |
| NVIDIA GB300 / sm100_gb300 | Yes | No | No | Profile/config only |
| AMD ROCm/RDNA3/CDNA3 | Yes | No | No ROCm | HIP source generator exists; compile not invoked |
| AMD MI350X | Yes | No | No | Profile/backend candidate only |
| AMD MI455X/CDNA5 | Yes | No | No | Profile/backend candidate only |
| Apple Metal M1–M5 claim | M1/M3 registry only | No | Windows | No M2/M4/M5 target coverage and no Metal run |
| Intel CPU/OpenVINO/NPU | CPU yes; OpenVINO NPU yes | CPU NumPy/native path only; OpenVINO unproven | CPU only | Partial CPU, NPU unverified |
| Qualcomm Cloud AI 100/QNN | Yes | No | No | Profile/backend candidate only |
| RISC-V MIPS S8200, SiFive X160, XuanTie C930, Cervell | Yes | No | No | Abstract target/config only |
| FPGA Xilinx VU9P / ternary FPGA | Yes | No | No | Backend names/config only |
| CPU x86/AVX2/AVX512/ARM NEON/ternary | AVX512, NEON and ternary profiles | AVX512 native/NumPy path only | AVX512 host | Ternary targets do not have proven executable kernels |

The current `KernelEmitter.emit()` returns a `KernelPlan` containing `target_id`, operation name, backend name and launch attributes. It does not compile or serialize CUDA PTX/cubin, ROCm HSACO, Metal metallib, OpenVINO blob, QNN binary, FPGA bitstream, or RISC-V executable. The compiler's Stage 3 uses hardware profiles and does not invoke the target source emitters.

All non-CPU hardware rows are therefore `UNVERIFIED — HARDWARE REQUIRED`, with the additional distinction that source-level kernel generation is not connected to the normal AEG package path.

## 8. AEG audit and end-to-end artifact test

Executed:

1. `Compiler(CompilerConfig(targets=["cpu_avx512"], overwrite=True, calibration_tokens=32, optimization_level=0)).compile("Qwen/Qwen3-0.6B", output_path=...)`.
2. Saved package contained 39 files and loaded successfully after the compiler process ended.
3. `load_aeg_package()` returned `has_weights=True` and `package_is_runnable=True`.
4. `verify_integrity()` returned successfully.
5. `load_engine_from_path()` loaded a 24-layer CPU engine and generated token IDs.
6. `Runtime.generate()` returned text.

The result is not a successful real-model E2E test. The compile log says the model path was not local and the graph was built without weights. `GraphWeightQuantizer` then synthesized random fallback tensors. The normal runtime backend returned:

`Aether loaded the compiled AEG artifact for qwen_family and executed a portable graph plan offline`

This is a deterministic graph-only response in `torch_backend.py`, not Qwen logits/tokenization. The CPU engine's numeric token generation is a valid synthetic execution test, but it cannot establish correctness against Qwen without authentic weights and tokenizer behavior.

AEG-specific failures:

- Normal compiler version is AEG/1.0, not AEG/1.1, AEG/2.0, or AEG/3.0 as required by the corresponding baselines/extensions.
- `AEGPackageV2` creates many files explicitly through `_write_stub()`; the ordinary compiler does not use it.
- Core `verify_integrity()` checks manifest and graph hashes, but does not verify every file in `manifest.artifacts` or all weight payloads.
- Extended files include defaults such as disabled reasoning, disabled multimodal, empty safety/audit events and unknown license/provenance. Presence is not operational behavior.

## 9. CLI audit

Actual `aether --help` command inventory:

`bench compile grammar graph green-profile hw info kernels kv-compress list logs mcp merge pull rm run serve status tee ttt-config version`

Successful probes:

- `aether --help`: works.
- `aether version`: `Aether Runtime 0.1.0`.
- `aether hw`: returns the Windows/CPU hardware fingerprint.

Failed or absent behavior:

- `aether kernels`: crashes with `AttributeError: 'PrintLogger' object has no attribute 'disabled'`.
- `aether graph <aeg>`: same logging crash during runtime/target initialization.
- `aether run <aeg>`: same logging crash before inference.
- `aether eval <aeg>`: Click reports no such command.
- No `aether hub` command group; only lower-level client code exists.
- No required `aether safety`, `trace`, `reasoning`, `mla-stats`, `precision-map`, `multi-agent`, `slo-status`, `kv-share`, `train grpo`, `kernel generate`, or `kv transfer-stats` commands.
- `grammar`, `merge`, `ttt-config`, `kv-compress`, `green-profile`, and `tee` are flat commands with different arguments from the PRD contracts. Several assign dynamic config attributes that are not dataclass fields and are lost by cloning/serialization.

The logging defect is in `utils/logging.py`: `structlog.stdlib.filter_by_level` is configured with `structlog.PrintLoggerFactory()`. That mismatch prevents normal CLI paths from starting.

## 10. Python SDK audit

Imports and construction work for `Compiler`, `CompilerConfig`, `Runtime`, and `RuntimeConfig`.

Observed public Runtime methods include `generate`, `chat`, `embed`, `rerank`, `transcribe`, `benchmark`, `compile_async`, `merge`, `generate_constrained`, `eval_gate`, `generate_video`, `grpo_train_step`, `semantic_cache_stats`, `kv_transfer_stats`, `cxl_pool_status`, and `multi_agent_session`.

Missing or contract-incompatible PRD examples:

- `generate_with_tools()` absent.
- `set_task_weights()` absent.
- `get_attestation_report()` absent on Runtime; TEE manager has a lower-level method.
- `quantization_report()` absent.
- `multi_agent_session()` is synchronous and returns a dictionary; PRD specifies a context-manager/session style API with model list and coordination mode.
- The real compile/generate SDK path works only for synthetic graph-only artifacts in this environment; configured `llamacpp` fails with `BackendNotAvailableError`, while default/pytorch returns the hardcoded artifact-prose path.

Direct invocation result: default Runtime against the compiled package returned successful `GenerationResponse` metadata, but text began `Aether loaded the`. Explicit `backend_name="llamacpp"` failed because the backend was unavailable.

## 11. REST API and gRPC audit

The app starts and exposes 28 total routes including docs/root/health. Working TestClient GETs:

- `/v1/health` → 200, `{"status":"healthy","version":"4.0"}`.
- `/v1/hardware` → 200 with CPU fingerprint.
- `/v1/models` → 200 with empty list.

Valid JSON requests failed:

- `POST /v1/generate` with `{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":2}` → 422, missing query field `req`.
- `POST /v1/compile` with a JSON body → same 422.

This is caused by request model classes being declared inside `create_router()` while the module uses postponed annotations; FastAPI does not bind them as body models in the installed environment.

Existing relevant routes: generate, chat, embeddings, rerank, transcribe, compile/job status, model management, hardware, kernels, metrics, health, tools/call, grammar compile/list, model merge/TTT, targets, green/status, TEE session.

Missing PRD routes include v3 graph/MLA/reasoning/eval/A-B/traces/cascade routes; v4 structured generation, async merge, multi-agent, SLO, TTT adapt/reset, MCP registration/listing, green metrics/carbon/route, TEE attestation/verify/status, Rubin routes; and v5 video, semantic cache controls, GRPO, KV transfer/CXL, sub2bit, kernel generate/verify routes.

No gRPC interface was found: no `.proto` service definitions, generated gRPC code, server, client, streaming implementation, auth, or tests. The protobuf dependency is used for ONNX/model parsing, not Aether RPC.

## 12. Model compatibility audit

| Model/format family | Download/ingest/compile/load/run/generate result |
|---|---|
| Qwen | Architecture detection and graph-only compile succeeded for `Qwen/Qwen3-0.6B`; real weights and correct generation not proven |
| Llama, DeepSeek, Gemma, Mistral | Not run with real checkpoints; no local weights available |
| Mixtral/MoE | No real checkpoint E2E; MoE pass is synthetic/isolated |
| DeepSeek MLA | No real checkpoint E2E; MLA module/metadata exists |
| Qwen-VL/video | No real model/video run; video pass skips without vision graph |
| Reasoning/long-context | No benchmark/1M-token run; metadata and sparse pass only |
| LoRA | No real adapter compile/hot-swap E2E |
| GGUF | No real GGUF file run |
| SafeTensors | No real checkpoint run |
| ONNX | Parser paths exist; ONNX backend returns `ONNX Runtime single forward pass placeholder.` |
| MLX | Not testable on Windows; no Apple execution |
| PyTorch/HF | Real Transformers backend exists in source, but tested AEG route did not load authentic weights |
| Ternary/sub-2-bit | Synthetic tensor/report smoke only; no real model/backend |

Therefore “any Hugging Face model” is not established. The only real model identifier test reached graph compilation, not faithful model inference.

## 13. Evaluation and performance audit

`Runtime.eval_gate()` does not implement the PRD gate. It uses a bounded hardcoded prompt list and treats a response longer than ten characters as success. It does not run HellaSwag, MMLU, GSM8K, Math-500, HumanEval, coding/reasoning/video suites, baseline comparisons, or artifact rejection on regression.

No PRD performance claim was validated. The tested graph-only runtime reported millions of tokens/sec for a short deterministic string; that is not comparable LLM inference. No same-model/batch/context/temperature baseline was run against llama.cpp, vLLM, SGLang, TensorRT-LLM, or MLX. GPU utilization, VRAM, P50/P95/P99, energy, CO2, KV savings, speculative acceptance, and quality regression are consequently `CLAIM NOT VALIDATED`.

## 14. Security, safety, observability, Hub, and distributed execution

Security findings:

- `create_app()` installs permissive CORS with `allow_origins=["*"]`, `allow_credentials=True`, and does not install `AuthMiddleware`.
- TEE fallback is explicitly software simulation and states it provides no confidential-compute guarantees.
- Generated kernel execution is not a real compiled path, so there is no demonstrated sandbox/signature policy for generated native kernels.
- AEG artifact verification does not cover every declared artifact payload.
- Provenance defaults include unknown license and disabled fingerprinting; presence of JSON does not establish provenance enforcement.
- No hostile AEG/model/path-traversal/kernel/MCP tenant-isolation test was found in the executed evidence.

Observability:

- `AetherTracer`, percentile metrics, Prometheus text, and OTLP-shaped JSON export exist and unit tests exercise them.
- The tracer explicitly simulates elapsed span time, and live Runtime generation is not shown exporting a trace. This is an instrumentation library, not proven live observability.

Hub:

- `HubClient` implements HTTP retries/auth headers and local caching.
- Its documented behavior transparently falls back to “local simulation” when Hub is unreachable.
- Local `download()` writes a JSON manifest file rather than reconstructing the uploaded AEG package. No live Hub, permissions, deduplication, or artifact-integrity E2E was run.

Distributed execution:

- Fleet and multi-node deployment plans are serialized into metadata.
- No multi-process/multi-node execution, session migration, GPU allocator, failure recovery, or tenant-isolated KV network was executed. A single-process session metadata response is not distributed execution.

## 15. Test quality and execution results

Static inventory: 43 Python test files, 1,201 test function definitions, and 60 parametrization markers. These are definitions, not 1,201 verified passing tests.

Executed results:

| Command | Result |
|---|---|
| `pytest tests/unit/test_format_loaders.py tests/unit/test_aeg_format.py tests/unit/test_aeg_format_v2.py -q` | 84 passed, 3 warnings, 16% overall coverage for the selected run; v2 tests validate explicit stubs |
| `pytest tests/test_passes_v2.py -q` | 42 passed, 8 failed, 1 warning; failures are pass-report status contract mismatches for passes 14–19/22 |
| `pytest tests/test_runtime_v2.py -q` | 77 passed, 1 failed, 1 warning; TEE attestation token length assertion failed |
| `pytest tests/unit/test_compiler.py ... --maxfail=1` | Timed out after 60s at unknown-model planning while attempting Hugging Face config download; 16 tests passed before hang |
| `pytest tests/unit/test_e2e_compile_run_cpu.py -q` | Timed out after 181.6s after 18 dots; no completion summary |
| `pytest -q --disable-warnings --maxfail=20` | Timed out after 120s without completion summary |
| `pytest --collect-only` | Timed out during collection in the available window |

The tests are heavily synthetic in the v4/v5 areas: `tests/test_passes_v2.py` constructs `MagicMock` graphs, and the CPU E2E fixture deliberately uses synthetic weights. Passing those tests is useful for local data-structure behavior but does not prove real model compatibility.

## 16. Fake, stub, placeholder, and broken findings

The repository-wide Python search found 16 `NotImplemented` occurrences, 22 placeholder references, 25 stub references, 32 simulation references, 1 dummy reference, 1 fake reference, 107 `return True` matches, 59 `return False` matches, 11 empty-dict returns, 94 `return None` matches, and many `pass` statements. Counts are triage signals, not proof by themselves; the material findings are:

- `weight_quantizer.py` explicitly synthesizes random weights when no real model is loaded.
- `torch_backend.py` graph-only generation emits a fixed word list and deterministic artifact description.
- `onnx_backend.py` returns placeholder text.
- `trtllm_backend.py` raises `NotImplementedError`.
- `aeg_format_v2.py` writes explicit empty config stubs for v4 directories.
- `kernel_emitter.py` emits plans rather than compiled native kernels.
- TEE logs explicitly identify software simulation.
- `observability/otel.py` simulates time passing in trace spans.
- `observability/ci_pipeline.py` simulates noisy results near a baseline.
- Abstract base-class `NotImplementedError` methods are legitimate in isolation, but specialized backend classes and advertised paths remain incomplete.

## 17. Critical bugs

1. **CLI logger crash:** `filter_by_level` + `PrintLoggerFactory` causes `PrintLogger.disabled` AttributeError; breaks run/graph/kernels and likely other commands.
2. **REST body binding bug:** local request model annotations resolve as query parameters; intended POST JSON API returns 422.
3. **Synthetic-weight fallback is silent at the product boundary:** graph-only HF identifiers become runnable artifacts with random weights instead of a hard error or clearly non-production artifact state.
4. **Hardcoded graph-only generation:** successful text output can be mistaken for model inference.
5. **AEG version mismatch:** normal compiler emits AEG/1.0 despite v3.1/v4/v5 requirements.
6. **`CompilerConfig.clone()` serialization loss:** `to_dict/from_dict` does not preserve all v4/v5 fields; flags set by users can disappear before Stage 2.
7. **v4/v5 runtime reachability:** optional layers are not on the ordinary generation path.
8. **Evaluation gate is non-gating:** nonempty text is treated as quality success; no benchmark regression block.
9. **Test-suite hangs on network access:** unknown-model planning has no bounded/offline Hugging Face failure path.
10. **Clean build failure:** declared Hatch build backend is unavailable in a fresh environment; no isolated installation proof.
11. **TEE test contract failure:** attestation token format does not match its test/declared digest contract.
12. **Integrity scope is incomplete:** declared artifact hashes are not all verified.

## 18. Missing functionality and unverified functionality

Missing or materially incomplete: AEG/3.0 writer/reader, gRPC, real target kernel compilation, real Hub service/CLI, complete REST surface, mandatory quality gates, distributed execution, CXL/NIXL transfer, live TEE attestation, real v5 training/video paths, and faithful real-HF compile/run.

Hardware-unverified: CUDA sm70–sm130/GB300, NVIDIA TEE, ROCm MI350X/MI455X, Metal, OpenVINO NPU, Qualcomm, all RISC-V NPUs, FPGA, ternary accelerators, CXL, and NIXL. These are not promoted to “implemented but unverified” where the source itself only provides profiles/plans or simulations.

## 19. Requirements that are unrealistic as currently specified

- “Compile once, run anywhere” cannot mean one hardware-specific native kernel artifact runs unchanged on CUDA, ROCm, Metal, CPU, NPU, FPGA, and RISC-V. A technically honest design needs portable IR plus target-specific compiled variants or a target-local compilation step.
- A single AEG cannot guarantee optimal performance across hardware generations without target-specific kernels, calibration, and runtime capability negotiation.
- Hardware-backed TEE, CXL, NIXL, NVIDIA Rubin/GB300, MI455X, FPGA, and RISC-V targets cannot be validated by software profile objects alone.
- Quality gates require real datasets, baseline checkpoints, deterministic methodology, and acceptance thresholds; a length/nonempty heuristic cannot substitute for benchmark evaluation.
- “Any Hugging Face model” requires an explicit supported architecture/format matrix and a fail-closed behavior for unknown/unsupported models, rather than random-weight fallback.

## 20. Exact fixes required before a full-implementation claim

1. Make real model ingestion mandatory for production compile; remove or quarantine synthetic-weight fallback and mark synthetic fixtures explicitly.
2. Implement authentic tokenizer/weight/graph paths and E2E tests for representative Qwen, Llama, DeepSeek/MLA, Mistral/MoE, VLM/video, GGUF, SafeTensors, ONNX, LoRA, and ternary models.
3. Repair `CompilerConfig` serialization/clone to preserve every v4/v5 option and register all 22 passes in the installed plugin/discovery path.
4. Define and emit the correct AEG/1.1, AEG/2.0, and AEG/3.0 formats, with migration/version validation and complete payload hash verification.
5. Replace kernel plans/placeholders with invoked, verified target toolchains and executable artifacts, or narrow the supported target claim.
6. Integrate each runtime layer into the actual generation lifecycle, including concurrency, isolation, fallback, metrics, and failure recovery.
7. Fix structlog configuration and add subprocess CLI tests for every documented command.
8. Move FastAPI request models to module scope or explicitly annotate body parameters; add contract tests for every JSON endpoint.
9. Implement the missing REST and gRPC interfaces or remove them from the PRD/documentation.
10. Replace the fake eval gate with real benchmark adapters and enforce artifact rejection on configured regression thresholds.
11. Implement real authentication/authorization, safe CORS defaults, tenant isolation, MCP policy, kernel sandbox/signatures, malicious artifact checks, and hardware attestation.
12. Make Hub offline behavior fail clearly instead of writing a manifest JSON as if it were a downloadable AEG.
13. Add bounded offline/network behavior, reproducible dependency locking, native build instructions, and a fresh Linux/Windows/macOS installation matrix.
14. Finish the test suite with unit/integration/E2E/hardware/performance/quality/security categories and eliminate hangs.

## 21. Final scorecard

Percentages are weighted audit judgments: “Completion” measures PRD surface attempted/represented, “Functional” measures real behavior, “Tested” measures meaningful exercised coverage, and “Production Ready” requires reliable external installation and operation. Hardware-only rows are not counted as functional without physical execution.

| Category | Completion | Functional | Tested | Production Ready |
|---|---:|---:|---:|---:|
| Model ingestion | 55% | 20% | 20% | 10% |
| AEG format | 65% | 35% | 40% | 20% |
| Optimizer | 92% | 32% | 55% | 12% |
| Hardware backends | 100% profiles | 5% | 10% | 0% |
| Runtime | 72% | 25% | 55% | 8% |
| CLI | 52% | 25% | 10% | 5% |
| Python SDK | 58% | 25% | 30% | 8% |
| REST API | 45% | 12% | 10% | 3% |
| gRPC | 0% | 0% | 0% | 0% |
| Evaluation | 20% | 2% | 2% | 0% |
| Performance | 15% | 0% | 0% | 0% |
| Observability | 50% | 15% | 25% | 5% |
| Safety | 45% | 10% | 5% | 2% |
| Hub | 50% | 10% | 5% | 2% |
| Distributed execution | 35% | 2% | 0% | 0% |
| Documentation | 70% | 45% | 20% | 25% |
| Installation/distribution | 55% | 20% | 10% | 5% |

Weighted true completion score, emphasizing ingestion/compiler/AEG/runtime/backend correctness over documentation: **29% PRD surface completion, 13% functional completion, 17% meaningfully tested, 5% production readiness.** The “tested” number is deliberately not code coverage; it reflects the presence of tests that reach real behavior, reduced for synthetic/mocked coverage and incomplete suite execution.

## 22. Final answers

### If I give this repository to a completely new developer today, can they install Aether, take a real Hugging Face model, compile it into a real AEG artifact, run it, serve it through the API, and receive correct model output without manually fixing source code?

**NO.** Source import and a graph-only synthetic AEG are possible, but clean installation fails in the tested fresh build, CLI inference crashes, REST JSON inference returns 422, and the successful runtime output is not correct output from the named Hugging Face model.

### If I claim on GitHub that Aether is fully implemented according to both PRDs, is that technically honest?

**NO.** That claim becomes honest only after the exact fixes in Section 20 are completed and demonstrated with real model, artifact, backend, API, quality-gate, security, installation, and hardware evidence. A qualified statement such as “architecture and experimental scaffolding for v3.1–v5.0; CPU synthetic execution and partial compiler/runtime modules; hardware and production paths incomplete” would be accurate today.

### Questions that genuinely require a decision

1. Do you have physical access to the claimed CUDA TEE, Rubin/GB300, MI350X/MI455X, Metal, Qualcomm, RISC-V, FPGA, CXL, and NIXL environments, or should those claims be removed/narrowed to `UNVERIFIED — HARDWARE REQUIRED`?
2. Should graph-only/synthetic compilation be removed from the production CLI, or exposed only under an explicit `--synthetic-fixture` mode that cannot produce a normal runnable model artifact?
3. Should the support promise be narrowed to a tested model/format/target matrix until real compatibility tests exist?

## Recommendation

**FIX FIRST — MAJOR REWORK.** Do not ship a “fully implemented according to both PRDs” claim. The repository is a substantial architectural scaffold with some functioning CPU/data-structure components, but the central compile-real-model → portable-AEG → correct-inference → serve-through-API promise is not currently met.
