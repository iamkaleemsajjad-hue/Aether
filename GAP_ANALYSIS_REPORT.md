# Aether Runtime — Gap Analysis Report

> Read-only analysis. No source files were modified.
> Project: `C:\Users\pc\Desktop\Aether Runtime` — a Python ML model compiler / runtime ("compile once, run anywhere") that turns any model into a portable **AEG (Aether Execution Graph)** artifact.

---

## 1. Overall summary

### What the project claims to be
Aether markets itself (README, PRDs) as "the LLVM for AI models": a five-stage compiler (ingestion → optimizer → hardware targeting → runtime → developer interface) that compiles any Hugging Face / GGUF / ONNX / MLX / PyTorch model into a self-contained AEG package runnable unchanged on CUDA (Hopper/Ada/Blackwell), AMD MI300X, Apple Silicon, Intel NPU, Qualcomm AI 100, RISC-V NPUs, and CPU AVX-512. Feature surface spans **22 optimizer passes**, **12 runtime layers (R1–R12)**, three AEG format versions (1.1 / 2.0 / 3.0), REST + gRPC APIs, a Python SDK/CLI, a model Hub, distributed execution, TEE confidential computing, evaluation gates, observability, and safety guardrails.

### Version inconsistency (documentation defect)
- `pyproject.toml`: `version = "1.0.0"`, classifier `Development Status :: 5 - Production/Stable`.
- `README.md`: "AEG Package Format v3.1", "production-grade".
- `CHANGELOG.md`: latest is `[0.4.0]` (Aug 2026), with `[0.1.0]` the only tagged release, plus a large `[Unreleased]` block.
- `PRD.md` = v3.0/3.1 baseline; `PRD_v2.md` = v4.0 + v5.0 net-new features.
The version numbers across the packaging metadata, README, and changelog do not agree.

### Biggest gaps (the honest picture)
The bundled **`AUDIT_REPORT.md`** (a self-described "Final Adversarial Audit") is unusually candid and its verdict is **"NOT COMPLETE — FIX FIRST / MAJOR REWORK."** The overall audit-weighted scores it reports:

| Metric | Score |
|---|---|
| PRD/code coverage | 60% |
| Functional coverage | 42% |
| Tested coverage | 52% |
| **Production readiness** | **20%** |

The central verified capability is a **single CPU path**: a *local* tokenizer-backed SafeTensors (or `torch.save`) checkpoint can be ingested → compiled → packaged into AEG/1.1 → reloaded → generate → served over REST/gRPC. Everything beyond that (arbitrary HF models, GPU/accelerator targets, most v4/v5 payloads, distributed execution, performance claims) is **partial, unverified, or not implemented**. The code's redeeming quality is that it now **fails closed** (raises explicit errors) rather than fabricating weights/output.

Biggest concrete gaps:
1. **Hardware backends (~35% / 10% functional):** 28 target profiles are registered but only `cpu_avx512` actually emits and executes a native kernel. No PTX/cubin/HSACO/metallib/QNN/FPGA/RISC-V binaries are produced; accelerator targets fail closed.
2. **Performance (~10% / 5% functional):** No PRD performance claim (tokens/sec, TTFT, TBT, latency percentiles, VRAM, energy, KV reduction, speculative acceptance) is validated. All figures are "CLAIM NOT VALIDATED."
3. **Distributed execution (~20% / 10% functional):** "collectives are CPU reference operations." No multi-process/multi-node/NCCL/RCCL/NIXL/GPU execution.
4. **Evaluation (~25% / 20% functional):** measured local evaluators exist but no official version-pinned benchmark corpora are bundled; gates are opt-in.
5. **v4/v5 payloads:** many optimizer passes and runtime layers (R8 TEE, R12 CXL, Pass 17 TEE, Pass 18 MDLM) are metadata/simulation/fail-closed rather than executable.

---

## 2. Source tree survey

Root layout (28 top-level items). Key documents: `PRD.md`, `PRD_v2.md`, `AUDIT_REPORT.md`, `REMEDIATION.md`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `pyproject.toml`, `Makefile`, `coverage.xml`, `LICENSE` (Apache-2.0), and an empty stray file named `set`.

### `src/aether/` — the package (the bulk of the code)
Top-level modules: `cli.py`, `sdk.py`, `__init__.py`, `__main__.py`. Subpackages:

| Package | Files (key) | What it does |
|---|---|---|
| `compiler/` | `compiler.py`, `config.py`, `plan.py`, `report.py`, `weight_quantizer.py`, `aeg_format_v2.py` + `calibration/`, `stage1_ingestion/`, `stage2_optimizer/`, `stage3_targeting/` | The five-stage compiler orchestrator. |
| `compiler/stage1_ingestion/` | `ingestion.py`, `architecture_detector.py`, `safetensors_loader.py`, `gguf_loader.py`, `onnx_loader.py`, `mlx_loader.py`, `pytorch_loader.py`, `vlm_loader.py`, `ssm_loader.py`, `graph_tracer.py` | Model loading + architecture detection + graph extraction. |
| `compiler/stage2_optimizer/` | `optimizer.py` (orchestrator, all 9 v3.1 passes), `pass1..pass22_*.py` entry points, `pass8_minference.py`, `base_pass.py` | 22-pass optimizer pipeline. Passes 1–9 live in `optimizer.py`; `passN_*.py` are façade/re-export entry points. |
| `compiler/stage3_targeting/` | `hardware_profile.py`, `target_registry.py`, `backend_selector.py`, `kernel_emitter.py`, `riscv_npu_ir.py`, `target_cpu/cuda/metal/rocm/openvino/riscv_*.py` | Hardware target profiles + kernel emission. Emits *plans*; only CPU emits a real library. |
| `compiler/calibration/` | `datasets.py`, `perplexity.py`, `sensitivity.py` | Calibration data + perplexity/sensitivity scoring (synthetic samples). |
| `core/` | `aeg_format.py`, `aeg_format_v31.py`, `aeg_ir.py`, `graph.py`, `weight_store.py`, `types.py`, `hash_utils.py`, `constants.py`, `exceptions.py` | AEG package format, AEG-IR, computation graph, content hashing. |
| `runtime/` | `runtime.py`, `cpu_engine.py`, `aeg_loader.py`, `kv_cache.py`, `scheduler.py`, `executor.py`, `eagle.py`, `speculative.py`, `precision_manager.py`, `model_registry.py`, `long_context.py`, `hot_reload.py`, `agentic_session.py`, `compute_controller.py`, `cascade_router.py`, `session.py`, `r1..r12_*.py` | Execution engine + all v4/v5 runtime layers R1–R12. |
| `backends/` | `base.py`, `torch_backend.py`, `onnx_backend.py`, `vllm_backend.py`, `llamacpp_backend.py`, `trtllm_backend.py`, `mlx_backend.py`, `hardware_backends.py`, `registry.py` | Backend plugin system. Real: torch, ONNX (tokenizer-backed). Others fail closed / CPU simulation. |
| `server/` | `app.py`, `routes.py`, `grpc_service.py`, `metrics.py`, `middleware.py`, `proto/` (`aether.proto`, `aether_pb2.py`, `aether_pb2_grpc.py`) | FastAPI REST (67 `/v1` routes) + typed gRPC service/client. |
| `quantization/` | `codecs.py`, `formats.py`, `assignment.py`, `packing.py`, `pruning.py`, `sensitivity.py`, `quantizer.py` | Quantization codecs (incl. distinct MXFP4 vs FP4) + orchestrator. |
| `hub/` | `client.py`, `server.py`, `cache.py`, `auth.py` | Hub client (real HTTP + local archive fallback) + a FastAPI Hub server. |
| `safety/` | `guardrails.py`, `policy.py`, `production_safety.py` | Prompt injection/toxicity detection, output PII/secret redaction, audit log. |
| `observability/` | `otel.py`, `ci_pipeline.py`, `gates.py`, `evaluators.py` | Custom OTLP-shaped tracing/metrics, eval gate, drift/AB rollout, benchmark evaluators. |
| `parallelism/` | `distributed.py`, `mesh.py`, `planner.py`, `sharding.py` | Sharding plans + CPU-reference collectives (not real distributed). |
| `attention/` | `mla.py` | MLA detector + compile-time planner (`MLAPlanner`, `MLACompressionPlan`). |
| `inference/` | `multimodal.py`, `rag.py` | Compile-time multimodal graph plan + RAG pipeline plan. |
| `adapters/` | `lora.py` | LoRA adapter/hot-swap engine (dual API generations). |
| `moe/` | `router.py`, `expert_manager.py`, `sparsity.py`, `planner.py` | MoE routing/planning (no real MoE model run). |
| `hybrid/` | `state.py` | SSM/Mamba/RWKV state + `HybridMemoryPool`. |
| `provenance/` | `manifest.py`, `fingerprint.py`, `watermark.py` | EU AI Act / C2PA provenance, IP fingerprint (ZK proof is a placeholder), watermark config. |
| `distillation/` | `pipeline.py`, `reward_model.py` | Distillation + Process Reward Model. |
| `cuda/` | `graphs.py`, `graph_manifest.py` | CUDA graph capture plan + manifest writer (metadata; no CUDA on this host). |
| `ecosystem/` | `sdks.py`, `vscode_plugin.py` | TS/Go/Rust SDK + VS Code extension generators. |
| `fleet/` | `manager.py`, `health.py` | Fleet placement, health monitor, autoscaler (planning layer). |
| `targets/` | `cpu_kernels.py`, `cuda_kernels.py`, `metal_kernels.py`, `rocm_kernels.py`, `templates.py`, `registry.py` | Kernel template strings + CPU kernels. |
| `kernels/` | `native_cpu.py` (real C++→ctypes), `attention.py`, `ffn.py`, `gemm.py`, `norm.py`, `rope.py`, `base.py` | Native CPU kernels with numpy references + graceful degradation. |
| `utils/` | `logging.py`, `profiling.py`, `telemetry.py`, `memory.py`, `file_io.py`, `threading.py` | Cross-cutting utilities. |

### `tests/` (test suite)
- `tests/unit/` — 44 unit test modules (e.g. `test_aeg_format.py`, `test_aeg_format_v2.py`, `test_compiler.py`, `test_optimizer_passes.py`, `test_runtime*.py`, `test_quantization.py`, `test_kernels.py`, `test_native_cpu_kernels.py`, `test_hardening_contracts.py`, `test_cli_contracts.py`, `test_hub*.py`, `test_server.py`, `test_targets.py`, `test_riscv_and_hardware.py`, `test_phase{2..6}_*.py`, `test_v31_*.py`, `test_e2e_compile_run_cpu.py`, etc.).
- `tests/integration/` — `test_end_to_end.py` (13 tests) and `test_local_safetensors_aeg_roundtrip.py` (18 tests: the real local SafeTensors/PyTorch/GGUF/BitNet/LoRA/MTP/gRPC/eval/merge/KV round-trips).
- `tests/` root — `test_passes_v2.py` (passes 10–22), `test_runtime_v2.py` (R1–R12, 83 tests), `conftest.py` (fixtures incl. the tiny local Llama checkpoint), `__init__.py`.
- `tests/data/` and `tests/fixtures/` are **empty** — no bundled datasets or model fixtures (fixtures are generated at runtime via `pytest.importorskip` on safetensors/tokenizers/transformers).

### `proto/`
Single file `aether.proto` — proto3 `AetherRuntime` service with `Generate`, `GenerateStream`, `Health` and typed request/response/chunk messages. A copy plus generated bindings live under `src/aether/server/proto/`.

### `examples/`
`01_compile_and_run.py`, `02_serve_openai.py`, `03_benchmark.py`, `04_custom_kernel_target.py` — call the public `Compiler`/`Runtime`/`create_app`/`TargetRegistry` APIs. Examples 1–3 reference `Qwen/Qwen3-0.6B` which requires network access.

### `benchmarks/`
`benchmark_runner.py`, `bench_suite.py`, `latency_bench.py`, `models.csv`, `README.md`. Measures TTFT/TBT/throughput/latency/memory/energy, but per the audit no real-model/baseline comparison has been run.

### `scripts/`
`check_env.py`, `setup_dev.py`, `compile_model.py`, `run_inference.py`, `inspect_aeg.py`, `convert_weights.py`, `benchmark_kernels.py`, `profile_memory.py`, `ci_smoke_test.py` (standalone 15-test smoke test, no pytest/network/GPU), and `_analyze_stubs.py` (an AST stub-counter). `check_env.py` fails under Windows CP1252 consoles because it prints Unicode check-marks.

### `docs/`
`index.md`, `getting-started.md`, `architecture.md`, `aeg-format.md`, `optimizer-passes.md`, `runtime.md`, `runtime_layers_v2.md`, `api-reference.md`, `research.md`, `roadmap.md`, `conf.py` (Sphinx). Note: `optimizer-passes.md` and `architecture.md` still describe **"six"/"9" passes** while the code has 22 — docs lag the implementation.

Other: `research/research_foundation.md` (paper→feature mapping), `build/`, `dist/`, `src/aether_runtime.egg-info/` (build artifacts, not readable).

---

## 3. The 17 audit categories

For each: **(a)** implementing files, **(b)** what the audit says is missing/incomplete, **(c)** concrete stubs / fail-closed / NotImplementedError / skips found by grep. Scorecard values are quoted from `AUDIT_REPORT.md §27`.

### 3.1 Model ingestion — 65% / 45% functional / 25% prod
**(a) Files:** `compiler/stage1_ingestion/ingestion.py`, `architecture_detector.py`, `safetensors_loader.py`, `gguf_loader.py`, `onnx_loader.py`, `mlx_loader.py`, `pytorch_loader.py`, `vlm_loader.py`, `ssm_loader.py`, `graph_tracer.py`.
**(b) Audit:** SafeTensors (M1), GGUF (M2), ONNX (M3), PyTorch/HF (M5) are "FUNCTIONAL BUT INCOMPLETE" — proven only for *tiny local* checkpoints; broader/remote HF variants unverified. MLX (M4) "NOT TESTABLE ON CURRENT HARDWARE." VLM/video/MLA/MoE/SSM/reasoning/MTP detection (M6) is "PARTIAL" — only DeepSeek-style MTP and Qwen2/Gemma/Mistral tiny fixtures pass; no real MLA/MoE/VLM/video/SSM run. "Aether cannot honestly claim any Hugging Face model support based on this evidence."
**(c) Concrete findings:**
- `gguf_loader.py`: `Dequantization not implemented for GGML type ...` (unsupported quant types raise); GGUF files without embedded tokenizer vocab fail explicitly.
- Multimodal `process_image()` rejects a configuration-only dispatcher (fails closed without learned weights).
- Compiler fails closed for unknown model identifiers (no invented default architecture).

### 3.2 AEG format — 76% / 64% / 33%
**(a) Files:** `core/aeg_format.py` (AEG/1.1 + 3.0), `core/aeg_format_v31.py`, `compiler/aeg_format_v2.py` (AEG/2.0), `core/weight_store.py`, `core/aeg_ir.py`.
**(b) Audit:** AEG/1.1 (A1) "FUNCTIONAL BUT INCOMPLETE" (CPU path only). AEG integrity (A4) is the one **✅ COMPLETE** item — tampered manifests/weights/artifacts are rejected before execution. AEG/2.0 (A2) and AEG/3.0 (A3) "PARTIAL" — created packages write *explicit disabled descriptors*; enabled manifest claims are validated against real payloads, but "many v5 payloads … remain incomplete"; legacy↔canonical migration incomplete.
**(c):** `test_aeg_format_v2.py::test_config_stubs_created` — the format intentionally writes disabled/stub descriptors; validation rejects enabled-flag-without-payload.

### 3.3 Optimizer — 85% / 55% / 31%
**(a) Files:** `compiler/stage2_optimizer/optimizer.py` (passes 1–9) + `pass10..pass22_*.py`, `pass8_minference.py`, `base_pass.py`.
**(b) Audit (per-pass §6):**
- Passes 1,3,4 → FUNCTIONAL BUT INCOMPLETE (no target megakernel / no quality validation).
- Passes 2,5,6,7,20,22 → PARTIAL (synthetic calibration; plans only; no real MoE/VLM/quality effect; RLVR has no gradient/optimizer path).
- Passes 8,9,10,11,12,13,14,15,16,19,21 → FUNCTIONAL BUT INCOMPLETE (CPU-consumed but heuristic classifiers, no vendor kernels, no quality gates).
- **Pass 17 (TEE)** → **NOT IMPLEMENTED** (skips with `tee_backend_artifacts_unavailable`).
- **Pass 18 (MDLM drafter)** → **STUB / PLACEHOLDER** (fails closed with `mdlm_drafter_weights_unavailable`).
**(c):** `pass18_mdlm_drafter.py` fails closed without a trained drafter bundle; `pass19_sub2bit_quant.py`, `pass12_model_merging.py`, `calibration/datasets.py` contain `raise NotImplementedError`; `test_optimizer_passes.py` asserts `pytest.raises(NotImplementedError)`.

### 3.4 Hardware backends — 35% / 10% / 5% (one of the biggest gaps)
**(a) Files:** `compiler/stage3_targeting/*` (`hardware_profile.py`, `target_registry.py`, `kernel_emitter.py`, `riscv_npu_ir.py`, `target_*.py`), `backends/hardware_backends.py`, `kernels/native_cpu.py`, `targets/*_kernels.py`.
**(b) Audit (§8):** 28 target profiles registered; **only `cpu_avx512` is FUNCTIONAL BUT INCOMPLETE.** All CUDA/ROCm/Metal/QNN/RISC-V/FPGA targets are PARTIAL/UNVERIFIED, STUB/PLACEHOLDER, or NOT TESTABLE. "The repository does not emit PTX, cubin, HSACO, metallib, QNN binaries, FPGA bitstreams, or RISC-V binaries; those requests fail closed." H2 kernel generation "PARTIAL." TensorRT-LLM backend has no executable engine loader.
**(c):** `kernel_emitter.py`: `executable kernel generation is not implemented for target {...}; ... fail closed here`; `kernels/base.py`: `raise NotImplementedError(...)`; `hardware_backends.py` header explicitly states non-available hardware uses a "CPU simulation layer"; `backends/base.py` has abstract `raise NotImplementedError`; `hardware_profile.py` has Rubin `is_placeholder: True`.

### 3.5 Runtime — 71% / 51% / 26%
**(a) Files:** `runtime/runtime.py`, `cpu_engine.py`, `aeg_loader.py`, `kv_cache.py`, `scheduler.py`, `executor.py`, `precision_manager.py`, `model_registry.py`, `eagle.py`, `speculative.py`, `r1..r12_*.py`.
**(b) Audit (§7 R1–R12):** Most layers "FUNCTIONAL BUT INCOMPLETE" and only partially connected to normal generation. Wired to the CPU path: R1 P-EAGLE (local MTP), R2 multi-agent KV (exact-prefix reuse), R3 grammar FSM, R4 SLO scheduler, R5 TTT, R6 MCP, R7 green accounting, R10 KV transfer (local tier only), R11 semantic cache. **Not real:** **R8 Confidential TEE → STUB/PLACEHOLDER** (software simulation, `hardware_backed=false`); **R9 diffusion → PARTIAL** (no drafter loaded by normal generation); **R12 CXL → STUB/PLACEHOLDER** (mmap/in-memory fallback, no physical CXL).
**(c):** `r8_tee_manager.py` — extensive "software simulation" / "simulation mode" fallbacks; `r12_cxl_kv_pool.py` — mmap/in-memory fallback; `r9_diffusion_spec_engine.py` fails closed without a drafter; `runtime.py` — "public convenience method must fail closed until one is configured."

### 3.6 CLI — 70% / 55% / 30%
**(a) Files:** `cli.py`, `__main__.py`.
**(b) Audit (§10):** Diagnostic commands work (`version`, `hw`, `kernels`, `list`, `graph`, `info`, `grammar`, `mcp`). `eval/safety/trace/reasoning/mla-stats/multi-agent/slo-status/kv-share/hub/train grpo/kernel generate/kv transfer-stats` are registered and smoke-tested but "backend-specific limitations remain." v5 valued flags (`--sub2bit ternary`, `--mdlm-K/-T`, `--video-compression`) now parse; `quantize-report`/`cache stats|flush` execute but need a real model artifact.
**(c):** `cli.py` — passes are "allowed to skip when their real inputs or backend [are unavailable]" and the CLI rejects turning a skipped pass into a "successful-looking artifact message"; `test_cli_contracts.py` asserts rejection of false-success.

### 3.7 Python SDK — 60% / 40% / 20%
**(a) Files:** `sdk.py`, `runtime/runtime.py`, `compiler/compiler.py`, `compiler/config.py`, `runtime/config.py`.
**(b) Audit (§11):** Imports/objects work; Runtime exposes the full PRD method surface (`generate/chat/embed/rerank/transcribe/benchmark/compile_async/eval_gate/generate_constrained/generate_video/generate_with_tools/get_attestation_report/grpo_train_step/…/merge/set_task_weights/multi_agent_session`). But several **fail closed**: `grpo_train_step` (no gradient backend), `generate_video` (no video encoder), TEE attestation (no hardware), cross-agent KV. "Accepted API shape is not counted as functional backend execution."

### 3.8 REST API — 60% / 40% / 20%
**(a) Files:** `server/app.py`, `server/routes.py`, `server/middleware.py`, `server/metrics.py`.
**(b) Audit (§12):** **67 `/v1` routes** registered (generate, chat, embeddings, rerank, transcribe, compile, models, hardware, kernels, metrics, health, tools/call, grammar, merge, ttt, targets, green, tee, video, train/grpo, sub2bit, …). `/v1/generate` + `/v1/chat` route real streaming + grammar. Unavailable grammar/TEE/merge return explicit **503/501**; video/GRPO report 501/failed when unsupported. Auth works when `AETHER_API_KEYS` set; with no keys the server intentionally accepts requests. "Real semantics remain partial."
**(c):** `routes.py`: `Rubin kernel profiling backend is not implemented in this runtime`.

### 3.9 gRPC — 50% / 45% / 20%
**(a) Files:** `proto/aether.proto`, `server/grpc_service.py`, `server/proto/aether_pb2.py`, `aether_pb2_grpc.py`.
**(b) Audit (§13):** Typed `Generate/GenerateStream/Health` exercised against a real local AEG with bearer auth + token streaming + unauthorized rejection. **Incomplete for production:** default channels are insecure TCP; optional TLS/mTLS exists but its cert-chain path was **not validated** on Windows; authZ is a single bearer token; bindings are runtime-built from the descriptor rather than `protoc`-generated in the build pipeline.
**(c):** `server/proto/aether_pb2_grpc.py` — default servicer methods `raise NotImplementedError("Generate"/"GenerateStream"/"Health")` (standard generated stubs); `test_hardening_contracts.py` asserts TLS "must be supplied together" fail-closed.

### 3.10 Evaluation — 25% / 20% / 5% (a big gap)
**(a) Files:** `observability/ci_pipeline.py`, `observability/evaluators.py`, `observability/gates.py`, `runtime.py`, `cli.py`, `server/routes.py`.
**(b) Audit (§15):** Gate framework + regression logic + measured evaluators exist. `JsonlBenchmarkEvaluator` and `DatasetBenchmarkEvaluator` (HellaSwag/MMLU/GSM8K/Math-500/AIME/HumanEval) run a real model callback; a deliberately poor result was blocked. **But:** no official version-pinned corpora bundled, no reference prompting/decontamination, HumanEval execution opt-in (not sandboxed), `score_override` is a CI replay, and the default compiler invocation is opt-in. Status PARTIAL.
**(c):** `observability/evaluators.py`, `ci_pipeline.py` — `raise NotImplementedError`; `ci_pipeline.py` "fails closed until such an evaluator is configured."

### 3.11 Performance — 10% / 5% / 0% (the biggest gap)
**(a) Files:** `runtime.py`, `benchmarks/*`, `scripts/benchmark_kernels.py`, `scripts/profile_memory.py`.
**(b) Audit (§16):** **No PRD performance claim validated.** Not reproduced: tokens/sec, TTFT, TBT, P50/P95/P99 latency, GPU utilization, VRAM, energy, CO₂, real-model compile time, real AEG size, KV reduction, speculative acceptance, or comparisons vs llama.cpp/vLLM/SGLang/TensorRT-LLM/MLX. "All PRD performance figures must be labeled: CLAIM NOT VALIDATED."

### 3.12 Observability — 65% / 55% / 30%
**(a) Files:** `observability/otel.py`, `server/metrics.py`, `observability/gates.py`.
**(b) Audit (§18):** Implemented: request spans, metrics, latency percentiles, throughput, KV/speculation fields, Prometheus text, OTLP-shaped JSON, and a real OTLP/HTTP JSON POST path (validated by a local collector test). **Custom, dependency-light** implementation — **not** the OpenTelemetry SDK; interoperability with external Jaeger/Tempo/OTLP unverified. Status PARTIAL.

### 3.13 Safety — 45% / 35% / 15%
**(a) Files:** `safety/guardrails.py`, `safety/policy.py`, `safety/production_safety.py`, `provenance/*`.
**(b) Audit (§17):** Positives — AEG hash verification, tamper/traversal/symlink rejection, cache-id normalization, API-key auth, MCP fail-closed, ONNX/PyTorch refuse fabricated output, `enable_safety_layer=True` enforces prompt/output policy + audit log, TEE attestation fails closed without hardware. **Risks:** remote code opt-in but not sandboxed; native kernel compilation runs subprocess/toolchains without isolation; multi-tenant isolation not demonstrated; authZ is a basic token check; no malicious-model fuzzing. Default safety is opt-in. Status PARTIAL.
**(c):** `provenance/fingerprint.py` — ZK proof is `placeholder — real ZK proof would be generated here`.

### 3.14 Hub — 35% / 15% / 5%
**(a) Files:** `hub/client.py`, `hub/server.py`, `hub/cache.py`, `hub/auth.py`.
**(b) Audit (§19):** `HubClient` has real HTTP + auth. Offline: upload stores a local manifest + ZIP bytes, search hits local manifests, download extracts the retained ZIP and refuses metadata-only payloads. **No live Hub server proof** in the audited environment; no real remote permissions/dedup/content-addressed workflow. (`hub/server.py` exists as a FastAPI registry but is unvalidated.) "offline mode remains a local cache fallback rather than a live Hub proof." Status PARTIAL.
**(c):** `hub/client.py` docstring: "local cache simulation" / "Local simulation — when the server is unreachable."

### 3.15 Distributed execution — 20% / 10% / 0% (a big gap)
**(a) Files:** `parallelism/distributed.py`, `mesh.py`, `planner.py`, `sharding.py`, `fleet/manager.py`, `fleet/health.py`.
**(b) Audit (§20):** Contains fleet placement, deployment manifests, sharding plans, CPU-reference collectives, hot-reload routing, scheduling. **Does not demonstrate** multi-process inference, multi-node comms, NCCL/RCCL/NIXL/UCCL, GPU allocation, node failure recovery, session migration, tenant-isolated KV, or cross-machine prefill/decode. "NOT IMPLEMENTED for real distributed execution."
**(c):** `parallelism/distributed.py` — all-reduce comment "For CPU collective simulation: just multiply by world_size"; barrier is "No-op in single-process simulation." (The module docstring optimistically claims "real multi-process distributed inference," contradicting the runtime reality.)

### 3.16 Documentation — 55% / 50% / 20%
**(a) Files:** `docs/*.md`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `research/research_foundation.md`, PRDs, `docs/conf.py` (Sphinx).
**(b) Assessment:** Extensive prose docs, but **out of sync with code**: `docs/optimizer-passes.md`/`architecture.md` describe 6/9 passes while code has 22; README says "v3.1"/"production-grade" while `pyproject.toml` says version 1.0.0; CHANGELOG top is 0.4.0. README lists CLI commands (`sign`, `verify`, `hub push/pull`, `sdk generate`) whose full semantics the audit marks partial. Tested-coverage of docs is low (25% column).

### 3.17 Installation / distribution — 45% / 30% / 15%
**(a) Files:** `pyproject.toml`, `Makefile`, `scripts/setup_dev.py`, `scripts/check_env.py`.
**(b) Audit (§3, I1):** Wheel builds (`aether_runtime-0.1.0-py3-none-any.whl`, 723,312 bytes) and installs with `--no-deps` in a system-site-packages venv; `aether --help` works. **But** normal isolated install fails offline while resolving build deps; a dependency-complete clean install + compile/run/serve/test workflow was **not achieved** (package index unreachable). `check_env.py` breaks under Windows CP1252 (Unicode check-marks). Status PARTIAL. (Note the wheel is `0.1.0` while `pyproject.toml` now declares `1.0.0`.)

---

## 4. Test setup, counts, skips, coverage

- **Runner:** pytest ≥ 8.0. `pyproject.toml [tool.pytest.ini_options]`: `testpaths=["tests"]`, `asyncio_mode="auto"`, `--strict-markers --strict-config -ra`, coverage on by default (`--cov=aether --cov-report=term-missing --cov-report=xml`). Markers: `slow`, `integration`, `gpu`, `cuda`, `metal`, `rocm`, `network`, `requires_backend`. README/REMEDIATION warn to **run serially** (`test_e2e_compile_run_cpu.py` and `test_v31_features.py` race on the shared `~/.aether` cache).
- **Current count (audit §3):** `pytest --collect-only` = **1,792 tests**. Last completed full no-coverage run = **1,777 passed, 15 skipped, 0 failed** in 208.89 s. Focused CPU/native slice = 100 passed, 2 skipped. `scripts/ci_smoke_test.py --verbose` = **15/15 PASS**.
- **Skips/xfail:** No `xfail` in the suite. Skips are limited to: network tests (auto-skipped unless `AETHER_RUN_NETWORK_TESTS=1`, via `conftest.py::pytest_collection_modifyitems`), `pytest.importorskip` on `safetensors`/`tokenizers`/`transformers`/`torch`/`grpc`, native-compiler-gated tests (`requires_compiler = pytest.mark.skipif(...)` in `test_native_cpu_kernels.py`; `test_hardening_contracts.py` skips if "host native compiler unavailable"), format loaders (`test_format_loaders.py` several `@pytest.mark.skipif`), and `test_e2e_compile_run_cpu.py` (`pytest.skip` when no HF weights). The 15 reported skips are network/HF-access or synthetic-model availability only.
- **Coverage report:** `coverage.xml` (Cobertura, coverage.py 7.15.2) exists at repo root. Its header reports **line-rate = 0.2351 (~23.5%)** (lines-valid 29,626 / covered 6,966) and **branch-rate = 0.0033** (25 of 7,650 branches). ⚠️ This on-disk XML (~23.5% line / ~0.3% branch) is **far lower** than the audit's prose mention of a prior "~68% combined statement/branch" run — likely a partial/interrupted or `--no-cov`-adjacent artifact. Either way, coverage ≠ feature completeness, and the checked-in number is low.
- **Test realism caveat (audit §22):** "many tests use MagicMock or synthetic graphs and do not exercise real model execution."

---

## 5. Verbatim gap items

### 5.1 From `AUDIT_REPORT.md`

**Executive verdict (§1):**
> **NOT COMPLETE** … The v4/v5 additions are mostly metadata planners, configuration emitters, isolated components, or emulation layers. AEG version selection now exists, but many v5 payloads and gRPC remain incomplete.

**Requirements matrix (§4), verbatim status cells:**
> P1 | Performance claims | … | **NOT IMPLEMENTED** — No real model/baseline comparison
> D1 | Distributed execution | fleet, parallelism | Planning layer | No | … Collectives are CPU reference operations | **NOT IMPLEMENTED**
> H1 | Hardware target registry | … Profiles and backend candidates only | **PARTIAL**
> HUB1 | Aether Hub | Client only | … no live Hub | **PARTIAL**

**Optimizer passes (§6):**
> 17 — TEE Enclave Emission | **NOT IMPLEMENTED** | The pass now skips with `tee_backend_artifacts_unavailable` unless a real backend-emitted executable kernel bundle is attached.
> 18 — Diffusion Drafter Compilation | **STUB / PLACEHOLDER** | The pass now fails closed with `mdlm_drafter_weights_unavailable` … R9 still lacks a complete compiler weight-ingestion path.

**Runtime layers (§7):**
> R8 Confidential TEE | **STUB / PLACEHOLDER** | Software simulation can initialize with hardware_backed=false; this is not confidential computing.
> R12 CXL Rack-Scale KV Pool | **STUB / PLACEHOLDER** | File-backed mmap and in-memory fallback exist; no physical CXL or rack-scale pool.

**Hardware (§8):**
> The repository does not emit PTX, cubin, HSACO, metallib, QNN binaries, FPGA bitstreams, or RISC-V binaries; those requests fail closed.

**Performance (§16):**
> No PRD performance claim was validated. … All PRD performance figures must be labeled: CLAIM NOT VALIDATED

**Critical bugs / blockers (§23):**
> 2. TensorRT-LLM backend has no executable engine loader in this repository.
> 3. Several v4/v5 optimizer artifacts are plans/configuration, not executable kernels or runtime tensors.
> 5. Runtime layers are only partially connected to normal generation; hardware/network layers remain unavailable here.
> 11. Distributed collectives are CPU reference operations, not multi-node execution.

**Final answer (§28):**
> If I claim on GitHub that Aether is fully implemented according to both PRDs, is that technically honest? **NO.**

**Recommendation (§30):**
> **FIX FIRST / MAJOR REWORK** — Do not ship the repository as fully implemented according to both PRDs.

### 5.2 From `REMEDIATION.md`

**Baseline vs result (header):**
> **Baseline:** 15 failed · 1169 passed · 3 collection errors
> **Result:** 0 collection errors; all 15 baseline failures resolved (last observed full run: 2 failed · 1322 passed, both since fixed)

**Root cause (Summary):**
> The dominant root cause was not missing modules. It was a set of **placeholder aliases** in package `__init__.py` files that bound v3.1 API names to unrelated classes … These satisfied `import` but not use.

**Collection errors (§1):**
> All three were missing re-exports, not missing implementations. … `pass9`'s error was masked behind `pass7`'s … so the audit reported 3 errors where there were 4 causes. **Unblocked:** … 161 tests that had never run.

**MXFP4 (§9):**
> **This is the one change made to a test rather than to source.** … The test encoded a factually incorrect claim, so it was corrected and **strengthened** into two tests.

**Audit Corrections (table):**
> The `passN_*.py` files are a **façade pattern**, not stubs: each is the stable public path for a pass whose implementation lives in the orchestrator. Two were genuinely broken (missing re-exports); the rest were correct as written.

**Regression guards / no-regression check:**
> modules @HEAD=165  @now=165 · Modules removed: none · Public symbols removed: NONE

---

## 6. Prioritized gap list (from AUDIT_REPORT §24)

**Priority 0:** (1) complete AEG/2.0 & 3.0 migration + executable consumption of every declared payload; (2) replace unavailable backend paths with real engine integrations; (3) complete gRPC production transport (protoc stubs, TLS/mTLS, authZ, interop); (4) connect v4/v5 runtime layers to executable inference; (5) run standard local dataset adapters against official version-pinned corpora and enforce measured quality gates.

**Priority 1:** real target-kernel compilation for CUDA/ROCm/Metal/OpenVINO/QNN/CPU; VLM/video ingestion+generation; validate MTP/P-EAGLE against a real DeepSeek/MTP model; complete BTC-LLM/NanoQuant + quality gate; production embedding packaging + distributed semantic cache; real KV transfer + distributed execution; replace Hub simulation with a real server or label client-only; harden model-loading trust boundaries; clean-install CI on Windows+Linux; real model-compatibility + quality regression suites.

---

*End of report — analysis only; no fixes applied.*
