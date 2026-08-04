# Changelog

All notable changes to Aether Runtime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-04 — PRD v5.0 (RLVR, LoRA+, Sub-2-Bit, Video, PEFT)

### Added — Compiler Passes (Batch 3: Passes 19–22)
- **Pass 19 — Sub-2-Bit Quantization**: BitNet b1.58 ternary (1.58-bit), BTC-LLM
  binary codebook, and NanoQuant trellis quantization. Enforces perplexity quality
  gate before committing compressed weights. Writes `sub2bit_manifest.json`.
- **Pass 20 — Video Token Compression**: STC (Spatial-Temporal Compression), STORM
  temporal projector, and StreamingTOM-style compression plan generation for
  VLM video inference. Skips non-VLM architectures automatically.
- **Pass 21 — Advanced PEFT Compilation**: LoRA+ (lambda-scaled B matrix),
  LoRAMoE (mixture of experts), MoLF (mixture of LoRA functions), LoRAFusion
  (task-specific adapter selection). Compiles adapter blobs to `.aeg/adapters/`.
- **Pass 22 — RLVR Verifier Head Injection**: Injects GRPO verifier head with
  sympy / pytest / LLM-judge / human-feedback backends. K2V sub-task decomposition
  opcodes. Writes `rlvr_config.json` for Runtime R12.

### Added — Runtime Layers (R9–R12)
- **R9 — Sub-2-Bit KV + Weight Cache**: Ternary KV cache (4 vals per byte, absmean
  quantization), ternary GEMM reference kernel, LRU decompressed weight cache.
- **R10 — Video Frame KV Manager**: Scene-adaptive frame sampling via inter-frame
  cosine similarity, StreamingTOM time-decaying importance eviction, STC compression,
  three-tier temporal attention routing (recent / mid-term / summary).
- **R11 — Semantic KV Cache**: HNSW ANN index with hnswlib acceleration (pure-Python
  fallback), similarity-threshold cross-request KV deduplication, LRU eviction with
  configurable memory budget.
- **R12 — RLVR Training Harness**: Full GRPO rollout loop (K sampling), sympy /
  pytest (sandboxed subprocess) / LLM-judge verifiers, K2V dense reward
  decomposition, pass@k metric, GRPO clipped surrogate loss.

### Added — Tests
- `tests/test_passes_v2.py`: 40+ tests covering passes 10–22 and pipeline integration.
- `tests/test_runtime_v2.py`: 50+ tests covering R1–R12 algorithm correctness.

## [0.2.0] - 2026-08-03 — PRD v4.0 (Speculation, Grammar, Merging, TTT, Green, TEE)

### Added — Compiler Passes (Batch 2: Passes 10–18)
- **Pass 10 — MTP Head Compilation**: Detects Multi-Token Prediction heads from
  DeepSeek V3 / L-MTP style models and compiles them to AEG speculation blobs.
- **Pass 11 — Grammar Constraint Compilation**: Thompson NFA → DFA → minimised FSA
  pipeline for JSON, regex, EBNF, and CFG grammars. Writes FSA binary blob
  for Runtime R3 enforcement. < 50 µs decode overhead (XGrammar 2026).
- **Pass 12 — Model Merging**: Task Arithmetic, DARE (random weight dropping),
  TIES (sign consensus), and FREE (Fisher-weighted) model merging at compile time.
- **Pass 13 — TTT Fast-Weight Injection**: Injects per-layer LoRA-rank fast-weight
  slot descriptors (A, B, µ, σ) for test-time training. Writes TTT config.
- **Pass 14 — Semantic KV Compression**: ChunkKV, SentenceKV (boundary-aware),
  and PyramidKV (pyramid-shaped per-layer retention ratio) compression plan
  generation. Writes `kv_compression_plan.json`.
- **Pass 15 — Cross-Layer KV Sharing**: Middle-outward xKV sharing pattern with
  exponential cosine similarity model. Reduces KV memory 15–40%.
- **Pass 16 — Green Energy Compilation**: DVFS (freq/voltage) breakpoint assignment
  per operator based on compute-to-memory ratio. Carbon intensity profile from
  region table (MELODI 2026). Writes `green_profile.json`.
- **Pass 17 — TEE Kernel Wrapping**: AES-256-GCM weight encryption, HMAC-SHA256
  kernel guards, SHA-256 weight hash manifest. Supports NVIDIA CC, Intel TDX,
  AMD SEV-SNP. Writes `tee_config.json` + `weight_hash_manifest.json`.
- **Pass 18 — MDLM Drafter Compilation**: Masked Diffusion Language Model drafter
  with cosine noise schedule (Sahoo ICML 2025). DiffuSpec / SpecDiff speculative
  integration. Writes `drafter_config.json` + `schedule.json`.

### Added — Runtime Layers (R1–R8)
- **R1 — P-EAGLE Engine**: SM-partitioned parallel speculative decoding. MTP/EAGLE-3/
  MDLM/Hybrid modes. Optimal-transport acceptance criterion (Leviathan 2023).
  Adaptive K adjustment. Up to 4× AR throughput on H100 SXM5.
- **R2 — Multi-Agent KV Coordinator**: Shared KV block registry with reference
  counting, copy-on-write private tails, prefix hash deduplication, LRU eviction.
- **R3 — Grammar FSM Engine**: Loads FSA binary blob, O(1) mask lookup, per-request
  session isolation, multi-grammar registry.
- **R4 — SLO Scheduler**: MLFQ / SJF / FCFS scheduling with chunked prefill
  (Sarathi-Serve), 4-tier priority queues, deadline-aware priority boosting.
- **R5 — TTT Fast-Weight Engine**: In-Place TTT (arXiv 2026) online LoRA A/B
  gradient steps and LayerNorm µ/σ updates, per-request and session-scoped modes.
- **R6 — MCP Integration Layer**: JSON-RPC 2.0 MCP protocol support. Three-strategy
  tool call detection (JSON pattern / XML tag / function-call XML). Multi-server
  registry, concurrent call semaphore.
- **R7 — Green Power Manager**: DVFS hint enforcement, TDP cap throttling with
  hysteresis, carbon routing across geo regions, energy/carbon estimation.
- **R8 — TEE Runtime Manager**: NVIDIA CC / Intel TDX / AMD SEV-SNP enclave
  lifecycle. SHA-256 weight hash verification, kernel enter/exit guards,
  attestation token generation, periodic heartbeat re-attestation.

### Added — Core Infrastructure
- Extended `HardwareTarget` enum: 13 new targets (Rubin R100, Blackwell Ultra GB300,
  AMD MI450X, AWS Trainium3, Google TPU v5p, Intel Gaudi 3, etc.)
- Extended `DType` enum: TERNARY_158, BINARY, UINT2, FP4, FP6, FLOAT8_E4M3,
  FLOAT8_E5M2.
- 35+ new `CompilerConfig` fields for passes 10–22.
- `src/aether/runtime/__init__.py` updated to export all R1–R12 classes.

## [Unreleased]

### Added
- Initial project structure and foundational packages.
- `aether.core` package: AEG format, AEG-IR, computation graph, tensor types, and content-addressed hashing.
- `aether.compiler` skeleton: `Compiler` and `CompilerConfig`, dry-run planning, quality reports, and optimizer pass manager.
- Stage 1 ingestion loaders: SafeTensors, GGUF, ONNX, MLX, PyTorch.
- Architecture detector for Llama, Qwen, Gemma, Mistral, DeepSeek, and MoE families.
- Stage 2 optimizer passes 1–6: operator fusion, sensitivity analysis, precision assignment, KV cache structuring, MoE routing, and automatic parallelism discovery.
- Stage 3 targeting: hardware profiles, target registry, backend selector, and kernel emitter.
- Backend plugin system with stable `Backend` interface.
- Reference backend plugins: vLLM, llama.cpp, TensorRT-LLM, MLX, ONNX Runtime, and PyTorch fallback.
- Stage 4 runtime: hardware fingerprinting, model registry, executor, disaggregated prefill/decode scheduler, global KV cache manager, tree-speculative decoding engine, and dynamic precision manager.
- REST server with OpenAI-compatible endpoints and Prometheus metrics.
- `aether` CLI with `compile`, `pull`, `run`, `serve`, `bench`, `info`, `graph`, `list`, `rm`, `hw`, `kernels`, and `logs` commands.
- Python SDK: `Runtime`, `Compiler`, `CompilerConfig`, `RuntimeConfig`, and response objects.
- Aether Hub client with local content-addressed kernel cache.
- Quantization package: formats, sensitivity scoring, precision assignment, and packing utilities.
- MoE package: router, expert manager, sparsity analysis, and planner.
- Parallelism package: planner, mesh, sharding, and distributed primitives.
- Targets package: hardware target profiles and Triton-style kernel templates.
- Utilities: logging, profiling, telemetry, memory management, file I/O, and threading.
- Comprehensive test suite: unit tests, integration tests with real HuggingFace models, and backend discovery tests.
- CI/CD pipelines: Linux, macOS, and Windows test matrix, integration tests, documentation build, and release pipeline.
- Documentation: README, architecture docs, AEG format specification, optimizer passes, runtime, API reference, and research foundation.
- Examples: compile-and-run, serve OpenAI, benchmark, and custom kernel target.
- Benchmark suite with comparison tooling and public leaderboard hooks.

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2026-07-27

### Added
- Initial alpha release of Aether Runtime.
- Foundation compiler and runtime as described in the PRD v2.0.
- Public Python SDK and CLI.
- OpenAI-compatible REST API.
- Research-backed optimizer passes and runtime intelligence.

[Unreleased]: https://github.com/aether-dev/aether-runtime/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aether-dev/aether-runtime/releases/tag/v0.1.0
