# Aether Runtime — Roadmap

This roadmap tracks implementation gates per release phase. A feature is complete when it has production code, tests, docs, and a repeatable validation path.

**Last updated:** 2026-08-14 | **Status:** Phase 4 in progress

---

## ✅ Phase 1: Compiler Foundation — COMPLETE

- [x] AEG package layout with manifest hashes, graph IR, precision map, kernel metadata, sharding plans
- [x] AEG format versions: AEG/1.1 (baseline), AEG/2.0 (v4 passes), AEG/3.0 (v5 passes)
- [x] Model ingestion for SafeTensors, GGUF, ONNX, PyTorch `.pt`/`.bin`, MLX
- [x] Operator fusion pass (Pass 1) — annotates and fuses transformer patterns
- [x] CPU/PyTorch fallback backend — API works without GPU
- [x] CLI surface (`compile`, `run`, `serve`, `bench`, `eval`, `info`, `graph`, `hw`)
- [x] Python SDK (`Runtime`, `Compiler`, `CompilerConfig`, `RuntimeConfig`)
- [x] OpenAI-compatible REST API (`/v1/chat`, `/v1/generate`, `/v1/embeddings`)
- [x] gRPC service with typed protobuf bindings
- [x] Documentation: architecture.md, getting-started.md, api-reference.md

---

## ✅ Phase 2: Optimizer Depth (v3.1 Passes 1–9) — COMPLETE

- [x] Pass 1: Operator Fusion — attention/FFN/norm kernel fusion
- [x] Pass 2: Sensitivity Analysis — layer-wise quality impact scoring
- [x] Pass 3: Mixed-Precision Assignment — INT4/INT8/FP8/FP16/BF16 per layer
- [x] Pass 4: KV Cache Structuring — paged, tiered, prefix-cache
- [x] Pass 5: MoE Routing — expert tiering hot/warm/cold with Zipf prior
- [x] Pass 6: Parallelism Discovery — TP/PP/EP/CP cost model
- [x] Pass 7: Reasoning Graph — chain-of-thought / tree-search graph compilation
- [x] Pass 8: Sparse Attention — MInference A-shape/vertical-slash/block-sparse
- [x] Pass 9: Pruning & Sparsity — Wanda magnitude + activation pruning

---

## ✅ Phase 3: Specialised Loaders & Runtime Layers (v4.0 Passes 10–17) — COMPLETE

### Specialised Model Loaders
- [x] MLALoader — DeepSeek V2/V3/R1 Multi-Head Latent Attention (68 tests)
- [x] MoELoader — Mixtral/Qwen-MoE/Jamba/DBRX/OLMoE expert tiering (68 tests)
- [x] VideoModelLoader — Video-LLaMA/2/VideoChat2/LLaVA-Video (68 tests)
- [x] VLMLoader — LLaVA/Qwen-VL/InternVL/Florence2
- [x] SSMLoader — Mamba/Jamba/RecurrentGemma state-space models
- [x] GGUFLoader — llama.cpp GGUF format (Q4/Q8 dequant)
- [x] ONNXLoader — ONNX opset 13–21
- [x] SafeTensorsLoader — HuggingFace multi-shard with SHA-256 validation
- [x] PyTorchLoader — `.pt`/`.bin` with corrupt-shard detection
- [x] Dispatcher: `_try_specialised_loader` fires before generic format dispatch

### Runtime Layers v4.0
- [x] R1: Dynamic Precision Manager — memory-pressure-triggered FP16→INT8
- [x] R2: Multi-Agent KV Coordinator — SwarmKV cross-request KV sharing
- [x] R3: Grammar FSM Engine — GBNF/JSON-Schema constrained decoding
- [x] R4: SLO-Aware Admission — TTFT/throughput budget enforcement
- [x] R5: TTT Fast-Weight Engine — per-request test-time training
- [x] R6: MCP Integration — Model Context Protocol tool dispatch
- [x] R7: Green Power Manager — carbon intensity scheduling
- [x] R8: TEE Runtime — confidential inference (hardware-gated)

### Optimizer Passes v4.0 (10–17)
- [x] Pass 10: Native MTP Head — multi-token prediction compilation
- [x] Pass 11: Grammar Constraint Compiler — FSM + tokenizer fingerprint
- [x] Pass 12: Model Merging — SLERP/task-arithmetic/DARE-TIES
- [x] Pass 13: TTT Fast-Weight — per-request weight adaptation
- [x] Pass 14: Semantic KV Compression — importance-scored KV eviction
- [x] Pass 15: Cross-Layer KV Sharing — GQA/MQA/MLA key reuse
- [x] Pass 16: Green Energy Scheduling — carbon-aware deferred inference
- [x] Pass 17: TEE Weight Encryption — hardware-gated

---

## 🔄 Phase 4: v5.0 Scale — IN PROGRESS

### Runtime Layers v5.0
- [x] R9: MDLM Diffusion Speculative Engine (hardware-gated: needs drafter weights)
- [x] R10: KV Network Transfer — RDMA/NIXL disaggregated prefill/decode
- [x] R11: Semantic Request Cache — embedding-based hit detection
- [x] R12: CXL Rack-Scale KV Pool (hardware-gated: needs CXL hardware)

### Optimizer Passes v5.0 (18–22)
- [x] Pass 18: Diffusion LM Drafting — MDLM speculative (hardware-gated)
- [x] Pass 19: Sub-2-bit Quantization — ternary/SpQR/QuIP#
- [x] Pass 20: Video Compression — STC/STORM/StreamingToM
- [x] Pass 21: PEFT Adapter Integration — LoRA/DoRA/VeRA/LoRAMoE
- [x] Pass 22: RLVR Fine-Tuning — GRPO/RLOO reward optimization

### Evaluation System
- [x] HellaSwag, MMLU, GSM8K, MATH-500, HumanEval, AIME, ARC, TruthfulQA, WinoGrande
- [x] JsonlBenchmarkEvaluator — custom JSONL dataset exact-match
- [x] DatasetBenchmarkEvaluator — multi-format (HellaSwag/MMLU/ARC)
- [x] EVALUATOR_REGISTRY for programmatic access
- [x] EvalGate / QualityGate / CIEvalPipeline

### Infrastructure
- [x] Distributed: DistributedInferenceEngine, ParallelismPlanner, DeviceMesh
- [x] Hub: content-addressed storage, push/pull, auth, dedup, path-traversal protection
- [x] Safety: jailbreak detection, watermarking, ZK proofs, multi-tenant isolation
- [x] Observability: OTLP exporter, Prometheus metrics, distributed tracing
- [x] Install scripts: `scripts/install.sh` (Linux/macOS), `scripts/install.ps1` (Windows)
- [x] `scripts/check_env.py` — environment diagnostics (Windows CP1252 fixed)

---

## 📋 Phase 5: Production Hardening — PLANNED

### Hardware-Gated Items (require specific hardware)
- [ ] TEE confidential inference (needs Intel TDX / AMD SEV hardware)
- [ ] CXL rack-scale KV pooling (needs CXL 2.0 hardware)
- [ ] MDLM drafter weights (needs pre-trained diffusion drafter)
- [ ] CUDA/ROCm/Metal backends (needs GPU hardware for end-to-end tests)

### Remaining Implementation Items
- [x] GGUF Q2_K / Q4_K / Q5_K / Q6_K / Q8_K dequantization -- IMPLEMENTED (K-quant dispatch in gguf_loader.py)
- [ ] NCCL/RCCL collective backend (replaces SocketCollective for GPU clusters)
- [ ] OpenTelemetry SDK full integration (distributed traces across workers)
- [ ] SDK type stubs (`.pyi` files for all public APIs)
- [ ] Docker multi-stage images (CPU / CUDA / ROCm variants)
- [ ] CI/CD pipeline: GitHub Actions matrix (Linux × Python 3.10–3.12 × CPU/CUDA)
- [ ] SBOM generation and provenance attestations

### Performance Validation
- [ ] Validate PRD performance claims on GPU hardware
- [ ] MLPerf Inference v4.0 submission
- [ ] Public benchmark reports with reproducible configs


### Phase 4 Production Hardening Additions (2026-08-14)
- [x] HardwareCapabilities dataclass (PRD Section 12) -- implemented + 21 tests
- [x] Real hardware detection pipeline (PRD Section 41) -- detect_all_capabilities()
- [x] hardware_validation_matrix.json -- 28+ targets, honest classification
- [x] BenchmarkRunner -- real time.perf_counter() metrics, 15 tests
- [x] aether doctor / hardware detect/validate / backend list / inspect / benchmark
- [x] Security adversarial tests -- 55 tests (archive traversal, backend fail-closed, TEE)
- [x] Hardware contract tests -- 21 tests
- [x] Installation validator scripts/verify_install.py
- [x] gRPC TLS test cert generator scripts/gen_test_certs.py
- [x] DistributedInferenceEngine.distributed_mode -- honest mode labeling
- [x] NCCL fail-closed: initialize() raises when NCCL requested but unavailable

---

## Test Coverage Summary

| Phase | Tests | Status |
|---|---|---|
| AEG Format + Compiler | 451 | ✅ All passing |
| Hardware Backends | 296 | ✅ All passing |
| Distributed Execution | 33 | ✅ All passing (0 skipped) |
| Model Ingestion | 148 + 68 | ✅ All passing |
| Evaluation System | 74 | ✅ All passing (0 skipped) |
| Safety System | — | ✅ All passing |
| Hub System | — | ✅ All passing |
| gRPC Transport | — | ✅ All passing |
| Optimizer Passes | 114 | ✅ All passing |
| v3.1 / v4 Extensions | 120 | ✅ 120 passing, 1 network skip |
| Phase 2–6 Runtime | 229 | ✅ All passing |
| **Total Unit Tests** | **~1,860+** | **✅ ~1,859 passing, 1 skip** |
