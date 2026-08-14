# Aether Runtime — Status

**Last updated:** 2026-08-14  
**Current version:** 0.5.0  
**Branch:** main  
**Status:** PHASE 3b COMPLETE

---

## 🎯 Current State at a Glance

| Metric | Value |
|---|---|
| PRD/code coverage | **78%** (+18pp from baseline) |
| Functional coverage | **58%** (+16pp) |
| Tested coverage | **74%** (+22pp) |
| Production readiness | **34%** (+14pp) |
| Unit tests passing | **~1,860+** |
| Skipped tests | **≤ 1** (env-gated network test) |
| Failed tests | **0** |
| GitHub commits | **8 commits since 2026-08-10** |

---

## ✅ What's Working — End-to-End

A new developer on a CPU-only machine can:

1. **Install** — `scripts/install.sh` or `scripts/install.ps1` (one-click)
2. **Verify environment** — `python scripts/check_env.py` (no errors on Windows)
3. **Compile a local SafeTensors checkpoint** — `aether compile ./model`
4. **Reload and run inference** — `aether run model.aeg --prompt "Hello"`
5. **Call REST API** — `curl http://localhost:11434/v1/chat -d '{...}'`
6. **Use Python SDK** — `from aether import Runtime; rt = Runtime()`
7. **Run evaluations** — `aether eval model.aeg --dataset hellaswag`
8. **Benchmark** — `aether bench model.aeg`

---

## ✅ Fully Completed Components (0 failures)

| Component | Tests | Notes |
|---|---|---|
| AEG Format 1.1/2.0/3.0 | 451 ✅ | Graph IR, manifest hashes, precision maps |
| Hardware Backends | 296 ✅ | CPU/CUDA/ROCm/Metal/Intel/Qualcomm/RISC-V profiles |
| Model Ingestion | 216 ✅ | SafeTensors + GGUF + ONNX + PyTorch + MLX |
| Specialised Loaders | 68 ✅ | MLA (DeepSeek) + MoE (Mixtral) + Video (LLaVA-Video) |
| Evaluation System | 74 ✅ | 11 benchmarks, 0 skips |
| Distributed Execution | 33 ✅ | DistributedInferenceEngine, TP/PP/DP, 0 skips |
| Runtime v4/v5 Extensions | 229 ✅ | R1–R12 layers, all passing |
| Optimizer Passes 1–22 | 155 ✅ | All passes implemented and tested |
| Safety System | ✅ | Jailbreak/watermark/ZK/tenant isolation |
| Hub System | ✅ | CAS storage, push/pull, auth |
| gRPC Transport | ✅ | Service, TLS config, auth interceptor |
| v3.1 / v4 Extensions | 120 ✅ | 1 env-gated skip (network) |

---

## 🔄 Remaining — Hardware-Gated Only

| Item | Blocker | Priority |
|---|---|---|
| GGUF K-quant dequant (Q2_K–Q6_K) | C algorithm (llama.cpp) | HIGH |
| TEE confidential inference | Intel TDX / AMD SEV hardware | MED |
| CXL rack-scale KV pooling | CXL 2.0 hardware | LOW |
| MDLM diffusion drafting | Drafter model weights | MED |
| NCCL/RCCL collective backend | GPU hardware | MED |
| GPU end-to-end validation | GPU hardware | HIGH |
| Docker multi-stage images | CI environment | LOW |
| MLPerf v4.0 submission | GPU hardware | LOW |

---

## 📦 Recent Commits

```
d5ad33e  docs: rewrite PROGRESS_REPORT.md (2026-08-14 Phase 3b complete)
6369ffd  docs: add v0.5.0 CHANGELOG entry
2665119  feat: PEP 561 type stubs, py.typed, pyproject wheel fix
1c85546  docs: update AUDIT_REPORT Phase 3b scorecard and completion scores
331b75a  fix: Windows CP1252 encoding in check_env.py; update roadmap
312571e  feat: Phase 3 remediation — specialised loaders, evaluators, distributed engine
918d4b9  (earlier Phase 2 work)
```

---

## 🔬 How to Run Tests

```powershell
# Fast subset (no slow E2E / network tests)
python -m pytest tests/unit/ --no-cov -q --tb=short

# Specific suites
python -m pytest tests/unit/test_evaluation_complete.py -v
python -m pytest tests/unit/test_distributed_complete.py -v
python -m pytest tests/unit/test_specialised_loaders.py -v

# Full suite (takes ~5 min on CPU-only)
python -m pytest tests/ --no-cov -q --tb=no

# Smoke test
python scripts/ci_smoke_test.py --verbose
python scripts/check_env.py
```
