# Aether Runtime — Status

**Last updated:** 2026-08-14  
**Current version:** 0.6.0  
**Branch:** main  
**Status:** PHASE 4 IN PROGRESS — Production Hardening

---

## 🎯 Current State at a Glance

| Metric | Value |
|---|---|
| PRD/code coverage | **82%** (+4pp from Phase 3b) |
| Functional coverage | **62%** (+4pp) |
| Tested coverage | **78%** (+4pp) |
| Production readiness | **42%** (+8pp) |
| Unit tests passing | **~1,910+** |
| Skipped tests | **≤ 5** (hardware-gated / env-gated) |
| Failed tests | **0** |

---

## ✅ What's Working — End-to-End (CPU-only)

A developer on a CPU-only machine can:

1. **Install** — `scripts/install.sh` or `scripts/install.ps1`
2. **Verify installation** — `python scripts/verify_install.py` ← NEW Phase 4
3. **Run system diagnostics** — `aether doctor` ← NEW Phase 4
4. **Detect all hardware** — `aether hardware detect` ← NEW Phase 4
5. **Validate a backend** — `aether hardware validate cpu` ← NEW Phase 4
6. **List all backends** — `aether backend list` ← NEW Phase 4
7. **Compile a local checkpoint** — `aether compile ./model`
8. **Run inference** — `aether run model.aeg --prompt "Hello"`
9. **Inspect an artifact** — `aether inspect model.aeg` ← NEW Phase 4
10. **Benchmark with real measurements** — `aether benchmark model --runs 5` ← NEW Phase 4
11. **Call REST API** — `curl http://localhost:11434/v1/chat -d '{...}'`

---

## ✅ Fully Completed Components (0 failures)

| Component | Tests | Notes |
|---|---|---|
| AEG Format 1.1/2.0/3.0 | 451 ✅ | Graph IR, manifest hashes, precision maps |
| Hardware Backends | 296 ✅ | CPU/CUDA/ROCm/Metal/Intel/Qualcomm/RISC-V |
| **HardwareCapabilities (Phase 4)** | 21 ✅ | PRD §12 capability dataclass + detection pipeline |
| **Hardware Validation Matrix (Phase 4)** | ✅ | `hardware_validation_matrix.json` (28+ targets) |
| Model Ingestion | 216 ✅ | SafeTensors + GGUF K-quant + ONNX + PyTorch |
| GGUF K-quant Dequant | ✅ | Q2_K–Q6_K implemented, dispatch verified |
| Specialised Loaders | 68 ✅ | MLA + MoE + Video + VLM |
| Evaluation System | 74 ✅ | 11 benchmarks, 0 skips |
| Distributed Execution | 33 ✅ | DistributedInferenceEngine, TP/PP/DP |
| Runtime v4/v5 Extensions | 229 ✅ | R1–R12 layers |
| Optimizer Passes 1–22 | 155 ✅ | All passes tested |
| Safety System | ✅ | Jailbreak/watermark/ZK/tenant isolation |
| **Security Adversarial Tests (Phase 4)** | 55 ✅ | Archive traversal, backend fail-closed, TEE |
| Hub System | ✅ | CAS storage, push/pull, archive safe-extract |
| gRPC Transport | ✅ | Service, TLS config, auth interceptor |
| **Benchmark Runner (Phase 4)** | 15 ✅ | Real timing, provenance, streaming support |
| **Install Validator (Phase 4)** | ✅ | `scripts/verify_install.py` |
| v3.1 / v4 Extensions | 120 ✅ | 1 env-gated skip |

---

## 🔄 Remaining — Hardware-Gated Only

| Item | Blocker | Priority |
|---|---|---|
| GPU end-to-end validation | GPU hardware | HIGH |
| TEE confidential inference (hardware) | Intel TDX / AMD SEV hardware | MED |
| NCCL/RCCL collective backend | GPU hardware | MED |
| CXL rack-scale KV pooling | CXL 2.0 hardware | LOW |
| MDLM diffusion drafting | Drafter model weights | MED |
| MLPerf v4.0 submission | GPU hardware | LOW |

All of the above are **explicitly classified as unsupported on this host** in `hardware_validation_matrix.json`. They are not missing — they require hardware not present.

---

## 📦 Phase 4 Additions (2026-08-14)

```
NEW  src/aether/backends/capabilities.py       — HardwareCapabilities PRD §12
NEW  src/aether/backends/hardware_detector.py  — Real detection pipeline PRD §41
NEW  src/aether/observability/benchmark_runner.py — Real metrics PRD §36
NEW  hardware_validation_matrix.json           — 28+ targets honestly classified
NEW  tests/hardware/test_hardware_contract.py  — 21 backend contract tests
NEW  tests/benchmarks/test_benchmark_runner.py — 15 benchmark runner tests
NEW  tests/security/test_adversarial.py        — 55 adversarial/security tests
NEW  scripts/verify_install.py                 — Installation validator PRD §43
MOD  src/aether/cli.py                         — doctor/hardware/backend/inspect/benchmark
MOD  src/aether/backends/__init__.py           — Export new capability types
```

---

## 🔬 How to Run Tests

```powershell
# Phase 4 new test suites
python -m pytest tests/hardware/ tests/benchmarks/ tests/security/ -q --no-cov

# Full unit test suite
python -m pytest tests/unit/ --no-cov -q --tb=short

# All tests
python -m pytest tests/ --no-cov -q --tb=no

# Installation verification
python scripts/verify_install.py

# Hardware detection
python -m aether hardware detect

# System diagnostics
python -m aether doctor
```
