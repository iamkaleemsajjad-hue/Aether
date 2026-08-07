# Aether Runtime — Agent Progress Tracker

> **For the next agent**: Read this file first. It tells you exactly where we are,
> what was done, and what remains. Every section is kept current after each work session.

---

## Project Overview

**Aether Runtime** is a production-grade compiler + runtime for LLM inference.

| Document | Status | Description |
|---|---|---|
| `PRD.md` | ✅ **Fully implemented** | Original 9-pass compiler + 9-layer runtime |
| `PRD_v2.md` | ✅ **~95% implemented** | 13 new passes (10–22) + 12 new runtime layers (R1–R12) + hardware targets |

---

## What Has Been Built (PRD v4.0 + v5.0)

### Compiler Passes (`src/aether/compiler/stage2_optimizer/`)

| Pass | File | Status | Description |
|---|---|---|---|
| 10 | `pass10_mtp_head.py` | ✅ Done | Multi-Token Prediction head detection + compilation |
| 11 | `pass11_grammar_constraint.py` | ✅ Done | FSA grammar pre-compilation (Thompson NFA → DFA) |
| 12 | `pass12_model_merging.py` | ✅ Done | Task Arithmetic / DARE / TIES / FREE merging |
| 13 | `pass13_ttt_fast_weight.py` | ✅ Done | TTT fast-weight slot injection |
| 14 | `pass14_semantic_kv_compression.py` | ✅ Done | ChunkKV + SentenceKV + PyramidKV |
| 15 | `pass15_cross_layer_kv.py` | ✅ Done | Middle-outward xKV sharing |
| 16 | `pass16_green_energy.py` | ✅ Done | DVFS breakpoints + carbon profile |
| 17 | `pass17_tee_wrapping.py` | ✅ Done | TEE kernel wrapping (NVIDIA CC / TDX / SEV-SNP) |
| 18 | `pass18_mdlm_drafter.py` | ✅ Done | MDLM diffusion drafter |
| 19 | `pass19_sub2bit_quant.py` | ✅ Done | BitNet b1.58 / BTC-LLM / NanoQuant |
| 20 | `pass20_video_compression.py` | ✅ Done | STC / STORM / StreamingTOM VLM video compression |
| 21 | `pass21_advanced_peft.py` | ✅ Done | LoRA+ / LoRAMoE / MoLF / LoRAFusion |
| 22 | `pass22_rlvr_verifier.py` | ✅ Done | RLVR verifier head + GRPO K2V opcodes |

### Runtime Layers (`src/aether/runtime/`)

| Layer | File | Status | Description |
|---|---|---|---|
| R1 | `r1_peagle_engine.py` | ✅ Done | P-EAGLE SM-partitioned speculative decoding |
| R2 | `r2_multi_agent_kv.py` | ✅ Done | Multi-agent KV coordinator |
| R3 | `r3_grammar_fsm.py` | ✅ Done | Grammar FSM runtime enforcement |
| R4 | `r4_slo_scheduler.py` | ✅ Done | SLO-aware MLFQ scheduler |
| R5 | `r5_ttt_engine.py` | ✅ Done | TTT fast-weight online adaptation |
| R6 | `r6_mcp_integration.py` | ✅ Done | MCP native integration (JSON-RPC 2.0) |
| R7 | `r7_green_power_manager.py` | ✅ Done | DVFS + carbon routing + TDP throttling |
| R8 | `r8_tee_manager.py` | ✅ Done | TEE enclave lifecycle + weight attestation |
| R9 | `r9_sub2bit_kv_cache.py` | ✅ Done | Ternary KV + decompressed weight LRU cache |
| R10 | `r10_video_kv_manager.py` | ✅ Done | Video frame KV manager |
| R11 | `r11_semantic_kv_cache.py` | ✅ Done | HNSW semantic cross-request KV dedup |
| R12 | `r12_rlvr_harness.py` | ✅ Done | RLVR GRPO training harness |

### Hardware Targets (`src/aether/compiler/stage3_targeting/hardware_profile.py`)

28 total profiles in `_TARGET_PROFILES`. New fields:
- `flops_fp4`, `supports_fp4` — FP4 tensor core (B200, Rubin R100+)
- `supports_ternary`, `supports_mxfp6`, `supports_tee`, `tee_backend`
- `nvlink_bandwidth_gb_s`, `tdp_watts`
- `is_riscv_npu`, `abstract_ir_family`

**v4.0 NEW targets (9):** cuda_sm130, cuda_sm100_tee, riscv_mips_s8200,
riscv_sifive_x160, riscv_xuantie_c930, fpga_xilinx_vu9p, amd_mi350x,
qualcomm_cloud_ai100

**v5.0 NEW targets (6):** cuda_sm100_gb300, rocm_cdna5_mi455x,
cpu_avx512_ternary, cpu_neon_ternary, fpga_ternary, riscv_cervell

### RISC-V NPU Abstract IR (PRD §3.2)

- `riscv_npu_ir.py` — core `RISCVNPUIRBuilder` + `RISCV_NPU_BACKEND_REGISTRY`
- `target_riscv_mips.py` — MIPS S8200 (RV32IM + MIPS.NPU, 64 TOPS)
- `target_riscv_sifive.py` — SiFive X160 (RVV-1.0 + RMMM-0.7, 128 TOPS)
- `target_riscv_xuantie.py` — XuanTie C930 (RVV-1.0 + XPU, 256 TOPS)
- `target_riscv_cervell.py` — Semidynamics Cervell (Quadric qdIR, 512 TOPS)

Tiling: `3 * T^2 * dtype_bytes <= scratchpad_bytes`; T always power-of-2.

### AEG Format 2.0 (`src/aether/compiler/aeg_format_v2.py`)

`AEGPackageV2` creates full PRD §5 directory tree. Key new dirs:
- `speculation/` — R1 P-EAGLE + Saguaro configs
- `structured_output/` — R3 grammar FSM binaries
- `merging/` — Pass 12 task vector artifacts
- `ttt/` — Pass 13 / R5 TTT config
- `green/` — Pass 16 / R7 energy profile + DVFS hints
- `tee/` — Pass 17 / R8 enclave config + attestation policy
- `multi_agent/` — R2 KV coordination + DroidSpeak
- `mcp/` — R6 server registry + tool schemas
- `kernels/` with all 25 v4.0+v5.0 target subdirectories

Exports: `AEGManifest`, `SpeculationConfig`, `GrammarManifest`,
`GreenEnergyProfile`, `TEEConfig`, `MultiAgentConfig`, `MCPConfig`

### Server Routes v4.0 (`src/aether/server/routes.py`)

New endpoints:
- `POST /v1/tools/call` — MCP tool call (R6)
- `POST /v1/grammar/compile` — Pre-compile grammar FSM (Pass 11)
- `GET  /v1/grammar/list` — List compiled grammars
- `POST /v1/models/{name}/merge` — Task vector merge (Pass 12)
- `POST /v1/models/{name}/ttt` — TTT fast-weight update (Pass 13)
- `GET  /v1/targets` — All 28 hardware targets with v4.0 fields
- `GET  /v1/targets/{target_id}` — Single target profile
- `GET  /v1/green/status` — Carbon/DVFS status (R7)
- `POST /v1/tee/session` — Start TEE session (R8)
- `DELETE /v1/tee/session/{id}` — Close TEE session

### Tests

| File | Status | Tests |
|---|---|---|
| `tests/test_passes_v2.py` | ✅ Done | Passes 10–22 + pipeline integration |
| `tests/test_runtime_v2.py` | ✅ Done | R1–R12 full algorithm correctness |
| `tests/unit/test_aeg_format_v2.py` | ✅ Done | 64 tests, 100% pass, 93% cov |
| `tests/unit/test_riscv_and_hardware.py` | ✅ Done | RISC-V backends + HardwareProfile v4.0 fields |
| `tests/unit/test_optimizer_passes.py` | ✅ Fixed | Updated for 22-pass pipeline |

---

## What Remains

### Phase E — Documentation + CI/CD
- [ ] Update `CHANGELOG.md` with v4.0 + v5.0 entries
- [ ] Update `CONTRIBUTING.md` with new pass authoring guide
- [ ] Update `.github/workflows/` CI with new test file discovery
- [ ] Export all new runtime classes in `src/aether/runtime/__init__.py`
- [ ] Export new pass classes in `src/aether/compiler/stage2_optimizer/__init__.py`

### Phase F — Runtime Integration
- [ ] `src/aether/compiler/compiler.py` — wire `AEGPackageV2.create()` into main compile flow
- [ ] `src/aether/compiler/compiler.py` — add `compile_async()` + `get_compile_status()`
- [ ] `src/aether/runtime/runtime.py` — add `grammar_engine`, `ttt_engine`, `tee_manager`, `green_power_manager`, `mcp_layer` attributes

---

## Known Gotchas

1. `hardware_profile.py` had a duplicate class definition (appended mid-file).
   Fixed by truncating at line 925. File is now clean (925 lines, 28 profiles).

2. PowerShell `&&` not supported — use `;` to chain commands.

3. All `from __future__ import annotations` must be the FIRST statement in a file.

---

## Quick Start for Next Agent

```bash
cd "c:\Users\pc\Desktop\Aether Runtime"

# Verify hardware profiles
python -c "from aether.compiler.stage3_targeting.hardware_profile import _TARGET_PROFILES; print(len(_TARGET_PROFILES), 'profiles')"

# Verify AEG Format 2.0
python -c "from aether.compiler.aeg_format_v2 import AEGPackageV2; print('AEG v2.0 OK')"

# Full test suite
python -m pytest tests/ -v --tb=short -q
```

---

*Last updated: 2026-08-07 by Aether Agent*
*Session: Hardware targets (28), RISC-V NPU IR, AEG Format 2.0, server routes v4.0, 317+ tests passing*
