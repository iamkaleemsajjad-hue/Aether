# Aether Runtime — Agent Progress Tracker

> **For the next agent**: Read this file first. It tells you exactly where we are,
> what was done, and what remains. Every section is kept current after each work session.

---

## Project Overview

**Aether Runtime** is a production-grade compiler + runtime for LLM inference.
It has two phases of requirements:

| Document | Status | Description |
|---|---|---|
| `PRD.md` | ✅ **Fully implemented** | Original 9-pass compiler + 9-layer runtime |
| `PRD_v2.md` | 🔄 **~85% implemented** | 13 new passes (10–22) + 12 new runtime layers (R1–R12) |

---

## What Has Been Built (PRD v3.1 — original)

All 9 original compiler passes and all original runtime layers are implemented
under `src/aether/compiler/stage2_optimizer/` and `src/aether/runtime/`.

---

## What Has Been Built (PRD v4.0 + v5.0 — current session)

### Compiler Passes (all in `src/aether/compiler/stage2_optimizer/`)

| Pass | File | Status | Description |
|---|---|---|---|
| 10 | `pass10_mtp_head.py` | ✅ Done | Multi-Token Prediction head detection + compilation |
| 11 | `pass11_grammar_constraint.py` | ✅ Done | FSA grammar pre-compilation (Thompson NFA → DFA) |
| 12 | `pass12_model_merging.py` | ✅ Done | Task Arithmetic / DARE / TIES / FREE merging |
| 13 | `pass13_ttt_fast_weight.py` | ✅ Done | TTT fast-weight slot injection |
| 14 | `pass14_semantic_kv_compression.py` | ✅ Done | ChunkKV + SentenceKV + PyramidKV per-layer compression |
| 15 | `pass15_cross_layer_kv.py` | ✅ Done | Middle-outward xKV sharing with exponential similarity model |
| 16 | `pass16_green_energy.py` | ✅ Done | DVFS breakpoints + carbon profile (MELODI 2026) |
| 17 | `pass17_tee_wrapping.py` | ✅ Done | TEE kernel wrapping (NVIDIA CC / Intel TDX / AMD SEV-SNP) |
| 18 | `pass18_mdlm_drafter.py` | ✅ Done | MDLM diffusion drafter (cosine schedule, DiffuSpec) |
| 19 | `pass19_sub2bit_quant.py` | ✅ Done | BitNet b1.58 / BTC-LLM / NanoQuant quantization |
| 20 | `pass20_video_compression.py` | ✅ Done | STC / STORM / StreamingTOM VLM video token compression |
| 21 | `pass21_advanced_peft.py` | ✅ Done | LoRA+ / LoRAMoE / MoLF / LoRAFusion adapter compilation |
| 22 | `pass22_rlvr_verifier.py` | ✅ Done | RLVR verifier head + GRPO K2V opcodes |

### Optimizer Pipeline Registration

- `optimizer.py` updated to register all 22 passes in order
- All new pass flags wired through `config.py`

### Runtime Layers (all in `src/aether/runtime/`)

| Layer | File | Status | Description |
|---|---|---|---|
| R1 | `r1_peagle_engine.py` | ✅ Done | P-EAGLE SM-partitioned speculative decoding |
| R2 | `r2_multi_agent_kv.py` | ✅ Done | Multi-agent KV coordinator (CoW, RadixAttention-style) |
| R3 | `r3_grammar_fsm.py` | ✅ Done | Grammar FSM runtime enforcement |
| R4 | `r4_slo_scheduler.py` | ✅ Done | SLO-aware MLFQ scheduler (Sarathi-Serve chunked prefill) |
| R5 | `r5_ttt_engine.py` | ✅ Done | TTT fast-weight online adaptation |
| R6 | `r6_mcp_integration.py` | ✅ Done | MCP native integration (JSON-RPC 2.0, tool registry) |
| R7 | `r7_green_power_manager.py` | ✅ Done | DVFS enforcement + carbon routing + TDP throttling |
| R8 | `r8_tee_manager.py` | ✅ Done | TEE enclave lifecycle + weight attestation |
| R9 | `r9_sub2bit_kv_cache.py` | ✅ Done | Ternary KV + decompressed weight LRU cache |
| R10 | `r10_video_kv_manager.py` | ✅ Done | Video frame KV manager (StreamingTOM eviction) |
| R11 | `r11_semantic_kv_cache.py` | ✅ Done | HNSW semantic cross-request KV deduplication |
| R12 | `r12_rlvr_harness.py` | ✅ Done | RLVR GRPO training harness |

### Test Files

| File | Status | Coverage |
|---|---|---|
| `tests/test_passes_v2.py` | ✅ Done | Passes 10–22 + pipeline integration |
| `tests/test_runtime_v2.py` | ✅ Done | R1–R12 full algorithm correctness |

---

## Core Infrastructure Changes

### `src/aether/core/types.py`
- Extended `HardwareTarget` enum: 13 new targets (Rubin R100, Blackwell Ultra GB300, AMD MI450X, AWS Trainium3, etc.)
- Extended `DType` enum: TERNARY_158, BINARY, UINT2, FP4, FP6, FLOAT8_E4M3, FLOAT8_E5M2

### `src/aether/core/constants.py`
- New target descriptions, backend maps, precision bits/byte sizes
- Default pass flags 10–22

### `src/aether/compiler/config.py`
- 35+ new fields for passes 10–22 (all typed, with defaults)

---

## What Remains

### Phase E — Documentation + CI/CD (CURRENT)
- [ ] Update `CHANGELOG.md` with v4.0 + v5.0 entries
- [ ] Update `CONTRIBUTING.md` with new pass authoring guide
- [ ] Update `.github/workflows/` CI with new test discovery
- [ ] Update `docs/` with runtime layer API docs
- [ ] Export all new runtime classes in `src/aether/runtime/__init__.py`
- [ ] Export all new pass classes in `src/aether/compiler/stage2_optimizer/__init__.py`

### Phase F — Hardware Targets (OPTIONAL)
- CUDA SM120 / Rubin R100 kernel stubs
- AMD MI450X ROCm backend stubs

---

## AEG Artifact Directory Layout

After compilation with all passes enabled, `.aeg/` contains:

```
.aeg/
├── graph/
│   ├── kv_compression_plan.json      # Pass 14
│   ├── cross_layer_kv_plan.json      # Pass 15
│   └── video_compression_plan.json   # Pass 20
├── speculation/
│   └── mtp_config.json               # Pass 10
├── grammar/
│   ├── fsm.bin                        # Pass 11
│   └── fsm_config.json
├── diffusion/
│   ├── drafter_config.json           # Pass 18
│   └── schedule.json
├── quantization/
│   └── sub2bit_manifest.json         # Pass 19
├── adapters/
│   ├── adapter_manifest.json         # Pass 21
│   └── {name}/lora_A.bin, lora_B.bin
├── metadata/
│   └── green_profile.json            # Pass 16
├── security/
│   ├── tee_config.json               # Pass 17
│   └── weight_hash_manifest.json
└── training/
    └── rlvr_config.json              # Pass 22
```

---

## Known Architecture Decisions

1. **All passes are pure-Python reference implementations** that emit JSON/binary
   artifacts into `.aeg/` and annotate the graph via `metadata` dict when
   graph doesn't have specific methods. This makes them testable without a full GPU.

2. **Runtime layers are hardware-agnostic** by default. They load config from
   AEG artifacts and degrade gracefully when optional libraries (hnswlib, sympy,
   safetensors, hnswlib) are not installed.

3. **Config fields use safe defaults**: opt-in passes (`enable_tee=False`,
   `enable_sub2bit=False`, etc.) default to disabled so existing pipelines
   are not broken.

4. **Test files** use `MagicMock` for graph objects. All algorithm tests
   use pure-Python inputs (no GPU required).

---

## Research Papers Referenced

The implementation is grounded in 200+ papers. Key ones per component:

| Component | Papers |
|---|---|
| MTP | FastMTP, L-MTP 2026; original DeepSeek V3 MTP |
| Grammar | XGrammar MLC 2026; LLGuidance MSR 2026; CRANE ICML 2026 |
| Merging | Task Arithmetic 2023; DARE 2024; TIES 2024; FREE 2025 |
| TTT | In-Place TTT 2026; VDS-TTT 2026; Sun et al. 2024 |
| Semantic KV | ChunkKV 2026; SentenceKV EMNLP 2025; PyramidKV 2024 |
| Cross-Layer KV | xKV 2026; CommonKV 2026; Wu/Tu 2025 |
| Green | MELODI 2026; CodeCarbon 2026; DVFS arXiv 2025 |
| TEE | NVIDIA CC 2025; Intel TDX Rev 1.5; AMD SEV-SNP Rev 1.58; Guardian OSDI 2026 |
| MDLM | Sahoo ICML 2025; DiffuSpec ACL 2026; SpecDiff ACL 2026 |
| Sub-2-bit | BitNet b1.58 2024; BTC-LLM 2026; NanoQuant 2026 |
| Video | STC CVPR 2026; STORM 2026; StreamingTOM 2026 |
| PEFT | LoRA 2022; LoRA+ 2024; LoRAMoE 2024; LoRAFusion 2026 |
| RLVR | GRPO DeepSeek-R1 2025; K2V 2026; OpenRLHF 2025 |
| P-EAGLE | EAGLE-3 2025; HPSD 2026; Leviathan 2023 |
| Semantic Cache | SemantiCache 2026; HNSW 2018; GPTCache 2023 |

---

## Quick Start for Next Agent

```bash
# Run all new tests
cd "c:\Users\pc\Desktop\Aether Runtime"
python -m pytest tests/test_passes_v2.py tests/test_runtime_v2.py -v

# Run full test suite
python -m pytest tests/ -v --tb=short

# Check imports
python -c "from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline; p = OptimizerPipeline(); print(p)"
```

---

*Last updated: 2026-08-04 by Aether Agent*
*Session: PRD v4.0 + v5.0 full implementation — passes 10–22 + runtime R1–R12*
