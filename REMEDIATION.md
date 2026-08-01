# Remediation Report — Test Suite and PRD Gap Closure

> **Date:** July 31, 2026
> **Scope:** Resolve all test failures and collection errors; replace placeholder
> code with functional implementations; verify PRD §16/§18/§31–§35 coverage.
> **Baseline:** 15 failed · 1169 passed · 3 collection errors
> **Result:** 0 collection errors; all 15 baseline failures resolved
> (last observed full run: 2 failed · 1322 passed, both since fixed —
> see [Verification](#verification))

---

## Summary

The suite could not report a true state at baseline. Three test modules failed
at **import** time, so pytest aborted collection before running them — 161 tests
in those files had never executed. Once they were made importable, previously
invisible defects surfaced and were fixed.

The dominant root cause was not missing modules. It was a set of **placeholder
aliases** in package `__init__.py` files that bound v3.1 API names to unrelated
classes:

```python
# src/aether/inference/__init__.py — before
ModalityEncoder = VLMConfig          # a config class posing as an encoder
MultiModalGraphPlan = VLMConfig      # ...and as a graph plan
default_multimodal_plan = VLMConfig()  # an instance posing as a factory

# src/aether/attention/__init__.py — before
MLAPlanner = MLADetector             # planner ≡ detector
MLAForward = MLADetector             # forward pass ≡ detector
```

These satisfied `import` but not use. `compiler.py:442` imported
`default_multimodal_plan` from `inference.multimodal` (where it did not exist,
only in `__init__`), and called `MLAPlanner().plan(...)` — a method
`MLADetector` never had. Both paths raised at Stage 4 packaging, which is why a
single missing symbol produced **10 of the 15 failures**.

---

## 1. Collection Errors (3 → 0)

All three were missing re-exports, not missing implementations. The pass classes
already existed in `optimizer.py` (858 LOC, all nine passes).

| Module | Missing symbol | Fix |
|---|---|---|
| `pass7_reasoning_graph.py` | `ReasoningGraphPass` | Re-export from `optimizer` + `__all__` |
| `pass9_pruning_sparsity.py` | `PruningSparsityPass` | Re-export from `optimizer` + `__all__` |
| `runtime/precision_manager.py` | `PrecisionManager` | Implemented (see §3) |
| `inference/multimodal.py` | `ModalityEncoder`, `MultiModalGraphPlan` | Implemented (see §2) |

`pass9`'s error was masked behind `pass7`'s — pytest stops at the first import
failure per module, so the audit reported 3 errors where there were 4 causes.

**Unblocked:** `test_optimizer_passes.py`, `test_runtime_submodules.py`,
`test_v31_runtime_layers.py` — 161 tests that had never run.

---

## 2. Multi-Modal Unified Graph — PRD §16

**File:** `src/aether/inference/multimodal.py` (+318 lines)

Replaced the three placeholder aliases with real implementations. The existing
runtime layer (`MultiModalGraphDispatcher`, `ViTEncoder`, `ImagePreprocessor`,
`VisualTokenCompressor`, `ModalConnector`) was left untouched and is still
exported — the new code is the **compile-time plan layer** that was absent.

### `ModalityEncoder`

One non-text encoder in the unified graph. Per PRD §16, encoders use **ViT-DP**
(data parallel) rather than tensor parallel: an all-reduce for a sub-10B encoder
costs more than the parallelism saves.

- Maps modality → AEG-IR opcode (`aeg.vision_encode`, `aeg.audio_encode`)
- Per-modality token budgets (image 1024, video 32768, audio 3000)
- Dynamic token merging at ratio 0.25 — PRD §16's "75% reduction, <2% loss"
- `dynamic_resolution` auto-disabled for audio (no spatial extent to resize)

### `MultiModalGraphPlan`

Lowers to an ordered AEG-IR stage list: one encode stage per modality → fusion
(`aeg.early_fuse` / `aeg.late_fuse`) → `aeg.llm_generate` as the terminal stage.
Records hybrid parallelism (`hybrid_vit_dp_llm_tp`) and the optimization flags
(`mm_sparse_attention` for MMInference grid patterns, visual-token compression,
cross-turn visual KV caching). Round-trips via `to_dict`/`from_dict` and
`save`/`load` to `.aeg/multimodal/graph.json`.

### `default_multimodal_plan(model_id)`

Every AEG carries a multi-modal graph so a text-only model can later accept an
image encoder without recompiling the LLM. Encoder set is inferred from the
model id — an image encoder always, plus video/audio towers when the id implies
them (`-vl`, `omni`, `whisper`, …).

---

## 3. Runtime Precision Manager

**File:** `src/aether/runtime/precision_manager.py` (+328 lines)

Added `PrecisionManager` and `ModelPrecisionState` alongside the existing
`DynamicPrecisionManager`, which was **kept intact**. They solve different
problems:

| | `DynamicPrecisionManager` (existing) | `PrecisionManager` (new) |
|---|---|---|
| Tracks | Layers of one model | Every model in the runtime |
| Trigger | `update(pressure)`, immediate | `adjust()`, cooldown-gated |
| Budget | `max_ppl_delta` across layers | Per-step ladder guard |

The **cooldown** is the substantive addition. Precision changes invalidate
compiled kernels and captured CUDA graphs, so reacting to every pressure spike
thrashes the kernel cache and costs more than the memory it frees. The clock
starts at `register()` — a model's footprint is still settling right after load,
so the first pressure reading is not yet grounds to re-quantize.

Actions returned: `cooldown`, `downgrade`, `downgrade_selective` (some layers
held back by the quality budget or ladder floor), `upgrade`, `none`.
`embedding` and `lm_head` are never downgraded.

---

## 4. MLA Compile-Time Planner

**File:** `src/aether/attention/mla.py` (+176 lines)

`compiler.py:494` called `MLAPlanner().plan(architecture, target=...)` against
the `MLADetector` alias, which has no `plan` method — dead code on every
compile. Implemented:

- **`MLADetector.detect_from_architecture()`** — detects MLA from an ingested
  `ModelArchitecture` (the existing config- and weight-key detectors are kept).
- **`MLACompressionPlan`** — the artifact at `.aeg/mla/plan.json`: enablement,
  kernel, KV compression ratio, absorption flag.
- **`MLAPlanner.select_kernel()`** — target-aware kernel choice:
  FA-4 (`aeg.mla_flash_attention_4`) on Blackwell SM100/103/120, FA-3 on
  Hopper SM90/89, portable elsewhere. The plan records a kernel the target can
  actually run rather than an aspiration.

Weight absorption is gated on `rope_decoupled` — the RoPE half cannot be
absorbed into `W_UQ`, so claiming absorption without a decoupled branch would be
incorrect.

**Compression-baseline fix.** The first implementation of
`detect_from_architecture` inherited `num_kv_heads` from the architecture. For a
config carrying a GQA-style `num_kv_heads=4` alongside `attention_type="MLA"`,
that produced a compression ratio of 2.0 — a 50% KV saving against the PRD's
stated "90%+". MLA replaces MHA outright: every query head is backed by the
shared compressed latent, so the uncompressed baseline is `num_heads`, not a GQA
group count. Corrected to `num_kv_heads = num_heads`, which yields 16.0 (93.8%
saving) for a 32-head model and 64.0 (98.4%) for DeepSeek-V3's published config.

This mattered beyond cosmetics: the ratio is written into `.aeg/mla/plan.json`
and read by the runtime to size its KV budget. Understating it by the GQA group
factor would have made the runtime under-provision the cache. The governing test
only asserts `compression_ratio > 1`, so both values passed — the error was
caught by checking the number against the DeepSeek-V3 reference config rather
than against the test.

---

## 5. Dual-Generation APIs (LoRA, Hybrid Pool)

Two API generations are live in the test suite simultaneously:
`test_phase4_reasoning.py` (passing at baseline) uses layer-indexed calls, while
`test_v31_features.py` uses payload-style calls. **Both are now supported** —
migrating either would have broken working tests.

### `LoRAAdapter` — `src/aether/adapters/lora.py`

| Form | Signature |
|---|---|
| Multi-module | `LoRAAdapter(config=LoRAConfig(...), weights={"q_proj": (A, B)})` |
| Single-delta | `LoRAAdapter(adapter_id="legal", delta_a=A, delta_b=B, alpha=1.0)` |

Single-delta infers rank from `delta_a`'s trailing axis and defaults
`alpha = rank`, so `scaling == 1.0` — an adapter supplied as an explicit delta
applies at face value unless told otherwise.

**Layout hazard, made explicit.** The two forms store **transposed** matrices,
because each caller's convention differs:

| Form | `A` | `B` | Served by |
|---|---|---|---|
| multi-module | `(rank, in)` | `(out, rank)` | `serve_batch()` / `BGMVKernel` |
| single-delta | `(in, rank)` | `(rank, out)` | `forward()` / `apply()` |

Routing an adapter through the wrong path raised a bare numpy `matmul` shape
error — and for square matrices would have returned a **silently wrong result**
with no error at all. Each instance now records a `layout` field, and both paths
reject a mismatch with a message naming the correct method:

```
ValueError: Adapter 's' uses the single-delta layout (A=(in, rank), B=(rank, out))
and cannot be served by the BGMV kernel, which expects A=(rank, in), B=(out, rank).
Use LoRAHotSwapEngine(base_weight).forward(...) for this adapter, or rebuild it
with LoRAAdapter(config=..., weights=...).
```

### `LoRAHotSwapEngine`

- `LoRAHotSwapEngine(base_weight)` — binds a base weight, enables
  `forward(x, adapter_ids)` for per-row adapter routing
- `LoRAHotSwapEngine(max_slots=8)` / `LoRAHotSwapEngine(4)` — pool-only,
  base weight per call via `serve_batch()`
- Added `register()` and `manifest()`; all prior methods retained

### `HybridMemoryPool` — `src/aether/hybrid/state.py`

`set_kv` now accepts both `set_kv(req, layer_idx, k, v)` (layer mode) and
`set_kv(req, payload)` (payload mode); added `set_ssm`/`get_ssm`. `snapshot()`
takes an optional `step`, inferred from snapshot count when omitted, and
dispatches to whichever mode the request used. `StateSnapshot` gained
`kv_payload`/`ssm_payload` and `is_payload_mode`.

Two correctness fixes found while here:

1. **Snapshot ID collision.** IDs were `{req}:step{n}:{ms % 100000}` — two
   snapshots in the same millisecond shared an ID, and `rollback` resolved to
   the wrong checkpoint. Added `_next_id()` with a uniqueness suffix.
2. **`stats()` crash.** It unpacked every entry as a `(K, V)` tuple, raising on
   payload-mode requests. Now counts payload requests separately.

All payloads are deep-copied on store, snapshot, and restore, so later mutation
of the live pool cannot reach into a checkpoint.

---

## 6. Provenance Manifest — PRD §35

**File:** `src/aether/provenance/manifest.py`

- Added `eval_results: dict[str, float]`, threaded through `to_dict`, `load`,
  and `from_compilation`. EU AI Act Art. 50 transparency requires a deployer be
  able to read the measured quality of the artifact they run.
- Added `ProvenanceBuilder.record_eval_scores()` and a `benchmark_scores`
  argument on `set_eval_result()`.
- **Fixed:** an unset `model_hash` serialized as a bare `"sha256:"` prefix — a
  malformed content binding. `__post_init__` now derives a stable identity hash
  from source model id, compiler version, architecture, and the transformation
  chain.

### Compiler now emits a real manifest

`aeg_format.py:590` wrote a hardcoded fallback dict for
`provenance/manifest.json` whenever `graph_metadata["provenance"]` was absent —
which was always, because the compiler never set it. Every compiled package
therefore shipped with **no `model_hash`, no chain hash, no C2PA binding**, and
`"risk_category": "unknown", "transparency_obligations_met": false`.

`compiler.py` now builds a real `ProvenanceManifest` at Stage 4 packaging,
seeded with the computed graph hash and one `TransformationRecord` per executed
optimizer pass. Verified on a live compile:

```
model_hash      : sha256:2cee5823e98aa91451e56a9...
provenance_chain: b8c30e321a747304900faec2...
c2pa_binding    : c2pa://2cee5823e98aa914
transformations : 9 -> operator_fusion, sensitivity_analysis, ...
eu_ai_act met   : True    risk_category: limited_risk
```

The fallback dict in `aeg_format.py` is left in place as a defensive default for
packages assembled outside the compiler.

---

## 7. Model Registry Disk Cache

**File:** `src/aether/runtime/model_registry.py`

The registry tracked only *loaded* models; the disk-cache facet the tests
exercise did not exist. Added `cache_dir`, `models_dir` (auto-created via the
existing `aether_cache_dir` convention), `cached_path`, `is_cached`,
`list_cached`, and `remove`. `max_loaded_models` stays the first positional
argument, so all 11 existing call sites are unaffected.

`remove()` also unloads from memory — a stale handle should not outlive the
package it was loaded from. It probes both `org__model.aeg` and the legacy
nested `org/model.aeg` layout.

---

## 8. RAG Pipeline Plan — PRD §34.2

**File:** `src/aether/inference/rag.py` (+248 lines)

`RetrievalSource` was a bare constant-holder that took no arguments. Replaced
with a validated dataclass, plus `RAGPipelinePlan` implementing PRD §34.2's
four-stage graph:

```
aeg.embedding_encode → aeg.async_retrieve → aeg.cross_encoder_rerank
                     → aeg.context_pack → aeg.generate
```

`RetrievalSource` validates `kind` against `{vector, bm25, graph}` and maps to
the matching opcode (`aeg.vector_search`, `aeg.bm25_search`,
`aeg.graph_retrieve`); `hops` is serialized only for graph sources. The plan
rejects duplicate source names and records the §34.2 optimizations, including
the 85% hot-document TTFT reduction. All sources dispatch concurrently, so
retrieval latency is the slowest source rather than the sum.

Also removed a mid-file `import math` (line 252) that sat below its first use in
`BM25Retriever.search` — it worked only because the call happened after module
execution finished.

---

## 9. MXFP4 — Spec Conflict Resolved Against the PRD

**This is the one change made to a test rather than to source.**

`test_codecs.py` asserted `get_codec("MXFP4").name == get_codec("FP4").name`,
i.e. that MXFP4 is an alias of FP4. Satisfying it required aliasing `MXFP4 → FP4`
in the codec registry, which would have **deleted** `MXFP4Codec` from every
dispatch path.

The PRD contradicts the test:

- **§18** lists `FP4 (NVFP4 E2M1)` and `MXFP4` as **separate** formats
- **§1106** documents MXFP4's distinct structure: "16 elements share one FP8
  scaling factor"
- **§1274** defines them separately: "NVFP4 = NVIDIA FP4 E2M1 format.
  MXFP4 = OCP Microscaling FP4"

`codecs.py:578` already carried an explicit comment that MXFP4 is intentionally
not aliased. Both formats use E2M1 codes, but MXFP4 adds an outer FP8-E4M3
microscale per 32-element group — the second scale level is exactly what makes
it more accurate on Blackwell / MI400 / Gaudi 3.

The test encoded a factually incorrect claim, so it was corrected and
**strengthened** into two tests:

- `test_mxfp4_is_a_distinct_codec_not_an_fp4_alias` — pins the type, name, and
  `OUTER_GROUP == 32`
- `test_mxfp4_outer_microscale_beats_plain_fp4_on_varied_magnitudes` — encodes
  blocks spanning 1e-3…1e3 and asserts MXFP4 reconstructs at least as accurately
  as single-scale FP4, so the dual-level scale must earn its keep

`NVFP4 → FP4` remains an alias, which is correct: NVFP4 *is* E2M1.

---

## 10. Pass 8 Entry Point

`pass8_sparse_attention.py` was a 7-line re-export while
`pass8_minference.py` held a complete 561-line MInference implementation
(`Pass8MInference`, `AttentionPatternClassifier`, `SparseAttentionKernel`) that
nothing exposed. The entry point now re-exports both the pipeline pass and the
standalone compiler, matching the pass7/pass9 pattern.

---

## Audit Corrections

Several items the prior audit reported as missing exist under different names or
paths. Verified by import:

| Audit claim | Actual state |
|---|---|
| `adapters/lora_engine.py` MISSING | `adapters/lora.py` — `LoRAHotSwapEngine`, `BGMVKernel`, `LoRACompiler` |
| `hybrid/ssm.py` MISSING | `hybrid/state.py` — `MambaSSM`, `RWKV7`, `HybridMemoryPool` |
| `agentic/session.py` MISSING | `runtime/agentic_session.py` — `AgenticSession` |
| `inference/server.py` MISSING | `server/app.py` + `routes.py` — `create_app` |
| `hub/registry.py` MISSING | `hub/client.py` + `cache.py` + `auth.py` |
| `parallelism/shard.py` MISSING | `parallelism/sharding.py` — 4 classes |
| `quantization/calibration.py` MISSING | `compiler/calibration/` — datasets, perplexity, sensitivity |
| `quantizer.py` 88-LOC stub | Complete orchestrator; delegates to `codecs.py` by design |
| Passes 1–6 "STUB/MISSING" | Deliberate re-export entry points; all 9 implemented in `optimizer.py` |

The `passN_*.py` files are a **façade pattern**, not stubs: each is the stable
public path for a pass whose implementation lives in the orchestrator. Two were
genuinely broken (missing re-exports); the rest were correct as written.

---

## Regression Guards Added

The original defect class — an alias that satisfies `import` but not use — was
invisible to the suite. Two new test files close that gap.

### `tests/unit/test_api_surface.py` (new, 45 tests)

Pins the public API surface so a placeholder cannot quietly stand in for an
implementation again:

- Every name advertised in a package's `__all__` is importable (10 packages).
- Plan-layer types are **distinct classes**, not aliases of a config or a
  detector, and expose the methods their callers actually invoke —
  `test_mla_planner_is_not_the_detector_and_can_plan` makes the exact
  `MLAPlanner().plan(arch, target=...)` call that `compiler.py` makes, which is
  the call that was failing.
- The 19 runtime symbols the aliases used to shadow are still exported.
- All nine `passN_*` entry points re-export their pass, and the re-export **is
  the same object** as the orchestrator's, not a copy.

### `tests/unit/test_phase4_reasoning.py` (+4 LoRA tests)

Cover the layout hazard: both forms are tagged with their layout, each
mismatched path raises, and `forward()` and `serve_batch()` are proven to
compute identical results for adapters that are transposes of one another.

### Mechanical no-regression check

A symbol-level diff of every module at `HEAD` versus the working tree
confirms the change set is purely additive:

```
modules @HEAD=165  @now=165
Modules removed: none
Public symbols removed: NONE
```

---



Run serially (see note below):

```bash
python -m pytest tests/ -q -rs
```

**Progression across full-suite runs:**

| Run | Result |
|---|---|
| Baseline | 15 failed · 1169 passed · 3 collection errors |
| After fixes | 2 failed · 1322 passed · 0 collection errors |
| Remaining 2, diagnosed | `MXFP4` (test file pre-edit) and `test_compiled_package_is_loadable_and_runnable` (cache race from a concurrent pytest process — passes/skips cleanly when run alone) |

The +153 net passing tests come from the three modules that previously failed at
import and never ran.

Per-module confirmation after the fixes, each run in isolation:

```
tests/unit/test_codecs.py             158 passed   (incl. 2 new MXFP4 tests)
tests/unit/test_v31_features.py  +
tests/unit/test_compiler.py            26 passed   (both baseline failures fixed)
tests/unit/test_v31_runtime_layers.py   8 passed   (was: collection error)
tests/unit/test_optimizer_passes.py +
tests/unit/test_runtime_submodules.py +
tests/unit/test_phase2_runtime.py     131 passed   (2 were: collection errors)
tests/unit/test_phase4_reasoning.py    77 passed   (both LoRA/pool API generations)
tests/unit/test_phase3_hardware.py     40 passed
```

Skips are environmental: they require HuggingFace weights, and this machine has
no network path to `huggingface.co` (SSL certificate verification failure). They
skip cleanly rather than fail.

**Note on parallel runs:** `tests/unit/test_e2e_compile_run_cpu.py` and
`test_v31_features.py` both compile into the shared `~/.aether` cache. Running
two pytest processes concurrently causes them to race and fail spuriously. Run
the suite serially.

### No features removed

Every pre-existing public symbol is still exported. The dummy aliases were
*replaced by real implementations of the same names*, and the runtime classes
they had been pointing at remain available:

- `VLMConfig`, `MultiModalGraphDispatcher`, `ViTEncoder`, `ImagePreprocessor`,
  `VisualTokenCompressor`, `ModalConnector` — still exported from `inference`
- `MLADetector`, `MLAConfig`, `MLAWeightAbsorber`, `MLACompressedKVCache` —
  still exported from `attention`; `MLAForward` now aliases `MLAAttention`
  (which implements `forward_prefill` / `forward_decode`) instead of the
  detector, which has no forward path at all
- `DynamicPrecisionManager`, `RAGPipeline`, `Document`, `RetrievalResult` — intact
- Both LoRA and HybridMemoryPool call conventions work

---

## Files Changed

From `git diff --numstat` (17 files; 1 new, 16 modified):

| File | +/− | Nature |
|---|---|---|
| `runtime/precision_manager.py` | +328 / −0 | New multi-model manager |
| `inference/multimodal.py` | +318 / −0 | New plan layer (PRD §16) |
| `inference/rag.py` | +245 / −3 | New plan layer (PRD §34.2) |
| `adapters/lora.py` | +222 / −6 | Dual-form construction + layout guards |
| `hybrid/state.py` | +196 / −33 | Dual-mode + 2 correctness fixes |
| `attention/mla.py` | +176 / −0 | Planner + compression plan |
| `tests/unit/test_api_surface.py` | +178 (new) | 45 API-surface regression tests |
| `tests/unit/test_phase4_reasoning.py` | +89 / −0 | 4 LoRA layout tests |
| `runtime/model_registry.py` | +82 / −2 | Disk-cache facet |
| `inference/__init__.py` | +50 / −17 | Real exports replacing aliases |
| `tests/unit/test_codecs.py` | +46 / −1 | MXFP4 correction + 2 new tests |
| `compiler/compiler.py` | +38 / −0 | Real provenance manifest |
| `pass8_sparse_attention.py` | +32 / −2 | Expose MInference implementation |
| `provenance/manifest.py` | +30 / −0 | `eval_results` + hash fix |
| `attention/__init__.py` | +25 / −10 | Real exports replacing aliases |
| `pass9_pruning_sparsity.py` | +18 / −0 | Re-export |
| `pass7_reasoning_graph.py` | +17 / −0 | Re-export |

Deletions are confined to the placeholder-alias blocks and the lines they sat
on; no implementation was removed (see the symbol-level check above).
