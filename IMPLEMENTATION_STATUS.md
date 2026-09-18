# Aether Runtime — Implementation Status

**Last Updated:** 2026-09-17  
**Audit Type:** Code-level forensic verification (not README-based)  
**Test Results:** 77/77 new unit tests passing

---

## Summary

This document records the **actual implementation status** of every subsystem advertised by Aether, verified against source code. It replaces any previous status based on README claims.

---

## Phase 0: Audit Baseline (COMPLETE)

| Claim | Status | Evidence |
|---|---|---|
| AEG compilation pipeline | ✅ Real | `compiler/compiler.py`, stage1/stage2 |
| Architecture detection (25+ families) | ✅ Real | `stage1_ingestion/architecture_detection.py` |
| AEG graph format | ✅ Real | `core/aeg_format.py`, `aeg_format_v2.py` |
| TorchAEGEngine forward pass | ✅ Real | `runtime/torch_engine.py` (3366 lines) |
| RoPE (all variants) | ✅ Real | `runtime/rope_scaling.py`, `torch_engine.py` |
| KV cache (monotonic grow) | ✅ Real | `TorchKVCache._append_kv()` |
| GQA/MHA/MQA attention | ✅ Real | `torch_engine.py:_attention()` |
| ALiBi attention (slow path) | ✅ Real → **Optimized** | einsum → SDPA fast path added |
| Optimizer passes 1–6 | ✅ Real (validated) | `stage2_optimizer/optimizer.py` |
| Optimizer passes 7–22 | ✅ Real (metadata-producing) | `optimizer.py` |
| Benchmarking infrastructure | ✅ Real | `benchmark/suite/` |

---

## Phase 1: New Implementations (THIS SESSION — COMPLETE)

### ✅ Quantized Linear Dispatch
**File:** `src/aether/kernels/quantized_linear.py`

| Format | Execution Path | Backend |
|---|---|---|
| FP8 E4M3 | `torch._scaled_mm` (sm89+), dequant fallback | CUDA / NumPy |
| INT8 symmetric | `torch._int_mm` (sm80+), dequant fallback | CUDA / NumPy |
| INT8 asymmetric | dequant fallback | NumPy |
| NF4 | bitsandbytes path, codec dequant fallback | bnb / NumPy |
| INT4 packed | dequant fallback (no native INT4 GEMM yet) | NumPy |
| BF16/FP16/FP32 | reference `F.linear` | PyTorch |

Every fallback explicitly labeled `execution_mode="dequant_fallback"`. No silent FP32 conversions.  
**Codec math verified** via 23 unit tests.

---

### ✅ Paged KV Cache (Real Tensor Storage)
**File:** `src/aether/runtime/paged_kv_cache.py`

The previous `kv_cache.py` stored only page IDs and tier metadata — no tensors. This stores actual KV data.

| Feature | Implementation |
|---|---|
| Pool tensor | `np.zeros((max_pages, page_size, num_layers, 2, num_kv_heads, head_dim))` |
| PyTorch mirror | `torch.zeros(same_shape, device=device)` |
| Block allocator | O(1) alloc/free with ref-counting (thread-safe) |
| KV write | `pool[page_id, slot, layer_idx, 0/1] = k/v` |
| KV read | gather pages → `(total_tokens, heads, dim)` |
| Prefix sharing | SHA-256 content hash + radix-prefix scan |
| CPU offload | `pool[page_id].cpu().numpy()` → numpy mirror |
| Pool exhaustion | `RuntimeError("KV pool exhausted")` |

**Research basis:** Kwon et al. (2023) "PagedAttention" (OSDI 2023).  
**Verified** via 21 unit tests.

---

### ✅ Speculative Decoding (Real Algorithm 1)
**File:** `src/aether/runtime/speculative_engine.py`

The previous `speculative.py` used `(token_id * 31 + branch_index) % 100000` to generate "draft" tokens. That is not speculative decoding.

| Algorithm | Reference | Status |
|---|---|---|
| DraftTargetSpecDecoder | Leviathan et al. 2023 (ICML) | ✅ Real — Algorithm 1 exact |
| SelfSpeculativeDecoder | Zhang et al. 2024 | ✅ Real — explicit fallback if no early_exit |
| MedusaDecoder | Cai et al. 2024 | ✅ Real — explicit fallback if no medusa_heads |

**Algorithm 1 correctness:**
- Draft tokens: real forward passes (not fabricated)
- Target verifies γ+1 positions in ONE forward pass  
- Acceptance: `min(1, p_target(x) / p_draft(x))` — exact
- Rejection: sample from `(p_target - p_draft)⁺` — corrected distribution
- Bonus token always emitted from target distribution
- Provably samples from target distribution (Leviathan et al. Theorem 1)

**Verified** via 33 unit tests.

---

### ✅ ALiBi Fast Path via SDPA
**File:** `src/aether/runtime/torch_engine.py`

Previous: ALiBi always used slow `torch.einsum` path, materializing full score matrix, blocking FlashAttention.

Fix: ALiBi bias built as float `attn_mask` `(1, heads, query, key)` and passed to `scaled_dot_product_attention`. PyTorch 2.x SDPA dispatches to FA2 with additive float bias on supported hardware.

Expected speedup: **2–4× for GPT-Neo / BLOOM decode on CUDA**.

---

### ✅ ForwardStep Protocol
**File:** `src/aether/runtime/torch_engine.py`

Added `forward_step(token_ids, kv_cache, return_all_logits) → (np.ndarray, cache)` and `init_kv_cache() → TorchKVCache` to `TorchAEGEngine`.

Enables `ContinuousBatcher` and `DraftTargetSpecDecoder` to drive any engine without device-specific code.

---

### ✅ Continuous Batcher (Orca-style)
**File:** `src/aether/runtime/continuous_batcher.py`

| Feature | Implementation |
|---|---|
| Waiting queue | FIFO with configurable max_queue_size |
| Prefill | Chunked (Sarathi-Serve) or full-sequence |
| Decode | Continuous: all active seqs run each step |
| Admission | max_decode_seqs capacity gate |
| Lifecycle | WAITING → PREFILLING → DECODING → FINISHED/ERROR |
| Stats | Throughput (TPS), TTFT per request |
| Sync API | `generate_sync()` |

**Research basis:** Yu et al. (2022) "Orca" (OSDI 2022); Agrawal et al. (2024) "Sarathi-Serve".

---

### ✅ Quality Validator
**File:** `src/aether/evaluation/quality_validator.py`

| Check | Metric | Default Threshold |
|---|---|---|
| Logit KL divergence | KL(ref \|\| cand) | ≤ 0.01 |
| Logit MAE | mean \|ref - cand\| | ≤ 0.10 |
| Cosine similarity | cos(ref, cand) | ≥ 0.999 |
| Token agreement | greedy match rate | ≥ 95% |
| Perplexity | relative PPL increase | ≤ 1% |
| Activation | per-layer relative error | ≤ 5% |

---

## Phase 2: Remaining Work (Priority Order)

| Subsystem | Status | Priority |
|---|---|---|
| Quantized dispatch wired into `_matmul` | Not wired | HIGH |
| PagedKVPool wired into TorchAEGEngine | Not wired | HIGH |
| Continuous batcher end-to-end test | Missing | HIGH |
| Quality validator integration test | Missing | MEDIUM |
| `speculative.py` → `speculative_engine.py` migration | Parallel files | LOW |
| Prefix sharing in ContinuousBatcher | Not connected | LOW |

---

## What Is NOT Implemented (Honest)

1. **Triton flash attention** — `kernels/flash_attn_triton.py` is a skeleton. Requires GPU.
2. **FP8 `torch._scaled_mm`** — CPU always dequants. Correctly labeled.
3. **INT8 W8A8 on CPU** — CPU always dequants. Correctly labeled.
4. **Medusa heads** — Requires fine-tuned checkpoints. Fallback is labeled.
5. **MLA (Multi-head Latent Attention)** — `attention/mla.py` at 21% coverage.
6. **CPU offload scheduling** — `PagedKVPool.offload_to_cpu()` works but no eviction policy calls it.
