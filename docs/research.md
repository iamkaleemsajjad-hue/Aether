# Research Notes

Aether is an AI model compiler and runtime. The PRD maps to established systems ideas: compiler IR stability, graph-level optimization, memory-aware serving, disaggregated scheduling, speculative decoding, quantization, and automatic parallelism.

## Design Evidence

| Area | Source | Product Decision |
|------|--------|------------------|
| Compiler IR | MLIR, A Compiler Infrastructure for the End of Moore's Law | Keep AEG-IR semantic and versioned so hardware-specific lowering can happen late. |
| KV memory | PagedAttention, SOSP 2023 | Represent KV cache as paged blocks instead of contiguous request buffers. |
| Disaggregation | DistServe, OSDI 2024 | Split prefill and decode plans because prefill is compute-heavy and decode is memory-bandwidth-heavy. |
| Parallelism | Alpa, OSDI 2022 | Search tensor, pipeline, context, and expert parallelism with cost models instead of user flags. |
| Quantization | GPTQ and AWQ | Use calibration-aware sensitivity before assigning low-bit precision. |
| Speculation | SpecInfer and OPT-Tree | Verify a branch tree against the target model and adapt based on acceptance rate. |

## Implementation Notes

- The current calibration layer is deterministic and offline-friendly. It estimates corpus entropy and precision penalties when backend logits are unavailable.
- The production evaluator should replace the proxy with backend next-token log-probabilities once real model execution is present.
- The scheduler now chunks long prefills and tracks decode work separately, which is the minimum local abstraction needed for later prefill/decode worker pools.
- The speculative engine now records accepted versus proposed draft tokens and rejects branches when target logits disagree.

## Source Links

- MLIR: https://arxiv.org/abs/2002.11054
- PagedAttention: https://doi.org/10.1145/3600006.3613165
- DistServe: https://arxiv.org/abs/2401.09670
- Alpa: https://arxiv.org/abs/2201.12023
- AWQ: https://arxiv.org/abs/2306.00978
- GPTQ: https://arxiv.org/abs/2210.17323
- SpecInfer: https://arxiv.org/abs/2305.09781

## Research Guardrails

- Claims in user-facing docs must distinguish implemented behavior from roadmap targets.
- Benchmarks must report hardware, backend, model, precision map, context length, batch shape, and warmup policy.
- Any paper-derived speedup should be cited as source-system evidence until reproduced in Aether Bench.


## v3.1 Research Addendum

| Area | Source | Aether Implementation |
|------|--------|-----------------------|
| Long-context sparse attention | MInference, NeurIPS 2024 | Pass 8 emits `attention_head_patterns.json` with vertical-slash, block-sparse, and A-shape plans. |
| Pruning and sparsity | SparseGPT and Wanda | Pass 9 emits `sparsity_masks.json` with sensitivity-aware sparsity targets. |
| LoRA serving | LoRA, S-LoRA, Punica | `LoRAHotSwapEngine` applies per-request low-rank deltas in one batch. |
| Hybrid SSM models | Mamba, Mamba-2, Jamba, RWKV | `HybridMemoryPool` snapshots KV and recurrent SSM state for speculative rollback. |
| Watermarking | SynthID-Text and green-list token watermarking | `AetherOutputWatermark` applies deterministic green-list logit bias and z-score detection. |
| Provenance | C2PA and EU AI Act transparency obligations | `ProvenanceManifest` records source model, compiler passes, eval gate results, and compliance status. |

Additional source links:

- MInference: https://arxiv.org/abs/2407.02490
- SparseGPT: https://arxiv.org/abs/2301.00774
- Wanda: https://arxiv.org/abs/2306.11695
- LoRA: https://arxiv.org/abs/2106.09685
- Punica: https://arxiv.org/abs/2310.18547
- Mamba: https://arxiv.org/abs/2312.00752
- SynthID-Text: https://www.nature.com/articles/s41586-024-08025-4
