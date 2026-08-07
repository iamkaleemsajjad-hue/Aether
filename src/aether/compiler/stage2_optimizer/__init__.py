"""
Stage 2: Aether Optimizer — 22-pass graph-level compiler.

PRD v3.1 passes (1–9):
  1. Operator Fusion         — merge sequential ops into megakernels
  2. Sensitivity Analysis    — compute d(perplexity)/d(precision) per layer
  3. Precision Assignment    — assign mixed precision using sensitivity map
  4. KV Cache Structuring    — paged blocks, radix-tree hints, tiering
  5. MoE Expert Routing      — hot/warm/cold tiering, threshold-based routing
  6. Parallelism Discovery   — tensor/pipeline/expert/context search
  7. Reasoning Graph         — chain-of-thought budget + verifier subgraph
  8. Sparse Attention        — sparse/linear/sliding-window pattern selection
  9. Pruning + Sparsity      — magnitude, SparseGPT, and structured pruning

PRD v4.0 passes (10–17):
  10. MTP Head Compilation      — native Multi-Token Prediction head fusion
  11. Grammar Constraint         — FSM grammar pre-compilation (XGrammar 2026)
  12. Model Merging              — Task Arithmetic / DARE / TIES / FREE
  13. TTT Fast-Weight Injection  — per-layer TTT LoRA slot allocation
  14. Semantic KV Compression    — ChunkKV / SentenceKV / PyramidKV
  15. Cross-Layer KV Sharing     — middle-outward xKV pattern
  16. Green Energy               — DVFS breakpoints + carbon profile
  17. TEE Kernel Wrapping        — NVIDIA CC / Intel TDX / AMD SEV-SNP

PRD v5.0 passes (18–22):
  18. MDLM Drafter               — masked diffusion speculative decoding
  19. Sub-2-Bit Quantization      — BitNet b1.58 / BTC-LLM / NanoQuant
  20. Video Token Compression     — STC / STORM / StreamingTOM VLM
  21. Advanced PEFT              — LoRA+ / LoRAMoE / MoLF / LoRAFusion
  22. RLVR Verifier Injection     — GRPO verifier head + K2V opcodes
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import (
    OptimizerPipeline,
    BasePass,
    OperatorFusionPass,
    SensitivityAnalysisPass,
    PrecisionAssignmentPass,
    KVCacheStructuringPass,
    MoERoutingPass,
    ParallelismDiscoveryPass,
    ReasoningGraphPass,
    SparseAttentionPass,
    PruningSparsityPass,
)

__all__ = [
    # Core pipeline
    "OptimizerPipeline",
    "BasePass",
    # PRD v3.1 passes (1–9)
    "OperatorFusionPass",
    "SensitivityAnalysisPass",
    "PrecisionAssignmentPass",
    "KVCacheStructuringPass",
    "MoERoutingPass",
    "ParallelismDiscoveryPass",
    "ReasoningGraphPass",
    "SparseAttentionPass",
    "PruningSparsityPass",
]
