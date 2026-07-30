"""
Stage 2: Aether Optimizer — six graph-level compiler passes.

The optimizer transforms the raw AEG-IR into an optimized graph that no human
would hand-write. Each pass is a self-contained transformation:

1. Operator Fusion — merge sequential ops into megakernels
2. Sensitivity Analysis — compute d(perplexity)/d(precision) per layer
3. Precision Assignment — assign mixed precision using sensitivity map
4. KV Cache Structuring — paged blocks, radix-tree hints, tiering
5. MoE Expert Routing — hot/warm/cold tiering, threshold-based routing
6. Automatic Parallelism Discovery — search tensor/pipeline/expert/context parallelism
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline

__all__ = ["OptimizerPipeline"]
