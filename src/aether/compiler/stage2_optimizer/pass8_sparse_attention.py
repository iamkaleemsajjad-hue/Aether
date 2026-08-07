"""
Pass 8: Sparse Attention Compiler.

Classifies each attention head into a sparse pattern (A-shape,
vertical-slash, or block-sparse) at compile time and bakes the assignment
into the AEG artifact, following MInference (NeurIPS 2024).

Two entry points are exported:

* :class:`SparseAttentionPass` — the pipeline-facing pass run by
  :class:`~aether.compiler.stage2_optimizer.optimizer.OptimizerPipeline`.
* :class:`Pass8MInference` — the standalone MInference compiler, plus its
  pattern classifier and kernel selector, for callers that drive pattern
  detection directly against real attention maps.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import SparseAttentionPass
from aether.compiler.stage2_optimizer.pass8_minference import (
    AttentionPatternClassifier,
    HeadPattern,
    MInferenceProfile,
    Pass8MInference,
    SparseAttentionKernel,
    SparsePattern,
)

__all__ = [
    "SparseAttentionPass",
    "Pass8MInference",
    "AttentionPatternClassifier",
    "SparseAttentionKernel",
    "MInferenceProfile",
    "HeadPattern",
    "SparsePattern",
]


