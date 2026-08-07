"""
Pass 5: MoE Expert Routing Optimization.

For MoE models, this pass profiles expert activations, classifies experts into
hot/warm/cold tiers, adds threshold-based routing hints, and emits intra-expert
sparsity annotations.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import MoERoutingPass

__all__ = ["MoERoutingPass"]


