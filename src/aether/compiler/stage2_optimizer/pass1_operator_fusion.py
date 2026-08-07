"""
Pass 1: Operator Fusion.

Identifies fuseable operation sequences in the AEG-IR and merges them into
megakernel operations. This reduces kernel launch overhead and eliminates
intermediate memory round-trips.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import OperatorFusionPass

__all__ = ["OperatorFusionPass"]


