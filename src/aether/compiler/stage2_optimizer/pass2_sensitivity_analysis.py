"""
Pass 2: Sensitivity Analysis.

Computes a per-layer sensitivity score: the change in model perplexity per
saved bit when a layer is quantized. This is the mathematical foundation for
Aether's mixed-precision quantization.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import SensitivityAnalysisPass

__all__ = ["SensitivityAnalysisPass"]
