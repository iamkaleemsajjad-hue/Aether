"""
Pass 2: Sensitivity Analysis.

Reports per-layer weight reconstruction sensitivity or an explicitly labeled
text entropy proxy. These scores do not measure model perplexity or establish
a task-quality budget; model evaluation is a separate operation.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import SensitivityAnalysisPass

__all__ = ["SensitivityAnalysisPass"]


