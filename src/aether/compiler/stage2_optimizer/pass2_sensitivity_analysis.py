"""
Pass 2: Sensitivity Analysis.

Computes a per-layer sensitivity score: the change in model perplexity per
saved bit when a layer is quantized. This is the mathematical foundation for
Aether's mixed-precision quantization.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import SensitivityAnalysisPass
from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SensitivityAnalysisPass"]


class SensitivityAnalysisPass(SensitivityAnalysisPass):
    """Exported alias of the sensitivity analysis pass."""
