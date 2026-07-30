"""
Pass 1: Operator Fusion.

Identifies fuseable operation sequences in the AEG-IR and merges them into
megakernel operations. This reduces kernel launch overhead and eliminates
intermediate memory round-trips.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import OperatorFusionPass
from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["OperatorFusionPass"]


class OperatorFusionPass(OperatorFusionPass):
    """Exported alias of the operator fusion pass."""
