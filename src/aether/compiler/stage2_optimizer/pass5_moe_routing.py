"""
Pass 5: MoE Expert Routing Optimization.

For MoE models, this pass profiles expert activations, classifies experts into
hot/warm/cold tiers, adds threshold-based routing hints, and emits intra-expert
sparsity annotations.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import MoERoutingPass
from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["MoERoutingPass"]


class MoERoutingPass(MoERoutingPass):
    """Exported alias of the MoE routing pass."""
