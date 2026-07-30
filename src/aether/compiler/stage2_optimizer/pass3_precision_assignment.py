"""
Pass 3: Precision Assignment.

Using the sensitivity map from Pass 2, assigns an optimal precision format to
each layer subject to the quality budget constraint.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import PrecisionAssignmentPass
from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["PrecisionAssignmentPass"]


class PrecisionAssignmentPass(PrecisionAssignmentPass):
    """Exported alias of the precision assignment pass."""
