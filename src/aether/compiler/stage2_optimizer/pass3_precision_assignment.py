"""
Pass 3: Precision Assignment.

Using the sensitivity map from Pass 2, assigns an optimal precision format to
each layer subject to the quality budget constraint.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import PrecisionAssignmentPass

__all__ = ["PrecisionAssignmentPass"]


