"""
Aether compiler package — ingestion, optimization, hardware targeting, and calibration.
"""

from __future__ import annotations

from aether.compiler.compiler import Compiler
from aether.compiler.config import CompilerConfig
from aether.compiler.plan import CompilationPlan, OptimizationOpportunity
from aether.compiler.report import QualityReport

__all__ = [
    "Compiler",
    "CompilerConfig",
    "CompilationPlan",
    "OptimizationOpportunity",
    "QualityReport",
]
