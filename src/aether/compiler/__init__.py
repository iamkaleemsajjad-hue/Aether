"""
Aether compiler package — ingestion, optimization, hardware targeting, and calibration.

Exports:
    Core compiler:
        Compiler, CompilerConfig, CompilationPlan, OptimizationOpportunity, QualityReport

    AEG Format 2.0 (v4.0 NEW):
        AEGPackageV2, AEGManifest, SpeculationConfig, GrammarManifest,
        GreenEnergyProfile, TEEConfig, MultiAgentConfig, MCPConfig
"""

from __future__ import annotations

from aether.compiler.compiler import Compiler
from aether.compiler.config import CompilerConfig
from aether.compiler.plan import CompilationPlan, OptimizationOpportunity
from aether.compiler.report import QualityReport
from aether.compiler.aeg_format_v2 import (
    AEGPackageV2,
    AEGManifest,
    SpeculationConfig,
    GrammarManifest,
    GreenEnergyProfile,
    TEEConfig,
    MultiAgentConfig,
    MCPConfig,
    AEG_FORMAT_VERSION_V2,
)

__all__ = [
    # Core compiler
    "Compiler",
    "CompilerConfig",
    "CompilationPlan",
    "OptimizationOpportunity",
    "QualityReport",
    # AEG Format 2.0 (v4.0 NEW)
    "AEGPackageV2",
    "AEGManifest",
    "SpeculationConfig",
    "GrammarManifest",
    "GreenEnergyProfile",
    "TEEConfig",
    "MultiAgentConfig",
    "MCPConfig",
    "AEG_FORMAT_VERSION_V2",
]

