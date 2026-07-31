"""Attention-specific compiler planners."""

from aether.attention.mla import (
    MLAConfig,
    MLAWeightAbsorber,
    MLACompressedKVCache,
    MLADetector,
)

# Backward-compat aliases expected by stage2 optimizer and other consumers
MLACompressionPlan = MLAConfig    # plan ≡ config at this level
MLAPlanner = MLADetector          # planner ≡ detector
MLAForward = MLADetector          # legacy alias — points to detector

__all__ = [
    "MLAConfig",
    "MLAWeightAbsorber",
    "MLACompressedKVCache",
    "MLADetector",
    "MLAForward",
    "MLACompressionPlan",
    "MLAPlanner",
]
