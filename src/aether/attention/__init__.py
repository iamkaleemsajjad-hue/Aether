"""
Attention-specific compiler planners and runtime kernels.

Multi-Head Latent Attention (MLA) is exported as two layers:

* **Plan layer** — :class:`MLAPlanner` produces an :class:`MLACompressionPlan`
  recorded at ``.aeg/mla/plan.json`` during compilation.
* **Runtime layer** — :class:`MLAAttention`, :class:`MLACompressedKVCache`, and
  :class:`MLAWeightAbsorber` execute that plan.
"""

from aether.attention.mla import (
    MLAAttention,
    MLACompressedKVCache,
    MLACompressionPlan,
    MLAConfig,
    MLADetector,
    MLAPlanner,
    MLAWeightAbsorber,
)

#: Legacy alias. Points at the class implementing the MLA forward path
#: (``forward_prefill`` / ``forward_decode``); it previously aliased
#: ``MLADetector``, which has no forward path.
MLAForward = MLAAttention

__all__ = [
    # Plan layer
    "MLAPlanner",
    "MLACompressionPlan",
    "MLADetector",
    "MLAConfig",
    # Runtime layer
    "MLAAttention",
    "MLAWeightAbsorber",
    "MLACompressedKVCache",
    "MLAForward",
]
