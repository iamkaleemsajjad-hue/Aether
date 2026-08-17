"""
Aether runtime package — execution engine, scheduler, KV cache, and speculative decoding.

PRD v3.1 (original):
  Runtime, RuntimeConfig, KVCacheManager, DisaggregatedScheduler,
  TreeSpeculativeEngine, HardwareDetector.

PRD v4.0 + v5.0 (new runtime layers R1–R12):
  PEAGLEEngine, MultiAgentKVCoordinator, GrammarFSMEngine,
  SLOScheduler, TTTFastWeightEngine, MCPIntegrationLayer,
  GreenPowerManager, TEERuntimeManager, Sub2BitKVWeightCache,
  VideoFrameKVManager, SemanticKVCache, RLVRTrainingHarness.

FRAMEWORK-INDEPENDENCE NOTE:
  The core runtime (Runtime, RuntimeConfig, KVCacheManager, HardwareDetector)
  does NOT require torch. It is imported eagerly.

  The R1–R12 runtime layers may optionally use torch for accelerated paths.
  They are imported LAZILY via __getattr__ so that:
    - `import aether` does NOT pull torch into sys.modules
    - torch is only loaded when a specific R-layer is actually instantiated
"""

from __future__ import annotations

# ── PRD v3.1 core — always available, no torch required ───────────────────────
from aether.runtime.runtime import Runtime
from aether.runtime.config import RuntimeConfig
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.scheduler import DisaggregatedScheduler
from aether.runtime.speculative import TreeSpeculativeEngine
from aether.runtime.hardware import HardwareDetector

# ── Lazy import map for R1–R12 runtime layers ────────────────────────────────
# These are resolved on first attribute access via __getattr__, not at import
# time. This prevents torch (and other optional deps) from loading when Aether
# is merely imported.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # name: (module_path, attr_name)
    "PEAGLEEngine":          ("aether.runtime.r1_peagle_engine",  "PEAGLEEngine"),
    "SpeculativeProposal":   ("aether.runtime.r1_peagle_engine",  "SpeculativeProposal"),
    "MultiAgentKVCoordinator": ("aether.runtime.r2_multi_agent_kv", "MultiAgentKVCoordinator"),
    "AgentKVSession":        ("aether.runtime.r2_multi_agent_kv", "AgentKVSession"),
    "SharedKVBlock":         ("aether.runtime.r2_multi_agent_kv", "SharedKVBlock"),
    "GrammarFSMEngine":      ("aether.runtime.r3_grammar_fsm",    "GrammarFSMEngine"),
    "SLOScheduler":          ("aether.runtime.r4_slo_scheduler",  "SLOScheduler"),
    "SLOTier":               ("aether.runtime.r4_slo_scheduler",  "SLOTier"),
    "ScheduledRequest":      ("aether.runtime.r4_slo_scheduler",  "ScheduledRequest"),
    "TTTFastWeightEngine":   ("aether.runtime.r5_ttt_engine",     "TTTFastWeightEngine"),
    "MCPIntegrationLayer":   ("aether.runtime.r6_mcp_integration","MCPIntegrationLayer"),
    "MCPClient":             ("aether.runtime.r6_mcp_integration","MCPClient"),
    "MCPToolRegistry":       ("aether.runtime.r6_mcp_integration","MCPToolRegistry"),
    "GreenPowerManager":     ("aether.runtime.r7_green_power_manager", "GreenPowerManager"),
    "TEERuntimeManager":     ("aether.runtime.r8_tee_manager",    "TEERuntimeManager"),
    "Sub2BitKVWeightCache":  ("aether.runtime.r9_sub2bit_kv_cache","Sub2BitKVWeightCache"),
    "VideoFrameKVManager":   ("aether.runtime.r10_video_kv_manager","VideoFrameKVManager"),
    "VideoKVSlot":           ("aether.runtime.r10_video_kv_manager","VideoKVSlot"),
    "SemanticKVCache":       ("aether.runtime.r11_semantic_kv_cache","SemanticKVCache"),
    "HNSWIndex":             ("aether.runtime.r11_semantic_kv_cache","HNSWIndex"),
    "RLVRTrainingHarness":   ("aether.runtime.r12_rlvr_harness",  "RLVRTrainingHarness"),
}


def __getattr__(name: str):
    """Lazy-load R1–R12 runtime layers on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib  # noqa: PLC0415
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'aether.runtime' has no attribute {name!r}")


__all__ = [
    # PRD v3.1 — eagerly loaded, no torch required
    "Runtime",
    "RuntimeConfig",
    "KVCacheManager",
    "DisaggregatedScheduler",
    "TreeSpeculativeEngine",
    "HardwareDetector",
    # PRD v4.0 + v5.0 — R1–R12 (lazy loaded)
    "PEAGLEEngine",
    "SpeculativeProposal",
    "MultiAgentKVCoordinator",
    "AgentKVSession",
    "SharedKVBlock",
    "GrammarFSMEngine",
    "SLOScheduler",
    "SLOTier",
    "ScheduledRequest",
    "TTTFastWeightEngine",
    "MCPIntegrationLayer",
    "MCPClient",
    "MCPToolRegistry",
    "GreenPowerManager",
    "TEERuntimeManager",
    "Sub2BitKVWeightCache",
    "VideoFrameKVManager",
    "VideoKVSlot",
    "SemanticKVCache",
    "HNSWIndex",
    "RLVRTrainingHarness",
]
