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
"""

from __future__ import annotations

# ── PRD v3.1 core ─────────────────────────────────────────────────────────────
from aether.runtime.runtime import Runtime
from aether.runtime.config import RuntimeConfig
from aether.runtime.kv_cache import KVCacheManager
from aether.runtime.scheduler import DisaggregatedScheduler
from aether.runtime.speculative import TreeSpeculativeEngine
from aether.runtime.hardware import HardwareDetector

# ── PRD v4.0 + v5.0 runtime layers (R1–R12) ──────────────────────────────────
from aether.runtime.r1_peagle_engine import PEAGLEEngine, SpeculativeProposal
from aether.runtime.r2_multi_agent_kv import MultiAgentKVCoordinator, AgentKVSession, SharedKVBlock
from aether.runtime.r3_grammar_fsm import GrammarFSMEngine
from aether.runtime.r4_slo_scheduler import SLOScheduler, SLOTier, ScheduledRequest
from aether.runtime.r5_ttt_engine import TTTFastWeightEngine
from aether.runtime.r6_mcp_integration import MCPIntegrationLayer, MCPClient, MCPToolRegistry
from aether.runtime.r7_green_power_manager import GreenPowerManager
from aether.runtime.r8_tee_manager import TEERuntimeManager
from aether.runtime.r9_sub2bit_kv_cache import Sub2BitKVWeightCache
from aether.runtime.r10_video_kv_manager import VideoFrameKVManager, VideoKVSlot
from aether.runtime.r11_semantic_kv_cache import SemanticKVCache, HNSWIndex
from aether.runtime.r12_rlvr_harness import RLVRTrainingHarness

__all__ = [
    # PRD v3.1
    "Runtime",
    "RuntimeConfig",
    "KVCacheManager",
    "DisaggregatedScheduler",
    "TreeSpeculativeEngine",
    "HardwareDetector",
    # PRD v4.0 + v5.0 — R1–R12
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
