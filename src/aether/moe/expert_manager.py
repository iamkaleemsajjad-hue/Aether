"""
MoE expert lifecycle manager.

Manages the hot/warm/cold tier assignment lifecycle. Hot experts stay in GPU
HBM, warm experts are prefetched from CPU DRAM, and cold experts are loaded
on demand. Also handles expert offload and preload scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpertInfo:
    """Metadata for a single expert."""

    index: int
    name: str
    tier: str = "cold"
    activation_rate: float = 0.0
    is_loaded: bool = False
    memory_bytes: int = 0
    last_access: float = 0.0
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "tier": self.tier,
            "activation_rate": self.activation_rate,
            "is_loaded": self.is_loaded,
            "memory_bytes": self.memory_bytes,
            "access_count": self.access_count,
        }


class ExpertManager:
    """Manages the expert lifecycle across memory tiers.

    Responsible for:
    - Assigning experts to hot/warm/cold tiers
    - Tracking which experts are loaded
    - Scheduling expert prefetch and offload
    - Reporting tier statistics
    """

    def __init__(self, memory_budget_gb: float = 0.0) -> None:
        self.memory_budget_gb = memory_budget_gb
        self._experts: dict[str, ExpertInfo] = {}

    def register(self, expert_index: int, name: str, memory_bytes: int = 0) -> ExpertInfo:
        """Register an expert with the manager."""
        info = ExpertInfo(
            index=expert_index,
            name=name,
            memory_bytes=memory_bytes,
        )
        self._experts[name] = info
        return info

    def update_tiers(self, classifications: dict[str, list[int]]) -> None:
        """Update expert tiers based on classification results."""
        for tier, indices in classifications.items():
            for idx in indices:
                name = f"expert_{idx}"
                info = self._experts.get(name)
                if info is not None:
                    info.tier = tier
        self._update_loaded_state()

    def _update_loaded_state(self) -> None:
        """Load/unload experts to fit within memory budget."""
        total_hot_memory = sum(
            info.memory_bytes for info in self._experts.values() if info.tier == "hot"
        )
        if self.memory_budget_gb > 0:
            budget_bytes = int(self.memory_budget_gb * (1024**3))
            for info in sorted(self._experts.values(), key=lambda x: x.activation_rate, reverse=True):
                if info.tier == "hot" and total_hot_memory <= budget_bytes:
                    info.is_loaded = True
                elif info.tier == "warm":
                    info.is_loaded = False
            for info in self._experts.values():
                if info.tier != "hot":
                    info.is_loaded = False
        else:
            for info in self._experts.values():
                info.is_loaded = info.tier in ("hot", "warm")

    def prefetch_candidates(self) -> list[str]:
        """Return the names of warm experts that should be prefetched."""
        return [
            name for name, info in self._experts.items()
            if info.tier == "warm" and not info.is_loaded
        ]

    def offload_candidates(self) -> list[str]:
        """Return the names of cold experts that should be offloaded."""
        return [
            name for name, info in self._experts.items()
            if info.tier == "cold" and info.is_loaded
        ]

    def tier_counts(self) -> dict[str, int]:
        """Return the count of experts in each tier."""
        counts: dict[str, int] = {"hot": 0, "warm": 0, "cold": 0}
        for info in self._experts.values():
            if info.tier in counts:
                counts[info.tier] += 1
        return counts

    def loaded_memory_bytes(self) -> int:
        """Return total memory used by loaded experts."""
        return sum(info.memory_bytes for info in self._experts.values() if info.is_loaded)

    def record_access(self, expert_idx: int) -> None:
        """Record an access to an expert."""
        name = f"expert_{expert_idx}"
        info = self._experts.get(name)
        if info is None:
            return
        info.access_count += 1
        info.last_access = __import__("time").time()

    def __repr__(self) -> str:
        tier = self.tier_counts()
        return f"ExpertManager({len(self._experts)} experts: H={tier['hot']} W={tier['warm']} C={tier['cold']})"
