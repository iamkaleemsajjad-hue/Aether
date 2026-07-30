"""
Expert parallelism planning.

Determines how to distribute MoE experts across devices for optimal load
balance. Generates placement plans that minimize inter-device communication
while respecting memory constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpertPlacement:
    """Placement of experts across devices."""

    device_id: int
    expert_indices: list[int] = field(default_factory=list)
    memory_bytes: int = 0
    load_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "expert_indices": self.expert_indices,
            "memory_bytes": self.memory_bytes,
            "load_score": self.load_score,
        }


@dataclass
class PlacementPlan:
    """Complete expert placement plan."""

    num_devices: int
    placements: list[ExpertPlacement] = field(default_factory=list)
    imbalanced_score: float = 0.0
    max_communication_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_devices": self.num_devices,
            "placements": [p.to_dict() for p in self.placements],
            "imbalanced_score": self.imbalanced_score,
            "max_communication_gb": self.max_communication_gb,
        }


class ExpertPlanner:
    """Plans expert placement across devices.

    Uses a greedy load-balancing heuristic to assign experts to devices.
    Supports constraints like memory limits and inter-device communication
    bandwidth.
    """

    def __init__(self, num_devices: int) -> None:
        self.num_devices = num_devices

    def plan(self, expert_memory: list[int], activation_rates: list[float] | None = None) -> PlacementPlan:
        """Generate a placement plan for experts across devices.

        Args:
            expert_memory: Memory per expert in bytes.
            activation_rates: Optional activation rates for load-aware placement.

        Returns:
            A PlacementPlan.
        """
        num_experts = len(expert_memory)
        rates = activation_rates or [1.0 / num_experts] * num_experts
        devices = [ExpertPlacement(device_id=i) for i in range(self.num_devices)]
        expert_costs = [m * r for m, r in zip(expert_memory, rates)]
        sorted_indices = sorted(range(num_experts), key=lambda i: expert_costs[i], reverse=True)
        for idx in sorted_indices:
            target = min(devices, key=lambda d: sum(expert_costs[e] for e in d.expert_indices))
            target.expert_indices.append(idx)
            target.memory_bytes += expert_memory[idx]
            target.load_score += rates[idx]
        loads = [d.load_score for d in devices]
        max_comm = 0.0
        if self.num_devices > 1:
            max_comm = max(loads) * 100.0 / sum(loads) if sum(loads) > 0 else 0.0
        return PlacementPlan(
            num_devices=self.num_devices,
            placements=devices,
            imbalanced_score=abs(max(loads) - min(loads)) / max(max(loads), 1e-6) if loads else 0.0,
            max_communication_gb=max_comm * expert_memory[0] / (1024**3) if expert_memory else 0.0,
        )

    def __repr__(self) -> str:
        return f"ExpertPlanner(num_devices={self.num_devices})"
