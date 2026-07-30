"""
Parallelism planner.

Searches over tensor/pipeline/expert/context parallelism degrees and produces
a sharding plan for a given model and GPU count. Used by the compiler and by
the disaggregated scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import (
    MAX_CONTEXT_PARALLEL_DEGREE,
    MAX_EXPERT_PARALLEL_DEGREE,
    MAX_PIPELINE_PARALLEL_STAGES,
    MAX_TENSOR_PARALLEL_DEGREE,
)
from aether.core.exceptions import ParallelismError
from aether.core.types import ModelArchitecture, ShardingPlan
from aether.parallelism.mesh import DeviceMesh
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelismConfig:
    """Candidate parallelism configuration."""

    tensor_parallel_degree: int = 1
    pipeline_stages: int = 1
    expert_parallel_degree: int = 1
    context_parallel_degree: int = 1

    def total_devices(self) -> int:
        return (
            self.tensor_parallel_degree
            * self.pipeline_stages
            * self.expert_parallel_degree
            * self.context_parallel_degree
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_parallel_degree": self.tensor_parallel_degree,
            "pipeline_stages": self.pipeline_stages,
            "expert_parallel_degree": self.expert_parallel_degree,
            "context_parallel_degree": self.context_parallel_degree,
            "total_devices": self.total_devices(),
        }


class ParallelismPlanner:
    """Discovers parallelism strategies for a given model and device count.

    Evaluates candidate configurations and returns a ShardingPlan. The scoring
    heuristic prefers tensor parallelism for decode and pipeline parallelism for
    prefill, with expert and context parallelism for MoE and long-context models.
    """

    def __init__(self, architecture: ModelArchitecture) -> None:
        self.architecture = architecture

    def generate_plans(self, max_gpus: int = 8) -> dict[int, ShardingPlan]:
        """Generate sharding plans for 1, 2, 4, ..., max_gpus."""
        plans: dict[int, ShardingPlan] = {}
        for num_gpus in range(1, max_gpus + 1):
            if num_gpus & (num_gpus - 1) != 0 and num_gpus != 1:
                continue
            try:
                plans[num_gpus] = self.plan_for_gpus(num_gpus)
            except ParallelismError as exc:
                logger.warning("Failed to plan for %d GPUs: %s", num_gpus, exc)
        return plans

    def plan_for_gpus(self, num_gpus: int, phase: str = "decode") -> ShardingPlan:
        """Generate a sharding plan for a given number of GPUs and phase."""
        if num_gpus < 1:
            msg = "num_gpus must be >= 1"
            raise ParallelismError(msg)

        config = self._search_config(num_gpus, phase)
        memory_per_gpu = self._estimate_memory_per_gpu(config, phase)
        return ShardingPlan(
            num_gpus=num_gpus,
            phase=phase,
            tensor_parallel_degree=config.tensor_parallel_degree,
            pipeline_stages=config.pipeline_stages,
            expert_parallel_degree=config.expert_parallel_degree,
            context_parallel_degree=config.context_parallel_degree,
            memory_per_gpu_gb=memory_per_gpu,
        )

    def _search_config(self, num_gpus: int, phase: str) -> ParallelismConfig:
        """Search over parallelism degrees to find a valid configuration."""
        best_score = -1.0
        best_config = ParallelismConfig()
        for tp in self._divisors(num_gpus, MAX_TENSOR_PARALLEL_DEGREE):
            for pp in self._divisors(num_gpus // tp, MAX_PIPELINE_PARALLEL_STAGES):
                for ep in self._divisors(num_gpus // (tp * pp), MAX_EXPERT_PARALLEL_DEGREE):
                    cp = num_gpus // (tp * pp * ep)
                    if cp > MAX_CONTEXT_PARALLEL_DEGREE:
                        continue
                    if cp < 1:
                        continue
                    config = ParallelismConfig(
                        tensor_parallel_degree=tp,
                        pipeline_stages=pp,
                        expert_parallel_degree=ep,
                        context_parallel_degree=cp,
                    )
                    if config.total_devices() != num_gpus:
                        continue
                    score = self._score_config(config, phase)
                    if score > best_score:
                        best_score = score
                        best_config = config
        return best_config

    def _divisors(self, n: int, max_val: int) -> list[int]:
        """Return divisors of n up to max_val in descending order."""
        return sorted([d for d in range(1, min(n, max_val) + 1) if n % d == 0], reverse=True)

    def _score_config(self, config: ParallelismConfig, phase: str) -> float:
        """Score a parallelism configuration. Higher is better."""
        score = 0.0
        if phase == "decode":
            # Decode prefers high TP for low latency, lower PP/CP overhead
            score += config.tensor_parallel_degree * 2.0
            score -= config.pipeline_stages * 0.5
            score -= config.context_parallel_degree * 0.3
        elif phase == "prefill":
            # Prefill prefers PP and CP for throughput
            score += config.pipeline_stages * 1.5
            score += config.context_parallel_degree * 1.0
            score += config.tensor_parallel_degree * 0.5
        if self.architecture.is_moe and config.expert_parallel_degree > 1:
            score += config.expert_parallel_degree * 0.8
        # Penalize fragmentation
        score -= (config.total_devices() - config.tensor_parallel_degree) * 0.1
        return score

    def _estimate_memory_per_gpu(self, config: ParallelismConfig, phase: str) -> float:
        """Estimate memory per GPU in GB."""
        params_gb = self.architecture.params_billion
        if config.tensor_parallel_degree > 1:
            params_gb /= config.tensor_parallel_degree
        if config.pipeline_stages > 1:
            params_gb /= config.pipeline_stages
        if config.expert_parallel_degree > 1 and self.architecture.is_moe:
            params_gb /= config.expert_parallel_degree
        # Activation overhead
        overhead = 1.2 if phase == "prefill" else 1.1
        return params_gb * overhead

    def __repr__(self) -> str:
        return f"ParallelismPlanner({self.architecture.family}, {self.architecture.params_billion}B)"
