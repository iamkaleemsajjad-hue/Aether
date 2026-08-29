"""
Parallelism planner.

Searches over tensor/pipeline/expert/context parallelism degrees and produces
a sharding plan for a given model and GPU count. Used by the compiler and by
the disaggregated scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether.core.constants import (
    MAX_CONTEXT_PARALLEL_DEGREE,
    MAX_EXPERT_PARALLEL_DEGREE,
    MAX_PIPELINE_PARALLEL_STAGES,
    MAX_TENSOR_PARALLEL_DEGREE,
)
from aether.core.exceptions import ParallelismError
from aether.core.types import ModelArchitecture, ShardingPlan
from aether.parallelism.sharding import DeviceCapacity
from aether.placement.waterfill import WaterfillInfeasible, water_fill
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HeterogeneousShardingPlan:
    """One model-wide placement plan for mixed CPU/GPU/NPU execution."""

    phase: str
    devices: tuple[DeviceCapacity, ...]
    weight_ranges: dict[str, tuple[int, int]]
    weight_fractions: dict[str, float]
    model_copies: int
    estimated_weight_bytes: int
    estimated_all_reduce_bytes: int
    bottleneck_time_units: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "aether_heterogeneous_sharding_v1",
            "phase": self.phase,
            "model_copies": self.model_copies,
            "devices": [device.__dict__ for device in self.devices],
            "weight_ranges": {key: list(value) for key, value in self.weight_ranges.items()},
            "weight_fractions": self.weight_fractions,
            "estimated_weight_bytes": self.estimated_weight_bytes,
            "estimated_all_reduce_bytes": self.estimated_all_reduce_bytes,
            "bottleneck_time_units": self.bottleneck_time_units,
            "invariant": "each weight element has exactly one device owner",
        }


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
        """Generate plans for every available GPU count up to ``max_gpus``."""
        plans: dict[int, ShardingPlan] = {}
        for num_gpus in range(1, max_gpus + 1):
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

        # For inference, the portable default is a single model-wide tensor
        # partition.  It guarantees that each physical GPU owns one shard of
        # every splittable projection rather than receiving a full replica.
        # Pipeline/context parallelism remains available through explicit plans,
        # but is not silently selected for a normal multi-GPU request.
        config = ParallelismConfig(tensor_parallel_degree=num_gpus)
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

    def plan_for_devices(
        self,
        devices: list[DeviceCapacity],
        phase: str = "decode",
        precision_bits: float = 16.0,
    ) -> HeterogeneousShardingPlan:
        """Build a capacity-weighted, single-copy plan for any device mix.

        The split is capped water-filling
        (:func:`aether.placement.waterfill.water_fill`): distribute in proportion to
        throughput, pin any device that would exceed its memory, redistribute the
        remainder, repeat.  That is the min-max optimum under a linear resource with
        per-device caps, and it is the reason an asymmetric pair gets an asymmetric
        split instead of an error.

        The previous behaviour split purely by ``compute_units`` and then *raised* if
        a shard exceeded 90% of a device — so the very case this method exists for,
        a 16 GB device beside a 24 GB one, failed rather than repartitioning.

        Raises:
            ParallelismError: Only when the devices cannot hold the model at all.
        """
        if not devices:
            raise ParallelismError("at least one execution device is required")
        if precision_bits <= 0:
            raise ParallelismError("precision_bits must be positive")
        total_bytes = max(
            1,
            int(self.architecture.params_billion * 1_000_000_000 * precision_bits / 8.0),
        )
        # A device with no declared memory is treated as unbounded rather than as
        # zero: an unknown capacity must not silently exclude a device.
        caps = [
            float(device.memory_bytes * 0.90) if device.memory_bytes else float("inf")
            for device in devices
        ]
        throughputs = [max(device.compute_units, 1e-9) for device in devices]
        try:
            weights = water_fill(float(total_bytes), throughputs, caps)
        except WaterfillInfeasible as exc:
            raise ParallelismError(
                f"the given devices cannot hold {total_bytes} bytes of weights: {exc}"
            ) from exc

        boundaries: list[tuple[int, int]] = []
        cursor = 0
        for index, weight in enumerate(weights):
            end = total_bytes if index == len(weights) - 1 else cursor + int(total_bytes * weight)
            boundaries.append((cursor, end))
            cursor = end
        weight_ranges = {
            device.device_id: value
            for device, value in zip(devices, boundaries, strict=False)
        }
        fractions = {
            device.device_id: (end - start) / total_bytes
            for device, (start, end) in zip(devices, boundaries, strict=False)
        }
        activation_bytes = int(
            self.architecture.hidden_size * max(1, self.architecture.layers) * precision_bits / 8.0
        )
        participants = len(devices)
        all_reduce_bytes = (
            int(activation_bytes * 2.0 * (participants - 1) / participants)
            if participants > 1 else 0
        )
        work_time = max(
            fractions[device.device_id] / max(device.compute_units, 1e-9)
            for device in devices
        )
        communication_time = (
            max(
                all_reduce_bytes / (device.bandwidth_gbps * 1_000_000_000)
                for device in devices
            )
            if participants > 1 else 0.0
        )
        return HeterogeneousShardingPlan(
            phase=phase,
            devices=tuple(devices),
            weight_ranges=weight_ranges,
            weight_fractions=fractions,
            model_copies=1,
            estimated_weight_bytes=total_bytes,
            estimated_all_reduce_bytes=all_reduce_bytes,
            bottleneck_time_units=work_time + communication_time,
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
