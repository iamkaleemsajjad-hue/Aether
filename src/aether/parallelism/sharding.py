"""
Sharding strategies and annotations for distributed tensor parallelism.

Defines how tensor dimensions are split across devices and provides utilities
for shard shape computation and all-gather/all-reduce decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class ShardingAxis:
    """Axis along which a tensor is sharded."""

    ROW = "row"
    COLUMN = "column"
    SEQUENCE = "sequence"
    HEAD = "head"
    REPLICATED = "replicated"


@dataclass(frozen=True)
class DeviceCapacity:
    """Measured capacity of one participant in a heterogeneous model mesh.

    ``compute_units`` is a relative sustained-throughput estimate, not a
    vendor peak.  ``memory_bytes`` is a hard placement constraint when it is
    non-zero.  Keeping these values explicit prevents the common mistake of
    treating a CPU and a GPU as equal workers merely because they are both
    present.
    """

    device_id: str
    kind: str
    memory_bytes: int = 0
    compute_units: float = 1.0
    bandwidth_gbps: float = 1.0

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        if self.compute_units <= 0 or self.bandwidth_gbps <= 0:
            raise ValueError("device compute and bandwidth capacities must be positive")
        if self.memory_bytes < 0:
            raise ValueError("memory_bytes must be non-negative")


def capacity_weighted_partition(length: int, capacities: list[float]) -> list[tuple[int, int]]:
    """Partition ``length`` contiguously in proportion to device capacity.

    For cumulative normalized capacity ``c_i`` the boundary is
    ``b_i = floor(length * c_i / sum(c))``.  This is lossless, deterministic,
    and minimizes the maximum fractional rounding error.  It is the placement
    primitive used by the heterogeneous planner; unlike equal splitting it
    gives a CPU a smaller shard when its measured throughput is lower.
    """
    if length < 0 or not capacities or any(value <= 0 for value in capacities):
        raise ValueError("length must be non-negative and capacities must be positive")
    # A tensor-parallel mesh may be wider than a particular tensor dimension
    # (for example one KV head across eight ranks). Empty local shards are
    # valid: concatenation reconstructs the global tensor and no weight is
    # replicated merely to satisfy the mesh width.
    if length < len(capacities):
        return balanced_partition(length, len(capacities))
    total = float(sum(capacities))
    boundaries = [0]
    cumulative = 0.0
    for value in capacities[:-1]:
        cumulative += float(value)
        boundaries.append(int(length * cumulative / total))
    boundaries.append(length)
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if any(end <= start for start, end in ranges):
        # Very skewed capacities can round a small shard to zero.  Allocate
        # one item to each device and distribute the remainder proportionally.
        ranges = balanced_partition(length, len(capacities))
    return ranges


def balanced_partition(length: int, parts: int) -> list[tuple[int, int]]:
    """Return contiguous, lossless ranges with at most one element of skew.

    The range for rank ``i`` is
    ``[floor(i*length/parts), floor((i+1)*length/parts))``.  This is the
    standard block partition used by model-parallel runtimes: every element is
    assigned exactly once and ``max(range)-min(range) <= 1``.  In particular,
    it avoids the common ``length // parts`` bug that silently drops remainder
    rows when a hidden/intermediate dimension is not divisible by GPU count.

    The invariant is the same equal-work objective used by tensor model
    parallelism in Megatron-LM (Shoeybi et al., 2019), while allowing real
    model dimensions that are not divisible by the device count.
    """
    if length < 0 or parts < 1:
        raise ValueError("length must be non-negative and parts must be positive")
    return [
        ((index * length) // parts, ((index + 1) * length) // parts)
        for index in range(parts)
    ]


@dataclass
class TensorShard:
    """Description of a single shard of a tensor."""

    device_id: int
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    axis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "shape": list(self.shape),
            "offset": list(self.offset),
            "axis": self.axis,
        }


@dataclass
class ShardingSpec:
    """Full sharding specification for a tensor."""

    tensor_name: str
    global_shape: tuple[int, ...]
    axis: str
    num_shards: int
    shards: list[TensorShard] = field(default_factory=list)
    requires_all_gather: bool = False
    requires_all_reduce: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "global_shape": list(self.global_shape),
            "axis": self.axis,
            "num_shards": self.num_shards,
            "shards": [s.to_dict() for s in self.shards],
            "requires_all_gather": self.requires_all_gather,
            "requires_all_reduce": self.requires_all_reduce,
        }


class ShardingStrategy:
    """Computes sharding annotations for common transformer tensors."""

    def __init__(self, tensor_parallel_degree: int) -> None:
        self.tp_degree = tensor_parallel_degree

    def shard_linear_weight(self, tensor_name: str, weight_shape: tuple[int, ...]) -> ShardingSpec:
        """Shard a 2D linear weight matrix column-wise."""
        if len(weight_shape) != 2:
            return ShardingSpec(
                tensor_name=tensor_name,
                global_shape=weight_shape,
                axis=ShardingAxis.REPLICATED,
                num_shards=1,
            )
        out_dim, in_dim = weight_shape
        shards: list[TensorShard] = []
        for i, (start, end) in enumerate(balanced_partition(out_dim, self.tp_degree)):
            shards.append(
                TensorShard(
                    device_id=i,
                    shape=(end - start, in_dim),
                    offset=(start, 0),
                    axis=ShardingAxis.COLUMN,
                )
            )
        return ShardingSpec(
            tensor_name=tensor_name,
            global_shape=weight_shape,
            axis=ShardingAxis.COLUMN,
            num_shards=self.tp_degree,
            shards=shards,
            requires_all_reduce=True,
        )

    def shard_attention_heads(self, tensor_name: str, num_heads: int, head_dim: int, batch_seq: tuple[int, int]) -> ShardingSpec:
        """Shard attention heads across the tensor parallel group."""
        shards: list[TensorShard] = []
        for i, (start, end) in enumerate(balanced_partition(num_heads, self.tp_degree)):
            shards.append(
                TensorShard(
                    device_id=i,
                    shape=(batch_seq[0], batch_seq[1], end - start, head_dim),
                    offset=(0, 0, start, 0),
                    axis=ShardingAxis.HEAD,
                )
            )
        return ShardingSpec(
            tensor_name=tensor_name,
            global_shape=(batch_seq[0], batch_seq[1], num_heads, head_dim),
            axis=ShardingAxis.HEAD,
            num_shards=self.tp_degree,
            shards=shards,
            requires_all_gather=True,
        )

    def shard_kv_cache(self, tensor_name: str, num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int) -> ShardingSpec:
        """Shard KV cache heads across the tensor parallel group."""
        shards: list[TensorShard] = []
        for i, (start, end) in enumerate(balanced_partition(num_kv_heads, self.tp_degree)):
            shards.append(
                TensorShard(
                    device_id=i,
                    shape=(num_layers, 2, seq_len, end - start, head_dim),
                    offset=(0, 0, 0, start, 0),
                    axis=ShardingAxis.HEAD,
                )
            )
        return ShardingSpec(
            tensor_name=tensor_name,
            global_shape=(num_layers, 2, seq_len, num_kv_heads, head_dim),
            axis=ShardingAxis.HEAD,
            num_shards=self.tp_degree,
            shards=shards,
            requires_all_gather=True,
        )

    def __repr__(self) -> str:
        return f"ShardingStrategy(tp_degree={self.tp_degree})"
