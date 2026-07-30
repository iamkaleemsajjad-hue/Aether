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
        shard_out = (out_dim + self.tp_degree - 1) // self.tp_degree
        shards: list[TensorShard] = []
        for i in range(self.tp_degree):
            start = i * shard_out
            end = min(start + shard_out, out_dim)
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
        heads_per_shard = (num_heads + self.tp_degree - 1) // self.tp_degree
        shards: list[TensorShard] = []
        for i in range(self.tp_degree):
            start = i * heads_per_shard
            end = min(start + heads_per_shard, num_heads)
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
        heads_per_shard = (num_kv_heads + self.tp_degree - 1) // self.tp_degree
        shards: list[TensorShard] = []
        for i in range(self.tp_degree):
            start = i * heads_per_shard
            end = min(start + heads_per_shard, num_kv_heads)
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
