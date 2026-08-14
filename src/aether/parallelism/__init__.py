"""
Parallelism package initialization.
"""

from __future__ import annotations

from aether.parallelism.mesh import DeviceMesh
from aether.parallelism.sharding import ShardingStrategy, ShardingSpec, TensorShard, ShardingAxis
from aether.parallelism.planner import ParallelismPlanner, ParallelismConfig
from aether.parallelism.distributed import CollectiveBackend, CommunicationGroup
from aether.parallelism.collective_backends import (
    CollectiveBackendError,
    NCCLCollectiveBackend,
    PlaceholderCollectiveBackend,
    RCCLCollectiveBackend,
    SocketCollectiveBackend,
    get_collective_backend,
)


__all__ = [
    "DeviceMesh",
    "ShardingStrategy",
    "ShardingSpec",
    "TensorShard",
    "ShardingAxis",
    "ParallelismPlanner",
    "ParallelismConfig",
    "CollectiveBackend",
    "CommunicationGroup",
]
