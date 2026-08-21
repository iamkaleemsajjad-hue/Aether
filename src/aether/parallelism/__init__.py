"""
Parallelism package initialization.
"""

from __future__ import annotations

from aether.parallelism.mesh import DeviceMesh
from aether.parallelism.sharding import (
    DeviceCapacity,
    ShardingStrategy,
    ShardingSpec,
    TensorShard,
    ShardingAxis,
    capacity_weighted_partition,
)
from aether.parallelism.planner import (
    HeterogeneousShardingPlan,
    ParallelismPlanner,
    ParallelismConfig,
)
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
    "DeviceCapacity",
    "capacity_weighted_partition",
    "HeterogeneousShardingPlan",
    "ShardingStrategy",
    "ShardingSpec",
    "TensorShard",
    "ShardingAxis",
    "ParallelismPlanner",
    "ParallelismConfig",
    "CollectiveBackend",
    "CommunicationGroup",
]
