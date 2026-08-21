"""Tests for the parallelism package."""

from __future__ import annotations

import pytest

from aether.core.types import ModelArchitecture
from aether.parallelism import (
    CollectiveBackend,
    DeviceCapacity,
    DeviceMesh,
    ParallelismPlanner,
    ShardingStrategy,
    capacity_weighted_partition,
)
from aether.parallelism.sharding import balanced_partition
from aether.runtime.config import RuntimeConfig


class TestDeviceMesh:
    def test_create_mesh(self) -> None:
        mesh = DeviceMesh(shape=(2, 4))
        assert len(mesh.device_ids) == 8

    def test_get_coords(self) -> None:
        mesh = DeviceMesh(shape=(2, 4))
        coords = mesh.get_coords(0)
        assert len(coords) == 2

    def test_get_group_along_axis(self) -> None:
        mesh = DeviceMesh(shape=(2, 4))
        group = mesh.get_group_along_axis(1, (0, 0))
        assert len(group) == 4


class TestShardingStrategy:
    def test_capacity_partition_is_lossless_and_weighted(self) -> None:
        ranges = capacity_weighted_partition(100, [1.0, 3.0])
        assert ranges == [(0, 25), (25, 100)]
        assert sum(end - start for start, end in ranges) == 100

    def test_partition_is_lossless_and_balanced_for_remainders(self) -> None:
        ranges = balanced_partition(10, 3)
        assert ranges == [(0, 3), (3, 6), (6, 10)]
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 10
        assert sum(end - start for start, end in ranges) == 10
        assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1

    def test_shard_linear_weight(self) -> None:
        strategy = ShardingStrategy(tensor_parallel_degree=4)
        spec = strategy.shard_linear_weight("q_proj", (4096, 4096))
        assert spec.num_shards == 4
        assert len(spec.shards) == 4
        assert spec.requires_all_reduce is True

    def test_shard_linear_weight_does_not_drop_remainder(self) -> None:
        spec = ShardingStrategy(tensor_parallel_degree=3).shard_linear_weight("q_proj", (10, 4))
        assert [shard.shape[0] for shard in spec.shards] == [3, 3, 4]
        assert [shard.offset[0] for shard in spec.shards] == [0, 3, 6]

    def test_shard_attention_heads(self) -> None:
        strategy = ShardingStrategy(tensor_parallel_degree=2)
        spec = strategy.shard_attention_heads("attn_out", 32, 128, (1, 1024))
        assert spec.num_shards == 2
        assert spec.requires_all_gather is True

    def test_kv_cache_shard(self) -> None:
        strategy = ShardingStrategy(tensor_parallel_degree=2)
        spec = strategy.shard_kv_cache("kv_cache", 32, 8, 128, 4096)
        assert spec.num_shards == 2
        assert spec.requires_all_gather is True


class TestParallelismPlanner:
    def test_heterogeneous_plan_has_one_model_copy(self) -> None:
        arch = ModelArchitecture(
            family="llama_family", params_billion=1.0, layers=4,
            hidden_size=128, num_attention_heads=8, num_kv_heads=2,
        )
        devices = [
            DeviceCapacity("cuda:0", "gpu", compute_units=10.0, bandwidth_gbps=900.0),
            DeviceCapacity("cuda:1", "gpu", compute_units=10.0, bandwidth_gbps=900.0),
            DeviceCapacity("cpu", "cpu", compute_units=1.0, bandwidth_gbps=50.0),
        ]
        plan = ParallelismPlanner(arch).plan_for_devices(devices)
        assert plan.model_copies == 1
        assert sum(plan.weight_fractions.values()) == pytest.approx(1.0)
        assert plan.weight_fractions["cpu"] < plan.weight_fractions["cuda:0"]
        assert plan.to_dict()["invariant"].startswith("each weight element")

    def test_generate_plans(self) -> None:
        arch = ModelArchitecture(
            family="llama_family",
            params_billion=70.0,
            layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_kv_heads=8,
        )
        planner = ParallelismPlanner(arch)
        plans = planner.generate_plans(max_gpus=4)
        assert 1 in plans
        assert 2 in plans
        assert 4 in plans

    def test_plan_decode(self) -> None:
        arch = ModelArchitecture(family="test", layers=8, hidden_size=1024, num_attention_heads=8, params_billion=0.1)
        planner = ParallelismPlanner(arch)
        plan = planner.plan_for_gpus(4, phase="decode")
        assert plan.num_gpus == 4
        assert plan.phase == "decode"
        assert plan.tensor_parallel_degree == 4
        assert plan.pipeline_stages == 1
        assert plan.expert_parallel_degree == 1
        assert plan.context_parallel_degree == 1

    def test_generate_plans_includes_non_power_of_two_gpu_counts(self) -> None:
        arch = ModelArchitecture(family="test", layers=8, hidden_size=1024, num_attention_heads=8, params_billion=0.1)
        plans = ParallelismPlanner(arch).generate_plans(max_gpus=3)
        assert set(plans) == {1, 2, 3}

    def test_explicit_heterogeneous_mesh_round_trips_in_runtime_config(self) -> None:
        config = RuntimeConfig(execution_devices=["cpu", "cuda:0", "cuda:1"])
        restored = RuntimeConfig.from_dict(config.to_dict())
        assert restored.execution_devices == ["cpu", "cuda:0", "cuda:1"]


class TestCollectiveBackend:
    def test_all_reduce(self) -> None:
        import numpy as np
        backend = CollectiveBackend([0, 1, 2, 3])
        t = np.array([1.0, 2.0, 3.0])
        result = backend.all_reduce(t)
        assert result is not None

    def test_broadcast(self) -> None:
        import numpy as np
        backend = CollectiveBackend([0, 1])
        t = np.array([1.0, 2.0])
        result = backend.broadcast(t)
        assert result is not None
