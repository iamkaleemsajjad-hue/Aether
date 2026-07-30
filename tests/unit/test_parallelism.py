"""Tests for the parallelism package."""

from __future__ import annotations

import pytest

from aether.core.types import ModelArchitecture
from aether.parallelism import CollectiveBackend, DeviceMesh, ParallelismPlanner, ShardingStrategy


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
    def test_shard_linear_weight(self) -> None:
        strategy = ShardingStrategy(tensor_parallel_degree=4)
        spec = strategy.shard_linear_weight("q_proj", (4096, 4096))
        assert spec.num_shards == 4
        assert len(spec.shards) == 4
        assert spec.requires_all_reduce is True

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
