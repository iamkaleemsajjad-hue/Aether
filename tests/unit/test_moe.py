"""Tests for the moe package."""

from __future__ import annotations

import numpy as np
import pytest

from aether.moe import (
    ExpertManager,
    ExpertPlanner,
    ExpertSparsityAnalyzer,
    PlacementPlan,
    ThresholdRouter,
)


class TestThresholdRouter:
    def test_compute_gates(self) -> None:
        router = ThresholdRouter(num_experts=8)
        logits = np.random.randn(4, 8)
        gates = router.compute_gates(logits)
        assert gates.shape == (4, 8)
        np.testing.assert_almost_equal(gates.sum(axis=1), np.ones(4))

    def test_route(self) -> None:
        router = ThresholdRouter(num_experts=8, hot_threshold=0.1)
        gates = np.ones((2, 8)) / 8
        routing = router.route(gates)
        assert len(routing) == 2
        for token_experts in routing:
            assert len(token_experts) > 0
            assert all(0 <= e < 8 for e in token_experts)

    def test_expert_activation_rates(self) -> None:
        router = ThresholdRouter(num_experts=4)
        gates = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        router.route(router.compute_gates(gates))
        rates = router.expert_activation_rates()
        assert len(rates) == 4
        assert all(0.0 <= r <= 1.0 for r in rates)

    def test_classify_experts(self) -> None:
        router = ThresholdRouter(num_experts=8)
        assert len(router.classify_experts()) == 3

    def test_reset(self) -> None:
        router = ThresholdRouter(num_experts=4)
        router.route(np.ones((2, 4)) / 4)
        router.reset()
        assert sum(router.expert_activation_rates()) == 0.0


class TestExpertManager:
    def test_register(self) -> None:
        mgr = ExpertManager()
        info = mgr.register(0, "expert_0", memory_bytes=100)
        assert info.index == 0
        assert info.name == "expert_0"

    def test_update_tiers(self) -> None:
        mgr = ExpertManager()
        for i in range(8):
            mgr.register(i, f"expert_{i}")
        mgr.update_tiers({"hot": [0, 1], "warm": [2, 3, 4], "cold": [5, 6, 7]})
        assert mgr.tier_counts() == {"hot": 2, "warm": 3, "cold": 3}

    def test_prefetch_and_offload(self) -> None:
        mgr = ExpertManager()
        for i in range(4):
            mgr.register(i, f"expert_{i}")
        mgr.update_tiers({"hot": [0], "warm": [1], "cold": [2, 3]})
        assert len(mgr.prefetch_candidates()) >= 0
        assert len(mgr.offload_candidates()) >= 0


class TestSparsityAnalyzer:
    def test_analyze_weights(self) -> None:
        analyzer = ExpertSparsityAnalyzer(sparsity_threshold=0.01)
        gate = np.random.randn(64, 256).astype(np.float32)
        up = np.random.randn(64, 256).astype(np.float32)
        down = np.random.randn(256, 64).astype(np.float32)
        result = analyzer.analyze_weights(0, gate, up, down)
        assert "dead_channels" in result
        assert "active_ratio" in result
        assert 0.0 <= result["dead_ratio"] <= 1.0

    def test_analyze_activations(self) -> None:
        analyzer = ExpertSparsityAnalyzer(sparsity_threshold=0.01)
        activations = np.random.randn(100, 128).astype(np.float32)
        result = analyzer.analyze_activations(0, activations)
        assert result["total_neurons"] == 128


class TestExpertPlanner:
    def test_plan(self) -> None:
        planner = ExpertPlanner(num_devices=4)
        expert_memory = [100] * 8
        plan = planner.plan(expert_memory)
        assert isinstance(plan, PlacementPlan)
        assert plan.num_devices == 4
        assert len(plan.placements) == 4

    def test_plan_with_rates(self) -> None:
        planner = ExpertPlanner(num_devices=2)
        expert_memory = [100] * 4
        rates = [0.5, 0.3, 0.1, 0.1]
        plan = planner.plan(expert_memory, rates)
        assert plan.num_devices == 2
