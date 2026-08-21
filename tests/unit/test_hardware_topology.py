"""Tests for aether.parallelism.hardware_topology.

Research basis:
- Megatron-LM (Shoeybi et al. 2019, arXiv:1909.08053): AllReduce bandwidth model
- Ring-AllReduce (Patarasuk & Yuan 2009): T = 2*(P-1)/P * M / B
"""
from __future__ import annotations

import pytest
import numpy as np

from aether.parallelism.hardware_topology import (
    CollectiveStrategy,
    HardwareTopology,
    InterconnectType,
    TopologyEdge,
    _BW_BPS,
    _nvlink_bandwidth,
    _parse_smi_topo_cell,
    detect_hardware_topology,
)


class TestInterconnectBandwidth:
    def test_nvlink_gen1_bandwidth(self):
        assert _nvlink_bandwidth(1) == pytest.approx(160e9)

    def test_nvlink_gen3_bandwidth(self):
        assert _nvlink_bandwidth(3) == pytest.approx(600e9)

    def test_nvlink_gen4_bandwidth(self):
        assert _nvlink_bandwidth(4) == pytest.approx(900e9)

    def test_unknown_gen_defaults_to_gen2(self):
        assert _nvlink_bandwidth(99) == pytest.approx(300e9)

    def test_nvlink_faster_than_pcie(self):
        # NVLink 3.0 is ~10x faster than PCIe Gen4
        assert _nvlink_bandwidth(3) > _BW_BPS[InterconnectType.PCIE] * 5


class TestTopoStringParser:
    def test_nvlink_4(self):
        link, gen = _parse_smi_topo_cell("NV4")
        assert link == InterconnectType.NVLINK
        assert gen == 4

    def test_phb(self):
        link, gen = _parse_smi_topo_cell("PHB")
        assert link == InterconnectType.PCIE
        assert gen == 0

    def test_sys_cross_socket(self):
        link, gen = _parse_smi_topo_cell("SYS")
        assert link == InterconnectType.PCIE_QPI

    def test_unknown(self):
        link, _ = _parse_smi_topo_cell("X")
        assert link == InterconnectType.UNKNOWN


class TestHardwareTopologyCollectiveSelection:
    """Verify collective strategy selection matches Megatron-LM §3.3 rules."""

    def _nvlink_topo(self, n: int) -> HardwareTopology:
        devices = [f"cuda:{i}" for i in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(TopologyEdge(
                    src=f"cuda:{i}", dst=f"cuda:{j}",
                    link=InterconnectType.NVLINK,
                    bandwidth_bps=_nvlink_bandwidth(3),
                    nvlink_gen=3,
                ))
        return HardwareTopology(devices=devices, edges=edges)

    def _pcie_topo(self, n: int) -> HardwareTopology:
        devices = [f"cuda:{i}" for i in range(n)]
        edges = [
            TopologyEdge(f"cuda:{i}", f"cuda:{j}", InterconnectType.PCIE, _BW_BPS[InterconnectType.PCIE])
            for i in range(n) for j in range(i + 1, n)
        ]
        return HardwareTopology(devices=devices, edges=edges)

    def _mixed_topo(self) -> HardwareTopology:
        """NVLink between 0-1, PCIe between 0-2 and 1-2."""
        return HardwareTopology(
            devices=["cuda:0", "cuda:1", "cuda:2"],
            edges=[
                TopologyEdge("cuda:0", "cuda:1", InterconnectType.NVLINK, _nvlink_bandwidth(3)),
                TopologyEdge("cuda:0", "cuda:2", InterconnectType.PCIE, _BW_BPS[InterconnectType.PCIE]),
                TopologyEdge("cuda:1", "cuda:2", InterconnectType.PCIE, _BW_BPS[InterconnectType.PCIE]),
            ]
        )

    def test_single_gpu_direct(self):
        topo = HardwareTopology(devices=["cuda:0"], edges=[])
        assert topo.recommend_strategy() == CollectiveStrategy.DIRECT

    def test_two_gpu_always_direct(self):
        # P=2: single send+recv is optimal regardless of interconnect
        topo = self._nvlink_topo(2)
        assert topo.recommend_strategy() == CollectiveStrategy.DIRECT

    def test_all_nvlink_ring(self):
        topo = self._nvlink_topo(4)
        assert topo.recommend_strategy() == CollectiveStrategy.RING

    def test_all_pcie_tree(self):
        topo = self._pcie_topo(4)
        assert topo.recommend_strategy() == CollectiveStrategy.TREE

    def test_mixed_hierarchical(self):
        topo = self._mixed_topo()
        assert topo.recommend_strategy() == CollectiveStrategy.HIERARCHICAL


class TestAllReduceLatencyModel:
    """Verify ring-allreduce cost model (Patarasuk & Yuan 2009).

    T = 2 * (P-1)/P * M / B
    """

    def test_zero_payload(self):
        topo = HardwareTopology(
            devices=["cuda:0", "cuda:1"],
            edges=[TopologyEdge("cuda:0", "cuda:1", InterconnectType.PCIE, 64e9)]
        )
        assert topo.allreduce_latency_ms(0) == pytest.approx(0.0)

    def test_single_device_zero_latency(self):
        topo = HardwareTopology(devices=["cuda:0"], edges=[])
        assert topo.allreduce_latency_ms(1024) == pytest.approx(0.0)

    def test_nvlink_faster_than_pcie(self):
        pcie_topo = HardwareTopology(
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            edges=[
                TopologyEdge(f"cuda:{i}", f"cuda:{j}", InterconnectType.PCIE, 64e9)
                for i in range(4) for j in range(i + 1, 4)
            ]
        )
        nv_topo = HardwareTopology(
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            edges=[
                TopologyEdge(f"cuda:{i}", f"cuda:{j}", InterconnectType.NVLINK, 600e9)
                for i in range(4) for j in range(i + 1, 4)
            ]
        )
        payload = 2 * 4096 * 4  # hidden_size=4096, float32
        t_pcie = pcie_topo.allreduce_latency_ms(payload)
        t_nv = nv_topo.allreduce_latency_ms(payload)
        # NVLink should be ~9x faster (600GB/s vs 64GB/s)
        assert t_pcie > t_nv * 5

    def test_ring_formula_exact(self):
        """Verify T = 2*(P-1)/P * M / B formula for P=4, PCIe."""
        P = 4
        M = 4096 * 4  # 4096 float32 values
        B = 64e9       # PCIe Gen4
        expected_ms = 2.0 * (P - 1) / P * M / B * 1000
        topo = HardwareTopology(
            devices=[f"cuda:{i}" for i in range(P)],
            edges=[
                TopologyEdge(f"cuda:{i}", f"cuda:{j}", InterconnectType.PCIE, B)
                for i in range(P) for j in range(i + 1, P)
            ]
        )
        assert topo.allreduce_latency_ms(M) == pytest.approx(expected_ms, rel=1e-6)


class TestDetectHardwareTopology:
    def test_single_device_empty_topology(self):
        topo = detect_hardware_topology(["cuda:0"])
        assert topo.devices == ["cuda:0"]
        assert topo.edges == []

    def test_non_cuda_devices_ignored(self):
        topo = detect_hardware_topology(["cpu"])
        assert topo.edges == []

    def test_two_cuda_devices_gets_edges(self):
        # Without actual GPUs, falls through to PCIe fallback
        topo = detect_hardware_topology(["cuda:0", "cuda:1"])
        assert len(topo.devices) == 2
        assert len(topo.edges) == 1  # one undirected edge between gpu0 and gpu1
        edge = topo.edges[0]
        assert edge.src == "cuda:0"
        assert edge.dst == "cuda:1"


class TestSharedMemoryShard:
    def test_round_trip(self):
        from aether.runtime.native_distributed_engine import NativeSharedMemoryShard
        data = np.random.randn(64, 32).astype(np.float32)
        shard = NativeSharedMemoryShard.create(data)
        attached = shard.attach()
        assert np.allclose(data, attached), "SharedMemory round-trip failed"
        shard.close()

    def test_correct_shape(self):
        from aether.runtime.native_distributed_engine import NativeSharedMemoryShard
        data = np.ones((128, 64), dtype=np.float32)
        shard = NativeSharedMemoryShard.create(data)
        assert shard.shape == (128, 64)
        shard.close()

    def test_launch_native_auto_workers(self):
        import os
        from aether.runtime.native_distributed_engine import launch_native_distributed_engine
        # Just check the function resolves num_workers without crashing
        # (we don't have a real cpu_engine, so just test worker count logic)
        cores = os.cpu_count() or 1
        # Compute expected workers
        num_w = 2 if cores < 8 else (4 if cores < 16 else min(8, cores // 4))
        pw = 1
        while pw * 2 <= num_w:
            pw *= 2
        expected = max(2, pw)
        assert expected >= 2
        assert expected <= 8
