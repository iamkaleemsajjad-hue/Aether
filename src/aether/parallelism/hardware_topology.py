"""
Hardware topology detection and optimal collective strategy selection.

Determines the physical interconnect topology between GPUs and recommends
the most efficient AllReduce algorithm for tensor-parallel inference.

Research basis
--------------
* **Megatron-LM** (Shoeybi et al. 2019, arXiv:1909.08053): Tensor and pipeline
  parallelism in transformer training. Section 3 establishes that AllReduce
  bandwidth is the dominant cost in TP; interconnect topology drives algorithm choice.
* **Ring-AllReduce** (Patarasuk & Yuan 2009, JPDC; popularized by Baidu Research 2017):
  Bandwidth-optimal for homogeneous rings: cost = 2*(P-1)/P * M / B.
* **Tree-AllReduce / Recursive halving** (Rabenseifner 2004, EuroPVM/MPI):
  Latency-optimal: O(log2(P)) steps. Better for small messages or mixed bandwidths.
* **NVLink bandwidth** (NVIDIA 2024): NVLink 4.0 = 900 GB/s (H100 SXM),
  NVLink 3.0 = 600 GB/s (A100); PCIe Gen4 x16 = ~64 GB/s bidirectional.
* **Two-level hierarchical AllReduce** (NCCL 2016+): NVLink ring intra-node,
  PCIe/InfiniBand tree inter-node. Used when topology is mixed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "InterconnectType",
    "CollectiveStrategy",
    "TopologyEdge",
    "HardwareTopology",
    "detect_hardware_topology",
]


class InterconnectType(Enum):
    """Physical interconnect between two GPUs."""
    NVLINK = auto()      # NVIDIA NVLink ~300-900 GB/s
    PCIE = auto()        # PCIe Gen4 x16 ~64 GB/s
    PCIE_QPI = auto()    # PCIe cross-socket through UPI ~40 GB/s
    XGMI = auto()        # AMD Infinity Fabric / XGMI ~92-192 GB/s
    UNKNOWN = auto()


class CollectiveStrategy(Enum):
    """Recommended AllReduce algorithm for this topology."""
    RING = "ring"
    """Ring-AllReduce: bandwidth-optimal for homogeneous NVLink meshes.
    Cost: 2*(P-1)/P * M / B.  Reference: Patarasuk & Yuan 2009."""
    TREE = "tree"
    """Recursive-halving tree: latency-optimal for slow/mixed links.
    O(log2(P)) steps.  Reference: Rabenseifner 2004."""
    DIRECT = "direct"
    """Pair-to-pair for P=2 — single send+recv, no coordination overhead."""
    HIERARCHICAL = "hierarchical"
    """Two-level: NVLink ring within node + PCIe tree between nodes.
    Used when some pairs have NVLink and others have PCIe.
    Reference: NCCL design docs (2016+)."""


# Approximate peak bandwidths by interconnect type (bytes/second)
# Source: NVIDIA/AMD white papers and MLSys 2023 benchmarks
_BW_BPS: dict[InterconnectType, float] = {
    InterconnectType.NVLINK:   900e9,   # NVLink 4.0 H100
    InterconnectType.XGMI:     192e9,   # AMD CDNA2 Infinity Fabric
    InterconnectType.PCIE:      64e9,   # PCIe Gen4 x16 bidirectional
    InterconnectType.PCIE_QPI:  40e9,   # Cross-socket through UPI
    InterconnectType.UNKNOWN:   16e9,   # Conservative fallback
}


@dataclass(frozen=True)
class TopologyEdge:
    """Undirected edge in the GPU interconnect graph."""
    src: str
    dst: str
    link: InterconnectType
    bandwidth_bps: float
    nvlink_gen: int = 0   # NVLink generation (0 = not NVLink)

    @property
    def latency_us_per_gb(self) -> float:
        return 1e15 / self.bandwidth_bps

    def __str__(self) -> str:
        return f"{self.src}<->{self.dst}[{self.link.name} {self.bandwidth_bps/1e9:.0f}GB/s]"


@dataclass
class HardwareTopology:
    """Weighted undirected graph of GPU interconnects.

    Used by the tensor-parallel engine to select the optimal AllReduce
    collective (Megatron-LM section 3.3, Shoeybi et al. 2019).
    """
    devices: list[str] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)

    def recommend_strategy(self) -> CollectiveStrategy:
        """Select the optimal AllReduce algorithm for this topology.

        Decision rule (from Megatron-LM section 3.3 and NCCL internals):
          P <= 2         -> DIRECT   (no overhead for trivial case)
          All NVLink/XGMI -> RING   (bandwidth-optimal, equal links)
          Mixed          -> HIERARCHICAL (two-level)
          All PCIe       -> TREE    (latency-optimal for slow links)
        """
        n = len(self.devices)
        if n <= 2:
            return CollectiveStrategy.DIRECT

        link_types = {e.link for e in self.edges}
        fast = {InterconnectType.NVLINK, InterconnectType.XGMI}
        all_fast = link_types.issubset(fast)
        any_fast = bool(link_types & fast)

        if all_fast:
            return CollectiveStrategy.RING
        if any_fast:
            return CollectiveStrategy.HIERARCHICAL
        return CollectiveStrategy.TREE

    def min_bandwidth_bps(self) -> float:
        """Bottleneck bandwidth on the critical AllReduce path."""
        if not self.edges:
            return _BW_BPS[InterconnectType.UNKNOWN]
        return min(e.bandwidth_bps for e in self.edges)

    def allreduce_latency_ms(self, payload_bytes: int) -> float:
        """Estimated AllReduce latency (ms) using the ring-allreduce model.

        Formula (Patarasuk & Yuan 2009):
            T = 2 * (P-1)/P * M / B
        where P = number of ranks, M = payload bytes, B = bottleneck bandwidth.
        """
        p = len(self.devices)
        if p <= 1:
            return 0.0
        b = self.min_bandwidth_bps()
        return 2.0 * (p - 1) / p * payload_bytes / b * 1000

    def throughput_ratio_vs_pcie(self) -> float:
        """Speedup relative to hypothetical all-PCIe baseline."""
        return self.min_bandwidth_bps() / _BW_BPS[InterconnectType.PCIE]

    def summary(self) -> str:
        strat = self.recommend_strategy()
        bw_gb = self.min_bandwidth_bps() / 1e9
        return (
            f"HardwareTopology({len(self.devices)} GPUs, "
            f"strategy={strat.value}, bottleneck={bw_gb:.0f}GB/s)"
        )


# ── Detection pipeline ────────────────────────────────────────────────────

def _nvlink_bandwidth(gen: int) -> float:
    """NVLink peak bandwidth by generation (bidirectional, bytes/sec).

    Source: NVIDIA GPU interconnect white papers (2016-2024).
    Gen1 (Pascal 2016): 160 GB/s  Gen2 (Volta 2018): 300 GB/s
    Gen3 (Ampere 2020): 600 GB/s  Gen4 (Hopper 2022): 900 GB/s
    """
    return {1: 160e9, 2: 300e9, 3: 600e9, 4: 900e9}.get(gen, 300e9)


def _parse_smi_topo_cell(cell: str) -> tuple[InterconnectType, int]:
    """Convert nvidia-smi topo code to (link type, nvlink_gen).

    Codes:
      NVx  -> NVLink generation x
      PHB/PIX -> same PCIe host bridge
      NODE -> different PCIe root complex (NUMA)
      SYS  -> cross-socket QPI/UPI
    """
    t = cell.strip().upper()
    if t.startswith("NV"):
        try:
            gen = int(t[2:])
        except ValueError:
            gen = 1
        return InterconnectType.NVLINK, gen
    if t in ("PHB", "PIX", "PXB"):
        return InterconnectType.PCIE, 0
    if t == "NODE":
        return InterconnectType.PCIE, 0
    if t == "SYS":
        return InterconnectType.PCIE_QPI, 0
    return InterconnectType.UNKNOWN, 0


def _step1_nvidia_smi(cuda_ids: list[int]) -> list[TopologyEdge] | None:
    """Parse nvidia-smi topo -m for per-pair interconnect type."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().splitlines()
        # Parse GPU rows: "GPU0  X   NV4  PHB  ..."
        gpu_rows: list[tuple[int, list[str]]] = []
        for line in lines:
            if not line.startswith("GPU"):
                continue
            parts = line.split()
            try:
                idx = int(parts[0][3:])
            except (IndexError, ValueError):
                continue
            gpu_rows.append((idx, parts[1:]))

        edges: list[TopologyEdge] = []
        num_gpus = len(gpu_rows)
        for row_pos, (src, cells) in enumerate(gpu_rows):
            for col_pos in range(row_pos + 1, num_gpus):
                if col_pos >= len(cells):
                    break
                dst = gpu_rows[col_pos][0]
                if src not in cuda_ids or dst not in cuda_ids:
                    continue
                cell = cells[col_pos]
                link, gen = _parse_smi_topo_cell(cell)
                bw = _nvlink_bandwidth(gen) if link == InterconnectType.NVLINK else _BW_BPS[link]
                edges.append(TopologyEdge(
                    src=f"cuda:{src}", dst=f"cuda:{dst}",
                    link=link, bandwidth_bps=bw, nvlink_gen=gen,
                ))
        return edges if edges else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _step2_pynvml(cuda_ids: list[int]) -> list[TopologyEdge] | None:
    """Use pynvml P2P NVLink capability matrix as a fallback."""
    try:
        try:
            import warnings as _w; _w.filterwarnings('ignore', category=FutureWarning, module='pynvml'); import pynvml as nvml
        except ImportError:
            import nvidia.ml as nvml  # type: ignore[no-redef]
        nvml.nvmlInit()
        n_total = nvml.nvmlDeviceGetCount()
        handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_total)]
        edges: list[TopologyEdge] = []
        for i_pos, src in enumerate(cuda_ids):
            for dst in cuda_ids[i_pos + 1:]:
                if src >= n_total or dst >= n_total:
                    continue
                try:
                    status = nvml.nvmlDeviceGetP2PStatus(
                        handles[src], handles[dst],
                        nvml.NVML_P2P_CAPS_INDEX_NVLINK,
                    )
                    has_nvlink = (status == nvml.NVML_P2P_STATUS_OK)
                except Exception:  # noqa: BLE001
                    has_nvlink = False
                link = InterconnectType.NVLINK if has_nvlink else InterconnectType.PCIE
                edges.append(TopologyEdge(
                    src=f"cuda:{src}", dst=f"cuda:{dst}",
                    link=link, bandwidth_bps=_BW_BPS[link],
                ))
        nvml.nvmlShutdown()
        return edges if edges else None
    except Exception:  # noqa: BLE001
        return None


def _step3_rocm_xgmi(cuda_ids: list[int]) -> list[TopologyEdge] | None:
    """Detect AMD XGMI / Infinity Fabric using rocm-smi.

    XGMI is AMD's die-to-die interconnect on multi-chip module GPUs.
    Reference: AMD CDNA2 Architecture White Paper (2021).
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtoponuma"],
            capture_output=True, text=True, timeout=5,
        )
        has_xgmi = "xgmi" in result.stdout.lower() or "infinity" in result.stdout.lower()
        if not has_xgmi:
            return None
        edges = []
        for i_pos, src in enumerate(cuda_ids):
            for dst in cuda_ids[i_pos + 1:]:
                edges.append(TopologyEdge(
                    src=f"cuda:{src}", dst=f"cuda:{dst}",
                    link=InterconnectType.XGMI,
                    bandwidth_bps=_BW_BPS[InterconnectType.XGMI],
                ))
        return edges if edges else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _step4_pcie_fallback(cuda_ids: list[int]) -> list[TopologyEdge]:
    """Conservative fallback: assume PCIe Gen4 for all pairs."""
    edges = []
    for i_pos, src in enumerate(cuda_ids):
        for dst in cuda_ids[i_pos + 1:]:
            edges.append(TopologyEdge(
                src=f"cuda:{src}", dst=f"cuda:{dst}",
                link=InterconnectType.PCIE,
                bandwidth_bps=_BW_BPS[InterconnectType.PCIE],
            ))
    return edges


def detect_hardware_topology(devices: list[str]) -> HardwareTopology:
    """Build the GPU interconnect topology graph for the given devices.

    4-step detection pipeline:
    1. nvidia-smi topo -m   (most accurate — reads NVLink routing tables)
    2. pynvml P2P matrix    (fallback via Python NVML bindings)
    3. rocm-smi             (AMD XGMI detection)
    4. PCIe fallback        (always succeeds, conservative estimate)

    Args:
        devices: Device identifiers, e.g. ``["cuda:0", "cuda:1"]``.

    Returns:
        :class:`HardwareTopology` with all detected interconnect edges and
        a recommended :class:`CollectiveStrategy` for AllReduce.
    """
    cuda_ids = [int(d.split(":", 1)[1]) for d in devices if d.startswith("cuda:")]
    topology = HardwareTopology(devices=list(devices))

    if len(cuda_ids) < 2:
        return topology

    # Step 1
    edges = _step1_nvidia_smi(cuda_ids)
    if edges:
        topology.edges = edges
        logger.info("GPU topology via nvidia-smi: %s", topology.summary())
        return topology

    # Step 2
    edges = _step2_pynvml(cuda_ids)
    if edges:
        topology.edges = edges
        logger.info("GPU topology via pynvml: %s", topology.summary())
        return topology

    # Step 3
    edges = _step3_rocm_xgmi(cuda_ids)
    if edges:
        topology.edges = edges
        logger.info("GPU topology via rocm-smi (XGMI): %s", topology.summary())
        return topology

    # Step 4
    logger.warning(
        "Could not detect GPU interconnect topology "
        "(nvidia-smi/pynvml/rocm-smi unavailable). "
        "Assuming PCIe Gen4 for all %d pairs.",
        len(cuda_ids) * (len(cuda_ids) - 1) // 2,
    )
    topology.edges = _step4_pcie_fallback(cuda_ids)
    return topology
