"""NCCL-free multi-GPU collectives over peer-to-peer device copies.

Aether's single-process tensor-parallel executor holds one shard of every weight
matrix on each device and needs a real all-reduce after every row-parallel GEMM.
This module provides that without NCCL, RCCL or ``torch.distributed``: the
collectives are built from device-to-device tensor copies, which the driver
services over NVLink or PCIe peer-to-peer when the pair supports it and stages
through pinned host memory when it does not.

Three algorithms, and the choice between them is arithmetic, not a preference.
With ``P`` devices, payload ``D`` bytes, per-hop latency ``α`` and link bandwidth
``B`` (the α–β model):

===============  ============================  ==================
Algorithm        Time                          Volume per device
===============  ============================  ==================
one-shot         ``α + (P−1)·D/B``             ``(P−1)·D``
two-shot         ``2α + 2(P−1)/P·D/B``         ``2(P−1)/P·D``
ring             ``2(P−1)·α + 2(P−1)/P·D/B``   ``2(P−1)/P·D``
===============  ============================  ==================

So:

* **one-shot** wins for small payloads, where latency dominates: every device
  reads every other device's buffer once and reduces locally.
* **two-shot** wins for large payloads on a fully peer-connected node: it moves
  the same bandwidth-optimal volume as a ring in two steps instead of
  ``2(P−1)``.  This is a reduce-scatter followed by an all-gather, both done with
  direct copies rather than hop-by-hop.
* **ring** is for topologies that are *not* fully connected — a device only talks
  to its neighbours.  It is bandwidth-optimal and latency-heavy.

The one-shot/two-shot crossover follows from setting the two times equal:

    D* = α·P·B / ((P−1)·(P−2))    for P > 2

Below ``D*`` one-shot is faster; above it two-shot is.  For ``P = 2`` one-shot is
always at least as good — the volumes are equal and it costs one hop instead of
two — so the ring degenerates correctly.

Reductions run in ascending device order in every algorithm, so a result is
reproducible run to run.  Floating-point addition is not associative, and a
collective whose summation order depends on arrival order silently produces
different logits on different runs.

References:
  * P. Patarasuk and X. Yuan, "Bandwidth optimal all-reduce algorithms for
    clusters of workstations", J. Parallel Distrib. Comput. 69(2), 2009 — the
    ring reduce-scatter + all-gather decomposition and its ``2(P−1)/P`` bound.
  * R. Thakur, R. Rabenseifner and W. Gropp, "Optimization of collective
    communication operations in MPICH", IJHPCA 19(1), 2005 — the α–β cost model
    and algorithm selection by message size.
  * M. Shoeybi et al., "Megatron-LM", arXiv:1909.08053, §3 — why the all-reduce
    after a row-parallel GEMM is the dominant cost in tensor-parallel inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from aether.parallelism.sharding import balanced_partition
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "P2PCollectiveError",
    "Algorithm",
    "LinkModel",
    "P2PRingCollective",
]


class P2PCollectiveError(RuntimeError):
    """Raised when a peer-to-peer collective cannot be performed."""


class Algorithm(str, Enum):
    """Which collective schedule to run."""

    ONE_SHOT = "one_shot"
    """Every device reads every other device's buffer, then reduces locally."""

    TWO_SHOT = "two_shot"
    """Direct-copy reduce-scatter followed by direct-copy all-gather."""

    RING = "ring"
    """Nearest-neighbour ring; for topologies without full peer connectivity."""

    AUTO = "auto"
    """Choose per call from the payload size and the measured topology."""


@dataclass(frozen=True)
class LinkModel:
    """α–β parameters of the slowest link in the device mesh.

    ``latency_s`` is the fixed per-hop cost — kernel launch plus copy setup — and
    ``bandwidth_bps`` is the sustained throughput of the bottleneck link.  Both are
    taken from the detected interconnect rather than assumed, because the crossover
    payload between one-shot and two-shot moves by two orders of magnitude between
    NVLink and PCIe.
    """

    latency_s: float = 8e-6
    """Per-hop latency. ~8 µs is a realistic device-to-device copy launch cost."""

    bandwidth_bps: float = 64e9
    """Bottleneck bandwidth. Defaults to PCIe Gen4 x16, the conservative case."""

    fully_connected: bool = True
    """Whether every device pair can copy directly (peer access or host staging)."""

    def crossover_bytes(self, devices: int) -> float:
        """Payload at which two-shot overtakes one-shot.

        Solving ``α + (P−1)D/B = 2α + 2(P−1)/P·D/B`` for ``D`` gives
        ``D* = α·P·B / ((P−1)(P−2))``.  For ``P ≤ 2`` one-shot is never worse, so
        the crossover is infinite and two-shot is never selected.
        """
        if devices <= 2:
            return float("inf")
        return (
            self.latency_s * devices * self.bandwidth_bps
            / ((devices - 1) * (devices - 2))
        )

    def one_shot_seconds(self, payload_bytes: int, devices: int) -> float:
        """Predicted one-shot all-reduce time."""
        if devices <= 1:
            return 0.0
        return self.latency_s + (devices - 1) * payload_bytes / self.bandwidth_bps

    def two_shot_seconds(self, payload_bytes: int, devices: int) -> float:
        """Predicted two-shot all-reduce time."""
        if devices <= 1:
            return 0.0
        return (
            2 * self.latency_s
            + 2 * (devices - 1) / devices * payload_bytes / self.bandwidth_bps
        )

    def ring_seconds(self, payload_bytes: int, devices: int) -> float:
        """Predicted ring all-reduce time."""
        if devices <= 1:
            return 0.0
        return (
            2 * (devices - 1) * self.latency_s
            + 2 * (devices - 1) / devices * payload_bytes / self.bandwidth_bps
        )


def _link_model_for(devices: Sequence[str]) -> LinkModel:
    """Build the α–β model from the detected interconnect topology."""
    try:
        from aether.parallelism.hardware_topology import (
            InterconnectType,
            detect_hardware_topology,
        )

        topology = detect_hardware_topology(list(devices))
        if not topology.edges:
            return LinkModel()
        bandwidth = topology.min_bandwidth_bps()
        fast = {InterconnectType.NVLINK, InterconnectType.XGMI}
        on_fast_fabric = all(edge.link in fast for edge in topology.edges)
        # An NVLink/XGMI copy launches in roughly a microsecond; a PCIe copy that
        # may stage through the host costs several. Using one number for both puts
        # the algorithm crossover in the wrong place by orders of magnitude.
        latency = 1.5e-6 if on_fast_fabric else 8e-6
        expected_pairs = len(devices) * (len(devices) - 1) // 2
        return LinkModel(
            latency_s=latency,
            bandwidth_bps=bandwidth,
            fully_connected=len(topology.edges) >= expected_pairs,
        )
    except Exception as exc:  # noqa: BLE001 - detection is advisory
        logger.debug("interconnect detection unavailable (%s); using defaults", exc)
        return LinkModel()


class P2PRingCollective:
    """Multi-GPU collectives built from device-to-device copies. No NCCL.

    Every method takes and returns a list of per-device tensors — one entry per
    device in ``devices``, all the same shape and dtype — which is how the
    tensor-parallel executor already holds its activations.

    Correctness properties worth naming:

    * reductions run in ascending device order, so results are bit-reproducible;
    * every method validates that it received one tensor per device with matching
      shapes, because a silently mismatched shard would produce a plausible tensor
      of the wrong values;
    * ``dtype`` is preserved, with half-precision accumulated in float32 and cast
      back — summing ``P`` half-precision partials in half precision loses low
      bits at every hop.
    """

    def __init__(
        self,
        devices: Sequence[str],
        *,
        torch_module: Any | None = None,
        algorithm: Algorithm | str = Algorithm.AUTO,
        link_model: LinkModel | None = None,
    ) -> None:
        if len(devices) < 1:
            raise ValueError("at least one device is required")
        if torch_module is None:
            try:
                import torch as torch_module  # type: ignore[no-redef]
            except ImportError as exc:  # pragma: no cover - guarded by callers
                raise P2PCollectiveError(
                    "peer-to-peer collectives require PyTorch for device tensors"
                ) from exc
        self.torch = torch_module
        self.devices = [self.torch.device(value) for value in devices]
        self.world_size = len(self.devices)
        self.algorithm = Algorithm(algorithm)
        self.link = link_model or _link_model_for([str(d) for d in devices])
        self._peer_access = self._probe_peer_access()
        self._streams = self._create_streams()
        self._calls = 0
        self._algorithm_counts: dict[str, int] = {}
        logger.info(
            "P2P collective ready: %d devices, peer_access=%s, bottleneck=%.0f GB/s, "
            "one-shot/two-shot crossover=%.1f KiB",
            self.world_size,
            self._peer_access,
            self.link.bandwidth_bps / 1e9,
            self.link.crossover_bytes(self.world_size) / 1024,
        )

    # ── topology probing ──────────────────────────────────────────────────────

    def _probe_peer_access(self) -> bool:
        """Whether every CUDA pair can copy directly.

        Without peer access a cross-device copy stages through host memory, which
        still works and is still correct — it is simply slower, and it makes the
        mesh effectively non-fully-connected for algorithm-selection purposes.
        """
        cuda_devices = [d for d in self.devices if d.type == "cuda"]
        if not cuda_devices:
            # Every "device" is host memory, so every pair is directly reachable.
            return True
        if len(cuda_devices) < 2:
            return self.world_size <= 1
        try:
            if not self.torch.cuda.is_available():
                return False
            for first in cuda_devices:
                for second in cuda_devices:
                    if first.index == second.index:
                        continue
                    if not self.torch.cuda.can_device_access_peer(
                        first.index, second.index
                    ):
                        return False
            return True
        except Exception as exc:  # noqa: BLE001 - treat unknown as "not peered"
            logger.debug("peer-access probe failed (%s); assuming staged copies", exc)
            return False

    def _create_streams(self) -> list[Any]:
        """One stream per device, so copies on different devices overlap.

        Issuing every copy on the default stream serializes the collective behind
        whatever compute is already queued there, which turns a parallel copy
        schedule back into a sequential one.
        """
        if not any(device.type == "cuda" for device in self.devices):
            return [None] * self.world_size
        try:
            return [
                self.torch.cuda.Stream(device=device) if device.type == "cuda" else None
                for device in self.devices
            ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not create per-device streams (%s)", exc)
            return [None] * self.world_size

    # ── validation ────────────────────────────────────────────────────────────

    def _validate(self, shards: Sequence[Any]) -> None:
        if len(shards) != self.world_size:
            raise P2PCollectiveError(
                f"expected {self.world_size} tensors, one per device, got {len(shards)}"
            )
        reference = shards[0]
        for index, tensor in enumerate(shards):
            if tuple(tensor.shape) != tuple(reference.shape):
                raise P2PCollectiveError(
                    f"shard {index} has shape {tuple(tensor.shape)}, expected "
                    f"{tuple(reference.shape)}; a collective over mismatched shards "
                    "would produce a plausible tensor of wrong values"
                )
            if tensor.dtype != reference.dtype:
                raise P2PCollectiveError(
                    f"shard {index} has dtype {tensor.dtype}, expected {reference.dtype}"
                )

    def _accumulation_dtype(self, dtype: Any) -> Any:
        """float32 for half precision, the input dtype otherwise."""
        if dtype in (self.torch.float16, self.torch.bfloat16):
            return self.torch.float32
        return dtype

    # ── algorithm selection ───────────────────────────────────────────────────

    def select_algorithm(self, payload_bytes: int) -> Algorithm:
        """Pick the schedule with the lowest predicted time for this payload.

        The decision is the α–β model, evaluated per call.  A fixed choice is wrong
        at one end or the other: a ring on a small payload pays ``2(P−1)`` launch
        latencies to save bandwidth it was never short of, and one-shot on a large
        payload moves ``P/2`` times more bytes than it needs to.
        """
        if self.world_size <= 1:
            return Algorithm.ONE_SHOT
        if self.algorithm is not Algorithm.AUTO:
            return self.algorithm
        if not (self.link.fully_connected and self._peer_access):
            # Only neighbours are cheap; the ring is the schedule that assumes that.
            return Algorithm.RING
        if payload_bytes <= self.link.crossover_bytes(self.world_size):
            return Algorithm.ONE_SHOT
        return Algorithm.TWO_SHOT

    def predicted_seconds(self, payload_bytes: int) -> dict[str, float]:
        """Predicted time for each algorithm, for benchmarking and diagnostics."""
        return {
            Algorithm.ONE_SHOT.value: self.link.one_shot_seconds(
                payload_bytes, self.world_size
            ),
            Algorithm.TWO_SHOT.value: self.link.two_shot_seconds(
                payload_bytes, self.world_size
            ),
            Algorithm.RING.value: self.link.ring_seconds(
                payload_bytes, self.world_size
            ),
        }

    # ── copy helpers ──────────────────────────────────────────────────────────

    def _to(self, tensor: Any, device: Any) -> Any:
        """Copy a tensor to ``device``, non-blocking where that is safe."""
        if tensor.device == device:
            return tensor
        return tensor.to(device, non_blocking=tensor.device.type == "cuda")

    def _synchronize(self) -> None:
        """Wait for every device's queued copies before reading results.

        Skipping this is the bug that makes a P2P collective return partially
        copied buffers under load while passing every single-threaded test.
        """
        for device in self.devices:
            if device.type == "cuda":
                try:
                    self.torch.cuda.synchronize(device)
                except Exception as exc:  # noqa: BLE001
                    raise P2PCollectiveError(
                        f"could not synchronize {device}: {exc}"
                    ) from exc

    # ── collectives ───────────────────────────────────────────────────────────

    def all_reduce(
        self,
        shards: Sequence[Any],
        op: str = "sum",
        *,
        algorithm: Algorithm | str | None = None,
    ) -> list[Any]:
        """All-reduce across devices; every device receives the full result.

        Args:
            shards: One tensor per device, identical shape and dtype.
            op: ``"sum"``, ``"avg"``, ``"max"`` or ``"min"``.
            algorithm: Overrides selection, for benchmarking one schedule.

        Returns:
            One tensor per device, each on its own device, holding the reduction.
        """
        self._validate(shards)
        if self.world_size == 1:
            return [shards[0].clone()]
        payload_bytes = shards[0].numel() * shards[0].element_size()
        chosen = (
            Algorithm(algorithm) if algorithm is not None
            else self.select_algorithm(payload_bytes)
        )
        if chosen is Algorithm.AUTO:
            chosen = self.select_algorithm(payload_bytes)
        self._calls += 1
        self._algorithm_counts[chosen.value] = (
            self._algorithm_counts.get(chosen.value, 0) + 1
        )
        if chosen is Algorithm.ONE_SHOT:
            result = self._all_reduce_one_shot(shards, op)
        elif chosen is Algorithm.TWO_SHOT:
            result = self._all_reduce_two_shot(shards, op)
        else:
            result = self._all_reduce_ring(shards, op)
        self._synchronize()
        return result

    def _combine(self, target: Any, incoming: Any, op: str) -> Any:
        if op in ("sum", "avg"):
            return target + incoming
        if op == "max":
            return self.torch.maximum(target, incoming)
        if op == "min":
            return self.torch.minimum(target, incoming)
        if op == "prod":
            return target * incoming
        raise P2PCollectiveError(f"unsupported reduction {op!r}")

    def _finalize(self, tensor: Any, op: str, dtype: Any) -> Any:
        if op == "avg":
            tensor = tensor / self.world_size
        return tensor.to(dtype) if tensor.dtype != dtype else tensor

    def _all_reduce_one_shot(self, shards: Sequence[Any], op: str) -> list[Any]:
        """Every device pulls every other buffer and reduces locally.

        ``(P-1)·D`` per device in a single logical step.

        The accumulation starts from device 0 and proceeds in ascending order on
        *every* device — not from the local shard first.  Starting locally is the
        obvious optimization and it is wrong: float addition is not associative, so
        device 1 computing ``s1+s0+s2+s3`` lands a few ULPs away from device 0
        computing ``s0+s1+s2+s3``, and the devices then disagree about the result of
        a collective. NCCL guarantees bitwise-identical all-reduce output across
        ranks, and so does this.
        """
        dtype = shards[0].dtype
        work_dtype = self._accumulation_dtype(dtype)
        outputs: list[Any] = []
        for device in self.devices:
            accumulator = self._to(shards[0], device).to(work_dtype)
            if accumulator is shards[0]:
                accumulator = accumulator.clone()
            for source in range(1, self.world_size):
                accumulator = self._combine(
                    accumulator, self._to(shards[source], device).to(work_dtype), op
                )
            outputs.append(self._finalize(accumulator, op, dtype))
        return outputs

    def _all_reduce_two_shot(self, shards: Sequence[Any], op: str) -> list[Any]:
        """Direct-copy reduce-scatter, then direct-copy all-gather.

        Device ``i`` owns slice ``i`` of the flattened buffer: it pulls that slice
        from every peer and reduces it, which makes it the sole authority for those
        elements. Then every device pulls each owner's reduced slice. Total volume
        is ``2(P−1)/P·D`` — the same bound as a ring — in two steps rather than
        ``2(P−1)``.
        """
        dtype = shards[0].dtype
        work_dtype = self._accumulation_dtype(dtype)
        shape = shards[0].shape
        flat = [tensor.reshape(-1) for tensor in shards]
        length = flat[0].numel()
        bounds = balanced_partition(length, self.world_size)

        # Step 1 — reduce-scatter: owner i reduces slice i.
        owned: list[Any] = []
        for index, device in enumerate(self.devices):
            start, end = bounds[index]
            accumulator = flat[index][start:end].to(work_dtype)
            for source in range(self.world_size):
                if source == index:
                    continue
                piece = self._to(flat[source][start:end], device).to(work_dtype)
                accumulator = self._combine(accumulator, piece, op)
            owned.append(self._finalize(accumulator, op, work_dtype))

        # Step 2 — all-gather: every device assembles all owners' slices.
        outputs: list[Any] = []
        for device in self.devices:
            assembled = self.torch.empty(length, dtype=work_dtype, device=device)
            for source in range(self.world_size):
                start, end = bounds[source]
                assembled[start:end] = self._to(owned[source], device)
            outputs.append(assembled.reshape(shape).to(dtype))
        return outputs

    def _all_reduce_ring(self, shards: Sequence[Any], op: str) -> list[Any]:
        """Nearest-neighbour ring reduce-scatter + all-gather.

        Used when the mesh is not fully peer-connected, so only neighbour copies
        are cheap. Chunk indices follow the standard schedule: at reduce-scatter
        step ``s`` rank ``r`` sends chunk ``(r−s) mod P`` and accumulates into
        ``(r−s−1) mod P``.
        """
        dtype = shards[0].dtype
        work_dtype = self._accumulation_dtype(dtype)
        shape = shards[0].shape
        length = shards[0].numel()
        bounds = balanced_partition(length, self.world_size)
        # chunks[rank][chunk] — each rank's working copy of every chunk.
        chunks = [
            [
                shards[rank].reshape(-1)[start:end].to(work_dtype).clone()
                for start, end in bounds
            ]
            for rank in range(self.world_size)
        ]
        for step in range(self.world_size - 1):
            staged: list[tuple[int, int, Any]] = []
            for rank in range(self.world_size):
                successor = (rank + 1) % self.world_size
                send_index = (rank - step) % self.world_size
                staged.append((
                    successor,
                    send_index,
                    self._to(chunks[rank][send_index], self.devices[successor]),
                ))
            for successor, chunk_index, payload in staged:
                chunks[successor][chunk_index] = self._combine(
                    chunks[successor][chunk_index], payload, op
                )
        for step in range(self.world_size - 1):
            staged = []
            for rank in range(self.world_size):
                successor = (rank + 1) % self.world_size
                send_index = (rank - step + 1) % self.world_size
                staged.append((
                    successor,
                    send_index,
                    self._to(chunks[rank][send_index], self.devices[successor]),
                ))
            for successor, chunk_index, payload in staged:
                chunks[successor][chunk_index] = payload
        outputs: list[Any] = []
        for rank, device in enumerate(self.devices):
            assembled = self.torch.cat(
                [chunk.reshape(-1) for chunk in chunks[rank]]
            ).reshape(shape)
            outputs.append(self._finalize(assembled, op, dtype).to(device))
        return outputs

    def all_gather(self, shards: Sequence[Any], dim: int = -1) -> list[Any]:
        """Concatenate every device's shard, in device order, on every device.

        Device order is not negotiable. A gather that concatenates in completion
        order produces a different tensor on different devices, which corrupts the
        column-parallel output of a tensor-parallel layer in a way no shape check
        catches.
        """
        if len(shards) != self.world_size:
            raise P2PCollectiveError(
                f"expected {self.world_size} shards, got {len(shards)}"
            )
        if self.world_size == 1:
            return [shards[0].clone()]
        outputs = [
            self.torch.cat(
                [self._to(shards[source], device) for source in range(self.world_size)],
                dim=dim,
            )
            for device in self.devices
        ]
        self._synchronize()
        return outputs

    def reduce_to_root(
        self, shards: Sequence[Any], op: str = "sum", root: int = 0
    ) -> Any:
        """Reduce onto one device using a binary tree: ``ceil(log2 P)`` rounds.

        This is the shape the single-process tensor-parallel executor actually
        needs — after a row-parallel GEMM only the primary device consumes the
        result, so replicating it to all ``P`` devices is work nobody reads.

        A tree is used rather than accumulating into the root sequentially: the
        sequential form is ``P−1`` dependent copies on one device's link, while the
        tree is ``ceil(log2 P)`` rounds of copies that proceed in parallel. Pairing
        is by fixed index, so the summation order — and therefore the result — is
        reproducible.
        """
        self._validate(shards)
        if not 0 <= root < self.world_size:
            raise P2PCollectiveError(f"root {root} out of range")
        if self.world_size == 1:
            return shards[0].clone()
        dtype = shards[0].dtype
        work_dtype = self._accumulation_dtype(dtype)
        # Relabel so the root is index 0; the tree then always folds toward it.
        order = [(root + offset) % self.world_size for offset in range(self.world_size)]
        working = [shards[index].to(work_dtype) for index in order]
        devices = [self.devices[index] for index in order]
        stride = 1
        while stride < self.world_size:
            for position in range(0, self.world_size, stride * 2):
                partner = position + stride
                if partner >= self.world_size:
                    continue
                working[position] = self._combine(
                    working[position],
                    self._to(working[partner], devices[position]),
                    op,
                )
            stride *= 2
        self._synchronize()
        return self._finalize(working[0], op, dtype)

    def stats(self) -> dict[str, Any]:
        """Collective counts by algorithm, plus the topology model in use."""
        return {
            "devices": [str(device) for device in self.devices],
            "world_size": self.world_size,
            "peer_access": self._peer_access,
            "fully_connected": self.link.fully_connected,
            "bandwidth_gbps": round(self.link.bandwidth_bps / 1e9, 1),
            "latency_us": round(self.link.latency_s * 1e6, 2),
            "crossover_bytes": self.link.crossover_bytes(self.world_size),
            "calls": self._calls,
            "algorithm_counts": dict(self._algorithm_counts),
            "requires_nccl": False,
        }
