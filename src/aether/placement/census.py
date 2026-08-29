"""Device capability census — measured, not assumed.

The planner's first input is a description of the machine, and every number in it
is either read from the driver, timed by a micro-benchmark, or taken from the
target registry and *labelled as a prior*.  Nothing is silently invented, because
a capability number the planner trusts and cannot justify is the same failure as a
hard-coded threshold, one level down.

Two fields deserve their own explanation.

``fabric_class``
    Devices are partitioned into fabric classes by the interconnect graph, and a
    tensor-parallel group may not span a class (Law II).  The partition is the
    connected components of the topology graph restricted to edges whose bandwidth
    is within :data:`FABRIC_BANDWIDTH_RATIO` of the best edge on the host — a link
    four times slower than the machine's best link is a different fabric.  This
    adapts to any topology instead of hard-coding "NVLink versus PCIe".

``achieved_fraction``
    Peak FLOPs are marketing; sustained FLOPs are engineering.  The ratio is
    calibrated per device signature and starts from a conservative prior, so the
    cost model is never comparing a measured bandwidth against a peak FLOP rate.

References:
  * device bandwidth priors: ``aether.compiler.stage3_targeting.hardware_profile``
  * interconnect detection: ``aether.parallelism.hardware_topology``
"""

from __future__ import annotations

import contextlib
import os
import platform
from dataclasses import dataclass
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "FABRIC_BANDWIDTH_RATIO",
    "DeviceCapability",
    "FabricLink",
    "DeviceCensus",
    "measure_bandwidth_bps",
    "measure_dispatch_seconds",
    "take_census",
]

FABRIC_BANDWIDTH_RATIO = 4.0
"""A link this many times slower than the host's best link is a separate fabric."""

_BANDWIDTH_PROBE_BYTES = 64 << 20
"""Payload for the device-to-device copy that measures effective bandwidth."""

_DISPATCH_PROBE_OPS = 400
"""Operations timed per dispatch-probe pass.

Large enough that the perf_counter resolution is negligible against the total, small
enough that the whole probe costs well under 100 ms on any host."""

_DISPATCH_PROBE_PASSES = 12
"""Upper bound on probe passes. Convergence normally stops it after three or four."""

_DISPATCH_PROBE_SETTLED = 1.25
"""Pass-to-pass agreement that counts as converged.

Comfortably inside the staleness ratio, so a converged probe cannot trip the staleness
check against the value a converged probe stored a moment earlier."""

# Conservative sustained-throughput priors, used only until a measurement exists.
# CPUs are deliberately pessimistic: a CPU that turns out to be faster than this
# will be promoted by its own measurement, whereas an optimistic prior would put a
# CPU into a plan it cannot carry.
_CPU_BANDWIDTH_BPS = 40e9
_CPU_FLOPS = 300e9
_UNKNOWN_ACCEL_BANDWIDTH_BPS = 200e9
_UNKNOWN_ACCEL_FLOPS = 10e12

_DEFAULT_ACHIEVED_FLOPS = 0.45
"""Fraction of peak FLOPs a real kernel sustains, before calibration."""

_DEFAULT_ACHIEVED_BANDWIDTH = 0.80
"""Fraction of peak memory bandwidth a real kernel sustains, before calibration."""


@dataclass(frozen=True)
class DeviceCapability:
    """One execution device, as measured on this host at this moment.

    ``free_bytes`` and ``external_bytes`` are volatile and are re-read on every
    census; everything else is stable for a given device and is cacheable against
    :attr:`signature`.
    """

    device_id: str
    """Backend-addressable identifier: ``cuda:0``, ``mps``, ``cpu``."""

    kind: str
    """``cuda`` | ``rocm`` | ``mps`` | ``cpu`` | ``other``."""

    name: str
    total_bytes: int
    free_bytes: int
    external_bytes: int
    """Bytes held on this device by processes that are not this one."""

    bandwidth_bps: float
    """Peak memory bandwidth. Multiply by :attr:`achieved_bandwidth` to use it."""

    flops: float
    """Peak dense FLOP/s at the model's compute precision."""

    achieved_flops: float = _DEFAULT_ACHIEVED_FLOPS
    achieved_bandwidth: float = _DEFAULT_ACHIEVED_BANDWIDTH
    fabric_class: int = 0
    supports_peer_access: bool = False
    unified_memory: bool = False
    measured: tuple[str, ...] = ()
    """Which fields came from a live measurement rather than a prior."""

    @property
    def signature(self) -> str:
        """Stable ledger key for this device model.

        Deliberately excludes the device index: two identical GPUs in one box share
        calibration, and pinning the index would throw away half the evidence.
        """
        return (
            f"{self.kind}:{self.name.replace(' ', '_')}:"
            f"{int(self.total_bytes) // 1024 ** 3}GiB"
        )

    @property
    def effective_bandwidth_bps(self) -> float:
        return self.bandwidth_bps * self.achieved_bandwidth

    @property
    def effective_flops(self) -> float:
        return self.flops * self.achieved_flops

    @property
    def is_accelerator(self) -> bool:
        return self.kind in ("cuda", "rocm", "mps", "other")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "name": self.name,
            "total_gib": round(self.total_bytes / 1024 ** 3, 2),
            "free_gib": round(self.free_bytes / 1024 ** 3, 2),
            "external_gib": round(self.external_bytes / 1024 ** 3, 2),
            "bandwidth_gbps": round(self.bandwidth_bps / 1e9, 1),
            "tflops": round(self.flops / 1e12, 2),
            "achieved_flops": self.achieved_flops,
            "achieved_bandwidth": self.achieved_bandwidth,
            "fabric_class": self.fabric_class,
            "signature": self.signature,
            "measured": list(self.measured),
        }


@dataclass(frozen=True)
class FabricLink:
    """One interconnect edge, in the units the cost model needs."""

    src: str
    dst: str
    kind: str
    bandwidth_bps: float
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "bandwidth_gbps": round(self.bandwidth_bps / 1e9, 1),
            "latency_us": round(self.latency_s * 1e6, 2),
        }


@dataclass(frozen=True)
class DeviceCensus:
    """Every execution device on this host, plus the fabric between them."""

    devices: tuple[DeviceCapability, ...]
    links: tuple[FabricLink, ...] = ()
    host_bytes: int = 0
    """Usable host RAM, for offload-tier planning."""

    backend_build: str = ""
    """Runtime identity — the dispatch cost is a property of this, not the device."""

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("a census must contain at least one device")

    def by_id(self, device_id: str) -> DeviceCapability:
        for device in self.devices:
            if device.device_id == device_id:
                return device
        raise KeyError(f"no device {device_id!r} in census")

    @property
    def accelerators(self) -> tuple[DeviceCapability, ...]:
        return tuple(device for device in self.devices if device.is_accelerator)

    def link(self, src: str, dst: str) -> FabricLink | None:
        """Return the link between two devices, in either direction."""
        for edge in self.links:
            if {edge.src, edge.dst} == {src, dst}:
                return edge
        return None

    def slowest_link(self, device_ids: "tuple[str, ...] | list[str]") -> FabricLink | None:
        """The bottleneck edge among a set of devices — what a collective pays.

        A collective over a set runs at the speed of its worst pair, so this is the
        edge the cost model must use, not the average and not the best.
        """
        edges = [
            edge for edge in self.links
            if edge.src in device_ids and edge.dst in device_ids
        ]
        return min(edges, key=lambda edge: edge.bandwidth_bps) if edges else None

    def fabric_groups(self) -> dict[int, tuple[DeviceCapability, ...]]:
        """Devices grouped by fabric class, in ascending class order."""
        groups: dict[int, list[DeviceCapability]] = {}
        for device in self.devices:
            groups.setdefault(device.fabric_class, []).append(device)
        return {key: tuple(value) for key, value in sorted(groups.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_build": self.backend_build,
            "host_gib": round(self.host_bytes / 1024 ** 3, 2),
            "devices": [device.to_dict() for device in self.devices],
            "links": [edge.to_dict() for edge in self.links],
            "fabric_classes": len(self.fabric_groups()),
        }


# ── detection ─────────────────────────────────────────────────────────────────

def _execution_mode() -> str:
    """The host-side execution shape, which is half of what sets ``t_dispatch``.

    A framework version alone does not identify the dispatch cost.  Capturing a CUDA
    graph or compiling the decode path collapses hundreds of launches into one, moving
    the cost by an order of magnitude without moving any version string — and a
    dispatch cost attributed to the wrong mode makes the planner refuse to shard
    models it should.  So the mode is part of the key, and the key is checked against a
    probe (see :meth:`~aether.placement.ledger.CalibrationLedger.reconcile_dispatch`)
    rather than trusted.
    """
    modes: list[str] = []
    for variable, token in (
        ("AETHER_CUDA_GRAPHS", "graph"),
        ("AETHER_TORCH_COMPILE", "compile"),
    ):
        value = os.environ.get(variable, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            modes.append(token)
    try:
        import torch

        # A live capture or a compiled callable is authoritative over the request.
        if getattr(torch.cuda, "is_current_stream_capturing", None) is not None:
            with contextlib.suppress(Exception):
                if torch.cuda.is_current_stream_capturing():
                    modes.append("capturing")
        threads = int(torch.get_num_threads())
        if threads > 0:
            modes.append(f"t{threads}")
    except Exception:  # noqa: BLE001 - a torch-free install has no mode to report
        pass
    return "+".join(modes) if modes else "eager"


def _backend_build() -> str:
    """Identity of the runtime whose dispatch cost we are calibrating.

    ``t_dispatch`` is a property of the host-side stack, not of the accelerator, so
    the key carries every layer of that stack that can move it: the interpreter, the
    framework and its CUDA build, Aether's own version — the decode loop that issues
    the operations is Aether code — and the execution mode.

    Mis-keying is the failure mode the design flagged as the one to watch, because an
    inflated dispatch roof does not merely mispredict a time: it makes every wider plan
    look worse, so the planner refuses to shard models it should.  Widening the key
    narrows the chance of a collision; it cannot eliminate it, which is why the value
    is also verified against a fresh probe before it is used.
    """
    parts = [f"py{platform.python_version()}"]
    try:
        import torch

        parts.append(f"torch{torch.__version__}")
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version:
            parts.append(f"cu{cuda_version}")
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            parts.append(f"hip{hip_version}")
    except Exception:  # noqa: BLE001 - a torch-free install is a valid backend
        parts.append("numpy")
    try:
        from aether.core.constants import AETHER_VERSION

        parts.append(f"aether{AETHER_VERSION}")
    except Exception:  # noqa: BLE001 - version metadata is advisory
        pass
    parts.append(_execution_mode())
    return "|".join(parts)


def _host_memory() -> tuple[int, int]:
    """Return ``(total, available)`` host bytes.

    The distinction is load-bearing: the kappa ceiling is a fraction of *installed*
    memory, while what may be committed right now is what is *available*. Using
    available for both would make a busy machine look permanently tiny, and using
    total for both would ignore every other process on the box.
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:  # noqa: BLE001
        return 0, 0


def _host_bytes() -> int:
    """Host bytes usable for an offload tier — what is available, not installed."""
    return _host_memory()[1]


def _cuda_memory(torch: Any, index: int) -> tuple[int, int, int]:
    """Return ``(total, free, external)`` bytes for one CUDA device.

    ``external`` is what other processes hold: the driver reports device-wide free
    memory, and anything used that this process did not reserve belongs to someone
    else.  Charging it to ourselves is how a planner OOMs on a shared GPU.
    """
    free, total = torch.cuda.mem_get_info(index)
    reserved = int(torch.cuda.memory_reserved(index))
    used_by_anyone = int(total) - int(free)
    external = max(0, used_by_anyone - reserved)
    return int(total), int(free), external


def _registry_prior(name: str, kind: str) -> tuple[float, float]:
    """Bandwidth and FLOPs priors from the target registry, by device name."""
    try:
        from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile

        for target_id, data in getattr(HardwareProfile, "PROFILES", {}).items():
            device_name = str(data.get("device_name", "")).lower()
            if device_name and device_name in name.lower():
                bandwidth = float(data.get("memory_bandwidth_gb_s", 0.0)) * 1e9
                flops = float(
                    data.get("flops_fp16", data.get("flops_bf16", data.get("flops_fp32", 0.0)))
                )
                if bandwidth > 0 and flops > 0:
                    logger.debug("capability prior for %s from %s", name, target_id)
                    return bandwidth, flops
    except Exception as exc:  # noqa: BLE001 - the registry is advisory
        logger.debug("capability registry lookup failed: %s", exc)
    if kind == "cpu":
        return _CPU_BANDWIDTH_BPS, _CPU_FLOPS
    return _UNKNOWN_ACCEL_BANDWIDTH_BPS, _UNKNOWN_ACCEL_FLOPS


def measure_bandwidth_bps(device_id: str, *, payload_bytes: int = _BANDWIDTH_PROBE_BYTES) -> float:
    """Time a large on-device copy to get effective memory bandwidth.

    A copy reads ``payload_bytes`` and writes ``payload_bytes``, so the achieved
    bandwidth is ``2·payload / elapsed``.  This is a *sustained* number and it is
    what the bandwidth roof needs — decode is a streaming read of the weights, not
    a peak-rate burst.  Returns ``0.0`` when the probe cannot run, so the caller
    keeps its prior rather than trusting a failed measurement.
    """
    try:
        import torch

        device = torch.device(device_id)
        if device.type not in ("cuda", "mps"):
            return 0.0
        count = payload_bytes // 4
        source = torch.empty(count, dtype=torch.float32, device=device)
        target = torch.empty_like(source)
        for _ in range(2):  # warm up allocator and clocks
            target.copy_(source)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                target.copy_(source)
            end.record()
            torch.cuda.synchronize(device)
            elapsed = start.elapsed_time(end) / 1000.0 / 5
        else:
            import time as _time

            torch.mps.synchronize()
            begin = _time.perf_counter()
            for _ in range(5):
                target.copy_(source)
            torch.mps.synchronize()
            elapsed = (_time.perf_counter() - begin) / 5
        del source, target
        if elapsed <= 0:
            return 0.0
        return 2.0 * payload_bytes / elapsed
    except Exception as exc:  # noqa: BLE001 - probing must never block planning
        logger.debug("bandwidth probe on %s failed: %s", device_id, exc)
        return 0.0


def measure_dispatch_seconds(
    device_id: str, *, operations: int = _DISPATCH_PROBE_OPS
) -> float:
    """Time the host cost of issuing one graph operation on this device.

    The dispatch roof is ``n_ops · t_dispatch``, and ``t_dispatch`` is *host-side
    issue* cost: the Python call, the framework's dispatcher, the kernel launch.  So
    the probe measures exactly that and nothing else — a stream of tiny operations
    timed on the **host clock with no synchronisation inside the loop**, so the queue
    never drains and the wall time is the issue cost rather than the device time.

    Synchronising per operation would measure launch latency plus device execution plus
    a round trip, which is a different and much larger quantity; using it as
    ``t_dispatch`` would inflate the dispatch roof and make the planner refuse to shard
    bandwidth-bound models.  One synchronise happens after the timed region, purely so
    the probe does not leave work queued behind it.

    The measurement is taken to *convergence* rather than for a fixed number of passes.
    That matters more than it looks: a cold process pays dispatcher and autograd
    initialisation on its first few hundred calls, so a fixed-count probe reports a
    number the same machine will not reproduce a second later — and a value that does
    not reproduce would trip the staleness check on the next load and thrash the stored
    calibration.  Passes therefore continue until two consecutive ones agree, and the
    minimum is kept because the fastest pass is the one least polluted by the OS
    scheduler.

    Returns:
        Seconds per operation, or ``0.0`` when the probe cannot run — in which case the
        caller keeps its prior and says so, rather than trusting a failed measurement.
    """
    try:
        import time as _time

        import torch

        device = torch.device(device_id)
        tensor = torch.ones(16, dtype=torch.float32, device=device)
        synchronize = None
        if device.type == "cuda":
            def synchronize() -> None:
                torch.cuda.synchronize(device)
        elif device.type == "mps":
            synchronize = torch.mps.synchronize

        def one_pass() -> float:
            start = _time.perf_counter()
            for _ in range(operations):
                tensor.add_(1.0)
            elapsed = _time.perf_counter() - start
            if synchronize is not None:
                synchronize()
            return elapsed / operations if elapsed > 0 else 0.0

        best = 0.0
        previous = 0.0
        settled = 0
        for _ in range(_DISPATCH_PROBE_PASSES):
            per_op = one_pass()
            if per_op <= 0:
                continue
            best = per_op if best == 0.0 else min(best, per_op)
            if previous > 0:
                spread = max(per_op / previous, previous / per_op)
                settled = settled + 1 if spread <= _DISPATCH_PROBE_SETTLED else 0
                if settled >= 2:
                    break
            previous = per_op
        return best
    except Exception as exc:  # noqa: BLE001 - probing must never block planning
        logger.debug("dispatch probe on %s failed: %s", device_id, exc)
        return 0.0


def _detect_links(device_ids: list[str]) -> tuple[FabricLink, ...]:
    """Build the interconnect edge list from the existing topology detector."""
    if len(device_ids) < 2:
        return ()
    try:
        from aether.parallelism.hardware_topology import (
            InterconnectType,
            detect_hardware_topology,
        )

        topology = detect_hardware_topology(device_ids)
        fast = {InterconnectType.NVLINK, InterconnectType.XGMI}
        edges: list[FabricLink] = []
        for edge in topology.edges:
            # A fast fabric launches a copy in roughly a microsecond; a PCIe copy
            # that may stage through the host costs several. One latency figure for
            # both would put the collective crossover orders of magnitude out.
            latency = 1.5e-6 if edge.link in fast else 8e-6
            edges.append(
                FabricLink(
                    src=edge.src,
                    dst=edge.dst,
                    kind=edge.link.name,
                    bandwidth_bps=edge.bandwidth_bps,
                    latency_s=latency,
                )
            )
        return tuple(edges)
    except Exception as exc:  # noqa: BLE001 - topology detection is advisory
        logger.debug("interconnect detection unavailable: %s", exc)
        return ()


def _assign_fabric_classes(
    device_ids: list[str], links: tuple[FabricLink, ...]
) -> dict[str, int]:
    """Partition devices into fabric classes (Law II's domain).

    Connected components of the graph restricted to edges within
    :data:`FABRIC_BANDWIDTH_RATIO` of the host's best edge.  A device with no fast
    edge to anything becomes its own class, which is exactly the behaviour Law II
    needs: it can hold a pipeline stage but cannot join someone else's TP group.
    """
    classes = {device_id: index for index, device_id in enumerate(device_ids)}
    if not links:
        return {device_id: 0 for device_id in device_ids} if len(device_ids) <= 1 else classes
    best = max(edge.bandwidth_bps for edge in links)
    threshold = best / FABRIC_BANDWIDTH_RATIO

    parent = {device_id: device_id for device_id in device_ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in links:
        if edge.bandwidth_bps < threshold:
            continue
        if edge.src not in parent or edge.dst not in parent:
            continue
        left, right = find(edge.src), find(edge.dst)
        if left != right:
            parent[right] = left

    roots: dict[str, int] = {}
    for device_id in device_ids:
        root = find(device_id)
        if root not in roots:
            roots[root] = len(roots)
        classes[device_id] = roots[root]
    return classes


def take_census(
    *,
    device_ids: list[str] | None = None,
    ledger: Any | None = None,
    probe_bandwidth: bool = True,
    probe_dispatch: bool = True,
) -> DeviceCensus:
    """Measure this host and return the planner's hardware input.

    Args:
        device_ids: Restrict the census to these devices. Defaults to every
            accelerator plus the CPU.
        ledger: A :class:`~aether.placement.ledger.CalibrationLedger`. When given,
            measured bandwidth is cached against the device signature so the probe
            runs once per machine rather than once per model load.
        probe_bandwidth: Time an on-device copy to get sustained bandwidth. Costs
            roughly 20 ms per device and replaces a registry prior with a fact.
        probe_dispatch: Time a stream of trivial operations to get the host cost per
            graph op, and reconcile it against whatever the ledger holds for this
            backend build. Costs a few tens of milliseconds per device and is the
            check that catches a mis-keyed or stale dispatch cost — the failure mode
            that silently stops the planner sharding models it should.

    Returns:
        A :class:`DeviceCensus`. Always succeeds: a host with no accelerator
        yields a CPU-only census rather than an error.
    """
    devices: list[DeviceCapability] = []
    discovered: list[tuple[str, str, str, int, int, int, bool, bool]] = []
    backend_build = _backend_build()

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        try:
            if torch.cuda.is_available():
                is_rocm = bool(getattr(torch.version, "hip", None))
                for index in range(torch.cuda.device_count()):
                    identifier = f"cuda:{index}"
                    if device_ids is not None and identifier not in device_ids:
                        continue
                    total, free, external = _cuda_memory(torch, index)
                    properties = torch.cuda.get_device_properties(index)
                    discovered.append((
                        identifier,
                        "rocm" if is_rocm else "cuda",
                        properties.name,
                        total, free, external,
                        True, False,
                    ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA census failed: %s", exc)
        try:
            mps = getattr(torch.backends, "mps", None)
            wants_mps = device_ids is None or "mps" in device_ids
            if wants_mps and mps is not None and mps.is_available():
                # Unified memory: the GPU and the host share one pool, so the census
                # reports the same totals for both and the planner's residual
                # naturally accounts for the sharing.
                total, available = _host_memory()
                discovered.append(
                    ("mps", "mps", "Apple Silicon GPU", total, available, 0, False, True)
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("MPS census failed: %s", exc)

    if device_ids is None or "cpu" in device_ids:
        total, available = _host_memory()
        # No external term for the CPU: ``available`` is already net of every other
        # process, so charging the difference again would report zero capacity on any
        # machine that is doing anything at all.
        discovered.append((
            "cpu", "cpu", platform.processor() or platform.machine() or "CPU",
            total, available, 0, False, True,
        ))

    accelerator_ids = [
        entry[0] for entry in discovered if entry[1] in ("cuda", "rocm", "mps", "other")
    ]
    links = _detect_links(accelerator_ids)
    classes = _assign_fabric_classes(accelerator_ids, links)

    for identifier, kind, name, total, free, external, peer, unified in discovered:
        bandwidth, flops = _registry_prior(name, kind)
        achieved_flops = _DEFAULT_ACHIEVED_FLOPS
        achieved_bandwidth = _DEFAULT_ACHIEVED_BANDWIDTH
        measured: list[str] = []

        entry = None
        signature = DeviceCapability(
            device_id=identifier, kind=kind, name=name,
            total_bytes=total, free_bytes=free, external_bytes=external,
            bandwidth_bps=bandwidth, flops=flops,
        ).signature
        if ledger is not None:
            entry = ledger.get(signature, backend_build)
            if entry is not None:
                if entry.measured_bandwidth_bps > 0:
                    bandwidth = entry.measured_bandwidth_bps
                    achieved_bandwidth = 1.0  # already a sustained figure
                    measured.append("bandwidth")
                if entry.achieved_flops > 0:
                    achieved_flops = entry.achieved_flops
                    measured.append("achieved_flops")

        wants_probe = (
            probe_bandwidth
            and "bandwidth" not in measured
            and kind in ("cuda", "rocm", "mps")
        )
        if wants_probe:
            observed = measure_bandwidth_bps(identifier)
            if observed > 0:
                bandwidth = observed
                achieved_bandwidth = 1.0
                measured.append("bandwidth")
                if ledger is not None:
                    ledger.record_bandwidth(signature, observed)

        # ── dispatch cost ────────────────────────────────────────────────────
        # Probed on every census rather than cached-and-trusted, because the stored
        # value is keyed by a *description* of the runtime and a description can be
        # wrong. Reconciling a fresh measurement against the ledger turns a silent
        # systematic bias into a logged replacement.
        if probe_dispatch and ledger is not None:
            probed = measure_dispatch_seconds(identifier)
            if probed > 0:
                in_force, replaced = ledger.reconcile_dispatch(
                    signature, backend_build, probed
                )
                if in_force > 0:
                    measured.append("dispatch")
                if replaced:
                    measured.append("dispatch_refreshed")
        elif ledger is not None and entry is not None and entry.dispatch_measured:
            measured.append("dispatch")

        devices.append(DeviceCapability(
            device_id=identifier, kind=kind, name=name,
            total_bytes=total, free_bytes=free, external_bytes=external,
            bandwidth_bps=bandwidth, flops=flops,
            achieved_flops=achieved_flops, achieved_bandwidth=achieved_bandwidth,
            fabric_class=classes.get(identifier, 0),
            supports_peer_access=peer, unified_memory=unified,
            measured=tuple(measured),
        ))

    census = DeviceCensus(
        devices=tuple(devices),
        links=links,
        host_bytes=_host_bytes(),
        backend_build=backend_build,
    )
    logger.info(
        "device census: %d devices, %d fabric class(es), backend %s",
        len(census.devices), len(census.fabric_groups()), census.backend_build,
    )
    return census
