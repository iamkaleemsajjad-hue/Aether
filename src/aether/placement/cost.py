"""The three-roof cost kernel, and the communication terms around it.

The standard roofline has a compute ceiling and a bandwidth ceiling.  A Python-
dispatched runtime has a third, and for small-model decode it is the binding one:

    t_stage = max( F/(θ_flops·u) , M/θ_bw , n_ops·t_dispatch )

The third term is not a patch.  Aether's own benchmark puts Qwen3-0.6B at 41.96
tok/s on one T4 — 23.8 ms per token — where the two-roof model predicts 3.75 ms.
The model was nowhere near its bandwidth roof, so the α–β analysis that says
"sharding halves the weight read, therefore shard" is arithmetically correct and
operationally wrong.  Adding a device raises the dispatch ceiling, and that is the
regression the planner has to predict.

Communication is the α–β model already calibrated in
:mod:`aether.parallelism.p2p_ring`, so the collective cost the planner predicts is
the cost the collective it selects will actually pay.

References:
  * Williams, Waterman & Patterson, "Roofline", CACM 52(4), 2009.
  * Thakur, Rabenseifner & Gropp, IJHPCA 19(1), 2005 — α–β and algorithm choice.
  * Patarasuk & Yuan, JPDC 69(2), 2009 — the 2(P−1)/P bandwidth bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aether.placement.census import DeviceCapability, DeviceCensus
from aether.placement.ledger import LedgerEntry
from aether.placement.model_profile import ModelProfile
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "RoofBreakdown",
    "StageCost",
    "decode_cost",
    "prefill_cost",
    "allreduce_seconds",
    "tp_comm_seconds",
    "pp_comm_seconds",
    "tp_prefill_comm_ratio",
    "link_model_for",
]


@dataclass(frozen=True)
class RoofBreakdown:
    """The three ceilings, and which one binds.

    Keeping all three rather than only the max is what lets the planner emit a
    recommendation instead of a verdict: *dispatch-bound at 23.8 ms against a 3.8 ms
    bandwidth roof* tells an operator to capture CUDA graphs, not to buy a GPU.
    """

    compute_s: float
    bandwidth_s: float
    dispatch_s: float

    @property
    def seconds(self) -> float:
        return max(self.compute_s, self.bandwidth_s, self.dispatch_s)

    @property
    def binding(self) -> str:
        ceilings = {
            "compute": self.compute_s,
            "bandwidth": self.bandwidth_s,
            "dispatch": self.dispatch_s,
        }
        return max(ceilings, key=lambda key: ceilings[key])

    @property
    def headroom_ratio(self) -> float:
        """How far the binding roof sits above the next one.

        A ratio near 1.0 means the workload is balanced; a large ratio names a
        single fixable bottleneck.
        """
        ordered = sorted((self.compute_s, self.bandwidth_s, self.dispatch_s), reverse=True)
        return ordered[0] / ordered[1] if ordered[1] > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_ms": round(self.compute_s * 1e3, 3),
            "bandwidth_ms": round(self.bandwidth_s * 1e3, 3),
            "dispatch_ms": round(self.dispatch_s * 1e3, 3),
            "binding": self.binding,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class StageCost:
    """One pipeline stage's predicted time, including its share of communication."""

    device_ids: tuple[str, ...]
    roofs: RoofBreakdown
    comm_s: float
    layers: int

    @property
    def seconds(self) -> float:
        return self.roofs.seconds + self.comm_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": list(self.device_ids),
            "layers": self.layers,
            "roofs": self.roofs.to_dict(),
            "comm_ms": round(self.comm_s * 1e3, 3),
            "total_ms": round(self.seconds * 1e3, 3),
        }


# ── the three roofs ───────────────────────────────────────────────────────────

def decode_cost(
    profile: ModelProfile,
    device: DeviceCapability,
    entry: LedgerEntry,
    *,
    weight_bytes: int,
    kv_bytes: int,
    ops: int,
    batch: int,
    context: int,
    tp_degree: int = 1,
) -> RoofBreakdown:
    """Per-token decode time on one device, as three ceilings.

    Decode streams **every** resident weight byte and **the whole live KV cache**
    once per step, which is why the bandwidth term is the sum of the two and why
    decode is the phase where extra aggregate bandwidth actually pays.
    """
    shards = max(1, tp_degree)
    flops = 2 * max(1, batch) * int(weight_bytes / max(profile.weight_dtype_bytes, 1e-9))
    flops += profile.attention_flops(max(1, batch), max(1, context)) // shards
    compute = flops / max(device.effective_flops, 1.0)
    bandwidth = (weight_bytes + kv_bytes) / max(device.effective_bandwidth_bps, 1.0)
    dispatch = max(0, ops) * max(entry.dispatch_seconds, 0.0)
    return RoofBreakdown(compute_s=compute, bandwidth_s=bandwidth, dispatch_s=dispatch)


def prefill_cost(
    profile: ModelProfile,
    device: DeviceCapability,
    entry: LedgerEntry,
    *,
    weight_bytes: int,
    ops: int,
    batch: int,
    sequence: int,
    tp_degree: int = 1,
    layers: int = 0,
) -> RoofBreakdown:
    """Whole-prefill time on one device, as three ceilings.

    Prefill amortises both the weight read and the dispatch cost over ``sequence``
    tokens, which is exactly why a model that is dispatch-bound in decode can be
    compute-bound in prefill — and why the two phases can prefer different plans.
    """
    shards = max(1, tp_degree)
    tokens = max(1, batch) * max(1, sequence)
    params = int(weight_bytes / max(profile.weight_dtype_bytes, 1e-9))
    flops = 2 * tokens * params
    # Attention is quadratic in the sequence: 2·b·h·S²·2 for scores plus context.
    flops += int(
        4 * max(1, batch) * profile.num_heads * profile.head_dim
        * max(1, sequence) ** 2 / shards
    )
    compute = flops / max(device.effective_flops, 1.0)
    # Activation traffic dominates the weight read once S is large, so both count.
    activation_traffic = profile.activation_bytes(
        batch, sequence, sequence, tp_degree=shards
    ) * max(1, layers or profile.layers)
    bandwidth = (weight_bytes + activation_traffic) / max(device.effective_bandwidth_bps, 1.0)
    dispatch = max(0, ops) * max(entry.dispatch_seconds, 0.0)
    return RoofBreakdown(compute_s=compute, bandwidth_s=bandwidth, dispatch_s=dispatch)


# ── communication ─────────────────────────────────────────────────────────────

def link_model_for(census: DeviceCensus, device_ids: "tuple[str, ...] | list[str]") -> Any:
    """Build the α–β model for a collective over these devices.

    Uses the *bottleneck* edge, because a collective runs at the speed of its worst
    pair. ``fully_connected`` is reported honestly so the collective's own algorithm
    selection can fall back to a ring when only neighbours are cheap.
    """
    from aether.parallelism.p2p_ring import LinkModel

    if len(device_ids) < 2:
        return LinkModel()
    edge = census.slowest_link(tuple(device_ids))
    if edge is None:
        # No topology information: assume the conservative PCIe default rather than
        # an optimistic one, so an unknown fabric never flatters a TP plan.
        return LinkModel()
    expected_pairs = len(device_ids) * (len(device_ids) - 1) // 2
    present = sum(
        1 for link in census.links
        if link.src in device_ids and link.dst in device_ids
    )
    return LinkModel(
        latency_s=edge.latency_s,
        bandwidth_bps=edge.bandwidth_bps,
        fully_connected=present >= expected_pairs,
    )


def allreduce_seconds(link_model: Any, payload_bytes: int, devices: int) -> float:
    """Predicted time for one all-reduce, using the schedule the runtime will pick.

    Takes the minimum over one-shot, two-shot and ring because that is precisely
    what :class:`~aether.parallelism.p2p_ring.P2PRingCollective` does at call time —
    the planner must predict the collective that will actually run, not the one that
    happens to be simplest to model.
    """
    if devices <= 1 or payload_bytes <= 0:
        return 0.0
    candidates = [
        link_model.one_shot_seconds(payload_bytes, devices),
        link_model.ring_seconds(payload_bytes, devices),
    ]
    if getattr(link_model, "fully_connected", True):
        candidates.append(link_model.two_shot_seconds(payload_bytes, devices))
    return min(candidates)


def tp_comm_seconds(
    profile: ModelProfile,
    census: DeviceCensus,
    device_ids: "tuple[str, ...]",
    *,
    layers: int,
    batch: int,
    step_tokens: int,
    dispatch_seconds: float = 0.0,
) -> float:
    """Communication for a tensor-parallel stage.

    Megatron's forward pass all-reduces twice per transformer block — once after the
    attention output projection, once after the FFN down projection — each carrying
    ``batch · step_tokens · hidden`` elements.  Two per layer, every layer, and every
    one is a barrier.

    ``dispatch_seconds`` is added per collective because a cross-device copy on the
    host critical path costs a launch and a synchronise regardless of payload, and on
    a dispatch-bound workload that term is the whole story.
    """
    devices = len(device_ids)
    if devices <= 1 or layers <= 0:
        return 0.0
    payload = int(
        max(1, batch) * max(1, step_tokens) * profile.hidden_size * profile.weight_dtype_bytes
    )
    link = link_model_for(census, device_ids)
    per_collective = allreduce_seconds(link, payload, devices) + max(0.0, dispatch_seconds)
    return 2.0 * layers * per_collective


def pp_comm_seconds(
    profile: ModelProfile,
    census: DeviceCensus,
    boundaries: "list[tuple[str, str]]",
    *,
    batch: int,
    step_tokens: int,
    dispatch_seconds: float = 0.0,
) -> float:
    """Communication for pipeline-stage boundaries.

    One activation send per boundary per step — not per layer — which is the whole
    reason pipeline parallelism survives a slow link where tensor parallelism does
    not. The payload crosses on the *fastest* edge available between the two stages,
    since a stage leader can be chosen for exactly that.
    """
    if not boundaries:
        return 0.0
    payload = int(
        max(1, batch) * max(1, step_tokens) * profile.hidden_size * profile.weight_dtype_bytes
    )
    total = 0.0
    for source, target in boundaries:
        edge = census.link(source, target)
        if edge is None:
            # Unknown edge between stages: charge the host-staged default rather
            # than assuming the boundary is free.
            latency, bandwidth = 8e-6, 6e9
        else:
            latency, bandwidth = edge.latency_s, edge.bandwidth_bps
        total += latency + payload / max(bandwidth, 1.0) + max(0.0, dispatch_seconds)
    return total


def tp_prefill_comm_ratio(
    profile: ModelProfile,
    device: DeviceCapability,
    link_bandwidth_bps: float,
    devices: int,
) -> float:
    """Tensor-parallel prefill communication as a fraction of its compute.

    The closed form, and the batch and sequence terms cancel:

        comm/compute = (P−1)·θ_flops / ( 6·h·β )

    because communication grows as ``b·s·h`` while compute grows as ``b·s·L·12h²``.
    So the question is never "is the context long" — it is whether the hidden size
    and the link are large enough for the device's arithmetic rate.  Reported in the
    decision record because it is the single number that explains why the same plan
    is right on NVLink and wrong on host-staged PCIe.
    """
    if devices <= 1 or profile.hidden_size <= 0 or link_bandwidth_bps <= 0:
        return 0.0
    return (
        (devices - 1) * device.effective_flops
        / (6.0 * profile.hidden_size * link_bandwidth_bps)
    )
