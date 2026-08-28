"""NCCL-free multi-GPU collectives: correctness, ordering, and algorithm choice.

The claim these tests defend is narrow and checkable: Aether performs multi-device
all-reduce, all-gather and reduce-to-root without NCCL, RCCL or
``torch.distributed``, and the results equal the mathematical definition.

Every test runs on CPU "devices". That is not a weakness of the test — the
schedules are device-agnostic, and the arithmetic they must reproduce is identical.
What CPU execution cannot check is throughput, which is what the α–β cost model in
:class:`LinkModel` is separately asserted against.

References:
  * Patarasuk & Yuan, JPDC 69(2), 2009 — the 2(P-1)/P bandwidth bound.
  * Thakur, Rabenseifner & Gropp, IJHPCA 19(1), 2005 — algorithm choice by size.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aether.parallelism.p2p_ring import (  # noqa: E402
    Algorithm,
    LinkModel,
    P2PCollectiveError,
    P2PRingCollective,
)

ALL_ALGORITHMS = [Algorithm.ONE_SHOT, Algorithm.TWO_SHOT, Algorithm.RING]


def _collective(world_size: int, **kwargs: object) -> P2PRingCollective:
    """A fully-connected CPU mesh of ``world_size`` devices."""
    return P2PRingCollective(
        ["cpu"] * world_size,
        torch_module=torch,
        link_model=LinkModel(fully_connected=True),
        **kwargs,  # type: ignore[arg-type]
    )


def _shards(world_size: int, size: int = 37, seed: int = 0) -> list[object]:
    torch.manual_seed(seed)
    return [torch.randn(size) for _ in range(world_size)]


# ── all_reduce correctness ────────────────────────────────────────────────────

@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_all_reduce_sum_equals_the_stacked_sum(world_size: int, algorithm: Algorithm) -> None:
    collective = _collective(world_size)
    shards = _shards(world_size)
    expected = torch.stack(shards).sum(0)
    for result in collective.all_reduce(shards, "sum", algorithm=algorithm):
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
@pytest.mark.parametrize(
    ("op", "reference"),
    [
        ("max", lambda stacked: stacked.max(0).values),
        ("min", lambda stacked: stacked.min(0).values),
        ("avg", lambda stacked: stacked.mean(0)),
    ],
)
def test_all_reduce_other_reductions(algorithm: Algorithm, op: str, reference) -> None:
    collective = _collective(4)
    shards = _shards(4)
    expected = reference(torch.stack(shards))
    for result in collective.all_reduce(shards, op, algorithm=algorithm):
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_every_device_receives_the_identical_result(algorithm: Algorithm) -> None:
    """A collective where devices disagree is worse than one that fails."""
    collective = _collective(4)
    outputs = collective.all_reduce(_shards(4), "sum", algorithm=algorithm)
    for result in outputs[1:]:
        torch.testing.assert_close(result, outputs[0], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_all_reduce_is_bit_reproducible(algorithm: Algorithm) -> None:
    """Fixed device order means fixed summation order means identical bits.

    Float addition is not associative, so a reduction ordered by arrival would give
    different logits from one run to the next.
    """
    collective = _collective(4)
    shards = _shards(4)
    first = collective.all_reduce(shards, "sum", algorithm=algorithm)[0]
    second = collective.all_reduce(shards, "sum", algorithm=algorithm)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
@pytest.mark.parametrize("size", [1, 2, 7, 8, 9, 1024])
def test_sizes_not_divisible_by_the_device_count(algorithm: Algorithm, size: int) -> None:
    """Padding must not leak into the result, and remainders must not be dropped."""
    collective = _collective(4)
    shards = _shards(4, size=size)
    expected = torch.stack(shards).sum(0)
    for result in collective.all_reduce(shards, "sum", algorithm=algorithm):
        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_multidimensional_shapes_survive_the_flatten(algorithm: Algorithm) -> None:
    collective = _collective(4)
    torch.manual_seed(1)
    shards = [torch.randn(3, 5, 7) for _ in range(4)]
    expected = torch.stack(shards).sum(0)
    for result in collective.all_reduce(shards, "sum", algorithm=algorithm):
        assert result.shape == (3, 5, 7)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_half_precision_accumulates_in_float32_and_casts_back(algorithm: Algorithm) -> None:
    """Summing P half-precision partials in half precision loses bits per hop."""
    collective = _collective(8)
    shards = [torch.full((16,), 1e-4, dtype=torch.float16) for _ in range(8)]
    result = collective.all_reduce(shards, "sum", algorithm=algorithm)[0]
    assert result.dtype == torch.float16
    torch.testing.assert_close(
        result, torch.full((16,), 8e-4, dtype=torch.float16), rtol=1e-3, atol=1e-6
    )


def test_world_size_one_is_the_identity() -> None:
    collective = _collective(1)
    shard = torch.randn(5)
    torch.testing.assert_close(collective.all_reduce([shard])[0], shard)


# ── all_gather ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_concatenates_in_device_order(world_size: int) -> None:
    """Order is by device, always — a completion-ordered gather is a silent bug."""
    collective = _collective(world_size)
    shards = [torch.full((2, 3), float(index)) for index in range(world_size)]
    for result in collective.all_gather(shards, dim=-1):
        assert result.shape == (2, 3 * world_size)
        assert result[0].tolist() == [
            float(index) for index in range(world_size) for _ in range(3)
        ]


def test_all_gather_rejects_a_wrong_shard_count() -> None:
    collective = _collective(4)
    with pytest.raises(P2PCollectiveError, match="expected 4 shards"):
        collective.all_gather([torch.zeros(2)] * 3)


# ── reduce_to_root ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 8])
def test_reduce_to_root_equals_the_stacked_sum(world_size: int) -> None:
    collective = _collective(world_size)
    shards = _shards(world_size)
    expected = torch.stack(shards).sum(0)
    for root in range(world_size):
        result = collective.reduce_to_root(shards, "sum", root=root)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_reduce_to_root_rejects_an_out_of_range_root() -> None:
    collective = _collective(4)
    with pytest.raises(P2PCollectiveError, match="root 9 out of range"):
        collective.reduce_to_root(_shards(4), root=9)


def test_reduce_to_root_is_bit_reproducible() -> None:
    collective = _collective(8)
    shards = _shards(8)
    first = collective.reduce_to_root(shards, "sum")
    second = collective.reduce_to_root(shards, "sum")
    torch.testing.assert_close(first, second, rtol=0, atol=0)


# ── Validation ────────────────────────────────────────────────────────────────

def test_mismatched_shard_shapes_are_rejected() -> None:
    """A shape-mismatched shard would yield a plausible tensor of wrong values."""
    collective = _collective(2)
    with pytest.raises(P2PCollectiveError, match="expected"):
        collective.all_reduce([torch.zeros(4), torch.zeros(5)])


def test_mismatched_shard_dtypes_are_rejected() -> None:
    collective = _collective(2)
    with pytest.raises(P2PCollectiveError, match="dtype"):
        collective.all_reduce([torch.zeros(4), torch.zeros(4, dtype=torch.float64)])


def test_wrong_shard_count_is_rejected() -> None:
    collective = _collective(4)
    with pytest.raises(P2PCollectiveError, match="one per device"):
        collective.all_reduce([torch.zeros(4)] * 2)


def test_unsupported_reduction_is_rejected() -> None:
    collective = _collective(2)
    with pytest.raises(P2PCollectiveError, match="unsupported reduction"):
        collective.all_reduce(_shards(2), "median")


# ── Cost model and algorithm selection ────────────────────────────────────────

def test_two_shot_matches_the_ring_bandwidth_bound() -> None:
    """Both move 2(P-1)/P of the payload; two-shot just spends less latency."""
    link = LinkModel(latency_s=1e-5, bandwidth_bps=64e9)
    payload = 64 << 20
    volume_term = 2 * 3 / 4 * payload / 64e9
    assert link.two_shot_seconds(payload, 4) == pytest.approx(2e-5 + volume_term)
    assert link.ring_seconds(payload, 4) == pytest.approx(6e-5 + volume_term)
    assert link.two_shot_seconds(payload, 4) < link.ring_seconds(payload, 4)


def test_one_shot_moves_more_bytes_than_two_shot_for_large_payloads() -> None:
    link = LinkModel(latency_s=1e-6, bandwidth_bps=64e9)
    payload = 256 << 20
    assert link.one_shot_seconds(payload, 8) > link.two_shot_seconds(payload, 8)


def test_crossover_is_where_the_two_predictions_meet() -> None:
    """The selection rule is the algebra, not a tuned constant."""
    link = LinkModel(latency_s=8e-6, bandwidth_bps=64e9)
    for world_size in (3, 4, 8):
        crossover = link.crossover_bytes(world_size)
        assert link.one_shot_seconds(crossover, world_size) == pytest.approx(
            link.two_shot_seconds(crossover, world_size), rel=1e-9
        )


def test_two_devices_never_prefer_two_shot() -> None:
    """At P=2 one-shot has equal volume and half the hops."""
    assert LinkModel().crossover_bytes(2) == float("inf")
    assert _collective(2).select_algorithm(1 << 30) is Algorithm.ONE_SHOT


def test_small_payloads_select_one_shot_and_large_select_two_shot() -> None:
    collective = _collective(8)
    crossover = collective.link.crossover_bytes(8)
    assert collective.select_algorithm(int(crossover // 2)) is Algorithm.ONE_SHOT
    assert collective.select_algorithm(int(crossover * 4)) is Algorithm.TWO_SHOT


def test_a_mesh_without_full_peer_access_selects_the_ring() -> None:
    """Only neighbour copies are cheap there, which is the ring's assumption."""
    collective = P2PRingCollective(
        ["cpu"] * 4, torch_module=torch, link_model=LinkModel(fully_connected=False)
    )
    assert collective.select_algorithm(1 << 20) is Algorithm.RING


def test_explicit_algorithm_overrides_selection() -> None:
    collective = _collective(4, algorithm=Algorithm.RING)
    assert collective.select_algorithm(1) is Algorithm.RING


def test_stats_report_the_algorithms_actually_used_and_no_nccl() -> None:
    collective = _collective(4)
    collective.all_reduce(_shards(4), algorithm=Algorithm.ONE_SHOT)
    collective.all_reduce(_shards(4), algorithm=Algorithm.TWO_SHOT)
    stats = collective.stats()
    assert stats["requires_nccl"] is False
    assert stats["algorithm_counts"] == {"one_shot": 1, "two_shot": 1}
    assert stats["world_size"] == 4


def test_p2p_backend_is_registered_and_labels_itself_honestly() -> None:
    from aether.parallelism.collective_backends import get_collective_backend

    backend = get_collective_backend("p2p")
    assert backend.mode == "p2p_multi_gpu_single_process"
    assert backend.production_capable is True


def test_p2p_backend_round_trips_an_all_reduce() -> None:
    from aether.parallelism.collective_backends import get_collective_backend

    backend = get_collective_backend("p2p", devices=["cpu", "cpu", "cpu", "cpu"])
    backend.initialize(rank=0, world_size=4, devices=["cpu"] * 4)
    shards = _shards(4)
    expected = torch.stack(shards).sum(0)
    for result in backend.all_reduce(shards):
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    assert backend.stats()["requires_nccl"] is False
    backend.shutdown()


def test_p2p_backend_rejects_a_device_count_mismatch() -> None:
    from aether.parallelism.collective_backends import (
        CollectiveBackendError,
        get_collective_backend,
    )

    backend = get_collective_backend("p2p")
    with pytest.raises(CollectiveBackendError, match="one device per rank"):
        backend.initialize(rank=0, world_size=4, devices=["cpu", "cpu"])
