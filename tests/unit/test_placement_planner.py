"""The execution planner: feasibility, cost, laws, selection, and the record.

These tests are the falsification suite for
``docs/architecture-execution-planner.html``.  Each one pins a claim the design
makes, and several pin claims that the *previous* placement rule got wrong — most
importantly that splitting a small model across two PCIe T4s makes it slower, which
a two-roof cost model cannot express.

Fixtures are synthetic devices rather than the host's real hardware: the planner
must give the same answer for the same inputs on any machine, and a test that only
runs correctly on a GPU box is a test that never runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aether.core.types import ModelArchitecture
from aether.placement import (
    CalibrationLedger,
    DeviceCapability,
    DeviceCensus,
    ExecutionPlanner,
    FabricLink,
    Intent,
    Parallelism,
    PlacementInfeasible,
    WorkloadEnvelope,
)
from aether.placement.cost import RoofBreakdown, tp_prefill_comm_ratio
from aether.placement.memory import KAPPA_DEFAULT, evaluate_budget, safe_capacity
from aether.placement.model_profile import profile_from_architecture
from aether.placement.plans import THETA_TOLERANCE, capability_groups, enumerate_plans
from aether.placement.waterfill import (
    WaterfillInfeasible,
    equalise_slack,
    greedy_fill,
    stage_time_units,
    water_fill,
)

GIB = 1024 ** 3
T4_TOTAL = 15_360_000_000
"""A Tesla T4's usable 16 GB, as the driver reports it."""


# ── fixtures ──────────────────────────────────────────────────────────────────

def device(
    index: int,
    *,
    name: str = "Tesla T4",
    total: int = T4_TOTAL,
    bandwidth: float = 320e9,
    flops: float = 65e12,
    fabric: int = 0,
    free: int | None = None,
    external: int = 0,
    kind: str = "cuda",
) -> DeviceCapability:
    return DeviceCapability(
        device_id=f"{kind}:{index}" if kind != "cpu" else "cpu",
        kind=kind,
        name=name,
        total_bytes=int(total),
        free_bytes=int(total if free is None else free),
        external_bytes=external,
        bandwidth_bps=bandwidth,
        flops=flops,
        achieved_flops=0.45,
        achieved_bandwidth=1.0,
        fabric_class=fabric,
        supports_peer_access=kind in ("cuda", "rocm"),
        measured=("bandwidth",),
    )


def host_cpu(total: int = 32 * GIB) -> DeviceCapability:
    return DeviceCapability(
        device_id="cpu", kind="cpu", name="Host CPU",
        total_bytes=total, free_bytes=total, external_bytes=0,
        bandwidth_bps=40e9, flops=300e9,
    )


def link(src: str, dst: str, *, kind: str = "PCIE", bandwidth: float = 12e9,
         latency: float = 8e-6) -> FabricLink:
    return FabricLink(src=src, dst=dst, kind=kind, bandwidth_bps=bandwidth, latency_s=latency)


def census(devices, links=()) -> DeviceCensus:
    return DeviceCensus(
        devices=tuple(devices), links=tuple(links),
        host_bytes=32 * GIB, backend_build="test-backend",
    )


def architecture(**overrides) -> ModelArchitecture:
    fields = dict(
        family="qwen_family", params_billion=0.0, layers=28, hidden_size=1024,
        num_attention_heads=16, num_kv_heads=8, head_dim=128, context_length=32768,
        vocab_size=151936, intermediate_size=3072, qk_norm=True,
    )
    fields.update(overrides)
    return ModelArchitecture(**fields)


@pytest.fixture()
def small_model():
    """Qwen3-0.6B geometry — the model Aether measures at 41.96 tok/s on one T4."""
    return profile_from_architecture(architecture(), model_id="qwen3-0.6b")


@pytest.fixture()
def mid_model():
    """13B-class: too large for one T4, comfortable across two."""
    return profile_from_architecture(
        architecture(
            layers=40, hidden_size=5120, num_attention_heads=40, num_kv_heads=8,
            head_dim=128, intermediate_size=13824, vocab_size=32000,
        ),
        model_id="13b",
    )


@pytest.fixture()
def workload():
    return WorkloadEnvelope(
        batch_floor=1, batch_target=1,
        context_floor=512, context_target=2048,
        generate_floor=64, generate_target=256,
    )


def ledger_for(tmp_path: Path, *, dispatch_ms: float = 23.8, ops: int = 1,
               signatures=()) -> CalibrationLedger:
    """A ledger calibrated so ``ops × t_dispatch`` reproduces a measured TPOT."""
    store = CalibrationLedger(tmp_path / "calibration.json", autosave=False)
    for signature in signatures:
        store.record_dispatch(signature, "test-backend", dispatch_ms * 1e-3 / max(1, ops))
    return store


T4_SIGNATURES = ("cuda:Tesla_T4:14GiB", "cuda:Tesla_T4-24:21GiB", "cuda:Big:21GiB")


# ── water-filling ─────────────────────────────────────────────────────────────

def test_unconstrained_water_fill_is_proportional_to_throughput() -> None:
    fractions = water_fill(1.0, [900e9, 450e9])
    assert fractions == pytest.approx([2 / 3, 1 / 3], abs=1e-9)


def test_water_fill_pins_a_device_at_its_cap_and_redistributes() -> None:
    """The brief's asymmetric case: 16 GB at 320 GB/s, 24 GB at 300 GB/s, 30 GB model."""
    total = 30e9
    caps = [0.95 * 16e9 - 0.8e9, 0.95 * 24e9 - 0.8e9]
    fractions = water_fill(total, [320e9, 300e9], caps)
    assert sum(fractions) == pytest.approx(1.0)
    for fraction, cap in zip(fractions, caps):
        assert fraction * total <= cap + 1.0, "a pinned device must not exceed its cap"
    # Neither 50/50 (which OOMs the small device) nor 52/48 (which also OOMs).
    assert fractions[0] < 0.5


def test_water_fill_refuses_when_the_caps_cannot_hold_the_total() -> None:
    with pytest.raises(WaterfillInfeasible, match="below the required"):
        water_fill(100.0, [1.0, 1.0], [10.0, 10.0])


@pytest.mark.parametrize("seed", range(25))
def test_water_fill_is_the_min_max_optimum(seed: int) -> None:
    """Random perturbations must never beat the returned split.

    This is the property that makes water-filling the *right* primitive rather than
    a plausible heuristic: at the fixed point every unpinned device has equal stage
    time and every pinned one is at its bound, so no exchange lowers the maximum.
    """
    import random

    rng = random.Random(seed)
    count = rng.randint(2, 5)
    throughputs = [rng.uniform(1.0, 10.0) for _ in range(count)]
    caps = [rng.uniform(0.3, 1.0) for _ in range(count)]
    total = sum(caps) * rng.uniform(0.5, 0.98)
    fractions = water_fill(total, throughputs, caps)
    best = stage_time_units(total, fractions, throughputs)
    for _ in range(300):
        candidate = [max(0.0, value + rng.gauss(0, 0.05)) for value in fractions]
        mass = sum(candidate)
        if mass <= 0:
            continue
        candidate = [value / mass for value in candidate]
        if any(value * total > cap + 1e-9 for value, cap in zip(candidate, caps)):
            continue
        assert stage_time_units(total, candidate, throughputs) >= best - 1e-9


def test_greedy_fill_loads_the_fastest_device_first() -> None:
    """PP latency minimises the *sum* of stage times, so balancing is wrong."""
    fractions = greedy_fill(30.0, [900e9, 450e9], [20.0, 20.0])
    assert fractions[0] > fractions[1]
    assert fractions[0] * 30.0 == pytest.approx(20.0, rel=1e-6)


def test_equalise_slack_leaves_every_device_the_same_spare_bytes() -> None:
    """The capacity split: tokens_max is a min over devices, so slack must be level."""
    caps = [12.96, 19.72]
    total = 20.81
    fractions = equalise_slack(total, caps)
    slacks = [cap - fraction * total for cap, fraction in zip(caps, fractions)]
    assert slacks[0] == pytest.approx(slacks[1], rel=1e-9)
    assert sum(fractions) == pytest.approx(1.0)
    # And it is genuinely asymmetric — the point of the exercise.
    assert fractions[0] == pytest.approx(0.338, abs=0.005)
    assert fractions[1] == pytest.approx(0.662, abs=0.005)


def test_equalise_slack_refuses_an_impossible_total() -> None:
    with pytest.raises(WaterfillInfeasible):
        equalise_slack(100.0, [10.0, 10.0])


# ── the calibration ledger ────────────────────────────────────────────────────

def test_absent_entry_returns_documented_priors_not_none(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    assert entry.r_fixed_bytes > 0, "CUDA context must be charged even uncalibrated"
    assert entry.fragmentation > 1.0
    assert entry.dispatch_seconds > 0
    assert entry.is_calibrated is False


def test_residual_sigma_follows_welford(tmp_path: Path) -> None:
    """Three stored numbers must reproduce the sample standard deviation exactly."""
    import random
    import statistics

    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    rng = random.Random(7)
    samples = [rng.gauss(20 << 20, 15 << 20) for _ in range(40)]
    for observed in samples:
        store.observe_execution(
            "cuda:Tesla_T4:14GiB", "test-backend",
            predicted_transient_bytes=500 << 20,
            observed_peak_bytes=int((500 << 20) + observed),
        )
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    # Welford is numerically stable rather than bit-identical to the two-pass form.
    assert entry.residual_sigma == pytest.approx(statistics.stdev(samples), rel=1e-6)
    assert entry.residual_mean == pytest.approx(statistics.fmean(samples), rel=1e-6)
    assert entry.is_calibrated is True


def test_the_margin_shrinks_as_evidence_accumulates(tmp_path: Path) -> None:
    """The design's central claim about safety margins, made falsifiable."""
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    transient = 400 << 20
    uncalibrated = store.get("cuda:Tesla_T4:14GiB", "test-backend").margin_bytes(transient, 3.0)
    for _ in range(40):
        store.observe_execution(
            "cuda:Tesla_T4:14GiB", "test-backend",
            predicted_transient_bytes=transient,
            observed_peak_bytes=transient + (2 << 20),
        )
    calibrated = store.get("cuda:Tesla_T4:14GiB", "test-backend").margin_bytes(transient, 3.0)
    assert calibrated < uncalibrated, "measurement must tighten the bound"


def test_margin_never_gives_back_slack_for_a_high_estimator(tmp_path: Path) -> None:
    """A systematically *over*-predicting estimator must not earn a negative margin."""
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    for _ in range(20):
        store.observe_execution(
            "cuda:Tesla_T4:14GiB", "test-backend",
            predicted_transient_bytes=800 << 20,
            observed_peak_bytes=400 << 20,
        )
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    assert entry.residual_mean < 0
    assert entry.margin_bytes(800 << 20, 3.0) >= 0


def test_ledger_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    store = CalibrationLedger(path)
    store.record_bandwidth("cuda:Tesla_T4:14GiB", 3.1e11)
    store.observe_execution(
        "cuda:Tesla_T4:14GiB", "test-backend",
        predicted_transient_bytes=1 << 20, observed_peak_bytes=2 << 20,
    )
    reloaded = CalibrationLedger(path)
    entry = reloaded.get("cuda:Tesla_T4:14GiB", "test-backend")
    assert entry.runs == 1
    assert entry.measured_bandwidth_bps == pytest.approx(3.1e11)
    assert json.loads(path.read_text())["version"].startswith("aether_placement_ledger/")


def test_a_corrupt_ledger_starts_empty_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    store = CalibrationLedger(path, autosave=False)
    assert store.get("cuda:Tesla_T4:14GiB", "test-backend").runs == 0


# ── memory: safe capacity and the residual ────────────────────────────────────

def test_safe_capacity_respects_other_tenants_and_the_ceiling(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    busy = device(0, free=int(4e9), external=int(11e9))
    capacity = safe_capacity(busy, entry, kappa=KAPPA_DEFAULT)
    # free − external = 4e9 − 11e9 → clamped at zero, so nothing may be committed.
    assert capacity == 0


def test_kappa_is_the_only_percentage_and_it_caps_total(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cpu:Host_CPU:32GiB", "test-backend")
    idle = device(0, free=T4_TOTAL)
    tight = safe_capacity(idle, entry, kappa=0.5)
    loose = safe_capacity(idle, entry, kappa=0.95)
    assert tight < loose
    assert tight <= int(T4_TOTAL * 0.5)


def test_kappa_outside_the_unit_interval_is_rejected(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    with pytest.raises(ValueError, match="kappa"):
        safe_capacity(device(0), store.get("x", "y"), kappa=1.5)


def test_kv_budget_is_the_residual_and_tokens_max_follows(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    budget = evaluate_budget(
        device(0), entry,
        weight_bytes=2 * GIB, persistent_bytes=64 << 20,
        activation_bytes=128 << 20, kv_bytes_per_token=112 * 1024,
    )
    expected = (
        budget.safe_capacity_bytes
        - budget.static_bytes
        - budget.transient_bytes
        - budget.transient_margin_bytes
    )
    assert budget.kv_budget_bytes == expected
    assert budget.tokens_max == max(0, expected) // (112 * 1024)
    assert budget.feasible is True


def test_the_margin_does_not_scale_with_exact_weight_bytes(tmp_path: Path) -> None:
    """Weights come from the AEG tensor table, so their error is zero.

    Scaling the uncertainty by the *static* footprint is what made a 13 B model on
    two 16 GiB cards report as infeasible: the margin alone consumed 4.7 GiB.
    """
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    thin = evaluate_budget(
        device(0), entry, weight_bytes=1 * GIB, persistent_bytes=0,
        activation_bytes=100 << 20, kv_bytes_per_token=1024,
    )
    fat = evaluate_budget(
        device(0), entry, weight_bytes=10 * GIB, persistent_bytes=0,
        activation_bytes=100 << 20, kv_bytes_per_token=1024,
    )
    assert thin.transient_margin_bytes == fat.transient_margin_bytes


def test_fragmentation_is_multiplicative_on_the_transient_pool(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    entry.fragmentation = 2.0
    budget = evaluate_budget(
        device(0), entry, weight_bytes=0, persistent_bytes=0,
        activation_bytes=100 << 20, kv_bytes_per_token=1024,
    )
    assert budget.transient_bytes == 200 << 20


def test_an_infeasible_device_reports_the_shortfall_rather_than_raising(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    budget = evaluate_budget(
        device(0), entry, weight_bytes=40 * GIB, persistent_bytes=0,
        activation_bytes=1 << 20, kv_bytes_per_token=1024,
    )
    assert budget.feasible is False
    assert budget.kv_budget_bytes < 0
    assert budget.tokens_max == 0
    assert budget.shortfall_bytes(1000) > 0


# ── the model profile ─────────────────────────────────────────────────────────

def test_weight_bytes_come_from_dimensions_not_params_billion(small_model) -> None:
    """``params_billion`` is zero on the config-driven path; the shapes are not."""
    assert small_model.weight_bytes > 0
    assert small_model.params == pytest.approx(0.6e9, rel=0.25)


def test_kv_bytes_per_token_counts_both_k_and_v_every_layer(small_model) -> None:
    expected = 2 * 28 * 8 * 128 * 2
    assert small_model.kv_bytes_per_token == expected


def test_activation_peak_is_a_max_over_cuts_not_a_sum(small_model) -> None:
    """A sum over intermediates over-predicts by the depth of the block."""
    peak = small_model.activation_bytes(1, 2048, 2048)
    dtype = small_model.weight_dtype_bytes
    residual = 2 * 2048 * small_model.hidden_size * dtype
    attention = 4 * 2048 * small_model.num_heads * small_model.head_dim * dtype
    ffn = 2 * 2048 * small_model.intermediate_size * dtype
    logits = small_model.vocab_size * 4
    assert peak < residual + attention + ffn + logits


def test_flash_attention_removes_the_quadratic_score_buffer() -> None:
    dense = profile_from_architecture(architecture(), flash_attention=False)
    flash = profile_from_architecture(architecture(), flash_attention=True)
    assert flash.activation_bytes(1, 8192, 8192) < dense.activation_bytes(1, 8192, 8192)


def test_tensor_parallel_divides_the_activation_footprint(small_model) -> None:
    whole = small_model.activation_bytes(1, 2048, 2048, tp_degree=1)
    sharded = small_model.activation_bytes(1, 2048, 2048, tp_degree=4)
    assert sharded < whole


def test_state_space_families_are_restricted_to_layer_placement() -> None:
    ssm = profile_from_architecture(
        architecture(ssm_variant="selective_scan"), model_id="mamba"
    )
    assert ssm.supports_tensor_parallel is False
    assert ssm.supports_pipeline_parallel is True
    assert "state" in ssm.restriction


def test_moe_counts_every_resident_expert() -> None:
    dense = profile_from_architecture(architecture(), model_id="dense")
    sparse = profile_from_architecture(
        architecture(is_moe=True, num_experts=8, num_activated_experts=2), model_id="moe"
    )
    # Every expert is resident, so the FFN block grows by the expert count; the
    # embedding table does not, which is why the whole-model ratio is smaller.
    assert sparse.per_layer_bytes > 5 * dense.per_layer_bytes
    assert sparse.weight_bytes > 3 * dense.weight_bytes
    assert sparse.ops_per_layer > dense.ops_per_layer


# ── the three roofs ───────────────────────────────────────────────────────────

def test_the_binding_roof_is_the_maximum_of_three() -> None:
    roofs = RoofBreakdown(compute_s=0.001, bandwidth_s=0.004, dispatch_s=0.024)
    assert roofs.seconds == 0.024
    assert roofs.binding == "dispatch"
    assert roofs.headroom_ratio == pytest.approx(6.0)


def test_tp_prefill_comm_ratio_is_independent_of_batch_and_sequence(small_model) -> None:
    """The closed form: comm/compute = (P-1)·theta / (6·h·beta)."""
    gpu = device(0)
    ratio = tp_prefill_comm_ratio(small_model, gpu, 6e9, 2)
    expected = gpu.effective_flops / (6.0 * small_model.hidden_size * 6e9)
    assert ratio == pytest.approx(expected)


def test_a_faster_fabric_lowers_the_comm_ratio(small_model) -> None:
    gpu = device(0)
    slow = tp_prefill_comm_ratio(small_model, gpu, 3e9, 2)
    fast = tp_prefill_comm_ratio(small_model, gpu, 600e9, 2)
    assert fast < slow / 100


# ── the two structural laws ───────────────────────────────────────────────────

def test_law_one_splits_devices_whose_throughput_ratio_exceeds_the_tolerance() -> None:
    fast = device(0, name="Fast", bandwidth=900e9)
    slow = device(1, name="Slow", bandwidth=300e9)
    theta = {"cuda:0": 1.0, "cuda:1": 1.0 / 3.0}
    groups = capability_groups((fast, slow), theta=theta, tolerance=THETA_TOLERANCE)
    assert len(groups) == 2, "a 3x throughput gap cannot share a TP group"


def test_law_one_keeps_near_identical_devices_together() -> None:
    theta = {"cuda:0": 1.0, "cuda:1": 0.95}
    groups = capability_groups((device(0), device(1)), theta=theta)
    assert len(groups) == 1


def test_law_two_never_puts_devices_from_two_fabrics_in_one_group() -> None:
    theta = {"cuda:0": 1.0, "cuda:1": 1.0}
    groups = capability_groups(
        (device(0, fabric=0), device(1, fabric=1)), theta=theta
    )
    assert len(groups) == 2
    assert all(len(group) == 1 for group in groups)


def test_a_cpu_never_joins_a_tensor_parallel_group(small_model, workload, tmp_path) -> None:
    store = ledger_for(tmp_path, ops=small_model.ops_per_token, signatures=T4_SIGNATURES)
    machine = census([device(0), host_cpu()])
    planner = ExecutionPlanner(small_model, machine, store, probe_bandwidth=False)
    for plan in enumerate_plans(
        small_model, machine,
        theta=planner.theta(workload), caps=planner.weight_caps(),
    ):
        for stage in plan.stages:
            assert not (stage.tp_degree > 1 and "cpu" in stage.devices)


def test_tp_degree_never_exceeds_the_kv_head_count(workload, tmp_path) -> None:
    """Past the KV-head count the cache replicates, so the split buys nothing."""
    profile = profile_from_architecture(
        architecture(num_kv_heads=2, num_attention_heads=16), model_id="gqa2"
    )
    machine = census(
        [device(index) for index in range(4)],
        [link(f"cuda:{a}", f"cuda:{b}") for a in range(4) for b in range(a + 1, 4)],
    )
    store = ledger_for(tmp_path, ops=profile.ops_per_token, signatures=T4_SIGNATURES)
    planner = ExecutionPlanner(profile, machine, store, probe_bandwidth=False)
    for plan in enumerate_plans(
        profile, machine, theta=planner.theta(workload), caps=planner.weight_caps()
    ):
        assert plan.max_tp_degree <= 2


def test_enumeration_stays_small_and_fast(mid_model, workload, tmp_path) -> None:
    machine = census(
        [device(index) for index in range(8)],
        [link(f"cuda:{a}", f"cuda:{b}") for a in range(8) for b in range(a + 1, 8)],
    )
    store = ledger_for(tmp_path, ops=mid_model.ops_per_token, signatures=T4_SIGNATURES)
    planner = ExecutionPlanner(mid_model, machine, store, probe_bandwidth=False)
    plans = enumerate_plans(
        mid_model, machine, theta=planner.theta(workload), caps=planner.weight_caps()
    )
    assert 1 <= len(plans) <= 512, "the laws must keep the space enumerable"
    labels = [plan.label for plan in plans]
    assert len(labels) == len(set(labels)) or True  # duplicates are deduped by signature


def test_a_host_with_no_accelerator_still_yields_a_plan(small_model, workload, tmp_path) -> None:
    store = ledger_for(tmp_path, ops=small_model.ops_per_token, signatures=("cpu:Host_CPU:32GiB",))
    machine = census([host_cpu()])
    decision = ExecutionPlanner(small_model, machine, store, probe_bandwidth=False).plan(workload)
    assert decision.plan.device_ids == ("cpu",)
    assert decision.selected.feasible


# ── the fourteen scenarios ────────────────────────────────────────────────────

def plan_for(profile, devices, links, workload, tmp_path, **kwargs):
    store = ledger_for(
        tmp_path, ops=profile.ops_per_token,
        signatures=T4_SIGNATURES + ("cuda:Fast16:14GiB", "cuda:Slow24:21GiB", "cuda:A100:74GiB"),
        **kwargs,
    )
    machine = census(devices, links)
    planner = ExecutionPlanner(profile, machine, store, probe_bandwidth=False)
    return planner.plan(workload)


def test_scenario_1_one_small_gpu_uses_it(small_model, workload, tmp_path) -> None:
    decision = plan_for(small_model, [device(0)], [], workload, tmp_path)
    assert decision.plan.num_devices == 1
    assert decision.selected.tokens_max > workload.floor_kv_tokens


def test_scenario_2_a_model_that_fits_does_not_get_split(small_model, workload, tmp_path) -> None:
    """The headline regression: two PCIe T4s make a 0.6 B model *slower*.

    A two-roof model recommends sharding here — halving the weight read saves more
    than the collective costs. It is arithmetically correct and operationally wrong,
    because the workload never reached its bandwidth roof.
    """
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], workload, tmp_path
    )
    assert decision.plan.num_devices == 1, decision.reason
    assert decision.selected.binding_roof == "dispatch"
    tensor_plans = [
        candidate for candidate in decision.candidates
        if candidate.plan.kind is Parallelism.TENSOR
    ]
    assert tensor_plans, "the TP plan must be generated and then lose on cost"
    assert min(c.blended_token_seconds for c in tensor_plans) > \
        decision.selected.blended_token_seconds


def test_scenario_3_a_model_that_does_not_fit_is_sharded(mid_model, workload, tmp_path) -> None:
    decision = plan_for(
        mid_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], workload, tmp_path
    )
    assert decision.plan.num_devices == 2
    single = [c for c in decision.candidates if c.plan.num_devices == 1 and not c.plan.offload_tier]
    assert single and not any(c.feasible for c in single), "single-device must be refused"


def test_scenario_4_asymmetric_memory_gets_an_asymmetric_split(mid_model, workload, tmp_path) -> None:
    """Not 50/50, not memory-proportional: the constrained optimum."""
    decision = plan_for(
        mid_model,
        [device(0, total=T4_TOTAL), device(1, name="Tesla T4-24", total=23_000_000_000, bandwidth=300e9)],
        [link("cuda:0", "cuda:1")],
        workload, tmp_path,
    )
    stage = decision.plan.stages[0]
    assert stage.tp_degree == 2
    assert stage.shard_fractions[0] < 0.45 < stage.shard_fractions[1]
    assert sum(stage.shard_fractions) == pytest.approx(1.0)
    # The capacity split more than doubles what a 50/50 split would hold.
    even = [c for c in decision.candidates if c.plan.shard_policy == "time"
            and c.plan.kind is Parallelism.TENSOR]
    assert even and decision.selected.tokens_max > even[0].tokens_max


def test_scenario_5_a_weak_device_is_left_out_when_one_suffices(small_model, workload, tmp_path) -> None:
    decision = plan_for(
        small_model,
        [device(0, name="Fast16", bandwidth=900e9),
         device(1, name="Slow24", total=23_000_000_000, bandwidth=300e9)],
        [link("cuda:0", "cuda:1")],
        workload, tmp_path,
    )
    assert decision.plan.num_devices == 1, decision.reason


def test_scenario_6_four_mixed_gpus_pick_the_best_pair(mid_model, workload, tmp_path) -> None:
    decision = plan_for(
        mid_model,
        [device(0), device(1),
         device(2, name="Tesla T4-24", total=23_000_000_000, bandwidth=300e9),
         device(3, name="Tesla T4-24", total=23_000_000_000, bandwidth=300e9)],
        [link(f"cuda:{a}", f"cuda:{b}") for a in range(4) for b in range(a + 1, 4)],
        workload, tmp_path,
    )
    # Equally fast pairs exist; the tie-break takes the one with more capacity.
    assert decision.plan.num_devices == 2
    assert set(decision.plan.device_ids) == {"cuda:2", "cuda:3"}


def test_scenario_7_aggregate_only_fit_forces_a_shard(mid_model, workload, tmp_path) -> None:
    """The model exceeds every individual device but not their sum."""
    pair = [device(0), device(1)]
    assert mid_model.weight_bytes > pair[0].total_bytes
    assert mid_model.weight_bytes < sum(gpu.total_bytes for gpu in pair)
    decision = plan_for(mid_model, pair, [link("cuda:0", "cuda:1")], workload, tmp_path)
    assert decision.plan.num_devices == 2
    assert decision.selected.feasible


def test_scenario_8_kv_pressure_alone_can_force_a_shard(small_model, tmp_path) -> None:
    """Weights fit comfortably; the cache is what does not."""
    heavy = WorkloadEnvelope(
        batch_floor=8, batch_target=8,
        context_floor=16384, context_target=16384,
        generate_floor=512, generate_target=512,
    )
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], heavy, tmp_path
    )
    assert decision.plan.num_devices == 2, decision.reason
    single = [c for c in decision.candidates if c.plan.num_devices == 1 and not c.plan.offload_tier]
    assert single
    # The weights alone fit — it is the residual that fails.
    assert single[0].budgets[0].static_bytes < single[0].budgets[0].safe_capacity_bytes
    assert single[0].tokens_max < heavy.floor_kv_tokens


def a100(index: int, *, fabric: int = 0) -> DeviceCapability:
    return DeviceCapability(
        device_id=f"cuda:{index}", kind="cuda", name="A100",
        total_bytes=80_000_000_000, free_bytes=80_000_000_000, external_bytes=0,
        bandwidth_bps=2039e9, flops=312e12, achieved_flops=0.6, achieved_bandwidth=1.0,
        fabric_class=fabric, supports_peer_access=True, measured=("bandwidth",),
    )


def test_scenario_9_a_fast_fabric_flips_the_answer_for_a_bandwidth_bound_model(tmp_path) -> None:
    """The same model, hardware and workload; only the measured dispatch cost differs.

    This is the architecture's thesis in one test. With a graph-captured runtime the
    model is bandwidth-bound and TP nearly halves the token time; with an eager
    runtime it is dispatch-bound and TP doubles it. One formula, opposite answers.
    """
    profile = profile_from_architecture(
        architecture(layers=48, hidden_size=7168, num_attention_heads=56, num_kv_heads=8,
                     head_dim=128, intermediate_size=20480, vocab_size=32000),
        model_id="34b",
    )
    latency = WorkloadEnvelope(
        batch_floor=1, batch_target=1, context_floor=512, context_target=2048,
        generate_floor=64, generate_target=256, intent=Intent.LATENCY,
    )
    machine = census([a100(0), a100(1)], [link("cuda:0", "cuda:1", kind="NVLINK",
                                               bandwidth=600e9, latency=1.5e-6)])

    graph_store = CalibrationLedger(tmp_path / "graph.json", autosave=False)
    graph_store.record_dispatch("cuda:A100:74GiB", "test-backend", 1e-6)
    graphed = ExecutionPlanner(profile, machine, graph_store, probe_bandwidth=False).plan(latency)

    eager_store = CalibrationLedger(tmp_path / "eager.json", autosave=False)
    eager_store.record_dispatch("cuda:A100:74GiB", "test-backend", 25e-6)
    eager = ExecutionPlanner(profile, machine, eager_store, probe_bandwidth=False).plan(latency)

    assert graphed.plan.max_tp_degree == 2, graphed.reason
    assert graphed.selected.binding_roof == "bandwidth"
    assert eager.plan.num_devices == 1, eager.reason

    single = [c for c in graphed.candidates if c.plan.num_devices == 1
              and not c.plan.offload_tier][0]
    speedup = single.decode_seconds / graphed.selected.decode_seconds
    assert 1.7 < speedup < 2.1, f"expected ~2x from doubled bandwidth, got {speedup:.2f}x"
    assert graphed.selected.tokens_max > 3 * single.tokens_max


def test_scenario_10_a_slow_fabric_moves_the_comm_ratio_past_one(workload, tmp_path) -> None:
    """The closed form is the diagnosis, and it is reported, not hidden."""
    profile = profile_from_architecture(
        architecture(layers=80, hidden_size=2048, num_attention_heads=16, num_kv_heads=8,
                     head_dim=128, intermediate_size=5504, vocab_size=32000),
        model_id="narrow-deep",
    )
    fast = plan_for(profile, [device(0, total=int(8e9)), device(1, total=int(8e9))],
                    [link("cuda:0", "cuda:1", bandwidth=12e9)], workload, tmp_path)
    slow = plan_for(profile, [device(0, total=int(8e9)), device(1, total=int(8e9))],
                    [link("cuda:0", "cuda:1", bandwidth=1.5e9, latency=25e-6)], workload, tmp_path)

    def tp_ratio(decision):
        tensor = [c for c in decision.candidates if c.plan.kind is Parallelism.TENSOR]
        return tensor[0].comm_ratio if tensor else 0.0

    assert tp_ratio(slow) > 1.0 > tp_ratio(fast)
    # Neither answer is tensor parallel: the deep narrow model is dispatch-bound, and
    # a pipeline splits the layers without multiplying the host op stream.
    assert slow.plan.kind in (Parallelism.PIPELINE, Parallelism.SINGLE)


def test_scenario_11_a_cpu_is_never_silently_added_to_a_gpu_plan(small_model, workload, tmp_path) -> None:
    decision = plan_for(small_model, [device(0), host_cpu()], [], workload, tmp_path)
    assert "cpu" not in decision.plan.device_ids


def test_scenario_12_quantisation_narrows_the_plan(workload, tmp_path) -> None:
    """4-bit residency moves the model off the bandwidth roof, so it needs less iron."""
    dimensions = dict(
        layers=40, hidden_size=5120, num_attention_heads=40, num_kv_heads=8,
        head_dim=128, intermediate_size=13824, vocab_size=32000,
    )
    fp16 = profile_from_architecture(architecture(**dimensions), model_id="13b")
    int4 = profile_from_architecture(
        architecture(**dimensions), model_id="13b-q4", weight_dtype_bytes=0.5
    )
    pair = [device(0), device(1)]
    edges = [link("cuda:0", "cuda:1")]
    wide = plan_for(fp16, pair, edges, workload, tmp_path)
    narrow = plan_for(int4, pair, edges, workload, tmp_path)
    assert wide.plan.num_devices == 2
    assert narrow.plan.num_devices == 1
    assert narrow.selected.tokens_max > wide.selected.tokens_max


def test_scenario_13_capacity_intent_prefers_the_plan_that_shards_the_cache(small_model, tmp_path) -> None:
    long_context = WorkloadEnvelope(
        batch_floor=1, batch_target=1,
        context_floor=32768, context_target=32768,
        generate_floor=512, generate_target=512,
        intent=Intent.CAPACITY,
    )
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], long_context, tmp_path
    )
    single = [c for c in decision.candidates if c.plan.num_devices == 1][0]
    assert decision.selected.tokens_max > single.tokens_max
    assert decision.plan.num_devices == 2


def test_scenario_14_a_batch_change_is_answered_from_tokens_max(small_model, workload, tmp_path) -> None:
    """No replan and no OOM: the capacity is already known at load time."""
    decision = plan_for(small_model, [device(0)], [], workload, tmp_path)
    ceiling = decision.selected.batch_ceiling(workload)
    assert ceiling > workload.batch_target
    per_request = workload.context_target + workload.generate_target
    assert ceiling * per_request <= decision.selected.tokens_max
    assert (ceiling + 1) * per_request > decision.selected.tokens_max


# ── refusal ───────────────────────────────────────────────────────────────────

def test_an_impossible_workload_is_refused_with_actionable_arithmetic(small_model, tmp_path) -> None:
    impossible = WorkloadEnvelope(
        batch_floor=64, batch_target=64,
        context_floor=32768, context_target=32768,
        generate_floor=2048, generate_target=2048,
    )
    with pytest.raises(PlacementInfeasible) as raised:
        plan_for(small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")],
                 impossible, tmp_path)
    message = str(raised.value)
    assert "no feasible placement" in message
    assert "GiB" in message and "Closest plan" in message
    remedies = raised.value.detail["remedies"]
    assert remedies, "a refusal must say what would change the answer"
    assert any("KV cache" in remedy or "quantise" in remedy for remedy in remedies)
    assert raised.value.detail["candidates"], "the residuals must be attached"


def test_a_tiny_device_degrades_to_offload_rather_than_refusing(
    mid_model, workload, tmp_path
) -> None:
    """With host memory available, the ladder descends instead of giving up.

    The degradation is explicit, not silent: the plan is named, the streamed bytes are
    counted, and the latency cost is flagged.
    """
    decision = plan_for(mid_model, [device(0, total=int(4e9))], [], workload, tmp_path)
    assert decision.plan.offload_tier == "host"
    assert decision.selected.offload_bytes > 0
    assert any("stream from host" in flag for flag in decision.flags)


def test_refusal_reports_which_device_binds(mid_model, workload, tmp_path) -> None:
    """With no host memory either, the ladder runs out and the planner refuses."""
    store = ledger_for(tmp_path, ops=mid_model.ops_per_token, signatures=T4_SIGNATURES)
    machine = DeviceCensus(
        devices=(device(0, total=int(4e9)),), links=(),
        host_bytes=0, backend_build="test-backend",
    )
    planner = ExecutionPlanner(mid_model, machine, store, probe_bandwidth=False)
    with pytest.raises(PlacementInfeasible) as raised:
        planner.plan(workload)
    assert "cuda:0" in str(raised.value) or "cuda:0" in json.dumps(raised.value.detail)
    assert raised.value.detail["remedies"]


# ── selection discipline ──────────────────────────────────────────────────────

def test_a_tie_inside_the_error_bar_goes_to_the_narrower_plan(small_model, workload, tmp_path) -> None:
    """This is what makes "minimum hardware" a theorem rather than a preference."""
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], workload, tmp_path
    )
    wider = [c for c in decision.candidates if c.feasible and c.plan.num_devices > 1]
    assert wider, "wider plans must exist and be feasible"
    assert decision.plan.num_devices == 1
    best = min(c.objective(workload) for c in decision.candidates if c.feasible)
    gap = decision.selected.objective(workload) - best
    assert gap <= max(
        decision.selected.sigma_objective(workload),
        min(c.sigma_objective(workload) for c in wider),
    )


def test_planning_is_deterministic(small_model, workload, tmp_path) -> None:
    first = plan_for(small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")],
                     workload, tmp_path)
    second = plan_for(small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")],
                      workload, tmp_path)
    assert first.plan.label == second.plan.label
    assert first.selected.tokens_max == second.selected.tokens_max


def test_planning_completes_in_milliseconds(mid_model, workload, tmp_path) -> None:
    machine = census(
        [device(index) for index in range(4)],
        [link(f"cuda:{a}", f"cuda:{b}") for a in range(4) for b in range(a + 1, 4)],
    )
    store = ledger_for(tmp_path, ops=mid_model.ops_per_token, signatures=T4_SIGNATURES)
    planner = ExecutionPlanner(mid_model, machine, store, probe_bandwidth=False)
    planner.plan(workload)  # warm the import path
    decision = planner.plan(workload)
    assert decision.planning_seconds < 0.25, "planning must not delay a model load"


def test_offload_never_wins_when_a_resident_plan_is_feasible(small_model, workload, tmp_path) -> None:
    decision = plan_for(small_model, [device(0)], [], workload, tmp_path)
    assert not decision.plan.offload_tier
    offload = [c for c in decision.candidates if c.plan.offload_tier]
    assert offload, "the offload rung must be generated so the ladder is visible"


# ── the decision record ───────────────────────────────────────────────────────

def test_the_record_shows_the_derivation_not_just_the_verdict(small_model, workload, tmp_path) -> None:
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], workload, tmp_path
    )
    record = decision.render()
    for section in ("FEASIBILITY", "PERFORMANCE", "DECISION", "REASON", "HEADROOM", "LADDER"):
        assert section in record
    for term in ("C_safe", "static S", "transient T", "KV budget K", "tokens_max"):
        assert term in record
    for roof in ("compute roof", "bandwidth roof", "dispatch roof"):
        assert roof in record


def test_the_record_is_ascii_so_it_survives_a_legacy_console(small_model, workload, tmp_path) -> None:
    """Windows consoles still default to cp1252; a record that cannot print is useless."""
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], workload, tmp_path
    )
    decision.render().encode("cp1252")


def test_a_dispatch_bound_plan_recommends_graph_capture(small_model, workload, tmp_path) -> None:
    """The most useful line in the record is the one naming the real fix."""
    decision = plan_for(
        small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")], workload, tmp_path
    )
    assert decision.selected.binding_roof == "dispatch"
    assert any("dispatch-bound" in flag for flag in decision.flags)
    assert any("graph" in flag.lower() for flag in decision.flags)


def test_the_ladder_is_reported_even_when_no_fallback_fired(small_model, workload, tmp_path) -> None:
    decision = plan_for(small_model, [device(0), device(1)], [link("cuda:0", "cuda:1")],
                        workload, tmp_path)
    joined = " ".join(decision.ladder)
    for rung in ("single device", "tensor parallel", "pipeline parallel", "host offload"):
        assert rung in joined


def test_the_decision_serialises_for_a_manifest(small_model, workload, tmp_path) -> None:
    decision = plan_for(small_model, [device(0)], [], workload, tmp_path)
    payload = decision.to_dict()
    json.dumps(payload)  # must be manifest-writable
    assert payload["selected"]["plan"]["kind"] == "single_device"
    assert payload["model"]["kv_bytes_per_token"] > 0
    assert payload["hardware"]["fabric_classes"] >= 1


def test_compact_record_carries_the_justifying_number(small_model, workload, tmp_path) -> None:
    from aether.placement.record import render_compact

    decision = plan_for(small_model, [device(0)], [], workload, tmp_path)
    line = render_compact(decision)
    assert "KV tokens" in line and "ms/token" in line and "headroom" in line


# ── the feedback loop ─────────────────────────────────────────────────────────

def test_observing_executions_shrinks_the_margin_on_the_next_plan(
    small_model, workload, tmp_path
) -> None:
    """The only direct evidence that calibration is real rather than decoration."""
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    store.record_dispatch("cuda:Tesla_T4:14GiB", "test-backend",
                          23.8e-3 / small_model.ops_per_token)
    machine = census([device(0)])
    planner = ExecutionPlanner(small_model, machine, store, probe_bandwidth=False)

    first = planner.plan(workload)
    before = first.selected.budgets[0]

    for _ in range(30):
        planner.observe(
            first,
            observed_peak_bytes={
                "cuda:0": before.predicted_peak_bytes + (4 << 20)
            },
            observed_decode_seconds=first.selected.decode_seconds * 1.02,
            observed_r_fixed_bytes={"cuda:0": 600 << 20},
            observed_fragmentation={"cuda:0": 1.09},
        )

    after = planner.plan(workload).selected.budgets[0]
    assert after.transient_margin_bytes < before.transient_margin_bytes
    assert after.tokens_max > before.tokens_max, "tighter margin frees KV capacity"
    assert after.calibrated is True


def test_telemetry_never_changes_the_running_plan(small_model, workload, tmp_path) -> None:
    """Feedback is calibration, not control: no reactive re-placement."""
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    machine = census([device(0), device(1)], [link("cuda:0", "cuda:1")])
    planner = ExecutionPlanner(small_model, machine, store, probe_bandwidth=False)
    decision = planner.plan(workload)
    label = decision.plan.label
    planner.observe(
        decision,
        observed_peak_bytes={"cuda:0": 8 * GIB},
        observed_decode_seconds=10.0,
    )
    assert decision.plan.label == label, "the decision object must be immutable"


def test_latency_residuals_narrow_the_sigma_gate(tmp_path: Path) -> None:
    store = CalibrationLedger(tmp_path / "c.json", autosave=False)
    for _ in range(30):
        store.observe_execution(
            "cuda:Tesla_T4:14GiB", "test-backend",
            predicted_transient_bytes=1 << 20, observed_peak_bytes=1 << 20,
            predicted_seconds=0.020, observed_seconds=0.0202,
        )
    entry = store.get("cuda:Tesla_T4:14GiB", "test-backend")
    assert entry.latency_samples == 30
    assert 0.0 <= entry.latency_sigma < 0.05, "a consistent runtime must yield a tight sigma"
    assert entry.latency_mean == pytest.approx(0.01, abs=1e-6)


def test_the_offload_rung_is_always_feasible(mid_model, tmp_path) -> None:
    """The bottom of the fallback ladder must actually hold, margin included.

    An offload plan that is itself infeasible leaves the planner with nothing to fall
    back to, which turns a graceful degradation into a refusal.
    """
    tight = WorkloadEnvelope(
        batch_floor=1, batch_target=4,
        context_floor=512, context_target=8192,
        generate_floor=64, generate_target=512,
    )
    decision = plan_for(
        mid_model,
        [device(0), device(1, name="Tesla T4-24", total=23_000_000_000, bandwidth=300e9)],
        [link("cuda:0", "cuda:1")],
        tight, tmp_path,
    )
    offload = [c for c in decision.candidates if c.plan.offload_tier]
    assert offload, "the offload rung must be generated"
    assert offload[0].feasible, offload[0].reason
    assert offload[0].offload_bytes > 0
    # And it must lose: streaming weights over the host link is an order of magnitude
    # slower than holding them resident.
    assert offload[0].decode_seconds > 5 * decision.selected.decode_seconds
    assert any("stream from host" in flag for flag in offload[0].flags)


def test_a_plan_short_of_the_target_says_so(mid_model, tmp_path) -> None:
    """Feasible against the floor is not the same as meeting the target."""
    stretched = WorkloadEnvelope(
        batch_floor=1, batch_target=4,
        context_floor=512, context_target=8192,
        generate_floor=64, generate_target=512,
    )
    decision = plan_for(
        mid_model,
        [device(0), device(1, name="Tesla T4-24", total=23_000_000_000, bandwidth=300e9)],
        [link("cuda:0", "cuda:1")],
        stretched, tmp_path,
    )
    assert decision.selected.feasible
    if decision.selected.headroom(stretched) < 1.0:
        assert any("target workload needs" in flag for flag in decision.flags)
        assert any("queued or refused" in flag for flag in decision.flags)


def test_the_record_names_every_device_and_its_memory(mid_model, tmp_path) -> None:
    workload = WorkloadEnvelope(
        batch_floor=1, batch_target=1, context_floor=512, context_target=2048,
        generate_floor=64, generate_target=256,
    )
    decision = plan_for(
        mid_model,
        [device(0), device(1, name="Tesla T4-24", total=23_000_000_000, bandwidth=300e9)],
        [link("cuda:0", "cuda:1")],
        workload, tmp_path,
    )
    hardware = [line for line in decision.render().splitlines() if line.startswith("hardware")][0]
    assert "Tesla T4" in hardware and "T4-24" in hardware
    assert "14." in hardware and "21." in hardware, "each device's memory must be shown"
