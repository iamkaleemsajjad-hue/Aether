"""Tests for the multi-engine benchmark suite.

These target the properties that make the suite trustworthy rather than its
plumbing: that a missing measurement can never become a zero, that a comparison
Aether loses is produced by the same code path as one it wins, and that the
representation label is attached wherever the two sides are not holding the same
weights.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from benchmark.suite import analysis, charts, hardware, plan, report
from benchmark.suite import engines as registry
from benchmark.suite import status as status_mod

# ── Status vocabulary ───────────────────────────────────────────────────────

def test_missing_measurement_is_never_measured() -> None:
    """Every non-MEASURED status must fail the predicate aggregation gates on."""
    for value in status_mod.ALL_STATUSES:
        assert status_mod.is_measured(value) is (value == status_mod.MEASURED)
    assert status_mod.is_measured({"status": status_mod.OOM}) is False
    assert status_mod.is_measured(None) is False
    assert status_mod.is_measured({}) is False


def test_exceptions_classify_into_distinguishable_statuses() -> None:
    """OOM, unsupported and a plain defect must not collapse into one label."""
    from benchmark.backends import UnsupportedConfiguration

    assert status_mod.from_exception(UnsupportedConfiguration("no"))[0] == \
        status_mod.NOT_SUPPORTED
    assert status_mod.from_exception(RuntimeError("CUDA out of memory"))[0] == \
        status_mod.OOM
    assert status_mod.from_exception(MemoryError())[0] == status_mod.OOM
    assert status_mod.from_exception(ImportError("no module"))[0] == \
        status_mod.NOT_INSTALLED
    assert status_mod.from_exception(ValueError("boom"))[0] == status_mod.FAILED


def test_runner_statuses_map_onto_the_suite_vocabulary() -> None:
    assert status_mod.from_runner({"status": "ok"}) == status_mod.MEASURED
    assert status_mod.from_runner({"status": "oom"}) == status_mod.OOM
    assert status_mod.from_runner({"status": "unsupported"}) == status_mod.NOT_SUPPORTED
    # An unrecognized status must not be optimistically read as a measurement.
    assert status_mod.from_runner({"status": "something new"}) == status_mod.FAILED


# ── Precision resolution ────────────────────────────────────────────────────

def _hardware(**overrides: Any) -> hardware.Hardware:
    base = hardware.Hardware(
        platform="x86_64", os_name="Linux", logical_cores=8, physical_cores=4,
        ram_bytes=32 * 1024 ** 3,
    )
    return replace(base, **overrides)


def test_cuda_resolves_to_bf16_so_every_engine_holds_the_same_weights() -> None:
    """bf16 is chosen for weight fidelity, including where it is not native.

    The checkpoints and Aether's artifact are both bf16. Picking a format the device
    runs faster would make one engine round another's weights, which is the thing the
    suite exists not to do.
    """
    native = _hardware(nvidia=True, bf16_native=True, compute_capabilities=["8.0"])
    emulated = _hardware(nvidia=True, bf16_native=False, compute_capabilities=["7.5"])
    assert hardware.resolve_precision("auto", native)[0] == "bf16"
    resolved, reason = hardware.resolve_precision("auto", emulated)
    assert resolved == "bf16"
    assert "emulated" in reason and "--precision fp16" in reason


def test_cpu_resolves_to_fp32_and_an_explicit_request_is_honoured() -> None:
    resolved, reason = hardware.resolve_precision("auto", _hardware())
    assert resolved == "fp32"
    assert "no accelerator" in reason
    assert hardware.resolve_precision("fp16", _hardware())[0] == "fp16"


def test_weight_fit_guard_uses_the_smallest_visible_device() -> None:
    """A guard against a doomed load, sized from the smallest card, not the largest."""
    box = _hardware(nvidia=True, gpu_vram_bytes=[16 * 1024 ** 3, 40 * 1024 ** 3])
    assert hardware.can_hold_weights(box, 3_800_000_000, "bf16")
    assert not hardware.can_hold_weights(box, 3_800_000_000, "fp32")
    # An unknown parameter count must not be turned into a refusal to try.
    assert hardware.can_hold_weights(box, None, "fp32")


def test_thread_pinning_reports_what_it_pinned(monkeypatch: Any) -> None:
    for name in hardware.THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    record = hardware.pin_threads(3)
    assert record["controlled"] is True
    assert record["env"]["OMP_NUM_THREADS"] == "3"
    inherited = hardware.pin_threads(None)
    assert inherited["controlled"] is False


# ── Engine registry ─────────────────────────────────────────────────────────

def test_every_engine_declares_the_full_adapter_contract() -> None:
    """A registered engine must be describable and probeable without running."""
    assert len(registry.KEYS) == len(set(registry.KEYS)), "engine keys must be unique"
    for key in registry.KEYS:
        module = registry.module_for(key)
        spec = module.SPEC
        assert spec.key == key
        assert spec.taxonomy, f"{key} must declare what kind of system it is"
        assert spec.summary
        assert callable(module.probe)
        assert callable(module.build)
        assert spec.artifact_persistence in {
            "none", "process-local", "on-disk-cache", "portable-artifact",
        }
        # A build phase and a persistence class have to agree: something that
        # compiles nothing cannot leave an artifact behind.
        if not spec.has_build_phase:
            assert spec.artifact_persistence == "none"


def test_reference_and_subject_are_distinct_registered_engines() -> None:
    assert registry.REFERENCE in registry.KEYS
    assert registry.SUBJECT in registry.KEYS
    assert registry.REFERENCE != registry.SUBJECT


def test_taxonomy_does_not_call_eager_frameworks_compilers() -> None:
    """The classification has to be accurate, not flattering or convenient."""
    from benchmark.suite.engines import base

    compilers = {base.AOT_COMPILER, base.JIT_COMPILER, base.GRAPH_COMPILER}
    for key in ("transformers", "pytorch_native"):
        assert not (set(registry.spec_for(key).taxonomy) & compilers)
    for key in ("torch_compile", "aether", "openvino", "tensorrt_llm", "mlc"):
        assert set(registry.spec_for(key).taxonomy) & compilers
    for key in ("vllm", "sglang"):
        assert base.SERVING_ENGINE in registry.spec_for(key).taxonomy


def test_cuda_only_engines_report_not_applicable_on_a_cpu_host() -> None:
    """An engine that could never run here is not a failure and not a zero."""
    cpu = _hardware()
    config = plan.SuiteConfig()
    for key in ("vllm", "sglang", "tensorrt_llm", "deepspeed", "exllamav2"):
        result = registry.probe(key, cpu, "Qwen/Qwen3-0.6B", "fp32", config)
        assert result.status == status_mod.NOT_APPLICABLE
        assert "NVIDIA" in result.reason or "CUDA" in result.reason


def test_engines_needing_a_supplied_artifact_say_so_rather_than_inventing_one() -> None:
    gpu = _hardware(nvidia=True, gpu_vram_bytes=[16 * 1024 ** 3])
    config = plan.SuiteConfig()
    for key in ("exllamav2", "mlc", "llama_cpp"):
        result = registry.probe(key, gpu, "Qwen/Qwen3-0.6B", "bf16", config)
        assert result.status in {
            status_mod.NOT_APPLICABLE, status_mod.NOT_INSTALLED,
        }
        assert result.reason


# ── Plan validation ─────────────────────────────────────────────────────────

def test_plan_rejects_models_outside_the_charter() -> None:
    with pytest.raises(SystemExit) as caught:
        plan.parse_args(["--models", "meta-llama/Llama-3-70B"])
    assert "fixed by the charter" in str(caught.value)


def test_plan_rejects_an_unknown_engine_before_the_run_starts() -> None:
    with pytest.raises(SystemExit) as caught:
        plan.parse_args(["--engines", "definitely-not-an-engine"])
    assert "unknown engine" in str(caught.value)


def test_plan_restores_the_measurements_every_derived_figure_needs() -> None:
    """Batch 1 and the primary lengths are denominators, not preferences."""
    config = plan.parse_args([
        "--batch-sizes", "4,8", "--prompt-tokens", "64", "--output-tokens", "64",
        "--primary-prompt-tokens", "256", "--primary-output-tokens", "128",
    ])
    assert config.batch_sizes[0] == 1
    assert 256 in config.prompt_tokens
    assert 128 in config.output_tokens


def test_plan_refuses_a_single_measured_iteration() -> None:
    with pytest.raises(SystemExit) as caught:
        plan.parse_args(["--measure-iters", "1"])
    assert "dispersion" in str(caught.value)


def test_workload_signature_covers_everything_that_defines_the_work() -> None:
    signature = plan.SuiteConfig().workload_signature()
    for field in ("precision", "batch_sizes", "prompt_tokens", "output_tokens",
                  "warmup_iters", "measure_iters", "seed", "temperature", "threads"):
        assert field in signature


# ── Comparison arithmetic ───────────────────────────────────────────────────

def test_compare_reports_both_operands_with_every_ratio() -> None:
    """A ratio must always be checkable against the numbers it came from."""
    faster = analysis.compare(150.0, 100.0)
    assert faster["ratio"] == pytest.approx(1.5)
    assert faster["subject_improvement_percent"] == pytest.approx(50.0)
    assert faster["subject"] == 150.0 and faster["other"] == 100.0


def test_compare_inverts_direction_for_metrics_where_lower_is_better() -> None:
    latency = analysis.compare(0.5, 1.0, lower_is_better=True)
    assert latency["subject_improvement_percent"] == pytest.approx(50.0)
    assert latency["ratio"] == pytest.approx(2.0)
    worse = analysis.compare(2.0, 1.0, lower_is_better=True)
    assert worse["subject_improvement_percent"] == pytest.approx(-100.0)


def test_a_missing_operand_never_becomes_a_zero() -> None:
    for subject, other in ((None, 100.0), (100.0, None), (0.0, 100.0), (100.0, 0.0)):
        result = analysis.compare(subject, other)
        assert result["comparable"] is False
        assert result["ratio"] is None
        assert result["subject_improvement_percent"] is None


def test_verdict_is_symmetric_around_the_stated_tie_threshold() -> None:
    threshold = analysis.TIE_THRESHOLD * 100.0
    assert analysis.verdict(threshold + 0.1) == "aether"
    assert analysis.verdict(-(threshold + 0.1)) == "competitor"
    assert analysis.verdict(0.0) == "tie"
    assert analysis.verdict(threshold - 0.1) == "tie"
    assert analysis.verdict(-(threshold - 0.1)) == "tie"
    assert analysis.verdict(None) == "no comparison"


def test_representation_difference_is_labelled_on_both_sides() -> None:
    same = {"quantized": False, "representation": "published checkpoint", "precision": "bf16"}
    assert analysis.comparability(same, dict(same))[0] == analysis.SAME_REPRESENTATION

    quantized = {**same, "quantized": True}
    assert analysis.comparability(same, quantized)[0] == \
        analysis.REPRESENTATION_DIFFERENCE
    exported = {**same, "representation": "ONNX graph exported from the checkpoint"}
    assert analysis.comparability(same, exported)[0] == \
        analysis.REPRESENTATION_DIFFERENCE
    other_precision = {**same, "precision": "fp16"}
    assert analysis.comparability(same, other_precision)[0] == \
        analysis.REPRESENTATION_DIFFERENCE
    # The subject's own artifact residency counts too: at fp32 Aether holds BF16
    # weights, so it is the side that differs and the label must still appear.
    aether_fp32 = {
        "quantized": False, "precision": "fp32",
        "representation": "compiled AEG artifact; compiler default residency BF16",
    }
    competitor_fp32 = {**same, "precision": "fp32"}
    label, note = analysis.comparability(aether_fp32, competitor_fp32)
    assert label == analysis.REPRESENTATION_DIFFERENCE
    assert "BF16" in note


# ── End-to-end analysis over a synthetic payload ────────────────────────────

def _cell(batch: int, prompt: int, output: int, throughput: float, latency: float,
          *, primary: bool = False, sweeps: list[str] | None = None,
          status: str = status_mod.MEASURED, reason: str = "") -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "primary" if primary else "batch",
        "batch_size": batch, "prompt_tokens": prompt, "output_tokens": output,
        "sweeps": sweeps or ["batch"], "is_primary": primary, "status": status,
        "reason": reason,
    }
    if status != status_mod.MEASURED:
        return record
    record["measurement"] = {
        "status": "ok", "prompt_tokens": prompt, "completion_tokens": output,
        "latency_s": {"n": 10, "median": latency, "mean": latency, "stdev": 0.01,
                      "min": latency, "max": latency, "p95": latency, "p99": latency,
                      "coefficient_of_variation": 0.01},
        "tokens_per_s": {"median": throughput},
        "host_during_inference": {"rss_peak_bytes": 1_000_000_000},
        "gpu_peak": {"available": True,
                     "devices": [{"peak_reserved_bytes": 2_000_000_000}]},
    }
    record["derived"] = {
        "total_tokens_per_s": throughput, "per_request_tokens_per_s": throughput / batch,
        "decode_tokens_per_s": throughput, "prompt_tokens_per_s": 500.0,
        "ttft_s": 0.05, "tpot_ms": 10.0, "end_to_end_latency_s": latency,
        "cold_latency_s": latency * 2, "iterations": 10,
        "coefficient_of_variation": 0.01,
        "latency_stats": record["measurement"]["latency_s"],
        "throughput_stats": record["measurement"]["tokens_per_s"],
    }
    return record


def _run(engine: str, cells: list[dict[str, Any]], *, build_s: float | None,
         load_s: float, quantized: bool = False,
         representation: str = "published checkpoint",
         persistence: str = "none") -> dict[str, Any]:
    return {
        "engine": engine, "model": "Qwen/Qwen3-0.6B", "precision": "bf16",
        "status": status_mod.MEASURED, "cells": cells,
        "spec": {"has_build_phase": build_s is not None, "taxonomy": ["runtime"],
                 "artifact_persistence": persistence},
        "describe": {"representation": representation, "quantized": quantized},
        "load": {"status": "ok", "prepare_s": build_s, "load_s": load_s,
                 "total_s": (build_s or 0.0) + load_s, "notes": {}},
        "artifact": {"has_build_phase": build_s is not None, "persistence": persistence,
                     "build_s": build_s, "load_s": load_s,
                     "total_startup_s": (build_s or 0.0) + load_s,
                     "artifact_bytes": 1024 if build_s else None},
        "correctness_sample": {
            "status": status_mod.MEASURED, "token_ids": [1, 2, 3, 4],
            "text": "hello world", "completion_tokens": 4,
        },
    }


def _payload(runs: list[dict[str, Any]], reuse: list[dict[str, Any]] | None = None,
             ) -> dict[str, Any]:
    return {
        "suite_version": "test", "generated_at": "now",
        "plan": {"temperature": 0.0, "amortization_runs": [1, 100],
                 "resolved_precision": "bf16", "precision_reason": "test",
                 "models": ["Qwen/Qwen3-0.6B"], "engines": [r["engine"] for r in runs],
                 "warmup_iters": 3, "measure_iters": 10, "seed": 1, "top_p": 1.0,
                 "top_k": 0, "threads": 4, "invocation": "pytest"},
        "workload_signature": {}, "hardware": {"accelerator": "cuda"},
        "environment": {"software": {}}, "engine_catalogue": {},
        "models": {}, "runs": runs, "reuse_runs": reuse or [],
        "worker_processes": [],
    }


def test_unmeasured_cells_survive_flattening_with_their_reason() -> None:
    """A row that was not measured has to reach the report, not vanish from it."""
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(8, 256, 128, 0.0, 0.0, status=status_mod.OOM,
                  reason="CUDA out of memory"),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
    ]
    rows = analysis.flatten(_payload(runs))
    assert len(rows) == 2
    unmeasured = next(row for row in rows if row["batch_size"] == 8)
    assert unmeasured["status"] == status_mod.OOM
    assert unmeasured["total_tokens_per_s"] is None
    assert "out of memory" in unmeasured["reason"]


def test_wins_and_losses_are_both_reported_from_the_same_cells() -> None:
    """The anti-bias property: a loss is recorded exactly like a win."""
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 90.0, 5.7),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [
            _cell(1, 256, 128, 25.0, 5.1, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 120.0, 4.3),
        ], build_s=None, load_s=6.0),
    ]
    result = analysis.analyze(_payload(runs))
    comparisons = result["comparisons"]
    assert len(comparisons) == 2
    by_batch = {item["batch_size"]: item for item in comparisons}
    assert by_batch[1]["winner"] == "aether"
    assert by_batch[1]["throughput"]["subject_improvement_percent"] == pytest.approx(60.0)
    assert by_batch[4]["winner"] == "competitor"
    assert by_batch[4]["throughput"]["subject_improvement_percent"] == pytest.approx(-25.0)

    win_loss = result["win_loss"]["all"]
    assert win_loss["aether_wins"] == 1
    assert win_loss["aether_losses"] == 1
    assert win_loss["compared"] == 2
    # The extremes must name the real best and worst case, not the best twice.
    assert result["win_loss"]["largest_advantage"]["batch_size"] == 1
    assert result["win_loss"]["largest_disadvantage"]["batch_size"] == 4


def test_scaling_efficiency_is_measured_against_the_engine_own_batch_one() -> None:
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 80.0, 6.4),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
    ]
    scaling = {
        (entry["batch_size"]): entry
        for entry in analysis.analyze(_payload(runs))["batch_scaling"]
    }
    assert scaling[1]["scaling_efficiency_percent"] == pytest.approx(100.0)
    # Double the throughput at four times the width is 50% of linear.
    assert scaling[4]["scaling_vs_batch1"] == pytest.approx(2.0)
    assert scaling[4]["scaling_efficiency_percent"] == pytest.approx(50.0)
    assert scaling[4]["per_request_tokens_per_s"] == pytest.approx(20.0)


def test_break_even_solves_the_crossing_point_and_names_the_never_case() -> None:
    """Aether pays 16s more to start and saves 1.9s per request: about 8 requests."""
    runs = [
        _run("aether", [_cell(1, 256, 128, 40.0, 3.2, primary=True,
                              sweeps=["batch", "prompt", "output"])],
             build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [_cell(1, 256, 128, 25.0, 5.1, primary=True,
                                    sweeps=["batch", "prompt", "output"])],
             build_s=None, load_s=8.0),
    ]
    reuse = [{
        "engine": "aether", "model": "Qwen/Qwen3-0.6B", "mode": "reuse",
        "status": status_mod.MEASURED, "load": {"total_s": 4.5},
        "first_inference": {"cold_latency_s": 3.6},
    }]
    economics = analysis.analyze(_payload(runs, reuse))["compile_economics"]
    entry = next(item for item in economics["entries"] if item["engine"] == "aether")
    assert entry["second_process_load_s"] == pytest.approx(4.5)
    assert entry["total_cost_s"]["100"]["cold_first_deployment"] == pytest.approx(
        24.0 + 100 * 3.2
    )
    assert entry["total_cost_s"]["100"]["warm_reused_artifact"] == pytest.approx(
        4.5 + 100 * 3.2
    )
    crossing = economics["break_even"][0]
    assert crossing["break_even_runs"] == pytest.approx(16.0 / 1.9, rel=1e-6)

    # Reverse it: when Aether is slower per request, no run count repays the build.
    slower = [
        _run("aether", [_cell(1, 256, 128, 20.0, 6.0, primary=True,
                              sweeps=["batch", "prompt", "output"])],
             build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [_cell(1, 256, 128, 25.0, 5.1, primary=True,
                                    sweeps=["batch", "prompt", "output"])],
             build_s=None, load_s=8.0),
    ]
    never = analysis.analyze(_payload(slower))["compile_economics"]["break_even"][0]
    assert never["break_even_runs"] is None
    assert "never" in never["interpretation"]


def test_correctness_classes_separate_rounding_from_a_different_computation() -> None:
    identical = {"identical": True, "matching_prefix_fraction": 1.0}
    text_same = {"identical": False, "matching_prefix_fraction": 0.0,
                 "first_divergence_index": 0}
    late_split = {"identical": False, "matching_prefix_fraction": 0.9,
                  "first_divergence_index": 57}
    early_split = {"identical": False, "matching_prefix_fraction": 0.02,
                   "first_divergence_index": 1}

    assert analysis._classify(identical, {"identical": True}, True)[0] == \
        analysis.EXACT_MATCH
    # Identical decoded text with different ids is still an exact match, and the
    # basis has to say which observable it rests on.
    label, basis = analysis._classify(text_same, {"identical": True}, True)
    assert label == analysis.EXACT_MATCH
    assert "text" in basis
    assert analysis._classify(late_split, {"identical": False}, True)[0] == \
        analysis.NUMERICALLY_EQUIVALENT
    assert analysis._classify(early_split, {"identical": False}, True)[0] == \
        analysis.DIFFERENT_OUTPUT
    assert analysis._classify(early_split, {"identical": False}, False)[0] == \
        analysis.EXPECTED_SAMPLING_DIFFERENCE


# ── Output artifacts ────────────────────────────────────────────────────────

def _analyzed() -> tuple[dict[str, Any], dict[str, Any]]:
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 90.0, 5.7),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [
            _cell(1, 256, 128, 25.0, 5.1, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 120.0, 4.3),
            _cell(8, 256, 128, 0.0, 0.0, status=status_mod.OOM, reason="out of memory"),
        ], build_s=None, load_s=6.0),
    ]
    payload = _payload(runs)
    return payload, analysis.analyze(payload)


def test_csv_keeps_unmeasured_rows_so_nothing_averages_them_as_zero(
    tmp_path: Path,
) -> None:
    import csv

    _, analyzed = _analyzed()
    target = tmp_path / "results.csv"
    report.write_csv(analyzed, target)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert len(rows) == len(analyzed["rows"])
    oom = next(row for row in rows if row["status"] == status_mod.OOM)
    assert oom["total_tokens_per_s"] == ""
    assert oom["batch_size"] == "8"


def test_comparison_csv_records_the_sign_of_every_result(tmp_path: Path) -> None:
    import csv

    _, analyzed = _analyzed()
    target = tmp_path / "comparisons.csv"
    report.write_comparison_csv(analyzed, target)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    signs = {row["winner"] for row in rows}
    assert signs == {"aether", "competitor"}
    losing = next(row for row in rows if row["winner"] == "competitor")
    assert float(losing["aether_improvement_percent"]) < 0


def test_report_states_both_the_win_and_the_loss(tmp_path: Path) -> None:
    payload, analyzed = _analyzed()
    text = report.build_report(payload, analyzed, {"written": [], "skipped": [],
                                                   "directory": "graphs"})
    assert "## Aether win/loss analysis" in text
    assert "### Where Aether loses" in text
    assert "-25.0%" in text, "the losing comparison must appear with its sign"
    assert "+60.0%" in text
    # The unmeasured cell has to be visible as unmeasured, not as absence.
    assert "OOM" in text
    for heading in ("## Compilation economics", "## Correctness",
                    "## Statistical quality", "## Final rankings",
                    "## Limitations", "## Reproducibility", "## Methodology"):
        assert heading in text


def test_charts_skip_rather_than_invent_when_data_is_absent(tmp_path: Path) -> None:
    empty = {
        "rows": [], "primary_metric": "total_tokens_per_s",
        "primary_metric_label": "tok/s", "batch_scaling": [], "per_competitor": {},
        "compile_economics": {"entries": []},
    }
    manifest = charts.write_all(empty, tmp_path)
    assert manifest["written"] == []
    assert manifest["skipped"], "a skipped figure must be recorded with a reason"
    assert not list(tmp_path.glob("*.png"))


def test_charts_are_written_for_measured_data(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _, analyzed = _analyzed()
    manifest = charts.write_all(analyzed, tmp_path)
    assert manifest["written"]
    for name in manifest["written"]:
        assert (tmp_path / name).stat().st_size > 0


# ── Worker planning and derivation ──────────────────────────────────────────

def test_the_primary_cell_is_planned_once_and_serves_every_sweep() -> None:
    """Measuring it three times would spend budget to sample the same thing."""
    from benchmark.suite import worker

    cells = worker._plan_cells(
        {"primary_prompt_tokens": 256, "primary_output_tokens": 128,
         "batch_sizes": [1, 2, 4], "output_tokens": [32, 128, 512]},
        {"32": {}, "256": {}, "1024": {}},
    )
    primaries = [cell for cell in cells if cell.get("is_primary")]
    assert len(primaries) == 1
    assert set(primaries[0]["sweeps"]) == {"batch", "prompt", "output"}
    keys = [(c["batch_size"], c["prompt_tokens"], c["output_tokens"]) for c in cells]
    assert len(keys) == len(set(keys)), "no configuration may be planned twice"
    assert (2, 256, 128) in keys and (4, 256, 128) in keys
    assert (1, 32, 128) in keys and (1, 1024, 128) in keys
    assert (1, 256, 32) in keys and (1, 256, 512) in keys


def test_derivation_separates_prefill_from_decode_and_admits_when_it_cannot() -> None:
    from benchmark.suite import worker

    measurement = {
        "prompt_tokens": 100, "completion_tokens": 11,
        "latency_s": {"median": 2.0, "n": 10},
        "tokens_per_s": {"median": 5.5},
    }
    derived = worker._derive(measurement, batch=1, prefill_s=1.0, ttft_s=1.05)
    # Ten of the eleven tokens come out of the decode loop, in the second left
    # after prefill.
    assert derived["decode_tokens_per_s"] == pytest.approx(10.0)
    assert derived["tpot_ms"] == pytest.approx(100.0)
    assert derived["prompt_tokens_per_s"] == pytest.approx(100.0)
    assert derived["ttft_s"] == pytest.approx(1.05)

    # With no prefill measurement, decode must stay undefined rather than borrow
    # the end-to-end rate under a name that means something narrower.
    without = worker._derive(measurement, batch=1, prefill_s=None, ttft_s=None)
    assert without["decode_tokens_per_s"] is None
    assert without["prompt_tokens_per_s"] is None
    assert without["total_tokens_per_s"] == pytest.approx(5.5)


def test_batch_throughput_is_aggregate_and_per_request_is_derived_from_it() -> None:
    from benchmark.suite import worker

    measurement = {
        "prompt_tokens": 100, "completion_tokens": 10,
        "latency_s": {"median": 2.0, "n": 10},
        "tokens_per_s": {"median": 40.0},
    }
    derived = worker._derive(measurement, batch=8, prefill_s=None, ttft_s=None)
    assert derived["total_tokens_per_s"] == pytest.approx(40.0)
    assert derived["per_request_tokens_per_s"] == pytest.approx(5.0)
    assert derived["generated_tokens_total"] == 80


def test_a_crashed_worker_leaves_a_record_with_no_measurement_in_it() -> None:
    from benchmark.suite import orchestrate

    record = orchestrate._orphan_record(
        "vllm", "Qwen/Qwen3-0.6B", "bf16",
        {"returncode": None, "elapsed_s": 900.0, "timed_out": True},
    )
    assert record["status"] == status_mod.FAILED
    assert record["cells"] == []
    assert "timeout" in record["reason"]
    assert not status_mod.is_measured(record)
