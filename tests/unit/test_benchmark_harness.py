"""Guards on the benchmark harness's neutrality and correctness.

A benchmark is only worth its methodology, and methodology decays silently. These
tests assert the properties the harness claims — that it never special-cases a
backend, that timing synchronizes, that telemetry and profiling stay out of the
official numbers — so a later edit that introduces bias fails here rather than
producing a quietly skewed report.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

BENCHMARK = pathlib.Path(__file__).resolve().parents[2] / "benchmark"


def _read(name: str) -> str:
    return (BENCHMARK / name).read_text(encoding="utf-8")


def test_runner_never_special_cases_a_backend_by_name() -> None:
    """The measurement path must be generic over backends.

    If the runner can name a backend it can also treat one differently, so the
    absence of those literals is the structural guarantee of even handling.
    """
    tree = ast.parse(_read("runner.py"))
    named = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.lower() in {"aether", "transformers"}
    ]
    assert not named, f"runner.py references backends by name: {named}"


def test_performance_mode_has_a_single_generation_call_site() -> None:
    """One call site means both backends cannot receive different arguments."""
    entry = _read("run_benchmark.py")
    performance = entry.split("def run_performance")[1].split("def run_correctness")[0]
    assert performance.count("runner.measure_generation(") == 1


def test_generation_settings_come_from_configuration() -> None:
    """Iteration counts, seed and sampling must not be literals in the runner."""
    entry = _read("run_benchmark.py")
    for expected in (
        "warmup_iters=config.warmup_iters",
        "measure_iters=config.measure_iters",
        "seed=config.seed",
        "temperature=config.temperature",
        "top_p=config.top_p",
        "top_k=config.top_k",
        "max_new_tokens=config.max_new_tokens",
    ):
        assert expected in entry, f"performance path does not use {expected}"


def test_timing_synchronizes_on_both_edges() -> None:
    """A wall clock around asynchronous device work is meaningless without this."""
    metrics = _read("metrics.py")
    timed = metrics.split("def timed(")[1].split("def summarize(")[0]
    assert timed.count("synchronize()") >= 2, "timed() must synchronize before and after"


def test_telemetry_is_not_collected_during_measured_iterations() -> None:
    """Sampling costs host time, which is the resource decode contends for."""
    runner = _read("runner.py")
    measured = runner.split("latencies: list[float] = []")[1].split("peak_gpu")[0]
    assert "GPUTelemetryMonitor" not in measured
    assert "HostMemoryMonitor" not in measured


def test_performance_mode_does_not_use_the_profiler() -> None:
    """Instrumentation perturbs a launch-bound loop, so it must stay separate."""
    entry = _read("run_benchmark.py")
    performance = entry.split("def run_performance")[1].split("def run_correctness")[0]
    assert "profiling." not in performance


def test_unsupported_configurations_are_recorded_not_silently_skipped() -> None:
    """An unsupported cell must reach the report as such."""
    from benchmark.runner import _failure
    from benchmark.backends import UnsupportedConfiguration

    record = _failure(UnsupportedConfiguration("no batch dimension"))
    assert record["status"] == "unsupported"
    assert "no batch dimension" in record["message"]
    oom = _failure(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert oom["status"] == "oom"


def test_charter_models_are_fixed() -> None:
    """The three benchmark models are part of the definition, not a default."""
    from benchmark.config import MODELS, is_charter_model

    assert MODELS == (
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "Qwen/Qwen3-0.6B",
        "SummerSigh/GPTNeo350M-Instruct-SFT",
    )
    assert is_charter_model(MODELS[0])
    assert not is_charter_model("./some/local/checkpoint")


def test_model_argument_accepts_paths_containing_spaces() -> None:
    """Splitting model lists on whitespace would corrupt a path silently."""
    from benchmark.config import _str_list

    assert _str_list("Qwen/Qwen3-0.6B,./a path/with spaces") == [
        "Qwen/Qwen3-0.6B", "./a path/with spaces",
    ]


def test_unknown_model_is_rejected_rather_than_substituted() -> None:
    """A typo must fail loudly, never fall back to a different model."""
    from benchmark.config import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--models", "meta-llama/Llama-3.1-8B"])


def test_statistics_report_dispersion_and_sample_size() -> None:
    """A centre without a spread and an n is not a measurement."""
    from benchmark.metrics import summarize

    summary = summarize([1.0, 2.0, 3.0, 4.0])
    for field in ("n", "mean", "median", "stdev", "min", "max", "p50", "p90"):
        assert field in summary, field
    assert summary["n"] == 4
    assert summarize([])["n"] == 0


def test_throughput_is_averaged_per_iteration_not_derived_from_mean_latency() -> None:
    """Mean-of-ratios and ratio-of-means differ; the report claims the former."""
    from benchmark.metrics import throughput_samples

    samples = throughput_samples([1.0, 2.0], tokens=10)
    assert samples == [10.0, 5.0]


def test_speedup_reports_both_operands() -> None:
    """A ratio a reader cannot check against its inputs is not evidence."""
    from benchmark.metrics import speedup

    result = speedup({"median": 100.0}, {"median": 80.0})
    assert result["ratio"] == pytest.approx(1.25)
    assert result["percent"] == pytest.approx(25.0)
    assert result["aether"] == 100.0 and result["transformers"] == 80.0


def test_token_ids_are_flattened_consistently() -> None:
    """Tokenizers return flat lists, batched lists or arrays interchangeably."""
    numpy = pytest.importorskip("numpy")
    from benchmark.prompts import flatten_ids

    assert flatten_ids([1, 2, 3]) == [1, 2, 3]
    assert flatten_ids([[1, 2, 3]]) == [1, 2, 3]
    assert flatten_ids(numpy.array([[4, 5]])) == [4, 5]


def test_view_operations_are_separated_from_kernels() -> None:
    """Metadata-only operations launch nothing and must not be counted as work."""
    from benchmark.profiling import VIEW_OPS, categorize

    assert "aten.transpose.int" in VIEW_OPS
    assert categorize("aten.addmm.default") == "gemm"
    assert categorize("aten.scaled_dot_product_attention.default") == "attention"
    assert categorize("aten._local_scalar_dense.default") == "sync"


def test_correctness_verdict_is_tolerance_based_not_bit_exact() -> None:
    """Two runtimes computing the same mathematics differ in the last bits."""
    from benchmark.correctness import LOGIT_TOLERANCE_OVER_STD, verdict

    noise = {
        "comparable": True, "max_abs_diff": 1e-5, "max_abs_diff_over_std": 1e-6,
        "argmax_agrees": True,
    }
    assert verdict(noise, {"identical": True})["equivalent"] is True

    structural = {
        "comparable": True, "max_abs_diff": 5.0, "max_abs_diff_over_std": 1.0,
        "argmax_agrees": False,
    }
    result = verdict(structural, {"identical": False, "first_divergence_index": 0})
    assert result["equivalent"] is False
    assert result["concerns"]
    assert LOGIT_TOLERANCE_OVER_STD > 0


def test_report_states_its_limitations() -> None:
    """The caveats that bound every number must ship with the numbers."""
    from benchmark.reporting import LIMITATIONS

    joined = " ".join(LIMITATIONS).lower()
    for topic in ("batch", "bf16", "compil", "streaming", "sampl", "profiler"):
        assert topic in joined, f"limitations do not mention {topic}"


def test_report_renders_with_no_successful_measurement() -> None:
    """A failed run must still produce an honest report, not a crash."""
    from benchmark.reporting import build_report

    payload = {
        "generated_at": "now", "config": {"mode": "performance"},
        "environment": {}, "performance": [], "skips": [
            {"model": "m", "backend": "both", "status": "load-failed"}
        ],
    }
    text = build_report(payload, charts=[])
    assert "supports no performance conclusion" in text
    assert "load-failed" in text


def test_semantic_response_cache_is_disabled_for_measured_runs() -> None:
    """A repeated prompt must be generated, not recalled.

    Aether's runtime enables a semantic response cache by default: an identical
    prompt returns a stored completion without running the model. Measured on this
    repository, that turns a ~15 s generation into ~0.001 s. A benchmark issues the
    same prompt many times, so with the cache on it would compare a dictionary
    lookup against real inference. Transformers has no equivalent, so such a
    comparison would be meaningless rather than merely flattering.
    """
    from benchmark.backend_aether import BENCHMARK_RUNTIME_FLAGS

    assert BENCHMARK_RUNTIME_FLAGS["enable_semantic_cache"] is False


def test_the_disabled_flag_is_reported_not_hidden() -> None:
    """Any deviation from default configuration must appear in the report."""
    from benchmark.backend_aether import AetherBackend

    described = AetherBackend().describe()
    assert described["runtime_flags_overridden"] == {"enable_semantic_cache": False}
    assert "cache" in described["runtime_flags_reason"]


def test_aether_runtime_config_still_defaults_the_cache_on() -> None:
    """The benchmark overrides a flag; it does not change Aether's default.

    If this fails, someone altered the runtime to suit the benchmark, which is
    exactly the thing the benchmark must not do.
    """
    from aether.runtime.config import RuntimeConfig

    assert RuntimeConfig().enable_semantic_cache is True


def test_default_target_cuda_capability_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA sm_75 (e.g. Tesla T4) must resolve to nearest supported target cuda_sm70."""
    import torch
    from benchmark.backend_aether import _default_target

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _idx=0: (7, 5))

    target = _default_target()
    assert target == "cuda_sm70"


def test_multigpu_formatting_handles_none() -> None:
    """multigpu_section must handle None engine/status fields gracefully without TypeError."""
    from benchmark.reporting import multigpu_section

    payload = {
        "cases": [
            {
                "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "configuration": "aether forced tensor-parallel",
                "gpus": 2,
                "tokens_per_s": {"median": None},
                "latency_s": {"median": None},
                "per_gpu_peak_bytes": [],
                "engine": None,
                "status": None,
            }
        ],
        "notes": ["test note"]
    }
    result = multigpu_section(payload)
    assert "—" in result
