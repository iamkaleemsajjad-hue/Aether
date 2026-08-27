"""Single entry point for the benchmark.

    python benchmark/run_benchmark.py --mode performance
    python benchmark/run_benchmark.py --mode correctness
    python benchmark/run_benchmark.py --mode profile
    python benchmark/run_benchmark.py --mode multigpu
    python benchmark/run_benchmark.py --quick

Every mode is independent so that a limited GPU budget is not spent running the
same work twice.  Nothing here needs editing to change an experiment; see
``--help`` for the full set of switches.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow `python benchmark/run_benchmark.py` from a clone without installation.
_ROOT = Path(__file__).resolve().parent.parent
for candidate in (_ROOT, _ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark import correctness as correctness_mod  # noqa: E402
from benchmark import gpu_monitor, profiling, prompts, reporting, runner  # noqa: E402
from benchmark import system_info  # noqa: E402
from benchmark.backend_aether import AetherBackend  # noqa: E402
from benchmark.backend_transformers import TransformersBackend  # noqa: E402
from benchmark.config import BenchmarkConfig, is_charter_model, parse_args  # noqa: E402

BACKENDS = ("transformers", "aether")


def _log(message: str) -> None:
    print(message, flush=True)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _limit_devices(count: int | None) -> None:
    """Pin GPU visibility before torch initializes its CUDA context."""
    if count is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(count))


def _make_backend(name: str, config: BenchmarkConfig, devices: list[str] | None = None) -> Any:
    device = _device()
    if name == "transformers":
        return TransformersBackend(device=device)
    return AetherBackend(
        device=device, cache_dir=config.cache_dir, execution_devices=devices,
        keep_artifact=config.keep_aeg,
    )


def _load_pair(config: BenchmarkConfig, model: str, precision: str) -> tuple[dict, dict, dict]:
    """Load both backends for one (model, precision) and return their handles."""
    backends: dict[str, Any] = {}
    loads: dict[str, Any] = {}
    for name in BACKENDS:
        backend = _make_backend(name, config)
        _log(f"    loading {name} ...")
        record = runner.measure_load(backend, model, precision)
        loads[name] = record
        if record.get("status") != "ok":
            _log(f"      {name}: {record.get('status')}: {record.get('message', '')[:120]}")
            backend.unload()
            continue
        backends[name] = backend
        _log(
            f"      {name}: load {record['load_s']:.2f}s"
            + (f", compile {record['prepare_s']:.2f}s" if record.get("prepare_s") else "")
        )
    return backends, loads, {}


def run_performance(config: BenchmarkConfig) -> tuple[list[dict], list[dict]]:
    """Mode 1 and 2: throughput, latency, phases, memory and utilization."""
    cells: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    repetition = 0
    for model in config.models:
        for precision in config.precisions:
            _log(f"\n=== {model} @ {precision} ===")
            backends, loads, _ = _load_pair(config, model, precision)
            if not backends:
                skips.append({
                    "model": model, "precision": precision, "backend": "both",
                    "status": "load-failed", "phase": "load",
                    "message": "neither backend loaded; see load records",
                })
                continue
            reference_tokenizer = None
            for name in BACKENDS:
                if name in backends:
                    reference_tokenizer = backends[name].tokenizer()
                    break
            prompt_set = prompts.build_prompt_set(reference_tokenizer, config.prompt_tokens)
            for requested, prompt in prompt_set.items():
                if not prompt["exact"]:
                    skips.append({
                        "model": model, "precision": precision,
                        "prompt_tokens": requested, "backend": "both",
                        "status": "approximate-prompt", "phase": "prompt",
                        "message": f"tokenizer round trip yielded {prompt['achieved_tokens']} tokens",
                    })
                for batch in config.batch_sizes:
                    repetition += 1
                    order = runner.alternated_order(repetition, BACKENDS)
                    cell = {
                        "cell": {
                            "model": model, "precision": precision,
                            "prompt_tokens": prompt["achieved_tokens"],
                            "requested_prompt_tokens": requested,
                            "batch_size": batch, "order": list(order),
                        },
                        "backends": {}, "load": loads, "prefill": {}, "ttft": {},
                    }
                    for name in order:
                        backend = backends.get(name)
                        if backend is None:
                            cell["backends"][name] = {
                                "status": "load-failed", "phase": "load",
                                "message": "backend did not load",
                            }
                            continue
                        _log(f"  [{name}] p{prompt['achieved_tokens']} b{batch} ...")
                        cell["backends"][name] = runner.measure_generation(
                            backend, prompt["text"],
                            max_new_tokens=config.max_new_tokens,
                            temperature=config.temperature, top_p=config.top_p,
                            top_k=config.top_k, seed=config.seed, batch_size=batch,
                            warmup_iters=config.warmup_iters,
                            measure_iters=config.measure_iters,
                            gpu_interval_s=config.gpu_sample_interval_s,
                        )
                        status = cell["backends"][name].get("status")
                        if status == "ok":
                            throughput = cell["backends"][name]["tokens_per_s"]["median"]
                            _log(f"      {throughput:.2f} tok/s (median)")
                        else:
                            _log(f"      {status}: {cell['backends'][name].get('message','')[:110]}")
                        if batch == 1:
                            cell["prefill"][name] = runner.measure_prefill(
                                backend, prompt["text"],
                                warmup_iters=1, measure_iters=max(3, config.measure_iters),
                            )
                            cell["ttft"][name] = runner.measure_ttft(
                                backend, prompt["text"],
                                max_new_tokens=config.max_new_tokens,
                                seed=config.seed, measure_iters=3,
                            )
                    cells.append(cell)
                    runner.cooldown(config.cooldown_s)
            for backend in backends.values():
                backend.unload()
            gpu_monitor.empty_cache()
    return cells, skips


def run_correctness(config: BenchmarkConfig) -> dict[str, Any]:
    """Mode 4: do the two runtimes compute the same thing?

    Compared at three levels — prompt logits, greedy token ids, decoded text —
    with the tokenizers themselves verified first, since a tokenizer difference
    would invalidate every other comparison in the suite.
    """
    cases: list[dict[str, Any]] = []
    agreement: dict[str, Any] = {}
    for model in config.models:
        for precision in config.precisions:
            _log(f"\n=== correctness: {model} @ {precision} ===")
            backends, loads, _ = _load_pair(config, model, precision)
            if len(backends) < 2:
                cases.append({
                    "model": model, "precision": precision,
                    "logits": {"comparable": False, "reason": "a backend failed to load"},
                    "tokens": {}, "text": {}, "verdict": {"equivalent": None},
                    "load": loads,
                })
                for backend in backends.values():
                    backend.unload()
                continue
            reference, candidate = backends["transformers"], backends["aether"]
            prompt_set = prompts.build_prompt_set(
                reference.tokenizer(), config.prompt_tokens[:1] or [32]
            )
            prompt = next(iter(prompt_set.values()))
            agreement[model] = prompts.verify_tokenizer_agreement(
                reference.tokenizer(), candidate.tokenizer(), [prompt["text"]]
            )
            try:
                reference_logits = reference.prefill(prompt["text"])
                candidate_logits = candidate.prefill(prompt["text"])
                logit_comparison = correctness_mod.compare_logits(
                    reference_logits, candidate_logits
                )
            except BaseException as exc:  # noqa: BLE001
                logit_comparison = {"comparable": False, "reason": f"{type(exc).__name__}: {exc}"}
            reference_text = candidate_text = ""
            token_comparison: dict[str, Any] = {}
            try:
                reference_out = reference.generate(
                    prompt["text"], max_new_tokens=config.max_new_tokens,
                    temperature=0.0, top_p=1.0, top_k=0, seed=config.seed, batch_size=1,
                )
                candidate_ids = candidate.generate_token_ids(
                    prompt["text"], max_new_tokens=config.max_new_tokens, temperature=0.0
                )
                candidate_text = candidate.tokenizer().decode(
                    candidate_ids, skip_special_tokens=True
                )
                reference_text = reference_out.text
                token_comparison = correctness_mod.compare_token_ids(
                    reference_out.token_ids, candidate_ids
                )
            except BaseException as exc:  # noqa: BLE001
                token_comparison = {"identical": None, "error": f"{type(exc).__name__}: {exc}"}
            cases.append({
                "model": model, "precision": precision,
                "prompt_tokens": prompt["achieved_tokens"],
                "logits": logit_comparison,
                "tokens": token_comparison,
                "text": correctness_mod.compare_text(reference_text, candidate_text),
                "verdict": correctness_mod.verdict(logit_comparison, token_comparison),
            })
            verdict = cases[-1]["verdict"]
            _log(f"    equivalent={verdict.get('equivalent')} "
                 f"deviation/std={verdict.get('observed_deviation_over_std')}")
            for backend in backends.values():
                backend.unload()
            gpu_monitor.empty_cache()
    return {"cases": cases, "tokenizer_agreement": agreement,
            "logit_tolerance_over_std": correctness_mod.LOGIT_TOLERANCE_OVER_STD}


def run_profile(config: BenchmarkConfig) -> dict[str, Any]:
    """Mode 3: kernel counts and time attribution.

    Deliberately separate from the performance mode: instrumentation perturbs a
    launch-bound decode loop, so nothing measured here is allowed to inform a
    throughput claim.
    """
    dispatch: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    precision = config.precisions[0]
    steps = 8
    for model in config.models:
        _log(f"\n=== profile: {model} @ {precision} ===")
        backends, _loads, _ = _load_pair(config, model, precision)
        if not backends:
            continue
        prompt_set = prompts.build_prompt_set(
            next(iter(backends.values())).tokenizer(), [config.prompt_tokens[0]]
        )
        prompt = next(iter(prompt_set.values()))["text"]
        dispatch_entry: dict[str, Any] = {"model": model, "precision": precision}
        profile_entry: dict[str, Any] = {"model": model, "precision": precision}
        for name, backend in backends.items():
            layers = _layer_count(backend, name)

            def step(backend: Any = backend) -> Any:
                return backend.generate(
                    prompt, max_new_tokens=steps, temperature=0.0, top_p=1.0,
                    top_k=0, seed=config.seed, batch_size=1,
                )

            try:
                dispatch_entry[name] = profiling.count_dispatches(
                    step, steps=steps, layers=layers
                )
                _log(f"    {name}: "
                     f"{dispatch_entry[name]['kernel_calls_per_step']:.1f} kernels/token")
            except BaseException as exc:  # noqa: BLE001
                dispatch_entry[name] = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                profile_entry[name] = profiling.profile_step(step, steps=3)
            except BaseException as exc:  # noqa: BLE001
                profile_entry[name] = {"error": f"{type(exc).__name__}: {exc}"}
        dispatch.append(dispatch_entry)
        profiles.append(profile_entry)
        for backend in backends.values():
            backend.unload()
        gpu_monitor.empty_cache()
    return {"dispatch": dispatch, "profile": profiles,
            "note": "Instrumented runs. Not used for any throughput figure."}


def _layer_count(backend: Any, name: str) -> int | None:
    """Layer depth, used to normalize kernel counts across models."""
    if name == "aether":
        return getattr(getattr(backend, "_engine", None), "num_layers", None)
    model = getattr(backend, "_model", None)
    config = getattr(model, "config", None)
    return getattr(config, "num_hidden_layers", None) if config else None


def run_multigpu(config: BenchmarkConfig) -> dict[str, Any] | None:
    """Mode 5: how each runtime uses more than one accelerator.

    Aether shards a dense decoder only when the model does not fit on the
    smallest visible device, so for these three models the default on a
    multi-GPU host is single-device execution.  Both the default and the forced
    sharded path are measured, and each is labelled, because comparing a sharded
    Aether against a single-device Transformers would not be a same-hardware
    comparison.
    """
    try:
        import torch

        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        gpus = 0
    if gpus < 2:
        return {"cases": [], "notes": [f"Only {gpus} accelerator(s) visible; nothing to compare."]}

    notes = [
        f"{gpus} accelerators visible.",
        "Aether's default placement keeps a model that fits on one device on that "
        "device; the sharded row is a forced diagnostic configuration.",
        "Transformers has no equivalent single-process tensor-parallel path for "
        "these models, so its row is single-device in every case.",
    ]
    cases: list[dict[str, Any]] = []
    precision = config.precisions[0]
    model = config.models[0]
    devices = [f"cuda:{index}" for index in range(gpus)]
    for label, execution_devices, force in (
        ("aether default placement", None, False),
        ("aether forced tensor-parallel", devices, True),
    ):
        previous = os.environ.get("AETHER_FORCE_TENSOR_PARALLEL")
        if force:
            os.environ["AETHER_FORCE_TENSOR_PARALLEL"] = "1"
        else:
            os.environ.pop("AETHER_FORCE_TENSOR_PARALLEL", None)
        backend = AetherBackend(
            device=_device(), cache_dir=config.cache_dir,
            execution_devices=execution_devices, keep_artifact=config.keep_aeg,
        )
        record = runner.measure_load(backend, model, precision)
        entry: dict[str, Any] = {
            "model": model, "configuration": label, "gpus": gpus,
            "status": record.get("status"),
            "engine": (record.get("notes") or {}).get("engine"),
        }
        if record.get("status") == "ok":
            prompt_set = prompts.build_prompt_set(
                backend.tokenizer(), [config.prompt_tokens[0]]
            )
            prompt = next(iter(prompt_set.values()))["text"]
            measurement = runner.measure_generation(
                backend, prompt, max_new_tokens=config.max_new_tokens,
                temperature=0.0, top_p=1.0, top_k=0, seed=config.seed, batch_size=1,
                warmup_iters=config.warmup_iters, measure_iters=config.measure_iters,
                gpu_interval_s=config.gpu_sample_interval_s,
            )
            entry["status"] = measurement.get("status")
            entry["tokens_per_s"] = measurement.get("tokens_per_s")
            entry["latency_s"] = measurement.get("latency_s")
            entry["per_gpu_peak_bytes"] = [
                device["peak_reserved_bytes"]
                for device in ((measurement.get("gpu_peak") or {}).get("devices") or [])
            ]
            entry["gpu_telemetry"] = measurement.get("gpu_telemetry")
        cases.append(entry)
        backend.unload()
        gpu_monitor.empty_cache()
        if previous is None:
            os.environ.pop("AETHER_FORCE_TENSOR_PARALLEL", None)
        else:
            os.environ["AETHER_FORCE_TENSOR_PARALLEL"] = previous

    reference = TransformersBackend(device=_device())
    record = runner.measure_load(reference, model, precision)
    entry = {"model": model, "configuration": "transformers single device",
             "gpus": 1, "status": record.get("status"), "engine": "transformers"}
    if record.get("status") == "ok":
        prompt_set = prompts.build_prompt_set(
            reference.tokenizer(), [config.prompt_tokens[0]]
        )
        prompt = next(iter(prompt_set.values()))["text"]
        measurement = runner.measure_generation(
            reference, prompt, max_new_tokens=config.max_new_tokens,
            temperature=0.0, top_p=1.0, top_k=0, seed=config.seed, batch_size=1,
            warmup_iters=config.warmup_iters, measure_iters=config.measure_iters,
            gpu_interval_s=config.gpu_sample_interval_s,
        )
        entry["status"] = measurement.get("status")
        entry["tokens_per_s"] = measurement.get("tokens_per_s")
        entry["latency_s"] = measurement.get("latency_s")
        entry["per_gpu_peak_bytes"] = [
            device["peak_reserved_bytes"]
            for device in ((measurement.get("gpu_peak") or {}).get("devices") or [])
        ]
    cases.append(entry)
    reference.unload()
    return {"cases": cases, "notes": notes}


def run_batch_scaling(config: BenchmarkConfig) -> dict[str, Any]:
    """Mode 6: what batching actually buys, with latency and throughput separated.

    The performance matrix already sweeps batch sizes, but it reports one
    throughput column, and for batching that single number hides the trade-off:
    aggregate tokens per second rises while each individual request gets no
    faster — and usually gets slower. This mode reports both figures side by side
    for every batch width, so a reader cannot mistake one for the other.

    ``batch tok/s`` is the whole pass's output over the pass's wall time.
    ``per-request tok/s`` is one row's output over that same wall time: what a
    single caller waiting in the batch experiences. At batch 1 they coincide,
    which is the baseline both are measured against.
    """
    cases: list[dict[str, Any]] = []
    precision = config.precisions[0]
    prompt_tokens = config.prompt_tokens[0]
    notes = [
        "Each cell replicates one prompt to the batch width, which is what the "
        "Transformers backend does for its own batch cells, so both runtimes are "
        "measured on identical work and neither batch carries padding.",
        "`batch tok/s` is aggregate output per second of wall time; "
        "`per-request tok/s` is what one caller inside the batch sees. Batching is "
        "a throughput mechanism, so the first is expected to rise and the second "
        "is not.",
        "`scaling` divides a cell's aggregate throughput by that backend's own "
        "batch-1 aggregate throughput. 1.00x means batching bought nothing.",
        "The `engine` column records which executor served each cell. On a host "
        "with no accelerator, Aether's batch>1 cells run on the portable tensor "
        "executor while batch=1 runs on the NumPy kernels, so a scaling ratio "
        "across those two rows compares different engines; such rows are flagged "
        "rather than presented as a batching speedup.",
    ]
    for model in config.models:
        _log(f"\n=== batch scaling: {model} @ {precision} ===")
        backends, loads, _ = _load_pair(config, model, precision)
        if not backends:
            cases.append({
                "model": model, "precision": precision, "status": "load-failed",
                "message": "neither backend loaded; see load records",
            })
            continue
        reference_tokenizer = next(iter(backends.values())).tokenizer()
        prompt_set = prompts.build_prompt_set(reference_tokenizer, [prompt_tokens])
        prompt = next(iter(prompt_set.values()))
        baseline: dict[str, float] = {}
        baseline_engine: dict[str, Any] = {}
        for batch in config.batch_sizes:
            for name in BACKENDS:
                backend = backends.get(name)
                entry: dict[str, Any] = {
                    "model": model, "precision": precision,
                    "prompt_tokens": prompt["achieved_tokens"], "batch_size": batch,
                    "backend": name,
                }
                if backend is None:
                    entry.update(status="load-failed", message="backend did not load")
                    cases.append(entry)
                    continue
                _log(f"  [{name}] b{batch} ...")
                measurement = runner.measure_generation(
                    backend, prompt["text"],
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature, top_p=config.top_p,
                    top_k=config.top_k, seed=config.seed, batch_size=batch,
                    warmup_iters=config.warmup_iters,
                    measure_iters=config.measure_iters,
                    gpu_interval_s=config.gpu_sample_interval_s,
                )
                entry["status"] = measurement.get("status")
                if measurement.get("status") != "ok":
                    entry["message"] = measurement.get("message", "")
                    _log(f"      {entry['status']}: {entry['message'][:110]}")
                    cases.append(entry)
                    continue
                latency = (measurement.get("latency_s") or {}).get("median")
                # ``measure_generation`` already scales by batch width, so this is
                # the aggregate figure; one row's share is it divided by the width.
                aggregate = (measurement.get("tokens_per_s") or {}).get("median")
                row_tokens = int(measurement.get("completion_tokens") or 0)
                entry.update({
                    "latency_s": measurement.get("latency_s"),
                    "batch_tokens_per_s": aggregate,
                    "per_request_tokens_per_s": (
                        None if aggregate is None else aggregate / max(batch, 1)
                    ),
                    "requests_per_s": (
                        None if not latency else batch / latency
                    ),
                    "row_completion_tokens": row_tokens,
                    "gpu_inference_delta_bytes": measurement.get("gpu_inference_delta_bytes"),
                    "cold_latency_s": measurement.get("cold_latency_s"),
                    "backend_metrics": measurement.get("backend_metrics"),
                    # Which executor actually served the cell.  On a CPU host
                    # Aether's batch>1 cells run on the portable tensor executor
                    # while batch=1 runs on the NumPy kernels, so a scaling ratio
                    # across them would compare two different engines.  Recording
                    # the engine is what lets the report say so instead of hiding it.
                    "engine": (measurement.get("backend_metrics") or {}).get("engine"),
                })
                if batch == 1 and aggregate:
                    baseline[name] = float(aggregate)
                    baseline_engine[name] = entry.get("engine")
                if aggregate and baseline.get(name):
                    entry["scaling_vs_batch1"] = float(aggregate) / baseline[name]
                    entry["scaling_is_same_engine"] = (
                        entry.get("engine") == baseline_engine.get(name)
                    )
                _log(
                    f"      batch {aggregate:.2f} tok/s, "
                    f"per-request {aggregate / max(batch, 1):.2f} tok/s, "
                    f"latency {latency:.4f}s"
                )
                cases.append(entry)
            runner.cooldown(config.cooldown_s)
        for backend in backends.values():
            backend.unload()
        gpu_monitor.empty_cache()
    return {"cases": cases, "notes": notes, "loads": {}}


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    _limit_devices(config.devices)

    _log("=" * 72)
    _log("Aether Runtime vs Hugging Face Transformers — benchmark")
    _log("=" * 72)
    _log("\nConfiguration:")
    _log(config.describe())

    off_charter = [name for name in config.models if not is_charter_model(name)]
    if off_charter:
        _log(
            "\nNOTE: these entries are not benchmark models; they are being used "
            f"to validate the harness only: {off_charter}"
        )

    environment = system_info.collect(config.models)
    gpu = environment.get("gpu", {})
    _log(f"\nAccelerators: {gpu.get('count', 0)}")
    for device in gpu.get("devices", []):
        _log(f"  cuda:{device['index']}  {device['name']}  "
             f"{device['total_memory_bytes'] / 1024 ** 3:.1f} GiB  "
             f"sm_{device['compute_capability'].replace('.', '')}")
    if not gpu.get("available"):
        _log("  none — running on CPU; results will not reflect accelerator behaviour.")

    thermal_before = gpu_monitor.temperature_snapshot()
    payload: dict[str, Any] = {
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "config": json.loads(config.describe()),
        "environment": environment,
        "thermal": {"before": thermal_before},
        "skips": [],
    }
    if off_charter:
        payload["harness_validation_only"] = off_charter

    modes = ["performance", "correctness", "profile", "batch", "multigpu"] \
        if config.mode == "all" else [config.mode]
    if config.mode == "memory":
        # Memory is measured inside the performance path; the mode exists to make
        # that explicit rather than to run a second, differently-timed experiment.
        modes = ["performance"]
        payload["memory_mode_note"] = (
            "Memory figures are collected during the performance run; this mode "
            "runs that path and reports the memory sections."
        )

    for mode in modes:
        _log(f"\n{'=' * 72}\nMODE: {mode}\n{'=' * 72}")
        if mode == "performance":
            cells, skips = run_performance(config)
            payload["performance"] = cells
            payload["skips"].extend(skips)
        elif mode == "correctness":
            payload["correctness"] = run_correctness(config)
        elif mode == "profile":
            payload["kernels"] = run_profile(config)
        elif mode == "batch":
            payload["batch_scaling"] = run_batch_scaling(config)
        elif mode == "multigpu":
            if config.multi_gpu:
                payload["multigpu"] = run_multigpu(config)
            else:
                payload["multigpu"] = {"cases": [], "notes": ["Disabled with --no-multi-gpu."]}
        runner.cooldown(config.cooldown_s)

    payload["thermal"]["after"] = gpu_monitor.temperature_snapshot()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = reporting.write_charts(payload.get("performance", []), output_dir)
    written = reporting.save_raw(payload, output_dir)
    payload["csv"] = written.get("csv")
    report_path = output_dir / "REPORT.md"
    report_path.write_text(reporting.build_report(payload, charts), encoding="utf-8")

    _log(f"\n{'=' * 72}")
    _log("Results written:")
    _log(f"  {report_path}")
    for name in written.values():
        _log(f"  {output_dir / name}")
    for name in charts:
        _log(f"  {output_dir / name}")
    _log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
