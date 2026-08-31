"""One engine, one model, one process.

Every measurement in the suite is taken here, in a process that holds exactly one
engine. That isolation buys three things the in-process harness cannot have:

* a serving engine that reserves all of device memory, or a native library that
  segfaults, takes down its own worker and nothing else;
* peak host memory is attributable to one engine, because no other engine's
  weights were ever resident;
* cold start is real. The first inference in this process is genuinely the first
  inference, not the first after another engine warmed the allocator and the
  filesystem cache.

The worker measures; it does not compare. Everything it writes is a raw
observation plus the status that says whether the observation exists.

    python -m benchmark.suite.worker --plan plan.json --workload workload.json \
        --engine aether --model Qwen/Qwen3-0.6B --out raw/aether__Qwen--Qwen3-0.6B.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (_ROOT, _ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

MODE_FULL = "full"
MODE_REUSE = "reuse"


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class Options:
    """Engine knobs, reconstructed from the serialized plan.

    The adapters read their configuration off an attribute bag rather than off the
    dataclass directly, so a new engine can add an option without every other
    engine's signature changing.
    """

    def __init__(self, plan: dict[str, Any]) -> None:
        for key, value in plan.items():
            setattr(self, key, value)


def _cell(kind: str, batch: int, prompt_tokens: int, output_tokens: int,
          **extra: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "batch_size": batch,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        **extra,
    }


def _plan_cells(plan: dict[str, Any], prompts: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every configuration this worker will attempt, once each.

    The batch-1 primary cell belongs to all three sweeps, so it is emitted once and
    tagged with every sweep it serves. Measuring it three times would spend budget
    to produce three samples of the same thing and then have to explain which one
    the report quotes.
    """
    primary_prompt = int(plan["primary_prompt_tokens"])
    primary_output = int(plan["primary_output_tokens"])
    available = {int(key) for key in prompts}
    if primary_prompt not in available:
        primary_prompt = min(available, key=lambda value: abs(value - primary_prompt))

    cells: list[dict[str, Any]] = [
        _cell("primary", 1, primary_prompt, primary_output,
              sweeps=["batch", "prompt", "output"], is_primary=True)
    ]
    for batch in plan["batch_sizes"]:
        if int(batch) == 1:
            continue
        cells.append(_cell("batch", int(batch), primary_prompt, primary_output,
                           sweeps=["batch"]))
    for length in sorted(available):
        if length == primary_prompt:
            continue
        cells.append(_cell("prompt", 1, length, primary_output, sweeps=["prompt"]))
    for output in plan["output_tokens"]:
        if int(output) == primary_output:
            continue
        cells.append(_cell("output", 1, primary_prompt, int(output), sweeps=["output"]))
    return cells


def _batch_support(status: str) -> str:
    """Translate a cell status into the per-batch support vocabulary."""
    from benchmark.suite import status as status_mod

    if status == status_mod.MEASURED:
        return status_mod.BATCH_SUPPORTED
    if status == status_mod.OOM:
        return status_mod.BATCH_OOM
    return status_mod.BATCH_UNSUPPORTED


def _derive(measurement: dict[str, Any], batch: int, prefill_s: float | None,
            ttft_s: float | None) -> dict[str, Any]:
    """Turn one timed generation into the metric set the report needs.

    Every quantity here is arithmetic over measured values, and each one names its
    operands. Decode throughput is derived by removing measured prefill from
    measured end-to-end latency, so it carries the uncertainty of both - which is
    why the raw latency stays in the record beside it.
    """
    latency = measurement.get("latency_s") or {}
    end_to_end = latency.get("median")
    completion = int(measurement.get("completion_tokens") or 0)
    prompt_tokens = int(measurement.get("prompt_tokens") or 0)
    rows = max(int(batch), 1)
    generated_total = completion * rows

    derived: dict[str, Any] = {
        "end_to_end_latency_s": end_to_end,
        "total_tokens_per_s": (measurement.get("tokens_per_s") or {}).get("median"),
        "per_request_tokens_per_s": None,
        "decode_tokens_per_s": None,
        "prompt_tokens_per_s": None,
        "ttft_s": ttft_s,
        "tpot_ms": None,
        "inter_token_latency_ms": None,
        "prefill_s": prefill_s,
        "generated_tokens_total": generated_total,
        "prompt_tokens_per_row": prompt_tokens,
    }
    aggregate = derived["total_tokens_per_s"]
    if aggregate:
        derived["per_request_tokens_per_s"] = aggregate / rows
    if end_to_end and prefill_s is not None and end_to_end > prefill_s and completion > 1:
        decode_seconds = end_to_end - prefill_s
        # The first token comes out of prefill, so the decode loop produced
        # completion-1 tokens in the remaining time.
        derived["decode_tokens_per_s"] = (completion - 1) * rows / decode_seconds
        derived["tpot_ms"] = decode_seconds / (completion - 1) * 1000.0
        derived["inter_token_latency_ms"] = derived["tpot_ms"]
    elif end_to_end and completion:
        # No prefill measurement for this engine: report the end-to-end rate under
        # its own name and leave decode undefined rather than passing one off as
        # the other.
        derived["tpot_ms"] = end_to_end / completion * 1000.0
        derived["inter_token_latency_ms"] = derived["tpot_ms"]
    if prefill_s and prompt_tokens:
        derived["prompt_tokens_per_s"] = prompt_tokens * rows / prefill_s
    derived["latency_stats"] = latency
    derived["throughput_stats"] = measurement.get("tokens_per_s")
    derived["latency_per_token_stats"] = measurement.get("latency_per_token_ms")
    derived["cold_latency_s"] = measurement.get("cold_latency_s")
    derived["iterations"] = latency.get("n")
    derived["coefficient_of_variation"] = latency.get("coefficient_of_variation")
    return derived


def run_full(plan: dict[str, Any], workload: dict[str, Any], engine_key: str,
             model_id: str) -> dict[str, Any]:
    """Measure one engine on one model across the whole workload matrix."""
    from benchmark import gpu_monitor, runner, system_info
    from benchmark import prompts as prompt_mod
    from benchmark.memory_monitor import HostMemoryMonitor
    from benchmark.memory_monitor import snapshot as host_snapshot
    from benchmark.suite import engines as registry
    from benchmark.suite import hardware as hardware_mod
    from benchmark.suite import status as status_mod

    options = Options(plan)
    thread_record = hardware_mod.pin_threads(plan.get("threads"))
    # Device visibility is restricted before anything touches torch, because
    # CUDA_VISIBLE_DEVICES is only read when the CUDA context is created. Every
    # engine therefore sees the same device count, and none of them has its own
    # placement logic altered.
    device_record = hardware_mod.visible_devices(plan.get("devices"))
    hardware = hardware_mod.detect()
    precision = workload["precision"]
    prompts = {int(key): value for key, value in workload["prompts"].items()}

    record: dict[str, Any] = {
        "engine": engine_key,
        "model": model_id,
        "precision": precision,
        "mode": MODE_FULL,
        "started_at": _now(),
        "spec": registry.spec_for(engine_key).to_dict(),
        "threads": thread_record,
        "devices": device_record,
        "hardware": hardware.to_dict(),
        "host_before": host_snapshot(),
        "cells": [],
        "batch_support": {},
        "status": status_mod.FAILED,
    }

    availability = registry.probe(engine_key, hardware, model_id, precision, options)
    record["availability"] = availability.to_dict()
    if not availability.usable:
        record["status"] = availability.status
        record["reason"] = availability.reason
        record["finished_at"] = _now()
        return record

    try:
        engine = registry.build(engine_key, hardware, model_id, precision, options)
    except BaseException as exc:  # noqa: BLE001
        status, message = status_mod.from_exception(exc)
        record.update(status=status, reason=message,
                      traceback=traceback.format_exc(limit=8)[-2000:],
                      finished_at=_now())
        return record

    process_monitor = HostMemoryMonitor(max(plan.get("gpu_sample_interval_s") or 0.1, 0.05))
    process_monitor.__enter__()
    try:
        _measure_everything(record, plan, engine, engine_key, model_id, precision,
                            prompts, prompt_mod, runner, gpu_monitor, status_mod)
    except BaseException as exc:  # noqa: BLE001
        status, message = status_mod.from_exception(exc)
        record.setdefault("reason", message)
        record["status"] = status
        record["traceback"] = traceback.format_exc(limit=8)[-2000:]
    finally:
        process_monitor.__exit__(None, None, None)
        record["host_process"] = process_monitor.report()
        record["host_after"] = host_snapshot()
        # Teardown must never mask a result that has already been measured.
        with contextlib.suppress(BaseException):
            engine.unload()
        record["environment"] = system_info.collect([model_id])
        record["finished_at"] = _now()
    return record


def _measure_everything(record: dict[str, Any], plan: dict[str, Any], engine: Any,
                        engine_key: str, model_id: str, precision: str,
                        prompts: dict[int, Any], prompt_mod: Any, runner: Any,
                        gpu_monitor: Any, status_mod: Any) -> None:
    """Load, then walk the cell list, recording each outcome as data."""
    load = runner.measure_load(engine, model_id, precision)
    record["load"] = load
    if load.get("status") != "ok":
        record["status"] = status_mod.from_runner(load)
        record["reason"] = load.get("message", "load failed")
        return

    try:
        record["describe"] = engine.describe()
    except BaseException as exc:  # noqa: BLE001
        record["describe"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Confirm this engine tokenizes the shared prompt strings the same way the
    # orchestrator did. If it does not, the engines were not given the same work,
    # and every throughput comparison for this row has to be read with that in mind.
    try:
        from transformers import AutoTokenizer

        reference_tokenizer = AutoTokenizer.from_pretrained(model_id)
        record["tokenizer_agreement"] = prompt_mod.verify_tokenizer_agreement(
            reference_tokenizer, engine.tokenizer(),
            [entry["text"] for entry in prompts.values()],
        )
    except BaseException as exc:  # noqa: BLE001
        record["tokenizer_agreement"] = {
            "identical": None, "reason": f"{type(exc).__name__}: {exc}"[:200],
        }

    cells = _plan_cells(plan, {str(k): v for k, v in prompts.items()})
    for cell in cells:
        prompt = prompts.get(cell["prompt_tokens"])
        if prompt is None:
            cell.update(status=status_mod.SKIPPED,
                        reason=f"no prompt built for {cell['prompt_tokens']} tokens")
            record["cells"].append(cell)
            continue
        if not engine.supports_batch(cell["batch_size"]):
            cell.update(
                status=status_mod.NOT_SUPPORTED,
                reason=(
                    f"{engine_key} reports it cannot execute batch "
                    f"{cell['batch_size']} as one pass"
                ),
            )
            record["cells"].append(cell)
            record["batch_support"].setdefault(
                str(cell["batch_size"]), status_mod.BATCH_UNSUPPORTED
            )
            continue

        measurement = runner.measure_generation(
            engine, prompt["text"],
            max_new_tokens=cell["output_tokens"],
            temperature=plan["temperature"], top_p=plan["top_p"], top_k=plan["top_k"],
            seed=plan["seed"], batch_size=cell["batch_size"],
            warmup_iters=plan["warmup_iters"], measure_iters=plan["measure_iters"],
            gpu_interval_s=plan.get("gpu_sample_interval_s") or 0.1,
        )
        status = status_mod.from_runner(measurement)
        cell["status"] = status
        cell["measurement"] = measurement
        if status != status_mod.MEASURED:
            cell["reason"] = measurement.get("message", "")
        record["batch_support"].setdefault(str(cell["batch_size"]), _batch_support(status))
        if cell["batch_size"] in {int(value) for value in plan["batch_sizes"]}:
            record["batch_support"][str(cell["batch_size"])] = _batch_support(status)

        prefill_s = ttft_s = None
        if cell.get("is_primary") and status == status_mod.MEASURED:
            prefill_s, ttft_s = _measure_primary_extras(
                record, plan, engine, prompt, runner, status_mod
            )
        cell["derived"] = (
            _derive(measurement, cell["batch_size"], prefill_s, ttft_s)
            if status == status_mod.MEASURED else {}
        )
        record["cells"].append(cell)
        if plan.get("cooldown_s"):
            runner.cooldown(float(plan["cooldown_s"]))
        gpu_monitor.empty_cache()

    record["status"] = (
        status_mod.MEASURED
        if any(item.get("status") == status_mod.MEASURED for item in record["cells"])
        else status_mod.FAILED
    )
    if plan.get("correctness"):
        record["correctness_sample"] = _capture_output(
            engine, plan, prompts, status_mod
        )
    record["artifact"] = _artifact_record(record)


def _measure_primary_extras(record: dict[str, Any], plan: dict[str, Any], engine: Any,
                            prompt: dict[str, Any], runner: Any,
                            status_mod: Any) -> tuple[float | None, float | None]:
    """Prefill and time-to-first-token for the primary cell.

    Measured once, at batch 1 and the primary lengths, because these are the
    single-request latency figures and a batched aggregate cannot stand in for
    them. Both are recorded in two forms where the engine supports it: prefill over
    all prompt positions, and prefill as generation actually pays for it.
    """
    iterations = max(3, int(plan["measure_iters"]) // 2)
    all_logits = runner.measure_prefill(
        engine, prompt["text"], warmup_iters=1, measure_iters=iterations
    )
    serving = runner.measure_serving_prefill(
        engine, prompt["text"], warmup_iters=1, measure_iters=iterations
    )
    ttft = runner.measure_ttft(
        engine, prompt["text"], max_new_tokens=int(plan["primary_output_tokens"]),
        seed=plan["seed"], measure_iters=3,
    )
    record["prefill"] = {"all_logits": all_logits, "serving": serving}
    record["ttft"] = ttft

    # Prefer the serving configuration for derived decode figures: it is the work a
    # served request actually performs, and it is what every engine's own generate
    # path does.
    prefill_s = None
    for candidate in (serving, all_logits):
        if candidate.get("status") == "ok":
            prefill_s = (candidate.get("latency_s") or {}).get("median")
            record["prefill_source"] = (
                "serving" if candidate is serving else "all_logits"
            )
            break
    ttft_s = (ttft.get("ttft_s") or {}).get("median") if ttft.get("status") == "ok" else None
    return prefill_s, ttft_s


def _capture_output(engine: Any, plan: dict[str, Any], prompts: dict[int, Any],
                    status_mod: Any) -> dict[str, Any]:
    """Generate once, greedily, and keep exactly what came out.

    The comparison itself happens in the orchestrator, against the reference
    engine's capture. The worker's only job is to record this engine's own output
    faithfully, including the token ids where the engine can supply real ones.
    """
    length = int(plan.get("correctness_tokens") or 64)
    prompt_tokens = min(prompts, key=lambda value: abs(value - 64)) if prompts else None
    if prompt_tokens is None:
        return {"status": status_mod.SKIPPED, "reason": "no prompt available"}
    prompt = prompts[prompt_tokens]
    try:
        outcome = engine.generate(
            prompt["text"], max_new_tokens=length, temperature=0.0, top_p=1.0,
            top_k=0, seed=int(plan["seed"]), batch_size=1,
        )
    except BaseException as exc:  # noqa: BLE001
        status, message = status_mod.from_exception(exc)
        return {"status": status, "reason": message}
    ids_source = (outcome.backend_metrics or {}).get("token_ids_source", "engine output")
    return {
        "status": status_mod.MEASURED,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": length,
        "completion_tokens": outcome.completion_tokens,
        "token_ids": list(outcome.token_ids or []),
        "token_ids_source": ids_source,
        "text": outcome.text,
        "greedy": True,
    }


#: Keys a load record may carry that name a build artifact's size. Different
#: engines call it different things; the report needs one column.
_ARTIFACT_SIZE_KEYS = ("artifact_bytes", "gguf_bytes", "ir_bytes", "plan_bytes")


def _artifact_record(record: dict[str, Any]) -> dict[str, Any]:
    """What this engine's build phase produced, if it has one.

    Reported per engine rather than only for Aether, because the compile-once
    question is not "does Aether have an artifact" but "which of these systems can
    hand a built model to a second process, and what does that cost". An engine with
    no build phase answers with zeros and a plain statement that it has none.
    """
    load = record.get("load") or {}
    notes = load.get("notes") or {}
    describe = record.get("describe") or {}
    spec = record.get("spec") or {}
    size = next(
        (notes[key] for key in _ARTIFACT_SIZE_KEYS if notes.get(key) is not None), None
    )
    return {
        "has_build_phase": bool(spec.get("has_build_phase")),
        "persistence": spec.get("artifact_persistence"),
        "build_s": load.get("prepare_s"),
        "load_s": load.get("load_s"),
        "download_s": load.get("download_s"),
        "total_startup_s": load.get("total_s"),
        "artifact_bytes": size,
        "artifact_path": describe.get("artifact") or notes.get("gguf_path")
        or describe.get("exl2_path") or describe.get("mlc_model"),
        "built_this_run": bool(
            notes.get("compiled_this_run")
            or notes.get("exported_this_run")
            or notes.get("converted_this_run")
            or (notes.get("initial_compile_s") is not None)
        ),
        "notes": notes,
    }


def run_reuse(plan: dict[str, Any], workload: dict[str, Any], engine_key: str,
              model_id: str) -> dict[str, Any]:
    """Reload an already-built artifact in a fresh process, and time it.

    This is the evidence for compile-once-use-everywhere, and it has to be a
    separate process to be evidence at all: an in-process reload would still hold
    the engine's caches, its allocator arenas and the OS page cache from the build.
    Here the only thing carried over from the build is what was written to disk.

    A single generation follows, so the record also shows what the first inference
    after a cold artifact load costs - the number a deployment actually feels.
    """
    from benchmark import prompts as prompt_mod
    from benchmark import runner
    from benchmark.memory_monitor import snapshot as host_snapshot
    from benchmark.suite import engines as registry
    from benchmark.suite import hardware as hardware_mod
    from benchmark.suite import status as status_mod

    del prompt_mod  # prompts arrive pre-built; nothing to construct here
    options = Options(plan)
    thread_record = hardware_mod.pin_threads(plan.get("threads"))
    device_record = hardware_mod.visible_devices(plan.get("devices"))
    hardware = hardware_mod.detect()
    precision = workload["precision"]
    prompts = {int(key): value for key, value in workload["prompts"].items()}
    primary = min(prompts, key=lambda value: abs(value - int(plan["primary_prompt_tokens"])))

    record: dict[str, Any] = {
        "engine": engine_key,
        "model": model_id,
        "precision": precision,
        "mode": MODE_REUSE,
        "started_at": _now(),
        "spec": registry.spec_for(engine_key).to_dict(),
        "threads": thread_record,
        "devices": device_record,
        "host_before": host_snapshot(),
        "status": status_mod.FAILED,
    }
    availability = registry.probe(engine_key, hardware, model_id, precision, options)
    record["availability"] = availability.to_dict()
    if not availability.usable:
        record.update(status=availability.status, reason=availability.reason,
                      finished_at=_now())
        return record
    try:
        engine = registry.build(engine_key, hardware, model_id, precision, options)
        load = runner.measure_load(engine, model_id, precision)
        record["load"] = load
        if load.get("status") != "ok":
            record.update(status=status_mod.from_runner(load),
                          reason=load.get("message", "load failed"), finished_at=_now())
            return record
        record["describe"] = engine.describe()
        record["artifact"] = _artifact_record(record)
        first = runner.measure_generation(
            engine, prompts[primary]["text"],
            max_new_tokens=int(plan["primary_output_tokens"]),
            temperature=plan["temperature"], top_p=plan["top_p"], top_k=plan["top_k"],
            seed=plan["seed"], batch_size=1, warmup_iters=0, measure_iters=2,
            gpu_interval_s=plan.get("gpu_sample_interval_s") or 0.1,
        )
        record["first_inference"] = first
        record["status"] = status_mod.from_runner(first)
        engine.unload()
    except BaseException as exc:  # noqa: BLE001
        status, message = status_mod.from_exception(exc)
        record.update(status=status, reason=message,
                      traceback=traceback.format_exc(limit=8)[-2000:])
    record["host_after"] = host_snapshot()
    record["finished_at"] = _now()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.suite.worker")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default=MODE_FULL, choices=[MODE_FULL, MODE_REUSE])
    args = parser.parse_args(argv)

    plan = _read_json(args.plan)
    workload = _read_json(args.workload)
    runner_fn = run_full if args.mode == MODE_FULL else run_reuse
    try:
        record = runner_fn(plan, workload, args.engine, args.model)
    except BaseException as exc:  # noqa: BLE001 - a worker always leaves a record
        from benchmark.suite import status as status_mod

        record = {
            "engine": args.engine,
            "model": args.model,
            "mode": args.mode,
            "status": status_mod.FAILED,
            "reason": f"{type(exc).__name__}: {exc}"[:400],
            "traceback": traceback.format_exc(limit=10)[-3000:],
            "finished_at": _now(),
        }
    _write_json(args.out, record)
    # Exit 0 even for a failed engine: the failure is the result, and a non-zero
    # exit would make the orchestrator's process bookkeeping ambiguous between
    # "engine failed" and "worker never wrote anything".
    print(f"[worker] {args.engine} / {args.model} -> {record.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
