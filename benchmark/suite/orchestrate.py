"""Drive the whole suite: survey the field, spawn the workers, collect the raw data.

The orchestrator never runs a model itself. It decides what should be measured,
hands each measurement to a worker process, and keeps whatever comes back -
including the records that say a measurement did not happen and why.

Two ordering rules it is responsible for:

* **workers run one at a time.** Two engines measured concurrently would contend
  for the same cores, the same memory bandwidth and the same device, so every
  number would be a measurement of the contention instead of the engine.
* **engine order rotates per model.** A host that drifts thermally or ramps clocks
  over a long run would otherwise always penalize whichever engine goes last. The
  order actually used is recorded with each model's results.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.suite import SUITE_VERSION
from benchmark.suite import engines as registry
from benchmark.suite import hardware as hardware_mod
from benchmark.suite import status as status_mod
from benchmark.suite.plan import SuiteConfig

#: Directory layout. Fixed, because a report that points at its own inputs is only
#: useful if the paths are predictable.
RAW_DIR = "raw"
GRAPH_DIR = "graphs"
REPORT_DIR = "reports"
ARTIFACT_DIR = "artifacts"


def _log(message: str) -> None:
    print(message, flush=True)


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _slug(text: str) -> str:
    return text.replace("/", "--").replace(" ", "_")


#: Failure reasons are printed in full, wrapped, rather than truncated. A reason cut
#: off at 100 characters routinely loses the actionable half - the required version,
#: the compute capability, the flag to pass - which turns a diagnosable result into a
#: dead end. The whole text is in the raw JSON either way; this is about the operator
#: watching the run.
_REASON_WIDTH = 96


def _wrap(text: str, indent: str = "") -> list[str]:
    """Wrap one reason for the console, preserving its own line breaks."""
    import textwrap

    if not text:
        return []
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        stripped = paragraph.strip()
        if not stripped:
            continue
        lines.extend(
            textwrap.wrap(
                stripped, width=_REASON_WIDTH, initial_indent=indent,
                subsequent_indent=indent, break_long_words=False,
                break_on_hyphens=False,
            )
            or [indent + stripped]
        )
    return lines


def model_facts(model_id: str) -> dict[str, Any]:
    """Everything about the checkpoint the report is required to state.

    Read from the repository's own config and tokenizer, plus the hub's file
    metadata for the parameter count. Each field is None when it could not be
    determined, never a guess: an unresolvable revision printed as a dash is
    honest, and a fabricated one is not.
    """
    facts: dict[str, Any] = {
        "model_id": model_id,
        "revision": None,
        "architecture": None,
        "parameters": None,
        "hidden_size": None,
        "layers": None,
        "attention_heads": None,
        "kv_heads": None,
        "vocab_size": None,
        "context_length": None,
        "checkpoint_dtype": None,
        "tokenizer_class": None,
        "tokenizer_vocab_size": None,
        "chat_template": None,
    }
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        facts.update(
            architecture=(getattr(config, "architectures", None) or [None])[0],
            hidden_size=getattr(config, "hidden_size", None),
            layers=getattr(config, "num_hidden_layers", None),
            attention_heads=getattr(config, "num_attention_heads", None),
            kv_heads=getattr(config, "num_key_value_heads", None),
            vocab_size=getattr(config, "vocab_size", None),
            context_length=getattr(config, "max_position_embeddings", None),
            checkpoint_dtype=str(
                getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
                or "unspecified"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        facts["config_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        facts.update(
            tokenizer_class=type(tokenizer).__name__,
            tokenizer_vocab_size=len(tokenizer),
            chat_template=bool(getattr(tokenizer, "chat_template", None)),
        )
    except Exception as exc:  # noqa: BLE001
        facts["tokenizer_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=False)
        facts["revision"] = info.sha
        safetensors = getattr(info, "safetensors", None)
        if safetensors is not None:
            facts["parameters"] = getattr(safetensors, "total", None)
            facts["parameter_dtypes"] = dict(getattr(safetensors, "parameters", {}) or {})
    except Exception as exc:  # noqa: BLE001 - offline is not fatal
        facts["hub_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return facts


def build_workload(config: SuiteConfig, model_id: str, precision: str,
                   precision_reason: str) -> dict[str, Any]:
    """Build the prompts once, here, so every engine receives identical strings.

    This is the single most important fairness mechanism in the suite. If each
    worker built its own prompts, a tokenizer difference would silently give two
    engines different amounts of work while the table claimed they had the same. The
    prompts are constructed with the model's own tokenizer to an exact token count,
    written into the workload file, and every worker then verifies that its engine's
    tokenizer encodes those exact strings to the same ids.
    """
    from transformers import AutoTokenizer

    from benchmark import prompts as prompt_mod

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    lengths = sorted({*config.prompt_tokens, config.primary_prompt_tokens})
    built = prompt_mod.build_prompt_set(tokenizer, lengths)
    prompts = {
        str(entry["requested_tokens"]): {
            "text": entry["text"],
            "requested_tokens": entry["requested_tokens"],
            "achieved_tokens": entry["achieved_tokens"],
            "exact": entry["exact"],
        }
        for entry in built.values()
    }
    return {
        "model": model_id,
        "precision": precision,
        "precision_reason": precision_reason,
        "prompt_builder_tokenizer": type(tokenizer).__name__,
        "prompts": prompts,
        "inexact_lengths": [
            entry["requested_tokens"] for entry in built.values() if not entry["exact"]
        ],
    }


def _worker_command(plan_path: Path, workload_path: Path, engine: str, model: str,
                    out_path: Path, mode: str) -> list[str]:
    return [
        sys.executable, "-m", "benchmark.suite.worker",
        "--plan", str(plan_path),
        "--workload", str(workload_path),
        "--engine", engine,
        "--model", model,
        "--out", str(out_path),
        "--mode", mode,
    ]


def _spawn(command: list[str], timeout_s: float, cwd: Path) -> dict[str, Any]:
    """Run one worker to completion, capturing what it printed.

    A worker that dies without writing its result file is the one case the
    orchestrator has to synthesize a record for, and it records exactly that: the
    process outcome, not a measurement.
    """
    start = time.perf_counter()
    environment = dict(os.environ)
    # Keep each worker's Python path pointing at this checkout, so a worker started
    # from an installed console script still imports the code under test.
    existing = environment.get("PYTHONPATH", "")
    roots = os.pathsep.join([str(cwd), str(cwd / "src")])
    environment["PYTHONPATH"] = f"{roots}{os.pathsep}{existing}" if existing else roots
    try:
        completed = subprocess.run(
            command, cwd=str(cwd), env=environment, capture_output=True, text=True,
            timeout=timeout_s,
        )
        return {
            "returncode": completed.returncode,
            "elapsed_s": time.perf_counter() - start,
            "stdout_tail": (completed.stdout or "")[-4000:],
            "stderr_tail": (completed.stderr or "")[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "elapsed_s": time.perf_counter() - start,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def _rotate(keys: list[str], offset: int) -> list[str]:
    """Rotate the engine order, so no engine is always last."""
    if not keys:
        return keys
    index = offset % len(keys)
    return keys[index:] + keys[:index]


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def run(config: SuiteConfig) -> dict[str, Any]:
    """Execute the plan and return the raw payload, unanalyzed."""
    from benchmark import gpu_monitor, system_info

    root = Path(config.output_dir)
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]

    hardware = hardware_mod.detect()
    precision, precision_reason = hardware_mod.resolve_precision(config.precision, hardware)
    if config.pin_threads and config.threads is None:
        config.threads = hardware_mod.default_threads(hardware)
    # Build artifacts belong with the run that produced them, so --output-dir moves
    # them too. Set here rather than in the dataclass defaults so the resolved paths
    # land in plan.json, where a reader can see exactly where each engine's artifact
    # went and how big it was.
    artifacts = root / ARTIFACT_DIR
    for attribute, name in (
        ("aeg_cache_dir", "aeg"), ("onnx_cache_dir", "onnx"),
        ("openvino_cache_dir", "openvino"), ("gguf_dir", "gguf"),
    ):
        if getattr(config, attribute) is None:
            setattr(config, attribute, str(artifacts / name))
    plan_path = raw_dir / "plan.json"
    plan_dict = config.to_dict()
    plan_dict.update(
        resolved_precision=precision,
        precision_reason=precision_reason,
        correctness_tokens=64,
    )
    plan_path.write_text(json.dumps(plan_dict, indent=2, sort_keys=True), encoding="utf-8")

    _log("=" * 78)
    _log(f"Multi-engine inference benchmark - suite {SUITE_VERSION}")
    _log("=" * 78)
    _log(f"\nHardware: {hardware.accelerator}, "
         f"{hardware.gpu_count} accelerator(s) present, "
         f"{hardware.physical_cores or hardware.logical_cores} physical cores")
    for index, name in enumerate(hardware.gpu_names):
        vram = hardware.gpu_vram_bytes[index] / 1024 ** 3
        _log(f"  cuda:{index}  {name}  {vram:.1f} GiB  "
             f"sm_{hardware.compute_capabilities[index].replace('.', '')}")
    if hardware.nvidia:
        _log("Accelerators visible to each engine: "
             + (f"{config.devices} (every engine sees the same device, so no runtime is "
                "measured on more hardware than another)" if config.devices
                else "unrestricted - engines that shard will use more hardware than "
                     "engines that do not"))
    _log(f"Precision: {precision}")
    for line in _wrap(precision_reason, indent="  "):
        _log(line)
    _log(f"Threads pinned to {config.threads}" if config.pin_threads
         else "Thread counts inherited from the environment and NOT controlled")
    if config.excluded_engines:
        _log(f"Engines excluded by request: {', '.join(config.excluded_engines)}")
    _log(f"Models: {len(config.models)}   Engines requested: {len(config.engines)}")

    payload: dict[str, Any] = {
        "suite_version": SUITE_VERSION,
        "generated_at": _now(),
        "plan": plan_dict,
        "workload_signature": config.workload_signature(),
        "hardware": hardware.to_dict(),
        "environment": system_info.collect(list(config.models)),
        "engine_catalogue": {
            key: spec.to_dict() for key, spec in registry.specs().items()
        },
        "thermal": {"before": gpu_monitor.temperature_snapshot()},
        "models": {},
        "runs": [],
        "reuse_runs": [],
        "worker_processes": [],
    }

    from benchmark.config import is_charter_model

    off_charter = [name for name in config.models if not is_charter_model(name)]
    if off_charter:
        # Accepted so the harness itself can be validated without the charter
        # models, but labelled everywhere so no number from such a run can be
        # mistaken for a benchmark result.
        payload["harness_validation_only"] = off_charter
        _log("")
        _log("NOTE: these entries are not benchmark models. They exercise the "
             f"harness only and are labelled as such in the report: {off_charter}")

    for model_index, model_id in enumerate(config.models):
        _log(f"\n{'=' * 78}\nMODEL: {model_id}\n{'=' * 78}")
        # Preparation can fail for reasons that have nothing to do with any engine:
        # an unreachable hub, a missing tokenizer, a checkpoint that will not parse.
        # One model failing must not end the run, so the failure is recorded against
        # that model and the next one is attempted.
        try:
            facts = model_facts(model_id)
            workload = build_workload(config, model_id, precision, precision_reason)
        except BaseException as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"[:400]
            _log(f"  preparation failed: {reason}")
            payload["models"][model_id] = {
                "facts": {"model_id": model_id, "error": reason},
                "workload": None,
                "availability": {},
                "engine_order": [],
                "status": status_mod.FAILED,
                "reason": (
                    "the model could not be prepared, so no engine was measured on "
                    "it: " + reason
                ),
            }
            for key in config.engines:
                payload["runs"].append({
                    "engine": key, "model": model_id, "precision": precision,
                    "mode": "full", "status": status_mod.SKIPPED,
                    "reason": f"model preparation failed: {reason}",
                    "spec": registry.spec_for(key).to_dict(), "cells": [],
                })
            continue
        workload_path = raw_dir / f"workload__{_slug(model_id)}.json"
        workload_path.write_text(json.dumps(workload, indent=2), encoding="utf-8")
        if workload["inexact_lengths"]:
            _log(f"  note: tokenizer could not hit exactly {workload['inexact_lengths']} "
                 "tokens; the achieved counts are recorded and identical for every engine")

        availability = registry.probe_all(
            hardware, model_id, precision, config, keys=list(config.engines)
        )
        for key in config.excluded_engines:
            # An engine the operator removed is still part of the field survey. It is
            # reported as SKIPPED with that reason, so a reader can tell "left out"
            # from "could not run here".
            payload["runs"].append({
                "engine": key, "model": model_id, "precision": precision,
                "mode": "full", "status": status_mod.SKIPPED,
                "reason": "excluded from this run with --exclude-engines",
                "spec": registry.spec_for(key).to_dict(), "cells": [],
            })
        # A model too large for the smallest visible accelerator is reported as
        # inapplicable for every device-bound engine, up front, rather than as a
        # sequence of identical out-of-memory failures.
        fits = hardware_mod.can_hold_weights(hardware, facts.get("parameters"), precision)
        if not fits:
            _log(f"  warning: {facts.get('parameters')} parameters at {precision} may not "
                 f"fit the smallest visible device; engines that fail are recorded as OOM")
        order = _rotate(list(config.engines), model_index)
        payload["models"][model_id] = {
            "facts": facts,
            "workload": workload,
            "availability": {key: value.to_dict() for key, value in availability.items()},
            "engine_order": order,
            "fits_smallest_device": fits,
        }
        _log("\n  engine availability:")
        for key in order:
            entry = availability[key]
            mark = "run " if entry.usable else "skip"
            _log(f"    [{mark}] {key:15s} {entry.status:15s} "
                 f"{(entry.version or '-'):12s}")
            # The reason is what the operator acts on, so it is wrapped rather than
            # truncated. A cut-off explanation is the same as no explanation.
            for line in _wrap(entry.reason, indent=" " * 14):
                _log(line)

        for key in order:
            entry = availability[key]
            out_path = raw_dir / f"{key}__{_slug(model_id)}.json"
            if not entry.usable:
                record = {
                    "engine": key, "model": model_id, "precision": precision,
                    "mode": "full", "status": entry.status, "reason": entry.reason,
                    "spec": registry.spec_for(key).to_dict(),
                    "availability": entry.to_dict(), "cells": [],
                }
                out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                payload["runs"].append(record)
                continue
            if config.resume and out_path.exists():
                existing = _read_record(out_path)
                if existing is not None:
                    _log(f"  [{key}] reusing existing result ({existing.get('status')})")
                    payload["runs"].append(existing)
                    continue
            _log(f"\n  [{key}] measuring ...")
            process = _spawn(
                _worker_command(plan_path, workload_path, key, model_id, out_path, "full"),
                config.worker_timeout_s, repository,
            )
            process.update(engine=key, model=model_id, mode="full")
            payload["worker_processes"].append(process)
            record = _read_record(out_path)
            if record is None:
                record = _orphan_record(key, model_id, precision, process)
                out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            payload["runs"].append(record)
            _log(f"  [{key}] {record.get('status')} in {process['elapsed_s']:.1f}s")
            if record.get("reason"):
                for line in _wrap(str(record["reason"]), indent=" " * 8):
                    _log(line)

        if config.reuse_probe:
            _run_reuse_probes(config, payload, model_id, order, raw_dir, plan_path,
                              workload_path, repository, precision)

    payload["thermal"]["after"] = gpu_monitor.temperature_snapshot()
    payload["finished_at"] = _now()
    return payload


#: Persistence classes for which a second-process reload is a meaningful
#: measurement. An engine that compiles nothing has nothing to reload, and one
#: whose compilation is process-local would simply pay it again.
_REUSABLE = {
    "portable-artifact",
    "on-disk-cache",
}


def _run_reuse_probes(config: SuiteConfig, payload: dict[str, Any], model_id: str,
                      order: list[str], raw_dir: Path, plan_path: Path,
                      workload_path: Path, repository: Path, precision: str) -> None:
    """Reload each build-phase engine's artifact in a brand-new process.

    Answers the compile-once question with evidence instead of architecture talk:
    for every engine that claims to leave something behind, a fresh process is asked
    to load it and run once. What it costs, and whether it works at all, is recorded
    per engine - including for the engines whose answer is that there was nothing to
    reuse.
    """
    measured = {
        record["engine"] for record in payload["runs"]
        if record.get("model") == model_id and record.get("status") == status_mod.MEASURED
    }
    _log("\n  compile-once probe (fresh process, artifact reload):")
    for key in order:
        spec = registry.spec_for(key)
        if key not in measured:
            payload["reuse_runs"].append({
                "engine": key, "model": model_id, "mode": "reuse",
                "status": status_mod.SKIPPED,
                "reason": "the full run for this engine produced no measurement",
            })
            continue
        if spec.artifact_persistence not in _REUSABLE:
            payload["reuse_runs"].append({
                "engine": key, "model": model_id, "mode": "reuse",
                "status": status_mod.NOT_APPLICABLE,
                "persistence": spec.artifact_persistence,
                "reason": (
                    f"{spec.display} leaves no reusable build behind "
                    f"({spec.artifact_persistence}), so a second process has nothing "
                    "to load and pays the same start-up cost as the first"
                ),
            })
            _log(f"    [{key:15s}] not applicable - {spec.artifact_persistence}")
            continue
        out_path = raw_dir / f"reuse__{key}__{_slug(model_id)}.json"
        if config.resume and out_path.exists():
            existing = _read_record(out_path)
            if existing is not None:
                payload["reuse_runs"].append(existing)
                _log(f"    [{key:15s}] reusing existing probe "
                     f"({existing.get('status')})")
                continue
        process = _spawn(
            _worker_command(plan_path, workload_path, key, model_id, out_path, "reuse"),
            config.worker_timeout_s, repository,
        )
        process.update(engine=key, model=model_id, mode="reuse")
        payload["worker_processes"].append(process)
        record = _read_record(out_path) or _orphan_record(key, model_id, precision, process)
        record.setdefault("mode", "reuse")
        payload["reuse_runs"].append(record)
        load = (record.get("load") or {}).get("total_s")
        _log(f"    [{key:15s}] {record.get('status')}"
             + (f", artifact reload {load:.2f}s" if isinstance(load, (int, float)) else ""))


def _orphan_record(engine: str, model_id: str, precision: str,
                   process: dict[str, Any]) -> dict[str, Any]:
    """A record for a worker that died without writing one.

    This is the only record the orchestrator authors, and it contains no
    measurement: just the process outcome and the tail of what the worker printed,
    which is what a crash actually leaves behind.
    """
    reason = (
        f"worker exceeded the {process.get('elapsed_s', 0):.0f}s timeout and was killed"
        if process.get("timed_out")
        else f"worker exited {process.get('returncode')} without writing a result"
    )
    return {
        "engine": engine,
        "model": model_id,
        "precision": precision,
        "mode": "full",
        "status": status_mod.FAILED,
        "reason": reason,
        "spec": registry.spec_for(engine).to_dict(),
        "cells": [],
        "process": process,
    }
