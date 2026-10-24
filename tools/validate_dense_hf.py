"""Offline dense AEG source parity, fresh-process load memory and decode profiles.

Uses complete, untrained tiny/27M checkpoints from the pass-audit fixtures.
Current compilation requires original-HF admission. Historical compilation is
explicitly diagnostic. Process ownership and RAM guards reuse the pass audit.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import time
import traceback
import tracemalloc
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

if __package__:
    from .pass_audit import (
        PASSES,
        admission,
        inventory,
        mark_phase,
        memory_point,
        reserve_bytes,
        run_worker,
    )
    from .pass_audit_fixtures import CONTINUATION, PROMPTS, create_fixture, prompt_text, write_json
else:
    from pass_audit import (
        PASSES,
        admission,
        inventory,
        mark_phase,
        memory_point,
        reserve_bytes,
        run_worker,
    )
    from pass_audit_fixtures import CONTINUATION, PROMPTS, create_fixture, prompt_text, write_json

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "core/weight_store.py",
    "core/quality_gate.py",
    "quantization/formats.py",
    "quantization/codecs.py",
    "runtime/aeg_loader.py",
    "runtime/cpu_engine.py",
    "runtime/torch_engine.py",
    "runtime/onnx_export.py",
    "kernels/native_cpu.py",
    "compiler/compiler.py",
    "compiler/weight_quantizer.py",
    "evaluation/admission_worker.py",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _array_bytes(value: Any) -> int:
    """Count the union of contiguous array spans, including aliased views once."""
    import numpy as np

    spans: list[tuple[int, int]] = []

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            if not (item.flags.c_contiguous or item.flags.f_contiguous):
                raise ValueError("Memory accounting requires contiguous array spans")
            start = int(item.__array_interface__["data"][0])
            spans.append((start, start + int(item.nbytes)))
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for member in item.values():
                visit(member)
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)

    visit(value)
    total, end = 0, 0
    for start, stop in sorted(spans):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def _compile(job: dict[str, Any], work: Path) -> dict[str, Any]:
    from aether import Compiler, CompilerConfig
    from aether.core.quality_gate import read_gate

    disabled: dict[str, Any] = {entry.flag: False for entry in PASSES}
    mark_phase(work, "compile")
    began = time.perf_counter()
    package = Compiler(
        CompilerConfig(
            **disabled,
            quality_mode="diagnostic" if job["diagnostic"] else "require_hf",
            quality_dataset=job["dataset"],
            quality_greedy_tokens=8,
            precision="FP32",
            precision_assignment_mode="uniform",
            targets=["cpu_generic"],
            skip_download=True,
            calibration_tokens=4,
            cache_dir=str(work / "compile-cache"),
        )
    ).compile(job["fixture"]["model"], output_path=job["artifact"])
    seconds = time.perf_counter() - began
    package.verify_integrity()
    return {
        "status": "ok",
        "compile_seconds": seconds,
        "artifact": job["artifact"],
        "files": inventory(package.root),
        "quality_gate": read_gate(package, allow_uncertified=job["diagnostic"]),
    }


def _rows_and_tokens(engine: Any, job: dict[str, Any], work: Path) -> dict[str, Any]:
    import numpy as np

    observations = {}
    for index, probe in enumerate(job["probes"]):
        prompt, labels = probe["prompt_ids"], probe["continuation_ids"]
        ids = np.asarray(prompt + labels[:-1], np.int64)
        rows, _ = engine.forward(ids)
        actual = rows[len(prompt) - 1 :]
        if not np.isfinite(actual).all():
            raise ValueError("Executor produced nonfinite logits")
        np.save(work / f"actual-{index}.npy", actual, allow_pickle=False)
        _, cache = engine.forward(np.asarray(prompt, np.int64))
        suffix, _ = engine.forward(np.asarray(labels[:-1], np.int64), cache)
        np.testing.assert_allclose(suffix, rows[len(prompt) :], rtol=3e-5, atol=3e-6)
        tokens = engine.generate(np.asarray(prompt, np.int64), max_tokens=8, temperature=0)
        observations[probe["id"]] = {
            "tokens": [int(token) for token in tokens],
            "logits_sha256": hashlib.sha256(
                (work / f"actual-{index}.npy").read_bytes()
            ).hexdigest(),
        }
    return observations


def _load_profile(job: dict[str, Any], work: Path) -> dict[str, Any]:
    import numpy as np

    from aether.kernels.native_cpu import NativeCPUKernels
    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.cpu_engine import CPUExecutionEngine

    # Warm code/native-library preparation, not model tensors or a prior model.
    provider = NativeCPUKernels()
    provider.ensure_compiled()
    gc.collect()
    before = mark_phase(work, "load")
    tracemalloc.start()
    began = time.perf_counter()
    engine = load_engine_from_path(job["artifact"], allow_uncertified=job["diagnostic"])
    seconds = time.perf_counter() - began
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = mark_phase(work, "loaded")
    if not isinstance(engine, CPUExecutionEngine):
        raise TypeError("Expected actual dense CPUExecutionEngine")
    result: dict[str, Any] = {
        "status": "ok",
        "load_seconds": seconds,
        "rss_before_load": before,
        "rss_after_load": after,
        "rss_load_delta": after - before,
        "traced_current_bytes": traced_current,
        "traced_peak_bytes": traced_peak,
        "weight_array_span_bytes": _array_bytes(engine.weights),
        "native_library_loaded": engine.kernels.is_native,
        "native_build_identity": engine.kernels.build_identity(),
        "memory_scope": (
            "Fresh process, warm code/library and filesystem cache; sampled RSS and "
            "traced NumPy/Python allocation. No cold-filesystem or native-allocation attribution."
        ),
    }
    if not job["exercise"]:
        return result
    mark_phase(work, "inference")
    result["observations"] = _rows_and_tokens(engine, job, work)
    series = []
    for probe in job["probes"]:
        prompt = np.asarray(probe["prompt_ids"], np.int64)
        for repeat in range(9):
            began = time.perf_counter()
            iterator = engine.generate_iter(prompt, max_tokens=8, temperature=0)
            first = next(iterator)
            first_seconds = time.perf_counter() - began
            tokens = [first, *iterator]
            elapsed = time.perf_counter() - began
            if tokens != result["observations"][probe["id"]]["tokens"]:
                raise ValueError("Repeated greedy generation changed IDs")
            if repeat >= 2:
                series.append(
                    {
                        "probe": probe["id"],
                        "first_output_seconds": first_seconds,
                        "decode_ids_per_second": (len(tokens) - 1) / (elapsed - first_seconds),
                    }
                )
    result["timings"] = series
    _, cache = engine.forward(np.asarray(job["probes"][1]["prompt_ids"], np.int64))
    result["cache_array_span_bytes"] = _array_bytes(cache)
    result["logical_kv_bytes"] = sum(
        int(length * key.shape[1] * key.shape[2] * key.dtype.itemsize * 2)
        for key, length in zip(cache.keys, cache.stored_lengths, strict=True)
        if key is not None
    )
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(20):
        engine.generate(np.asarray(job["probes"][0]["prompt_ids"], np.int64), 8)
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(30)
    (work / "decode-cprofile.txt").write_text(stream.getvalue().rstrip() + "\n", encoding="utf-8")
    return result


def _backend(job: dict[str, Any], work: Path) -> dict[str, Any]:
    from aether import Runtime, RuntimeConfig
    from aether.runtime.aeg_loader import load_engine_from_path

    mark_phase(work, "backend")
    source = load_engine_from_path(job["artifact"])
    if job["executor"] == "torch_cpu":
        from aether.runtime.torch_engine import TorchAEGEngine

        engine: Any = TorchAEGEngine(source, "cpu")
    elif job["executor"] == "onnx_cpu":
        import onnxruntime as ort

        from aether.runtime.onnx_export import ONNXAEGEngine, export_engine_to_onnx

        bundle = export_engine_to_onnx(source, work / "onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        engine = ONNXAEGEngine(
            bundle,
            *[
                ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                for path in (bundle.prefill_path, bundle.decode_path)
            ],
        )
    else:
        raise ValueError(f"Unsupported executor: {job['executor']!r}")
    result: dict[str, Any] = {"status": "ok", "observations": _rows_and_tokens(engine, job, work)}
    if job["executor"] == "torch_cpu":
        for backend in ("aether_cpu", "pytorch"):
            runtime = Runtime(
                RuntimeConfig(
                    backend_name=backend,
                    execution_devices=["cpu"],
                    hf_offline=True,
                    speculative_decoding=False,
                    enable_semantic_cache=False,
                    enable_prefix_caching=False,
                    model_cache_dir=str(work / ("runtime-" + backend)),
                )
            )
            try:
                response = runtime.generate(
                    job["artifact"], job["public_prompt"], max_tokens=8, temperature=0
                )
                text = "".join(
                    runtime.generate_stream(
                        job["artifact"],
                        job["public_prompt"],
                        max_tokens=8,
                        temperature=0,
                    )
                )
                observed = {"text": response.text, "tokens": response.usage["completion_tokens"]}
                if observed != job["public_expected"] or text != response.text:
                    raise ValueError(f"{backend} public generation/stream differs from original HF")
                result[backend] = observed
            finally:
                runtime.close()
    return result


def _worker(path: Path) -> int:
    job = _read(path)
    work = Path(job["work"])
    mark_phase(work, "imports")
    sys.path.insert(0, job["expected_src"])
    try:
        import aether

        package = Path(aether.__file__).resolve().parent
        if not package.is_relative_to(Path(job["expected_src"]).resolve()):
            raise RuntimeError(f"Wrong checkout imported: {package}")
        operation = job["operation"]
        actions = {"compile": _compile, "load": _load_profile, "backend": _backend}
        if operation not in actions:
            raise ValueError(f"Unsupported worker operation: {operation!r}")
        result = actions[operation](job, work)
        result["code_root"] = str(package)
        result["code_sha256"] = {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in CODE_FILES
        }
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "failed", "reason": str(exc)}
        code = 1
    result["final_memory_point"] = memory_point("complete")
    write_json(work / "result.json", result)
    return code


def main() -> int:
    """Run admitted workers serially and retain every measured or blocked result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--fixtures", nargs="+", choices=("tiny", "mid"), default=["tiny", "mid"])
    parser.add_argument("--load-repeats", type=int, default=7)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        return _worker(args.worker)
    if args.output is None or args.load_repeats < 3:
        parser.error("--output and at least three --load-repeats are required")
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AETHER_TORCH_DTYPE="float32")
    import numpy as np
    import psutil

    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    policy = argparse.Namespace(
        min_free_mib=512, native=False, epoch=1700000000, threads=1, timeout=1200
    )
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Untrained dense FP32 source parity and load/decode measurements",
        "all_optimizer_passes_disabled": True,
        "load_sampling": "Two warmup processes, then alternating role order for each sample",
        "source_revisions": {
            role: subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            for role, source in (("current", ROOT), ("baseline", args.baseline_src))
            if source is not None
        },
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "host_ram_bytes": psutil.virtual_memory().total,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "transformers",
                "safetensors",
                "tokenizers",
                "psutil",
                "onnxruntime",
            )
        },
        "fixture_sha256": hashlib.sha256(
            (ROOT / "tools/pass_audit_fixtures.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }

    def launch(job: dict[str, Any], destination: Path) -> dict[str, Any]:
        observed: dict[str, Any] = run_worker(
            job, destination, policy, worker_script=Path(__file__)
        )
        if (
            observed["status"] != "ok"
            or observed["memory"]["worker_identity_status"] != "verified_owned"
        ):
            report["failed_worker"] = {"directory": str(destination), "observation": observed}
            write_json(work / "report.json", report)
            raise RuntimeError(f"Worker refused or failed: {destination}: {observed}")
        return observed

    for name in args.fixtures:
        case = work / name
        case.mkdir()
        check = admission(
            psutil.virtual_memory().available, reserve_bytes(name, "fixture"), 512 * 1024 * 1024
        )
        row: dict[str, Any] = {"fixture_name": name, "fixture_admission": check}
        report["cases"].append(row)
        write_json(work / "report.json", report)
        if check["status"] != "ok":
            raise RuntimeError(f"Fixture RAM admission refused before allocation: {check}")
        from transformers import AutoTokenizer

        from aether import CompilerConfig
        from aether.compiler.numerical_admission import run_worker as reference_worker
        from aether.compiler.numerical_admission import source_binding
        from aether.evaluation.quality_validator import QualityValidator, _log_probs

        fixture = create_fixture(case / "source", name)
        row["fixture"] = fixture
        row["source"] = source_binding(Path(fixture["model"]))
        probes = [
            {"id": key, "prompt_ids": ids, "continuation_ids": CONTINUATION}
            for key, ids in PROMPTS.items()
        ]
        dataset = case / "probes.jsonl"
        dataset.write_text("".join(json.dumps(probe) + "\n" for probe in probes), encoding="utf-8")
        reference_dir = case / "reference"
        reference_dir.mkdir()
        try:
            reference = reference_worker(
                reference_dir,
                "hf",
                {"source": fixture["model"], "probes": probes, "greedy_tokens": 8},
                CompilerConfig(),
                reserve_bytes(name, "hf"),
            )
        except Exception as exc:
            row["reference_error"] = str(exc)
            write_json(work / "report.json", report)
            raise
        row["reference"] = reference
        tokenizer_factory: Callable[..., Any] = AutoTokenizer.from_pretrained
        tokenizer = tokenizer_factory(
            fixture["model"], local_files_only=True, trust_remote_code=False
        )
        expected = {
            "text": tokenizer.decode(reference["tokens"]["A"], skip_special_tokens=True),
            "tokens": 8,
        }
        jobs: dict[str, Any] = {}
        role_jobs: dict[str, dict[str, Any]] = {}
        for role, code_root in (("current", ROOT / "src"), ("baseline", args.baseline_src)):
            if code_root is None:
                continue
            common = {
                "fixture_name": name,
                "fixture": fixture,
                "probes": probes,
                "expected_src": str(code_root.resolve()),
                "artifact": str(case / (role + ".aeg")),
                "diagnostic": role == "baseline",
                "dataset": str(dataset),
                "passes": [],
                "public_prompt": prompt_text(PROMPTS["A"]),
                "public_expected": expected,
            }
            row[role] = {
                "compile": launch(
                    {**common, "action": "compile", "operation": "compile"},
                    case / (role + "-compile"),
                ),
                "loads": [],
            }
            role_jobs[role] = common
            write_json(work / "report.json", report)
        for sample in range(args.load_repeats + 2):
            roles = list(role_jobs)
            if sample % 2:
                roles.reverse()
            for role in roles:
                common = role_jobs[role]
                destination = case / f"{role}-load-{sample}"
                observed = launch(
                    {**common, "action": "numerics", "operation": "load", "exercise": sample == 2},
                    destination,
                )
                row[role]["loads"].append(observed)
                if sample == 2:
                    jobs[role] = (destination, observed)
                write_json(work / "report.json", report)
        for backend in ("torch_cpu", "onnx_cpu"):
            destination = case / backend
            observed = launch(
                {
                    **role_jobs["current"],
                    "action": "numerics",
                    "operation": "backend",
                    "executor": backend,
                },
                destination,
            )
            row[backend] = observed
            jobs[backend] = (destination, observed)
            write_json(work / "report.json", report)
        row["comparisons"] = {}
        for executor, (destination, observed) in jobs.items():
            checks = {}
            for index, probe in enumerate(probes):
                original = np.load(reference_dir / f"reference-{index}.npy", allow_pickle=False)
                actual = np.load(destination / f"actual-{index}.npy", allow_pickle=False)
                labels = np.asarray(probe["continuation_ids"], np.int64)
                validator = QualityValidator()
                checks[probe["id"]] = {
                    "logits": asdict(validator.validate_logits(original, actual)),
                    "tokens": asdict(
                        validator.validate_greedy_agreement(
                            reference["tokens"][probe["id"]],
                            observed["observations"][probe["id"]]["tokens"],
                        )
                    ),
                    "perplexity": asdict(
                        validator.validate_perplexity(
                            -_log_probs(original.astype(np.float64))[
                                np.arange(len(labels)), labels
                            ],
                            -_log_probs(actual.astype(np.float64))[np.arange(len(labels)), labels],
                        )
                    ),
                }
            row["comparisons"][executor] = checks
            write_json(work / "report.json", report)
            if not all(metrics[kind]["passed"] for metrics in checks.values() for kind in metrics):
                raise RuntimeError(f"Original-HF parity failed: {name}/{executor}")
        command = [
            sys.executable,
            "-m",
            "aether",
            "run",
            str(case / "current.aeg"),
            "--prompt",
            prompt_text(PROMPTS["A"]),
            "--max-tokens",
            "8",
            "--temperature",
            "0",
            "--raw",
            "--stream",
            "--non-interactive",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (case / "cli.stdout").write_text(completed.stdout, encoding="utf-8")
        (case / "cli.stderr").write_text(completed.stderr, encoding="utf-8")
        row["cli_exit"] = completed.returncode
        if (
            completed.returncode
            or completed.stdout.strip() != ("Response: " + expected["text"]).strip()
        ):
            raise RuntimeError("Ordinary public CLI differs from original HF")
        write_json(work / "report.json", report)
        print(f"{name}: HF / CPU / Torch / ONNX / API / CLI passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
