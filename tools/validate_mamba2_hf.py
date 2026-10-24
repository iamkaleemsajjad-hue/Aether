"""Strict offline Mamba-1/2 compile/save/load/generate parity against original HF.

Requires test extras, Torch and Transformers. Reference workers retain the
compiler's normal RAM guards. Optionally measure a separate historical src
checkout in explicit diagnostic mode; historical failures are reported, never
accepted as current quality. All source weights are synthetic, untrained FP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import runpy
import statistics
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _array_bytes(value: Any) -> int:
    import numpy as np

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if is_dataclass(value):
        return sum(_array_bytes(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, list):
        return sum(_array_bytes(item) for item in value)
    return 0


def _compile(job: dict[str, Any]) -> None:
    """Execute one actual checkout, persisting arrays and public API observations."""
    import numpy as np

    import aether
    from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
    from aether.core.aeg_format import load_aeg_package
    from aether.core.quality_gate import read_gate
    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.mamba2_engine import Mamba2ExecutionEngine
    from aether.runtime.mamba_engine import MambaExecutionEngine

    package_dir = Path(aether.__file__).resolve().parent
    if not package_dir.is_relative_to(Path(job["expected_src"]).resolve()):
        raise RuntimeError(f"Wrong checkout imported: {package_dir}")
    parameter_module = package_dir / "core/mamba2_parameters.py"
    parameter_hash = (
        {"core/mamba2_parameters.py": hashlib.sha256(parameter_module.read_bytes()).hexdigest()}
        if parameter_module.is_file()
        else {}
    )
    output = Path(job["output"])
    output.mkdir()
    artifact = output / "model.aeg"
    diagnostic = job["diagnostic"]
    began = time.perf_counter()
    Compiler(
        CompilerConfig(
            quality_mode="diagnostic" if diagnostic else "require_hf",
            quality_dataset=job["dataset"],
            quality_greedy_tokens=8,
            targets=["cpu_generic"],
            skip_download=True,
            calibration_tokens=4,
            precision_assignment_mode="uniform",
            precision="FP32",
            cache_dir=str(output / "cache"),
        )
    ).compile(job["source"], output_path=artifact)
    compiled_seconds = time.perf_counter() - began
    package = load_aeg_package(artifact)
    if package.manifest is None:
        raise RuntimeError("The compiled artifact has no manifest")
    gate = read_gate(package, allow_uncertified=diagnostic)
    if not diagnostic and gate is None:
        raise RuntimeError("The ordinary artifact has no measured numerical admission")
    engine = load_engine_from_path(artifact, allow_uncertified=diagnostic)
    family = job.get("family", "mamba2")
    expected_engine = MambaExecutionEngine if family == "mamba" else Mamba2ExecutionEngine
    if not isinstance(engine, expected_engine):
        raise RuntimeError(f"Expected the actual {family} executor, got {type(engine).__name__}")
    probes = [json.loads(line) for line in Path(job["dataset"]).read_text().splitlines()]
    timings = []
    for index, probe in enumerate(probes):
        prefix, labels = probe["prompt_ids"], probe["continuation_ids"]
        rows, _ = engine.forward(np.asarray(prefix + labels[:-1], np.int64))
        if not np.isfinite(rows).all():
            raise RuntimeError("The actual candidate produced nonfinite logits")
        np.save(output / f"logits-{index}.npy", rows[len(prefix) - 1 :], allow_pickle=False)
        for repeat in range(9):
            began = time.perf_counter()
            iterator = engine.generate_iter(
                np.asarray(prefix, np.int64), max_tokens=8, temperature=0
            )
            first = next(iterator)
            first_seconds = time.perf_counter() - began
            tokens = [int(first), *[int(token) for token in iterator]]
            elapsed = time.perf_counter() - began
            if repeat >= 2:
                timings.append(
                    {
                        "probe": probe["id"],
                        "first_token_seconds": first_seconds,
                        "decode_tokens_per_second": (len(tokens) - 1) / (elapsed - first_seconds),
                        "generated_ids": tokens,
                    }
                )
    _, cache = engine.forward(np.asarray(probes[0]["prompt_ids"], np.int64))
    public = {}
    for backend in ["aether_cpu"] if diagnostic else ["aether_cpu", "pytorch"]:
        runtime = Runtime(
            RuntimeConfig(
                backend_name=backend,
                execution_devices=["cpu"],
                hf_offline=True,
                allow_uncertified=diagnostic,
                speculative_decoding=False,
                enable_semantic_cache=False,
                enable_prefix_caching=False,
                model_cache_dir=str(output / f"runtime-{backend}"),
            )
        )
        try:
            response = runtime.generate(str(artifact), "hello", max_tokens=8, temperature=0)
            streamed = "".join(
                runtime.generate_stream(str(artifact), "hello", max_tokens=8, temperature=0)
            )
            if streamed != response.text:
                raise RuntimeError(f"{backend} public generation and streaming disagree")
            public[backend] = {"text": response.text, "tokens": response.usage["completion_tokens"]}
        finally:
            runtime.close()
    result = {
        "status": "executed",
        "diagnostic": diagnostic,
        "artifact": str(artifact),
        "code_root": str(package_dir),
        "compile_seconds": compiled_seconds,
        "artifact_bytes": sum(
            path.stat().st_size for path in artifact.rglob("*") if path.is_file()
        ),
        "loaded_weight_array_bytes": _array_bytes(engine.weights),
        "ssm_state_logical_bytes": _array_bytes(cache.states),
        "convolution_history_logical_bytes": _array_bytes(cache.conv_history),
        "activation_peak_bytes": None,
        "runtime_overhead_bytes": None,
        "memory_note": "Array geometry, not peak RSS; activations/overhead not measured.",
        "outer_norm": engine.weights.layers[0].norm.tolist(),
        "output_norm": (
            output_norm.tolist()
            if (output_norm := getattr(engine.weights.layers[0], "output_norm", None)) is not None
            else None
        ),
        "architecture": package.manifest.architecture.to_dict(),
        "numerical_gate": gate,
        "timings": timings,
        "public": public,
        "timing_summary": {
            probe["id"]: {
                "median_decode_tokens_per_second": statistics.median(rates),
                "min_decode_tokens_per_second": min(rates),
                "max_decode_tokens_per_second": max(rates),
                "median_first_token_seconds": statistics.median(first_times),
                "min_first_token_seconds": min(first_times),
                "max_first_token_seconds": max(first_times),
            }
            for probe in probes
            for rates in [
                [
                    item["decode_tokens_per_second"]
                    for item in timings
                    if item["probe"] == probe["id"]
                ]
            ]
            for first_times in [
                [item["first_token_seconds"] for item in timings if item["probe"] == probe["id"]]
            ]
        },
        "code_sha256": {
            **parameter_hash,
            **{
                name: hashlib.sha256((package_dir / name).read_bytes()).hexdigest()
                for name in (
                    f"runtime/{family}_engine.py",
                    "runtime/torch_state_engine.py",
                    "runtime/torch_engine.py",
                    "hybrid/state.py",
                    "core/types.py",
                    "runtime/aeg_loader.py",
                    "compiler/compiler.py",
                    "compiler/stage1_ingestion/ingestion.py",
                    "compiler/stage1_ingestion/architecture_detector.py",
                    "evaluation/admission_worker.py",
                    "backends/native_cpu_backend.py",
                )
            },
        },
    }
    _write(output / "result.json", result)


def _launch(job: dict[str, Any], path: Path) -> dict[str, Any]:
    _write(path, job)
    environment = dict(os.environ, PYTHONPATH=job["expected_src"])
    with path.with_suffix(".log").open("wb") as log:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--compile-job", str(path)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
    if completed.returncode:
        return {
            "status": "failed",
            "exit_code": completed.returncode,
            "log": str(path.with_suffix(".log")),
            "reason": path.with_suffix(".log").read_text(encoding="utf-8", errors="replace")[
                -2000:
            ],
        }
    result = json.loads((Path(job["output"]) / "result.json").read_text())
    if not isinstance(result, dict):
        raise ValueError("The execution result must be a JSON object")
    return result


def main() -> None:
    """Require measured current parity and retain actual historical losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--compile-job", type=Path)
    parser.add_argument("--family", choices=("mamba", "mamba2"), default="mamba2")
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", AETHER_TORCH_DTYPE="float32")
    if args.compile_job is not None:
        _compile(json.loads(args.compile_job.read_text()))
        return
    if args.output is None:
        parser.error("--output is required")
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    fixture_path = (
        root
        / "tests/unit"
        / (
            "mamba1_fidelity_fixtures.py"
            if args.family == "mamba"
            else "mamba2_normalization_fixtures.py"
        )
    )
    fixture = runpy.run_path(str(fixture_path))
    import psutil

    cpu_name = platform.processor()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        cpu_name = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_info.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_name,
        )
    report: dict[str, Any] = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"untrained synthetic original-HF FP32 {args.family} fidelity, CPU hardware only",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": {
            "cpu": cpu_name,
            "logical_cpus": os.cpu_count(),
            "host_ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes_at_start": psutil.virtual_memory().available,
        },
        "versions": {
            name: version(name)
            for name in ("numpy", "torch", "transformers", "psutil", "safetensors", "tokenizers")
        },
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "cases": [],
    }
    _write(work / "report.json", report)
    import numpy as np

    from aether import CompilerConfig
    from aether.compiler.numerical_admission import MIB, run_worker, source_binding
    from aether.evaluation.quality_validator import QualityValidator, _log_probs

    fixture_cases: list[tuple[str, dict[str, Any]]] = [
        ("default", {}),
        ("expanded-biased", {"expansion": 4, "with_bias": True, "rms_norm": False}),
    ]
    if args.family == "mamba":
        fixture_cases.extend(
            [
                ("zero-epsilon", {"epsilon": 0.0}),
                ("large-delta", {"large_delta": True}),
                ("negative-gate", {"with_bias": True, "large_silu": "gate"}),
                ("negative-conv", {"with_bias": True, "large_silu": "conv"}),
                ("swish-alias", {"activation": "swish"}),
                ("default-activation", {"omit_activation": True}),
                ("multi-layer", {"layers": 2, "with_bias": True, "expansion": 3}),
            ]
        )
    else:
        fixture_cases.extend([
            ("zero-epsilon", {"epsilon": 0.0}),
            ("large-delta", {"large_delta": True}),
            ("negative-gate", {"with_bias": True, "large_silu": "gate"}),
            ("negative-conv", {"large_silu": "conv"}),
            ("upper-limit", {"time_step_limit": (0.0, 0.001)}),
            ("lower-limit", {"time_step_limit": (1.0, 2.0)}),
            ("zero-limit", {"time_step_limit": (0.0, 0.0)}),
            ("open-upper-limit", {"time_step_limit": (1.0, None)}),
            ("open-lower-limit", {"time_step_limit": (None, 0.001)}),
            ("inverted-limits", {"time_step_limit": (0.2, 0.1)}),
            ("relu-convolution", {"activation": "relu"}),
            ("relu-bounded", {"activation": "relu", "time_step_limit": (0.1, 0.2)}),
        ])
    for case_name, options in fixture_cases:
        case = work / case_name
        case.mkdir()
        source = case / "source"
        fixture["write_checkpoint"](source, **options)
        dataset = case / "probes.jsonl"
        probes: list[dict[str, Any]] = [
            {"id": "public", "prompt_ids": [1], "continuation_ids": [2, 3, 4, 5]},
            {
                "id": "long",
                "prompt_ids": [1 + i % 15 for i in range(64)],
                "continuation_ids": [9, 10, 11, 12],
            },
        ]
        dataset.write_text("".join(json.dumps(probe) + "\n" for probe in probes))
        binding = source_binding(source)
        reference_dir = case / "reference"
        reference_dir.mkdir()
        reference = run_worker(
            reference_dir,
            "hf",
            {"source": str(source), "probes": probes, "greedy_tokens": 8},
            CompilerConfig(),
            512 * MIB + 4 * sum(item["bytes"] for item in binding["files"].values()),
        )
        job = {
            "family": args.family,
            "source": str(source),
            "dataset": str(dataset),
            "diagnostic": False,
            "expected_src": str(root / "src"),
            "output": str(case / "current"),
        }
        current = _launch(job, case / "current-job.json")
        baseline = None
        if args.baseline_src is not None:
            baseline = _launch(
                {
                    **job,
                    "diagnostic": True,
                    "expected_src": str(args.baseline_src.resolve()),
                    "output": str(case / "baseline"),
                },
                case / "baseline-job.json",
            )
        row: dict[str, Any] = {
            "case": case_name,
            "source": binding,
            "reference": reference,
            "current": current,
            "baseline": baseline,
            "comparisons": {},
        }
        report["cases"].append(row)
        for role, measured in (("current", current), ("baseline", baseline)):
            if measured is None or measured["status"] != "executed":
                continue
            checks = {}
            for index, probe in enumerate(probes):
                original = np.load(reference_dir / f"reference-{index}.npy", allow_pickle=False)
                actual = np.load(
                    Path(measured["artifact"]).parent / f"logits-{index}.npy", allow_pickle=False
                )
                validator = QualityValidator()
                labels = probe["continuation_ids"]
                from dataclasses import asdict

                checks[probe["id"]] = {
                    "logits": asdict(validator.validate_logits(original, actual)),
                    "perplexity": asdict(
                        validator.validate_perplexity(
                            -_log_probs(original.astype(np.float64))[
                                np.arange(len(labels)), labels
                            ],
                            -_log_probs(actual.astype(np.float64))[np.arange(len(labels)), labels],
                        )
                    ),
                    "tokens": asdict(
                        validator.validate_greedy_agreement(
                            reference["tokens"][probe["id"]],
                            next(
                                item["generated_ids"]
                                for item in measured["timings"]
                                if item["probe"] == probe["id"]
                            ),
                        )
                    ),
                }
            row["comparisons"][role] = checks
        _write(work / "report.json", report)
        if current["status"] != "executed":
            raise RuntimeError(f"Current compile/execution failed; see {current['log']}")
        if not all(
            check[metric]["passed"]
            for check in row["comparisons"]["current"].values()
            for metric in ("logits", "perplexity", "tokens")
        ):
            raise RuntimeError("Current model failed independent original-HF checks")
        from transformers import AutoTokenizer

        tokenizer_factory: Callable[..., Any] = AutoTokenizer.from_pretrained
        tokenizer = tokenizer_factory(source, local_files_only=True, trust_remote_code=False)
        expected = reference["tokens"]["public"]
        stop = next(
            (index + 1 for index, token in enumerate(expected) if token == 16), len(expected)
        )
        expected = expected[:stop]
        text = tokenizer.decode(expected, skip_special_tokens=True)
        for backend, observed in current["public"].items():
            if observed != {"text": text, "tokens": len(expected)}:
                raise RuntimeError(f"{backend} public API differs from original HF: {observed!r}")
        command = [
            sys.executable,
            "-m",
            "aether",
            "run",
            current["artifact"],
            "--prompt",
            "hello",
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
        if completed.returncode or completed.stdout.strip() != ("Response: " + text).strip():
            raise RuntimeError(f"Strict public CLI differs from HF, exit={completed.returncode}")
        row["public_expected"] = {"text": text, "tokens": expected}
        row["cli_exit"] = completed.returncode
        _write(work / "report.json", report)
        print(f"{case_name}: original HF / CPU / Torch CPU / public API / CLI passed", flush=True)


if __name__ == "__main__":
    main()
