"""Prepare, run and report Aether versus ONNX Runtime with Qwen3-8B on Mac."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.macbook.engine import (
    FILES,
    GIB,
    MODEL,
    PROVIDER,
    REPOSITORY,
    SUBFOLDER,
    Engine,
    estimate_working_bytes,
)
from benchmark.macbook.aether_engine import (
    AETHER_COMPILE_AVAILABLE,
    AETHER_LOAD_ESTIMATE,
    ENGINES,
    AetherEngine,
    read_lock,
)
from benchmark.metrics import summarize
from benchmark.prompts import _CORPUS
from benchmark.suite.plan import BATCH_SIZES, OUTPUT_TOKENS, PROMPT_TOKENS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "macbook-onnx-int4"
DEFAULT_OUTPUT = ROOT / "benchmark_results_macbook"
DEFAULT_AETHER_DIR = ROOT / "models" / "macbook-aether"
SCHEMA = 2


def prepare_aether(directory: Path, revision: str) -> None:
    """Compile the fixed source model on a host with sufficient available memory."""
    import psutil
    from dataclasses import fields
    from huggingface_hub import HfApi, snapshot_download
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig

    if psutil.virtual_memory().available < AETHER_COMPILE_AVAILABLE:
        raise ValueError(
            "Aether's current full-weight compiler needs at least 56 GiB available RAM "
            "for this preparation guard. A ~16 GB Mac cannot prepare/run its Qwen3-8B CPU "
            "path. Use a larger-memory host; this command will not download 16 GB of "
            "source weights or substitute another engine."
        )
    if (directory / "model.aeg").exists():
        raise ValueError("Aether artifact already exists; choose a new --aether-dir")
    info = HfApi().model_info(MODEL, revision=revision)
    start = time.perf_counter()
    source = snapshot_download(
        MODEL, revision=info.sha, allow_patterns=["*.json", "*.safetensors", "*.jinja"]
    )
    download_s = time.perf_counter() - start
    source_config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    if source_config.get("model_type") != "qwen3":
        raise ValueError("Expected the Qwen3 source checkpoint")
    # Disable approximation/speculation passes and automatic mixed precision.
    # The checkpoint's BF16 values are retained in AEG storage; execution is FP32.
    flags = {
        field.name: False for field in fields(CompilerConfig) if field.name.startswith("enable_")
    }
    config = CompilerConfig(
        **flags,
        optimization_level=0,
        skip_download=True,
        targets=["cpu_neon" if platform.machine() == "arm64" else "cpu_avx2"],
    )
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    package = Compiler(config).compile(source, output_path=directory / "model.aeg")
    lock = {
        "schema": SCHEMA,
        "model": MODEL,
        "engine": "aether",
        "revision": info.sha,
        "artifact": "model.aeg",
        "graph_hash": package.manifest.graph_hash,
        "download_s": download_s,
        "compile_s": time.perf_counter() - start,
        "compiler_flags": flags,
        "optimization_level": 0,
        "representation": "source BF16 values in AEG storage; expanded FP32 compute",
    }
    write_json(directory / "aether.lock.json", lock)
    read_lock(directory)
    print(f"Prepared Aether Qwen3-8B: {directory}", flush=True)


def write_json(path: Path, value: Any) -> None:
    """Atomically checkpoint a record so interruption preserves completed cases."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_info() -> dict[str, Any]:
    import psutil
    import aether

    versions = {}
    for name in (
        "aether-runtime",
        "onnxruntime",
        "numpy",
        "tokenizers",
        "huggingface-hub",
        "psutil",
        "matplotlib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    host = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version,
        "logical_cores": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "available_bytes": psutil.virtual_memory().available,
        "swap_used_bytes": psutil.swap_memory().used,
        "versions": versions,
        "aether_source_file": str(Path(aether.__file__).resolve()),
        "aether_source_version": getattr(aether, "__version__", None),
    }
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model", "sysctl.proc_translated"):
            result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            host[key] = result.stdout.strip() if result.returncode == 0 else None
    for key, command in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--porcelain"]),
    ):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            host[key] = result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            host[key] = None
    return host


def prepare(directory: Path, revision: str) -> None:
    """Download only the CPU INT4 artifact at one resolved immutable Hub revision."""
    from huggingface_hub import HfApi, hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "model.lock.json").exists():
        raise ValueError("Model already prepared; use it as-is or choose a new --model-dir")
    info = HfApi().model_info(REPOSITORY, revision=revision, files_metadata=True)
    metadata = {item.rfilename: item for item in info.siblings}
    records = {}
    start = time.perf_counter()
    for name in FILES:
        remote = f"{SUBFOLDER}/{name}"
        print(f"Downloading {name} at {info.sha}", flush=True)
        downloaded = Path(
            hf_hub_download(REPOSITORY, remote, revision=info.sha, local_dir=directory / "snapshot")
        )
        size = downloaded.stat().st_size
        expected = metadata[remote]
        if expected.size is not None and size != expected.size:
            raise ValueError(f"Incomplete download: {name}")
        digest = sha256(downloaded)
        lfs = getattr(expected, "lfs", None)
        expected_hash = getattr(lfs, "sha256", None)
        if isinstance(lfs, dict):
            expected_hash = lfs.get("sha256")
        if expected_hash and digest != expected_hash:
            raise ValueError(f"Checksum mismatch: {name}")
        records[name] = {
            "path": str(downloaded.relative_to(directory)),
            "bytes": size,
            "sha256": digest,
        }
    lock = {
        "schema": SCHEMA,
        "model": MODEL,
        "repository": REPOSITORY,
        "revision": info.sha,
        "subfolder": SUBFOLDER,
        "representation": "INT4 KLD, block size 128; float32 KV cache",
        "files": records,
        "download_and_verify_s": time.perf_counter() - start,
    }
    write_json(directory / "model.lock.json", lock)
    load_artifact(directory)
    print(f"Prepared {MODEL}: {directory / 'model.lock.json'}", flush=True)


def load_artifact(directory: Path, verify_hashes: bool = False) -> tuple[dict, Path, dict]:
    lock = json.loads((directory / "model.lock.json").read_text(encoding="utf-8"))
    if (lock.get("model"), lock.get("repository"), lock.get("subfolder")) != (
        MODEL,
        REPOSITORY,
        SUBFOLDER,
    ):
        raise ValueError("The model lock must identify the documented Qwen3-8B CPU INT4 artifact")
    paths = {}
    for name in FILES:
        entry = lock["files"][name]
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Artifact path escapes --model-dir")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Model file size changed: {name}; prepare into a new directory")
        if verify_hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"Model checksum changed: {name}")
        paths[name] = path
    artifact = paths["model.onnx"].parent
    if any(path.parent != artifact for path in paths.values()):
        raise ValueError("ONNX graph, external weights and tokenizer must be in one directory")
    config = json.loads(paths["genai_config.json"].read_text(encoding="utf-8"))
    decoder = config["model"]["decoder"]
    if (config["model"]["type"], decoder["num_hidden_layers"], decoder["hidden_size"]) != (
        "qwen3",
        36,
        4096,
    ):
        raise ValueError("Artifact configuration is not Qwen3-8B")
    return lock, artifact, config


def int_list(value: str) -> list[int]:
    try:
        values = sorted({int(item) for item in value.split(",")})
        if not values or min(values) < 1:
            raise ValueError
        return values
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers") from exc


def cells_for(args: argparse.Namespace) -> list[dict[str, int]]:
    """Use the existing suite's three independent sweeps, or an explicit full grid."""
    import itertools

    batches = sorted({1, *args.batch_sizes})
    if args.smoke:
        keys = [(1, 32, 4)]
    elif args.matrix == "grid":
        keys = list(itertools.product(batches, args.prompt_tokens, args.output_tokens))
    else:
        keys = [(b, 256, 128) for b in batches]
        keys += [(1, p, 128) for p in args.prompt_tokens]
        keys += [(1, 256, o) for o in args.output_tokens]
    return [
        dict(zip(("batch_size", "prompt_tokens", "output_tokens"), key))
        for key in dict.fromkeys(keys)
    ]


def case_id(cell: dict[str, int]) -> str:
    return f"b{cell['batch_size']}_p{cell['prompt_tokens']}_o{cell['output_tokens']}"


def build_prompt(tokenizer: Any, length: int) -> list[int]:
    ids = tokenizer.encode(_CORPUS, add_special_tokens=False).ids
    if not ids:
        raise ValueError("Tokenizer produced an empty corpus")
    # Pass token IDs directly: the model receives exactly the requested count.
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def worker(job: dict[str, Any]) -> dict[str, Any]:
    import psutil
    from tokenizers import Tokenizer

    from benchmark.memory_monitor import HostMemoryMonitor

    cell = job["cell"]
    lock, artifact, _ = load_artifact(Path(job["model_dir"]))
    tokenizer = Tokenizer.from_file(str(artifact / "tokenizer.json"))
    ids = build_prompt(tokenizer, cell["prompt_tokens"])
    swap_before = psutil.swap_memory().used
    engine_key = job["engine"]
    if engine_key == "aether":
        engine = AetherEngine(Path(job["aether_dir"]), job["threads"])
        if engine.tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Aether and ONNX artifact tokenizers have different token-ID mappings")
        if (
            engine.tokenizer.encode(_CORPUS, add_special_tokens=False).ids
            != tokenizer.encode(_CORPUS, add_special_tokens=False).ids
        ):
            raise ValueError("Aether and ONNX corpus tokenization differs")
        engine_lock, _ = read_lock(Path(job["aether_dir"]))
    elif engine_key == "onnxruntime":
        engine = Engine(artifact, job["threads"])
        engine_lock = lock
    else:
        raise ValueError(f"Unknown engine: {engine_key}")

    def call() -> dict[str, Any]:
        return engine.generate(ids, cell["batch_size"], cell["output_tokens"])

    cold = call()
    for _ in range(job["warmup_iters"]):
        call()
    samples = [call() for _ in range(job["measure_iters"])]
    expected = samples[0]["token_ids"]
    for sample in [cold, *samples]:
        rows = sample["token_ids"]
        if len(rows) != cell["batch_size"] or any(
            len(row) != cell["output_tokens"] for row in rows
        ):
            raise ValueError("The requested batch or output length was not executed")
        if rows != expected or any(row != rows[0] for row in rows):
            raise ValueError("Greedy repeatability / replicated-batch consistency failed")
    # RSS / utilization sampling is in an extra generation, outside timing samples.
    with HostMemoryMonitor(0.05) as monitor:
        call()
    memory = monitor.report()
    metric_names = (
        "latency_s",
        "ttft_s",
        "decode_s",
        "tpot_ms",
        "aggregate_tokens_per_s",
        "per_request_tokens_per_s",
    )
    statistics = {
        name: summarize([sample[name] for sample in samples if sample[name] is not None])
        for name in metric_names
    }
    return {
        **cell,
        "status": "MEASURED",
        "engine": engine_key,
        "model": MODEL,
        "provider": PROVIDER if engine_key == "onnxruntime" else "Aether native CPU",
        "revision": engine_lock["revision"],
        "representation": engine_lock["representation"],
        "execution_info": getattr(engine, "execution_info", {"provider": PROVIDER}),
        "load_s": engine.load_s,
        "cold_latency_s": cold["latency_s"],
        "metrics": statistics,
        "samples": [
            {key: value for key, value in sample.items() if key != "token_ids"}
            for sample in samples
        ],
        "memory": memory,
        "swap_used_bytes_before": swap_before,
        "swap_used_bytes_after": psutil.swap_memory().used,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "sample_token_ids": expected[0],
        "repeatability": "passed",
        "sample_text": tokenizer.decode(expected[0], skip_special_tokens=False),
    }


def spawn(
    job: dict[str, Any], directory: Path, memory_limit: int, timeout: float
) -> dict[str, Any]:
    """Watch a fresh worker; a timeout or memory kill is a recorded missing result."""
    import psutil

    key = case_id(job["cell"])
    job_path, result_path = directory / f"{key}.job.json", directory / f"{key}.json"
    write_json(job_path, job)
    # Never accept a stale completed record if this invocation crashes before writing.
    if result_path.exists():
        raise ValueError(f"Refusing to overwrite existing case: {result_path}")
    start, reason, peak = time.monotonic(), None, 0
    with (directory / f"{key}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmark.macbook",
                "_worker",
                "--job",
                str(job_path),
                "--result",
                str(result_path),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                **{
                    name: str(job.get("threads", 1))
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                try:
                    rss = psutil.Process(process.pid).memory_info().rss
                    peak = max(peak, rss)
                    if rss > memory_limit:
                        reason = "MEMORY_LIMIT"
                    elif time.monotonic() - start > timeout:
                        reason = "TIMEOUT"
                    if reason:
                        process.kill()
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    if reason or process.returncode != 0 or not result_path.exists():
        result = {
            **job["cell"],
            "status": reason or "FAILED",
            "reason": f"Worker {reason or 'failed'} (exit {process.returncode}); see {key}.log",
        }
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        worker_elapsed_s=time.monotonic() - start,
        worker_peak_rss_bytes=peak,
        worker_rss_sample_interval_s=0.2,
    )
    write_json(result_path, result)
    return result


def run(args: argparse.Namespace) -> int:
    import psutil

    from benchmark.macbook.reporting import render

    if platform.system() != "Darwin":
        raise ValueError(
            "Run this benchmark on the MacBook. Offline tests and report generation work elsewhere."
        )
    output = args.output_dir.resolve()
    if (output / "run.json").exists() or any((output / "raw").rglob("*.json")):
        raise ValueError("Choose a new --output-dir; existing evidence is never overwritten")
    lock, _, config = load_artifact(args.model_dir.resolve(), verify_hashes=True)
    aether_lock = None
    if (args.aether_dir / "aether.lock.json").exists():
        aether_lock, _ = read_lock(args.aether_dir.resolve())
    host = host_info()
    if host.get("sysctl.proc_translated") == "1":
        raise ValueError(
            "Python is running under Rosetta. Use native arm64 Python on Apple Silicon"
        )
    threads = args.threads or psutil.cpu_count(logical=False) or 1
    memory_limit = int(min(args.max_memory_gib * GIB, host["ram_bytes"] * 0.85))
    plan = {
        "schema": SCHEMA,
        "model": MODEL,
        "engines": list(ENGINES),
        "providers": {"onnxruntime": PROVIDER, "aether": "Aether native CPU"},
        "model_dir": str(args.model_dir.resolve()),
        "aether_dir": str(args.aether_dir.resolve()),
        "aether_lock": aether_lock,
        "model_lock": lock,
        "threads": threads,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "matrix": args.matrix,
        "smoke": args.smoke,
        "memory_limit_bytes": memory_limit,
        "worker_timeout_s": args.worker_timeout,
        "cooldown_s": args.cooldown,
        "cells": cells_for(args),
        "host": host,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "invocation": sys.argv,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in Path(__file__).parent.glob("*.py")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", plan)
    records = []
    try:
        for index, cell in enumerate(plan["cells"]):
            # Alternate engine order at each matched workload to reduce order bias.
            order = ENGINES if index % 2 == 0 else tuple(reversed(ENGINES))
            for engine_key in order:
                estimate = (
                    AETHER_LOAD_ESTIMATE
                    if engine_key == "aether"
                    else estimate_working_bytes(
                        cell, config, lock["files"]["model.onnx.data"]["bytes"]
                    )
                )
                available_budget = max(0, psutil.virtual_memory().available - GIB)
                reason = case_unavailable(
                    engine_key,
                    cell,
                    estimate,
                    memory_limit,
                    available_budget,
                    aether_lock is not None,
                )
                if (
                    cell["prompt_tokens"] + cell["output_tokens"]
                    > config["model"]["context_length"]
                ):
                    reason = ("CONTEXT_LIMIT", "Exceeds model context")
                print(
                    f"{engine_key}/{case_id(cell)}: estimate {estimate / GIB:.1f} GiB", flush=True
                )
                if reason:
                    result = {**cell, "status": reason[0], "reason": reason[1]}
                else:
                    result = spawn(
                        {
                            **{
                                key: plan[key]
                                for key in (
                                    "model_dir",
                                    "aether_dir",
                                    "threads",
                                    "warmup_iters",
                                    "measure_iters",
                                )
                            },
                            "engine": engine_key,
                            "cell": cell,
                        },
                        output / "raw" / engine_key,
                        memory_limit,
                        args.worker_timeout,
                    )
                result.update(
                    engine=engine_key,
                    model=MODEL,
                    estimated_working_bytes=estimate,
                    available_budget_bytes=available_budget,
                    engine_order=list(order),
                )
                records.append(result)
                write_json(output / "raw" / engine_key / f"{case_id(cell)}.json", result)
                print(f"  {result['status']}: {result.get('reason', '')}", flush=True)
                if args.cooldown:
                    time.sleep(args.cooldown)
    finally:
        # Also usable after Ctrl-C: missing cells are made explicit by the reporter.
        render(output)
    measured = sum(row["status"] == "MEASURED" for row in records)
    total = len(plan["cells"]) * len(ENGINES)
    print(f"{measured}/{total} engine/cases measured. Report: {output / 'reports' / 'REPORT.html'}")
    return 0 if measured == total else (2 if measured else 1)


def case_unavailable(
    engine: str, cell: dict, estimate: int, limit: int, available: int, aether_prepared: bool
) -> tuple[str, str] | None:
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine}")
    if engine == "aether" and cell["batch_size"] != 1:
        return (
            "NOT_SUPPORTED",
            "Aether CPU forward is single-sequence; no serial batching substitute",
        )
    if estimate > min(limit, available):
        detail = (
            "Aether expands Qwen3-8B weights to FP32; this path cannot fit ~16 GB RAM. "
            if engine == "aether"
            else ""
        )
        return "SKIPPED_MEMORY", detail + "Estimate exceeds RSS cap or available RAM minus 1 GiB"
    if engine == "aether" and not aether_prepared:
        return (
            "NOT_PREPARED",
            "Run prepare-aether on a sufficient-memory host and supply --aether-dir",
        )
    return None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MacBook: Aether vs ONNX Runtime, Qwen3-8B only")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Print hardware, package versions and CPU provider availability"
    )
    prep = commands.add_parser(
        "prepare", help="Download and verify the pinned CPU INT4 artifact (~6.1 GB)"
    )
    prep.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    prep.add_argument(
        "--revision", default="main", help="Hub ref, resolved and locked to a commit SHA"
    )
    aether = commands.add_parser(
        "prepare-aether", help="Compile Qwen3-8B AEG on a high-memory host"
    )
    aether.add_argument("--aether-dir", type=Path, default=DEFAULT_AETHER_DIR)
    aether.add_argument("--revision", default="main")
    bench = commands.add_parser("run", help="Execute isolated cases and generate reports / charts")
    bench.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    bench.add_argument("--aether-dir", type=Path, default=DEFAULT_AETHER_DIR)
    bench.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    bench.add_argument("--smoke", action="store_true", help="One case: b1/p32/o4, still Qwen3-8B")
    bench.add_argument("--batch-sizes", type=int_list, default=list(BATCH_SIZES))
    bench.add_argument("--prompt-tokens", type=int_list, default=list(PROMPT_TOKENS))
    bench.add_argument("--output-tokens", type=int_list, default=list(OUTPUT_TOKENS))
    bench.add_argument("--matrix", choices=("sweeps", "grid"), default="sweeps")
    bench.add_argument("--warmup-iters", type=int, default=2)
    bench.add_argument("--measure-iters", type=int, default=5)
    bench.add_argument("--threads", type=int)
    bench.add_argument("--max-memory-gib", type=float, default=12.0)
    bench.add_argument("--worker-timeout", type=float, default=7200.0)
    bench.add_argument("--cooldown", type=float, default=5.0)
    report = commands.add_parser(
        "report", help="Regenerate reports / graphs without running inference"
    )
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    internal = commands.add_parser("_worker", help="Internal isolated worker (used by run)")
    internal.add_argument("--job", type=Path, required=True)
    internal.add_argument("--result", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            import onnxruntime as ort
            from benchmark.macbook.probe import check_ort_kernels

            host = host_info()
            host["available_providers"] = ort.get_available_providers()
            host["selected_provider"] = PROVIDER
            host["engines"] = list(ENGINES)
            host["aether_load_planning_gib"] = AETHER_LOAD_ESTIMATE / GIB
            host["aether_true_batch_sizes"] = [1]
            host["aether_fits_available_memory"] = host["available_bytes"] > AETHER_LOAD_ESTIMATE
            try:
                host["ort_kernel_checks"] = check_ort_kernels()
            except Exception as exc:
                host["ort_kernel_checks"] = {"error": str(exc)}
                print(json.dumps(host, indent=2))
                return 1
            print(json.dumps(host, indent=2))
            return 0 if PROVIDER in ort.get_available_providers() else 1
        if args.command == "prepare":
            prepare(args.model_dir.resolve(), args.revision)
        elif args.command == "prepare-aether":
            prepare_aether(args.aether_dir.resolve(), args.revision)
        elif args.command == "report":
            from benchmark.macbook.reporting import render

            render(args.output_dir.resolve())
        elif args.command == "_worker":
            job = json.loads(args.job.read_text(encoding="utf-8"))
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=job["threads"]):
                write_json(args.result, worker(job))
        else:
            if (
                args.measure_iters < 2
                or args.warmup_iters < 0
                or args.max_memory_gib <= 0
                or args.worker_timeout <= 0
                or args.cooldown < 0
                or (args.threads is not None and args.threads < 1)
            ):
                raise ValueError(
                    "Need >=2 measurements, >=0 warmups/cooldown and positive "
                    "memory/timeout/threads"
                )
            return run(args)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
