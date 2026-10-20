"""Measure the convolution regression fixtures; requires Aether's test extras.

Run separately on each source revision, with no other tests/benchmarks running:
python tools/benchmark_mamba_convolution.py --output report.json

This small CPU microbenchmark isolates a cache bug. It is not a trained-model
benchmark or original-HF admission. The companion test supplies an independent
full-sequence convolution oracle and zero B/C projections to isolate it.
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
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _history_bytes(cache: Any) -> tuple[int, int]:
    """Return logical history bytes and its retained backing allocation bytes."""
    logical, owners = 0, {}
    for history in cache.conv_history:
        if callable(getattr(history, "untyped_storage", None)):
            logical += history.numel() * history.element_size()
            storage = history.untyped_storage()
            owners[storage.data_ptr()] = storage.nbytes()
        else:
            logical += history.nbytes
            owner = history
            while getattr(owner, "base", None) is not None:
                owner = owner.base
            owners[id(owner)] = owner.nbytes
    return int(logical), int(sum(owners.values()))


def main() -> None:
    """Save raw repeated timings, cache sizes and fixed-label oracle comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--backends", nargs="+", choices=("cpu", "torch_cpu"), default=["cpu", "torch_cpu"]
    )
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 3 or args.warmup < 1:
        parser.error("tokens >= 2, repeats >= 3 and warmup >= 1 are required")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    import numpy as np
    import psutil

    from aether.evaluation.quality_validator import QualityValidator

    root = Path(__file__).resolve().parents[1]
    fixture_path = root / "tests/unit/test_mamba_convolution_history.py"
    fixture = runpy.run_path(str(fixture_path))
    process = psutil.Process()
    report: dict[str, Any] = {
        "scope": "isolated synthetic convolution; not original-HF or trained-model parity",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {name: version(name) for name in ("numpy", "psutil", "pytest")},
        "python": platform.python_version(),
        "available_memory_at_start": psutil.virtual_memory().available,
        "threads": {
            name: os.environ[name]
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "tokens": args.tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                fixture_path,
                root / "src/aether/runtime/mamba_engine.py",
                root / "src/aether/runtime/mamba2_engine.py",
                root / "src/aether/runtime/torch_state_engine.py",
            )
        },
        "artifact_bytes": None,
        "artifact_note": "No artifact is created; weight/format changes are outside this fix.",
        "cases": [],
    }
    ids = np.random.default_rng(7).integers(0, 17, args.tokens, dtype=np.int64)
    report["tokens_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
    for backend in args.backends:
        if backend == "torch_cpu":
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            report["versions"]["torch"] = torch.__version__
        for family in ("mamba", "mamba2"):
            for width in (1, 4):
                source = fixture["_engine"](family, width, True, 3)
                engine = fixture["_portable"](source, backend)
                expected = fixture["_oracle"](source, ids[:65])
                actual, _ = engine.forward(ids[:65])
                validator = QualityValidator()
                quality = asdict(validator.validate_logits(expected, actual))

                def nll(logits: Any) -> Any:
                    values = np.asarray(logits[:-1], dtype=np.float64)
                    shifted = values - values.max(axis=-1, keepdims=True)
                    return (
                        np.log(np.exp(shifted).sum(axis=-1))
                        - shifted[np.arange(len(values)), ids[1 : len(values) + 1]]
                    )

                quality["fixed_label_ppl"] = asdict(
                    validator.validate_perplexity(nll(expected), nll(actual))
                )
                samples, peak_sampled_rss = [], 0
                for repeat in range(args.warmup + args.repeats):
                    began = time.perf_counter()
                    _, cache = engine.forward(ids[:1])
                    first = time.perf_counter() - began
                    began = time.perf_counter()
                    for token in ids[1:]:
                        _, cache = engine.forward(np.asarray([token]), cache)
                    elapsed = time.perf_counter() - began
                    # Sample after each complete repeat, not a peak-RSS claim.
                    peak_sampled_rss = max(peak_sampled_rss, process.memory_info().rss)
                    if repeat >= args.warmup:
                        samples.append(
                            {
                                "first_forward_seconds": first,
                                "decode_seconds": elapsed,
                                "decode_tokens_per_second": (len(ids) - 1) / elapsed,
                            }
                        )
                logical, backing = _history_bytes(cache)
                rates = [item["decode_tokens_per_second"] for item in samples]
                case = {
                    "family": family,
                    "backend": backend,
                    "kernel_width": width,
                    "quality": quality,
                    "samples": samples,
                    "median_decode_tokens_per_second": statistics.median(rates),
                    "min_decode_tokens_per_second": min(rates),
                    "max_decode_tokens_per_second": max(rates),
                    "history_logical_bytes": logical,
                    "history_backing_bytes": backing,
                    "max_repeat_end_sampled_process_rss_bytes": peak_sampled_rss,
                }
                report["cases"].append(case)
                print(
                    f"{backend} {family} width={width}: {statistics.median(rates):.2f} tokens/s",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
