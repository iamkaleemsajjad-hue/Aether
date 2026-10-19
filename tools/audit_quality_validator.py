"""Measure quality-validator revisions on recorded real CPU/HF evidence, serially.

An executed default-policy report supplies original-FP32 HF evidence
and an integrity-checked AEG. This does not certify a compiler or trained model.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .pass_audit import ROOT, mark_phase, memory_point, run_worker, summarize
    from .pass_audit_fixtures import CONTINUATION, PROMPTS, write_json
else:
    from pass_audit import ROOT, mark_phase, memory_point, run_worker, summarize
    from pass_audit_fixtures import CONTINUATION, PROMPTS, write_json


def digest(path: Path) -> str:
    """Hash actual source/evidence bytes without normalizing fields."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one revision with real logits, tokens and append-only engine caches."""
    sys.path.insert(0, str(ROOT / "src"))
    work = Path(job["work"])
    source = Path(job["validator_source"])
    if digest(source) != job["validator_sha256"]:
        raise ValueError("Validator source changed after admission")
    spec = importlib.util.spec_from_file_location("_quality_audit_revision", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load validator revision")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    validator = module.QualityValidator()
    evidence = {}
    for name in ("reference", "candidate"):
        path = Path(job[name]["logits"])
        if digest(path) != job[name + "_sha256"]:
            raise ValueError("Recorded numerical evidence changed")
        with np.load(path, allow_pickle=False) as data:
            evidence[name] = {label: data[label].copy() for label in PROMPTS}
    nll: dict[str, list[float]] = {}
    for name, prompts in evidence.items():
        nll[name] = []
        for rows in prompts.values():
            if rows.ndim != 2 or rows.shape[0] != len(CONTINUATION) + 1:
                raise ValueError("Expected fixed-label rows plus one unlabelled final row")
            shifted = rows[: len(CONTINUATION)].astype(np.float64)
            shifted -= shifted.max(axis=-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            nll[name].extend(
                -float(row[label]) for row, label in zip(log_probs, CONTINUATION, strict=True)
            )
    kwargs = {
        "reference_logits": np.concatenate(list(evidence["reference"].values())),
        "candidate_logits": np.concatenate(list(evidence["candidate"].values())),
        "reference_tokens": [
            token for label in PROMPTS for token in job["reference"]["tokens"][label]
        ],
        "candidate_tokens": [
            token for label in PROMPTS for token in job["candidate"]["tokens"][label]
        ],
        "reference_nll": nll["reference"],
        "candidate_nll": nll["candidate"],
    }
    mark_phase(work, "metrics")
    metric_samples = []
    measured = validator.validate_all(**kwargs)
    for index in range(job["warmup"] + job["metric_repeats"]):
        started = time.perf_counter()
        measured = validator.validate_all(**kwargs)
        ended = time.perf_counter()
        if index >= job["warmup"]:
            metric_samples.append((ended - started) * 1000)

    from aether.runtime.aeg_loader import load_engine_from_path

    mark_phase(work, "load")
    engine = load_engine_from_path(job["artifact"], prepare_native_kernels=False)
    prompt = PROMPTS["A"]
    mark_phase(work, "full_prefix_oracle")
    expected: list[int] = []
    for _ in range(job["tokens"]):
        logits, _ = engine.forward_step(prompt + expected, None)
        expected.append(int(np.argmax(logits)))

    class Trace:
        def __init__(self) -> None:
            self.requests: list[list[dict[str, Any]]] = []

        def forward_step(self, ids: list[int], cache: Any, **options: Any) -> tuple[Any, Any]:
            if cache is None:
                self.requests.append([])
            logits, cache = engine.forward_step(ids, cache, **options)
            arrays = [
                array
                for group in (cache.keys, cache.values, cache.positions)
                for array in group
                if array is not None
            ]
            allocated = sum(array.nbytes for array in {id(a): a for a in arrays}.values())
            self.requests[-1].append(
                {
                    "input_count": len(ids),
                    "logical_length": cache.length,
                    "allocated_kv_bytes": allocated,
                    "greedy_id": int(np.argmax(logits)),
                }
            )
            return logits, cache

    def compare() -> tuple[Any, list[list[list[dict[str, Any]]]]]:
        reference, candidate = Trace(), Trace()
        result = validator.compare_engines(reference, candidate, prompt, max_tokens=job["tokens"])
        return result, [reference.requests, candidate.requests]

    compare_samples = []
    last = None
    traces: list[list[list[dict[str, Any]]]] = []
    for index in range(job["warmup"] + job["repeats"]):
        mark_phase(work, "compare_warmup" if index < job["warmup"] else "engine_compare")
        started = time.perf_counter()
        last, traces = compare()
        ended = time.perf_counter()
        if index >= job["warmup"]:
            compare_samples.append((ended - started) * 1000)
    if last is None:
        raise ValueError("No engine comparison was executed")
    generated = [step["greedy_id"] for step in traces[0][0]]
    if job["variant"] == "current" and generated != expected:
        raise ValueError("Corrected validation history disagrees with fresh-prefix AEG oracle")
    mark_phase(work, "profile")
    profile = cProfile.Profile()
    profile.runcall(compare)
    profile.dump_stats(str(work / "compare.pstats"))
    return {
        "status": "ok",
        "metrics": measured.to_dict(),
        "metrics_ms": summarize(metric_samples),
        "comparison_ms": summarize(compare_samples),
        "comparison": last.to_dict(),
        "cache_traces": traces,
        "oracle_ids": expected,
        "generated_ids": generated,
        "matches_correct_prefix_history": generated == expected,
        "scope": (
            "NumPy CPU AEG; recorded HF metric inputs; "
            "fixed-count validation, not serving throughput"
        ),
        "final_memory_point": memory_point("profile"),
    }


def main(argv: list[str] | None = None) -> int:
    """Compare an explicit Git baseline against current source in fresh admitted workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-revision")
    parser.add_argument("--fixture", default="tiny")
    parser.add_argument("--profile", default="configured_default")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--metric-repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if args.worker:
        job = json.loads(args.worker.read_text(encoding="utf-8"))
        work = Path(job["work"])
        mark_phase(work, "imports")
        try:
            result = measure(job)
        except Exception as exc:
            traceback.print_exc()
            result = {
                "status": "failed",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "final_memory_point": memory_point("failed"),
            }
        write_json(work / "result.json", result)
        return int(result["status"] != "ok")
    if not args.report or not args.output or not args.baseline_revision:
        parser.error("Need --report, --output and an explicit --baseline-revision")
    if (
        min(args.blocks, args.tokens, args.repeats, args.metric_repeats) < 1
        or args.warmup < 0
        or args.min_free_mib < 0
    ):
        parser.error("Counts must be positive; warmup and free-memory headroom must be nonnegative")
    args.native, args.epoch, args.threads, args.timeout = False, 1700000000, 1, 600
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_report = json.loads(args.report.read_text(encoding="utf-8"))
    row = next(
        row
        for row in source_report["results"]
        if row["fixture"] == args.fixture and row["profile"] == args.profile
    )
    reference = source_report["fixtures"][args.fixture]["hf_reference"]
    candidate = row["runtime"]["numerics"]
    if reference["status"] != "ok" or candidate["status"] != "ok":
        raise ValueError("Need actually executed original-FP32 HF and candidate numerical evidence")
    source_path = "src/aether/evaluation/quality_validator.py"
    before = subprocess.check_output(
        ["git", "show", f"{args.baseline_revision}:{source_path}"], cwd=ROOT
    )
    baseline = args.output / "baseline_validator.py"
    baseline.write_bytes(before)
    current = ROOT / source_path
    report: dict[str, Any] = {
        "source_report": str(args.report.resolve()),
        "source_report_sha256": digest(args.report),
        "baseline_revision": args.baseline_revision,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scope": "Serial diagnostic; archived real HF evidence is reused, not re-executed",
        "jobs": [],
    }
    for block in range(args.blocks):
        variants = [("baseline", baseline), ("current", current)]
        for variant, source in variants if block % 2 == 0 else list(reversed(variants)):
            job = {
                "action": "quality_validation",
                "fixture_name": args.fixture,
                "variant": variant,
                "validator_source": str(source),
                "validator_sha256": digest(source),
                "artifact": row["builds"][0]["artifact"],
                "reference": reference,
                "candidate": candidate,
                "reference_sha256": digest(Path(reference["logits"])),
                "candidate_sha256": digest(Path(candidate["logits"])),
                "tokens": args.tokens,
                "repeats": args.repeats,
                "metric_repeats": args.metric_repeats,
                "warmup": args.warmup,
            }
            print(f"{block}/{variant}", flush=True)
            outcome = run_worker(
                job, args.output / f"{block}-{variant}", args, worker_script=Path(__file__)
            )
            report["jobs"].append(
                {"block": block, "variant": variant, "job": job, "result": outcome}
            )
            write_json(args.output / "report.json", report)
    report["successful_execution"] = all(row["result"]["status"] == "ok" for row in report["jobs"])
    write_json(args.output / "report.json", report)
    return int(not report["successful_execution"])


if __name__ == "__main__":
    raise SystemExit(main())
