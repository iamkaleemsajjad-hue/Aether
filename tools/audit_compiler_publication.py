"""Compare a real compiler revision with current candidate publication, serially.

All other source modules come from the current checkout. This isolates the
compiler/publication change; it is not a whole-checkout release comparison.
Synthetic checkpoints establish numerical parity, not natural-language quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if __package__:
    from .pass_audit import (
        PASSES,
        ROOT,
        compare_numerics,
        diff_artifacts,
        inventory,
        mark_phase,
        memory_point,
        run_worker,
        summarize,
    )
    from .pass_audit_fixtures import write_json
else:
    from pass_audit import (
        PASSES,
        ROOT,
        compare_numerics,
        diff_artifacts,
        inventory,
        mark_phase,
        memory_point,
        run_worker,
        summarize,
    )
    from pass_audit_fixtures import write_json


def digest(path: Path) -> str:
    """Hash the exact source bytes loaded by a worker."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_revision(job: dict[str, Any]) -> dict[str, Any]:
    """Compile an actual checkpoint through one recorded Compiler implementation."""
    sys.path.insert(0, str(ROOT / "src"))
    from aether import CompilerConfig

    source = Path(job["compiler_source"])
    if digest(source) != job["compiler_sha256"]:
        raise ValueError("Compiler source changed after admission")
    spec = importlib.util.spec_from_file_location("_publication_audit_revision", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the requested compiler revision")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    work = Path(job["work"])
    config = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        skip_download=True,
        overwrite=True,
        reproducible_builds=True,
        cache_dir=str(work / "compiler_cache"),
    )
    mark_phase(work, "compile")
    started = time.perf_counter()
    package = module.Compiler(config).compile(
        job["fixture"]["model"], output_path=work / "model.aeg"
    )
    package.verify_integrity()
    elapsed = time.perf_counter() - started
    files = inventory(package.root)
    return {
        "status": "ok",
        "artifact": str(package.root),
        "compile_and_verify_seconds": elapsed,
        "artifact_files": files,
        "artifact_bytes": sum(row["bytes"] for row in files.values()),
        "enabled_passes": [entry.number for entry in PASSES if getattr(config, entry.flag)],
        "evaluation_status": package.metadata.get("evaluation_status"),
        "quality_scope": "Explicit diagnostic compilation; not compiler quality certification",
        "final_memory_point": memory_point("compile"),
    }


def main(argv: list[str] | None = None) -> int:
    """Retain every executed, failed or RAM-blocked result without relaxing admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-revision")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if args.worker:
        job = json.loads(args.worker.read_text(encoding="utf-8"))
        work = Path(job["work"])
        mark_phase(work, "imports")
        try:
            outcome = compile_revision(job)
        except Exception as exc:
            traceback.print_exc()
            outcome = {
                "status": "failed",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "final_memory_point": memory_point("failed"),
            }
        write_json(work / "result.json", outcome)
        return int(outcome["status"] != "ok")
    if not args.output or not args.baseline_revision:
        parser.error("Need --output and an explicit --baseline-revision")
    if min(args.blocks, args.tokens, args.repeats) < 1 or min(args.warmup, args.min_free_mib) < 0:
        parser.error("Counts must be positive; warmup/headroom must be nonnegative")
    args.native, args.epoch, args.threads, args.timeout = True, 1700000000, 1, 600
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = "src/aether/compiler/compiler.py"
    baseline = args.output / "baseline_compiler.py"
    baseline.write_bytes(
        subprocess.check_output(
            ["git", "show", f"{args.baseline_revision}:{source_path}"],
            cwd=ROOT,
        )
    )
    report: dict[str, Any] = {
        "schema": "aether-compiler-publication-audit/1",
        "baseline_revision": args.baseline_revision,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "safetensors",
                "psutil",
                "tokenizers",
            )
        },
        "scope": __doc__,
        "sources": {
            str(path): digest(path)
            for path in (
                baseline,
                ROOT / source_path,
                ROOT / "src/aether/compiler/publication.py",
            )
        },
        "builds": [],
        "runtime": {},
    }

    def save() -> None:
        write_json(args.output / "report.json", report)

    fixture_result = run_worker(
        {"action": "fixture", "fixture_name": "tiny"},
        args.output / "fixture",
        args,
    )
    report["fixture"] = fixture_result
    save()
    if fixture_result["status"] != "ok":
        return 1
    fixture = fixture_result["fixture"]
    for block in range(args.blocks):
        variants = [("baseline", baseline), ("current", ROOT / source_path)]
        for variant, source in variants if block % 2 == 0 else list(reversed(variants)):
            print(f"compile {block}/{variant}", flush=True)
            job = {
                "action": "compile",
                "fixture_name": "tiny",
                "fixture": fixture,
                "compiler_source": str(source),
                "compiler_sha256": digest(source),
            }
            outcome = run_worker(
                job,
                args.output / f"{block}-{variant}",
                args,
                worker_script=Path(__file__),
            )
            report["builds"].append({"block": block, "variant": variant, "result": outcome})
            save()
    complete = all(row["result"]["status"] == "ok" for row in report["builds"])
    if complete:
        for variant in ("baseline", "current"):
            builds = [row["result"] for row in report["builds"] if row["variant"] == variant]
            report[variant + "_compile_seconds"] = summarize(
                [row["compile_and_verify_seconds"] for row in builds]
            )
            artifact = builds[0]["artifact"]
            runtime: dict[str, Any] = {}
            report["runtime"][variant] = runtime
            base = {
                "fixture_name": "tiny",
                "fixture": fixture,
                "artifact": artifact,
                "passes": builds[0]["enabled_passes"],
                "tokens": args.tokens,
            }
            for action in ("load", "numerics", "public"):
                runtime[action] = run_worker(
                    {**base, "action": action},
                    args.output / f"{variant}-{action}",
                    args,
                )
                complete &= runtime[action]["status"] == "ok"
                save()
            if runtime["numerics"]["status"] == "ok":
                runtime["timing"] = run_worker(
                    {
                        **base,
                        "action": "timing",
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "expected_tokens": runtime["numerics"]["tokens"],
                    },
                    args.output / f"{variant}-timing",
                    args,
                )
                complete &= runtime["timing"]["status"] == "ok"
                save()
        first = Path(report["builds"][0]["result"]["artifact"])
        report["artifact_diffs"] = [
            diff_artifacts(first, Path(row["result"]["artifact"])) for row in report["builds"]
        ]
        complete &= all(row["byte_identical"] for row in report["artifact_diffs"])
        report["numerical_comparison"] = compare_numerics(
            report["runtime"]["baseline"]["numerics"],
            report["runtime"]["current"]["numerics"],
        )
        complete &= all(
            row["max_abs_logit_error"] == 0 and row["greedy_exact_match"]
            for row in report["numerical_comparison"].get("prompts", {}).values()
        ) and report["numerical_comparison"]["status"] == "ok"
    report["successful_execution"] = complete
    save()
    return int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
