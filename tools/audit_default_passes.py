"""Measure compiler defaults versus explicit removal of unverified default passes.

Workers, admission, native execution, numerical/timing checks and raw artifact
diffs use pass_audit.py. This never changes production defaults or algorithms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .pass_audit import (
        PASSES,
        ROOT,
        compare_numerics,
        diff_artifacts,
        inventory,
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
        run_worker,
        summarize,
    )
    from pass_audit_fixtures import write_json

REMOVED = (1, 2, 4, 5, 6, 7, 8)


def main(argv: list[str] | None = None) -> int:
    """Execute serial real-checkpoint A/Bs; retain blocked/failed outcomes."""
    # Match run_worker(): profiles and source hashes must describe this checkout,
    # even if a different Aether installation exists in the calling interpreter.
    sys.path.insert(0, str(ROOT / "src"))
    from aether import CompilerConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", nargs="+", choices=("tiny", "moe", "mid"), default=["tiny"])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--builds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-free-mib", type=int, default=512)
    parser.add_argument("--hf-reference", action="store_true")
    parser.add_argument(
        "--historical-baseline",
        action="store_true",
        help="Compare the seven former opt-ins forced on against actual current defaults",
    )
    args = parser.parse_args(argv)
    if min(args.builds, args.tokens, args.repeats) < 2 or args.context < 128:
        parser.error("Need builds/tokens/repeats >=2 and context >=128")
    if args.warmup < 0 or args.threads < 1 or args.min_free_mib < 0:
        parser.error("Need nonnegative warmup/headroom and positive threads")
    args.native, args.epoch, args.timeout = True, 1700000000, 600
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    default = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        skip_download=True,
        reproducible_builds=True,
        cache_dir=str(output / "config-cache"),
    ).to_dict()
    candidate = dict(default)
    candidate.update({PASSES[number - 1].flag: False for number in REMOVED})
    profiles = {"configured_default": default, "unverified_off": candidate}
    reference_name, candidate_name = "configured_default", "unverified_off"
    if args.historical_baseline:
        historical = dict(default)
        historical.update({PASSES[number - 1].flag: True for number in REMOVED})
        reference_name, candidate_name = "historical_seven_on", "configured_default"
        profiles = {reference_name: historical, candidate_name: default}
    report: dict[str, Any] = {
        "schema": "aether-default-pass-audit/1",
        "status": "running",
        "profiles": profiles,
        "removed_passes": REMOVED,
        "python": sys.version,
        "platform": platform.platform(),
        "threads": args.threads,
        "model_context": args.context,
        "compile_scope": (
            "Fresh workers and cold per-worker native caches, including integrity check"
        ),
        "runtime_scope": "Native CPU, fixed A/B prompts, fresh cache for each request",
        "quality_scope": (
            "Explicit diagnostic compilation/execution of untrained fixtures; "
            "original-FP32 HF only when its worker succeeds"
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "safetensors", "psutil", "tokenizers")
        },
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/aether").rglob("*.py"))
        },
        "fixtures": {},
        "results": [],
    }

    def save() -> None:
        write_json(output / "report.json", report)

    save()
    failed = False
    for name in args.fixtures:
        print(f"fixture {name}", flush=True)
        prepared = run_worker(
            {"action": "fixture", "fixture_name": name}, output / name / "fixture", args
        )
        report["fixtures"][name] = prepared
        if prepared["status"] != "ok":
            failed = True
            save()
            continue
        fixture = prepared["fixture"]
        # Set the same declared context in both deterministic copies and merge
        # sources. Prompt lengths remain 4/64; this is not a 32k-token benchmark.
        work = output / name / "fixture"
        for copy in ("first", "second"):
            for source in ("model", "merge_source"):
                path = work / copy / source / "config.json"
                source_config = json.loads(path.read_text(encoding="utf-8"))
                source_config["max_position_embeddings"] = args.context
                write_json(path, source_config)
        fixture["config"]["max_position_embeddings"] = args.context
        prepared["determinism"] = diff_artifacts(work / "first", work / "second")
        prepared["source_files"] = inventory(work / "first")
        if not prepared["determinism"]["byte_identical"]:
            failed = True
            save()
            continue
        hf = (
            run_worker(
                {
                    "action": "hf",
                    "fixture_name": name,
                    "fixture": fixture,
                    "tokens": args.tokens,
                    "threads": args.threads,
                },
                output / name / "hf",
                args,
            )
            if args.hf_reference
            else {"status": "skipped", "reason": "not_requested"}
        )
        prepared["hf_reference"] = hf
        if hf["status"] in {"blocked", "failed", "refused"}:
            failed = True
        rows = {}
        for profile, options in profiles.items():
            config = CompilerConfig(**options).clone()
            enabled = [p.number for p in PASSES if getattr(config, p.flag)]
            rows[profile] = {
                "fixture": name,
                "profile": profile,
                "enabled_passes": enabled,
                "builds": [],
                "runtime": {},
            }
            report["results"].append(rows[profile])
        # Alternate profiles between fresh cold builds to expose order/spread.
        for index in range(args.builds):
            for profile in list(profiles) if index % 2 == 0 else list(reversed(profiles)):
                row = rows[profile]
                print(f"  {name}/{profile}/build_{index}", flush=True)
                built = run_worker(
                    {
                        "action": "compile",
                        "fixture_name": name,
                        "fixture": fixture,
                        "passes": row["enabled_passes"],
                        "compiler_config": profiles[profile],
                    },
                    output / name / profile / f"build_{index}",
                    args,
                )
                row["builds"].append(built)
                if built["status"] == "ok":
                    built["artifact_files"] = inventory(Path(built["artifact"]))
                    built["artifact_bytes"] = sum(
                        item["bytes"] for item in built["artifact_files"].values()
                    )
                else:
                    failed = True
                save()
        for profile, row in rows.items():
            directory = output / name / profile
            if any(build["status"] != "ok" for build in row["builds"]):
                row["status"] = "incomplete_compile"
                save()
                continue
            first = row["builds"][0]
            artifact = Path(first["artifact"])
            row["compile_and_verify_seconds"] = summarize(
                [build["compile_and_verify_seconds"] for build in row["builds"]]
            )
            row["repeat_diffs"] = [
                diff_artifacts(artifact, Path(build["artifact"])) for build in row["builds"][1:]
            ]
            if any(not diff["byte_identical"] for diff in row["repeat_diffs"]):
                failed = True
            base = {
                "fixture_name": name,
                "fixture": fixture,
                "passes": row["enabled_passes"],
                "artifact": str(artifact),
                "tokens": args.tokens,
            }
            for action in ("load", "numerics", "public"):
                row["runtime"][action] = run_worker(
                    {**base, "action": action}, directory / action, args
                )
            numerical = row["runtime"]["numerics"]
            if numerical["status"] == "ok":
                row["runtime"]["timing"] = run_worker(
                    {
                        **base,
                        "action": "timing",
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "expected_tokens": numerical["tokens"],
                    },
                    directory / "timing",
                    args,
                )
            row["versus_original_fp32_hf"] = compare_numerics(hf, numerical)
            row["runtime_artifact_diff"] = diff_artifacts(
                artifact, artifact, first["artifact_files"], inventory(artifact)
            )
            states = [item["status"] for item in row["runtime"].values()]
            row["status"] = "ok" if all(state == "ok" for state in states) else "failed"
            failed |= row["status"] != "ok" or not row["runtime_artifact_diff"]["byte_identical"]
            save()
        original, actual = rows[reference_name], rows[candidate_name]
        if original.get("status") == actual.get("status") == "ok":
            actual[f"versus_{reference_name}"] = compare_numerics(
                original["runtime"]["numerics"], actual["runtime"]["numerics"]
            )
            actual[f"versus_{reference_name}_artifacts"] = diff_artifacts(
                Path(original["builds"][0]["artifact"]), Path(actual["builds"][0]["artifact"])
            )
        save()
    report["status"] = "complete"
    report["successful_execution"] = not failed
    save()
    print(output / "report.json")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
