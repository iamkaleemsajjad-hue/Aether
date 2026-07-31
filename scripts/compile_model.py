#!/usr/bin/env python3
"""
compile_model.py — Compile any model to an AEG artifact from the command line.

This is the primary developer CLI for Aether compilation. It wraps the
Compiler pipeline with rich progress output, dry-run support, profiling,
and detailed result reporting.

Usage examples:
    # Compile with defaults (Q4_K_M, auto targets)
    python scripts/compile_model.py Qwen/Qwen3-0.6B

    # Preview compilation plan without running it
    python scripts/compile_model.py Qwen/Qwen3-0.6B --dry-run

    # Compile to specific output path with INT8 precision
    python scripts/compile_model.py meta-llama/Llama-3.1-8B \
        --output ./artifacts/llama3-8b.aeg \
        --precision INT8

    # Compile for specific hardware targets
    python scripts/compile_model.py Qwen/Qwen3-72B \
        --targets cuda_sm90 cuda_sm89 cpu_avx512 \
        --output ./artifacts/qwen3-72b.aeg

    # Compile a local model directory
    python scripts/compile_model.py ./my-model/ --output ./my-model.aeg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()


# ── ANSI helpers ─────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"


def _header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{msg}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {DIM}·{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _error(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}", file=sys.stderr)


# ── Compilation ───────────────────────────────────────────────────────────────

def run_dry_run(model: str, targets: list[str]) -> int:
    """Print the compilation plan without running it."""
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig

    config = CompilerConfig(targets=targets)
    compiler = Compiler(config=config)

    _header("Compilation Plan")
    try:
        plan = compiler.plan(model)
        _info(f"Model             : {model}")
        _info(f"Architecture      : {plan.architecture.family} {plan.architecture.params_billion:.1f}B")
        _info(f"Layers            : {plan.architecture.layers}")
        _info(f"Hidden size       : {plan.architecture.hidden_size}")
        _info(f"Attention heads   : {plan.architecture.num_attention_heads}")
        _info(f"Est. compile time : {plan.estimated_compile_time_s:.1f}s")
        _info(f"Est. memory (BF16): {plan.estimated_memory_gb:.2f} GB")
        _info(f"Default precision : {config.default_precision}")
        _info(f"Targets           : {', '.join(targets) if targets else 'auto'}")
        _info(f"Optimizer passes  : {', '.join(plan.optimizer_passes)}")
    except Exception as exc:
        _error(f"Plan failed: {exc}")
        return 1

    _ok("Dry run complete — use without --dry-run to compile")
    return 0


def run_compile(
    model: str,
    output: str | None,
    targets: list[str],
    precision: str,
    opt_level: int,
    overwrite: bool,
    profile: bool,
) -> int:
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig

    config = CompilerConfig(
        default_precision=precision,
        optimization_level=opt_level,
        targets=targets or ["cpu_avx512"],
        overwrite=overwrite,
    )
    compiler = Compiler(config=config)

    _header(f"Compiling  {model}")
    _info(f"Precision   : {precision}")
    _info(f"Opt level   : {opt_level}")
    _info(f"Targets     : {', '.join(config.targets)}")
    if output:
        _info(f"Output      : {output}")
    print()

    t0 = time.perf_counter()
    try:
        package = compiler.compile(
            model,
            output_path=output,
            targets=config.targets,
        )
    except Exception as exc:
        _error(f"Compilation failed: {exc}")
        if profile:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - t0

    _header("Result")
    _ok(f"Package saved to: {package.root}")

    # Weight blob stats
    store = package.weight_store()
    if store.exists:
        _ok(f"Weight tensors  : {len(store)}")
        size_mb = store.total_bytes / (1024 ** 2)
        _ok(f"Weight blob     : {size_mb:.1f} MB")
    else:
        _warn("No weight blob written (graph-only package)")

    manifest = package.manifest
    if manifest:
        _ok(f"Model ID        : {manifest.model_id}")
        _ok(f"Format version  : {manifest.format_version}")

    _ok(f"Compile time    : {elapsed:.2f}s")

    if profile:
        _header("Profile")
        try:
            from aether.compiler.report import CompilationReport
            report_path = package.root / "compile_report.json"
            if report_path.exists():
                import json
                data = json.loads(report_path.read_text())
                for stage, duration in data.get("stage_durations", {}).items():
                    _info(f"  {stage:30s}  {duration:.3f}s")
        except Exception:
            pass

    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a model to an AEG artifact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Model ID (HuggingFace), local path, or .gguf file")
    parser.add_argument("--output", "-o", help="Output .aeg directory path")
    parser.add_argument("--targets", nargs="+", default=[], metavar="TARGET",
                        help="Hardware targets (e.g. cuda_sm90 cpu_avx512)")
    parser.add_argument("--precision", default="Q4_K_M",
                        help="Default quantization precision (default: Q4_K_M)")
    parser.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Optimizer pass level 0-3 (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan only, do not compile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .aeg package")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-stage timing after compilation")

    args = parser.parse_args()

    if args.dry_run:
        return run_dry_run(args.model, args.targets)

    return run_compile(
        model=args.model,
        output=args.output,
        targets=args.targets,
        precision=args.precision,
        opt_level=args.opt_level,
        overwrite=args.overwrite,
        profile=args.profile,
    )


if __name__ == "__main__":
    sys.exit(main())
