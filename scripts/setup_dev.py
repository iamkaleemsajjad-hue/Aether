#!/usr/bin/env python3
"""
setup_dev.py — Automated development environment setup for Aether Runtime.

Installs all dependencies, verifies the toolchain, runs a quick smoke test,
and sets up pre-commit hooks. Run this once after cloning the repo.

Usage:
    python scripts/setup_dev.py
    python scripts/setup_dev.py --no-hooks      # skip pre-commit hooks
    python scripts/setup_dev.py --extras vllm mlx onnxruntime
    python scripts/setup_dev.py --check-only    # verify env without installing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def _run(cmd: list[str], desc: str, check: bool = True) -> bool:
    print(f"  {CYAN}▸{RESET} {desc}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=ROOT, check=False
        )
        if result.returncode == 0 or not check:
            print(f"{GREEN}done{RESET}")
            return True
        print(f"{RED}failed{RESET}")
        if result.stderr:
            print(f"    {result.stderr.strip()[:300]}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"{RED}not found{RESET}")
        return False


def _check_python() -> bool:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"{RED}✗ Python 3.10+ required (got {major}.{minor}){RESET}")
        return False
    print(f"  {GREEN}✓{RESET} Python {major}.{minor}.{sys.version_info.micro}")
    return True


def _install_package(extras: list[str]) -> bool:
    extras_str = ",".join(["dev"] + extras) if extras else "dev"
    spec = f".[{extras_str}]"
    return _run(
        [sys.executable, "-m", "pip", "install", "-e", spec, "--quiet"],
        f"Installing aether[{extras_str}]",
    )


def _setup_hooks() -> bool:
    return _run(
        [sys.executable, "-m", "pre_commit", "install"],
        "Installing pre-commit hooks",
        check=False,
    )


def _verify_import() -> bool:
    print(f"  {CYAN}▸{RESET} Verifying aether import...", end=" ", flush=True)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import aether  # type: ignore
        print(f"{GREEN}done{RESET}  (version={getattr(aether, '__version__', 'unknown')})")
        return True
    except Exception as exc:
        print(f"{RED}failed: {exc}{RESET}")
        return False


def _smoke_test() -> bool:
    print(f"  {CYAN}▸{RESET} Running smoke test...", end=" ", flush=True)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        from aether.kernels.native_cpu import NativeCPUKernels
        from aether.quantization.formats import dequantize_tensor, quantize_tensor
        from aether.runtime.cpu_engine import CPUExecutionEngine, LayerWeights, ModelWeights

        # Build a minimal 1-layer model and run one forward pass.
        rng = np.random.default_rng(0)
        h, v, heads, kv = 64, 256, 4, 2
        hd = h // heads
        inter = 128

        lw = LayerWeights(
            attention_norm=np.ones(h, dtype=np.float32),
            q_proj=rng.standard_normal((heads * hd, h)).astype(np.float32) * 0.02,
            k_proj=rng.standard_normal((kv * hd, h)).astype(np.float32) * 0.02,
            v_proj=rng.standard_normal((kv * hd, h)).astype(np.float32) * 0.02,
            o_proj=rng.standard_normal((h, heads * hd)).astype(np.float32) * 0.02,
            ffn_norm=np.ones(h, dtype=np.float32),
            gate_proj=rng.standard_normal((inter, h)).astype(np.float32) * 0.02,
            up_proj=rng.standard_normal((inter, h)).astype(np.float32) * 0.02,
            down_proj=rng.standard_normal((h, inter)).astype(np.float32) * 0.02,
        )
        mw = ModelWeights(
            embedding=rng.standard_normal((v, h)).astype(np.float32) * 0.02,
            layers=[lw],
            final_norm=np.ones(h, dtype=np.float32),
            lm_head=rng.standard_normal((v, h)).astype(np.float32) * 0.02,
        )
        engine = CPUExecutionEngine(mw, num_heads=heads, num_kv_heads=kv)
        logits, _ = engine.forward(np.array([1, 2, 3], dtype=np.int64))
        assert logits.shape == (3, v)

        # Quick quantize round-trip.
        w = rng.standard_normal((32, 64)).astype(np.float32)
        qt = quantize_tensor(w, "Q4_K_M", block_size=32)
        out = dequantize_tensor(qt)
        assert out.shape == w.shape

        print(f"{GREEN}done{RESET}")
        return True
    except Exception as exc:
        print(f"{RED}failed: {exc}{RESET}")
        return False


def _run_unit_tests() -> bool:
    return _run(
        [sys.executable, "-m", "pytest", "tests/unit", "-q", "--tb=short", "-x"],
        "Running unit tests",
        check=False,
    )


def _print_summary(steps: dict[str, bool]) -> None:
    print(f"\n{BOLD}Setup Summary{RESET}")
    print("=" * 40)
    all_ok = True
    for step, ok in steps.items():
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} {step}")
        if not ok:
            all_ok = False
    print()
    if all_ok:
        print(f"{GREEN}{BOLD}Development environment is ready!{RESET}")
        print(f"\nNext steps:")
        print(f"  python scripts/check_env.py            # full env report")
        print(f"  python scripts/compile_model.py --help # compile a model")
        print(f"  make test                              # run full test suite")
    else:
        print(f"{RED}Some steps failed — check output above.{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up Aether Runtime dev environment")
    parser.add_argument("--no-hooks",    action="store_true", help="Skip pre-commit hooks")
    parser.add_argument("--no-tests",    action="store_true", help="Skip unit tests")
    parser.add_argument("--check-only",  action="store_true", help="Verify without installing")
    parser.add_argument("--extras", nargs="*", default=[], metavar="EXTRA",
                        help="Additional pip extras (e.g. vllm mlx onnxruntime)")
    args = parser.parse_args()

    print(f"{BOLD}{CYAN}Aether Runtime — Dev Setup{RESET}")
    print(f"Root: {ROOT}\n")

    steps: dict[str, bool] = {}

    steps["Python 3.10+"] = _check_python()

    if not args.check_only:
        steps["Install package"] = _install_package(args.extras)

    steps["Import check"]  = _verify_import()
    steps["Smoke test"]    = _smoke_test()

    if not args.check_only and not args.no_hooks:
        steps["Pre-commit hooks"] = _setup_hooks()

    if not args.no_tests:
        steps["Unit tests"] = _run_unit_tests()

    _print_summary(steps)
    return 0 if all(steps.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
