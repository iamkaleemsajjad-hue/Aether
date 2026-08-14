"""
Aether Runtime Installation Validator (PRD §43).

Runs a step-by-step verification of the Aether installation:
  1. Imports all public modules
  2. Checks core dependencies
  3. Runs hardware detection
  4. Validates CPU backend environment
  5. Runs smoke test (compile + run) if a model is available
  6. Reports PASS/FAIL per step

Usage:
    python scripts/verify_install.py
    python scripts/verify_install.py --verbose
    python scripts/verify_install.py --smoke-model ./my_model.safetensors
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


# ANSI colors (work on Windows 10+ terminals with VT processing)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg: str) -> str:
    return f"{GREEN}PASS{RESET} {msg}"


def _fail(msg: str, detail: str = "") -> str:
    s = f"{RED}FAIL{RESET} {msg}"
    if detail:
        s += f"\n     {RED}{detail}{RESET}"
    return s


def _warn(msg: str) -> str:
    return f"{YELLOW}WARN{RESET} {msg}"


def _run_check(
    name: str,
    fn: Callable[[], Any],
    *,
    warn_on_fail: bool = False,
) -> tuple[bool, str]:
    """Run a check function, return (passed, message)."""
    try:
        detail = fn()
        return True, _ok(f"{name}: {detail}")
    except Exception as exc:
        msg = f"{name}: {exc}"
        if warn_on_fail:
            return True, _warn(msg)
        return False, _fail(name, str(exc))


def check_python_version() -> str:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        raise RuntimeError(f"Python 3.9+ required; got {sys.version}")
    return sys.version.split()[0]


def check_import(pkg: str) -> str:
    mod = importlib.import_module(pkg)
    return getattr(mod, "__version__", "installed")


def check_aether_version() -> str:
    from aether.core.constants import AETHER_VERSION
    return AETHER_VERSION


def check_hardware_detection() -> str:
    from aether.backends.hardware_detector import detect_all_capabilities
    caps = detect_all_capabilities()
    available = [c.target_id for c in caps if c.available]
    unavailable_count = sum(1 for c in caps if not c.available)
    return f"{len(available)} available targets: {available[:3]}... ({unavailable_count} unavailable)"


def check_cpu_backend_validation() -> str:
    from aether.backends.hardware_detector import validate_backend_environment
    result = validate_backend_environment("cpu")
    if not result.all_passed:
        raise RuntimeError(f"CPU backend validation failed: {result.checks_failed}")
    return f"{len(result.checks_passed)} checks passed"


def check_native_kernels() -> str:
    from aether.kernels.native_cpu import get_native_kernels
    nk = get_native_kernels()
    if nk.ensure_compiled():
        return f"available={nk.available_kernels()}"
    return f"WARNING: {nk.build_error}"


def check_runtime_init() -> str:
    from aether import Runtime, RuntimeConfig
    rt = Runtime(RuntimeConfig(hf_offline=True))
    return f"Runtime initialized with hf_offline=True"


def check_compiler_init() -> str:
    from aether import Compiler, CompilerConfig
    c = Compiler(CompilerConfig())
    return "Compiler initialized"


def check_grpc_server_import() -> str:
    import aether.server.grpc_service  # noqa: F401
    return "gRPC service module importable"


def check_rest_server_import() -> str:
    import aether.server.rest  # noqa: F401
    return "REST server module importable"


def check_validation_matrix() -> str:
    candidates = [
        Path(__file__).parent.parent / "hardware_validation_matrix.json",
        Path.cwd() / "hardware_validation_matrix.json",
    ]
    for p in candidates:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            n_targets = len(data.get("targets", []))
            return f"Found at {p} ({n_targets} targets)"
    raise RuntimeError("hardware_validation_matrix.json not found — run: aether hardware detect --save")


def check_smoke_compile(model_path: str | None) -> str:
    if not model_path:
        return "SKIPPED (no --smoke-model provided)"
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    from aether import Compiler, CompilerConfig
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = CompilerConfig(cache_dir=tmpdir, hf_offline=True)
        c = Compiler(cfg)
        aeg = c.compile(model_path)
        return f"Compiled to {aeg.root}"


def check_smoke_run(model_path: str | None) -> str:
    if not model_path:
        return "SKIPPED (no --smoke-model provided)"
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    from aether import Runtime, RuntimeConfig
    rt = Runtime(RuntimeConfig(hf_offline=True))
    result = rt.generate(model_path, "Hello", max_tokens=5, temperature=0.0)
    return f"Generated {len(result.text.split())} tokens"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aether Runtime Installation Verifier (PRD §43)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--smoke-model", type=str, default=None,
                        help="Path to a real model checkpoint to compile and run as smoke test")
    parser.add_argument("--json", "as_json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    print(f"\n{BOLD}Aether Runtime Installation Verifier{RESET}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    results: list[dict[str, Any]] = []

    checks: list[tuple[str, Callable[[], Any], bool]] = [
        # (name, fn, warn_on_fail)
        ("python_version", check_python_version, False),
        ("import:aether", lambda: check_import("aether"), False),
        ("aether_version", check_aether_version, False),
        ("import:torch", lambda: check_import("torch"), True),  # warn not fail
        ("import:numpy", lambda: check_import("numpy"), False),
        ("import:safetensors", lambda: check_import("safetensors"), True),
        ("import:click", lambda: check_import("click"), False),
        ("import:rich", lambda: check_import("rich"), False),
        ("hardware_detection", check_hardware_detection, False),
        ("cpu_backend_validation", check_cpu_backend_validation, False),
        ("native_cpu_kernels", check_native_kernels, True),
        ("runtime_init", check_runtime_init, False),
        ("compiler_init", check_compiler_init, False),
        ("grpc_import", check_grpc_server_import, True),
        ("rest_import", check_rest_server_import, True),
        ("validation_matrix", check_validation_matrix, True),
        ("smoke_compile", lambda: check_smoke_compile(args.smoke_model), True),
        ("smoke_run", lambda: check_smoke_run(args.smoke_model), True),
    ]

    passed = 0
    failed = 0

    for name, fn, warn_on_fail in checks:
        ok, msg = _run_check(name, fn, warn_on_fail=warn_on_fail)
        print(msg)
        results.append({"name": name, "passed": ok, "message": msg})
        if ok:
            passed += 1
        else:
            failed += 1
            if args.verbose:
                traceback.print_exc()

    print()
    if failed:
        print(f"{RED}{BOLD}RESULT: {failed} checks failed, {passed} passed{RESET}")
    else:
        print(f"{GREEN}{BOLD}RESULT: All {passed} checks passed{RESET}")

    if args.as_json:
        print(json.dumps({
            "passed": passed,
            "failed": failed,
            "total": len(results),
            "checks": results,
        }, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
