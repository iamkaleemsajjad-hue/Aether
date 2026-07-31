#!/usr/bin/env python3
"""
check_env.py — Aether Runtime environment diagnostics.

Checks Python version, required dependencies, optional backends,
hardware capabilities (CUDA, Metal, ROCm), and C++ toolchain availability.
Prints a structured report and exits with code 0 (all good) or 1 (issues found).

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --json
    python scripts/check_env.py --strict   # exits 1 on any missing optional dep
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Minimum versions ─────────────────────────────────────────────────────────

PYTHON_MIN = (3, 10)

REQUIRED_PACKAGES = [
    "numpy",
    "safetensors",
    "huggingface_hub",
    "tqdm",
    "psutil",
]

OPTIONAL_PACKAGES = {
    "torch":         "PyTorch (GPU inference, CUDA/MPS backends)",
    "transformers":  "HuggingFace Transformers (model loading helpers)",
    "accelerate":    "HuggingFace Accelerate (multi-GPU helpers)",
    "gguf":          "llama.cpp GGUF loader",
    "onnxruntime":   "ONNX Runtime backend",
    "mlx":           "Apple MLX backend",
    "vllm":          "vLLM backend",
    "tritonclient":  "Triton inference server client",
    "fastapi":       "REST API server",
    "uvicorn":       "ASGI server for the REST API",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PackageStatus:
    name: str
    available: bool
    version: str | None = None
    note: str = ""


@dataclass
class HardwareStatus:
    device: str
    available: bool
    detail: str = ""


@dataclass
class ToolchainStatus:
    name: str
    available: bool
    path: str | None = None
    version: str | None = None


@dataclass
class EnvReport:
    python_ok: bool
    python_version: str
    platform: str
    required: list[PackageStatus] = field(default_factory=list)
    optional: list[PackageStatus] = field(default_factory=list)
    hardware: list[HardwareStatus] = field(default_factory=list)
    toolchains: list[ToolchainStatus] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.python_ok and all(p.available for p in self.required) and not self.issues


# ── Checks ───────────────────────────────────────────────────────────────────

def _check_package(name: str) -> PackageStatus:
    try:
        import importlib.metadata
        version = importlib.metadata.version(name)
        return PackageStatus(name=name, available=True, version=version)
    except Exception:
        pass
    try:
        __import__(name)
        return PackageStatus(name=name, available=True)
    except ImportError:
        return PackageStatus(name=name, available=False)


def _check_cuda() -> HardwareStatus:
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            return HardwareStatus("CUDA", True, f"{count} device(s): {', '.join(names)}")
        return HardwareStatus("CUDA", False, "torch.cuda.is_available() = False")
    except ImportError:
        return HardwareStatus("CUDA", False, "PyTorch not installed")


def _check_mps() -> HardwareStatus:
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return HardwareStatus("Apple MPS", True, "Metal Performance Shaders available")
        return HardwareStatus("Apple MPS", False, "Not available on this system")
    except ImportError:
        return HardwareStatus("Apple MPS", False, "PyTorch not installed")


def _check_rocm() -> HardwareStatus:
    try:
        import torch
        if torch.cuda.is_available() and "rocm" in torch.__version__.lower():
            return HardwareStatus("ROCm", True, f"PyTorch {torch.__version__}")
        return HardwareStatus("ROCm", False, "Not detected")
    except ImportError:
        return HardwareStatus("ROCm", False, "PyTorch not installed")


def _check_toolchain(name: str) -> ToolchainStatus:
    import shutil
    exe = shutil.which(name)
    if exe is None:
        return ToolchainStatus(name=name, available=False)
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        version_line = (result.stdout or result.stderr).splitlines()[0].strip()
        return ToolchainStatus(name=name, available=True, path=exe, version=version_line)
    except Exception:
        return ToolchainStatus(name=name, available=True, path=exe)


def _aether_toolchain_check() -> ToolchainStatus:
    """Test whether Aether's native kernel compilation actually works."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from aether.kernels.native_cpu import NativeCPUKernels
        k = NativeCPUKernels()
        if k.ensure_compiled():
            return ToolchainStatus(
                "aether-native-kernels", True,
                path=str(k.library_path),
                version=f"toolchain={k.toolchain.name if k.toolchain else 'none'}",
            )
        return ToolchainStatus(
            "aether-native-kernels", False,
            version=f"error: {k.build_error}",
        )
    except Exception as exc:
        return ToolchainStatus("aether-native-kernels", False, version=str(exc))


# ── Rendering ─────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _tick(ok: bool, warn: bool = False) -> str:
    if ok:
        return f"{GREEN}✓{RESET}"
    if warn:
        return f"{YELLOW}~{RESET}"
    return f"{RED}✗{RESET}"


def _print_report(report: EnvReport, strict: bool) -> None:
    print(f"\n{BOLD}Aether Runtime — Environment Check{RESET}")
    print("=" * 50)
    print(f"Python  {report.python_version}  {'✓' if report.python_ok else '✗'}")
    print(f"OS      {report.platform}")
    print()

    print(f"{BOLD}Required packages{RESET}")
    for p in report.required:
        ver = f"  ({p.version})" if p.version else ""
        print(f"  {_tick(p.available)} {p.name}{ver}")

    print(f"\n{BOLD}Optional packages{RESET}")
    for p in report.optional:
        ver = f"  ({p.version})" if p.version else ""
        note = f"  — {OPTIONAL_PACKAGES.get(p.name, '')}" if not p.available else ""
        print(f"  {_tick(p.available, warn=True)} {p.name}{ver}{note}")

    print(f"\n{BOLD}Hardware{RESET}")
    for h in report.hardware:
        print(f"  {_tick(h.available, warn=True)} {h.device}  {h.detail}")

    print(f"\n{BOLD}Toolchains{RESET}")
    for t in report.toolchains:
        path = f"  {t.path}" if t.path else ""
        ver  = f"  [{t.version}]" if t.version else ""
        print(f"  {_tick(t.available, warn=True)} {t.name}{path}{ver}")

    print()
    if report.ok:
        print(f"{GREEN}{BOLD}All checks passed — Aether is ready.{RESET}")
    else:
        print(f"{RED}{BOLD}Issues found:{RESET}")
        for issue in report.issues:
            print(f"  • {issue}")

    if strict and report.optional:
        missing_opt = [p.name for p in report.optional if not p.available]
        if missing_opt:
            print(f"\n{YELLOW}Missing optional packages (--strict): {', '.join(missing_opt)}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_report() -> EnvReport:
    py_ver = sys.version_info
    py_ok  = py_ver >= PYTHON_MIN
    report = EnvReport(
        python_ok=py_ok,
        python_version=f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        platform=f"{platform.system()} {platform.machine()} {platform.release()}",
    )
    if not py_ok:
        report.issues.append(
            f"Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ required, got {report.python_version}"
        )

    for name in REQUIRED_PACKAGES:
        status = _check_package(name)
        report.required.append(status)
        if not status.available:
            report.issues.append(f"Required package '{name}' is not installed")

    for name in OPTIONAL_PACKAGES:
        report.optional.append(_check_package(name))

    report.hardware = [_check_cuda(), _check_mps(), _check_rocm()]
    report.toolchains = [
        _check_toolchain("g++"),
        _check_toolchain("clang++"),
        _check_toolchain("cl"),
        _aether_toolchain_check(),
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Aether Runtime environment check")
    parser.add_argument("--json",   action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Fail on missing optional deps")
    args = parser.parse_args()

    report = build_report()

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        _print_report(report, strict=args.strict)

    if not report.ok:
        return 1
    if args.strict and any(not p.available for p in report.optional):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
