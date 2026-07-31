#!/usr/bin/env python3
"""
convert_weights.py — Convert between weight formats for Aether Runtime.

Supports:
  • SafeTensors → AEG (compile & quantize)
  • GGUF → AEG (compile & quantize)
  • AEG → dequantized SafeTensors (for debugging / fine-tuning)
  • AEG → weight precision report (no conversion, just analysis)

Usage:
    # Convert safetensors model directory to AEG
    python scripts/convert_weights.py safetensors ./llama3-8b/ --output ./llama3-8b.aeg

    # Convert GGUF file to AEG
    python scripts/convert_weights.py gguf ./llama3-8b-q4_k_m.gguf --output ./llama3.aeg

    # Export AEG weights back to safetensors (dequantized, for debugging)
    python scripts/convert_weights.py aeg-to-safetensors ./llama3.aeg --output ./recovered/

    # Print precision distribution of an AEG package
    python scripts/convert_weights.py analyze ./llama3.aeg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
DIM    = "\033[2m"


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_safetensors(args: argparse.Namespace) -> int:
    """Compile a SafeTensors model directory to AEG."""
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig

    src = Path(args.source)
    if not src.exists():
        print(f"{RED}Error: {src} does not exist{RESET}", file=sys.stderr)
        return 1

    print(f"{BOLD}SafeTensors → AEG{RESET}")
    print(f"  Source    : {src}")
    print(f"  Output    : {args.output}")
    print(f"  Precision : {args.precision}")

    config = CompilerConfig(
        default_precision=args.precision,
        optimization_level=args.opt_level,
        targets=args.targets or ["cpu_avx512"],
        overwrite=args.overwrite,
    )
    compiler = Compiler(config=config)
    package = compiler.compile(str(src), output_path=args.output, targets=config.targets)

    store = package.weight_store()
    print(f"\n  {GREEN}✓{RESET} Done — {len(store)} tensors, package at {package.root}")
    return 0


def cmd_gguf(args: argparse.Namespace) -> int:
    """Compile a GGUF file to AEG."""
    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig

    src = Path(args.source)
    if not src.exists() or src.suffix.lower() not in (".gguf", ".ggml"):
        print(f"{RED}Error: {src} is not a valid GGUF file{RESET}", file=sys.stderr)
        return 1

    print(f"{BOLD}GGUF → AEG{RESET}")
    print(f"  Source : {src}")
    print(f"  Output : {args.output}")

    config = CompilerConfig(
        default_precision=args.precision,
        targets=args.targets or ["cpu_avx512"],
        overwrite=args.overwrite,
    )
    compiler = Compiler(config=config)
    package = compiler.compile(str(src), output_path=args.output, targets=config.targets)
    store = package.weight_store()
    print(f"\n  {GREEN}✓{RESET} Done — {len(store)} tensors at {package.root}")
    return 0


def cmd_aeg_to_safetensors(args: argparse.Namespace) -> int:
    """Export AEG quantized weights back to dequantized SafeTensors."""
    try:
        import safetensors.torch as st
        import torch
    except ImportError:
        print(f"{RED}Error: safetensors and torch are required for this command{RESET}")
        print(f"  pip install safetensors torch")
        return 1

    import numpy as np
    from aether.core.aeg_format import AEGPackage

    aeg_path = Path(args.source)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{BOLD}AEG → SafeTensors (dequantized){RESET}")
    print(f"  Source : {aeg_path}")
    print(f"  Output : {out_path}")

    pkg = AEGPackage(aeg_path)
    pkg.load()
    store = pkg.weight_store()

    if not store.exists:
        print(f"{RED}Error: No weight blob in package{RESET}", file=sys.stderr)
        return 1

    print(f"\n  Loading and dequantizing {len(store)} tensors...")
    flat = store.dequantize_all()

    tensors = {name: torch.from_numpy(arr) for name, arr in flat.items()}
    out_file = out_path / "model.safetensors"
    st.save_file(tensors, str(out_file))

    size_mb = out_file.stat().st_size / (1024 ** 2)
    print(f"  {GREEN}✓{RESET} Saved {len(tensors)} tensors ({size_mb:.1f} MB) to {out_file}")

    # Copy config.json if present.
    manifest = pkg.manifest
    if manifest and manifest.architecture:
        import json
        arch = manifest.architecture
        config_dict = {
            "model_type": arch.family,
            "hidden_size": arch.hidden_size,
            "num_hidden_layers": arch.layers,
            "num_attention_heads": arch.num_attention_heads,
            "num_key_value_heads": arch.num_kv_heads,
            "intermediate_size": arch.intermediate_size,
            "vocab_size": arch.vocab_size,
            "rms_norm_eps": arch.norm_eps,
            "rope_theta": arch.rope_theta,
        }
        (out_path / "config.json").write_text(json.dumps(config_dict, indent=2))
        print(f"  {GREEN}✓{RESET} config.json written")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze precision distribution of an AEG package."""
    from collections import Counter
    from aether.core.aeg_format import AEGPackage

    aeg_path = Path(args.source)
    pkg = AEGPackage(aeg_path)
    pkg.load()
    store = pkg.weight_store()

    if not store.exists:
        print(f"{YELLOW}No weight blob in package{RESET}")
        return 0

    store.load_index()
    counter: Counter = Counter()
    total_bytes = 0
    bytes_by_precision: dict[str, int] = {}

    for entry in store.entries.values():
        counter[entry.precision] += 1
        b = entry.codes_bytes + entry.scales_bytes + entry.zero_points_bytes
        total_bytes += b
        bytes_by_precision[entry.precision] = bytes_by_precision.get(entry.precision, 0) + b

    print(f"\n{BOLD}Precision Analysis — {aeg_path.name}{RESET}")
    print(f"  Total tensors : {len(store.entries)}")
    print(f"  Total bytes   : {total_bytes / (1024**2):.1f} MB\n")

    print(f"  {'Precision':<14} {'Tensors':>8} {'Size (MB)':>10} {'Share':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*8}")
    for prec, count in counter.most_common():
        size_mb = bytes_by_precision[prec] / (1024 ** 2)
        share   = bytes_by_precision[prec] / max(total_bytes, 1) * 100
        print(f"  {prec:<14} {count:>8} {size_mb:>10.1f} {share:>7.1f}%")

    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert between AI model weight formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # safetensors
    p_st = sub.add_parser("safetensors", help="SafeTensors directory → AEG")
    p_st.add_argument("source", help="Source directory with .safetensors files")
    p_st.add_argument("--output", "-o", required=True)
    p_st.add_argument("--precision", default="Q4_K_M")
    p_st.add_argument("--opt-level", type=int, default=3)
    p_st.add_argument("--targets", nargs="+", default=[])
    p_st.add_argument("--overwrite", action="store_true")

    # gguf
    p_gguf = sub.add_parser("gguf", help="GGUF file → AEG")
    p_gguf.add_argument("source", help="Source .gguf file")
    p_gguf.add_argument("--output", "-o", required=True)
    p_gguf.add_argument("--precision", default="Q4_K_M")
    p_gguf.add_argument("--targets", nargs="+", default=[])
    p_gguf.add_argument("--overwrite", action="store_true")

    # aeg-to-safetensors
    p_exp = sub.add_parser("aeg-to-safetensors", help="AEG → dequantized SafeTensors")
    p_exp.add_argument("source", help="Source .aeg directory")
    p_exp.add_argument("--output", "-o", required=True)

    # analyze
    p_ana = sub.add_parser("analyze", help="Print precision distribution of an AEG package")
    p_ana.add_argument("source", help="Path to .aeg directory")

    args = parser.parse_args()

    dispatch = {
        "safetensors":        cmd_safetensors,
        "gguf":               cmd_gguf,
        "aeg-to-safetensors": cmd_aeg_to_safetensors,
        "analyze":            cmd_analyze,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
