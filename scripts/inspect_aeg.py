#!/usr/bin/env python3
"""
inspect_aeg.py — Inspect and validate a compiled AEG package.

Prints a structured report of everything inside a .aeg directory:
manifest metadata, architecture, precision map, weight blob stats,
kernel set, graph node count, and integrity hashes.

Usage:
    python scripts/inspect_aeg.py ./my-model.aeg
    python scripts/inspect_aeg.py ./my-model.aeg --weights       # include per-tensor table
    python scripts/inspect_aeg.py ./my-model.aeg --graph         # print graph nodes
    python scripts/inspect_aeg.py ./my-model.aeg --verify        # verify all hashes
    python scripts/inspect_aeg.py ./my-model.aeg --json          # output as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}── {title} {'─' * max(0, 50 - len(title))}{RESET}")


def _row(label: str, value: str | int | float | None, unit: str = "") -> None:
    if value is None:
        return
    print(f"  {label:<28} {value} {unit}")


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def inspect_package(aeg_path: Path, args: argparse.Namespace) -> dict:
    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import package_is_runnable

    if not aeg_path.exists():
        print(f"{RED}Error: {aeg_path} does not exist{RESET}", file=sys.stderr)
        sys.exit(1)

    pkg = AEGPackage(aeg_path)
    pkg.load()

    report: dict = {"path": str(aeg_path.resolve())}

    # ── Package layout ────────────────────────────────────────────────────────
    _section("Package")
    _row("Path", str(aeg_path.resolve()))
    format_ver_file = aeg_path / "FORMAT_VERSION"
    if format_ver_file.exists():
        fv = format_ver_file.read_text().strip()
        _row("Format version", fv)
        report["format_version"] = fv

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = pkg.manifest
    if manifest:
        _section("Manifest")
        _row("Model ID",        manifest.model_id)
        _row("Aether version",  manifest.aether_version)
        _row("Compiled at",     manifest.compiled_at)
        _row("Graph hash",      manifest.graph_hash)
        _row("Manifest hash",   manifest.manifest_hash)

        report["manifest"] = {
            "model_id":       manifest.model_id,
            "aether_version": manifest.aether_version,
            "compiled_at":    manifest.compiled_at,
        }

        arch = manifest.architecture
        if arch:
            _section("Architecture")
            _row("Family",           arch.family)
            _row("Parameters",       f"{arch.params_billion:.2f}B")
            _row("Layers",           arch.layers)
            _row("Hidden size",      arch.hidden_size)
            _row("Attention heads",  arch.num_attention_heads)
            _row("KV heads",         arch.num_kv_heads)
            _row("Head dim",         arch.head_dim)
            _row("Intermediate",     arch.intermediate_size)
            _row("Vocab size",       arch.vocab_size)
            _row("RoPE θ",           arch.rope_theta)
            _row("Norm ε",           arch.norm_eps)
            _row("Is MoE",           str(arch.is_moe))
            if arch.is_moe:
                _row("Experts",      arch.num_experts)
                _row("Active experts", arch.num_activated_experts)
            report["architecture"] = arch.to_dict()

        opt = manifest.optimization
        if opt:
            _section("Optimization")
            _row("Fusion passes",    ", ".join(opt.fusion_passes_applied) or "none")
            _row("Fused ops",        opt.fused_ops_count)
            _row("PPL budget",       f"{opt.quality_budget_ppl_increase:.3f}")
            if opt.actual_ppl_increase is not None:
                _row("Actual PPL Δ", f"{opt.actual_ppl_increase:.4f}")

        mem = manifest.memory_requirements
        if mem:
            _section("Memory Requirements")
            _row("BF16 full",        f"{mem.bf16_gb:.2f} GB")
            _row("Compiled min",     f"{mem.compiled_min_gb:.2f} GB")
            _row("Recommended",      f"{mem.recommended_gb:.2f} GB")

    # ── Precision map ─────────────────────────────────────────────────────────
    precision_path = aeg_path / "graph" / "precision_map.json"
    if precision_path.exists():
        _section("Precision Map (sample)")
        pm = json.loads(precision_path.read_text())
        items = list(pm.items())[:10]
        for k, v in items:
            print(f"  {k:<30} {v}")
        if len(pm) > 10:
            print(f"  ... and {len(pm) - 10} more layers")
        report["precision_map_count"] = len(pm)

    # ── Weights ───────────────────────────────────────────────────────────────
    store = pkg.weight_store()
    _section("Weights")
    if store.exists:
        store.load_index()
        _row("Tensor count",     len(store))
        _row("Blob size",        _format_bytes(store.total_bytes))
        _row("Runnable",         f"{GREEN}yes{RESET}" if package_is_runnable(pkg) else f"{RED}no{RESET}")
        report["weights"] = {"tensor_count": len(store), "total_bytes": store.total_bytes}

        if args.weights:
            _section("Weight Tensors (detail)")
            print(f"  {'Name':<40} {'Precision':<12} {'Shape':<20} {'Packed'}")
            print(f"  {'-'*40} {'-'*12} {'-'*20} {'-'*6}")
            for entry in store.entries.values():
                shape_str = str(list(entry.shape))
                print(f"  {entry.name:<40} {entry.precision:<12} {shape_str:<20} {'yes' if entry.packed else 'no'}")
    else:
        _row("Weight blob", f"{YELLOW}not present (graph-only package){RESET}")
        report["weights"] = None

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph_path = aeg_path / "graph" / "aeg_graph.json"
    if graph_path.exists():
        _section("Graph")
        gdata = json.loads(graph_path.read_text())
        n_nodes = len(gdata.get("nodes", {}))
        n_edges = len(gdata.get("edges", []))
        _row("Nodes",  n_nodes)
        _row("Edges",  n_edges)
        report["graph"] = {"nodes": n_nodes, "edges": n_edges}

        if args.graph:
            _section("Graph Nodes")
            for node_id, node in list(gdata.get("nodes", {}).items())[:50]:
                op = node.get("op_type", "?")
                layer = node.get("layer_index", "")
                layer_str = f"  L{layer}" if layer != "" else ""
                print(f"  {node_id:<35} op={op:<18}{layer_str}")

    # ── Kernels ───────────────────────────────────────────────────────────────
    kernels_dir = aeg_path / "kernels"
    if kernels_dir.exists():
        kernel_files = list(kernels_dir.glob("**/*"))
        kernel_files = [f for f in kernel_files if f.is_file()]
        _section("Kernels")
        _row("Kernel files", len(kernel_files))
        for kf in kernel_files[:8]:
            print(f"  {DIM}{kf.relative_to(aeg_path)}{RESET}")

    # ── Hash verification ─────────────────────────────────────────────────────
    if args.verify:
        _section("Integrity Verification")
        try:
            from aether.core.hash_utils import verify_file_hash
            passed = 0
            failed = 0
            for artifact_path, expected_hash in (manifest.artifacts or {}).items():
                full_path = aeg_path / artifact_path
                if not full_path.exists():
                    print(f"  {RED}✗{RESET} MISSING  {artifact_path}")
                    failed += 1
                    continue
                if verify_file_hash(full_path, expected_hash):
                    print(f"  {GREEN}✓{RESET} OK       {artifact_path}")
                    passed += 1
                else:
                    print(f"  {RED}✗{RESET} MISMATCH {artifact_path}")
                    failed += 1
            if passed + failed == 0:
                print(f"  {YELLOW}No artifact hashes in manifest to verify{RESET}")
            else:
                print(f"\n  {passed} passed, {failed} failed")
        except Exception as exc:
            print(f"  {RED}Verification error: {exc}{RESET}")

    print()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a compiled AEG package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("aeg_path", help="Path to the .aeg package directory")
    parser.add_argument("--weights", action="store_true", help="Print per-tensor weight table")
    parser.add_argument("--graph",   action="store_true", help="Print graph node list")
    parser.add_argument("--verify",  action="store_true", help="Verify content hashes")
    parser.add_argument("--json",    action="store_true", help="Output as JSON (no ANSI)")
    args = parser.parse_args()

    aeg_path = Path(args.aeg_path)

    if args.json:
        # Suppress color output for JSON mode.
        global BOLD, DIM, GREEN, CYAN, YELLOW, RED, RESET
        BOLD = DIM = GREEN = CYAN = YELLOW = RED = RESET = ""

    report = inspect_package(aeg_path, args)

    if args.json:
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
