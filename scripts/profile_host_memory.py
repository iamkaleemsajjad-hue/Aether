"""Attribute host RAM after an AEG load, on a real compiled artifact.

The benchmark report shows Aether holding substantially more host RSS than
Transformers after loading the same model — 5.306 GiB vs 2.233 GiB on Qwen3-0.6B.
Host RAM is not a rounding error at scale: it is linear in parameter count, so
whatever the ratio is at 0.6B it is the same ratio at 70B or 500B, where it decides
whether the model loads at all.

This measures where that memory is, rather than guessing. Three checkpoints:

  1. interpreter baseline
  2. after the AEG loader materializes host-side weights
  3. after the portable tensor executor materializes device tensors

The gap between (3) and (2) is the executor's own footprint. What (2) holds *after*
(3) exists is the question that matters: a host copy that no longer serves
execution is pure residency.

Run:  python scripts/profile_host_memory.py [--aeg PATH]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for candidate in (_ROOT, _ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

DEFAULT_AEG = _ROOT / "benchmark" / "results" / "aeg-cache" / "qwen 0.6B.aeg"
GIB = 1024.0 ** 3


def _rss() -> float:
    """Resident set size in GiB, read from the OS rather than from Python."""
    import psutil

    return psutil.Process().memory_info().rss / GIB


def _weight_bytes(weights) -> tuple[int, int]:
    """Total bytes and element count of every host-resident weight array."""
    total = 0
    count = 0
    seen: set[int] = set()

    def visit(value) -> None:
        nonlocal total, count
        if value is None or not isinstance(value, np.ndarray):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        total += int(value.nbytes)
        count += int(value.size)

    for name in ("embedding", "lm_head", "final_norm", "final_norm_bias",
                 "embedding_norm", "embedding_norm_bias", "position_embedding"):
        visit(getattr(weights, name, None))
    for layer in getattr(weights, "layers", None) or []:
        for name in (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            "attention_norm", "ffn_norm", "q_norm", "k_norm",
            "post_attention_norm", "post_ffn_norm", "router",
            "q_proj_bias", "k_proj_bias", "v_proj_bias", "o_proj_bias",
            "gate_proj_bias", "up_proj_bias", "down_proj_bias",
        ):
            visit(getattr(layer, name, None))
        for expert in getattr(layer, "experts", None) or []:
            for name in ("gate_proj", "up_proj", "down_proj"):
                visit(getattr(expert, name, None))
    return total, count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aeg", type=Path, default=DEFAULT_AEG)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not args.aeg.exists():
        print(f"AEG not found: {args.aeg}")
        return 2

    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.torch_engine import TorchAEGEngine

    print("=" * 78)
    print("Host memory attribution across an AEG load")
    print("=" * 78)
    baseline = _rss()
    print(f"  [1] interpreter baseline                     {baseline:7.3f} GiB")

    package = AEGPackage(args.aeg)
    package.load()
    package.verify_integrity()

    cpu_engine = load_engine_from_path(args.aeg)
    gc.collect()
    after_loader = _rss()
    host_bytes, host_elements = _weight_bytes(cpu_engine.weights)
    print(f"  [2] after AEG loader (host weights)          {after_loader:7.3f} GiB "
          f"(+{after_loader - baseline:.3f})")
    print(f"      host weight arrays: {host_bytes / GIB:6.3f} GiB over "
          f"{host_elements / 1e6:7.1f}M elements "
          f"({host_bytes / max(host_elements, 1):.1f} bytes/element)")

    engine = TorchAEGEngine(cpu_engine, args.device)
    gc.collect()
    after_engine = _rss()
    device_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in (engine.embedding, engine.lm_head, engine.final_norm)
    )
    for layer in engine.layers:
        for value in layer.values():
            if hasattr(value, "numel"):
                device_bytes += int(value.numel()) * int(value.element_size())
            elif isinstance(value, list):
                for expert in value:
                    if isinstance(expert, dict):
                        for tensor in expert.values():
                            if hasattr(tensor, "numel"):
                                device_bytes += (
                                    int(tensor.numel()) * int(tensor.element_size())
                                )
    print(f"  [3] after tensor executor                    {after_engine:7.3f} GiB "
          f"(+{after_engine - after_loader:.3f})")
    print(f"      device/execution tensors: {device_bytes / GIB:6.3f} GiB "
          f"(dtype={engine.compute_dtype})")

    still_held = engine.weights is cpu_engine.weights
    print(f"\n  executor still references the host weight set: {still_held}")
    print(
        f"  host weights resident after conversion:       {host_bytes / GIB:6.3f} GiB"
    )
    ratio = host_bytes / max(device_bytes, 1)
    print(f"  host:device weight bytes                     {ratio:6.2f}x")
    print(
        "\nReading: rows [2] and [3] are both live at steady state. The host copy in\n"
        "[2] served the conversion in [3] and is not read again by the decode loop, so\n"
        "it is residency rather than working set. It is linear in parameter count, so\n"
        "the ratio above is what a larger model pays too."
    )
    release = getattr(engine, "release_host_weights", None)
    if callable(release):
        freed = release()
        gc.collect()
        after_release = _rss()
        print(
            f"\n  release_host_weights() reported {freed / GIB:.3f} GiB; "
            f"RSS {after_engine:.3f} -> {after_release:.3f} GiB "
            f"({after_engine - after_release:+.3f})"
        )
    else:
        print("\n  (engine exposes no release_host_weights(); nothing to reclaim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
