"""Profile where prefill time actually goes, on a real compiled AEG.

Written to answer one question with evidence rather than intuition: the benchmark
report shows Aether's prefill *winning* at 32 tokens and *losing* by 1.4-1.9x at
256 and 1024 tokens, across all three models. A constant overhead cannot produce a
crossover like that, so something in the prefill path must scale worse than the
reference does.

The instrument is deliberately crude and direct: run the real forward pass, then
run it again with one stage removed, and attribute the difference. No sampling
profiler, because the thing under suspicion is a single large GEMM whose cost a
sampler would report without explaining.

Run:  python scripts/profile_prefill.py [--aeg PATH] [--lengths 32,256,1024]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for candidate in (_ROOT, _ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

DEFAULT_AEG = _ROOT / "benchmark" / "results" / "aeg-cache" / "qwen 0.6B.aeg"


def _load(aeg_path: Path):
    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.torch_engine import TorchAEGEngine

    package = AEGPackage(aeg_path)
    package.load()
    package.verify_integrity()
    return TorchAEGEngine(load_engine_from_path(aeg_path), "cpu")


def _sync(engine) -> None:
    if engine.device.type == "cuda":
        engine.torch.cuda.synchronize()


def _time(fn, repeats: int, engine) -> float:
    """Median wall time of ``fn``, warmed once."""
    fn()
    _sync(engine)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(engine)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aeg", type=Path, default=DEFAULT_AEG)
    parser.add_argument("--lengths", default="32,128,256,512,1024")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.aeg.exists():
        print(f"AEG not found: {args.aeg}")
        return 2
    lengths = [int(value) for value in args.lengths.replace(",", " ").split()]

    engine = _load(args.aeg)
    torch = engine.torch
    vocab, hidden = int(engine.lm_head.shape[0]), int(engine.lm_head.shape[1])
    print("=" * 78)
    print("Prefill stage attribution")
    print("=" * 78)
    print(
        f"  device={engine.device} dtype={engine.compute_dtype} layers={engine.num_layers} "
        f"hidden={hidden} vocab={vocab} heads={engine.num_heads}/{engine.num_kv_heads}"
    )
    print(f"  batch={args.batch}  repeats={args.repeats}\n")

    # Analytic FLOP model, printed alongside the measurement so the two can be
    # checked against each other rather than either being taken on faith.
    body_params = sum(
        int(np.asarray(getattr(layer, name)).size)
        for layer in engine.weights.layers
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        if getattr(layer, name, None) is not None
    )
    head_params = vocab * hidden
    print(f"  transformer-body matmul params: {body_params/1e6:8.1f}M")
    print(f"  lm_head params:                 {head_params/1e6:8.1f}M "
          f"({head_params/(body_params+head_params)*100:.1f}% of matmul FLOPs/token)\n")

    header = (
        f"{'S':>6} {'logits=all s':>13} {'logits=last s':>14} {'saved s':>9} "
        f"{'saved %':>8} {'all logits MiB':>15} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for length in lengths:
        ids = np.tile(
            np.arange(length, dtype=np.int64) % min(vocab, 1000), (args.batch, 1)
        )
        batched = args.batch > 1

        def full() -> None:
            engine._forward_device(ids, None, batched=batched)

        total = _time(full, args.repeats, engine)

        # The same pass, projecting only the final position.  Everything else --
        # embedding, every layer, attention, FFN -- is identical work, so the
        # difference is the discarded part of the vocabulary projection plus the
        # cost of materializing and writing the logits tensor that held it.
        def last_only() -> None:
            engine._forward_device(ids, None, batched=batched, logits="last")

        body_time = _time(last_only, args.repeats, engine)
        head_time = total - body_time
        logits_bytes = args.batch * length * vocab * torch.finfo(engine.compute_dtype).bits // 8
        print(
            f"{length:>6} {total:>13.4f} {body_time:>14.4f} {head_time:>9.4f} "
            f"{head_time / total * 100:>7.1f}% {logits_bytes / 1024**2:>14.1f} "
            f"{total / body_time:>7.2f}x"
        )

    print(
        "\nReading: `lm_head %` is the share of prefill spent projecting *every*\n"
        "position to the vocabulary. Generation reads only the last row, so all but\n"
        "one position of that column is discarded work, and `logits MiB` is the\n"
        "tensor allocated to hold it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
