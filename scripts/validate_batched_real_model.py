"""Validate batched inference against a real compiled AEG, end to end.

Not a fixture: this loads an actual Qwen3-0.6B ``.aeg`` through the real loader and
the real portable executor, then asserts the property the whole design rests on -
a row of a batch produces exactly what that sequence produces alone.

Run directly:  python scripts/validate_batched_real_model.py [--aeg PATH]
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

#: Greedy decode is compared exactly (argmax is deterministic); logits are
#: compared with a tolerance because a batched GEMM reduces in a different order.
LOGIT_TOLERANCE = 3e-3


def _load(aeg_path: Path):
    """Bring up the real engine and tokenizer from a compiled artifact."""
    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.backends.native_cpu_backend import PackagedTokenizer
    from aether.core.aeg_format import AEGPackage
    from aether.runtime.torch_engine import TorchAEGEngine

    package = AEGPackage(aeg_path)
    package.load()
    package.verify_integrity()
    print(f"  artifact verified: {aeg_path.name}")

    cpu_engine = load_engine_from_path(aeg_path)
    engine = TorchAEGEngine(cpu_engine, "cpu")
    tokenizer = PackagedTokenizer(aeg_path / "tokenizer" / "tokenizer.json")
    print(
        f"  engine: {type(engine).__name__}  layers={engine.num_layers} "
        f"heads={engine.num_heads}/{engine.num_kv_heads} head_dim={engine.head_dim} "
        f"dtype={engine.compute_dtype}"
    )
    return engine, tokenizer


def _encode(tokenizer, text: str) -> np.ndarray:
    return np.asarray(tokenizer(text, return_tensors="np")["input_ids"][0], dtype=np.int64)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aeg", type=Path, default=DEFAULT_AEG)
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args(argv)

    if not args.aeg.exists():
        print(f"AEG not found: {args.aeg}")
        print("Compile one first, e.g.:")
        print("  python -c \"from aether.compiler.compiler import Compiler; "
              "from aether.compiler.config import CompilerConfig; "
              "Compiler(CompilerConfig(targets=['cpu_avx2'])).compile('qwen 0.6B', "
              "output_path='out.aeg')\"")
        return 2

    print("=" * 70)
    print("Batched inference - real compiled AEG validation")
    print("=" * 70)
    engine, tokenizer = _load(args.aeg)

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "In mathematics, a prime number is",
        "The three primary colours are",
    ]
    ids = [_encode(tokenizer, text) for text in prompts]
    print(f"\n  prompt token counts: {[int(row.size) for row in ids]}")

    passed = True

    # -- Solo baselines ------------------------------------------------------
    print("\n[1] Decoding each prompt alone (baseline)")
    solo: list[list[int]] = []
    for index, row in enumerate(ids):
        start = time.perf_counter()
        tokens = engine.generate(row, max_tokens=args.max_tokens, temperature=0.0)
        solo.append(tokens)
        print(
            f"  p{index}: {time.perf_counter() - start:5.2f}s  "
            f"{tokenizer.decode(tokens, skip_special_tokens=True)!r}"
        )

    # -- B=1 regression ------------------------------------------------------
    print("\n[2] Batch=1 must equal the single-sequence path")
    batched_one = engine.generate_batch([ids[0]], max_tokens=args.max_tokens, temperature=0.0)
    passed &= _check("B=1 tokens identical", batched_one[0] == solo[0])

    logits_solo, _ = engine.forward(ids[0])
    logits_batch, cache_one = engine.forward_batch([ids[0]])
    deviation = float(
        np.abs(
            cache_one.last_logits.detach().float().cpu().numpy()[0]
            - np.asarray(logits_solo)[-1]
        ).max()
    )
    passed &= _check(
        "B=1 final logits within tolerance",
        deviation <= LOGIT_TOLERANCE,
        f"max|delta| = {deviation:.3e}",
    )

    # -- Equal-length batch (no padding) -------------------------------------
    print("\n[3] Uniform batch (equal lengths - no padding)")
    uniform = [ids[0][: min(int(row.size) for row in ids)] for row in ids]
    uniform_solo = [
        engine.generate(row, max_tokens=args.max_tokens, temperature=0.0) for row in uniform[:1]
    ]
    _, uniform_cache = engine.forward_batch(uniform)
    passed &= _check(
        "no mask materialized for an unpadded batch", uniform_cache.live is None
    )
    passed &= _check(
        "uniform row 0 matches solo",
        engine.generate_batch(uniform, max_tokens=args.max_tokens, temperature=0.0)[0]
        == uniform_solo[0],
    )

    # -- Ragged batches: B=2 and B=4 -----------------------------------------
    for width in (2, 4):
        print(f"\n[4] Ragged batch, B={width} (prompts differ in length)")
        subset, expected = ids[:width], solo[:width]
        _, cache = engine.forward_batch(subset)
        print(f"  pad counts: {cache.layout.pad_counts}  live mask: "
              f"{'materialized' if cache.live is not None else 'none'}")

        worst = 0.0
        final = cache.last_logits.detach().float().cpu().numpy()
        for index, row in enumerate(subset):
            reference, _ = engine.forward(row)
            worst = max(
                worst, float(np.abs(final[index] - np.asarray(reference)[-1]).max())
            )
        passed &= _check(
            f"B={width} prefill logits match solo runs",
            worst <= LOGIT_TOLERANCE,
            f"worst max|delta| = {worst:.3e}",
        )

        start = time.perf_counter()
        together = engine.generate_batch(
            subset, max_tokens=args.max_tokens, temperature=0.0
        )
        elapsed = time.perf_counter() - start
        for index in range(width):
            match = together[index] == expected[index]
            passed &= _check(
                f"B={width} row {index} greedy tokens identical to solo",
                match,
                "" if match else f"batched={together[index]} solo={expected[index]}",
            )
        produced = sum(len(row) for row in together)
        print(
            f"  batch wall time {elapsed:5.2f}s  "
            f"aggregate {produced / elapsed:6.2f} tok/s  "
            f"per-request {len(together[0]) / elapsed:6.2f} tok/s"
        )

    # -- Isolation -----------------------------------------------------------
    print("\n[5] Isolation - row 0 must not move when its neighbours change")
    alone_in_pair = engine.generate_batch(
        [ids[0], ids[3]], max_tokens=args.max_tokens, temperature=0.0
    )
    passed &= _check("row 0 stable across differing batch composition",
                     alone_in_pair[0] == solo[0])

    print("\n[6] Pad content must be unobservable")
    _, pad_zero = engine.forward_batch(ids[:2], pad_token_id=0)
    _, pad_other = engine.forward_batch(ids[:2], pad_token_id=1)
    shift = float(
        np.abs(
            pad_zero.last_logits.detach().float().cpu().numpy()
            - pad_other.last_logits.detach().float().cpu().numpy()
        ).max()
    )
    passed &= _check(
        "changing the pad token does not change any row's logits",
        shift <= LOGIT_TOLERANCE,
        f"max|delta| = {shift:.3e}",
    )

    # -- Vocabulary projection ------------------------------------------------
    print("\n[7] Restricting the vocabulary projection must not change decoding")
    # Generation projects only each row's final position. That is the same
    # arithmetic as projecting every position and discarding the rest, but a
    # one-row GEMV accumulates the contracted sum in a different order than an
    # S-row GEMM, so the logits differ at rounding scale. With a 151936-wide
    # vocabulary the question is whether that can flip an argmax. Decode a
    # reference sequence driven entirely by all-position projections and compare.
    subject = ids[2]
    reference: list[int] = []
    cache = None
    step_ids = subject
    for _ in range(args.max_tokens):
        step_logits, cache = engine._forward_device(
            step_ids, cache, validate_ids=True, logits="all"
        )
        token = int(engine.torch.argmax(step_logits[-1]).item())
        reference.append(token)
        step_ids = np.asarray([token], dtype=np.int64)

    production = engine.generate(subject, max_tokens=args.max_tokens, temperature=0.0)
    passed &= _check(
        "greedy tokens identical to an all-logits reference decode",
        production == reference,
        "" if production == reference else f"last={production} all={reference}",
    )

    full_logits, _ = engine._forward_device(subject, None, validate_ids=True, logits="all")
    last_logits, _ = engine._forward_device(subject, None, validate_ids=True, logits="last")
    deviation = float(
        np.abs(
            last_logits[0].detach().float().cpu().numpy()
            - full_logits[-1].detach().float().cpu().numpy()
        ).max()
    )
    passed &= _check(
        "final-position logits agree at rounding scale",
        deviation <= LOGIT_TOLERANCE,
        f"max|delta| = {deviation:.3e}",
    )

    # -- Throughput, for the record ------------------------------------------
    print("\n[8] Throughput by batch width (CPU, informational)")
    baseline = None
    for width in (1, 2, 4):
        subset = [ids[0]] * width
        engine.generate_batch(subset, max_tokens=2, temperature=0.0)  # warm
        start = time.perf_counter()
        rows = engine.generate_batch(subset, max_tokens=args.max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - start
        aggregate = sum(len(row) for row in rows) / elapsed
        baseline = baseline or aggregate
        print(
            f"  B={width}: {elapsed:5.2f}s  aggregate {aggregate:6.2f} tok/s  "
            f"per-request {len(rows[0]) / elapsed:6.2f} tok/s  "
            f"scaling {aggregate / baseline:4.2f}x"
        )

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
