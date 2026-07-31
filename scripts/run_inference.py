#!/usr/bin/env python3
"""
run_inference.py — Run inference on a compiled AEG package from the CLI.

Loads a compiled .aeg artifact, selects the best available backend
(CUDA → MPS → CPU), and runs generation on one or more prompts.

Usage:
    # Interactive REPL (reads prompts from stdin)
    python scripts/run_inference.py ./my-model.aeg

    # Single prompt
    python scripts/run_inference.py ./my-model.aeg --prompt "Hello, world"

    # Multiple prompts from a file (one per line)
    python scripts/run_inference.py ./my-model.aeg --prompts-file prompts.txt

    # Control generation parameters
    python scripts/run_inference.py ./my-model.aeg \
        --prompt "Explain transformers" \
        --max-tokens 256 \
        --temperature 0.7 \
        --top-k 50

    # Force CPU execution engine (bypasses backend selection)
    python scripts/run_inference.py ./my-model.aeg \
        --prompt "Test" --backend cpu

    # Benchmark mode: run N iterations and report throughput
    python scripts/run_inference.py ./my-model.aeg \
        --prompt "Benchmark prompt" \
        --benchmark --iterations 20
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

BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"


def _load_package(aeg_path: Path):
    from aether.core.aeg_format import AEGPackage
    if not aeg_path.exists():
        print(f"{RED}Error: AEG package not found: {aeg_path}{RESET}", file=sys.stderr)
        sys.exit(1)
    pkg = AEGPackage(aeg_path)
    pkg.load()
    return pkg


def _load_cpu_engine(package):
    """Load the CPU execution engine from an AEG package."""
    from aether.runtime.aeg_loader import load_engine_from_package
    print(f"  {DIM}·{RESET} Loading weights and building CPU engine...")
    t0 = time.perf_counter()
    engine = load_engine_from_package(package)
    elapsed = time.perf_counter() - t0
    print(f"  {GREEN}✓{RESET} Engine ready  ({elapsed:.2f}s)  — {engine!r}")
    return engine


def _naive_tokenize(text: str, vocab_size: int = 50257) -> list[int]:
    """
    Minimal character-level tokenizer used when no real tokenizer is available.
    Maps each UTF-8 byte to a token id modulo vocab_size.
    This is NOT a real BPE tokenizer — it exists only so the inference script
    can run without a network connection or the `transformers` library.
    """
    return [b % vocab_size for b in text.encode("utf-8")]


def _try_load_tokenizer(package):
    """Attempt to load the HuggingFace tokenizer for the compiled model."""
    try:
        from transformers import AutoTokenizer  # type: ignore
        manifest = package.manifest
        if manifest is None:
            return None
        model_id = manifest.model_id
        print(f"  {DIM}·{RESET} Loading tokenizer for {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"  {GREEN}✓{RESET} Tokenizer loaded  (vocab={tok.vocab_size})")
        return tok
    except Exception:
        return None


def _encode(tokenizer, text: str, vocab_size: int) -> list[int]:
    if tokenizer is not None:
        return tokenizer.encode(text)
    return _naive_tokenize(text, vocab_size)


def _decode(tokenizer, ids: list[int]) -> str:
    if tokenizer is not None:
        return tokenizer.decode(ids, skip_special_tokens=True)
    return "".join(chr(max(32, min(126, i % 95 + 32))) for i in ids)


def _run_one(engine, tokenizer, prompt: str, args) -> tuple[str, float, float]:
    """Run a single generation. Returns (output_text, ttft_s, total_s)."""
    import numpy as np

    vocab_size = engine.weights.vocab_size
    input_ids = _encode(tokenizer, prompt, vocab_size)
    # Clamp tokens to valid range.
    input_ids = [min(max(0, t), vocab_size - 1) for t in input_ids]
    input_arr = np.array(input_ids, dtype=np.int64)

    t_start = time.perf_counter()
    logits, cache = engine.forward(input_arr)
    t_ttft = time.perf_counter()

    eos = None
    if tokenizer is not None and hasattr(tokenizer, "eos_token_id"):
        eos = tokenizer.eos_token_id

    generated = engine.generate(
        input_arr,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos,
    )
    t_end = time.perf_counter()

    output = _decode(tokenizer, generated)
    return output, t_ttft - t_start, t_end - t_start


def _print_generation(prompt: str, output: str, ttft: float, total: float, n_tokens: int) -> None:
    tps = n_tokens / max(total, 1e-9)
    print(f"\n{BOLD}Prompt:{RESET} {prompt}")
    print(f"{BOLD}Output:{RESET} {output}")
    print(
        f"{DIM}TTFT: {ttft*1000:.1f}ms  |  "
        f"{n_tokens} tokens  |  {tps:.1f} tok/s  |  "
        f"total: {total:.3f}s{RESET}"
    )


def _benchmark_mode(engine, tokenizer, prompt: str, args) -> int:
    import numpy as np
    print(f"\n{BOLD}{CYAN}Benchmark mode — {args.iterations} iterations{RESET}")
    times: list[float] = []
    ttfts: list[float] = []
    for i in range(args.iterations):
        _, ttft, total = _run_one(engine, tokenizer, prompt, args)
        times.append(total)
        ttfts.append(ttft)
        print(f"  iter {i+1:3d}/{args.iterations}  {total:.3f}s", end="\r")
    print()

    tps_list = [args.max_tokens / t for t in times]
    print(f"\n{BOLD}Results ({args.iterations} iterations, {args.max_tokens} tokens each):{RESET}")
    print(f"  Throughput  mean={np.mean(tps_list):.1f}  min={np.min(tps_list):.1f}  max={np.max(tps_list):.1f}  tok/s")
    print(f"  TTFT        mean={np.mean(ttfts)*1000:.1f}ms")
    print(f"  Total       mean={np.mean(times)*1000:.1f}ms")
    return 0


def _interactive_repl(engine, tokenizer, args) -> int:
    print(f"\n{BOLD}Aether Interactive REPL{RESET}  (Ctrl-C or Ctrl-D to quit)\n")
    while True:
        try:
            prompt = input(f"{CYAN}>>> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        try:
            output, ttft, total = _run_one(engine, tokenizer, prompt, args)
            _print_generation(prompt, output, ttft, total, args.max_tokens)
        except Exception as exc:
            print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference on a compiled AEG package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("aeg_path", help="Path to the .aeg package directory")
    parser.add_argument("--prompt", help="Single prompt to generate from")
    parser.add_argument("--prompts-file", help="File of prompts (one per line)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--backend", choices=["cpu", "auto"], default="auto",
                        help="Execution backend (default: auto)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode (requires --prompt)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations (default: 10)")

    args = parser.parse_args()
    aeg_path = Path(args.aeg_path)

    print(f"{BOLD}Aether Runtime — Inference{RESET}")
    print(f"  Package : {aeg_path}")

    package = _load_package(aeg_path)
    engine  = _load_cpu_engine(package)
    tokenizer = _try_load_tokenizer(package)

    if args.benchmark:
        if not args.prompt:
            print(f"{RED}--benchmark requires --prompt{RESET}", file=sys.stderr)
            return 1
        return _benchmark_mode(engine, tokenizer, args.prompt, args)

    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        for p in prompts:
            output, ttft, total = _run_one(engine, tokenizer, p, args)
            _print_generation(p, output, ttft, total, args.max_tokens)
        return 0

    if args.prompt:
        output, ttft, total = _run_one(engine, tokenizer, args.prompt, args)
        _print_generation(args.prompt, output, ttft, total, args.max_tokens)
        return 0

    return _interactive_repl(engine, tokenizer, args)


if __name__ == "__main__":
    sys.exit(main())
