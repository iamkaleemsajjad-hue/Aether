"""
Aether benchmark suite.

Runs throughput and latency benchmarks for compiled models on the current hardware.
Supports comparisons and result serialization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from aether import Runtime, RuntimeConfig


DEFAULT_PROMPT = "Hello, my name is"
DEFAULT_MAX_TOKENS = 128


def benchmark_model(
    model_id: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Benchmark a single model."""
    rt = Runtime(config or RuntimeConfig())
    rt.pull(model_id)
    return rt.benchmark(model_id, prompt=prompt, max_tokens=max_tokens)


def run_benchmarks(
    models: list[str],
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Run benchmarks for a list of models."""
    results: list[dict[str, Any]] = []
    for model_id in models:
        try:
            result = benchmark_model(model_id, prompt=prompt, max_tokens=max_tokens)
            results.append(result)
            if smoke:
                break
        except Exception as exc:
            results.append({
                "model_id": model_id,
                "error": str(exc),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aether benchmark suite")
    parser.add_argument("--models", type=str, default="Qwen/Qwen3-0.6B", help="Comma-separated model IDs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--smoke", action="store_true", help="Run one quick benchmark only")
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",")]
    results = run_benchmarks(
        models=model_ids,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        smoke=args.smoke,
    )

    output = json.dumps(results, indent=2, default=str)
    print(output)
    if args.output:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
