"""Example: Benchmark a model on the current hardware."""

from __future__ import annotations

import json

from aether import Runtime


def main() -> None:
    model_id = "Qwen/Qwen3-0.6B"
    rt = Runtime()
    rt.pull(model_id)
    result = rt.benchmark(model_id, prompt="Hello, my name is", max_tokens=128)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
