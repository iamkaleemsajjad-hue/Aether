"""Example: Compile a model and run it with Aether."""

from __future__ import annotations

from aether import Compiler, Runtime


def main() -> None:
    model_id = "Qwen/Qwen3-0.6B"

    # Compile the model to an AEG artifact
    print(f"Compiling {model_id}...")
    compiler = Compiler()
    plan = compiler.plan(model_id)
    print(f"Estimated memory: {plan.estimated_memory_gb:.1f} GB")
    print(f"Estimated compile time: {plan.estimated_compile_time_s:.1f} s")
    aeg = compiler.compile(model_id)
    print(f"AEG saved to {aeg.root}")

    # Run the model
    print(f"\nRunning {model_id}...")
    rt = Runtime()
    response = rt.generate(
        model_id=model_id,
        prompt="Explain the AEG format in one sentence.",
        max_tokens=64,
    )
    print(f"Response: {response.text}")
    print(f"TPS: {response.metrics.throughput_tps:.1f}")
    print(f"TTFT: {response.metrics.ttft_ms:.1f}ms")
    print(f"Backend: {response.metrics.backend_name}")


if __name__ == "__main__":
    main()
