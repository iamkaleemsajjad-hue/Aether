"""The real Aether CPU executor, with no third-party inference fallback.

This adapter deliberately reports the current implementation's limitations:
weights are expanded to floating point on load and true batching is unavailable.
It must not flatten a batch into one sequence or relabel ONNX execution as Aether.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.macbook.engine import GIB, MODEL

ENGINES = ("aether", "onnxruntime")
# Qwen3-8B has about 8.2B parameters. Loading also holds serialized weight data,
# transient conversion buffers and interpreter state. This is a planning estimate.
AETHER_LOAD_ESTIMATE = 48 * GIB
AETHER_COMPILE_AVAILABLE = 56 * GIB


def read_lock(directory: Path) -> tuple[dict, Path]:
    """Validate the prepared artifact identity without loading the weight blob."""
    from aether.core.aeg_format import AEGPackage

    lock = json.loads((directory / "aether.lock.json").read_text(encoding="utf-8"))
    if lock.get("model") != MODEL or lock.get("engine") != "aether":
        raise ValueError("The Aether lock must identify Qwen/Qwen3-8B and engine=aether")
    artifact = (directory / lock["artifact"]).resolve()
    if not artifact.is_relative_to(directory.resolve()) or not artifact.is_dir():
        raise ValueError("Aether artifact must be a prepared directory within --aether-dir")
    package = AEGPackage(artifact).load()
    architecture = package.manifest.architecture
    geometry = (
        architecture.layers,
        architecture.hidden_size,
        architecture.num_attention_heads,
        architecture.num_kv_heads,
    )
    if geometry != (36, 4096, 32, 8):
        raise ValueError(f"Expected Qwen3-8B geometry, found {geometry}")
    if package.manifest.graph_hash != lock["graph_hash"]:
        raise ValueError("Aether artifact graph hash differs from the prepared lock")
    return lock, artifact


class AetherEngine:
    """Benchmark Aether's native CPUExecutionEngine over its compiled AEG."""

    def __init__(self, directory: Path, threads: int) -> None:
        from aether.runtime.aeg_loader import load_engine_from_path
        from aether.runtime.cpu_engine import CPUExecutionEngine
        from tokenizers import Tokenizer

        _, artifact = read_lock(directory)
        start = time.perf_counter()
        self.executor = load_engine_from_path(artifact)
        self.load_s = time.perf_counter() - start
        if type(self.executor) is not CPUExecutionEngine:
            raise ValueError("Expected Aether CPUExecutionEngine; external engine fallback refused")
        self.tokenizer = Tokenizer.from_file(str(artifact / "tokenizer" / "tokenizer.json"))
        self.execution_info = {
            "implementation": "aether.runtime.cpu_engine.CPUExecutionEngine",
            "native_kernels_available": bool(self.executor.kernels.is_native),
            "native_kernel_build_error": self.executor.kernels.build_error,
            "generation": "Aether forward passes with the benchmark greedy loop",
            "semantic_response_cache": "not used: direct executor calls",
            "weight_compute": "FP32 expanded from the compiled AEG",
            "true_batch_support": "batch 1 only",
            "requested_threads": threads,
        }

    def generate(self, prompt_ids: list[int], batch: int, output_tokens: int) -> dict[str, Any]:
        if batch != 1:
            raise ValueError("Aether CPUExecutionEngine does not support a true batch above 1")
        if not prompt_ids or output_tokens < 1:
            raise ValueError("Generation requires a nonempty prompt and positive output length")
        start = time.perf_counter()
        ids = np.asarray(prompt_ids, dtype=np.int64)
        cache = None
        generated = []
        first_at = None
        for step in range(output_tokens):
            logits, cache = self.executor.forward(ids, cache)
            if logits.ndim != 2 or not np.isfinite(logits[-1]).all():
                raise ValueError("Invalid Aether last-token logits")
            token = int(np.argmax(logits[-1]))
            generated.append(token)
            ids = np.asarray([token], dtype=np.int64)
            if step == 0:
                first_at = time.perf_counter()
            del logits
        end = time.perf_counter()
        latency = end - start
        return {
            "latency_s": latency,
            "ttft_s": first_at - start,
            "decode_s": end - first_at if output_tokens > 1 else None,
            "tpot_ms": (end - first_at) * 1000 / (output_tokens - 1) if output_tokens > 1 else None,
            "aggregate_tokens_per_s": output_tokens / latency,
            "per_request_tokens_per_s": output_tokens / latency,
            "generated_tokens_total": len(generated),
            "token_ids": [generated],
        }
