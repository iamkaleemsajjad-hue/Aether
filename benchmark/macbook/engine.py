"""A small NumPy greedy loop; every model operation executes in ONNX Runtime.

The supported artifact is the published Qwen3-8B CPU INT4 graph. No framework
export or full precision checkpoint is needed on the benchmark machine.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL = "Qwen/Qwen3-8B"
REPOSITORY = "onnx-community/Qwen3-8B-ONNX"
SUBFOLDER = "onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128"
FILES = ("model.onnx", "model.onnx.data", "genai_config.json", "tokenizer.json")
PROVIDER = "CPUExecutionProvider"
GIB = 1024**3


def estimate_working_bytes(cell: dict[str, int], config: dict[str, Any], weight_bytes: int) -> int:
    """Conservative planning estimate, not a guarantee or measured memory value."""
    decoder = config["model"]["decoder"]
    batch, prompt, output = (cell[key] for key in ("batch_size", "prompt_tokens", "output_tokens"))
    # Two float32 K/V copies can coexist across a session.run boundary.
    kv = (
        2
        * decoder["num_hidden_layers"]
        * batch
        * decoder["num_key_value_heads"]
        * (prompt + output)
        * decoder["head_size"]
        * 4
    )
    logits = batch * prompt * config["model"]["vocab_size"] * 4
    return int(weight_bytes * 1.2 + 2 * kv + logits + GIB)


class Engine:
    """Execute real batches with an independent KV cache for each generation."""

    def __init__(self, directory: Path, threads: int) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Avoid retaining arena allocations from the largest prompt on a 16 GB host.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        start = time.perf_counter()
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=[PROVIDER],
        )
        self.load_s = time.perf_counter() - start
        self.session.disable_fallback()
        if self.session.get_providers() != [PROVIDER]:
            raise RuntimeError(f"Unexpected execution providers: {self.session.get_providers()}")
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.outputs = [item.name for item in self.session.get_outputs()]
        self.cache_names = [name for name in self.inputs if name.startswith("past_key_values.")]
        if set(self.inputs) != {"input_ids", "attention_mask", *self.cache_names}:
            raise ValueError(
                "Unsupported ONNX input signature; use the documented CPU INT4 artifact"
            )
        expected_cache = {
            f"past_key_values.{layer}.{kind}" for layer in range(36) for kind in ("key", "value")
        }
        if set(self.cache_names) != expected_cache or "logits" not in self.outputs:
            raise ValueError("Expected the 36-layer Qwen3-8B decoder with logits and K/V outputs")
        for name in ("input_ids", "attention_mask"):
            if self.inputs[name].type != "tensor(int64)" or len(self.inputs[name].shape) != 2:
                raise ValueError(f"Expected rank-2 int64 input: {name}")
        for name in self.cache_names:
            item = self.inputs[name]
            if item.type != "tensor(float)" or len(item.shape) != 4 or item.shape[1::2] != [8, 128]:
                raise ValueError(
                    f"Unexpected Qwen3-8B KV tensor: {name}: {item.shape}, {item.type}"
                )
            if name.replace("past_key_values.", "present.") not in self.outputs:
                raise ValueError(f"Missing KV output for {name}")

    def generate(self, prompt_ids: list[int], batch: int, output_tokens: int) -> dict[str, Any]:
        """Time generation from prepared token IDs, including prefill and Python loop.

        EOS is deliberately ignored: all rows execute exactly output_tokens steps.
        Tokenization and text rendering are outside the measured region.
        """
        if not prompt_ids or batch < 1 or output_tokens < 1:
            raise ValueError("Generation requires a nonempty prompt and positive batch/output")
        start = time.perf_counter()
        ids = np.repeat(np.asarray(prompt_ids, dtype=np.int64)[None, :], batch, axis=0)
        cache = {name: np.empty((batch, 8, 0, 128), dtype=np.float32) for name in self.cache_names}
        generated = np.empty((batch, output_tokens), dtype=np.int64)
        first_at = None
        for step in range(output_tokens):
            feed = {
                "input_ids": ids,
                "attention_mask": np.ones((batch, len(prompt_ids) + step), dtype=np.int64),
                **cache,
            }
            values = self.session.run(self.outputs, feed)
            result = dict(zip(self.outputs, values))
            logits = result["logits"]
            if (
                logits.ndim != 3
                or logits.shape[0] != batch
                or not np.isfinite(logits[:, -1, :]).all()
            ):
                raise ValueError("Invalid batch dimension or non-finite last-token logits")
            ids = np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)[:, None]
            generated[:, step] = ids[:, 0]
            if step == 0:
                first_at = time.perf_counter()
            cache = {
                name: result[name.replace("past_key_values.", "present.")]
                for name in self.cache_names
            }
            if any(
                value.shape != (batch, 8, len(prompt_ids) + step, 128) for value in cache.values()
            ):
                raise ValueError(
                    "ONNX KV outputs did not grow to the expected batch/sequence length"
                )
            # Release logits and the previous cache before the next call.
            del feed, values, result, logits
        end = time.perf_counter()
        latency = end - start
        ttft = first_at - start
        return {
            "latency_s": latency,
            "ttft_s": ttft,
            "decode_s": end - first_at if output_tokens > 1 else None,
            "tpot_ms": (end - first_at) * 1000 / (output_tokens - 1) if output_tokens > 1 else None,
            "aggregate_tokens_per_s": batch * output_tokens / latency,
            "per_request_tokens_per_s": output_tokens / latency,
            "generated_tokens_total": int(generated.size),
            "token_ids": generated.tolist(),
        }
