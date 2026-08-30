"""SGLang: a serving runtime built around RadixAttention prefix reuse.

Like vLLM it is a serving system rather than a compiler, but its distinctive
mechanism is different: a radix tree over KV cache prefixes, so requests that
share a prefix skip recomputing it. That matters for how this benchmark must
treat it - the suite issues the *same* prompt many times, which is precisely the
pattern prefix caching is designed to shortcut.

The suite therefore disables prefix caching for SGLang, exactly as it disables
Aether's semantic response cache, so that repeated iterations measure inference
rather than a cache hit. Both overrides are public flags, both are recorded, and
neither changes the engine's own default outside this benchmark.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    set_seed,
)
from benchmark.prompts import flatten_ids
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="sglang",
    display="SGLang",
    taxonomy=(base.SERVING_ENGINE, base.RUNTIME, base.EXECUTION_ENGINE),
    summary=(
        "Optimized LLM serving runtime: RadixAttention prefix cache, a continuous "
        "scheduler and fused kernels. Measured through its offline Engine API so no "
        "HTTP time is attributed to inference."
    ),
    package="sglang",
    requires=("torch", "sglang"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_DISK_CACHE,
    requires_cuda=True,
    ttft_method="single_token_call",
    notes=(
        "Prefix caching (RadixAttention) is disabled for every measured run via the "
        "public disable_radix_cache flag. The benchmark issues one prompt "
        "repeatedly, so with it on the second and later iterations would reuse the "
        "prefill instead of performing it, and the row would report a cache lookup "
        "as decode throughput.",
        "Engine start-up includes weight loading, KV-pool sizing and CUDA-graph "
        "capture, timed together because SGLang does not separate them.",
        "Generated token ids are not returned by the offline API by default, so "
        "correctness for this engine is compared on decoded text and on completion "
        "counts taken from the engine's own meta_info, with ids re-encoded from the "
        "text and labelled as such.",
    ),
)

_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class Engine(base.BackendAdapterMixin):
    """Drive SGLang's offline engine under the suite's fixed settings."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "cuda", memory_fraction: float = 0.80, **_: Any) -> None:
        self.device = device
        self.memory_fraction = memory_fraction
        self._llm: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._startup_s: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "mem_fraction_static": self.memory_fraction,
            "engine_startup_s": self._startup_s,
            "generation": "sglang.Engine.generate (offline)",
            "representation": "published checkpoint, loaded at the benchmark precision",
            "quantized": False,
            "runtime_flags_overridden": {"disable_radix_cache": True},
            "runtime_flags_reason": (
                "prefix caching is disabled so repeated prompts measure inference "
                "rather than a prefix-cache hit"
            ),
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("sglang"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        import sglang as sgl
        from transformers import AutoTokenizer

        if precision not in _DTYPE:
            raise UnsupportedConfiguration(f"unknown precision {precision!r}")
        self._precision = precision
        start = time.perf_counter()
        try:
            self._llm = sgl.Engine(
                model_path=model_id,
                dtype=_DTYPE[precision],
                mem_fraction_static=self.memory_fraction,
                disable_radix_cache=True,
                log_level="warning",
                random_seed=0,
            )
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"SGLang could not start for {model_id}: {type(exc).__name__}: {exc}"[:400]
            ) from exc
        self._startup_s = time.perf_counter() - start
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        return LoadOutcome(
            download_s=None,
            prepare_s=None,
            load_s=self._startup_s,
            total_s=self._startup_s,
            notes={
                "engine_startup_s": self._startup_s,
                "mem_fraction_static": self.memory_fraction,
                "sglang_version": base.package_version("sglang"),
                "startup_includes": "weight load, KV pool allocation, graph capture",
                "radix_cache": "disabled for measurement",
            },
        )

    def tokenizer(self) -> Any:
        return self._tokenizer

    def _params(self, max_new_tokens: int, temperature: float, top_p: float,
                top_k: int) -> dict[str, Any]:
        return {
            "temperature": temperature,
            "top_p": top_p if temperature > 0.0 else 1.0,
            "top_k": top_k if (temperature > 0.0 and top_k > 0) else -1,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "ignore_eos": True,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        batch_size: int = 1,
    ) -> GenerationOutcome:
        set_seed(seed)
        params = self._params(max_new_tokens, temperature, top_p, top_k)
        outputs = self._llm.generate([prompt] * batch_size, params)
        if isinstance(outputs, dict):
            outputs = [outputs]
        if len(outputs) != batch_size:
            raise UnsupportedConfiguration(
                f"SGLang returned {len(outputs)} rows for batch {batch_size}"
            )
        rows = [
            int((item.get("meta_info") or {}).get("completion_tokens") or 0)
            for item in outputs
        ]
        text = outputs[0].get("text", "")
        meta = outputs[0].get("meta_info") or {}
        ids = (
            flatten_ids(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            if self._tokenizer else []
        )
        return GenerationOutcome(
            text=text,
            token_ids=[int(value) for value in ids],
            prompt_tokens=int(meta.get("prompt_tokens") or 0),
            # Taken from the engine's own accounting, not from re-encoded text, so a
            # decode/encode round trip cannot change the token count this row is
            # credited with.
            completion_tokens=rows[0] or len(ids),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": len(outputs),
                "engine": "sglang",
                "row_completion_tokens": rows,
                "token_ids_source": "re-encoded from decoded text",
                "finish_reason": meta.get("finish_reason"),
            },
        )

    def unload(self) -> None:
        shutdown = getattr(self._llm, "shutdown", None)
        if callable(shutdown):
            with contextlib.suppress(Exception):
                shutdown()
        self._llm = None
        self._tokenizer = None
        super().unload()


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    return base.generic_probe(SPEC, hardware)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(
        device="cuda",
        memory_fraction=getattr(options, "sglang_memory_fraction", None) or 0.80,
    )
