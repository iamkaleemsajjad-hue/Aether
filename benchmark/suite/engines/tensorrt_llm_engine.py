"""TensorRT-LLM: NVIDIA's AOT engine builder and inference runtime.

A checkpoint is compiled into a TensorRT engine plan - kernels selected and
autotuned for one specific GPU architecture - and executed by TensorRT's runtime.
It is the strongest compiled competitor on NVIDIA hardware and the most
hardware-bound: the plan is not portable across GPU architectures or TensorRT
versions, which is a real difference from Aether's artifact and is stated as one.

Measured through the offline ``tensorrt_llm.LLM`` API, whose build step is timed as
this engine's build phase. On a host with no NVIDIA device the engine is reported
``NOT_APPLICABLE``, never as a zero.
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
from benchmark.suite.engines import base

#: TensorRT-LLM's published wheels are built for Ampere and newer. Turing support
#: exists in source builds but not in the packages a pip install produces, so a
#: pre-Ampere host is reported as inapplicable rather than told to install something
#: that would not run.
MIN_CAPABILITY = (8, 0)

SPEC = base.EngineSpec(
    key="tensorrt_llm",
    display="TensorRT-LLM",
    taxonomy=(
        base.AOT_COMPILER, base.GRAPH_COMPILER, base.KERNEL_OPTIMIZER, base.RUNTIME,
        base.SERVING_ENGINE,
    ),
    summary=(
        "NVIDIA's ahead-of-time engine builder plus TensorRT runtime: kernels are "
        "selected and autotuned for one GPU architecture and baked into an engine "
        "plan, which the runtime then executes."
    ),
    package="tensorrt_llm",
    requires=("tensorrt_llm",),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    requires_cuda=True,
    min_capability=MIN_CAPABILITY,
    ttft_method="single_token_call",
    notes=(
        "The engine plan is a file, but it is tied to the GPU architecture, the "
        "TensorRT version and the build-time shape profile. It is portable in the "
        "sense that another process can load it; it is not portable across "
        "hardware, which distinguishes it from Aether's artifact.",
        "Engine build time is charged to the build phase and never amortized into "
        "throughput.",
    ),
)


class Engine(base.BackendAdapterMixin):
    """Build a TensorRT engine, then generate through the offline LLM API."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "cuda", **_: Any) -> None:
        self.device = device
        self._llm: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._build_s: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "engine_build_s": self._build_s,
            "generation": "tensorrt_llm.LLM.generate (offline)",
            "representation": "TensorRT engine plan built from the checkpoint",
            "weight_storage_bits": 32 if self._precision == "fp32" else 16,
            "weight_storage_format": self._precision,
            "quantized": False,
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("tensorrt_llm"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from tensorrt_llm import LLM

        self._precision = precision
        start = time.perf_counter()
        try:
            self._llm = LLM(model=model_id, dtype=_DTYPE.get(precision, "auto"))
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"TensorRT-LLM could not build an engine for {model_id}: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc
        self._build_s = time.perf_counter() - start
        self._tokenizer = getattr(self._llm, "tokenizer", None)
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        return LoadOutcome(
            download_s=None,
            prepare_s=self._build_s,
            load_s=0.0,
            total_s=self._build_s,
            notes={
                "engine_build_s": self._build_s,
                "tensorrt_llm_version": base.package_version("tensorrt_llm"),
                "build_includes": "graph conversion, kernel autotuning, plan serialization",
            },
        )

    def tokenizer(self) -> Any:
        return self._tokenizer

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
        from tensorrt_llm import SamplingParams

        set_seed(seed)
        params = SamplingParams(
            max_tokens=max_new_tokens,
            min_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else 1.0,
            top_k=top_k if (temperature > 0.0 and top_k > 0) else None,
        )
        outputs = self._llm.generate([prompt] * batch_size, params)
        if len(outputs) != batch_size:
            raise UnsupportedConfiguration(
                f"TensorRT-LLM returned {len(outputs)} rows for batch {batch_size}"
            )
        first = outputs[0].outputs[0]
        ids = [int(value) for value in (first.token_ids or [])]
        return GenerationOutcome(
            text=first.text,
            token_ids=ids,
            prompt_tokens=len(getattr(outputs[0], "prompt_token_ids", None) or []),
            completion_tokens=len(ids),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": len(outputs),
                "engine": "tensorrt-llm",
                "row_completion_tokens": [
                    len(item.outputs[0].token_ids or []) for item in outputs
                ],
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


_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if generic.status == "NOT_INSTALLED":
        # Say what installing it would take, because on most hosts it is not a
        # one-line pip install: the wheels are multi-gigabyte, they pin a CUDA and
        # TensorRT pair, and the published builds target Ampere and newer.
        capabilities = ", ".join(hardware.compute_capabilities) or "unknown"
        return base.not_installed(
            "tensorrt_llm is not installed. Its published wheels require a matching "
            "CUDA and TensorRT installation and target compute capability 8.0 or "
            f"newer; this host reports {capabilities}. Install it deliberately "
            "(pip install tensorrt-llm, several GB) if this hardware supports it."
        )
    return generic


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(device="cuda")
