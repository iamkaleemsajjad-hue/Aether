"""MLC LLM: a TVM-based AOT compiler and its generated runtime library.

MLC compiles a model through TVM into a device-specific shared library plus
quantized weight shards, then executes that library. Architecturally it is the
closest thing in the field to Aether's ahead-of-time story - compile once, ship an
artifact, run it - which makes it the most interesting comparison for the
compile-once question and the most demanding to set up, because the compilation is
a separate toolchain step outside this benchmark's scope.

The suite therefore measures MLC only when an already-compiled model is supplied,
and otherwise reports why it could not: never a fabricated row, never a zero.
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
    key="mlc",
    display="MLC LLM (TVM)",
    taxonomy=(
        base.AOT_COMPILER, base.GRAPH_COMPILER, base.KERNEL_OPTIMIZER, base.RUNTIME,
        base.QUANTIZED_ENGINE,
    ),
    summary=(
        "TVM-based ahead-of-time compiler: the model is lowered to a device-specific "
        "shared library plus quantized weight shards, which the MLC runtime loads "
        "and executes."
    ),
    package="mlc-llm",
    requires=("mlc_llm",),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    alters_representation=True,
    ttft_method="single_token_call",
    notes=(
        "MLC's default conversion quantizes weights (commonly 4-bit group "
        "quantization), so comparisons against it are labelled "
        "REPRESENTATION_DIFFERENCE unless an unquantized conversion is supplied.",
        "Compilation happens in the mlc_llm toolchain, not in this harness. A "
        "pre-compiled model must be supplied with --mlc-map; the build cost is then "
        "outside the measured window and is reported as not measured rather than as "
        "zero.",
    ),
)


def locate(model_id: str, options: Any) -> tuple[str | None, str]:
    mapping = getattr(options, "mlc_map", None) or {}
    if model_id not in mapping:
        return None, (
            f"no pre-compiled MLC model supplied for {model_id} (--mlc-map). MLC "
            "executes a TVM-compiled library, which this harness does not build."
        )
    return str(mapping[model_id]), ""


class Engine(base.BackendAdapterMixin):
    """Load a pre-compiled MLC model and generate through MLCEngine."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, model: str, hf_model_id: str, **_: Any) -> None:
        self.model = model
        self.hf_model_id = hf_model_id
        self._llm: Any = None
        self._tokenizer: Any = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "mlc_model": self.model,
            "generation": "MLCEngine.completions.create",
            "representation": "TVM-compiled library with quantized weight shards",
            "weight_storage_bits": 4,
            "weight_storage_format": "mlc-group-quantized",
            "quantized": True,
            "build_cost": "not measured; compilation happens in the mlc_llm toolchain",
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("mlc-llm"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from mlc_llm import MLCEngine
        from transformers import AutoTokenizer

        start = time.perf_counter()
        try:
            self._llm = MLCEngine(self.model)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"MLC could not load {self.model}: {type(exc).__name__}: {exc}"[:400]
            ) from exc
        load_s = time.perf_counter() - start
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        return LoadOutcome(
            download_s=None,
            prepare_s=None,
            load_s=load_s,
            total_s=load_s,
            notes={"mlc_model": self.model, "compiled_outside_harness": True},
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
        set_seed(seed)
        if batch_size != 1:
            raise UnsupportedConfiguration(
                "the MLCEngine completion API is driven one request at a time here; "
                "a batch would be serialized, which is not batching"
            )
        response = self._llm.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else 1.0,
            stream=False,
        )
        text = response.choices[0].text
        usage = getattr(response, "usage", None)
        ids = flatten_ids(self._tokenizer(text, add_special_tokens=False)["input_ids"])
        return GenerationOutcome(
            text=text,
            token_ids=[int(value) for value in ids],
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or len(ids)),
            backend_metrics={
                "batch_size": 1,
                "returned_rows": 1,
                "engine": "mlc-llm",
                "token_ids_source": "re-encoded from decoded text",
            },
        )

    def unload(self) -> None:
        terminate = getattr(self._llm, "terminate", None)
        if callable(terminate):
            with contextlib.suppress(Exception):
                terminate()
        self._llm = None
        self._tokenizer = None
        super().unload()


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    model, reason = locate(model_id, options)
    if model is None:
        return base.not_applicable(reason)
    return base.available(base.package_version("mlc-llm"))


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    model, reason = locate(model_id, options)
    if model is None:
        raise UnsupportedConfiguration(reason)
    return Engine(model=model, hf_model_id=model_id)
