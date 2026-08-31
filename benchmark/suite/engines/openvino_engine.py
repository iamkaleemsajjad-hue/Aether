"""OpenVINO: Intel's AOT graph compiler and CPU/iGPU inference runtime.

The checkpoint is converted once into OpenVINO IR - a real ahead-of-time build
producing a portable directory - and executed by the OpenVINO runtime, which
compiles the IR for the specific CPU at load time and dispatches its own kernels.
On a CPU-only host this is the most direct competitor to what Aether does, and the
comparison the local-inference story actually hinges on.

Conversion defaults to the benchmark precision where OpenVINO supports it; the
precision actually stored in the IR is read back and reported rather than assumed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    set_seed,
)
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="openvino",
    display="OpenVINO",
    taxonomy=(
        base.AOT_COMPILER, base.GRAPH_COMPILER, base.RUNTIME, base.EXECUTION_ENGINE,
        base.KERNEL_OPTIMIZER,
    ),
    summary=(
        "Ahead-of-time conversion to OpenVINO IR, then execution by the OpenVINO "
        "runtime, which compiles the IR for the host CPU at load time. The IR is a "
        "portable directory another process or machine can load."
    ),
    package="optimum-intel",
    requires=("openvino", "optimum"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    alters_representation=True,
    ttft_method="single_token_call",
    notes=(
        "The IR's stored element type is read back after conversion and printed. "
        "When it differs from the benchmark precision, every percentage derived "
        "against this engine is labelled REPRESENTATION_DIFFERENCE.",
        "Targets CPU by default. An Intel GPU or NPU target is only selected when "
        "one is present and named explicitly, and the chosen device is recorded.",
    ),
)


class Engine(base.BackendAdapterMixin):
    """Convert once to OpenVINO IR, cache it, then generate through the runtime."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "CPU", cache_dir: str | None = None, **_: Any) -> None:
        self.device = device
        self.cache_dir = Path(cache_dir or "benchmark_results/artifacts/openvino")
        self._model: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._convert_s: float = 0.0
        self._reused: bool | None = None
        self._artifact: Path | None = None
        self._ir_element_type: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "ir_element_type": self._ir_element_type,
            "artifact": str(self._artifact) if self._artifact else None,
            "artifact_reused": self._reused,
            "conversion_s": self._convert_s,
            "generation": "OVModelForCausalLM.generate",
            "representation": f"OpenVINO IR ({self._ir_element_type or 'unknown'})",
            "weight_storage_bits": _element_bits(self._ir_element_type),
            "weight_storage_format": self._ir_element_type,
            "quantized": False,
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("openvino"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer

        self._precision = precision
        download_start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        download_s = time.perf_counter() - download_start

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.cache_dir / model_id.replace("/", "--")
        self._artifact = artifact
        reuse = (artifact / "openvino_model.xml").exists()
        self._reused = reuse

        start = time.perf_counter()
        try:
            if reuse:
                self._model = OVModelForCausalLM.from_pretrained(
                    artifact, device=self.device
                )
                self._convert_s = 0.0
                load_s = time.perf_counter() - start
            else:
                self._model = OVModelForCausalLM.from_pretrained(
                    model_id, export=True, device=self.device
                )
                self._convert_s = time.perf_counter() - start
                self._model.save_pretrained(artifact)
                load_s = 0.0
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"OpenVINO conversion/load failed for {model_id}: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc
        self._ir_element_type = _ir_element_type(self._model)
        return LoadOutcome(
            download_s=download_s,
            prepare_s=self._convert_s or None,
            load_s=load_s,
            total_s=download_s + self._convert_s + load_s,
            notes={
                "converted_this_run": not reuse,
                "artifact_bytes": _tree_size(artifact),
                "ir_element_type": self._ir_element_type,
                "openvino_device": self.device,
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
        set_seed(seed)
        encoded = self._tokenizer([prompt] * batch_size, return_tensors="pt")
        prompt_len = int(encoded["input_ids"].shape[1])
        sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "do_sample": sample,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if sample:
            kwargs.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                kwargs["top_k"] = top_k
        output = self._model.generate(**encoded, **kwargs)
        generated = output[0, prompt_len:].tolist()
        return GenerationOutcome(
            text=self._tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=[int(value) for value in generated],
            prompt_tokens=prompt_len,
            completion_tokens=len(generated),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": int(output.shape[0]),
                "engine": "openvino",
                "openvino_device": self.device,
            },
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        super().unload()


def _ir_element_type(model: Any) -> str | None:
    """Read the element type OpenVINO actually stored the weights as."""
    try:
        graph = getattr(model, "model", None)
        for parameter in getattr(graph, "get_parameters", lambda: [])():
            return str(parameter.get_element_type())
        for node in getattr(graph, "get_ordered_ops", lambda: [])():
            if node.get_type_name() == "Constant":
                return str(node.get_element_type())
    except Exception:  # noqa: BLE001 - reported as unknown rather than guessed
        return None
    return None


def _element_bits(element_type: str | None) -> int | None:
    """Width, in bits, of the element type OpenVINO stored the weights as.

    Read from the IR rather than assumed from the requested precision, because the
    conversion decides this and the comparability label depends on it.
    """
    if not element_type:
        return None
    text = str(element_type).lower()
    for name, bits in (("f32", 32), ("float32", 32), ("f16", 16), ("float16", 16),
                       ("bf16", 16), ("bfloat16", 16), ("i8", 8), ("u8", 8),
                       ("i4", 4), ("u4", 4)):
        if name in text:
            return bits
    return None


def _tree_size(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    ok, reason = base.module_importable("optimum.intel")
    if not ok:
        return base.not_installed(
            f"openvino is installed but optimum.intel is not ({reason}); "
            "install optimum[openvino]"
        )
    return base.available(base.package_version("openvino"))


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(
        device=getattr(options, "openvino_device", None) or "CPU",
        cache_dir=getattr(options, "openvino_cache_dir", None),
    )
