"""ExLlamaV2: a specialized runtime for its own EXL2 quantized format.

ExLlamaV2 does not run the published 16-bit checkpoint. It runs an EXL2
quantization of it - mixed bit-widths chosen per tensor - with kernels written for
that format. It is included because it is one of the fastest single-GPU local
engines in practice, and excluded from same-weights speed claims for the same
reason: fewer bits per weight is less memory traffic, and decode is memory bound.

Without an EXL2 conversion on disk there is nothing for this engine to execute, so
it reports ``NOT_APPLICABLE`` with that reason rather than being dropped from the
compatibility table.
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
    key="exllamav2",
    display="ExLlamaV2",
    taxonomy=(base.QUANTIZED_ENGINE, base.RUNTIME, base.EXECUTION_ENGINE),
    summary=(
        "Specialized CUDA runtime for the EXL2 mixed-bit quantized format, with "
        "kernels written for that representation. Quantization is the build step."
    ),
    package="exllamav2",
    requires=("torch", "exllamav2"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    requires_cuda=True,
    alters_representation=True,
    ttft_method="single_token_call",
    notes=(
        "Runs an EXL2 quantization, not the published checkpoint, so every "
        "comparison against it is labelled REPRESENTATION_DIFFERENCE and it is "
        "excluded from same-weights speed claims.",
        "Requires an EXL2 directory supplied with --exl2-map; the suite does not "
        "quantize models itself, because the bit-width allocation would then be the "
        "benchmark's choice rather than a published configuration.",
    ),
)


def locate(model_id: str, options: Any) -> tuple[Path | None, str]:
    mapping = getattr(options, "exl2_map", None) or {}
    if model_id not in mapping:
        return None, (
            f"no EXL2 conversion supplied for {model_id} (--exl2-map). ExLlamaV2 "
            "executes EXL2 weights only, so there is nothing for it to run."
        )
    path = Path(mapping[model_id])
    if not path.is_dir():
        return None, f"--exl2-map points at {path}, which is not a directory"
    return path, ""


class Engine(base.BackendAdapterMixin):
    """Load an EXL2 directory and generate through the dynamic generator."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, path: Path, hf_model_id: str, **_: Any) -> None:
        self.path = path
        self.hf_model_id = hf_model_id
        self._model: Any = None
        self._generator: Any = None
        self._cache: Any = None
        self._tokenizer: Any = None
        self._bits: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": "cuda",
            "exl2_path": str(self.path),
            "bits_per_weight": self._bits,
            "generation": "ExLlamaV2DynamicGenerator",
            "representation": f"EXL2 quantized ({self._bits or 'unknown'} bpw)",
            "quantized": True,
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("exllamav2"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator
        from transformers import AutoTokenizer

        start = time.perf_counter()
        try:
            config = ExLlamaV2Config(str(self.path))
            config.prepare()
            self._model = ExLlamaV2(config)
            self._cache = ExLlamaV2Cache(self._model, lazy=True, max_seq_len=4096)
            self._model.load_autosplit(self._cache, progress=False)
            exl_tokenizer = ExLlamaV2Tokenizer(config)
            self._generator = ExLlamaV2DynamicGenerator(
                model=self._model, cache=self._cache, tokenizer=exl_tokenizer
            )
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"ExLlamaV2 could not load {self.path}: {type(exc).__name__}: {exc}"[:400]
            ) from exc
        load_s = time.perf_counter() - start
        self._bits = getattr(getattr(self._model, "config", None), "bits", None)
        # Scored on the source repository's tokenizer, like every other engine.
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        return LoadOutcome(
            download_s=None,
            prepare_s=None,
            load_s=load_s,
            total_s=load_s,
            notes={
                "exl2_path": str(self.path),
                "artifact_bytes": sum(
                    item.stat().st_size for item in self.path.rglob("*") if item.is_file()
                ),
                "bits_per_weight": self._bits,
                "quantization_supplied": True,
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
        from exllamav2.generator import ExLlamaV2Sampler

        set_seed(seed)
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = max(temperature, 0.0)
        settings.top_p = top_p if temperature > 0.0 else 0.0
        settings.top_k = top_k if (temperature > 0.0 and top_k > 0) else 0
        outputs = self._generator.generate(
            prompt=[prompt] * batch_size,
            max_new_tokens=max_new_tokens,
            gen_settings=settings,
            completion_only=True,
            stop_conditions=[],
            add_bos=True,
            encode_special_tokens=False,
        )
        texts = outputs if isinstance(outputs, list) else [outputs]
        if len(texts) != batch_size:
            raise UnsupportedConfiguration(
                f"ExLlamaV2 returned {len(texts)} rows for batch {batch_size}"
            )
        from benchmark.prompts import flatten_ids

        ids = flatten_ids(self._tokenizer(texts[0], add_special_tokens=False)["input_ids"])
        return GenerationOutcome(
            text=texts[0],
            token_ids=[int(value) for value in ids],
            prompt_tokens=len(
                flatten_ids(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])
            ),
            completion_tokens=len(ids),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": len(texts),
                "engine": "exllamav2",
                "token_ids_source": "re-encoded from decoded text",
            },
        )

    def unload(self) -> None:
        for attribute in ("_generator", "_cache", "_model", "_tokenizer"):
            setattr(self, attribute, None)
        super().unload()


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    path, reason = locate(model_id, options)
    if path is None:
        return base.not_applicable(reason)
    return base.available(base.package_version("exllamav2"))


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    path, reason = locate(model_id, options)
    if path is None:
        raise UnsupportedConfiguration(reason)
    return Engine(path=path, hf_model_id=model_id)
