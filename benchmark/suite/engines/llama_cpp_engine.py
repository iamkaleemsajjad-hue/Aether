"""llama.cpp: a native C/C++ inference runtime over its own GGUF format.

llama.cpp does not execute the published checkpoint. It executes a GGUF file
converted from it, usually quantized, with hand-written SIMD and CUDA kernels and
no Python in the decode loop. Two consequences the suite has to handle explicitly:

* a GGUF must exist before the engine can be measured at all, so when none is
  available this engine is reported ``NOT_APPLICABLE`` with that reason rather
  than silently omitted;
* when the GGUF is quantized, the row is a *representation difference*. It is
  still worth measuring - it is how people actually run local inference - but
  every percentage derived against it is labelled, because fewer bits per weight
  is less memory traffic, and that is most of what decode speed is.

Conversion to an unquantized F16 GGUF is supported when llama.cpp's converter is
reachable, because that is the one configuration where the comparison is
like-for-like on weights.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
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
    key="llama_cpp",
    display="llama.cpp",
    taxonomy=(base.RUNTIME, base.EXECUTION_ENGINE, base.QUANTIZED_ENGINE),
    summary=(
        "Native C/C++ inference runtime with its own GGUF weight format and "
        "hand-written SIMD and CUDA kernels. No graph compiler and no Python in the "
        "decode loop; the build step is a format conversion, not a compilation."
    ),
    package="llama_cpp_python",
    requires=("llama_cpp",),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    alters_representation=True,
    ttft_method="single_token_call",
    notes=(
        "Requires a GGUF conversion of the checkpoint. The quantization type "
        "actually loaded is read from the file and printed with the result; an "
        "F16 GGUF is the only configuration where weights match the other engines.",
        "The high-level Python binding drives one sequence at a time, so batch "
        "widths above 1 are reported UNSUPPORTED rather than emulated by a loop, "
        "which would report serialization as batching.",
        "n_gpu_layers is set to offload everything when a CUDA build is present, "
        "and recorded either way, since a partially offloaded model is a different "
        "configuration from a fully offloaded one.",
    ),
)


def _gguf_dir(options: Any) -> Path:
    return Path(getattr(options, "gguf_dir", None) or "benchmark_results/artifacts/gguf")


def locate_gguf(model_id: str, options: Any) -> tuple[Path | None, str]:
    """Find a GGUF for this model, or say what is missing.

    Looks only where the operator pointed it: an explicit ``model_id=path``
    mapping, then a conventional filename inside the GGUF directory. It never
    downloads a third-party quantization of its own accord, because a stranger's
    quantization is not the checkpoint the rest of the suite is measuring.
    """
    mapping = getattr(options, "gguf_map", None) or {}
    if model_id in mapping:
        path = Path(mapping[model_id])
        if path.is_file():
            return path, ""
        return None, f"--gguf-map points at {path}, which does not exist"
    directory = _gguf_dir(options)
    stem = model_id.replace("/", "--")
    for candidate in sorted(directory.glob(f"{stem}*.gguf")):
        return candidate, ""
    return None, (
        f"no GGUF for {model_id}: none supplied with --gguf-map and none found as "
        f"{directory / (stem + '*.gguf')}"
    )


def convert_to_gguf(model_id: str, options: Any) -> tuple[Path | None, float, str]:
    """Convert the checkpoint to an F16 GGUF using llama.cpp's own converter.

    F16 rather than a quantization, deliberately: it is the only conversion that
    leaves this engine holding the same values as every other engine, so it is the
    only one that can support a same-weights speed claim. Returns the conversion
    time as the engine's build cost.
    """
    script = getattr(options, "gguf_convert_script", None)
    if not script:
        return None, 0.0, "no --gguf-convert-script given, so no conversion attempted"
    script_path = Path(script)
    if not script_path.is_file():
        return None, 0.0, f"--gguf-convert-script {script_path} does not exist"
    directory = _gguf_dir(options)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{model_id.replace('/', '--')}-f16.gguf"

    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    try:
        local = snapshot_download(model_id)
        completed = subprocess.run(
            [sys.executable, str(script_path), local, "--outfile", str(target),
             "--outtype", "f16"],
            capture_output=True, text=True, timeout=3600,
        )
    except Exception as exc:  # noqa: BLE001
        return None, time.perf_counter() - start, f"conversion raised: {exc}"[:300]
    if completed.returncode != 0 or not target.is_file():
        tail = (completed.stderr or completed.stdout or "").strip()[-300:]
        return None, time.perf_counter() - start, f"converter exited {completed.returncode}: {tail}"
    return target, time.perf_counter() - start, ""


class Engine(base.BackendAdapterMixin):
    """Load a GGUF into llama.cpp and generate through the Python binding."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "cpu", options: Any = None, threads: int | None = None,
                 context: int = 4096, **_: Any) -> None:
        self.device = device
        self.options = options
        self.threads = threads
        self.context = context
        self._llm: Any = None
        self._tokenizer: Any = None
        self._path: Path | None = None
        self._quantization: str | None = None
        self._convert_s: float = 0.0
        self._gpu_layers: int = 0
        self._precision: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "gguf_path": str(self._path) if self._path else None,
            "gguf_quantization": self._quantization,
            "n_gpu_layers": self._gpu_layers,
            "n_threads": self.threads,
            "n_ctx": self.context,
            "conversion_s": self._convert_s,
            "generation": "llama_cpp.Llama.create_completion (single sequence)",
            "representation": f"GGUF ({self._quantization or 'unknown'})",
            "quantized": bool(self._quantization and "F16" not in str(self._quantization).upper()),
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("llama_cpp_python"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        self._precision = precision
        path, reason = locate_gguf(model_id, self.options)
        if path is None:
            path, self._convert_s, convert_reason = convert_to_gguf(model_id, self.options)
            if path is None:
                raise UnsupportedConfiguration(f"{reason}; {convert_reason}")
        self._path = path

        # The tokenizer comes from the original repository so this engine is scored
        # on the same token counts as every other engine, not on GGUF's own
        # vocabulary bookkeeping.
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._gpu_layers = -1 if self.device == "cuda" else 0
        start = time.perf_counter()
        try:
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.context,
                n_threads=self.threads,
                n_gpu_layers=self._gpu_layers,
                logits_all=False,
                verbose=False,
                seed=0,
            )
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"llama.cpp could not load {path.name}: {type(exc).__name__}: {exc}"[:400]
            ) from exc
        load_s = time.perf_counter() - start
        metadata = dict(getattr(self._llm, "metadata", {}) or {})
        self._quantization = str(
            metadata.get("general.file_type")
            or _quant_from_name(path.name)
            or "unknown"
        )
        return LoadOutcome(
            download_s=None,
            prepare_s=self._convert_s or None,
            load_s=load_s,
            total_s=self._convert_s + load_s,
            notes={
                "gguf_path": str(path),
                "gguf_bytes": path.stat().st_size,
                "gguf_quantization": self._quantization,
                "n_gpu_layers": self._gpu_layers,
                "converted_this_run": bool(self._convert_s),
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
        if batch_size != 1:
            raise UnsupportedConfiguration(
                "the llama.cpp Python binding drives one sequence at a time; a batch "
                f"of {batch_size} would have to be serialized, which is not batching"
            )
        self._llm.reset()
        result = self._llm.create_completion(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else 1.0,
            top_k=top_k if (temperature > 0.0 and top_k > 0) else 0,
            # No stop sequences: every engine is asked for the same fixed amount of
            # work, so none can finish early and appear faster for doing less.
            stop=[],
            echo=False,
        )
        text = result["choices"][0]["text"]
        usage = result.get("usage") or {}
        ids = [int(value) for value in self._llm.tokenize(text.encode("utf-8"),
                                                          add_bos=False)]
        return GenerationOutcome(
            text=text,
            token_ids=ids,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or len(ids)),
            backend_metrics={
                "batch_size": 1,
                "returned_rows": 1,
                "engine": "llama.cpp",
                "finish_reason": result["choices"][0].get("finish_reason"),
                "token_ids_source": "llama.cpp tokenizer over the generated text",
            },
        )

    def supports_batch(self, batch_size: int) -> bool:
        return batch_size == 1

    def unload(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        self._llm = None
        self._tokenizer = None
        super().unload()


def _quant_from_name(name: str) -> str | None:
    """Read the quantization label out of a conventional GGUF filename."""
    upper = name.upper()
    for tag in ("F32", "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0",
                "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q2_K", "IQ4_XS"):
        if tag in upper:
            return tag
    return None


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    path, reason = locate_gguf(model_id, options)
    if path is None and not getattr(options, "gguf_convert_script", None):
        return base.not_applicable(
            f"{reason}. llama.cpp executes GGUF, not the published checkpoint, so "
            "without one there is nothing for it to run on this model."
        )
    return base.available(
        base.package_version("llama_cpp_python"),
        "" if path else "a GGUF will be converted from the checkpoint before measuring",
    )


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(
        device="cuda" if hardware.nvidia else "cpu",
        options=options,
        threads=getattr(options, "threads", None),
        context=getattr(options, "llama_cpp_context", None) or 4096,
    )
