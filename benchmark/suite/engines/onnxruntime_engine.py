"""ONNX Runtime: an exported graph executed by a cross-platform runtime.

The checkpoint is exported to ONNX once - a real build phase that leaves a real
directory on disk - and then executed by ONNX Runtime's own graph optimizer and
kernels. Like Aether, and unlike torch.compile, what the build leaves behind is a
portable artifact: the same folder loads in another process, and on another
machine with the same runtime.

The honest caveat, recorded and printed rather than buried: an ONNX export of a
16-bit checkpoint is float32 unless it is explicitly converted, so this row is
normally a representation difference and is labelled as one.
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
    key="onnxruntime",
    display="ONNX Runtime",
    taxonomy=(
        base.RUNTIME, base.EXECUTION_ENGINE, base.GRAPH_COMPILER, base.KERNEL_OPTIMIZER,
    ),
    summary=(
        "The model exported to an ONNX graph ahead of time, then executed by ONNX "
        "Runtime with its own graph-level optimizations and kernel library. The "
        "export is a separate build step that produces a portable directory."
    ),
    package="optimum",
    requires=("torch", "optimum", "onnxruntime"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    alters_representation=True,
    ttft_method="single_token_call",
    notes=(
        "Exported through optimum.onnxruntime. The exported graph carries float32 "
        "weights unless a conversion pass is applied, so unless the benchmark "
        "precision is fp32 this row is a representation difference, not a "
        "same-weights comparison, and every derived percentage against it is "
        "labelled REPRESENTATION_DIFFERENCE.",
        "The execution provider actually selected is recorded per run. CUDA "
        "execution requires the onnxruntime-gpu build; the CPU build reports "
        "CPUExecutionProvider even on a GPU host.",
    ),
)


class Engine(base.BackendAdapterMixin):
    """Export once to ONNX, cache the export, then generate through ORT."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "cpu", cache_dir: str | None = None, **_: Any) -> None:
        self.device = device
        self.cache_dir = Path(cache_dir or "benchmark_results/artifacts/onnx")
        self._model: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._export_s: float = 0.0
        self._export_reused: bool | None = None
        self._artifact: Path | None = None
        self._providers: list[str] = []

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "execution_providers": self._providers,
            "artifact": str(self._artifact) if self._artifact else None,
            "artifact_reused": self._export_reused,
            "export_s": self._export_s,
            "generation": "ORTModelForCausalLM.generate (io-binding managed by optimum)",
            "representation": "ONNX graph exported from the checkpoint; float32 weights",
            "quantized": False,
            "ttft_method": SPEC.ttft_method,
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        self._precision = precision
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"

        download_start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        download_s = time.perf_counter() - download_start

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.cache_dir / model_id.replace("/", "--")
        self._artifact = artifact
        reuse = (artifact / "model.onnx").exists() or any(artifact.glob("*.onnx"))
        self._export_reused = reuse

        export_start = time.perf_counter()
        try:
            if reuse:
                self._model = ORTModelForCausalLM.from_pretrained(
                    artifact, provider=provider
                )
                self._export_s = 0.0
            else:
                self._model = ORTModelForCausalLM.from_pretrained(
                    model_id, export=True, provider=provider
                )
                self._model.save_pretrained(artifact)
                self._export_s = time.perf_counter() - export_start
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"ONNX export/load failed for {model_id}: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc
        load_s = time.perf_counter() - export_start - self._export_s
        session = getattr(self._model, "model", None)
        self._providers = list(getattr(session, "get_providers", lambda: [])())
        return LoadOutcome(
            download_s=download_s,
            prepare_s=self._export_s,
            load_s=max(load_s, 0.0),
            total_s=download_s + self._export_s + max(load_s, 0.0),
            notes={
                "exported_this_run": not reuse,
                "artifact_bytes": _tree_size(artifact),
                "execution_providers": self._providers,
                "requested_provider": provider,
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
                "engine": "onnxruntime",
                "execution_providers": self._providers,
            },
        )

    def prefill(self, prompt: str) -> Any:
        """One forward pass through the ONNX graph, logits at every position."""
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**encoded)
        return output.logits[0, -1].detach().float().cpu()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        super().unload()


def _tree_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    ok, reason = base.module_importable("optimum.onnxruntime")
    if not ok:
        return base.not_installed(
            "optimum is installed but optimum.onnxruntime is not importable "
            f"({reason}); install optimum[onnxruntime] or optimum[onnxruntime-gpu]"
        )
    version = base.package_version("optimum")
    if hardware.nvidia and base.package_version("onnxruntime-gpu") is None:
        return base.available(
            version,
            "only the CPU build of onnxruntime is installed, so this engine will "
            "execute on CPU on a GPU host; recorded with its execution provider",
        )
    return base.available(version)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    gpu_build = base.package_version("onnxruntime-gpu") is not None
    return Engine(
        device="cuda" if (hardware.nvidia and gpu_build) else "cpu",
        cache_dir=getattr(options, "onnx_cache_dir", None),
    )
