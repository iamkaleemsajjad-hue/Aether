"""DeepSpeed-Inference: kernel injection over a Hugging Face model.

Not a compiler and not a separate runtime: DeepSpeed replaces attention and MLP
modules in a loaded PyTorch model with its own fused CUDA kernels, then lets
Transformers' own ``generate`` drive them. So this row isolates one variable
against the baseline - fused kernels instead of composed eager ops - with the
framework, the weights and the generation loop held fixed.

Kernel injection only covers architectures DeepSpeed has policies for. When it has
none for a model it silently changes nothing, which would make this row a duplicate
of the baseline; the adapter therefore checks whether injection actually replaced
anything and reports ``NOT_SUPPORTED`` when it did not.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.backend_transformers import TransformersBackend
from benchmark.backends import LoadOutcome, UnsupportedConfiguration
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="deepspeed",
    display="DeepSpeed-Inference",
    taxonomy=(base.KERNEL_OPTIMIZER, base.RUNTIME),
    summary=(
        "Kernel-injection layer over a loaded Transformers model: fused CUDA "
        "attention and MLP kernels replacing the eager modules, driven by the "
        "library's own generate. No graph compilation."
    ),
    package="deepspeed",
    requires=("torch", "deepspeed", "transformers"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PROCESS_LOCAL,
    requires_cuda=True,
    ttft_method="streaming",
    notes=(
        "Injection is verified after initialization by counting replaced modules. "
        "If DeepSpeed has no policy for the architecture it changes nothing, and "
        "this row is reported NOT_SUPPORTED rather than shipped as a second copy of "
        "the eager baseline.",
    ),
)


class Engine(TransformersBackend):
    """The baseline model with DeepSpeed's kernels injected."""

    spec = SPEC

    def __init__(self, device: str = "cuda", **_: Any) -> None:
        super().__init__(device=device)
        self.name = SPEC.key
        self._inject_s: float | None = None
        self._replaced: int = 0

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record.update(
            engine_key=SPEC.key,
            taxonomy=list(SPEC.taxonomy),
            generation="model.generate over DeepSpeed-injected kernels",
            injected_modules=self._replaced,
            injection_s=self._inject_s,
            representation="published checkpoint, loaded at the benchmark precision",
            quantized=False,
            ttft_method=SPEC.ttft_method,
            version=base.package_version("deepspeed"),
        )
        return record

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        import deepspeed

        from benchmark.backends import resolve_dtype

        outcome = super().load(model_id, precision)
        before = _module_signature(self._model)
        start = time.perf_counter()
        try:
            engine = deepspeed.init_inference(
                self._model,
                dtype=resolve_dtype(precision),
                replace_with_kernel_inject=True,
                mp_size=1,
            )
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"DeepSpeed could not initialize inference for {model_id}: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc
        self._inject_s = time.perf_counter() - start
        self._model = getattr(engine, "module", engine)
        after = _module_signature(self._model)
        self._replaced = sum(1 for name in after if after[name] != before.get(name))
        if self._replaced == 0:
            raise UnsupportedConfiguration(
                "DeepSpeed kernel injection replaced no module for this "
                "architecture, so this row would be a duplicate of the eager "
                "Transformers baseline rather than a distinct engine"
            )
        notes = dict(outcome.notes)
        notes.update(injection_s=self._inject_s, injected_modules=self._replaced)
        return LoadOutcome(
            download_s=outcome.download_s,
            prepare_s=self._inject_s,
            load_s=outcome.load_s,
            total_s=(outcome.download_s or 0.0) + outcome.load_s + self._inject_s,
            notes=notes,
        )


def _module_signature(model: Any) -> dict[str, str]:
    """Map module path to class name, so injection can be detected by comparison."""
    try:
        return {name: type(module).__name__ for name, module in model.named_modules()}
    except Exception:  # noqa: BLE001
        return {}


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    return base.generic_probe(SPEC, hardware)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(device="cuda")
