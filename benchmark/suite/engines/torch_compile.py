"""torch.compile: PyTorch's JIT graph capture and kernel generation.

The same weights and the same framework as the Transformers baseline, with one
thing added: the forward pass is traced, lowered through TorchInductor, and
executed as generated kernels instead of interpreted op by op. Paired with a
static KV cache, which is what makes the decode step's shapes stable enough to
compile once and reuse.

This is the closest comparison in the field to what Aether does, and the most
informative one for the compile-once question, because the two systems pay a
build cost for the same reason and get to keep it for different lengths of time:
Inductor's output lives in a keyed on-disk cache tied to this library, this
device and this graph; Aether's lives in a file.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.backend_transformers import TransformersBackend
from benchmark.backends import LoadOutcome, UnsupportedConfiguration
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="torch_compile",
    display="PyTorch torch.compile (Inductor)",
    taxonomy=(
        base.JIT_COMPILER, base.GRAPH_COMPILER, base.KERNEL_OPTIMIZER, base.FRAMEWORK,
    ),
    summary=(
        "Transformers' model graph captured by TorchDynamo and lowered by "
        "TorchInductor into generated kernels, with a static KV cache so decode "
        "shapes stay compilable. Compilation is just-in-time on first call."
    ),
    package="torch",
    requires=("torch", "transformers"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_DISK_CACHE,
    ttft_method="streaming",
    notes=(
        "Inductor caches generated code on disk, keyed to the torch build, the "
        "device and the traced graph. A later process on the same machine reuses it; "
        "it is not a portable artifact and cannot be shipped to a different host.",
        "Each distinct input shape can trigger a fresh trace. Warm-up iterations "
        "run the measured shape, so recompilation is paid before the timed "
        "iterations rather than inside them, and the first unwarmed call is "
        "reported separately as the cold latency.",
    ),
)

#: Compilation mode. ``reduce-overhead`` is the mode PyTorch documents for
#: latency-bound decoding (it enables CUDA graph capture); it is also the mode a
#: user following the Transformers guide would apply, which is the point.
COMPILE_MODE = "reduce-overhead"


class Engine(TransformersBackend):
    """The baseline model with a compiled forward and a static cache."""

    spec = SPEC

    def __init__(self, device: str = "cuda", mode: str = COMPILE_MODE, **_: Any) -> None:
        super().__init__(device=device)
        self.name = SPEC.key
        self.mode = mode
        self._compile_s: float | None = None
        self._cache_implementation: str | None = None

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record.update(
            engine_key=SPEC.key,
            taxonomy=list(SPEC.taxonomy),
            generation="model.generate with a compiled forward and a static KV cache",
            compile_mode=self.mode,
            cache_implementation=self._cache_implementation,
            initial_compile_s=self._compile_s,
            representation="published checkpoint, loaded at the benchmark precision",
            quantized=False,
            ttft_method=SPEC.ttft_method,
        )
        return record

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        """Load as the baseline does, then compile and pay the first trace.

        The initial compilation is timed into ``prepare_s`` - the same field the
        Aether adapter reports its ahead-of-time compile in - so the report can put
        the two build costs in one column without either being folded into a
        throughput figure.
        """
        import torch

        outcome = super().load(model_id, precision)
        try:
            self._model.generation_config.cache_implementation = "static"
            self._cache_implementation = "static"
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"this Transformers version does not accept a static cache: {exc}"
            ) from exc
        compile_start = time.perf_counter()
        try:
            self._model.forward = torch.compile(
                self._model.forward, mode=self.mode, fullgraph=True
            )
            # Force the first trace here rather than inside a measured iteration.
            probe_ids = self._tokenizer("compile probe", return_tensors="pt")
            probe_ids = {k: v.to(self.device) for k, v in probe_ids.items()}
            with torch.no_grad():
                self._model.generate(
                    **probe_ids, max_new_tokens=4, min_new_tokens=4, do_sample=False,
                    use_cache=True, pad_token_id=self._tokenizer.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"torch.compile could not build this model here: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc
        self._compile_s = time.perf_counter() - compile_start
        notes = dict(outcome.notes)
        notes.update(
            compile_mode=self.mode,
            initial_compile_s=self._compile_s,
            cache_implementation=self._cache_implementation,
            inductor_cache_dir=_inductor_cache_dir(),
        )
        return LoadOutcome(
            download_s=outcome.download_s,
            prepare_s=self._compile_s,
            load_s=outcome.load_s,
            total_s=(outcome.download_s or 0.0) + outcome.load_s + self._compile_s,
            notes=notes,
        )


def _inductor_cache_dir() -> str | None:
    """Where Inductor keeps its generated code, when it can be determined.

    Recorded as the evidence for this engine's artifact-persistence
    classification: a real directory on this machine, not a portable file.
    """
    import os
    import tempfile

    explicit = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if explicit:
        return explicit
    try:
        from torch._inductor import codecache

        getter = getattr(codecache, "cache_dir", None)
        if callable(getter):
            return str(getter())
    except Exception:  # noqa: BLE001 - private API, absence is fine
        pass
    return os.path.join(tempfile.gettempdir(), "torchinductor_*")


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - generic_probe already checked
        return base.not_installed(str(exc))
    if not hasattr(torch, "compile"):
        return base.not_supported(
            f"torch {torch.__version__} has no torch.compile; it was added in 2.0"
        )
    if hardware.os_name == "Windows" and not hardware.nvidia:
        # Inductor's CPU path needs a C++ toolchain on PATH; on Windows that is
        # MSVC, which is usually absent. Say so rather than failing mid-run.
        ok, _ = base.module_importable("torch._inductor")
        if not ok:
            return base.not_supported("TorchInductor is unavailable in this build")
    return base.available(torch.__version__)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(
        device="cuda" if hardware.nvidia else "cpu",
        mode=getattr(options, "compile_mode", None) or COMPILE_MODE,
    )
