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
        "Whole-graph capture (fullgraph=True) is attempted first and falls back to "
        "graph-broken compilation when the model or the generation loop is not "
        "capturable in one piece - which is the common case, not the exception. The "
        "configuration that actually compiled is recorded per model, because they "
        "represent different amounts of compilation.",
    ),
)

#: Compilation configurations, tried in order, best first.
#:
#: ``fullgraph=True`` is the strictest and the most flattering when it works: one
#: whole-graph capture with no breaks. It also fails routinely on real generation
#: code - Transformers' ``generate`` calls functions Dynamo marks as skipped, and
#: models with length-dependent branches (Phi-3's LongRoPE among them) are
#: data-dependent by construction. Refusing to fall back would report those models
#: as "torch.compile unsupported", which is not true: what is unsupported is
#: whole-graph capture. A graph-broken Inductor compilation is still a real
#: torch.compile, and it is what a user following PyTorch's own guidance gets.
#:
#: Whichever configuration succeeds is recorded and printed, because they are not
#: the same amount of compilation and the reader has to know which one produced the
#: number.
COMPILE_ATTEMPTS: tuple[dict[str, Any], ...] = (
    {"mode": "reduce-overhead", "fullgraph": True},
    {"mode": "reduce-overhead", "fullgraph": False},
    {"mode": "default", "fullgraph": False},
)
COMPILE_MODE = COMPILE_ATTEMPTS[0]["mode"]


class Engine(TransformersBackend):
    """The baseline model with a compiled forward and a static cache."""

    spec = SPEC

    def __init__(self, device: str = "cuda", mode: str | None = None, **_: Any) -> None:
        super().__init__(device=device)
        self.name = SPEC.key
        #: When the operator names a mode, it replaces the mode in every attempt but
        #: the fullgraph ladder is kept, since that is what makes the engine runnable.
        self.requested_mode = mode
        self.mode: str | None = None
        self.fullgraph: bool | None = None
        self._compile_s: float | None = None
        self._cache_implementation: str | None = None
        self._attempts: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record.update(
            engine_key=SPEC.key,
            taxonomy=list(SPEC.taxonomy),
            generation="model.generate with a compiled forward and a static KV cache",
            compile_mode=self.mode,
            compile_fullgraph=self.fullgraph,
            compile_attempts=self._attempts,
            cache_implementation=self._cache_implementation,
            initial_compile_s=self._compile_s,
            representation=(
                f"published checkpoint cast to {self._precision or '?'} tensors"
            ),
            weight_storage_bits=32 if self._precision == "fp32" else 16,
            weight_storage_format=self._precision,
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

        pristine = self._model.forward
        compile_start = time.perf_counter()
        for attempt in COMPILE_ATTEMPTS:
            mode = self.requested_mode or attempt["mode"]
            fullgraph = bool(attempt["fullgraph"])
            try:
                self._model.forward = torch.compile(
                    pristine, mode=mode, fullgraph=fullgraph
                )
                self._trigger_first_trace(torch)
            except Exception as exc:  # noqa: BLE001
                # Restore the uncompiled forward before the next attempt, so a failed
                # trace cannot leave a half-wrapped callable behind.
                self._model.forward = pristine
                self._attempts.append({
                    "mode": mode, "fullgraph": fullgraph, "outcome": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                })
                _reset_dynamo()
                continue
            self.mode, self.fullgraph = mode, fullgraph
            self._attempts.append({
                "mode": mode, "fullgraph": fullgraph, "outcome": "compiled",
            })
            break
        else:
            raise UnsupportedConfiguration(
                "torch.compile could not build this model in any configuration. "
                + "; ".join(
                    f"[mode={item['mode']} fullgraph={item['fullgraph']}] {item['error']}"
                    for item in self._attempts
                )
            )
        self._compile_s = time.perf_counter() - compile_start
        notes = dict(outcome.notes)
        notes.update(
            compile_mode=self.mode,
            compile_fullgraph=self.fullgraph,
            compile_attempts=self._attempts,
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

    def _trigger_first_trace(self, torch: Any) -> None:
        """Force compilation now, so no measured iteration pays for it."""
        probe = self._tokenizer("compile probe", return_tensors="pt")
        probe = {key: value.to(self.device) for key, value in probe.items()}
        with torch.no_grad():
            self._model.generate(
                **probe, max_new_tokens=4, min_new_tokens=4, do_sample=False,
                use_cache=True, pad_token_id=self._tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _reset_dynamo() -> None:
    """Clear Dynamo's state between compilation attempts.

    Without this, a failed trace leaves guards and cached graphs behind that make the
    next attempt fail for the previous attempt's reason.
    """
    try:
        import torch

        torch._dynamo.reset()
    except Exception:  # noqa: BLE001 - private API; absence is not fatal
        pass


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
        mode=getattr(options, "compile_mode", None),
    )
