"""What an engine adapter is, and the vocabulary used to describe one honestly.

Every adapter in this package is a thin shim onto somebody else's inference
stack. The shim's only jobs are to satisfy :class:`benchmark.backends.Backend`
so the shared measurement primitives can time it, and to declare, in fields the
report prints verbatim, what the stack actually *is* and what representation of
the model it actually ran.

The second job is the important one. Calling every system here a "compiler" would
be wrong, and comparing a 4-bit engine against a 16-bit one without saying so
would be worse. :class:`EngineSpec` exists so neither can happen silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.suite import status as status_mod

# ── Taxonomy ────────────────────────────────────────────────────────────────
# Accurate classifications. An engine carries every label that applies to it, so
# `torch.compile` is a JIT compiler *and* part of a framework, and vLLM is a
# serving engine *and* a runtime, without either being called something it is not.

FRAMEWORK = "inference framework"
RUNTIME = "runtime"
EXECUTION_ENGINE = "execution engine"
GRAPH_COMPILER = "graph compiler"
JIT_COMPILER = "JIT compiler"
AOT_COMPILER = "AOT compiler"
KERNEL_OPTIMIZER = "kernel optimization system"
SERVING_ENGINE = "serving engine"
QUANTIZED_ENGINE = "quantized inference engine"

# ── Artifact persistence, for the compile-once question ─────────────────────
#: No compilation step at all: the checkpoint is interpreted every time.
ARTIFACT_NONE = "none"
#: Compilation happens per process and is lost on exit.
ARTIFACT_PROCESS_LOCAL = "process-local"
#: Compilation is cached on disk but keyed to the installed library, the device
#: and the exact graph; it accelerates a later process but is not a distributable
#: artifact.
ARTIFACT_DISK_CACHE = "on-disk-cache"
#: Compilation produces a self-contained file or directory that a different
#: process, and in principle a different machine, can load and execute.
ARTIFACT_PORTABLE = "portable-artifact"


@dataclass(frozen=True)
class EngineSpec:
    """A factual description of one inference stack under test."""

    key: str
    display: str
    taxonomy: tuple[str, ...]
    summary: str
    #: Distribution name used to look up an installed version.
    package: str | None = None
    #: Modules that must import for the engine to be usable at all.
    requires: tuple[str, ...] = ()
    #: True when the stack has a distinct build/compile/export phase whose cost is
    #: paid before serving.
    has_build_phase: bool = False
    #: Which of the ARTIFACT_* constants describes what that phase leaves behind.
    artifact_persistence: str = ARTIFACT_NONE
    #: The engine cannot run without an NVIDIA CUDA device.
    requires_cuda: bool = False
    #: The engine cannot run on this OS. Empty means no restriction.
    supported_os: tuple[str, ...] = ()
    #: True when the engine executes a representation that is not the published
    #: 16-bit checkpoint (a quantization, a re-export, a different container).
    #: Comparisons against such an engine are labelled, not suppressed.
    alters_representation: bool = False
    #: How time-to-first-token is obtained: a real streaming API, or a
    #: single-token generation call. Disclosed because they are not the same
    #: measurement.
    ttft_method: str = "streaming"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display": self.display,
            "taxonomy": list(self.taxonomy),
            "summary": self.summary,
            "package": self.package,
            "has_build_phase": self.has_build_phase,
            "artifact_persistence": self.artifact_persistence,
            "requires_cuda": self.requires_cuda,
            "alters_representation": self.alters_representation,
            "ttft_method": self.ttft_method,
            "notes": list(self.notes),
        }


@dataclass
class Availability:
    """Whether an engine can run here, and the reason when it cannot.

    The reason is a sentence, not a code: it is printed in the compatibility table
    so a reader can tell "this box has no NVIDIA GPU" from "this package is not
    installed" from "this engine does not implement this architecture".
    """

    status: str
    reason: str = ""
    version: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status == status_mod.MEASURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "version": self.version,
            **({"detail": self.detail} if self.detail else {}),
        }


def available(version: str | None = None, reason: str = "") -> Availability:
    """The engine can run: report it as measurable."""
    return Availability(status_mod.MEASURED, reason, version)


def not_installed(reason: str) -> Availability:
    return Availability(status_mod.NOT_INSTALLED, reason)


def not_applicable(reason: str) -> Availability:
    return Availability(status_mod.NOT_APPLICABLE, reason)


def not_supported(reason: str) -> Availability:
    return Availability(status_mod.NOT_SUPPORTED, reason)


def package_version(name: str | None) -> str | None:
    """Installed version of a distribution, or None when it is absent."""
    if not name:
        return None
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - absence is the answer
        return None


def module_importable(name: str) -> tuple[bool, str]:
    """Whether a module can be imported, without importing it.

    ``find_spec`` is used rather than a real import so that probing the field of
    engines does not pull a dozen heavyweight CUDA libraries into the
    orchestrator process, where their allocators would then be resident for the
    rest of the run.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return False, f"{name}: {exc}"
    if spec is None:
        return False, f"{name} is not installed"
    return True, ""


def generic_probe(spec: EngineSpec, hardware: Any) -> Availability:
    """The applicability checks every engine shares.

    Returns ``None``-equivalent only by returning an available Availability: a
    caller that needs extra checks runs them after this one passes. Hardware
    applicability is settled before installation, because "this machine has no
    NVIDIA GPU" is a more informative answer than "TensorRT-LLM is not
    installed" on a machine where it could never have been.
    """
    if spec.requires_cuda and not getattr(hardware, "nvidia", False):
        return not_applicable(
            f"{spec.display} requires an NVIDIA CUDA device; this host reports "
            f"accelerator={getattr(hardware, 'accelerator', 'unknown')}"
        )
    if spec.supported_os and getattr(hardware, "os_name", "") not in spec.supported_os:
        return not_applicable(
            f"{spec.display} supports {', '.join(spec.supported_os)}; this host is "
            f"{getattr(hardware, 'os_name', 'unknown')}"
        )
    for module in spec.requires:
        ok, reason = module_importable(module)
        if not ok:
            return not_installed(reason)
    return available(package_version(spec.package))


class BackendAdapterMixin:
    """Defaults for the parts of the Backend protocol an engine may not expose.

    A stack that cannot return prompt logits, or cannot stream, says so through
    :class:`benchmark.backends.UnsupportedConfiguration`. The runner records that
    against the cell. What it must never do is silently substitute a different
    measurement under the same label, so there is no fallback here that changes
    what is being timed without changing what it is called.
    """

    spec: EngineSpec

    def prefill(self, prompt: str) -> Any:
        from benchmark.backends import UnsupportedConfiguration

        raise UnsupportedConfiguration(
            f"{self.spec.display} exposes no forward pass that returns prompt logits"
        )

    def serving_prefill(self, prompt: str) -> Any:
        from benchmark.backends import UnsupportedConfiguration

        raise UnsupportedConfiguration(
            f"{self.spec.display} exposes no last-position-only prefill path"
        )

    def first_token_latency(self, prompt: str, *, max_new_tokens: int, seed: int) -> float:
        """Time to the first token via a one-token generation call.

        Used by engines whose Python surface is not a token stream. Timing a
        one-token generation is a real time-to-first-token for such an API, but it
        is not the same machinery as a streaming decode, so the adapter declares
        ``ttft_method="single_token_call"`` and the report prints which method each
        row used.
        """
        import time

        from benchmark import metrics

        metrics.synchronize()
        start = time.perf_counter()
        self.generate(  # type: ignore[attr-defined]
            prompt, max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0,
            seed=seed, batch_size=1,
        )
        metrics.synchronize()
        return time.perf_counter() - start

    def supports_batch(self, batch_size: int) -> bool:
        return batch_size >= 1

    def unload(self) -> None:
        import gc

        for attribute in ("_model", "_engine", "_llm", "_runtime", "_session"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
