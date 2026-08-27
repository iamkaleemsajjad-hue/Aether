"""The Aether Runtime backend.

Aether is a compiler plus a runtime: a checkpoint is compiled once into a
self-contained ``.aeg`` artifact, and the runtime executes that artifact.  Those
are separate phases and are timed separately here — compilation is a one-time
cost that a deployment pays before serving, not part of inference.

Everything below uses Aether's public surface (``Compiler``, ``Runtime``) with
default settings.  The only configuration applied is the one a user would also
apply: the execution precision and the device mesh.  No internal behaviour is
altered for the benchmark.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from benchmark.prompts import flatten_ids
from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    set_seed,
)

#: Aether selects its compute dtype from this variable, read when the accelerator
#: engine is constructed.  It must therefore be set before ``Runtime.generate``
#: first loads the artifact.
DTYPE_ENV = "AETHER_TORCH_DTYPE"

#: Compiler target chosen from the live device rather than hard-coded, so the
#: artifact is built the way ``aether compile`` would build it on this host.
def _default_target() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            # Map compute capability to the nearest supported target using the
            # same tiered logic as hardware_detector._cuda_target_id, so that
            # e.g. sm_75 (Tesla T4) resolves to cuda_sm70 (Volta) rather than
            # the non-existent cuda_sm75.
            sm = major * 10 + minor
            if sm >= 130:
                return "cuda_sm130"
            if sm >= 120:
                return "cuda_sm120"
            if sm >= 100:
                return "cuda_sm100"
            if sm >= 90:
                return "cuda_sm90"
            if sm >= 89:
                return "cuda_sm89"
            if sm >= 80:
                return "cuda_sm80"
            if sm >= 70:
                return "cuda_sm70"
            return f"cuda_sm{major}{minor}"
    except Exception:  # noqa: BLE001
        pass
    return "cpu_avx512"


#: Aether's runtime enables a semantic *response* cache by default, which returns
#: a stored completion for a repeated prompt without running the model.  Measured
#: on this repository: with it on, a repeated prompt takes ~0.001 s instead of
#: ~15 s.  A benchmark issues the same prompt many times, so leaving it enabled
#: would compare a dictionary lookup against real generation.  It is therefore
#: disabled for every measured run, and the fact is reported.
#:
#: This is not a modification to Aether — it is a configuration flag on the
#: public ``RuntimeConfig``, set so that the thing being timed is inference.
BENCHMARK_RUNTIME_FLAGS: dict[str, Any] = {
    "enable_semantic_cache": False,
}


class AetherBackend:
    """Compiles a checkpoint to an ``.aeg`` and runs it through ``Runtime``."""

    name = "aether"

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: str | None = None,
        execution_devices: list[str] | None = None,
        keep_artifact: bool = True,
    ) -> None:
        self.device = device
        self.cache_dir = Path(cache_dir or "benchmark/results/aeg-cache")
        self.execution_devices = execution_devices
        self.keep_artifact = keep_artifact
        self._runtime: Any = None
        self._artifact: Path | None = None
        self._tokenizer: Any = None
        self._engine: Any = None
        self._precision: str | None = None
        self._model_id: str | None = None
        self._target: str | None = None
        self._compiled_fresh: bool | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": self.device,
            "precision": self._precision,
            "compiler_target": self._target,
            "execution_devices": self.execution_devices or "auto-detected",
            "artifact": str(self._artifact) if self._artifact else None,
            "artifact_reused": (None if self._compiled_fresh is None else not self._compiled_fresh),
            "engine": type(self._engine).__name__ if self._engine else None,
            "generation": "Runtime.generate over a compiled AEG (default settings)",
            "weight_source": "AEG blob; compiler default residency is BF16",
            "runtime_flags_overridden": dict(BENCHMARK_RUNTIME_FLAGS),
            "runtime_flags_reason": (
                "the semantic response cache is disabled so that repeated prompts "
                "measure inference rather than a cache hit"
            ),
            DTYPE_ENV: os.environ.get(DTYPE_ENV),
        }

    # ── Phases ──────────────────────────────────────────────────────────────

    def compile(self, model_id: str) -> tuple[Path, float, bool]:
        """Compile ``model_id`` to an artifact, reusing one if already present."""
        from aether.compiler.compiler import Compiler
        from aether.compiler.config import CompilerConfig

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.cache_dir / (model_id.replace("/", "--") + ".aeg")
        self._target = _default_target()
        if artifact.exists() and self.keep_artifact:
            return artifact, 0.0, False
        start = time.perf_counter()
        Compiler(CompilerConfig(targets=[self._target])).compile(
            model_id, output_path=artifact
        )
        return artifact, time.perf_counter() - start, True

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        """Compile if needed, then bring the runtime and engine up."""
        import torch

        from aether.runtime.config import RuntimeConfig
        from aether.runtime.runtime import Runtime

        if precision not in {"fp32", "fp16", "bf16"}:
            raise UnsupportedConfiguration(f"unknown precision {precision!r}")
        os.environ[DTYPE_ENV] = precision
        self._precision = precision
        self._model_id = model_id

        artifact, compile_s, fresh = self.compile(model_id)
        self._artifact = artifact
        self._compiled_fresh = fresh

        load_start = time.perf_counter()
        config_kwargs: dict[str, Any] = {"hf_offline": False}
        config_kwargs.update(BENCHMARK_RUNTIME_FLAGS)
        if self.execution_devices:
            config_kwargs["execution_devices"] = self.execution_devices
        self._runtime = Runtime(RuntimeConfig(**config_kwargs))
        # Runtime.generate loads lazily, so a one-token call is what actually
        # materializes weights on the device.  It is charged to load, not to
        # inference, and the warm-up phase follows it.
        self._runtime.generate(str(artifact), prompt="a", max_tokens=1, temperature=0.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_s = time.perf_counter() - load_start

        handle = self._runtime_handle()
        self._engine = getattr(handle, "engine", None)
        self._tokenizer = getattr(handle, "tokenizer", None)
        return LoadOutcome(
            download_s=None,
            prepare_s=compile_s,
            load_s=load_s,
            total_s=compile_s + load_s,
            notes={
                "compiled_this_run": fresh,
                "compiler_target": self._target,
                "engine": type(self._engine).__name__ if self._engine else None,
                "compute_dtype": str(getattr(self._engine, "compute_dtype", None)),
                "artifact_bytes": _tree_size(artifact),
                "context_length": getattr(self._engine, "max_positions", None),
            },
        )

    def _runtime_handle(self) -> Any:
        """Reach the loaded model handle the runtime is holding.

        Read-only introspection: it is how the benchmark reports which engine
        Aether selected, and how it reaches the same forward pass the
        Transformers side is measured on for the prefill and logit experiments.
        """
        for attribute in ("_loaded_backends", "_backends"):
            registry = getattr(self._runtime, attribute, None)
            if not isinstance(registry, dict):
                continue
            for backend in registry.values():
                for value in getattr(backend, "_models", {}).values():
                    if getattr(value, "engine", None) is not None:
                        return value
        return None

    # ── Measured operations ─────────────────────────────────────────────────

    def tokenizer(self) -> Any:
        return self._tokenizer

    def prefill(self, prompt: str) -> Any:
        """One forward pass over the prompt, at the engine level.

        Deliberately the same abstraction the Transformers side is measured at
        (``model(**inputs)``): a single forward, no sampling, no generation loop.
        """
        import numpy as np

        if self._engine is None:
            raise UnsupportedConfiguration("engine unavailable for prefill measurement")
        ids = np.asarray(
            flatten_ids(self._tokenizer(prompt, return_tensors="np")["input_ids"]),
            dtype=np.int64,
        )
        logits, _cache = self._engine.forward(ids)
        return _to_cpu_float(logits[-1])

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
            return self._generate_batched(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                batch_size=batch_size,
            )
        result = self._runtime.generate(
            str(self._artifact),
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        ids = (
            flatten_ids(self._tokenizer(result.text, add_special_tokens=False)["input_ids"])
            if self._tokenizer else []
        )
        # ``Runtime.generate`` reports counts in a ``usage`` mapping rather than as
        # attributes; the completion count is taken from there, not from the
        # re-encoded text, so a decode/encode round trip cannot change it.
        usage = dict(getattr(result, "usage", {}) or {})
        metrics = getattr(result, "metrics", None)
        return GenerationOutcome(
            text=result.text,
            token_ids=ids,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", len(ids))),
            backend_metrics=(
                metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics or {})
            ),
        )

    def _generate_batched(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        batch_size: int,
    ) -> GenerationOutcome:
        """Run ``batch_size`` copies of the prompt as one batched pass.

        The prompt is replicated because that is exactly what the Transformers
        backend does for its own batch cells (``tokenizer([prompt] * batch_size)``),
        so the two backends are measured on the same work.  Replication also makes
        every row the same length, so the batch carries no padding — the rows are
        genuinely independent sequences that merely happen to be identical.

        This goes through ``Runtime.generate_batch``, the same public surface the
        ``batch_size=1`` cell uses via ``Runtime.generate``, so both cells are
        measured through the full runtime stack rather than one at the engine level
        and one above it.  The runtime decodes the rows together in one KV tensor;
        if no executor can batch, it raises and the cell is recorded as unsupported
        rather than being quietly serialized into a loop.
        """
        try:
            responses = self._runtime.generate_batch(
                str(self._artifact),
                [prompt] * batch_size,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            # A backend that cannot batch says so; record that, do not approximate.
            raise UnsupportedConfiguration(str(exc)[:400]) from exc
        if len(responses) != batch_size:
            raise UnsupportedConfiguration(
                f"batched generation returned {len(responses)} rows for batch {batch_size}"
            )
        first = responses[0]
        metrics = getattr(first, "metrics", None)
        extra = dict(getattr(metrics, "extra", {}) or {}) if metrics is not None else {}
        usage = dict(getattr(first, "usage", {}) or {})
        ids = (
            flatten_ids(self._tokenizer(first.text, add_special_tokens=False)["input_ids"])
            if self._tokenizer else []
        )
        # The harness normalizes throughput by one row's tokens times the batch
        # width, matching how it treats the Transformers backend, so report the
        # first row here and carry the per-row counts as evidence.
        return GenerationOutcome(
            text=first.text,
            token_ids=ids,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", len(ids))),
            backend_metrics={
                **extra,
                "batch_size": batch_size,
                "returned_rows": len(responses),
                "row_completion_tokens": [
                    int(dict(getattr(item, "usage", {}) or {}).get("completion_tokens", 0))
                    for item in responses
                ],
                "execution": "batched AEG forward pass (single KV tensor)",
            },
        )

    def generate_token_ids(
        self, prompt: str, *, max_new_tokens: int, temperature: float = 0.0
    ) -> list[int]:
        """Exact generated ids, for the correctness comparison.

        ``Runtime.generate`` returns decoded text; comparing token ids requires
        the engine's own output, so the correctness mode reads it there.  The
        sampling settings are the same ones the runtime would apply.
        """
        import numpy as np

        ids = np.asarray(
            flatten_ids(self._tokenizer(prompt, return_tensors="np")["input_ids"]),
            dtype=np.int64,
        )
        eos = getattr(self._tokenizer, "eos_token_id", None)
        return [
            int(value)
            for value in self._engine.generate(
                ids, max_tokens=max_new_tokens, temperature=temperature,
                eos_token_id=eos,
            )
        ]

    def first_token_latency(self, prompt: str, *, max_new_tokens: int, seed: int) -> float:
        """Time until the runtime's stream yields its first non-empty chunk."""
        import torch

        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        stream = self._runtime.generate_stream(
            str(self._artifact), prompt=prompt, max_tokens=max_new_tokens, temperature=0.0
        )
        for chunk in stream:
            if chunk:
                break
        elapsed = time.perf_counter() - start
        for _ in stream:  # drain, so the next measurement starts clean
            pass
        return elapsed

    def supports_batch(self, batch_size: int) -> bool:
        """Whether this backend can execute ``batch_size`` rows as one real pass.

        Asked of the loaded backend rather than the raw engine, because a CPU-loaded
        AEG can be promoted onto the portable tensor executor for batched work — a
        capability the engine object alone does not report.
        """
        if batch_size == 1:
            return True
        handle = self._runtime_handle()
        for registry_name in ("_loaded_backends", "_backends"):
            registry = getattr(self._runtime, registry_name, None)
            if not isinstance(registry, dict):
                continue
            for backend in registry.values():
                probe = getattr(backend, "supports_batched_generation", None)
                if callable(probe) and handle is not None:
                    return bool(probe(handle.model_id, batch_size))
        supports = getattr(self._engine, "supports_batch", None)
        return bool(
            getattr(self._engine, "generate_batch", None) is not None
            and callable(supports)
            and supports(batch_size)
        )

    def unload(self) -> None:
        import gc

        import torch

        self._runtime = None
        self._engine = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _to_cpu_float(value: Any) -> Any:
    """Bring a backend's logit row to CPU float32 for comparison."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def _tree_size(path: Path) -> int:
    """Total bytes of an artifact, which may be a directory tree."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
