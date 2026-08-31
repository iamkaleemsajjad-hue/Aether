"""Aether Runtime: an AOT compiler plus the runtime that executes its artifact.

Aether is the system under test. It compiles a checkpoint once into a
self-contained ``.aeg`` artifact and then executes that artifact, so its cost
structure is genuinely different from a framework's: a one-time build, then
inference that never re-derives the graph. The suite times those phases apart and
never amortizes one into the other.

The adapter reuses the existing :class:`benchmark.backend_aether.AetherBackend`
verbatim, including its one configuration override (the semantic response cache
is off, so a repeated prompt measures inference rather than a cache hit).
"""

from __future__ import annotations

from typing import Any

from benchmark.backend_aether import AetherBackend
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="aether",
    display="Aether Runtime",
    taxonomy=(
        base.AOT_COMPILER, base.GRAPH_COMPILER, base.RUNTIME, base.EXECUTION_ENGINE,
    ),
    summary=(
        "Ahead-of-time compiler that lowers a checkpoint into a self-contained "
        ".aeg artifact, plus the runtime that executes that artifact. Compilation "
        "is a separate, one-time phase; the artifact is a file another process can "
        "load."
    ),
    package="aether-runtime",
    requires=("torch", "aether"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    ttft_method="streaming",
    notes=(
        "The compiler's default weight residency is BF16. At fp16 or fp32 the "
        "artifact therefore does not hold bit-identical values to the published "
        "checkpoint that the framework engines load, which is why bf16 is the "
        "primary comparison wherever the hardware supports it natively.",
        "The semantic response cache is disabled for every measured run through "
        "the public RuntimeConfig flag. It is on by default and returns a stored "
        "completion for a repeated prompt in about a millisecond, so leaving it on "
        "would time a lookup instead of inference. No other engine has an "
        "equivalent, and no other default is changed.",
    ),
)


class Engine(AetherBackend):
    """Aether, with the suite's descriptive fields attached."""

    spec = SPEC

    def __init__(self, device: str = "cuda", cache_dir: str | None = None,
                 execution_devices: list[str] | None = None, **_: Any) -> None:
        super().__init__(device=device, cache_dir=cache_dir, keep_artifact=True,
                         execution_devices=execution_devices)
        self.name = SPEC.key

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record.update(
            engine_key=SPEC.key,
            taxonomy=list(SPEC.taxonomy),
            # The AEG blob stores the checkpoint's bf16 values; the compute dtype is
            # whatever the run's precision is. Both halves are stated because the
            # comparability check keys off the compute dtype and discloses the storage.
            representation=(
                f"compiled AEG artifact, bf16 weight storage, {self._precision or '?'} "
                "compute"
            ),
            weight_storage_bits=16,
            weight_storage_format="bf16",
            quantized=False,
            ttft_method=SPEC.ttft_method,
        )
        return record

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate, and record how the reported token ids were obtained.

        ``Runtime.generate`` returns decoded text, so the ids the adapter reports are
        re-encoded from it. The correctness comparison needs to know that: a round trip
        can renumber tokens that decode to the identical string, and a difference from
        it is not a difference in the model.
        """
        outcome = super().generate(prompt, **kwargs)
        outcome.backend_metrics = {
            **(outcome.backend_metrics or {}),
            "token_ids_source": "re-encoded from decoded text",
        }
        return outcome


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    return base.generic_probe(SPEC, hardware)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    # Name the single device explicitly as well as restricting visibility in the
    # worker. Visibility alone is enough, but stating the execution device means the
    # artifact's placement is recorded in the result rather than inferred, and it
    # holds even if a future runtime learns to look past CUDA_VISIBLE_DEVICES.
    devices = getattr(options, "devices", None)
    execution_devices = (
        [f"cuda:{index}" for index in range(devices)]
        if hardware.nvidia and devices and devices >= 1 else None
    )
    return Engine(
        device="cuda" if hardware.nvidia else "cpu",
        cache_dir=getattr(options, "aeg_cache_dir", None),
        execution_devices=execution_devices,
    )
