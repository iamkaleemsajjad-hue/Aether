"""Hugging Face Transformers on PyTorch eager: the reference baseline.

This is the stack almost every published model is first run on, so it is the
suite's reference point for both throughput and correctness. It is *not* a
compiler: the checkpoint is interpreted by Python model classes on every forward
pass, and the only build-like step is loading weights.

The measured configuration is the existing two-backend harness's baseline,
unchanged and reused rather than reimplemented, so V2 numbers for this engine
remain comparable with the A/B report's numbers.
"""

from __future__ import annotations

from typing import Any

from benchmark.backend_transformers import TransformersBackend
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="transformers",
    display="HF Transformers (PyTorch eager)",
    taxonomy=(base.FRAMEWORK, base.RUNTIME),
    summary=(
        "Reference implementation: Python model classes executing eager PyTorch "
        "ops, the library's own KV cache, and model.generate. No compilation step."
    ),
    package="transformers",
    requires=("torch", "transformers"),
    has_build_phase=False,
    artifact_persistence=base.ARTIFACT_NONE,
    ttft_method="streaming",
    notes=(
        "Attention implementation is whatever the library selects for the device "
        "(SDPA, and FlashAttention within SDPA where the GPU supports it). Nothing "
        "is disabled to make the comparison easier and nothing exotic is enabled.",
    ),
)


class Engine(TransformersBackend):
    """The baseline, with the suite's descriptive fields attached."""

    spec = SPEC

    def __init__(self, device: str = "cuda", **_: Any) -> None:
        super().__init__(device=device)
        self.name = SPEC.key

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record.update(
            engine_key=SPEC.key,
            taxonomy=list(SPEC.taxonomy),
            execution_device=self.device,
            execution_device_class=base.device_class(self.device),
            threads=base.torch_thread_budget(),
            representation=(
                f"published checkpoint cast to {self._precision or '?'} tensors"
            ),
            weight_storage_bits=32 if self._precision == "fp32" else 16,
            weight_storage_format=self._precision,
            quantized=False,
            ttft_method=SPEC.ttft_method,
        )
        return record


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    return base.generic_probe(SPEC, hardware)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(device="cuda" if hardware.nvidia else "cpu")
