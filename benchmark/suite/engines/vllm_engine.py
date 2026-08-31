"""vLLM: an LLM serving engine with a paged KV cache and a continuous scheduler.

vLLM is not primarily a compiler; it is a serving system. Its throughput comes
from PagedAttention, a continuous batching scheduler, and CUDA-graph capture of
the decode step. That makes it the field's strongest opponent at large batch and a
different kind of system from the ones whose advantage is a build phase, so the
report classifies it accordingly rather than lumping it in with the compilers.

It is measured through the offline ``LLM`` API - the same engine and scheduler the
server uses, with the HTTP layer removed so no network time is attributed to
inference.
"""

from __future__ import annotations

import os
import time
from typing import Any

from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    set_seed,
)
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="vllm",
    display="vLLM",
    taxonomy=(base.SERVING_ENGINE, base.RUNTIME, base.EXECUTION_ENGINE),
    summary=(
        "Optimized LLM serving engine: paged KV cache, continuous batching "
        "scheduler, fused kernels, and CUDA-graph capture of the decode step. "
        "Measured through its offline LLM API so no HTTP time is included."
    ),
    package="vllm",
    requires=("torch", "vllm"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_DISK_CACHE,
    requires_cuda=True,
    ttft_method="single_token_call",
    notes=(
        "Engine start-up includes CUDA-graph capture and, on recent versions, a "
        "torch.compile pass whose output is cached under the vLLM cache directory. "
        "That start-up cost is timed as this engine's build phase; it is a "
        "machine-local cache, not a portable artifact.",
        "vLLM pre-allocates a large fraction of device memory for its KV pool by "
        "design, so its peak-memory row reflects a reservation policy rather than "
        "the working set of one request. gpu_memory_utilization is recorded with "
        "the result.",
        "Prompts are submitted as one batch of independent requests, which is what "
        "the scheduler is built for; the batch-1 rows submit exactly one.",
    ),
)

#: Fraction of device memory vLLM may reserve for weights plus its KV pool.
#: Deliberately below 1.0 so a second engine's leftovers or another tenant on a
#: shared host cannot turn this into an allocation failure. Recorded in the result.
GPU_MEMORY_UTILIZATION = 0.85


_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class Engine(base.BackendAdapterMixin):
    """Drive vLLM's offline engine under the suite's fixed generation settings."""

    spec = SPEC
    name = SPEC.key

    def __init__(
        self,
        device: str = "cuda",
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        **_: Any,
    ) -> None:
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self._llm: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._startup_s: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "enforce_eager": self.enforce_eager,
            "engine_startup_s": self._startup_s,
            "generation": "vllm.LLM.generate with SamplingParams (offline engine)",
            "representation": (
                f"published checkpoint cast to {self._precision or '?'} tensors"
            ),
            "weight_storage_bits": 32 if self._precision == "fp32" else 16,
            "weight_storage_format": self._precision,
            "quantized": False,
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("vllm"),
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from vllm import LLM

        if precision not in _DTYPE:
            raise UnsupportedConfiguration(f"unknown precision {precision!r}")
        self._precision = precision
        # vLLM logs at INFO by default and the volume distorts a terminal summary;
        # quieting it does not change execution.
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        # vLLM's V1 engine runs its core in a child process, which means its real
        # traceback lands in that child's stderr and the parent only sees "engine core
        # initialization failed". Keeping the core in-process makes the actual cause
        # (an unsupported dtype, a missing runner, an allocation failure) reach the
        # record, which is the difference between a diagnosable result and a dead end.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": model_id,
                "dtype": _DTYPE[precision],
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": False,
                "seed": 0,
                # One device, like every other engine in the run. Visibility is already
                # restricted by the worker; stating it keeps the record explicit.
                "tensor_parallel_size": 1,
            }
            if self.max_model_len:
                kwargs["max_model_len"] = self.max_model_len
            if self.enforce_eager:
                kwargs["enforce_eager"] = True
            self._llm = LLM(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"vLLM could not start for {model_id}: {type(exc).__name__}: {exc}"
            ) from exc
        self._startup_s = time.perf_counter() - start
        self._tokenizer = self._llm.get_tokenizer()
        return LoadOutcome(
            download_s=None,
            # Engine start-up subsumes weight loading, KV-pool sizing and graph
            # capture. vLLM does not separate them, so the whole cost is reported
            # under the build column and labelled, rather than split by guesswork.
            prepare_s=None,
            load_s=self._startup_s,
            total_s=self._startup_s,
            notes={
                "engine_startup_s": self._startup_s,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "vllm_version": base.package_version("vllm"),
                "startup_includes": "weight load, KV pool allocation, graph capture",
                "tensor_parallel_size": 1,
            },
        )

    def tokenizer(self) -> Any:
        return self._tokenizer

    def _params(self, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature,
            top_p=top_p if temperature > 0.0 else 1.0,
            top_k=top_k if (temperature > 0.0 and top_k > 0) else -1,
            max_tokens=max_new_tokens,
            min_tokens=max_new_tokens,
            # Every engine in the suite is asked for exactly max_new_tokens of work,
            # so an early stop cannot make one row look faster by doing less.
            ignore_eos=True,
            seed=None,
        )

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
        params = self._params(max_new_tokens, temperature, top_p, top_k)
        outputs = self._llm.generate([prompt] * batch_size, params, use_tqdm=False)
        if len(outputs) != batch_size:
            raise UnsupportedConfiguration(
                f"vLLM returned {len(outputs)} rows for batch {batch_size}"
            )
        first = outputs[0].outputs[0]
        ids = [int(value) for value in first.token_ids]
        return GenerationOutcome(
            text=first.text,
            token_ids=ids,
            prompt_tokens=len(outputs[0].prompt_token_ids or []),
            completion_tokens=len(ids),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": len(outputs),
                "engine": "vllm",
                "row_completion_tokens": [
                    len(item.outputs[0].token_ids) for item in outputs
                ],
                "finish_reasons": sorted({item.outputs[0].finish_reason for item in outputs}),
            },
        )

    def generate_mixed(self, prompts: list[str], *, max_new_tokens: int,
                       temperature: float, top_p: float, top_k: int) -> Any:
        """One scheduled pass over prompts of differing length.

        This is the case vLLM's scheduler exists for: rows are not padded to a
        common width, they are packed into pages, so the pad overhead a padded
        batch pays is not paid here. The report notes that difference next to the
        numbers rather than pretending the two mechanisms are the same.
        """
        from benchmark.backends import MixedBatchOutcome

        params = self._params(max_new_tokens, temperature, top_p, top_k)
        outputs = self._llm.generate(list(prompts), params, use_tqdm=False)
        return MixedBatchOutcome(
            texts=[item.outputs[0].text for item in outputs],
            row_prompt_tokens=[len(item.prompt_token_ids or []) for item in outputs],
            row_completion_tokens=[len(item.outputs[0].token_ids) for item in outputs],
            backend_metrics={"engine": "vllm", "scheduling": "paged, no row padding"},
        )

    def unload(self) -> None:
        self._llm = None
        self._tokenizer = None
        super().unload()


#: Architectures vLLM does not implement. Checked before a run rather than
#: discovered from a pydantic validation error 25 seconds in, so the compatibility
#: table can say *why* instead of quoting a stack trace. Kept short and specific:
#: this is a list of known gaps, not a guess at vLLM's whole coverage, and anything
#: not listed here is attempted for real.
UNSUPPORTED_ARCHITECTURES: dict[str, str] = {
    "GPTNeoForCausalLM": (
        "vLLM implements GPT-NeoX and GPT-J but not the original GPT-Neo "
        "architecture, so it has no model runner for this checkpoint"
    ),
    "GPT2LMHeadModel": (
        "vLLM's GPT-2 support is limited and not present in every release; this "
        "build reports no runner for it"
    ),
}

#: bf16 needs Ampere. vLLM refuses it below compute capability 8.0 rather than
#: emulating, so on an older card the benchmark precision decides whether this
#: engine can be measured at all.
BF16_MIN_CAPABILITY = (8, 0)


def _architecture(model_id: str) -> str | None:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        return (getattr(config, "architectures", None) or [None])[0]
    except Exception:  # noqa: BLE001 - an unreadable config is not this engine's fault
        return None


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    from benchmark.suite import hardware as hardware_mod

    if precision == "bf16" and not hardware_mod.meets_capability(
        hardware, BF16_MIN_CAPABILITY
    ):
        capabilities = ", ".join(hardware.compute_capabilities) or "unknown"
        return base.not_supported(
            f"vLLM requires compute capability 8.0 or newer for bf16; this host is "
            f"{capabilities}. Re-run with --precision fp16 (or let --precision auto "
            "choose it) and this engine becomes measurable."
        )
    architecture = _architecture(model_id)
    if architecture in UNSUPPORTED_ARCHITECTURES:
        return base.not_supported(
            f"{architecture}: {UNSUPPORTED_ARCHITECTURES[architecture]}"
        )
    return base.available(base.package_version("vllm"))


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(
        device="cuda",
        gpu_memory_utilization=getattr(options, "vllm_gpu_utilization", None)
        or GPU_MEMORY_UTILIZATION,
        max_model_len=getattr(options, "vllm_max_model_len", None),
        enforce_eager=bool(getattr(options, "vllm_enforce_eager", False)),
    )
