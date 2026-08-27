"""Batched generation over a compiled AEG, shared by every AEG-executing backend.

Both the native CPU backend and the PyTorch backend load an AEG into a
:class:`~aether.backends.compiled_handle.CompiledAEGHandle` and then execute it.
Batched execution is identical work in both cases — pack the prompts, run one
batched pass, split the rows back out — so it lives here once rather than being
reimplemented per backend, where the two copies would drift.

Two rules this module exists to enforce:

* **A batch is one pass, or it is refused.** If no available executor carries a
  batch axis, :func:`generate_batch` raises. It never loops over the requests and
  returns the result labelled as a batch, because that would attribute serial work
  to batched throughput — the one measurement error batching invites.
* **Promotion is explicit and paid once.** A CPU-loaded AEG sits on the NumPy
  reference executor, whose kernels are sequence-major. The same authenticated
  weights are promoted onto the portable tensor executor to serve a batch. That
  materializes the weights a second time, so it happens lazily, at most once per
  handle, and only when a batch is actually requested.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from aether.backends.base import GenerationRequest, GenerationResult
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.core.exceptions import BackendError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

#: Engine class that holds authenticated weights but cannot batch, and can be
#: promoted onto the portable tensor executor.  Matched by name so this module
#: needs no import of either engine.
_PROMOTABLE_ENGINE = "CPUExecutionEngine"


def engine_can_batch(engine: Any, batch_size: int) -> bool:
    """Whether ``engine`` advertises real batched execution at this width."""
    supports = getattr(engine, "supports_batch", None)
    return bool(
        engine is not None
        and getattr(engine, "generate_batch", None) is not None
        and callable(supports)
        and supports(batch_size)
    )


def can_batch(handle: CompiledAEGHandle | None, batch_size: int = 2) -> bool:
    """Whether ``handle`` could serve a batch of this width as one pass.

    A capability probe: it loads nothing and promotes nothing, so it is safe to
    call before deciding whether to build a batch at all.
    """
    if not isinstance(handle, CompiledAEGHandle):
        return False
    if batch_size <= 1:
        return True
    if engine_can_batch(handle.engine, batch_size):
        return True
    if engine_can_batch(handle.batched_engine, batch_size):
        return True
    if type(handle.engine).__name__ != _PROMOTABLE_ENGINE:
        return False
    try:  # promotion needs the portable tensor executor
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def batch_capable_engine(
    handle: CompiledAEGHandle, batch_size: int, *, backend_name: str
) -> Any:
    """Return an executor for ``handle`` that runs ``batch_size`` rows in one pass.

    On an accelerator the AEG has already loaded onto the portable tensor executor,
    which carries a batch axis, and it is returned unchanged. On a CPU host the AEG
    is on the NumPy reference executor; the same weights are promoted onto the
    portable tensor executor at ``device="cpu"``.

    Raises :class:`BackendError` when neither applies. Refusing is deliberate: see
    the module docstring.
    """
    engine = handle.engine
    if engine is None:
        raise BackendError(
            f"AEG {handle.aeg_path} has no executable engine", backend_name=backend_name
        )
    if engine_can_batch(engine, batch_size):
        return engine
    if engine_can_batch(handle.batched_engine, batch_size):
        return handle.batched_engine
    if type(engine).__name__ != _PROMOTABLE_ENGINE:
        raise BackendError(
            f"the loaded execution engine ({type(engine).__name__}) is "
            f"single-sequence, so a batch of {batch_size} cannot be executed as one "
            "pass. Submit the requests individually; they are deliberately not "
            "looped here, because serial work reported as a batch would misstate "
            "batched throughput.",
            backend_name=backend_name,
        )
    try:
        from aether.runtime.torch_engine import TorchAEGEngine
    except Exception as exc:  # noqa: BLE001 - any import failure means no promotion
        raise BackendError(
            "batched inference on a CPU host needs the portable tensor executor, "
            "which requires PyTorch (install the [pytorch] extra). Batching is "
            "refused rather than emulated by a sequential loop.",
            backend_name=backend_name,
        ) from exc
    logger.info(
        "Promoting the NumPy reference executor onto the portable tensor executor "
        "(device=cpu) to serve a batch of %d; this materializes the weights once "
        "more as tensors and is done once per loaded model.",
        batch_size,
    )
    promoted = TorchAEGEngine(engine, "cpu")
    if not engine_can_batch(promoted, batch_size):
        raise BackendError(
            f"the portable tensor executor declined a batch of {batch_size}; "
            "refusing to serialize it",
            backend_name=backend_name,
        )
    handle.batched_engine = promoted
    return promoted


def generate_batch(
    handle: CompiledAEGHandle,
    requests: list[GenerationRequest],
    *,
    backend_name: str,
    request_text: Any,
    truncate_stop_text: Any,
    default_device: str = "cpu",
) -> list[GenerationResult]:
    """Serve ``requests`` in one batched forward pass over ``handle``.

    ``request_text`` and ``truncate_stop_text`` are the calling backend's own
    prompt-rendering and stop-sequence helpers, passed in so this module imposes no
    opinion on either.

    A batch shares one set of weights, one KV tensor and one decode loop, so the
    requests must agree on what shapes those: the model, and the sampling settings
    (temperature, top-p, top-k are applied to the whole logits tensor, and per-row
    settings would need per-row masking).

    ``max_tokens`` may differ. The decode loop runs to the longest budget and each
    row is truncated to its own — correct because rows are independent, so
    over-running one cannot affect another.
    """
    if not requests:
        return []
    if handle.tokenizer is None:
        raise BackendError(
            "batched generation requires the AEG's packaged tokenizer",
            backend_name=backend_name,
        )

    settings = {
        (float(request.temperature), float(request.top_p), int(request.top_k))
        for request in requests
    }
    if len(settings) != 1:
        raise BackendError(
            "every request in a batch must share temperature, top_p and top_k; "
            "per-row sampling settings are not supported",
            backend_name=backend_name,
        )
    temperature, top_p, top_k = next(iter(settings))

    engine = batch_capable_engine(handle, len(requests), backend_name=backend_name)

    prompts: list[np.ndarray] = []
    for request in requests:
        text = request_text(request, handle.tokenizer)
        encoded = handle.tokenizer(text, return_tensors="np")
        prompts.append(np.asarray(encoded["input_ids"][0], dtype=np.int64))

    horizon = max(int(request.max_tokens) for request in requests)
    start = time.perf_counter()
    rows = engine.generate_batch(
        prompts,
        max_tokens=horizon,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(handle.tokenizer, "eos_token_id", None),
    )
    elapsed = time.perf_counter() - start
    if len(rows) != len(requests):
        raise BackendError(
            f"batched generation returned {len(rows)} rows for {len(requests)} "
            "requests",
            backend_name=backend_name,
        )

    produced = sum(len(row) for row in rows)
    results: list[GenerationResult] = []
    for request, prompt_ids, generated in zip(requests, prompts, rows, strict=True):
        generated = [int(value) for value in generated][: int(request.max_tokens)]
        text = handle.tokenizer.decode(generated, skip_special_tokens=True)
        completion_tokens = len(generated)
        finish_reason = "length" if completion_tokens >= int(request.max_tokens) else "stop"
        if request.stop:
            text, completion_tokens, stopped = truncate_stop_text(
                handle.tokenizer, generated, request.stop
            )
            if stopped:
                finish_reason = "stop"
        results.append(
            GenerationResult(
                text=text,
                prompt_tokens=int(prompt_ids.size),
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                backend_name=backend_name,
                metrics={
                    # The wall time belongs to the batch, so a per-row rate would
                    # understate what the pass achieved and the aggregate would
                    # overstate what one caller saw.  Both are reported, labelled.
                    "batch_size": len(requests),
                    "batch_latency_s": elapsed,
                    "batch_throughput_tps": produced / max(elapsed, 1e-9),
                    "row_throughput_tps": completion_tokens / max(elapsed, 1e-9),
                    "engine": type(engine).__name__,
                    "device": str(getattr(engine, "device", default_device)),
                },
            )
        )
    return results
