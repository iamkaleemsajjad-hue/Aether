"""The measurement loop: warm-up, alternated repetitions, and per-cell recording.

Three properties this module is responsible for:

* **order alternation.** Which backend runs first is flipped on every repetition,
  so a thermal drift or a clock ramp during the run cannot land preferentially on
  one of them.
* **phase separation.** Download, compile, load, warm-up and steady-state are
  distinct measurements. Warm-up iterations are executed and discarded.
* **failure as data.** An unsupported configuration or a runtime error is
  recorded against the cell; it never silently changes what was measured.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from benchmark import gpu_monitor, metrics
from benchmark.backends import UnsupportedConfiguration
from benchmark.memory_monitor import HostMemoryMonitor, snapshot as host_snapshot


def _failure(exc: BaseException) -> dict[str, Any]:
    """Record a failure with enough detail to diagnose it after the fact."""
    kind = "unsupported" if isinstance(exc, UnsupportedConfiguration) else "error"
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        kind = "oom"
    return {
        "status": kind,
        "exception": type(exc).__name__,
        "message": str(exc)[:600],
        "traceback": None if kind == "unsupported" else traceback.format_exc(limit=8)[-1600:],
    }


def measure_load(backend: Any, model_id: str, precision: str) -> dict[str, Any]:
    """Time bringing a model up, with host and device memory around it."""
    gpu_monitor.empty_cache()
    gpu_monitor.reset_peak_stats()
    before_gpu = gpu_monitor.memory_snapshot()
    before_host = host_snapshot()
    try:
        with HostMemoryMonitor(0.05) as host:
            outcome = backend.load(model_id, precision)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised by caller
        return {**_failure(exc), "phase": "load"}
    metrics.synchronize()
    return {
        "status": "ok",
        "download_s": outcome.download_s,
        "prepare_s": outcome.prepare_s,
        "load_s": outcome.load_s,
        "total_s": outcome.total_s,
        "notes": outcome.notes,
        "gpu_before": before_gpu,
        "gpu_after": gpu_monitor.memory_snapshot(),
        "host_before": before_host,
        "host_after": host_snapshot(),
        "host_during": host.report(),
    }


def _run_generation(
    backend: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    batch_size: int,
) -> Any:
    return backend.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        batch_size=batch_size,
    )


def measure_generation(
    backend: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    gpu_interval_s: float,
) -> dict[str, Any]:
    """Warm up, then time ``measure_iters`` steady-state generations.

    The first (cold) iteration is reported separately from the steady state
    rather than averaged into it: a first call carries autotuning and allocator
    growth that a served request would not.

    Device telemetry is gathered in one extra iteration *after* the timed ones,
    so the sampler's host cost cannot contaminate the reported latencies.
    """
    call: Callable[..., Any] = lambda: _run_generation(
        backend, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k, seed=seed, batch_size=batch_size,
    )

    cold: list[float] = []
    try:
        with metrics.timed(cold):
            first = call()
    except BaseException as exc:  # noqa: BLE001
        return {**_failure(exc), "phase": "generate"}

    for _ in range(max(0, warmup_iters)):
        try:
            call()
        except BaseException as exc:  # noqa: BLE001
            return {**_failure(exc), "phase": "warmup"}

    # Both backends stay resident in one process so that execution order can be
    # alternated per cell, which protects the headline throughput numbers from
    # thermal drift.  The consequence is that the allocator's *absolute* peak is
    # process-wide and includes the other backend's weights.  Recording the
    # baseline here makes the per-backend inference footprint (KV cache plus
    # activations) recoverable as a delta, which is the figure that is actually
    # attributable to this backend.
    gpu_monitor.reset_peak_stats()
    baseline_gpu = gpu_monitor.memory_snapshot()
    latencies: list[float] = []
    completions: list[int] = []
    outcome = first
    for _ in range(max(1, measure_iters)):
        try:
            with metrics.timed(latencies):
                outcome = call()
        except BaseException as exc:  # noqa: BLE001
            return {**_failure(exc), "phase": "measure",
                    "completed_iterations": len(latencies)}
        completions.append(outcome.completion_tokens)

    peak_gpu = gpu_monitor.memory_snapshot()

    telemetry: dict[str, Any] = {"sampled": False}
    host_report: dict[str, Any] = {"sampled": False}
    try:
        with gpu_monitor.GPUTelemetryMonitor(gpu_interval_s) as gpu, \
                HostMemoryMonitor(gpu_interval_s) as host:
            call()
        telemetry = gpu.report()
        host_report = host.report()
    except BaseException:  # noqa: BLE001 - telemetry is never fatal
        pass

    generated = completions[0] if completions else 0
    consistent = len(set(completions)) <= 1
    return {
        "status": "ok",
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": generated,
        "total_tokens": outcome.prompt_tokens + generated,
        "completion_tokens_consistent": consistent,
        "completion_tokens_observed": sorted(set(completions)),
        "cold_latency_s": cold[0] if cold else None,
        "latency_s": metrics.summarize(latencies),
        "tokens_per_s": metrics.summarize(
            metrics.throughput_samples(latencies, generated * max(batch_size, 1))
        ),
        "latency_per_token_ms": metrics.summarize(
            [value / max(generated, 1) * 1000.0 for value in latencies]
        ),
        "gpu_peak": peak_gpu,
        "gpu_baseline_at_reset": baseline_gpu,
        "gpu_inference_delta_bytes": _inference_delta(baseline_gpu, peak_gpu),
        "gpu_telemetry": telemetry,
        "host_during_inference": host_report,
        "sample_text": outcome.text[:400],
        "backend_metrics": outcome.backend_metrics,
    }


def _inference_delta(baseline: dict[str, Any], peak: dict[str, Any]) -> int | None:
    """Memory this backend's inference added beyond what was already resident.

    The allocator's counters are process-wide, so the absolute peak is not
    attributable to one backend when two are loaded.  The difference from the
    baseline taken immediately after resetting the peak counters is.
    """
    if not baseline.get("available") or not peak.get("available"):
        return None
    before = sum(d["allocated_bytes"] for d in baseline["devices"])
    after = sum(d["peak_allocated_bytes"] for d in peak["devices"])
    return max(0, after - before)

def measure_mixed_batch(
    backend: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    """Time one batched pass over prompts that genuinely differ in length.

    Distinct from :func:`measure_generation`, which replicates a single prompt to the
    batch width. Here every row is a different prompt, so the batch carries real
    padding and the measurement includes what padding costs.

    Throughput is the aggregate: total tokens the pass produced over its wall time.
    Every row shares that wall time, so the latency percentiles describe the pass,
    not individual rows — a per-row percentile would be the same number repeated.
    """
    from benchmark.backends import set_seed

    call = getattr(backend, "generate_mixed", None)
    if call is None:
        return {
            "status": "unsupported",
            "phase": "mixed_batch",
            "message": (
                f"{type(backend).__name__} cannot run a batch of differing prompts; "
                "reported rather than serialized into a loop"
            ),
        }

    def once() -> Any:
        set_seed(seed)
        return call(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    try:
        outcome = once()
    except BaseException as exc:  # noqa: BLE001
        return {**_failure(exc), "phase": "mixed_batch"}
    for _ in range(max(0, warmup_iters)):
        try:
            once()
        except BaseException as exc:  # noqa: BLE001
            return {**_failure(exc), "phase": "mixed_warmup"}

    gpu_monitor.reset_peak_stats()
    baseline_gpu = gpu_monitor.memory_snapshot()
    latencies: list[float] = []
    for _ in range(max(1, measure_iters)):
        try:
            with metrics.timed(latencies):
                outcome = once()
        except BaseException as exc:  # noqa: BLE001
            return {**_failure(exc), "phase": "mixed_measure",
                    "completed_iterations": len(latencies)}
    peak_gpu = gpu_monitor.memory_snapshot()

    produced = sum(outcome.row_completion_tokens)
    return {
        "status": "ok",
        "rows": len(outcome.row_completion_tokens),
        "row_completion_tokens": list(outcome.row_completion_tokens),
        "total_completion_tokens": produced,
        "latency_s": metrics.summarize(latencies),
        "tokens_per_s": metrics.summarize(
            metrics.throughput_samples(latencies, produced)
        ),
        "requests_per_s": metrics.summarize(
            [len(outcome.row_completion_tokens) / value for value in latencies if value > 0]
        ),
        "gpu_peak": peak_gpu,
        "gpu_inference_delta_bytes": _inference_delta(baseline_gpu, peak_gpu),
    }


def measure_prefill(
    backend: Any, prompt: str, *, warmup_iters: int, measure_iters: int
) -> dict[str, Any]:
    """Time a single forward pass over the prompt: the prefill phase alone.

    Both backends are measured at the same abstraction — one forward call, no
    sampling and no generation loop — so the number isolates prompt processing
    from decoding.
    """
    try:
        for _ in range(max(1, warmup_iters)):
            backend.prefill(prompt)
        latencies: list[float] = []
        for _ in range(max(1, measure_iters)):
            with metrics.timed(latencies):
                backend.prefill(prompt)
    except BaseException as exc:  # noqa: BLE001
        return {**_failure(exc), "phase": "prefill"}
    return {"status": "ok", "latency_s": metrics.summarize(latencies)}


def measure_serving_prefill(
    backend: Any, prompt: str, *, warmup_iters: int, measure_iters: int
) -> dict[str, Any]:
    """Time prefill as a served request pays for it: final-position logits only.

    ``measure_prefill`` asks for logits at every prompt position. That is a fair
    like-for-like comparison of that operation, but it is not the work generation
    does: a served request reads one row of logits, and both runtimes can skip the
    rest. Measuring both configurations keeps the comparison honest in either
    direction — neither backend is credited with a shortcut the other was denied.

    A backend without a last-position path reports ``unsupported`` rather than
    falling back to the full-logits measurement under this label.
    """
    call = getattr(backend, "serving_prefill", None)
    if call is None:
        return {
            "status": "unsupported",
            "phase": "serving_prefill",
            "message": f"{type(backend).__name__} exposes no last-position prefill path",
        }
    try:
        for _ in range(max(1, warmup_iters)):
            call(prompt)
        latencies: list[float] = []
        for _ in range(max(1, measure_iters)):
            with metrics.timed(latencies):
                call(prompt)
    except BaseException as exc:  # noqa: BLE001
        return {**_failure(exc), "phase": "serving_prefill"}
    return {"status": "ok", "latency_s": metrics.summarize(latencies)}


def measure_ttft(
    backend: Any, prompt: str, *, max_new_tokens: int, seed: int, measure_iters: int
) -> dict[str, Any]:
    """Time until the first token reaches the caller through each streaming API.

    Streaming machinery differs between the two libraries, so this is reported as
    its own experiment rather than folded into the throughput figure.
    """
    try:
        backend.first_token_latency(prompt, max_new_tokens=max_new_tokens, seed=seed)
        samples = [
            backend.first_token_latency(prompt, max_new_tokens=max_new_tokens, seed=seed)
            for _ in range(max(1, measure_iters))
        ]
    except BaseException as exc:  # noqa: BLE001
        return {**_failure(exc), "phase": "ttft"}
    return {"status": "ok", "ttft_s": metrics.summarize(samples)}


def alternated_order(repetition: int, names: tuple[str, str]) -> tuple[str, str]:
    """Flip which backend goes first on odd repetitions."""
    return names if repetition % 2 == 0 else (names[1], names[0])


def cooldown(seconds: float) -> None:
    """Idle so the device can shed heat between sections, if asked to."""
    if seconds > 0:
        metrics.synchronize()
        time.sleep(seconds)
