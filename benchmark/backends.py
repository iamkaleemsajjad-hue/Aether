"""The contract both backends implement, so the runner cannot treat them unevenly.

The runner only ever calls methods declared here.  Anything a backend needs to do
differently — compiling an artifact, choosing an attention implementation — happens
behind this interface, and every such choice is reported in ``describe()`` so it
lands in the report instead of staying implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationOutcome:
    """The result of one generation call, as both backends must report it."""

    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    #: Backend-reported metrics, kept separate from anything the harness times
    #: itself so a backend cannot influence the official numbers.
    backend_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedBatchOutcome:
    """The result of one batched pass over prompts that differ in length.

    Separate from :class:`GenerationOutcome` because the interesting quantity is
    per row: a mixed batch's rows produce different token counts, and collapsing
    them into one number is what hides the cost of raggedness.
    """

    texts: list[str]
    row_prompt_tokens: list[int]
    row_completion_tokens: list[int]
    backend_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadOutcome:
    """What happened while bringing a model up, timed in distinct phases."""

    download_s: float | None
    prepare_s: float | None
    load_s: float
    total_s: float
    notes: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    """A benchmarkable inference backend."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Every configuration choice that could affect the measurement."""

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        """Bring the model to a state where generation can be requested."""

    def tokenizer(self) -> Any:
        """The tokenizer this backend will actually use for generation."""

    def prefill(self, prompt: str) -> Any:
        """Run a single forward pass over the prompt and return its last logits."""

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
        """Generate a completion under exactly the given settings."""

    def first_token_latency(self, prompt: str, *, max_new_tokens: int, seed: int) -> float:
        """Seconds until the first generated token is available to a caller."""

    def unload(self) -> None:
        """Release model memory so the next measurement starts from a clean state."""


class UnsupportedConfiguration(RuntimeError):
    """Raised when a backend genuinely cannot run a requested configuration.

    This is a reportable outcome, not a failure to work around: the runner
    records it against the cell and moves on, so the report can state plainly
    that a configuration is unavailable rather than quietly substituting one
    that is.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


TORCH_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def resolve_dtype(precision: str) -> Any:
    """Map a benchmark precision label onto a torch dtype."""
    import torch

    if precision not in TORCH_DTYPES:
        raise UnsupportedConfiguration(f"unknown precision {precision!r}")
    return getattr(torch, TORCH_DTYPES[precision])


def set_seed(seed: int) -> None:
    """Seed every generator that could affect sampling, on both backends."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stream_first_token_latency(model: Any, tokenizer: Any, inputs: Any, *,
                               max_new_tokens: int) -> float:
    """Seconds until the first generated token is decodable, through the streamer.

    One implementation, shared by every engine whose ``generate`` accepts a
    Transformers streamer, so that ``ttft_method="streaming"`` names the same
    machinery on each row it appears on instead of two copies that could drift.
    The caller seeds and encodes, because those steps are genuinely the engine's
    own; the timed region -- streamer, thread, first token out -- is not.

    Timing stops at the streamer's first emission, so whatever the generation was
    asked to produce after that cannot affect the figure. Nothing is caught and
    turned into a number: a generation that fails re-raises here, and one that
    returns no token beyond its prompt is reported as unmeasurable, so the runner
    records the cell rather than publishing a latency for a token that never came.
    """
    import threading
    import time

    from transformers import TextIteratorStreamer

    from benchmark import metrics

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=120.0)
    failure: list[BaseException] = []
    produced: list[Any] = []

    def worker() -> None:
        try:
            produced.append(model.generate(
                **inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
            ))
        except BaseException as exc:  # re-raised on the caller's thread, not swallowed
            failure.append(exc)
            streamer.end()

    metrics.synchronize()
    thread = threading.Thread(target=worker, daemon=True)
    start = time.perf_counter()
    thread.start()
    for _ in streamer:
        break
    elapsed = time.perf_counter() - start
    thread.join(timeout=300.0)
    if failure:
        raise failure[0]
    if not _generation_grew(produced, inputs):
        raise UnsupportedConfiguration(
            "the generation returned no token beyond the prompt, so there is no "
            "time-to-first-token to report"
        )
    return elapsed


def _generation_grew(produced: list[Any], inputs: Any) -> bool:
    """Whether the finished generation holds a token the prompt did not.

    The streamer signals the end of a stream with an emission of its own, so the
    first thing off the queue is not by itself proof that a token was generated.
    A generation still running when the join times out is not evidence of anything
    either way, and is left alone: its first token did arrive and was timed.
    """
    if not produced or produced[0] is None:
        return True
    prompt_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    try:
        return int(produced[0].shape[-1]) > int(prompt_ids.shape[-1])
    except (AttributeError, TypeError, IndexError):
        return True
