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
