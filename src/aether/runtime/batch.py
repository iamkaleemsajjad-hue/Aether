"""The batch execution model shared by Aether's portable executors.

Aether's decoder executors are sequence-major: one sequence, one monotonic run of
positions, one KV cache per layer.  Serving several *independent* sequences in one
forward pass needs an explicit, testable description of how those sequences share
a tensor, because every correctness hazard in batched autoregressive decode comes
out of that arrangement:

* a sequence must never attend to another sequence's tokens;
* a sequence's positions must not shift because a *different* sequence in the
  batch happens to be longer;
* a sequence that emits its stop token must not truncate the others.

This module owns that description and nothing else.  It imports no tensor
framework, so the same layout drives the PyTorch executor, can drive the NumPy
executor, and is unit-testable on its own.

Left padding
------------

Sequences are right-aligned inside the padded window::

    prompt A (5 tokens)   .  .  .  a  a  a  a  a
    prompt B (8 tokens)   b  b  b  b  b  b  b  b
    prompt C (2 tokens)   .  .  .  .  .  .  c  c
                          ^ pad                ^ every row's last real token

Right alignment is what makes decode uniform: after prefill every row's next
token lands at the *same* padded index, so one write index serves the whole batch
and a decode step is a single slice assignment rather than a per-row scatter.
Right-padding would leave each row's frontier at a different index and force that
scatter, plus a gather to find each row's final logits.

The cost is that pad slots occupy KV rows and pass through the prefill GEMMs.
That cost is bounded by the spread of prompt lengths and is zero when the rows are
equal-length.

Positions are per row
---------------------

A padded slot contributes no position.  Row ``b``'s first *real* token sits at
position 0 however much padding precedes it::

    position(b, i) = max(0, i - pad_count(b))

This is the invariant that makes a batched result *equal* the same sequence
decoded alone.  Using the padded index as the position would shift every rotary
angle — and every learned absolute position embedding, which is how GPT-Neo is
positioned — by that row's pad count.  It is a silent, model-wide numerical error
that no shape check would catch, which is why the layout computes positions here
rather than leaving them to each call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Token id written into padded slots.  Any in-vocabulary id is sound because
#: pad slots are excluded from attention by the ``live`` mask and their own
#: outputs are never read, but it must be in range or the executor's vocabulary
#: validation would reject the batch.
DEFAULT_PAD_TOKEN_ID = 0


@dataclass(frozen=True)
class BatchLayout:
    """How a set of variable-length sequences is arranged into one padded batch.

    Immutable, and carries the row lengths rather than only the padded width, so
    every derived quantity (pad counts, positions, the validity mask) has exactly
    one definition.
    """

    lengths: tuple[int, ...]
    """True token count of each row, in batch order."""

    padded_length: int
    """Width of the padded window: ``max(lengths)``."""

    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch."""
        return len(self.lengths)

    @property
    def pad_counts(self) -> tuple[int, ...]:
        """Leading pad slots per row.  ``0`` for the longest row(s)."""
        return tuple(self.padded_length - length for length in self.lengths)

    @property
    def is_uniform(self) -> bool:
        """Whether every row has the same length, so the batch carries no padding.

        The executor uses this to keep the fused attention path: with no padding
        the validity mask is all-true, so prefill stays ``is_causal=True`` and
        decode needs no mask at all.
        """
        return len(set(self.lengths)) <= 1

    @property
    def total_real_tokens(self) -> int:
        """Tokens that are real, excluding every pad slot."""
        return sum(self.lengths)

    @property
    def padding_overhead(self) -> float:
        """Fraction of padded slots that are pad.  ``0.0`` for a uniform batch."""
        slots = self.batch_size * self.padded_length
        return 0.0 if slots == 0 else 1.0 - (self.total_real_tokens / slots)

    def decode_positions(self, step: int) -> np.ndarray:
        """Per-row positions for decode ``step`` (0-based, after prefill).

        Decode step ``t`` writes padded index ``padded_length + t``, so row ``b``
        is at position ``padded_length + t - pad_count(b)``.
        """
        if step < 0:
            raise ValueError("decode step must be non-negative")
        index = self.padded_length + int(step)
        return np.asarray(
            [index - pad for pad in self.pad_counts], dtype=np.int64
        ).reshape(self.batch_size, 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the layout for logging and report payloads."""
        return {
            "batch_size": self.batch_size,
            "padded_length": self.padded_length,
            "lengths": list(self.lengths),
            "pad_counts": list(self.pad_counts),
            "uniform": self.is_uniform,
            "padding_overhead": self.padding_overhead,
        }


@dataclass(frozen=True)
class PackedBatch:
    """A left-padded batch of token ids with everything derived from its layout."""

    token_ids: np.ndarray
    """``(batch, padded_length)`` int64 ids; pad slots hold the pad token."""

    live: np.ndarray
    """``(batch, padded_length)`` bool; ``True`` where the slot is a real token."""

    position_ids: np.ndarray
    """``(batch, padded_length)`` int64 positions; pad slots hold 0."""

    layout: BatchLayout

    @property
    def batch_size(self) -> int:
        """Number of sequences in this batch."""
        return self.layout.batch_size

    @property
    def padded_length(self) -> int:
        """Width of the padded window."""
        return self.layout.padded_length


def normalize_sequences(sequences: Any) -> list[np.ndarray]:
    """Coerce a batch of prompts into a list of rank-1 int64 id arrays.

    Accepts a list of sequences, or a rank-2 array where each row is a sequence.
    A rank-1 array is treated as a single sequence rather than as a batch of
    scalars, which matches how every single-sequence caller already passes ids.
    """
    if sequences is None:
        raise ValueError("a batch requires at least one sequence")
    if isinstance(sequences, np.ndarray):
        if sequences.ndim == 1:
            sequences = [sequences]
        elif sequences.ndim == 2:
            sequences = list(sequences)
        else:
            raise ValueError(
                f"a token id batch must be rank 1 or 2, got rank {sequences.ndim}"
            )
    elif not isinstance(sequences, Sequence):
        raise TypeError("sequences must be a sequence of token id arrays")

    rows: list[np.ndarray] = []
    for index, item in enumerate(sequences):
        row = np.asarray(item, dtype=np.int64).reshape(-1)
        if row.size == 0:
            raise ValueError(
                f"sequence {index} is empty; batched generation needs at least one "
                "token per row"
            )
        rows.append(row)
    if not rows:
        raise ValueError("a batch requires at least one sequence")
    return rows


def pack_left_padded(
    sequences: Any, *, pad_token_id: int = DEFAULT_PAD_TOKEN_ID
) -> PackedBatch:
    """Right-align ``sequences`` into one padded batch.

    Returns the padded ids together with the validity mask and the per-row
    positions, all three built from a single :class:`BatchLayout` so they cannot
    disagree.
    """
    rows = normalize_sequences(sequences)
    lengths = tuple(int(row.size) for row in rows)
    padded = max(lengths)
    layout = BatchLayout(lengths=lengths, padded_length=padded)

    batch = len(rows)
    token_ids = np.full((batch, padded), int(pad_token_id), dtype=np.int64)
    live = np.zeros((batch, padded), dtype=bool)
    position_ids = np.zeros((batch, padded), dtype=np.int64)
    for index, row in enumerate(rows):
        offset = padded - row.size
        token_ids[index, offset:] = row
        live[index, offset:] = True
        # Positions restart at 0 for the row's first real token, so a shorter row
        # is not rotated as though it began mid-sequence.
        position_ids[index, offset:] = np.arange(row.size, dtype=np.int64)
    return PackedBatch(
        token_ids=token_ids, live=live, position_ids=position_ids, layout=layout
    )
