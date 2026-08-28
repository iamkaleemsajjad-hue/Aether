"""Stop-criteria helpers shared by every Aether executor.

An instruction-tuned checkpoint routinely ends a turn on a delimiter that is *not*
the tokenizer's canonical ``eos_token``. Phi-3.5-mini is the clear case: its
``tokenizer_config.eos_token`` is ``<|endoftext|>`` (32000), but a chat turn ends
with ``<|end|>`` (32007), and its ``generation_config.eos_token_id`` lists both.

Comparing a sampled token against a single id therefore means *never stopping* for
such a model: generation runs to ``max_tokens`` and the tail degenerates into
repetition. That reads as a model-quality failure and is a stopping failure, which
is why this lives in one place rather than being re-derived per engine — every
executor must agree on what "stop" means.

Framework-free by design, so the NumPy reference executors, the portable tensor
executor and the state-space executors can all share it.
"""

from __future__ import annotations

from typing import Any


def stop_token_set(eos_token_id: Any) -> frozenset[int]:
    """Normalize a stop-token argument into a set of token ids.

    Accepts ``None`` (no stop condition), a single id, or any iterable of ids, so a
    caller may pass whatever its tokenizer or ``generation_config`` provides without
    the engines needing to care which shape it was.

    ``bool`` is rejected even though it is an ``int`` in Python: silently treating
    ``True`` as token id 1 would halt generation on a real vocabulary entry.
    """
    if eos_token_id is None or isinstance(eos_token_id, bool):
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset({int(eos_token_id)})
    if isinstance(eos_token_id, (str, bytes)):
        # Iterable, but not a sequence of ids; treating it as one would produce
        # nonsense stop ids from character codes.
        return frozenset()
    try:
        return frozenset(
            int(value)
            for value in eos_token_id
            if isinstance(value, int) and not isinstance(value, bool)
        )
    except TypeError:
        return frozenset()
