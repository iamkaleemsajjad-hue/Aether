"""Deterministic prompts of an exact tokenized length.

Prompt size is controlled in *tokens*, not characters, because a character count
means different work for each of the three tokenizers.  The text is built from a
fixed corpus, encoded, truncated, and then re-encoded to confirm the round trip
still yields the requested count — decoding a truncated id sequence does not
always re-encode to the same length, and silently accepting a different length
would make two backends look like they were given the same work when they were
not.

Both backends receive the identical *string*, and both use the same tokenizer
(Aether packages the source tokenizer into the artifact), so identical text is
identical token ids.  ``verify_tokenizer_agreement`` asserts that rather than
assuming it.
"""

from __future__ import annotations

from typing import Any

#: A fixed, license-free corpus with no model-specific content.  Repeated as
#: needed; the exact wording is irrelevant, its stability across runs is not.
_CORPUS = (
    "The history of computing machinery begins with mechanical calculation. "
    "Early engines were designed to evaluate polynomials by finite differences. "
    "Later designs introduced stored programs, conditional branching, and "
    "addressable memory, which together made general purpose computation "
    "practical. Progress in semiconductor fabrication reduced the cost of "
    "arithmetic by many orders of magnitude, and the resulting abundance moved "
    "the binding constraint from computation to memory bandwidth and to the "
    "movement of data between processing elements. Modern accelerators expose "
    "wide parallel execution units and a deep memory hierarchy, so the "
    "performance of a program depends less on the number of arithmetic "
    "operations it performs and more on how those operations are scheduled, "
    "how often data is reloaded, and how much time is spent coordinating work "
    "rather than doing it. "
)


def flatten_ids(value: Any) -> list[int]:
    """Return token ids as a flat list of Python ints.

    Tokenizer implementations differ in what ``["input_ids"]`` contains: a flat
    list, a single-row batch, or a NumPy array.  Both backends' tokenizers are
    called through this so neither is advantaged by an encoding quirk.
    """
    import numpy as np

    array = np.asarray(value)
    return [int(item) for item in array.reshape(-1)]


def _encode(tokenizer: Any, text: str) -> list[int]:
    return flatten_ids(tokenizer(text, add_special_tokens=False)["input_ids"])


def make_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    """Return ``(text, achieved_token_count)`` for a prompt of ``target_tokens``.

    The achieved count is returned rather than assumed: when a tokenizer cannot
    represent exactly the requested length after a decode/encode round trip, the
    benchmark records what it actually used.  The same text is then handed to
    both backends, so the comparison stays controlled either way.
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    repeats = max(1, target_tokens // 8 + 2)
    ids = _encode(tokenizer, _CORPUS * repeats)
    while len(ids) < target_tokens:
        repeats *= 2
        ids = _encode(tokenizer, _CORPUS * repeats)

    text = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
    achieved = len(_encode(tokenizer, text))
    # Nudge the id count until the round trip lands on the target, or until it
    # is clear that this tokenizer cannot hit it exactly.
    budget = 24
    cut = target_tokens
    while achieved != target_tokens and budget > 0:
        cut += 1 if achieved < target_tokens else -1
        if cut < 1 or cut >= len(ids):
            break
        text = tokenizer.decode(ids[:cut], skip_special_tokens=True)
        achieved = len(_encode(tokenizer, text))
        budget -= 1
    return text, achieved


def build_prompt_set(tokenizer: Any, lengths: list[int]) -> dict[int, dict[str, Any]]:
    """Build one prompt per requested length, keyed by the requested length."""
    prompts: dict[int, dict[str, Any]] = {}
    for length in lengths:
        text, achieved = make_prompt(tokenizer, length)
        prompts[length] = {
            "requested_tokens": length,
            "achieved_tokens": achieved,
            "exact": achieved == length,
            "text": text,
        }
    return prompts


def verify_tokenizer_agreement(
    reference: Any, candidate: Any, texts: list[str]
) -> dict[str, Any]:
    """Check that two tokenizers encode the same texts to the same ids.

    Aether packages the source tokenizer into the ``.aeg``, so this should hold
    exactly.  It is verified rather than assumed because a tokenizer difference
    would invalidate every throughput and correctness comparison in the suite.
    """
    mismatches = []
    for text in texts:
        left = _encode(reference, text)
        right = _encode(candidate, text)
        if left != right:
            mismatches.append({
                "text_prefix": text[:60],
                "reference_len": len(left),
                "candidate_len": len(right),
                "first_divergence": next(
                    (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                    min(len(left), len(right)),
                ),
            })
    return {"identical": not mismatches, "mismatches": mismatches, "checked": len(texts)}


def context_fits(model_context: int | None, prompt_tokens: int, max_new_tokens: int) -> bool:
    """Whether a prompt plus its completion fits the model's declared context."""
    if not model_context:
        return True
    return prompt_tokens + max_new_tokens <= model_context
