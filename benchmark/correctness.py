"""Does Aether compute the same thing as Transformers?

Throughput is only meaningful if the two runtimes agree on the model's output, so
this module compares them at three levels of increasing strictness:

1. **logits** on the same prompt — the raw forward pass, before any sampling;
2. **greedy token ids** — whether the same argmax path is taken at every step;
3. **decoded text** — what a user would actually see.

Bit-for-bit equality is not the criterion. Both runtimes evaluate the same
mathematics with different kernels, different reduction orders, and (for Aether)
weights that have been through the AEG's BF16 residency, so small floating-point
differences are expected and legitimate. What matters is whether the differences
are at the scale of floating-point noise or at the scale of a different
computation, so the absolute and relative magnitudes are both reported and the
verdict is stated against an explicit tolerance.
"""

from __future__ import annotations

from typing import Any


def compare_logits(reference: Any, candidate: Any) -> dict[str, Any]:
    """Compare two logit vectors for the same prompt position.

    ``max_abs_diff`` alone is not interpretable — logit magnitudes differ by
    model — so it is also normalized by the reference's standard deviation, which
    is the scale on which a softmax actually distinguishes tokens.
    """
    import numpy as np

    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return {
            "comparable": False,
            "reason": f"vocabulary mismatch: {a.shape} vs {b.shape}",
        }
    difference = np.abs(a - b)
    spread = float(a.std()) or 1.0
    denominator = np.maximum(np.abs(a), 1e-6)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    # Softmax agreement is the quantity that actually governs sampling.
    probability_a = _softmax(a)
    probability_b = _softmax(b)
    return {
        "comparable": True,
        "vocab_size": int(a.size),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "max_rel_diff": float((difference / denominator).max()),
        "reference_std": spread,
        "max_abs_diff_over_std": float(difference.max() / spread),
        "cosine_similarity": cosine,
        "argmax_agrees": bool(a.argmax() == b.argmax()),
        "top5_agrees": bool(
            set(np.argsort(-a)[:5].tolist()) == set(np.argsort(-b)[:5].tolist())
        ),
        "max_prob_diff": float(np.abs(probability_a - probability_b).max()),
        "total_variation_distance": float(0.5 * np.abs(probability_a - probability_b).sum()),
    }


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = values - values.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def compare_token_ids(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    """Compare two greedy decodes, and locate where they first diverge.

    Greedy decoding is a sequence of argmax decisions, so a single near-tie can
    make two numerically equivalent runtimes diverge and then stay diverged. The
    divergence *index* therefore matters more than the raw match count: an early
    split suggests a real computational difference, a late one suggests a tie
    broken differently.
    """
    length = min(len(reference), len(candidate))
    first_divergence = next(
        (index for index in range(length) if reference[index] != candidate[index]),
        None,
    )
    matching_prefix = length if first_divergence is None else first_divergence
    return {
        "identical": reference == candidate,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "matching_prefix_tokens": matching_prefix,
        "matching_prefix_fraction": matching_prefix / length if length else 0.0,
        "first_divergence_index": first_divergence,
        "reference_at_divergence": (
            reference[first_divergence] if first_divergence is not None else None
        ),
        "candidate_at_divergence": (
            candidate[first_divergence] if first_divergence is not None else None
        ),
    }


def compare_text(reference: str, candidate: str) -> dict[str, Any]:
    """Compare decoded strings, including their longest common prefix."""
    limit = min(len(reference), len(candidate))
    shared = next(
        (index for index in range(limit) if reference[index] != candidate[index]), limit
    )
    return {
        "identical": reference == candidate,
        "reference_chars": len(reference),
        "candidate_chars": len(candidate),
        "common_prefix_chars": shared,
        "reference_prefix": reference[:200],
        "candidate_prefix": candidate[:200],
    }


#: Logit agreement is judged against the reference's own logit spread, so the
#: threshold means the same thing for a model with large logits and one with
#: small.  1e-2 of a standard deviation is far below what changes a softmax
#: ranking, and far above float32/BF16 rounding noise.
LOGIT_TOLERANCE_OVER_STD = 1e-2


def verdict(logits: dict[str, Any], tokens: dict[str, Any]) -> dict[str, Any]:
    """State plainly whether the two runtimes computed the same thing."""
    if not logits.get("comparable"):
        return {"equivalent": None, "reason": logits.get("reason", "logits not comparable")}
    scaled = logits["max_abs_diff_over_std"]
    within = scaled <= LOGIT_TOLERANCE_OVER_STD
    reasons = []
    if not within:
        reasons.append(
            f"logit deviation {scaled:.2e} exceeds {LOGIT_TOLERANCE_OVER_STD:.0e} of the "
            "reference logit spread, which is larger than floating-point noise"
        )
    if not logits["argmax_agrees"]:
        reasons.append("greedy next-token choice differs at the prompt position")
    if not tokens.get("identical") and tokens.get("first_divergence_index") == 0:
        reasons.append("greedy decode diverges at the very first generated token")
    return {
        "equivalent": bool(within and logits["argmax_agrees"]),
        "numerically_identical": logits["max_abs_diff"] == 0.0,
        "tokens_identical": bool(tokens.get("identical")),
        "logit_tolerance_over_std": LOGIT_TOLERANCE_OVER_STD,
        "observed_deviation_over_std": scaled,
        "concerns": reasons,
    }
