"""
Weight pruning and sparsity mask computation.

Implements the importance metrics and mask construction used by Pass 9. Unlike a
sparsity *plan* (which only records target ratios), these functions compute real
boolean masks over real weight tensors.

Metrics
-------
* **Magnitude** — ``|W|``. The classic baseline; ignores activations.
* **Wanda** — ``|W_ij| * ||X_j||_2``, the per-output-row importance from
  *A Simple and Effective Pruning Approach for Large Language Models* (Sun et al.).
  Scores are compared **within each output row**, which is what makes Wanda work
  without weight update.
* **SparseGPT-style diagonal** — ``W^2 / diag(H)^-1`` approximated with the
  activation second moment, giving a Hessian-aware ordering.

Patterns
--------
* **Unstructured** — prune the globally lowest-importance fraction per row.
* **2:4 semi-structured** — within every contiguous group of 4 weights, keep
  exactly the 2 highest-importance entries. This is the pattern NVIDIA Ampere+
  sparse tensor cores accelerate, so the structural guarantee must hold exactly.
* **N:M generalisation** — keep ``n`` of every ``m``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ImportanceMetric",
    "SparsityPattern",
    "PruningMask",
    "compute_importance",
    "build_unstructured_mask",
    "build_nm_mask",
    "build_mask",
    "apply_mask",
    "verify_nm_pattern",
]

ImportanceMetric = Literal["magnitude", "wanda", "sparsegpt"]
SparsityPattern = Literal["unstructured", "2:4", "4:8", "1:2"]

#: Parsed (n, m) for each supported semi-structured pattern.
_NM_PATTERNS: dict[str, tuple[int, int]] = {"2:4": (2, 4), "4:8": (4, 8), "1:2": (1, 2)}


@dataclass
class PruningMask:
    """A boolean keep-mask over a weight tensor.

    ``mask`` is True where the weight is **kept**. Storing keep-semantics (rather
    than prune-semantics) matches how sparse kernels consume the metadata.
    """

    mask: np.ndarray
    pattern: str
    metric: str
    target_sparsity: float
    shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def achieved_sparsity(self) -> float:
        """Fraction of weights actually pruned."""
        if self.mask.size == 0:
            return 0.0
        return float(1.0 - self.mask.sum() / self.mask.size)

    @property
    def kept_count(self) -> int:
        return int(self.mask.sum())

    @property
    def pruned_count(self) -> int:
        return int(self.mask.size - self.mask.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "metric": self.metric,
            "target_sparsity": round(self.target_sparsity, 4),
            "achieved_sparsity": round(self.achieved_sparsity, 4),
            "shape": list(self.shape),
            "kept": self.kept_count,
            "pruned": self.pruned_count,
            **self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"PruningMask({self.pattern}, {self.metric}, "
            f"sparsity={self.achieved_sparsity:.3f}, shape={self.shape})"
        )


def compute_importance(
    weights: np.ndarray,
    metric: ImportanceMetric = "magnitude",
    activation_norms: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a per-weight importance score.

    Args:
        weights: 2-D weight matrix ``(out_features, in_features)``.
        metric: Importance metric to apply.
        activation_norms: Per-input-feature activation statistic of shape
            ``(in_features,)``. Required for ``"wanda"`` and ``"sparsegpt"``;
            for Wanda this is ``||X_j||_2`` over the calibration set.

    Returns:
        Non-negative importance scores with the same shape as ``weights``.

    Raises:
        ValueError: If ``weights`` is not 2-D, the metric is unknown, or an
            activation-dependent metric is requested without ``activation_norms``.
    """
    if weights.ndim != 2:
        msg = f"pruning importance requires a 2-D weight matrix, got shape {weights.shape}"
        raise ValueError(msg)

    w = np.abs(np.ascontiguousarray(weights, dtype=np.float32))

    if metric == "magnitude":
        return w

    if metric not in ("wanda", "sparsegpt"):
        msg = f"Unknown importance metric '{metric}' (expected magnitude, wanda, or sparsegpt)"
        raise ValueError(msg)

    if activation_norms is None:
        msg = f"metric '{metric}' requires activation_norms of shape (in_features,)"
        raise ValueError(msg)

    norms = np.ascontiguousarray(activation_norms, dtype=np.float32).ravel()
    if norms.size != weights.shape[1]:
        msg = (
            f"activation_norms has {norms.size} entries but weights have "
            f"{weights.shape[1]} input features"
        )
        raise ValueError(msg)

    if metric == "wanda":
        # |W_ij| * ||X_j||: broadcast the per-column activation norm across rows.
        return w * norms[None, :]

    # SparseGPT diagonal approximation: W^2 scaled by the activation second moment.
    return (w**2) * (norms[None, :] ** 2)


def build_unstructured_mask(importance: np.ndarray, target_sparsity: float, per_row: bool = True) -> np.ndarray:
    """Build an unstructured keep-mask pruning the lowest-importance weights.

    Args:
        importance: Non-negative importance scores.
        target_sparsity: Fraction to prune, in ``[0, 1)``.
        per_row: When True, prune the same fraction within each output row
            (Wanda's comparison group). When False, compare globally.

    Returns:
        Boolean keep-mask matching ``importance``'s shape.
    """
    if not 0.0 <= target_sparsity < 1.0:
        msg = f"target_sparsity must be in [0, 1), got {target_sparsity}"
        raise ValueError(msg)
    if target_sparsity == 0.0:
        return np.ones_like(importance, dtype=bool)

    if not per_row:
        flat = importance.ravel()
        n_prune = int(round(flat.size * target_sparsity))
        if n_prune <= 0:
            return np.ones_like(importance, dtype=bool)
        threshold_idx = np.argpartition(flat, n_prune - 1)[:n_prune]
        mask = np.ones(flat.size, dtype=bool)
        mask[threshold_idx] = False
        return mask.reshape(importance.shape)

    n_cols = importance.shape[1]
    n_prune = int(round(n_cols * target_sparsity))
    if n_prune <= 0:
        return np.ones_like(importance, dtype=bool)
    # Rank within each row; prune the n_prune smallest per row.
    order = np.argsort(importance, axis=1, kind="stable")
    mask = np.ones_like(importance, dtype=bool)
    rows = np.arange(importance.shape[0])[:, None]
    mask[rows, order[:, :n_prune]] = False
    return mask


def build_nm_mask(importance: np.ndarray, n: int, m: int) -> np.ndarray:
    """Build an N:M semi-structured keep-mask.

    Keeps exactly the ``n`` highest-importance weights in every contiguous group
    of ``m`` along the input dimension — the structural guarantee sparse tensor
    cores rely on.

    Args:
        importance: Non-negative importance scores, ``(out_features, in_features)``.
        n: Weights to keep per group.
        m: Group size.

    Returns:
        Boolean keep-mask matching ``importance``'s shape.

    Raises:
        ValueError: If ``n >= m``, either is non-positive, or the input dimension
            is not divisible by ``m``.
    """
    if n <= 0 or m <= 0 or n >= m:
        msg = f"invalid N:M pattern {n}:{m} (need 0 < n < m)"
        raise ValueError(msg)
    rows, cols = importance.shape
    if cols % m != 0:
        msg = (
            f"input dimension {cols} is not divisible by group size {m}; "
            f"N:M sparsity requires aligned groups"
        )
        raise ValueError(msg)

    groups = importance.reshape(rows, cols // m, m)
    # Rank descending within each group and keep the top n.
    order = np.argsort(-groups, axis=2, kind="stable")
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, order[:, :, :n], True, axis=2)
    return mask.reshape(rows, cols)


def build_mask(
    weights: np.ndarray,
    target_sparsity: float = 0.5,
    pattern: SparsityPattern = "unstructured",
    metric: ImportanceMetric = "magnitude",
    activation_norms: np.ndarray | None = None,
) -> PruningMask:
    """Compute a :class:`PruningMask` for a weight matrix.

    Args:
        weights: 2-D weight matrix ``(out_features, in_features)``.
        target_sparsity: Desired pruned fraction. Ignored for N:M patterns, whose
            sparsity is fixed at ``1 - n/m``.
        pattern: ``"unstructured"`` or an N:M pattern such as ``"2:4"``.
        metric: Importance metric.
        activation_norms: Per-input-feature activation norms for Wanda/SparseGPT.

    Returns:
        The computed mask with achieved-sparsity metadata attached.
    """
    importance = compute_importance(weights, metric=metric, activation_norms=activation_norms)

    if pattern in _NM_PATTERNS:
        n, m = _NM_PATTERNS[pattern]
        mask = build_nm_mask(importance, n, m)
        effective_target = 1.0 - n / m
        metadata: dict[str, Any] = {"n": n, "m": m, "structured": True}
    elif pattern == "unstructured":
        mask = build_unstructured_mask(importance, target_sparsity)
        effective_target = target_sparsity
        metadata = {"structured": False, "comparison_group": "per_output_row"}
    else:
        msg = (
            f"Unknown sparsity pattern '{pattern}' "
            f"(expected 'unstructured' or one of {sorted(_NM_PATTERNS)})"
        )
        raise ValueError(msg)

    return PruningMask(
        mask=mask,
        pattern=pattern,
        metric=metric,
        target_sparsity=effective_target,
        shape=tuple(weights.shape),
        metadata=metadata,
    )


def apply_mask(weights: np.ndarray, mask: PruningMask | np.ndarray) -> np.ndarray:
    """Zero out pruned weights, returning a new array.

    Args:
        weights: Weight tensor to prune.
        mask: A :class:`PruningMask` or a raw boolean keep-mask.

    Returns:
        A copy of ``weights`` with pruned entries set to exactly zero.
    """
    keep = mask.mask if isinstance(mask, PruningMask) else np.asarray(mask, dtype=bool)
    if keep.shape != weights.shape:
        msg = f"mask shape {keep.shape} does not match weight shape {weights.shape}"
        raise ValueError(msg)
    return np.where(keep, weights, np.zeros_like(weights))


def verify_nm_pattern(mask: np.ndarray, n: int, m: int) -> bool:
    """Check that every group of ``m`` keeps exactly ``n`` weights.

    Sparse tensor cores produce wrong results if this invariant is violated, so
    it is verified explicitly rather than assumed.
    """
    rows, cols = mask.shape
    if cols % m != 0:
        return False
    per_group = mask.reshape(rows, cols // m, m).sum(axis=2)
    return bool(np.all(per_group == n))
