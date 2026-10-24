"""Legacy candidate-tree inspection and greedy verification utilities.

This compatibility API has no model executor. It cannot generate a draft from
model identifiers. Executable draft/target decoding lives in speculative_engine;
these utilities only verify explicitly supplied candidates and target logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

from aether.core.constants import MINIMUM_ACCEPTANCE_RATE
from aether.core.exceptions import SpeculativeDecodingError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DraftTreeNode:
    """A node in a draft tree."""

    token_id: int
    """Token ID proposed by the draft model."""

    children: list[DraftTreeNode] = field(default_factory=list)
    """Child nodes in the draft tree."""

    depth: int = 0
    """Depth from the root."""

    probability: float = 1.0
    """Draft model probability."""

    def add_child(self, token_id: int, probability: float = 1.0) -> DraftTreeNode:
        """Add a child node."""
        child = DraftTreeNode(token_id=token_id, depth=self.depth + 1, probability=probability)
        self.children.append(child)
        return child

    def to_dict(self) -> dict[str, Any]:
        """Describe the supplied tree without performing inference."""
        return {
            "token_id": self.token_id,
            "depth": self.depth,
            "probability": self.probability,
            "children": [c.to_dict() for c in self.children],
        }


class TreeSpeculativeEngine:
    """Compatibility utilities; drafting requires a separate executing backend."""

    def __init__(self, target_model_id: str, draft_model_id: str | None = None) -> None:
        self.target_model_id = target_model_id
        self.draft_model_id = draft_model_id or self._select_draft_model(target_model_id)
        self._acceptance_rate = 0.0
        self._accepted_count = 0
        self._total_count = 0

    def _select_draft_model(self, target_model_id: str) -> str | None:
        """Return no implicit draft model.

        A target identifier does not tell the runtime which compatible draft
        checkpoint is installed, tokenizer-compatible, or licensed for use.
        Selecting one from a family-name table was both Qwen-biased and unsafe
        for arbitrary open models.  Callers must pass an explicit
        ``draft_model_id`` (or bind a model-aware scheduler) instead.
        """
        return None

    def build_draft_tree(
        self, prefix_tokens: list[int], max_depth: int = 4, branching_factor: int = 3
    ) -> DraftTreeNode:
        """Reject drafting because this API never loads a draft executor.

        Use Runtime with an explicit compatible draft model, or the executing
        DraftTargetSpecDecoder. An identifier, depth, or branching factor is
        insufficient evidence for a candidate token or its probability.
        """
        if not prefix_tokens:
            raise ValueError("Drafting requires a nonempty prefix; no root token is synthesized")
        raise SpeculativeDecodingError(
            "Legacy tree drafting has no draft executor. Use Runtime with an explicit "
            "compatible draft model or DraftTargetSpecDecoder with real engines."
        )

    def verify_tree(self, draft_tree: DraftTreeNode, target_logits: list[list[float]]) -> list[int]:
        """Verify the best supplied path against finite, equal-width target rows.

        Greedy acceptance requires the target argmax at each candidate's prefix
        (Leviathan et al., 2023, Section 2.2:
        https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).
        This utility does not execute the target or implement stochastic tree
        sampling. The returned root is existing context, not a generated token.
        Missing or malformed evidence raises before counters are committed.
        """
        if not isinstance(draft_tree, DraftTreeNode):
            raise SpeculativeDecodingError("Expected a supplied DraftTreeNode")
        if not isinstance(target_logits, (list, tuple)):
            raise SpeculativeDecodingError("Target logits must contain vocabulary rows")
        vocabulary: int | None = None
        for row in target_logits:
            if not isinstance(row, (list, tuple)) or not row:
                raise SpeculativeDecodingError("Each target logit row must be nonempty")
            if vocabulary is None:
                vocabulary = len(row)
            elif len(row) != vocabulary:
                raise SpeculativeDecodingError("Target logit rows must share one vocabulary size")
            if any(
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                for value in row
            ):
                raise SpeculativeDecodingError("Target logits must contain finite numeric values")
        accepted: list[int] = [draft_tree.token_id]
        node = draft_tree
        proposed = 0
        depth = 0
        visited: set[int] = set()
        while node.children:
            if id(node) in visited:
                raise SpeculativeDecodingError("Candidate path contains a cycle")
            visited.add(id(node))
            if depth >= len(target_logits):
                raise SpeculativeDecodingError("Missing target logits for a proposed continuation")
            for child in node.children:
                if not isinstance(child, DraftTreeNode):
                    raise SpeculativeDecodingError(
                        "Candidate children must be DraftTreeNode values"
                    )
                token = child.token_id
                if (
                    isinstance(token, bool)
                    or not isinstance(token, Integral)
                    or token < 0
                    or vocabulary is None
                    or token >= vocabulary
                ):
                    raise SpeculativeDecodingError(
                        "Candidate token is outside the target vocabulary"
                    )
                probability = child.probability
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, Real)
                    or not math.isfinite(probability)
                    or not 0.0 <= probability <= 1.0
                ):
                    raise SpeculativeDecodingError(
                        "Candidate probability must be finite and in [0, 1]"
                    )
            proposed += 1
            best = max(node.children, key=lambda child: child.probability)
            target_token = self._argmax_token(target_logits[depth])
            if target_token != best.token_id:
                break
            accepted.append(int(best.token_id))
            node = best
            depth += 1
        accepted_draft_tokens = max(0, len(accepted) - 1)
        self._accepted_count += accepted_draft_tokens
        self._total_count += proposed
        self._acceptance_rate = self._accepted_count / max(1, self._total_count)
        return accepted

    def prune_tree(self, draft_tree: DraftTreeNode, min_probability: float = 0.05) -> DraftTreeNode:
        """Remove candidate branches below a probability threshold in place."""
        draft_tree.children = [
            child for child in draft_tree.children if child.probability >= min_probability
        ]
        for child in draft_tree.children:
            self.prune_tree(child, min_probability=min_probability)
        return draft_tree

    def _argmax_token(self, logits: list[float]) -> int:
        """Return the token index with the largest target logit."""
        if not logits:
            raise SpeculativeDecodingError("A target logit row cannot be empty")
        return max(range(len(logits)), key=logits.__getitem__)

    def acceptance_rate(self) -> float:
        """Return the current acceptance rate."""
        return self._acceptance_rate

    def should_use_speculation(self) -> bool:
        """Recommend speculation only after target-verified observations."""
        return self._total_count > 0 and self._acceptance_rate >= MINIMUM_ACCEPTANCE_RATE

    def __repr__(self) -> str:
        return (
            f"TreeSpeculativeEngine(target={self.target_model_id}, "
            f"draft={self.draft_model_id}, acceptance={self._acceptance_rate:.2f})"
        )
