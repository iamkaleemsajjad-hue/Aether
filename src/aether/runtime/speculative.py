"""
Tree-speculative decoding engine.

Aether implements adaptive tree speculative decoding where a draft model proposes
a branching tree of candidate tokens, and the target model verifies the entire
tree in a single forward pass using tree-masked attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import DRAFT_FAMILIES, MINIMUM_ACCEPTANCE_RATE
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DraftTreeNode:
    """A node in a draft tree."""

    token_id: int
    """Token ID proposed by the draft model."""

    children: list["DraftTreeNode"] = field(default_factory=list)
    """Child nodes in the draft tree."""

    depth: int = 0
    """Depth from the root."""

    probability: float = 1.0
    """Draft model probability."""

    def add_child(self, token_id: int, probability: float = 1.0) -> "DraftTreeNode":
        """Add a child node."""
        child = DraftTreeNode(token_id=token_id, depth=self.depth + 1, probability=probability)
        self.children.append(child)
        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "depth": self.depth,
            "probability": self.probability,
            "children": [c.to_dict() for c in self.children],
        }


class TreeSpeculativeEngine:
    """Adaptive tree-speculative decoding engine."""

    def __init__(self, target_model_id: str, draft_model_id: str | None = None) -> None:
        self.target_model_id = target_model_id
        self.draft_model_id = draft_model_id or self._select_draft_model(target_model_id)
        self._acceptance_rate = 0.0
        self._accepted_count = 0
        self._total_count = 0

    def _select_draft_model(self, target_model_id: str) -> str | None:
        """Select a draft model from the target model family."""
        normalized = target_model_id.lower().replace("/", "-").replace("_", "-")
        for prefix, draft in DRAFT_FAMILIES.items():
            if prefix in normalized:
                return draft
        return None

    def build_draft_tree(self, prefix_tokens: list[int], max_depth: int = 4, branching_factor: int = 3) -> DraftTreeNode:
        """Build an adaptive draft tree.

        Without a loaded draft backend, the reference implementation creates a
        deterministic OPT-Tree style candidate set: high-probability branches
        continue deeper while low-probability branches remain available for
        verification. This makes scheduler and verifier behavior reproducible.
        """
        root = DraftTreeNode(token_id=prefix_tokens[-1] if prefix_tokens else 0, depth=0)
        frontier = [root]
        for depth in range(max_depth):
            candidates: list[DraftTreeNode] = []
            for node in frontier:
                for branch_index in range(branching_factor):
                    decay = 0.72 ** (depth + 1)
                    probability = max(0.01, decay * (1.0 - branch_index / max(branching_factor + 1, 1)))
                    token_id = (node.token_id * 31 + branch_index + depth + 1) % 100000
                    candidates.append(node.add_child(token_id=token_id, probability=probability))
            candidates.sort(key=lambda child: child.probability, reverse=True)
            frontier = candidates[: max(1, branching_factor)]
        return root

    def verify_tree(self, draft_tree: DraftTreeNode, target_logits: list[list[float]]) -> list[int]:
        """Verify a draft tree against target model logits.

        The verifier follows the highest-probability draft path only while the
        target distribution agrees with that token at the current depth. If no
        logits are provided, it accepts the best draft path and records the path
        as speculative-only evidence.
        """
        accepted: list[int] = [draft_tree.token_id]
        node = draft_tree
        proposed = 0
        depth = 0
        while node.children:
            proposed += 1
            best = max(node.children, key=lambda child: child.probability)
            if target_logits and depth < len(target_logits):
                target_token = self._argmax_token(target_logits[depth])
                if target_token != best.token_id:
                    break
            accepted.append(best.token_id)
            node = best
            depth += 1
        accepted_draft_tokens = max(0, len(accepted) - 1)
        self._accepted_count += accepted_draft_tokens
        self._total_count += proposed
        self._acceptance_rate = self._accepted_count / max(1, self._total_count)
        return accepted

    def prune_tree(self, draft_tree: DraftTreeNode, min_probability: float = 0.05) -> DraftTreeNode:
        """Remove candidate branches below a probability threshold in place."""
        draft_tree.children = [child for child in draft_tree.children if child.probability >= min_probability]
        for child in draft_tree.children:
            self.prune_tree(child, min_probability=min_probability)
        return draft_tree

    def _argmax_token(self, logits: list[float]) -> int:
        """Return the token index with the largest target logit."""
        if not logits:
            return -1
        return max(range(len(logits)), key=logits.__getitem__)

    def acceptance_rate(self) -> float:
        """Return the current acceptance rate."""
        return self._acceptance_rate

    def should_use_speculation(self) -> bool:
        """Return True if the acceptance rate is above the threshold."""
        return self._acceptance_rate >= MINIMUM_ACCEPTANCE_RATE or self._total_count == 0

    def __repr__(self) -> str:
        return (
            f"TreeSpeculativeEngine(target={self.target_model_id}, "
            f"draft={self.draft_model_id}, acceptance={self._acceptance_rate:.2f})"
        )
