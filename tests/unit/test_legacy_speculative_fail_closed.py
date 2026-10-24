"""Legacy speculation must never invent candidates or accept missing evidence.

Positive verifier cases use explicitly constructed trees and supplied logits.
The legacy builder is called only to prove that unsupported generation fails.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from aether.core.constants import MINIMUM_ACCEPTANCE_RATE
from aether.core.exceptions import SpeculativeDecodingError
from aether.runtime import TreeSpeculativeEngine as PackageExport
from aether.runtime.runtime import TreeSpeculativeEngine as RuntimeExport
from aether.runtime.speculative import DraftTreeNode, TreeSpeculativeEngine


def _row(token: int, vocabulary: int = 4) -> list[float]:
    logits = [-3.0] * vocabulary
    logits[token] = 7.0
    return logits


def _chain(second_token: Any = 2) -> DraftTreeNode:
    root = DraftTreeNode(token_id=0)
    root.add_child(1, probability=0.9).add_child(second_token, probability=0.8)
    return root


def _branching_tree() -> DraftTreeNode:
    root = DraftTreeNode(token_id=3)
    first = root.add_child(1, probability=0.9)
    root.add_child(2, probability=0.1)
    second = first.add_child(4, probability=0.8)
    first.add_child(5, probability=0.2)
    second.add_child(2, probability=0.7)
    return root


def _stats(engine: TreeSpeculativeEngine) -> tuple[int, int, float, bool]:
    return (
        engine._accepted_count,
        engine._total_count,
        engine.acceptance_rate(),
        engine.should_use_speculation(),
    )


@pytest.fixture
def verified_engine() -> TreeSpeculativeEngine:
    """Seed nonzero counters so malformed evidence must preserve real observations."""
    engine = TreeSpeculativeEngine("target")
    root = DraftTreeNode(token_id=3)
    root.add_child(1, probability=0.9)
    assert engine.verify_tree(root, [_row(1)]) == [3, 1]
    assert _stats(engine) == (1, 1, 1.0, True)
    return engine


def _assert_atomic_rejection(
    engine: TreeSpeculativeEngine, tree: DraftTreeNode, evidence: Any
) -> None:
    before = _stats(engine)
    original_tree = tree.to_dict()
    with pytest.raises((SpeculativeDecodingError, ValueError)) as error:
        engine.verify_tree(tree, evidence)
    assert str(error.value).strip(), "Invalid evidence needs an explanatory error"
    assert _stats(engine) == before
    assert tree.to_dict() == original_tree


def test_existing_imports_constructor_and_method_signatures_remain_compatible() -> None:
    """Preserve imports and call signatures while removing unsupported generation."""
    assert PackageExport is RuntimeExport is TreeSpeculativeEngine
    parameters = inspect.signature(TreeSpeculativeEngine).parameters
    assert tuple(parameters) == ("target_model_id", "draft_model_id")
    assert parameters["draft_model_id"].default is None
    engine = PackageExport("target", "explicit-draft")
    assert engine.target_model_id == "target"
    assert engine.draft_model_id == "explicit-draft"
    assert TreeSpeculativeEngine("unknown/model").draft_model_id is None
    expected = {
        "build_draft_tree": ("self", "prefix_tokens", "max_depth", "branching_factor"),
        "verify_tree": ("self", "draft_tree", "target_logits"),
        "prune_tree": ("self", "draft_tree", "min_probability"),
    }
    for name, names in expected.items():
        assert tuple(inspect.signature(getattr(TreeSpeculativeEngine, name)).parameters) == names
    build = inspect.signature(TreeSpeculativeEngine.build_draft_tree).parameters
    assert build["max_depth"].default == 4
    assert build["branching_factor"].default == 3
    prune = inspect.signature(TreeSpeculativeEngine.prune_tree).parameters
    assert prune["min_probability"].default == 0.05


@pytest.mark.parametrize("draft_id", [None, "explicit-but-unloaded-draft"])
@pytest.mark.parametrize(
    "options",
    [{}, {"max_depth": 0}, {"max_depth": 2, "branching_factor": 1}],
    ids=["defaults", "zero-depth", "positive-depth"],
)
def test_legacy_builder_always_fails_without_an_executing_draft(
    draft_id: str | None, options: dict[str, int]
) -> None:
    """Neither an identifier nor a zero-depth request supplies a draft executor."""
    engine = TreeSpeculativeEngine("target", draft_id)
    before = _stats(engine)
    with pytest.raises(
        SpeculativeDecodingError,
        match=r"(?i)(draft|backend|executor|unavailable|unsupported)",
    ):
        engine.build_draft_tree([0], **options)
    assert _stats(engine) == before


@pytest.mark.parametrize("draft_id", [None, "explicit-but-unloaded-draft"])
def test_empty_prefix_fails_instead_of_inventing_root_token_zero(draft_id: str | None) -> None:
    """An absent context must never synthesize token zero."""
    engine = TreeSpeculativeEngine("target", draft_id)
    before = _stats(engine)
    with pytest.raises((SpeculativeDecodingError, ValueError), match=r"(?i)(prefix|empty|token)"):
        engine.build_draft_tree([])
    assert _stats(engine) == before


@pytest.mark.parametrize("draft_id", [None, "explicit-but-unloaded-draft"])
def test_speculation_is_not_recommended_before_any_target_verified_observation(
    draft_id: str | None,
) -> None:
    """An unobserved decoder has no verified acceptance evidence."""
    engine = TreeSpeculativeEngine("target", draft_id)
    assert _stats(engine) == (0, 0, 0.0, False)


def test_manual_tree_accepts_only_the_greedy_verified_path_and_excludes_root_from_stats() -> None:
    """Count supplied continuations, preserving existing context and the input tree."""
    engine = TreeSpeculativeEngine("target")
    tree = _branching_tree()
    original_tree = tree.to_dict()
    assert engine.verify_tree(tree, [_row(1, 6), _row(4, 6), _row(2, 6)]) == [3, 1, 4, 2]
    assert _stats(engine) == (3, 3, 1.0, True)
    assert tree.to_dict() == original_tree
    # Returning existing context from a leaf is not another accepted proposal.
    before = _stats(engine)
    assert engine.verify_tree(DraftTreeNode(token_id=3), [_row(1, 6)]) == [3]
    assert _stats(engine) == before


@pytest.mark.parametrize(
    "evidence,expected,accepted,proposed",
    [
        ([_row(2, 6)], [3], 0, 1),
        ([_row(1, 6), _row(5, 6)], [3, 1], 1, 2),
    ],
    ids=["first-candidate-mismatch", "later-candidate-mismatch"],
)
def test_manual_greedy_mismatch_stops_without_accepting_a_lower_ranked_sibling(
    evidence: list[list[float]], expected: list[int], accepted: int, proposed: int
) -> None:
    """A mismatch ends the selected path with exact proposal accounting."""
    engine = TreeSpeculativeEngine("target")
    assert engine.verify_tree(_branching_tree(), evidence) == expected
    assert engine._accepted_count == accepted
    assert engine._total_count == proposed
    assert engine.acceptance_rate() == pytest.approx(accepted / proposed)
    assert engine.should_use_speculation() is False


def test_verified_acceptance_counts_accumulate_exactly_across_calls() -> None:
    """Accumulate only accepted continuations and actually checked proposals."""
    engine = TreeSpeculativeEngine("target")
    assert engine.verify_tree(_chain(), [_row(1), _row(2)]) == [0, 1, 2]
    assert engine.verify_tree(_chain(), [_row(1), _row(3)]) == [0, 1]
    assert engine._accepted_count == 3
    assert engine._total_count == 4
    assert engine.acceptance_rate() == pytest.approx(0.75)
    assert engine.should_use_speculation() is (MINIMUM_ACCEPTANCE_RATE <= 0.75)


@pytest.mark.parametrize(
    "evidence", [None, [], [_row(1)]], ids=["none", "empty", "exhausted-after-match"]
)
def test_missing_or_exhausted_target_evidence_rejects_atomically(
    verified_engine: TreeSpeculativeEngine, evidence: Any
) -> None:
    """Missing rows cannot certify a partial or complete draft path."""
    _assert_atomic_rejection(verified_engine, _chain(), evidence)


@pytest.mark.parametrize(
    "evidence",
    [
        [0.0, 7.0, 0.0, 0.0],
        [_row(1), []],
        [_row(1), None],
        [_row(1), 2.0],
        [_row(1), "abcd"],
        [_row(1), [0.0, 0.0, "invalid", 0.0]],
        [_row(1), [[0.0, 1.0], [2.0, 3.0]]],
        {0: _row(1), 1: _row(2)},
    ],
    ids=[
        "flat-vector",
        "empty-row",
        "none-row",
        "scalar-row",
        "text-row",
        "nonnumeric-coordinate",
        "nested-row",
        "mapping-container",
    ],
)
def test_malformed_target_evidence_rejects_atomically(
    verified_engine: TreeSpeculativeEngine, evidence: Any
) -> None:
    """Malformed containers and coordinates produce an explicit unchanged-state error."""
    _assert_atomic_rejection(verified_engine, _chain(), evidence)


@pytest.mark.parametrize("second_vocabulary", [3, 5], ids=["narrower", "wider"])
def test_inconsistent_target_vocabulary_widths_reject_atomically(
    verified_engine: TreeSpeculativeEngine, second_vocabulary: int
) -> None:
    """Every position in one model must describe the same vocabulary."""
    _assert_atomic_rejection(verified_engine, _chain(), [_row(1), _row(2, second_vocabulary)])


@pytest.mark.parametrize("row_index", [0, 1], ids=["first-row", "after-match"])
@pytest.mark.parametrize(
    "nonfinite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_nonfinite_target_evidence_rejects_atomically(
    verified_engine: TreeSpeculativeEngine, row_index: int, nonfinite: float
) -> None:
    """Even an unused nonfinite logit invalidates the supplied target evidence."""
    evidence = [_row(1), _row(2)]
    # Even an unselected coordinate must not make invalid evidence acceptable.
    evidence[row_index][0] = nonfinite
    _assert_atomic_rejection(verified_engine, _chain(), evidence)


@pytest.mark.parametrize(
    "candidate",
    [-1, 4, 100000, True, 2.0, "2"],
    ids=["negative", "vocabulary-boundary", "large", "boolean", "float", "string"],
)
def test_oov_or_malformed_candidate_after_a_match_rejects_atomically(
    verified_engine: TreeSpeculativeEngine, candidate: Any
) -> None:
    """An invalid continuation must not commit preceding verified candidates."""
    _assert_atomic_rejection(verified_engine, _chain(candidate), [_row(1), _row(2)])
