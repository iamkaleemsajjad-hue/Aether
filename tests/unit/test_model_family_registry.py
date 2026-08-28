"""The published model-family counts must equal what the registry implements.

The failure this guards against is a documentation number that drifts away from
the code — "100+ model families" in a README while the compiler implements a few
dozen contracts.  Every count in ``README.md`` and ``SUPPORTED_MODELS.md`` is
derived from :mod:`aether.core.model_families`, and these tests fail the build
when a document and the registry disagree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.core.constants import SUPPORTED_ARCHITECTURES
from aether.core.model_families import (
    MODEL_FAMILIES,
    FamilyKind,
    SupportLevel,
    detection_key_count,
    executable_families,
    families_by_level,
    family_counts,
    support_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def counts() -> dict[str, int]:
    return family_counts()


# ── Registry integrity ────────────────────────────────────────────────────────

def test_family_keys_are_unique() -> None:
    keys = [family.key for family in MODEL_FAMILIES]
    assert len(keys) == len(set(keys)), "duplicate family keys collapse counts"


def test_family_names_are_unique() -> None:
    names = [family.name for family in MODEL_FAMILIES]
    assert len(names) == len(set(names))


def test_every_degraded_level_states_why() -> None:
    """A non-verified level without an explanation is an undocumented gap."""
    for family in MODEL_FAMILIES:
        if family.level is not SupportLevel.PARITY_VERIFIED:
            assert family.note, f"{family.key} has level {family.level.value} and no note"


def test_parity_verified_families_declare_their_numerics() -> None:
    for family in families_by_level(SupportLevel.PARITY_VERIFIED):
        assert family.numerics, f"{family.key} claims parity with no stated contract"
        assert family.representative_models, f"{family.key} names no checkpoint"


def test_counts_are_internally_consistent(counts: dict[str, int]) -> None:
    assert counts["total"] == len(MODEL_FAMILIES)
    assert counts["executable"] == len(executable_families())
    assert (
        counts["executable"]
        == counts["parity_verified"] + counts["runs"] + counts["known_incorrect"]
    )
    assert counts["total"] == counts["executable"] + counts["unsupported"]
    by_kind = sum(counts[kind.value] for kind in FamilyKind)
    assert by_kind == counts["total"], "every family has exactly one kind"
    assert counts["detection_keys"] == detection_key_count()


def test_detection_keys_outnumber_families(counts: dict[str, int]) -> None:
    """Names are coverage, not families — the distinction the old claim lost."""
    assert counts["detection_keys"] > counts["total"]


def test_declared_aether_family_exists_in_the_detection_registry() -> None:
    for family in MODEL_FAMILIES:
        if not family.aether_family:
            continue
        assert family.aether_family in SUPPORTED_ARCHITECTURES, (
            f"{family.key} compiles through {family.aether_family!r}, "
            "which is not a registered architecture family"
        )


def test_every_detection_key_resolves_to_some_family() -> None:
    """A key that resolves to nothing is a documented family Aether cannot find."""
    detector = ArchitectureDetector()
    unresolved: list[tuple[str, str]] = []
    for family in MODEL_FAMILIES:
        for key in family.detection_keys:
            resolved = detector._detect_family_from_arch_type(key) or detector._match_family(key)
            if resolved is None:
                unresolved.append((family.key, key))
    assert not unresolved, f"detection keys with no family mapping: {unresolved}"


# ── Documentation agreement ───────────────────────────────────────────────────

def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _table_counts(text: str) -> list[int]:
    """Extract the bolded family counts from a support-level markdown table."""
    return [int(value) for value in re.findall(r"\|\s*\*\*(\d+)\*\*\s*\|", text)]


@pytest.mark.parametrize("document", ["README.md", "SUPPORTED_MODELS.md"])
def test_documents_publish_the_registry_counts(document: str, counts: dict[str, int]) -> None:
    text = _read(document)
    published = _table_counts(text)
    expected = [
        counts["parity_verified"],
        counts["runs"],
        counts["known_incorrect"],
        counts["unsupported"],
    ]
    assert published[: len(expected)] == expected, (
        f"{document} publishes support-level counts {published[:len(expected)]} "
        f"but the registry has {expected}"
    )


@pytest.mark.parametrize("document", ["README.md", "SUPPORTED_MODELS.md"])
def test_documents_publish_the_family_and_key_totals(document: str, counts: dict[str, int]) -> None:
    text = _read(document)
    assert f"{counts['total']} architecture families" in text
    assert str(counts["detection_keys"]) in text
    assert f"{counts['executable']}" in text


def test_no_document_claims_a_round_marketing_number() -> None:
    """The specific regression: an unfalsifiable "100+ families" claim."""
    pattern = re.compile(r"(100\+|over 100|more than 100)\s*(model|architecture)?\s*famil", re.I)
    offenders = []
    for name in ("README.md", "SUPPORTED_MODELS.md", "REQUIREMENTS.md"):
        path = REPO_ROOT / name
        if path.is_file() and pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(name)
    assert not offenders, f"unverifiable family-count claim in: {offenders}"


def test_readme_decoder_table_lists_every_verified_decoder() -> None:
    text = _read("README.md")
    section = text.split("### The 26 parity-verified families", 1)
    assert len(section) == 2, "README is missing the verified-family table"
    body = section[1]
    for family in families_by_level(SupportLevel.PARITY_VERIFIED):
        assert family.name.split(" (")[0] in body, f"{family.name} missing from README table"


def test_support_summary_states_the_same_numbers(counts: dict[str, int]) -> None:
    sentence = support_summary()
    for key in ("parity_verified", "runs", "known_incorrect", "executable", "detection_keys"):
        assert str(counts[key]) in sentence
