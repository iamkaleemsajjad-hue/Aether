"""Prevent documentation drift and promotion of unrecorded hardware evidence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tools.render_hardware_readme import END, ROOT, START, main, render, replace_table

if TYPE_CHECKING:
    from pathlib import Path


def test_readme_matches_every_catalog_target() -> None:
    """The committed README must match the current generator and catalog exactly."""
    matrix = json.loads((ROOT / "hardware_validation_matrix.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generated = render(matrix)
    assert replace_table(readme, generated) == readme
    for target in matrix["targets"]:
        assert generated.count(f"| `{target['target_id']}` |") == 1


def test_execution_claim_requires_dated_scoped_evidence() -> None:
    """Availability or an execution boolean alone cannot produce a verified row."""
    target = {
        "target_id": "example",
        "display_name": "Example device",
        "implemented": True,
        "compile_tested": False,
        "execution_tested": False,
        "available": True,
    }
    assert "awaiting hardware validation" in render({"targets": [target]})
    target["execution_tested"] = True
    with pytest.raises(ValueError, match="validation_date"):
        render({"targets": [target]})


def test_check_detects_drift_without_writing(tmp_path: Path) -> None:
    """A stale generated section fails its check and preserves the file bytes."""
    readme = tmp_path / "README.md"
    original = f"Before\n{START}\nStale\n{END}\nAfter\n"
    readme.write_text(original, encoding="utf-8")
    matrix = ROOT / "hardware_validation_matrix.json"
    args = ["--matrix", str(matrix), "--readme", str(readme)]
    assert main([*args, "--check"]) == 1
    assert readme.read_text(encoding="utf-8") == original
    assert main(args) == 0
    assert main([*args, "--check"]) == 0
    assert readme.read_text(encoding="utf-8").startswith("Before\n")
    assert readme.read_text(encoding="utf-8").endswith("\nAfter\n")


@pytest.mark.parametrize("text", ["Missing", f"{END}\n{START}", f"{START}{START}{END}"])
def test_missing_or_ambiguous_region_fails_closed(text: str) -> None:
    """Invalid marker placement must never replace unrelated README content."""
    with pytest.raises(ValueError):
        replace_table(text, "Generated")
