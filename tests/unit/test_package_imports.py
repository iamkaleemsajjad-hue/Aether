"""The distribution must be importable on its own, from anywhere.

These are packaging tests, not runtime tests.  They exist because the failure they
pin is silent: when the ``aether`` name resolves to an implicit *namespace* package
instead of to the installed distribution, ``import aether`` succeeds,
``aether.__file__`` is ``None``, and the first submodule import fails with a
``ModuleNotFoundError`` naming the submodule rather than the cause.  Nothing in the
runtime can detect that, because the package's own ``__init__.py`` never runs.

Every check runs in a **subprocess**.  That is the whole point: the parent pytest
process has already imported ``aether``, so asserting anything about it here would
only confirm that a cached module stays cached.  A fresh interpreter re-runs the
import system from ``site`` onward, which is the thing under test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_python(code: str, *, cwd: Path, extra_path: "list[str] | None" = None) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout.

    ``extra_path`` is prepended to ``PYTHONPATH`` so the child sees it at *startup*,
    the way a real environment would, rather than through a mid-flight ``sys.path``
    edit that the import system may already have cached around.
    """
    import os

    environment = dict(os.environ)
    if extra_path:
        joined = os.pathsep.join(extra_path)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            f"{joined}{os.pathsep}{existing}" if existing else joined
        )
    completed = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(cwd), env=environment, capture_output=True, text=True, timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"child interpreter failed (exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


PROBE = """
import json, importlib.util
spec = importlib.util.find_spec("aether")
import aether
from aether.compiler.compiler import Compiler
print(json.dumps({
    "file": aether.__file__,
    "paths": list(aether.__path__),
    "origin": spec.origin,
    "compiler": Compiler.__name__,
}))
"""


# ── the package resolves to a real distribution, not a namespace ───────────────

def test_aether_is_a_regular_package_not_a_namespace_package(tmp_path: Path) -> None:
    """``__file__ is None`` is the signature of the bug this suite exists for."""
    report = json.loads(run_python(PROBE, cwd=tmp_path))
    assert report["file"] is not None, (
        "aether resolved to an implicit namespace package; the installed "
        "distribution was not found"
    )
    assert report["origin"] is not None
    assert Path(report["file"]).name == "__init__.py"


def test_the_compiler_imports_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    """The published import must not depend on where the process happens to be."""
    report = json.loads(run_python(PROBE, cwd=tmp_path))
    assert report["compiler"] == "Compiler"


def test_the_compiler_imports_from_the_repository_root() -> None:
    """The developer's own cwd must behave the same as any other."""
    report = json.loads(run_python(PROBE, cwd=REPO_ROOT))
    assert report["compiler"] == "Compiler"


def test_a_directory_named_aether_on_sys_path_does_not_shadow_the_package(
    tmp_path: Path,
) -> None:
    """The exact Kaggle shape: a clone directory named ``aether`` beside the cwd.

    A regular package must win over a namespace portion wherever the portion appears
    on the path. Python's own resolution order guarantees this *provided* the real
    distribution is importable at all — which is what makes this a packaging test
    rather than an interpreter test.
    """
    shadow_root = tmp_path / "working"
    (shadow_root / "aether").mkdir(parents=True)
    report = json.loads(
        run_python(PROBE, cwd=shadow_root, extra_path=[str(shadow_root)])
    )
    assert report["file"] is not None
    assert str(shadow_root) not in report["paths"], (
        f"the empty directory shadowed the distribution: {report['paths']}"
    )
    assert report["compiler"] == "Compiler"


def test_a_shadow_directory_that_is_also_the_cwd_does_not_win(tmp_path: Path) -> None:
    """Running *inside* the clone must not resolve the clone as the package either."""
    clone = tmp_path / "aether"
    clone.mkdir()
    report = json.loads(run_python(PROBE, cwd=clone, extra_path=[str(tmp_path)]))
    assert report["file"] is not None
    assert report["compiler"] == "Compiler"


# ── the declared layout matches the tree ──────────────────────────────────────

def test_the_packaging_configuration_declares_the_src_layout_explicitly() -> None:
    """Inference is what made the failure silent, so the layout is stated outright."""
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        toml_reader = pytest.importorskip("tomli")
    data = toml_reader.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    setuptools_config = data["tool"]["setuptools"]
    assert setuptools_config["package-dir"] == {"": "src"}
    find = setuptools_config["packages"]["find"]
    assert find["where"] == ["src"]
    assert find["include"] == ["aether*"]
    assert find["namespaces"] is False, (
        "namespace discovery must stay off: it is what lets a directory without an "
        "__init__.py be packaged as if it were a package"
    )


def test_the_build_backend_is_the_one_the_configuration_describes() -> None:
    """Configuration for a backend that is not in use is a false lead when debugging."""
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        toml_reader = pytest.importorskip("tomli")
    data = toml_reader.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "hatch" not in data.get("tool", {}), (
        "hatch configuration describes a different layout and is not used by the "
        "declared build backend"
    )


def test_every_declared_package_has_an_init_file() -> None:
    """``namespaces = false`` is only meaningful if the tree actually satisfies it."""
    source_root = REPO_ROOT / "src" / "aether"
    missing = [
        str(directory.relative_to(REPO_ROOT))
        for directory in sorted(source_root.rglob("*"))
        if directory.is_dir()
        and directory.name != "__pycache__"
        and "__pycache__" not in directory.parts
        and not (directory / "__init__.py").is_file()
    ]
    assert not missing, (
        "these directories would be discovered as implicit namespace packages, or "
        f"silently dropped from the wheel: {missing}"
    )


def test_the_import_name_and_the_distribution_name_are_both_resolvable() -> None:
    """They differ (``aether`` vs ``aether-runtime``), which is a common source of
    "installed but not importable" confusion. Both must answer."""
    report = json.loads(run_python(
        """
        import json
        from importlib.metadata import version
        import aether
        print(json.dumps({
            "distribution": version("aether-runtime"),
            "module": aether.__file__,
        }))
        """,
        cwd=REPO_ROOT,
    ))
    assert report["distribution"]
    assert report["module"] is not None


# ── the entry point refuses to paper over a broken install ────────────────────

def test_the_profiling_script_reports_a_shadowed_package_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """A cryptic ModuleNotFoundError is what sent this investigation the wrong way."""
    script = REPO_ROOT / "scripts" / "profile_batch_scaling.py"
    source = script.read_text(encoding="utf-8")
    assert "sys.path.insert" not in source, (
        "the entry point must not repair sys.path: that hides a broken install and "
        "only works for callers who run this one file"
    )
    assert "require_installed_aether" in source
